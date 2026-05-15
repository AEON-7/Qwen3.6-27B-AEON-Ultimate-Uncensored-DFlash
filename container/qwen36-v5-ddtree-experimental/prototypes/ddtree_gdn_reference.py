#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def tree_conv1d_reference(
    x: torch.Tensor,
    weight: torch.Tensor,
    parent_ids: torch.Tensor,
    initial_state: torch.Tensor,
    *,
    bias: torch.Tensor | None = None,
    apply_silu: bool = False,
) -> torch.Tensor:
    """Tree-aware causal conv1d reference for Qwen3.6 GDN blocks.

    Args:
        x: [T, D] current tree-token projection.
        weight: [D, K] depthwise conv kernel.
        parent_ids: [T] parent node index in flattened tree, -1 for root.
        initial_state: [K - 1, D] pre-tree conv history, oldest -> newest.
    """
    if x.ndim != 2:
        raise ValueError("x must be [T, D]")
    if weight.ndim != 2:
        raise ValueError("weight must be [D, K]")
    T, D = x.shape
    if parent_ids.shape != (T,):
        raise ValueError("parent_ids must be [T]")
    if weight.shape[0] != D:
        raise ValueError("weight dim mismatch")
    K = weight.shape[1]
    if initial_state.shape != (K - 1, D):
        raise ValueError("initial_state must be [K - 1, D]")

    sx = torch.cat([initial_state, x], dim=0)
    outputs: list[torch.Tensor] = []
    for token_index in range(T):
        virtual_slots = [token_index]
        cursor = token_index
        for _ in range(K - 1):
            if cursor >= 0:
                cursor = int(parent_ids[cursor].item())
            else:
                cursor -= 1
            virtual_slots.append(cursor)
        virtual_slots.reverse()
        sx_slots = [K - 1 + slot for slot in virtual_slots]
        window = sx[sx_slots]
        out = torch.sum(window * weight.transpose(0, 1), dim=0)
        if bias is not None:
            out = out + bias
        if apply_silu:
            out = torch.nn.functional.silu(out)
        outputs.append(out)
    return torch.stack(outputs, dim=0)


def _delta_rule_step(
    state: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gate: torch.Tensor,
    beta: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    # State layout mirrors vLLM's Triton kernel: [HV, V, K].
    decayed = state * torch.exp(gate).view(-1, 1, 1)
    kv = torch.einsum("hvk,hk->hv", decayed, k)
    corrected_v = (v - kv) * torch.sigmoid(beta).unsqueeze(-1)
    updated = decayed + torch.einsum("hv,hk->hvk", corrected_v, k)
    out = torch.einsum("hvk,hk->hv", updated, q)
    return out, updated


def tree_gated_delta_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gate: torch.Tensor,
    beta: torch.Tensor,
    parent_ids: torch.Tensor,
    initial_state: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Tree-aware Gated DeltaNet recurrence reference.

    Args:
        q, k: [T, H, K]
        v: [T, HV, V]
        gate, beta: [T, HV]
        parent_ids: [T] parent node index in flattened tree, -1 for root.
        initial_state: [HV, V, K] pre-tree state.
    """
    T = q.shape[0]
    if parent_ids.shape != (T,):
        raise ValueError("parent_ids must be [T]")
    outputs: list[torch.Tensor] = []
    states: list[torch.Tensor] = []
    for token_index in range(T):
        parent = int(parent_ids[token_index].item())
        state = initial_state if parent < 0 else states[parent]
        out, updated = _delta_rule_step(
            state.clone(),
            q[token_index],
            k[token_index],
            v[token_index],
            gate[token_index],
            beta[token_index],
        )
        outputs.append(out)
        states.append(updated)
    return torch.stack(outputs, dim=0), torch.stack(states, dim=0)


def replay_path_gated_delta(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gate: torch.Tensor,
    beta: torch.Tensor,
    path: list[int],
    initial_state: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    state = initial_state.clone()
    outputs: list[torch.Tensor] = []
    for token_index in path:
        out, state = _delta_rule_step(
            state,
            q[token_index],
            k[token_index],
            v[token_index],
            gate[token_index],
            beta[token_index],
        )
        outputs.append(out)
    return torch.stack(outputs, dim=0), state


def path_to_root(parent_ids: torch.Tensor, token_index: int) -> list[int]:
    path: list[int] = []
    cursor = token_index
    while cursor >= 0:
        path.append(cursor)
        cursor = int(parent_ids[cursor].item())
    path.reverse()
    return path


def run_reference(seed: int = 7) -> dict[str, object]:
    torch.manual_seed(seed)
    # DFS-flattened tree:
    # 0, 1, 2 is the main chain; 3 branches from node 1; 4 branches from root.
    parent_ids = torch.tensor([-1, 0, 1, 1, -1], dtype=torch.int64)
    T, D, KERNEL = 5, 8, 4
    HV, K_DIM, V_DIM = 2, 4, 4

    x = torch.randn(T, D, dtype=torch.float64)
    weight = torch.randn(D, KERNEL, dtype=torch.float64)
    conv_initial = torch.randn(KERNEL - 1, D, dtype=torch.float64)
    conv_out = tree_conv1d_reference(x, weight, parent_ids, conv_initial)

    q = torch.randn(T, HV, K_DIM, dtype=torch.float64)
    k = torch.randn(T, HV, K_DIM, dtype=torch.float64)
    v = torch.randn(T, HV, V_DIM, dtype=torch.float64)
    gate = torch.randn(T, HV, dtype=torch.float64) * 0.1
    beta = torch.randn(T, HV, dtype=torch.float64)
    gdn_initial = torch.randn(HV, V_DIM, K_DIM, dtype=torch.float64)
    tree_out, tree_states = tree_gated_delta_reference(
        q, k, v, gate, beta, parent_ids, gdn_initial
    )

    max_conv_diff = 0.0
    max_gdn_out_diff = 0.0
    max_gdn_state_diff = 0.0
    for token_index in range(T):
        path = path_to_root(parent_ids, token_index)
        # Conv replay uses the same reference on just the path with a chain
        # topology, then compares the final path token.
        path_x = x[path]
        path_parent = torch.arange(-1, len(path) - 1, dtype=torch.int64)
        path_conv = tree_conv1d_reference(path_x, weight, path_parent, conv_initial)
        max_conv_diff = max(
            max_conv_diff,
            float(torch.max(torch.abs(conv_out[token_index] - path_conv[-1])).item()),
        )

        path_out, path_state = replay_path_gated_delta(
            q, k, v, gate, beta, path, gdn_initial
        )
        max_gdn_out_diff = max(
            max_gdn_out_diff,
            float(torch.max(torch.abs(tree_out[token_index] - path_out[-1])).item()),
        )
        max_gdn_state_diff = max(
            max_gdn_state_diff,
            float(torch.max(torch.abs(tree_states[token_index] - path_state)).item()),
        )

    return {
        "seed": seed,
        "parent_ids": parent_ids.tolist(),
        "max_conv_diff_vs_path_replay": max_conv_diff,
        "max_gdn_out_diff_vs_path_replay": max_gdn_out_diff,
        "max_gdn_state_diff_vs_path_replay": max_gdn_state_diff,
        "matches_path_replay": (
            max_conv_diff < 1e-10
            and max_gdn_out_diff < 1e-10
            and max_gdn_state_diff < 1e-10
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tree-aware GDN/conv reference check.")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = run_reference(args.seed)
    output = json.dumps(result, indent=2)
    if args.output:
        Path(args.output).write_text(output + "\n")
    print(output)
    return 0 if result["matches_path_replay"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
