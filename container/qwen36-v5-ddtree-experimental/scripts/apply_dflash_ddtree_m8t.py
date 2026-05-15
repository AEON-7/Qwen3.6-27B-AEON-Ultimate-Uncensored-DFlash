#!/usr/bin/env python3
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path


MARKER = "aeon_dflash_ddtree_m8t"


def replace_exact(path: Path, old: str, new: str) -> bool:
    text = path.read_text()
    if new in text:
        return False
    if old not in text:
        raise RuntimeError(f"Could not find expected text in {path}:\n{old}")
    path.write_text(text.replace(old, new, 1))
    return True


def replace_between(path: Path, start_marker: str, end_marker: str, new: str) -> None:
    text = path.read_text()
    if "aeon_dflash_ddtree_m8t" in text and new.strip() in text:
        return
    start = text.find(start_marker)
    if start < 0:
        raise RuntimeError(f"Could not find start marker in {path}: {start_marker}")
    end = text.find(end_marker, start)
    if end < 0:
        raise RuntimeError(f"Could not find end marker in {path}: {end_marker}")
    path.write_text(text[:start] + new + text[end:])


def clear_python_caches(pkg_root: Path) -> None:
    for pyc in pkg_root.rglob("*.pyc"):
        pyc.unlink(missing_ok=True)
    for pycache in pkg_root.rglob("__pycache__"):
        shutil.rmtree(pycache, ignore_errors=True)


M8T_RECURRENT_COMPACT_FN = r'''    def _compact_ddtree_recurrent_states(
        self,
        num_reqs: int,
        spec_decode_common_attn_metadata: CommonAttentionMetadata | None,
    ) -> None:
        # aeon_dflash_ddtree_m8t
        # Use the exact GDN spec_state_indices_tensor produced by the metadata
        # builder during the target verifier forward pass. Re-deriving this
        # from common attention metadata can accidentally use the full-attention
        # cache group's block ids, which are not valid for GDN state caches.
        if os.environ.get("DDTREE_USE_RUNTIME_SAMPLER", "0") != "1":
            return
        if os.environ.get("DDTREE_COMPACT_RECURRENT_STATE", "1") != "1":
            return
        entries = getattr(self, "_last_ddtree_gdn_state_metadata", None)
        if not entries:
            return
        runtime_req_ids = getattr(self, "_last_ddtree_runtime_req_ids", None)
        bonus_parent_by_req = getattr(
            self, "_last_ddtree_bonus_parent_compact_indices", None
        )
        if not runtime_req_ids or not bonus_parent_by_req:
            return

        accepted_counts = self.num_accepted_tokens.gpu[:num_reqs].detach().cpu()
        req_to_batch_index = {
            req_id: idx for idx, req_id in enumerate(self.input_batch.req_ids[:num_reqs])
        }
        compacted = 0

        for layer_names, spec_req_indices, state_indices in entries:
            if state_indices is None or state_indices.numel() == 0:
                continue
            row_by_req_index = {
                int(req_index): row_i
                for row_i, req_index in enumerate(spec_req_indices.tolist())
            }
            for req_id, bonus_parent in zip(
                runtime_req_ids, bonus_parent_by_req, strict=False
            ):
                req_i = req_to_batch_index.get(req_id)
                if req_i is None or req_i not in row_by_req_index:
                    continue
                output_count = int(accepted_counts[req_i].item())
                if output_count <= 0:
                    continue
                src_compact = int(bonus_parent)
                dst_compact = output_count - 1
                row = state_indices[row_by_req_index[req_i]]
                if (
                    src_compact == dst_compact
                    or src_compact < 0
                    or dst_compact < 0
                    or src_compact >= row.numel()
                    or dst_compact >= row.numel()
                ):
                    continue

                base_block_id = int(row[0].item())
                src_block_id = int(row[src_compact].item())
                dst_ssm_block_id = int(row[dst_compact].item())
                if base_block_id <= 0 or src_block_id <= 0 or dst_ssm_block_id <= 0:
                    continue

                for layer_name in layer_names:
                    attention = self.compilation_config.static_forward_context.get(
                        layer_name
                    )
                    kv_caches = getattr(attention, "kv_cache", None)
                    if not isinstance(kv_caches, (list, tuple)) or len(kv_caches) < 2:
                        continue

                    conv_state, ssm_state = kv_caches[0], kv_caches[1]
                    if (
                        base_block_id >= conv_state.shape[0]
                        or src_block_id >= conv_state.shape[0]
                        or dst_ssm_block_id >= ssm_state.shape[0]
                    ):
                        continue

                    conv_width = int(getattr(attention, "conv_kernel_size", 1)) - 1
                    if conv_width <= 0 or conv_state.ndim < 3:
                        continue

                    conv_dim = getattr(attention, "conv_dim", None)
                    if conv_dim is not None and int(conv_state.shape[-1]) == int(conv_dim):
                        state_len = int(conv_state.shape[1])
                        if dst_compact + conv_width > state_len:
                            continue
                        conv_state[
                            base_block_id,
                            dst_compact : dst_compact + conv_width,
                        ].copy_(conv_state[src_block_id, :conv_width].clone())
                    else:
                        state_len = int(conv_state.shape[-1])
                        if dst_compact + conv_width > state_len:
                            continue
                        conv_state[
                            base_block_id,
                            :,
                            dst_compact : dst_compact + conv_width,
                        ].copy_(conv_state[src_block_id, :, :conv_width].clone())

                    if src_block_id != dst_ssm_block_id:
                        ssm_state[dst_ssm_block_id].copy_(
                            ssm_state[src_block_id].clone()
                        )
                    compacted += 1

        if os.environ.get("DDTREE_LOG_STATE_COMPACT", "0") == "1" and not getattr(
            self, "_ddtree_m8t_logged", False
        ):
            logger.warning(
                "DDTree M8T recurrent state compaction from GDN metadata compacted=%s",
                compacted,
            )
            self._ddtree_m8t_logged = True

'''


def patch_gpu_model_runner(pkg_root: Path) -> None:
    path = pkg_root / "v1/worker/gpu_model_runner.py"
    replace_between(
        path,
        "    def _compact_ddtree_recurrent_states(\n",
        "    def _ddtree_state_token_counts(\n",
        M8T_RECURRENT_COMPACT_FN,
    )
    replace_exact(
        path,
        """        cached_attn_metadata: dict[
            tuple[KVCacheSpec, type[AttentionMetadataBuilder]], AttentionMetadata
        ] = {}

        def _build_attn_group_metadata(
""",
        """        cached_attn_metadata: dict[
            tuple[KVCacheSpec, type[AttentionMetadataBuilder]], AttentionMetadata
        ] = {}
        # aeon_dflash_ddtree_m8t
        # Saved after each build for sampler-side recurrent-state compaction.
        self._last_ddtree_gdn_state_metadata = []

        def _build_attn_group_metadata(
""",
    )
    replace_exact(
        path,
        """            if ubid is None:
                assert isinstance(attn_metadata, dict)
""",
        """            # aeon_dflash_ddtree_m8t
            # Capture the exact GDN state-index rows used by the verifier
            # forward pass. These are already filtered to spec rows by the GDN
            # builder, so preserve the active request indices alongside them.
            if (
                use_spec_decode
                and isinstance(builder, GDNAttentionMetadataBuilder)
                and hasattr(attn_metadata_i, "spec_state_indices_tensor")
            ):
                spec_state_indices = getattr(
                    attn_metadata_i, "spec_state_indices_tensor", None
                )
                spec_sequence_masks = getattr(
                    attn_metadata_i, "spec_sequence_masks", None
                )
                if spec_state_indices is not None and spec_sequence_masks is not None:
                    live_mask = spec_sequence_masks[:num_reqs]
                    if getattr(live_mask, "dtype", None) == torch.bool:
                        spec_req_indices = (
                            torch.nonzero(live_mask, as_tuple=False)
                            .flatten()
                            .detach()
                            .cpu()
                        )
                    else:
                        spec_req_indices = torch.arange(
                            min(num_reqs, spec_state_indices.shape[0]),
                            dtype=torch.long,
                        )
                    if spec_req_indices.numel() > 0:
                        self._last_ddtree_gdn_state_metadata.append(
                            (
                                tuple(attn_group.layer_names),
                                spec_req_indices,
                                spec_state_indices[: spec_req_indices.numel()]
                                .detach()
                                .cpu(),
                            )
                        )

            if ubid is None:
                assert isinstance(attn_metadata, dict)
""",
    )


def verify_static(pkg_root: Path) -> None:
    runner = (pkg_root / "v1/worker/gpu_model_runner.py").read_text()
    for needle in (
        "aeon_dflash_ddtree_m8t",
        "_last_ddtree_gdn_state_metadata",
        "DDTree M8T recurrent state compaction from GDN metadata",
        "spec_state_indices[: spec_req_indices.numel()]",
        "row_by_req_index",
    ):
        if needle not in runner:
            raise RuntimeError(f"Static M8T verification failed: missing {needle}")


def main() -> int:
    root_override = os.environ.get("VLLM_PACKAGE_ROOT")
    if root_override:
        pkg_root = Path(root_override).resolve()
    else:
        import vllm

        pkg_root = Path(vllm.__file__).resolve().parent

    print(f"[{MARKER}] vLLM package root: {pkg_root}")
    patch_gpu_model_runner(pkg_root)
    clear_python_caches(pkg_root)
    verify_static(pkg_root)
    print(f"[{MARKER}] DDTree GDN metadata-backed recurrent compaction verified")
    return 0


if __name__ == "__main__":
    sys.exit(main())
