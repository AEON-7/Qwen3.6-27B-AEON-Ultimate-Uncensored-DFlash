#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "prototypes"))

from ddtree_gdn_reference import run_reference  # noqa: E402


def test_reference_matches_path_replay() -> None:
    for seed in (1, 7, 17):
        result = run_reference(seed)
        assert result["matches_path_replay"], result


if __name__ == "__main__":
    test_reference_matches_path_replay()
    print("ddtree GDN reference tests passed")
