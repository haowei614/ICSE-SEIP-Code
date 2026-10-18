#!/usr/bin/env python3
"""Download one Pillar-1 HF checkpoint at a time onto /workspace/models.

Safety: sequential only; never load into GPU; leave headroom on the shared node.
PVC (/aisin-repair) has ~6G free — do NOT write weights there.
"""
from __future__ import annotations

import argparse
import os
import sys

# Open models first; gated need HF_TOKEN + license accept on huggingface.co
OPEN = [
    ("Qwen/Qwen2.5-7B-Instruct", "Qwen2.5-7B-Instruct"),
    ("mistralai/Mistral-7B-Instruct-v0.3", "Mistral-7B-Instruct-v0.3"),
]
GATED = [
    ("meta-llama/Llama-3.1-8B-Instruct", "Llama-3.1-8B-Instruct"),
    ("google/gemma-2-9b-it", "Gemma-2-9B-it"),
]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--which", choices=["qwen25", "mistral", "llama", "gemma", "open"],
                   default="qwen25")
    p.add_argument("--out-root", default="/workspace/models")
    args = p.parse_args()

    mapping = {
        "qwen25": [OPEN[0]],
        "mistral": [OPEN[1]],
        "llama": [GATED[0]],
        "gemma": [GATED[1]],
        "open": OPEN,
    }
    targets = mapping[args.which]
    os.makedirs(args.out_root, exist_ok=True)

    from huggingface_hub import snapshot_download

    for repo, dirname in targets:
        dest = os.path.join(args.out_root, dirname)
        print(f"[download] {repo} -> {dest}", flush=True)
        snapshot_download(
            repo_id=repo,
            local_dir=dest,
            local_dir_use_symlinks=False,
            resume_download=True,
        )
        print(f"[ok] {dest}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
