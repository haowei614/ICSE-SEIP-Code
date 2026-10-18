"""Host and cell provenance for Pillar-1 logs.

The config hash identifies a cell without the git commit. Resume skips a cell
only when both the hash and the commit match, so a new commit reruns the cell.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any, Optional


def repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / ".git").exists():
            return parent
    return here.parents[2]


def git_commit(root: Optional[Path] = None) -> str:
    root = root or repo_root()
    out = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    )
    return out.strip()


def git_dirty(root: Optional[Path] = None) -> bool:
    """True when source files differ from HEAD.

    Result logs under replication/results/ are ignored. The runner writes them
    between cells, and those files must not abort the next cell.
    """
    root = root or repo_root()
    out = subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=root, text=True
    )
    for line in out.splitlines():
        path = line[3:].strip()
        if " -> " in path:
            path = path.split(" -> ", 1)[1].strip()
        if path.startswith('"') and path.endswith('"'):
            path = path[1:-1]
        if path == "replication/results" or path.startswith("replication/results/"):
            continue
        return True
    return False


def llama_cpp_version() -> Optional[str]:
    return _pkg_version("llama-cpp-python")


def _pkg_version(name: str) -> Optional[str]:
    try:
        from importlib.metadata import version
        return version(name)
    except Exception:
        return None


def cuda_and_driver() -> tuple[Optional[str], Optional[str]]:
    """Return (cuda_version, driver_version) from nvidia-smi, or (None, None)."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            text=True, stderr=subprocess.DEVNULL,
        )
        driver = out.strip().splitlines()[0].strip() if out.strip() else None
    except Exception:
        driver = None
    cuda = None
    try:
        banner = subprocess.check_output(
            ["nvidia-smi"], text=True, stderr=subprocess.DEVNULL
        )
        for line in banner.splitlines():
            if "CUDA Version" in line:
                # e.g. "| CUDA Version: 12.4     |"
                cuda = line.split("CUDA Version:", 1)[1].split("|", 1)[0].strip()
                break
    except Exception:
        pass
    return cuda, driver


def file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def config_hash(parts: dict[str, Any]) -> str:
    blob = json.dumps(parts, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def cell_identity(
    *,
    model_id: str,
    precision: str,
    task_id: str,
    prompt_version: str,
    wrapper: str,
    input_mode: str,
    decoding: dict,
    serving: dict,
    sampling_seed: int,
    sampling_n: int,
    gguf_sha256: Optional[str],
) -> dict[str, Any]:
    out = {
        "model_id": model_id,
        "precision": precision,
        "task_id": task_id,
        "prompt_version": prompt_version,
        "wrapper": wrapper,
        "input_mode": input_mode,
        "temperature": decoding.get("temperature"),
        "top_p": decoding.get("top_p"),
        "seed": decoding.get("seed"),
        "max_new_tokens": decoding.get("max_new_tokens"),
        "n_ctx": serving.get("n_ctx"),
        "n_batch": serving.get("n_batch"),
        "n_gpu_layers": serving.get("n_gpu_layers"),
        "sampling_seed": sampling_seed,
        "sampling_n": sampling_n,
        "gguf_sha256": gguf_sha256,
    }
    # Absent when false, so hashes of the aligned_v2 cells stay stable.
    if decoding.get("use_grammar"):
        out["use_grammar"] = True
    return out


def host_provenance(root: Optional[Path] = None) -> dict[str, Any]:
    cuda, driver = cuda_and_driver()
    return {
        "git_commit": git_commit(root),
        "git_dirty": git_dirty(root),
        "llama_cpp_python": llama_cpp_version(),
        "transformers": _pkg_version("transformers"),
        "torch": _pkg_version("torch"),
        "cuda_version": cuda,
        "driver_version": driver,
    }


def load_jsonl(path: str) -> tuple[list[dict], bool]:
    """Return records and whether a torn final line was dropped."""
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return [], False
    with open(path, "r", encoding="utf-8") as fh:
        text = fh.read()
    lines = text.splitlines()
    records: list[dict] = []
    torn = False
    for i, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            if i == len(lines) - 1:
                torn = True
                continue
            raise
    if torn:
        kept = [json.dumps(r, ensure_ascii=False) for r in records]
        body = ("\n".join(kept) + "\n") if kept else ""
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(body)
    return records, torn
