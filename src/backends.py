"""Model backends: (prompt, decoding) -> Generation with resources.

Backends available:
  - MockBackend       : dry-run only; NEVER emits quality; mock=True.
  - TransformersBackend: HF safetensors on Apple MPS (or CUDA/CPU). Works with the
                         torch/transformers already installed on the pilot host.
                         Peak memory via torch.mps.* (Apple) or torch.cuda.* (CUDA).
                         NOTE: gives fp16/bf16 only on MPS (no bitsandbytes on MPS),
                         so it is NOT the quantization-ladder path.
  - LlamaCppBackend   : GGUF K-quant via llama-cpp-python (Metal on Apple / CUDA).
                         Quantization-study path (Q4_K_M/Q6_K/Q8_0) plus F16 GGUF
                         same-stack control.
  - CloudBackend      : OpenAI- or Anthropic-compatible API; upper-bound reference
                         only under Basis A (never ground truth / Basis B yardstick).

All heavy imports are lazy so the harness/tests import without GPU deps.
The driver (run_experiments.py) wraps generate() to catch OOM/timeout/exception and
record them as flags -- backends may raise; they must not fabricate numbers.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional, Protocol


@dataclass
class Generation:
    text: str
    prompt_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    total_latency_s: Optional[float] = None
    ttft_s: Optional[float] = None
    tokens_per_s: Optional[float] = None
    peak_memory_mb: Optional[float] = None
    peak_memory_method: Optional[str] = None
    mock: bool = False
    finish_reason: Optional[str] = None


class Backend(Protocol):
    name: str

    def generate(self, prompt: str, decoding: dict) -> Generation: ...


# --------------------------------------------------------------------------- #
# Mock (dry-run)                                                               #
# --------------------------------------------------------------------------- #
class MockBackend:
    """Deterministic dry-run backend. Emits a placeholder; NEVER a quality signal."""

    name = "mock"

    def __init__(self, model_id: str, precision: str, **kw):
        self.model_id = model_id
        self.precision = precision

    def close(self) -> None:
        return None

    def generate(self, prompt: str, decoding: dict) -> Generation:
        t0 = time.time()
        text = ("[MOCK OUTPUT - dry-run only. No real model was executed. "
                "This record cannot populate paper macros.]")
        return Generation(
            text=text,
            prompt_tokens=None,
            output_tokens=None,
            total_latency_s=time.time() - t0,   # pipeline overhead only
            ttft_s=None,
            tokens_per_s=None,
            peak_memory_mb=None,
            peak_memory_method=None,
            mock=True,
        )


# --------------------------------------------------------------------------- #
# Memory helpers                                                               #
# --------------------------------------------------------------------------- #
def _reset_peak_memory(device: str) -> None:
    try:
        import torch
        if device == "cuda" and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        elif device == "mps" and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            fn = getattr(torch.mps, "reset_peak_memory_stats", None)
            if fn:
                fn()
    except Exception:
        pass


def _read_peak_memory(device: str) -> tuple[Optional[float], Optional[str]]:
    try:
        import torch
        if device == "cuda" and torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / (1024 ** 2), "torch.cuda.max_memory_allocated"
        if device == "mps" and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            fn = getattr(torch.mps, "driver_allocated_memory", None) or getattr(torch.mps, "current_allocated_memory", None)
            if fn:
                return fn() / (1024 ** 2), "torch.mps.driver_allocated_memory"
    except Exception:
        pass
    return None, None


def _rss_mb() -> Optional[float]:
    """Process resident-set size in MB (fallback memory proxy for llama.cpp Metal)."""
    try:
        import resource
        # ru_maxrss is bytes on macOS, kilobytes on Linux.
        maxrss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        import sys
        return maxrss / (1024 ** 2) if sys.platform == "darwin" else maxrss / 1024
    except Exception:
        return None


def _pick_device(prefer: Optional[str]) -> str:
    try:
        import torch
        if prefer:
            return prefer
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


# --------------------------------------------------------------------------- #
# Transformers backend (Apple MPS / CUDA / CPU)                               #
# --------------------------------------------------------------------------- #
class TransformersBackend:
    name = "transformers"

    def __init__(self, model_id: str, precision: str, hf_name: str,
                 device: Optional[str] = None, revision: Optional[str] = None, **kw):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.model_id = model_id
        self.precision = precision
        self.device = _pick_device(device)
        dtype = torch.float16 if precision.upper() in ("FP16", "F16", "Q8_0", "Q6_K", "Q4_K_M") else torch.bfloat16
        # NOTE: MPS cannot do bitsandbytes int-quant; precision here is fp16/bf16 only.
        # trust_remote_code: needed for some Qwen checkpoints; local dirs also supported.
        self.tok = AutoTokenizer.from_pretrained(
            hf_name, revision=revision, trust_remote_code=True)
        load_kw = dict(revision=revision, torch_dtype=dtype, trust_remote_code=True)
        # device_map=auto places layers on GPU when VRAM allows (CUDA servers).
        if self.device == "cuda":
            load_kw["device_map"] = "auto"
            # Keep weights on GPU; avoid silent CPU offload that makes generation crawl.
            load_kw["max_memory"] = {0: "22GiB"}
            self.model = AutoModelForCausalLM.from_pretrained(hf_name, **load_kw)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(hf_name, **load_kw).to(self.device)
        self.model.eval()
        if self.tok.pad_token_id is None:
            self.tok.pad_token = self.tok.eos_token

    def close(self) -> None:
        """Release GPU/CPU weights so the next model can load (one-at-a-time)."""
        import gc
        try:
            import torch
            del self.model
            del self.tok
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        except Exception:
            pass
        gc.collect()

    def generate(self, prompt: str, decoding: dict) -> Generation:
        import torch
        from .prompt_format import apply_hf_chat_template

        _reset_peak_memory(self.device)
        # Soft cap: avoid accidental OOM on long TF_Reason catalogs during smoke.
        max_input = int(decoding.get("max_input_tokens") or 8192)
        text_in = apply_hf_chat_template(self.tok, prompt, decoding)
        # Prefer keeping the END of the prompt (task instruction / "Links:") when
        # long TF_Reason catalogs exceed max_input_tokens.
        if getattr(self.tok, "truncation_side", None) is not None:
            self.tok.truncation_side = "left"
        inputs = self.tok(
            text_in, return_tensors="pt", truncation=True, max_length=max_input)
        if self.device == "cuda":
            inputs = {k: v.to("cuda") for k, v in inputs.items()}
        else:
            inputs = inputs.to(self.device)
        prompt_tokens = int(inputs["input_ids"].shape[1])
        t0 = time.time()
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=decoding.get("max_new_tokens", 1024),
                do_sample=False,                        # deterministic
                temperature=None, top_p=None, top_k=None,
                pad_token_id=self.tok.eos_token_id,
            )
        latency = time.time() - t0
        gen_ids = out[0][prompt_tokens:]
        text = self.tok.decode(gen_ids, skip_special_tokens=True)
        out_tokens = int(gen_ids.shape[0])
        peak_mb, method = _read_peak_memory(self.device)
        return Generation(
            text=text, prompt_tokens=prompt_tokens, output_tokens=out_tokens,
            total_latency_s=latency, ttft_s=None,       # not instrumented (mark N/M)
            tokens_per_s=(out_tokens / latency if latency and out_tokens else None),
            peak_memory_mb=peak_mb, peak_memory_method=method, mock=False,
        )


# --------------------------------------------------------------------------- #
# llama.cpp backend (GGUF K-quant / F16; CUDA or Metal)                        #
# --------------------------------------------------------------------------- #
class LlamaCppBackend:
    name = "llama_cpp"

    def __init__(
        self,
        model_id: str,
        precision: str,
        gguf_path: str,
        hf_name: Optional[str] = None,
        revision: Optional[str] = None,
        **kw,
    ):
        try:
            from llama_cpp import Llama
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                "llama-cpp-python not installed. On Apple Silicon install a Metal "
                "build: `CMAKE_ARGS=\"-DLLAMA_METAL=on\" pip install llama-cpp-python`."
            ) from e
        self.model_id = model_id
        self.precision = precision
        self.hf_name = hf_name
        self.revision = revision
        self.input_mode = "token_ids"
        self.gguf_path = gguf_path
        self.gguf_filename = gguf_path.replace("\\", "/").rsplit("/", 1)[-1]
        self.gguf_sha256 = None
        self._hf_tok = None
        self._llm = Llama(
            model_path=gguf_path,
            n_gpu_layers=kw.get("n_gpu_layers", -1),
            n_ctx=kw.get("n_ctx", 8192),
            n_batch=kw.get("n_batch", 512),
            logits_all=False, verbose=False,
        )

    def _tokenizer(self):
        if self._hf_tok is None and self.hf_name:
            from .prompt_format import load_hf_tokenizer
            self._hf_tok = load_hf_tokenizer(self.hf_name, self.revision)
        return self._hf_tok

    def close(self) -> None:
        import gc
        try:
            del self._llm
        except Exception:
            pass
        self._hf_tok = None
        gc.collect()

    def _prompt_token_ids(self, text_in: str) -> list[int]:
        """Ids actually passed to llama.cpp.

        The HF chat template already includes the BOS piece for Mistral, Llama 3,
        and Gemma. llama.cpp's string completion path prepends another BOS when
        GGUF metadata sets add_bos_token, so those models saw the token twice.
        Tokenizing the rendered string with the HF tokenizer (specials already
        in the text, add_special_tokens=False) keeps a single BOS and matches
        the transformers ids. Qwen does not add a second BOS; this is a no-op.
        """
        tok = self._tokenizer()
        if tok is not None:
            return list(tok(text_in, add_special_tokens=False)["input_ids"])
        ids = list(self._llm.tokenize(
            text_in.encode("utf-8"), add_bos=False, special=True))
        bos = self._llm.token_bos()
        try:
            add_bos = bool(self._llm._model.add_bos_token())
        except Exception:
            add_bos = False
        if add_bos and bos != -1 and (not ids or ids[0] != bos):
            ids = [bos] + ids
        return ids

    def generate(self, prompt: str, decoding: dict) -> Generation:
        from .prompt_format import apply_hf_chat_template, grammar_for_task_kind

        # Align with transformers: same HF chat template → completion on that text.
        text_in = apply_hf_chat_template(self._tokenizer(), prompt, decoding)
        prompt_ids = self._prompt_token_ids(text_in)

        grammar = None
        if decoding.get("use_grammar"):
            gstr = decoding.get("grammar_string") or grammar_for_task_kind(
                decoding.get("task_kind"))
            if gstr:
                from llama_cpp import LlamaGrammar
                grammar = LlamaGrammar.from_string(gstr)

        rss0 = _rss_mb()
        t0 = time.time()
        ttft = None
        pieces = []
        first = True
        finish_reason = None
        gen_kw = dict(
            max_tokens=decoding.get("max_new_tokens", 1024),
            temperature=decoding.get("temperature", 0.0),
            top_p=decoding.get("top_p", 1.0),
            top_k=decoding.get("top_k", 0) or 0,
            seed=decoding.get("seed", 0),
            stream=True,
        )
        if grammar is not None:
            gen_kw["grammar"] = grammar
        stream = self._llm(prompt_ids, **gen_kw)
        for chunk in stream:
            choice = chunk["choices"][0]
            if first:
                ttft = time.time() - t0
                first = False
            pieces.append(choice.get("text") or "")
            if choice.get("finish_reason"):
                finish_reason = choice["finish_reason"]
        latency = time.time() - t0
        text = "".join(pieces)
        try:
            output_tokens = len(self._llm.tokenize(text.encode("utf-8"), add_bos=False, special=True)) if text else 0
        except Exception:
            output_tokens = None
        prompt_tokens = len(prompt_ids)
        rss1 = _rss_mb()
        peak = (rss1 if rss1 is not None else None)
        return Generation(
            text=text, prompt_tokens=prompt_tokens, output_tokens=output_tokens,
            total_latency_s=latency, ttft_s=ttft,
            tokens_per_s=(output_tokens / latency if latency and output_tokens else None),
            peak_memory_mb=peak,
            peak_memory_method="process_rss_max (llama.cpp; unified or host RSS)",
            mock=False,
            finish_reason=finish_reason,
        )


# --------------------------------------------------------------------------- #
# Cloud frontier reference (upper bound only; never ground truth / Basis B)   #
# --------------------------------------------------------------------------- #
class CloudBackend:
    """OpenAI- or Anthropic-compatible chat completion for the cloud upper bound.

    Credentials come from env vars named in the experiment config
    (defaults: OPENAI_API_KEY / ANTHROPIC_API_KEY). Never hard-code keys.
    """

    name = "cloud"

    def __init__(
        self,
        model_id: str,
        precision: str,
        provider: str,
        api_model: str,
        key_env: str,
        base_url: Optional[str] = None,
        **kw,
    ):
        import os

        self.model_id = model_id
        self.precision = precision
        self.provider = provider.lower().strip()
        self.api_model = api_model
        key = os.environ.get(key_env)
        if not key:
            raise RuntimeError(
                f"Cloud backend requires env {key_env} (provider={self.provider}).")
        if self.provider == "anthropic":
            import anthropic
            self._client = anthropic.Anthropic(api_key=key)
        elif self.provider in ("openai", "openai_compatible"):
            from openai import OpenAI
            self._client = OpenAI(api_key=key, base_url=base_url or None)
        else:
            raise ValueError(
                f"Unknown cloud provider '{provider}'. Use anthropic | openai.")

    def close(self) -> None:
        return None

    def generate(self, prompt: str, decoding: dict) -> Generation:
        max_tokens = int(decoding.get("max_new_tokens", 1024))
        temperature = float(decoding.get("temperature", 0.0))
        t0 = time.time()
        if self.provider == "anthropic":
            kwargs = dict(
                model=self.api_model,
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}],
            )
            # Anthropic rejects temperature=0 on some models; 0 is still greedy.
            if temperature > 0:
                kwargs["temperature"] = temperature
            resp = self._client.messages.create(**kwargs)
            text = "".join(
                block.text for block in resp.content
                if getattr(block, "type", None) == "text"
            )
            prompt_tokens = getattr(resp.usage, "input_tokens", None)
            output_tokens = getattr(resp.usage, "output_tokens", None)
        else:
            resp = self._client.chat.completions.create(
                model=self.api_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature,
            )
            text = resp.choices[0].message.content or ""
            usage = getattr(resp, "usage", None)
            prompt_tokens = getattr(usage, "prompt_tokens", None) if usage else None
            output_tokens = getattr(usage, "completion_tokens", None) if usage else None
        latency = time.time() - t0
        return Generation(
            text=text,
            prompt_tokens=prompt_tokens,
            output_tokens=output_tokens,
            total_latency_s=latency,
            ttft_s=None,
            tokens_per_s=(output_tokens / latency if latency and output_tokens else None),
            peak_memory_mb=None,
            peak_memory_method=None,
            mock=False,
        )


# --------------------------------------------------------------------------- #
def build_backend(kind: str, model_id: str, precision: str, **kw) -> Backend:
    if kind == "mock":
        return MockBackend(model_id, precision, **kw)
    if kind == "transformers":
        return TransformersBackend(model_id, precision, **kw)
    if kind == "llama_cpp":
        return LlamaCppBackend(model_id, precision, **kw)
    if kind == "cloud":
        return CloudBackend(model_id, precision, **kw)
    raise ValueError(
        f"Unknown backend '{kind}'. Use mock | transformers | llama_cpp | cloud.")
