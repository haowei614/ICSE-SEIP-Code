"""Shared prompt rendering so transformers and llama.cpp see the same text.

When ``use_chat_template`` is true, both backends must wrap the task prompt with
the Hugging Face tokenizer's ``apply_chat_template`` (same messages, same
kwargs). llama.cpp then runs completion on that rendered string — not on the
raw task prompt, and not on a divergent GGUF-embedded chat wrapper.
"""
from __future__ import annotations

from typing import Any, Optional


def apply_hf_chat_template(
    tokenizer: Any,
    prompt: str,
    decoding: Optional[dict] = None,
) -> str:
    """Return the string that should be tokenized / completed.

    If chat templating is disabled or unavailable, returns ``prompt`` unchanged.
    """
    decoding = decoding or {}
    if not decoding.get("use_chat_template", True):
        return prompt
    if tokenizer is None or not hasattr(tokenizer, "apply_chat_template"):
        return prompt
    enable_thinking = bool(decoding.get("enable_thinking", False))
    messages = [{"role": "user", "content": prompt}]
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        return prompt


def load_hf_tokenizer(hf_name: str, revision: Optional[str] = None):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(
        hf_name, revision=revision, trust_remote_code=True)


# Minimal GBNF grammars for structured RE outputs (llama.cpp native).
# Kept deliberately loose so labels/filenames remain free strings.
GRAMMAR_TRACE_LINK = r'''
root   ::= ws "[" ws (string (ws "," ws string)*)? ws "]" ws
string ::= "\"" chars "\""
chars  ::= char*
char   ::= [^"\\] | "\\" ["\\/bfnrt]
ws     ::= [ \t\n]*
'''

GRAMMAR_SPAN_LIST = r'''
root   ::= ws "[" ws (obj (ws "," ws obj)*)? ws "]" ws
obj    ::= "{" ws "\"start\"" ws ":" ws number ws "," ws "\"end\"" ws ":" ws number ws "," ws "\"label\"" ws ":" ws string ws "}"
number ::= "-"? [0-9]+
string ::= "\"" chars "\""
chars  ::= char*
char   ::= [^"\\] | "\\" ["\\/bfnrt]
ws     ::= [ \t\n]*
'''

GRAMMAR_BY_TASK_KIND = {
    "trace_link": GRAMMAR_TRACE_LINK,
    "span_detection": GRAMMAR_SPAN_LIST,
}


def grammar_for_task_kind(task_kind: Optional[str]) -> Optional[str]:
    if not task_kind:
        return None
    return GRAMMAR_BY_TASK_KIND.get(task_kind)
