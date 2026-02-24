from __future__ import annotations

from typing import Optional

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer

_DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
_tokenizer = None
_qa_model = None
_loaded_model_name: Optional[str] = None
_loaded_is_encoder_decoder: Optional[bool] = None


def _get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _safe_tokenizer_max_length(tokenizer, *, fallback: int = 512) -> int:
    """
    Some HF tokenizers expose `model_max_length` as a very large sentinel (e.g. 1e30)
    to mean "unbounded/unknown". Fast tokenizers pass this into a Rust backend that
    expects a 32-bit length, which can overflow.
    """
    max_len = getattr(tokenizer, "model_max_length", None)
    try:
        max_len_int = int(max_len)
    except Exception:
        return fallback

    # Treat unbounded/unknown sentinels as "use a reasonable cap".
    if max_len_int <= 0 or max_len_int > 1_000_000:
        return fallback
    return max_len_int


def _load(model_name: str = _DEFAULT_MODEL):
    global _tokenizer, _qa_model, _loaded_model_name, _loaded_is_encoder_decoder
    if (
        _qa_model is not None
        and _tokenizer is not None
        and _loaded_model_name == model_name
        and _loaded_is_encoder_decoder is not None
    ):
        return _tokenizer, _qa_model, _loaded_is_encoder_decoder

    cfg = AutoConfig.from_pretrained(model_name)
    is_encoder_decoder = bool(getattr(cfg, "is_encoder_decoder", False))

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if is_encoder_decoder:
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name)

        # Some causal LMs don't define a pad token; make generation inputs well-formed.
        if getattr(tokenizer, "pad_token_id", None) is None:
            if getattr(tokenizer, "eos_token", None) is not None:
                tokenizer.pad_token = tokenizer.eos_token

    model.to(_get_device())
    model.eval()

    _tokenizer = tokenizer
    _qa_model = model
    _loaded_model_name = model_name
    _loaded_is_encoder_decoder = is_encoder_decoder
    return tokenizer, model, is_encoder_decoder


def answer_question(
    question: str,
    context_chunks: list[str],
    *,
    model_name: str = _DEFAULT_MODEL,
    max_new_tokens: int = 128,
) -> str:
    tokenizer, qa_model, is_encoder_decoder = _load(model_name)

    # Instruction-style prompt tends to reduce repetition vs raw concat.
    context = "\n\n".join(c.strip() for c in context_chunks if c and c.strip())
    instruction = (
        "Answer the question using only the context. "
        "If the answer is not in the context, say \"I don't know\"."
    )

    if is_encoder_decoder:
        prompt = (
            f"{instruction}\n\n"
            f"Question: {question.strip()}\n\n"
            f"Context:\n{context}\n\n"
            "Answer:"
        )
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=_safe_tokenizer_max_length(tokenizer, fallback=512),
        ).to(qa_model.device)
    else:
        user_content = (
            f"{instruction}\n\n"
            f"Question: {question.strip()}\n\n"
            f"Context:\n{context}"
        )

        if hasattr(tokenizer, "apply_chat_template") and getattr(tokenizer, "chat_template", None):
            text = tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": user_content},
                ],
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            text = f"{user_content}\n\nAnswer:"

        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=_safe_tokenizer_max_length(tokenizer, fallback=2048),
        ).to(qa_model.device)

    with torch.no_grad():
        outputs = qa_model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=4 if is_encoder_decoder else 1,
            repetition_penalty=1.2,
            no_repeat_ngram_size=3,
        )

    if is_encoder_decoder:
        return tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

    input_len = int(inputs["input_ids"].shape[-1])
    gen_ids = outputs[0][input_len:]
    return tokenizer.decode(gen_ids, skip_special_tokens=True).strip()