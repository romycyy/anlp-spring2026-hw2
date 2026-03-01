"""LLM reader: generates answers from retrieved context chunks."""

from __future__ import annotations

from typing import Optional

import torch
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    GenerationConfig,
)

_DEFAULT_MODEL = "Qwen/Qwen2.5-14B-Instruct"

# Character-based context budget (mirrors reference repo's MAX_CONTEXT_CHARS).
MAX_CONTEXT_CHARS: int = 10_000

# Prompt template from reference repo: explicit reasoning steps.
PROMPT_TEMPLATE = """\
Your task:
1. Read the question and the retrieved context carefully.
2. If the retrieved context is missing or irrelevant, answer based on your external knowledge.
3. Answer clearly and concisely in English.

Question:
{question}

Retrieved Context:
{context}

Answer:
Provide the final answer below. Do not show your reasoning steps. Answer in english.
"""

# Prompt for evaluating a single chunk's relevance and extracting a partial answer.
CHUNK_RELEVANCE_TEMPLATE = """\
Your task:
1. Read the question and the single retrieved chunk carefully.
2. Decide whether the chunk contains information relevant to answering the question.
3. If relevant, provide a concise partial answer based solely on this chunk.
4. If not relevant, respond with exactly: NOT RELEVANT

Question:
{question}

Retrieved Chunk ({chunk_idx}/{total_chunks}):
{chunk_text}

Response:
If this chunk is relevant, provide a partial answer. Otherwise, write "NOT RELEVANT".
"""

# Prompt for synthesizing a final answer from all per-chunk responses.
SYNTHESIS_TEMPLATE = """\
Your task:
1. Read the question and the partial responses collected from evaluating each retrieved chunk.
2. Synthesize a single, concise final answer in English based on the relevant responses.
3. If none of the chunks were relevant, answer based on your external knowledge.
4. Do not show your reasoning steps.

Question:
{question}

Per-chunk responses:
{chunk_responses}

Final Answer:
Provide the final answer below. Answer in English.
"""

_tokenizer = None
_qa_model = None
_loaded_model_name: Optional[str] = None
_loaded_is_encoder_decoder: Optional[bool] = None


def _get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _get_dtype() -> torch.dtype:
    """Return the best supported half-precision dtype for the current device."""
    if torch.cuda.is_available():
        return torch.bfloat16
    if torch.backends.mps.is_available():
        return torch.float16  # MPS does not support bfloat16
    return torch.float32


def _safe_tokenizer_max_length(tokenizer, *, fallback: int = 512) -> int:
    max_len = getattr(tokenizer, "model_max_length", None)
    try:
        max_len_int = int(max_len)
    except Exception:
        return fallback
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

    dtype = _get_dtype()
    device = _get_device()
    # device_map="auto" is incompatible with MPS; load to CPU then move manually.
    use_device_map = torch.cuda.is_available()

    if is_encoder_decoder:
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name, torch_dtype=dtype)
    else:
        kwargs = {"torch_dtype": dtype}
        if use_device_map:
            kwargs["device_map"] = "auto"
        model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
        if getattr(tokenizer, "pad_token_id", None) is None:
            if getattr(tokenizer, "eos_token", None) is not None:
                tokenizer.pad_token = tokenizer.eos_token

    if not use_device_map:
        model.to(device)
    model.eval()

    _tokenizer = tokenizer
    _qa_model = model
    _loaded_model_name = model_name
    _loaded_is_encoder_decoder = is_encoder_decoder
    return tokenizer, model, is_encoder_decoder


def _build_context(
    retrieved_chunks: list, *, max_chars: int = MAX_CONTEXT_CHARS
) -> str:
    """
    Assemble context from retrieved results.
    Accepts either list[str] or list[dict] (structured results from retrievers).
    Sorts by relevance score descending when available so the best chunk is first.
    Respects max_chars budget.
    """
    # Sort by score descending so most relevant chunks come first when truncating
    chunks = list(retrieved_chunks)
    if chunks and isinstance(chunks[0], dict) and "score" in chunks[0]:
        chunks = sorted(chunks, key=lambda x: x.get("score", 0.0), reverse=True)

    ctxs: list[str] = []
    total_len = 0
    for item in chunks:
        text = item["text"].strip() if isinstance(item, dict) else str(item).strip()
        if not text:
            continue
        if total_len + len(text) > max_chars:
            break
        ctxs.append(text)
        total_len += len(text)
    return "\n\n".join(ctxs)


def _split_thinking(content: str) -> tuple[str, str]:
    """
    DeepSeek-R1 models emit a <think>...</think> block before the answer.
    Split into (thinking, answer). If no thinking block, return ("", content).
    """
    if "<think>" in content and "</think>" in content:
        end = content.index("</think>") + len("</think>")
        thinking = content[:end].strip()
        answer = content[end:].strip()
        return thinking, answer
    # Fallback: some models use blank-line separation
    if "\n \n\n" in content:
        parts = content.split("\n \n\n", 1)
        return parts[0].strip(), parts[1].strip()
    return "", content.strip()


def _generate_from_prompt(
    prompt_body: str,
    tokenizer,
    qa_model,
    is_encoder_decoder: bool,
    *,
    max_new_tokens: int = 256,
) -> str:
    """Run inference on a single prompt_body string and return the decoded answer."""
    if is_encoder_decoder:
        inputs = tokenizer(
            prompt_body,
            return_tensors="pt",
            truncation=True,
            max_length=_safe_tokenizer_max_length(tokenizer, fallback=512),
        ).to(qa_model.device)
    else:
        if hasattr(tokenizer, "apply_chat_template") and getattr(
            tokenizer, "chat_template", None
        ):
            user_content = (
                "You are an expert assistant with access to external retrieved documents.\n\n"
                + prompt_body
            )
            text = tokenizer.apply_chat_template(
                [{"role": "user", "content": user_content}],
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            text = prompt_body

        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=_safe_tokenizer_max_length(tokenizer, fallback=2048),
        ).to(qa_model.device)

    num_beams = 4 if is_encoder_decoder else 1
    gen_cfg = GenerationConfig(
        max_new_tokens=max_new_tokens,
        do_sample=False,
        num_beams=num_beams,
        repetition_penalty=1.2,
        no_repeat_ngram_size=3,
        temperature=1.0,
        top_p=1.0,
    )

    with torch.no_grad():
        outputs = qa_model.generate(**inputs, generation_config=gen_cfg)

    if is_encoder_decoder:
        raw = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    else:
        input_len = int(inputs["input_ids"].shape[-1])
        gen_ids = outputs[0][input_len:]
        raw = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

    _thinking, answer = _split_thinking(raw)
    return answer


def answer_question(
    question: str,
    context_chunks: list,
    *,
    model_name: str = _DEFAULT_MODEL,
    max_new_tokens: int = 256,
    max_context_chars: int = MAX_CONTEXT_CHARS,
) -> str:
    """
    Generate an answer for *question* given *context_chunks*.

    context_chunks can be list[str] or list[dict] (from structured retrievers).
    Returns only the final answer string (thinking stripped for reasoning models).
    """
    tokenizer, qa_model, is_encoder_decoder = _load(model_name)
    context = _build_context(context_chunks, max_chars=max_context_chars)
    prompt_body = PROMPT_TEMPLATE.format(question=question.strip(), context=context)
    return _generate_from_prompt(
        prompt_body,
        tokenizer,
        qa_model,
        is_encoder_decoder,
        max_new_tokens=max_new_tokens,
    )


def answer_question_iterative(
    question: str,
    context_chunks: list,
    *,
    model_name: str = _DEFAULT_MODEL,
    max_new_tokens: int = 256,
) -> str:
    """
    Iterative chunk-by-chunk RAG reader.

    For each retrieved chunk, the LLM is asked independently whether the chunk
    is relevant to the question.  If relevant, it provides a partial answer;
    otherwise it responds "NOT RELEVANT".  After all chunks are processed, a
    final synthesis call consolidates the partial answers into one response.

    context_chunks can be list[str] or list[dict] (from structured retrievers).
    Returns only the final answer string (thinking stripped for reasoning models).
    """
    tokenizer, qa_model, is_encoder_decoder = _load(model_name)

    # Normalise chunks to plain strings, best-first when scores available.
    chunks = list(context_chunks)
    if chunks and isinstance(chunks[0], dict) and "score" in chunks[0]:
        chunks = sorted(chunks, key=lambda x: x.get("score", 0.0), reverse=True)

    chunk_texts: list[str] = []
    for item in chunks:
        text = item["text"].strip() if isinstance(item, dict) else str(item).strip()
        if text:
            chunk_texts.append(text)

    total = len(chunk_texts)
    per_chunk_responses: list[str] = []

    for idx, chunk_text in enumerate(chunk_texts, start=1):
        prompt_body = CHUNK_RELEVANCE_TEMPLATE.format(
            question=question.strip(),
            chunk_idx=idx,
            total_chunks=total,
            chunk_text=chunk_text,
        )
        response = _generate_from_prompt(
            prompt_body,
            tokenizer,
            qa_model,
            is_encoder_decoder,
            max_new_tokens=max_new_tokens,
        )
        print(f"  [Chunk {idx}/{total}] {response[:120].rstrip()}{'...' if len(response) > 120 else ''}")
        per_chunk_responses.append(f"Chunk {idx}: {response}")

    # Synthesise a final answer from all per-chunk responses.
    chunk_responses_text = "\n\n".join(per_chunk_responses)
    synthesis_prompt = SYNTHESIS_TEMPLATE.format(
        question=question.strip(),
        chunk_responses=chunk_responses_text,
    )
    final_answer = _generate_from_prompt(
        synthesis_prompt,
        tokenizer,
        qa_model,
        is_encoder_decoder,
        max_new_tokens=max_new_tokens,
    )
    return final_answer
