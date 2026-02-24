from typing import List

def chunk_text(text: str, chunk_size:int=500, overlap:int=50) -> List[str]:
    """Break text into chunks with overlap."""
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be > 0 (got {chunk_size})")
    if overlap < 0:
        raise ValueError(f"overlap must be >= 0 (got {overlap})")
    if overlap >= chunk_size:
        raise ValueError(
            f"overlap must be < chunk_size (got overlap={overlap}, chunk_size={chunk_size})"
        )
    tokens = text.split()
    chunks = []
    start = 0
    while start < len(tokens):
        end = start + chunk_size
        chunk = tokens[start:end]
        chunks.append(" ".join(chunk))
        start = end - overlap
    return chunks