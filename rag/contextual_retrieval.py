"""Context enrichment for chunks before they enter the retrieval indexes."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any


Contextualizer = Callable[[str, str], str]


def select_document_context(
    source_text: str,
    chunks: Sequence[str],
    chunk_index: int,
    *,
    max_source_chars: int,
    start_chunks: int,
    previous_chunks: int,
) -> str:
    """Use the source document when it fits; otherwise use a bounded context window."""
    if len(source_text) <= max_source_chars:
        return source_text

    if not 0 <= chunk_index < len(chunks):
        raise IndexError(f"chunk_index out of range: {chunk_index}")

    selected_indices = list(range(min(start_chunks, len(chunks))))
    selected_indices.extend(range(max(0, chunk_index - previous_chunks), chunk_index))

    unique_indices = dict.fromkeys(selected_indices)
    return "\n\n".join(chunks[index] for index in unique_indices if chunks[index].strip())


def build_context_prompt(chunk: str, document_context: str) -> str:
    """Build the prompt used by Anthropic's contextual-retrieval pattern."""
    return f"""Write a short and succinct snippet of text to situate this chunk within the overall source document for the purposes of improving search retrieval of the chunk.

Here is the source-document context:
<document>
{document_context}
</document>

Here is the chunk we want to situate within the whole document:
<chunk>
{chunk}
</chunk>

Answer only with the succinct context and nothing else."""


class AnthropicContextualizer:
    """Call Claude to add document context to one chunk at a time."""

    def __init__(
        self,
        *,
        model: str,
        max_tokens: int,
        timeout: float,
        client: Any | None = None,
    ) -> None:
        self.model = model
        self.max_tokens = max_tokens
        self.client = client if client is not None else self._build_client(timeout)

    @staticmethod
    def _build_client(timeout: float) -> Any:
        try:
            from anthropic import Anthropic
        except ImportError as error:
            raise RuntimeError(
                "Contextual retrieval needs the optional Anthropic dependency. "
                "Install `ragmcp[anthropic]` and set ANTHROPIC_API_KEY."
            ) from error
        return Anthropic(timeout=timeout)

    def __call__(self, chunk: str, document_context: str) -> str:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=0,
            messages=[
                {
                    "role": "user",
                    "content": build_context_prompt(chunk, document_context),
                }
            ],
        )
        text_parts = [
            block.text
            for block in response.content
            if getattr(block, "text", None)
        ]
        if not text_parts:
            raise RuntimeError("Anthropic returned no contextual text")
        return "\n".join(text_parts).strip()
