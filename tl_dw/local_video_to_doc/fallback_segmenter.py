from __future__ import annotations

import re


class SimpleParagraphSegmenter:
    """Fallback segmenter used when SaT cannot be loaded."""

    def __init__(
        self, max_sentences_per_paragraph: int = 4, max_chars_per_paragraph: int = 700
    ) -> None:
        self.max_sentences_per_paragraph = max_sentences_per_paragraph
        self.max_chars_per_paragraph = max_chars_per_paragraph

    def split(self, text: str, do_paragraph_segmentation: bool = True):
        sentences = self._split_sentences(text)
        if not do_paragraph_segmentation:
            return sentences
        return self._group_sentences(sentences)

    def _split_sentences(self, text: str) -> list[str]:
        normalized = re.sub(r"\s+", " ", text).strip()
        if not normalized:
            return []

        parts = re.split(r"(?<=[.!?])\s+", normalized)
        sentences = [part.strip() for part in parts if part.strip()]
        if sentences:
            return [f"{sentence} " for sentence in sentences]
        return [f"{normalized} "]

    def _group_sentences(self, sentences: list[str]) -> list[list[str]]:
        paragraphs: list[list[str]] = []
        current: list[str] = []
        current_length = 0

        for sentence in sentences:
            sentence_length = len(sentence)
            too_many_sentences = len(current) >= self.max_sentences_per_paragraph
            too_many_chars = (
                current
                and current_length + sentence_length > self.max_chars_per_paragraph
            )
            if too_many_sentences or too_many_chars:
                paragraphs.append(current)
                current = []
                current_length = 0

            current.append(sentence)
            current_length += sentence_length

        if current:
            paragraphs.append(current)
        return paragraphs
