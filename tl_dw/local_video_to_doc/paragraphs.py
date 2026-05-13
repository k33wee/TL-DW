from __future__ import annotations

from typing import Any, Iterable, Sequence

from .models import Paragraph, TranscriptSegment


def normalize_segmented_paragraphs(raw_paragraphs: Iterable[Any]) -> list[list[str]]:
    normalized: list[list[str]] = []
    for paragraph in raw_paragraphs:
        if isinstance(paragraph, str):
            sentences = [paragraph]
        else:
            sentences = [
                sentence for sentence in paragraph if isinstance(sentence, str)
            ]
        if sentences:
            normalized.append(sentences)
    return normalized


def align_paragraphs_to_timestamps(
    transcription_segments: Sequence[TranscriptSegment],
    raw_paragraphs: Sequence[Sequence[str] | str],
) -> list[Paragraph]:
    if not transcription_segments:
        return []

    normalized_paragraphs = normalize_segmented_paragraphs(raw_paragraphs)
    if not normalized_paragraphs:
        return []

    segments_text = "".join(segment.text for segment in transcription_segments)
    segments_pos = 0
    current_segment_index = 0
    current_segment_offset = 0
    result: list[Paragraph] = []

    for paragraph in normalized_paragraphs:
        start_index: int | None = None
        paragraph_text_parts: list[str] = []
        for sentence in paragraph:
            if not sentence:
                continue

            if start_index is None:
                start_index = current_segment_index

            sentence_pos = 0
            while sentence_pos < len(sentence) and segments_pos < len(segments_text):
                if sentence[sentence_pos] == segments_text[segments_pos]:
                    sentence_pos += 1
                segments_pos += 1
                current_segment_offset += 1
                while current_segment_index < len(
                    transcription_segments
                ) - 1 and current_segment_offset >= len(
                    transcription_segments[current_segment_index].text
                ):
                    current_segment_index += 1
                    current_segment_offset = 0

            paragraph_text_parts.append(sentence)

        paragraph_text = "".join(paragraph_text_parts).strip()
        if paragraph_text and start_index is not None:
            result.append(
                Paragraph(
                    start=transcription_segments[start_index].start,
                    text=paragraph_text,
                )
            )

    return result


def segment_paragraphs(
    transcription_segments: Sequence[TranscriptSegment], sat_model: Any
) -> list[Paragraph]:
    if not transcription_segments:
        return []

    full_text = "".join(segment.text for segment in transcription_segments)
    if not full_text.strip():
        return []

    raw_paragraphs = sat_model.split(full_text, do_paragraph_segmentation=True)
    return align_paragraphs_to_timestamps(transcription_segments, raw_paragraphs)
