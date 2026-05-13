# TL-DW (Too Long; Didn't Watch)

Local video to readable Markdown extraction with speech transcription plus OCR-derived visual context.

## Main entrypoint

Primary command:

- `uv run -m main --video "path-to-video/video.mp4" `

Example with GPU, Italian, timestamps, OCR, and heuristic chapters:

- `uv run -m main --video "path-to-video/video.mp4" --device gpu --whisper-language it --segment-unchaptered --timestamp-paragraphs --add-table-of-contents`

If you want help:

- `uv run python main.py --help`

## What it does

- extracts audio from a local video
- transcribes speech with `faster-whisper`
- samples frames and extracts meaningful on-screen text with OCR
- merges spoken and visual context into a readable Markdown document
- writes structured artifacts to `output/<video-slug>/`

## Requirements

- Python 3.13+
- `ffmpeg` and `ffprobe` available on your system path
- optional NVIDIA CUDA support for GPU transcription

## Install

Using `uv`:

- `uv sync`

## Basic usage

Process a single video:

- `uv run -m main --video "path-to-video/video.mp4"`

Process every supported file in `videos/` directory:

- `uv run -m main`

Write to a specific file:

- `uv run -m main --video "path-to-video/video.mp4" -o output/my-doc.md`

## Important arguments

Input/output:

- `--video <path>` — process one local file
- `--video-dir <path>` — directory scan fallback, defaults to `videos/`
- `-o, --output <path>` — output directory, or a `.md` file for a single input

Transcription:

- `--device <cpu|gpu>` — high-level runtime choice
- `--whisper-model <name>` — ASR model to load, default `small`
- `--whisper-language <code>` — language hint like `it` or `en`
- `--initial-prompt <text>` — optional prompt override

Low-level Whisper overrides:

- `--whisper-device <cpu|cuda>`
- `--whisper-compute-type <type>`

Segmentation:

- `--sat-model <name>` — SaT segmentation model, default `sat-3l-sm`
- `--segment-unchaptered` — create heuristic chapters when the source has none
- `--ignore-source-chapters` — ignore embedded chapters completely
- `--max-generated-chapter-seconds <float>`
- `--min-generated-chapter-gap <float>`

Markdown formatting:

- `--timestamp-paragraphs`
- `--add-table-of-contents`

OCR / visual context:

- `--no-ocr` — disable OCR extraction
- `--ocr-sample-sec <float>` — frame sampling interval
- `--ocr-min-score <float>` — OCR confidence threshold
- `--ocr-min-chars <int>` — minimum OCR text length for generic lines
- `--ocr-max-lines <int>` — max lines kept per note
- `--ocr-max-note-chars <int>` — max rendered note length
- `--ocr-dedupe-window-sec <float>` — suppress repeated nearby OCR notes

Other:

- `--verbose`

## Model selection

You can already choose which transcription model to load.

Examples:

- faster startup:
  - `--whisper-model tiny`
- balanced default:
  - `--whisper-model small`
- stronger ASR quality:
  - `--whisper-model medium`
- strongest current Whisper option in this project:
  - `--whisper-model large-v3`

Example:

- `uv run python main.py --video "path-to-video/video.mp4" --device gpu --whisper-model large-v3`

## GPU notes

- `--device gpu` maps to `cuda`
- default GPU compute type is `int8_float16`
- if your GPU does not support that efficiently, the loader retries more compatible CUDA compute types automatically
- this is useful for older GPUs, including Pascal-era setups

## SaT fallback behavior

If `wtpsplit` cannot load its full SaT runtime, the app falls back to a built-in paragraph segmenter so the pipeline still runs.

## Output structure

By default, outputs are written to:

- `output/<video-slug>/`

Typical files:

- `metadata.json` — ffprobe metadata
- `audio.wav` — extracted mono 16k audio
- `transcript.txt` — readable transcript segments
- `transcript.jsonl` — structured transcript segments
- `transcript_info.json` — transcription settings and language info
- `ocr.jsonl` — OCR observations per sampled frame
- `visual_notes.json` — deduplicated OCR notes
- `chapters.json` — final chapter/paragraph structure
- `document.md` — rendered Markdown document

## Testing

Run the test suite with:

- `uv run pytest`

## Current limitations

- OCR visual-note extraction still needs better relevance filtering for noisy desktop recordings
- SaT may fall back to the built-in segmenter if its runtime dependencies are unavailable
- the document pipeline is already usable, but visual-context prioritization can still be improved for meeting-heavy recordings
