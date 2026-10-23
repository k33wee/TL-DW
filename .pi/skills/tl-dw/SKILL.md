---
name: tl-dw
description: Turn a local meeting video into a timestamped transcript, evidence-based summary or operational plan, with optional context documents and diagrams. Use for transcription, call analysis, decisions, action items or planning from a recording or an existing TL-DW bundle.
compatibility: Requires uv, ffmpeg/ffprobe, the TL-DW local Pi package, and an authenticated image-capable Pi model.
---

# TL-DW video summary

Use `tl_dw_video` for a recording and `tl_dw_analyze` for an existing `analysis.json` or artifact directory. Local preparation probes media, checks disk space, selects audio conservatively, transcribes and extracts frames/OCR. Explicit document attachments are converted locally with embedded raster figures. Model analysis sends bounded evidence to the selected provider.

## Procedure

1. Reuse an existing prepared bundle when one belongs to the task: call `tl_dw_analyze`, not `tl_dw_video`, to avoid repeating ASR/OCR. Otherwise identify one local recording; a directory may be passed as `video` only when it contains exactly one supported video. If ambiguous, ask rather than choosing silently.
2. For a new recording call `tl_dw_video` once. If cancelled, resume only when the user asks to continue; verified completed preparation and model checkpoints are reused. Preserve quality defaults unless the user requests otherwise:
   - omit `whisper_language` to auto-detect, or pass an explicit code such as `it` or `en`;
   - omit `hardware_profile` to auto-select;
   - keep frames and OCR enabled;
   - omit `model` to use the active model when it supports images;
   - default artifacts belong in `./tl-dw-output`; pass `output_directory` only when the user asks for another location;
   - set `force` only when the user explicitly asks to discard reusable Pi analysis checkpoints; `force_preparation` separately rebuilds local ASR/frame stages and also needs explicit intent;
   - pass `attachments` only for context files/directories explicitly supplied for the task. Do not discover unrelated documents. Converted text and figures are sent to the model, and prepared context is reused by later analyses of that bundle;
   - for a requested planning deliverable set `report_mode: "planning"` and optionally `planning_request`. Do not manually reconstruct a plan from scratch if the tool can produce it from existing evidence;
   - pass user-requested call/image ceilings as `max_model_calls` / `max_images`. These fail before spending rather than dropping evidence; they are not currency guarantees;
   - do not compress or manually probe/merge audio by default: preparation does those checks. If distinct tracks are ambiguous, ask the user to select an absolute FFprobe stream (`audio_stream`) or approve a mix (`audio_mix`). Quiet samples do not prove full-track silence.
3. Treat the returned Markdown report as the authoritative summary. Give the user a concise completion note with the report and artifact paths; do not redo or dilute the report unless they ask for a shorter summary or a specific extraction.
4. If the tool result says its report was truncated, read the complete report from the returned path before answering detailed questions.

## Privacy and cost

Only run analysis when the user has asked to process the recording/bundle. Local preparation can take substantial CPU, RAM and disk. Model analysis sends transcript excerpts, selected frames and any explicit prepared context/diagrams to the configured provider and may incur cost. Document conversion is local and never opts into hosted OCR. Scanned PDFs fail explicitly; PDF vector/embedded diagrams require explicit raster attachments. Files remain local artifacts, not material to commit.

For prepared input call `tl_dw_analyze` directly, with `input` set to the bundle path and the same planning/context options as needed. The manual alternative is `/tl-dw-analyze <bundle> --report-mode planning --attachment <file>`. Summary and planning share meeting checkpoints; context changes only redo context analysis and synthesis. Do not force a replay merely to change the deliverable.

Planning must distinguish meeting agreements, attachment-only requirements, proposed estimates, unassigned/parked work and unresolved decisions. It is not proof of live implementation, deployment, approval or operational authorization. If a generic repeatable gap still needs manual work, report the gap for a tested project improvement rather than treating a one-off script as an implemented capability.
