import { afterEach, expect, test } from "bun:test";
import { mkdtemp, mkdir, readFile, rm, symlink, writeFile } from "node:fs/promises";
import { unlinkSync } from "node:fs";
import { createHash } from "node:crypto";
import { tmpdir } from "node:os";
import * as path from "node:path";
import type { Api, Model, Usage } from "@earendil-works/pi-ai";
import type { ExtensionContext } from "@earendil-works/pi-coding-agent";
import tlDwExtension, {
	analyzeBundle,
	enforceAnalysisLimits,
	loadDocumentContext,
	parseArguments,
	retainAllowedScreenshots,
} from "../.pi/extensions/tl-dw";

const roots: string[] = [];
afterEach(async () => { for (const root of roots.splice(0)) await rm(root, { recursive: true, force: true }); });
const model = { provider: "fixture", id: "vision", name: "Synthetic test model", input: ["text", "image"],
	contextWindow: 64000, maxTokens: 4000, reasoning: false, inputLimits: { images: { maxPerMessage: 1, maxPerRequest: 1 } } } as Model<Api>;
const usage: Usage = { input: 10, output: 5, totalTokens: 15, cacheRead: 0, cacheWrite: 0,
	cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 } };
async function fixture() {
	const root = await mkdtemp(path.join(tmpdir(), "tl-dw-ts-"));
	roots.push(root);
	await writeFile(path.join(root, "transcript.txt"), "[00:00:00] A proposed feature, not assigned.\n");
	const bundle = { schema_version: 1, kind: "tl-dw-meeting-analysis", title: "Synthetic meeting",
		source: { path: "synthetic.mp4", duration: 20 }, sections: [1, 2].map((index) => ({
			id: `section-00${index}`, index, start: (index - 1) * 10, end: index * 10,
			transcript: [], transcript_text: "A proposed feature, not an assignment.", frames: [],
		})) };
	await writeFile(path.join(root, "analysis.json"), JSON.stringify(bundle));
	const calls: Array<{ systemPrompt: string; messages: unknown }> = [];
	let afterCall = () => {};
	let statusHook = (_status: string | undefined) => {};
	const ctx = { ui: { setStatus: (_key: string, status?: string) => statusHook(status) }, thinkingLevel: "off", modelRegistry: {
		streamSimple: (_model: unknown, request: { systemPrompt: string; messages: unknown }) => {
			calls.push(request);
			return { result: async () => {
				afterCall();
				const text = request.systemPrompt.includes("exactly two XML")
					? "<section_notes>Proposed feature at 00:00:01. No assigned owner.</section_notes><meeting_state>Open: owner.</meeting_state>"
					: request.systemPrompt.startsWith("Analyze this explicit")
						? "Attachment-only requirement, status pending."
						: "<report_title>Synthetic plan</report_title>\n# Synthetic plan\n\nOwner: Unassigned. Estimate is a proposal, not a meeting commitment.";
				return { stopReason: "stop", content: [{ type: "text", text }], usage };
			} };
		},
	} } as unknown as ExtensionContext;
	const run = (workflow = {}, force = false, signal = new AbortController().signal) => analyzeBundle({
		ctx, model, bundle, artifactDir: root, signature: "meeting-signature", force, workflow, signal,
	});
	return { root, bundle, calls, run, setAfterCall: (fn: () => void) => { afterCall = fn; },
		setStatusHook: (fn: (status: string | undefined) => void) => { statusHook = fn; } };
}
async function addDocuments(root: string, figures = 1) {
	await mkdir(path.join(root, "documents"), { recursive: true });
	await writeFile(path.join(root, "documents/spec.md"), "# Specification\nA separate program in stand-by. Not an assignment.");
	for (let i = 0; i < figures; i++) await writeFile(path.join(root, `documents/figure-${i}.png`), Buffer.from("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+a2ioAAAAASUVORK5CYII=", "base64"));
	const analysisSha = createHash("sha256").update(await readFile(path.join(root, "analysis.json"))).digest("hex");
	await writeFile(path.join(root, "documents.json"), JSON.stringify({ schema_version: 1, analysis_sha256: analysisSha, documents: [{
		id: "D1", name: "Synthetic specification", markdown_path: "documents/spec.md",
		figures: Array.from({ length: figures }, (_, i) => ({ image_path: `documents/figure-${i}.png` })),
		warnings: ["Document requirement only; not approved in meeting."],
	}] }));
}

test("report images are limited to declared local artifacts, including titled/reference syntax", () => {
	const input = '![valid](frames/a.jpg "caption") ![outside](https://example.test/leak.png "title") ![ref][remote] ![shortcut] ![encoded](frames/%2e%2e/leak.png)';
	const result = retainAllowedScreenshots(input, ["frames/a.jpg", "frames/%2e%2e/leak.png"]);
	expect(result).toContain("![valid](frames/a.jpg)");
	expect(result).not.toContain("example.test");
	expect(result).not.toContain("%2e");
	expect(result).not.toContain("[ref]");
	expect(result).not.toContain("[shortcut]");
});

test("registers both complete-video and prepared-bundle tools", () => {
	const tools: string[] = [];
	tlDwExtension({ registerTool: (tool: { name: string }) => tools.push(tool.name), registerCommand: () => {} } as never);
	expect(tools).toEqual(["tl_dw_video", "tl_dw_analyze"]);
});

test("parses attachments, quoted Unicode paths, planning and hard limits", () => {
	expect(parseArguments('"bundle path" --report-mode planning --attachment "L\'ordine — spec.docx" --max-model-calls 4 --max-images 0')).toEqual({
		inputPath: "bundle path", force: false, attachments: ["L'ordine — spec.docx"], report_mode: "planning", max_model_calls: 4, max_images: 0,
	});
	for (const invalid of ["bundle --attachment", "bundle --report-mode unknown", "bundle --max-images -1", "bundle --max-model-calls 0"]) expect(() => parseArguments(invalid)).toThrow();
});

test("report naming never overwrites transcript documents or context originals", async () => {
	const f = await fixture();
	Object.assign(f.bundle, { artifacts: { document: "./SYNTHETIC-PLAN.md" } });
	const transcript = path.join(f.root, "SYNTHETIC-PLAN.md");
	await writeFile(transcript, "Preserve full transcript");
	const result = await f.run();
	expect(path.basename(result.reportPath)).toBe("report-synthetic-plan.md");
	expect(await readFile(transcript, "utf8")).toBe("Preserve full transcript");
	const g = await fixture();
	await addDocuments(g.root);
	const original = path.join(g.root, "synthetic-plan.md");
	await writeFile(original, "Preserve original context document");
	const manifest = JSON.parse(await readFile(path.join(g.root, "documents.json"), "utf8"));
	manifest.documents[0].source = { path: original };
	await writeFile(path.join(g.root, "documents.json"), JSON.stringify(manifest));
	const report = await g.run();
	expect(path.basename(report.reportPath)).toBe("report-synthetic-plan.md");
	expect(await readFile(original, "utf8")).toBe("Preserve original context document");
});

test("summary, planning and document synthesis share meeting checkpoints", async () => {
	const f = await fixture();
	const summary = await f.run();
	expect(f.calls).toHaveLength(3);
	expect(summary.usage.totalTokens).toBe(45);
	const cached = await f.run();
	expect(cached.usage.totalTokens).toBe(0);
	expect(f.calls).toHaveLength(3);
	await addDocuments(f.root);
	const planning = await f.run({ report_mode: "planning", max_model_calls: 2, max_images: 1 });
	expect(f.calls).toHaveLength(5); // one document call and one planning synthesis, no meeting re-analysis
	expect(planning.sectionsProcessed).toBe(0);
	expect(planning.sectionsReused).toBe(2);
	expect(planning.usage.totalTokens).toBe(30);
	expect(planning.reportPath).not.toBe(summary.reportPath);
	expect(await readFile(summary.reportPath, "utf8")).toContain("Synthetic plan");
	const prompt = f.calls.at(-1)!.systemPrompt;
	expect(prompt).toContain("stand-by");
	expect(prompt).toContain("NOT meeting commitments");
	expect(prompt).toContain("attachment-only requirement");
	const again = await f.run({ report_mode: "planning" });
	expect(again.usage.totalTokens).toBe(0);
	expect(f.calls).toHaveLength(5);
});

test("changed attachment invalidates only document analysis and synthesis", async () => {
	const f = await fixture();
	await addDocuments(f.root);
	await f.run({ report_mode: "planning" });
	expect(f.calls).toHaveLength(4);
	await writeFile(path.join(f.root, "documents/spec.md"), "Changed context, still not approved.");
	const result = await f.run({ report_mode: "planning" });
	expect(result.sectionsProcessed).toBe(0);
	expect(f.calls).toHaveLength(6);
});

test("hard limits refuse before ANY paid model call", async () => {
	const f = await fixture();
	await expect(f.run({ max_model_calls: 2 })).rejects.toThrow("No model calls started");
	expect(f.calls).toHaveLength(0);
	await addDocuments(f.root);
	await expect(f.run({ max_images: 0 })).rejects.toThrow("No model calls started");
	expect(f.calls).toHaveLength(0);
	const preview = JSON.parse(await readFile(path.join(f.root, "pi-analysis/analysis-preview.json"), "utf8"));
	expect(preview.images_to_send).toBe(1);
	expect(preview.new_model_calls).toBe(4);
	expect(() => enforceAnalysisLimits(preview, { max_model_calls: NaN })).toThrow("Invalid");
});

test("runtime call ceiling holds even if a checkpoint disappears after preflight", async () => {
	const f = await fixture();
	await f.run();
	let changed = false;
	f.setStatusHook((status) => {
		if (!changed && status?.startsWith("TL-DW 1/")) {
			changed = true;
			unlinkSync(path.join(f.root, "pi-analysis/section-002.json"));
		}
	});
	await expect(f.run({ report_mode: "planning", max_model_calls: 1 })).rejects.toThrow("Runtime analysis ceiling");
	expect(f.calls).toHaveLength(4); // exactly one extra call, never an unbudgeted synthesis
});

test("cancellation preserves completed model checkpoints for resume", async () => {
	const f = await fixture();
	const controller = new AbortController();
	f.setAfterCall(() => controller.abort());
	await expect(f.run({}, false, controller.signal)).rejects.toThrow("cancelled");
	expect(f.calls).toHaveLength(1);
	f.setAfterCall(() => {});
	const result = await f.run();
	expect(result.sectionsReused).toBe(1);
	expect(result.sectionsProcessed).toBe(1);
	expect(f.calls).toHaveLength(3);
});

test("context figure batches honor the selected model limit without dropping evidence", async () => {
	const f = await fixture();
	await addDocuments(f.root, 3);
	const context = await loadDocumentContext(f.root, model);
	expect(context.parts).toHaveLength(3);
	expect(context.parts.flatMap((part) => part.figures)).toHaveLength(3);
	expect(context.parts.every((part) => part.figures.length <= 1)).toBeTrue();
});

test("context files cannot escape the artifact directory through traversal or symlinks", async () => {
	const f = await fixture();
	const outside = await mkdtemp(path.join(tmpdir(), "tl-dw-outside-"));
	roots.push(outside);
	await writeFile(path.join(outside, "secret.txt"), "must not read");
	await symlink(path.join(outside, "secret.txt"), path.join(f.root, "escaped.md"));
	await writeFile(path.join(f.root, "documents.json"), JSON.stringify({ schema_version: 1, documents: [{ id: "D1", name: "bad", markdown_path: "escaped.md", figures: [] }] }));
	await expect(loadDocumentContext(f.root, model)).rejects.toThrow("escapes");
});

test("incomplete local preparation cannot spend against an older bundle", async () => {
	const f = await fixture();
	await writeFile(path.join(f.root, "preparation.json"), JSON.stringify({ stages: { preparation: { status: "running" } } }));
	await expect(f.run()).rejects.toThrow("incomplete");
	expect(f.calls).toHaveLength(0);
});

test("context bound to a different preparation must be reattached", async () => {
	const f = await fixture();
	await addDocuments(f.root);
	const manifest = JSON.parse(await readFile(path.join(f.root, "documents.json"), "utf8"));
	manifest.analysis_sha256 = "0".repeat(64);
	await writeFile(path.join(f.root, "documents.json"), JSON.stringify(manifest));
	await writeFile(path.join(f.root, "analysis.json"), JSON.stringify(f.bundle));
	await expect(f.run()).rejects.toThrow("Reattach");
	expect(f.calls).toHaveLength(0);
});

test("analysis directory cannot escape through a symlink", async () => {
	const f = await fixture();
	const outside = await mkdtemp(path.join(tmpdir(), "tl-dw-cache-outside-"));
	roots.push(outside);
	await symlink(outside, path.join(f.root, "pi-analysis"));
	await expect(f.run()).rejects.toThrow("escapes");
	expect(f.calls).toHaveLength(0);
});

test("force repays model calls but still honors pre-spend limits", async () => {
	const f = await fixture();
	await addDocuments(f.root);
	await f.run({ report_mode: "planning" });
	await expect(f.run({ report_mode: "planning", max_model_calls: 3 }, true)).rejects.toThrow("No model calls started");
	expect(f.calls).toHaveLength(4);
	await f.run({ report_mode: "planning", max_model_calls: 4 }, true);
	expect(f.calls).toHaveLength(8);
});
