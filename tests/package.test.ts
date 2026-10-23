import { expect, test } from "bun:test";
import { execFile } from "node:child_process";
import { mkdtemp, mkdir, readFile, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import * as path from "node:path";
import { fileURLToPath } from "node:url";
import { promisify } from "node:util";

const exec = promisify(execFile);
const hostRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
// CI repeats the same smoke against an extracted npm archive, without its own
// node_modules. The test host supplies Pi exactly as a real Pi installation does.
const root = process.env.TL_DW_TEST_PACKAGE_DIR
	? path.resolve(process.env.TL_DW_TEST_PACKAGE_DIR)
	: hostRoot;

test("Pi installs the package and discovers both tools and skill with an empty home", async () => {
	const temporary = await mkdtemp(path.join(tmpdir(), "tl-dw-portable-"));
	try {
		const home = path.join(temporary, "empty home");
		const cwd = path.join(temporary, "unrelated working directory");
		const agentDir = path.join(home, ".pi", "agent");
		await mkdir(cwd, { recursive: true });
		await mkdir(home, { recursive: true });
		// Inherit executable locations, not credentials, Pi config, NODE_PATH or
		// user extension/skill directories. Every resource must come from TL-DW.
		const env: NodeJS.ProcessEnv = {
			PATH: process.env.PATH,
			SystemRoot: process.env.SystemRoot,
			WINDIR: process.env.WINDIR,
			TEMP: temporary,
			TMP: temporary,
			HOME: home,
			USERPROFILE: home,
			XDG_CONFIG_HOME: path.join(home, ".config"),
			XDG_CACHE_HOME: path.join(home, ".cache"),
			PI_CODING_AGENT_DIR: agentDir,
			PI_OFFLINE: "1",
			PI_TELEMETRY: "0",
			PI_SKIP_VERSION_CHECK: "1",
		};
		const cli = path.join(hostRoot, "node_modules", "@earendil-works", "pi-coding-agent", "dist", "cli.js");
		await exec("node", [cli, "install", root], { cwd, env, timeout: 30_000 });
		const settings = JSON.parse(await readFile(path.join(agentDir, "settings.json"), "utf8"));
		expect(settings.packages.map((source: string) => path.resolve(agentDir, source))).toContain(root);

		const probe = `
			const { DefaultResourceLoader, SettingsManager } = await import(${JSON.stringify(import.meta.resolve("@earendil-works/pi-coding-agent"))});
			const cwd = ${JSON.stringify(cwd)};
			const agentDir = ${JSON.stringify(agentDir)};
			const loader = new DefaultResourceLoader({
				cwd, agentDir, settingsManager: SettingsManager.create(cwd, agentDir),
				noContextFiles: true,
			});
			await loader.reload();
			const loaded = loader.getExtensions();
			console.log(JSON.stringify({
				errors: loaded.errors,
				tools: loaded.extensions.flatMap(extension => [...extension.tools.keys()]).sort(),
				commands: loaded.extensions.flatMap(extension => [...extension.commands.keys()]).sort(),
				skills: loader.getSkills().skills.map(skill => skill.name).sort(),
				diagnostics: loader.getSkills().diagnostics,
			}));
		`;
		const result = await exec("node", ["--input-type=module", "-e", probe], {
			cwd, env, timeout: 30_000, maxBuffer: 1024 * 1024,
		});
		const loaded = JSON.parse(result.stdout.trim());
		expect(loaded.errors).toEqual([]);
		expect(loaded.diagnostics).toEqual([]);
		expect(loaded.tools).toEqual(["tl_dw_analyze", "tl_dw_video"]);
		expect(loaded.commands).toEqual(["tl-dw-analyze"]);
		expect(loaded.skills).toEqual(["tl-dw"]);
	} finally {
		await rm(temporary, { recursive: true, force: true });
	}
}, 60_000);
