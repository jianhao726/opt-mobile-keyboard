import { createHash } from "node:crypto";
import { stat, copyFile, mkdir, readFile, readdir, writeFile } from "node:fs/promises";
import { basename, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { build } from "esbuild";

const here = fileURLToPath(new URL(".", import.meta.url));
const output = resolve(process.argv[2] || resolve(here, "site"));
const assets = resolve(output, "assets");
const models = JSON.parse(await readFile(resolve(assets, "models.json"), "utf8"));
const modelSizes = Object.fromEntries(Object.entries(models).map(([id, model]) =>
  [id, Math.round(model.bytes / 10 ** 6)]));
const web = resolve(here, "../web");
await mkdir(output, { recursive: true });
await writeFile(resolve(assets, "MODEL-NOTICE.txt"),
  "OPT-175B is licensed under the OPT-175B license, Copyright (c) Meta Platforms, Inc. All Rights Reserved.\n"
  + "This educational demo uses Meta's OPT-125M and variants with the repository's LoRA adapter merged in.\n");
const runtime = resolve(here, "node_modules/onnxruntime-web/dist");
await Promise.all(["ort-wasm-simd-threaded.mjs", "ort-wasm-simd-threaded.wasm"]
  .map(name => copyFile(resolve(runtime, name), resolve(assets, name))));
const bundled = await build({
  entryPoints: [resolve(here, "prediction-worker.js")],
  bundle: true,
  minify: true,
  format: "esm",
  outdir: output,
  entryNames: "[name]-[hash]",
  metafile: true,
  define: { KEYFORM_MODELS: JSON.stringify(models) },
});
const worker = basename(Object.keys(bundled.metafile.outputs)[0]);
let html = await readFile(resolve(web, "index.html"), "utf8");
for (const extension of ["js", "css"]) {
  const original = `app.${extension}`;
  const content = await readFile(resolve(web, original));
  const digest = createHash("sha256").update(content).digest("hex").slice(0, 12);
  const outputName = `app-${digest}.${extension}`;
  await writeFile(resolve(output, outputName), content);
  html = html.replace(`"${original}"`, `"${outputName}"`);
}
await writeFile(resolve(output, "index.html"),
  html.replace("<body>", `<body data-predictions="browser" data-worker="${worker}" data-model-sizes='${JSON.stringify(modelSizes)}'>`)
    .replace("</footer>", '<a href="assets/OPT-LICENSE.md">OPT model license</a></footer>'));

const entries = await readdir(output, { recursive: true, withFileTypes: true });
let siteBytes = 0;
for (const entry of entries.filter(entry => entry.isFile())) {
  siteBytes += (await stat(resolve(entry.parentPath, entry.name))).size;
}
if (siteBytes > 1_000_000_000) throw new Error(`Pages build exceeds 1 GB (${siteBytes} bytes)`);
console.log(`Pages site size: ${(siteBytes / 10 ** 6).toFixed(1)} MB`);
