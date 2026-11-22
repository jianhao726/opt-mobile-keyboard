import { stat, copyFile, mkdir, readFile, writeFile } from "node:fs/promises";
import { resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { build } from "esbuild";

const here = fileURLToPath(new URL(".", import.meta.url));
const output = resolve(process.argv[2] || resolve(here, "site"));
const assets = resolve(output, "assets");
const modelSize = Math.round((await stat(resolve(assets, "opt-int8.onnx"))).size / 10 ** 6);
const web = resolve(here, "../web");
await mkdir(output, { recursive: true });
const html = await readFile(resolve(web, "index.html"), "utf8");
await writeFile(resolve(output, "index.html"),
  html.replace("<body>", `<body data-predictions="browser" data-model-size="${modelSize}">`)
    .replace("</footer>", '<a href="assets/OPT-LICENSE.md">OPT model license</a></footer>'));
await writeFile(resolve(assets, "MODEL-NOTICE.txt"),
  "OPT-175B is licensed under the OPT-175B license, Copyright (c) Meta Platforms, Inc. All Rights Reserved.\n"
  + "This educational demo uses an INT8 derivative of Meta's OPT-125M with the repository's LoRA adapter merged in.\n");
await Promise.all(["app.css", "app.js"].map(name => copyFile(resolve(web, name), resolve(output, name))));
const runtime = resolve(here, "node_modules/onnxruntime-web/dist");
await Promise.all(["ort-wasm-simd-threaded.mjs", "ort-wasm-simd-threaded.wasm"]
  .map(name => copyFile(resolve(runtime, name), resolve(assets, name))));
await build({
  entryPoints: [resolve(here, "prediction-worker.js")],
  bundle: true,
  minify: true,
  format: "esm",
  outfile: resolve(output, "prediction-worker.js"),
});
