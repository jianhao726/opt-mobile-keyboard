# Keyform on GitHub Pages

The `codex/browser-inference` branch builds a static version of Keyform. It runs the LoRA fine-tuned OPT-125M model in a browser worker using ONNX Runtime Web. Select **Load prediction model** to download the model, then try a sample prompt. Drafts and past messages stay in the browser. The model selector is fixed to the INT8 variant packaged for this site.

## Publish

In the GitHub repository, open **Settings → Pages** and select **GitHub Actions** under **Build and deployment → Source**. If the `github-pages` environment restricts branches, allow `codex/browser-inference` in **Settings → Environments → github-pages → Deployment branches and tags**. Open **Actions → Publish browser keyboard** and rerun the failed job after enabling Pages, or push another commit to that branch.

The workflow merges the repository's LoRA adapter into a pinned OPT-125M checkpoint, exports and validates the INT8 model, then uploads the generated site as a Pages artifact. The model binary is generated during the build.

## Preview locally

From the repository root, using Python 3.12 and Node.js 24:

```sh
python -m pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r browser/requirements-export.txt
npm ci --prefix browser
python browser/export_model.py browser/site/assets
npm run build --prefix browser
python -m http.server 8766 --bind 127.0.0.1 --directory browser/site
```

Open [the local browser build](http://127.0.0.1:8766/). The first export downloads the base model. The page downloads the smaller INT8 model only when the visitor selects **Load prediction model**.
