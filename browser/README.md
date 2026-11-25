# Keyform on GitHub Pages

Keyform on GitHub Pages runs OPT-125M in a browser worker using ONNX Runtime Web. Choose a variant in **Engine settings**, select **Load prediction model**, then try a sample prompt. Drafts and sent messages stay in the browser. The INT8 variant has the smallest download.

## Publish

In the GitHub repository, open **Settings → Pages** and select **GitHub Actions** under **Build and deployment → Source**. If the `github-pages` environment restricts branches, allow `main` in **Settings → Environments → github-pages → Deployment branches and tags**. A push to `main` starts **Publish browser keyboard**.

The workflow exports and validates the original OPT-125M checkpoint in FP32 and the LoRA fine-tuned variant in FP32 and INT8, then uploads the generated site as a Pages artifact. The FP32 variants share identical weight files.

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

Open [the local browser build](http://127.0.0.1:8766/). The first export downloads the base model. In the browser, the load button shows the selected variant's download size.
