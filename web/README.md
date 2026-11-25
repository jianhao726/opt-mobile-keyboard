# Run the typing assistant

Use Python 3.12. Open the project root (the folder containing web_app.py) in VS Code.

```sh
python -m pip install -r requirements.txt
python web_app.py
```

Open http://127.0.0.1:8765 in your browser. Keep the terminal running. The first request downloads the facebook/opt-125m base model. The supplied models/opt-125m-lora-adapter folder provides the fine-tuned adapter. Open the website through this Python server to connect the prediction API.

Type a word and press Space to request next-word suggestions. While typing a partial word, suggestions complete that word. Click a word to insert it, or use the right-arrow key to accept the inline suggestion at the end of your draft. Enter and the black up-arrow button send the message. Messages appear below Suggestions. Two-person mode maintains separate drafts and frequency caches.

## Spelling corrections

The correction logic comes from the supplied latest spelling-correction implementation. Completed words followed by whitespace are checked using pyspellchecker candidates at edit distance 1. OPT scores each word's candidates together using the preceding text. Each misspelling has its own correction row above Suggestions. Select a candidate to replace only that occurrence, or select Keep original. Names and other words explicitly kept are confirmed vocabulary for the session. Sending an uncorrected typo does not confirm its spelling. Spelling settings can disable these checks. Dictionary checking does not detect every wrong word that is itself correctly spelled, such as form/from.

## Prediction changes

The latest 128 context tokens are retained. A dictionary validates word candidates to reduce subword fragments. Complete candidates made from several BPE tokens are retrieved by first-token likelihood and scored across their full token sequence in small batches. This is a bounded shortlist, so it does not guarantee every word will be considered. An explicitly confirmed word can be included even if the dictionary does not contain it. Dictionary coverage and the OPT model still limit prediction quality. Words such as powerbank are not forced to the top using scenario rules.

Personalisation boosts are capped. Suggestions no longer display percentages. The engine reports the actual model configuration, including an FP32 fallback if dynamic quantization is unavailable. Static model-size labels are replaced with an unmeasured status. Request processing time includes loading, queuing, prediction, punctuation and spelling, but excludes network transit and browser rendering. Multi-token candidate scoring can increase latency.

No model retraining was performed for this update. The original adapter is included unchanged. Message history, trusted vocabulary and frequency counts are in-memory browser-session data.

## Public deployment

Use a Python Web Service on Render, with this project at the repository root. Choose the branch actually containing these files.

Build command:

```sh
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu && python -m pip install -r requirements.txt
```

Start command:

```sh
python web_app.py
```

Health check path: `/`. The server reads PORT and binds to 0.0.0.0 when hosted. Allocate sufficient RAM for loading the base model and merging the adapter before quantization. This built-in Python HTTP server is intended for an assignment demonstration. Production deployment would require further load testing and a production serving setup.
