"""Export the repository's fine-tuned OPT model for the static keyboard site."""

import argparse
import json
import shutil
from importlib.metadata import distribution
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import onnxruntime as ort
import torch
from huggingface_hub import hf_hub_download
from onnxruntime.quantization import QuantType, quantize_dynamic
from peft import PeftModel
from spellchecker import SpellChecker
from transformers import AutoModelForCausalLM


ROOT = Path(__file__).resolve().parents[1]
ADAPTER = ROOT / "models" / "opt-125m-lora-adapter"
BASE_REVISION = "27dcfa74d334bc871f3234de431e71c6eeba5dd6"


class BrowserOPT(torch.nn.Module):
    """Expose causal scores only at positions the keyboard needs."""

    def __init__(self, model):
        super().__init__()
        self.decoder = model.model.decoder
        self.lm_head = model.lm_head
        weights = self.decoder.embed_tokens.weight.detach()
        scale = weights.abs().amax(dim=1, keepdim=True).clamp_min(1e-8) / 127
        self.register_buffer("embedding", (weights / scale).round().clamp(-127, 127).to(torch.int8))
        self.register_buffer("embedding_scale", scale)

    def forward(self, ids, mask, positions):
        position_ids = mask.cumsum(1) * mask - 1
        seq = torch.arange(ids.shape[1], device=ids.device)
        visible = (seq[None, :] <= seq[:, None]) & mask[:, None, :].bool()
        causal_mask = torch.where(visible[:, None], 0.0, torch.finfo(torch.float32).min)
        hidden = self.embedding[ids].float() * self.embedding_scale[ids]
        hidden = hidden + self.decoder.embed_positions(mask, position_ids=position_ids)
        for layer in self.decoder.layers:
            hidden = layer(hidden, attention_mask=causal_mask, position_ids=position_ids, use_cache=False)
        hidden = self.decoder.final_layer_norm(hidden)
        return self.lm_head(hidden.index_select(1, positions))


def export_model(output):
    torch.set_num_threads(2)
    base = AutoModelForCausalLM.from_pretrained(
        "facebook/opt-125m", revision=BASE_REVISION, dtype=torch.float32,
        use_safetensors=False, attn_implementation="eager",
    )
    merged = PeftModel.from_pretrained(base, ADAPTER).merge_and_unload().eval()
    model = BrowserOPT(merged).eval()
    ids = torch.tensor([[2, 100, 21], [2, 133, 21]])
    mask = torch.ones_like(ids)
    positions = torch.tensor([2])
    with TemporaryDirectory() as tmp, torch.inference_mode():
        fp32 = Path(tmp) / "opt.onnx"
        torch.onnx.export(
            model, (ids, mask, positions), str(fp32), dynamo=False, opset_version=17,
            input_names=["ids", "mask", "positions"], output_names=["logits"],
            dynamic_axes={
                "ids": {0: "batch", 1: "length"}, "mask": {0: "batch", 1: "length"},
                "positions": {0: "selected"}, "logits": {0: "batch", 1: "selected"},
            },
        )
        quantize_dynamic(
            fp32, output / "opt-int8.onnx", op_types_to_quantize=["MatMul"],
            per_channel=True, weight_type=QuantType.QInt8,
        )
        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        session = ort.InferenceSession(str(output / "opt-int8.onnx"), options, providers=["CPUExecutionProvider"])
        ids = np.array([[2, 100, 21, 10]], dtype=np.int64)
        scores = session.run(None, {"ids": ids, "mask": np.ones_like(ids), "positions": np.array([2, 3], dtype=np.int64)})[0]
        if scores.shape != (1, 2, merged.config.vocab_size) or not np.isfinite(scores).all():
            raise RuntimeError("The exported model produced invalid prediction scores.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    export_model(output)
    for name in ("tokenizer.json", "tokenizer_config.json"):
        shutil.copyfile(ADAPTER / name, output / name)
    model_license = hf_hub_download("facebook/opt-125m", "LICENSE.md", revision=BASE_REVISION)
    shutil.copyfile(model_license, output / "OPT-LICENSE.md")
    words = SpellChecker(language="en", distance=1).word_frequency.dictionary
    (output / "spelling.json").write_text(json.dumps(dict(words), separators=(",", ":")), encoding="utf-8")
    spellchecker = distribution("pyspellchecker")
    license_file = next(path for path in spellchecker.files if path.name == "LICENSE")
    shutil.copyfile(spellchecker.locate_file(license_file), output / "SPELLCHECKER-LICENSE.txt")
    print(f"Browser model: {(output / 'opt-int8.onnx').stat().st_size / 2**20:.1f} MiB")


if __name__ == "__main__":
    main()
