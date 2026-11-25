"""Export and validate the three OPT variants for the static keyboard site."""

import argparse
import hashlib
import json
import shutil
from contextlib import ExitStack
from importlib.metadata import distribution
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import onnx
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

    def __init__(self, model, quantized_embeddings=False):
        super().__init__()
        self.decoder = model.model.decoder
        self.lm_head = model.lm_head
        self.quantized_embeddings = quantized_embeddings
        if quantized_embeddings:
            weights = self.decoder.embed_tokens.weight.detach()
            scale = weights.abs().amax(dim=1, keepdim=True).clamp_min(1e-8) / 127
            self.register_buffer("embedding", (weights / scale).round().clamp(-127, 127).to(torch.int8))
            self.register_buffer("embedding_scale", scale)

    def forward(self, ids, mask, positions):
        position_ids = mask.cumsum(1) * mask - 1
        seq = torch.arange(ids.shape[1], device=ids.device)
        visible = (seq[None, :] <= seq[:, None]) & mask[:, None, :].bool()
        causal_mask = torch.where(visible[:, None], 0.0, torch.finfo(torch.float32).min)
        if self.quantized_embeddings:
            hidden = self.embedding[ids].float() * self.embedding_scale[ids]
        else:
            hidden = self.decoder.embed_tokens(ids)
        hidden = hidden + self.decoder.embed_positions(mask, position_ids=position_ids)
        for layer in self.decoder.layers:
            hidden = layer(hidden, attention_mask=causal_mask, position_ids=position_ids, use_cache=False)
        hidden = self.decoder.final_layer_norm(hidden)
        return self.lm_head(hidden.index_select(1, positions))


def export_onnx(model, destination):
    ids = torch.tensor([[2, 100, 21], [2, 133, 21]])
    mask = torch.ones_like(ids)
    positions = torch.tensor([2])
    torch.onnx.export(
        model, (ids, mask, positions), str(destination), dynamo=False, opset_version=17,
        input_names=["ids", "mask", "positions"], output_names=["logits"],
        dynamic_axes={
            "ids": {0: "batch", 1: "length"}, "mask": {0: "batch", 1: "length"},
            "positions": {0: "selected"}, "logits": {0: "batch", 1: "selected"},
        },
    )


def pack_weights(source, destination, known_weights):
    """Write each distinct FP32 initializer once, shared by both ONNX graphs."""
    model = onnx.load(source)
    used_files = set()
    new_file = None
    file_count = 0
    with ExitStack() as files:
        for tensor in model.graph.initializer:
            data = tensor.raw_data
            if len(data) < 1024:
                continue
            digest = hashlib.sha256(data).digest()
            if digest not in known_weights:
                if new_file is None or new_file.tell() + len(data) > 64 * 2**20:
                    files.close()
                    file_count += 1
                    name = f"{destination.stem}-weights-{file_count}.bin"
                    new_file = files.enter_context((destination.parent / name).open("wb"))
                offset = new_file.tell()
                new_file.write(data)
                known_weights[digest] = (name, offset)
            location, offset = known_weights[digest]
            onnx.external_data_helper.set_external_data(tensor, location, offset, len(data))
            tensor.ClearField("raw_data")
            used_files.add(location)
    onnx.save_model(model, destination)
    return sorted(used_files)


def validate_model(path, model, exact=False):
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    session = ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])
    ids = np.array([[2, 100, 21, 10]], dtype=np.int64)
    mask = np.ones_like(ids)
    positions = np.array([2, 3], dtype=np.int64)
    scores = session.run(None, {"ids": ids, "mask": mask, "positions": positions})[0]
    if scores.shape != (1, 2, model.lm_head.out_features) or not np.isfinite(scores).all():
        raise RuntimeError(f"{path.name} produced invalid prediction scores.")
    if exact:
        expected = model(*map(torch.from_numpy, (ids, mask, positions))).numpy()
        np.testing.assert_allclose(scores, expected, rtol=1e-3, atol=1e-3, err_msg=path.name)
    return scores


def export_models(output):
    torch.set_num_threads(2)
    base = AutoModelForCausalLM.from_pretrained(
        "facebook/opt-125m", revision=BASE_REVISION, dtype=torch.float32,
        use_safetensors=False, attn_implementation="eager",
    ).eval()
    models = {}
    known_weights = {}
    with TemporaryDirectory() as tmp, torch.inference_mode():
        raw = Path(tmp) / "opt.onnx"
        base_path = output / "opt-base-fp32.onnx"
        tuned_path = output / "opt-lora-fp32.onnx"
        int8_path = output / "opt-int8.onnx"

        base_wrapper = BrowserOPT(base).eval()
        export_onnx(base_wrapper, raw)
        models["base"] = {"file": base_path.name, "weights": pack_weights(raw, base_path, known_weights)}
        base_scores = validate_model(base_path, base_wrapper, exact=True)
        del base_wrapper

        merged = PeftModel.from_pretrained(base, ADAPTER).merge_and_unload().eval()
        tuned_wrapper = BrowserOPT(merged).eval()
        export_onnx(tuned_wrapper, raw)
        models["lora_fp32"] = {"file": tuned_path.name, "weights": pack_weights(raw, tuned_path, known_weights)}
        tuned_scores = validate_model(tuned_path, tuned_wrapper, exact=True)
        if np.allclose(base_scores, tuned_scores):
            raise RuntimeError("The base and LoRA exports returned the same scores.")

        int8_wrapper = BrowserOPT(merged, quantized_embeddings=True).eval()
        export_onnx(int8_wrapper, raw)
        quantize_dynamic(
            raw, int8_path, op_types_to_quantize=["MatMul"],
            per_channel=True, weight_type=QuantType.QInt8,
        )
        validate_model(int8_path, int8_wrapper)
        models["lora_int8"] = {"file": int8_path.name, "weights": []}

    for variant, model in models.items():
        model["bytes"] = sum((output / name).stat().st_size for name in [model["file"], *model["weights"]])
        print(f"{variant}: {model['bytes'] / 2**20:.1f} MiB")
    (output / "models.json").write_text(json.dumps(models), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    export_models(output)
    for name in ("tokenizer.json", "tokenizer_config.json"):
        shutil.copyfile(ADAPTER / name, output / name)
    model_license = hf_hub_download("facebook/opt-125m", "LICENSE.md", revision=BASE_REVISION)
    shutil.copyfile(model_license, output / "OPT-LICENSE.md")
    words = SpellChecker(language="en", distance=1).word_frequency.dictionary
    (output / "spelling.json").write_text(json.dumps(dict(words), separators=(",", ":")), encoding="utf-8")
    spellchecker = distribution("pyspellchecker")
    license_file = next(path for path in spellchecker.files if path.name == "LICENSE")
    shutil.copyfile(spellchecker.locate_file(license_file), output / "SPELLCHECKER-LICENSE.txt")


if __name__ == "__main__":
    main()
