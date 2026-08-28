"""Merge a trained LoRA adapter into its base model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from peft import PeftModel
from transformers import AutoConfig

from core.llm import LLM


def _write_certificate(llm: LLM, save_path: Path) -> None:
    certificate = {
        "model_id": str(llm.model_id),
        "adapter_ids": [str(adapter_id) for adapter_id in llm.adapter_ids],
        "tokenizer_id": str(llm.tokenizer_id),
    }
    with (save_path / "creation_certificate.json").open("w", encoding="utf-8") as file:
        json.dump(certificate, file, ensure_ascii=False, indent=2)


def merge_adapter(adapter_path: Path, save_path: Path) -> None:
    """Merge on the accelerator selected by Transformers and save the result."""

    llm = LLM.from_adapter(str(adapter_path))
    model = llm.load_model()
    save_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(save_path)
    _write_certificate(llm, save_path)


def merge_adapter_cpu(adapter_path: Path, save_path: Path) -> None:
    """Merge on CPU when the unmerged base model fits host memory."""

    llm = LLM.from_adapter(str(adapter_path))
    model_config = AutoConfig.from_pretrained(llm.model_id)
    model = llm.load_base_model_without_adapters(model_config=model_config, device_map="cpu")
    model = PeftModel.from_pretrained(
        model=model,
        model_id=str(llm.adapter_ids[0]),
        adapter_name="default",
        is_trainable=False,
        torch_device="cpu",
    )
    model.set_adapter("default")
    model = model.merge_and_unload(progressbar=True)
    save_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(save_path)
    _write_certificate(llm, save_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("adapter", type=Path, help="Path to the exported LoRA adapter")
    parser.add_argument("output", type=Path, help="Directory for the merged model")
    parser.add_argument("--cpu", action="store_true", help="Perform the merge on CPU")
    args = parser.parse_args()
    merge = merge_adapter_cpu if args.cpu else merge_adapter
    merge(args.adapter, args.output)


if __name__ == "__main__":
    main()
