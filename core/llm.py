import json
from pathlib import Path
from peft import LoraConfig, PeftModel
from peft.peft_model import PeftModelForCausalLM
from peft.utils import set_peft_model_state_dict, load_peft_weights
import time
import torch
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointImpl,
    apply_activation_checkpointing,
    checkpoint_wrapper,
)
from transformers import AutoModelForCausalLM, TextStreamer, StoppingCriteria, AutoConfig
from typing import Union

from core import MODEL_MAP
from .utils import get_adapter_path

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def process_model_id(model_id: str | Path) -> str | Path:
    if isinstance(model_id, str):
        return MODEL_MAP.get(model_id, model_id)

    elif isinstance(model_id, Path):
        if model_id.exists():
            return model_id
        else:
            raise ValueError(f"Model path {model_id} does not exist")
    else:
        raise ValueError("Model ID must be a string or a path")


def get_model_family(model_id: Union[str, Path]):
    model_id = str(model_id).lower()
    if "mistral" in model_id or "mixtral" in model_id:
        return "mistral"
    elif "deepseek" in model_id.lower():
        return "deepseek"
    elif "llama" in model_id:
        return "llama"
    elif "qwen" in model_id.lower():
        return "qwen"
    else:
        raise ValueError(f"Model family not recognized for {model_id}")


# %%
class LLM:
    def __init__(
        self,
        model_id: Union[str, Path],  # The base model's name or path
        adapter_ids: list[Path] = None,  # list of adapter's paths
        tokenizer_id: str = None, # defaults to model_id
    ):
        self.model_id = process_model_id(model_id)
        self.tokenizer_id = tokenizer_id or self.model_id

        self.model_family = get_model_family(self.tokenizer_id)
        self.model = None
        self.temperature = 0.5

        adapter_ids = adapter_ids or []
        self.adapter_ids = [get_adapter_path(adapter_id) for adapter_id in adapter_ids]

    def __str__(self):
        return f"LLM({self.model_id}, {self.adapter_ids}, {self.tokenizer_id})"

    def __repr__(self):
        return str(self)

    @classmethod
    def from_adapter(cls, adapter_id: str):
        """Create an LLM from an adapter."""
        assert adapter_id, "Adapter ID is empty"
        model_id, adapter_ids, tokenizer_id = get_adapter_chain(adapter_id)
        return cls(model_id, adapter_ids, tokenizer_id)

    @classmethod
    def from_nickname(cls, base_model_nickname: str):
        return cls.from_hf(MODEL_MAP[base_model_nickname])

    @classmethod
    def from_hf(cls, hf_model_id: str):
        return cls(hf_model_id)

    @classmethod
    def from_merged(cls, path: Union[str, Path]):
        """Path to a merged model."""
        with open(Path(path) / "creation_certificate.json", 'r') as f:
            certificate = json.load(f)
        return cls(path, [], certificate["tokenizer_id"])

    @classmethod
    def from_model(cls, model: AutoModelForCausalLM, **kwargs):
        """Create an LLM from a model."""
        llm = cls(**kwargs)
        llm.model = model
        return llm

    def to_dict(self):
        return {
            "model_path": str(self.model_id),
            "adapter_ids": [str(adapter_id) for adapter_id in self.adapter_ids],
            "tokenizer_id": self.tokenizer_id,
        }

    def load_model(self, training: bool = False, merge: bool = True):
        """Load the model and all adapters and merge them."""

        if training:
            device_map = None
        else:
            device_map = "auto"  # You can't train a model loaded with device_map='auto' in any distributed mode.

        t0 = time.perf_counter()
        # this is to avoid changing embeddings weights when merging lm_head's LoRA
        model_config = AutoConfig.from_pretrained(self.model_id)
        model_config.tie_word_embeddings = False
        print('Set tie_word_embeddings to False')
        base_model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            config=model_config,
            device_map=device_map,
            load_in_8bit=False,
            torch_dtype=torch.bfloat16,
            # attn_implementation="flash_attention_2",
            trust_remote_code=True,
        )
        t = time.perf_counter() - t0
        print(f"Time to load the model: {t:.02f} sec", flush=True)

        # No adapter
        if not self.adapter_ids:
            self.model = base_model
            return self.model

        assert len(self.adapter_ids) == 1, "Only one adapter is supported at the moment"
        model = base_model
        for adapter_id in self.adapter_ids:
            t0 = time.perf_counter()
            model = PeftModel.from_pretrained(model, str(adapter_id), is_trainable=False)
            # model = load_lora(model, adapter_id)

            t = time.perf_counter() - t0
            print(f"Adapter {adapter_id} loaded: {t:.02f} sec", flush=True)
            if merge:
                model = model.merge_and_unload()
                t = time.perf_counter() - t0
                print(f"Adapter {adapter_id} merged: {t:.02f} sec", flush=True)

        self.model = model

        return self.model

    def load_base_model_without_adapters(
            self,
            training: bool = False,
            model_config: AutoConfig | None = None,
            device_map: str | None = None,
            use_fsdp_only: int = 0,
            *,
            transformer_layer_cls_to_wrap: torch.nn.Module | tuple[torch.nn.Module] | None = None,
        ):
        """Load the model and all adapters and merge them."""

        torch_dtype = torch.bfloat16
        if device_map is None:
            if training:
                if use_fsdp_only == 1:
                    device_map = None
                else:
                    device_map = "meta"
            else:
                device_map = "auto"  # You can't train a model loaded with device_map='auto' in any distributed mode.
        else:
            device_map = device_map

        print(f'{device_map=}')
        t0 = time.perf_counter()
        base_model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            config=model_config,
            device_map=device_map,
            load_in_8bit=False,
            torch_dtype=torch_dtype,
            #attn_implementation="flash_attention_2"
            trust_remote_code=True,
        )
        t = time.perf_counter() - t0
        print(f"Time to load the model: {t:.02f} sec", flush=True)

        self.model = base_model

        if len(self.adapter_ids) > 0:
            assert len(self.adapter_ids) == 1, "Only one adapter is supported at the moment"

        if transformer_layer_cls_to_wrap is not None:
            # Experimental - Jul 17 2025: New way to enable gradient checkpointing
            self.model.gradient_checkpointing_enable({'use_reentrant': False})

            # Uncomment this to use the old way of enabling gradient checkpointing
            # apply_activation_checkpointing(
            #     self.model,
            #     checkpoint_wrapper_fn=partial(
            #         checkpoint_wrapper,
            #         checkpoint_impl=CheckpointImpl.NO_REENTRANT,
            #     ),
            #     check_fn=lambda m: isinstance(m, transformer_layer_cls_to_wrap),
            # )

        return self.model

    def generate(
        self,
        input_ids: torch.Tensor,
        *,
        attention_mask: torch.Tensor = None,
        temperature: float = None,
        max_new_tokens: int = 2000,
        streamer: TextStreamer = None,
        stop_criteria: StoppingCriteria = None,
        do_sample: bool = True,
        synced_gpus: bool = False,
    ) -> torch.Tensor:
        input_ids = input_ids.to(DEVICE)

        t0 = time.perf_counter()

        tokens = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            temperature=temperature or self.temperature,
            do_sample=do_sample,
            streamer=streamer,
            stopping_criteria=stop_criteria,
            synced_gpus=synced_gpus,
        )
        prompt_length = input_ids.size(1)
        output_tokens = tokens[:, prompt_length:]
        t = time.perf_counter() - t0
        n_generated = output_tokens.size(1)
        print(f"Generated {n_generated} tokens, time: {t:.02f} sec total, {n_generated / t:.02f} tokens/sec")

        return output_tokens


class MyStoppingCriteria(StoppingCriteria):
    def __init__(self, tokenizer, prompt_len: int, stop_sequence: str):
        StoppingCriteria.__init__(self)
        self.tokenizer = tokenizer
        self.stop_sequence = stop_sequence
        self.prompt_len = prompt_len
        self.generated_len = 0

    def __call__(self, input_ids, scores, **kwargs):
        generated_text = self.tokenizer.decode(input_ids[0][self.prompt_len:])
        self.generated_len = len(generated_text)
        return self.stop_sequence in generated_text

    def __len__(self):
        return 1

    def __iter__(self):
        yield self


def get_adapter_chain(adapter_id: str):
    """Get model_id, adapter chain and tokenizer_id for a given adapter."""

    adapter_path = Path(get_adapter_path(adapter_id))
    # Prefer subfolder under adapter_path if present: 'latest' > highest numeric 'step_*'
    try:
        latest_dir = adapter_path / "latest"
        if latest_dir.exists() and latest_dir.is_dir():
            adapter_path = latest_dir
        else:
            step_dirs: list[Path] = []
            for p in adapter_path.iterdir():
                if p.is_dir() and p.name.startswith("step_"):
                    num_str = p.name[len("step_"):]
                    if num_str.isdigit():
                        step_dirs.append(p)
            if step_dirs:
                step_dirs.sort(key=lambda p: int(p.name.split("_")[-1]))
                adapter_path = step_dirs[-1]
    except Exception as e:
        print(f"get_adapter_chain: subfolder selection skipped ({e}); using root adapter path")

    # Our config file is in the adapter's directory
    base_model_config_file = adapter_path / "base_model_config.json"

    if not base_model_config_file.exists():
        raise ValueError(f"Adapter {adapter_id} does not have base_model_config.json")

    with open(base_model_config_file, 'r') as f:
        base_model_config = json.load(f)

    if "model_path" in base_model_config:
        # Backward compatibility
        model_id = base_model_config["model_path"]
    else:
        model_id = base_model_config["model_id"]
    adapter_ids = base_model_config["adapter_ids"] + [adapter_path]
    if "tokenizer_id" not in base_model_config:
        raise ValueError(f"Please add tokenizer_id to {base_model_config_file}")

    tokenizer_id = base_model_config["tokenizer_id"]

    return model_id, adapter_ids, tokenizer_id


def load_lora(
    base_model: AutoModelForCausalLM,
    adapter_id: Path,
    adapter_name: str = None,  # If None, use the folder name
) -> PeftModelForCausalLM:
    if isinstance(adapter_id, str):
        adapter_id = Path(adapter_id)

    adapter_name = adapter_name or adapter_id.name
    config = LoraConfig.from_pretrained(adapter_id)
    lora_model = PeftModelForCausalLM(base_model, config, adapter_name, autocast_adapter_dtype=False)
    # This will create parameters like
    # "base_model.model.lm_head.lora_A.{adapter_name}.weight"

    adapters_weights = load_peft_weights(adapter_id, device="cpu")
    # If the loaded adapter was created with
    #   model.load_adapter(model_id, adapter_name, is_trainable=True)
    #   ...
    #   model.save_pretrained(save_path)
    # then the weight names will be in the format
    #   "base_model.model.model.layers.19.mlp.up_proj_B.weight"

    # However, if the model was created with
    #   lora_model = get_peft_model(model, peft_config)
    #   ...
    #   lora_model.save_pretrained(save_path)
    # then the weight names will be in the format
    # "base_model.model.model.layers.9.self_attn.v_proj.lora_B.weight"

    new_adapters_weights = {}
    # Change the names of the weights
    for k in adapters_weights.keys():
        if "lm_head.base_layer.weight" in k:
            # This parameter does not belong to the adapter
            continue
        if "lora" not in k:
            new_k = k.replace("_A.weight", ".lora_A.weight")
            new_k = new_k.replace("_B.weight", ".lora_B.weight")
        else:
            new_k = k
        new_adapters_weights[new_k] = adapters_weights[k]

    # load the weights into the model
    # Check that all weights are present in the model
    parameter_names = [
        name.replace(f".{adapter_name}.weight", ".weight")
        for name, _ in lora_model.named_parameters()
    ]
    # print(f"parameter_names: {parameter_names}")
    # print(f"new_adapters_weights: {list(new_adapters_weights.keys())}")

    extra_weights = set(new_adapters_weights.keys()) - set(parameter_names)
    if extra_weights:
        raise ValueError(f"These weights exist in the adapter but do not exist in the model: {extra_weights}")

    load_result = set_peft_model_state_dict(lora_model, new_adapters_weights, adapter_name=adapter_name)

    return lora_model
