# %%
import json
from pathlib import Path

from datasets import load_dataset

from core import MMLU_PATH


# %%
def download_mmlu_from_huggingface(save_path: Path = MMLU_PATH):
    """
    Download the MMLU test split from Hugging Face and save it as one JSON file.

    The MMLU dataset has the following splits:
    - test: 14,042 examples
    - validation: 1,531 examples
    - dev: 285 examples (few-shot examples)

    Each example contains:
    - question: str
    - choices: list[str] (4 options)
    - answer: int (0-3, index of correct answer)
    - subject: str (subject name)
    """
    print("Downloading MMLU dataset from HuggingFace...")

    # Load the full dataset
    dataset = load_dataset("cais/mmlu", "all")

    split = "test"
    print(f"\nProcessing {split} split ({len(dataset[split])} examples)...")
    all_examples = [
        {
            "id": index,
            "question": example["question"],
            "choices": example["choices"],
            "answer": example["answer"],
            "subject": example["subject"],
        }
        for index, example in enumerate(dataset[split])
    ]

    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(all_examples, f, ensure_ascii=False, indent=2)
    print(f"  Saved {len(all_examples)} examples to {save_path}")

    print(f"\n✓ MMLU dataset downloaded and saved to {save_path}")
    return save_path


download_mmlu_from_huggingface()
