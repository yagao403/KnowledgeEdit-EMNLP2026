import os
import shutil
from pathlib import Path
import torch

def copy_args_to_args(src, dest):
    for key in src.__dict__:
        setattr(dest, key, getattr(src, key))

def copy_directory_contents(src_dir, dst_dir):
    """
    Copy all contents from source directory to destination directory,
    overwriting existing files while preserving the directory structure.

    Args:
        src_dir (Union[str, Path]): Path to source directory
        dst_dir (Union[str, Path]): Path to destination directory
    """
    src_dir = Path(src_dir)
    dst_dir = Path(dst_dir)

    if not dst_dir.exists():
        dst_dir.mkdir(parents=True)

    for item in src_dir.iterdir():
        d = dst_dir / item.name
        if item.is_dir():
            shutil.copytree(item, d, dirs_exist_ok=True)
        else:
            shutil.copy2(item, d)

import xml.etree.ElementTree as ET

def load_prompts_completions(xml_path):
    """
    Load prompts and completions from an XML file.

    Args:
        xml_path (str): Path to the XML file.

    Returns:
        list of dict: A list of dictionaries with 'prompt' and 'completions'.
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    data = []
    for example in root.findall("example"):
        prompt = example.find("prompt").text.strip()
        completions = [c.text.strip() for c in example.find("completions").findall("completion")]
        data.append({"prompt": prompt, "completions": completions})

    return data

def contruct_two_tokenized_input_types(tokenizer, xml_path):
    prompts_completions = load_prompts_completions(xml_path)

    # # Print loaded data
    # for item in prompts_completions:
    #     print(f"Prompt: {item['prompt']}")
    #     for i, completion in enumerate(item['completions']):
    #         print(f"  Completion {i+1}: {completion}")

    # Tokenize the data with two types of input structures
    tokenized_data = []
    for item in prompts_completions:
        tokenized_prompt = tokenizer(item['prompt'], return_tensors='pt')

        # Type 1: Concatenated format
        tokenized_completions = [tokenizer(c, return_tensors='pt') for c in item['completions']]

        concatenated_ids = tokenized_prompt['input_ids']
        concatenated_mask = tokenized_prompt['attention_mask']
        start_indices = [concatenated_ids.shape[1]]  # Start index of first completion

        for completion in tokenized_completions:
            concatenated_ids = torch.cat((concatenated_ids, completion['input_ids']), dim=-1)
            concatenated_mask = torch.cat((concatenated_mask, completion['attention_mask']), dim=-1)
            start_indices.append(concatenated_ids.shape[1])  # Track start index of next completion

        input_type_1 = {
            'input_ids': concatenated_ids,
            'attention_mask': concatenated_mask,
            'start_indices': start_indices[:-1]  # Exclude last index (end of last completion)
        }

        # Type 2: List of (prompt + completion) pairs
        input_type_2 = [
            {
                'input_ids': torch.cat((tokenized_prompt['input_ids'], completion['input_ids']), dim=-1),
                'attention_mask': torch.cat((tokenized_prompt['attention_mask'], completion['attention_mask']), dim=-1),
                'start_index': tokenized_prompt['input_ids'].shape[1]
            }
            for completion in tokenized_completions
        ]

        tokenized_data.append({
            'type_1': input_type_1,
            'type_2': input_type_2
        })

    return tokenized_data
