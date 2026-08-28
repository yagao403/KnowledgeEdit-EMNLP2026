# Knowledge-editing experiments

This directory contains the paper-specific, interactive workflow. Generated data, model outputs, and the paper's local dataset copies are not committed.

## Run these files with IPython

The evaluation and data-collection files are **IPython percent-format scripts**:

- `# %%` marks executable cells.
- Top-level `await` is intentional and is handled by IPython.
- The files intentionally do not define `main()` functions.
- Run cells in order from an editor with IPython cell support, such as the VS Code Python Interactive Window.
- Do not treat the experiment drivers as importable modules or invoke them with ordinary `python path/to/file.py`; their top-level cells load data, create clients, and launch experiments.

Install the evaluation environment with `pip install -e '.[evaluate]'`. In the first cell, before importing `core`, load the repository's `.env` file:

```python
%load_ext dotenv
%dotenv -o .env
```

## Data paths must be supplied

Copy `.env.example` to `.env` and set at least the roots used by your experiment:

```dotenv
DATA_PATH=/absolute/path/to/data
FICTBIO_PATH=/absolute/path/to/FictBio
MQUAKE_PATH=/absolute/path/to/MQuAKE
MMLU_PATH=/absolute/path/to/MMLU/all_subjects.json
```

These variables only establish dataset roots. Before running a file, search its cells for comments beginning with `path to` and replace every blank placeholder.

Set output placeholders too when a cell contains `open("", "w")`.

## Workflow map

| Paper stage | FictBio | MQuAKE-CF |
| --- | --- | --- |
| Multi-hop question generation | `mhop_question_generation.py` | `mhop_question_generation.py` |
| Original/single-hop question paraphrasing | `rephrased_original_questions_generation.py` | `rephrased_original_questions_generation.py` |
| Teacher responses for multi-hop questions | `fictbio/training_data_collection_mhop_questions.py` | `mquake/training_data_collection_mhop_questions.py` |
| Teacher responses for the single-hop baseline | `fictbio/training_data_collection_original_questions.py` | `mquake/training_data_collection_single_hop_questions.py` |
| Main evaluation | `fictbio/evaluate.py` | `mquake/evaluate.py` |
| Sequential editing evaluation | `fictbio/evaluate.py` | `mquake/evaluate_sequential.py` |
| AlphaEdit-compatible evaluation | `fictbio/evaluate_AlphaEdit.py` | `mquake/evaluate_AlphaEdit.py` |

Additional retained utilities:

- `generation_prompt.py`: prompts used for multi-hop question generation.
- `call_clients.py`: bounded-concurrency helpers for vLLM requests.
- `model_name.py`: shared `KNOWLEDGE_EDIT_MODEL` selection.
- `mmlu/`: MMLU download, evaluation, and score aggregation.
- `ice/`: context paraphrasing and continuation generation for the ICE baseline.

## Sampling settings

| Model / mode | temperature | top-p | min-p |
| --- | ---: | ---: | ---: |
| Qwen3 answer-only | 0.7 | 0.8 | 0.0 |
| Qwen3 reasoning trace | 0.6 | 0.95 | 0.0 |
| Llama 3.1 | 0.6 | 0.9 | 0.2 |
