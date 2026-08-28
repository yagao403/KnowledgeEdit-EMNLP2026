# Edit Knowledge, Not Just Facts

Official code for **Edit Knowledge, Not Just Facts via Multi-Step Reasoning over Background Stories**.

We propose a self-/context-distillation framework for continual knowledge editing, designed particularly for reasoning models. The method turns knowledge updating from a memorization problem into a “learn to reason” problem: new knowledge is internalized into the model’s reasoning process, enabling it to flexibly use updated knowledge in downstream tasks while continually incorporating new updates.

Paper: [arXiv:2602.02028](https://arxiv.org/abs/2602.02028)

## Repository layout

```text
core/                         Custom vLLM server/client and shared data types
training/                     Context-distillation and SFT training framework
tensor_parallelism/           FSDP/tensor-parallel training support
tasks/t_knowledge_edit/       Data construction, baselines, and evaluation
configs/                      Public training entry points
scripts/                      Launch examples
```

See [`tasks/t_knowledge_edit/README.md`](tasks/t_knowledge_edit/README.md) for the experiment-to-script map and expected data layout.

## Installation

Python 3.10 is used by the original experiments. Create an environment and install the components you need:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[train,evaluate]'
```

For the custom inference server, also install `.[serve]`. Install vLLM
separately on a compatible CUDA or ROCm machine; the retained server targets
the vLLM 0.11.x API used by the original runs. On ROCm, install the monitoring
dependency too:

```bash
pip install -e '.[train,evaluate,serve,rocm]'
```

The training code was run on four AMD MI250X accelerators (eight logical ROCm
devices). Other distributed setups may require changes to the launcher or
tensor-parallel configuration.

Copy `.env.example` to `.env` or export the variables in your shell. Dataset roots and output paths are configurable; no workstation-specific paths are required.

## Important: IPython task workflow

The experiment drivers in `tasks/t_knowledge_edit/` are **IPython percent-format scripts**, not conventional command-line programs. Open them in an editor with IPython cell support and run the cells in order. Running an entire evaluation file with ordinary `python` may fail with `SyntaxError: 'await' outside function` or immediately start a full experiment.

Load the path configuration before importing `core` in the first cell of a new IPython session:

```python
%load_ext dotenv
%dotenv -o .env
```

## vLLM server and client

The repository includes the minimal custom server required by the retained
task scripts. It understands this project's serialized messages and is not the
standard vLLM OpenAI-compatible endpoint. The paper-only implementation uses
vLLM V1, supports text models, and exposes only `GET /health` and
`POST /generate`. Start it on a compatible GPU host:

```bash
python -m core.server \
  --model Qwen/Qwen3-32B \
  --tokenizer Qwen/Qwen3-32B \
  --dtype auto \
  --tensor-parallel-size 8
```

Then point the client at it:

```python
from core.client import Client
from core.message import Message

client = Client(model="qwen3-32b", base_url="http://localhost:8000")
response = client.sync_call([Message("user", "What is knowledge editing?")])
print(response)
```

The task scripts use the same client for question generation, teacher-response collection, and evaluation.

## Training

Training examples are XML teacher/student conversations. A minimal single-node launch is:

```bash
python -m configs.train_distillation \
  --train 'knowledge_edit/fictbio/train/**/*.xml' \
  --validation 'knowledge_edit/fictbio/validation/**/*.xml' \
  --dataset fictbio \
  --base-model llama3.1-70b \
  --run-name fictbio-llama3.1-70b \
  --checkpoint-interval 240 \
  --val-interval 320 \
  --log-interval 1
```

Paths supplied to `--train` and `--validation` are resolved relative to `DATA_PATH` unless absolute. Use `--dry-run` to validate configuration and file discovery without loading a model.

Dataset-specific checkpoint,
validation, and logging intervals remain explicit command-line arguments.

## Evaluation and baselines

The retained task code covers data collection and evaluation, MMLU, ICE, single-hop training, and AlphaEdit-compatible evaluation.

## Citation

```bibtex
@article{gao2026edit,
  title={Edit Knowledge, Not Just Facts via Multi-Step Reasoning over Background Stories},
  author={Gao, Ya and Kujanpää, Kalle and Marttinen, Pekka and Valpola, Harri and Ilin, Alexander},
  journal={arXiv preprint arXiv:2602.02028},
  year={2026}
}
```