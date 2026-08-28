"""Shared model selection for legacy experiment drivers.

Set ``KNOWLEDGE_EDIT_MODEL`` to the vLLM model name or adapter identifier.
Command-line entry points should prefer an explicit ``--model`` argument.
"""

import os


MODEL_NAME = os.environ.get("KNOWLEDGE_EDIT_MODEL", "qwen3-32b")
