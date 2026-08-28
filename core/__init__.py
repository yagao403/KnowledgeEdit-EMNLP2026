import os
from pathlib import Path

def is_docker():
    if os.path.exists("/.dockerenv"):
        return True
    # Check for "docker" or "kubepods" in cgroup (used by Docker and Kubernetes)
    try:
        with open("/proc/1/cgroup", "rt") as f:
            return any("docker" in line or "kubepods" in line for line in f)
    except FileNotFoundError:
        return False

BASE_PATH = Path(__file__).parent.parent

# All paths are configurable so the public repository does not depend on a
# particular workstation or cluster layout.
DATA_PATH = Path(os.environ.get("DATA_PATH", BASE_PATH / "data"))
MODEL_PATH = Path(os.environ.get("MODEL_PATH", BASE_PATH / "checkpoints"))
ADAPTER_PATH = Path(os.environ.get("ADAPTER_PATH", BASE_PATH / "adapters"))
LOGBOOK_PATH = Path(os.environ.get("LOGBOOK_PATH", BASE_PATH / "logbook"))
PROFILE_PATH = Path(os.environ.get("PROFILE_PATH", BASE_PATH / "web-profiles"))
DATA_COLLECTION_LOGS_PATH = LOGBOOK_PATH / "data_collection_logs"

# Prefer conventional uppercase environment-variable names. Keep the old
# mixed-case spelling as a compatibility fallback for existing experiments.
FictBio_PATH = Path(os.environ.get("FICTBIO_PATH", os.environ.get("FictBio_PATH", DATA_PATH / "FictBio")))
MQuAKE_PATH = Path(os.environ.get("MQUAKE_PATH", DATA_PATH / "MQuAKE"))
ReCoE_PATH = Path(os.environ.get("RECOE_PATH", DATA_PATH / "ReCoE"))
MMLU_PATH = Path(os.environ.get("MMLU_PATH", DATA_PATH / "MMLU" / "all_subjects.json"))

def load_model_maps():
    """
    Load the model map and family map from a TSV file.
    """
    filename = Path(__file__).parent / "model_map.tsv"

    model_map = {}  # short_name -> full_name
    family_map = {}  # short_name, full_name -> family
    with open(filename, encoding='utf-8') as f:
        next(f)  # Skip header
        for line in f:
            parts = line.strip().split('|')
            if len(parts) == 3:
                short_name, full_name, family = parts
                model_map[short_name] = full_name

                family_map[short_name] = family
                family_map[full_name] = family
    return model_map, family_map

MODEL_MAP, FAMILY_MAP = load_model_maps()

# full_name -> short_name
S2_MODEL_NAME = {
    full_name: short_name
    for short_name, full_name in MODEL_MAP.items()
}
