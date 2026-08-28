import logging
import torch
from pathlib import Path
import torch.distributed as dist
import contextlib

logging.basicConfig(
    format="%(asctime)s %(message)s", datefmt="%m/%d/%Y %I:%M:%S %p", level=logging.INFO
)

def get_logger():
    return logging.getLogger(__name__)

def rank_print(rank, msg):
    """helper function to log only on global rank 0"""
    if rank == 0:
        print(msg)

def rank_log(_rank, logger, msg):
    """helper function to log only on global rank 0"""
    if _rank == 0:
        logger.info(f" {msg}")

def verify_min_gpu_count(min_gpus: int = 2) -> bool:
    """ verification that we have at least 2 gpus to run dist examples """
    has_cuda = torch.cuda.is_available()
    gpu_count = torch.cuda.device_count()
    return has_cuda and gpu_count >= min_gpus

def trace_handler(rank, filepath: str, prof: torch.profiler.profile):
    # if rank == 0:
    #     # Construct the trace file.
    #    prof.export_chrome_trace(f"{filepath}.json.gz")

    # Construct the memory timeline file.
    prof.export_memory_timeline(f"{filepath}_{rank}.html", device=rank)

    # dist.barrier()

# Dummy context manager for ranks != 0
@contextlib.contextmanager
def dummy_profiler():
    yield  # Do nothing, just provide a context
