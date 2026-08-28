from __future__ import annotations

from collections import Counter
from functools import partial
import random
from typing import Callable, Iterable, Iterator, Literal, Optional, TYPE_CHECKING

import torch
from torch.utils.data import BatchSampler
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.distributed import DistributedSampler

from training.st_dataset import STDataset, ConcatSTDataset, STStepMessages
from training.utils import TrainingData, Tokenizer

if TYPE_CHECKING:  # Avoid runtime circular imports for type hints
    from training.utils import ValidationData


class InfDataLoader:
    """
    Infinite data loader designed for mp.spawn and DDP.

    New lifecycle (deferred loading):
    - Construct with from_training_data(...): stores only lightweight config.
    - Send this instance to worker processes (it's picklable and LIGHT).
    - In each worker, call reset(dp_rank, dp_size[, seed]) to LOAD data and
      build the sampler tied to that rank. Iteration starts afterwards.

    Attempting to iterate before calling reset will raise a RuntimeError.
    """

    def __init__(self) -> None:
        # Heavy state created in reset(...)
        self._dataset: Optional[ConcatSTDataset] = None
        self._collate_fn: Optional[Callable[..., dict]] = None
        self._sampler: Optional[Iterable[list[int]]] = None

        # Lightweight construction config captured by from_training_data(...)
        self.data: Optional[TrainingData] = None
        self._cfg_only_student: Optional[bool] = None
        self._cfg_tokenizer: Optional[Tokenizer] = None
        self._cfg_sampling_mode: Optional[Literal["standard", "batch_optimized", "episodic"]] = None
        self._cfg_tags_assigner: Callable[[STStepMessages], set[str]] | None = None
        self._cfg_verbose: bool = True
        self._cfg_seed: int = 42

        # Lazily created persistent iterator over batch indices; not pickled
        self._idx_iter: Optional[Iterator[list[int]]] = None

    def __iter__(self) -> InfDataLoader:
        return self

    def __next__(self) -> dict:
        # Ensure reset(...) was called inside the worker
        if self._dataset is None or self._collate_fn is None or self._sampler is None:
            raise RuntimeError(
                "InfDataLoader is not initialized. Call reset(dp_rank, dp_size[, seed]) in the worker before iterating."
            )
        # Ensure we have a persistent iterator so we don't restart the RNG sequence.
        # The sampler itself is constructed on-demand here (first use in-process)
        # to satisfy the constraints discussed above.
        if self._idx_iter is None:
            self._idx_iter = iter(self._sampler)
        indices = next(self._idx_iter)  # sampler is infinite under normal operation
        assert self._dataset is not None and self._collate_fn is not None
        samples = [self._dataset[i] for i in indices]
        return self._collate_fn(samples)

    # Optional convenience to get a single batch
    def next_batch(self) -> dict:
        return next(self)

    def __len__(self) -> int:
        if self._dataset is None:
            raise RuntimeError("Dataset not loaded yet. Call reset(...) first.")
        return len(self._dataset)

    def to_dict(self) -> dict | None:
        """
        Lightweight serialization of the training data configuration used by
        this loader. Returns the same structure as TrainingData.to_dict().

        Returns None if the loader hasn't been configured via
        from_training_data(...).
        """
        if self.data is None:
            return None
        # Keep the exact structure produced by TrainingData.to_dict()
        return self.data.to_dict()

    # Make pickling safe for mp.spawn by dropping non-picklable iterator state
    def __getstate__(self):
        # Drop non-picklable iterator state so the loader can be sent across
        # processes (e.g., via mp.spawn). The sampler will be rebuilt lazily
        # on first use in the target process.
        state = self.__dict__.copy()
        state["_idx_iter"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

    @classmethod
    def from_training_data(
        cls,
        data: TrainingData,
        only_student: bool,
        tokenizer: Tokenizer,
        sampling_mode: Literal["standard", "batch_optimized", "episodic"] = "standard",
        *,
        verbose: bool = True,
        seed: int = 42,
    ) -> "InfDataLoader":
        """
        Create a lightweight, picklable loader that defers data loading.

        The returned object contains no heavy dataset/sampler state and can be
        safely sent to worker processes. Call reset(dp_rank, dp_size[, seed])
        inside each worker to actually load data and build the sampler.
        """
        assert sampling_mode in ["standard", "batch_optimized", "episodic"]

        self = cls()
        self.data = data
        self._cfg_only_student = only_student
        self._cfg_tokenizer = tokenizer
        self._cfg_sampling_mode = sampling_mode
        self._cfg_verbose = bool(verbose)
        self._cfg_seed = int(seed)
        return self

    def reset(self, dp_rank: int, dp_size: int, seed: Optional[int] = None) -> int:
        """
        Load data and build sampler for a specific worker rank.

        Returns the dataset length (number of samples) after loading, which can
        be used for scheduling if needed.
        """
        # Validate config presence
        if self.data is None or self._cfg_tokenizer is None or self._cfg_sampling_mode is None or self._cfg_only_student is None:
            raise RuntimeError("InfDataLoader.reset() called before from_training_data().")

        data = self.data
        tokenizer = self._cfg_tokenizer
        only_student = self._cfg_only_student
        sampling_mode = self._cfg_sampling_mode
        verbose = self._cfg_verbose
        seed = int(self._cfg_seed if seed is None else seed)

        # 1) Build per-group datasets (HEAVY) — done inside worker
        datasets: list[STDataset] = []
        for _group, paths in data.group_xml_paths.items():
            ds = STDataset.from_xml_paths(
                tokenizer,
                paths,
                verbose=verbose,
                student_dropout_rate=data.student_dropout_rate,
                max_teacher_seq_len=data.max_teacher_seq_len,
                tags_assigner=data.tags_assigner,
                add_filename_tags=data.add_filename_tags,
            )
            datasets.append(ds)

        # 2) Concat into a single dataset
        train_dataset = ConcatSTDataset(
            datasets,
            tokenizer=tokenizer,
            student_dropout_rate=data.student_dropout_rate,
        )

        # 3) Per-group checks and optional loss weights/tags
        if data.group_loss_weights is not None:
            for (group, ds) in zip(data.group_xml_paths.keys(), datasets, strict=True):
                ds.set_loss_weight(data.group_loss_weights[group])
                ds.add_tags_to_all_exercises({group})

        for (group, ds) in zip(data.group_xml_paths.keys(), datasets, strict=True):
            if len(ds) == 0:
                raise ValueError(f"No logit training data available for {group} group")

        if verbose:
            print("Number of training samples:", len(train_dataset), flush=True)

        assert data.group_mix_props is not None, "group_mix_props must be provided"

        collate_fn: Callable[..., dict] = partial(
            train_dataset.collate_fn,
            padding_value=0,
            div8=True,
            only_student=only_student,
        )

        # Precompute compact metadata needed for sampler construction
        data_lengths: list[int] = [len(ds) for ds in datasets]

        # Build sampler directly here, using class constructors
        if sampling_mode == "standard":
            sampler = SyncInfiniteBatchSampler.from_data(
                data=data,
                data_lengths=data_lengths,
                dp_size=int(dp_size),
                seed=seed,
                dp_rank=int(dp_rank),
            )
        elif sampling_mode == "episodic":
            sampler = SyncInfiniteShuffledBatchSampler.from_data(
                data=data,
                data_lengths=data_lengths,
                dp_size=int(dp_size),
                seed=seed,
                dp_rank=int(dp_rank),
            )
        else:
            # batch_optimized requires per-example teacher seq lengths
            teacher_seq_lengths: list[list[int]] = []
            for ds in datasets:
                assert isinstance(ds, STDataset)
                # All exercises must have teacher_seq_len populated
                assert all(ex.teacher_seq_len is not None for ex in ds.exercises)
                teacher_seq_lengths.append([int(ex.teacher_seq_len) for ex in ds.exercises])  # type: ignore[attr-defined]

            sampler = GroupBinBatchSampler.from_data(
                data=data,
                data_lengths=data_lengths,
                teacher_seq_lengths=teacher_seq_lengths,
                dp_size=int(dp_size),
                seed=seed,
                dp_rank=int(dp_rank),
            )

        # Commit heavy state and reset iterator stream
        self._dataset = train_dataset
        self._collate_fn = collate_fn
        self._sampler = sampler
        self._idx_iter = None

        return len(train_dataset)

    # Convenience properties exposing config-sourced batch parameters
    @property
    def micro_batch_size(self) -> int:
        if self.data is None:
            raise RuntimeError("InfDataLoader not configured; call from_training_data(...) first.")
        return int(self.data.micro_batch_size)

    @property
    def n_micro_batches_in_batch(self) -> int:
        if self.data is None:
            raise RuntimeError("InfDataLoader not configured; call from_training_data(...) first.")
        return int(self.data.n_micro_batches_in_batch)


class SyncInfiniteBatchSampler(BatchSampler):
    """
    Synchronized infinite batch sampler that samples full batches of indices
    (with replacement) according to the provided weights.
    """
    @classmethod
    def from_data(
        cls,
        *,
        data: TrainingData,
        data_lengths: list[int],
        dp_size: int,
        seed: int | None,
        dp_rank: int,
    ) -> "SyncInfiniteBatchSampler":
        """
        Build per-sample weights from group mix proportions and dataset sizes,
        then construct the sampler.
        """
        assert data.group_mix_props is not None, "group_mix_props must be provided"
        group_mix_props: dict[str, float] = data.group_mix_props
        total = sum(group_mix_props.values())
        group_mix_props_normalized = {
            group: float(group_mix_props[group]) / float(total)
            for group in group_mix_props.keys()
        }

        sample_weights = torch.concat(
            [
                torch.full(
                    (int(L),),
                    group_mix_props_normalized[group] / max(1, int(L)),
                    dtype=torch.float,
                )
                for group, L in zip(data.group_xml_paths.keys(), data_lengths, strict=True)
            ]
        )

        return cls(
            weights=sample_weights,
            batch_size=data.micro_batch_size,
            dp_size=dp_size,
            dp_rank=dp_rank,
            seed=seed,
        )

    def __init__(
        self,
        weights,
        batch_size: int,
        dp_size: int = 1,
        dp_rank: int = 0,
        *,
        seed: int | None = None
    ):
        """
        Args:
            weights (Tensor | list[float]): Per-sample weights
            seed (int | None): Base seed.
            batch_size (int): Micro batch size
            dp_size (int): Data-parallel size
            dp_rank (int): Data-parallel rank of this sampler instance
        """
        self.weights = weights
        self.local_batch_size = int(batch_size)
        self.global_batch_size = int(batch_size) * int(dp_size)
        self.seed = seed
        self.dp_size = dp_size
        self.dp_rank = dp_rank
        self.generator = torch.Generator()
        if self.seed is not None:
            self.generator.manual_seed(self.seed)
            random.seed(self.seed)  # IMPORTANT: prevents hangs observed without it

    def __iter__(self):
        while True:
            # Draw len(self.weights) samples (for efficiency), then yield in batch chunks
            idxs = torch.multinomial(
                self.weights,
                num_samples=len(self.weights),
                replacement=True,
                generator=self.generator,
            ).tolist()

            # Yield batches of batch_size
            for i in range(
                0,
                len(self.weights)//self.global_batch_size * self.global_batch_size,
                self.global_batch_size
            ):
                global_yield = idxs[i:i + self.global_batch_size]
                local_yield = global_yield[self.dp_rank*self.local_batch_size:(self.dp_rank+1)*self.local_batch_size]
                assert len(local_yield) == self.local_batch_size and len(global_yield) == self.global_batch_size
                yield local_yield

    def __len__(self):
        return float("inf")


class SyncInfiniteShuffledBatchSampler(BatchSampler):
    """
    Synchronized infinite batch sampler that samples full batches of indices
    (without replacement) according to the provided weights.
    """
    @classmethod
    def from_data(
        cls,
        *,
        data: TrainingData,
        data_lengths: list[int],
        dp_size: int,
        seed: int | None,
        dp_rank: int,
    ) -> "SyncInfiniteShuffledBatchSampler":
        assert data.group_mix_props is not None, "group_mix_props must be provided"
        group_names = list(data.group_xml_paths.keys())
        group_offsets = [int(L) for L in data_lengths]
        # compute cumulative offsets
        cum_off = [0]
        for L in group_offsets[:-1]:
            cum_off.append(cum_off[-1] + L)

        group_global_indices = {
            group_name: list(range(cum_off[i], cum_off[i] + int(data_lengths[i])))
            for i, group_name in enumerate(group_names)
        }

        return cls(
            group_mix_probs=data.group_mix_props,
            group_global_indices=group_global_indices,
            batch_size=data.micro_batch_size,
            dp_size=dp_size,
            dp_rank=dp_rank,
            seed=seed,
        )

    def __init__(
        self,
        group_mix_probs: dict[str, float],
        group_global_indices: dict[str, list[int]],
        batch_size: int,
        dp_size: int = 1,
        dp_rank: int = 0,
        *,
        seed: int | None = None
    ):
        """
        Args:
            group_mix_probs (dict[str, float]): Per-group mixing proportions (will be normalized)
            group_global_indices (dict[str, list[int]]): Per-group list of GLOBAL indices in Dataset space.
            seed (int | None): Base seed.
            batch_size (int): Micro batch size
            dp_size (int): Data-parallel size
            dp_rank (int): Data-parallel rank of this sampler instance
        """
        self.group_mix_probs = group_mix_probs
        self.local_batch_size = int(batch_size)
        self.global_batch_size = int(batch_size) * int(dp_size)
        self.group_global_indices = group_global_indices
        self.seed = seed
        self.dp_size = dp_size
        self.dp_rank = dp_rank
        self.generator = torch.Generator()
        if self.seed is not None:
            self.generator.manual_seed(self.seed)
            random.seed(self.seed)  # IMPORTANT: prevents hangs observed without it

        # Normalize group mix probs
        gprobs = torch.tensor(list(self.group_mix_probs.values()), dtype=torch.float)
        self.group_probs = gprobs / gprobs.sum()
        self.group_name_to_group_id_map = {g: i for i, g in enumerate(self.group_mix_probs.keys())}

        # Create pools
        assert set(self.group_mix_probs.keys()) == set(self.group_global_indices.keys())
        self.pools: dict[str, list[int]] = {group_name: [] for group_name in self.group_mix_probs.keys()}

    @property
    def pool_sizes(self):
        return {group: len(self.pools[group]) for group in self.group_mix_probs.keys()}

    def _to_group_name(self, group_id: int) -> str:
        return list(self.group_mix_probs.keys())[group_id]

    def _check_and_refill_pools(self, num_requested_ids_per_group: dict[str, int]):
        """
        Check if the pools for the given groups need to be refilled based on the number of requested ids.
        If any pool has fewer ids than requested, refill it.
        """
        to_be_refilled = []
        for group in num_requested_ids_per_group:
            if len(self.pools[group]) < num_requested_ids_per_group[group]:
                to_be_refilled.append(group)
        if to_be_refilled:
            for group in to_be_refilled:
                self._refill(group, min_size=num_requested_ids_per_group[group])

    def _refill(self, group_name: str, min_size: int):
        """
        Refill the pool for the given group name by shuffling the global indices of that group
        and extending its pool until it is full again.
        """
        while len(self.pools[group_name]) < min_size:
            indices = self.group_global_indices[group_name]
            perm = torch.randperm(len(indices), generator=self.generator).tolist()
            shuffled_indices = [indices[i] for i in perm]
            self.pools[group_name].extend(shuffled_indices)

    def __iter__(self):
        while True:
            # Draw self.global_batch_size samples using self.group_probs to determine the classes of each sample within the batch
            chosen_group_ids = torch.multinomial(
                self.group_probs,
                num_samples=self.global_batch_size,
                replacement=True,
                generator=self.generator,
            ).tolist()
            chosen_groups = [self._to_group_name(g) for g in chosen_group_ids]

            # Count number of samples requested per group
            group_counts = dict(Counter(chosen_groups))
            self._check_and_refill_pools(group_counts)
            # Collect sampled indices from pools
            idxs = []
            for group in chosen_groups:
                sampled_idx = self.pools[group].pop(0)
                idxs.append(sampled_idx)

            # Yield batches of batch_size
            local_yield = idxs[self.dp_rank*self.local_batch_size:(self.dp_rank+1)*self.local_batch_size]
            assert len(local_yield) == self.local_batch_size and len(idxs) == self.global_batch_size
            yield local_yield

    def __len__(self):
        return float("inf")


class GroupBinBatchSampler(BatchSampler):
    """
    Synchronized infinite batch sampler that:
      1) samples a group according to provided group_probs,
      2) samples a bin within that group with probability proportional to its size,
      3) samples a full batch of indices from that bin (no replacement within batch if possible).

    Args:
        group_bin_indices: list over groups -> list over bins -> list of GLOBAL indices in. ConcatDataset space
        group_probs: per-group mixing proportions (will be normalized)
        batch_size: micro batch size
        dp_size: data-parallel size
        dp_rank: data-parallel rank of this sampler instance
    """
    @classmethod
    def from_data(
        cls,
        *,
        data: TrainingData,
        data_lengths: list[int],
        teacher_seq_lengths: list[list[int]],
        dp_size: int,
        seed: int | None,
        dp_rank: int,
    ) -> "GroupBinBatchSampler":
        assert data.group_mix_props is not None, "group_mix_props must be provided"

        # min bin capacity = 4 × global batch (micro × dp_size)
        min_bin_size = 4 * data.micro_batch_size * dp_size

        # Compute ConcatDataset offsets per group using provided lengths
        group_offsets = [0]
        for L in data_lengths[:-1]:
            group_offsets.append(group_offsets[-1] + int(L))

        # Build bins per group in GLOBAL index space
        group_bins_global: list[list[list[int]]] = []
        for gi, lengths in enumerate(teacher_seq_lengths):
            local_bins = build_length_bins([int(x) for x in lengths], min_bin_size)
            off = group_offsets[gi]
            global_bins = [[off + idx for idx in b] for b in local_bins]
            assert all(len(b) > 0 for b in global_bins), f"Empty bin found in group {gi}"
            assert len(global_bins) > 0, f"No bins found in group {gi}"
            group_bins_global.append(global_bins)

        # Normalize group probs
        group_names = list(data.group_xml_paths.keys())
        total = sum(float(data.group_mix_props[g]) for g in group_names)
        group_probs = [float(data.group_mix_props[g]) / float(total) for g in group_names]

        return cls(
            group_bin_indices=group_bins_global,
            group_probs=group_probs,
            batch_size=data.micro_batch_size,
            dp_size=dp_size,
            dp_rank=dp_rank,
            seed=seed,
        )

    def __init__(
        self,
        group_bin_indices: list[list[list[int]]],
        group_probs: list[float],
        batch_size: int,
        dp_size: int = 1,
        dp_rank: int = 0,
        *,
        seed: int | None = None,
    ):
        assert len(group_bin_indices) == len(group_probs) and len(group_bin_indices) > 0
        self.group_bin_indices = group_bin_indices
        self.local_batch_size = int(batch_size)
        self.global_batch_size = int(batch_size) * int(dp_size)
        self.dp_size = dp_size
        self.dp_rank = dp_rank

        gprobs = torch.tensor(group_probs, dtype=torch.float)
        self.group_probs = gprobs / gprobs.sum()

        # Precompute per-group bin probs ~ bin size
        self.bin_probs_per_group = []
        for bins in self.group_bin_indices:
            sizes = torch.tensor([len(b) for b in bins], dtype=torch.float)
            probs = sizes / sizes.sum() if sizes.sum() > 0 else torch.ones_like(sizes) / max(1, len(sizes))
            self.bin_probs_per_group.append(probs)

        # RNG
        self.seed = seed
        self.generator = torch.Generator()
        if self.seed is not None:
            self.generator.manual_seed(self.seed)
            random.seed(self.seed)  # IMPORTANT: prevents hangs observed without it

    def __iter__(self):
        while True:
            # 1) choose group
            g = torch.multinomial(self.group_probs, 1, generator=self.generator).item()
            g = int(g)

            # 2) choose bin proportional to its size
            bin_probs = self.bin_probs_per_group[g]
            b = torch.multinomial(bin_probs, 1, generator=self.generator).item()
            b = int(b)
            pool = self.group_bin_indices[g][b]
            L = len(pool)

            # 3) sample within bin; no replacement
            assert L >= self.global_batch_size
            perm = torch.randperm(L, generator=self.generator).tolist()
            take = perm[: self.global_batch_size]
            global_idxs = [pool[i] for i in take]
            idxs = global_idxs[self.dp_rank*self.local_batch_size:(self.dp_rank+1)*self.local_batch_size]
            yield idxs

    def __len__(self):
        return float("inf")

def build_length_bins(lengths: list[int], min_bin_size: int) -> list[list[int]]:
    """
    Build quantile-like bins of indices, sorted by sequence length.
    We choose the maximum number of bins such that each bin has at least `min_bin_size`.

    Returns a list of bins, each a list of local indices within a group dataset.
    """
    n = len(lengths)
    if n == 0:
        return []

    # Sort indices by length (ascending)
    sorted_idxs = sorted(range(n), key=lambda i: lengths[i])

    # Max number of bins s.t. each bin has at least min_bin_size
    n_bins = max(1, n // max(1, min_bin_size))

    # Split sorted indices into n_bins chunks as evenly as possible
    base = n // n_bins
    rem = n % n_bins
    sizes = [base + (1 if i < rem else 0) for i in range(n_bins)]
    bins = []
    start = 0
    for sz in sizes:
        bins.append(sorted_idxs[start:start + sz])
        start += sz
    # # Defensive: filter out any empty bins if min_bin_size > n
    # bins = [b for b in bins if len(b) > 0]
    assert all(len(b) > 0 for b in bins), "Empty bin detected, this is unexpected and might indicate a bug"
    return bins


# -----------------------------
# Validation data loaders (finite)
# -----------------------------

class ValidationDataLoader:
    """
    Finite validation data loader built in the same spirit as InfDataLoader:
    - Lightweight, picklable configuration via from_validation_data(...)
    - Heavy dataset/sampler/dataloader are constructed in reset(dp_rank, dp_size)

    Iterates over a standard torch DataLoader backed by DistributedSampler.
    """

    def __init__(self) -> None:
        # Heavy state (created in reset)
        self._dataset: Optional[STDataset] = None
        self._sampler: Optional[DistributedSampler] = None
        self._dataloader: Optional[DataLoader] = None

        # Lightweight construction config
        self.data: Optional["ValidationData"] = None
        self._cfg_only_student: Optional[bool] = None
        self._cfg_tokenizer: Optional[Tokenizer] = None
        self._cfg_verbose: bool = True

    @classmethod
    def from_validation_data(
        cls,
        val_data: ValidationData,
        only_student: bool,
        tokenizer: Tokenizer,
        *,
        verbose: bool = True,
    ) -> ValidationDataLoader:
        self = cls()
        self.data = val_data
        self._cfg_only_student = only_student
        self._cfg_tokenizer = tokenizer
        self._cfg_verbose = bool(verbose)
        return self

    def reset(self, dp_rank: int, dp_size: int) -> int:
        if self.data is None or self._cfg_tokenizer is None or self._cfg_only_student is None:
            raise RuntimeError("ValidationDataLoader.reset() called before from_validation_data().")

        val_data = self.data
        tokenizer = self._cfg_tokenizer
        only_student = self._cfg_only_student

        # Build dataset (heavy)
        val_dataset = STDataset.from_xml_paths(
            tokenizer,
            val_data.xml_paths,
            student_dropout_rate=0.5,
            verbose=self._cfg_verbose,
            div=8,
            tags_per_xml=val_data.tags_per_xml,
            add_filename_tags=val_data.add_filename_tags,
            tags_assigner=val_data.tags_assigner,
        )

        # Handle empty validation set as error (mirrors previous behavior)
        if len(val_dataset) == 0:
            raise ValueError("No validation data found in val_data.")

        # Sampler and torch DataLoader
        val_sampler = DistributedSampler(
            val_dataset,
            num_replicas=int(dp_size),
            rank=int(dp_rank),
            shuffle=False,
        )

        collate = partial(
            val_dataset.collate_fn,
            padding_value=0,
            div8=True,
            only_student=only_student,
        )

        val_dataloader = DataLoader(
            val_dataset,
            batch_size=val_data.micro_batch_size,
            collate_fn=collate,
            shuffle=False,
            sampler=val_sampler,
            num_workers=1,
            persistent_workers=True,
        )

        # Commit heavy state
        self._dataset = val_dataset
        self._sampler = val_sampler
        self._dataloader = val_dataloader

        return len(val_dataset)

    def __iter__(self) -> Iterator[dict]:
        assert self._dataloader is not None, "ValidationDataLoader not initialized. Call reset(...) first."
        return iter(self._dataloader)

    def __len__(self) -> int:
        if self._dataset is None:
            raise RuntimeError("Dataset not loaded yet. Call reset(...) first.")
        return len(self._dataset)

    def to_dict(self) -> dict | None:
        """
        Lightweight serialization of the validation data configuration used by
        this loader. Returns the same structure as ValidationData.to_dict().

        Returns None if the loader hasn't been configured via
        from_validation_data(...).
        """
        if self.data is None:
            return None
        return self.data.to_dict()


class OfflineValidationDataLoader(ValidationDataLoader):
    """
    Convenience variant for offline evaluation where dp_size and dp_rank are
    known at construction time. Call reset() without arguments to build the
    underlying DataLoader.
    """

    def __init__(self) -> None:
        super().__init__()
        self._cfg_dp_rank: Optional[int] = None
        self._cfg_dp_size: Optional[int] = None

    @classmethod
    def from_offline_validation_data(
        cls,
        val_data: ValidationData,
        only_student: bool,
        tokenizer: Tokenizer,
        *,
        verbose: bool = True,
        dp_size: int = 1,
        dp_rank: int = 0,
    ) -> OfflineValidationDataLoader:
        self = cls()
        self.data = val_data
        self._cfg_only_student = only_student
        self._cfg_tokenizer = tokenizer
        self._cfg_verbose = bool(verbose)
        self._cfg_dp_size = int(dp_size)
        self._cfg_dp_rank = int(dp_rank)
        return self

    def reset(self) -> int:  # type: ignore[override]
        if self._cfg_dp_rank is None or self._cfg_dp_size is None:
            raise RuntimeError("OfflineValidationDataLoader.reset() requires preconfigured dp_rank and dp_size.")
        return super().reset(self._cfg_dp_rank, self._cfg_dp_size)

