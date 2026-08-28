from __future__ import annotations

from pathlib import Path
import random
import xml.etree.ElementTree as ET
import tempfile
import fsspec
from typing import Callable

from dataclasses import dataclass
import torch
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F

from core.tokenizer import Tokenizer
from core.steps import STStepMessages, get_steps_from_run


class STExercise:
    def __init__(
        self,
        step: STStepMessages,
        tags: set[str] | None = None,  # Classification tags for the exercise
    ):
        step_or_none = step.strip_after_last_target()
        if step_or_none is None:
            raise ValueError(f"Step does not contain a target\n{step.metadata}")
        else:
            step = step_or_none
        self.step = step
        self.tags = tags or set()
        self.drop = False
        # Instance attribute set later by dataset when weighting is enabled
        self.loss_weight: float | None = None
        self.teacher_seq_len: int | None = None  # Optional attribute to store precomputed teacher seq len

    @classmethod
    def from_string(cls, xml_string: str, tags: set[str] | None = None) -> STExercise:
        step = STStepMessages.from_string(xml_string)
        return cls(step, tags)

    @classmethod
    def from_element(cls, element: ET.Element, tags: set[str] | None = None) -> STExercise:
        step = STStepMessages.from_element(element)
        return cls(step, tags)

    @property
    def contains_target(self) -> bool:
        return self.step.contains_target

    def to_sample(self, tokenizer: Tokenizer, dropout_rate: float = 0.5) -> dict:
        """Convert to a sample that contains tokens.
        Random dropout if there are sections or messages with student_dropout.

        by_section (bool): Whether to tokenize sections separately.

        This is a version in which sections are tokenized separately.
        """

        step = self.step.apply_dropout(dropout_rate)
        def get_token_ids_and_mask(teacher: bool, dropout_rate: float = 1.) -> tuple[torch.Tensor, torch.Tensor]:
            messages = step.to_messages(teacher=teacher, dropout_rate=dropout_rate)
            token_ids, target_mask = tokenizer.tokenize_for_training(messages)

            # Remove all tokens after the last target token
            mask = get_mask(target_mask)
            token_ids, target_mask = token_ids[mask], target_mask[mask]

            return token_ids.unsqueeze(0), target_mask.unsqueeze(0)

        student_seq, student_mask = get_token_ids_and_mask(teacher=False, dropout_rate=dropout_rate)
        teacher_seq, teacher_mask = get_token_ids_and_mask(teacher=True)

        return {
            "student_seq": student_seq,  # (1, student_seq_len)
            "student_mask": student_mask,  # (1, student_seq_len)
            "teacher_seq": teacher_seq,  # (1, teacher_seq_len)
            "teacher_mask": teacher_mask,  # (1, teacher_seq_len)
            "tags": self.tags,  # set[str]
        }

    def apply_dropout(self, dropout_rate: float) -> STExercise:
        return STExercise(self.step.apply_dropout(dropout_rate), self.tags)

    def get_student_seq_len(self, tokenizer: Tokenizer) -> int:
        # This function is meant to be used for deterministic exercises, so dropout_rate is set to 0
        sample = self.to_sample(tokenizer, dropout_rate=0)
        return sample["student_seq"].shape[1]

    def __str__(self):
        return str(self.step)

    def __repr__(self):
        return str(self.step)

    def to_string(self):
        return self.step.to_string()


def get_mask(target_mask: torch.Tensor) -> torch.Tensor:
    """Get the mask of all position before the last target token.

    Args:
        target_mask: A tensor of shape (seq_len,)

    TODO: this function is very similar to get_attention_mask in losses.py.
    """
    dims = [0]
    dim = target_mask.dim() - 1

    # flip so the last 1 becomes the first 1
    y = torch.flip(target_mask, dims=dims).to(torch.int)
    y_filled = (y.cumsum(dim=dim) > 0)

    # flip back to original order
    mask = torch.flip(y_filled, dims=dims)
    return mask


def read_exercises(
    xml_path: Path,
    must_exist: bool = False,  # Raise error if file does not exist
    must_parse: bool = False,  # Raise error if parsing fails
    verbose: bool = True,
    tags: set[str] | None = None,
    add_filename_tags: bool = True,
    tags_assigner: Callable[[STStepMessages], set[str]] | None = None,
) -> list[STExercise]:
    from core.steps import get_multiple_steps_from_xml
    steps = get_multiple_steps_from_xml(xml_path, must_exist, must_parse, verbose=verbose, only_with_target=True)

    step_tags = [tags or set() for _ in range(len(steps))]

    if tags_assigner is not None:
        for step, step_tag in zip(steps, step_tags):
            assigned_tags = tags_assigner(step)
            step_tag.update(assigned_tags)
            # If no tags were assigned, and add_filename_tags is True, add the filename tag
            # If tags were assigned, we do not need to add the filename tag again
            if not assigned_tags:
                step_tag.add(xml_path.stem)
    elif add_filename_tags:
        for step_tag in step_tags:
            step_tag.add(xml_path.stem)
    return [STExercise(step, tags=tags) for step, tags in zip(steps, step_tags)]

@dataclass
class ZipTemplate:
    zip_path: Path
    pattern: str  # e.g. "*.xml" or "data/**/*.json"

    def __post_init__(self):
        self.zip_path = Path(self.zip_path)

    def __repr__(self):
        return f"ZipTemplate(zip_path={self.zip_path!s}, pattern={self.pattern!r})"

    @classmethod
    def from_path(cls, s: str | Path) -> ZipTemplate:
        if isinstance(s, Path):
            s = str(s)
        # split at first ".zip"
        zip_path, pattern = s.split(".zip", 1)
        return cls(Path(zip_path + ".zip"), pattern)

def resolve_zip_exercise_glob_and_move_to_tmp(zip_path: Path, pattern: str) -> tuple[list[Path], tempfile.TemporaryDirectory]:
    """Resolve a glob path inside a zip file.
    Example inputs:
    zip_path = Path("/path/to/file.zip")
    pattern = "data/*.xml"

    Returns:
        - list of Paths to temporary files extracted from the zip
        - TemporaryDirectory object that should be cleaned up later
    """
    assert zip_path.suffix == ".zip" and zip_path.exists()
    fs = fsspec.filesystem("zip", fo=str(zip_path))
    all_files = fs.glob(pattern)
    tmp_dir = tempfile.TemporaryDirectory(dir="/tmp", prefix="exercise-")  # e.g. /tmp/exercise-ad2e23dp4
    tmp_filepaths = []
    for file in all_files:
        if file.endswith(".xml"):
            dest_path = Path(tmp_dir.name) / Path(file).name  # e.g. /tmp/exercise-ad2e23dp4/data/file.xml
            fs.get(file, str(dest_path))
            tmp_filepaths.append(dest_path)
    return tmp_filepaths, tmp_dir

class STDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        tokenizer: Tokenizer,
        exercises: list[STExercise],
        student_dropout_rate: float,
        div: int | None = None,  # The dataset size must be divisible by this number
    ):

        self.tokenizer = tokenizer
        self.student_dropout_rate = student_dropout_rate
        self.exercises = exercises
        self.exercise_ids = list(range(len(exercises)))

        if div is not None:
            # Add extra exercises to make the dataset size divisible by div
            if (remainder := len(exercises) % div) != 0:
                print()
                ex = STExercise(exercises[-1].step, tags=exercises[-1].tags)
                ex.drop = True  # Drop this exercise from metric calculations
                self.exercises.extend([ex] * (div - remainder))

    def set_loss_weight(self, weight: float):
        for exercise in self.exercises:
            exercise.loss_weight = weight

    def add_tags_to_all_exercises(self, tags: set[str]):
        for exercise in self.exercises:
            exercise.tags.update(tags)

    @classmethod
    def from_xml_paths(
        cls,
        tokenizer: Tokenizer,
        xml_paths: list[Path],
        student_dropout_rate: float,
        verbose: bool = False,
        strict: bool = False,
        max_teacher_seq_len: int = 8192,
        div: int | None = None,
        tags_per_xml: list[set[str]] | None = None,  # Tags to assign to exercises from each XML file
        add_filename_tags: bool = True,
        tags_assigner: Callable[[STStepMessages], set[str]] | None = None,
    ) -> STDataset:
        if tags_per_xml is not None:
            assert len(tags_per_xml) == len(xml_paths), "tags_per_xml must have the same length as xml_paths"

        def my_print(*args, **kw):
            if verbose:
                print(*args, **kw)

        my_print("==== StudentTeacherDataset ====", flush=True)
        all_exercises: list[STExercise] = []
        filepaths = list(xml_paths)
        if len(filepaths) == 0:
            my_print("No exercise files found")
            return cls(tokenizer, all_exercises, student_dropout_rate, div=div)

        for i, filepath in enumerate(filepaths):
            tags = tags_per_xml[i] if tags_per_xml is not None else None
            exercises = read_exercises(
                filepath, must_exist=strict, must_parse=strict, verbose=verbose,
                tags=tags, add_filename_tags=add_filename_tags,
                tags_assigner=tags_assigner,
            )
            if not exercises:
                my_print(f"No exercises found in {filepath}")
                continue

            teacher_seq_lens = get_seq_lens(
                exercises,
                teacher=True,
                tokenizer=tokenizer,
            )
            my_print(f"\n{filepath}:")
            my_print(f"  found {len(exercises)} exercises")
            my_print(f"  teacher_seq_lens: {min(teacher_seq_lens)} to {max(teacher_seq_lens)}, avg {sum(teacher_seq_lens)/len(teacher_seq_lens):.0f}")
            filtered_exercises = []
            for ex, tsl in zip(exercises, teacher_seq_lens, strict=True):
                if tsl <= max_teacher_seq_len:
                    ex.teacher_seq_len = tsl
                    filtered_exercises.append(ex)
            my_print(f"  kept {len(filtered_exercises)} exercises with teacher_seq_len <= {max_teacher_seq_len}")
            all_exercises.extend(filtered_exercises)
        my_print("==== /StudentTeacherDataset ====", flush=True)
        return cls(tokenizer, all_exercises, student_dropout_rate, div=div)

    def __len__(self):
        return len(self.exercises)

    def __getitem__(self, idx) -> STExercise:
        return self.exercises[idx]

    def collate_fn(self,
        exercises: list[STExercise],
        padding_value: int,
        div8: bool = False,
        only_student: bool = False,
        include_weights: bool = False,
    ):
        batch = exercise_collate_fn(
            tokenizer=self.tokenizer,
            student_dropout_rate=self.student_dropout_rate,
            exercises=exercises,
            padding_value=padding_value,
            div8=div8,
            only_student=only_student
        )
        if include_weights:
            if any(ex.loss_weight is None for ex in exercises):
                raise ValueError("All exercises must have loss_weight set if include_weights is True")
            else:
                batch['loss_weights'] = torch.tensor([ex.loss_weight for ex in exercises], dtype=torch.float)
        return batch

    @classmethod
    def from_zip(
        cls,
        tokenizer: Tokenizer,
        zip_templates: list[ZipTemplate],
        student_dropout_rate: float,
        verbose: bool = False,
        strict: bool = False,
        max_teacher_seq_len: int = 8192,
        div: int | None = None,
        tags_per_xml: list[set[str]] | None = None,  # Tags to assign to exercises from each XML file
        add_filename_tags: bool = True,
    ) -> STDataset:
        all_tmp_xml_paths = []
        tmp_dirs = []
        for zip_template in zip_templates:
            zip_path = zip_template.zip_path
            pattern = zip_template.pattern
            tmp_xml_paths, tmp_dir = resolve_zip_exercise_glob_and_move_to_tmp(zip_path, pattern)  # Pre-fetch and cache the zip file
            if verbose:
                print(f"Resolved {len(tmp_xml_paths)} files from {zip_path} with pattern {pattern}", flush=True)
            all_tmp_xml_paths.extend(tmp_xml_paths)
            # Keep track of temporary directories to clean up later
            tmp_dirs.append(tmp_dir)

        assert len(all_tmp_xml_paths) > 0, "No XML files found in the provided zip templates"
        dataset = cls.from_xml_paths(
            tokenizer=tokenizer,
            xml_paths=all_tmp_xml_paths,
            verbose=verbose,
            strict=strict,
            max_teacher_seq_len=max_teacher_seq_len,
            student_dropout_rate=student_dropout_rate,
            div=div,
            tags_per_xml=tags_per_xml,
            add_filename_tags=add_filename_tags,
        )
        # Clean up temporary files
        for tmp_dir in tmp_dirs:
            tmp_dir.cleanup()

        return dataset

class ConcatSTDataset(torch.utils.data.ConcatDataset):
    def __init__(
            self,
            datasets: list[STDataset],
            tokenizer: Tokenizer,
            student_dropout_rate: float,
        ):
        if not datasets:
            raise ValueError("ConcatSTDataset requires at least one dataset")

        # Initialize as a regular ConcatDataset
        super().__init__(datasets)

        self.tokenizer = tokenizer
        self.student_dropout_rate = student_dropout_rate

    def collate_fn(self,
        exercises: list[STExercise],
        padding_value: int,
        div8: bool = False,
        only_student: bool = False,
        include_weights: bool = False,
    ):
        batch = exercise_collate_fn(
            tokenizer=self.tokenizer,
            student_dropout_rate=self.student_dropout_rate,
            exercises=exercises,
            padding_value=padding_value,
            div8=div8,
            only_student=only_student
        )
        if include_weights:
            if any(ex.loss_weight is None for ex in exercises):
                raise ValueError("All exercises must have loss_weight set if include_weights is True")
            else:
                batch['loss_weights'] = torch.tensor([ex.loss_weight for ex in exercises], dtype=torch.float)

        return batch

class RandomSTDataset(torch.utils.data.Dataset):
    def __init__(self, seq_len: int, batch_size: int, devices: int):
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.devices = devices
        self.teacher_answer_len = 200
        if self.seq_len < self.teacher_answer_len:
            self.teacher_answer_len = self.seq_len
            print("Warning: teacher_answer_len is greater than seq_len")

    def __len__(self):
        return 32

    def __getitem__(self, _idx):
        return None

    def collate_fn(self, _samples, _padding_value):
        student_seqs = torch.randint(0, 100, (self.batch_size, self.seq_len))
        student_masks = torch.zeros_like(student_seqs, dtype=torch.bool)
        student_masks[:, -self.teacher_answer_len:] = 1
        teacher_seqs = torch.randint(0, 100, (self.batch_size, self.seq_len))
        teacher_masks = torch.zeros_like(teacher_seqs, dtype=torch.bool)
        teacher_masks[:, -self.teacher_answer_len:] = 1
        lesson_ixs = torch.zeros(self.batch_size)

        return {
            'student_seqs': student_seqs,
            'student_masks': student_masks,
            'teacher_seqs': teacher_seqs,
            'teacher_masks': teacher_masks,
            'lesson_ixs': lesson_ixs,
        }


def combine_steps_for_validation(
    run_paths: list[Path],
    max_student_seq_len: int | None,
    base_model: str,
    max_num_exercises: int | None = None,
) -> tuple[list[STStepMessages], list[STStepMessages]]:

    max_num_exercises = max_num_exercises or len(run_paths)

    steps_0 = []
    steps_1 = []

    run_paths = list(run_paths)
    random.shuffle(run_paths)
    for run_path in run_paths:
        step_0, step_1 = sample_step_for_validation(run_path, max_student_seq_len, base_model)
        if step_0 and step_1:
            steps_0.append(step_0)
            steps_1.append(step_1)

        if len(steps_0) == max_num_exercises:
            break

    return steps_0, steps_1


def sample_step_for_validation(
    run_path: Path,
    max_student_seq_len: int | None,
    base_model: str,
) -> tuple[STStepMessages | None, STStepMessages | None]:
    """Sample a single step from a run. The length of the student sequence must be smaller than max_student_seq_len.
    This is used to create validation datasets.

    Returns:
      - exercise_dropout_0: an exercise with the student_dropout_rate=0.0
      - exercise_dropout_1: an exercise with the student_dropout_rate=1.0
    """
    steps = []
    for step in get_steps_from_run(run_path):
        stripped_step = step.strip_after_last_target()
        if stripped_step is not None and stripped_step.contains_target:
            steps.append(stripped_step)

    candidates: list[tuple[STStepMessages, STStepMessages]] = []
    for step in steps:
        step_0 = step.apply_dropout(dropout_rate=0.0)

        if max_student_seq_len is not None:
            student_token_ids = Tokenizer(base_model).tokenize_for_inference(
                step_0.to_messages(teacher=False),
            )
            num_student_tokens = len(student_token_ids)
            if num_student_tokens > max_student_seq_len:
                continue

        step_1 = step.apply_dropout(dropout_rate=1.0)
        candidates.append((step_0, step_1))

    if not candidates:
        return None, None

    return random.choice(candidates)

def pad_sequence_to_multiple_of_8(tensor_list, batch_first, padding_value):
    if len(tensor_list) == 0:
        return torch.tensor([])  # Handle empty case

    max_len = max(tensor.size(0) for tensor in tensor_list)  # Get longest sequence
    padded_len = (max_len + 7) // 8 * 8 + 1 # Round up to nearest multiple of 8

    # Pad sequences to padded_len explicitly
    padded_tensors = pad_sequence(tensor_list, batch_first=batch_first, padding_value=padding_value)

    # If already correct length, return as is
    if padded_tensors.size(1) == padded_len:
        return padded_tensors

    # Otherwise, pad manually
    padding_needed = padded_len - padded_tensors.size(1)
    return F.pad(padded_tensors, (0, padding_needed), value=padding_value)

def exercise_collate_fn(tokenizer, student_dropout_rate, exercises: list[STExercise], padding_value: int, div8: bool = False, only_student: bool = False):
    pad_sequence_fn = pad_sequence_to_multiple_of_8 if div8 else pad_sequence

    samples = [
        exercise.to_sample(tokenizer, dropout_rate=student_dropout_rate)
        for exercise in exercises
    ]

    student_seqs = [sample["student_seq"][0] for sample in samples]
    student_masks = [sample["student_mask"][0] for sample in samples]

    batch = {
        "student_seqs": pad_sequence_fn(student_seqs, batch_first=True, padding_value=padding_value),
        "student_masks": pad_sequence_fn(student_masks, batch_first=True, padding_value=0).bool(),
        "tags": [exercise.tags for exercise in exercises],
    }

    if any(hasattr(exercise, "drop") for exercise in exercises):
        batch["drop"] = torch.tensor([getattr(exercise, "drop", False) for exercise in exercises])

    if only_student:
        return batch

    teacher_seqs = [sample["teacher_seq"][0] for sample in samples]
    teacher_masks = [sample["teacher_mask"][0] for sample in samples]

    batch["teacher_seqs"] = pad_sequence_fn(teacher_seqs, batch_first=True, padding_value=padding_value)
    batch["teacher_masks"] = pad_sequence_fn(teacher_masks, batch_first=True, padding_value=0).bool()

    return batch

def get_seq_lens(
    exercises: list[STExercise],
    teacher: bool,
    tokenizer: Tokenizer,
    dropout_rate: float = 0.0
) -> list[int]:
    step_list = [ex.step.to_messages(teacher=teacher, dropout_rate=dropout_rate) for ex in exercises]
    return tokenizer.get_step_lengths(step_list)
