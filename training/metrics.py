from dataclasses import dataclass
import torch
from collections import defaultdict
import torch.distributed as dist
from typing import NamedTuple


class RunningAverage:
    def __init__(self, total=0, count=0):
        self.total = total
        self.count = count

    def add(self, value: float, count: int = 1):
        self.total += value
        self.count += count

    def get_average(self):
        if self.count == 0:
            return 0
        return self.total / self.count

    def __repr__(self):
        return f"RunningAverage(total={self.total}, count={self.count})"


class RunningAverageTensor:
    def __init__(self, size: int, device=None):
        self.total = torch.zeros(size, device=device)
        self.count = torch.zeros(size, device=device)

    def add(
        self,
        group_idxs: torch.Tensor,  # group index of each metric, (sample_size,)
        values: torch.Tensor,  # metric values, (sample_size,)
    ):
        dim = 0
        self.total.index_add_(dim, group_idxs, values.to(self.total.dtype))
        self.count.index_add_(dim, group_idxs, torch.ones_like(values, dtype=self.count.dtype))

    def get_average(self):
        total = self.total
        count = self.count
        # Perform element-wise division where count is not zero, else put NaN
        return torch.where(count != 0, total / count, torch.empty_like(total).fill_(torch.nan))

    def get_total_average(self):
        total = self.total.sum()
        count = self.count.sum()
        return 0 if count == 0 else total / count

    def __repr__(self):
        print(type(self.total), type(self.count))
        return f"RunningAverageTensor(total={self.total}, count={self.count})"

    def to(self, device):
        self.total = self.total.to(device)
        self.count = self.count.to(device)


class AggregatorAccelerate:
    """
    Calculation in which all metrics for every sample in the validation set are gathered
    in all devices.
    """
    def __init__(self, group_names: list[str], device):
        self.group_names = group_names
        def ra_factory():
            return RunningAverageTensor(len(group_names), device=device)

        self._ra = defaultdict(ra_factory)  # (metric_name, group_id) -> RunningAverage

    def to(self, device):
        for v in self._ra.values():
            v.to(device)

    def add_batch(
        self,
        batch_group_ixs: torch.Tensor,  # Every element is the index of the group to which
                                        # the corresponding sample belongs, shape (batch_size,)
        batch_metrics: dict[str, torch.Tensor],  # metric_name -> tensor with batch_size elements
        accelerator
    ):
        metric_names = list(batch_metrics.keys())
        batch_metrics["group_ixs"] = batch_group_ixs
        # Gather all metrics in all devices
        gathered_metrics = accelerator.gather_for_metrics(batch_metrics)
        group_ixs = gathered_metrics["group_ixs"]
        for metric_name in metric_names:
            values = gathered_metrics[metric_name]
            # Update the running statistics of every group
            self._ra[metric_name].add(group_ixs.flatten(), values.flatten())

    def gather_for_metrics(self, run, batch_metrics):
        current_device = torch.cuda.current_device()
        process_group = run.device_mesh.get_group(mesh_dim="dp")
        group_size = dist.get_world_size(group=process_group)
        dp_rank = run.dp_rank
        tp_rank = run.tp_rank
        gathered_metrics = {}
        for key in batch_metrics.keys():
            if batch_metrics[key] is not None:
                gathered_metrics[key] = [torch.zeros_like(batch_metrics[key]).to(current_device) for _ in range(group_size)]
            else:
                gathered_metrics[key] = [None for _ in range(group_size)]

        for key in batch_metrics.keys():
            if dp_rank == 0 and tp_rank == 0:
                dist.gather(batch_metrics[key].to(current_device), gather_list=gathered_metrics[key], group=process_group)
            elif dp_rank != 0 and tp_rank == 0:
                dist.gather(batch_metrics[key].to(current_device), dst=0, group=process_group)

        # Convert each list of tensors into a single tensor
        for key in batch_metrics.keys():
            if batch_metrics[key] is not None:
                gathered_metrics[key] = torch.cat(gathered_metrics[key], dim=0).to(torch.cuda.current_device())

        return gathered_metrics

    def key_to_string(self, key):
        metric_name, group_id = key
        group_name = self.group_names[int(group_id)]
        return "/".join((metric_name, group_name))

    def get_average(self) -> tuple[dict[str, float], dict[str, float]]:
        metrics_by_group = {}
        metrics_total = {}
        for metric_name, ra in self._ra.items():
            av = ra.get_average()
            metrics_by_group.update({
                self.key_to_string((metric_name, i)): av[i]
                for i in range(len(self.group_names))
                if ra.count[i] > 0
            })
            metrics_total[metric_name] = ra.get_total_average()

        return metrics_total, metrics_by_group



@dataclass
class BatchMetrics:
    group_ixs: torch.Tensor  # Group indexes of each sample, shape (batch_size,)
    metrics: dict[str, torch.Tensor]  # metric_name -> tensor with batch_size elements

class AggregatorTorch:
    group_names: list[str]
    global_rank: int
    dp_group: dist.ProcessGroup
    dp_rank: int
    tp_rank: int

    def __init__(self, group_names: list[str], global_rank, dp_group, dp_rank, tp_rank):
        self.group_names = group_names
        self.global_rank = global_rank
        self.dp_group = dp_group
        self.dp_rank = dp_rank
        self.tp_rank = tp_rank

        self.batch_metrics_list: list[BatchMetrics] = []

    def to(self, device):
        for v in self._ra.values():
            v.to(device)

    def add_batch(
        self,
        batch_group_ixs: torch.Tensor,  # Every element is the index of the group to which
                                        # the corresponding sample belongs, shape (batch_size,)
        batch_metrics: dict[str, torch.Tensor],  # metric_name -> tensor with batch_size elements
        drop: torch.Tensor,  # whether the sample should be dropped from the average
    ):
        metric_names = list(batch_metrics.keys())
        batch_metrics["group_ixs"] = batch_group_ixs  # We need to gather this too
        batch_metrics["drop"] = drop  # We need to gather this too
        # Gather all metrics in all devices
        gathered_metrics = self.gather_for_metrics(batch_metrics, self.dp_group, self.dp_rank, self.tp_rank)
        print(f"GRank {self.global_rank} gathered metrics: {gathered_metrics}")

        keep = ~gathered_metrics["drop"]
        self.batch_metrics_list.append(
            BatchMetrics(
                group_ixs=gathered_metrics["group_ixs"][keep],
                metrics={metric_name: gathered_metrics[metric_name][keep] for metric_name in metric_names},
            )
        )

    def gather_for_metrics(
        self,
        batch_metrics: dict[str, torch.Tensor],  # metric_name -> tensor with batch_size elements
        process_group,  # process group to gather the metrics
        dp_rank,  # rank of the current process in the data parallel group
        tp_rank,  # rank of the current process in the tensor parallel group
    ) -> dict[str, torch.Tensor]:
        device = torch.cuda.current_device()
        group_size = dist.get_world_size(group=process_group)
        gathered_metrics = {}
        for key, metric in batch_metrics.items():
            if metric is None:
                continue

            gather_list = [torch.zeros_like(metric).to(device) for _ in range(group_size)]

            if dp_rank == 0 and tp_rank == 0:
                dist.gather(metric.to(device), gather_list=gather_list, group=process_group)
            elif dp_rank != 0 and tp_rank == 0:
                dist.gather(metric.to(device), dst=0, group=process_group)

            gathered_metrics[key] = torch.cat(gather_list, dim=0).to(device)

        return gathered_metrics

    def get_average(self) -> tuple[dict[str, float], dict[str, float]]:
        """
        Get the average of all metrics in all groups.
        """
        metric_names = []
        for batch_metrics in self.batch_metrics_list:
            metric_names.extend(batch_metrics.metrics.keys())
        # Remove duplicates while preserving order
        metric_names = list(dict.fromkeys(metric_names))

        ra = {
            metric_name: RunningAverageTensor(len(self.group_names), device=torch.cuda.current_device())
            for metric_name in metric_names
        }
        for batch_metrics in self.batch_metrics_list:
            for metric_name in batch_metrics.metrics:
                ra[metric_name].add(batch_metrics.group_ixs, batch_metrics.metrics[metric_name])

        averages = {
            metric_name: ra[metric_name].get_total_average()
            for metric_name in metric_names
        }

        group_average_tensors = {
            metric_name: ra[metric_name].get_average()
            for metric_name in metric_names
        }

        # Split the group averages into a dictionary with group names as keys
        # and filter out groups with no samples
        group_averages = {
            f"{metric_name}/{group_name}": metric_tensor[group_ix]
            for metric_name, metric_tensor in group_average_tensors.items()
            for group_ix, group_name in enumerate(self.group_names)
            if not metric_tensor[group_ix].isnan()
        }

        return averages, group_averages


class TaggedMetric(NamedTuple):
    tags: set[str]  # Tags of the sample
    metric_name: str  # Name of the metric
    value: float  # Value of the metric
    drop: bool  # Whether the sample should be dropped from the average (some samples in the last batch may be dropped)
    n_tokens: int  # Number of tokens in the sample

class Aggregator:
    def __init__(self, global_rank: int, dp_group: dist.ProcessGroup, dp_rank: int, tp_rank: int):
        self.global_rank = global_rank
        self.dp_group = dp_group
        self.dp_rank = dp_rank
        self.tp_rank = tp_rank

        self.gathered_metrics: list[TaggedMetric] = []

    def add_batch(
        self,
        tagged_metrics: list[TaggedMetric],
    ):
        if self.tp_rank == 0:
            if self.dp_rank == 0:
                gathered = [None for _ in range(dist.get_world_size(group=self.dp_group))]
                dist.gather_object(
                    tagged_metrics,
                    object_gather_list=gathered,
                    dst=0,
                    group=self.dp_group,
                )
                # gathered is now a list of list[TaggedMetric] with len == world_size(group)
                gathered = sum(gathered, [])
                self.gathered_metrics.extend(gathered)
            else:
                dist.gather_object(
                    tagged_metrics,
                    dst=0,
                    group=self.dp_group,
                )

    def get_average(self) -> tuple[dict[str, float], dict[str, float]]:
        """
        Get the average of all metrics in all groups.
        """
        if self.tp_rank != 0 or self.dp_rank != 0:
            return {}, {}

        averages = defaultdict(RunningAverage)
        group_averages = defaultdict(RunningAverage)
        for tagged_metric in self.gathered_metrics:
            if tagged_metric.drop:
                continue

            metric_name = tagged_metric.metric_name
            if "/" in metric_name:
                i = metric_name.find("/")
                per_token_metric_name = metric_name[:i] + "_per_token/" + metric_name[i+1:]
            else:
                per_token_metric_name = metric_name + "_per_token"
            averages[metric_name].add(tagged_metric.value)
            averages[per_token_metric_name].add(
                tagged_metric.value * tagged_metric.n_tokens, tagged_metric.n_tokens)

            # We create a separate group of WnB plots for every metric with multiple tags
            metric_name = metric_name.replace("/", "_")
            per_token_metric_name = per_token_metric_name.replace("/", "_")
            for tag in tagged_metric.tags:
                group_averages[f"{metric_name}/{tag}"].add(tagged_metric.value)
                group_averages[f"{per_token_metric_name}/{tag}"].add(
                    tagged_metric.value * tagged_metric.n_tokens, tagged_metric.n_tokens)

        averages = {k: v.get_average() for k, v in averages.items()}
        group_averages = {k: v.get_average() for k, v in group_averages.items()}
        return averages, group_averages
