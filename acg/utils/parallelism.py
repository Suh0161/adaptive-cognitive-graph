"""
Tensor and pipeline parallelism utilities for ACG model.

Provides utilities for distributing model across GPUs using tensor and pipeline parallelism.
"""

import torch
import torch.nn as nn
import torch.distributed as dist
from typing import Optional, List, Dict, Any, Tuple
import logging

logger = logging.getLogger(__name__)


def check_distributed_available() -> bool:
    """
    Check if distributed training is available.

    Returns:
        True if distributed is available and initialized
    """
    return dist.is_available() and dist.is_initialized()


def get_world_size() -> int:
    """Get distributed world size."""
    if check_distributed_available():
        return dist.get_world_size()
    return 1


def get_rank() -> int:
    """Get distributed rank."""
    if check_distributed_available():
        return dist.get_rank()
    return 0


class TensorParallelConfig:
    """
    Configuration for tensor parallelism.

    """

    def __init__(
        self,
        tensor_parallel_size: int = 1,
        sequence_parallel: bool = False,
        async_tensor_model_parallel_allreduce: bool = False,
    ):
        """
        Initialize tensor parallel configuration.

        Args:
            tensor_parallel_size: Number of GPUs for tensor parallelism
            sequence_parallel: Enable sequence parallelism
            async_tensor_model_parallel_allreduce: Use async allreduce
        """
        self.tensor_parallel_size = tensor_parallel_size
        self.sequence_parallel = sequence_parallel
        self.async_tensor_model_parallel_allreduce = (
            async_tensor_model_parallel_allreduce
        )


class PipelineParallelConfig:
    """
    Configuration for pipeline parallelism.

    """

    def __init__(
        self,
        pipeline_parallel_size: int = 1,
        num_microbatches: int = 1,
        pipeline_model_parallel_split_rank: Optional[int] = None,
    ):
        """
        Initialize pipeline parallel configuration.

        Args:
            pipeline_parallel_size: Number of pipeline stages
            num_microbatches: Number of microbatches for pipeline
            pipeline_model_parallel_split_rank: Rank to split pipeline
        """
        self.pipeline_parallel_size = pipeline_parallel_size
        self.num_microbatches = num_microbatches
        self.pipeline_model_parallel_split_rank = pipeline_model_parallel_split_rank


def split_tensor_along_dim(
    tensor: torch.Tensor,
    num_partitions: int,
    dim: int = -1,
    contiguous_split_chunks: bool = False,
) -> List[torch.Tensor]:
    """
    Split tensor along specified dimension for tensor parallelism.

    Args:
        tensor: Tensor to split
        num_partitions: Number of partitions
        dim: Dimension to split along
        contiguous_split_chunks: Make chunks contiguous

    Returns:
        List of tensor chunks

    """
    dim_size = tensor.size(dim)
    assert (
        dim_size % num_partitions == 0
    ), f"Dimension size {dim_size} not divisible by {num_partitions}"

    chunk_size = dim_size // num_partitions
    tensor_list = torch.split(tensor, chunk_size, dim=dim)

    if contiguous_split_chunks:
        tensor_list = [chunk.contiguous() for chunk in tensor_list]

    return list(tensor_list)


def gather_tensor_along_dim(
    tensor: torch.Tensor, dim: int = -1, group: Optional[Any] = None
) -> torch.Tensor:
    """
    Gather tensor from all ranks along specified dimension.

    Args:
        tensor: Local tensor
        dim: Dimension to gather along
        group: Process group

    Returns:
        Gathered tensor

    """
    if not check_distributed_available():
        return tensor

    world_size = get_world_size() if group is None else dist.get_world_size(group)

    if world_size == 1:
        return tensor

    # Gather tensors from all ranks
    tensor_list = [torch.empty_like(tensor) for _ in range(world_size)]
    dist.all_gather(tensor_list, tensor, group=group)

    # Concatenate along dimension
    output = torch.cat(tensor_list, dim=dim)

    return output


class ColumnParallelLinear(nn.Module):
    """
    Linear layer with column parallelism.

    Splits weight matrix along output dimension.

    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        gather_output: bool = True,
        tensor_parallel_size: int = 1,
    ):
        """
        Initialize column parallel linear layer.

        Args:
            in_features: Input features
            out_features: Output features (will be split)
            bias: Use bias
            gather_output: Gather output across ranks
            tensor_parallel_size: Tensor parallel size
        """
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.gather_output = gather_output
        self.tensor_parallel_size = tensor_parallel_size

        # Split output features
        assert out_features % tensor_parallel_size == 0
        self.out_features_per_partition = out_features // tensor_parallel_size

        # Create weight and bias
        self.weight = nn.Parameter(
            torch.empty(self.out_features_per_partition, in_features)
        )
        if bias:
            self.bias = nn.Parameter(torch.empty(self.out_features_per_partition))
        else:
            self.register_parameter("bias", None)

        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights."""
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor (..., in_features)

        Returns:
            Output tensor (..., out_features) or (..., out_features_per_partition)
        """
        # Linear transformation
        output = torch.nn.functional.linear(x, self.weight, self.bias)

        # Gather output if needed
        if self.gather_output and self.tensor_parallel_size > 1:
            output = gather_tensor_along_dim(output, dim=-1)

        return output


class RowParallelLinear(nn.Module):
    """
    Linear layer with row parallelism.

    Splits weight matrix along input dimension.

    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        input_is_parallel: bool = False,
        tensor_parallel_size: int = 1,
    ):
        """
        Initialize row parallel linear layer.

        Args:
            in_features: Input features (will be split)
            out_features: Output features
            bias: Use bias
            input_is_parallel: Whether input is already partitioned
            tensor_parallel_size: Tensor parallel size
        """
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.input_is_parallel = input_is_parallel
        self.tensor_parallel_size = tensor_parallel_size

        # Split input features
        assert in_features % tensor_parallel_size == 0
        self.in_features_per_partition = in_features // tensor_parallel_size

        # Create weight and bias
        self.weight = nn.Parameter(
            torch.empty(out_features, self.in_features_per_partition)
        )
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)

        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights."""
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor (..., in_features) or (..., in_features_per_partition)

        Returns:
            Output tensor (..., out_features)
        """
        # Split input if not already parallel
        if not self.input_is_parallel and self.tensor_parallel_size > 1:
            rank = get_rank()
            x_list = split_tensor_along_dim(x, self.tensor_parallel_size, dim=-1)
            x = x_list[rank]

        # Linear transformation
        output = torch.nn.functional.linear(x, self.weight)

        # All-reduce across tensor parallel group
        if self.tensor_parallel_size > 1 and check_distributed_available():
            dist.all_reduce(output)

        # Add bias
        if self.bias is not None:
            output = output + self.bias

        return output


class PipelineStage(nn.Module):
    """
    Single stage in pipeline parallelism.

    """

    def __init__(self, module: nn.Module, stage_id: int, num_stages: int):
        """
        Initialize pipeline stage.

        Args:
            module: Module for this stage
            stage_id: Stage identifier (0 to num_stages-1)
            num_stages: Total number of stages
        """
        super().__init__()
        self.module = module
        self.stage_id = stage_id
        self.num_stages = num_stages
        self.is_first_stage = stage_id == 0
        self.is_last_stage = stage_id == num_stages - 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through stage."""
        return self.module(x)


def create_pipeline_stages(
    model: nn.Module, num_stages: int, split_points: Optional[List[str]] = None
) -> List[PipelineStage]:
    """
    Create pipeline stages from model.

    Args:
        model: Model to split into stages
        num_stages: Number of pipeline stages
        split_points: Optional list of module names to split at

    Returns:
        List of pipeline stages

    """
    if num_stages == 1:
        return [PipelineStage(model, 0, 1)]

    # Auto-split if no split points provided
    if split_points is None:
        # Try to split evenly across major components
        stages = []

        if hasattr(model, "encoder"):
            stages.append(PipelineStage(model.encoder, 0, num_stages))

        if hasattr(model, "router") and hasattr(model, "experts"):
            combined = nn.ModuleDict({"router": model.router, "experts": model.experts})
            stages.append(PipelineStage(combined, 1, num_stages))

        if hasattr(model, "verifier") and hasattr(model, "fusion"):
            combined = nn.ModuleDict(
                {"verifier": model.verifier, "fusion": model.fusion}
            )
            stages.append(PipelineStage(combined, 2, num_stages))

        # Pad or trim to match num_stages
        while len(stages) < num_stages:
            stages.append(stages[-1])
        stages = stages[:num_stages]

        # Update stage IDs
        for i, stage in enumerate(stages):
            stage.stage_id = i
            stage.num_stages = num_stages
            stage.is_first_stage = i == 0
            stage.is_last_stage = i == num_stages - 1

        return stages

    # Manual split based on split points
    # This would require more complex logic to split at specific module names
    raise NotImplementedError("Manual split points not yet implemented")


class ParallelismManager:
    """
    Manager for tensor and pipeline parallelism.

    """

    def __init__(
        self,
        tensor_parallel_config: Optional[TensorParallelConfig] = None,
        pipeline_parallel_config: Optional[PipelineParallelConfig] = None,
    ):
        """
        Initialize parallelism manager.

        Args:
            tensor_parallel_config: Tensor parallelism configuration
            pipeline_parallel_config: Pipeline parallelism configuration
        """
        self.tensor_parallel_config = tensor_parallel_config or TensorParallelConfig()
        self.pipeline_parallel_config = (
            pipeline_parallel_config or PipelineParallelConfig()
        )

        self.tensor_parallel_size = self.tensor_parallel_config.tensor_parallel_size
        self.pipeline_parallel_size = (
            self.pipeline_parallel_config.pipeline_parallel_size
        )

        logger.info(
            f"Parallelism manager initialized: "
            f"TP={self.tensor_parallel_size}, PP={self.pipeline_parallel_size}"
        )

    def apply_tensor_parallelism(
        self, model: nn.Module, target_modules: Optional[List[str]] = None
    ) -> None:
        """
        Apply tensor parallelism to model.

        Args:
            model: Model to parallelize
            target_modules: Optional list of module names to parallelize
        """
        if self.tensor_parallel_size <= 1:
            logger.info("Tensor parallelism disabled (size=1)")
            return

        logger.info(
            f"Applying tensor parallelism with size={self.tensor_parallel_size}"
        )

        # Replace linear layers with parallel versions
        if target_modules is None:
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "ffn"]

        for name, module in model.named_modules():
            if any(target in name for target in target_modules):
                if isinstance(module, nn.Linear):
                    # Determine if column or row parallel
                    if "o_proj" in name or "down" in name:
                        # Row parallel for output projections
                        parallel_module = RowParallelLinear(
                            in_features=module.in_features,
                            out_features=module.out_features,
                            bias=module.bias is not None,
                            tensor_parallel_size=self.tensor_parallel_size,
                        )
                    else:
                        # Column parallel for input projections
                        parallel_module = ColumnParallelLinear(
                            in_features=module.in_features,
                            out_features=module.out_features,
                            bias=module.bias is not None,
                            tensor_parallel_size=self.tensor_parallel_size,
                        )

                    # Replace module
                    parent_name = ".".join(name.split(".")[:-1])
                    child_name = name.split(".")[-1]
                    parent = model.get_submodule(parent_name) if parent_name else model
                    setattr(parent, child_name, parallel_module)

                    logger.info(f"Replaced {name} with parallel version")

    def create_pipeline_stages(self, model: nn.Module) -> List[PipelineStage]:
        """
        Create pipeline stages from model.

        Args:
            model: Model to split

        Returns:
            List of pipeline stages
        """
        if self.pipeline_parallel_size <= 1:
            logger.info("Pipeline parallelism disabled (size=1)")
            return [PipelineStage(model, 0, 1)]

        logger.info(f"Creating {self.pipeline_parallel_size} pipeline stages")

        stages = create_pipeline_stages(model, self.pipeline_parallel_size)

        logger.info(f"Created {len(stages)} pipeline stages")
        return stages

    def get_parallelism_info(self) -> Dict[str, Any]:
        """
        Get parallelism configuration info.

        Returns:
            Dictionary with parallelism information
        """
        return {
            "tensor_parallel_size": self.tensor_parallel_size,
            "pipeline_parallel_size": self.pipeline_parallel_size,
            "world_size": get_world_size(),
            "rank": get_rank(),
            "sequence_parallel": self.tensor_parallel_config.sequence_parallel,
        }
