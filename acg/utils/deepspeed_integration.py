"""
DeepSpeed-MoE integration for expert parallelism.

Provides utilities for distributing experts across GPUs using DeepSpeed-MoE.

"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any, List
import logging

logger = logging.getLogger(__name__)


def check_deepspeed_available() -> bool:
    """
    Check if DeepSpeed is available.

    Returns:
        True if DeepSpeed is installed
    """
    try:
        import deepspeed

        return True
    except ImportError:
        return False


class DeepSpeedMoEConfig:
    """
    Configuration for DeepSpeed MoE integration.

    Requirement 11.2
    """

    def __init__(
        self,
        expert_parallel_size: int = 1,
        ep_group: Optional[Any] = None,
        enable_expert_tensor_parallelism: bool = False,
        use_residual: bool = True,
        use_tutel: bool = False,
        min_capacity: int = 0,
        drop_tokens: bool = True,
        use_rts: bool = True,
        capacity_factor: float = 1.0,
    ):
        """
        Initialize DeepSpeed MoE configuration.

        Args:
            expert_parallel_size: Number of GPUs for expert parallelism
            ep_group: Expert parallel process group
            enable_expert_tensor_parallelism: Enable tensor parallelism within experts
            use_residual: Use residual connections
            use_tutel: Use Tutel optimization library
            min_capacity: Minimum capacity per expert
            drop_tokens: Whether to drop tokens when capacity exceeded
            use_rts: Use random token selection
            capacity_factor: Capacity factor for expert buffers
        """
        self.expert_parallel_size = expert_parallel_size
        self.ep_group = ep_group
        self.enable_expert_tensor_parallelism = enable_expert_tensor_parallelism
        self.use_residual = use_residual
        self.use_tutel = use_tutel
        self.min_capacity = min_capacity
        self.drop_tokens = drop_tokens
        self.use_rts = use_rts
        self.capacity_factor = capacity_factor

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for DeepSpeed config."""
        return {
            "expert_parallel_size": self.expert_parallel_size,
            "ep_group": self.ep_group,
            "enable_expert_tensor_parallelism": self.enable_expert_tensor_parallelism,
            "use_residual": self.use_residual,
            "use_tutel": self.use_tutel,
            "min_capacity": self.min_capacity,
            "drop_tokens": self.drop_tokens,
            "use_rts": self.use_rts,
        }


def create_deepspeed_config(
    train_batch_size: int,
    gradient_accumulation_steps: int = 1,
    fp16_enabled: bool = False,
    bf16_enabled: bool = True,
    zero_stage: int = 1,
    moe_config: Optional[DeepSpeedMoEConfig] = None,
    gradient_clipping: float = 1.0,
    learning_rate: float = 3e-4,
) -> Dict[str, Any]:
    """
    Create DeepSpeed configuration dictionary.

    Args:
        train_batch_size: Total training batch size
        gradient_accumulation_steps: Number of gradient accumulation steps
        fp16_enabled: Enable fp16 training
        bf16_enabled: Enable bf16 training
        zero_stage: ZeRO optimization stage (0, 1, 2, or 3)
        moe_config: MoE-specific configuration
        gradient_clipping: Gradient clipping value
        learning_rate: Learning rate

    Returns:
        DeepSpeed configuration dictionary

    Requirement 11.2
    """
    config = {
        "train_batch_size": train_batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "gradient_clipping": gradient_clipping,
        "steps_per_print": 100,
        "wall_clock_breakdown": False,
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": learning_rate,
                "betas": [0.9, 0.999],
                "eps": 1e-8,
                "weight_decay": 0.01,
            },
        },
        "scheduler": {
            "type": "WarmupDecayLR",
            "params": {
                "warmup_min_lr": 0,
                "warmup_max_lr": learning_rate,
                "warmup_num_steps": 1000,
                "total_num_steps": 100000,
            },
        },
    }

    # Add fp16 config
    if fp16_enabled:
        config["fp16"] = {
            "enabled": True,
            "loss_scale": 0,
            "loss_scale_window": 1000,
            "initial_scale_power": 16,
            "hysteresis": 2,
            "min_loss_scale": 1,
        }

    # Add bf16 config
    if bf16_enabled:
        config["bf16"] = {"enabled": True}

    # Add ZeRO config
    if zero_stage > 0:
        config["zero_optimization"] = {
            "stage": zero_stage,
            "allgather_partitions": True,
            "allgather_bucket_size": 5e8,
            "overlap_comm": True,
            "reduce_scatter": True,
            "reduce_bucket_size": 5e8,
            "contiguous_gradients": True,
        }

        # Stage 3 specific options
        if zero_stage == 3:
            config["zero_optimization"].update(
                {
                    "stage3_prefetch_bucket_size": 5e8,
                    "stage3_param_persistence_threshold": 1e6,
                    "stage3_max_live_parameters": 1e9,
                    "stage3_max_reuse_distance": 1e9,
                }
            )

    # Add MoE config
    if moe_config is not None:
        config["moe"] = {
            "enabled": True,
            "expert_parallel_size": moe_config.expert_parallel_size,
            "enable_expert_tensor_parallelism": moe_config.enable_expert_tensor_parallelism,
            "use_residual": moe_config.use_residual,
            "use_tutel": moe_config.use_tutel,
            "min_capacity": moe_config.min_capacity,
            "drop_tokens": moe_config.drop_tokens,
            "use_rts": moe_config.use_rts,
        }

    return config


def wrap_experts_with_deepspeed(
    expert_module: nn.Module,
    num_experts: int,
    hidden_size: int,
    moe_config: DeepSpeedMoEConfig,
) -> nn.Module:
    """
    Wrap expert module with DeepSpeed MoE layer.

    Args:
        expert_module: Expert module to wrap
        num_experts: Total number of experts
        hidden_size: Hidden dimension size
        moe_config: MoE configuration

    Returns:
        Wrapped expert module with DeepSpeed MoE

    Requirement 11.2
    """
    if not check_deepspeed_available():
        logger.warning("DeepSpeed not available, returning unwrapped experts")
        return expert_module

    try:
        from deepspeed.moe.layer import MoE

        # Create DeepSpeed MoE layer
        moe_layer = MoE(
            hidden_size=hidden_size,
            expert=expert_module,
            num_experts=num_experts,
            ep_size=moe_config.expert_parallel_size,
            use_residual=moe_config.use_residual,
            use_tutel=moe_config.use_tutel,
            enable_expert_tensor_parallelism=moe_config.enable_expert_tensor_parallelism,
            min_capacity=moe_config.min_capacity,
            drop_tokens=moe_config.drop_tokens,
        )

        logger.info(
            f"Wrapped experts with DeepSpeed MoE: "
            f"num_experts={num_experts}, ep_size={moe_config.expert_parallel_size}"
        )

        return moe_layer

    except Exception as e:
        logger.error(f"Failed to wrap experts with DeepSpeed: {e}")
        logger.warning("Returning unwrapped experts")
        return expert_module


def initialize_deepspeed_engine(
    model: nn.Module, config: Dict[str, Any], model_parameters: Optional[List] = None
) -> Any:
    """
    Initialize DeepSpeed engine for training.

    Args:
        model: Model to train
        config: DeepSpeed configuration dictionary
        model_parameters: Optional list of model parameters

    Returns:
        DeepSpeed engine

    Requirement 11.2
    """
    if not check_deepspeed_available():
        raise RuntimeError("DeepSpeed is not installed")

    import deepspeed

    if model_parameters is None:
        model_parameters = model.parameters()

    engine, optimizer, _, scheduler = deepspeed.initialize(
        model=model, model_parameters=model_parameters, config=config
    )

    logger.info("DeepSpeed engine initialized")
    logger.info(f"  World size: {engine.world_size}")
    logger.info(f"  Local rank: {engine.local_rank}")
    logger.info(f"  Global rank: {engine.global_rank}")

    return engine


class ExpertParallelWrapper:
    """
    Wrapper for expert-parallel training with DeepSpeed.

    Handles expert distribution across GPUs and communication.

    """

    def __init__(
        self,
        model: nn.Module,
        expert_parallel_size: int = 1,
        use_deepspeed: bool = True,
    ):
        """
        Initialize expert parallel wrapper.

        Args:
            model: ACG model
            expert_parallel_size: Number of GPUs for expert parallelism
            use_deepspeed: Whether to use DeepSpeed for parallelism
        """
        self.model = model
        self.expert_parallel_size = expert_parallel_size
        self.use_deepspeed = use_deepspeed

        if use_deepspeed and not check_deepspeed_available():
            logger.warning("DeepSpeed not available, disabling expert parallelism")
            self.use_deepspeed = False
            self.expert_parallel_size = 1

    def distribute_experts(self) -> None:
        """
        Distribute experts across GPUs.

        """
        if not self.use_deepspeed or self.expert_parallel_size <= 1:
            logger.info("Expert parallelism disabled")
            return

        if not hasattr(self.model, "experts"):
            logger.warning("Model does not have experts attribute")
            return

        try:
            import deepspeed

            # Get expert parallel group
            ep_group = deepspeed.utils.groups.get_expert_parallel_group()

            # Distribute experts
            experts = self.model.experts
            if hasattr(experts, "expert_blocks"):
                num_experts = len(experts.expert_blocks)
                experts_per_gpu = num_experts // self.expert_parallel_size

                logger.info(
                    f"Distributing {num_experts} experts across "
                    f"{self.expert_parallel_size} GPUs "
                    f"({experts_per_gpu} experts per GPU)"
                )

                # Set expert parallel group
                if hasattr(experts, "set_expert_parallel_group"):
                    experts.set_expert_parallel_group(ep_group)

        except Exception as e:
            logger.error(f"Failed to distribute experts: {e}")

    def get_expert_parallel_rank(self) -> int:
        """
        Get expert parallel rank.

        Returns:
            Expert parallel rank (0 if not using expert parallelism)
        """
        if not self.use_deepspeed:
            return 0

        try:
            import deepspeed

            return deepspeed.utils.groups.get_expert_parallel_rank()
        except:
            return 0

    def get_expert_parallel_world_size(self) -> int:
        """
        Get expert parallel world size.

        Returns:
            Expert parallel world size (1 if not using expert parallelism)
        """
        if not self.use_deepspeed:
            return 1

        try:
            import deepspeed

            return deepspeed.utils.groups.get_expert_parallel_world_size()
        except:
            return 1


def setup_expert_parallel_training(
    model: nn.Module,
    train_batch_size: int,
    expert_parallel_size: int = 1,
    zero_stage: int = 1,
    bf16_enabled: bool = True,
    gradient_clipping: float = 1.0,
    learning_rate: float = 3e-4,
) -> Any:
    """
    Setup complete expert-parallel training with DeepSpeed.

    Args:
        model: ACG model
        train_batch_size: Training batch size
        expert_parallel_size: Number of GPUs for expert parallelism
        zero_stage: ZeRO optimization stage
        bf16_enabled: Enable bf16 training
        gradient_clipping: Gradient clipping value
        learning_rate: Learning rate

    Returns:
        DeepSpeed engine

    """
    # Create MoE config
    moe_config = DeepSpeedMoEConfig(
        expert_parallel_size=expert_parallel_size,
        enable_expert_tensor_parallelism=False,
        use_residual=True,
        use_tutel=False,
        drop_tokens=True,
        use_rts=True,
    )

    # Create DeepSpeed config
    ds_config = create_deepspeed_config(
        train_batch_size=train_batch_size,
        gradient_accumulation_steps=1,
        fp16_enabled=False,
        bf16_enabled=bf16_enabled,
        zero_stage=zero_stage,
        moe_config=moe_config,
        gradient_clipping=gradient_clipping,
        learning_rate=learning_rate,
    )

    # Initialize DeepSpeed engine
    engine = initialize_deepspeed_engine(model, ds_config)

    # Setup expert parallelism
    wrapper = ExpertParallelWrapper(
        model=model, expert_parallel_size=expert_parallel_size, use_deepspeed=True
    )
    wrapper.distribute_experts()

    logger.info("Expert-parallel training setup complete")

    return engine
