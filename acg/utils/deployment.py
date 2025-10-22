"""
Deployment utilities for ACG model.

Provides PyTorch 2.1+ compatibility, mixed precision support, and checkpointing.

"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any, Callable
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def check_pytorch_version() -> bool:
    """
    Check if PyTorch version is 2.1 or higher.

    Returns:
        True if version is compatible

    """
    version = torch.__version__.split("+")[0]  # Remove +cu118 suffix
    major, minor = map(int, version.split(".")[:2])

    is_compatible = (major > 2) or (major == 2 and minor >= 1)

    if not is_compatible:
        logger.warning(
            f"PyTorch version {torch.__version__} detected. "
            f"ACG requires PyTorch 2.1 or higher for optimal performance."
        )

    return is_compatible


def compile_model(
    model: nn.Module,
    mode: str = "default",
    fullgraph: bool = False,
    dynamic: bool = False,
) -> nn.Module:
    """
    Compile model using torch.compile for PyTorch 2.1+.

    Args:
        model: Model to compile
        mode: Compilation mode ("default", "reduce-overhead", "max-autotune")
        fullgraph: Whether to require full graph compilation
        dynamic: Whether to enable dynamic shapes

    Returns:
        Compiled model

    """
    if not check_pytorch_version():
        logger.warning("torch.compile not available, returning uncompiled model")
        return model

    try:
        logger.info(
            f"Compiling model with mode={mode}, fullgraph={fullgraph}, dynamic={dynamic}"
        )
        compiled_model = torch.compile(
            model, mode=mode, fullgraph=fullgraph, dynamic=dynamic
        )
        logger.info("Model compilation successful")
        return compiled_model
    except Exception as e:
        logger.error(f"Model compilation failed: {e}")
        logger.warning("Returning uncompiled model")
        return model


def enable_torch_compile_for_components(
    model: nn.Module,
    compile_encoder: bool = True,
    compile_router: bool = True,
    compile_experts: bool = True,
    compile_verifier: bool = True,
    mode: str = "default",
) -> None:
    """
    Selectively compile model components.

    Args:
        model: ACG model
        compile_encoder: Whether to compile encoder
        compile_router: Whether to compile router
        compile_experts: Whether to compile experts
        compile_verifier: Whether to compile verifier
        mode: Compilation mode

    """
    if not check_pytorch_version():
        return

    try:
        if compile_encoder and hasattr(model, "encoder"):
            logger.info("Compiling encoder...")
            model.encoder = torch.compile(model.encoder, mode=mode)

        if compile_router and hasattr(model, "router"):
            logger.info("Compiling router...")
            model.router = torch.compile(model.router, mode=mode)

        if compile_experts and hasattr(model, "experts"):
            logger.info("Compiling experts...")
            model.experts = torch.compile(model.experts, mode=mode)

        if compile_verifier and hasattr(model, "verifier"):
            logger.info("Compiling verifier...")
            model.verifier = torch.compile(model.verifier, mode=mode)

        logger.info("Component compilation complete")
    except Exception as e:
        logger.error(f"Component compilation failed: {e}")


class MixedPrecisionManager:
    """
    Manager for mixed precision training with automatic loss scaling.

    Supports fp16 and bf16 precision modes.

    """

    def __init__(
        self,
        precision: str = "bf16",
        enabled: bool = True,
        init_scale: float = 2.0**16,
        growth_factor: float = 2.0,
        backoff_factor: float = 0.5,
        growth_interval: int = 2000,
    ):
        """
        Initialize mixed precision manager.

        Args:
            precision: Precision mode ("fp32", "fp16", "bf16")
            enabled: Whether to enable mixed precision
            init_scale: Initial loss scale for fp16
            growth_factor: Factor to grow loss scale
            backoff_factor: Factor to reduce loss scale on overflow
            growth_interval: Steps between scale growth attempts
        """
        self.precision = precision
        self.enabled = enabled and precision != "fp32"

        # Determine dtype
        if precision == "bf16":
            self.dtype = torch.bfloat16
            self.use_scaler = False  # bf16 doesn't need loss scaling
        elif precision == "fp16":
            self.dtype = torch.float16
            self.use_scaler = True
        else:
            self.dtype = torch.float32
            self.use_scaler = False

        # Create gradient scaler for fp16
        if self.use_scaler:
            self.scaler = torch.cuda.amp.GradScaler(
                init_scale=init_scale,
                growth_factor=growth_factor,
                backoff_factor=backoff_factor,
                growth_interval=growth_interval,
                enabled=self.enabled,
            )
        else:
            self.scaler = None

        logger.info(
            f"Mixed precision initialized: precision={precision}, "
            f"dtype={self.dtype}, use_scaler={self.use_scaler}"
        )

    def autocast(self):
        """
        Get autocast context manager.

        Returns:
            Context manager for automatic mixed precision
        """
        return torch.cuda.amp.autocast(enabled=self.enabled, dtype=self.dtype)

    def scale_loss(self, loss: torch.Tensor) -> torch.Tensor:
        """
        Scale loss for fp16 training.

        Args:
            loss: Unscaled loss

        Returns:
            Scaled loss
        """
        if self.scaler is not None:
            return self.scaler.scale(loss)
        return loss

    def step(self, optimizer: torch.optim.Optimizer) -> None:
        """
        Perform optimizer step with gradient unscaling.

        Args:
            optimizer: Optimizer to step
        """
        if self.scaler is not None:
            self.scaler.step(optimizer)
            self.scaler.update()
        else:
            optimizer.step()

    def unscale_gradients(self, optimizer: torch.optim.Optimizer) -> None:
        """
        Unscale gradients before gradient clipping.

        Args:
            optimizer: Optimizer with scaled gradients
        """
        if self.scaler is not None:
            self.scaler.unscale_(optimizer)

    def state_dict(self) -> Dict[str, Any]:
        """Get state dict for checkpointing."""
        if self.scaler is not None:
            return {"scaler": self.scaler.state_dict()}
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load state dict from checkpoint."""
        if self.scaler is not None and "scaler" in state_dict:
            self.scaler.load_state_dict(state_dict["scaler"])


def apply_gradient_checkpointing(
    model: nn.Module, checkpoint_encoder: bool = True, checkpoint_experts: bool = True
) -> None:
    """
    Apply gradient checkpointing to model components.

    Reduces memory usage during training by recomputing activations
    during backward pass instead of storing them.

    Args:
        model: ACG model
        checkpoint_encoder: Whether to checkpoint encoder blocks
        checkpoint_experts: Whether to checkpoint expert blocks

    """
    logger.info("Applying gradient checkpointing...")

    # Checkpoint encoder blocks
    if checkpoint_encoder and hasattr(model, "encoder"):
        if hasattr(model.encoder, "blocks"):
            for block in model.encoder.blocks:
                if hasattr(block, "forward"):
                    # Wrap forward method with checkpoint
                    block.forward = torch.utils.checkpoint.checkpoint(
                        block.forward, use_reentrant=False
                    )
            logger.info(f"Checkpointed {len(model.encoder.blocks)} encoder blocks")

    # Checkpoint expert blocks
    if checkpoint_experts and hasattr(model, "experts"):
        if hasattr(model.experts, "expert_blocks"):
            for expert in model.experts.expert_blocks:
                if hasattr(expert, "forward"):
                    expert.forward = torch.utils.checkpoint.checkpoint(
                        expert.forward, use_reentrant=False
                    )
            logger.info(
                f"Checkpointed {len(model.experts.expert_blocks)} expert blocks"
            )

    logger.info("Gradient checkpointing applied")


class ModularCheckpoint:
    """
    Modular checkpointing system for saving/loading model components independently.

    Allows saving and loading encoder, router, experts, adapters, verifier,
    and fusion components separately for flexible deployment.

    """

    @staticmethod
    def save_checkpoint(
        model: nn.Module,
        save_dir: Path,
        optimizer: Optional[torch.optim.Optimizer] = None,
        epoch: Optional[int] = None,
        step: Optional[int] = None,
        save_components_separately: bool = True,
        **extra_state,
    ) -> None:
        """
        Save model checkpoint with optional component separation.

        Args:
            model: Model to save
            save_dir: Directory to save checkpoint
            optimizer: Optional optimizer state
            epoch: Optional epoch number
            step: Optional step number
            save_components_separately: Whether to save components separately
            **extra_state: Additional state to save
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        # Save full model
        full_checkpoint = {
            "model_state_dict": model.state_dict(),
            "config": model.config if hasattr(model, "config") else None,
        }

        if optimizer is not None:
            full_checkpoint["optimizer_state_dict"] = optimizer.state_dict()

        if epoch is not None:
            full_checkpoint["epoch"] = epoch

        if step is not None:
            full_checkpoint["step"] = step

        full_checkpoint.update(extra_state)

        torch.save(full_checkpoint, save_dir / "checkpoint.pt")
        logger.info(f"Saved full checkpoint to {save_dir / 'checkpoint.pt'}")

        # Save components separately if requested
        if save_components_separately:
            ModularCheckpoint._save_components(model, save_dir)

    @staticmethod
    def _save_components(model: nn.Module, save_dir: Path) -> None:
        """Save individual model components."""
        components = {
            "encoder": "encoder",
            "router": "router",
            "experts": "experts",
            "adapters": "adapters",
            "verifier": "verifier",
            "fusion": "fusion",
        }

        for name, attr in components.items():
            if hasattr(model, attr):
                component = getattr(model, attr)
                if component is not None:
                    torch.save(component.state_dict(), save_dir / f"{name}.pt")
                    logger.info(f"Saved {name} to {save_dir / name}.pt")

    @staticmethod
    def load_checkpoint(
        model: nn.Module,
        checkpoint_path: Path,
        optimizer: Optional[torch.optim.Optimizer] = None,
        strict: bool = True,
        device: str = "cpu",
    ) -> Dict[str, Any]:
        """
        Load model checkpoint.

        Args:
            model: Model to load into
            checkpoint_path: Path to checkpoint file
            optimizer: Optional optimizer to load state into
            strict: Whether to strictly enforce state dict keys match
            device: Device to load checkpoint to

        Returns:
            Dictionary with epoch, step, and other metadata
        """
        checkpoint = torch.load(checkpoint_path, map_location=device)

        # Load model state
        model.load_state_dict(checkpoint["model_state_dict"], strict=strict)
        logger.info(f"Loaded model from {checkpoint_path}")

        # Load optimizer state
        if optimizer is not None and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            logger.info("Loaded optimizer state")

        # Return metadata
        metadata = {
            "epoch": checkpoint.get("epoch"),
            "step": checkpoint.get("step"),
        }

        return metadata

    @staticmethod
    def load_component(
        model: nn.Module,
        component_name: str,
        component_path: Path,
        strict: bool = True,
        device: str = "cpu",
    ) -> None:
        """
        Load individual model component.

        Args:
            model: Model containing the component
            component_name: Name of component ('encoder', 'router', etc.)
            component_path: Path to component checkpoint
            strict: Whether to strictly enforce state dict keys match
            device: Device to load to
        """
        if not hasattr(model, component_name):
            raise ValueError(f"Model does not have component: {component_name}")

        component = getattr(model, component_name)
        if component is None:
            raise ValueError(f"Component {component_name} is None")

        state_dict = torch.load(component_path, map_location=device)
        component.load_state_dict(state_dict, strict=strict)
        logger.info(f"Loaded {component_name} from {component_path}")


def verify_trillion_param_constraints(
    model: nn.Module, max_active_params: int = 40_000_000_000  # 40B
) -> Dict[str, Any]:
    """
    Verify model satisfies trillion-parameter constraints.

    For models with over 1T parameters, ensures active parameters
    stay below 40B per token.

    Args:
        model: Model to verify
        max_active_params: Maximum active parameters (default 40B)

    Returns:
        Dictionary with verification results

    """
    total_params = sum(p.numel() for p in model.parameters())

    # Get active parameters if model supports it
    if hasattr(model, "estimate_active_params"):
        active_params = model.estimate_active_params()
    else:
        # Estimate based on config if available
        if hasattr(model, "config"):
            config = model.config
            active_ratio = config.active_experts / config.n_experts
            active_params = int(total_params * active_ratio)
        else:
            active_params = total_params

    is_trillion_param = total_params > 1_000_000_000_000  # 1T
    satisfies_constraint = not is_trillion_param or active_params < max_active_params

    results = {
        "total_params": total_params,
        "active_params": active_params,
        "is_trillion_param": is_trillion_param,
        "max_active_params": max_active_params,
        "satisfies_constraint": satisfies_constraint,
        "active_ratio": active_params / total_params if total_params > 0 else 0,
    }

    if is_trillion_param:
        if satisfies_constraint:
            logger.info(
                f"✓ Trillion-parameter model constraint satisfied: "
                f"{active_params:,} active params < {max_active_params:,}"
            )
        else:
            logger.error(
                f"✗ Trillion-parameter model constraint violated: "
                f"{active_params:,} active params >= {max_active_params:,}"
            )

    return results


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    path: Path,
    config: Optional[Any] = None,
    **extra_state,
) -> None:
    """
    Save training checkpoint.

    Args:
        model: Model to save
        optimizer: Optimizer to save
        step: Training step
        path: Path to save checkpoint
        config: Optional model configuration
        **extra_state: Additional state to save
    """
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "step": step,
    }

    if config is not None:
        checkpoint["config"] = config

    checkpoint.update(extra_state)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, path)
    logger.info(f"Saved checkpoint to {path}")


def load_checkpoint(
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    path: Path,
    device: str = "cpu",
    strict: bool = True,
) -> int:
    """
    Load training checkpoint.

    Args:
        model: Model to load into
        optimizer: Optional optimizer to load into
        path: Path to checkpoint
        device: Device to load to
        strict: Whether to strictly match state dict keys

    Returns:
        Training step from checkpoint
    """
    checkpoint = torch.load(path, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"], strict=strict)

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    step = checkpoint.get("step", 0)
    logger.info(f"Loaded checkpoint from {path}, step={step}")

    return step


def save_modular_checkpoint(
    model: nn.Module, step: int, checkpoint_dir: Path, config: Optional[Any] = None
) -> None:
    """
    Save modular checkpoint with separate component files.

    Args:
        model: Model to save
        step: Training step
        checkpoint_dir: Directory to save components
        config: Optional model configuration
    """
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Save each component
    components = ["encoder", "router", "experts", "adapters", "verifier", "fusion"]

    for component_name in components:
        if hasattr(model, component_name):
            component = getattr(model, component_name)
            if component is not None:
                torch.save(
                    component.state_dict(), checkpoint_dir / f"{component_name}.pt"
                )
                logger.info(f"Saved {component_name}")

    # Save metadata
    import json

    metadata = {
        "step": step,
        "components": components,
    }
    if config is not None:
        metadata["config"] = str(config)

    with open(checkpoint_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Saved modular checkpoint to {checkpoint_dir}")


def load_modular_checkpoint(
    model: nn.Module,
    checkpoint_dir: Path,
    components: Optional[list] = None,
    device: str = "cpu",
    strict: bool = True,
) -> int:
    """
    Load modular checkpoint.

    Args:
        model: Model to load into
        checkpoint_dir: Directory with component files
        components: Optional list of components to load (None = all)
        device: Device to load to
        strict: Whether to strictly match state dict keys

    Returns:
        Training step from checkpoint
    """
    checkpoint_dir = Path(checkpoint_dir)

    # Load metadata
    import json

    with open(checkpoint_dir / "metadata.json", "r") as f:
        metadata = json.load(f)

    # Determine which components to load
    if components is None:
        components = metadata.get(
            "components", ["encoder", "router", "experts", "verifier", "fusion"]
        )

    # Load each component
    for component_name in components:
        component_path = checkpoint_dir / f"{component_name}.pt"
        if component_path.exists() and hasattr(model, component_name):
            component = getattr(model, component_name)
            if component is not None:
                state_dict = torch.load(component_path, map_location=device)
                component.load_state_dict(state_dict, strict=strict)
                logger.info(f"Loaded {component_name}")

    step = metadata.get("step", 0)
    logger.info(f"Loaded modular checkpoint from {checkpoint_dir}, step={step}")

    return step


def estimate_memory_usage(
    model: nn.Module, config: Any, batch_size: int = 1, seq_len: Optional[int] = None
) -> Dict[str, float]:
    """
    Estimate memory usage for model.

    Args:
        model: Model to estimate
        config: Model configuration
        batch_size: Batch size for estimation
        seq_len: Sequence length (uses config.max_seq_len if None)

    Returns:
        Dictionary with memory estimates in MB
    """
    if seq_len is None:
        seq_len = config.max_seq_len

    # Model parameters
    total_params = sum(p.numel() for p in model.parameters())

    # Active parameters
    if hasattr(model, "estimate_active_params"):
        active_params = model.estimate_active_params()
    else:
        active_params = total_params

    # Memory for model weights (4 bytes per float32 param)
    model_memory_mb = (total_params * 4) / (1024**2)

    # Estimate activation memory
    # Rough estimate: batch_size * seq_len * d_model * n_layers * 4 (bytes) * 2 (forward + backward)
    d_model = config.d_model
    n_layers = config.n_layers
    activation_memory_mb = (batch_size * seq_len * d_model * n_layers * 4 * 2) / (
        1024**2
    )

    # Total memory
    total_memory_mb = model_memory_mb + activation_memory_mb

    return {
        "total_params": total_params,
        "active_params": active_params,
        "model_memory_mb": model_memory_mb,
        "activation_memory_mb": activation_memory_mb,
        "total_memory_mb": total_memory_mb,
    }
