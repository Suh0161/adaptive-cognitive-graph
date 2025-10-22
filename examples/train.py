"""
Training script for ACG model with example configuration.

This script demonstrates how to train an ACG model with:
- Data loading and preprocessing
- Checkpoint saving and loading
- Metric logging and visualization
- Mixed precision training support

"""

import os
import json
import argparse
from pathlib import Path
from typing import Optional, Dict
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# Import ACG components
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from acg.config import ACGConfig
from acg.model import ACGModel
from acg.training import ACGTrainer, compute_total_loss
from acg.utils.metrics import MetricsTracker


class DummyTextDataset(Dataset):
    """
    Dummy dataset for demonstration purposes.
    
    In production, replace with your actual dataset that loads text data,
    tokenizes it, and returns token IDs.
    """
    
    def __init__(self, num_samples: int, seq_len: int, vocab_size: int):
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.vocab_size = vocab_size
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # Generate random token IDs for demonstration
        token_ids = torch.randint(0, self.vocab_size, (self.seq_len,))
        
        # Targets are shifted by 1 (next token prediction)
        targets = torch.cat([token_ids[1:], torch.tensor([0])])
        
        return {
            'token_ids': token_ids,
            'targets': targets
        }


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train ACG model')
    
    # Model configuration
    parser.add_argument('--config', type=str, default=None,
                       help='Path to config JSON file')
    parser.add_argument('--model-size', type=str, default='small',
                       choices=['small', 'medium', 'large'],
                       help='Predefined model size')
    
    # Training configuration
    parser.add_argument('--batch-size', type=int, default=8,
                       help='Training batch size')
    parser.add_argument('--num-epochs', type=int, default=10,
                       help='Number of training epochs')
    parser.add_argument('--learning-rate', type=float, default=3e-4,
                       help='Learning rate')
    parser.add_argument('--grad-clip', type=float, default=1.0,
                       help='Gradient clipping max norm')
    
    # Data configuration
    parser.add_argument('--seq-len', type=int, default=512,
                       help='Sequence length')
    parser.add_argument('--num-samples', type=int, default=1000,
                       help='Number of training samples')
    
    # Checkpoint configuration
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints',
                       help='Directory to save checkpoints')
    parser.add_argument('--save-every', type=int, default=100,
                       help='Save checkpoint every N steps')
    parser.add_argument('--resume-from', type=str, default=None,
                       help='Path to checkpoint to resume from')
    
    # Logging configuration
    parser.add_argument('--log-every', type=int, default=10,
                       help='Log metrics every N steps')
    parser.add_argument('--log-dir', type=str, default='logs',
                       help='Directory to save logs')
    
    # Device configuration
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to use for training')
    parser.add_argument('--mixed-precision', action='store_true',
                       help='Use mixed precision training')
    
    return parser.parse_args()


def load_config(args) -> ACGConfig:
    """
    Load model configuration from file or use predefined size.
    
    Args:
        args: Command line arguments
        
    Returns:
        ACGConfig instance
    """
    if args.config:
        # Load from JSON file
        with open(args.config, 'r') as f:
            config_dict = json.load(f)
        config = ACGConfig(**config_dict)
    else:
        # Use predefined size from examples/configs/
        config_path = Path(__file__).parent / 'configs' / f'{args.model_size}.json'
        if config_path.exists():
            with open(config_path, 'r') as f:
                config_dict = json.load(f)
            config = ACGConfig(**config_dict)
        else:
            # Use default config
            config = ACGConfig()
    
    # Override with command line arguments
    config.batch_size = args.batch_size
    config.learning_rate = args.learning_rate
    config.grad_clip = args.grad_clip
    config.max_seq_len = args.seq_len
    
    return config


def save_checkpoint(
    model: ACGModel,
    optimizer: torch.optim.Optimizer,
    step: int,
    epoch: int,
    metrics: Dict,
    checkpoint_dir: str,
    is_best: bool = False
):
    """
    Save model checkpoint.
    
    Args:
        model: ACG model
        optimizer: Optimizer
        step: Current training step
        epoch: Current epoch
        metrics: Current metrics
        checkpoint_dir: Directory to save checkpoint
        is_best: Whether this is the best checkpoint
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    checkpoint = {
        'step': step,
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics,
        'config': model.config.__dict__
    }
    
    # Save regular checkpoint
    checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_step_{step}.pt')
    torch.save(checkpoint, checkpoint_path)
    print(f"Saved checkpoint to {checkpoint_path}")
    
    # Save best checkpoint
    if is_best:
        best_path = os.path.join(checkpoint_dir, 'best_checkpoint.pt')
        torch.save(checkpoint, best_path)
        print(f"Saved best checkpoint to {best_path}")
    
    # Save latest checkpoint (for easy resuming)
    latest_path = os.path.join(checkpoint_dir, 'latest_checkpoint.pt')
    torch.save(checkpoint, latest_path)


def load_checkpoint(
    checkpoint_path: str,
    model: ACGModel,
    optimizer: Optional[torch.optim.Optimizer] = None
) -> Dict:
    """
    Load model checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        model: ACG model
        optimizer: Optional optimizer to load state
        
    Returns:
        Dictionary with checkpoint metadata
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    print(f"Loaded checkpoint from {checkpoint_path}")
    print(f"  Step: {checkpoint['step']}")
    print(f"  Epoch: {checkpoint['epoch']}")
    
    return checkpoint


def train_epoch(
    model: ACGModel,
    trainer: ACGTrainer,
    dataloader: DataLoader,
    metrics_tracker: MetricsTracker,
    device: str,
    epoch: int,
    args
) -> Dict:
    """
    Train for one epoch.
    
    Args:
        model: ACG model
        trainer: ACG trainer
        dataloader: Training data loader
        metrics_tracker: Metrics tracker
        device: Device to use
        epoch: Current epoch number
        args: Command line arguments
        
    Returns:
        Dictionary with epoch metrics
    """
    model.train()
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
    
    for batch_idx, batch in enumerate(pbar):
        # Move batch to device
        token_ids = batch['token_ids'].to(device)
        targets = batch['targets'].to(device)
        
        # Training step
        step_metrics = trainer.training_step(
            token_ids=token_ids,
            targets=targets
        )
        
        # Update metrics tracker
        metrics_tracker.update(step_metrics)
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f"{step_metrics['total_loss']:.4f}",
            'grad': f"{step_metrics['grad_norm']:.2f}"
        })
        
        # Log metrics
        if trainer.step_count % args.log_every == 0:
            avg_metrics = metrics_tracker.get_average_metrics()
            print(f"\nStep {trainer.step_count}:")
            for key, value in avg_metrics.items():
                print(f"  {key}: {value:.4f}")
            
            # Save metrics to file
            metrics_tracker.save_to_file(
                os.path.join(args.log_dir, 'metrics.json')
            )
        
        # Save checkpoint
        if trainer.step_count % args.save_every == 0:
            save_checkpoint(
                model=model,
                optimizer=trainer.optimizer,
                step=trainer.step_count,
                epoch=epoch,
                metrics=metrics_tracker.get_average_metrics(),
                checkpoint_dir=args.checkpoint_dir
            )
    
    # Get epoch metrics
    epoch_metrics = metrics_tracker.get_average_metrics()
    
    return epoch_metrics


def main():
    """Main training function."""
    args = parse_args()
    
    print("=" * 80)
    print("ACG Model Training")
    print("=" * 80)
    
    # Load configuration
    print("\nLoading configuration...")
    config = load_config(args)
    print(f"Model configuration:")
    print(f"  Model dimension: {config.d_model}")
    print(f"  Number of experts: {config.n_experts}")
    print(f"  Active experts: {config.active_experts}")
    print(f"  Number of layers: {config.n_layers}")
    print(f"  Max sequence length: {config.max_seq_len}")
    
    # Create model
    print("\nInitializing model...")
    model = ACGModel(config)
    device = torch.device(args.device)
    model = model.to(device)
    
    print(f"Total parameters: {model.get_num_params():,}")
    print(f"Active parameters: {model.estimate_active_params():,}")
    
    # Create trainer
    print("\nInitializing trainer...")
    trainer = ACGTrainer(
        model=model,
        config=config,
        use_mixed_precision=args.mixed_precision
    )
    
    # Load checkpoint if resuming
    start_epoch = 0
    if args.resume_from:
        print(f"\nResuming from checkpoint: {args.resume_from}")
        checkpoint = load_checkpoint(args.resume_from, model, trainer.optimizer)
        start_epoch = checkpoint['epoch']
        trainer.step_count = checkpoint['step']
    
    # Create dataset and dataloader
    print("\nCreating dataset...")
    dataset = DummyTextDataset(
        num_samples=args.num_samples,
        seq_len=args.seq_len,
        vocab_size=config.vocab_size
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0  # Set to > 0 for parallel data loading
    )
    
    print(f"Dataset size: {len(dataset)} samples")
    print(f"Batch size: {args.batch_size}")
    print(f"Steps per epoch: {len(dataloader)}")
    
    # Create metrics tracker
    os.makedirs(args.log_dir, exist_ok=True)
    metrics_tracker = MetricsTracker(
        log_interval=args.log_every,
        log_dir=args.log_dir
    )
    
    # Training loop
    print("\n" + "=" * 80)
    print("Starting training...")
    print("=" * 80)
    
    best_loss = float('inf')
    
    for epoch in range(start_epoch, args.num_epochs):
        print(f"\n{'='*80}")
        print(f"Epoch {epoch + 1}/{args.num_epochs}")
        print(f"{'='*80}")
        
        # Train for one epoch
        epoch_metrics = train_epoch(
            model=model,
            trainer=trainer,
            dataloader=dataloader,
            metrics_tracker=metrics_tracker,
            device=device,
            epoch=epoch,
            args=args
        )
        
        # Print epoch summary
        print(f"\nEpoch {epoch + 1} Summary:")
        for key, value in epoch_metrics.items():
            print(f"  {key}: {value:.4f}")
        
        # Save checkpoint at end of epoch
        is_best = epoch_metrics['total_loss'] < best_loss
        if is_best:
            best_loss = epoch_metrics['total_loss']
        
        save_checkpoint(
            model=model,
            optimizer=trainer.optimizer,
            step=trainer.step_count,
            epoch=epoch + 1,
            metrics=epoch_metrics,
            checkpoint_dir=args.checkpoint_dir,
            is_best=is_best
        )
        
        # Reset metrics for next epoch
        metrics_tracker.reset()
    
    print("\n" + "=" * 80)
    print("Training completed!")
    print("=" * 80)
    print(f"Best loss: {best_loss:.4f}")
    print(f"Final checkpoint saved to: {args.checkpoint_dir}")


if __name__ == '__main__':
    main()
