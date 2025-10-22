# Quick Start Guide

Get up and running with ACG in 5 minutes!

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/acg.git
cd acg

# Install dependencies
pip install -r requirements.txt

# Install ACG package
pip install -e .
```

## Your First Model

### 1. Create a Simple Model

```python
from acg import ACGConfig, ACGModel
import torch

# Create configuration (small model for testing)
config = ACGConfig(
    d_model=512,
    n_experts=8,
    active_experts=2,
    n_layers=4,
    max_seq_len=512
)

# Initialize model
model = ACGModel(config)

print(f"Model created with {model.get_num_params():,} parameters")
```

### 2. Run a Forward Pass

```python
# Create dummy input
batch_size = 2
seq_len = 128
token_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))

# Forward pass
logits, confidence = model(token_ids)

print(f"Output shape: {logits.shape}")  # (2, 128, 50257)
print(f"Confidence: {confidence}")       # (2,)
```

### 3. Train the Model

```python
from acg.training import ACGTrainer

# Create trainer
trainer = ACGTrainer(model, config)

# Training step
targets = torch.randint(0, config.vocab_size, (batch_size, seq_len))
metrics = trainer.training_step(token_ids, targets)

print(f"Loss: {metrics['total_loss']:.4f}")
```

## Using Example Scripts

### Train a Model

```bash
# Train small model (safe for laptops)
python examples/train.py \
    --model-size small \
    --batch-size 4 \
    --num-epochs 5 \
    --checkpoint-dir checkpoints
```

### Run Inference

```bash
# Generate text
python examples/inference.py \
    --checkpoint checkpoints/best_checkpoint.pt \
    --mode generate \
    --prompt "Hello world" \
    --max-length 50
```

## Configuration Examples

### Small Model (Laptop-Friendly)
```python
config = ACGConfig(
    d_model=512,
    n_experts=8,
    active_experts=2,
    n_layers=4,
    max_seq_len=2048
)
# ~50M parameters, ~15M active
```

### Medium Model (Multi-GPU)
```python
config = ACGConfig(
    d_model=2048,
    n_experts=32,
    active_experts=4,
    n_layers=8,
    max_seq_len=32768
)
# ~7B parameters, ~1.5B active
```

### Large Model (Cluster)
```python
config = ACGConfig(
    d_model=4096,
    n_experts=64,
    active_experts=8,
    n_layers=16,
    max_seq_len=256000
)
# ~100B+ parameters, ~20B active
```

## Common Tasks

### Load a Checkpoint

```python
checkpoint = torch.load('checkpoints/best_checkpoint.pt')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
```

### Save a Checkpoint

```python
torch.save({
    'model_state_dict': model.state_dict(),
    'config': config.__dict__
}, 'my_checkpoint.pt')
```

### Use Mixed Precision

```python
trainer = ACGTrainer(
    model,
    config,
    use_mixed_precision=True,
    mixed_precision_dtype=torch.bfloat16
)
```

### Enable Gradient Checkpointing

```python
config = ACGConfig(
    use_gradient_checkpointing=True,
    # ... other params
)
```

## Troubleshooting

### Out of Memory?
- Reduce batch size: `--batch-size 1`
- Use gradient checkpointing: `use_gradient_checkpointing=True`
- Use smaller model: `--model-size small`
- Reduce sequence length: `--seq-len 128`

### Slow Training?
- Enable mixed precision: `--mixed-precision`
- Use GPU: `--device cuda`
- Increase batch size (if memory allows)
- Use multiple GPUs with DeepSpeed

### NaN Loss?
- Reduce learning rate: `--learning-rate 1e-5`
- Enable gradient clipping (already enabled by default)
- Check input data for NaN values
- Use mixed precision with bf16 instead of fp16

## Next Steps

1. **Read the full README**: Comprehensive documentation
2. **Explore examples**: Check `examples/` directory
3. **Run tests**: `pytest tests/`
4. **Customize config**: See `examples/configs/`
5. **Join discussions**: GitHub Discussions

## Resources

- **Documentation**: [README.md](README.md)
- **Examples**: [examples/](examples/)
- **Tests**: [tests/](tests/)
- **Issues**: [GitHub Issues](https://github.com/yourusername/acg/issues)

Happy modeling! 🚀
