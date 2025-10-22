# Contributing to ACG

Thank you for your interest in contributing to the Adaptive Cognitive Graph project! We welcome contributions from the community.

## How to Contribute

### Reporting Bugs

If you find a bug, please open an issue on GitHub with:

1. **Clear title**: Describe the issue concisely
2. **Description**: Detailed explanation of the problem
3. **Steps to reproduce**: How to recreate the issue
4. **Expected behavior**: What should happen
5. **Actual behavior**: What actually happens
6. **Environment**: OS, Python version, PyTorch version, GPU info
7. **Code snippet**: Minimal code to reproduce (if applicable)

### Suggesting Features

Feature requests are welcome! Please open an issue with:

1. **Use case**: Why is this feature needed?
2. **Proposed solution**: How should it work?
3. **Alternatives**: Other approaches you've considered
4. **Additional context**: Any relevant information

### Pull Requests

We love pull requests! Here's the process:

1. **Fork the repository**
   ```bash
   git clone https://github.com/yourusername/acg.git
   cd acg
   ```

2. **Create a branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Make your changes**
   - Write clean, readable code
   - Follow existing code style
   - Add tests for new functionality
   - Update documentation as needed

4. **Test your changes**
   ```bash
   pytest tests/
   ```

5. **Commit your changes**
   ```bash
   git add .
   git commit -m "Add feature: description"
   ```

6. **Push to your fork**
   ```bash
   git push origin feature/your-feature-name
   ```

7. **Open a Pull Request**
   - Provide a clear description of changes
   - Reference any related issues
   - Ensure all tests pass

## Code Style

- Follow PEP 8 for Python code
- Use type hints where appropriate
- Write docstrings for all public functions/classes
- Keep functions focused and modular
- Add comments for complex logic

### Example

```python
def compute_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Compute cross-entropy loss with optional masking.
    
    Args:
        logits: (batch, seq_len, vocab_size) - Model predictions
        targets: (batch, seq_len) - Target token IDs
        mask: (batch, seq_len) - Optional mask (1=valid, 0=ignore)
        
    Returns:
        loss: Scalar tensor with computed loss
    """
    # Implementation here
    pass
```

## Testing

All new features should include tests:

```python
def test_new_feature():
    """Test description."""
    # Setup
    config = ACGConfig()
    model = ACGModel(config)
    
    # Test
    result = model.new_feature()
    
    # Assert
    assert result is not None
```

Run tests with:
```bash
pytest tests/ -v
```

## Documentation

Update documentation when:
- Adding new features
- Changing existing behavior
- Adding configuration options
- Modifying APIs

Documentation locations:
- `README.md` - Main documentation
- Docstrings - Inline code documentation
- `examples/` - Usage examples

## Areas for Contribution

We especially welcome contributions in:

### High Priority
- [ ] Pre-training scripts for large datasets
- [ ] Hugging Face integration
- [ ] Flash Attention 2 support
- [ ] Quantization (INT8, INT4)
- [ ] Benchmark suite

### Medium Priority
- [ ] Additional expert topologies
- [ ] Multi-modal extensions
- [ ] ONNX export
- [ ] TensorRT optimization
- [ ] Distributed training improvements

### Low Priority
- [ ] Additional example scripts
- [ ] Visualization tools
- [ ] Documentation improvements
- [ ] Tutorial notebooks

## Code Review Process

1. Maintainers will review your PR within 1-2 weeks
2. Address any requested changes
3. Once approved, your PR will be merged
4. Your contribution will be acknowledged in release notes

## Questions?

If you have questions about contributing:
- Open a GitHub Discussion
- Comment on relevant issues
- Reach out to maintainers

## Code of Conduct

Be respectful and constructive:
- Welcome newcomers
- Be patient with questions
- Provide constructive feedback
- Focus on the code, not the person

## License

By contributing, you agree that your contributions will be licensed under the Apache License 2.0.

Thank you for contributing to ACG! 🚀
