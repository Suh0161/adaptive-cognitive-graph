# ACG Component Connection Guide

This guide shows you exactly how all components are connected and how to verify they work together.

## Quick Verification

Run these commands to verify everything is connected:

```bash
# 1. Test all connections (RECOMMENDED)
python test_integration.py

# 2. Visualize component structure
python visualize_connections.py

# 3. Quick smoke test
python -c "from acg import ACGConfig, ACGModel; import torch; m = ACGModel(ACGConfig()); print('✅ All imports work!')"
```

## Component Connection Map

### 1. Configuration Flow

```
ACGConfig (config.py)
    ↓
ACGModel (model.py)
    ↓
All 6 Components (models/*.py)
```

**Verification:**
```python
from acg import ACGConfig, ACGModel

config = ACGConfig()  # Creates config
model = ACGModel(config)  # Passes config to all components
```

### 2. Data Flow (Forward Pass)

```
Input tokens
    ↓
SemanticEncoder (encoder.py)
    ↓
IntentRouter (router.py)
    ↓
GraphOfExperts (experts.py)
    ↓
MemoryAdapter (adapter.py)
    ↓
VerifierExpert (verifier.py)
    ↓
OutputFusionLayer (fusion.py)
    ↓
Output logits + confidence
```

**Verification:**
```python
import torch
from acg import ACGConfig, ACGModel

model = ACGModel(ACGConfig())
token_ids = torch.randint(0, 50257, (2, 128))
logits, confidence = model(token_ids)  # Data flows through all components
print(f"✅ Output shape: {logits.shape}")
```

### 3. Training Flow

```
ACGModel
    ↓
ACGTrainer (training.py)
    ↓
Forward pass → Compute losses → Backward pass → Update weights
```

**Verification:**
```python
from acg import ACGConfig, ACGModel
from acg.training import ACGTrainer
import torch

model = ACGModel(ACGConfig())
trainer = ACGTrainer(model, model.config)

token_ids = torch.randint(0, 50257, (2, 128))
targets = torch.randint(0, 50257, (2, 128))

metrics = trainer.training_step(token_ids, targets)
print(f"✅ Loss: {metrics['total_loss']:.4f}")
```

### 4. Example Scripts Connection

```
examples/train.py
    ↓
imports: ACGConfig, ACGModel, ACGTrainer
    ↓
Uses: Complete training pipeline

examples/inference.py
    ↓
imports: ACGConfig, ACGModel
    ↓
Uses: Model for generation
```

**Verification:**
```bash
# Check scripts exist and are valid
python -m py_compile examples/train.py
python -m py_compile examples/inference.py
```

## Import Dependency Graph

```
acg/__init__.py
    ├── exports ACGConfig
    └── exports ACGModel

acg/config.py
    └── ACGConfig (no dependencies)

acg/model.py
    ├── imports ACGConfig
    ├── imports SemanticEncoder
    ├── imports IntentRouter
    ├── imports GraphOfExperts
    ├── imports MemoryAdapter
    ├── imports VerifierExpert
    └── imports OutputFusionLayer

acg/models/*.py
    ├── imports BaseModule
    ├── imports ACGConfig
    └── imports utils (RoutingMap, validation, etc.)

acg/training.py
    ├── imports ACGConfig
    └── imports torch modules

examples/*.py
    ├── imports ACGConfig
    ├── imports ACGModel
    └── imports ACGTrainer
```

## How to Verify Each Connection

### ✅ Config → Model
```python
from acg import ACGConfig, ACGModel
config = ACGConfig(d_model=512)
model = ACGModel(config)
assert model.config.d_model == 512
print("✅ Config connected to model")
```

### ✅ Model → Encoder
```python
from acg import ACGConfig, ACGModel
import torch

model = ACGModel(ACGConfig())
token_ids = torch.randint(0, 50257, (2, 128))
hidden = model.encoder(token_ids)
assert hidden.shape[-1] == model.config.d_model
print("✅ Encoder connected")
```

### ✅ Encoder → Router
```python
from acg import ACGConfig, ACGModel
import torch

model = ACGModel(ACGConfig())
token_ids = torch.randint(0, 50257, (2, 128))
hidden = model.encoder(token_ids)
routing = model.router(hidden)
assert routing.expert_ids.shape[1] == model.config.active_experts
print("✅ Router connected")
```

### ✅ Router → Experts
```python
from acg import ACGConfig, ACGModel
import torch

model = ACGModel(ACGConfig())
token_ids = torch.randint(0, 50257, (2, 128))
hidden = model.encoder(token_ids)
routing = model.router(hidden)
expert_out = model.experts(hidden, routing.expert_ids, routing.expert_gates)
assert len(expert_out) > 0
print("✅ Experts connected")
```

### ✅ Experts → Verifier
```python
from acg import ACGConfig, ACGModel
import torch

model = ACGModel(ACGConfig())
token_ids = torch.randint(0, 50257, (2, 128))
hidden = model.encoder(token_ids)
routing = model.router(hidden)
expert_out = model.experts(hidden, routing.expert_ids, routing.expert_gates)
verified, conf = model.verifier(expert_out[0])
assert conf.shape[0] == token_ids.shape[0]
print("✅ Verifier connected")
```

### ✅ Verifier → Fusion
```python
from acg import ACGConfig, ACGModel
import torch

model = ACGModel(ACGConfig())
token_ids = torch.randint(0, 50257, (2, 128))
hidden = model.encoder(token_ids)
routing = model.router(hidden)
expert_out = model.experts(hidden, routing.expert_ids, routing.expert_gates)
verified, conf = model.verifier(expert_out[0])
logits = model.fusion([verified], conf.unsqueeze(1), hidden)
assert logits.shape[-1] == model.config.vocab_size
print("✅ Fusion connected")
```

### ✅ Complete Forward Pass
```python
from acg import ACGConfig, ACGModel
import torch

model = ACGModel(ACGConfig())
token_ids = torch.randint(0, 50257, (2, 128))
logits, confidence = model(token_ids)
assert logits.shape == (2, 128, 50257)
assert confidence.shape == (2,)
print("✅ Complete forward pass works!")
```

## Common Connection Issues

### Issue: Import Error
```
ImportError: cannot import name 'ACGModel'
```
**Solution:** Make sure you're in the project root directory
```bash
cd /path/to/acg
python -c "from acg import ACGModel"
```

### Issue: Shape Mismatch
```
RuntimeError: shape mismatch
```
**Solution:** Check that all components use the same `d_model`
```python
config = ACGConfig(d_model=512)  # All components will use 512
```

### Issue: Missing Component
```
AttributeError: 'ACGModel' object has no attribute 'encoder'
```
**Solution:** Model initialization failed. Check error messages during `ACGModel(config)`

## Testing Checklist

Use this checklist to verify all connections:

- [ ] Configuration creates successfully: `ACGConfig()`
- [ ] Model initializes: `ACGModel(config)`
- [ ] All 6 components exist: encoder, router, experts, adapters, verifier, fusion
- [ ] Forward pass works: `model(token_ids)`
- [ ] Returns correct shapes: logits and confidence
- [ ] Training step works: `trainer.training_step()`
- [ ] All losses computed: lm, balance, verify, phase
- [ ] Example scripts exist: train.py, inference.py
- [ ] Config files valid: small.json, medium.json, large.json

## Automated Testing

Run the comprehensive test suite:

```bash
# Full integration test
python test_integration.py

# Expected output:
# ✅ PASS: Configuration
# ✅ PASS: Model Initialization
# ✅ PASS: Forward Pass
# ✅ PASS: Routing Information
# ✅ PASS: Training Step
# ✅ PASS: Component Connections
# ✅ PASS: Config Files
# ✅ PASS: Example Scripts
# 
# 🎉 ALL TESTS PASSED!
```

## Visual Verification

Generate visual diagrams:

```bash
python visualize_connections.py
```

This will show:
- Component tree structure
- Data flow diagram
- Parameter flow during training
- Module import connections

## Summary

**All components are connected if:**

1. ✅ You can import: `from acg import ACGConfig, ACGModel`
2. ✅ You can create: `model = ACGModel(ACGConfig())`
3. ✅ You can run: `logits, conf = model(token_ids)`
4. ✅ You can train: `trainer.training_step(token_ids, targets)`
5. ✅ Tests pass: `python test_integration.py`

**Quick verification command:**
```bash
python -c "from acg import ACGModel, ACGConfig; import torch; m = ACGModel(ACGConfig()); t = torch.randint(0, 50257, (2, 128)); l, c = m(t); print(f'✅ All connected! Output: {l.shape}')"
```

If this runs without errors, everything is properly connected! 🎉
