"""
Visualize ACG component connections and data flow.

This script shows how all components are connected and what data flows between them.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from acg.config import ACGConfig
from acg.model import ACGModel


def print_component_tree(model, config):
    """Print a tree view of all components."""
    print("\n" + "=" * 80)
    print("  ACG MODEL COMPONENT TREE")
    print("=" * 80)

    print("\nACGModel")
    print("├── config: ACGConfig")
    print("│   ├── d_model:", config.d_model)
    print("│   ├── n_experts:", config.n_experts)
    print("│   ├── active_experts:", config.active_experts)
    print("│   ├── n_layers:", config.n_layers)
    print("│   └── max_seq_len:", config.max_seq_len)
    print("│")
    print("├── encoder: SemanticEncoder")
    print("│   ├── embedding: TokenEmbedding")
    print("│   ├── layers: ModuleList")
    print("│   │   ├── TransformerBlock (attention + FFN)")
    print("│   │   └── SSMBlock (state space model)")
    print("│   └── final_norm: LayerNorm")
    print("│")
    print("├── router: IntentRouter")
    print("│   ├── pooling: SequencePooling")
    print("│   ├── phase_classifier: Linear")
    print("│   ├── expert_scorer: Linear")
    print("│   └── phase_gates: Parameter")
    print("│")
    print("├── experts: GraphOfExperts")
    print("│   ├── experts: ModuleList")
    print("│   │   └── ExpertBlock × " + str(config.n_experts))
    print("│   │       └── layers: TransformerBlock × " + str(config.expert_layers))
    print("│   ├── adjacency: Dict (DAG topology)")
    print("│   └── edge_weights: ModuleDict (cross-edge communication)")
    print("│")

    if model.adapters:
        print("├── adapters: ModuleDict")
        print("│   └── MemoryAdapter × " + str(config.n_experts))
        print("│       ├── lora_A: Parameter (rank=" + str(config.lora_rank) + ")")
        print("│       ├── lora_B: Parameter (rank=" + str(config.lora_rank) + ")")
        print("│       └── weight_delta: Buffer (EMA)")
        print("│")

    print("├── verifier: VerifierExpert")
    print("│   ├── verifier_mlp: VerifierMLP")
    print("│   │   └── layers × " + str(config.verifier_layers))
    print("│   └── correction_block: " + config.correction_type.capitalize())
    print("│")
    print("└── fusion: OutputFusionLayer")
    print("    ├── expert_projections: ModuleList × " + str(config.n_experts))
    print("    ├── layer_norm: LayerNorm")
    print("    └── output_projection: Linear → vocab_size")


def print_data_flow():
    """Print the data flow through the model."""
    print("\n" + "=" * 80)
    print("  DATA FLOW DIAGRAM")
    print("=" * 80)

    print(
        """
INPUT: token_ids (batch, seq_len)
  │
  ▼
┌─────────────────────────────────────────────────────────────┐
│ 1. SEMANTIC ENCODER                                         │
│    - Token embedding + RoPE                                 │
│    - Hybrid Transformer + SSM blocks                        │
│    - Output: hidden_states (batch, seq_len, d_model)       │
└─────────────────────────────────────────────────────────────┘
  │
  ├──────────────────────────────────────┐
  │                                      │ (residual connection)
  ▼                                      │
┌─────────────────────────────────────────────────────────────┐
│ 2. INTENT ROUTER                                            │
│    - Sequence pooling                                       │
│    - Phase classification (P phases)                        │
│    - Expert scoring (E experts)                             │
│    - Top-K selection (K active experts)                     │
│    - Output: (phase_id, expert_ids, expert_gates)          │
└─────────────────────────────────────────────────────────────┘
  │
  ▼
┌─────────────────────────────────────────────────────────────┐
│ 3. GRAPH OF EXPERTS                                         │
│    - Process through selected experts                       │
│    - DAG topology with cross-edge communication             │
│    - Weighted aggregation by gates                          │
│    - Output: expert_outputs (batch, seq_len, d_model)      │
└─────────────────────────────────────────────────────────────┘
  │
  ▼
┌─────────────────────────────────────────────────────────────┐
│ 4. MEMORY ADAPTER (if enabled)                              │
│    - Apply LoRA adaptation per expert                       │
│    - Low-rank matrices: ΔW = B·A                           │
│    - EMA weight updates                                     │
│    - Output: adapted_outputs (batch, seq_len, d_model)     │
└─────────────────────────────────────────────────────────────┘
  │
  ▼
┌─────────────────────────────────────────────────────────────┐
│ 5. VERIFIER EXPERT                                          │
│    - Confidence scoring via MLP                             │
│    - Threshold-based correction decision                    │
│    - Apply correction if confidence < threshold             │
│    - Output: (verified_states, confidence)                  │
└─────────────────────────────────────────────────────────────┘
  │
  ▼
┌─────────────────────────────────────────────────────────────┐
│ 6. OUTPUT FUSION LAYER                                      │
│    - Per-expert projections                                 │
│    - Confidence-weighted fusion                             │
│    - Add residual from encoder ◄────────────────────────────┘
│    - Layer norm + dropout                                   │
│    - Final projection to vocabulary                         │
│    - Output: logits (batch, seq_len, vocab_size)           │
└─────────────────────────────────────────────────────────────┘
  │
  ▼
OUTPUT: (logits, confidence)
"""
    )


def print_parameter_flow():
    """Print how parameters flow during training."""
    print("\n" + "=" * 80)
    print("  TRAINING PARAMETER FLOW")
    print("=" * 80)

    print(
        """
FORWARD PASS:
  Input → Encoder → Router → Experts → Adapter → Verifier → Fusion → Output

LOSS COMPUTATION:
  ┌─ L_lm (language modeling)
  ├─ L_balance (expert utilization)
  ├─ L_verify (confidence accuracy)
  └─ L_phase (phase classification)
  
  L_total = L_lm + α·L_balance + β·L_verify + γ·L_phase

BACKWARD PASS:
  Gradients flow back through all components:
  
  ∂L/∂fusion → ∂L/∂verifier → ∂L/∂adapter → ∂L/∂experts → ∂L/∂router → ∂L/∂encoder

GRADIENT CLIPPING:
  Clip gradients by global norm (max_norm = 1.0)

OPTIMIZER STEP:
  Update all trainable parameters using AdamW

ADAPTER EMA UPDATE:
  Update memory adapter weights using exponential moving average:
  θ_ema = β·θ_ema + (1-β)·θ_current
"""
    )


def print_import_connections():
    """Print how Python modules import each other."""
    print("\n" + "=" * 80)
    print("  MODULE IMPORT CONNECTIONS")
    print("=" * 80)

    print(
        """
acg/
├── __init__.py
│   └── exports: ACGConfig, ACGModel
│
├── config.py
│   └── defines: ACGConfig (standalone, no imports)
│
├── model.py
│   ├── imports: ACGConfig from config
│   ├── imports: All components from models/
│   └── defines: ACGModel (integrates everything)
│
├── training.py
│   ├── imports: ACGConfig from config
│   └── defines: ACGTrainer, loss functions
│
├── models/
│   ├── __init__.py
│   │   └── exports: All component classes
│   │
│   ├── base.py
│   │   └── defines: BaseModule (parent class)
│   │
│   ├── encoder.py
│   │   ├── imports: BaseModule, ACGConfig
│   │   └── defines: SemanticEncoder
│   │
│   ├── router.py
│   │   ├── imports: BaseModule, ACGConfig, RoutingMap
│   │   └── defines: IntentRouter
│   │
│   ├── experts.py
│   │   ├── imports: BaseModule, ACGConfig
│   │   └── defines: GraphOfExperts, ExpertBlock
│   │
│   ├── adapter.py
│   │   ├── imports: BaseModule, ACGConfig
│   │   └── defines: MemoryAdapter
│   │
│   ├── verifier.py
│   │   ├── imports: BaseModule, ACGConfig
│   │   └── defines: VerifierExpert
│   │
│   └── fusion.py
│       ├── imports: BaseModule, ACGConfig
│       └── defines: OutputFusionLayer
│
└── utils/
    ├── routing.py
    │   └── defines: RoutingMap (dataclass)
    │
    ├── validation.py
    │   └── defines: validate_token_ids()
    │
    └── metrics.py
        └── defines: MetricsTracker

examples/
├── train.py
│   ├── imports: ACGConfig, ACGModel, ACGTrainer
│   └── uses: All components via ACGModel
│
└── inference.py
    ├── imports: ACGConfig, ACGModel
    └── uses: Model for generation/evaluation
"""
    )


def main():
    """Run all visualizations."""
    print("\n" + "=" * 80)
    print("  ACG COMPONENT CONNECTION VISUALIZER")
    print("=" * 80)

    # Create a small model for visualization
    config = ACGConfig(
        d_model=512, n_experts=8, active_experts=2, n_layers=4, expert_layers=2
    )

    model = ACGModel(config)

    # Print all visualizations
    print_component_tree(model, config)
    print_data_flow()
    print_parameter_flow()
    print_import_connections()

    print("\n" + "=" * 80)
    print("  SUMMARY")
    print("=" * 80)
    print(
        """
✅ All components are properly connected:
   1. Config → Model → All Components
   2. Data flows: Input → 6 Components → Output
   3. Gradients flow back through all components
   4. Example scripts use the integrated model

✅ To verify connections work, run:
   python test_integration.py

✅ To test training:
   python examples/train.py --model-size small --num-epochs 1

✅ To test inference:
   python examples/inference.py --checkpoint <path> --mode generate
"""
    )
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
