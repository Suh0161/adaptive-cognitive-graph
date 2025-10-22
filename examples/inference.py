"""
Inference script for ACG model evaluation.

This script demonstrates how to use a trained ACG model for inference with:
- Model loading from checkpoint
- Batch processing support
- Optimized inference mode
- Text generation examples

"""

import os
import json
import argparse
from pathlib import Path
from typing import List, Optional
import torch
import torch.nn.functional as F
from tqdm import tqdm

# Import ACG components
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from acg.config import ACGConfig
from acg.model import ACGModel


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run inference with ACG model")

    # Model configuration
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to model checkpoint"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config JSON file (optional, will use checkpoint config)",
    )

    # Inference configuration
    parser.add_argument(
        "--mode",
        type=str,
        default="generate",
        choices=["generate", "evaluate", "interactive"],
        help="Inference mode",
    )
    parser.add_argument(
        "--prompt", type=str, default=None, help="Input prompt for generation"
    )
    parser.add_argument(
        "--max-length", type=int, default=100, help="Maximum generation length"
    )
    parser.add_argument(
        "--temperature", type=float, default=1.0, help="Sampling temperature"
    )
    parser.add_argument("--top-k", type=int, default=50, help="Top-k sampling")
    parser.add_argument(
        "--top-p", type=float, default=0.9, help="Top-p (nucleus) sampling"
    )

    # Batch processing
    parser.add_argument(
        "--input-file",
        type=str,
        default=None,
        help="File with input prompts (one per line)",
    )
    parser.add_argument(
        "--output-file", type=str, default=None, help="File to save outputs"
    )
    parser.add_argument(
        "--batch-size", type=int, default=1, help="Batch size for processing"
    )

    # Device configuration
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use for inference",
    )
    parser.add_argument(
        "--compile", action="store_true", help="Use torch.compile for optimization"
    )

    return parser.parse_args()


def load_model(
    checkpoint_path: str, config_path: Optional[str] = None, device: str = "cpu"
) -> ACGModel:
    """
    Load ACG model from checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file
        config_path: Optional path to config file
        device: Device to load model on

    Returns:
        Loaded ACG model in eval mode
    """
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    # Load configuration
    if config_path:
        with open(config_path, "r") as f:
            config_dict = json.load(f)
        config = ACGConfig(**config_dict)
    else:
        # Use config from checkpoint
        config = ACGConfig(**checkpoint["config"])

    # Create model
    print("Initializing model...")
    model = ACGModel(config)

    # Load weights
    model.load_state_dict(checkpoint["model_state_dict"])

    # Move to device and set to eval mode
    model = model.to(device)
    model.eval()

    print(f"Model loaded successfully")
    print(f"  Total parameters: {model.get_num_params():,}")
    print(f"  Active parameters: {model.estimate_active_params():,}")

    return model


@torch.no_grad()
def generate_text(
    model: ACGModel,
    input_ids: torch.Tensor,
    max_length: int = 100,
    temperature: float = 1.0,
    top_k: int = 50,
    top_p: float = 0.9,
    device: str = "cpu",
) -> torch.Tensor:
    """
    Generate text using the model with sampling.

    Args:
        model: ACG model
        input_ids: (batch, seq_len) - Input token IDs
        max_length: Maximum generation length
        temperature: Sampling temperature
        top_k: Top-k sampling parameter
        top_p: Top-p (nucleus) sampling parameter
        device: Device to use

    Returns:
        generated_ids: (batch, seq_len + max_length) - Generated token IDs

    """
    model.eval()

    batch_size, seq_len = input_ids.shape
    generated = input_ids.clone()

    for _ in range(max_length):
        # Forward pass
        logits, confidence = model(generated)

        # Get logits for last token
        next_token_logits = logits[:, -1, :]  # (batch, vocab_size)

        # Apply temperature
        if temperature != 1.0:
            next_token_logits = next_token_logits / temperature

        # Apply top-k filtering
        if top_k > 0:
            indices_to_remove = (
                next_token_logits
                < torch.topk(next_token_logits, top_k)[0][..., -1, None]
            )
            next_token_logits[indices_to_remove] = float("-inf")

        # Apply top-p (nucleus) filtering
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(
                next_token_logits, descending=True
            )
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

            # Remove tokens with cumulative probability above threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
                ..., :-1
            ].clone()
            sorted_indices_to_remove[..., 0] = 0

            indices_to_remove = sorted_indices_to_remove.scatter(
                1, sorted_indices, sorted_indices_to_remove
            )
            next_token_logits[indices_to_remove] = float("-inf")

        # Sample from distribution
        probs = F.softmax(next_token_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)  # (batch, 1)

        # Append to generated sequence
        generated = torch.cat([generated, next_token], dim=1)

        # Check if all sequences have generated EOS token (assuming 0 is EOS)
        # In practice, you'd check for your specific EOS token
        if (next_token == 0).all():
            break

    return generated


@torch.no_grad()
def evaluate_batch(
    model: ACGModel, token_ids: torch.Tensor, targets: torch.Tensor, device: str = "cpu"
) -> dict:
    """
    Evaluate model on a batch of data.

    Args:
        model: ACG model
        token_ids: (batch, seq_len) - Input token IDs
        targets: (batch, seq_len) - Target token IDs
        device: Device to use

    Returns:
        Dictionary with evaluation metrics

    """
    model.eval()

    # Forward pass
    logits, confidence = model(token_ids)

    # Compute loss
    batch_size, seq_len, vocab_size = logits.shape
    logits_flat = logits.view(-1, vocab_size)
    targets_flat = targets.view(-1)

    loss = F.cross_entropy(logits_flat, targets_flat)

    # Compute perplexity
    perplexity = torch.exp(loss)

    # Compute accuracy
    predictions = logits.argmax(dim=-1)
    accuracy = (predictions == targets).float().mean()

    # Average confidence
    avg_confidence = confidence.mean()

    return {
        "loss": loss.item(),
        "perplexity": perplexity.item(),
        "accuracy": accuracy.item(),
        "confidence": avg_confidence.item(),
    }


def generate_mode(args, model: ACGModel, device: str):
    """
    Run generation mode.

    Args:
        args: Command line arguments
        model: ACG model
        device: Device to use
    """
    if args.prompt is None:
        print("Error: --prompt is required for generate mode")
        return

    print(f"\nPrompt: {args.prompt}")
    print(
        f"Generating with max_length={args.max_length}, temperature={args.temperature}..."
    )

    # Tokenize prompt (dummy tokenization for demonstration)
    # In practice, use your actual tokenizer
    prompt_ids = torch.randint(0, model.config.vocab_size, (1, 10)).to(device)

    # Generate
    generated_ids = generate_text(
        model=model,
        input_ids=prompt_ids,
        max_length=args.max_length,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        device=device,
    )

    print(f"\nGenerated {generated_ids.shape[1]} tokens")
    print(f"Token IDs: {generated_ids[0].tolist()[:20]}...")  # Show first 20 tokens

    # In practice, decode tokens to text using your tokenizer
    print("\nNote: Use your tokenizer to decode token IDs to text")


def evaluate_mode(args, model: ACGModel, device: str):
    """
    Run evaluation mode.

    Args:
        args: Command line arguments
        model: ACG model
        device: Device to use
    """
    if args.input_file is None:
        print("Error: --input-file is required for evaluate mode")
        return

    print(f"\nEvaluating on {args.input_file}...")

    # Load data (dummy data for demonstration)
    # In practice, load your actual evaluation dataset
    num_samples = 100
    seq_len = 512

    print(f"Processing {num_samples} samples with batch_size={args.batch_size}...")

    all_metrics = []

    # Process in batches
    for i in tqdm(range(0, num_samples, args.batch_size)):
        batch_size = min(args.batch_size, num_samples - i)

        # Generate dummy batch (replace with actual data loading)
        token_ids = torch.randint(0, model.config.vocab_size, (batch_size, seq_len)).to(
            device
        )
        targets = torch.randint(0, model.config.vocab_size, (batch_size, seq_len)).to(
            device
        )

        # Evaluate batch
        metrics = evaluate_batch(model, token_ids, targets, device)
        all_metrics.append(metrics)

    # Compute average metrics
    avg_metrics = {
        key: sum(m[key] for m in all_metrics) / len(all_metrics)
        for key in all_metrics[0].keys()
    }

    print("\nEvaluation Results:")
    print(f"  Loss: {avg_metrics['loss']:.4f}")
    print(f"  Perplexity: {avg_metrics['perplexity']:.2f}")
    print(f"  Accuracy: {avg_metrics['accuracy']:.4f}")
    print(f"  Confidence: {avg_metrics['confidence']:.4f}")

    # Save results if output file specified
    if args.output_file:
        with open(args.output_file, "w") as f:
            json.dump(avg_metrics, f, indent=2)
        print(f"\nResults saved to {args.output_file}")


def interactive_mode(args, model: ACGModel, device: str):
    """
    Run interactive mode.

    Args:
        args: Command line arguments
        model: ACG model
        device: Device to use
    """
    print("\n" + "=" * 80)
    print("Interactive Mode")
    print("=" * 80)
    print("Enter prompts to generate text. Type 'quit' to exit.")
    print()

    while True:
        try:
            prompt = input("Prompt: ").strip()

            if prompt.lower() in ["quit", "exit", "q"]:
                break

            if not prompt:
                continue

            # Tokenize prompt (dummy tokenization)
            # In practice, use your actual tokenizer
            prompt_ids = torch.randint(0, model.config.vocab_size, (1, 10)).to(device)

            # Generate
            print("Generating...")
            generated_ids = generate_text(
                model=model,
                input_ids=prompt_ids,
                max_length=args.max_length,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                device=device,
            )

            print(f"Generated: {generated_ids.shape[1]} tokens")
            print(f"Token IDs: {generated_ids[0].tolist()[:50]}...")
            print()

        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")
            continue


def batch_process_mode(args, model: ACGModel, device: str):
    """
    Process multiple prompts from file in batches.

    Args:
        args: Command line arguments
        model: ACG model
        device: Device to use
    """
    if args.input_file is None:
        print("Error: --input-file is required for batch processing")
        return

    print(f"\nBatch processing prompts from {args.input_file}...")

    # Load prompts
    with open(args.input_file, "r") as f:
        prompts = [line.strip() for line in f if line.strip()]

    print(f"Loaded {len(prompts)} prompts")
    print(f"Processing with batch_size={args.batch_size}...")

    all_outputs = []

    # Process in batches
    for i in tqdm(range(0, len(prompts), args.batch_size)):
        batch_prompts = prompts[i : i + args.batch_size]
        batch_size = len(batch_prompts)

        # Tokenize batch (dummy tokenization)
        # In practice, use your actual tokenizer
        prompt_ids = torch.randint(0, model.config.vocab_size, (batch_size, 10)).to(
            device
        )

        # Generate
        generated_ids = generate_text(
            model=model,
            input_ids=prompt_ids,
            max_length=args.max_length,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            device=device,
        )

        # Store outputs
        for j, prompt in enumerate(batch_prompts):
            all_outputs.append(
                {"prompt": prompt, "generated_ids": generated_ids[j].tolist()}
            )

    # Save outputs if specified
    if args.output_file:
        with open(args.output_file, "w") as f:
            json.dump(all_outputs, f, indent=2)
        print(f"\nOutputs saved to {args.output_file}")
    else:
        # Print first few outputs
        print("\nFirst 3 outputs:")
        for output in all_outputs[:3]:
            print(f"  Prompt: {output['prompt']}")
            print(f"  Generated: {output['generated_ids'][:20]}...")
            print()


def main():
    """Main inference function."""
    args = parse_args()

    print("=" * 80)
    print("ACG Model Inference")
    print("=" * 80)

    # Load model
    device = torch.device(args.device)
    model = load_model(args.checkpoint, args.config, device)

    # Compile model if requested (PyTorch 2.0+)
    if args.compile:
        print("\nCompiling model with torch.compile...")
        try:
            model = torch.compile(model)
            print("Model compiled successfully")
        except Exception as e:
            print(f"Warning: Could not compile model: {e}")

    # Run inference based on mode
    if args.mode == "generate":
        generate_mode(args, model, device)
    elif args.mode == "evaluate":
        evaluate_mode(args, model, device)
    elif args.mode == "interactive":
        interactive_mode(args, model, device)

    # Batch processing if input file provided
    if args.input_file and args.mode == "generate":
        batch_process_mode(args, model, device)

    print("\n" + "=" * 80)
    print("Inference completed!")
    print("=" * 80)


if __name__ == "__main__":
    main()
