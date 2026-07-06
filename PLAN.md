Project TetraFT: Resource Planning & Memory Estimator

Quantization-Aware Fine-Tuning (QAFT) for 2-Bit Quaternary LLMs

In standard post-training quantization (PTQ), we only need enough memory to load the model weights. However, in Quantization-Aware Fine-Tuning (QAFT), we must maintain high-precision continuous latent weights $W$ while simulating the 2-bit quaternary forward pass. This creates a unique memory profile that requires careful planning to optimize iteration speed.

Below is the detailed technical resource blueprint for running your TetraFT research track.

1. The Core Equation: QAFT VRAM Breakdown

To estimate how much Video RAM (VRAM) you need on your GPU, we must map out where every byte goes. During training, memory is split into four distinct pools:

$$M_{\text{total}} = M_{\text{weights}} + M_{\text{gradients}} + M_{\text{optimizer}} + M_{\text{activation}} + M_{\text{overhead}}$$

Let $P$ be the number of active parameters in your model (in billions).

A. Model Weights ($M_{\text{weights}}$)

Even though the forward pass uses 2-bit simulated weights, the continuous latent weights $W$ must be updated in high-precision to capture tiny gradient steps.

Standard FP32 Master Weights: $4$ bytes per parameter.

Alternative BF16 Master Weights: $2$ bytes per parameter (requires highly stable gradients).

Formula: $M_{\text{weights}} = P \times 4\text{ GB}$ (assuming FP32 master weights for stability).

B. Gradients ($M_{\text{gradients}}$)

Gradients are calculated in floating-point precision to prevent underflow.

FP32 Gradients: $4$ bytes per parameter.

Formula: $M_{\text{gradients}} = P \times 4\text{ GB}$

C. Optimizer States ($M_{\text{optimizer}}$)

The choice of optimizer is the biggest lever for controlling VRAM. For LLMs, AdamW is standard. It tracks two states: momentum (mean) and variance (uncentered variance).

Standard FP32 AdamW: $8$ bytes per parameter ($4\text{ bytes} \times 2\text{ states}$).

8-bit Quantized AdamW (e.g., bitsandbytes): $2$ bytes per parameter ($1\text{ byte} \times 2\text{ states}$).

Formula (Standard): $M_{\text{optimizer}} = P \times 8\text{ GB}$

Formula (8-bit): $M_{\text{optimizer}} = P \times 2\text{ GB}$

D. Activation Memory ($M_{\text{activation}}$)

During the forward pass, PyTorch must cache the intermediate activations of every single layer so it can calculate gradients during the backward pass. Activation memory scales linearly with sequence length ($L$), batch size ($B$), and architectural width.

To drastically save activation memory without sacrificing performance, you should use Gradient Checkpointing (activation checkpointing). This trades compute for memory by re-evaluating layers during the backward pass instead of caching them.

2. Concrete VRAM Scenarios

Let's calculate the exact VRAM required for our two main candidate models using Standard FP32 AdamW and Gradient Checkpointing enabled.

Scenario A: Qwen2.5-0.5B (The Sandbox)

Parameter Count ($P$): $\approx 0.49\text{ Billion}$

Weights (FP32): $0.49 \times 4 = 1.96\text{ GB}$

Gradients (FP32): $0.49 \times 4 = 1.96\text{ GB}$

Optimizer States (FP32 AdamW): $0.49 \times 8 = 3.92\text{ GB}$

Activation Memory (with gradient checkpointing, batch size 4, 1024 seq length): $\approx 1.5\text{ GB}$

CUDA/PyTorch Overhead: $\approx 1.0\text{ GB}$

Estimated VRAM Required: $\approx 10.34\text{ GB}$

Optimization Note: Using 8-bit Adam reduces this to $\approx 7.4\text{ GB}$, easily fitting inside a budget 8GB or 12GB GPU.

Scenario B: Llama-3.2-1B (The Benchmark)

Parameter Count ($P$): $\approx 1.23\text{ Billion}$

Weights (FP32): $1.23 \times 4 = 4.92\text{ GB}$

Gradients (FP32): $1.23 \times 4 = 4.92\text{ GB}$

Optimizer States (FP32 AdamW): $1.23 \times 8 = 9.84\text{ GB}$

Activation Memory (with gradient checkpointing, batch size 4, 1024 seq length): $\approx 2.5\text{ GB}$

CUDA/PyTorch Overhead: $\approx 1.5\text{ GB}$

Estimated VRAM Required: $\approx 23.68\text{ GB}$

Optimization Note: Using 8-bit Adam reduces this to $\approx 16.3\text{ GB}$, fitting safely within a standard 24GB consumer GPU (like an RTX 3090 or 4090).

3. Hardware Recommendation Guide

Based on the calculations above, here is how you should allocate your physical or cloud budget:

Tier

Hardware Setup

Target Model

Running Cost

Strategy

Local (Low Cost)

1x RTX 3060 / 4060 (12GB VRAM)

Qwen2.5-0.5B

Free (Upfront cost)

Perfect for initial implementation of the custom quantization layers, verifying the custom autograd engine, and fast feedback loops.

Local (Standard)

1x RTX 3090 / 4090 / 5090 (24GB - 32GB VRAM)

Llama-3.2-1B

Free (Upfront cost)

The gold standard for independent researchers. Allows you to run full QAFT runs overnight without cloud bill anxiety.

Cloud (Pay-as-you-go)

Google Colab Pro or RunPod (1x A100 40GB/80GB)

Llama-3.2-1B

$\$0.50$ - $\$1.50$ / hour

Best for high-throughput scaling experiments, running evaluations (MMLU, GSM8k), and training on larger token batches.

4. Software Stack Requirements

You will not need heavy enterprise-level framework engineering to start. Stick to a clean, highly debuggable stack:

PyTorch (latest stable): Essential for writing your custom torch.autograd.Function to handle the Straight-Through Estimator (STE) backward pass.

Hugging Face transformers & accelerate: For handling model loading, pipeline execution, and model weight manipulation.

bitsandbytes: Crucial if you want to swap to 8-bit optimizers to save VRAM on tighter hardware setups.

deepspeed or FSDP (Optional for Phase 2): Only needed if you decide to scale to multi-GPU training. For single-GPU runs on 0.5B or 1B models, vanilla PyTorch is cleaner and much easier to debug.

5. Dataset & Pre-training Knowledge Baseline

Because TetraFT is a Fine-Tuning track, you do not need to pre-train a model from scratch. You are doing Quantization-Aware Continual Pre-training or Instruction Tuning to help the model heal the quantization noise.

Recommended Datasets:

For General Knowledge Retention (Continual Pre-training):

SlimPajama-6B (or a subset of C4): Use a clean, randomized subset (e.g., 50M to 100M tokens). This is enough to evaluate how well the model recovers its language modeling ability (perplexity) post-quantization.

For Instruction Fine-Tuning (SFT):

Alpaca-GPT4 or UltraFeedback (clean): This allows you to evaluate if the model can retain reasoning, instruction-following, and formatting abilities under 2-bit quantization constraint.

Token Budget:

For a 0.5B or 1B model, you will generally observe convergence of the quantization recovery within 50 million to 200 million tokens (which takes roughly 1 to 4 hours on a single modern GPU).
