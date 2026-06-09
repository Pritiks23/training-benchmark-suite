# FSDP vs PyTorch Baseline Training Benchmark

## Overview

This project benchmarks and compares two training paradigms for a causal language model (GPT-style architecture):

- **Standard PyTorch training loop (baseline)**
- **Fully Sharded Data Parallel (FSDP) training**

The goal is to empirically evaluate differences in:

- Step latency (ms/iteration)
- Training stability (loss dynamics)
- System overhead vs optimization efficiency tradeoffs

All experiments were run in a controlled single-GPU environment to isolate framework overhead rather than distributed scaling benefits.

---

## Experimental Setup

### Model
- Architecture: GPT-2–style causal language model
- Task: Autoregressive next-token prediction
- Loss: Cross-entropy loss over token sequences

### Hardware
- Single NVIDIA GPU (Vast.ai instance)
- CUDA-enabled runtime environment

### Frameworks

#### PyTorch Baseline
- Standard single-process training loop
- Direct forward and backward pass execution
- No distributed training abstraction overhead

#### FSDP (Fully Sharded Data Parallel)
- Parameters, gradients, and optimizer states are sharded
- Uses `torch.distributed.fsdp`
- Designed primarily for multi-GPU and memory scaling

---

## Metrics Collected

Each training step logs:

- `step_time_ms` → iteration latency
- `loss` → training objective value
- `gpu` → device identifier
- `mem` → GPU memory usage (MB)

---

## Results

### Average Step Time

| Method | Average Step Time (ms) |
|--------|------------------------|
| PyTorch Baseline | **20.39 ms** |
| FSDP | **26.92 ms** |

### Key Observations from Plots

- PyTorch baseline consistently achieves lower per-step latency
- FSDP shows higher overhead and slightly higher variance
- Both methods exhibit similar loss convergence behavior

---

## Key Findings

### 1. PyTorch Baseline is Faster on Single GPU

The baseline avoids distributed system overhead:

- No parameter sharding/unsharding
- No distributed autograd hooks
- No synchronization barriers

This results in lower execution latency per step.

---

### 2. FSDP Introduces Overhead Even Without Multi-GPU Scaling

Although FSDP is designed for large-scale distributed training, it still incurs overhead in single-GPU settings:

- Forward pass materialization of sharded parameters
- Backward hooks for gradient gathering
- Internal distributed context initialization

This overhead increases step latency without providing scaling benefits.

---

### 3. Why FSDP Still Exists

FSDP is not optimized for single-device performance. Its purpose is:

- Enable training of models that exceed single-GPU memory limits
- Reduce memory footprint via parameter sharding
- Scale efficiently across multiple GPUs/nodes

| Property | PyTorch Baseline | FSDP |
|----------|------------------|------|
| Single-GPU speed | Faster | Slower |
| Multi-GPU scalability | Limited | Strong |
| Memory efficiency | Moderate | High |
| Large model support | Limited | Excellent |

---

### 4. Loss Behavior

Both systems produce nearly identical loss curves:

- Rapid initial loss decrease
- Stable convergence trend
- Minor stochastic variation across steps

This indicates:

> FSDP does not change optimization dynamics, only system execution behavior.

---

## Interpretation

The tradeoff demonstrated is fundamental in ML systems design:

> **FSDP optimizes for scalability, not single-device speed.**

Thus:

- PyTorch baseline is optimal for small-scale / single-GPU training
- FSDP becomes essential when model or batch size exceeds GPU memory limits
- Performance overhead is the cost of enabling large-scale training

---

## Conclusion

This benchmark highlights a key principle in deep learning systems:

> System design choices depend on scaling constraints, not isolated performance.

FSDP introduces measurable overhead in small-scale settings, but enables training regimes that are otherwise infeasible due to memory constraints in large-scale models.

---

## Future Work

Potential extensions:

- Multi-GPU scaling efficiency (1 → 8 → 32 GPUs)
- Memory profiling (activation + optimizer state breakdown)
- Throughput analysis (tokens/sec)
- Comparison with DeepSpeed ZeRO-3
- Mixed precision (FP16/BF16) impact analysis
- Communication overhead breakdown per layer

---
