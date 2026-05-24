# SGLang Development Environment

## Python Environment

Always use the conda `sglang` environment. Activate it before running any Python command:

```bash
source ~/conda/bin/activate sglang
```

Do NOT install packages into the system Python (`/usr/bin/python`). All dependencies are already set up in the conda env.

## GPU Setup

- 8x NVIDIA H100 80GB HBM3
- CUDA driver 535.129.03 (native CUDA 12.2, forward-compatible with CUDA 12.x toolkit)

## Models

Models are stored at `/mnt/dolphinfs/ssd_pool/docker/user/hadoop-scale-llm/luxuhao/models/`:
- `qwen3_30B` — Qwen3-30B-A3B (MoE, 128 experts)
- `DeepSeek-V3-0324-AWQ` — DeepSeek V3 AWQ 4-bit (~187GB on disk)

## Running Benchmarks

```bash
# Offline throughput benchmark
source ~/conda/bin/activate sglang
python -m sglang.bench_offline_throughput --model-path <path> --dataset-name random --random-input 1024 --random-output 512 --num-prompts 100 --tp <N> [--ep <N>] [--trust-remote-code]

# Simple inference example
python examples/runtime/engine/offline_batch_inference.py --model-path <path> --tp <N> --trust-remote-code
```

## Known Issues

- `std::bit_cast` is not supported by nvcc 12.9 — already replaced with `reinterpret_cast` in JIT kernel sources under `python/sglang/jit_kernel/csrc/`
- `sgl-deep-gemm` requires libcudart.so.13 (cu13 libs) — needed for FP8/FP4 inference (e.g., DeepSeek V3 FP8), not needed for AWQ/BF16
