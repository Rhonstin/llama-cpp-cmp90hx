# CMP 90HX Benchmark Results

**GPU**: NVIDIA CMP 90HX (GA102, sm_86, 9877 MiB)  
**Branch**: `cmp90hx-optimizations` — IMAD + HFMA2 patches on `atomic/feature/turboquant-kv-cache`  
**Date**: 2026-05-15

## Patches

| Commit | Change | File |
|---|---|---|
| `3d8c364bc` | DP4A → PTX IMAD (unthrottled on sm_86) | `ggml/src/ggml-cuda/common.cuh` |
| `a5df16302` | FP32 vmmq accumulation → HFMA2 (unthrottled) | `ggml/src/ggml-cuda/vecdotq.cuh` |
| `e7b5d6bed` | DKQ=512 → tile kernel fix (Gemma MTP assistant) | `ggml/src/ggml-cuda/fattn.cu` |

## Benchmark Files

| File | Model | Key finding |
|---|---|---|
| [GEMMA_E4B_BENCH.md](GEMMA_E4B_BENCH.md) | gemma4 E4B Q5_K (6.2 GiB, 100% GPU) | Patches: **+57.5%** (42→67 tok/s) |
| [TURBOQUANT_BENCH.md](TURBOQUANT_BENCH.md) | gemma4 E4B Q5_K | TurboQuant KV: **−6.8%** — avoid on CMP 90HX |
| [QWEN35_9B_BENCH.md](QWEN35_9B_BENCH.md) | Qwen3.5-9B Q4_K (5.8 GiB, 100% GPU) | Patches: **+87.2%** (30→57 tok/s) |
| [QWEN36_35B_BENCH.md](QWEN36_35B_BENCH.md) | Qwen3.6-35B-A3B Q4_K (ncmoe=26, ~44% GPU) | Patches: **+7.0%**; ncmoe scan |
| [NEXTN_BENCH.md](NEXTN_BENCH.md) | Qwen3.6-35B-A3B MTP (ncmoe=28) | NextN: **+7.6%**, 87.7% acceptance |

## Key Findings Summary

### 1. Patch speedup scales with GPU weight fraction

| Model | GPU% | IMAD+HFMA2 speedup |
|---|---:|---:|
| Qwen3.5-9B Q4_K (100% GPU) | 100% | **+87.2%** |
| gemma E4B Q5_K (100% GPU) | 100% | **+57.5%** |
| Qwen3.6-35B MoE (ncmoe=26, ~44% GPU) | ~44% | **+7.0%** |

### 2. TurboQuant KV is net-negative on CMP 90HX

KV dequant in Flash Attention uses FP32 FFMA (throttled 14× on CMP 90HX).
Bandwidth savings do not cover the compute overhead. **Always use `-ctk f16 -ctv f16`.**

| Model | turbo3 vs f16 |
|---|---:|
| gemma E4B | −6.8% |
| Qwen3.5-9B | −3.2% |
| Qwen3.6-35B (ncmoe=26) | −1.5% |

### 3. NextN speculative decoding

| Model | NextN gain | Acceptance |
|---|---:|---:|
| Qwen3.5-9B dense | ≈ 0% | 86% |
| Qwen3.6-35B MoE (ncmoe=28) | +7.6% | 87.7% |

Dense models: draft-compute-bound, no gain.  
MoE targets with heavy verify step: modest gain from async overlap.

### 4. Optimal ncmoe for Qwen3.6-35B-A3B on CMP 90HX

- `llama-bench`: ncmoe=24 (30.81 tok/s, MTP GGUF)
- `llama-server` no NextN: ncmoe=26 (29.5 tok/s, non-MTP) or ncmoe=28
- `llama-server` + NextN: ncmoe=28 required (compute buffers need ~1 GiB VRAM headroom)

## CMP 90HX throttle reference

| Instruction | Latency | Status |
|---|---|---|
| FFMA / FADD | 17.8 ns | throttled 14× |
| DP4A | 35.6 ns | throttled 29× |
| **IMAD** | **1.4 ns** | **unthrottled** |
| **HFMA2** | **1.4 ns** | **unthrottled** |
| Tensor cores | 284.5 ns | severely throttled — do not use |
