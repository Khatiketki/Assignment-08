# Week 9 Assignment: DeepSeekMoE with ThunderKittens

## Assignment
> Reimplement your DeepseekMOE with ThunderKittens to use WMMA tensor cores
> and tensor memory acceleration on a B200.
> Measure the performance improvement compared to your current implementation.

---

## What's in this directory

| File | Purpose |
|------|---------|
| `moe_tk.cu`       | **Main deliverable** — TK WMMA kernel (tensor cores) |
| `moe_tk_tma.cu`   | Extended deliverable — TK WMMA + TMA kernel |
| `moe_naive.cu`    | Baseline — scalar CUDA (same logic as Week 7 C impl) |
| `benchmark.py`    | Python harness: times all three + plots results |
| `Makefile`        | Build all targets |

---

## Architecture: DeepSeekMoE Expert MLP

Each routed expert computes:
```
gate_out = silu(X @ gate_proj.T)   [T, I]
up_out   = X @ up_proj.T           [T, I]
hidden   = gate_out * up_out       [T, I]
output   = hidden @ down_proj.T    [T, H]
```

This is a **SiLU-gated MLP** (same structure as LLaMA FFN). The course's
**blockwise parallel** insight applies here: we fuse the gate+up GEMMs, the
SiLU activation, and the down GEMM into a single kernel pass, so `hidden`
is never fully materialized in global memory.

---

## ThunderKittens implementation

### WMMA kernel (`moe_tk.cu`)

Uses TK's typed tile system:
- `st_bf<32,32>` — shared memory tiles in bf16
- `rt_fl<32,32>` — register accumulator tiles in float32
- `warp::mma_AB` — warp-level tensor core MMA (WMMA)

Key steps per CTA tile `(t_tile, h_tile)`:
1. Loop over `i_tile` (intermediate dimension)
2. For each `i_tile`: loop over `k` tiles (hidden dim) → accumulate gate and up
3. Apply SiLU element-wise in registers
4. Accumulate `hidden @ down_proj` into output

**No global memory reads of intermediate activations** — everything stays in
registers/shared until the final store.

### TMA kernel (`moe_tk_tma.cu`)

Adds `tma::load_async` for global→shared transfers. This uses the B200's
**Tensor Memory Accelerator** hardware unit:
- Warp 0 issues async TMA load descriptors
- Warps 1–3 compute MMA while loads are in flight
- Communication and computation are overlapped

This corresponds to **Level 05** in the TK `educational_b200` sequence.

---

## Expected performance (B200)

| Config       | Naive CUDA | TK WMMA | TK TMA | TK Speedup |
|--------------|:----------:|:-------:|:------:|:----------:|
| tiny (T=8)   | ~0.1 ms    | ~0.05ms | ~0.04ms| ~2-3x      |
| small (512×1024×2048) | ~5ms | ~1ms | ~0.7ms | ~6-8x |
| medium (2048×2048×4096) | ~40ms | ~5ms | ~3ms | ~10-15x |
| large (4096×4096×8192) | ~300ms | ~30ms | ~18ms | ~15-20x |

The TK levels from the notes give us a rough ceiling:
- Naive (level 01/02): ~6 TFLOPS
- + Shared memory:      ~11 TFLOPS
- + WMMA (level 04):    ~26 TFLOPS
- + TMA  (level 05):    ~55 TFLOPS
- + tcgen05 (level 06): ~293 TFLOPS

Our fused kernel adds activation fusion on top, which reduces global
memory traffic further compared to running three separate GEMMs.

---

## Build & run

```bash
# Clone ThunderKittens
git clone https://github.com/HazyResearch/ThunderKittens
export TK_DIR=$PWD/ThunderKittens

# Build
make TK_DIR=$TK_DIR CUDA_ARCH=sm_90a

# Benchmark
make bench

# Verify against Week 7 test cases
make verify
```

---

## Connection to Week 7

The Week 7 implementation passed all tests:
```
[router]  max_abs_err = 0.000e+00  PASS
[mlp]     max_abs_err = 1.788e-07  PASS
[moe]     max_abs_err = 1.788e-07  PASS
```

The Week 9 TK kernel reimplements the same `mlp` block (per-expert FFN) using
tensor cores. The router logic (top-k selection, score normalization) remains
unchanged since it is not GEMM-bound. The full MoE forward pass dispatches each
token's top-k experts to the TK MLP kernel.
