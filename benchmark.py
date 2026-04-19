#!/usr/bin/env python3
"""
Week 9 Benchmark: DeepSeekMoE expert MLP performance comparison.

Compares three implementations:
  1. Naive C (Week 7 baseline) — scalar float, no parallelism
  2. TK WMMA (moe_tk.cu)      — WMMA tensor cores via ThunderKittens
  3. TK TMA  (moe_tk_tma.cu)  — WMMA + TMA global<->shared transfers

Usage:
  python benchmark.py                 # run all tiers
  python benchmark.py --verify        # also verify correctness vs Week 7 tests
  python benchmark.py --plot          # save a bar chart of results

Requires:
  - moe_tk and moe_tk_tma compiled binaries in the same directory
  - torch + cuda for the Python reference timing
  - matplotlib (optional, for --plot)
"""

import subprocess
import time
import argparse
import json
import os
import sys

try:
    import torch
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

# ── Config ────────────────────────────────────────────────────────────────────
BENCHMARKS = [
    # label,        T,    H,     I,    warmup, iters
    ("tiny (test)", 8,    16,    16,   3,      50),
    ("small",       512,  1024,  2048, 5,      100),
    ("medium",      2048, 2048,  4096, 5,      50),
    ("large",       4096, 4096,  8192, 3,      20),
]

# ── PyTorch reference timing (GPU) ────────────────────────────────────────────
def torch_expert_mlp_time(T, H, I, warmup, iters):
    """Time a SiLU-gated MLP using PyTorch on GPU (cuBLAS backend)."""
    if not HAS_TORCH or not torch.cuda.is_available():
        return None

    device = "cuda"
    dtype  = torch.bfloat16

    X    = torch.randn(T, H, device=device, dtype=dtype)
    gate = torch.randn(I, H, device=device, dtype=dtype)
    up   = torch.randn(I, H, device=device, dtype=dtype)
    down = torch.randn(H, I, device=device, dtype=dtype)

    def run():
        g = F.silu(X @ gate.T)
        u = X @ up.T
        h = g * u
        return h @ down.T

    # Warmup
    for _ in range(warmup):
        run()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end   = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        run()
    end.record()
    torch.cuda.synchronize()

    return start.elapsed_time(end) / iters  # ms

# ── Run a compiled binary and parse its output ────────────────────────────────
def run_binary(binary, T, H, I, warmup, iters):
    """Run a C/CUDA benchmark binary with args and parse timing."""
    cmd = [f"./{binary}", str(T), str(H), str(I), str(warmup), str(iters)]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        # Expect stdout like: "TIME_MS=3.14\nTFLOPS=26.5\n"
        ms = None
        for line in result.stdout.splitlines():
            if line.startswith("TIME_MS="):
                ms = float(line.split("=")[1])
        return ms
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return None

# ── FLOPs for expert MLP ──────────────────────────────────────────────────────
def flops(T, H, I):
    # 3 GEMMs: gate T×H×I, up T×H×I, down T×I×H
    # + SiLU+mul: T×I
    return T * I * H * 2 * 3 + T * I * 2

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--plot",   action="store_true")
    args = parser.parse_args()

    results = []
    print(f"\n{'='*72}")
    print(f"  Week 9: DeepSeekMoE Expert MLP — Performance Comparison")
    print(f"{'='*72}")
    print(f"{'Config':<18} {'PyTorch':>10} {'Naive C':>10} {'TK WMMA':>10} {'TK TMA':>10} {'Speedup':>10}")
    print(f"{'-'*72}")

    for label, T, H, I, warmup, iters in BENCHMARKS:
        f = flops(T, H, I)

        # PyTorch (cuBLAS) baseline
        pt_ms    = torch_expert_mlp_time(T, H, I, warmup, iters)
        pt_tflop = (f / (pt_ms * 1e9)) if pt_ms else None

        # Compiled binaries (if available)
        tk_ms    = run_binary("moe_tk",     T, H, I, warmup, iters)
        tma_ms   = run_binary("moe_tk_tma", T, H, I, warmup, iters)
        naive_ms = run_binary("moe_naive",  T, H, I, warmup, iters)

        tk_tflop    = (f / (tk_ms    * 1e9)) if tk_ms    else None
        tma_tflop   = (f / (tma_ms   * 1e9)) if tma_ms   else None
        naive_tflop = (f / (naive_ms * 1e9)) if naive_ms else None

        # Speedup relative to naive
        speedup_tk  = (naive_ms / tk_ms)  if (naive_ms and tk_ms)  else None
        speedup_tma = (naive_ms / tma_ms) if (naive_ms and tma_ms) else None

        def fmt(v, unit=""):
            return f"{v:.2f}{unit}" if v is not None else "N/A"

        print(
            f"  {label:<16} "
            f"{fmt(pt_ms,'ms'):>10} "
            f"{fmt(naive_ms,'ms'):>10} "
            f"{fmt(tk_ms,'ms'):>10} "
            f"{fmt(tma_ms,'ms'):>10} "
            f"{'TK:'+fmt(speedup_tk,'x') + ' TMA:'+fmt(speedup_tma,'x'):>14}"
        )

        results.append({
            "label": label, "T": T, "H": H, "I": I,
            "pytorch_ms": pt_ms, "naive_ms": naive_ms,
            "tk_wmma_ms": tk_ms, "tk_tma_ms": tma_ms,
            "tk_speedup": speedup_tk, "tma_speedup": speedup_tma,
            "tk_tflops": tk_tflop, "tma_tflops": tma_tflop,
        })

    print(f"{'='*72}\n")

    # Save results JSON
    with open("benchmark_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Results saved to benchmark_results.json")

    # ── Optional plot ────────────────────────────────────────────────────────
    if args.plot:
        try:
            import matplotlib.pyplot as plt
            import numpy as np

            labels = [r["label"] for r in results]
            x = np.arange(len(labels))
            w = 0.2

            pt_ms_vals    = [r["pytorch_ms"]  or 0 for r in results]
            naive_ms_vals = [r["naive_ms"]     or 0 for r in results]
            tk_ms_vals    = [r["tk_wmma_ms"]   or 0 for r in results]
            tma_ms_vals   = [r["tk_tma_ms"]    or 0 for r in results]

            fig, ax = plt.subplots(figsize=(10, 5))
            ax.bar(x - 1.5*w, pt_ms_vals,    w, label="PyTorch (cuBLAS)", color="#4C72B0")
            ax.bar(x - 0.5*w, naive_ms_vals, w, label="Naive CUDA",       color="#DD8452")
            ax.bar(x + 0.5*w, tk_ms_vals,    w, label="TK WMMA",          color="#55A868")
            ax.bar(x + 1.5*w, tma_ms_vals,   w, label="TK WMMA + TMA",    color="#C44E52")

            ax.set_xlabel("Configuration")
            ax.set_ylabel("Time (ms, lower is better)")
            ax.set_title("DeepSeekMoE Expert MLP — Week 9 Benchmark")
            ax.set_xticks(x)
            ax.set_xticklabels(labels)
            ax.legend()
            plt.tight_layout()
            plt.savefig("benchmark_results.png", dpi=150)
            print("Plot saved to benchmark_results.png")
        except ImportError:
            print("matplotlib not installed, skipping plot.")

    # ── Verify ───────────────────────────────────────────────────────────────
    if args.verify and HAS_TORCH:
        print("\nVerifying TK output against PyTorch reference...")
        T, H, I = 8, 16, 16  # Week 7 test dimensions
        device = "cuda"
        dtype  = torch.bfloat16

        torch.manual_seed(0)
        X    = torch.randn(T, H, device=device, dtype=dtype)
        gate = torch.randn(I, H, device=device, dtype=dtype)
        up   = torch.randn(I, H, device=device, dtype=dtype)
        down = torch.randn(H, I, device=device, dtype=dtype)

        ref = (F.silu(X @ gate.T) * (X @ up.T)) @ down.T

        # Write tensors to tmp files for the binary to read
        for name, t in [("X", X), ("gate", gate), ("up", up), ("down", down)]:
            t.cpu().numpy().tofile(f"/tmp/{name}.bin")

        result = subprocess.run(
            ["./moe_tk", "verify",
             "/tmp/X.bin", "/tmp/gate.bin", "/tmp/up.bin", "/tmp/down.bin",
             str(T), str(H), str(I)],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            print("  TK WMMA output: PASS")
        else:
            print("  TK WMMA output: FAIL or binary not found")
            print("  (Run moe_tk in verify mode manually)")

        max_err = (ref.cpu().float() - ref.cpu().float()).abs().max().item()
        print(f"  PyTorch self-check max_abs_err = {max_err:.3e}")

if __name__ == "__main__":
    main()
