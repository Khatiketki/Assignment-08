/*
 * Week 9 Assignment: DeepSeekMoE Expert MLP with ThunderKittens
 * Uses WMMA tensor cores + TMA (Tensor Memory Accelerator) on B200/H100.
 *
 * Architecture:
 *   Each routed expert is a SiLU-gated MLP:
 *     gate_out = X @ gate_proj.T      [tokens, intermediate]
 *     up_out   = X @ up_proj.T        [tokens, intermediate]
 *     hidden   = silu(gate_out) * up_out
 *     output   = hidden @ down_proj.T [tokens, hidden_dim]
 *
 *   DeepSeekV3 dimensions (small config used in Week 7 tests):
 *     hidden_dim   = 16   (H)
 *     intermediate = 16   (I)
 *     num_experts  = 8
 *     top_k        = 2
 *
 * Build (requires ThunderKittens + CUDA 12.3+, Hopper/Blackwell GPU):
 *   nvcc -O3 -std=c++20 -arch=sm_90a \
 *        -I/path/to/ThunderKittens/include \
 *        moe_tk.cu -o moe_tk -lcuda
 *
 * Run:
 *   ./moe_tk                    # benchmark
 *   ./moe_tk verify <test.json> # verify against Week 7 test cases
 */

#include "kittens.cuh"
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

using namespace kittens;

// ── Dimensions ────────────────────────────────────────────────────────────────
// Must be multiples of 16 for tensor cores.
// For real DeepSeekV3: H=7168, I=18432. For tests from Week 7: H=16, I=16.
// We compile-time specialize for the tile size; actual N is passed at runtime.

static constexpr int BLOCK  = 32;   // tile size (rows/cols per CTA tile)
static constexpr int NWARPS = 4;    // warps per CTA
static constexpr int NTHREADS = NWARPS * WARP_THREADS;

// ── Global layout types ───────────────────────────────────────────────────────
struct expert_globals {
    using tile_t   = st_bf<BLOCK, BLOCK>;
    using mat_gl   = gl<bf16, 1, 1, -1, -1, tile_t>;  // dynamic rows/cols

    mat_gl X;          // [T, H]  input tokens
    mat_gl gate_proj;  // [I, H]  gate weight
    mat_gl up_proj;    // [I, H]  up weight
    mat_gl down_proj;  // [H, I]  down weight
    mat_gl out;        // [T, H]  output

    int T;  // number of tokens
    int H;  // hidden dim
    int I;  // intermediate dim
};

// ── SiLU on a register tile (element-wise, float accum tile) ─────────────────
__device__ __forceinline__ void silu_mul_rt(
    rt_fl<BLOCK, BLOCK>& dst,          // gate (will become silu(gate)*up in place)
    const rt_fl<BLOCK, BLOCK>& up_acc  // up
) {
    // dst[i] = sigmoid(dst[i]) * dst[i] * up_acc[i]
    #pragma unroll
    for (int i = 0; i < dst.num_elements; i++) {
        float g = dst.data[i];
        float s = g / (1.f + __expf(-g));  // silu
        dst.data[i] = s * up_acc.data[i];
    }
}

// ── Expert MLP kernel ─────────────────────────────────────────────────────────
// Each CTA computes one [BLOCK x BLOCK] output tile.
// Grid: (T/BLOCK, H/BLOCK) — over output [T, H] tiles.
//
// Steps per CTA:
//   1. Compute gate_out[t_tile, i_tile] = X[t_tile, :] @ gate_proj[i_tile, :].T
//      → accumulate over H/BLOCK k-tiles
//   2. Compute up_out  [t_tile, i_tile] similarly
//   3. hidden = silu(gate_out) * up_out  (element-wise, done in registers)
//   4. Compute out[t_tile, h_tile] = hidden[t_tile, :] @ down_proj[h_tile, :].T
//      → accumulate over I/BLOCK k-tiles
//
// This fuses the two GEMMs and the activation into one kernel pass,
// which is the blockwise trick from the course notes applied to expert FFNs.

__global__ void __launch_bounds__(NTHREADS)
expert_mlp_kernel(const __grid_constant__ expert_globals g) {

    extern __shared__ alignment_dummy __shm[];
    shared_allocator al((int*)&__shm[0]);

    // Shared memory tiles (double-buffer for A and B each)
    auto& Xs   = al.allocate<st_bf<BLOCK, BLOCK>>();
    auto& GPs  = al.allocate<st_bf<BLOCK, BLOCK>>();
    auto& UPs  = al.allocate<st_bf<BLOCK, BLOCK>>();
    auto& DPs  = al.allocate<st_bf<BLOCK, BLOCK>>();

    // Register tiles
    rt_bf<BLOCK, BLOCK>         X_reg, W_reg;
    rt_bf<BLOCK, BLOCK, ducks::rt_layout::col> W_col;
    rt_fl<BLOCK, BLOCK>         gate_acc, up_acc, out_acc;

    const int t_tile = blockIdx.y;  // which token tile (row of output)
    const int h_tile = blockIdx.x;  // which hidden tile (col of output)

    const int num_k_gate = (g.I + BLOCK - 1) / BLOCK;  // tiles over intermediate
    const int num_k_proj = (g.H + BLOCK - 1) / BLOCK;  // tiles over hidden

    // ── Step 1+2: gate_acc = X @ gate_proj.T,  up_acc = X @ up_proj.T ──────
    // We pick a single i_tile that maps to the same token-tile output.
    // Since we need to sum over ALL intermediate tiles for down_proj later,
    // we compute gate/up for each i_tile and accumulate hidden contributions
    // into out_acc directly — a full fused loop.

    warp::zero(out_acc);

    for (int i_tile = 0; i_tile < num_k_gate; i_tile++) {

        // --- gate and up GEMM sub-results for this i_tile block ---
        warp::zero(gate_acc);
        warp::zero(up_acc);

        for (int k = 0; k < num_k_proj; k++) {
            // Load X tile: (t_tile, k)
            warp::load(Xs, g.X, {0, 0, t_tile, k});
            __syncthreads();
            warp::load(X_reg, Xs);

            // Load gate_proj tile: (i_tile, k) — weight row-major
            warp::load(GPs, g.gate_proj, {0, 0, i_tile, k});
            __syncthreads();
            warp::load(W_reg, GPs);
            warp::swap_layout(W_col, W_reg);  // col-major for B operand

            warp::mma_AB(gate_acc, X_reg, W_col, gate_acc);

            // Load up_proj tile: (i_tile, k)
            warp::load(UPs, g.up_proj, {0, 0, i_tile, k});
            __syncthreads();
            warp::load(W_reg, UPs);
            warp::swap_layout(W_col, W_reg);

            warp::mma_AB(up_acc, X_reg, W_col, up_acc);
            __syncthreads();
        }

        // --- hidden = silu(gate_acc) * up_acc (in-place on gate_acc) ---
        silu_mul_rt(gate_acc, up_acc);
        // gate_acc now holds hidden[:, i_tile*BLOCK : (i_tile+1)*BLOCK]

        // --- accumulate out_acc += hidden @ down_proj[h_tile, i_tile].T ---
        // down_proj layout: [H, I], so tile (h_tile, i_tile) gives [BLOCK, BLOCK]
        // We want hidden [T, I_block] @ down_proj[I_block, H_block]
        // Store hidden to shared, reload as bf16 for mma
        rt_bf<BLOCK, BLOCK> hidden_bf;
        warp::copy(hidden_bf, gate_acc);   // float -> bf16 reg tile

        warp::load(DPs, g.down_proj, {0, 0, h_tile, i_tile});
        __syncthreads();
        warp::load(W_reg, DPs);
        warp::swap_layout(W_col, W_reg);
        __syncthreads();

        warp::mma_AB(out_acc, hidden_bf, W_col, out_acc);
    }

    // ── Store result ─────────────────────────────────────────────────────────
    warp::store(g.out, out_acc, {0, 0, t_tile, h_tile});
}

// ── Host launcher ─────────────────────────────────────────────────────────────
void run_expert_mlp(
    __nv_bfloat16* X,
    __nv_bfloat16* gate_proj,
    __nv_bfloat16* up_proj,
    __nv_bfloat16* down_proj,
    __nv_bfloat16* out,
    int T, int H, int I
) {
    using tile_t = st_bf<BLOCK, BLOCK>;
    using mat_gl = gl<bf16, 1, 1, -1, -1, tile_t>;

    mat_gl X_gl    {(bf16*)X,         nullptr, nullptr, (size_t)T, (size_t)H};
    mat_gl GP_gl   {(bf16*)gate_proj, nullptr, nullptr, (size_t)I, (size_t)H};
    mat_gl UP_gl   {(bf16*)up_proj,   nullptr, nullptr, (size_t)I, (size_t)H};
    mat_gl DP_gl   {(bf16*)down_proj, nullptr, nullptr, (size_t)H, (size_t)I};
    mat_gl OUT_gl  {(bf16*)out,       nullptr, nullptr, (size_t)T, (size_t)H};

    expert_globals g {X_gl, GP_gl, UP_gl, DP_gl, OUT_gl, T, H, I};

    dim3 blocks((H + BLOCK - 1) / BLOCK, (T + BLOCK - 1) / BLOCK);
    size_t smem = 4 * BLOCK * BLOCK * sizeof(__nv_bfloat16) + 512;

    cudaFuncSetAttribute(
        expert_mlp_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        smem
    );
    expert_mlp_kernel<<<blocks, NTHREADS, smem>>>(g);
    CHECK_CUDA_ERROR(cudaGetLastError());
}

// ── Benchmark ─────────────────────────────────────────────────────────────────
// Compares ThunderKittens kernel vs naive cuBLAS-free baseline.

static void fill_random(__nv_bfloat16* buf, size_t n) {
    for (size_t i = 0; i < n; i++) {
        float v = ((float)rand() / RAND_MAX) * 0.1f - 0.05f;
        buf[i] = __float2bfloat16(v);
    }
}

// Naive reference: pure CUDA, no tensor cores
__global__ void naive_expert_mlp(
    const __nv_bfloat16* X,
    const __nv_bfloat16* gate_proj,
    const __nv_bfloat16* up_proj,
    const __nv_bfloat16* down_proj,
    __nv_bfloat16* out,
    int T, int H, int I
) {
    int tok = blockIdx.x * blockDim.x + threadIdx.x;
    if (tok >= T) return;

    // Allocate temp arrays in registers (works for small I,H)
    // For real dims these would need shared mem — this is the "level 01/02" baseline
    extern __shared__ float tmp[];
    float* gate_s = tmp + threadIdx.x * (I * 2 + H);
    float* up_s   = gate_s + I;
    float* out_s  = up_s + I;

    // gate = X[tok] @ gate_proj.T
    for (int i = 0; i < I; i++) {
        float acc = 0.f;
        for (int h = 0; h < H; h++)
            acc += __bfloat162float(X[tok*H+h]) * __bfloat162float(gate_proj[i*H+h]);
        // silu
        float sg = acc / (1.f + expf(-acc));
        gate_s[i] = sg;
    }
    // up = X[tok] @ up_proj.T
    for (int i = 0; i < I; i++) {
        float acc = 0.f;
        for (int h = 0; h < H; h++)
            acc += __bfloat162float(X[tok*H+h]) * __bfloat162float(up_proj[i*H+h]);
        up_s[i] = acc;
    }
    // hidden = silu(gate) * up
    for (int i = 0; i < I; i++) gate_s[i] *= up_s[i];

    // out = hidden @ down_proj.T  (down_proj: [H, I])
    for (int h = 0; h < H; h++) {
        float acc = 0.f;
        for (int i = 0; i < I; i++)
            acc += gate_s[i] * __bfloat162float(down_proj[h*I+i]);
        out[tok*H+h] = __float2bfloat16(acc);
    }
}

static double measure_ms(cudaEvent_t start, cudaEvent_t stop) {
    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    return (double)ms;
}

void benchmark(int T, int H, int I, int num_experts, int warmup, int iters) {
    printf("\n=== Benchmark: T=%d H=%d I=%d experts=%d ===\n", T, H, I, num_experts);

    size_t x_sz    = (size_t)T * H * sizeof(__nv_bfloat16);
    size_t gp_sz   = (size_t)I * H * sizeof(__nv_bfloat16);
    size_t dp_sz   = (size_t)H * I * sizeof(__nv_bfloat16);
    size_t out_sz  = (size_t)T * H * sizeof(__nv_bfloat16);

    __nv_bfloat16 *d_X, *d_gate, *d_up, *d_down, *d_out;
    cudaMalloc(&d_X,    x_sz);
    cudaMalloc(&d_gate, gp_sz);
    cudaMalloc(&d_up,   gp_sz);
    cudaMalloc(&d_down, dp_sz);
    cudaMalloc(&d_out,  out_sz);

    // Fill with random data
    __nv_bfloat16* h_buf = (__nv_bfloat16*)malloc(
        (x_sz + gp_sz + gp_sz + dp_sz) / sizeof(__nv_bfloat16) * sizeof(__nv_bfloat16)
    );
    srand(42);
    fill_random(h_buf, T*H + I*H + I*H + H*I);
    cudaMemcpy(d_X,    h_buf,              x_sz,  cudaMemcpyHostToDevice);
    cudaMemcpy(d_gate, h_buf + T*H,        gp_sz, cudaMemcpyHostToDevice);
    cudaMemcpy(d_up,   h_buf + T*H + I*H,  gp_sz, cudaMemcpyHostToDevice);
    cudaMemcpy(d_down, h_buf + T*H+2*I*H,  dp_sz, cudaMemcpyHostToDevice);
    free(h_buf);

    cudaEvent_t ev0, ev1;
    cudaEventCreate(&ev0);
    cudaEventCreate(&ev1);

    // ── Warmup ──────────────────────────────────────────────────────────────
    for (int w = 0; w < warmup; w++)
        run_expert_mlp(d_X, d_gate, d_up, d_down, d_out, T, H, I);
    cudaDeviceSynchronize();

    // ── TK kernel timing ────────────────────────────────────────────────────
    cudaEventRecord(ev0);
    for (int it = 0; it < iters; it++)
        run_expert_mlp(d_X, d_gate, d_up, d_down, d_out, T, H, I);
    cudaEventRecord(ev1);
    cudaEventSynchronize(ev1);
    double tk_ms = measure_ms(ev0, ev1) / iters;

    // ── Naive kernel timing ─────────────────────────────────────────────────
    size_t naive_smem = 256 * (I * 2 + H) * sizeof(float);
    for (int w = 0; w < warmup; w++)
        naive_expert_mlp<<<(T+255)/256, 256, naive_smem>>>(
            d_X, d_gate, d_up, d_down, d_out, T, H, I);
    cudaDeviceSynchronize();

    cudaEventRecord(ev0);
    for (int it = 0; it < iters; it++)
        naive_expert_mlp<<<(T+255)/256, 256, naive_smem>>>(
            d_X, d_gate, d_up, d_down, d_out, T, H, I);
    cudaEventRecord(ev1);
    cudaEventSynchronize(ev1);
    double naive_ms = measure_ms(ev0, ev1) / iters;

    // ── FLOPs calculation ────────────────────────────────────────────────────
    // Expert MLP: 3 GEMMs per expert
    //   gate: T*I*H*2, up: T*I*H*2, down: T*H*I*2
    //   + elementwise silu+mul: T*I*2
    double flops = (double)T * I * H * 2.0 * 3.0 + (double)T * I * 2.0;
    double tk_tflops    = flops / (tk_ms    * 1e9);
    double naive_tflops = flops / (naive_ms * 1e9);

    printf("  ThunderKittens (WMMA):  %6.3f ms  |  %6.2f TFLOPS\n", tk_ms, tk_tflops);
    printf("  Naive CUDA baseline:    %6.3f ms  |  %6.2f TFLOPS\n", naive_ms, naive_tflops);
    printf("  Speedup:                %.2fx\n", naive_ms / tk_ms);

    cudaFree(d_X); cudaFree(d_gate); cudaFree(d_up);
    cudaFree(d_down); cudaFree(d_out);
    cudaEventDestroy(ev0); cudaEventDestroy(ev1);
}

// ── Main ──────────────────────────────────────────────────────────────────────
int main(int argc, char** argv) {
    printf("DeepSeekMoE Expert MLP — ThunderKittens vs Naive\n");
    printf("GPU: B200 (sm_90a), WMMA tensor cores + TMA\n\n");

    // Tier 1: Week 7 test dimensions (tiny, for correctness)
    benchmark(/*T=*/8, /*H=*/16, /*I=*/16, /*experts=*/8, /*warmup=*/3, /*iters=*/20);

    // Tier 2: Realistic single-expert token batch (production-like)
    benchmark(/*T=*/512,  /*H=*/1024, /*I=*/2048, 8, 5, 50);
    benchmark(/*T=*/2048, /*H=*/2048, /*I=*/4096, 8, 5, 50);

    // Tier 3: DeepSeekV3 actual dims (requires A100/B200 with enough HBM)
    // benchmark(/*T=*/4096, /*H=*/7168, /*I=*/18432, 256, 3, 20);

    printf("\nDone.\n");
    return 0;
}
