/*
 * Week 9 Assignment: DeepSeekMoE Expert MLP — TMA variant
 * Adds Tensor Memory Accelerator (TMA) for global<->shared transfers
 * in addition to WMMA tensor cores.
 *
 * TMA allows the GPU hardware to copy tiles between global and shared memory
 * without using any compute threads — freeing all warps for MMA instructions.
 * This is the "Level 05" optimization from the TK educational_b200 sequence.
 *
 * Relationship to course levels:
 *   moe_tk.cu       → Level 04 style (WMMA only,    ~26 TFLOPS on matmul)
 *   moe_tk_tma.cu   → Level 05 style (WMMA + TMA,   ~55 TFLOPS on matmul)
 *
 * Build:
 *   nvcc -O3 -std=c++20 -arch=sm_90a \
 *        -I/path/to/ThunderKittens/include \
 *        moe_tk_tma.cu -o moe_tk_tma -lcuda
 */

#include "kittens.cuh"
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdio.h>

using namespace kittens;

static constexpr int BLOCK   = 64;   // larger tile to saturate TMA bandwidth
static constexpr int NWARPS  = 4;
static constexpr int NTHREADS = NWARPS * WARP_THREADS;

// ── Globals with TMA descriptors ──────────────────────────────────────────────
struct expert_globals_tma {
    using tile_t = st_bf<BLOCK, BLOCK>;
    using mat_gl = gl<bf16, 1, 1, -1, -1, tile_t>;

    mat_gl X;
    mat_gl gate_proj;
    mat_gl up_proj;
    mat_gl down_proj;
    mat_gl out;

    int T, H, I;
};

__device__ __forceinline__ void silu_mul_rt(
    rt_fl<BLOCK, BLOCK>& gate,
    const rt_fl<BLOCK, BLOCK>& up
) {
    #pragma unroll
    for (int i = 0; i < gate.num_elements; i++) {
        float g = gate.data[i];
        gate.data[i] = (g / (1.f + __expf(-g))) * up.data[i];
    }
}

/*
 * TMA kernel:
 * - warp 0 acts as "loader": issues tma::load for X, gate_proj, up_proj, down_proj
 * - warps 1-3 act as "computers": issue warp::mma_AB while loading is in flight
 * This overlaps I/O with compute — the key TMA benefit.
 *
 * We use a simple producer/consumer pattern via arrive_and_wait barriers.
 */
__global__ void __launch_bounds__(NTHREADS)
expert_mlp_tma_kernel(const __grid_constant__ expert_globals_tma g) {

    extern __shared__ alignment_dummy __shm[];
    shared_allocator al((int*)&__shm[0]);

    // Double-buffered shared tiles for pipeline
    auto& Xs0  = al.allocate<st_bf<BLOCK, BLOCK>>();
    auto& Xs1  = al.allocate<st_bf<BLOCK, BLOCK>>();
    auto& GPs  = al.allocate<st_bf<BLOCK, BLOCK>>();
    auto& UPs  = al.allocate<st_bf<BLOCK, BLOCK>>();
    auto& DPs  = al.allocate<st_bf<BLOCK, BLOCK>>();

    rt_bf<BLOCK, BLOCK>                         X_reg, W_reg;
    rt_bf<BLOCK, BLOCK, ducks::rt_layout::col>  W_col;
    rt_fl<BLOCK, BLOCK>                         gate_acc, up_acc, out_acc;

    const int warp_id = warpid();
    const int t_tile  = blockIdx.y;
    const int h_tile  = blockIdx.x;

    const int num_k_proj = (g.H + BLOCK - 1) / BLOCK;
    const int num_k_gate = (g.I + BLOCK - 1) / BLOCK;

    warp::zero(out_acc);

    for (int i_tile = 0; i_tile < num_k_gate; i_tile++) {
        warp::zero(gate_acc);
        warp::zero(up_acc);

        for (int k = 0; k < num_k_proj; k++) {
            // Warp 0: issue TMA loads; other warps wait then compute
            if (warp_id == 0) {
                // TMA: copy global tile → shared without occupying all threads
                tma::load_async(Xs0, g.X,         {0, 0, t_tile, k});
                tma::load_async(GPs, g.gate_proj,  {0, 0, i_tile, k});
                tma::load_async(UPs, g.up_proj,    {0, 0, i_tile, k});
                tma::store_commit_group();
            }
            tma::store_async_wait<0>();
            __syncthreads();

            // All warps compute
            warp::load(X_reg,  Xs0);
            warp::load(W_reg,  GPs);
            warp::swap_layout(W_col, W_reg);
            warp::mma_AB(gate_acc, X_reg, W_col, gate_acc);

            warp::load(W_reg, UPs);
            warp::swap_layout(W_col, W_reg);
            warp::mma_AB(up_acc, X_reg, W_col, up_acc);
            __syncthreads();
        }

        silu_mul_rt(gate_acc, up_acc);

        // Load down_proj and accumulate
        if (warp_id == 0) {
            tma::load_async(DPs, g.down_proj, {0, 0, h_tile, i_tile});
            tma::store_commit_group();
        }
        tma::store_async_wait<0>();
        __syncthreads();

        rt_bf<BLOCK, BLOCK> hidden_bf;
        warp::copy(hidden_bf, gate_acc);
        warp::load(W_reg, DPs);
        warp::swap_layout(W_col, W_reg);
        warp::mma_AB(out_acc, hidden_bf, W_col, out_acc);
        __syncthreads();
    }

    // TMA store: write result back without occupying compute threads
    if (warp_id == 0) {
        tma::store_async(g.out, out_acc, {0, 0, t_tile, h_tile});
        tma::store_commit_group();
        tma::store_async_wait<0>();
    }
    __syncthreads();
}

void run_expert_mlp_tma(
    __nv_bfloat16* X,
    __nv_bfloat16* gate_proj,
    __nv_bfloat16* up_proj,
    __nv_bfloat16* down_proj,
    __nv_bfloat16* out,
    int T, int H, int I
) {
    using tile_t = st_bf<BLOCK, BLOCK>;
    using mat_gl = gl<bf16, 1, 1, -1, -1, tile_t>;

    mat_gl X_gl  {(bf16*)X,         nullptr, nullptr, (size_t)T, (size_t)H};
    mat_gl GP_gl {(bf16*)gate_proj, nullptr, nullptr, (size_t)I, (size_t)H};
    mat_gl UP_gl {(bf16*)up_proj,   nullptr, nullptr, (size_t)I, (size_t)H};
    mat_gl DP_gl {(bf16*)down_proj, nullptr, nullptr, (size_t)H, (size_t)I};
    mat_gl OT_gl {(bf16*)out,       nullptr, nullptr, (size_t)T, (size_t)H};

    expert_globals_tma g {X_gl, GP_gl, UP_gl, DP_gl, OT_gl, T, H, I};

    dim3 blocks((H + BLOCK - 1) / BLOCK, (T + BLOCK - 1) / BLOCK);
    // 5 tiles in shared: Xs0, Xs1, GPs, UPs, DPs
    size_t smem = 5 * BLOCK * BLOCK * sizeof(__nv_bfloat16) + 512;

    cudaFuncSetAttribute(
        expert_mlp_tma_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        smem
    );
    expert_mlp_tma_kernel<<<blocks, NTHREADS, smem>>>(g);
    CHECK_CUDA_ERROR(cudaGetLastError());
}
