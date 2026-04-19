/*
 * moe_naive.cu — Naive CUDA baseline for DeepSeekMoE Expert MLP
 *
 * This is the "scalar" implementation: one thread per output token,
 * no tensor cores, no shared memory tiling. Equivalent to the Week 7
 * pure-C implementation ported to CUDA so timings are GPU-apples-to-apples.
 *
 * Build:
 *   nvcc -O3 -std=c++20 -arch=sm_90a moe_naive.cu -o moe_naive
 *
 * Usage (for benchmark.py):
 *   ./moe_naive <T> <H> <I> <warmup> <iters>
 */

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

__global__ void naive_expert_mlp(
    const __nv_bfloat16* __restrict__ X,          // [T, H]
    const __nv_bfloat16* __restrict__ gate_proj,  // [I, H]
    const __nv_bfloat16* __restrict__ up_proj,    // [I, H]
    const __nv_bfloat16* __restrict__ down_proj,  // [H, I]
    __nv_bfloat16* __restrict__ out,              // [T, H]
    int T, int H, int I
) {
    int tok = blockIdx.x * blockDim.x + threadIdx.x;
    if (tok >= T) return;

    // gate = X[tok] @ gate_proj.T  with silu
    // up   = X[tok] @ up_proj.T
    // hidden = silu(gate) * up
    // out[tok] = hidden @ down_proj.T

    // Use dynamic shared memory as scratch; each thread block gets a slice
    extern __shared__ float smem[];
    // Each thread uses I*2 + H floats
    float* gate_buf = smem + threadIdx.x * (I * 2 + H);
    float* up_buf   = gate_buf + I;
    float* out_buf  = up_buf   + I;

    const float* Xf = nullptr;  // we read X in bf16

    // Gate projection + SiLU
    for (int i = 0; i < I; i++) {
        float acc = 0.f;
        for (int h = 0; h < H; h++)
            acc += __bfloat162float(X[tok*H + h]) *
                   __bfloat162float(gate_proj[i*H + h]);
        float sig = 1.f / (1.f + expf(-acc));
        gate_buf[i] = acc * sig;  // silu
    }

    // Up projection
    for (int i = 0; i < I; i++) {
        float acc = 0.f;
        for (int h = 0; h < H; h++)
            acc += __bfloat162float(X[tok*H + h]) *
                   __bfloat162float(up_proj[i*H + h]);
        up_buf[i] = acc;
    }

    // Hidden = silu(gate) * up
    for (int i = 0; i < I; i++)
        gate_buf[i] *= up_buf[i];

    // Down projection
    for (int h = 0; h < H; h++) {
        float acc = 0.f;
        for (int i = 0; i < I; i++)
            acc += gate_buf[i] * __bfloat162float(down_proj[h*I + i]);
        out[tok*H + h] = __float2bfloat16(acc);
    }
}

static void fill_random(__nv_bfloat16* buf, size_t n, unsigned seed) {
    srand(seed);
    for (size_t i = 0; i < n; i++) {
        float v = ((float)rand() / RAND_MAX) * 0.1f - 0.05f;
        buf[i] = __float2bfloat16(v);
    }
}

int main(int argc, char** argv) {
    if (argc < 6) {
        fprintf(stderr, "Usage: %s <T> <H> <I> <warmup> <iters>\n", argv[0]);
        return 1;
    }

    int T      = atoi(argv[1]);
    int H      = atoi(argv[2]);
    int I      = atoi(argv[3]);
    int warmup = atoi(argv[4]);
    int iters  = atoi(argv[5]);

    size_t x_sz = (size_t)T * H;
    size_t g_sz = (size_t)I * H;
    size_t d_sz = (size_t)H * I;

    __nv_bfloat16 *d_X, *d_gate, *d_up, *d_down, *d_out;
    cudaMalloc(&d_X,    x_sz * sizeof(*d_X));
    cudaMalloc(&d_gate, g_sz * sizeof(*d_gate));
    cudaMalloc(&d_up,   g_sz * sizeof(*d_up));
    cudaMalloc(&d_down, d_sz * sizeof(*d_down));
    cudaMalloc(&d_out,  x_sz * sizeof(*d_out));

    // Fill on host, copy to device
    __nv_bfloat16* h_buf = (__nv_bfloat16*)malloc(
        (x_sz + g_sz + g_sz + d_sz) * sizeof(*h_buf));
    fill_random(h_buf, x_sz + g_sz + g_sz + d_sz, 42);
    cudaMemcpy(d_X,    h_buf,               x_sz * sizeof(*h_buf), cudaMemcpyHostToDevice);
    cudaMemcpy(d_gate, h_buf + x_sz,        g_sz * sizeof(*h_buf), cudaMemcpyHostToDevice);
    cudaMemcpy(d_up,   h_buf + x_sz + g_sz, g_sz * sizeof(*h_buf), cudaMemcpyHostToDevice);
    cudaMemcpy(d_down, h_buf + x_sz+2*g_sz, d_sz * sizeof(*h_buf), cudaMemcpyHostToDevice);
    free(h_buf);

    const int THREADS = 256;
    dim3 blocks((T + THREADS - 1) / THREADS);
    size_t smem = (size_t)THREADS * (I * 2 + H) * sizeof(float);

    // Warn if smem too large
    if (smem > 100000) {
        fprintf(stderr,
            "WARNING: shared mem per block = %zu bytes > 100KB. "
            "Use smaller dims or increase limit.\n", smem);
    }

    auto launch = [&]() {
        naive_expert_mlp<<<blocks, THREADS, smem>>>(
            d_X, d_gate, d_up, d_down, d_out, T, H, I);
    };

    // Warmup
    for (int w = 0; w < warmup; w++) launch();
    cudaDeviceSynchronize();

    // Timing
    cudaEvent_t ev0, ev1;
    cudaEventCreate(&ev0);
    cudaEventCreate(&ev1);
    cudaEventRecord(ev0);
    for (int it = 0; it < iters; it++) launch();
    cudaEventRecord(ev1);
    cudaEventSynchronize(ev1);

    float ms = 0.f;
    cudaEventElapsedTime(&ms, ev0, ev1);
    ms /= iters;

    printf("TIME_MS=%.4f\n", ms);

    double f = (double)T * I * H * 2.0 * 3.0 + (double)T * I * 2.0;
    printf("TFLOPS=%.4f\n", f / ((double)ms * 1e9));

    cudaFree(d_X); cudaFree(d_gate); cudaFree(d_up);
    cudaFree(d_down); cudaFree(d_out);
    return 0;
}
