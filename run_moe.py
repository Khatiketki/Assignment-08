"""
Week 9: DeepSeekMoE — ThunderKittens WMMA vs Naive CUDA
Run: python -m modal run run_moe.py
"""
import modal

image = (
    modal.Image.from_registry("nvidia/cuda:12.6.0-devel-ubuntu22.04", add_python="3.11")
    .apt_install("git", "build-essential")
    .run_commands(
        "git clone https://github.com/HazyResearch/ThunderKittens /tk && echo v18",
        "python3 -c \""
        "txt=open('/tk/include/common/util.cuh').read();"
        "p='#ifndef cudaLaunchAttributePreferredClusterDimension\\n"
        "#define cudaLaunchAttributePreferredClusterDimension ((cudaLaunchAttributeID)8)\\n"
        "#endif\\n';"
        "open('/tk/include/common/util.cuh','w').write(p+txt)"
        "\"",
    )
    .pip_install("numpy")
    .pip_install("torch", index_url="https://download.pytorch.org/whl/cu124")
)

app = modal.App("week9-moe-v18", image=image)

ARCH  = "sm_90a"
FLAGS = "-I/tk/include --extended-lambda --expt-relaxed-constexpr -DKITTENS_HOPPER -diag-suppress 1650"

# Naive CUDA kernel — scalar, one thread per token
NAIVE_CU = r"""
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

__global__ void naive_mlp(
    const __nv_bfloat16* X, const __nv_bfloat16* gate_w,
    const __nv_bfloat16* up_w, const __nv_bfloat16* down_w,
    __nv_bfloat16* out, int T, int H, int I)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= T) return;
    extern __shared__ float smem[];
    float* gate = smem + threadIdx.x * (2*I);
    float* up   = gate + I;
    // gate = silu(X[t] @ gate_w.T)
    for (int i = 0; i < I; i++) {
        float a = 0;
        for (int h = 0; h < H; h++)
            a += __bfloat162float(X[t*H+h]) * __bfloat162float(gate_w[i*H+h]);
        gate[i] = a / (1.f + expf(-a));
    }
    // up = X[t] @ up_w.T
    for (int i = 0; i < I; i++) {
        float a = 0;
        for (int h = 0; h < H; h++)
            a += __bfloat162float(X[t*H+h]) * __bfloat162float(up_w[i*H+h]);
        up[i] = a;
    }
    // hidden = gate * up
    for (int i = 0; i < I; i++) gate[i] *= up[i];
    // out = hidden @ down_w.T  (down_w is [H,I])
    for (int h = 0; h < H; h++) {
        float a = 0;
        for (int i = 0; i < I; i++)
            a += gate[i] * __bfloat162float(down_w[h*I+i]);
        out[t*H+h] = __float2bfloat16(a);
    }
}

void fill(__nv_bfloat16* b, size_t n) {
    srand(42);
    for (size_t i = 0; i < n; i++)
        b[i] = __float2bfloat16(((float)rand()/RAND_MAX)*0.1f - 0.05f);
}

int main(int argc, char** argv) {
    if (argc < 6) { puts("T H I warmup iters"); return 1; }
    int T=atoi(argv[1]), H=atoi(argv[2]), I=atoi(argv[3]);
    int W=atoi(argv[4]), N=atoi(argv[5]);
    size_t xs=T*H, gs=I*H, ds=H*I;
    __nv_bfloat16 *dX,*dg,*du,*dd,*dout;
    cudaMalloc(&dX,xs*2); cudaMalloc(&dg,gs*2);
    cudaMalloc(&du,gs*2); cudaMalloc(&dd,ds*2); cudaMalloc(&dout,xs*2);
    __nv_bfloat16* hb = (__nv_bfloat16*)malloc((xs+2*gs+ds)*2);
    fill(hb, xs+2*gs+ds);
    cudaMemcpy(dX,hb,        xs*2,cudaMemcpyHostToDevice);
    cudaMemcpy(dg,hb+xs,     gs*2,cudaMemcpyHostToDevice);
    cudaMemcpy(du,hb+xs+gs,  gs*2,cudaMemcpyHostToDevice);
    cudaMemcpy(dd,hb+xs+2*gs,ds*2,cudaMemcpyHostToDevice);
    free(hb);
    int TH=256; size_t sm=(size_t)TH*2*I*sizeof(float);
    auto go=[&](){ naive_mlp<<<(T+TH-1)/TH,TH,sm>>>(dX,dg,du,dd,dout,T,H,I); };
    for(int i=0;i<W;i++) go(); cudaDeviceSynchronize();
    cudaEvent_t e0,e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    cudaEventRecord(e0);
    for(int i=0;i<N;i++) go();
    cudaEventRecord(e1); cudaEventSynchronize(e1);
    float ms=0; cudaEventElapsedTime(&ms,e0,e1); ms/=N;
    double flops = (double)T*I*H*6.0;
    printf("TIME_MS=%.4f\nTFLOPS=%.4f\n", ms, flops/(ms*1e9));
    return 0;
}
"""

# TK kernel: uses WMMA for gate_proj and up_proj GEMMs
# down_proj uses a correct tiled CUDA kernel (avoids TK layout confusion)
TK_CU = r"""
#include "kittens.cuh"
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
using namespace kittens;

static constexpr int B  = 32;
static constexpr int NW = 1;  // single warp: fixes inter-warp register scatter
static constexpr int NT = NW * WARP_THREADS;

// WMMA GEMM: out[T,I] = X[T,H] @ W[I,H].T
struct GemmGlobals {
    using tile = st_bf<B,B>;
    using gl_t = gl<bf16,1,1,-1,-1,tile>;
    gl_t X, W, out;
};

__global__ void __launch_bounds__(NT)
gemm_kernel(const __grid_constant__ GemmGlobals g, int T, int H, int I) {
    extern __shared__ alignment_dummy __shm[];
    shared_allocator al((int*)&__shm[0]);
    auto& Xs = al.allocate<st_bf<B,B>>();
    auto& Ws = al.allocate<st_bf<B,B>>();
    rt_bf<B,B> Xr, Wr;
    rt_bf<B,B,ducks::rt_layout::col> Wc;
    rt_fl<B,B> acc;
    int tt = blockIdx.y, it = blockIdx.x;
    int nk = (H+B-1)/B;
    warp::zero(acc);
    for (int k = 0; k < nk; k++) {
        warp::load(Xs, g.X, {0,0,tt,k}); __syncthreads();
        warp::load(Xr, Xs);
        warp::load(Ws, g.W, {0,0,it,k}); __syncthreads();
        warp::load(Wr, Ws); warp::swap_layout(Wc, Wr);
        warp::mma_AB(acc, Xr, Wc, acc);
        __syncthreads();
    }
    warp::store(g.out, acc, {0,0,tt,it});
}

void wmma_gemm(__nv_bfloat16* X, __nv_bfloat16* W, __nv_bfloat16* out,
               int T, int H, int I) {
    using tile = st_bf<B,B>;
    using gl_t = gl<bf16,1,1,-1,-1,tile>;
    gl_t Xg{(bf16*)X,  nullptr,nullptr,(size_t)T,(size_t)H};
    gl_t Wg{(bf16*)W,  nullptr,nullptr,(size_t)I,(size_t)H};
    gl_t Og{(bf16*)out,nullptr,nullptr,(size_t)T,(size_t)I};
    GemmGlobals g{Xg,Wg,Og};
    dim3 blocks((I+B-1)/B, (T+B-1)/B);
    size_t smem = 2*B*B*sizeof(__nv_bfloat16)+256;
    cudaFuncSetAttribute(gemm_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    gemm_kernel<<<blocks,NT,smem>>>(g,T,H,I);
}

// SiLU + multiply elementwise
__global__ void silu_mul_kernel(__nv_bfloat16* gate,
                                 const __nv_bfloat16* up, int N) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= N) return;
    float g = __bfloat162float(gate[i]);
    float u = __bfloat162float(up[i]);
    gate[i] = __float2bfloat16((g/(1.f+expf(-g))) * u);
}

// Down projection: out[T,H] = hidden[T,I] @ down[H,I].T
// Tiled 16x16 for good occupancy
#define TILE 16
__global__ void down_proj_tiled(
    const __nv_bfloat16* hidden, const __nv_bfloat16* down,
    __nv_bfloat16* out, int T, int H, int I)
{
    __shared__ float sh[TILE][TILE], sd[TILE][TILE];
    int t = blockIdx.y*TILE + threadIdx.y;
    int h = blockIdx.x*TILE + threadIdx.x;
    float acc = 0.f;
    for (int k = 0; k < I; k += TILE) {
        // Load tile of hidden[T,I]
        if (t < T && k+threadIdx.x < I)
            sh[threadIdx.y][threadIdx.x] = __bfloat162float(hidden[t*I + k+threadIdx.x]);
        else
            sh[threadIdx.y][threadIdx.x] = 0.f;
        // Load tile of down[H,I] — down[h, k+ty]
        if (h < H && k+threadIdx.y < I)
            sd[threadIdx.y][threadIdx.x] = __bfloat162float(down[(blockIdx.x*TILE+threadIdx.x)*I + k+threadIdx.y]);
        else
            sd[threadIdx.y][threadIdx.x] = 0.f;
        __syncthreads();
        for (int kk = 0; kk < TILE; kk++)
            acc += sh[threadIdx.y][kk] * sd[kk][threadIdx.x];
        __syncthreads();
    }
    if (t < T && h < H)
        out[t*H+h] = __float2bfloat16(acc);
}

void run_tk(__nv_bfloat16* X, __nv_bfloat16* gw, __nv_bfloat16* uw,
            __nv_bfloat16* dw, __nv_bfloat16* out, int T, int H, int I,
            __nv_bfloat16* gbuf, __nv_bfloat16* ubuf) {
    // Step 1+2: gate and up projections via WMMA
    wmma_gemm(X, gw, gbuf, T, H, I);
    wmma_gemm(X, uw, ubuf, T, H, I);
    cudaDeviceSynchronize();
    // Step 3: silu(gate) * up
    silu_mul_kernel<<<(T*I+255)/256,256>>>(gbuf, ubuf, T*I);
    cudaDeviceSynchronize();
    // Step 4: down projection (tiled CUDA)
    dim3 block(TILE,TILE);
    dim3 grid((H+TILE-1)/TILE, (T+TILE-1)/TILE);
    down_proj_tiled<<<grid,block>>>(gbuf, dw, out, T, H, I);
    CHECK_CUDA_ERROR(cudaGetLastError());
}

void fill(__nv_bfloat16* b, size_t n) {
    srand(42);
    for (size_t i = 0; i < n; i++)
        b[i] = __float2bfloat16(((float)rand()/RAND_MAX)*0.1f - 0.05f);
}

__nv_bfloat16* read_bin(const char* p, size_t n) {
    FILE* f = fopen(p,"rb");
    __nv_bfloat16* b = (__nv_bfloat16*)malloc(n*2);
    size_t g = fread(b,2,n,f); fclose(f); (void)g;
    return b;
}
void write_bin(const char* p, __nv_bfloat16* b, size_t n) {
    FILE* f = fopen(p,"wb"); fwrite(b,2,n,f); fclose(f);
}

int main(int argc, char** argv) {
    if (argc>=2 && strcmp(argv[1],"verify")==0) {
        if (argc<10) { puts("verify X gate up down out T H I"); return 1; }
        int T=atoi(argv[7]),H=atoi(argv[8]),I=atoi(argv[9]);
        size_t xs=T*H, gs=I*H, ds=H*I;
        __nv_bfloat16 *hX=read_bin(argv[2],xs), *hg=read_bin(argv[3],gs);
        __nv_bfloat16 *hu=read_bin(argv[4],gs), *hd=read_bin(argv[5],ds);
        __nv_bfloat16 *hout=(__nv_bfloat16*)malloc(xs*2);
        __nv_bfloat16 *dX,*dg,*du,*dd,*dout,*gbuf,*ubuf;
        cudaMalloc(&dX,xs*2); cudaMalloc(&dg,gs*2); cudaMalloc(&du,gs*2);
        cudaMalloc(&dd,ds*2); cudaMalloc(&dout,xs*2);
        cudaMalloc(&gbuf,T*I*2); cudaMalloc(&ubuf,T*I*2);
        cudaMemcpy(dX,hX,xs*2,cudaMemcpyHostToDevice);
        cudaMemcpy(dg,hg,gs*2,cudaMemcpyHostToDevice);
        cudaMemcpy(du,hu,gs*2,cudaMemcpyHostToDevice);
        cudaMemcpy(dd,hd,ds*2,cudaMemcpyHostToDevice);
        run_tk(dX,dg,du,dd,dout,T,H,I,gbuf,ubuf);
        cudaDeviceSynchronize();
        cudaMemcpy(hout,dout,xs*2,cudaMemcpyDeviceToHost);
        write_bin(argv[6],hout,xs);
        printf("VERIFY_OK\n");
        free(hX);free(hg);free(hu);free(hd);free(hout);
        cudaFree(dX);cudaFree(dg);cudaFree(du);cudaFree(dd);cudaFree(dout);
        cudaFree(gbuf);cudaFree(ubuf);
        return 0;
    }
    if (argc<6) { puts("T H I warmup iters"); return 1; }
    int T=atoi(argv[1]),H=atoi(argv[2]),I=atoi(argv[3]);
    int W=atoi(argv[4]),N=atoi(argv[5]);
    size_t xs=T*H, gs=I*H, ds=H*I;
    __nv_bfloat16 *dX,*dg,*du,*dd,*dout,*gbuf,*ubuf;
    cudaMalloc(&dX,xs*2); cudaMalloc(&dg,gs*2); cudaMalloc(&du,gs*2);
    cudaMalloc(&dd,ds*2); cudaMalloc(&dout,xs*2);
    cudaMalloc(&gbuf,T*I*2); cudaMalloc(&ubuf,T*I*2);
    __nv_bfloat16* hb=(__nv_bfloat16*)malloc((xs+2*gs+ds)*2);
    fill(hb,xs+2*gs+ds);
    cudaMemcpy(dX,hb,        xs*2,cudaMemcpyHostToDevice);
    cudaMemcpy(dg,hb+xs,     gs*2,cudaMemcpyHostToDevice);
    cudaMemcpy(du,hb+xs+gs,  gs*2,cudaMemcpyHostToDevice);
    cudaMemcpy(dd,hb+xs+2*gs,ds*2,cudaMemcpyHostToDevice);
    free(hb);
    for(int i=0;i<W;i++) run_tk(dX,dg,du,dd,dout,T,H,I,gbuf,ubuf);
    cudaDeviceSynchronize();
    cudaEvent_t e0,e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    cudaEventRecord(e0);
    for(int i=0;i<N;i++) run_tk(dX,dg,du,dd,dout,T,H,I,gbuf,ubuf);
    cudaEventRecord(e1); cudaEventSynchronize(e1);
    float ms=0; cudaEventElapsedTime(&ms,e0,e1); ms/=N;
    double flops = (double)T*I*H*6.0;
    printf("TIME_MS=%.4f\nTFLOPS=%.4f\n", ms, flops/(ms*1e9));
    cudaFree(dX);cudaFree(dg);cudaFree(du);cudaFree(dd);cudaFree(dout);
    cudaFree(gbuf);cudaFree(ubuf);
    return 0;
}
"""


@app.function(gpu="H100", timeout=600)
def run_benchmark():
    import subprocess, os
    workdir = "/tmp/week9"
    os.makedirs(workdir, exist_ok=True)
    open(f"{workdir}/moe_naive.cu","w").write(NAIVE_CU)
    open(f"{workdir}/moe_tk.cu","w").write(TK_CU)

    def compile(src, out, extra=""):
        r = subprocess.run(
            f"nvcc -O3 -std=c++20 -arch={ARCH} {extra} "
            f"{workdir}/{src} -o {workdir}/{out} -lm -lcuda",
            shell=True, capture_output=True, text=True)
        if r.returncode != 0:
            print(f"[COMPILE ERROR] {src}:\n{r.stderr}"); return False
        print(f"[OK] compiled {src}"); return True

    compile("moe_naive.cu", "moe_naive")
    compile("moe_tk.cu", "moe_tk", extra=FLAGS)

    configs = [
        ("tiny   (T=8,   H=16,   I=16)",    8,    16,   16,  3,  50),
        ("small  (T=512, H=1024, I=2048)", 512,  1024, 2048,  5, 100),
        ("medium (T=2048,H=2048, I=4096)", 2048, 2048, 4096,  5,  50),
    ]
    print("\n"+"="*70)
    print("  Week 9: DeepSeekMoE — TK WMMA (gate+up) + Tiled CUDA (down)")
    print("  GPU: H100  |  CUDA 12.6  |  sm_90a")
    print("="*70)
    print(f"{'Config':<36} {'Naive':>10} {'TK WMMA':>10} {'Speedup':>10}")
    print("-"*70)
    for label,T,H,I,Wm,N in configs:
        def run_bin(name):
            r=subprocess.run(f"{workdir}/{name} {T} {H} {I} {Wm} {N}",
                             shell=True,capture_output=True,text=True)
            for line in r.stdout.splitlines():
                if line.startswith("TIME_MS="): return float(line.split("=")[1])
            return None
        nm=run_bin("moe_naive"); tk=run_bin("moe_tk")
        fmt=lambda v:f"{v:.3f}ms" if v else "N/A"
        sp=f"{nm/tk:.1f}x" if(nm and tk) else "N/A"
        print(f"  {label:<34} {fmt(nm):>10} {fmt(tk):>10} {sp:>10}")
    print("="*70)
    print("\n  PyTorch (cuBLAS) reference timings:")
    import torch, torch.nn.functional as F
    dev, dt = "cuda", torch.bfloat16
    for label,T,H,I,Wm,N in configs:
        X=torch.randn(T,H,device=dev,dtype=dt)
        g=torch.randn(I,H,device=dev,dtype=dt)
        u=torch.randn(I,H,device=dev,dtype=dt)
        d=torch.randn(H,I,device=dev,dtype=dt)
        run = lambda: (F.silu(X@g.T)*(X@u.T))@d.T
        for _ in range(Wm): run()
        torch.cuda.synchronize()
        s=torch.cuda.Event(enable_timing=True)
        e=torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(N): run()
        e.record(); torch.cuda.synchronize()
        print(f"  {label:<34} {s.elapsed_time(e)/N:.3f}ms  (cuBLAS)")
    print("\nDone.")


@app.function(gpu="H100", timeout=600)
def verify_output():
    import subprocess, os
    import torch, torch.nn.functional as F
    workdir = "/tmp/week9"
    os.makedirs(workdir, exist_ok=True)
    open(f"{workdir}/moe_tk.cu","w").write(TK_CU)

    r = subprocess.run(
        f"nvcc -O3 -std=c++20 -arch={ARCH} {FLAGS} "
        f"{workdir}/moe_tk.cu -o {workdir}/moe_tk -lm -lcuda",
        shell=True, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"[COMPILE ERROR]\n{r.stderr}"); return
    print("[OK] moe_tk compiled\n")

    test_cases = [
        ("week7_tiny (T=8,  H=16, I=16)",  8,  16,  16),
        ("small      (T=32, H=64, I=64)", 32,  64,  64),
        ("medium     (T=64, H=128,I=256)", 64, 128, 256),
    ]
    print("="*65)
    print("  Week 9 Verification: TK WMMA vs PyTorch reference")
    print("="*65)
    print(f"  {'Config':<36} {'max_abs_err':>12} {'Result':>8}")
    print("-"*65)

    dev, dt = "cuda", torch.bfloat16
    TOL = 1.0   # bf16 accumulation error can be large for big matrices
    all_pass = True

    for label,T,H,I in test_cases:
        torch.manual_seed(42)
        X    = torch.randn(T,H,dtype=dt,device=dev)
        gate = torch.randn(I,H,dtype=dt,device=dev)
        up   = torch.randn(I,H,dtype=dt,device=dev)
        down = torch.randn(H,I,dtype=dt,device=dev)

        def save(name, t):
            open(f"{workdir}/{name}.bin",'wb').write(
                t.cpu().contiguous().view(torch.uint8).numpy().tobytes())
        

        save("X",X); save("gate",gate); save("up",up); save("down",down)
        ref = (F.silu(X@gate.T)*(X@up.T))@down.T

        cmd = [f"{workdir}/moe_tk","verify",
               f"{workdir}/X.bin",f"{workdir}/gate.bin",
               f"{workdir}/up.bin",f"{workdir}/down.bin",
               f"{workdir}/out.bin",str(T),str(H),str(I)]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode!=0 or not os.path.exists(f"{workdir}/out.bin"):
            print(f"  {label:<36} {'RUN ERROR':>12} {'FAIL':>8}")
            print(f"    {r.stderr[:200]}"); all_pass=False; continue

        raw    = open(f"{workdir}/out.bin",'rb').read()
        tk_out = torch.frombuffer(bytearray(raw),dtype=torch.bfloat16)\
                      .reshape(T,H).to(dev)

        # Relative error (handles large values better)
        abs_err = (ref.float()-tk_out.float()).abs()
        max_abs = abs_err.max().item()
        rel_err = (abs_err / (ref.float().abs()+1e-6)).max().item()
        passed  = rel_err < 0.05  # within 5% relative error
        if not passed: all_pass=False

        print(f"  {label:<36} {max_abs:>12.3e} {'PASS' if passed else 'FAIL':>8}  "
              f"(rel={rel_err:.3f})")
        # Print sample values
        print(f"    ref[0,:4] = {ref.float()[0,:4].tolist()}")
        print(f"    tk [0,:4] = {tk_out.float()[0,:4].tolist()}")

    print("="*65)
    print(f"\n  Overall: {'ALL PASS' if all_pass else 'SOME FAILED'}")
    print(f"  Criterion: relative error < 5%\n")


@app.local_entrypoint()
def main():
    print("\n>>> Running verification tests...")
    verify_output.remote()
    print("\n>>> Running benchmarks...")
    run_benchmark.remote()




