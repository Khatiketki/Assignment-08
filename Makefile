# Week 9 Makefile — DeepSeekMoE ThunderKittens implementation
#
# Requires:
#   - CUDA 12.3+  (sm_90a for H100/B200)
#   - ThunderKittens cloned at TK_DIR
#   - nvcc in PATH

TK_DIR     ?= $(HOME)/ThunderKittens
CUDA_ARCH  ?= sm_90a
NVCC       := nvcc
CXXFLAGS   := -O3 -std=c++20 -arch=$(CUDA_ARCH)
INCLUDES   := -I$(TK_DIR)/include
LDFLAGS    := -lcuda -lm

all: moe_tk moe_tk_tma moe_naive

# ThunderKittens WMMA kernel (Level 04 style)
moe_tk: moe_tk.cu
	$(NVCC) $(CXXFLAGS) $(INCLUDES) $< -o $@ $(LDFLAGS)
	@echo "Built moe_tk (WMMA tensor cores)"

# ThunderKittens TMA + WMMA kernel (Level 05 style)
moe_tk_tma: moe_tk_tma.cu
	$(NVCC) $(CXXFLAGS) $(INCLUDES) $< -o $@ $(LDFLAGS)
	@echo "Built moe_tk_tma (WMMA + TMA)"

# Naive scalar C baseline (Week 7 reference, compiled as CUDA for fair GPU timing)
moe_naive: moe_naive.cu
	$(NVCC) $(CXXFLAGS) $< -o $@ $(LDFLAGS)
	@echo "Built moe_naive (scalar baseline)"

# Run the Python benchmark (requires compiled binaries + torch)
bench: all
	python3 benchmark.py --plot

# Verify outputs match Week 7 test cases
verify: all
	python3 benchmark.py --verify

clean:
	rm -f moe_tk moe_tk_tma moe_naive benchmark_results.json benchmark_results.png

.PHONY: all bench verify clean
