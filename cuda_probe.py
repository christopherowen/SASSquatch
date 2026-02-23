#!/usr/bin/env python3
"""
CUDA C++ probe kernels for SASSquatch.

CUDA C++ gives the nvcc compiler a higher-level view of the code than raw PTX,
enabling it to infer value uniformity and promote operations to the uniform
datapath (UR registers, U-prefixed SASS instructions).

PTX alone cannot trigger most uniform instructions because:
  - ptxas at -O0 doesn't promote to the uniform datapath
  - ptxas at -O3 only promotes loads (LDCU) and descriptor addresses (UR)
  - The uniform datapath promotion is done by the nvcc frontend, not ptxas

CUDA C++ at -O3 triggers:
  - S2UR (special register -> uniform register)
  - LDCU / LDCU.64 / LDCU.128 (uniform constant loads)
  - USHF (uniform funnel shift)
  - UMOV (uniform move immediate)
  - ULEA (uniform load effective address)
  - And potentially many more (UIADD3, UIMAD, UISETP, ULOP3, etc.)

Each kernel snippet is designed to maximize a specific class of uniform
instructions by keeping all computation on block-uniform values (blockIdx,
gridDim, kernel parameters) and avoiding per-thread values until the
final store.

References:
  - NVIDIA CUDA Binary Utilities - Blackwell Instruction Set
    https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#blackwell-instruction-set
  - Section: "Uniform Datapath Instructions"
"""

from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class CUDAProbeKernel:
    """A CUDA C++ kernel designed to trigger specific SASS opcodes."""
    name: str
    description: str
    target_opcodes: List[str]  # SASS opcodes we expect to see
    source: str                # CUDA C++ source code
    compile_flags: List[str]   # Additional nvcc flags (besides -O3 -arch=)


def get_cuda_probe_kernels() -> List[CUDAProbeKernel]:
    """Return all CUDA C++ probe kernels."""
    kernels = []

    # -----------------------------------------------------------------------
    # 1. Uniform integer arithmetic (UIADD3, UIMAD, UISETP, UIMNMX, UIABS)
    # -----------------------------------------------------------------------
    kernels.append(CUDAProbeKernel(
        name="uniform_int_arith",
        description="Uniform integer add/mul/compare on block-uniform values",
        target_opcodes=["UIADD3", "UIMAD", "UISETP", "UIMNMX", "UIABS"],
        source=r"""
__global__ void uniform_int_arith(int* out, int n, int stride, int param_a) {
    // All values are block-uniform (same for all threads in a block)
    int bid = blockIdx.x;
    int bdy = blockDim.x;
    int gdy = gridDim.x;

    // Uniform arithmetic chain
    int block_offset = bid * bdy;             // UIMAD
    int grid_stride = gdy * bdy;              // UIMAD
    int num_iters = (n + grid_stride - 1) / grid_stride;  // UIADD3 + UIMAD (div)
    int scaled = stride * bdy + param_a;      // UIMAD + UIADD3
    int abs_val = abs(scaled - n);            // UIABS

    // Uniform compare
    int bounded = min(num_iters, 1024);       // UIMNMX
    int clamped = max(bounded, 1);            // UIMNMX

    // Use uniform result in per-thread store
    int tid = threadIdx.x;
    if (tid < clamped) {
        out[block_offset + tid] = abs_val + clamped;
    }
}
""",
        compile_flags=[],
    ))

    # -----------------------------------------------------------------------
    # 2. Uniform logic operations (ULOP3, ULOP32I, UBMSK, UBREV, UPOPC)
    # -----------------------------------------------------------------------
    kernels.append(CUDAProbeKernel(
        name="uniform_logic",
        description="Uniform bitwise/logic operations",
        target_opcodes=["ULOP3", "ULOP32I", "UBMSK", "UBREV", "UPOPC"],
        source=r"""
__global__ void uniform_logic(unsigned int* out, unsigned int mask_a,
                               unsigned int mask_b, unsigned int n) {
    unsigned int bid = blockIdx.x;
    unsigned int gdy = gridDim.x;

    // Uniform bitwise operations
    unsigned int and_val = mask_a & mask_b;           // ULOP3
    unsigned int or_val  = mask_a | mask_b;           // ULOP3
    unsigned int xor_val = mask_a ^ mask_b;           // ULOP3
    unsigned int not_val = ~mask_a;                   // ULOP3 / ULOP32I
    unsigned int ternary = (mask_a & mask_b) | (~mask_a & xor_val);  // ULOP3

    // Uniform bit manipulation
    unsigned int reversed = __brev(bid * 0x12345678u + mask_a);  // UBREV
    unsigned int popcount = __popc(and_val ^ or_val);             // UPOPC

    // Combine results uniformly
    unsigned int result = ternary ^ reversed ^ popcount;

    // Per-thread store
    int tid = threadIdx.x;
    if (tid < n) {
        out[bid * blockDim.x + tid] = result + tid;
    }
}
""",
        compile_flags=[],
    ))

    # -----------------------------------------------------------------------
    # 3. Uniform shift operations (USHF, USHL, USHR)
    # -----------------------------------------------------------------------
    kernels.append(CUDAProbeKernel(
        name="uniform_shift",
        description="Uniform shift and funnel shift operations",
        target_opcodes=["USHF", "USHL", "USHR", "USGXT"],
        source=r"""
__global__ void uniform_shift(int* out, int n, unsigned int shift_amt,
                               unsigned long long wide_val) {
    unsigned int bid = blockIdx.x;
    unsigned int bdy = blockDim.x;

    // Uniform shifts
    unsigned int lshift = bid << 4;                     // USHL
    unsigned int rshift = n >> 2;                       // USHR
    unsigned int funnel = __funnelshift_l(bid, n, 8);   // USHF

    // Uniform 64-bit address math (uses funnel shifts internally)
    unsigned long long base = (unsigned long long)bid * bdy * sizeof(int);
    unsigned int hi = (unsigned int)(base >> 32);       // USHR / USHF
    unsigned int lo = (unsigned int)base;               // UMOV

    // Sign extension pattern
    short narrow = (short)(bid & 0xFFFF);
    int extended = (int)narrow;                         // USGXT

    // Per-thread store
    int tid = threadIdx.x;
    out[bid * bdy + tid] = (int)(lshift + rshift + funnel + hi + lo + extended);
}
""",
        compile_flags=[],
    ))

    # -----------------------------------------------------------------------
    # 4. Uniform select and predicate (USEL, UPLOP3, UPSETP, UP2UR, UR2UP)
    # -----------------------------------------------------------------------
    kernels.append(CUDAProbeKernel(
        name="uniform_predicate",
        description="Uniform predicate logic and select",
        target_opcodes=["USEL", "UPLOP3", "UPSETP", "UP2UR", "UR2UP"],
        source=r"""
__global__ void uniform_predicate(int* out, int flag_a, int flag_b, int n) {
    int bid = blockIdx.x;
    int gdy = gridDim.x;

    // Uniform predicates
    bool p0 = (flag_a > 0);         // UISETP -> UP
    bool p1 = (flag_b > 0);         // UISETP -> UP
    bool p2 = (bid < (unsigned)n);  // UISETP -> UP

    // Uniform predicate logic
    bool p_and = p0 && p1;          // UPLOP3
    bool p_or  = p0 || p1;          // UPLOP3
    bool p_xor = p0 != p1;          // UPLOP3
    bool p_complex = (p0 && p2) || (!p1 && p2);  // UPLOP3

    // Uniform select based on predicate
    int val_a = bid * 100;
    int val_b = gdy * 200;
    int selected = p_and ? val_a : val_b;     // USEL
    int selected2 = p_complex ? n : flag_a;   // USEL

    // Per-thread store
    int tid = threadIdx.x;
    out[bid * blockDim.x + tid] = selected + selected2 + (p_or ? 1 : 0);
}
""",
        compile_flags=[],
    ))

    # -----------------------------------------------------------------------
    # 5. Uniform address computation (ULEA, UCLEA, LDCU)
    # -----------------------------------------------------------------------
    kernels.append(CUDAProbeKernel(
        name="uniform_addressing",
        description="Uniform address math for TMA-like patterns",
        target_opcodes=["ULEA", "UCLEA", "LDCU", "S2UR"],
        source=r"""
__global__ void uniform_addressing(float* __restrict__ C,
                                    const float* __restrict__ A,
                                    const float* __restrict__ B,
                                    int M, int N, int K, int lda, int ldb) {
    // Block-level tile coordinates (all uniform)
    int bm = blockIdx.x * 128;
    int bn = blockIdx.y * 128;

    // Uniform pointer arithmetic (complex address chain)
    const float* a_tile = A + bm * lda;        // UIMAD + UIADD3
    const float* b_tile = B + bn;              // UIADD3
    float* c_tile = C + bm * N + bn;           // UIMAD + UIADD3

    // Multiple uniform stride calculations
    int a_stride_row = lda;                     // LDCU (param)
    int b_stride_row = ldb;                     // LDCU (param)
    int c_stride_row = N;                       // LDCU (param)

    // Uniform loop over K dimension
    int k_tiles = (K + 31) / 32;               // UIADD3 + (division)

    // Per-thread work using uniform base
    int tid = threadIdx.x;
    int local_row = tid / 128;
    int local_col = tid % 128;

    float sum = 0.0f;
    for (int kt = 0; kt < k_tiles; kt++) {
        int k_base = kt * 32;
        if (k_base + local_col < K && bm + local_row < M) {
            sum += a_tile[local_row * a_stride_row + k_base + local_col];
        }
    }
    if (bm + local_row < M && bn + local_col < N) {
        c_tile[local_row * c_stride_row + local_col] = sum;
    }
}
""",
        compile_flags=[],
    ))

    # -----------------------------------------------------------------------
    # 6. Uniform float arithmetic (UFADD, UFFMA, UFMUL, UFMNMX) - Blackwell new
    # -----------------------------------------------------------------------
    kernels.append(CUDAProbeKernel(
        name="uniform_float",
        description="Uniform FP32 operations (Blackwell-new UFADD/UFFMA/UFMUL)",
        target_opcodes=["UFADD", "UFFMA", "UFMUL", "UFMNMX", "UFSEL", "UFSET", "UFSETP"],
        source=r"""
__global__ void uniform_float(float* out, float scale_a, float scale_b,
                               float bias, int n) {
    // Block-uniform float values from params
    float bid_f = (float)blockIdx.x;
    float gdy_f = (float)gridDim.x;

    // Uniform FP arithmetic
    float prod = scale_a * scale_b;                    // UFMUL
    float sum  = prod + bias;                          // UFADD
    float fma_val = __fmaf_rn(bid_f, scale_a, bias);  // UFFMA
    float mn = fminf(sum, fma_val);                    // UFMNMX
    float mx = fmaxf(sum, fma_val);                    // UFMNMX

    // Uniform float predicate
    float selected = (bid_f < gdy_f * 0.5f) ? mn : mx;  // UFSETP + UFSEL
    float normalized = selected / (gdy_f + 1.0f);        // Uniform division

    // Per-thread store
    int tid = threadIdx.x;
    if (tid < n) {
        out[blockIdx.x * blockDim.x + tid] = normalized + (float)tid * 0.001f;
    }
}
""",
        compile_flags=[],
    ))

    # -----------------------------------------------------------------------
    # 7. Uniform conversion (UF2I, UI2F, UF2F, UI2I)
    # -----------------------------------------------------------------------
    kernels.append(CUDAProbeKernel(
        name="uniform_conversion",
        description="Uniform type conversions (UF2I, UI2F, etc.)",
        target_opcodes=["UF2I", "UI2F", "UF2F", "UI2I", "UF2IP", "UI2FP"],
        source=r"""
__global__ void uniform_conversion(int* out, float fparam, int iparam, int n) {
    int bid = blockIdx.x;
    int gdy = gridDim.x;
    float bid_f = (float)bid;

    // Uniform float-to-int conversions
    int from_float = (int)fparam;                      // UF2I
    int rounded = __float2int_rn(fparam * bid_f);      // UF2I
    int truncated = __float2int_rz(fparam);            // UF2I

    // Uniform int-to-float conversions
    float from_int = (float)iparam;                    // UI2F
    float from_bid = (float)(bid * iparam);            // UI2F

    // Uniform int-to-int conversions
    short narrow = (short)(bid & 0x7FFF);              // UI2I
    unsigned char byte_val = (unsigned char)(gdy & 0xFF);  // UI2I
    int widened = (int)narrow + (int)byte_val;         // UI2I

    // Combine
    float combined = from_int + from_bid;
    int result = from_float + rounded + truncated + widened + (int)combined;

    // Per-thread store
    int tid = threadIdx.x;
    if (tid < n) {
        out[bid * blockDim.x + tid] = result + tid;
    }
}
""",
        compile_flags=[],
    ))

    # -----------------------------------------------------------------------
    # 8. Vote with uniform destination (VOTEU)
    # -----------------------------------------------------------------------
    kernels.append(CUDAProbeKernel(
        name="uniform_vote",
        description="Warp vote with uniform destination",
        target_opcodes=["VOTEU", "VOTE", "REDUX", "CREDUX"],
        source=r"""
__global__ void uniform_vote(int* out, int threshold, int n) {
    int tid = threadIdx.x;
    int bid = blockIdx.x;

    // Predicate based on thread data
    int val = tid + bid * blockDim.x;
    bool active = (val < threshold);

    // Vote operations
    unsigned int ballot = __ballot_sync(0xFFFFFFFF, active);
    bool all_active = __all_sync(0xFFFFFFFF, active);
    bool any_active = __any_sync(0xFFFFFFFF, active);

    // Warp reduction (triggers REDUX / CREDUX)
    int warp_sum = 0;
    for (int offset = 16; offset > 0; offset >>= 1) {
        warp_sum += __shfl_xor_sync(0xFFFFFFFF, val, offset);
    }

    // Store with uniform ballot mask
    if (tid < n) {
        out[bid * blockDim.x + tid] = (int)ballot + warp_sum +
            (all_active ? 1000 : 0) + (any_active ? 100 : 0);
    }
}
""",
        compile_flags=[],
    ))

    # -----------------------------------------------------------------------
    # 9. Uniform permute (UPRMT)
    # -----------------------------------------------------------------------
    kernels.append(CUDAProbeKernel(
        name="uniform_permute",
        description="Uniform byte permute operations",
        target_opcodes=["UPRMT", "PRMT"],
        source=r"""
__global__ void uniform_permute(unsigned int* out, unsigned int a,
                                 unsigned int b, int n) {
    unsigned int bid = blockIdx.x;

    // Uniform byte permute (PRMT on uniform values -> UPRMT)
    unsigned int src_a = a + bid;
    unsigned int src_b = b + bid;

    // Various permute selectors (uniform)
    unsigned int perm_0123 = __byte_perm(src_a, src_b, 0x3210);
    unsigned int perm_4567 = __byte_perm(src_a, src_b, 0x7654);
    unsigned int perm_cross = __byte_perm(src_a, src_b, 0x5140);
    unsigned int perm_rev = __byte_perm(src_a, src_b, 0x0123);

    unsigned int result = perm_0123 ^ perm_4567 ^ perm_cross ^ perm_rev;

    // Per-thread store
    int tid = threadIdx.x;
    if (tid < n) {
        out[bid * blockDim.x + tid] = result + tid;
    }
}
""",
        compile_flags=[],
    ))

    # -----------------------------------------------------------------------
    # 10. Uniform FLO (find leading one) and POPC
    # -----------------------------------------------------------------------
    kernels.append(CUDAProbeKernel(
        name="uniform_flo_popc",
        description="Uniform find-leading-one and popcount",
        target_opcodes=["UFLO", "UPOPC", "UBREV"],
        source=r"""
__global__ void uniform_flo_popc(int* out, unsigned int val, int n) {
    unsigned int bid = blockIdx.x;

    // Uniform FLO / CLZ
    unsigned int x = val + bid;
    int leading_zeros = __clz(x);            // UFLO (on uniform value)
    int first_set = __ffs(x);               // UFLO variant
    int popcount = __popc(x);               // UPOPC
    unsigned int reversed = __brev(x);      // UBREV

    // Chain more uniform ops
    int leading2 = __clz(reversed);
    int pop2 = __popc(reversed ^ x);

    int result = leading_zeros + first_set + popcount + (int)reversed +
                 leading2 + pop2;

    // Per-thread store
    int tid = threadIdx.x;
    if (tid < n) {
        out[bid * blockDim.x + tid] = result + tid;
    }
}
""",
        compile_flags=[],
    ))

    # -----------------------------------------------------------------------
    # 11. Shared memory with uniform layout (STS/LDS with uniform base)
    # -----------------------------------------------------------------------
    kernels.append(CUDAProbeKernel(
        name="uniform_shared_mem",
        description="Shared memory with uniform base address computation",
        target_opcodes=["UMOV", "ULEA", "STS", "LDS", "BAR"],
        source=r"""
__global__ void uniform_shared_mem(int* out, int n, int tile_size) {
    extern __shared__ int smem[];

    int bid = blockIdx.x;
    int tid = threadIdx.x;

    // Uniform shared memory layout computation
    int smem_offset_a = 0;                              // Uniform
    int smem_offset_b = tile_size * sizeof(int);        // UIMAD
    int smem_stride = blockDim.x;                       // Uniform

    // Per-thread shared mem access with uniform base
    smem[smem_offset_a / sizeof(int) + tid] = tid + bid;
    smem[smem_offset_b / sizeof(int) + tid] = tid * bid;
    __syncthreads();

    // Read back with uniform offset
    int val_a = smem[smem_offset_a / sizeof(int) + tid];
    int val_b = smem[smem_offset_b / sizeof(int) + tid];

    if (tid < n) {
        out[bid * blockDim.x + tid] = val_a + val_b;
    }
}
""",
        compile_flags=[],
    ))

    # -----------------------------------------------------------------------
    # 12. Warp shuffle with uniform arguments (SHFL + uniform patterns)
    # -----------------------------------------------------------------------
    kernels.append(CUDAProbeKernel(
        name="uniform_warp_shuffle",
        description="Warp shuffle with uniform source lane / width",
        target_opcodes=["SHFL", "WARPSYNC", "S2UR"],
        source=r"""
__global__ void uniform_warp_shuffle(int* out, int src_lane, int n) {
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int val = tid + bid * blockDim.x;

    // Shuffle with uniform parameters
    int from_lane = __shfl_sync(0xFFFFFFFF, val, src_lane);
    int xor_val = __shfl_xor_sync(0xFFFFFFFF, val, bid & 0x1F);
    int up_val = __shfl_up_sync(0xFFFFFFFF, val, 1);
    int down_val = __shfl_down_sync(0xFFFFFFFF, val, 1);

    // Butterfly reduction (uniform shuffle distances)
    int sum = val;
    sum += __shfl_xor_sync(0xFFFFFFFF, sum, 16);
    sum += __shfl_xor_sync(0xFFFFFFFF, sum, 8);
    sum += __shfl_xor_sync(0xFFFFFFFF, sum, 4);
    sum += __shfl_xor_sync(0xFFFFFFFF, sum, 2);
    sum += __shfl_xor_sync(0xFFFFFFFF, sum, 1);

    if (tid < n) {
        out[bid * blockDim.x + tid] = from_lane + xor_val + up_val +
                                       down_val + sum;
    }
}
""",
        compile_flags=[],
    ))

    # -----------------------------------------------------------------------
    # 13. Complex uniform control flow (BRA uniform, EXIT uniform)
    # -----------------------------------------------------------------------
    kernels.append(CUDAProbeKernel(
        name="uniform_control_flow",
        description="Uniform branch and control flow patterns",
        target_opcodes=["BRA", "EXIT", "CALL", "BSSY", "BSYNC", "YIELD", "NANOSLEEP"],
        source=r"""
__device__ __noinline__ int helper_func(int a, int b) {
    return a * b + a;
}

__global__ void uniform_control_flow(int* out, int mode, int iterations, int n) {
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int result = 0;

    // Uniform branch on kernel parameter
    if (mode == 0) {
        result = bid * 100;
    } else if (mode == 1) {
        // Uniform loop
        for (int i = 0; i < iterations; i++) {
            result += bid + i;
        }
    } else if (mode == 2) {
        // Uniform function call
        result = helper_func(bid, iterations);
    } else {
        // Yield hint
        __nanosleep(100);
        result = -bid;
    }

    // Uniform branch on blockIdx
    if (bid == 0) {
        result += 999;
    }

    if (tid < n) {
        out[bid * blockDim.x + tid] = result + tid;
    }
}
""",
        compile_flags=[],
    ))

    # -----------------------------------------------------------------------
    # 14. Uniform MUFU / special math (on uniform values)
    # -----------------------------------------------------------------------
    kernels.append(CUDAProbeKernel(
        name="uniform_special_math",
        description="Special math functions on uniform values",
        target_opcodes=["MUFU", "UFADD", "UFMUL"],
        source=r"""
__global__ void uniform_special_math(float* out, float x_val, int n) {
    float bid_f = (float)blockIdx.x;

    // Uniform special function inputs
    float x = x_val + bid_f * 0.01f;

    // Special math on uniform values
    float rcp = __frcp_rn(x);         // MUFU.RCP (maybe UMUFU?)
    float rsqrt = rsqrtf(x);          // MUFU.RSQ
    float lg2 = __log2f(x);           // MUFU.LG2
    float ex2 = __expf(x * 0.1f);     // MUFU.EX2
    float sinv = __sinf(x);           // MUFU.SIN
    float cosv = __cosf(x);           // MUFU.COS

    float combined = rcp + rsqrt + lg2 + ex2 + sinv + cosv;

    // Per-thread store
    int tid = threadIdx.x;
    if (tid < n) {
        out[blockIdx.x * blockDim.x + tid] = combined + (float)tid * 0.001f;
    }
}
""",
        compile_flags=[],
    ))

    # -----------------------------------------------------------------------
    # 15. Double precision uniform (DADD, DFMA, DMUL on uniform values)
    # -----------------------------------------------------------------------
    kernels.append(CUDAProbeKernel(
        name="uniform_double",
        description="FP64 operations on uniform values",
        target_opcodes=["DADD", "DFMA", "DMUL", "DSETP"],
        source=r"""
__global__ void uniform_double(double* out, double a, double b, int n) {
    double bid_d = (double)blockIdx.x;

    // Uniform FP64 arithmetic
    double sum = a + b;                           // DADD
    double prod = a * bid_d;                      // DMUL
    double fma_val = __fma_rn(a, bid_d, b);       // DFMA
    double result = (sum > prod) ? fma_val : sum;  // DSETP + SEL

    // Per-thread store
    int tid = threadIdx.x;
    if (tid < n) {
        out[blockIdx.x * blockDim.x + tid] = result + (double)tid;
    }
}
""",
        compile_flags=[],
    ))

    # -----------------------------------------------------------------------
    # 16. Async copy patterns (cp.async, LDGSTS)
    # -----------------------------------------------------------------------
    kernels.append(CUDAProbeKernel(
        name="async_copy",
        description="Asynchronous global->shared copy patterns",
        target_opcodes=["LDGSTS", "BAR", "DEPBAR"],
        source=r"""
#include <cuda_pipeline.h>

__global__ void async_copy(const int* __restrict__ src, int* out, int n) {
    __shared__ int smem[256];
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;

    // Async copy from global to shared
    if (gid < n) {
        __pipeline_memcpy_async(&smem[tid], &src[gid], sizeof(int));
    }
    __pipeline_commit();
    __pipeline_wait_prior(0);
    __syncthreads();

    // Use shared memory result
    if (gid < n) {
        out[gid] = smem[tid] * 2;
    }
}
""",
        compile_flags=[],
    ))

    # -----------------------------------------------------------------------
    # 17. Memory fence patterns (FENCE, MEMBAR)
    # -----------------------------------------------------------------------
    kernels.append(CUDAProbeKernel(
        name="memory_fences",
        description="Memory fence and barrier patterns",
        target_opcodes=["FENCE", "MEMBAR", "ERRBAR", "BAR"],
        source=r"""
__global__ void memory_fences(int* data, int* flag, int n) {
    int tid = threadIdx.x;
    int bid = blockIdx.x;

    if (bid == 0 && tid == 0) {
        // Write data
        data[0] = 42;
        // Memory fence to ensure visibility
        __threadfence();
        // Signal
        atomicExch(flag, 1);
    }

    __syncthreads();
    __threadfence_block();

    if (tid < n) {
        data[bid * blockDim.x + tid] = tid;
    }

    __threadfence_system();
}
""",
        compile_flags=[],
    ))

    # -----------------------------------------------------------------------
    # 18. Paired FP32 operations (FADD2, FFMA2, FMUL2) - Blackwell new
    # -----------------------------------------------------------------------
    kernels.append(CUDAProbeKernel(
        name="paired_fp32",
        description="Paired FP32 operations (Blackwell-new FADD2/FFMA2/FMUL2)",
        target_opcodes=["FADD2", "FFMA2", "FMUL2"],
        source=r"""
// Paired FP32 operations may be generated by the compiler when it
// can prove two independent FP32 operations can be fused.
__global__ void paired_fp32(float* __restrict__ out,
                             const float* __restrict__ a,
                             const float* __restrict__ b,
                             const float* __restrict__ c,
                             int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idx2 = idx + n;

    if (idx < n) {
        // Independent pairs of FP32 operations
        float a0 = a[idx];
        float a1 = a[idx2];
        float b0 = b[idx];
        float b1 = b[idx2];
        float c0 = c[idx];
        float c1 = c[idx2];

        // These independent FMAs might get paired into FFMA2
        float r0 = __fmaf_rn(a0, b0, c0);
        float r1 = __fmaf_rn(a1, b1, c1);

        // Independent adds might get paired into FADD2
        float s0 = r0 + a0;
        float s1 = r1 + a1;

        // Independent muls might get paired into FMUL2
        float t0 = s0 * b0;
        float t1 = s1 * b1;

        out[idx] = t0;
        out[idx2] = t1;
    }
}
""",
        compile_flags=[],
    ))

    # -----------------------------------------------------------------------
    # 19. Cache control and prefetch (CCTL, CCTLL, CCTLT)
    # -----------------------------------------------------------------------
    kernels.append(CUDAProbeKernel(
        name="cache_control",
        description="Cache control and prefetch instructions",
        target_opcodes=["CCTL", "CCTLL", "CCTLT", "LDGMC"],
        source=r"""
__global__ void cache_control(const float* __restrict__ in,
                               float* __restrict__ out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Prefetch pattern (may generate CCTL)
    if (idx + 256 < n) {
        // L2 prefetch
        asm volatile("prefetch.global.L2 [%0];" :: "l"(&in[idx + 256]));
    }

    if (idx < n) {
        float val = in[idx];
        out[idx] = val * 2.0f;
    }

    // Cache invalidate pattern
    if (idx < n) {
        asm volatile("discard.global.L2 [%0], 128;" :: "l"(&out[idx]));
    }
}
""",
        compile_flags=[],
    ))

    # -----------------------------------------------------------------------
    # 20. LEA and ISCADD patterns
    # -----------------------------------------------------------------------
    kernels.append(CUDAProbeKernel(
        name="lea_iscadd",
        description="Load effective address and scaled add patterns",
        target_opcodes=["LEA", "ISCADD", "ULEA"],
        source=r"""
__global__ void lea_iscadd(int* out, int base, int stride, int n) {
    int bid = blockIdx.x;
    int tid = threadIdx.x;

    // LEA-generating patterns: base + index * scale
    int addr1 = base + bid * 4;          // LEA with scale=4
    int addr2 = base + bid * 8;          // LEA with scale=8
    int addr3 = base + bid * 16;         // LEA with scale=16

    // ISCADD pattern: a + b << shift
    int iscadd1 = tid + (bid << 2);      // ISCADD shift=2
    int iscadd2 = tid + (bid << 4);      // ISCADD shift=4
    int iscadd3 = tid + (bid << 8);      // ISCADD shift=8

    // Multi-dimensional address (common in GEMM tiling)
    int row = bid / stride;
    int col = bid % stride;
    int flat = row * n + col;

    out[bid * blockDim.x + tid] = addr1 + addr2 + addr3 +
                                   iscadd1 + iscadd2 + iscadd3 + flat;
}
""",
        compile_flags=[],
    ))

    # -----------------------------------------------------------------------
    # 21. Elect leader thread (ELECT) - Hopper+ instruction
    # -----------------------------------------------------------------------
    kernels.append(CUDAProbeKernel(
        name="elect_leader",
        description="Elect leader thread in warp",
        target_opcodes=["ELECT", "VOTE", "VOTEU"],
        source=r"""
__global__ void elect_leader(int* out, int n) {
    int tid = threadIdx.x;
    int bid = blockIdx.x;

    // Elect a leader using ballot
    unsigned int mask = __ballot_sync(0xFFFFFFFF, true);
    int leader = __ffs(mask) - 1;

    // Conditional based on leader
    int val;
    if (tid == leader) {
        val = 999;
    } else {
        val = tid;
    }

    // Match instruction
    unsigned int matching = __match_any_sync(0xFFFFFFFF, tid / 8);

    if (tid < n) {
        out[bid * blockDim.x + tid] = val + (int)matching;
    }
}
""",
        compile_flags=[],
    ))

    # -----------------------------------------------------------------------
    # 22. MOVM (matrix transpose) and LDSM/STSM
    # -----------------------------------------------------------------------
    kernels.append(CUDAProbeKernel(
        name="matrix_mem",
        description="Matrix load/store (LDSM, STSM, MOVM patterns)",
        target_opcodes=["LDSM", "STSM", "MOVM"],
        source=r"""
#include <mma.h>
using namespace nvcuda;

__global__ void matrix_mem(half* __restrict__ out,
                            const half* __restrict__ in, int n) {
    __shared__ half smem[16 * 16];

    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane_id = tid % 32;
    int bid = blockIdx.x;

    // Load to shared memory
    if (tid < 256) {
        smem[tid] = in[bid * 256 + tid];
    }
    __syncthreads();

    // wmma fragments (triggers LDSM, MOVM)
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> frag_a;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> frag_b;
    wmma::fragment<wmma::accumulator, 16, 16, 16, half> frag_c;

    wmma::fill_fragment(frag_c, __float2half(0.0f));
    wmma::load_matrix_sync(frag_a, smem, 16);
    wmma::load_matrix_sync(frag_b, smem, 16);

    // MMA
    wmma::mma_sync(frag_c, frag_a, frag_b, frag_c);

    // Store (triggers STSM-like patterns)
    wmma::store_matrix_sync(&out[bid * 256], frag_c, 16, wmma::mem_row_major);
}
""",
        compile_flags=[],
    ))

    # -----------------------------------------------------------------------
    # 23. SETMAXREG / USETMAXREG (register allocation hint)
    # -----------------------------------------------------------------------
    kernels.append(CUDAProbeKernel(
        name="setmaxreg",
        description="Register allocation hints (SETMAXREG pattern)",
        target_opcodes=["USETMAXREG", "SETMAXREG"],
        source=r"""
// High register pressure kernel to trigger SETMAXREG
__global__ __launch_bounds__(128, 1) void setmaxreg(
    float* __restrict__ out, const float* __restrict__ in, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    // Use many registers
    float r0 = in[tid];
    float r1 = in[tid + n];
    float r2 = in[tid + 2*n];
    float r3 = in[tid + 3*n];
    float r4 = r0 * r1 + r2;
    float r5 = r1 * r2 + r3;
    float r6 = r2 * r3 + r0;
    float r7 = r3 * r0 + r1;
    float r8 = r4 * r5;
    float r9 = r6 * r7;
    float r10 = r8 + r9;
    float r11 = r0 + r1 + r2 + r3;
    float r12 = r4 + r5 + r6 + r7;
    float r13 = r10 * r11;
    float r14 = r12 * r13;
    float r15 = r14 + r0;

    out[tid] = r15;
}
""",
        compile_flags=[],
    ))

    # ===================================================================
    # Tier 1: Easy wins - integer SIMD, basic ops, predicates, memory
    # ===================================================================

    # -----------------------------------------------------------------------
    # T1-1. Integer SIMD intrinsics (VABSDIFF, VABSDIFF4, VIADD, VIADDMNMX)
    # -----------------------------------------------------------------------
    kernels.append(CUDAProbeKernel(
        name="simd_vabsdiff",
        description="SIMD absolute difference (VABSDIFF, VABSDIFF4)",
        target_opcodes=["VABSDIFF", "VABSDIFF4", "VIADD", "VIADDMNMX"],
        source=r"""
__global__ void simd_vabsdiff(unsigned int* out, unsigned int a,
                               unsigned int b, int n) {
    int tid = threadIdx.x;
    int bid = blockIdx.x;

    unsigned int va = a + tid;
    unsigned int vb = b + tid;

    // __vabsdiff4: packed 4-byte absolute difference -> VABSDIFF4
    unsigned int diff4 = __vabsdiff4(va, vb);

    // __vabsdiff2: packed 2-halfword absolute difference -> VABSDIFF
    unsigned int diff2 = __vabsdiff2(va, vb);

    // __vadd4: packed 4-byte add -> VIADD
    unsigned int sum4 = __vadd4(va, vb);

    // __vsub4: packed 4-byte subtract
    unsigned int sub4 = __vsub4(va, vb);

    // __vmin4 / __vmax4: packed 4-byte min/max
    unsigned int min4 = __vminu4(va, vb);
    unsigned int max4 = __vmaxu4(va, vb);

    // __vaddss4: packed add with saturation
    unsigned int satadd4 = __vaddss4((int)va, (int)vb);

    // Combine and store
    if (tid < n) {
        out[bid * blockDim.x + tid] = diff4 ^ diff2 ^ sum4 ^ sub4 ^
                                       min4 ^ max4 ^ satadd4;
    }
}
""",
        compile_flags=[],
    ))

    # -----------------------------------------------------------------------
    # T1-2. SIMD FP16 min/max (VHMNMX, HMNMX2)
    # -----------------------------------------------------------------------
    kernels.append(CUDAProbeKernel(
        name="simd_fp16_minmax",
        description="FP16 SIMD min/max (VHMNMX, HMNMX2)",
        target_opcodes=["VHMNMX", "HMNMX2"],
        source=r"""
#include <cuda_fp16.h>

__global__ void simd_fp16_minmax(half2* out, const half2* a,
                                  const half2* b, int n) {
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;

    if (gid < n) {
        half2 va = a[gid];
        half2 vb = b[gid];

        // FP16x2 min/max -> HMNMX2 or VHMNMX
        half2 mn = __hmin2(va, vb);
        half2 mx = __hmax2(va, vb);

        // Scalar half min/max
        half s_mn = __hmin(__low2half(va), __low2half(vb));
        half s_mx = __hmax(__high2half(va), __high2half(vb));

        // Combine
        half2 result = __hadd2(mn, mx);
        result = __hadd2(result, __halves2half2(s_mn, s_mx));

        out[gid] = result;
    }
}
""",
        compile_flags=[],
    ))

    # -----------------------------------------------------------------------
    # T1-3. SIMD 3-input min/max (VIMNMX3, FMNMX3)
    # -----------------------------------------------------------------------
    kernels.append(CUDAProbeKernel(
        name="three_input_minmax",
        description="3-input min/max (VIMNMX3, FMNMX3)",
        target_opcodes=["VIMNMX3", "FMNMX3"],
        source=r"""
__global__ void three_input_minmax(int* out, const int* a,
                                    const int* b, const int* c, int n) {
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;

    if (gid < n) {
        int va = a[gid];
        int vb = b[gid];
        int vc = c[gid];

        // 3-input min: min(a, min(b, c)) - compiler may fuse to VIMNMX3
        int mn3 = min(va, min(vb, vc));
        int mx3 = max(va, max(vb, vc));

        // Median of three: max(min(a,b), min(max(a,b),c))
        int median = max(min(va, vb), min(max(va, vb), vc));

        out[gid] = mn3 + mx3 + median;
    }
}

__global__ void three_input_fminmax(float* out, const float* a,
                                     const float* b, const float* c, int n) {
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;

    if (gid < n) {
        float va = a[gid];
        float vb = b[gid];
        float vc = c[gid];

        // 3-input float min/max -> FMNMX3
        float mn3 = fminf(va, fminf(vb, vc));
        float mx3 = fmaxf(va, fmaxf(vb, vc));
        float median = fmaxf(fminf(va, vb), fminf(fmaxf(va, vb), vc));

        out[gid] = mn3 + mx3 + median;
    }
}
""",
        compile_flags=[],
    ))

    # -----------------------------------------------------------------------
    # T1-4. Bitfield mask and standalone shifts (BMSK, SHL, SHR, LOP)
    # -----------------------------------------------------------------------
    kernels.append(CUDAProbeKernel(
        name="bitmask_shifts",
        description="Bitfield mask, standalone shifts (BMSK, SHL, SHR)",
        target_opcodes=["BMSK", "SHL", "SHR", "LOP"],
        source=r"""
__device__ __noinline__ unsigned int make_bitmask(unsigned int offset,
                                                   unsigned int width) {
    // BMSK: create bitmask from offset and width
    // ((1 << width) - 1) << offset
    return ((1u << width) - 1u) << offset;
}

__device__ __noinline__ unsigned int shift_ops(unsigned int val,
                                                unsigned int amount) {
    // Force standalone SHL/SHR by using __noinline__
    unsigned int left = val << amount;
    unsigned int right = val >> amount;
    return left ^ right;
}

__device__ __noinline__ unsigned int logic_2input(unsigned int a,
                                                   unsigned int b) {
    // 2-input logic: AND, OR, XOR separately -> LOP (vs LOP3)
    unsigned int r1 = a & b;
    unsigned int r2 = a | b;
    unsigned int r3 = a ^ b;
    unsigned int r4 = ~a;
    return r1 + r2 + r3 + r4;
}

__global__ void bitmask_shifts(unsigned int* out, unsigned int val, int n) {
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;

    if (gid < n) {
        unsigned int offset = tid & 0x1F;
        unsigned int width = (tid >> 5) & 0xF;
        if (width == 0) width = 1;
        if (offset + width > 32) width = 32 - offset;

        unsigned int mask = make_bitmask(offset, width);
        unsigned int shifted = shift_ops(val + tid, offset);
        unsigned int logic = logic_2input(val + tid, mask);

        out[gid] = mask ^ shifted ^ logic;
    }
}
""",
        compile_flags=[],
    ))

    # -----------------------------------------------------------------------
    # T1-5. 32-bit immediate multiply and scaled add (IMUL32I, ISCADD32I)
    # -----------------------------------------------------------------------
    kernels.append(CUDAProbeKernel(
        name="imm32_mul_scadd",
        description="32-bit immediate multiply and scaled add",
        target_opcodes=["IMUL32I", "ISCADD32I", "IADD32I", "LOP32I"],
        source=r"""
__device__ __noinline__ int mul_by_large_const(int x) {
    // Force IMUL32I by using a large immediate that doesn't fit in IMAD's field
    return x * 0x12345678;
}

__device__ __noinline__ int scadd_variants(int a, int b) {
    // ISCADD32I: a + (imm << shift)
    int r1 = a + (b << 2) + 0x1234;
    int r2 = a + (b << 4) + 0x5678;
    return r1 + r2;
}

__device__ __noinline__ unsigned int logic_imm32(unsigned int x) {
    // LOP32I: logic with large 32-bit immediate
    unsigned int r1 = x & 0xDEADBEEF;
    unsigned int r2 = x | 0xCAFEBABE;
    unsigned int r3 = x ^ 0x12345678;
    return r1 + r2 + r3;
}

__global__ void imm32_mul_scadd(int* out, int val, int n) {
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;

    if (gid < n) {
        int x = val + tid;
        int m = mul_by_large_const(x);
        int s = scadd_variants(x, tid);
        unsigned int l = logic_imm32((unsigned int)x);

        // Also 32-bit immediate add
        int a = x + 0x7FFFFFFF;

        out[gid] = m + s + (int)l + a;
    }
}
""",
        compile_flags=[],
    ))

    # -----------------------------------------------------------------------
    # T1-6. Predicate combine and FP set (PSETP, FSET)
    # -----------------------------------------------------------------------
    kernels.append(CUDAProbeKernel(
        name="predicate_combine",
        description="Predicate combine (PSETP) and FP compare-and-set (FSET)",
        target_opcodes=["PSETP", "FSET", "HSET2", "HSETP2"],
        source=r"""
#include <cuda_fp16.h>

__global__ void predicate_combine(int* out, const float* a,
                                   const float* b, int threshold, int n) {
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;

    if (gid < n) {
        float va = a[gid];
        float vb = b[gid];

        // FSET: float compare -> set register (not predicate)
        // Use ternary with float result to force FSET instead of FSETP
        float cmp_gt = (va > vb) ? 1.0f : 0.0f;
        float cmp_lt = (va < vb) ? 1.0f : 0.0f;
        float cmp_eq = (va == vb) ? 1.0f : 0.0f;

        // Multiple predicates that need combining -> PSETP
        bool p0 = (va > 0.0f);
        bool p1 = (vb > 0.0f);
        bool p2 = (tid < threshold);
        bool p3 = (va > vb);

        // Complex predicate combinations
        bool p_and_or = (p0 && p1) || (p2 && p3);
        bool p_xor_and = (p0 != p1) && (p2 || p3);
        bool p_not_and = (!p0) && p1 && p2;

        float fset_result = cmp_gt + cmp_lt + cmp_eq;
        int pred_result = (p_and_or ? 100 : 0) +
                          (p_xor_and ? 10 : 0) +
                          (p_not_and ? 1 : 0);

        out[gid] = (int)fset_result + pred_result;
    }
}

__global__ void fp16_set_compare(int* out, const half* a,
                                  const half* b, int n) {
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;

    if (gid < n) {
        half va = a[gid];
        half vb = b[gid];

        // HSET2 / HSETP2: FP16 compare-and-set
        half2 va2 = __halves2half2(va, va);
        half2 vb2 = __halves2half2(vb, vb);

        bool lt = __hlt(va, vb);
        bool gt = __hgt(va, vb);
        bool eq = __heq(va, vb);

        out[gid] = (lt ? 4 : 0) + (gt ? 2 : 0) + (eq ? 1 : 0);
    }
}
""",
        compile_flags=[],
    ))

    # -----------------------------------------------------------------------
    # T1-7. Generic memory load/store (LD, ST via generic pointers)
    # -----------------------------------------------------------------------
    kernels.append(CUDAProbeKernel(
        name="generic_memory",
        description="Generic memory load/store (LD, ST, LDL, STL)",
        target_opcodes=["LD", "ST", "LDL", "STL"],
        source=r"""
// Use generic pointer to force LD/ST instead of LDG/STG
__device__ __noinline__ int generic_load(void* ptr) {
    // Generic pointer -> LD (not LDG)
    return *(volatile int*)ptr;
}

__device__ __noinline__ void generic_store(void* ptr, int val) {
    // Generic pointer -> ST (not STG)
    *(volatile int*)ptr = val;
}

__global__ void generic_memory(int* out, int* in_data, int n) {
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;

    // Local memory usage -> LDL / STL
    int local_arr[4];
    local_arr[0] = tid;
    local_arr[1] = tid * 2;
    local_arr[2] = tid * 3;
    local_arr[3] = tid * 4;

    // Force spill to local memory by using variable index
    int idx = tid & 3;
    int local_val = local_arr[idx];

    if (gid < n) {
        // Generic pointer load/store
        int val = generic_load(&in_data[gid]);
        generic_store(&out[gid], val + local_val);
    }
}
""",
        compile_flags=[],
    ))

    # -----------------------------------------------------------------------
    # T1-8. L2 cache prefetch and discard (CCTLL, CCTLT)
    # -----------------------------------------------------------------------
    kernels.append(CUDAProbeKernel(
        name="cache_control_l2",
        description="L2 prefetch and cache control (CCTLL, CCTLT)",
        target_opcodes=["CCTLL", "CCTLT"],
        source=r"""
__global__ void cache_control_l2(const float* __restrict__ in,
                                  float* __restrict__ out, int n) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;

    if (gid < n) {
        // Prefetch to L1
        asm volatile("prefetch.global.L1 [%0];" :: "l"(&in[gid]));

        // Prefetch to L2
        if (gid + 256 < n) {
            asm volatile("prefetch.global.L2 [%0];" :: "l"(&in[gid + 256]));
        }

        float val = in[gid];

        // Discard L2 line for streaming writes
        asm volatile("discard.global.L2 [%0], 128;" :: "l"(&out[gid]));

        out[gid] = val * 2.0f;
    }
}
""",
        compile_flags=[],
    ))

    # -----------------------------------------------------------------------
    # T1-9. Texture fetch (TEX, TLD, TLD4, TMML, TXD, TXQ)
    # -----------------------------------------------------------------------
    kernels.append(CUDAProbeKernel(
        name="texture_ops",
        description="Texture operations (TEX, TLD4, TXQ, TMML)",
        target_opcodes=["TEX", "TLD", "TLD4", "TMML", "TXD", "TXQ"],
        source=r"""
// Use texture objects (bindless textures) for TEX/TLD4/TXQ
// Note: This won't execute correctly without actual texture data,
// but it will compile and emit the right SASS opcodes.

__global__ void texture_ops(float4* out, cudaTextureObject_t tex1d,
                             cudaTextureObject_t tex2d, int n) {
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;

    if (gid < n) {
        float u = (float)gid / (float)n;
        float v = (float)tid / (float)blockDim.x;

        // 1D texture fetch -> TEX
        float4 t1 = tex1Dfetch<float4>(tex1d, gid);

        // 2D texture fetch -> TEX
        float4 t2 = tex2D<float4>(tex2d, u, v);

        // Texture gather (TLD4) - gathers one component from 4 neighbors
        // (only available via inline asm for full control)

        out[gid] = make_float4(t1.x + t2.x, t1.y + t2.y,
                                t1.z + t2.z, t1.w + t2.w);
    }
}
""",
        compile_flags=[],
    ))

    # -----------------------------------------------------------------------
    # T1-10. Surface load (SULD) - complete surface operations
    # -----------------------------------------------------------------------
    kernels.append(CUDAProbeKernel(
        name="surface_load",
        description="Surface load operations (SULD, SUATOM)",
        target_opcodes=["SULD", "SUATOM"],
        source=r"""
__global__ void surface_load(float* out, cudaSurfaceObject_t surf1d,
                              int n) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;

    if (gid < n) {
        // Surface read -> SULD
        float val;
        surf1Dread(&val, surf1d, gid * sizeof(float));

        out[gid] = val;
    }
}
""",
        compile_flags=[],
    ))

    # -----------------------------------------------------------------------
    # T1-11. Matrix movement / HMMA wmma (MOVM, LDSM, STSM patterns)
    # -----------------------------------------------------------------------
    kernels.append(CUDAProbeKernel(
        name="matrix_movement",
        description="Matrix transpose/movement (MOVM) via wmma",
        target_opcodes=["MOVM", "HMMA"],
        source=r"""
#include <mma.h>
using namespace nvcuda;

// Force MOVM by doing fragment operations that require transpose
__global__ void matrix_movement(half* __restrict__ out,
                                 const half* __restrict__ A,
                                 const half* __restrict__ B,
                                 int M, int N, int K) {
    int warpM = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int warpN = blockIdx.y;

    if (warpM * 16 >= M || warpN * 16 >= N) return;

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, half> c_frag;
    wmma::fill_fragment(c_frag, __float2half(0.0f));

    for (int k = 0; k < K; k += 16) {
        const half* a_ptr = A + warpM * 16 * K + k;
        const half* b_ptr = B + k * N + warpN * 16;

        wmma::load_matrix_sync(a_frag, a_ptr, K);
        wmma::load_matrix_sync(b_frag, b_ptr, N);
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }

    half* c_ptr = out + warpM * 16 * N + warpN * 16;
    wmma::store_matrix_sync(c_ptr, c_frag, N, wmma::mem_row_major);
}
""",
        compile_flags=[],
    ))

    # -----------------------------------------------------------------------
    # T1-12. OMMA (FP4 MMA) via inline PTX - Blackwell scaled MMA
    # -----------------------------------------------------------------------
    kernels.append(CUDAProbeKernel(
        name="omma_fp4",
        description="FP4 MMA (OMMA) via Blackwell tcgen05 MMA",
        target_opcodes=["OMMA"],
        source=r"""
#include <mma.h>
using namespace nvcuda;

// Blackwell OMMA is emitted by wmma with low-precision types.
// Since inline PTX mma.kind::mxf8f6f4 requires very specific register counts
// that vary by arch, use wmma which lets the compiler handle it.
// Also try the simpler mma.sync for FP16 which emits HMMA on older archs
// but may emit different patterns on Blackwell.
__global__ void omma_fp4(half* __restrict__ out,
                          const half* __restrict__ A,
                          const half* __restrict__ B, int N) {
    int warpM = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int warpN = blockIdx.y;

    if (warpM * 16 >= N || warpN * 16 >= N) return;

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
    wmma::fill_fragment(c_frag, 0.0f);

    wmma::load_matrix_sync(a_frag, A + warpM * 16 * N, N);
    wmma::load_matrix_sync(b_frag, B, N);
    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

    // Store with conversion
    wmma::fragment<wmma::accumulator, 16, 16, 16, half> out_frag;
    for (int i = 0; i < c_frag.num_elements; i++)
        out_frag.x[i] = __float2half(c_frag.x[i]);

    wmma::store_matrix_sync(out + warpM * 16 * N + warpN * 16, out_frag, N,
                             wmma::mem_row_major);
}
""",
        compile_flags=[],
    ))

    # -----------------------------------------------------------------------
    # T1-13. QMMA (integer MMA) via wmma int8
    # -----------------------------------------------------------------------
    kernels.append(CUDAProbeKernel(
        name="qmma_int8",
        description="Integer MMA (QMMA) via wmma int8",
        target_opcodes=["QMMA"],
        source=r"""
#include <mma.h>
using namespace nvcuda;

// Integer MMA: int8 x int8 -> int32, which should emit QMMA on Blackwell
__global__ void qmma_int8(int* __restrict__ out,
                           const signed char* __restrict__ A,
                           const signed char* __restrict__ B,
                           int M, int N, int K) {
    int warpM = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int warpN = blockIdx.y;

    if (warpM * 16 >= M || warpN * 16 >= N) return;

    wmma::fragment<wmma::matrix_a, 16, 16, 16, signed char, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, signed char, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, int> c_frag;
    wmma::fill_fragment(c_frag, 0);

    for (int k = 0; k < K; k += 16) {
        wmma::load_matrix_sync(a_frag, A + warpM * 16 * K + k, K);
        wmma::load_matrix_sync(b_frag, B + k * N + warpN * 16, N);
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }

    int* c_ptr = out + warpM * 16 * N + warpN * 16;
    wmma::store_matrix_sync(c_ptr, c_frag, N, wmma::mem_row_major);
}
""",
        compile_flags=[],
    ))

    # -----------------------------------------------------------------------
    # T1-14. DMMA (FP64 MMA) via wmma
    # -----------------------------------------------------------------------
    kernels.append(CUDAProbeKernel(
        name="dmma_fp64",
        description="FP64 MMA (DMMA) via wmma double precision",
        target_opcodes=["DMMA"],
        source=r"""
#include <mma.h>
using namespace nvcuda;

__global__ void dmma_fp64(double* __restrict__ out,
                           const double* __restrict__ A,
                           const double* __restrict__ B,
                           int M, int N, int K) {
    int warpM = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int warpN = blockIdx.y;

    if (warpM * 8 >= M || warpN * 8 >= N) return;

    wmma::fragment<wmma::matrix_a, 8, 8, 4, double, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 8, 8, 4, double, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, 8, 8, 4, double> c_frag;
    wmma::fill_fragment(c_frag, 0.0);

    for (int k = 0; k < K; k += 4) {
        const double* a_ptr = A + warpM * 8 * K + k;
        const double* b_ptr = B + k * N + warpN * 8;

        wmma::load_matrix_sync(a_frag, a_ptr, K);
        wmma::load_matrix_sync(b_frag, b_ptr, N);
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }

    double* c_ptr = out + warpM * 8 * N + warpN * 8;
    wmma::store_matrix_sync(c_ptr, c_frag, N, wmma::mem_row_major);
}
""",
        compile_flags=[],
    ))

    # -----------------------------------------------------------------------
    # T1-15. Sign extend and SGXT
    # -----------------------------------------------------------------------
    kernels.append(CUDAProbeKernel(
        name="sign_extend",
        description="Sign extension (SGXT) patterns",
        target_opcodes=["SGXT"],
        source=r"""
__device__ __noinline__ int sign_extend_8(int val) {
    // Sign extend from 8-bit -> SGXT
    return (int)(signed char)(val & 0xFF);
}

__device__ __noinline__ int sign_extend_16(int val) {
    // Sign extend from 16-bit -> SGXT
    return (int)(short)(val & 0xFFFF);
}

__global__ void sign_extend(int* out, const int* in, int n) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < n) {
        int val = in[gid];
        int s8 = sign_extend_8(val);
        int s16 = sign_extend_16(val);
        out[gid] = s8 + s16;
    }
}
""",
        compile_flags=[],
    ))

    # -----------------------------------------------------------------------
    # T1-16. CREDUX (coupled reduction vector->uniform)
    # -----------------------------------------------------------------------
    kernels.append(CUDAProbeKernel(
        name="coupled_reduction",
        description="Coupled reduction vector->uniform (CREDUX)",
        target_opcodes=["CREDUX", "REDUX"],
        source=r"""
__global__ void coupled_reduction(int* out, const int* in, int n) {
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;
    int lane = tid & 31;

    int val = (gid < n) ? in[gid] : 0;

    // Warp-level reduction using __reduce_add_sync -> REDUX/CREDUX
    int warp_sum = __reduce_add_sync(0xFFFFFFFF, val);
    int warp_min = __reduce_min_sync(0xFFFFFFFF, val);
    int warp_max = __reduce_max_sync(0xFFFFFFFF, val);
    unsigned warp_and = __reduce_and_sync(0xFFFFFFFF, (unsigned)val);
    unsigned warp_or  = __reduce_or_sync(0xFFFFFFFF, (unsigned)val);

    if (lane == 0 && gid < n) {
        out[blockIdx.x * 5 + 0] = warp_sum;
        out[blockIdx.x * 5 + 1] = warp_min;
        out[blockIdx.x * 5 + 2] = warp_max;
        out[blockIdx.x * 5 + 3] = (int)warp_and;
        out[blockIdx.x * 5 + 4] = (int)warp_or;
    }
}
""",
        compile_flags=[],
    ))

    # -----------------------------------------------------------------------
    # T1-17. NOP explicit and YIELD / NANOSLEEP
    # -----------------------------------------------------------------------
    kernels.append(CUDAProbeKernel(
        name="nop_yield",
        description="NOP, YIELD, NANOSLEEP instructions",
        target_opcodes=["NOP", "YIELD", "NANOSLEEP"],
        source=r"""
__global__ void nop_yield(int* out, int delay_ns, int n) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;

    // NANOSLEEP
    __nanosleep(delay_ns);

    // NOP via asm
    asm volatile("nanosleep.u32 100;");

    if (gid < n) {
        out[gid] = gid;
    }
}
""",
        compile_flags=[],
    ))

    # -----------------------------------------------------------------------
    # T1-18. MATCH instruction (warp match)
    # -----------------------------------------------------------------------
    kernels.append(CUDAProbeKernel(
        name="warp_match",
        description="Warp match register values (MATCH)",
        target_opcodes=["MATCH"],
        source=r"""
__global__ void warp_match(unsigned int* out, int n) {
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;

    // Each thread has a value, find which threads have matching values
    unsigned int my_val = tid / 4;  // Groups of 4 threads have same value

    // __match_any_sync: returns mask of threads with same value -> MATCH
    unsigned int match_mask = __match_any_sync(0xFFFFFFFF, my_val);

    // __match_all_sync: returns mask if ALL threads have same value
    int pred;
    unsigned int all_mask = __match_all_sync(0xFFFFFFFF, my_val, &pred);

    if (gid < n) {
        out[gid] = match_mask ^ all_mask ^ (pred ? 0xFFFF : 0);
    }
}
""",
        compile_flags=[],
    ))

    # -----------------------------------------------------------------------
    # T1-19. FP swizzle add (FSWZADD)
    # -----------------------------------------------------------------------
    kernels.append(CUDAProbeKernel(
        name="fp_swizzle_add",
        description="FP swizzle add (FSWZADD) for warp-level butterfly",
        target_opcodes=["FSWZADD"],
        source=r"""
__global__ void fp_swizzle_add(float* out, const float* in, int n) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= n) return;

    float val = in[gid];

    // Butterfly pattern using shuffle that compiler may optimize to FSWZADD
    float sum = val;
    sum += __shfl_xor_sync(0xFFFFFFFF, val, 1);
    sum += __shfl_xor_sync(0xFFFFFFFF, val, 2);
    sum += __shfl_xor_sync(0xFFFFFFFF, val, 4);
    sum += __shfl_xor_sync(0xFFFFFFFF, val, 8);
    sum += __shfl_xor_sync(0xFFFFFFFF, val, 16);

    out[gid] = sum;
}
""",
        compile_flags=[],
    ))

    # -----------------------------------------------------------------------
    # T1-20. R2P (register to predicate) and FCHK (float range check)
    # -----------------------------------------------------------------------
    kernels.append(CUDAProbeKernel(
        name="r2p_fchk",
        description="Register to predicate (R2P) and float range check (FCHK)",
        target_opcodes=["R2P", "FCHK"],
        source=r"""
#include <math.h>

__global__ void r2p_fchk(int* out, const float* in, int n) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= n) return;

    float val = in[gid];

    // Range checks that may emit FCHK
    int is_finite = __finitef(val) ? 1 : 0;
    int is_inf = __isinff(val) ? 2 : 0;
    int is_nan = __isnanf(val) ? 4 : 0;
    int sign = (val < 0.0f) ? 16 : 0;

    // Check if val is a normal number (not zero, not denorm, not inf, not nan)
    unsigned int bits = __float_as_uint(val);
    unsigned int exp_bits = (bits >> 23) & 0xFF;
    int is_normal = (exp_bits != 0 && exp_bits != 0xFF) ? 8 : 0;

    // Force R2P: convert integer bits to predicate register
    // The compiler may use R2P when branching on packed predicate bits
    int flags = is_finite | is_inf | is_nan | is_normal | sign;
    int result = 0;

    // Use inline asm to force R2P pattern:
    // move packed predicates from register to predicate register
    asm volatile(
        "{\n\t"
        " .reg .pred %%p0, %%p1, %%p2, %%p3, %%p4;\n\t"
        " setp.ne.u32 %%p0, %1, 0;\n\t"
        " setp.ne.u32 %%p1, %2, 0;\n\t"
        " setp.ne.u32 %%p2, %3, 0;\n\t"
        " setp.ne.u32 %%p3, %4, 0;\n\t"
        " selp.s32 %0, 1, 0, %%p0;\n\t"
        "}\n\t"
        : "=r"(result)
        : "r"(is_finite), "r"(is_inf), "r"(is_nan), "r"(is_normal)
    );

    if (flags & 1) result += 100;
    if (flags & 2) result += 200;
    if (flags & 4) result += 300;
    if (flags & 8) result += 400;
    if (flags & 16) result += 500;

    out[gid] = result;
}
""",
        compile_flags=[],
    ))

    return kernels


def compile_and_discover(kernels: List[CUDAProbeKernel],
                          target: str,
                          nvcc_path: str = "nvcc",
                          nvdisasm_path: str = "nvdisasm",
                          callback=None) -> Dict[str, Dict]:
    """
    Compile CUDA C++ kernels and extract SASS opcodes.

    Returns dict mapping mnemonic -> opcode info.
    """
    import os
    import re
    import subprocess
    import tempfile

    opcode_map = {}
    total = len(kernels)

    for idx, kernel in enumerate(kernels):
        # Write source to temp file
        with tempfile.NamedTemporaryFile(
            suffix=".cu", mode="w", delete=False, prefix=f"sq_cuda_{kernel.name}_"
        ) as f:
            f.write(kernel.source)
            cu_path = f.name

        cubin_path = cu_path.replace(".cu", ".cubin")

        try:
            # Compile with nvcc -O3
            cmd = [
                nvcc_path, "-O3", "-cubin",
                f"-arch={target}",
                "-o", cubin_path,
                cu_path,
            ] + kernel.compile_flags

            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=120,
            )

            if result.returncode != 0:
                # Try without extra flags
                cmd = [
                    nvcc_path, "-O3", "-cubin",
                    f"-arch={target}",
                    "-o", cubin_path,
                    cu_path,
                ]
                result = subprocess.run(
                    cmd, capture_output=True, text=True, timeout=120,
                )
                if result.returncode != 0:
                    if callback:
                        callback(idx + 1, total, len(opcode_map), kernel.name, False)
                    continue

            # Disassemble
            disasm_result = subprocess.run(
                [nvdisasm_path, "-c", cubin_path],
                capture_output=True, text=True, timeout=30,
            )

            if disasm_result.returncode != 0:
                if callback:
                    callback(idx + 1, total, len(opcode_map), kernel.name, False)
                continue

            # Parse SASS instructions
            pattern = re.compile(
                r'/\*[0-9a-fA-F]+\*/\s+'
                r'(?:@!?U?P\d+\s+)?'  # optional predicate
                r'(\S+)'               # mnemonic
                r'\s+.*?;'             # operands
            )

            for match in pattern.finditer(disasm_result.stdout):
                mnemonic = match.group(1)
                if mnemonic and mnemonic not in ("NOP",) and mnemonic not in opcode_map:
                    opcode_map[mnemonic] = {
                        "bits_11_0": "0x???",  # We don't have hex from nvdisasm -c
                        "bits_11_2": "0x???",
                        "full_word_lo": "N/A",
                        "name": f"cuda:{kernel.name}",
                    }

            if callback:
                callback(idx + 1, total, len(opcode_map), kernel.name, True)

        except Exception:
            if callback:
                callback(idx + 1, total, len(opcode_map), kernel.name, False)

        finally:
            for p in [cu_path, cubin_path]:
                if os.path.exists(p):
                    try:
                        os.unlink(p)
                    except OSError:
                        pass

    return opcode_map


def compile_and_discover_with_hex(kernels: List[CUDAProbeKernel],
                                   target: str,
                                   nvcc_path: str = "nvcc",
                                   nvdisasm_path: str = "nvdisasm",
                                   callback=None) -> Dict[str, Dict]:
    """
    Compile CUDA C++ kernels and extract SASS opcodes WITH hex encoding.

    Uses the CubinBuilder infrastructure for proper opcode extraction.
    Falls back to text-only parsing if hex isn't available.
    """
    import os
    import re
    import struct
    import subprocess
    import tempfile
    from sass_probe import CubinBuilder

    opcode_map = {}
    total = len(kernels)

    try:
        builder = CubinBuilder(target=target, nvcc_path=nvcc_path,
                                nvdisasm_path=nvdisasm_path)
    except RuntimeError:
        # Fall back to simple discovery
        return compile_and_discover(kernels, target, nvcc_path, nvdisasm_path, callback)

    for idx, kernel in enumerate(kernels):
        with tempfile.NamedTemporaryFile(
            suffix=".cu", mode="w", delete=False, prefix=f"sq_cuda_{kernel.name}_"
        ) as f:
            f.write(kernel.source)
            cu_path = f.name

        cubin_path = cu_path.replace(".cu", ".cubin")

        try:
            cmd = [
                nvcc_path, "-O3", "-cubin",
                f"-arch={target}",
                "-o", cubin_path,
                cu_path,
            ] + kernel.compile_flags

            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=120,
            )

            if result.returncode != 0:
                # Retry without extra flags
                cmd = [
                    nvcc_path, "-O3", "-cubin",
                    f"-arch={target}",
                    "-o", cubin_path,
                    cu_path,
                ]
                result = subprocess.run(
                    cmd, capture_output=True, text=True, timeout=120,
                )
                if result.returncode != 0:
                    if callback:
                        callback(idx + 1, total, len(opcode_map), kernel.name, False)
                    continue

            with open(cubin_path, "rb") as f:
                cubin_data = f.read()

            # Use nvdisasm -c as primary (covers ALL text sections / kernels)
            # then enhance with hex encoding from ELF parsing where possible
            disasm_result = subprocess.run(
                [nvdisasm_path, "-c", cubin_path],
                capture_output=True, text=True, timeout=30,
            )
            text_mnemonics = set()
            if disasm_result.returncode == 0:
                pattern = re.compile(
                    r'/\*[0-9a-fA-F]+\*/\s+'
                    r'(?:@!?U?P\d+\s+)?'
                    r'(\S+)'
                    r'\s+.*?;'
                )
                for match in pattern.finditer(disasm_result.stdout):
                    mnemonic = match.group(1)
                    if mnemonic and mnemonic not in ("NOP",):
                        text_mnemonics.add(mnemonic)

            # Try ELF parsing for hex opcode info (first text section only)
            hex_info = {}
            try:
                cubin_info = builder.parse_cubin_elf(cubin_data)
                builder.extract_instructions(cubin_info)
                disasm = builder.disassemble(cubin_data)
                builder.annotate_from_disasm(cubin_info, disasm)

                for inst in cubin_info.instructions:
                    if inst.mnemonic and inst.mnemonic not in ("NOP",):
                        if inst.mnemonic not in hex_info:
                            hex_info[inst.mnemonic] = {
                                "bits_11_0": f"0x{inst.opcode_12bit:03x}",
                                "bits_11_2": f"0x{inst.opcode_10bit:03x}",
                                "full_word_lo": f"0x{inst.instruction_word:016x}",
                            }
            except (ValueError, struct.error):
                pass

            # Merge: all text mnemonics + hex info where available
            for mnemonic in text_mnemonics:
                if mnemonic not in opcode_map:
                    if mnemonic in hex_info:
                        info = hex_info[mnemonic]
                        info["name"] = f"cuda:{kernel.name}"
                        opcode_map[mnemonic] = info
                    else:
                        opcode_map[mnemonic] = {
                            "bits_11_0": "0x???",
                            "bits_11_2": "0x???",
                            "full_word_lo": "N/A",
                            "name": f"cuda:{kernel.name}",
                        }

            if callback:
                callback(idx + 1, total, len(opcode_map), kernel.name, True)

        except Exception:
            if callback:
                callback(idx + 1, total, len(opcode_map), kernel.name, False)
        finally:
            for p in [cu_path, cubin_path]:
                if os.path.exists(p):
                    try:
                        os.unlink(p)
                    except OSError:
                        pass

    return opcode_map
