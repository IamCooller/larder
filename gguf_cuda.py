"""gguf_experts' kernels on CUDA: the backend prelude of their shared source."""
import mlx.core as mx

# NVRTC contracts a * b + c into one FMA by default, so the dequant's products and sums are rounded alone.
PRELUDE = r"""
#define GPTR(T) const T*
#define LOCAL
#define FN __device__ __forceinline__
#define CFN __host__ __device__ constexpr
#define H2F(u) __half2float(__ushort_as_half((unsigned short)(u)))
#define SHFL(v, m) __shfl_xor_sync(0xffffffffu, v, m)
#define POPC(v) __popc(v)
#define EXP(v) expf(v)
#define MUL(a, b) __fmul_rn(a, b)
#define ADD(a, b) __fadd_rn(a, b)
#define SUB(a, b) __fsub_rn(a, b)
#define U4(a, b, c, d) make_uint4(a, b, c, d)
#define TGMEM __shared__
#define BARRIER __syncthreads()
"""

BODY = r"""
const uint32_t _tid = threadIdx.x;
const uint3 _bid = blockIdx;
const uint32_t _gid = blockIdx.x * blockDim.x + threadIdx.x;
"""


def kernel(name, ins, outs, source, header):
    return mx.fast.cuda_kernel(name=name, input_names=ins, output_names=outs, source=BODY + source,
                               header=PRELUDE + header)
