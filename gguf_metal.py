"""gguf_experts' kernels on Metal: the backend prelude of their shared source."""
import mlx.core as mx

# MLX compiles custom kernels in its "safe" math mode, whose products and sums are not contracted into fused
# multiply-adds, with the dequant matching the reference bit for bit.
PRELUDE = r"""
#define GPTR(T) const device T*
#define LOCAL thread
#define FN static inline
#define CFN static inline constexpr
#define H2F(u) float(as_type<half>(ushort(u)))
#define U2F(u) as_type<float>(uint(u))
#define F2U(f) as_type<uint>(float(f))
#define SHFL(v, m) simd_shuffle_xor(v, ushort(m))
#define POPC(v) popcount(v)
#define EXP(v) metal::exp(v)
#define MUL(a, b) ((a) * (b))
#define ADD(a, b) ((a) + (b))
#define SUB(a, b) ((a) - (b))
#define U4(a, b, c, d) uint4(a, b, c, d)
#define TGMEM threadgroup
#define BARRIER threadgroup_barrier(mem_flags::mem_threadgroup)
"""

BODY = r"""
const uint _tid = thread_index_in_threadgroup;
const uint3 _bid = threadgroup_position_in_grid;
const uint _gid = thread_position_in_grid.x;
"""


def kernel(name, ins, outs, source, header):
    return mx.fast.metal_kernel(name=name, input_names=ins, output_names=outs, source=BODY + source,
                                header=PRELUDE + header)
