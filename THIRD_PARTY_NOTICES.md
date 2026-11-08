# Third-party notices

Parts of this project are ports or copies of other projects' code, under the MIT, BSD-3-Clause and
Apache-2.0/MPL-2.0 licences. Each section names the files that carry the derived code, its version
and copyright, and where the full licence text sits.

| project | copyright | derived files |
|---|---|---|
| ddalcu/sushi, `src/expert_exl3_kernels.zig` at commit 1ea9492 | Copyright (c) 2026 Theinruj Toranavikrai and David Dalcu | `exl3_experts.py` (the pair GEMV, the fused down GEMV, the reduce and the codebook helpers in Metal), `exl3_cuda.py` (their CUDA twins) |
| beamivalice/sushi, the upstream of ddalcu/sushi | Copyright (c) 2026 Theinruj Toranavikrai and David Dalcu | the same code as the ddalcu/sushi row: ddalcu/sushi is a fork of it and carries the same licence text |
| turboderp-org/exllamav3 | Copyright (c) 2025 Turboderp | `exl3_experts.py`, `exl3_cuda.py` (the EXL3 format: 16x16 trellis tiles, the MCG codebook, the tensor-core permutation, the suh / svh Hadamards) |
| ggml-org/llama.cpp (ggml, gguf-py), at commit b1c902f0 | Copyright (c) 2023-2026 The ggml authors | `gguf_ref.py` (the ggml block formats and their dequant: Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q2_0, Q4_K, Q5_K, IQ2_S, IQ3_XXS, IQ4_NL, IQ4_XS, with the IQ grids and tables), `gguf_experts.py` (the same decode in Metal and CUDA kernels), `tools/gguf_repack.py` (the GGUF container layout). The CPU kernel that ships, `cpu_ext/cpu_kern.cpp`, holds no ggml code |
| ml-explore/mlx | Copyright (c) 2023 Apple Inc. | `layer_glue.py` (rms norms, the column reduce, the sigmoid and silu values and the operation order of the decode-row kernels), `fused_router.py` (MLX's dtypes, operation order and argpartition tie order in the routing kernel), `route_cuda.py` (MLX's CUDA row reduce copied lane for lane, and the CUDA sort's tie order, in the CUDA routing kernel), `draftstep.py` (`qdot` from `kernels/quantized.h`, verbatim), `stream_fast.py` (the two-pass vector attention kernels of `sdpa_vector.h`, run over the picked keys), `qsa_tiles.py` (key tiles cut so that MLX's GEMM sums them in its own split-K order), `chunkattn.py` (MLX's softmax switch point and GEMM key alignment), `cpu_ext/cpu_kern.cpp` (MLX's affine-quantized matmul arithmetic on the CPU). `exl3_experts.py` patches the installed MLX's gather GEMM source at run time and ships none of it |
| Blaizzy/mlx-vlm, `mlx_vlm/models/qwen3_vl/vision.py` | Copyright (c) 2025 Prince Canuma | `vision.py` (the Qwen3-VL vision tower, `_Tower` and the pieces under it; the licence is also inline in the file) |
| markedjs/marked, 18.1.2 | Copyright (c) 2018+, MarkedJS (https://github.com/markedjs/); Copyright (c) 2011-2018, Christopher Jeffrey | `web/vendor/marked-18.1.2/marked.umd.js` (markdown to HTML in the chat page) |
| cure53/DOMPurify, 3.4.17 | Copyright (c) Cure53 and other contributors | `web/vendor/dompurify-3.4.17/purify.min.js` (sanitises rendered markdown) |
| highlightjs/highlight.js, 11.12.0 | Copyright (c) 2006, Ivan Sagalaev | `web/vendor/highlight.js-11.12.0/highlight.min.js` (code block highlighting) |

## ddalcu/sushi

MIT License

Copyright (c) 2026 Theinruj Toranavikrai
Copyright (c) 2026 David Dalcu

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

The upstream LICENSE adds that these MIT terms cover sushi's own code, including code it inherits from
mlx-serve (ddalcu/mlx-serve), and that third-party parts of that distribution keep their own licences.
sushi's NOTICE lists turboderp/exllamav3 as the source of the EXL3 format used in
`src/expert_exl3_kernels.zig` and says that no exllamav3 code is shipped there.

## beamivalice/sushi

The upstream of ddalcu/sushi. Its LICENSE is the same text as ddalcu/sushi's above, word for word:
MIT License, Copyright (c) 2026 Theinruj Toranavikrai, Copyright (c) 2026 David Dalcu.

## turboderp-org/exllamav3

MIT License

Copyright (c) 2025 Turboderp

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## ggml-org/llama.cpp

MIT License

Copyright (c) 2023-2026 The ggml authors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## ml-explore/mlx

MIT License

Copyright (c) 2023 Apple Inc.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## Blaizzy/mlx-vlm

MIT License

Copyright (c) 2025 Prince Canuma

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## markedjs/marked

Version 18.1.2, MIT License.

Copyright (c) 2018+, MarkedJS (https://github.com/markedjs/)
Copyright (c) 2011-2018, Christopher Jeffrey (https://github.com/chjj/)

The complete MIT licence text is in `web/vendor/marked-18.1.2/LICENSE`, next to the vendored
`marked.umd.js`.

## cure53/DOMPurify

Version 3.4.17, Apache License 2.0 or Mozilla Public License 2.0, at your option.

Copyright (c) Cure53 and other contributors.

The complete licence texts are in `web/vendor/dompurify-3.4.17/LICENSE` (Apache 2.0) and
`web/vendor/dompurify-3.4.17/LICENSE-MPL` (MPL 2.0), next to the vendored `purify.min.js`.

## highlightjs/highlight.js

Version 11.12.0, BSD 3-Clause License.

Copyright (c) 2006, Ivan Sagalaev.

The complete BSD 3-Clause licence text is in `web/vendor/highlight.js-11.12.0/LICENSE`, next to
the vendored `highlight.min.js`.
