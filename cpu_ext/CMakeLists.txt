# cpu_kern: threaded SIMD kernels for MLX's CPU device, built against the running env's MLX (cpu_ext/build.sh).
cmake_minimum_required(VERSION 3.27)
project(cpu_kern LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(Python 3.10 COMPONENTS Interpreter Development.Module REQUIRED)
execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE nanobind_ROOT)
find_package(nanobind CONFIG REQUIRED)
execute_process(
  COMMAND "${Python_EXECUTABLE}" -m mlx --cmake-dir
  OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE MLX_ROOT)
find_package(MLX CONFIG REQUIRED)
find_package(OpenMP REQUIRED)

# NB_DOMAIN mlx: share the mx.array type with mlx's own nanobind module.
nanobind_add_module(cpu_kern NB_STATIC STABLE_ABI LTO NOMINSIZE NB_DOMAIN mlx
                    ${CMAKE_CURRENT_LIST_DIR}/cpu_kern.cpp)
# No FMA contraction: the GatedDeltaNet update must round a multiply then an add, as MLX's elementwise ops do.
target_compile_options(cpu_kern PRIVATE -O3 -march=x86-64-v4 -mavx512bf16 -mavx512vbmi -ffp-contract=off)
target_link_libraries(cpu_kern PRIVATE mlx OpenMP::OpenMP_CXX)
