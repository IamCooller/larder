# Native expert agent for MOE_NATIVE=1 (stream_fast._Native). Build with the
# venv the engine runs in, so the extension links that venv's libmlx, with the
# nanobind mlx itself was built with (2.15.0 for mlx 0.32.2: another version
# does not share the mx.array type):
#   cmake -S native -B native/build -DCMAKE_BUILD_TYPE=Release \
#         -DPython_EXECUTABLE=$VENV/bin/python && cmake --build native/build -j
# Follows mlx/examples/extensions/CMakeLists.txt (MLX v0.32.2).
cmake_minimum_required(VERSION 3.27)
project(moe_native LANGUAGES CXX)

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

# NB_DOMAIN mlx: share the mx.array type with mlx's own nanobind module.
nanobind_add_module(moe_native NB_STATIC STABLE_ABI LTO NOMINSIZE NB_DOMAIN mlx
                    ${CMAKE_CURRENT_LIST_DIR}/moe_native.cpp)
target_link_libraries(moe_native PRIVATE mlx)
