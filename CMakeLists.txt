cmake_minimum_required(VERSION 3.24)

# nvcc 12.4 não aceita g++ >= 14 como compilador host; usamos g++-13 por padrão.
if(NOT DEFINED CMAKE_CUDA_HOST_COMPILER)
  find_program(GXX13 g++-13)
  if(GXX13)
    set(CMAKE_CUDA_HOST_COMPILER ${GXX13})
  endif()
endif()
if(NOT DEFINED CMAKE_CXX_COMPILER AND DEFINED CMAKE_CUDA_HOST_COMPILER)
  set(CMAKE_CXX_COMPILER ${CMAKE_CUDA_HOST_COMPILER})
endif()
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES native)
endif()
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

project(fp_growth_gpu LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_CXX_FLAGS_RELEASE "-O3")
set(CMAKE_CUDA_FLAGS_RELEASE "-O3")
add_compile_options(
  $<$<COMPILE_LANGUAGE:CXX>:-Wall>
  $<$<COMPILE_LANGUAGE:CXX>:-Wextra>
  $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra>
  $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
  $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>
)

find_package(CUDAToolkit REQUIRED)

add_library(fpg_common STATIC
  src/common/fimi.cpp
  src/common/output.cpp
  src/common/args.cpp)
target_include_directories(fpg_common PUBLIC src)

add_library(fpg_seq_lib STATIC src/seq/fpgrowth_seq.cpp)
target_link_libraries(fpg_seq_lib PUBLIC fpg_common)

add_library(fpg_gpu_lib STATIC
  src/gpu/preprocess.cu
  src/gpu/build_tree.cu
  src/gpu/mine.cu)
target_link_libraries(fpg_gpu_lib PUBLIC fpg_common CUDA::cudart)
set_target_properties(fpg_gpu_lib PROPERTIES CUDA_SEPARABLE_COMPILATION OFF)

add_executable(fpg_seq src/seq/main_seq.cpp)
target_link_libraries(fpg_seq PRIVATE fpg_seq_lib)

add_executable(fpg_gpu src/gpu/main_gpu.cu)
target_link_libraries(fpg_gpu PRIVATE fpg_gpu_lib fpg_seq_lib)

add_executable(tests tests/tests.cu)
target_link_libraries(tests PRIVATE fpg_gpu_lib fpg_seq_lib)
target_compile_definitions(tests PRIVATE TESTS_DATA_DIR="${CMAKE_SOURCE_DIR}/tests/data")

add_executable(microbench tests/microbench.cu)
target_link_libraries(microbench PRIVATE fpg_common CUDA::cudart)

# Baselines em GPU para comparação: Apriori (estilo GPApriori) e Eclat
# (estilo Frontier Expansion), com bitsets verticais.
add_library(fpg_vertical_lib STATIC src/baselines/vertical.cu)
target_link_libraries(fpg_vertical_lib PUBLIC fpg_common CUDA::cudart)

add_executable(fpg_apriori_gpu src/baselines/main_vertical.cu)
target_compile_definitions(fpg_apriori_gpu PRIVATE FPG_VERTICAL_ALGO=0)
target_link_libraries(fpg_apriori_gpu PRIVATE fpg_vertical_lib fpg_seq_lib)

add_executable(fpg_eclat_gpu src/baselines/main_vertical.cu)
target_compile_definitions(fpg_eclat_gpu PRIVATE FPG_VERTICAL_ALGO=1)
target_link_libraries(fpg_eclat_gpu PRIVATE fpg_vertical_lib fpg_seq_lib)

target_link_libraries(tests PRIVATE fpg_vertical_lib)
