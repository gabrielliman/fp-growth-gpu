// Microbenchmark de limites do hardware usados na análise do artigo:
// banda D2D, H2D (paginável e pinned) e radix sort de chaves de 32/64 bits.
// uso: microbench <n_transacoes> <n_itens>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include <thrust/device_vector.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>

#include "common/cuda_check.cuh"

namespace {

template <typename F>
float time_ms(F f, int reps) {
    cudaEvent_t a, b;
    CUDA_CHECK(cudaEventCreate(&a));
    CUDA_CHECK(cudaEventCreate(&b));
    f();  // aquecimento
    CUDA_CHECK(cudaEventRecord(a));
    for (int i = 0; i < reps; ++i) f();
    CUDA_CHECK(cudaEventRecord(b));
    CUDA_CHECK(cudaEventSynchronize(b));
    float ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms, a, b));
    return ms / reps;
}

__global__ void scramble(unsigned long long* k, size_t n) {
    size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x;
    if (i < n) k[i] = (i * 0x9E3779B97F4A7C15ull) ^ (i >> 7);
}

}  // namespace

int main(int argc, char** argv) {
    const size_t n_tx = argc > 1 ? std::strtoull(argv[1], nullptr, 10) : 1000000;
    const size_t n_items = argc > 2 ? std::strtoull(argv[2], nullptr, 10) : 8000000;
    CUDA_CHECK(cudaFree(nullptr));

    // Banda D2D (lê + escreve), 256 MB.
    const size_t big = size_t(256) << 20;
    char *x, *y;
    CUDA_CHECK(cudaMalloc(&x, big));
    CUDA_CHECK(cudaMalloc(&y, big));
    float d2d = time_ms([&] { CUDA_CHECK(cudaMemcpy(y, x, big, cudaMemcpyDeviceToDevice)); }, 20);
    std::printf("micro,d2d_GBps,%.1f\n", 2.0 * big / d2d / 1e6);

    // H2D com os bytes de entrada do dataset (itens + offsets).
    const size_t bytes = (n_items + n_tx + 1) * 4;
    std::vector<char> pageable(bytes, 1);
    char* pinned;
    CUDA_CHECK(cudaMallocHost(&pinned, bytes));
    float h2d_p = time_ms([&] { CUDA_CHECK(cudaMemcpy(x, pageable.data(), bytes, cudaMemcpyHostToDevice)); }, 10);
    float h2d_n = time_ms([&] { CUDA_CHECK(cudaMemcpy(x, pinned, bytes, cudaMemcpyHostToDevice)); }, 10);
    std::printf("micro,h2d_pageable_ms,%.4f\nmicro,h2d_pageable_GBps,%.2f\n", h2d_p, bytes / h2d_p / 1e6);
    std::printf("micro,h2d_pinned_ms,%.4f\nmicro,h2d_pinned_GBps,%.2f\n", h2d_n, bytes / h2d_n / 1e6);

    // Radix sort de n_tx chaves (32 e 64 bits) com índice: limite para uma
    // ordenação lexicográfica cujas chaves coubessem numa palavra.
    thrust::device_vector<unsigned long long> k64(n_tx), src(n_tx);
    thrust::device_vector<unsigned int> k32(n_tx), v(n_tx);
    scramble<<<(n_tx + 255) / 256, 256>>>(thrust::raw_pointer_cast(src.data()), n_tx);
    CUDA_CHECK_KERNEL();
    float s32 = time_ms([&] {
        thrust::transform(src.begin(), src.end(), k32.begin(), [] __device__(unsigned long long a) { return unsigned(a); });
        thrust::sequence(v.begin(), v.end());
        thrust::sort_by_key(k32.begin(), k32.end(), v.begin());
    }, 10);
    float s64 = time_ms([&] {
        thrust::copy(src.begin(), src.end(), k64.begin());
        thrust::sequence(v.begin(), v.end());
        thrust::sort_by_key(k64.begin(), k64.end(), v.begin());
    }, 10);
    std::printf("micro,radix32_pairs_ms,%.4f\nmicro,radix64_pairs_ms,%.4f\n", s32, s64);
    CUDA_CHECK(cudaFreeHost(pinned));
    return 0;
}
