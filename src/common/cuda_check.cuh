#pragma once
// Verificação de erros CUDA. CUDA_CHECK envolve toda chamada da API;
// CUDA_CHECK_KERNEL vem logo após cada lançamento de kernel.

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                                  \
    do {                                                                                  \
        cudaError_t err__ = (call);                                                       \
        if (err__ != cudaSuccess) {                                                       \
            std::fprintf(stderr, "CUDA error %s at %s:%d: %s\n", cudaGetErrorName(err__), \
                         __FILE__, __LINE__, cudaGetErrorString(err__));                  \
            std::exit(1);                                                                 \
        }                                                                                 \
    } while (0)

// Em modo debug (FPG_SYNC_CHECK definido) sincroniza para pegar erros de execução
// no ponto exato; caso contrário só verifica erros de lançamento.
#ifdef FPG_SYNC_CHECK
#define CUDA_CHECK_KERNEL()                       \
    do {                                          \
        CUDA_CHECK(cudaGetLastError());           \
        CUDA_CHECK(cudaDeviceSynchronize());      \
    } while (0)
#else
#define CUDA_CHECK_KERNEL() CUDA_CHECK(cudaGetLastError())
#endif
