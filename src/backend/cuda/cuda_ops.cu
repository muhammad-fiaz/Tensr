/**
 * @file cuda_ops.cu
 * @brief CUDA GPU-accelerated tensor operations
 * @author Muhammad Fiaz
 * 
 * Implements GPU kernels for high-performance tensor operations using CUDA.
 * Provides parallel implementations of arithmetic operations for NVIDIA GPUs.
 */

#include "tensr/tensr.h"
#include <cuda_runtime.h>

/**
 * @brief CUDA kernel for element-wise addition
 * @param a First input array
 * @param b Second input array
 * @param c Output array
 * @param n Number of elements
 * 
 * GPU kernel that performs parallel element-wise addition: c[i] = a[i] + b[i]
 */
__global__ void add_kernel(float* a, float* b, float* c, size_t n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

/**
 * @brief CUDA kernel for element-wise multiplication
 * @param a First input array
 * @param b Second input array
 * @param c Output array
 * @param n Number of elements
 * 
 * GPU kernel that performs parallel element-wise multiplication: c[i] = a[i] * b[i]
 */
__global__ void mul_kernel(float* a, float* b, float* c, size_t n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] * b[idx];
    }
}

extern "C" {

/**
 * @brief Launch CUDA addition kernel
 * @param a First input array on GPU
 * @param b Second input array on GPU
 * @param c Output array on GPU
 * @param n Number of elements
 * 
 * Host function that launches the GPU addition kernel with appropriate
 * grid and block dimensions.
 */
void cuda_add(float* a, float* b, float* c, size_t n) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    add_kernel<<<blocks, threads>>>(a, b, c, n);
    cudaDeviceSynchronize();
}

/**
 * @brief Launch CUDA multiplication kernel
 * @param a First input array on GPU
 * @param b Second input array on GPU
 * @param c Output array on GPU
 * @param n Number of elements
 * 
 * Host function that launches the GPU multiplication kernel with appropriate
 * grid and block dimensions.
 */
void cuda_mul(float* a, float* b, float* c, size_t n) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    mul_kernel<<<blocks, threads>>>(a, b, c, n);
    cudaDeviceSynchronize();
}

}
