/**
 * @file array.c
 * @brief Array creation functions for easy tensor initialization from C arrays
 * @author Muhammad Fiaz
 * 
 * Provides convenient functions for creating tensors from existing C arrays,
 * enabling easy initialization with static data.
 */

#include "tensr/tensr_array.h"
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>

/**
 * @brief Create a 1D tensor from variable arguments
 * @param dtype Data type of tensor elements
 * @param device Device to allocate tensor on
 * @param n Number of elements
 * @param ... Variable number of values
 * @return Pointer to newly created 1D tensor
 * 
 * Creates a 1D tensor from a variable number of arguments.
 * 
 * Example:
 *   Tensor* t = tensr_array_1d(TENSR_FLOAT32, TENSR_CPU, 3, 1.0, 2.0, 3.0);
 */
Tensor* tensr_array_1d(TensrDType dtype, TensrDevice device, size_t n, ...) {
    size_t shape[1] = {n};
    Tensor* t = tensr_create(shape, 1, dtype, device);
    if (!t) return NULL;

    va_list args;
    va_start(args, n);

    if (dtype == TENSR_FLOAT32) {
        float* data = (float*)t->data;
        for (size_t i = 0; i < n; i++) {
            data[i] = (float)va_arg(args, double);
        }
    } else if (dtype == TENSR_FLOAT64) {
        double* data = (double*)t->data;
        for (size_t i = 0; i < n; i++) {
            data[i] = va_arg(args, double);
        }
    } else if (dtype == TENSR_INT32) {
        int32_t* data = (int32_t*)t->data;
        for (size_t i = 0; i < n; i++) {
            data[i] = va_arg(args, int32_t);
        }
    }

    va_end(args);
    return t;
}

/**
 * @brief Create a 2D tensor from array data
 * @param dtype Data type of tensor elements
 * @param device Device to allocate tensor on
 * @param rows Number of rows
 * @param cols Number of columns
 * @param data Pointer to data array
 * @return Pointer to newly created 2D tensor
 * 
 * Creates a 2D tensor (matrix) from existing array data.
 * 
 * Example:
 *   float data[] = {1, 2, 3, 4, 5, 6};
 *   Tensor* t = tensr_array_2d(TENSR_FLOAT32, TENSR_CPU, 2, 3, data);
 */
Tensor* tensr_array_2d(TensrDType dtype, TensrDevice device, size_t rows, size_t cols, void* data) {
    size_t shape[2] = {rows, cols};
    return tensr_from_array(shape, 2, dtype, device, data);
}

/**
 * @brief Create a tensor from existing array data
 * @param shape Array of dimension sizes
 * @param ndim Number of dimensions
 * @param dtype Data type of tensor elements
 * @param device Device to allocate tensor on
 * @param data Pointer to data array
 * @return Pointer to newly created tensor
 * 
 * Creates a tensor from existing data array. The data is copied into the tensor.
 * This is the most flexible array creation function.
 * 
 * Example:
 *   float data[] = {1, 2, 3, 4, 5, 6};
 *   size_t shape[] = {2, 3};
 *   Tensor* t = tensr_from_array(shape, 2, TENSR_FLOAT32, TENSR_CPU, data);
 */
Tensor* tensr_from_array(size_t* shape, size_t ndim, TensrDType dtype, TensrDevice device, void* data) {
    Tensor* t = tensr_create(shape, ndim, dtype, device);
    if (!t) return NULL;

    size_t bytes = t->size * tensr_dtype_size(dtype);
    memcpy(t->data, data, bytes);

    return t;
}
