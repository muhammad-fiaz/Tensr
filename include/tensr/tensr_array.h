/**
 * @file tensr_array.h
 * @brief Simple array creation macros for easy tensor initialization
 * @author Muhammad Fiaz
 * @version 0.0.0
 * @license Apache-2.0
 * 
 * Provides convenient macros for creating tensors from static arrays,
 * similar to numpy.array() functionality in Python.
 */

#ifndef TENSR_ARRAY_H
#define TENSR_ARRAY_H

#include "tensr.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Create a 1D tensor from array values
 * @param dtype Data type (TENSR_FLOAT32, TENSR_INT32, etc.)
 * @param device Device type (TENSR_CPU, TENSR_CUDA, etc.)
 * @param ... Variable number of values
 * 
 * Example:
 *   Tensor* t = tensr_array_1d(TENSR_FLOAT32, TENSR_CPU, 3, 1.0f, 2.0f, 3.0f);
 */
Tensor* tensr_array_1d(TensrDType dtype, TensrDevice device, size_t n, ...);

/**
 * @brief Create a 2D tensor from array values
 * @param dtype Data type
 * @param device Device type
 * @param rows Number of rows
 * @param cols Number of columns
 * @param data Pointer to data array
 * 
 * Example:
 *   float data[] = {1, 2, 3, 4, 5, 6};
 *   Tensor* t = tensr_array_2d(TENSR_FLOAT32, TENSR_CPU, 2, 3, data);
 */
Tensor* tensr_array_2d(TensrDType dtype, TensrDevice device, size_t rows, size_t cols, void* data);

/**
 * @brief Create a tensor from existing data array
 * @param shape Array of dimension sizes
 * @param ndim Number of dimensions
 * @param dtype Data type
 * @param device Device type
 * @param data Pointer to data array
 * @return Pointer to newly created tensor
 * 
 * Creates a tensor from existing data. The data is copied into the tensor.
 * Similar to numpy.array() in Python.
 * 
 * Example:
 *   float data[] = {1, 2, 3, 4, 5, 6};
 *   size_t shape[] = {2, 3};
 *   Tensor* t = tensr_from_array(shape, 2, TENSR_FLOAT32, TENSR_CPU, data);
 */
Tensor* tensr_from_array(size_t* shape, size_t ndim, TensrDType dtype, TensrDevice device, void* data);

#ifdef __cplusplus
}
#endif

#endif /* TENSR_ARRAY_H */
