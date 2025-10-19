/**
 * @file tensor.c
 * @brief Core tensor implementation with creation and manipulation functions
 * @author Muhammad Fiaz
 * 
 * This file implements the fundamental tensor operations including creation,
 * memory management, shape manipulation, and utility functions.
 */

#include "tensr/tensr.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

/**
 * @brief Get the size in bytes of a data type
 * @param dtype The data type enum
 * @return Size in bytes of the data type
 * 
 * Returns the memory size required for one element of the specified data type.
 * Used internally for memory allocation calculations.
 */
size_t tensr_dtype_size(TensrDType dtype) {
    switch (dtype) {
        case TENSR_FLOAT32: return sizeof(float);
        case TENSR_FLOAT64: return sizeof(double);
        case TENSR_INT32: return sizeof(int32_t);
        case TENSR_INT64: return sizeof(int64_t);
        case TENSR_UINT8: return sizeof(uint8_t);
        case TENSR_BOOL: return sizeof(bool);
        default: return 0;
    }
}

/**
 * @brief Get the string name of a data type
 * @param dtype The data type enum
 * @return String representation of the data type
 * 
 * Returns a human-readable name for the data type, useful for debugging
 * and printing tensor information.
 */
const char* tensr_dtype_name(TensrDType dtype) {
    switch (dtype) {
        case TENSR_FLOAT32: return "float32";
        case TENSR_FLOAT64: return "float64";
        case TENSR_INT32: return "int32";
        case TENSR_INT64: return "int64";
        case TENSR_UINT8: return "uint8";
        case TENSR_BOOL: return "bool";
        default: return "unknown";
    }
}

/**
 * @brief Get the string name of a device type
 * @param device The device type enum
 * @return String representation of the device type
 * 
 * Returns a human-readable name for the device type (CPU, CUDA, etc.).
 */
const char* tensr_device_name(TensrDevice device) {
    switch (device) {
        case TENSR_CPU: return "CPU";
        case TENSR_CUDA: return "CUDA";
        case TENSR_XPU: return "XPU";
        case TENSR_NPU: return "NPU";
        case TENSR_TPU: return "TPU";
        default: return "unknown";
    }
}

/**
 * @brief Compute total number of elements from shape
 * @param shape Array of dimension sizes
 * @param ndim Number of dimensions
 * @return Total number of elements
 * 
 * Calculates the product of all dimensions to get the total tensor size.
 */
static size_t compute_size(size_t* shape, size_t ndim) {
    size_t size = 1;
    for (size_t i = 0; i < ndim; i++) {
        size *= shape[i];
    }
    return size;
}

/**
 * @brief Compute strides for row-major memory layout
 * @param strides Output array for strides
 * @param shape Array of dimension sizes
 * @param ndim Number of dimensions
 * 
 * Calculates strides for efficient element access in row-major order.
 * Strides determine the number of elements to skip in memory when moving
 * along each dimension.
 */
static void compute_strides(size_t* strides, size_t* shape, size_t ndim) {
    strides[ndim - 1] = 1;
    for (int i = (int)ndim - 2; i >= 0; i--) {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
}

/**
 * @brief Create an uninitialized tensor
 * @param shape Array of dimension sizes
 * @param ndim Number of dimensions
 * @param dtype Data type of tensor elements
 * @param device Device to allocate tensor on
 * @return Pointer to newly created tensor, or NULL on failure
 * 
 * Allocates memory for a new tensor with the specified shape and type.
 * The tensor data is uninitialized. Use tensr_zeros() or tensr_ones()
 * for initialized tensors.
 * 
 * Example:
 *   size_t shape[] = {3, 4};
 *   Tensor* t = tensr_create(shape, 2, TENSR_FLOAT32, TENSR_CPU);
 */
Tensor* tensr_create(size_t* shape, size_t ndim, TensrDType dtype, TensrDevice device) {
    Tensor* t = (Tensor*)malloc(sizeof(Tensor));
    if (!t) return NULL;

    t->ndim = ndim;
    t->dtype = dtype;
    t->device = device;
    t->device_id = 0;
    t->owns_data = true;

    t->shape = (size_t*)malloc(ndim * sizeof(size_t));
    t->strides = (size_t*)malloc(ndim * sizeof(size_t));
    if (!t->shape || !t->strides) {
        free(t->shape);
        free(t->strides);
        free(t);
        return NULL;
    }

    memcpy(t->shape, shape, ndim * sizeof(size_t));
    compute_strides(t->strides, t->shape, ndim);
    t->size = compute_size(shape, ndim);

    size_t bytes = t->size * tensr_dtype_size(dtype);
    t->data = malloc(bytes);
    if (!t->data) {
        free(t->shape);
        free(t->strides);
        free(t);
        return NULL;
    }

    return t;
}

/**
 * @brief Create a tensor filled with zeros
 * @param shape Array of dimension sizes
 * @param ndim Number of dimensions
 * @param dtype Data type of tensor elements
 * @param device Device to allocate tensor on
 * @return Pointer to newly created tensor filled with zeros
 * 
 * Creates a new tensor with all elements initialized to zero.
 * 
 * Example:
 *   size_t shape[] = {3, 3};
 *   Tensor* t = tensr_zeros(shape, 2, TENSR_FLOAT32, TENSR_CPU);
 */
Tensor* tensr_zeros(size_t* shape, size_t ndim, TensrDType dtype, TensrDevice device) {
    Tensor* t = tensr_create(shape, ndim, dtype, device);
    if (t) {
        memset(t->data, 0, t->size * tensr_dtype_size(dtype));
    }
    return t;
}

/**
 * @brief Create a tensor filled with ones
 * @param shape Array of dimension sizes
 * @param ndim Number of dimensions
 * @param dtype Data type of tensor elements
 * @param device Device to allocate tensor on
 * @return Pointer to newly created tensor filled with ones
 * 
 * Creates a new tensor with all elements initialized to one.
 * 
 * Example:
 *   size_t shape[] = {2, 4};
 *   Tensor* t = tensr_ones(shape, 2, TENSR_FLOAT32, TENSR_CPU);
 */
Tensor* tensr_ones(size_t* shape, size_t ndim, TensrDType dtype, TensrDevice device) {
    Tensor* t = tensr_create(shape, ndim, dtype, device);
    if (!t) return NULL;

    if (dtype == TENSR_FLOAT32) {
        float* data = (float*)t->data;
        for (size_t i = 0; i < t->size; i++) data[i] = 1.0f;
    } else if (dtype == TENSR_FLOAT64) {
        double* data = (double*)t->data;
        for (size_t i = 0; i < t->size; i++) data[i] = 1.0;
    } else if (dtype == TENSR_INT32) {
        int32_t* data = (int32_t*)t->data;
        for (size_t i = 0; i < t->size; i++) data[i] = 1;
    } else if (dtype == TENSR_INT64) {
        int64_t* data = (int64_t*)t->data;
        for (size_t i = 0; i < t->size; i++) data[i] = 1;
    }
    return t;
}

/**
 * @brief Create a tensor filled with a specific value
 * @param shape Array of dimension sizes
 * @param ndim Number of dimensions
 * @param value Fill value for all elements
 * @param dtype Data type of tensor elements
 * @param device Device to allocate tensor on
 * @return Pointer to newly created tensor filled with the specified value
 * 
 * Creates a new tensor with all elements initialized to the given value.
 * 
 * Example:
 *   size_t shape[] = {3, 3};
 *   Tensor* t = tensr_full(shape, 2, 5.0, TENSR_FLOAT32, TENSR_CPU);
 */
Tensor* tensr_full(size_t* shape, size_t ndim, double value, TensrDType dtype, TensrDevice device) {
    Tensor* t = tensr_create(shape, ndim, dtype, device);
    if (!t) return NULL;

    if (dtype == TENSR_FLOAT32) {
        float* data = (float*)t->data;
        for (size_t i = 0; i < t->size; i++) data[i] = (float)value;
    } else if (dtype == TENSR_FLOAT64) {
        double* data = (double*)t->data;
        for (size_t i = 0; i < t->size; i++) data[i] = value;
    } else if (dtype == TENSR_INT32) {
        int32_t* data = (int32_t*)t->data;
        for (size_t i = 0; i < t->size; i++) data[i] = (int32_t)value;
    } else if (dtype == TENSR_INT64) {
        int64_t* data = (int64_t*)t->data;
        for (size_t i = 0; i < t->size; i++) data[i] = (int64_t)value;
    }
    return t;
}

/**
 * @brief Create a 1D tensor with evenly spaced values
 * @param start Start value (inclusive)
 * @param stop End value (exclusive)
 * @param step Spacing between values
 * @param dtype Data type of tensor elements
 * @param device Device to allocate tensor on
 * @return Pointer to newly created 1D tensor with evenly spaced values
 * 
 * Creates a 1D tensor with values from start to stop (exclusive) with
 * the specified step size.
 * 
 * Example:
 *   Tensor* t = tensr_arange(0.0, 10.0, 2.0, TENSR_FLOAT32, TENSR_CPU);
 */
Tensor* tensr_arange(double start, double stop, double step, TensrDType dtype, TensrDevice device) {
    size_t n = (size_t)ceil((stop - start) / step);
    size_t shape[1] = {n};
    Tensor* t = tensr_create(shape, 1, dtype, device);
    if (!t) return NULL;

    if (dtype == TENSR_FLOAT32) {
        float* data = (float*)t->data;
        for (size_t i = 0; i < n; i++) data[i] = (float)(start + i * step);
    } else if (dtype == TENSR_FLOAT64) {
        double* data = (double*)t->data;
        for (size_t i = 0; i < n; i++) data[i] = start + i * step;
    } else if (dtype == TENSR_INT32) {
        int32_t* data = (int32_t*)t->data;
        for (size_t i = 0; i < n; i++) data[i] = (int32_t)(start + i * step);
    } else if (dtype == TENSR_INT64) {
        int64_t* data = (int64_t*)t->data;
        for (size_t i = 0; i < n; i++) data[i] = (int64_t)(start + i * step);
    }
    return t;
}

/**
 * @brief Create a 1D tensor with linearly spaced values
 * @param start Start value
 * @param stop End value
 * @param num Number of values to generate
 * @param dtype Data type of tensor elements
 * @param device Device to allocate tensor on
 * @return Pointer to newly created 1D tensor with linearly spaced values
 * 
 * Creates a 1D tensor with num evenly spaced values from start to stop
 * (both inclusive).
 * 
 * Example:
 *   Tensor* t = tensr_linspace(0.0, 1.0, 5, TENSR_FLOAT32, TENSR_CPU);
 */
Tensor* tensr_linspace(double start, double stop, size_t num, TensrDType dtype, TensrDevice device) {
    size_t shape[1] = {num};
    Tensor* t = tensr_create(shape, 1, dtype, device);
    if (!t) return NULL;

    double step = (stop - start) / (num - 1);
    if (dtype == TENSR_FLOAT32) {
        float* data = (float*)t->data;
        for (size_t i = 0; i < num; i++) data[i] = (float)(start + i * step);
    } else if (dtype == TENSR_FLOAT64) {
        double* data = (double*)t->data;
        for (size_t i = 0; i < num; i++) data[i] = start + i * step;
    }
    return t;
}

/**
 * @brief Create an identity matrix
 * @param n Size of the square matrix (n x n)
 * @param dtype Data type of tensor elements
 * @param device Device to allocate tensor on
 * @return Pointer to newly created identity matrix
 * 
 * Creates a 2D identity matrix with ones on the diagonal and zeros elsewhere.
 * 
 * Example:
 *   Tensor* t = tensr_eye(3, TENSR_FLOAT32, TENSR_CPU);
 */
Tensor* tensr_eye(size_t n, TensrDType dtype, TensrDevice device) {
    size_t shape[2] = {n, n};
    Tensor* t = tensr_zeros(shape, 2, dtype, device);
    if (!t) return NULL;

    if (dtype == TENSR_FLOAT32) {
        float* data = (float*)t->data;
        for (size_t i = 0; i < n; i++) data[i * n + i] = 1.0f;
    } else if (dtype == TENSR_FLOAT64) {
        double* data = (double*)t->data;
        for (size_t i = 0; i < n; i++) data[i * n + i] = 1.0;
    } else if (dtype == TENSR_INT32) {
        int32_t* data = (int32_t*)t->data;
        for (size_t i = 0; i < n; i++) data[i * n + i] = 1;
    } else if (dtype == TENSR_INT64) {
        int64_t* data = (int64_t*)t->data;
        for (size_t i = 0; i < n; i++) data[i * n + i] = 1;
    }
    return t;
}

/**
 * @brief Create a deep copy of a tensor
 * @param src Source tensor to copy
 * @return Pointer to newly created tensor copy
 * 
 * Creates a new tensor with the same shape, type, and data as the source.
 * The data is copied, so modifications to the copy won't affect the original.
 */
Tensor* tensr_copy(const Tensor* src) {
    Tensor* t = tensr_create(src->shape, src->ndim, src->dtype, src->device);
    if (t) {
        memcpy(t->data, src->data, src->size * tensr_dtype_size(src->dtype));
    }
    return t;
}

/**
 * @brief Free tensor memory
 * @param t Tensor to free
 * 
 * Releases all memory associated with the tensor including data, shape,
 * and strides arrays. Always call this when done with a tensor to prevent
 * memory leaks.
 */
void tensr_free(Tensor* t) {
    if (t) {
        if (t->owns_data && t->data) free(t->data);
        if (t->shape) free(t->shape);
        if (t->strides) free(t->strides);
        free(t);
    }
}

/**
 * @brief Reshape a tensor to a new shape
 * @param t Source tensor
 * @param new_shape Array of new dimension sizes
 * @param new_ndim Number of new dimensions
 * @return Pointer to reshaped tensor view, or NULL if incompatible
 * 
 * Returns a new view of the tensor with a different shape. The total
 * number of elements must remain the same.
 * 
 * Example:
 *   size_t new_shape[] = {2, 3};
 *   Tensor* reshaped = tensr_reshape(t, new_shape, 2);
 */
Tensor* tensr_reshape(const Tensor* t, size_t* new_shape, size_t new_ndim) {
    size_t new_size = compute_size(new_shape, new_ndim);
    if (new_size != t->size) return NULL;

    Tensor* result = (Tensor*)malloc(sizeof(Tensor));
    if (!result) return NULL;

    result->ndim = new_ndim;
    result->dtype = t->dtype;
    result->device = t->device;
    result->device_id = t->device_id;
    result->size = t->size;
    result->owns_data = false;
    result->data = t->data;

    result->shape = (size_t*)malloc(new_ndim * sizeof(size_t));
    result->strides = (size_t*)malloc(new_ndim * sizeof(size_t));
    memcpy(result->shape, new_shape, new_ndim * sizeof(size_t));
    compute_strides(result->strides, result->shape, new_ndim);

    return result;
}

/**
 * @brief Transpose a tensor by permuting dimensions
 * @param t Source tensor
 * @param axes Array specifying dimension permutation (NULL for reverse)
 * @param naxes Number of axes (0 for default transpose)
 * @return Pointer to transposed tensor
 * 
 * Permutes the dimensions of the tensor. If axes is NULL, reverses the
 * dimension order.
 * 
 * Example:
 *   Tensor* transposed = tensr_transpose(t, NULL, 0);
 */
Tensor* tensr_transpose(const Tensor* t, size_t* axes, size_t naxes) {
    if (naxes == 0) {
        axes = (size_t*)malloc(t->ndim * sizeof(size_t));
        naxes = t->ndim;
        for (size_t i = 0; i < t->ndim; i++) {
            axes[i] = t->ndim - 1 - i;
        }
    }

    Tensor* result = tensr_create(t->shape, t->ndim, t->dtype, t->device);
    if (!result) return NULL;

    for (size_t i = 0; i < t->ndim; i++) {
        result->shape[i] = t->shape[axes[i]];
    }
    compute_strides(result->strides, result->shape, result->ndim);

    return result;
}

void tensr_to_device(Tensor* t, TensrDevice device, int device_id) {
    t->device = device;
    t->device_id = device_id;
}

void tensr_synchronize(TensrDevice device, int device_id) {
    /* Implementation depends on device backend */
}

int tensr_device_count(TensrDevice device) {
    return 1; /* Default implementation */
}
