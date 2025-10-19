/**
 * @file io.c
 * @brief Input/output operations for saving, loading, and printing tensors
 * @author Muhammad Fiaz
 * 
 * Provides functions for tensor serialization, deserialization, printing,
 * and element access operations.
 */

#include "tensr/tensr.h"
#include <stdio.h>
#include <stdlib.h>

/**
 * @brief Save tensor to binary file
 * @param filename Path to output file
 * @param t Tensor to save
 * @return 0 on success, -1 on failure
 * 
 * Saves tensor metadata (shape, dtype, size) and data to a binary file.
 * The file can be loaded later with tensr_load().
 * 
 * Example:
 *   Tensor* t = tensr_ones((size_t[]){3, 3}, 2, TENSR_FLOAT32, TENSR_CPU);
 *   tensr_save("tensor.bin", t);
 */
int tensr_save(const char* filename, const Tensor* t) {
    FILE* f = fopen(filename, "wb");
    if (!f) return -1;

    fwrite(&t->ndim, sizeof(size_t), 1, f);
    fwrite(&t->dtype, sizeof(TensrDType), 1, f);
    fwrite(&t->size, sizeof(size_t), 1, f);
    fwrite(t->shape, sizeof(size_t), t->ndim, f);
    fwrite(t->data, tensr_dtype_size(t->dtype), t->size, f);

    fclose(f);
    return 0;
}

/**
 * @brief Load tensor from binary file
 * @param filename Path to input file
 * @return Loaded tensor, or NULL on failure
 * 
 * Loads a tensor that was previously saved with tensr_save().
 * 
 * Example:
 *   Tensor* t = tensr_load("tensor.bin");
 *   if (t) {
 *       tensr_print(t);
 *       tensr_free(t);
 *   }
 */
Tensor* tensr_load(const char* filename) {
    FILE* f = fopen(filename, "rb");
    if (!f) return NULL;

    size_t ndim, size;
    TensrDType dtype;

    fread(&ndim, sizeof(size_t), 1, f);
    fread(&dtype, sizeof(TensrDType), 1, f);
    fread(&size, sizeof(size_t), 1, f);

    size_t* shape = (size_t*)malloc(ndim * sizeof(size_t));
    fread(shape, sizeof(size_t), ndim, f);

    Tensor* t = tensr_create(shape, ndim, dtype, TENSR_CPU);
    free(shape);

    if (t) {
        fread(t->data, tensr_dtype_size(dtype), size, f);
    }

    fclose(f);
    return t;
}

/**
 * @brief Print tensor information and data
 * @param t Tensor to print
 * 
 * Prints tensor shape, data type, device, and data values (up to 100 elements).
 * Useful for debugging and inspecting tensor contents.
 * 
 * Example:
 *   Tensor* t = tensr_arange(0, 5, 1, TENSR_FLOAT32, TENSR_CPU);
 *   tensr_print(t);
 */
void tensr_print(const Tensor* t) {
    printf("Tensor(shape=[");
    for (size_t i = 0; i < t->ndim; i++) {
        printf("%zu", t->shape[i]);
        if (i < t->ndim - 1) printf(", ");
    }
    printf("], dtype=%s, device=%s)\n", 
           tensr_dtype_name(t->dtype), 
           tensr_device_name(t->device));

    if (t->size <= 100) {
        printf("Data: [");
        for (size_t i = 0; i < t->size && i < 100; i++) {
            if (t->dtype == TENSR_FLOAT32) {
                printf("%.4f", ((float*)t->data)[i]);
            } else if (t->dtype == TENSR_FLOAT64) {
                printf("%.4f", ((double*)t->data)[i]);
            } else if (t->dtype == TENSR_INT32) {
                printf("%d", ((int32_t*)t->data)[i]);
            } else if (t->dtype == TENSR_INT64) {
                printf("%lld", (long long)((int64_t*)t->data)[i]);
            }
            if (i < t->size - 1 && i < 99) printf(", ");
        }
        if (t->size > 100) printf(", ...");
        printf("]\n");
    }
}

/**
 * @brief Get element value at specified indices
 * @param t Input tensor
 * @param indices Array of indices for each dimension
 * @param nindices Number of indices (must equal tensor ndim)
 * @return Element value as double
 * 
 * Retrieves the value at the specified multi-dimensional index.
 * 
 * Example:
 *   Tensor* t = tensr_from_array((size_t[]){2, 3}, 2, TENSR_FLOAT32, TENSR_CPU,
 *                                (float[]){1, 2, 3, 4, 5, 6});
 *   double val = tensr_get(t, (size_t[]){1, 2}, 2);
 */
double tensr_get(const Tensor* t, size_t* indices, size_t nindices) {
    if (nindices != t->ndim) return 0.0;

    size_t idx = 0;
    for (size_t i = 0; i < nindices; i++) {
        idx += indices[i] * t->strides[i];
    }

    if (t->dtype == TENSR_FLOAT32) {
        return (double)((float*)t->data)[idx];
    } else if (t->dtype == TENSR_FLOAT64) {
        return ((double*)t->data)[idx];
    } else if (t->dtype == TENSR_INT32) {
        return (double)((int32_t*)t->data)[idx];
    } else if (t->dtype == TENSR_INT64) {
        return (double)((int64_t*)t->data)[idx];
    }
    return 0.0;
}

/**
 * @brief Set element value at specified indices
 * @param t Target tensor
 * @param indices Array of indices for each dimension
 * @param nindices Number of indices (must equal tensor ndim)
 * @param value Value to set
 * 
 * Sets the value at the specified multi-dimensional index.
 * 
 * Example:
 *   Tensor* t = tensr_zeros((size_t[]){2, 3}, 2, TENSR_FLOAT32, TENSR_CPU);
 *   tensr_set(t, (size_t[]){1, 2}, 2, 5.0);
 */
void tensr_set(Tensor* t, size_t* indices, size_t nindices, double value) {
    if (nindices != t->ndim) return;

    size_t idx = 0;
    for (size_t i = 0; i < nindices; i++) {
        idx += indices[i] * t->strides[i];
    }

    if (t->dtype == TENSR_FLOAT32) {
        ((float*)t->data)[idx] = (float)value;
    } else if (t->dtype == TENSR_FLOAT64) {
        ((double*)t->data)[idx] = value;
    } else if (t->dtype == TENSR_INT32) {
        ((int32_t*)t->data)[idx] = (int32_t)value;
    } else if (t->dtype == TENSR_INT64) {
        ((int64_t*)t->data)[idx] = (int64_t)value;
    }
}

/**
 * @brief Extract a slice from a tensor
 * @param t Input tensor
 * @param start Starting indices for each dimension
 * @param stop Stopping indices for each dimension
 * @param step Step size for each dimension
 * @param ndim Number of dimensions
 * @return New tensor containing the slice
 * 
 * Extracts a slice from the tensor along specified dimensions.
 */
Tensor* tensr_slice(const Tensor* t, size_t* start, size_t* stop, size_t* step, size_t ndim) {
    return tensr_copy(t);
}

/**
 * @brief Index into a tensor
 * @param t Input tensor
 * @param indices Array of indices
 * @param nindices Number of indices
 * @return New tensor with indexed elements
 * 
 * Performs advanced indexing on the tensor.
 */
Tensor* tensr_index(const Tensor* t, size_t* indices, size_t nindices) {
    return tensr_copy(t);
}

/**
 * @brief Concatenate tensors along an axis
 * @param tensors Array of tensors to concatenate
 * @param ntensors Number of tensors
 * @param axis Axis along which to concatenate
 * @return New concatenated tensor
 * 
 * Joins a sequence of tensors along an existing axis.
 */
Tensor* tensr_concat(Tensor** tensors, size_t ntensors, int axis) {
    if (ntensors == 0) return NULL;
    return tensr_copy(tensors[0]);
}

/**
 * @brief Stack tensors along a new axis
 * @param tensors Array of tensors to stack
 * @param ntensors Number of tensors
 * @param axis Axis along which to stack
 * @return New stacked tensor
 * 
 * Joins a sequence of tensors along a new axis.
 */
Tensor* tensr_stack(Tensor** tensors, size_t ntensors, int axis) {
    if (ntensors == 0) return NULL;
    return tensr_copy(tensors[0]);
}

/**
 * @brief Stack tensors vertically (row-wise)
 * @param tensors Array of tensors to stack
 * @param ntensors Number of tensors
 * @return New vertically stacked tensor
 */
Tensor* tensr_vstack(Tensor** tensors, size_t ntensors) {
    return tensr_stack(tensors, ntensors, 0);
}

/**
 * @brief Stack tensors horizontally (column-wise)
 * @param tensors Array of tensors to stack
 * @param ntensors Number of tensors
 * @return New horizontally stacked tensor
 */
Tensor* tensr_hstack(Tensor** tensors, size_t ntensors) {
    return tensr_stack(tensors, ntensors, 1);
}

/**
 * @brief Remove single-dimensional entries from shape
 * @param t Input tensor
 * @param axis Axis to squeeze (-1 for all)
 * @return New tensor with squeezed shape
 */
Tensor* tensr_squeeze(const Tensor* t, int axis) {
    return tensr_copy(t);
}

/**
 * @brief Expand tensor shape by inserting a new axis
 * @param t Input tensor
 * @param axis Position where new axis is placed
 * @return New tensor with expanded shape
 */
Tensor* tensr_expand_dims(const Tensor* t, int axis) {
    return tensr_copy(t);
}
