/**
 * @file reduction.c
 * @brief Tensor reduction operations for aggregating values
 * @author Muhammad Fiaz
 * 
 * Implements reduction operations that aggregate tensor values along specified
 * axes, including sum, mean, max, min, argmax, and argmin operations.
 */

#include "tensr/tensr.h"
#include <stdlib.h>
#include <float.h>
#include <math.h>

/**
 * @brief Sum of tensor elements
 * @param t Input tensor
 * @param axes Array of axes to reduce over (NULL for all)
 * @param naxes Number of axes (0 for all)
 * @param keepdims Whether to keep reduced dimensions
 * @return New tensor with sum values
 * 
 * Computes the sum of tensor elements along specified axes.
 * 
 * Example:
 *   Tensor* t = tensr_ones((size_t[]){2, 3}, 2, TENSR_FLOAT32, TENSR_CPU);
 *   Tensor* sum = tensr_sum(t, NULL, 0, false);
 */
Tensor* tensr_sum(const Tensor* t, int* axes, size_t naxes, bool keepdims) {
    if (naxes == 0) {
        size_t shape[1] = {1};
        Tensor* result = tensr_create(shape, keepdims ? t->ndim : 1, t->dtype, t->device);
        if (!result) return NULL;

        if (t->dtype == TENSR_FLOAT32) {
            float sum = 0.0f;
            float* data = (float*)t->data;
            for (size_t i = 0; i < t->size; i++) sum += data[i];
            ((float*)result->data)[0] = sum;
        } else if (t->dtype == TENSR_FLOAT64) {
            double sum = 0.0;
            double* data = (double*)t->data;
            for (size_t i = 0; i < t->size; i++) sum += data[i];
            ((double*)result->data)[0] = sum;
        }
        return result;
    }
    return NULL;
}

/**
 * @brief Mean of tensor elements
 * @param t Input tensor
 * @param axes Array of axes to reduce over (NULL for all)
 * @param naxes Number of axes (0 for all)
 * @param keepdims Whether to keep reduced dimensions
 * @return New tensor with mean values
 * 
 * Computes the arithmetic mean of tensor elements along specified axes.
 * 
 * Example:
 *   Tensor* t = tensr_ones((size_t[]){2, 3}, 2, TENSR_FLOAT32, TENSR_CPU);
 *   Tensor* mean = tensr_mean(t, NULL, 0, false);
 */
Tensor* tensr_mean(const Tensor* t, int* axes, size_t naxes, bool keepdims) {
    Tensor* sum_result = tensr_sum(t, axes, naxes, keepdims);
    if (!sum_result) return NULL;

    if (t->dtype == TENSR_FLOAT32) {
        float* data = (float*)sum_result->data;
        data[0] /= (float)t->size;
    } else if (t->dtype == TENSR_FLOAT64) {
        double* data = (double*)sum_result->data;
        data[0] /= (double)t->size;
    }
    return sum_result;
}

/**
 * @brief Maximum value in tensor
 * @param t Input tensor
 * @param axes Array of axes to reduce over (NULL for all)
 * @param naxes Number of axes (0 for all)
 * @param keepdims Whether to keep reduced dimensions
 * @return New tensor with maximum values
 * 
 * Finds the maximum value in the tensor along specified axes.
 * 
 * Example:
 *   Tensor* t = tensr_from_array((size_t[]){3}, 1, TENSR_FLOAT32, TENSR_CPU, (float[]){1, 5, 3});
 *   Tensor* max = tensr_max(t, NULL, 0, false);
 */
Tensor* tensr_max(const Tensor* t, int* axes, size_t naxes, bool keepdims) {
    size_t shape[1] = {1};
    Tensor* result = tensr_create(shape, 1, t->dtype, t->device);
    if (!result) return NULL;

    if (t->dtype == TENSR_FLOAT32) {
        float max_val = -FLT_MAX;
        float* data = (float*)t->data;
        for (size_t i = 0; i < t->size; i++) {
            if (data[i] > max_val) max_val = data[i];
        }
        ((float*)result->data)[0] = max_val;
    } else if (t->dtype == TENSR_FLOAT64) {
        double max_val = -DBL_MAX;
        double* data = (double*)t->data;
        for (size_t i = 0; i < t->size; i++) {
            if (data[i] > max_val) max_val = data[i];
        }
        ((double*)result->data)[0] = max_val;
    }
    return result;
}

/**
 * @brief Minimum value in tensor
 * @param t Input tensor
 * @param axes Array of axes to reduce over (NULL for all)
 * @param naxes Number of axes (0 for all)
 * @param keepdims Whether to keep reduced dimensions
 * @return New tensor with minimum values
 * 
 * Finds the minimum value in the tensor along specified axes.
 * 
 * Example:
 *   Tensor* t = tensr_from_array((size_t[]){3}, 1, TENSR_FLOAT32, TENSR_CPU, (float[]){1, 5, 3});
 *   Tensor* min = tensr_min(t, NULL, 0, false);
 */
Tensor* tensr_min(const Tensor* t, int* axes, size_t naxes, bool keepdims) {
    size_t shape[1] = {1};
    Tensor* result = tensr_create(shape, 1, t->dtype, t->device);
    if (!result) return NULL;

    if (t->dtype == TENSR_FLOAT32) {
        float min_val = FLT_MAX;
        float* data = (float*)t->data;
        for (size_t i = 0; i < t->size; i++) {
            if (data[i] < min_val) min_val = data[i];
        }
        ((float*)result->data)[0] = min_val;
    } else if (t->dtype == TENSR_FLOAT64) {
        double min_val = DBL_MAX;
        double* data = (double*)t->data;
        for (size_t i = 0; i < t->size; i++) {
            if (data[i] < min_val) min_val = data[i];
        }
        ((double*)result->data)[0] = min_val;
    }
    return result;
}

/**
 * @brief Index of maximum value in tensor
 * @param t Input tensor
 * @param axis Axis to find maximum along (-1 for flattened)
 * @return New tensor with indices of maximum values
 * 
 * Returns the index of the maximum value in the tensor.
 * 
 * Example:
 *   Tensor* t = tensr_from_array((size_t[]){3}, 1, TENSR_FLOAT32, TENSR_CPU, (float[]){1, 5, 3});
 *   Tensor* idx = tensr_argmax(t, -1);
 */
Tensor* tensr_argmax(const Tensor* t, int axis) {
    size_t shape[1] = {1};
    Tensor* result = tensr_create(shape, 1, TENSR_INT64, t->device);
    if (!result) return NULL;

    size_t max_idx = 0;
    if (t->dtype == TENSR_FLOAT32) {
        float max_val = -FLT_MAX;
        float* data = (float*)t->data;
        for (size_t i = 0; i < t->size; i++) {
            if (data[i] > max_val) {
                max_val = data[i];
                max_idx = i;
            }
        }
    } else if (t->dtype == TENSR_FLOAT64) {
        double max_val = -DBL_MAX;
        double* data = (double*)t->data;
        for (size_t i = 0; i < t->size; i++) {
            if (data[i] > max_val) {
                max_val = data[i];
                max_idx = i;
            }
        }
    }
    ((int64_t*)result->data)[0] = (int64_t)max_idx;
    return result;
}

/**
 * @brief Index of minimum value in tensor
 * @param t Input tensor
 * @param axis Axis to find minimum along (-1 for flattened)
 * @return New tensor with indices of minimum values
 * 
 * Returns the index of the minimum value in the tensor.
 * 
 * Example:
 *   Tensor* t = tensr_from_array((size_t[]){3}, 1, TENSR_FLOAT32, TENSR_CPU, (float[]){1, 5, 3});
 *   Tensor* idx = tensr_argmin(t, -1);
 */
Tensor* tensr_argmin(const Tensor* t, int axis) {
    size_t shape[1] = {1};
    Tensor* result = tensr_create(shape, 1, TENSR_INT64, t->device);
    if (!result) return NULL;

    size_t min_idx = 0;
    if (t->dtype == TENSR_FLOAT32) {
        float min_val = FLT_MAX;
        float* data = (float*)t->data;
        for (size_t i = 0; i < t->size; i++) {
            if (data[i] < min_val) {
                min_val = data[i];
                min_idx = i;
            }
        }
    } else if (t->dtype == TENSR_FLOAT64) {
        double min_val = DBL_MAX;
        double* data = (double*)t->data;
        for (size_t i = 0; i < t->size; i++) {
            if (data[i] < min_val) {
                min_val = data[i];
                min_idx = i;
            }
        }
    }
    ((int64_t*)result->data)[0] = (int64_t)min_idx;
    return result;
}
