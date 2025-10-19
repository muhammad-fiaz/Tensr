/**
 * @file tensr.h
 * @brief Main header for Tensr - A powerful, superfast multidimensional tensor library
 * @author Muhammad Fiaz
 * @version 0.0.0
 * @license Apache-2.0
 */

#ifndef TENSR_H
#define TENSR_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

/* Data types */
typedef enum {
    TENSR_FLOAT32,
    TENSR_FLOAT64,
    TENSR_INT32,
    TENSR_INT64,
    TENSR_UINT8,
    TENSR_BOOL
} TensrDType;

/* Device types */
typedef enum {
    TENSR_CPU,
    TENSR_CUDA,
    TENSR_XPU,
    TENSR_NPU,
    TENSR_TPU
} TensrDevice;

/* Tensor structure */
typedef struct {
    void* data;
    size_t* shape;
    size_t* strides;
    size_t ndim;
    size_t size;
    TensrDType dtype;
    TensrDevice device;
    int device_id;
    bool owns_data;
} Tensor;

/* Core tensor operations */
Tensor* tensr_create(size_t* shape, size_t ndim, TensrDType dtype, TensrDevice device);
Tensor* tensr_zeros(size_t* shape, size_t ndim, TensrDType dtype, TensrDevice device);
Tensor* tensr_ones(size_t* shape, size_t ndim, TensrDType dtype, TensrDevice device);
Tensor* tensr_full(size_t* shape, size_t ndim, double value, TensrDType dtype, TensrDevice device);
Tensor* tensr_arange(double start, double stop, double step, TensrDType dtype, TensrDevice device);
Tensor* tensr_linspace(double start, double stop, size_t num, TensrDType dtype, TensrDevice device);
Tensor* tensr_eye(size_t n, TensrDType dtype, TensrDevice device);
Tensor* tensr_copy(const Tensor* src);
Tensor* tensr_reshape(const Tensor* t, size_t* new_shape, size_t new_ndim);
Tensor* tensr_transpose(const Tensor* t, size_t* axes, size_t naxes);
Tensor* tensr_squeeze(const Tensor* t, int axis);
Tensor* tensr_expand_dims(const Tensor* t, int axis);
void tensr_free(Tensor* t);

/* Arithmetic operations */
Tensor* tensr_add(const Tensor* a, const Tensor* b);
Tensor* tensr_sub(const Tensor* a, const Tensor* b);
Tensor* tensr_mul(const Tensor* a, const Tensor* b);
Tensor* tensr_div(const Tensor* a, const Tensor* b);
Tensor* tensr_pow(const Tensor* a, double exponent);
Tensor* tensr_sqrt(const Tensor* t);
Tensor* tensr_exp(const Tensor* t);
Tensor* tensr_log(const Tensor* t);
Tensor* tensr_abs(const Tensor* t);
Tensor* tensr_neg(const Tensor* t);

/* Reduction operations */
Tensor* tensr_sum(const Tensor* t, int* axes, size_t naxes, bool keepdims);
Tensor* tensr_mean(const Tensor* t, int* axes, size_t naxes, bool keepdims);
Tensor* tensr_max(const Tensor* t, int* axes, size_t naxes, bool keepdims);
Tensor* tensr_min(const Tensor* t, int* axes, size_t naxes, bool keepdims);
Tensor* tensr_argmax(const Tensor* t, int axis);
Tensor* tensr_argmin(const Tensor* t, int axis);

/* Linear algebra */
Tensor* tensr_dot(const Tensor* a, const Tensor* b);
Tensor* tensr_matmul(const Tensor* a, const Tensor* b);
Tensor* tensr_inv(const Tensor* t);
Tensor* tensr_det(const Tensor* t);
Tensor* tensr_svd(const Tensor* t, Tensor** u, Tensor** s, Tensor** vt);
Tensor* tensr_eig(const Tensor* t, Tensor** eigvals, Tensor** eigvecs);
Tensor* tensr_solve(const Tensor* a, const Tensor* b);
Tensor* tensr_lstsq(const Tensor* a, const Tensor* b);

/* Trigonometric functions */
Tensor* tensr_sin(const Tensor* t);
Tensor* tensr_cos(const Tensor* t);
Tensor* tensr_tan(const Tensor* t);
Tensor* tensr_arcsin(const Tensor* t);
Tensor* tensr_arccos(const Tensor* t);
Tensor* tensr_arctan(const Tensor* t);

/* Random operations */
Tensor* tensr_rand(size_t* shape, size_t ndim, TensrDevice device);
Tensor* tensr_randn(size_t* shape, size_t ndim, TensrDevice device);
Tensor* tensr_randint(int low, int high, size_t* shape, size_t ndim, TensrDevice device);
void tensr_seed(unsigned int seed);

/* Comparison operations */
Tensor* tensr_equal(const Tensor* a, const Tensor* b);
Tensor* tensr_not_equal(const Tensor* a, const Tensor* b);
Tensor* tensr_greater(const Tensor* a, const Tensor* b);
Tensor* tensr_less(const Tensor* a, const Tensor* b);
Tensor* tensr_greater_equal(const Tensor* a, const Tensor* b);
Tensor* tensr_less_equal(const Tensor* a, const Tensor* b);

/* Logical operations */
Tensor* tensr_logical_and(const Tensor* a, const Tensor* b);
Tensor* tensr_logical_or(const Tensor* a, const Tensor* b);
Tensor* tensr_logical_not(const Tensor* t);

/* Indexing and slicing */
Tensor* tensr_slice(const Tensor* t, size_t* start, size_t* stop, size_t* step, size_t ndim);
Tensor* tensr_index(const Tensor* t, size_t* indices, size_t nindices);
void tensr_set(Tensor* t, size_t* indices, size_t nindices, double value);
double tensr_get(const Tensor* t, size_t* indices, size_t nindices);

/* Concatenation and stacking */
Tensor* tensr_concat(Tensor** tensors, size_t ntensors, int axis);
Tensor* tensr_stack(Tensor** tensors, size_t ntensors, int axis);
Tensor* tensr_vstack(Tensor** tensors, size_t ntensors);
Tensor* tensr_hstack(Tensor** tensors, size_t ntensors);

/* FFT operations */
Tensor* tensr_fft(const Tensor* t, int axis);
Tensor* tensr_ifft(const Tensor* t, int axis);
Tensor* tensr_fft2(const Tensor* t);
Tensor* tensr_ifft2(const Tensor* t);

/* I/O operations */
int tensr_save(const char* filename, const Tensor* t);
Tensor* tensr_load(const char* filename);
void tensr_print(const Tensor* t);

/* Device management */
void tensr_to_device(Tensor* t, TensrDevice device, int device_id);
void tensr_synchronize(TensrDevice device, int device_id);
int tensr_device_count(TensrDevice device);

/* Utility functions */
size_t tensr_dtype_size(TensrDType dtype);
const char* tensr_dtype_name(TensrDType dtype);
const char* tensr_device_name(TensrDevice device);

#ifdef __cplusplus
}
#endif

#endif /* TENSR_H */
