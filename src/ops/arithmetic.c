/**
 * @file arithmetic.c
 * @brief Element-wise arithmetic and comparison operations for tensors
 * @author Muhammad Fiaz
 * 
 * Implements element-wise operations including basic arithmetic (add, subtract,
 * multiply, divide), mathematical functions (pow, sqrt, exp, log), trigonometric
 * functions (sin, cos, tan), and comparison operations.
 */

#include "tensr/tensr.h"
#include <stdlib.h>
#include <math.h>

/**
 * @brief Macro to generate element-wise binary operations
 * @param name Operation name
 * @param op C operator to apply
 * 
 * Generates functions for element-wise binary operations that work across
 * all supported data types (float32, float64, int32, int64).
 */
#define BINARY_OP(name, op) \
Tensor* tensr_##name(const Tensor* a, const Tensor* b) { \
    if (a->size != b->size || a->dtype != b->dtype) return NULL; \
    Tensor* result = tensr_create(a->shape, a->ndim, a->dtype, a->device); \
    if (!result) return NULL; \
    if (a->dtype == TENSR_FLOAT32) { \
        float* ra = (float*)a->data; \
        float* rb = (float*)b->data; \
        float* rr = (float*)result->data; \
        for (size_t i = 0; i < a->size; i++) rr[i] = ra[i] op rb[i]; \
    } else if (a->dtype == TENSR_FLOAT64) { \
        double* ra = (double*)a->data; \
        double* rb = (double*)b->data; \
        double* rr = (double*)result->data; \
        for (size_t i = 0; i < a->size; i++) rr[i] = ra[i] op rb[i]; \
    } else if (a->dtype == TENSR_INT32) { \
        int32_t* ra = (int32_t*)a->data; \
        int32_t* rb = (int32_t*)b->data; \
        int32_t* rr = (int32_t*)result->data; \
        for (size_t i = 0; i < a->size; i++) rr[i] = ra[i] op rb[i]; \
    } else if (a->dtype == TENSR_INT64) { \
        int64_t* ra = (int64_t*)a->data; \
        int64_t* rb = (int64_t*)b->data; \
        int64_t* rr = (int64_t*)result->data; \
        for (size_t i = 0; i < a->size; i++) rr[i] = ra[i] op rb[i]; \
    } \
    return result; \
}

BINARY_OP(add, +)
BINARY_OP(sub, -)
BINARY_OP(mul, *)
BINARY_OP(div, /)

/**
 * @brief Macro to generate element-wise unary mathematical functions
 * @param name Function name
 * @param func C math function to apply
 * 
 * Generates functions for element-wise unary operations using standard
 * C math library functions (sqrt, exp, log, sin, cos, tan, etc.).
 */
#define UNARY_FUNC(name, func) \
Tensor* tensr_##name(const Tensor* t) { \
    Tensor* result = tensr_create(t->shape, t->ndim, t->dtype, t->device); \
    if (!result) return NULL; \
    if (t->dtype == TENSR_FLOAT32) { \
        float* rt = (float*)t->data; \
        float* rr = (float*)result->data; \
        for (size_t i = 0; i < t->size; i++) rr[i] = func##f(rt[i]); \
    } else if (t->dtype == TENSR_FLOAT64) { \
        double* rt = (double*)t->data; \
        double* rr = (double*)result->data; \
        for (size_t i = 0; i < t->size; i++) rr[i] = func(rt[i]); \
    } \
    return result; \
}

UNARY_FUNC(sqrt, sqrt)
UNARY_FUNC(exp, exp)
UNARY_FUNC(log, log)
UNARY_FUNC(abs, fabs)
UNARY_FUNC(sin, sin)
UNARY_FUNC(cos, cos)
UNARY_FUNC(tan, tan)
UNARY_FUNC(arcsin, asin)
UNARY_FUNC(arccos, acos)
UNARY_FUNC(arctan, atan)

/**
 * @brief Raise tensor elements to a power
 * @param a Input tensor
 * @param exponent Power to raise elements to
 * @return New tensor with elements raised to the power
 * 
 * Computes element-wise power operation: result[i] = a[i] ^ exponent
 * 
 * Example:
 *   Tensor* a = tensr_from_array((size_t[]){3}, 1, TENSR_FLOAT32, TENSR_CPU, (float[]){2, 3, 4});
 *   Tensor* squared = tensr_pow(a, 2.0);
 */
Tensor* tensr_pow(const Tensor* a, double exponent) {
    Tensor* result = tensr_create(a->shape, a->ndim, a->dtype, a->device);
    if (!result) return NULL;

    if (a->dtype == TENSR_FLOAT32) {
        float* ra = (float*)a->data;
        float* rr = (float*)result->data;
        for (size_t i = 0; i < a->size; i++) rr[i] = powf(ra[i], (float)exponent);
    } else if (a->dtype == TENSR_FLOAT64) {
        double* ra = (double*)a->data;
        double* rr = (double*)result->data;
        for (size_t i = 0; i < a->size; i++) rr[i] = pow(ra[i], exponent);
    }
    return result;
}

/**
 * @brief Negate all elements in a tensor
 * @param t Input tensor
 * @return New tensor with negated elements
 * 
 * Computes element-wise negation: result[i] = -t[i]
 * 
 * Example:
 *   Tensor* a = tensr_from_array((size_t[]){3}, 1, TENSR_FLOAT32, TENSR_CPU, (float[]){1, -2, 3});
 *   Tensor* neg = tensr_neg(a);
 */
Tensor* tensr_neg(const Tensor* t) {
    Tensor* result = tensr_create(t->shape, t->ndim, t->dtype, t->device);
    if (!result) return NULL;

    if (t->dtype == TENSR_FLOAT32) {
        float* rt = (float*)t->data;
        float* rr = (float*)result->data;
        for (size_t i = 0; i < t->size; i++) rr[i] = -rt[i];
    } else if (t->dtype == TENSR_FLOAT64) {
        double* rt = (double*)t->data;
        double* rr = (double*)result->data;
        for (size_t i = 0; i < t->size; i++) rr[i] = -rt[i];
    } else if (t->dtype == TENSR_INT32) {
        int32_t* rt = (int32_t*)t->data;
        int32_t* rr = (int32_t*)result->data;
        for (size_t i = 0; i < t->size; i++) rr[i] = -rt[i];
    } else if (t->dtype == TENSR_INT64) {
        int64_t* rt = (int64_t*)t->data;
        int64_t* rr = (int64_t*)result->data;
        for (size_t i = 0; i < t->size; i++) rr[i] = -rt[i];
    }
    return result;
}

/**
 * @brief Macro to generate element-wise comparison operations
 * @param name Operation name
 * @param op C comparison operator
 * 
 * Generates functions for element-wise comparison operations that return
 * boolean tensors indicating where the condition is true.
 */
#define COMPARISON_OP(name, op) \
Tensor* tensr_##name(const Tensor* a, const Tensor* b) { \
    if (a->size != b->size) return NULL; \
    Tensor* result = tensr_create(a->shape, a->ndim, TENSR_BOOL, a->device); \
    if (!result) return NULL; \
    bool* rr = (bool*)result->data; \
    if (a->dtype == TENSR_FLOAT32) { \
        float* ra = (float*)a->data; \
        float* rb = (float*)b->data; \
        for (size_t i = 0; i < a->size; i++) rr[i] = ra[i] op rb[i]; \
    } else if (a->dtype == TENSR_FLOAT64) { \
        double* ra = (double*)a->data; \
        double* rb = (double*)b->data; \
        for (size_t i = 0; i < a->size; i++) rr[i] = ra[i] op rb[i]; \
    } else if (a->dtype == TENSR_INT32) { \
        int32_t* ra = (int32_t*)a->data; \
        int32_t* rb = (int32_t*)b->data; \
        for (size_t i = 0; i < a->size; i++) rr[i] = ra[i] op rb[i]; \
    } else if (a->dtype == TENSR_INT64) { \
        int64_t* ra = (int64_t*)a->data; \
        int64_t* rb = (int64_t*)b->data; \
        for (size_t i = 0; i < a->size; i++) rr[i] = ra[i] op rb[i]; \
    } \
    return result; \
}

COMPARISON_OP(equal, ==)
COMPARISON_OP(not_equal, !=)
COMPARISON_OP(greater, >)
COMPARISON_OP(less, <)
COMPARISON_OP(greater_equal, >=)
COMPARISON_OP(less_equal, <=)

/**
 * @brief Element-wise logical AND operation
 * @param a First boolean tensor
 * @param b Second boolean tensor
 * @return New boolean tensor with AND results
 * 
 * Computes element-wise logical AND: result[i] = a[i] && b[i]
 */
Tensor* tensr_logical_and(const Tensor* a, const Tensor* b) {
    if (a->size != b->size) return NULL;
    Tensor* result = tensr_create(a->shape, a->ndim, TENSR_BOOL, a->device);
    if (!result) return NULL;

    bool* ra = (bool*)a->data;
    bool* rb = (bool*)b->data;
    bool* rr = (bool*)result->data;
    for (size_t i = 0; i < a->size; i++) rr[i] = ra[i] && rb[i];
    return result;
}

/**
 * @brief Element-wise logical OR operation
 * @param a First boolean tensor
 * @param b Second boolean tensor
 * @return New boolean tensor with OR results
 * 
 * Computes element-wise logical OR: result[i] = a[i] || b[i]
 */
Tensor* tensr_logical_or(const Tensor* a, const Tensor* b) {
    if (a->size != b->size) return NULL;
    Tensor* result = tensr_create(a->shape, a->ndim, TENSR_BOOL, a->device);
    if (!result) return NULL;

    bool* ra = (bool*)a->data;
    bool* rb = (bool*)b->data;
    bool* rr = (bool*)result->data;
    for (size_t i = 0; i < a->size; i++) rr[i] = ra[i] || rb[i];
    return result;
}

/**
 * @brief Element-wise logical NOT operation
 * @param t Input boolean tensor
 * @return New boolean tensor with NOT results
 * 
 * Computes element-wise logical NOT: result[i] = !t[i]
 */
Tensor* tensr_logical_not(const Tensor* t) {
    Tensor* result = tensr_create(t->shape, t->ndim, TENSR_BOOL, t->device);
    if (!result) return NULL;

    bool* rt = (bool*)t->data;
    bool* rr = (bool*)result->data;
    for (size_t i = 0; i < t->size; i++) rr[i] = !rt[i];
    return result;
}
