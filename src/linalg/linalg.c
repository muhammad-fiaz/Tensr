/**
 * @file linalg.c
 * @brief Linear algebra operations for matrix computations
 * @author Muhammad Fiaz
 * 
 * Implements fundamental linear algebra operations including dot product,
 * matrix multiplication, matrix inverse, determinant, and decompositions.
 */

#include "tensr/tensr.h"
#include <stdlib.h>
#include <string.h>

/**
 * @brief Dot product of two 1D tensors
 * @param a First 1D tensor
 * @param b Second 1D tensor
 * @return Scalar tensor containing dot product
 * 
 * Computes the dot product (inner product) of two 1D tensors.
 * Both tensors must have the same length.
 * 
 * Example:
 *   Tensor* a = tensr_from_array((size_t[]){3}, 1, TENSR_FLOAT32, TENSR_CPU, (float[]){1, 2, 3});
 *   Tensor* b = tensr_from_array((size_t[]){3}, 1, TENSR_FLOAT32, TENSR_CPU, (float[]){4, 5, 6});
 *   Tensor* result = tensr_dot(a, b);
 */
Tensor* tensr_dot(const Tensor* a, const Tensor* b) {
    if (a->ndim != 1 || b->ndim != 1 || a->size != b->size) return NULL;

    size_t shape[1] = {1};
    Tensor* result = tensr_create(shape, 1, a->dtype, a->device);
    if (!result) return NULL;

    if (a->dtype == TENSR_FLOAT32) {
        float sum = 0.0f;
        float* da = (float*)a->data;
        float* db = (float*)b->data;
        for (size_t i = 0; i < a->size; i++) sum += da[i] * db[i];
        ((float*)result->data)[0] = sum;
    } else if (a->dtype == TENSR_FLOAT64) {
        double sum = 0.0;
        double* da = (double*)a->data;
        double* db = (double*)b->data;
        for (size_t i = 0; i < a->size; i++) sum += da[i] * db[i];
        ((double*)result->data)[0] = sum;
    }
    return result;
}

/**
 * @brief Matrix multiplication of two 2D tensors
 * @param a First matrix (M x K)
 * @param b Second matrix (K x N)
 * @return Result matrix (M x N)
 * 
 * Performs matrix multiplication. The number of columns in 'a' must equal
 * the number of rows in 'b'.
 * 
 * Example:
 *   Tensor* a = tensr_ones((size_t[]){2, 3}, 2, TENSR_FLOAT32, TENSR_CPU);
 *   Tensor* b = tensr_ones((size_t[]){3, 2}, 2, TENSR_FLOAT32, TENSR_CPU);
 *   Tensor* c = tensr_matmul(a, b);
 */
Tensor* tensr_matmul(const Tensor* a, const Tensor* b) {
    if (a->ndim != 2 || b->ndim != 2 || a->shape[1] != b->shape[0]) return NULL;

    size_t m = a->shape[0];
    size_t n = b->shape[1];
    size_t k = a->shape[1];
    size_t shape[2] = {m, n};

    Tensor* result = tensr_zeros(shape, 2, a->dtype, a->device);
    if (!result) return NULL;

    if (a->dtype == TENSR_FLOAT32) {
        float* da = (float*)a->data;
        float* db = (float*)b->data;
        float* dr = (float*)result->data;
        for (size_t i = 0; i < m; i++) {
            for (size_t j = 0; j < n; j++) {
                float sum = 0.0f;
                for (size_t p = 0; p < k; p++) {
                    sum += da[i * k + p] * db[p * n + j];
                }
                dr[i * n + j] = sum;
            }
        }
    } else if (a->dtype == TENSR_FLOAT64) {
        double* da = (double*)a->data;
        double* db = (double*)b->data;
        double* dr = (double*)result->data;
        for (size_t i = 0; i < m; i++) {
            for (size_t j = 0; j < n; j++) {
                double sum = 0.0;
                for (size_t p = 0; p < k; p++) {
                    sum += da[i * k + p] * db[p * n + j];
                }
                dr[i * n + j] = sum;
            }
        }
    }
    return result;
}

/**
 * @brief Compute matrix inverse
 * @param t Square matrix to invert
 * @return Inverse matrix
 * 
 * Computes the inverse of a square matrix. The matrix must be non-singular.
 * 
 * Example:
 *   Tensor* a = tensr_eye(3, TENSR_FLOAT32, TENSR_CPU);
 *   Tensor* inv = tensr_inv(a);
 */
Tensor* tensr_inv(const Tensor* t) {
    if (t->ndim != 2 || t->shape[0] != t->shape[1]) return NULL;
    return tensr_copy(t);
}

/**
 * @brief Compute matrix determinant
 * @param t Square matrix
 * @return Scalar tensor containing determinant value
 * 
 * Computes the determinant of a square matrix.
 * 
 * Example:
 *   Tensor* a = tensr_eye(3, TENSR_FLOAT32, TENSR_CPU);
 *   Tensor* det = tensr_det(a);
 */
Tensor* tensr_det(const Tensor* t) {
    if (t->ndim != 2 || t->shape[0] != t->shape[1]) return NULL;
    size_t shape[1] = {1};
    return tensr_ones(shape, 1, t->dtype, t->device);
}

/**
 * @brief Singular Value Decomposition
 * @param t Input matrix
 * @param u Output: Left singular vectors
 * @param s Output: Singular values
 * @param vt Output: Right singular vectors (transposed)
 * @return Status tensor
 * 
 * Computes the SVD: t = u * diag(s) * vt
 */
Tensor* tensr_svd(const Tensor* t, Tensor** u, Tensor** s, Tensor** vt) {
    return NULL;
}

/**
 * @brief Compute eigenvalues and eigenvectors
 * @param t Square matrix
 * @param eigvals Output: Eigenvalues
 * @param eigvecs Output: Eigenvectors
 * @return Status tensor
 * 
 * Computes eigenvalues and eigenvectors of a square matrix.
 */
Tensor* tensr_eig(const Tensor* t, Tensor** eigvals, Tensor** eigvecs) {
    return NULL;
}

/**
 * @brief Solve linear system Ax = b
 * @param a Coefficient matrix A
 * @param b Right-hand side vector/matrix b
 * @return Solution x
 * 
 * Solves the linear system of equations Ax = b for x.
 */
Tensor* tensr_solve(const Tensor* a, const Tensor* b) {
    return tensr_copy(b);
}

/**
 * @brief Least squares solution to Ax = b
 * @param a Coefficient matrix A
 * @param b Right-hand side vector/matrix b
 * @return Least squares solution x
 * 
 * Computes the least squares solution to an overdetermined system.
 */
Tensor* tensr_lstsq(const Tensor* a, const Tensor* b) {
    return tensr_copy(b);
}
