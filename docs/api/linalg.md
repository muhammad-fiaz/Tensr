# Linear Algebra

## Matrix Operations

### dot - Dot product

Compute dot product of two 1D tensors.

=== "C"
    ```c
    Tensor* a = tensr_from_array((size_t[]){3}, 1, TENSR_FLOAT32, TENSR_CPU, (float[]){1, 2, 3});
    Tensor* b = tensr_from_array((size_t[]){3}, 1, TENSR_FLOAT32, TENSR_CPU, (float[]){4, 5, 6});
    Tensor* result = tensr_dot(a, b);  /* 1*4 + 2*5 + 3*6 = 32 */
    ```

=== "C++"
    ```cpp
    auto a = tensr::Tensor::ones({3});
    auto b = tensr::Tensor::ones({3});
    auto result = a.dot(b);
    ```

### matmul - Matrix multiplication

Multiply two matrices.

=== "C"
    ```c
    /* 2x3 matrix */
    Tensor* a = tensr_ones((size_t[]){2, 3}, 2, TENSR_FLOAT32, TENSR_CPU);
    /* 3x2 matrix */
    Tensor* b = tensr_ones((size_t[]){3, 2}, 2, TENSR_FLOAT32, TENSR_CPU);
    /* Result is 2x2 */
    Tensor* c = tensr_matmul(a, b);
    ```

=== "C++"
    ```cpp
    auto a = tensr::Tensor::ones({2, 3});
    auto b = tensr::Tensor::ones({3, 2});
    auto c = a.matmul(b);
    ```

### transpose - Matrix transpose

=== "C"
    ```c
    Tensor* a = tensr_ones((size_t[]){2, 3}, 2, TENSR_FLOAT32, TENSR_CPU);
    Tensor* at = tensr_transpose(a, NULL, 0);  /* Shape becomes 3x2 */
    ```

=== "C++"
    ```cpp
    auto a = tensr::Tensor::ones({2, 3});
    auto at = a.transpose();
    ```

## Matrix Decompositions

### inv - Matrix inverse

=== "C"
    ```c
    Tensor* a = tensr_eye(3, TENSR_FLOAT32, TENSR_CPU);
    Tensor* inv_a = tensr_inv(a);
    ```

=== "C++"
    ```cpp
    auto a = tensr::Tensor::eye(3);
    auto inv_a = a.inv();
    ```

### det - Determinant

=== "C"
    ```c
    Tensor* a = tensr_eye(3, TENSR_FLOAT32, TENSR_CPU);
    Tensor* det_a = tensr_det(a);  /* Determinant = 1.0 */
    ```

=== "C++"
    ```cpp
    auto a = tensr::Tensor::eye(3);
    auto det_a = a.det();
    ```

### svd - Singular Value Decomposition

=== "C"
    ```c
    Tensor* a = tensr_rand((size_t[]){4, 3}, 2, TENSR_CPU);
    Tensor *u, *s, *vt;
    tensr_svd(a, &u, &s, &vt);
    /* a = u * diag(s) * vt */
    ```

### eig - Eigenvalues and Eigenvectors

=== "C"
    ```c
    Tensor* a = tensr_rand((size_t[]){3, 3}, 2, TENSR_CPU);
    Tensor *eigvals, *eigvecs;
    tensr_eig(a, &eigvals, &eigvecs);
    ```

## Linear Systems

### solve - Solve Ax = b

=== "C"
    ```c
    Tensor* A = tensr_rand((size_t[]){3, 3}, 2, TENSR_CPU);
    Tensor* b = tensr_rand((size_t[]){3}, 1, TENSR_CPU);
    Tensor* x = tensr_solve(A, b);
    ```

### lstsq - Least squares solution

=== "C"
    ```c
    Tensor* A = tensr_rand((size_t[]){5, 3}, 2, TENSR_CPU);
    Tensor* b = tensr_rand((size_t[]){5}, 1, TENSR_CPU);
    Tensor* x = tensr_lstsq(A, b);
    ```

## Complete Example

```c
#include <tensr/tensr.h>

int main() {
    /* Create matrices */
    float data_a[] = {1, 2, 3, 4, 5, 6};
    float data_b[] = {7, 8, 9, 10, 11, 12};
    
    Tensor* a = tensr_from_array((size_t[]){2, 3}, 2, TENSR_FLOAT32, TENSR_CPU, data_a);
    Tensor* b = tensr_from_array((size_t[]){3, 2}, 2, TENSR_FLOAT32, TENSR_CPU, data_b);
    
    /* Matrix multiplication */
    Tensor* c = tensr_matmul(a, b);
    printf("Matrix multiplication result:\n");
    tensr_print(c);
    
    /* Transpose */
    Tensor* at = tensr_transpose(a, NULL, 0);
    printf("Transpose:\n");
    tensr_print(at);
    
    /* Identity matrix */
    Tensor* eye = tensr_eye(3, TENSR_FLOAT32, TENSR_CPU);
    Tensor* inv = tensr_inv(eye);
    printf("Inverse of identity:\n");
    tensr_print(inv);
    
    /* Cleanup */
    tensr_free(a);
    tensr_free(b);
    tensr_free(c);
    tensr_free(at);
    tensr_free(eye);
    tensr_free(inv);
    
    return 0;
}
```

## GPU Acceleration

Linear algebra operations benefit greatly from GPU acceleration:

```c
/* Create large matrices on GPU */
Tensor* a = tensr_rand((size_t[]){1000, 1000}, 2, TENSR_CUDA);
Tensor* b = tensr_rand((size_t[]){1000, 1000}, 2, TENSR_CUDA);

/* Fast GPU matrix multiplication */
Tensor* c = tensr_matmul(a, b);

tensr_synchronize(TENSR_CUDA, 0);
```
