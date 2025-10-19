# Quick Start

## NumPy-Style Usage in C

```c
#include <tensr/tensr.h>
#include <tensr/tensr_array.h>

int main() {
    /* Create arrays from data (like np.array) */
    float data_a[] = {1, 2, 3};
    float data_b[] = {4, 5, 6};
    size_t shape[] = {3};
    
    Tensor* a = tensr_from_array(shape, 1, TENSR_FLOAT32, TENSR_CPU, data_a);
    Tensor* b = tensr_from_array(shape, 1, TENSR_FLOAT32, TENSR_CPU, data_b);
    
    /* Element-wise operations */
    Tensor* sum = tensr_add(a, b);      /* a + b = [5, 7, 9] */
    Tensor* product = tensr_mul(a, b);  /* a * b = [4, 10, 18] */
    Tensor* squared = tensr_pow(a, 2.0); /* a ** 2 = [1, 4, 9] */
    
    /* Print results */
    tensr_print(sum);
    tensr_print(product);
    tensr_print(squared);
    
    /* Cleanup */
    tensr_free(a);
    tensr_free(b);
    tensr_free(sum);
    tensr_free(product);
    tensr_free(squared);
    
    return 0;
}
```

## Matrix Operations

```c
/* Create a 3x3 matrix (like np.array([[1,2,3], [4,5,6], [7,8,9]])) */
float matrix_data[] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
size_t shape[] = {3, 3};
Tensor* matrix = tensr_from_array(shape, 2, TENSR_FLOAT32, TENSR_CPU, matrix_data);

/* Transpose (like matrix.T) */
Tensor* transpose = tensr_transpose(matrix, NULL, 0);

/* Matrix multiplication (like np.dot(matrix, transpose)) */
Tensor* result = tensr_matmul(matrix, transpose);

/* Statistics (like np.mean(matrix), np.max(matrix)) */
Tensor* mean = tensr_mean(matrix, NULL, 0, false);
Tensor* max_val = tensr_max(matrix, NULL, 0, false);

tensr_print(result);
tensr_print(mean);
tensr_print(max_val);

/* Cleanup */
tensr_free(matrix);
tensr_free(transpose);
tensr_free(result);
tensr_free(mean);
tensr_free(max_val);
```

## C++ Usage

```cpp
#include <tensr/tensr.hpp>

int main() {
    /* Create tensors */
    auto a = tensr::Tensor::ones({3, 3});
    auto b = tensr::Tensor::rand({3, 3});
    
    /* Operations with operator overloading */
    auto c = a + b;
    auto d = a * b;
    auto e = a.pow(2.0);
    
    /* Statistics */
    auto sum = c.sum();
    auto mean = c.mean();
    
    /* Print */
    sum.print();
    mean.print();
    
    return 0;
}
```
