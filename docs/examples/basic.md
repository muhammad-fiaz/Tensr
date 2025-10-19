# Basic Examples

## Creating Tensors

```c
#include <tensr/tensr.h>

int main() {
    size_t shape[] = {3, 3};
    
    Tensor* zeros = tensr_zeros(shape, 2, TENSR_FLOAT32, TENSR_CPU);
    Tensor* ones = tensr_ones(shape, 2, TENSR_FLOAT32, TENSR_CPU);
    Tensor* rand = tensr_rand(shape, 2, TENSR_CPU);
    
    tensr_print(zeros);
    tensr_print(ones);
    tensr_print(rand);
    
    tensr_free(zeros);
    tensr_free(ones);
    tensr_free(rand);
    
    return 0;
}
```

## Arithmetic Operations

```c
size_t shape[] = {2, 2};
Tensor* a = tensr_full(shape, 2, 3.0, TENSR_FLOAT32, TENSR_CPU);
Tensor* b = tensr_full(shape, 2, 2.0, TENSR_FLOAT32, TENSR_CPU);

Tensor* sum = tensr_add(a, b);
Tensor* product = tensr_mul(a, b);

tensr_print(sum);
tensr_print(product);

tensr_free(a);
tensr_free(b);
tensr_free(sum);
tensr_free(product);
```

## Matrix Operations

```c
Tensor* eye = tensr_eye(3, TENSR_FLOAT32, TENSR_CPU);
size_t shape[] = {3, 3};
Tensor* mat = tensr_rand(shape, 2, TENSR_CPU);

Tensor* result = tensr_matmul(eye, mat);
tensr_print(result);

tensr_free(eye);
tensr_free(mat);
tensr_free(result);
```
