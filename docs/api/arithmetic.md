# Arithmetic Operations

## Element-wise Operations

### add - Addition

=== "C"
    ```c
    Tensor* a = tensr_from_array((size_t[]){3}, 1, TENSR_FLOAT32, TENSR_CPU, (float[]){1, 2, 3});
    Tensor* b = tensr_from_array((size_t[]){3}, 1, TENSR_FLOAT32, TENSR_CPU, (float[]){4, 5, 6});
    Tensor* c = tensr_add(a, b);  /* [5, 7, 9] */
    ```

=== "C++"
    ```cpp
    auto a = tensr::Tensor::ones({3});
    auto b = tensr::Tensor::ones({3});
    auto c = a + b;
    ```

### sub - Subtraction

=== "C"
    ```c
    Tensor* c = tensr_sub(a, b);
    ```

=== "C++"
    ```cpp
    auto c = a - b;
    ```

### mul - Multiplication

=== "C"
    ```c
    Tensor* c = tensr_mul(a, b);
    ```

=== "C++"
    ```cpp
    auto c = a * b;
    ```

### div - Division

=== "C"
    ```c
    Tensor* c = tensr_div(a, b);
    ```

=== "C++"
    ```cpp
    auto c = a / b;
    ```

## Mathematical Functions

### pow - Power

=== "C"
    ```c
    Tensor* squared = tensr_pow(a, 2.0);  /* a^2 */
    Tensor* cubed = tensr_pow(a, 3.0);    /* a^3 */
    ```

=== "C++"
    ```cpp
    auto squared = a.pow(2.0);
    ```

### sqrt - Square root

=== "C"
    ```c
    Tensor* result = tensr_sqrt(a);
    ```

=== "C++"
    ```cpp
    auto result = a.sqrt();
    ```

### exp - Exponential

=== "C"
    ```c
    Tensor* result = tensr_exp(a);
    ```

=== "C++"
    ```cpp
    auto result = a.exp();
    ```

### log - Natural logarithm

=== "C"
    ```c
    Tensor* result = tensr_log(a);
    ```

=== "C++"
    ```cpp
    auto result = a.log();
    ```

### abs - Absolute value

=== "C"
    ```c
    Tensor* result = tensr_abs(a);
    ```

=== "C++"
    ```cpp
    auto result = a.abs();
    ```

### neg - Negation

=== "C"
    ```c
    Tensor* result = tensr_neg(a);  /* -a */
    ```

=== "C++"
    ```cpp
    auto result = -a;
    ```

## Trigonometric Functions

### sin, cos, tan

=== "C"
    ```c
    Tensor* sin_result = tensr_sin(a);
    Tensor* cos_result = tensr_cos(a);
    Tensor* tan_result = tensr_tan(a);
    ```

=== "C++"
    ```cpp
    auto sin_result = a.sin();
    auto cos_result = a.cos();
    auto tan_result = a.tan();
    ```

### arcsin, arccos, arctan

=== "C"
    ```c
    Tensor* asin_result = tensr_arcsin(a);
    Tensor* acos_result = tensr_arccos(a);
    Tensor* atan_result = tensr_arctan(a);
    ```

=== "C++"
    ```cpp
    auto asin_result = a.arcsin();
    auto acos_result = a.arccos();
    auto atan_result = a.arctan();
    ```

## Comparison Operations

### equal, not_equal

=== "C"
    ```c
    Tensor* eq = tensr_equal(a, b);
    Tensor* neq = tensr_not_equal(a, b);
    ```

=== "C++"
    ```cpp
    auto eq = (a == b);
    auto neq = (a != b);
    ```

### greater, less

=== "C"
    ```c
    Tensor* gt = tensr_greater(a, b);
    Tensor* lt = tensr_less(a, b);
    Tensor* gte = tensr_greater_equal(a, b);
    Tensor* lte = tensr_less_equal(a, b);
    ```

=== "C++"
    ```cpp
    auto gt = (a > b);
    auto lt = (a < b);
    auto gte = (a >= b);
    auto lte = (a <= b);
    ```

## Logical Operations

### logical_and, logical_or, logical_not

=== "C"
    ```c
    Tensor* and_result = tensr_logical_and(a, b);
    Tensor* or_result = tensr_logical_or(a, b);
    Tensor* not_result = tensr_logical_not(a);
    ```

## Complete Example

```c
#include <tensr/tensr.h>
#include <tensr/tensr_array.h>

int main() {
    /* Create arrays */
    float data_a[] = {1, 2, 3, 4};
    float data_b[] = {2, 2, 2, 2};
    size_t shape[] = {4};
    
    Tensor* a = tensr_from_array(shape, 1, TENSR_FLOAT32, TENSR_CPU, data_a);
    Tensor* b = tensr_from_array(shape, 1, TENSR_FLOAT32, TENSR_CPU, data_b);
    
    /* Arithmetic */
    Tensor* sum = tensr_add(a, b);        /* [3, 4, 5, 6] */
    Tensor* product = tensr_mul(a, b);    /* [2, 4, 6, 8] */
    Tensor* squared = tensr_pow(a, 2.0);  /* [1, 4, 9, 16] */
    
    /* Math functions */
    Tensor* sqrt_a = tensr_sqrt(a);
    Tensor* exp_a = tensr_exp(a);
    
    /* Print results */
    tensr_print(sum);
    tensr_print(squared);
    
    /* Cleanup */
    tensr_free(a);
    tensr_free(b);
    tensr_free(sum);
    tensr_free(product);
    tensr_free(squared);
    tensr_free(sqrt_a);
    tensr_free(exp_a);
    
    return 0;
}
```

## Broadcasting

Operations automatically broadcast tensors of compatible shapes:

```c
/* Scalar + Vector */
Tensor* vec = tensr_arange(0, 5, 1, TENSR_FLOAT32, TENSR_CPU);
Tensor* scalar = tensr_full((size_t[]){1}, 1, 10.0, TENSR_FLOAT32, TENSR_CPU);
Tensor* result = tensr_add(vec, scalar);  /* Adds 10 to each element */
```
