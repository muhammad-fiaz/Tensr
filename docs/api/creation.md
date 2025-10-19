# Tensor Creation

## Creating Tensors from Data

### from_array - Create from existing data

Create tensors from C arrays.

=== "C"
    ```c
    #include <tensr/tensr_array.h>
    
    /* 1D array */
    float data[] = {1, 2, 3, 4, 5};
    size_t shape[] = {5};
    Tensor* t = tensr_from_array(shape, 1, TENSR_FLOAT32, TENSR_CPU, data);
    
    /* 2D array (matrix) */
    float matrix[] = {1, 2, 3, 4, 5, 6};
    size_t shape2d[] = {2, 3};
    Tensor* mat = tensr_from_array(shape2d, 2, TENSR_FLOAT32, TENSR_CPU, matrix);
    ```

=== "C++"
    ```cpp
    /* C++ uses factory methods */
    auto t = tensr::Tensor::zeros({2, 3});
    ```

## Initialization Functions

### zeros - All zeros

=== "C"
    ```c
    size_t shape[] = {3, 3};
    Tensor* t = tensr_zeros(shape, 2, TENSR_FLOAT32, TENSR_CPU);
    tensr_print(t);
    tensr_free(t);
    ```

=== "C++"
    ```cpp
    auto t = tensr::Tensor::zeros({3, 3});
    t.print();
    ```



### ones - All ones

=== "C"
    ```c
    size_t shape[] = {2, 4};
    Tensor* t = tensr_ones(shape, 2, TENSR_FLOAT32, TENSR_CPU);
    ```

=== "C++"
    ```cpp
    auto t = tensr::Tensor::ones({2, 4});
    ```



### full - Fill with value

=== "C"
    ```c
    size_t shape[] = {3, 3};
    Tensor* t = tensr_full(shape, 2, 5.0, TENSR_FLOAT32, TENSR_CPU);
    ```

=== "C++"
    ```cpp
    auto t = tensr::Tensor::full({3, 3}, 5.0);
    ```



## Range Functions

### arange - Evenly spaced values

=== "C"
    ```c
    /* Create [0, 2, 4, 6, 8] */
    Tensor* t = tensr_arange(0.0, 10.0, 2.0, TENSR_FLOAT32, TENSR_CPU);
    ```

=== "C++"
    ```cpp
    auto t = tensr::Tensor::arange(0.0, 10.0, 2.0);
    ```



### linspace - Linearly spaced values

=== "C"
    ```c
    /* Create 5 values from 0 to 1 */
    Tensor* t = tensr_linspace(0.0, 1.0, 5, TENSR_FLOAT32, TENSR_CPU);
    /* Result: [0.0, 0.25, 0.5, 0.75, 1.0] */
    ```

=== "C++"
    ```cpp
    auto t = tensr::Tensor::linspace(0.0, 1.0, 5);
    ```



## Special Matrices

### eye - Identity matrix

=== "C"
    ```c
    /* Create 3x3 identity matrix */
    Tensor* t = tensr_eye(3, TENSR_FLOAT32, TENSR_CPU);
    /* Result:
       [[1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]] */
    ```

=== "C++"
    ```cpp
    auto t = tensr::Tensor::eye(3);
    ```



## Complete Example

```c
#include <tensr/tensr.h>
#include <tensr/tensr_array.h>

int main() {
    /* Create from data */
    float data[] = {1, 2, 3, 4, 5, 6};
    size_t shape[] = {2, 3};
    Tensor* a = tensr_from_array(shape, 2, TENSR_FLOAT32, TENSR_CPU, data);
    
    /* Create initialized tensors */
    Tensor* zeros = tensr_zeros(shape, 2, TENSR_FLOAT32, TENSR_CPU);
    Tensor* ones = tensr_ones(shape, 2, TENSR_FLOAT32, TENSR_CPU);
    
    /* Create ranges */
    Tensor* range = tensr_arange(0.0, 10.0, 1.0, TENSR_FLOAT32, TENSR_CPU);
    
    /* Create identity */
    Tensor* eye = tensr_eye(3, TENSR_FLOAT32, TENSR_CPU);
    
    /* Print and cleanup */
    tensr_print(a);
    tensr_free(a);
    tensr_free(zeros);
    tensr_free(ones);
    tensr_free(range);
    tensr_free(eye);
    
    return 0;
}
```

## Memory Management

!!! warning "Always Free Tensors"
    Always call `tensr_free()` when done with a tensor to prevent memory leaks.
    
    ```c
    Tensor* t = tensr_zeros(shape, 2, TENSR_FLOAT32, TENSR_CPU);
    /* Use tensor */
    tensr_free(t);  /* Don't forget! */
    ```

## Data Types

Available data types:

- `TENSR_FLOAT32` - 32-bit float (default)
- `TENSR_FLOAT64` - 64-bit double
- `TENSR_INT32` - 32-bit integer
- `TENSR_INT64` - 64-bit integer
- `TENSR_UINT8` - 8-bit unsigned integer
- `TENSR_BOOL` - Boolean

## Device Selection

Create tensors on different devices:

```c
/* CPU */
Tensor* t_cpu = tensr_zeros(shape, 2, TENSR_FLOAT32, TENSR_CPU);

/* CUDA GPU */
Tensor* t_gpu = tensr_zeros(shape, 2, TENSR_FLOAT32, TENSR_CUDA);
```
