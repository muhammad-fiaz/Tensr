# Shape Manipulation

## reshape - Change tensor shape

Change the shape of a tensor without changing its data.

=== "C"
    ```c
    /* Create 1D tensor with 6 elements */
    Tensor* t = tensr_arange(0, 6, 1, TENSR_FLOAT32, TENSR_CPU);
    
    /* Reshape to 2x3 matrix */
    Tensor* reshaped = tensr_reshape(t, (size_t[]){2, 3}, 2);
    tensr_print(reshaped);
    ```

=== "C++"
    ```cpp
    auto t = tensr::Tensor::arange(0, 6, 1);
    auto reshaped = t.reshape({2, 3});
    ```

## transpose - Permute dimensions

Reverse or permute the dimensions of a tensor.

=== "C"
    ```c
    /* Create 2x3 matrix */
    Tensor* t = tensr_ones((size_t[]){2, 3}, 2, TENSR_FLOAT32, TENSR_CPU);
    
    /* Transpose to 3x2 */
    Tensor* transposed = tensr_transpose(t, NULL, 0);
    ```

=== "C++"
    ```cpp
    auto t = tensr::Tensor::ones({2, 3});
    auto transposed = t.transpose();
    ```

## squeeze - Remove single dimensions

Remove dimensions of size 1.

=== "C"
    ```c
    /* Create tensor with shape (1, 3, 1) */
    Tensor* t = tensr_ones((size_t[]){1, 3, 1}, 3, TENSR_FLOAT32, TENSR_CPU);
    
    /* Squeeze to shape (3,) */
    Tensor* squeezed = tensr_squeeze(t, -1);
    ```

=== "C++"
    ```cpp
    auto t = tensr::Tensor::ones({1, 3, 1});
    auto squeezed = t.squeeze();
    ```

## expand_dims - Add dimension

Insert a new dimension of size 1.

=== "C"
    ```c
    /* Create 1D tensor with shape (3,) */
    Tensor* t = tensr_ones((size_t[]){3}, 1, TENSR_FLOAT32, TENSR_CPU);
    
    /* Expand to shape (1, 3) */
    Tensor* expanded = tensr_expand_dims(t, 0);
    ```

=== "C++"
    ```cpp
    auto t = tensr::Tensor::ones({3});
    auto expanded = t.expand_dims(0);
    ```

## Complete Example

```c
#include <tensr/tensr.h>

int main() {
    /* Create 1D array */
    Tensor* vec = tensr_arange(0, 12, 1, TENSR_FLOAT32, TENSR_CPU);
    printf("Original shape: ");
    tensr_print(vec);
    
    /* Reshape to 3x4 matrix */
    Tensor* mat = tensr_reshape(vec, (size_t[]){3, 4}, 2);
    printf("Reshaped to 3x4:\n");
    tensr_print(mat);
    
    /* Transpose to 4x3 */
    Tensor* mat_t = tensr_transpose(mat, NULL, 0);
    printf("Transposed to 4x3:\n");
    tensr_print(mat_t);
    
    /* Reshape back to 1D */
    Tensor* flat = tensr_reshape(mat_t, (size_t[]){12}, 1);
    printf("Flattened:\n");
    tensr_print(flat);
    
    /* Cleanup */
    tensr_free(vec);
    tensr_free(mat);
    tensr_free(mat_t);
    tensr_free(flat);
    
    return 0;
}
```

## Use Cases

### Batch Processing

```c
/* Reshape data for batch processing */
Tensor* data = tensr_rand((size_t[]){100}, 1, TENSR_CPU);
Tensor* batched = tensr_reshape(data, (size_t[]){10, 10}, 2);
/* Now have 10 batches of 10 samples each */
```

### Matrix Operations

```c
/* Prepare vectors for matrix multiplication */
Tensor* vec = tensr_rand((size_t[]){5}, 1, TENSR_CPU);
Tensor* col = tensr_expand_dims(vec, 1);  /* Shape: (5, 1) */
Tensor* row = tensr_expand_dims(vec, 0);  /* Shape: (1, 5) */

/* Outer product */
Tensor* outer = tensr_matmul(col, row);  /* Shape: (5, 5) */
```

### Flattening

```c
/* Flatten multi-dimensional tensor */
Tensor* tensor = tensr_rand((size_t[]){2, 3, 4}, 3, TENSR_CPU);
Tensor* flat = tensr_reshape(tensor, (size_t[]){24}, 1);
```
