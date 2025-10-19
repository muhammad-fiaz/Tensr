# I/O Operations

## Saving and Loading

### save - Save tensor to file

Save tensor to binary file for later use.

=== "C"
    ```c
    Tensor* t = tensr_ones((size_t[]){3, 3}, 2, TENSR_FLOAT32, TENSR_CPU);
    int result = tensr_save("tensor.bin", t);
    if (result == 0) {
        printf("Tensor saved successfully\n");
    }
    ```

=== "C++"
    ```cpp
    auto t = tensr::Tensor::ones({3, 3});
    t.save("tensor.bin");
    ```

### load - Load tensor from file

Load a previously saved tensor.

=== "C"
    ```c
    Tensor* t = tensr_load("tensor.bin");
    if (t != NULL) {
        tensr_print(t);
        tensr_free(t);
    }
    ```

=== "C++"
    ```cpp
    auto t = tensr::Tensor::load("tensor.bin");
    t.print();
    ```

## Printing

### print - Display tensor information

Print tensor shape, dtype, device, and data.

=== "C"
    ```c
    Tensor* t = tensr_arange(0, 10, 1, TENSR_FLOAT32, TENSR_CPU);
    tensr_print(t);
    /* Output:
       Tensor(shape=[10], dtype=float32, device=CPU)
       Data: [0.0000, 1.0000, 2.0000, ..., 9.0000]
    */
    ```

=== "C++"
    ```cpp
    auto t = tensr::Tensor::arange(0, 10, 1);
    t.print();
    ```

## Element Access

### get - Get element value

Retrieve value at specific indices.

=== "C"
    ```c
    float data[] = {1, 2, 3, 4, 5, 6};
    Tensor* t = tensr_from_array((size_t[]){2, 3}, 2, TENSR_FLOAT32, TENSR_CPU, data);
    
    double val = tensr_get(t, (size_t[]){1, 2}, 2);
    printf("Value at [1,2]: %f\n", val);  /* 6.0 */
    ```

=== "C++"
    ```cpp
    auto t = tensr::Tensor::ones({2, 3});
    double val = t.get({1, 2});
    ```

### set - Set element value

Set value at specific indices.

=== "C"
    ```c
    Tensor* t = tensr_zeros((size_t[]){2, 3}, 2, TENSR_FLOAT32, TENSR_CPU);
    tensr_set(t, (size_t[]){1, 2}, 2, 5.0);
    tensr_print(t);
    ```

=== "C++"
    ```cpp
    auto t = tensr::Tensor::zeros({2, 3});
    t.set({1, 2}, 5.0);
    ```

## Complete Example

```c
#include <tensr/tensr.h>

int main() {
    /* Create and save tensor */
    Tensor* original = tensr_arange(0, 12, 1, TENSR_FLOAT32, TENSR_CPU);
    Tensor* matrix = tensr_reshape(original, (size_t[]){3, 4}, 2);
    
    printf("Original tensor:\n");
    tensr_print(matrix);
    
    /* Save to file */
    if (tensr_save("matrix.bin", matrix) == 0) {
        printf("\nTensor saved to matrix.bin\n");
    }
    
    /* Load from file */
    Tensor* loaded = tensr_load("matrix.bin");
    if (loaded) {
        printf("\nLoaded tensor:\n");
        tensr_print(loaded);
        
        /* Access elements */
        double val = tensr_get(loaded, (size_t[]){1, 2}, 2);
        printf("\nValue at [1,2]: %f\n", val);
        
        /* Modify element */
        tensr_set(loaded, (size_t[]){1, 2}, 2, 99.0);
        printf("\nAfter modification:\n");
        tensr_print(loaded);
        
        tensr_free(loaded);
    }
    
    tensr_free(original);
    tensr_free(matrix);
    
    return 0;
}
```

## Use Cases

### Checkpointing

```c
/* Save model weights during training */
Tensor* weights = tensr_randn((size_t[]){784, 128}, 2, TENSR_CPU);
tensr_save("checkpoint_epoch_10.bin", weights);

/* Load weights later */
Tensor* loaded_weights = tensr_load("checkpoint_epoch_10.bin");
```

### Data Persistence

```c
/* Save processed data */
Tensor* processed_data = tensr_rand((size_t[]){1000, 100}, 2, TENSR_CPU);
tensr_save("processed_data.bin", processed_data);

/* Load for analysis */
Tensor* data = tensr_load("processed_data.bin");
Tensor* mean = tensr_mean(data, NULL, 0, false);
tensr_print(mean);
```

### Debugging

```c
/* Print intermediate results */
Tensor* a = tensr_rand((size_t[]){3, 3}, 2, TENSR_CPU);
Tensor* b = tensr_rand((size_t[]){3, 3}, 2, TENSR_CPU);

printf("Input A:\n");
tensr_print(a);

printf("Input B:\n");
tensr_print(b);

Tensor* result = tensr_matmul(a, b);
printf("Result:\n");
tensr_print(result);
```

## File Format

The binary file format stores:
1. Number of dimensions (size_t)
2. Data type (TensrDType)
3. Total size (size_t)
4. Shape array (size_t[ndim])
5. Raw data (dtype[size])

This format is portable across platforms with the same endianness.
