# Reduction Operations

## Aggregation Functions

### sum - Sum of elements

=== "C"
    ```c
    Tensor* t = tensr_ones((size_t[]){2, 3}, 2, TENSR_FLOAT32, TENSR_CPU);
    Tensor* total = tensr_sum(t, NULL, 0, false);  /* Sum all elements = 6.0 */
    ```

=== "C++"
    ```cpp
    auto t = tensr::Tensor::ones({2, 3});
    auto total = t.sum();
    ```

### mean - Average value

=== "C"
    ```c
    Tensor* t = tensr_ones((size_t[]){2, 3}, 2, TENSR_FLOAT32, TENSR_CPU);
    Tensor* avg = tensr_mean(t, NULL, 0, false);  /* Mean = 1.0 */
    ```

=== "C++"
    ```cpp
    auto t = tensr::Tensor::ones({2, 3});
    auto avg = t.mean();
    ```

### max - Maximum value

=== "C"
    ```c
    Tensor* t = tensr_from_array((size_t[]){5}, 1, TENSR_FLOAT32, TENSR_CPU, 
                                 (float[]){1, 5, 3, 9, 2});
    Tensor* max_val = tensr_max(t, NULL, 0, false);  /* Max = 9.0 */
    ```

=== "C++"
    ```cpp
    auto t = tensr::Tensor::rand({5});
    auto max_val = t.max();
    ```

### min - Minimum value

=== "C"
    ```c
    Tensor* t = tensr_from_array((size_t[]){5}, 1, TENSR_FLOAT32, TENSR_CPU, 
                                 (float[]){1, 5, 3, 9, 2});
    Tensor* min_val = tensr_min(t, NULL, 0, false);  /* Min = 1.0 */
    ```

=== "C++"
    ```cpp
    auto t = tensr::Tensor::rand({5});
    auto min_val = t.min();
    ```

## Index Operations

### argmax - Index of maximum

=== "C"
    ```c
    Tensor* t = tensr_from_array((size_t[]){5}, 1, TENSR_FLOAT32, TENSR_CPU, 
                                 (float[]){1, 5, 3, 9, 2});
    Tensor* idx = tensr_argmax(t, -1);  /* Index = 3 */
    ```

=== "C++"
    ```cpp
    auto t = tensr::Tensor::rand({5});
    auto idx = t.argmax();
    ```

### argmin - Index of minimum

=== "C"
    ```c
    Tensor* t = tensr_from_array((size_t[]){5}, 1, TENSR_FLOAT32, TENSR_CPU, 
                                 (float[]){1, 5, 3, 9, 2});
    Tensor* idx = tensr_argmin(t, -1);  /* Index = 0 */
    ```

=== "C++"
    ```cpp
    auto t = tensr::Tensor::rand({5});
    auto idx = t.argmin();
    ```

## Complete Example

```c
#include <tensr/tensr.h>
#include <tensr/tensr_array.h>

int main() {
    /* Create a matrix */
    float data[] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    Tensor* matrix = tensr_from_array((size_t[]){3, 3}, 2, TENSR_FLOAT32, TENSR_CPU, data);
    
    /* Compute statistics */
    Tensor* sum = tensr_sum(matrix, NULL, 0, false);
    Tensor* mean = tensr_mean(matrix, NULL, 0, false);
    Tensor* max_val = tensr_max(matrix, NULL, 0, false);
    Tensor* min_val = tensr_min(matrix, NULL, 0, false);
    
    /* Find indices */
    Tensor* max_idx = tensr_argmax(matrix, -1);
    Tensor* min_idx = tensr_argmin(matrix, -1);
    
    /* Print results */
    printf("Sum: ");
    tensr_print(sum);
    
    printf("Mean: ");
    tensr_print(mean);
    
    printf("Max: ");
    tensr_print(max_val);
    
    printf("Min: ");
    tensr_print(min_val);
    
    printf("Max index: ");
    tensr_print(max_idx);
    
    printf("Min index: ");
    tensr_print(min_idx);
    
    /* Cleanup */
    tensr_free(matrix);
    tensr_free(sum);
    tensr_free(mean);
    tensr_free(max_val);
    tensr_free(min_val);
    tensr_free(max_idx);
    tensr_free(min_idx);
    
    return 0;
}
```

## Use Cases

### Data Analysis

```c
/* Analyze dataset */
Tensor* data = tensr_rand((size_t[]){1000}, 1, TENSR_CPU);

Tensor* mean = tensr_mean(data, NULL, 0, false);
Tensor* max = tensr_max(data, NULL, 0, false);
Tensor* min = tensr_min(data, NULL, 0, false);

printf("Dataset statistics:\n");
printf("Mean: ");
tensr_print(mean);
printf("Range: [");
tensr_print(min);
printf(", ");
tensr_print(max);
printf("]\n");
```

### Finding Extrema

```c
/* Find best prediction */
Tensor* predictions = tensr_rand((size_t[]){10}, 1, TENSR_CPU);
Tensor* best_idx = tensr_argmax(predictions, -1);

printf("Best prediction at index: ");
tensr_print(best_idx);
```
