# Random Operations

## Random Number Generation

### rand - Uniform distribution

Generate random values from uniform distribution [0, 1).

=== "C"
    ```c
    tensr_seed(42);  /* Set seed for reproducibility */
    Tensor* t = tensr_rand((size_t[]){3, 3}, 2, TENSR_CPU);
    tensr_print(t);
    ```

=== "C++"
    ```cpp
    tensr::seed(42);
    auto t = tensr::Tensor::rand({3, 3});
    t.print();
    ```

### randn - Normal distribution

Generate random values from standard normal distribution (mean=0, std=1).

=== "C"
    ```c
    tensr_seed(42);
    Tensor* t = tensr_randn((size_t[]){3, 3}, 2, TENSR_CPU);
    tensr_print(t);
    ```

=== "C++"
    ```cpp
    tensr::seed(42);
    auto t = tensr::Tensor::randn({3, 3});
    t.print();
    ```

### randint - Random integers

Generate random integers in range [low, high).

=== "C"
    ```c
    tensr_seed(42);
    Tensor* t = tensr_randint(0, 10, (size_t[]){3, 3}, 2, TENSR_CPU);
    tensr_print(t);  /* Random integers from 0 to 9 */
    ```

=== "C++"
    ```cpp
    tensr::seed(42);
    auto t = tensr::Tensor::randint(0, 10, {3, 3});
    t.print();
    ```

## Seeding

### seed - Set random seed

Set the random seed for reproducible results.

=== "C"
    ```c
    tensr_seed(42);
    Tensor* t1 = tensr_rand((size_t[]){5}, 1, TENSR_CPU);
    
    tensr_seed(42);  /* Same seed */
    Tensor* t2 = tensr_rand((size_t[]){5}, 1, TENSR_CPU);
    /* t1 and t2 will have identical values */
    ```

=== "C++"
    ```cpp
    tensr::seed(42);
    auto t1 = tensr::Tensor::rand({5});
    
    tensr::seed(42);
    auto t2 = tensr::Tensor::rand({5});
    /* t1 and t2 will have identical values */
    ```

## Complete Example

```c
#include <tensr/tensr.h>

int main() {
    /* Set seed for reproducibility */
    tensr_seed(42);
    
    /* Uniform random [0, 1) */
    Tensor* uniform = tensr_rand((size_t[]){3, 3}, 2, TENSR_CPU);
    printf("Uniform random:\n");
    tensr_print(uniform);
    
    /* Normal distribution */
    Tensor* normal = tensr_randn((size_t[]){3, 3}, 2, TENSR_CPU);
    printf("\nNormal distribution:\n");
    tensr_print(normal);
    
    /* Random integers [0, 10) */
    Tensor* integers = tensr_randint(0, 10, (size_t[]){3, 3}, 2, TENSR_CPU);
    printf("\nRandom integers:\n");
    tensr_print(integers);
    
    /* Cleanup */
    tensr_free(uniform);
    tensr_free(normal);
    tensr_free(integers);
    
    return 0;
}
```

## Use Cases

### Random Initialization

```c
/* Initialize weights for neural network */
tensr_seed(42);
Tensor* weights = tensr_randn((size_t[]){784, 128}, 2, TENSR_CPU);
Tensor* bias = tensr_zeros((size_t[]){128}, 1, TENSR_FLOAT32, TENSR_CPU);
```

### Data Augmentation

```c
/* Add random noise to data */
Tensor* data = tensr_ones((size_t[]){100, 100}, 2, TENSR_FLOAT32, TENSR_CPU);
Tensor* noise = tensr_randn((size_t[]){100, 100}, 2, TENSR_CPU);
Tensor* augmented = tensr_add(data, noise);
```

### Random Sampling

```c
/* Generate random samples */
tensr_seed(42);
Tensor* samples = tensr_rand((size_t[]){1000}, 1, TENSR_CPU);

/* Compute statistics */
Tensor* mean = tensr_mean(samples, NULL, 0, false);
printf("Sample mean: ");
tensr_print(mean);
```

## GPU Random Generation

Generate random tensors on GPU for better performance:

```c
/* Generate large random tensor on GPU */
tensr_seed(42);
Tensor* gpu_rand = tensr_rand((size_t[]){10000, 10000}, 2, TENSR_CUDA);

/* Use in GPU computations */
Tensor* result = tensr_mul(gpu_rand, gpu_rand);
tensr_synchronize(TENSR_CUDA, 0);
```
