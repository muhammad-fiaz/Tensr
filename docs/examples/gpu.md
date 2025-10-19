# GPU Examples

## Basic GPU Usage

```c
#include <tensr/tensr.h>

int main() {
    /* Check CUDA availability */
    int cuda_count = tensr_device_count(TENSR_CUDA);
    printf("CUDA devices available: %d\n", cuda_count);
    
    /* Create tensor on GPU */
    size_t shape[] = {1000, 1000};
    Tensor* a = tensr_rand(shape, 2, TENSR_CUDA);
    Tensor* b = tensr_rand(shape, 2, TENSR_CUDA);
    
    /* GPU operations */
    Tensor* c = tensr_matmul(a, b);
    
    /* Synchronize */
    tensr_synchronize(TENSR_CUDA, 0);
    
    /* Cleanup */
    tensr_free(a);
    tensr_free(b);
    tensr_free(c);
    
    return 0;
}
```

## CPU to GPU Transfer

```c
/* Create on CPU */
size_t shape[] = {100, 100};
Tensor* t_cpu = tensr_ones(shape, 2, TENSR_FLOAT32, TENSR_CPU);

/* Transfer to GPU */
tensr_to_device(t_cpu, TENSR_CUDA, 0);

/* Now tensor is on GPU */
Tensor* result = tensr_mul(t_cpu, t_cpu);

tensr_free(t_cpu);
tensr_free(result);
```
