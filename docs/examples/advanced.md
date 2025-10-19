# Advanced Examples

## Custom Computation Pipeline

```c
#include <tensr/tensr.h>

int main() {
    size_t shape[] = {100, 100};
    
    /* Create input data */
    Tensor* x = tensr_randn(shape, 2, TENSR_CPU);
    Tensor* w = tensr_randn(shape, 2, TENSR_CPU);
    
    /* Forward pass */
    Tensor* h = tensr_matmul(x, w);
    Tensor* h_relu = tensr_max(h, NULL, 0, false);
    
    /* Compute statistics */
    Tensor* mean = tensr_mean(h_relu, NULL, 0, false);
    Tensor* sum = tensr_sum(h_relu, NULL, 0, false);
    
    tensr_print(mean);
    tensr_print(sum);
    
    /* Cleanup */
    tensr_free(x);
    tensr_free(w);
    tensr_free(h);
    tensr_free(h_relu);
    tensr_free(mean);
    tensr_free(sum);
    
    return 0;
}
```

## Batch Processing

```c
#define BATCH_SIZE 32
#define INPUT_SIZE 784
#define OUTPUT_SIZE 10

size_t input_shape[] = {BATCH_SIZE, INPUT_SIZE};
size_t weight_shape[] = {INPUT_SIZE, OUTPUT_SIZE};

Tensor* inputs = tensr_randn(input_shape, 2, TENSR_CPU);
Tensor* weights = tensr_randn(weight_shape, 2, TENSR_CPU);

Tensor* outputs = tensr_matmul(inputs, weights);
Tensor* predictions = tensr_argmax(outputs, 1);

tensr_print(predictions);

tensr_free(inputs);
tensr_free(weights);
tensr_free(outputs);
tensr_free(predictions);
```
