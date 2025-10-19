# Performance Tips

## GPU Optimization

1. **Batch Operations**: Process multiple tensors together
2. **Minimize Transfers**: Keep data on GPU
3. **Use Appropriate Types**: float32 is faster than float64
4. **Async Operations**: Use streams for parallelism

## Memory Management

- Reuse tensors when possible
- Free unused tensors immediately
- Use memory pools for frequent allocations

## Profiling

Profile your code to identify bottlenecks:

```c
/* Time critical sections */
clock_t start = clock();
Tensor* result = tensr_matmul(a, b);
clock_t end = clock();
double time = (double)(end - start) / CLOCKS_PER_SEC;
```
