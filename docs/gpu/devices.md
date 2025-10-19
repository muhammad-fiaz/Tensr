# Device Management

## Supported Devices

- **CPU**: Standard CPU execution
- **CUDA**: NVIDIA GPU
- **XPU**: Intel XPU
- **NPU**: Neural Processing Unit
- **TPU**: Tensor Processing Unit

## Device Selection

```c
Tensor* t_cpu = tensr_zeros(shape, 2, TENSR_FLOAT32, TENSR_CPU);
Tensor* t_cuda = tensr_zeros(shape, 2, TENSR_FLOAT32, TENSR_CUDA);
```

## Best Practices

- Use GPU for large tensors (>1000 elements)
- Transfer data in batches
- Synchronize after GPU operations
- Check device availability before use
