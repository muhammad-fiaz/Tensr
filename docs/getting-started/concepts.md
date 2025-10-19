# Basic Concepts

## Tensors

A tensor is a multidimensional array with:
- **Shape**: Dimensions of the tensor
- **Data Type**: float32, float64, int32, int64, uint8, bool
- **Device**: CPU, CUDA, XPU, NPU, TPU
- **Data**: Actual values stored in memory

## Data Types

- `TENSR_FLOAT32` - 32-bit floating point
- `TENSR_FLOAT64` - 64-bit floating point
- `TENSR_INT32` - 32-bit integer
- `TENSR_INT64` - 64-bit integer
- `TENSR_UINT8` - 8-bit unsigned integer
- `TENSR_BOOL` - Boolean

## Devices

- `TENSR_CPU` - CPU execution
- `TENSR_CUDA` - NVIDIA GPU
- `TENSR_XPU` - Intel XPU
- `TENSR_NPU` - Neural Processing Unit
- `TENSR_TPU` - Tensor Processing Unit

## Memory Management

Always free tensors after use:

```c
Tensor* t = tensr_zeros(shape, 2, TENSR_FLOAT32, TENSR_CPU);
/* Use tensor */
tensr_free(t);
```
