# CUDA Support

## Creating GPU Tensors

**C API:**
```c
size_t shape[] = {1000, 1000};
Tensor* t = tensr_zeros(shape, 2, TENSR_FLOAT32, TENSR_CUDA);
```

**C++ API:**
```cpp
auto t = tensr::Tensor::zeros({1000, 1000}, tensr::DType::Float32, tensr::Device::CUDA);
```

## Device Transfer

**C API:**
```c
tensr_to_device(t, TENSR_CUDA, 0);
```

**C++ API:**
```cpp
t.to(tensr::Device::CUDA, 0);
```

## Synchronization

**C API:**
```c
tensr_synchronize(TENSR_CUDA, 0);
```

**C++ API:**
```cpp
tensr::synchronize(tensr::Device::CUDA, 0);
```

## Device Count

**C API:**
```c
int count = tensr_device_count(TENSR_CUDA);
```

**C++ API:**
```cpp
int count = tensr::device_count(tensr::Device::CUDA);
```
