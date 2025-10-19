# C++ API Reference

## Tensor Class

### Constructor
```cpp
Tensor(const std::vector<size_t>& shape, DType dtype = DType::Float32, Device device = Device::CPU);
```

### Factory Methods

#### zeros
```cpp
static Tensor zeros(const std::vector<size_t>& shape, DType dtype = DType::Float32, Device device = Device::CPU);
```

#### ones
```cpp
static Tensor ones(const std::vector<size_t>& shape, DType dtype = DType::Float32, Device device = Device::CPU);
```

#### rand
```cpp
static Tensor rand(const std::vector<size_t>& shape, Device device = Device::CPU);
```

### Operators

#### Addition
```cpp
Tensor operator+(const Tensor& other) const;
```

#### Subtraction
```cpp
Tensor operator-(const Tensor& other) const;
```

#### Multiplication
```cpp
Tensor operator*(const Tensor& other) const;
```

#### Division
```cpp
Tensor operator/(const Tensor& other) const;
```

### Methods

#### sum
```cpp
Tensor sum(const std::vector<int>& axes = {}, bool keepdims = false) const;
```

#### mean
```cpp
Tensor mean(const std::vector<int>& axes = {}, bool keepdims = false) const;
```

#### matmul
```cpp
Tensor matmul(const Tensor& other) const;
```

#### print
```cpp
void print() const;
```
