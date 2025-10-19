# C API Reference

## Tensor Creation

### tensr_create
```c
Tensor* tensr_create(size_t* shape, size_t ndim, TensrDType dtype, TensrDevice device);
```
Create an uninitialized tensor.

### tensr_zeros
```c
Tensor* tensr_zeros(size_t* shape, size_t ndim, TensrDType dtype, TensrDevice device);
```
Create a tensor filled with zeros.

### tensr_ones
```c
Tensor* tensr_ones(size_t* shape, size_t ndim, TensrDType dtype, TensrDevice device);
```
Create a tensor filled with ones.

### tensr_full
```c
Tensor* tensr_full(size_t* shape, size_t ndim, double value, TensrDType dtype, TensrDevice device);
```
Create a tensor filled with a specific value.

### tensr_arange
```c
Tensor* tensr_arange(double start, double stop, double step, TensrDType dtype, TensrDevice device);
```
Create a tensor with evenly spaced values.

### tensr_linspace
```c
Tensor* tensr_linspace(double start, double stop, size_t num, TensrDType dtype, TensrDevice device);
```
Create a tensor with linearly spaced values.

### tensr_eye
```c
Tensor* tensr_eye(size_t n, TensrDType dtype, TensrDevice device);
```
Create an identity matrix.

## Arithmetic Operations

### tensr_add
```c
Tensor* tensr_add(const Tensor* a, const Tensor* b);
```
Element-wise addition.

### tensr_sub
```c
Tensor* tensr_sub(const Tensor* a, const Tensor* b);
```
Element-wise subtraction.

### tensr_mul
```c
Tensor* tensr_mul(const Tensor* a, const Tensor* b);
```
Element-wise multiplication.

### tensr_div
```c
Tensor* tensr_div(const Tensor* a, const Tensor* b);
```
Element-wise division.

## Memory Management

### tensr_free
```c
void tensr_free(Tensor* t);
```
Free tensor memory.
