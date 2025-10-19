/**
 * @file tensr.hpp
 * @brief C++ interface for Tensr - A powerful, superfast multidimensional tensor library
 * @author Muhammad Fiaz
 * @version 0.0.0
 * @license Apache-2.0
 */

#ifndef TENSR_HPP
#define TENSR_HPP

#include "tensr.h"
#include <vector>
#include <initializer_list>
#include <memory>

namespace tensr {

enum class DType {
    Float32 = TENSR_FLOAT32,
    Float64 = TENSR_FLOAT64,
    Int32 = TENSR_INT32,
    Int64 = TENSR_INT64,
    UInt8 = TENSR_UINT8,
    Bool = TENSR_BOOL
};

enum class Device {
    CPU = TENSR_CPU,
    CUDA = TENSR_CUDA,
    XPU = TENSR_XPU,
    NPU = TENSR_NPU,
    TPU = TENSR_TPU
};

class Tensor {
private:
    ::Tensor* tensor_;

public:
    Tensor(::Tensor* t) : tensor_(t) {}
    Tensor(const std::vector<size_t>& shape, DType dtype = DType::Float32, Device device = Device::CPU);
    Tensor(const Tensor& other);
    Tensor(Tensor&& other) noexcept;
    ~Tensor();

    Tensor& operator=(const Tensor& other);
    Tensor& operator=(Tensor&& other) noexcept;

    /* Factory methods */
    static Tensor zeros(const std::vector<size_t>& shape, DType dtype = DType::Float32, Device device = Device::CPU);
    static Tensor ones(const std::vector<size_t>& shape, DType dtype = DType::Float32, Device device = Device::CPU);
    static Tensor full(const std::vector<size_t>& shape, double value, DType dtype = DType::Float32, Device device = Device::CPU);
    static Tensor arange(double start, double stop, double step = 1.0, DType dtype = DType::Float32, Device device = Device::CPU);
    static Tensor linspace(double start, double stop, size_t num, DType dtype = DType::Float32, Device device = Device::CPU);
    static Tensor eye(size_t n, DType dtype = DType::Float32, Device device = Device::CPU);
    static Tensor rand(const std::vector<size_t>& shape, Device device = Device::CPU);
    static Tensor randn(const std::vector<size_t>& shape, Device device = Device::CPU);
    static Tensor randint(int low, int high, const std::vector<size_t>& shape, Device device = Device::CPU);

    /* Shape operations */
    Tensor reshape(const std::vector<size_t>& new_shape) const;
    Tensor transpose(const std::vector<size_t>& axes = {}) const;
    Tensor squeeze(int axis = -1) const;
    Tensor expand_dims(int axis) const;

    /* Arithmetic operations */
    Tensor operator+(const Tensor& other) const;
    Tensor operator-(const Tensor& other) const;
    Tensor operator*(const Tensor& other) const;
    Tensor operator/(const Tensor& other) const;
    Tensor operator-() const;

    Tensor pow(double exponent) const;
    Tensor sqrt() const;
    Tensor exp() const;
    Tensor log() const;
    Tensor abs() const;

    /* Reduction operations */
    Tensor sum(const std::vector<int>& axes = {}, bool keepdims = false) const;
    Tensor mean(const std::vector<int>& axes = {}, bool keepdims = false) const;
    Tensor max(const std::vector<int>& axes = {}, bool keepdims = false) const;
    Tensor min(const std::vector<int>& axes = {}, bool keepdims = false) const;
    Tensor argmax(int axis = -1) const;
    Tensor argmin(int axis = -1) const;

    /* Linear algebra */
    Tensor dot(const Tensor& other) const;
    Tensor matmul(const Tensor& other) const;
    Tensor inv() const;
    Tensor det() const;

    /* Trigonometric functions */
    Tensor sin() const;
    Tensor cos() const;
    Tensor tan() const;
    Tensor arcsin() const;
    Tensor arccos() const;
    Tensor arctan() const;

    /* Comparison operations */
    Tensor operator==(const Tensor& other) const;
    Tensor operator!=(const Tensor& other) const;
    Tensor operator>(const Tensor& other) const;
    Tensor operator<(const Tensor& other) const;
    Tensor operator>=(const Tensor& other) const;
    Tensor operator<=(const Tensor& other) const;

    /* Indexing */
    double get(const std::vector<size_t>& indices) const;
    void set(const std::vector<size_t>& indices, double value);

    /* Device management */
    void to(Device device, int device_id = 0);
    Device device() const;

    /* I/O */
    void save(const char* filename) const;
    static Tensor load(const char* filename);
    void print() const;

    /* Properties */
    std::vector<size_t> shape() const;
    size_t ndim() const;
    size_t size() const;
    DType dtype() const;

    ::Tensor* raw() const { return tensor_; }
};

/* Global functions */
Tensor concat(const std::vector<Tensor>& tensors, int axis);
Tensor stack(const std::vector<Tensor>& tensors, int axis);
Tensor vstack(const std::vector<Tensor>& tensors);
Tensor hstack(const std::vector<Tensor>& tensors);

void seed(unsigned int s);
void synchronize(Device device, int device_id = 0);
int device_count(Device device);

} // namespace tensr

#endif /* TENSR_HPP */
