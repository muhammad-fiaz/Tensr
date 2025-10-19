/**
 * @file tensor.cpp
 * @brief C++ object-oriented wrapper for tensor operations
 * @author Muhammad Fiaz
 * 
 * Provides a modern C++ interface with RAII, operator overloading, and
 * STL-style containers for the Tensr library.
 */

#include "tensr/tensr.hpp"
#include <stdexcept>

namespace tensr {

/**
 * @brief Construct a tensor with specified shape and type
 * @param shape Vector of dimension sizes
 * @param dtype Data type (default: Float32)
 * @param device Device type (default: CPU)
 * @throws std::runtime_error if tensor creation fails
 */
Tensor::Tensor(const std::vector<size_t>& shape, DType dtype, Device device) {
    tensor_ = tensr_create(const_cast<size_t*>(shape.data()), shape.size(), 
                          static_cast<TensrDType>(dtype), 
                          static_cast<TensrDevice>(device));
    if (!tensor_) throw std::runtime_error("Failed to create tensor");
}

/**
 * @brief Copy constructor - creates a deep copy
 * @param other Tensor to copy from
 * @throws std::runtime_error if copy fails
 */
Tensor::Tensor(const Tensor& other) {
    tensor_ = tensr_copy(other.tensor_);
    if (!tensor_) throw std::runtime_error("Failed to copy tensor");
}

/**
 * @brief Move constructor - transfers ownership
 * @param other Tensor to move from
 */
Tensor::Tensor(Tensor&& other) noexcept : tensor_(other.tensor_) {
    other.tensor_ = nullptr;
}

/**
 * @brief Destructor - frees tensor resources
 */
Tensor::~Tensor() {
    if (tensor_) tensr_free(tensor_);
}

/**
 * @brief Copy assignment operator
 * @param other Tensor to copy from
 * @return Reference to this tensor
 */
Tensor& Tensor::operator=(const Tensor& other) {
    if (this != &other) {
        if (tensor_) tensr_free(tensor_);
        tensor_ = tensr_copy(other.tensor_);
    }
    return *this;
}

/**
 * @brief Move assignment operator
 * @param other Tensor to move from
 * @return Reference to this tensor
 */
Tensor& Tensor::operator=(Tensor&& other) noexcept {
    if (this != &other) {
        if (tensor_) tensr_free(tensor_);
        tensor_ = other.tensor_;
        other.tensor_ = nullptr;
    }
    return *this;
}

Tensor Tensor::zeros(const std::vector<size_t>& shape, DType dtype, Device device) {
    auto t = tensr_zeros(const_cast<size_t*>(shape.data()), shape.size(),
                        static_cast<TensrDType>(dtype), static_cast<TensrDevice>(device));
    return Tensor(t);
}

Tensor Tensor::ones(const std::vector<size_t>& shape, DType dtype, Device device) {
    auto t = tensr_ones(const_cast<size_t*>(shape.data()), shape.size(),
                       static_cast<TensrDType>(dtype), static_cast<TensrDevice>(device));
    return Tensor(t);
}

Tensor Tensor::full(const std::vector<size_t>& shape, double value, DType dtype, Device device) {
    auto t = tensr_full(const_cast<size_t*>(shape.data()), shape.size(), value,
                       static_cast<TensrDType>(dtype), static_cast<TensrDevice>(device));
    return Tensor(t);
}

Tensor Tensor::arange(double start, double stop, double step, DType dtype, Device device) {
    auto t = tensr_arange(start, stop, step, static_cast<TensrDType>(dtype), 
                         static_cast<TensrDevice>(device));
    return Tensor(t);
}

Tensor Tensor::linspace(double start, double stop, size_t num, DType dtype, Device device) {
    auto t = tensr_linspace(start, stop, num, static_cast<TensrDType>(dtype), 
                           static_cast<TensrDevice>(device));
    return Tensor(t);
}

Tensor Tensor::eye(size_t n, DType dtype, Device device) {
    auto t = tensr_eye(n, static_cast<TensrDType>(dtype), static_cast<TensrDevice>(device));
    return Tensor(t);
}

Tensor Tensor::rand(const std::vector<size_t>& shape, Device device) {
    auto t = tensr_rand(const_cast<size_t*>(shape.data()), shape.size(), 
                       static_cast<TensrDevice>(device));
    return Tensor(t);
}

Tensor Tensor::randn(const std::vector<size_t>& shape, Device device) {
    auto t = tensr_randn(const_cast<size_t*>(shape.data()), shape.size(), 
                        static_cast<TensrDevice>(device));
    return Tensor(t);
}

Tensor Tensor::randint(int low, int high, const std::vector<size_t>& shape, Device device) {
    auto t = tensr_randint(low, high, const_cast<size_t*>(shape.data()), shape.size(), 
                          static_cast<TensrDevice>(device));
    return Tensor(t);
}

Tensor Tensor::reshape(const std::vector<size_t>& new_shape) const {
    auto t = tensr_reshape(tensor_, const_cast<size_t*>(new_shape.data()), new_shape.size());
    return Tensor(t);
}

Tensor Tensor::transpose(const std::vector<size_t>& axes) const {
    auto t = tensr_transpose(tensor_, const_cast<size_t*>(axes.data()), axes.size());
    return Tensor(t);
}

Tensor Tensor::operator+(const Tensor& other) const {
    return Tensor(tensr_add(tensor_, other.tensor_));
}

Tensor Tensor::operator-(const Tensor& other) const {
    return Tensor(tensr_sub(tensor_, other.tensor_));
}

Tensor Tensor::operator*(const Tensor& other) const {
    return Tensor(tensr_mul(tensor_, other.tensor_));
}

Tensor Tensor::operator/(const Tensor& other) const {
    return Tensor(tensr_div(tensor_, other.tensor_));
}

Tensor Tensor::operator-() const {
    return Tensor(tensr_neg(tensor_));
}

Tensor Tensor::pow(double exponent) const {
    return Tensor(tensr_pow(tensor_, exponent));
}

Tensor Tensor::sqrt() const {
    return Tensor(tensr_sqrt(tensor_));
}

Tensor Tensor::exp() const {
    return Tensor(tensr_exp(tensor_));
}

Tensor Tensor::log() const {
    return Tensor(tensr_log(tensor_));
}

Tensor Tensor::abs() const {
    return Tensor(tensr_abs(tensor_));
}

Tensor Tensor::sum(const std::vector<int>& axes, bool keepdims) const {
    auto t = tensr_sum(tensor_, const_cast<int*>(axes.data()), axes.size(), keepdims);
    return Tensor(t);
}

Tensor Tensor::mean(const std::vector<int>& axes, bool keepdims) const {
    auto t = tensr_mean(tensor_, const_cast<int*>(axes.data()), axes.size(), keepdims);
    return Tensor(t);
}

Tensor Tensor::max(const std::vector<int>& axes, bool keepdims) const {
    auto t = tensr_max(tensor_, const_cast<int*>(axes.data()), axes.size(), keepdims);
    return Tensor(t);
}

Tensor Tensor::min(const std::vector<int>& axes, bool keepdims) const {
    auto t = tensr_min(tensor_, const_cast<int*>(axes.data()), axes.size(), keepdims);
    return Tensor(t);
}

Tensor Tensor::dot(const Tensor& other) const {
    return Tensor(tensr_dot(tensor_, other.tensor_));
}

Tensor Tensor::matmul(const Tensor& other) const {
    return Tensor(tensr_matmul(tensor_, other.tensor_));
}

Tensor Tensor::sin() const {
    return Tensor(tensr_sin(tensor_));
}

Tensor Tensor::cos() const {
    return Tensor(tensr_cos(tensor_));
}

double Tensor::get(const std::vector<size_t>& indices) const {
    return tensr_get(tensor_, const_cast<size_t*>(indices.data()), indices.size());
}

void Tensor::set(const std::vector<size_t>& indices, double value) {
    tensr_set(tensor_, const_cast<size_t*>(indices.data()), indices.size(), value);
}

void Tensor::to(Device device, int device_id) {
    tensr_to_device(tensor_, static_cast<TensrDevice>(device), device_id);
}

Device Tensor::device() const {
    return static_cast<Device>(tensor_->device);
}

void Tensor::save(const char* filename) const {
    tensr_save(filename, tensor_);
}

Tensor Tensor::load(const char* filename) {
    return Tensor(tensr_load(filename));
}

void Tensor::print() const {
    tensr_print(tensor_);
}

std::vector<size_t> Tensor::shape() const {
    return std::vector<size_t>(tensor_->shape, tensor_->shape + tensor_->ndim);
}

size_t Tensor::ndim() const {
    return tensor_->ndim;
}

size_t Tensor::size() const {
    return tensor_->size;
}

DType Tensor::dtype() const {
    return static_cast<DType>(tensor_->dtype);
}

void seed(unsigned int s) {
    tensr_seed(s);
}

void synchronize(Device device, int device_id) {
    tensr_synchronize(static_cast<TensrDevice>(device), device_id);
}

int device_count(Device device) {
    return tensr_device_count(static_cast<TensrDevice>(device));
}

} // namespace tensr
