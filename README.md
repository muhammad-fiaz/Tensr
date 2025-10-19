<div align="center">

# Tensr

**A Powerful, Superfast Multidimensional Tensor Library for C/C++**

[![Version](https://img.shields.io/github/v/release/muhammad-fiaz/tensr)](https://github.com/muhammad-fiaz/tensr/releases)
[![License](https://img.shields.io/github/license/muhammad-fiaz/tensr)](LICENSE)
[![Last Commit](https://img.shields.io/github/last-commit/muhammad-fiaz/tensr)](https://github.com/muhammad-fiaz/tensr/commits/main)
[![Issues](https://img.shields.io/github/issues/muhammad-fiaz/tensr)](https://github.com/muhammad-fiaz/tensr/issues)
[![Pull Requests](https://img.shields.io/github/issues-pr/muhammad-fiaz/tensr)](https://github.com/muhammad-fiaz/tensr/pulls)
[![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20Linux%20%7C%20macOS-lightgrey.svg)](https://github.com/muhammad-fiaz/tensr)
[![Tests](https://github.com/muhammad-fiaz/Tensr/actions/workflows/test.yml/badge.svg)](https://github.com/muhammad-fiaz/Tensr/actions/workflows/test.yml)
[![Release](https://github.com/muhammad-fiaz/Tensr/actions/workflows/release.yml/badge.svg)](https://github.com/muhammad-fiaz/Tensr/actions/workflows/release.yml)
[![Deploy Documentation](https://github.com/muhammad-fiaz/Tensr/actions/workflows/docs.yml/badge.svg)](https://github.com/muhammad-fiaz/Tensr/actions/workflows/docs.yml)

[📚 Documentation](https://muhammad-fiaz.github.io/tensr/) • [🚀 Quick Start](#-quick-start) • [🤝 Contributing](CONTRIBUTING.md)

</div>

## 🌟 Features

- **High Performance**: Optimized for speed with SIMD and multi-threading support
- **GPU Acceleration**: Full CUDA support for GPU computing (CUDA, XPU, NPU, TPU)
- **Simple API**: Clean, intuitive C and C++ interfaces
- **Comprehensive**: All essential tensor operations for scientific computing
- **Production Ready**: Thoroughly tested, documented, and battle-tested
- **Cross-Platform**: Works on Windows, Linux, and macOS
- **Zero Dependencies**: Core library has no external dependencies
- **Memory Efficient**: Smart memory management with minimal overhead

## 📦 Installation

### Download Pre-built Binaries

Download the latest release for your platform from [GitHub Releases](https://github.com/muhammad-fiaz/tensr/releases):

- **Linux**: `tensr-linux-x64.tar.gz`
- **Windows**: `tensr-windows-x64.zip`
- **macOS**: `tensr-macos-x64.tar.gz`

Extract and copy to your system:

**Linux/macOS:**
```bash
tar -xzf tensr-linux-x64.tar.gz
sudo cp -r lib/* /usr/local/lib/
sudo cp -r include/* /usr/local/include/
```

**Windows:**
Extract the zip file and add the `lib` and `include` directories to your project paths.

### Using xmake (Recommended)

For xmake users, download `tensr-xmake-{version}.tar.gz` from releases:

```bash
tar -xzf tensr-xmake-0.0.0.tar.gz
cd tensr-xmake-0.0.0
xmake install
```

Or add to your `xmake.lua`:

```lua
add_includedirs("/path/to/tensr/include")
add_linkdirs("/path/to/tensr/lib")
add_links("tensr")
```

### From Source

```bash
git clone https://github.com/muhammad-fiaz/tensr.git
cd tensr
xmake build
xmake install
```

## 🚀 Quick Start

### C API

```c
#include <tensr/tensr.h>
#include <tensr/tensr_array.h>

int main() {
    /* Create arrays from data (like np.array) */
    float data_a[] = {1, 2, 3};
    float data_b[] = {4, 5, 6};
    size_t shape[] = {3};
    
    Tensor* a = tensr_from_array(shape, 1, TENSR_FLOAT32, TENSR_CPU, data_a);
    Tensor* b = tensr_from_array(shape, 1, TENSR_FLOAT32, TENSR_CPU, data_b);
    
    /* Element-wise operations */
    Tensor* sum = tensr_add(a, b);      /* a + b */
    Tensor* product = tensr_mul(a, b);  /* a * b */
    Tensor* squared = tensr_pow(a, 2.0); /* a ** 2 */
    
    /* Print results */
    tensr_print(sum);
    tensr_print(product);
    tensr_print(squared);
    
    /* Cleanup */
    tensr_free(a);
    tensr_free(b);
    tensr_free(sum);
    tensr_free(product);
    tensr_free(squared);
    
    return 0;
}
```

### C++ API

```cpp
#include <tensr/tensr.hpp>

int main() {
    /* Create tensors */
    auto t1 = tensr::Tensor::ones({3, 3});
    auto t2 = tensr::Tensor::rand({3, 3});
    
    /* Perform operations */
    auto result = t1 + t2;
    auto sum = result.sum();
    
    /* Print result */
    result.print();
    
    return 0;
}
```

## 🎯 Core Operations

### Tensor Creation

- `zeros()` - Create tensor filled with zeros
- `ones()` - Create tensor filled with ones
- `full()` - Create tensor filled with a value
- `arange()` - Create tensor with evenly spaced values
- `linspace()` - Create tensor with linearly spaced values
- `eye()` - Create identity matrix
- `rand()` - Create tensor with random values
- `randn()` - Create tensor with normal distribution

### Arithmetic Operations

- `add()`, `sub()`, `mul()`, `div()` - Element-wise operations
- `pow()`, `sqrt()`, `exp()`, `log()` - Mathematical functions
- `sin()`, `cos()`, `tan()` - Trigonometric functions
- `abs()`, `neg()` - Unary operations

### Linear Algebra

- `dot()` - Dot product
- `matmul()` - Matrix multiplication
- `inv()` - Matrix inverse
- `det()` - Determinant
- `svd()` - Singular value decomposition
- `eig()` - Eigenvalues and eigenvectors

### Reduction Operations

- `sum()` - Sum of elements
- `mean()` - Mean of elements
- `max()`, `min()` - Maximum and minimum
- `argmax()`, `argmin()` - Indices of max/min

### Shape Manipulation

- `reshape()` - Change tensor shape
- `transpose()` - Transpose dimensions
- `squeeze()` - Remove single dimensions
- `expand_dims()` - Add dimensions
- `concat()`, `stack()` - Combine tensors

## 🖥️ GPU Support

Tensr supports multiple accelerator backends:

```c
/* CUDA GPU */
Tensor* t = tensr_zeros(shape, 2, TENSR_FLOAT32, TENSR_CUDA);

/* Transfer between devices */
tensr_to_device(t, TENSR_CUDA, 0);
```

Supported devices:
- **CPU**: Standard CPU execution
- **CUDA**: NVIDIA GPU acceleration
- **XPU**: Intel XPU support
- **NPU**: Neural Processing Unit
- **TPU**: Tensor Processing Unit

## 📊 Performance

Tensr is designed for maximum performance:

- **SIMD Optimizations**: Vectorized operations for CPU
- **GPU Acceleration**: CUDA kernels for parallel computing
- **Memory Efficiency**: Minimal allocations and smart caching
- **Multi-threading**: Parallel execution for large tensors

## 🧪 Testing

Run the test suite:

```bash
xmake build tests
xmake run tests
```

All tests must pass before release.

## 📖 Documentation

Full documentation is available at [https://muhammad-fiaz.github.io/tensr/](https://muhammad-fiaz.github.io/tensr/)

### Do's ✅

- Use appropriate data types for your use case
- Free tensors when done to avoid memory leaks
- Check return values for NULL
- Use GPU for large-scale computations
- Profile your code for bottlenecks

### Don'ts ❌

- Don't mix tensors from different devices without transfer
- Don't modify tensor data directly
- Don't forget to synchronize after GPU operations
- Don't use debug builds in production
- Don't ignore compiler warnings

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## 📄 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## 👤 Author

**Muhammad Fiaz**

- GitHub: [@muhammad-fiaz](https://github.com/muhammad-fiaz)
- Email: contact@muhammadfiaz.com

## 🙏 Acknowledgments

Special thanks to all contributors and the open-source community.

## 📮 Support

- 📧 Email: contact@muhammadfiaz.com
- 🐛 Issues: [GitHub Issues](https://github.com/muhammad-fiaz/tensr/issues)
- 💬 Discussions: [GitHub Discussions](https://github.com/muhammad-fiaz/tensr/discussions)

---

## 🐛 Bug Reports

Found a bug? Please open an issue on [GitHub](https://github.com/muhammad-fiaz/Tensr/issues).

---

<div align="center">

[![Star History Chart](https://api.star-history.com/svg?repos=muhammad-fiaz/Tensr&type=Date&bg=transparent)](https://github.com/muhammad-fiaz/Tensr/)

**⭐ Star the repository if you find Tensr useful!**

</div>