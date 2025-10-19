# Tensr Documentation

Welcome to **Tensr** - a powerful, superfast multidimensional tensor library for C/C++!

## Overview

Tensr is a high-performance tensor computation library designed for scientific computing, machine learning, and numerical analysis. It provides a clean, intuitive API for both C and C++ with full GPU acceleration support.

## Key Features

- ⚡ **High Performance**: Optimized for speed with SIMD and multi-threading
- 🚀 **GPU Acceleration**: Full CUDA support for parallel computing
- 🎯 **Simple API**: Clean, intuitive interfaces for C and C++
- 📦 **Comprehensive**: All essential tensor operations
- ✅ **Production Ready**: Thoroughly tested and documented
- 🌍 **Cross-Platform**: Windows, Linux, and macOS support
- 🔧 **Zero Dependencies**: No external dependencies for core library

## Quick Example

### C API (NumPy-Style)

```c
#include <tensr/tensr.h>
#include <tensr/tensr_array.h>

int main() {
    /* Create arrays from data (like np.array([1, 2, 3])) */
    float data_a[] = {1, 2, 3};
    float data_b[] = {4, 5, 6};
    size_t shape[] = {3};
    
    Tensor* a = tensr_from_array(shape, 1, TENSR_FLOAT32, TENSR_CPU, data_a);
    Tensor* b = tensr_from_array(shape, 1, TENSR_FLOAT32, TENSR_CPU, data_b);
    
    /* Element-wise operations (like a + b, a * b, a ** 2) */
    Tensor* sum = tensr_add(a, b);
    Tensor* product = tensr_mul(a, b);
    Tensor* squared = tensr_pow(a, 2.0);
    
    /* Print results */
    tensr_print(sum);      /* [5, 7, 9] */
    tensr_print(product);  /* [4, 10, 18] */
    tensr_print(squared);  /* [1, 4, 9] */
    
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
    auto eye = tensr::Tensor::eye(3);
    auto rand_mat = tensr::Tensor::rand({3, 3});
    
    /* Matrix multiplication */
    auto result = eye.matmul(rand_mat);
    
    /* Print result */
    result.print();
    
    return 0;
}
```

## Installation

Download prebuilt binaries from the GitHub Releases page. We publish platform-specific artifacts (headers + static/shared libraries) for each tagged release.

Windows (PowerShell)

```powershell
# 1. Go to: https://github.com/muhammad-fiaz/tensr/releases
# 2. Download the desired `tensr-<version>-windows-x64.zip` asset
Expand-Archive -Path .\tensr-<version>-windows-x64.zip -DestinationPath .\tensr
cd .\tensr
# Headers are in include\, libraries in lib\, and binaries (if any) in bin\
```

Linux / macOS

```bash
# 1. Go to: https://github.com/muhammad-fiaz/tensr/releases
# 2. Download the desired `tensr-<version>-linux-x64.tar.gz` (or macOS tarball)
tar -xzf tensr-<version>-linux-x64.tar.gz
cd tensr-<version>
# Headers in include/, libs in lib/, binaries in bin/
```

If you prefer to build from source, follow the instructions below.

=== "From Source"
    ```bash
    git clone https://github.com/muhammad-fiaz/tensr.git
    cd tensr
    xmake build
    xmake install
    ```

## Core Operations

### Tensor Creation

Create tensors with various initialization methods:

- `zeros()` - All zeros
- `ones()` - All ones
- `full()` - Fill with value
- `arange()` - Evenly spaced values
- `linspace()` - Linearly spaced values
- `eye()` - Identity matrix
- `rand()` - Random uniform
- `randn()` - Random normal

### Arithmetic Operations

Perform element-wise operations:

- Basic: `add()`, `sub()`, `mul()`, `div()`
- Mathematical: `pow()`, `sqrt()`, `exp()`, `log()`
- Trigonometric: `sin()`, `cos()`, `tan()`

### Linear Algebra

Advanced matrix operations:

- `dot()` - Dot product
- `matmul()` - Matrix multiplication
- `inv()` - Matrix inverse
- `det()` - Determinant
- `svd()` - Singular value decomposition
- `eig()` - Eigenvalues and eigenvectors

### Reduction Operations

Aggregate tensor values:

- `sum()` - Sum of elements
- `mean()` - Mean value
- `max()`, `min()` - Maximum and minimum
- `argmax()`, `argmin()` - Indices of extrema

## GPU Support

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

## Performance Tips

!!! tip "Optimization Guidelines"
    - Use GPU for large tensors (>1000 elements)
    - Batch operations when possible
    - Reuse tensors to minimize allocations
    - Use appropriate data types (float32 vs float64)
    - Profile your code to identify bottlenecks

## Best Practices

### Do's ✅

- Free tensors when done to avoid memory leaks
- Check return values for NULL
- Use GPU for large-scale computations
- Profile your code for bottlenecks
- Use appropriate data types

### Don'ts ❌

- Don't mix tensors from different devices without transfer
- Don't modify tensor data directly
- Don't forget to synchronize after GPU operations
- Don't use debug builds in production
- Don't ignore compiler warnings

## Getting Help

- 📚 [Documentation](https://muhammad-fiaz.github.io/tensr/)
- 🐛 [Issue Tracker](https://github.com/muhammad-fiaz/tensr/issues)
- 💬 [Discussions](https://github.com/muhammad-fiaz/tensr/discussions)
- 📧 [Email](mailto:contact@muhammadfiaz.com)

## Contributing

We welcome contributions! See our [Contributing Guide](contributing.md) for details.

## License

Tensr is licensed under the Apache License 2.0. See [LICENSE](license.md) for details.

---

**Author**: Muhammad Fiaz  
**Email**: contact@muhammadfiaz.com  
**GitHub**: [@muhammad-fiaz](https://github.com/muhammad-fiaz)
