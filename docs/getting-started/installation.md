# Installation

## Requirements

- C11 compiler (GCC, Clang, MSVC)
- C++17 compiler (for C++ API)
- xmake (optional, for building from source)
- CUDA Toolkit (optional, for GPU support)

## Download Pre-built Binaries

Download the latest release from [GitHub Releases](https://github.com/muhammad-fiaz/tensr/releases):

- **Linux**: `tensr-linux-x64.tar.gz`
- **Windows**: `tensr-windows-x64.zip`
- **macOS**: `tensr-macos-x64.tar.gz`

### Linux/macOS Installation

```bash
# Download and extract
wget https://github.com/muhammad-fiaz/tensr/releases/download/v0.0.0/tensr-linux-x64.tar.gz
tar -xzf tensr-linux-x64.tar.gz

# Install system-wide
sudo cp -r lib/* /usr/local/lib/
sudo cp -r include/* /usr/local/include/

# Or install locally
cp -r lib/* ~/.local/lib/
cp -r include/* ~/.local/include/
```

### Windows Installation

1. Download `tensr-windows-x64.zip`
2. Extract to a directory (e.g., `C:\tensr`)
3. Add to your project:
   - Include directories: `C:\tensr\include`
   - Library directories: `C:\tensr\lib`
   - Link: `tensr.lib`

## For xmake Users

Download `tensr-xmake-{version}.tar.gz` from releases:

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

## Build From Source

```bash
git clone https://github.com/muhammad-fiaz/tensr.git
cd tensr
xmake build
xmake install
```

## Verify Installation

```c
#include <tensr/tensr.h>
#include <stdio.h>

int main() {
    size_t shape[] = {2, 2};
    Tensor* t = tensr_ones(shape, 2, TENSR_FLOAT32, TENSR_CPU);
    tensr_print(t);
    tensr_free(t);
    printf("Tensr is working!\n");
    return 0;
}
```

Compile:

```bash
# Linux/macOS
gcc test.c -ltensr -o test
./test

# Windows (MSVC)
cl test.c tensr.lib
test.exe
```
