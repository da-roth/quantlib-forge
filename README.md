# QuantLib-Forge

Forge integration layer for QuantLib - enabling high-performance automatic differentiation (AAD) for financial derivatives pricing.

## Overview

This project provides:

1. **Integration Layer** - Connects Forge's JIT-compiled AAD with QuantLib's pricing engines
2. **ABool Patch** - Minimal patch for QuantLib to support conditional recording via `ABool`
3. **Benchmark Harness** - Compare performance: Forge AAD vs finite differences vs XAD

## Dependencies

- **[Forge](https://github.com/da-roth/forge)** - JIT compiler and AAD engine (zlib license)
- **[QuantLib](https://github.com/lballabio/QuantLib)** - Financial library (permissive license)
- **Boost** - Required by QuantLib

## Building

```bash
# Clone workspace
git clone <this-repo> quantlib-forge
cd quantlib-forge

# Build Forge package first (see forge/tools/packaging)
# Then configure and build:

cmake -B build -S . \
    -DForge_DIR=<path-to-forge-install>/lib/cmake/Forge \
    -DQuantLib_DIR=<path-to-quantlib-install>/lib/cmake/QuantLib

cmake --build build
```

## Project Structure

```
quantlib-forge/
  LICENSE              # AGPL-3.0-or-later
  NOTICE.md            # Third-party attributions
  CMakeLists.txt       # Main build configuration
  docs/                # Documentation
  patches/             # QuantLib ABool patch
  integration/         # Forge <-> QuantLib adapter code
    ql/                # QuantLib-specific headers
  harness/             # Benchmark harness (engine-agnostic)
  benchmarks/          # Benchmark executables
  examples/            # Usage examples
  cmake/               # CMake modules and presets
```

## License

AGPL-3.0-or-later. See [LICENSE](LICENSE) and [NOTICE.md](NOTICE.md) for details.

Some code is derived from [QuantLib-Risks-Cpp](https://github.com/auto-differentiation/QuantLib-Risks-Cpp) (Xcelerit Computing Ltd., AGPL-3.0).
