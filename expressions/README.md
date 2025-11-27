# QuantLib Template Layer

Header-only expression template layer enabling QuantLib integration with custom Automatic Differentiation (AD) backends.

> **⚠️ CRITICAL DEVELOPER NOTE**
>
> **DO NOT MODIFY ANY FUNCTION SIGNATURES, CLASS INTERFACES, OR OPERATOR OVERLOADS.**
>
> This template layer is precisely engineered for QuantLib integration. Any changes to function signatures, template parameters, operator overloads, or class interfaces **WILL BREAK QUANTLIB INTEGRATION**. The expression template system and operator overloading work in delicate balance - even minor changes cause compilation failures or silent bugs.

## Purpose

Provides infrastructure to replace `double` with `AReal<T, N>` in QuantLib, enabling automatic derivative computation (Greeks) without modifying QuantLib source code.

**Derived from**: [XAD](https://github.com/auto-differentiation/xad) by Xcelerit Computing Ltd.

## Architecture

### Core Components

**Expression Templates** (`Expression.hpp`, `ExpressionTemplates/`):
- CRTP-based expression tree building
- Lazy evaluation with automatic derivative propagation
- Binary/unary operators: `+`, `-`, `*`, `/`, `pow()`, `sin()`, `cos()`, `exp()`, `log()`, etc.
- Operator macros generate all type combinations (`AReal`×`AReal`, `AReal`×`double`, etc.)

**AReal Active Type** (`Literals.hpp`):
- Main AD type replacing `double`: `AReal<Scalar, N>`
- Wraps scalar value with derivative tracking and tape slot references
- Maintains `double`-like semantics with conversion operators

**Tape System** (`TapeStub` in `Literals.hpp`):
- **Interface stub only** - all methods throw `NoTapeException`
- Users must provide full `Tape` implementation with:
  - Computational graph recording
  - Memory management
  - Reverse-mode AD algorithm
  - Gradient computation

**Compatibility Layer** (`Compatibility/`):
- `MathFunctions.hpp`: Imports `std::` math functions to `xad::`
- `StdCompatibility.hpp`: Re-exports `xad::` functions to `std::` (enables `std::log(areal_var)`)
- `Complex.hpp`: Support for `std::complex<AReal>`

**Type Traits** (`Traits.hpp`):
- Compile-time type information (`ExprTraits`, `DerivativesTraits`)
- AD mode tracking (forward/reverse/none)

### Directory Structure

```
quantlib-template/
├── Expression.hpp, Literals.hpp, Traits.hpp, Macros.hpp, Exceptions.hpp
├── Config.hpp.in                  # Build configuration (generates Config.hpp)
├── ExpressionTemplates/           # 11 headers: operators, functors, macros
└── Compatibility/                 # 3 headers: std integration
```

## How It Works

```cpp
// Standard QuantLib - uses double
double price = blackScholesFormula(S, K, r, sigma, T);

// With quantlib-template - same code, different type
using Real = xad::AReal<double, 1>;
Real price = blackScholesFormula(S, K, r, sigma, T);  // Auto-records operations
// Reverse pass computes Greeks: Delta, Vega, etc.
```

**Mechanism**:
1. Operator overloading captures all operations
2. Expression templates build computation tree
3. Tape records graph when active
4. Reverse-mode AD computes derivatives in one pass

## What This Library Does NOT Include

This is **interface-only**. NOT included:
- ❌ Tape recording implementation
- ❌ Computational graph memory management
- ❌ Gradient computation algorithms
- ❌ Derivative storage/retrieval

**Users provide their own AD backend** replacing `TapeStub`.

## Integration

```cmake
add_subdirectory(forge/tools/quantlib-template)
target_link_libraries(your_target PRIVATE quantlib-template)
```

## Configuration (`Config.hpp.in`)

- `XAD_USE_STRONG_INLINE`: Aggressive inlining (slower compilation, faster runtime)
- `XAD_ALLOW_INT_CONVERSION`: Allow `AReal`→int conversions (enabled for QuantLib)

## Development Rules

### ⚠️ CRITICAL

1. **NO SIGNATURE CHANGES** - Breaks QuantLib integration
2. **NO API MODIFICATIONS** - Interface must match XAD
3. **NO BEHAVIORAL CHANGES** - Operator semantics preserved
4. **COMPREHENSIVE TESTING** - Any change requires full QuantLib test suite

### Testing Before Any Changes

1. Compile with full QuantLib test suite
2. Verify operator overload resolution
3. Check no implicit conversion issues
4. Validate expression template instantiation

## License

GNU Affero General Public License v3.0 (AGPL-3.0) - derived from XAD by Xcelerit Computing Ltd.

---

**Remember**: This is a delicate interface layer. When in doubt, don't change anything.
