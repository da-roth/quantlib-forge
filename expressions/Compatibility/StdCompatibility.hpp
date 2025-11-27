/*******************************************************************************

   Placing forge::expr math functions into the std namespace for std::log type
   expressions to work, as well as specialising numeric_limits.

   This partially violates the C++ standard's "don't specialize std templates"
   rule but is necessary for integration with other libraries.

   This file is part of QuantLib-Forge, a Forge integration layer for QuantLib.

   Copyright (C) 2025 The QuantLib-Forge Authors

   Derived from XAD (https://github.com/auto-differentiation/XAD)
   Original code: Copyright (C) 2010-2024 Xcelerit Computing Ltd.

   SPDX-License-Identifier: AGPL-3.0-or-later

******************************************************************************/

#pragma once

#include <expressions/ExpressionTemplates/BinaryOperators.hpp>
#include <expressions/Literals.hpp>
#include <expressions/Compatibility/MathFunctions.hpp>
#include <expressions/ExpressionTemplates/UnaryOperators.hpp>
#include <cmath>
#include <functional>
#include <limits>
#include <string>
#include <type_traits>

namespace std
{
using forge::expr::abs;
using forge::expr::acos;
using forge::expr::acosh;
using forge::expr::asin;
using forge::expr::asinh;
using forge::expr::atan;
using forge::expr::atan2;
using forge::expr::atanh;
using forge::expr::cbrt;
using forge::expr::ceil;
using forge::expr::copysign;
using forge::expr::cos;
using forge::expr::cosh;
using forge::expr::erf;
using forge::expr::erfc;
using forge::expr::exp;
using forge::expr::exp2;
using forge::expr::expm1;
using forge::expr::fabs;
using forge::expr::floor;
using forge::expr::fma;
using forge::expr::fmax;
using forge::expr::fmin;
using forge::expr::fmod;
using forge::expr::fpclassify;
using forge::expr::frexp;
using forge::expr::hypot;
using forge::expr::ilogb;
using forge::expr::isfinite;
using forge::expr::isinf;
using forge::expr::isnan;
using forge::expr::isnormal;
using forge::expr::ldexp;
using forge::expr::llround;
using forge::expr::log;
using forge::expr::log10;
using forge::expr::log1p;
using forge::expr::log2;
using forge::expr::lround;
using forge::expr::max;
using forge::expr::min;
using forge::expr::modf;
using forge::expr::nextafter;
using forge::expr::pow;
using forge::expr::remainder;
using forge::expr::remquo;
using forge::expr::round;
using forge::expr::scalbn;
using forge::expr::signbit;
using forge::expr::sin;
using forge::expr::sinh;
using forge::expr::sqrt;
using forge::expr::tan;
using forge::expr::tanh;
using forge::expr::trunc;

#if defined(_MSC_VER) || (__clang_major__ > 16 && defined(_LIBCPP_VERSION))
// we need these explicit instantiation to disambiguate templates in MSVC & Clang

template <class T, std::size_t N>
FEXPR_INLINE T copysign(const T& x, const forge::expr::AReal<T, N>& y)
{
    using std::copysign;
    return copysign(x, value(y));
}

template <class T, class T2, std::size_t N>
FEXPR_INLINE forge::expr::AReal<T, N> copysign(const T2& x, const forge::expr::AReal<T, N>& y)
{
    using std::copysign;
    return copysign(x, value(y));
}

template <class T, std::size_t N>
FEXPR_INLINE forge::expr::AReal<T, N> copysign(const forge::expr::AReal<T, N> x, const T& y)
{
    return ::forge::expr::copysign(x, forge::expr::value(y));
}

template <class T, class T2, std::size_t N>
FEXPR_INLINE forge::expr::AReal<T, N> copysign(const forge::expr::AReal<T, N> x, const T2& y)
{
    return ::forge::expr::copysign(x, forge::expr::value(y));
}

template <class T, std::size_t N>
FEXPR_INLINE forge::expr::AReal<T, N> copysign(const forge::expr::AReal<T, N> x, const forge::expr::AReal<T, N>& y)
{
    return ::forge::expr::copysign(x, forge::expr::value(y));
}

// FReal copysign functions removed - not used by QuantLib-Risks-Cpp

#endif

template <class Scalar, class Derived, class Deriv>
inline std::string to_string(const forge::expr::Expression<Scalar, Derived, Deriv>& _Val)
{
    return to_string(value(_Val));
}

}  // namespace std

namespace std
{

// note that these return the underlying doubles, not the active type,
// but since they are constant and convertible, it's the right behaviour
// for the majority of cases

template <class T, std::size_t N>
struct numeric_limits<forge::expr::AReal<T, N>> : std::numeric_limits<T>
{
};

// FReal numeric_limits specialization removed - not used by QuantLib-Risks-Cpp

}  // namespace std

// hashing for active types
namespace std
{

template <class T, std::size_t N>
struct hash<forge::expr::AReal<T, N>>
{
    std::size_t operator()(forge::expr::AReal<T, N> const& s) const noexcept
    {
        return std::hash<T>{}(forge::expr::value(s));
    }
};

// FReal hash specialization removed - not used by QuantLib-Risks-Cpp

// type traits
template <class T, std::size_t N>
struct is_floating_point<forge::expr::AReal<T, N>> : std::is_floating_point<T>
{
};
// FReal is_floating_point specialization removed - not used by QuantLib-Risks-Cpp
template <class T, std::size_t N>
struct is_arithmetic<forge::expr::AReal<T, N>> : std::is_arithmetic<T>
{
};
// FReal is_arithmetic specialization removed - not used by QuantLib-Risks-Cpp
template <class T, std::size_t N>
struct is_signed<forge::expr::AReal<T, N>> : std::is_signed<T>
{
};
// FReal is_signed specialization removed - not used by QuantLib-Risks-Cpp
template <class T, std::size_t N>
struct is_pod<forge::expr::AReal<T, N>> : std::false_type
{
};
// FReal is_pod specialization removed - not used by QuantLib-Risks-Cpp
template <class T, std::size_t N>
struct is_fundamental<forge::expr::AReal<T, N>> : std::false_type
{
};
// FReal is_fundamental specialization removed - not used by QuantLib-Risks-Cpp
// FReal is_trivially_copyable specialization removed - not used by QuantLib-Risks-Cpp
template <class T, std::size_t N>
struct is_scalar<forge::expr::AReal<T, N>> : std::false_type
{
};
// FReal is_scalar specialization removed - not used by QuantLib-Risks-Cpp
template <class T, std::size_t N>
struct is_compound<forge::expr::AReal<T, N>> : std::true_type
{
};
// FReal is_compound specialization removed - not used by QuantLib-Risks-Cpp

#if ((defined(_MSVC_LANG) && _MSVC_LANG >= 201703L) || __cplusplus >= 201703L)

// For some reason, in VS 2022, a generic template for is_floating_point_v is not used
// in overload resolution. We need to fully specialise the template for common types
// here (first and second order only for now)

#define FEXPR_TEMPLATE_TRAIT_FUNC_FIRST(name_v, N, value)                                           \
    template <>                                                                                    \
    inline constexpr bool name_v<forge::expr::AReal<double, N>> = value;                           \
    template <>                                                                                    \
    inline constexpr bool name_v<forge::expr::AReal<float, N>> = value;                            \
    template <>                                                                                    \
    inline constexpr bool name_v<forge::expr::AReal<long double, N>> = value

#define FEXPR_TEMPLATE_TRAIT_FUNC_SECOND(name_v, N, M, value)                                       \
    template <>                                                                                    \
    inline constexpr bool name_v<forge::expr::AReal<forge::expr::AReal<double, M>, N>> = value;    \
    template <>                                                                                    \
    inline constexpr bool name_v<forge::expr::AReal<forge::expr::AReal<float, M>, N>> = value;     \
    template <>                                                                                    \
    inline constexpr bool name_v<forge::expr::AReal<forge::expr::AReal<long double, M>, N>> = value

#define FEXPR_TEMPLATE_TRAIT_FUNC1(name_v, N, value)                                                \
    FEXPR_TEMPLATE_TRAIT_FUNC_FIRST(name_v, N, value);                                              \
    FEXPR_TEMPLATE_TRAIT_FUNC_SECOND(name_v, N, 1, value);                                          \
    FEXPR_TEMPLATE_TRAIT_FUNC_SECOND(name_v, N, 2, value);                                          \
    FEXPR_TEMPLATE_TRAIT_FUNC_SECOND(name_v, N, 4, value)

#define FEXPR_TEMPLATE_TRAIT_FUNC(name_v, value)                                                    \
    FEXPR_TEMPLATE_TRAIT_FUNC1(name_v, 1, value);                                                   \
    FEXPR_TEMPLATE_TRAIT_FUNC1(name_v, 2, value);                                                   \
    FEXPR_TEMPLATE_TRAIT_FUNC1(name_v, 4, value)

FEXPR_TEMPLATE_TRAIT_FUNC(is_floating_point_v, true);
FEXPR_TEMPLATE_TRAIT_FUNC(is_arithmetic_v, true);
FEXPR_TEMPLATE_TRAIT_FUNC(is_integral_v, false);
FEXPR_TEMPLATE_TRAIT_FUNC(is_fundamental_v, false);
FEXPR_TEMPLATE_TRAIT_FUNC(is_scalar_v, false);
FEXPR_TEMPLATE_TRAIT_FUNC(is_compound_v, true);

#undef FEXPR_TEMPLATE_TRAIT_FUNC
#undef FEXPR_TEMPLATE_TRAIT_FUNC1
#undef FEXPR_TEMPLATE_TRAIT_FUNC_FIRST
#undef FEXPR_TEMPLATE_TRAIT_FUNC_SECOND

// FReal is_trivially_copyable_v specializations removed - not used by QuantLib-Risks-Cpp

#endif

#if defined(_MSC_VER)

// for MSVC, we need this workaround so that the safety checks in their STL
// for floating point types are also passing for the forge::expr types

// VS 2017+, when the STL checks if a type is in the list of built-in floating point types,
// this should forward the check to the wrapped type by AReal or FReal.
//
// (In GCC, std::is_floating_point is used instead, where traits above work)

#if ((defined(_MSVC_LANG) && _MSVC_LANG >= 201703L) || __cplusplus >= 201703L || defined(__clang__))
#define _FEXPR_INLINE_VAR inline
#else
#define _FEXPR_INLINE_VAR
#endif

template <>
_FEXPR_INLINE_VAR constexpr bool _Is_any_of_v<forge::expr::AReal<double, 1>, float, double, long double> =
    true;
template <>
_FEXPR_INLINE_VAR constexpr bool _Is_any_of_v<forge::expr::AReal<float, 1>, float, double, long double> =
    true;
template <>
_FEXPR_INLINE_VAR constexpr bool
    _Is_any_of_v<forge::expr::AReal<long double, 1>, float, double, long double> = true;
template <>
_FEXPR_INLINE_VAR constexpr bool _Is_any_of_v<forge::expr::AReal<double, 2>, float, double, long double> =
    true;
template <>
_FEXPR_INLINE_VAR constexpr bool _Is_any_of_v<forge::expr::AReal<float, 2>, float, double, long double> =
    true;
template <>
_FEXPR_INLINE_VAR constexpr bool
    _Is_any_of_v<forge::expr::AReal<long double, 2>, float, double, long double> = true;
template <>
_FEXPR_INLINE_VAR constexpr bool _Is_any_of_v<forge::expr::AReal<double, 4>, float, double, long double> =
    true;
template <>
_FEXPR_INLINE_VAR constexpr bool _Is_any_of_v<forge::expr::AReal<float, 4>, float, double, long double> =
    true;
template <>
_FEXPR_INLINE_VAR constexpr bool
    _Is_any_of_v<forge::expr::AReal<long double, 4>, float, double, long double> = true;
// FReal _Is_any_of_v specializations removed - not used by QuantLib-Risks-Cpp

template <>
_FEXPR_INLINE_VAR constexpr bool
    _Is_any_of_v<forge::expr::AReal<forge::expr::AReal<double>>, float, double, long double> = true;
template <>
_FEXPR_INLINE_VAR constexpr bool
    _Is_any_of_v<forge::expr::AReal<forge::expr::AReal<float>>, float, double, long double> = true;
template <>
_FEXPR_INLINE_VAR constexpr bool
    _Is_any_of_v<forge::expr::AReal<forge::expr::AReal<long double>>, float, double, long double> = true;

#undef _FEXPR_INLINE_VAR

#endif

// https://github.com/auto-differentiation/xad/issues/169
#if __clang_major__ >= 16 && defined(_LIBCPP_VERSION)

template <typename T, std::size_t N>
struct __libcpp_random_is_valid_realtype<forge::expr::AReal<T, N>> : true_type
{
};

// FReal __libcpp_random_is_valid_realtype specialization removed - not used by QuantLib-Risks-Cpp

#endif

}  // namespace std

