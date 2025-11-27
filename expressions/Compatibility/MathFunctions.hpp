/*******************************************************************************

   Importing or declaring of math functions in our namespace.

   This file is part of QuantLib-Forge, a Forge integration layer for QuantLib.

   Copyright (C) 2025 The QuantLib-Forge Authors

   Derived from XAD (https://github.com/auto-differentiation/XAD)
   Original code: Copyright (C) 2010-2024 Xcelerit Computing Ltd.

   SPDX-License-Identifier: AGPL-3.0-or-later

******************************************************************************/

#pragma once

#include <expressions/Macros.hpp>
#include <expressions/Traits.hpp>
#include <type_traits>

#include <algorithm>
#include <cfloat>
#include <cmath>

namespace forge { namespace expr {
// import std functions to our namespace, and re-implement as needed
// (all for float and double types)

// functions that always are in cmath
using ::std::abs;
using ::std::acos;
using ::std::asin;
using ::std::atan;
using ::std::atan2;
using ::std::ceil;
using ::std::cos;
using ::std::cosh;
using ::std::exp;
using ::std::fabs;
using ::std::floor;
using ::std::fmod;
using ::std::frexp;
using ::std::hypot;
using ::std::ldexp;
using ::std::log;
using ::std::log10;
using ::std::max;
using ::std::min;
using ::std::modf;
using ::std::pow;
using ::std::sin;
using ::std::sinh;
using ::std::sqrt;
using ::std::tan;
using ::std::tanh;

FEXPR_INLINE double smooth_abs(double x, double c = 0.001)
{
    if (abs(x) > c)
        return abs(x);
    if (x < 0.0)
        return x * x * (2. / c + x / (c * c));
    else
        return x * x * (2. / c - x / (c * c));
}

FEXPR_INLINE float smooth_abs(float x, float c = 0.001f)
{
    if (abs(x) > c)
        return abs(x);
    if (x < 0.0f)
        return x * x * (2.f / c + x / (c * c));
    else
        return x * x * (2.f / c - x / (c * c));
}

FEXPR_INLINE double smooth_max(double x, double y, double c = 0.001)
{
    return 0.5 * (x + y + smooth_abs(x - y, c));
}

FEXPR_INLINE float smooth_max(float x, float y, float c = 0.001f)
{
    return 0.5f * (x + y + smooth_abs(x - y, c));
}

FEXPR_INLINE double smooth_min(double x, double y, double c = 0.001)
{
    return 0.5 * (x + y - smooth_abs(x - y, c));
}

FEXPR_INLINE float smooth_min(float x, float y, float c = 0.001f)
{
    return 0.5f * (x + y - smooth_abs(x - y, c));
}

using ::std::acosh;
using ::std::asinh;
using ::std::atan2;
using ::std::atanh;
using ::std::cbrt;
using ::std::erf;
using ::std::erfc;
using ::std::exp2;
using ::std::expm1;
using ::std::fmax;
using ::std::fmin;
using ::std::fmod;
using ::std::fpclassify;
using ::std::hypot;
using ::std::isfinite;
using ::std::isinf;
using ::std::isnan;
using ::std::isnormal;
using ::std::log1p;
using ::std::log2;
using ::std::nextafter;
using ::std::remainder;
using ::std::round;
using ::std::signbit;
using ::std::trunc;

}}  // namespace forge::expr

