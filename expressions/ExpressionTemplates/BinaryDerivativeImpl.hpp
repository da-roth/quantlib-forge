/*******************************************************************************

   Implementation template for binary derivatives, specialising if 2nd parameter
   is not needed.

   This file is part of QuantLib-Forge, a Forge integration layer for QuantLib.

   Copyright (C) 2025 The QuantLib-Forge Authors

   Derived from XAD (https://github.com/auto-differentiation/XAD)
   Original code: Copyright (C) 2010-2024 Xcelerit Computing Ltd.

   SPDX-License-Identifier: AGPL-3.0-or-later

******************************************************************************/

#pragma once
#include <expressions/Macros.hpp>

namespace forge { namespace expr {
namespace detail
{

template <bool>
struct BinaryDerivativeImpl
{
    template <class Op, class Scalar>
    static FEXPR_INLINE Scalar derivative_a(const Op& op, const Scalar& a, const Scalar& b,
                                          const Scalar&)
    {
        return op.derivative_a(a, b);
    }

    template <class Op, class Scalar>
    static FEXPR_INLINE Scalar derivative_b(const Op& op, const Scalar& a, const Scalar& b,
                                          const Scalar&)
    {
        return op.derivative_b(a, b);
    }
};

template <>
struct BinaryDerivativeImpl<true>
{
    template <class Op, class Scalar>
    static FEXPR_INLINE Scalar derivative_a(const Op& op, const Scalar& a, const Scalar& b,
                                          const Scalar& c)
    {
        return op.derivative_a(a, b, c);
    }

    template <class Op, class Scalar>
    static FEXPR_INLINE Scalar derivative_b(const Op& op, const Scalar& a, const Scalar& b,
                                          const Scalar& c)
    {
        return op.derivative_b(a, b, c);
    }
};

}  // namespace detail
}}  // namespace forge::expr

