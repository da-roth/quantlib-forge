/*******************************************************************************

   Functors for binary arithmetic operators.

   This file is part of QuantLib-Forge, a Forge integration layer for QuantLib.

   Copyright (C) 2025 The QuantLib-Forge Authors

   Derived from XAD (https://github.com/auto-differentiation/XAD)
   Original code: Copyright (C) 2010-2024 Xcelerit Computing Ltd.

   SPDX-License-Identifier: AGPL-3.0-or-later

******************************************************************************/

#pragma once
#include <expressions/Macros.hpp>

namespace forge { namespace expr {

template <class Scalar>
struct add_op
{
    FEXPR_INLINE Scalar operator()(const Scalar& a, const Scalar& b) const { return a + b; }

    FEXPR_INLINE Scalar derivative_a(const Scalar&, const Scalar&) const { return Scalar(1); }

    FEXPR_INLINE Scalar derivative_b(const Scalar&, const Scalar&) const { return Scalar(1); }
};

template <class Scalar>
struct prod_op
{
    FEXPR_INLINE Scalar operator()(const Scalar& a, const Scalar& b) const { return a * b; }

    FEXPR_INLINE Scalar derivative_a(const Scalar&, const Scalar& b) const { return b; }

    FEXPR_INLINE Scalar derivative_b(const Scalar& a, const Scalar&) const { return a; }
};

template <class Scalar>
struct sub_op
{
    FEXPR_INLINE Scalar operator()(const Scalar& a, const Scalar& b) const { return a - b; }

    FEXPR_INLINE Scalar derivative_a(const Scalar&, const Scalar&) const { return Scalar(1); }

    FEXPR_INLINE Scalar derivative_b(const Scalar&, const Scalar&) const { return Scalar(-1); }
};

template <class Scalar>
struct div_op
{
    FEXPR_INLINE Scalar operator()(const Scalar& a, const Scalar& b) const { return a / b; }

    FEXPR_INLINE Scalar derivative_a(const Scalar&, const Scalar& b) const { return Scalar(1) / b; }

    FEXPR_INLINE Scalar derivative_b(const Scalar& a, const Scalar& b) const { return -a / (b * b); }
};

}}  // namespace forge::expr

