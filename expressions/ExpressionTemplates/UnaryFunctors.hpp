/*******************************************************************************

   Functors capturing unary expressions.

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
struct negate_op
{
    FEXPR_INLINE Scalar operator()(const Scalar& a) const { return -a; }
    FEXPR_INLINE Scalar derivative(const Scalar&) const { return -Scalar(1); }
};

// binary operations with a scalar are actually unary functors

template <class Scalar, class T2>
struct scalar_add_op
{
    FEXPR_INLINE explicit scalar_add_op(const T2& b) : b_(Scalar(b)) {}
    FEXPR_INLINE Scalar operator()(const Scalar& a) const { return a + b_; }
    FEXPR_INLINE Scalar derivative(const Scalar&) const { return Scalar(1); }
    Scalar b_;
};

template <class Scalar, class T2>
struct scalar_prod_op
{
    FEXPR_INLINE explicit scalar_prod_op(const T2& b) : b_(Scalar(b)) {}
    FEXPR_INLINE Scalar operator()(const Scalar& a) const { return Scalar(a * b_); }
    FEXPR_INLINE Scalar derivative(const Scalar&) const { return Scalar(b_); }
    Scalar b_;
};

template <class Scalar, class T2>
struct scalar_sub1_op
{
    FEXPR_INLINE explicit scalar_sub1_op(const T2& b) : b_(Scalar(b)) {}
    FEXPR_INLINE Scalar operator()(const Scalar& a) const { return b_ - a; }
    FEXPR_INLINE Scalar derivative(const Scalar&) const { return Scalar(-1); }
    Scalar b_;
};

template <class Scalar, class T2>
struct scalar_sub2_op
{
    FEXPR_INLINE explicit scalar_sub2_op(const T2& b) : b_(Scalar(b)) {}
    FEXPR_INLINE Scalar operator()(const Scalar& a) const { return a - Scalar(b_); }
    FEXPR_INLINE Scalar derivative(const Scalar&) const { return Scalar(1); }
    Scalar b_;
};

template <class Scalar, class T2>
struct scalar_div1_op
{
    FEXPR_INLINE explicit scalar_div1_op(const T2& b) : b_(Scalar(b)) {}
    FEXPR_INLINE Scalar operator()(const Scalar& a) const { return b_ / a; }
    FEXPR_INLINE Scalar derivative(const Scalar& a) const { return -b_ / (a * a); }
    Scalar b_;
};

template <class Scalar, class T2>
struct scalar_div2_op
{
    FEXPR_INLINE explicit scalar_div2_op(const T2& b) : b_(Scalar(b)) {}
    FEXPR_INLINE Scalar operator()(const Scalar& a) const { return a / b_; }
    FEXPR_INLINE Scalar derivative(const Scalar&) const { return Scalar(1) / b_; }
    Scalar b_;
};

}}  // namespace forge::expr

