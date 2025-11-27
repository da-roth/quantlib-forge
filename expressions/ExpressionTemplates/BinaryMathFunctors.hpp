/*******************************************************************************

   Functors for binary math functions.

   This file is part of QuantLib-Forge, a Forge integration layer for QuantLib.

   Copyright (C) 2025 The QuantLib-Forge Authors

   Derived from XAD (https://github.com/auto-differentiation/XAD)
   Original code: Copyright (C) 2010-2024 Xcelerit Computing Ltd.

   SPDX-License-Identifier: AGPL-3.0-or-later

******************************************************************************/

#pragma once

#include <expressions/Macros.hpp>
#include <expressions/Compatibility/MathFunctions.hpp>

namespace forge { namespace expr {

////////////// Pow

template <class Scalar>
struct pow_op
{
    FEXPR_INLINE Scalar operator()(const Scalar& a, const Scalar& b) const { return pow(a, b); }
    FEXPR_INLINE Scalar derivative_a(const Scalar& a, const Scalar& b, const Scalar&) const
    {
        return b * pow(a, b - Scalar(1));
    }
    FEXPR_INLINE Scalar derivative_b(const Scalar& a, const Scalar&, const Scalar& v) const
    {
        return log(a) * v;
    }
};

template <class Scalar>
struct OperatorTraits<pow_op<Scalar> >
{
    enum
    {
        useResultBasedDerivatives = 1
    };
};

/// smooth ABS

template <class Scalar>
struct smooth_abs_op
{
    FEXPR_INLINE Scalar operator()(const Scalar& x, const Scalar& c) const
    {
        if (abs(x) > c)
            return abs(x);
        if (x < Scalar())
        {
            return x * x * (Scalar(2) / c + x / (c * c));
        }
        else
        {
            return x * x * (Scalar(2) / c - x / (c * c));
        }
    }

    FEXPR_INLINE Scalar derivative_a(const Scalar& x, const Scalar& c) const
    {
        if (x > c)
            return Scalar(1);
        else if (x < -c)
            return Scalar(-1);
        else if (x < Scalar())
        {
            return x / (c * c) * (Scalar(3) * x + Scalar(4) * c);
        }
        else
            return -x / (c * c) * (Scalar(3) * x - Scalar(4) * c);
    }

    FEXPR_INLINE Scalar derivative_b(const Scalar& x, const Scalar& c) const
    {
        if (x > c || x < -c)
            return Scalar();
        else if (x < Scalar())
        {
            return -Scalar(2) * x * x * (c + x) / (c * c * c);
        }
        else
        {
            return -Scalar(2) * x * x * (c - x) / (c * c * c);
        }
    }
};

//////// max

// need this complicated expression to have a kind-of smooth 2nd derivative
template <class Scalar>
struct max_op
{
    FEXPR_INLINE Scalar operator()(const Scalar& a, const Scalar& b) const
    {
        if (!ExprTraits<Scalar>::isExpr)
            return (a < b) ? b : a;
        else
            return (a + b + abs(a - b)) / Scalar(2);
    }
    FEXPR_INLINE Scalar derivative_a(const Scalar& a, const Scalar& b) const
    {
        if (!ExprTraits<Scalar>::isExpr)
        {
            if (b < a)
                return Scalar(1);
            else if (a < b)
                return Scalar();
            else
                return Scalar(0.5);
        }
        else
        {
            return (Scalar(1) + (Scalar((a - b) > Scalar()) - Scalar((a - b) < Scalar()))) /
                   Scalar(2);
        }
    }
    FEXPR_INLINE Scalar derivative_b(const Scalar& a, const Scalar& b) const
    {
        if (!ExprTraits<Scalar>::isExpr)
        {
            if (b < a)
                return Scalar();
            else if (a < b)
                return Scalar(1);
            else
                return Scalar(0.5);
        }
        else
        {
            return (Scalar(1) - (Scalar((a - b) > Scalar()) - Scalar((a - b) < Scalar()))) /
                   Scalar(2);
        }
    }
};

////// min

template <class Scalar>
struct min_op
{
    FEXPR_INLINE Scalar operator()(const Scalar& a, const Scalar& b) const
    {
        if (!ExprTraits<Scalar>::isExpr)
            return (a < b) ? a : b;
        else
            return (a + b - abs(a - b)) / Scalar(2);
    }
    FEXPR_INLINE Scalar derivative_a(const Scalar& a, const Scalar& b) const
    {
        if (!ExprTraits<Scalar>::isExpr)
        {
            if (a < b)
                return Scalar(1);
            else if (b < a)
                return Scalar();
            else
                return Scalar(0.5);
        }
        else
        {
            return (Scalar(1) - (Scalar((a - b) > Scalar()) - Scalar((a - b) < Scalar()))) /
                   Scalar(2);
        }
    }
    FEXPR_INLINE Scalar derivative_b(const Scalar& a, const Scalar& b) const
    {
        if (!ExprTraits<Scalar>::isExpr)
        {
            if (a < b)
                return Scalar();
            else if (b < a)
                return Scalar(1);
            else
                return Scalar(0.5);
        }
        else
        {
            return (Scalar(1) + (Scalar((a - b) > Scalar()) - Scalar((a - b) < Scalar()))) /
                   Scalar(2);
        }
    }
};

///////// fmax / fmin

template <class Scalar>
struct fmax_op : max_op<Scalar>
{
};
template <class Scalar>
struct fmin_op : min_op<Scalar>
{
};

/////////// fmod

template <class Scalar>
struct fmod_op
{
    FEXPR_INLINE Scalar operator()(const Scalar& a, const Scalar& b) const { return fmod(a, b); }
    FEXPR_INLINE Scalar derivative_a(const Scalar&, const Scalar&) const { return Scalar(1); }
    FEXPR_INLINE Scalar derivative_b(const Scalar& a, const Scalar& b) const { return -floor(a / b); }
};

template <class Scalar>
struct atan2_op
{
    FEXPR_INLINE Scalar operator()(const Scalar& a, const Scalar& b) const { return atan2(a, b); }
    FEXPR_INLINE Scalar derivative_a(const Scalar& a, const Scalar& b) const
    {
        return b / (a * a + b * b);
    }
    FEXPR_INLINE Scalar derivative_b(const Scalar& a, const Scalar& b) const
    {
        return -a / (a * a + b * b);
    }
};

template <class Scalar>
struct hypot_op
{
    FEXPR_INLINE Scalar operator()(const Scalar& a, const Scalar& b) const { return hypot(a, b); }
    FEXPR_INLINE Scalar derivative_a(const Scalar& a, const Scalar&, const Scalar& v) const
    {
        return a / v;
    }
    FEXPR_INLINE Scalar derivative_b(const Scalar&, const Scalar& b, const Scalar& v) const
    {
        return b / v;
    }
};

template <class Scalar>
struct OperatorTraits<hypot_op<Scalar> >
{
    enum
    {
        useResultBasedDerivatives = 1
    };
};

template <class Scalar>
struct remainder_op
{
    FEXPR_INLINE Scalar operator()(const Scalar& a, const Scalar& b) const { return remainder(a, b); }
    FEXPR_INLINE Scalar derivative_a(const Scalar&, const Scalar&) const { return Scalar(1); }
    FEXPR_INLINE Scalar derivative_b(const Scalar& a, const Scalar& b) const
    {
        // function is rare enough that there's no need to optimize this better
        int n_;
        using std::remquo;
        FEXPR_UNUSED_VARIABLE(remquo(a, b, &n_));
        return Scalar(-n_);
    }
};

template <class Scalar>
struct remquo_op
{
    FEXPR_INLINE explicit remquo_op(int* quo) : quo_(quo), q_() {}
    FEXPR_INLINE Scalar operator()(const Scalar& a, const Scalar& b) const
    {
        using std::remquo;
        Scalar v = remquo(a, b, &q_);
        *quo_ = q_;
        return v;
    }
    FEXPR_INLINE Scalar derivative_a(const Scalar&, const Scalar&) const { return Scalar(1); }
    FEXPR_INLINE Scalar derivative_b(const Scalar&, const Scalar&) const { return Scalar(-q_); }
    int* quo_;
    mutable int q_;
};

template <class Scalar>
struct nextafter_op
{
    FEXPR_INLINE explicit nextafter_op() {}
    FEXPR_INLINE Scalar operator()(const Scalar& from, const Scalar& to) const
    {
        return nextafter(from, to);
    }
    FEXPR_INLINE Scalar derivative_a(const Scalar&, const Scalar&) const { return Scalar(1); }
    FEXPR_INLINE Scalar derivative_b(const Scalar&, const Scalar&) const { return Scalar(0); }
};

}}  // namespace forge::expr

