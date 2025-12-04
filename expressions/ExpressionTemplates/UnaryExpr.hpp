/*******************************************************************************

   Unary expressions.

   This file is part of QuantLib-Forge, a Forge integration layer for QuantLib.

   Copyright (C) 2025 The QuantLib-Forge Authors

   Derived from XAD (https://github.com/auto-differentiation/XAD)
   Original code: Copyright (C) 2010-2024 Xcelerit Computing Ltd.

   SPDX-License-Identifier: AGPL-3.0-or-later

******************************************************************************/

#pragma once

#include <expressions/Expression.hpp>
#include <expressions/Macros.hpp>
#include <expressions/Traits.hpp>

#include <iostream>
#include <typeinfo>

// ========== Forge Integration: forward declarations and includes ==========
#include <types/fdouble.hpp>

namespace forge { namespace expr {
// Forward-declare unary functors we want to handle specially for Forge
template <class Scalar>
struct negate_op;

template <class Scalar, class T2>
struct scalar_add_op;

template <class Scalar, class T2>
struct scalar_prod_op;

template <class Scalar, class T2>
struct scalar_sub1_op;

template <class Scalar, class T2>
struct scalar_sub2_op;

template <class Scalar, class T2>
struct scalar_div1_op;

template <class Scalar, class T2>
struct scalar_div2_op;

template <class Scalar>
struct sqrt_op;

template <class Scalar>
struct exp_op;

template <class Scalar>
struct log_op;

template <class Scalar>
struct fabs_op;
}}  // namespace forge::expr
// ==========================================================================

namespace forge { namespace expr {

namespace detail
{
template <bool>
struct UnaryDerivativeImpl
{
    template <class Op, class Scalar>
    FEXPR_INLINE static Scalar derivative(const Op& op, const Scalar& a, const Scalar&)
    {
        return op.derivative(a);
    }
};

template <>
struct UnaryDerivativeImpl<true>
{
    template <class Op, class Scalar>
    FEXPR_INLINE static Scalar derivative(const Op& op, const Scalar& a, const Scalar& v)
    {
        return op.derivative(a, v);
    }
};
}  // namespace detail

template <class, class, class>
struct Expression;

/// Base class of all unary expressions
template <class Scalar, class Op, class Expr, class DerivativeType = Scalar>
struct UnaryExpr : Expression<Scalar, UnaryExpr<Scalar, Op, Expr, DerivativeType>, DerivativeType>
{
    typedef detail::UnaryDerivativeImpl<OperatorTraits<Op>::useResultBasedDerivatives == 1>
        der_impl;
    FEXPR_INLINE explicit UnaryExpr(const Expr& a, Op op = Op()) : a_(a), op_(op), v_(op_(a_.value()))
    {
    }
    FEXPR_INLINE Scalar value() const { return v_; }
    template <class Tape, int Size>
    FEXPR_INLINE void calc_derivatives(DerivInfo<Tape, Size>& info, Tape& s, const Scalar& mul) const
    {
        using forge::expr::value;
        a_.calc_derivatives(info, s, mul * der_impl::template derivative<>(op_, a_.value(), v_));
    }
    template <class Tape, int Size>
    FEXPR_INLINE void calc_derivatives(DerivInfo<Tape, Size>& info, Tape& s) const
    {
        using forge::expr::value;
        a_.calc_derivatives(info, s, der_impl::template derivative<>(op_, value(a_), v_));
    }

    FEXPR_INLINE bool shouldRecord() const { return a_.shouldRecord(); }

    FEXPR_INLINE DerivativeType derivative() const
    {
        using forge::expr::derivative;
        using forge::expr::value;
        return der_impl::template derivative<>(op_, value(a_), v_) * derivative(a_);
    }

    // ========== Forge Integration: propagate forge::fdouble through unary ops ==========
    FEXPR_INLINE ::forge::fdouble forgeValue() const
    {
        return applyForgeOp(getForgeValue(a_));
    }
    // ===============================================================================

  private:
    Expr a_;
    Op op_;
    Scalar v_;

    // ========== Forge Integration helpers =========================================
    template <class T>
    FEXPR_INLINE ::forge::fdouble getForgeValue(const T& expr) const
    {
        // Rely on nested expressions / literals to provide forgeValue().
        return expr.forgeValue();
    }

    FEXPR_INLINE ::forge::fdouble applyForgeOp(const ::forge::fdouble& fa) const
    {
        return performForgeOp(op_, fa);
    }

    // Negation
    static FEXPR_INLINE ::forge::fdouble performForgeOp(const negate_op<Scalar>&,
                                                    const ::forge::fdouble& a)
    {
        return -a;
    }

    // scalar_add_op: a + b
    template <class T2>
    static FEXPR_INLINE ::forge::fdouble performForgeOp(const scalar_add_op<Scalar, T2>& op,
                                                    const ::forge::fdouble& a)
    {
        ::forge::fdouble b(op.b_);
        return a + b;
    }

    // scalar_prod_op: a * b
    template <class T2>
    static FEXPR_INLINE ::forge::fdouble performForgeOp(const scalar_prod_op<Scalar, T2>& op,
                                                    const ::forge::fdouble& a)
    {
        ::forge::fdouble b(op.b_);
        return a * b;
    }

    // scalar_sub1_op: b - a
    template <class T2>
    static FEXPR_INLINE ::forge::fdouble performForgeOp(const scalar_sub1_op<Scalar, T2>& op,
                                                    const ::forge::fdouble& a)
    {
        ::forge::fdouble b(op.b_);
        return b - a;
    }

    // scalar_sub2_op: a - b
    template <class T2>
    static FEXPR_INLINE ::forge::fdouble performForgeOp(const scalar_sub2_op<Scalar, T2>& op,
                                                    const ::forge::fdouble& a)
    {
        ::forge::fdouble b(op.b_);
        return a - b;
    }

    // scalar_div1_op: b / a
    template <class T2>
    static FEXPR_INLINE ::forge::fdouble performForgeOp(const scalar_div1_op<Scalar, T2>& op,
                                                    const ::forge::fdouble& a)
    {
        ::forge::fdouble b(op.b_);
        return b / a;
    }

    // scalar_div2_op: a / b
    template <class T2>
    static FEXPR_INLINE ::forge::fdouble performForgeOp(const scalar_div2_op<Scalar, T2>& op,
                                                    const ::forge::fdouble& a)
    {
        ::forge::fdouble b(op.b_);
        return a / b;
    }

    // sqrt
    static FEXPR_INLINE ::forge::fdouble performForgeOp(const sqrt_op<Scalar>&,
                                                    const ::forge::fdouble& a)
    {
        // Use Forge's own sqrt to avoid ambiguity with std::sqrt overloads.
        return ::forge::sqrt(a);
    }

    // exp
    static FEXPR_INLINE ::forge::fdouble performForgeOp(const exp_op<Scalar>&,
                                                    const ::forge::fdouble& a)
    {
        // Use Forge's own exp to avoid ambiguity with std::exp overloads.
        return ::forge::exp(a);
    }

    // log
    static FEXPR_INLINE ::forge::fdouble performForgeOp(const log_op<Scalar>&,
                                                    const ::forge::fdouble& a)
    {
        return ::forge::log(a);
    }

    // fabs / abs
    static FEXPR_INLINE ::forge::fdouble performForgeOp(const fabs_op<Scalar>&,
                                                    const ::forge::fdouble& a)
    {
        // Use Forge's own abs to avoid ambiguity with std::abs/fabs overloads.
        return ::forge::abs(a);
    }

    // Fallback for other unary functors: re-create a Forge value from the numeric value.
    // This keeps dependencies correct for the expression templates but may need extending with more special-
    // cases if full Forge graph fidelity is required for additional math functions.
    template <class OpT>
    static FEXPR_INLINE ::forge::fdouble performForgeOp(const OpT&, const ::forge::fdouble& a)
    {
        // DEBUG: Log when unhandled unary operators hit the fallback path
        static std::size_t fallbackCount = 0;
        if (fallbackCount < 5) {
            ++fallbackCount;
            std::cerr << "[Forge][DEBUG] UnaryExpr fallback hit for unhandled operator type: "
                      << typeid(OpT).name() << " (occurrence " << fallbackCount << ")\n";
        }
        // Convert to plain double and back into a forge::fdouble node.
        return ::forge::fdouble(static_cast<double>(a));
    }
    // ===============================================================================
};

template <class Scalar, class Op, class Expr, class DerivativeType>
struct ExprTraits<UnaryExpr<Scalar, Op, Expr, DerivativeType> >
{
    static const bool isExpr = true;
    static const int numVariables = ExprTraits<Expr>::numVariables;
    static const bool isForward = ExprTraits<typename ExprTraits<Expr>::value_type>::isForward;
    static const bool isReverse = ExprTraits<typename ExprTraits<Expr>::value_type>::isReverse;
    static const bool isLiteral = false;
    static const Direction direction = ExprTraits<typename ExprTraits<Expr>::value_type>::direction;
    static const std::size_t vector_size =
        ExprTraits<typename ExprTraits<Expr>::value_type>::vector_size;

    typedef typename ExprTraits<Scalar>::nested_type nested_type;
    typedef typename ExprTraits<Expr>::value_type value_type;
};

}}  // namespace forge::expr

