/*******************************************************************************

   Binary expression template.

   This file is part of QuantLib-Forge, a Forge integration layer for QuantLib.

   Copyright (C) 2025 The QuantLib-Forge Authors

   Derived from XAD (https://github.com/auto-differentiation/XAD)
   Original code: Copyright (C) 2010-2024 Xcelerit Computing Ltd.

   SPDX-License-Identifier: AGPL-3.0-or-later

******************************************************************************/

#pragma once

#include <expressions/ExpressionTemplates/BinaryDerivativeImpl.hpp>
#include <expressions/ExpressionTemplates/BinaryFunctors.hpp>
#include <expressions/Expression.hpp>
#include <expressions/Macros.hpp>
#include <expressions/Traits.hpp>

// ========== Forge Integration: Add includes ==========
#include <types/fdouble.hpp>
// =====================================================

#include <type_traits>

namespace forge { namespace expr {

// Forward declarations for math functors used in Forge integration
template <class Scalar>
struct pow_op;

template <class Scalar>
struct max_op;

template <class Scalar>
struct min_op;

template <class Scalar, class Op, class Expr1, class Expr2, class DerivativeType = Scalar>
struct BinaryExpr
    : Expression<Scalar, BinaryExpr<Scalar, Op, Expr1, Expr2, DerivativeType>, DerivativeType>
{
    typedef detail::BinaryDerivativeImpl<OperatorTraits<Op>::useResultBasedDerivatives == 1>
        der_impl;
    FEXPR_INLINE BinaryExpr(const Expr1& a, const Expr2& b, Op op = Op())
        : a_(a), b_(b), op_(op), v_(op_(a_.value(), b_.value()))
    {
    }
    FEXPR_INLINE Scalar value() const { return v_; }

    template <class Tape, int Size>
    FEXPR_INLINE void calc_derivatives(DerivInfo<Tape, Size>& info, Tape& s, const Scalar& mul) const
    {
        using forge::expr::value;
        a_.calc_derivatives(info, s,
                            mul * der_impl::template derivative_a<>(op_, value(a_), value(b_), v_));
        b_.calc_derivatives(info, s,
                            mul * der_impl::template derivative_b<>(op_, value(a_), value(b_), v_));
    }
    template <class Tape, int Size>
    FEXPR_INLINE void calc_derivatives(DerivInfo<Tape, Size>& info, Tape& s) const
    {
        using forge::expr::value;
        a_.calc_derivatives(info, s,
                            der_impl::template derivative_a<>(op_, value(a_), value(b_), v_));
        b_.calc_derivatives(info, s,
                            der_impl::template derivative_b<>(op_, value(a_), value(b_), v_));
    }

    FEXPR_INLINE DerivativeType derivative() const
    {
        using forge::expr::derivative;
        using forge::expr::value;
        return der_impl::template derivative_a<>(op_, value(a_), value(b_), v_) * derivative(a_) +
               der_impl::template derivative_b<>(op_, value(a_), value(b_), v_) * derivative(b_);
    }

    FEXPR_INLINE bool shouldRecord() const { return a_.shouldRecord() || b_.shouldRecord(); }

    // ========== Forge Integration: Compute forge result from operands ==========
    FEXPR_INLINE ::forge::fdouble forgeValue() const {
        // Apply the operator to the forge values of the operands
        return applyForgeOp(getForgeValue(a_), getForgeValue(b_));
    }
    // ===========================================================================

  private:
    Expr1 a_;
    Expr2 b_;
    Op op_;
    Scalar v_;

    // ========== Forge Integration: Helper to extract forge values ==========
    template<class T>
    FEXPR_INLINE ::forge::fdouble getForgeValue(const T& expr) const {
        return expr.forgeValue();  // Recursive for nested expressions
    }

    // Apply the operator to forge values - delegates to the specific functor
    FEXPR_INLINE ::forge::fdouble applyForgeOp(const ::forge::fdouble& fa, const ::forge::fdouble& fb) const {
        return performForgeOp(op_, fa, fb);
    }

    // Overloads for each operator type
    FEXPR_INLINE static ::forge::fdouble performForgeOp(const add_op<Scalar>&,
                                                    const ::forge::fdouble& a, const ::forge::fdouble& b) {
        return a + b;
    }
    FEXPR_INLINE static ::forge::fdouble performForgeOp(const sub_op<Scalar>&,
                                                    const ::forge::fdouble& a, const ::forge::fdouble& b) {
        return a - b;
    }
    FEXPR_INLINE static ::forge::fdouble performForgeOp(const prod_op<Scalar>&,
                                                    const ::forge::fdouble& a, const ::forge::fdouble& b) {
        return a * b;
    }
    FEXPR_INLINE static ::forge::fdouble performForgeOp(const div_op<Scalar>&,
                                                    const ::forge::fdouble& a, const ::forge::fdouble& b) {
        return a / b;
    }

    // pow(a, b)
    FEXPR_INLINE static ::forge::fdouble performForgeOp(const pow_op<Scalar>&,
                                                    const ::forge::fdouble& a,
                                                    const ::forge::fdouble& b)
    {
        // Use Forge's own pow to avoid ambiguity with std::pow overloads.
        return ::forge::pow(a, b);
    }

    // max(a, b)
    FEXPR_INLINE static ::forge::fdouble performForgeOp(const max_op<Scalar>&,
                                                    const ::forge::fdouble& a,
                                                    const ::forge::fdouble& b)
    {
        return ::forge::max(a, b);
    }

    // min(a, b)
    FEXPR_INLINE static ::forge::fdouble performForgeOp(const min_op<Scalar>&,
                                                    const ::forge::fdouble& a,
                                                    const ::forge::fdouble& b)
    {
        return ::forge::min(a, b);
    }

    // Fallback for any other binary functor: evaluate numerically and wrap into forge::fdouble.
    template <class OpT>
    FEXPR_INLINE static ::forge::fdouble performForgeOp(const OpT& op,
                                                    const ::forge::fdouble& a,
                                                    const ::forge::fdouble& b)
    {
        const double va = static_cast<double>(a);
        const double vb = static_cast<double>(b);
        const double r = op(va, vb);
        return ::forge::fdouble(r);
    }
    // =======================================================================
};

template <class Scalar, class Op, class Expr1, class Expr2, class DerivativeType>
struct ExprTraits<BinaryExpr<Scalar, Op, Expr1, Expr2, DerivativeType>>
{
    static const bool isExpr = true;
    static const int numVariables =
        ExprTraits<Expr1>::numVariables + ExprTraits<Expr2>::numVariables;
    static const bool isForward = ExprTraits<typename ExprTraits<Expr1>::value_type>::isForward;
    static const bool isReverse = ExprTraits<typename ExprTraits<Expr1>::value_type>::isReverse;
    static const bool isLiteral = false;
    static const Direction direction =
        ExprTraits<typename ExprTraits<Expr1>::value_type>::direction;
    static const std::size_t vector_size =
        ExprTraits<typename ExprTraits<Expr1>::value_type>::vector_size;

    typedef typename ExprTraits<Scalar>::nested_type nested_type;
    typedef typename ExprTraits<Expr1>::value_type value_type;
    typedef Scalar scalar_type;
    // make sure that both sides of the binary expression have the same value_type
    // This should always be the case, as expressions with scalars are producing UnaryExpr
    // objects, and mixing different AReal/FReal types in a single expression is invalid
    static_assert(std::is_same<typename ExprTraits<Expr1>::value_type,
                               typename ExprTraits<Expr2>::value_type>::value,
                  "both expressions must be the same underlying type");
};

}}  // namespace forge::expr

