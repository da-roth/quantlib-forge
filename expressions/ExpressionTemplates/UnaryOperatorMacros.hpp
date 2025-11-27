/*******************************************************************************

   Macros for unary operators - to be included by UnaryOperators.hpp.

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

namespace forge { namespace expr {

template <class, std::size_t>
struct AReal;
template <class, std::size_t>
struct ADVar;

}}  // namespace forge::expr

#define FEXPR_UNARY_OPERATOR(_op, func)                                                            \
    template <class Scalar, class Expr, class DerivativeType>                                      \
    FEXPR_INLINE UnaryExpr<Scalar, func<Scalar>, Expr, DerivativeType>(_op)(                       \
        const Expression<Scalar, Expr, DerivativeType>& a)                                         \
    {                                                                                              \
        return UnaryExpr<Scalar, func<Scalar>, Expr, DerivativeType>(a.derived());                 \
    }                                                                                              \
    template <class Scalar1, std::size_t N = 1>                                                    \
    FEXPR_INLINE UnaryExpr<Scalar1, func<Scalar1>, ADVar<Scalar1, N>,                              \
                         typename DerivativesTraits<Scalar1, N>::type>(_op)(                       \
        const AReal<Scalar1, N>& a1)                                                               \
    {                                                                                              \
        return UnaryExpr<Scalar1, func<Scalar1>, ADVar<Scalar1, N>,                                \
                         typename DerivativesTraits<Scalar1, N>::type>(ADVar<Scalar1, N>(a1));     \
    }

#define FEXPR_UNARY_BINSCAL2(_op, func)                                                            \
    template <class Scalar, class Expr, class T2, class DerivativeType>                            \
    FEXPR_INLINE typename std::enable_if<                                                          \
        std::is_arithmetic<T2>::value && std::is_fundamental<T2>::value &&                         \
            !std::is_same<T2, typename ExprTraits<Expr>::nested_type>::value,                      \
        UnaryExpr<Scalar, func<Scalar, T2>, Expr, DerivativeType>>::                               \
        type(_op)(const Expression<Scalar, Expr, DerivativeType>& a, const T2& b)                  \
    {                                                                                              \
        return UnaryExpr<Scalar, func<Scalar, T2>, Expr, DerivativeType>(a.derived(),              \
                                                                         func<Scalar, T2>(b));     \
    }                                                                                              \
    template <class Scalar, class Expr, class DerivativeType>                                      \
    FEXPR_INLINE UnaryExpr<Scalar, func<Scalar, typename ExprTraits<Expr>::nested_type>, Expr,     \
                         DerivativeType>(_op)(const Expression<Scalar, Expr, DerivativeType>& a,   \
                                              typename ExprTraits<Expr>::nested_type b)            \
    {                                                                                              \
        return UnaryExpr<Scalar, func<Scalar, typename ExprTraits<Expr>::nested_type>, Expr,       \
                         DerivativeType>(a.derived(),                                              \
                                         func<Scalar, typename ExprTraits<Expr>::nested_type>(b)); \
    }                                                                                              \
    template <class Scalar1, class T21, std::size_t N>                                             \
    FEXPR_INLINE typename std::enable_if<                                                          \
        std::is_arithmetic<T21>::value && std::is_fundamental<T21>::value &&                       \
            !std::is_same<T21, typename ExprTraits<Scalar1>::nested_type>::value,                  \
        UnaryExpr<Scalar1, func<Scalar1, T21>, ADVar<Scalar1, N>,                                  \
                  typename DerivativesTraits<Scalar1, N>::type>>::type(_op)(const AReal<Scalar1,   \
                                                                                        N>& a1,    \
                                                                            const T21& b1)         \
    {                                                                                              \
        return UnaryExpr<Scalar1, func<Scalar1, T21>, ADVar<Scalar1, N>,                           \
                         typename DerivativesTraits<Scalar1, N>::type>(ADVar<Scalar1, N>(a1),      \
                                                                       func<Scalar1, T21>(b1));    \
    }                                                                                              \
    template <class Scalar, std::size_t M>                                                         \
    FEXPR_INLINE UnaryExpr<Scalar, func<Scalar, typename ExprTraits<Scalar>::nested_type>,         \
                         ADVar<Scalar, M>, typename DerivativesTraits<Scalar, M>::type>(_op)(      \
        const AReal<Scalar, M>& a, typename ExprTraits<Scalar>::nested_type b)                     \
    {                                                                                              \
        return UnaryExpr<Scalar, func<Scalar, typename ExprTraits<Scalar>::nested_type>,           \
                         ADVar<Scalar, M>, typename DerivativesTraits<Scalar, M>::type>(           \
            ADVar<Scalar, M>(a), func<Scalar, typename ExprTraits<Scalar>::nested_type>(b));       \
    }

#define FEXPR_UNARY_BINSCAL1(_op, func)                                                            \
    template <class Scalar2, class Expr1, class T22, class Deriv>                                  \
    FEXPR_INLINE typename std::enable_if<                                                          \
        std::is_arithmetic<T22>::value && std::is_fundamental<T22>::value &&                       \
            !std::is_same<T22, typename ExprTraits<Expr1>::nested_type>::value,                    \
        UnaryExpr<Scalar2, func<Scalar2, T22>, Expr1,                                              \
                  Deriv>>::type(_op)(const T22& a2, const Expression<Scalar2, Expr1, Deriv>& b2)   \
    {                                                                                              \
        return UnaryExpr<Scalar2, func<Scalar2, T22>, Expr1, Deriv>(b2.derived(),                  \
                                                                    func<Scalar2, T22>(a2));       \
    }                                                                                              \
    template <class Scalar2, class Expr1, class Deriv>                                             \
    FEXPR_INLINE                                                                                   \
    UnaryExpr<Scalar2, func<Scalar2, typename ExprTraits<Expr1>::nested_type>, Expr1, Deriv>(_op)( \
        typename ExprTraits<Expr1>::nested_type a2, const Expression<Scalar2, Expr1, Deriv>& b2)   \
    {                                                                                              \
        return UnaryExpr<Scalar2, func<Scalar2, typename ExprTraits<Expr1>::nested_type>, Expr1,   \
                         Deriv>(b2.derived(),                                                      \
                                func<Scalar2, typename ExprTraits<Expr1>::nested_type>(a2));       \
    }                                                                                              \
    template <class Scalar3, class T23, std::size_t N>                                             \
    FEXPR_INLINE typename std::enable_if<                                                          \
        std::is_arithmetic<T23>::value && std::is_fundamental<T23>::value &&                       \
            !std::is_same<T23, typename ExprTraits<Scalar3>::nested_type>::value,                  \
        UnaryExpr<Scalar3, func<Scalar3, T23>, ADVar<Scalar3, N>,                                  \
                  typename DerivativesTraits<Scalar3, N>::type>>::type(_op)(const T23& a3,         \
                                                                            const AReal<Scalar3,   \
                                                                                        N>& b3)    \
    {                                                                                              \
        return UnaryExpr<Scalar3, func<Scalar3, T23>, ADVar<Scalar3, N>,                           \
                         typename DerivativesTraits<Scalar3, N>::type>(ADVar<Scalar3, N>(b3),      \
                                                                       func<Scalar3, T23>(a3));    \
    }                                                                                              \
    template <class Scalar3, std::size_t N>                                                        \
    FEXPR_INLINE UnaryExpr<Scalar3, func<Scalar3, typename ExprTraits<Scalar3>::nested_type>,      \
                         ADVar<Scalar3, N>, typename DerivativesTraits<Scalar3, N>::type>(_op)(    \
        typename ExprTraits<Scalar3>::nested_type a3, const AReal<Scalar3, N>& b3)                 \
    {                                                                                              \
        return UnaryExpr<Scalar3, func<Scalar3, typename ExprTraits<Scalar3>::nested_type>,        \
                         ADVar<Scalar3, N>, typename DerivativesTraits<Scalar3, N>::type>(         \
            ADVar<Scalar3, N>(b3), func<Scalar3, typename ExprTraits<Scalar3>::nested_type>(a3));  \
    }

#define FEXPR_UNARY_BINSCAL(_op, func)                                                             \
    FEXPR_UNARY_BINSCAL1(_op, func)                                                                \
    FEXPR_UNARY_BINSCAL2(_op, func)

#define FEXPR_MAKE_UNARY_FUNC(func)                                                                \
    template <class Scalar, class Expr, class DerivativeType>                                      \
    FEXPR_INLINE UnaryExpr<Scalar, func##_op<Scalar>, Expr, DerivativeType>(func)(                 \
        const Expression<Scalar, Expr, DerivativeType>& x0)                                        \
    {                                                                                              \
        return UnaryExpr<Scalar, func##_op<Scalar>, Expr, DerivativeType>(x0.derived());           \
    }                                                                                              \
    template <class Scalar1, std::size_t N>                                                        \
    FEXPR_INLINE UnaryExpr<Scalar1, func##_op<Scalar1>, ADVar<Scalar1, N>,                         \
                         typename DerivativesTraits<Scalar1, N>::type>(func)(                      \
        const AReal<Scalar1, N>& x)                                                                \
    {                                                                                              \
        return UnaryExpr<Scalar1, func##_op<Scalar1>, ADVar<Scalar1, N>,                           \
                         typename DerivativesTraits<Scalar1, N>::type>(ADVar<Scalar1, N>(x));      \
    }                                                                                              \
    template <class Scalar, std::size_t N>                                                         \
    FEXPR_INLINE FRealDirect<Scalar, N>(func)(const FRealDirect<Scalar, N>& x)                     \
    {                                                                                              \
        return {FReal<Scalar, N>((func)(x.base()))};                                               \
    }                                                                                              \
    template <class Scalar, std::size_t N>                                                         \
    FEXPR_INLINE ARealDirect<Scalar, N>(func)(const ARealDirect<Scalar, N>& x)                     \
    {                                                                                              \
        return {AReal<Scalar, N>((func)(x.base()))};                                               \
    }

#define FEXPR_MAKE_FPCLASSIFY_FUNC_RET(ret, func, using)                                           \
    template <class Scalar, class Expr, class DerivativeType>                                      \
    FEXPR_INLINE ret(func)(const Expression<Scalar, Expr, DerivativeType>& x)                      \
    {                                                                                              \
        using;                                                                                     \
        return func(x.value());                                                                    \
    }                                                                                              \
    template <class Scalar, std::size_t M = 1>                                                     \
    FEXPR_INLINE ret(func)(const AReal<Scalar, M>& x)                                              \
    {                                                                                              \
        using;                                                                                     \
        return func(x.value());                                                                    \
    }                                                                                              \
    template <class Scalar, std::size_t N>                                                         \
    FEXPR_INLINE ret(func)(const FReal<Scalar, N>& x)                                              \
    {                                                                                              \
        return func(x.value());                                                                    \
    }                                                                                              \
    template <class Scalar, std::size_t N>                                                         \
    FEXPR_INLINE ret(func)(const forge::expr::FRealDirect<Scalar, N>& x)                           \
    {                                                                                              \
        return func(x.value());                                                                    \
    }                                                                                              \
    template <class Scalar, std::size_t N>                                                         \
    FEXPR_INLINE ret(func)(const forge::expr::ARealDirect<Scalar, N>& x)                           \
    {                                                                                              \
        return func(x.value());                                                                    \
    }                                                                                              \
    template <class Scalar, class Expr, class Op, class DerivativeType>                            \
    FEXPR_INLINE ret(func)(const UnaryExpr<Scalar, Op, Expr, DerivativeType>& x)                   \
    {                                                                                              \
        using;                                                                                     \
        return func(x.value());                                                                    \
    }                                                                                              \
    template <class Scalar, class Op, class Expr1, class Expr2, class Deriv>                       \
    FEXPR_INLINE ret(func)(const BinaryExpr<Scalar, Op, Expr1, Expr2, Deriv>& x)                   \
    {                                                                                              \
        using;                                                                                     \
        return func(x.value());                                                                    \
    }

#define FEXPR_MAKE_FPCLASSIFY_FUNC(func, using) FEXPR_MAKE_FPCLASSIFY_FUNC_RET(bool, func, using)

