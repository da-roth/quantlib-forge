/*******************************************************************************

   Overloaded operators for binary functions.

   This file is part of QuantLib-Forge, a Forge integration layer for QuantLib.

   Copyright (C) 2025 The QuantLib-Forge Authors

   Derived from XAD (https://github.com/auto-differentiation/XAD)
   Original code: Copyright (C) 2010-2024 Xcelerit Computing Ltd.

   SPDX-License-Identifier: AGPL-3.0-or-later

******************************************************************************/

#pragma once

#include <expressions/ExpressionTemplates/BinaryExpr.hpp>
#include <expressions/ExpressionTemplates/BinaryFunctors.hpp>
#include <expressions/ExpressionTemplates/BinaryMathFunctors.hpp>

#include <expressions/ExpressionTemplates/BinaryOperatorMacros.hpp>
#include <expressions/Macros.hpp>

namespace forge { namespace expr {

FEXPR_BINARY_OPERATOR(operator+, add_op)
FEXPR_BINARY_OPERATOR(operator*, prod_op)
FEXPR_BINARY_OPERATOR(operator-, sub_op)
FEXPR_BINARY_OPERATOR(operator/, div_op)
FEXPR_BINARY_OPERATOR(pow, pow_op)
FEXPR_BINARY_OPERATOR(max, max_op)
FEXPR_BINARY_OPERATOR(fmax, fmax_op)
FEXPR_BINARY_OPERATOR(min, min_op)
FEXPR_BINARY_OPERATOR(fmin, fmin_op)
FEXPR_BINARY_OPERATOR(fmod, fmod_op)
FEXPR_BINARY_OPERATOR(atan2, atan2_op)
FEXPR_BINARY_OPERATOR(hypot, hypot_op)
FEXPR_BINARY_OPERATOR(smooth_abs, smooth_abs_op)
FEXPR_BINARY_OPERATOR(nextafter, nextafter_op)

// note - this is C++11 only
template <class T1, class T2, class T3>
FEXPR_INLINE auto smooth_max(const T1& x, const T2& y,
                           const T3& c) -> decltype(0.5 * (x + y + smooth_abs(x - y, c)))
{
    return 0.5 * (x + y + smooth_abs(x - y, c));
}
template <class T1, class T2>
FEXPR_INLINE auto smooth_max(const T1& x, const T2& y) -> decltype(0.5 * (x + y + smooth_abs(x - y)))
{
    return 0.5 * (x + y + smooth_abs(x - y));
}
template <class T1, class T2, class T3>
FEXPR_INLINE auto smooth_min(const T1& x, const T2& y,
                           const T3& c) -> decltype(0.5 * (x + y - smooth_abs(x - y, c)))
{
    return 0.5 * (x + y - smooth_abs(x - y, c));
}
template <class T1, class T2>
FEXPR_INLINE auto smooth_min(const T1& x, const T2& y) -> decltype(0.5 * (x + y - smooth_abs(x - y)))
{
    return 0.5 * (x + y - smooth_abs(x - y));
}

template <class T, class = typename std::enable_if<forge::expr::ExprTraits<T>::isExpr>>
FEXPR_INLINE auto fma(const T& a, const T& b, const T& c) -> decltype(a * b + c)
{
    return a * b + c;
}

template <
    class T1, class T2, class T3,
    class = typename std::enable_if<(forge::expr::ExprTraits<T1>::isExpr || forge::expr::ExprTraits<T2>::isExpr ||
                                     forge::expr::ExprTraits<T3>::isExpr)>>
FEXPR_INLINE auto fma(const T1& a, const T2& b, const T3& c) -> decltype(a * b + c)
{
    return a * b + c;
}

/////////// comparisons - they just return bool

#define FEXPR_COMPARE_OPERATOR(op, opname)                                                           \
    template <class Scalar, class Expr1, class Expr2, class DerivativeType>                        \
    FEXPR_INLINE bool operator op(const Expression<Scalar, Expr1, DerivativeType>& a,                \
                                const Expression<Scalar, Expr2, DerivativeType>& b)                \
    {                                                                                              \
        return value(a) op value(b);                                                               \
    }                                                                                              \
    template <class Scalar, class Expr, class DerivativeType>                                      \
    FEXPR_INLINE bool operator op(const typename ExprTraits<Expr>::value_type& a,                    \
                                const Expression<Scalar, Expr, DerivativeType>& b)                 \
    {                                                                                              \
        return value(a) op value(b);                                                               \
    }                                                                                              \
    template <class Scalar, class Expr, class DerivativeType>                                      \
    FEXPR_INLINE bool operator op(const Expression<Scalar, Expr, DerivativeType>& a,                 \
                                const typename ExprTraits<Expr>::value_type& b)                    \
    {                                                                                              \
        return value(a) op value(b);                                                               \
    }                                                                                              \
    template <class Scalar, std::size_t M = 1>                                                     \
    FEXPR_INLINE bool operator op(const AReal<Scalar, M>& a, const AReal<Scalar, M>& b)              \
    {                                                                                              \
        return value(a) op value(b);                                                               \
    }                                                                                              \
    template <class Scalar, std::size_t N>                                                         \
    FEXPR_INLINE bool operator op(const FReal<Scalar, N>& a, const FReal<Scalar, N>& b)              \
    {                                                                                              \
        return value(a) op value(b);                                                               \
    }                                                                                              \
    template <class Scalar, std::size_t N>                                                         \
    FEXPR_INLINE bool operator op(const FRealDirect<Scalar, N>& a, const FRealDirect<Scalar, N>& b)  \
    {                                                                                              \
        return value(a) op value(b);                                                               \
    }                                                                                              \
    template <class Scalar, std::size_t N>                                                         \
    FEXPR_INLINE bool operator op(const Scalar& a, const FRealDirect<Scalar, N>& b)                  \
    {                                                                                              \
        return a op value(b);                                                                      \
    }                                                                                              \
    template <class Scalar, std::size_t N>                                                         \
    FEXPR_INLINE bool operator op(const FRealDirect<Scalar, N>& a, const Scalar& b)                  \
    {                                                                                              \
        return value(a) op b;                                                                      \
    }                                                                                              \
    template <class Scalar, std::size_t N>                                                         \
    FEXPR_INLINE bool operator op(const ARealDirect<Scalar, N>& a, const ARealDirect<Scalar, N>& b)  \
    {                                                                                              \
        return value(a) op value(b);                                                               \
    }                                                                                              \
    template <class Scalar, std::size_t N>                                                         \
    FEXPR_INLINE bool operator op(const Scalar& a, const ARealDirect<Scalar, N>& b)                  \
    {                                                                                              \
        return a op value(b);                                                                      \
    }                                                                                              \
    template <class Scalar, std::size_t N>                                                         \
    FEXPR_INLINE bool operator op(const ARealDirect<Scalar, N>& a, const Scalar& b)                  \
    {                                                                                              \
        return value(a) op b;                                                                      \
    }                                                                                              \
                                                                                                   \
    template <class Scalar, class Expr, std::size_t N>                                             \
    FEXPR_INLINE bool operator op(typename ExprTraits<Scalar>::nested_type a,                        \
                                const FRealDirect<Scalar, N>& b)                                   \
    {                                                                                              \
        return a op value(b);                                                                      \
    }                                                                                              \
    template <class Scalar, class Expr, std::size_t N>                                             \
    FEXPR_INLINE bool operator op(const FRealDirect<Scalar, N>& a,                                   \
                                typename ExprTraits<Scalar>::nested_type b)                        \
    {                                                                                              \
        return value(a) op b;                                                                      \
    }                                                                                              \
    template <class Scalar, class Expr, std::size_t N>                                             \
    FEXPR_INLINE bool operator op(typename ExprTraits<Scalar>::nested_type a,                        \
                                const ARealDirect<Scalar, N>& b)                                   \
    {                                                                                              \
        return a op value(b);                                                                      \
    }                                                                                              \
    template <class Scalar, class Expr, std::size_t N>                                             \
    FEXPR_INLINE bool operator op(const ARealDirect<Scalar, N>& a,                                   \
                                typename ExprTraits<Scalar>::nested_type b)                        \
    {                                                                                              \
        return value(a) op b;                                                                      \
    }                                                                                              \
                                                                                                   \
    template <class Scalar, class Expr, class DerivativeType>                                      \
    FEXPR_INLINE bool operator op(typename ExprTraits<Expr>::nested_type a,                          \
                                const Expression<Scalar, Expr, DerivativeType>& b)                 \
    {                                                                                              \
        return a op value(b);                                                                      \
    }                                                                                              \
    template <class Scalar, class Expr, class DerivativeType>                                      \
    FEXPR_INLINE bool operator op(const Expression<Scalar, Expr, DerivativeType>& a,                 \
                                typename ExprTraits<Expr>::nested_type b)                          \
    {                                                                                              \
        return value(a) op b;                                                                      \
    }

FEXPR_COMPARE_OPERATOR(==, "==")
FEXPR_COMPARE_OPERATOR(!=, "!=")
FEXPR_COMPARE_OPERATOR(<=, "<=")
FEXPR_COMPARE_OPERATOR(>=, ">=")
FEXPR_COMPARE_OPERATOR(<, "<")
FEXPR_COMPARE_OPERATOR(>, ">")

FEXPR_BINARY_OPERATOR(remainder, remainder_op)

// manual remquo due to additional argument
template <class Scalar, class Expr1, class Expr2>
FEXPR_INLINE BinaryExpr<Scalar, remquo_op<Scalar>, Expr1, Expr2> remquo(
    const Expression<Scalar, Expr1>& a, const Expression<Scalar, Expr2>& b, int* quo)
{
    return BinaryExpr<Scalar, remquo_op<Scalar>, Expr1, Expr2>(a.derived(), b.derived(),
                                                               remquo_op<Scalar>(quo));
}

template <class Scalar, std::size_t M>
FEXPR_INLINE BinaryExpr<Scalar, remquo_op<Scalar>, ADVar<Scalar, M>, ADVar<Scalar, M>> remquo(
    const AReal<Scalar, M>& a, const AReal<Scalar, M>& b, int* quo)
{
    return BinaryExpr<Scalar, remquo_op<Scalar>, ADVar<Scalar, M>, ADVar<Scalar, M>>(
        ADVar<Scalar, M>(a), ADVar<Scalar, M>(b), remquo_op<Scalar>(quo));
}

template <class Scalar, std::size_t N>
FEXPR_INLINE BinaryExpr<Scalar, remquo_op<Scalar>, FReal<Scalar, N>, FReal<Scalar, N>> remquo(
    const FReal<Scalar, N>& a, const FReal<Scalar, N>& b, int* quo)
{
    return BinaryExpr<Scalar, remquo_op<Scalar>, FReal<Scalar, N>, FReal<Scalar, N>>(
        a, b, remquo_op<Scalar>(quo));
}

template <class T, class = typename std::enable_if<float_or_double<T>::value>::type, std::size_t N>
FEXPR_INLINE forge::expr::FRealDirect<T, N> remquo(const forge::expr::FRealDirect<T, N>& a,
                                         const forge::expr::FRealDirect<T, N>& b, int* c)
{
    return forge::expr::FReal<T, N>(remquo(a.base(), b.base(), c));
}

template <class T, class = typename std::enable_if<float_or_double<T>::value>::type, std::size_t N>
FEXPR_INLINE forge::expr::FRealDirect<T, N> remquo(const forge::expr::FRealDirect<T, N>& a, const T& b, int* c)
{
    return forge::expr::FReal<T, N>(remquo(a.base(), b, c));
}

template <class T, class = typename std::enable_if<float_or_double<T>::value>::type, std::size_t N>
FEXPR_INLINE forge::expr::FRealDirect<T, N> remquo(const T& a, const forge::expr::FRealDirect<T, N>& b, int* c)
{
    return forge::expr::FReal<T, N>(remquo(a, b.base(), c));
}

template <class T, class = typename std::enable_if<float_or_double<T>::value>::type, std::size_t N>
FEXPR_INLINE forge::expr::ARealDirect<T, N> remquo(const forge::expr::ARealDirect<T, N>& a,
                                         const forge::expr::ARealDirect<T, N>& b, int* c)
{
    return forge::expr::AReal<T, N>(remquo(a.base(), b.base(), c));
}

template <class T, class = typename std::enable_if<float_or_double<T>::value>::type, std::size_t N>
FEXPR_INLINE forge::expr::ARealDirect<T, N> remquo(const forge::expr::ARealDirect<T, N>& a, const T& b, int* c)
{
    return forge::expr::AReal<T, N>(remquo(a.base(), b, c));
}

template <class T, class = typename std::enable_if<float_or_double<T>::value>::type, std::size_t N>
FEXPR_INLINE forge::expr::ARealDirect<T, N> remquo(const T& a, const forge::expr::ARealDirect<T, N>& b, int* c)
{
    return forge::expr::AReal<T, N>(remquo(a, b.base(), c));
}

template <class Scalar, class Expr>
FEXPR_INLINE BinaryExpr<Scalar, remquo_op<Scalar>, typename ExprTraits<Expr>::value_type, Expr>
remquo(const typename ExprTraits<Expr>::value_type& a, const Expression<Scalar, Expr>& b, int* quo)
{
    return BinaryExpr<Scalar, remquo_op<Scalar>, typename ExprTraits<Expr>::value_type, Expr>(
        typename ExprTraits<Expr>::value_type(a), b.derived(), remquo_op<Scalar>(quo));
}

template <class Scalar, class Expr>
FEXPR_INLINE BinaryExpr<Scalar, remquo_op<Scalar>, Expr, typename ExprTraits<Expr>::value_type>
remquo(const Expression<Scalar, Expr>& a, const typename ExprTraits<Expr>::value_type& b, int* quo)
{
    return BinaryExpr<Scalar, remquo_op<Scalar>, Expr, typename ExprTraits<Expr>::value_type>(
        a.derived(), typename ExprTraits<Expr>::value_type(b), remquo_op<Scalar>(quo));
}

template <class T, class = typename std::enable_if<float_or_double<T>::value>::type, std::size_t N>
FEXPR_INLINE forge::expr::FRealDirect<T, N> frexp(const forge::expr::FRealDirect<T, N>& a, int* exp)
{
    return forge::expr::FReal<T, N>(frexp(a.base(), exp));
}

template <class T, class = typename std::enable_if<float_or_double<T>::value>::type, std::size_t N>
FEXPR_INLINE forge::expr::FRealDirect<T, N> ldexp(const forge::expr::FRealDirect<T, N>& a, int b)
{
    return forge::expr::FReal<T, N>(ldexp(a.base(), b));
}

template <class T, class = typename std::enable_if<float_or_double<T>::value>::type, std::size_t N>
FEXPR_INLINE forge::expr::FRealDirect<T, N> modf(const forge::expr::FRealDirect<T, N>& a, forge::expr::FRealDirect<T, N>* b)
{
    return forge::expr::FReal<T, N>(modf(a.base(), &b->base()));
}

template <class T, class = typename std::enable_if<float_or_double<T>::value>::type, std::size_t N>
FEXPR_INLINE forge::expr::FRealDirect<T, N> modf(const forge::expr::FRealDirect<T, N>& a, T* b)
{
    return forge::expr::FReal<T, N>(modf(a.base(), b));
}

template <class T, class = typename std::enable_if<float_or_double<T>::value>::type, std::size_t N>
FEXPR_INLINE forge::expr::ARealDirect<T, N> frexp(const forge::expr::ARealDirect<T, N>& a, int* exp)
{
    return forge::expr::AReal<T, N>(frexp(a.base(), exp));
}

template <class T, class = typename std::enable_if<float_or_double<T>::value>::type, std::size_t N>
FEXPR_INLINE forge::expr::ARealDirect<T, N> ldexp(const forge::expr::ARealDirect<T, N>& a, int b)
{
    return forge::expr::AReal<T, N>(ldexp(a.base(), b));
}

template <class T, class = typename std::enable_if<float_or_double<T>::value>::type, std::size_t N>
FEXPR_INLINE forge::expr::ARealDirect<T, N> modf(const forge::expr::ARealDirect<T, N>& a, forge::expr::ARealDirect<T, N>* b)
{
    return forge::expr::AReal<T, N>(modf(a.base(), &b->base()));
}

template <class T, class = typename std::enable_if<float_or_double<T>::value>::type, std::size_t N>
FEXPR_INLINE forge::expr::ARealDirect<T, N> modf(const forge::expr::ARealDirect<T, N>& a, T* b)
{
    return forge::expr::AReal<T, N>(modf(a.base(), b));
}

#undef FEXPR_BINARY_OPERATOR
#undef FEXPR_COMPARE_OPERATOR

}}  // namespace forge::expr

