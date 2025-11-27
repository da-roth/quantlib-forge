/*******************************************************************************

   Macros used for binary operator declarations.

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
template <class, std::size_t>
struct FReal;
template <class, std::size_t>
struct FRealDirect;
template <class, std::size_t>
struct ARealDirect;

template <class Scalar, class Expr, class Enable = void>
struct wrapper_type
{
    typedef typename ExprTraits<Expr>::value_type type;
};

template <class Scalar, class Expr>
struct wrapper_type<Scalar, Expr, typename std::enable_if<ExprTraits<Expr>::isReverse>::type>
{
    typedef ADVar<Scalar, ExprTraits<Expr>::vector_size> type;
};

template <class T>
struct is_vec : std::false_type
{
};

template <class T, std::size_t N>
struct is_vec<Vec<T, N>> : std::true_type
{
};

}}  // namespace forge::expr

#define FEXPR_BINARY_OPERATOR(_op, func)                                                             \
    template <class Scalar, class Expr1, class Expr2, class DerivativeType>                        \
    FEXPR_INLINE BinaryExpr<Scalar, func<Scalar>, Expr1, Expr2, DerivativeType>(_op)(                \
        const Expression<Scalar, Expr1, DerivativeType>& a1,                                       \
        const Expression<Scalar, Expr2, DerivativeType>& b1)                                       \
    {                                                                                              \
        return BinaryExpr<Scalar, func<Scalar>, Expr1, Expr2, DerivativeType>(a1.derived(),        \
                                                                              b1.derived());       \
    }                                                                                              \
    template <class Scalar, std::size_t N>                                                         \
    FEXPR_INLINE typename std::enable_if<                                                            \
        std::is_floating_point<typename ExprTraits<Scalar>::nested_type>::value &&                 \
            std::is_fundamental<typename ExprTraits<Scalar>::nested_type>::value,                  \
        BinaryExpr<Scalar, func<Scalar>, ADVar<Scalar, N>, ADVar<Scalar, N>,                       \
                   typename DerivativesTraits<Scalar, N>::type>>::                                 \
        type(_op)(const AReal<Scalar, N>& a2, const AReal<Scalar, N>& b2)                          \
    {                                                                                              \
        return BinaryExpr<Scalar, func<Scalar>, ADVar<Scalar, N>, ADVar<Scalar, N>,                \
                          typename DerivativesTraits<Scalar, N>::type>(ADVar<Scalar, N>(a2),       \
                                                                       ADVar<Scalar, N>(b2));      \
    }                                                                                              \
    template <class Scalar, class Expr, class DerivativeType>                                      \
    FEXPR_INLINE BinaryExpr<Scalar, func<Scalar>, typename wrapper_type<Scalar, Expr>::type, Expr,   \
                          DerivativeType>(_op)(const typename ExprTraits<Expr>::value_type& a3,    \
                                               const Expression<Scalar, Expr, DerivativeType>& b3) \
    {                                                                                              \
        return BinaryExpr<Scalar, func<Scalar>, typename wrapper_type<Scalar, Expr>::type, Expr,   \
                          DerivativeType>(typename wrapper_type<Scalar, Expr>::type(a3),           \
                                          b3.derived());                                           \
    }                                                                                              \
    template <class Scalar, class Expr, class DerivativeType>                                      \
    FEXPR_INLINE BinaryExpr<Scalar, func<Scalar>, Expr, typename wrapper_type<Scalar, Expr>::type,   \
                          DerivativeType>(_op)(const Expression<Scalar, Expr, DerivativeType>& a4, \
                                               const typename ExprTraits<Expr>::value_type& b4)    \
    {                                                                                              \
        return BinaryExpr<Scalar, func<Scalar>, Expr, typename wrapper_type<Scalar, Expr>::type,   \
                          DerivativeType>(a4.derived(),                                            \
                                          typename wrapper_type<Scalar, Expr>::type(b4));          \
    }                                                                                              \
    template <class Scalar, std::size_t N>                                                         \
    FEXPR_INLINE typename std::enable_if<                                                            \
        std::is_floating_point<typename ExprTraits<Scalar>::nested_type>::value &&                 \
            std::is_fundamental<typename ExprTraits<Scalar>::nested_type>::value,                  \
        BinaryExpr<Scalar, func<Scalar>, FReal<Scalar, N>, FReal<Scalar, N>,                       \
                   typename FReal<Scalar, N>::derivative_type>>::                                  \
        type(_op)(const FReal<Scalar, N>& a2, const FReal<Scalar, N>& b2)                          \
    {                                                                                              \
        return BinaryExpr<Scalar, func<Scalar>, FReal<Scalar, N>, FReal<Scalar, N>,                \
                          typename FReal<Scalar, N>::derivative_type>(a2, b2);                     \
    }                                                                                              \
    template <class Scalar, class = typename std::enable_if<float_or_double<Scalar>::value>::type, \
              std::size_t N>                                                                       \
    FEXPR_INLINE FRealDirect<Scalar, N>(_op)(const FRealDirect<Scalar, N>& a,                        \
                                           const FRealDirect<Scalar, N>& b)                        \
    {                                                                                              \
        return {FReal<Scalar, N>((_op)(a.base(), b.base()))};                                      \
    }                                                                                              \
    template <class Scalar, class T, std::size_t N,                                                \
              class = typename std::enable_if<float_or_double<Scalar>::value &&                    \
                                              !is_vec<T>::value>::type>                            \
    FEXPR_INLINE FRealDirect<Scalar, N>(_op)(const FRealDirect<Scalar, N>& a, const T& b)            \
    {                                                                                              \
        return {FReal<Scalar, N>((_op)(a.base(), b))};                                             \
    }                                                                                              \
    template <class Scalar, class T, std::size_t N,                                                \
              class = typename std::enable_if<float_or_double<Scalar>::value &&                    \
                                              !is_vec<T>::value>::type>                            \
    FEXPR_INLINE FRealDirect<Scalar, N>(_op)(const T& a, const FRealDirect<Scalar, N>& b)            \
    {                                                                                              \
        return {FReal<Scalar, N>((_op)(a, b.base()))};                                             \
    }                                                                                              \
    template <class Scalar, class = typename std::enable_if<float_or_double<Scalar>::value>::type, \
              std::size_t N>                                                                       \
    FEXPR_INLINE ARealDirect<Scalar, N>(_op)(const ARealDirect<Scalar, N>& a,                        \
                                           const ARealDirect<Scalar, N>& b)                        \
    {                                                                                              \
        return {AReal<Scalar, N>((_op)(a.base(), b.base()))};                                      \
    }                                                                                              \
    template <class Scalar, class T, std::size_t N,                                                \
              class = typename std::enable_if<float_or_double<Scalar>::value &&                    \
                                              !is_vec<T>::value>::type>                            \
    FEXPR_INLINE ARealDirect<Scalar, N>(_op)(const ARealDirect<Scalar, N>& a, const T& b)            \
    {                                                                                              \
        return {AReal<Scalar, N>((_op)(a.base(), b))};                                             \
    }                                                                                              \
    template <class Scalar, class T, std::size_t N,                                                \
              class = typename std::enable_if<float_or_double<Scalar>::value &&                    \
                                              !is_vec<T>::value>::type>                            \
    FEXPR_INLINE ARealDirect<Scalar, N>(_op)(const T& a, const ARealDirect<Scalar, N>& b)            \
    {                                                                                              \
        return {AReal<Scalar, N>((_op)(a, b.base()))};                                             \
    }

