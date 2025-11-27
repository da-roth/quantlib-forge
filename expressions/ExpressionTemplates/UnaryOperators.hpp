/*******************************************************************************

   Overloads of operators that translate to unary functors.

   This file is part of QuantLib-Forge, a Forge integration layer for QuantLib.

   Copyright (C) 2025 The QuantLib-Forge Authors

   Derived from XAD (https://github.com/auto-differentiation/XAD)
   Original code: Copyright (C) 2010-2024 Xcelerit Computing Ltd.

   SPDX-License-Identifier: AGPL-3.0-or-later

******************************************************************************/

#pragma once

#include <expressions/ExpressionTemplates/BinaryExpr.hpp>
#include <expressions/ExpressionTemplates/UnaryExpr.hpp>
#include <expressions/ExpressionTemplates/UnaryMathFunctors.hpp>

#include <expressions/Macros.hpp>
#include <expressions/ExpressionTemplates/UnaryOperatorMacros.hpp>

namespace forge { namespace expr {

template <class, std::size_t>
struct FReal;
template <class, std::size_t>
struct AReal;
template <class, std::size_t>
struct FRealDirect;
template <class, std::size_t>
struct ARealDirect;

// unary plus - does nothing
template <class Scalar, class Expr, class DerivativeType>
FEXPR_INLINE const Expression<Scalar, Expr, DerivativeType>& operator+(
    const Expression<Scalar, Expr, DerivativeType>& a)
{
    return a;
}

template <class Scalar, std::size_t M = 1>
FEXPR_INLINE const ADVar<Scalar, M> operator+(const ADVar<Scalar, M>& a)
{
    return ADVar<Scalar, M>(a);
}

FEXPR_UNARY_OPERATOR(operator-, negate_op)
FEXPR_UNARY_BINSCAL(operator+, scalar_add_op)
FEXPR_UNARY_BINSCAL(operator*, scalar_prod_op)
FEXPR_UNARY_BINSCAL1(operator-, scalar_sub1_op)
FEXPR_UNARY_BINSCAL2(operator-, scalar_sub2_op)
FEXPR_UNARY_BINSCAL1(operator/, scalar_div1_op)
FEXPR_UNARY_BINSCAL2(operator/, scalar_div2_op)
FEXPR_UNARY_BINSCAL1(pow, scalar_pow1_op)
FEXPR_UNARY_BINSCAL2(pow, scalar_pow2_op)
FEXPR_UNARY_BINSCAL1(smooth_abs, scalar_smooth_abs1_op)
FEXPR_UNARY_BINSCAL2(smooth_abs, scalar_smooth_abs2_op)

template <class Scalar, class Expr, class DerivativeType>
FEXPR_INLINE UnaryExpr<Scalar, scalar_smooth_abs2_op<Scalar, typename ExprTraits<Expr>::nested_type>,
                     Expr, DerivativeType>
smooth_abs(const Expression<Scalar, Expr, DerivativeType>& a)
{
    return smooth_abs(a, typename ExprTraits<Expr>::nested_type(0.001));
}

template <class Scalar, std::size_t M>
FEXPR_INLINE UnaryExpr<Scalar, scalar_smooth_abs2_op<Scalar, typename AReal<Scalar, M>::nested_type>,
                     ADVar<Scalar, M>, typename DerivativesTraits<Scalar, M>::type>
smooth_abs(const AReal<Scalar, M>& a)
{
    return smooth_abs(a, typename AReal<Scalar, M>::nested_type(0.001));
}

FEXPR_UNARY_BINSCAL1(fmod, scalar_fmod1_op)
FEXPR_UNARY_BINSCAL2(fmod, scalar_fmod2_op)
FEXPR_UNARY_BINSCAL1(atan2, scalar_atan21_op)
FEXPR_UNARY_BINSCAL2(atan2, scalar_atan22_op)
FEXPR_UNARY_BINSCAL1(nextafter, scalar_nextafter1_op)
FEXPR_UNARY_BINSCAL2(nextafter, scalar_nextafter2_op)
FEXPR_UNARY_BINSCAL1(hypot, scalar_hypot1_op)
FEXPR_UNARY_BINSCAL2(hypot, scalar_hypot2_op)

// pown (integral exponents)
template <class Scalar, class Expr, class DerivativeType>
FEXPR_INLINE UnaryExpr<Scalar, scalar_pow2_op<Scalar, int>, Expr, DerivativeType> pown(
    const Expression<Scalar, Expr, DerivativeType>& x, int y)
{
    return pow(x, y);
}
template <class Scalar, std::size_t M = 1>
FEXPR_INLINE UnaryExpr<Scalar, scalar_pow2_op<Scalar, int>, ADVar<Scalar, M>,
                     typename DerivativesTraits<Scalar, M>::type>
pown(const AReal<Scalar, M>& x, int y)
{
    return pow(x, y);
}

// ldexp
template <class Scalar, class Expr, class DerivativeType>
FEXPR_INLINE UnaryExpr<Scalar, ldexp_op<Scalar>, Expr, DerivativeType> ldexp(
    const Expression<Scalar, Expr, DerivativeType>& x, int y)
{
    return UnaryExpr<Scalar, ldexp_op<Scalar>, Expr, DerivativeType>(x.derived(),
                                                                     ldexp_op<Scalar>(y));
}

template <class Scalar, std::size_t M = 1>
FEXPR_INLINE UnaryExpr<Scalar, ldexp_op<Scalar>, AReal<Scalar, M>,
                     typename DerivativesTraits<Scalar, M>::type>
ldexp(const AReal<Scalar, M>& x, int y)
{
    return UnaryExpr<Scalar, ldexp_op<Scalar>, AReal<Scalar, M>,
                     typename DerivativesTraits<Scalar, M>::type>(AReal<Scalar, M>(x),
                                                                  ldexp_op<Scalar>(y));
}

// frexp
template <class Scalar, class Expr, class DerivativeType>
FEXPR_INLINE UnaryExpr<Scalar, frexp_op<Scalar>, Expr, DerivativeType> frexp(
    const Expression<Scalar, Expr, DerivativeType>& x, int* exp)
{
    return UnaryExpr<Scalar, frexp_op<Scalar>, Expr, DerivativeType>(x.derived(),
                                                                     frexp_op<Scalar>(exp));
}

template <class Scalar, std::size_t M = 1>
FEXPR_INLINE UnaryExpr<Scalar, frexp_op<Scalar>, AReal<Scalar, M>,
                     typename DerivativesTraits<Scalar, M>::type>
frexp(const AReal<Scalar, M>& x, int* exp)
{
    return UnaryExpr<Scalar, frexp_op<Scalar>, AReal<Scalar, M>,
                     typename DerivativesTraits<Scalar, M>::type>(AReal<Scalar, M>(x),
                                                                  frexp_op<Scalar>(exp));
}

// modf - only enabled if iptr is nested type (double) or Scalar
template <class Scalar, class Expr, class T, class DerivativeType>
FEXPR_INLINE UnaryExpr<Scalar, modf_op<Scalar, T>, Expr, DerivativeType> modf(
    const Expression<Scalar, Expr, DerivativeType>& x, T* iptr)
{
    return UnaryExpr<Scalar, modf_op<Scalar, T>, Expr, DerivativeType>(x.derived(),
                                                                       modf_op<Scalar, T>(iptr));
}

template <class Scalar, class T, std::size_t M = 1>
FEXPR_INLINE UnaryExpr<Scalar, modf_op<Scalar, T>, AReal<Scalar, M>,
                     typename DerivativesTraits<Scalar, M>::type>
modf(const AReal<Scalar, M>& x, T* iptr)
{
    return UnaryExpr<Scalar, modf_op<Scalar, T>, AReal<Scalar, M>,
                     typename DerivativesTraits<Scalar, M>::type>(AReal<Scalar, M>(x),
                                                                  modf_op<Scalar, T>(iptr));
}

// we put max/min here explicitly, as the 2 arguments to them must match
// and we need to avoid conflicts with the standard versions

template <class Scalar, class Expr, class DerivativeType>
FEXPR_INLINE UnaryExpr<Scalar, scalar_max_op<Scalar, Scalar>, Expr, DerivativeType>(max)(
    Scalar a2, const Expression<Scalar, Expr, DerivativeType>& b2)
{
    return UnaryExpr<Scalar, scalar_max_op<Scalar, Scalar>, Expr, DerivativeType>(
        b2.derived(), scalar_max_op<Scalar, Scalar>(a2));
}

template <class Scalar, class Expr, class T, class DerivativeType>
FEXPR_INLINE typename std::enable_if<
    std::is_same<T, typename ExprTraits<Expr>::nested_type>::value,
    UnaryExpr<Scalar, scalar_max_op<Scalar, Scalar>, Expr,
              DerivativeType>>::type(max)(T a2, const Expression<Scalar, Expr, DerivativeType>& b2)
{
    return UnaryExpr<Scalar, scalar_max_op<Scalar, Scalar>, Expr, DerivativeType>(
        b2.derived(), scalar_max_op<Scalar, Scalar>(a2));
}

template <class T, class AT, std::size_t M = 1>
FEXPR_INLINE typename std::enable_if<
    std::is_same<T, typename ExprTraits<AT>::nested_type>::value &&
        std::is_same<AReal<typename ExprTraits<AT>::scalar_type, M>, AT>::value,
    UnaryExpr<
        typename ExprTraits<AT>::scalar_type,
        scalar_max_op<typename ExprTraits<AT>::scalar_type, typename ExprTraits<AT>::scalar_type>,
        ADVar<typename ExprTraits<AT>::scalar_type, M>>>::type(max)(T a3, const AT& b3)
{
    using Scalar = typename ExprTraits<AT>::scalar_type;
    return UnaryExpr<Scalar, scalar_max_op<Scalar, Scalar>, ADVar<Scalar, M>>(
        ADVar<Scalar, M>(b3), scalar_max_op<Scalar, Scalar>(a3));
}


template <class Scalar, class Expr, class DerivativeType>
FEXPR_INLINE UnaryExpr<Scalar, scalar_max_op<Scalar, Scalar>, Expr, DerivativeType>(max)(
    const Expression<Scalar, Expr, DerivativeType>& a2, Scalar b2)
{
    return UnaryExpr<Scalar, scalar_max_op<Scalar, Scalar>, Expr, DerivativeType>(
        a2.derived(), scalar_max_op<Scalar, Scalar>(b2));
}

template <class Scalar, class Expr, class T, class DerivativeType>
FEXPR_INLINE typename std::enable_if<
    std::is_same<T, typename ExprTraits<Expr>::nested_type>::value,
    UnaryExpr<Scalar, scalar_max_op<Scalar, Scalar>, Expr,
              DerivativeType>>::type(max)(const Expression<Scalar, Expr, DerivativeType>& a2, T b2)
{
    return UnaryExpr<Scalar, scalar_max_op<Scalar, Scalar>, Expr, DerivativeType>(
        a2.derived(), scalar_max_op<Scalar, Scalar>(b2));
}

template <class T, class AT, std::size_t N>
FEXPR_INLINE typename std::enable_if<
    std::is_same<T, typename ExprTraits<AT>::nested_type>::value &&
        std::is_same<AReal<typename ExprTraits<AT>::scalar_type, N>, AT>::value,
    UnaryExpr<
        typename ExprTraits<AT>::scalar_type,
        scalar_max_op<typename ExprTraits<AT>::scalar_type, typename ExprTraits<AT>::scalar_type>,
        ADVar<typename ExprTraits<AT>::scalar_type, N>>>::type(max)(const AT& a1, T b1)
{
    return max(b1, a1);
}



template <class Scalar, class Expr, class DerivativeType>
FEXPR_INLINE UnaryExpr<Scalar, scalar_min_op<Scalar, Scalar>, Expr, DerivativeType>(min)(
    Scalar a2, const Expression<Scalar, Expr, DerivativeType>& b2)
{
    return UnaryExpr<Scalar, scalar_min_op<Scalar, Scalar>, Expr, DerivativeType>(
        b2.derived(), scalar_min_op<Scalar, Scalar>(a2));
}

template <class Scalar, class Expr, class T, class DerivativeType>
FEXPR_INLINE typename std::enable_if<
    std::is_same<T, typename ExprTraits<Expr>::nested_type>::value,
    UnaryExpr<Scalar, scalar_min_op<Scalar, Scalar>, Expr,
              DerivativeType>>::type(min)(T a2, const Expression<Scalar, Expr, DerivativeType>& b2)
{
    return UnaryExpr<Scalar, scalar_min_op<Scalar, Scalar>, Expr, DerivativeType>(
        b2.derived(), scalar_min_op<Scalar, Scalar>(a2));
}

template <class T, class AT, std::size_t M = 1>
FEXPR_INLINE typename std::enable_if<
    std::is_same<T, typename ExprTraits<AT>::nested_type>::value &&
        std::is_same<AReal<typename ExprTraits<AT>::scalar_type, M>, AT>::value,
    UnaryExpr<
        typename ExprTraits<AT>::scalar_type,
        scalar_min_op<typename ExprTraits<AT>::scalar_type, typename ExprTraits<AT>::scalar_type>,
        ADVar<typename ExprTraits<AT>::scalar_type, M>>>::type(min)(T a3, const AT& b3)
{
    using Scalar = typename ExprTraits<AT>::scalar_type;
    return UnaryExpr<Scalar, scalar_min_op<Scalar, Scalar>, ADVar<Scalar, M>>(
        ADVar<Scalar, M>(b3), scalar_min_op<Scalar, Scalar>(a3));
}


template <class Scalar, class Expr, class DerivativeType>
FEXPR_INLINE UnaryExpr<Scalar, scalar_min_op<Scalar, Scalar>, Expr, DerivativeType>(min)(
    const Expression<Scalar, Expr, DerivativeType>& a2, Scalar b2)
{
    return UnaryExpr<Scalar, scalar_min_op<Scalar, Scalar>, Expr, DerivativeType>(
        a2.derived(), scalar_min_op<Scalar, Scalar>(b2));
}

template <class Scalar, class Expr, class T, class DerivativeType>
FEXPR_INLINE typename std::enable_if<
    std::is_same<T, typename ExprTraits<Expr>::nested_type>::value,
    UnaryExpr<Scalar, scalar_min_op<Scalar, Scalar>, Expr,
              DerivativeType>>::type(min)(const Expression<Scalar, Expr, DerivativeType>& a2, T b2)
{
    return UnaryExpr<Scalar, scalar_min_op<Scalar, Scalar>, Expr, DerivativeType>(
        a2.derived(), scalar_min_op<Scalar, Scalar>(b2));
}

template <class T, class AT>
FEXPR_INLINE typename std::enable_if<
    std::is_same<T, typename ExprTraits<AT>::nested_type>::value &&
        std::is_same<AReal<typename ExprTraits<AT>::scalar_type>, AT>::value,
    UnaryExpr<
        typename ExprTraits<AT>::scalar_type,
        scalar_min_op<typename ExprTraits<AT>::scalar_type, typename ExprTraits<AT>::scalar_type>,
        ADVar<typename ExprTraits<AT>::scalar_type>, typename AT::derivatives_type>>::type(min)(const AT& a1, T b1)
{
    return min(b1, a1);
}

FEXPR_UNARY_BINSCAL(fmax, scalar_fmax_op)
FEXPR_UNARY_BINSCAL(fmin, scalar_fmin_op)

/////////// Math functions

FEXPR_MAKE_UNARY_FUNC(degrees)
FEXPR_MAKE_UNARY_FUNC(radians)
FEXPR_MAKE_UNARY_FUNC(cos)
FEXPR_MAKE_UNARY_FUNC(sin)
FEXPR_MAKE_UNARY_FUNC(log)
FEXPR_MAKE_UNARY_FUNC(log10)
FEXPR_MAKE_UNARY_FUNC(log2)
FEXPR_MAKE_UNARY_FUNC(asin)
FEXPR_MAKE_UNARY_FUNC(acos)
FEXPR_MAKE_UNARY_FUNC(atan)
FEXPR_MAKE_UNARY_FUNC(sinh)
FEXPR_MAKE_UNARY_FUNC(cosh)
FEXPR_MAKE_UNARY_FUNC(expm1)
FEXPR_MAKE_UNARY_FUNC(exp2)
FEXPR_MAKE_UNARY_FUNC(log1p)
FEXPR_MAKE_UNARY_FUNC(asinh)
FEXPR_MAKE_UNARY_FUNC(acosh)
FEXPR_MAKE_UNARY_FUNC(atanh)
FEXPR_MAKE_UNARY_FUNC(abs)
FEXPR_MAKE_UNARY_FUNC(fabs)
FEXPR_MAKE_UNARY_FUNC(floor)
FEXPR_MAKE_UNARY_FUNC(ceil)
FEXPR_MAKE_UNARY_FUNC(trunc)
FEXPR_MAKE_UNARY_FUNC(round)
FEXPR_MAKE_UNARY_FUNC(exp)
FEXPR_MAKE_UNARY_FUNC(tanh)
FEXPR_MAKE_UNARY_FUNC(sqrt)
FEXPR_MAKE_UNARY_FUNC(cbrt)
FEXPR_MAKE_UNARY_FUNC(tan)
FEXPR_MAKE_UNARY_FUNC(erf)
FEXPR_MAKE_UNARY_FUNC(erfc)

// no special AD treatement here, but we need the overloads

FEXPR_MAKE_FPCLASSIFY_FUNC(isinf, using std::isinf)
FEXPR_MAKE_FPCLASSIFY_FUNC(isnan, using std::isnan)
FEXPR_MAKE_FPCLASSIFY_FUNC(isfinite, using std::isfinite)
FEXPR_MAKE_FPCLASSIFY_FUNC(signbit, using std::signbit)
FEXPR_MAKE_FPCLASSIFY_FUNC(isnormal, using std::isnormal)
FEXPR_MAKE_FPCLASSIFY_FUNC(__isinf, )
FEXPR_MAKE_FPCLASSIFY_FUNC(__isnan, )
FEXPR_MAKE_FPCLASSIFY_FUNC(__isfinite, )
FEXPR_MAKE_FPCLASSIFY_FUNC_RET(int, fpclassify, using std::fpclassify)
FEXPR_MAKE_FPCLASSIFY_FUNC_RET(long, lround, using std::lround)
FEXPR_MAKE_FPCLASSIFY_FUNC_RET(long long, llround, using std::llround)

FEXPR_UNARY_BINSCAL1(remainder, scalar_remainder1_op)
FEXPR_UNARY_BINSCAL2(remainder, scalar_remainder2_op)

template <class Scalar, class Expr, class T2, class DerivativeType>
FEXPR_INLINE typename std::enable_if<
    std::is_arithmetic<T2>::value && std::is_fundamental<T2>::value &&
        !std::is_same<T2, typename ExprTraits<Expr>::nested_type>::value,
    UnaryExpr<Scalar, scalar_remquo1_op<Scalar, T2>, Expr, DerivativeType>>::type
remquo(const T2& a, const Expression<Scalar, Expr, DerivativeType>& b, int* quo)
{
    return UnaryExpr<Scalar, scalar_remquo1_op<Scalar, T2>, Expr, DerivativeType>(
        b.derived(), scalar_remquo1_op<Scalar, T2>(a, quo));
}
template <class Scalar, class Expr, class DerivativeType>
FEXPR_INLINE UnaryExpr<Scalar, scalar_remquo1_op<Scalar, typename ExprTraits<Expr>::nested_type>,
                     Expr, DerivativeType>
remquo(typename ExprTraits<Expr>::nested_type a, const Expression<Scalar, Expr, DerivativeType>& b,
       int* quo)
{
    return UnaryExpr<Scalar, scalar_remquo1_op<Scalar, typename ExprTraits<Expr>::nested_type>,
                     Expr, DerivativeType>(
        b.derived(), scalar_remquo1_op<Scalar, typename ExprTraits<Expr>::nested_type>(a, quo));
}
template <class Scalar, class T2, std::size_t M = 1>
FEXPR_INLINE
    typename std::enable_if<std::is_arithmetic<T2>::value && std::is_fundamental<T2>::value &&
                                !std::is_same<T2, typename ExprTraits<Scalar>::nested_type>::value,
                            UnaryExpr<Scalar, scalar_remquo1_op<Scalar, T2>, ADVar<Scalar, M>,
                                      typename DerivativesTraits<Scalar, M>::type>>::type
    remquo(const T2& a, const AReal<Scalar, M>& b, int* quo)
{
    return UnaryExpr<Scalar, scalar_remquo1_op<Scalar, T2>, ADVar<Scalar, M>,
                     typename DerivativesTraits<Scalar, M>::type>(
        ADVar<Scalar, M>(b), scalar_remquo1_op<Scalar, T2>(a, quo));
}
template <class Scalar, std::size_t M = 1>
FEXPR_INLINE UnaryExpr<Scalar, scalar_remquo1_op<Scalar, typename ExprTraits<Scalar>::nested_type>,
                     AReal<Scalar, M>, typename DerivativesTraits<Scalar, M>::type>
remquo(typename ExprTraits<Scalar>::nested_type a, const AReal<Scalar, M>& b, int* quo)
{
    return UnaryExpr<Scalar, scalar_remquo1_op<Scalar, typename ExprTraits<Scalar>::nested_type>,
                     AReal<Scalar, M>, typename DerivativesTraits<Scalar, M>::type>(
        AReal<Scalar, M>(b),
        scalar_remquo1_op<Scalar, typename ExprTraits<Scalar>::nested_type>(a, quo));
}

template <class Scalar, class Expr, class T2, class DerivativeType>
FEXPR_INLINE typename std::enable_if<
    std::is_arithmetic<T2>::value && std::is_fundamental<T2>::value &&
        !std::is_same<T2, typename ExprTraits<Expr>::nested_type>::value,
    UnaryExpr<Scalar, scalar_remquo2_op<Scalar, T2>, Expr, DerivativeType>>::type
remquo(const Expression<Scalar, Expr, DerivativeType>& a, const T2& b, int* quo)
{
    return UnaryExpr<Scalar, scalar_remquo2_op<Scalar, T2>, Expr, DerivativeType>(
        a.derived(), scalar_remquo2_op<Scalar, T2>(b, quo));
}
template <class Scalar, class Expr, class DerivativeType>
FEXPR_INLINE UnaryExpr<Scalar, scalar_remquo2_op<Scalar, typename ExprTraits<Expr>::nested_type>,
                     Expr, DerivativeType>
remquo(const Expression<Scalar, Expr, DerivativeType>& a, typename ExprTraits<Expr>::nested_type b,
       int* quo)
{
    return UnaryExpr<Scalar, scalar_remquo2_op<Scalar, typename ExprTraits<Expr>::nested_type>,
                     Expr, DerivativeType>(
        a.derived(), scalar_remquo2_op<Scalar, typename ExprTraits<Expr>::nested_type>(b, quo));
}
template <class Scalar, class T2, std::size_t M = 1>
FEXPR_INLINE
    typename std::enable_if<std::is_arithmetic<T2>::value && std::is_fundamental<T2>::value &&
                                !std::is_same<T2, typename ExprTraits<Scalar>::nested_type>::value,
                            UnaryExpr<Scalar, scalar_remquo2_op<Scalar, T2>, ADVar<Scalar, M>,
                                      typename DerivativesTraits<Scalar, M>::type>>::type
    remquo(const AReal<Scalar, M>& a, const T2& b, int* quo)
{
    return UnaryExpr<Scalar, scalar_remquo2_op<Scalar, T2>, ADVar<Scalar, M>,
                     typename DerivativesTraits<Scalar, M>::type>(
        ADVar<Scalar, M>(a), scalar_remquo2_op<Scalar, T2>(b, quo));
}

template <class Scalar, std::size_t M = 1>
FEXPR_INLINE UnaryExpr<Scalar, scalar_remquo2_op<Scalar, typename ExprTraits<Scalar>::nested_type>,
                     ADVar<Scalar, M>, typename DerivativesTraits<Scalar, M>::type>
remquo(const AReal<Scalar, M>& a, typename ExprTraits<Scalar>::nested_type b, int* quo)
{
    return UnaryExpr<Scalar, scalar_remquo2_op<Scalar, typename ExprTraits<Scalar>::nested_type>,
                     ADVar<Scalar, M>, typename DerivativesTraits<Scalar, M>::type>(
        ADVar<Scalar, M>(a),
        scalar_remquo2_op<Scalar, typename ExprTraits<Scalar>::nested_type>(b, quo));
}

template <class Scalar, class Derived, class Deriv>
FEXPR_INLINE int ilogb(const Expression<Scalar, Derived, Deriv>& x)
{
    using std::ilogb;
    return ilogb(x.value());
}

template <class Scalar, class Derived, class Deriv>
FEXPR_INLINE typename ExprTraits<Derived>::value_type scalbn(
    const Expression<Scalar, Derived, Deriv>& x, int exp)
{
    using std::scalbn;
    using T = typename ExprTraits<Derived>::value_type;
    return T(x) * scalbn(1.0, exp);
}

template <class Scalar, class Derived, class T2, class DerivativeType>
FEXPR_INLINE typename ExprTraits<Derived>::value_type copysign(
    const Expression<Scalar, Derived, DerivativeType>& x, const T2& y)
{
    using T = typename ExprTraits<Derived>::value_type;
    bool sign = signbit(y);
    if (x < 0)
    {
        if (sign)
            return T(x);
        else
            return T(-x);
    }
    else
    {
        if (sign)
            return T(-x);
        else
            return T(x);
    }
}

template <class Scalar, std::size_t N>
FEXPR_INLINE Scalar copysign(double x, const FRealDirect<Scalar, N>& y)
{
    using std::copysign;
    return copysign(x, value(y));
}

template <class Scalar, std::size_t N>
FEXPR_INLINE Scalar copysign(float x, const FRealDirect<Scalar, N>& y)
{
    using std::copysign;
    return copysign(x, value(y));
}

template <class Scalar, std::size_t N>
FEXPR_INLINE Scalar copysign(double x, const ARealDirect<Scalar, N>& y)
{
    using std::copysign;
    return copysign(x, value(y));
}

template <class Scalar, std::size_t N>
FEXPR_INLINE Scalar copysign(float x, const ARealDirect<Scalar, N>& y)
{
    using std::copysign;
    return copysign(x, value(y));
}

template <class Scalar, class Derived, class DerivativeType>
FEXPR_INLINE double copysign(double x, const Expression<Scalar, Derived, DerivativeType>& y)
{
    using std::copysign;
    return copysign(x, value(y));
}

template <class Scalar, class Derived, class DerivativeType>
FEXPR_INLINE float copysign(float x, const Expression<Scalar, Derived, DerivativeType>& y)
{
    using std::copysign;
    return copysign(x, value(y));
}

#undef FEXPR_UNARY_BINSCAL
#undef FEXPR_UNARY_BINSCAL1
#undef FEXPR_UNARY_BINSCAL2
#undef FEXPR_MAKE_UNARY_FUNC
#undef FEXPR_MAKE_FPCLASSIFY_FUNC
#undef FEXPR_MAKE_FPCLASSIFY_FUNC_RET

}}  // namespace forge::expr

