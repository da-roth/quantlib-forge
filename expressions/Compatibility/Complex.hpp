/*******************************************************************************

   An AD-enabled equivalent of std::complex.

   This file is part of QuantLib-Forge, a Forge integration layer for QuantLib.

   Copyright (C) 2025 The QuantLib-Forge Authors

   Derived from XAD (https://github.com/auto-differentiation/XAD)
   Original code: Copyright (C) 2010-2024 Xcelerit Computing Ltd.

   SPDX-License-Identifier: AGPL-3.0-or-later

******************************************************************************/

#pragma once
#include <expressions/ExpressionTemplates/BinaryOperators.hpp>
#include <expressions/Expression.hpp>
#include <expressions/Literals.hpp>
#include <expressions/Traits.hpp>
#include <expressions/ExpressionTemplates/UnaryOperators.hpp>
#include <cmath>
#include <complex>
#include <iostream>
#include <limits>
#include <type_traits>

// Note: The seemingly high number of overloads in this file, containing partially
// duplicated code, is necessary to make sure all supported compilers pick the overloads
// from here instead of the the std::complex versions provided by the standard library.

namespace forge { namespace expr {

namespace detail
{
template <class T>
class complex_impl
{
  public:
    typedef T value_type;

    explicit complex_impl(const T& areal = T(), const T& aimag = T()) : real_(areal), imag_(aimag)
    {
    }

    template <class X>
    explicit complex_impl(const std::complex<X>& o) : real_(o.real()), imag_(o.imag())
    {
    }

    template <class X>
    FEXPR_INLINE complex_impl& operator=(const X& other)
    {
        real_ = other;
        imag_ = T();
        return *this;
    }

    template <class X>
    FEXPR_INLINE complex_impl& operator=(const std::complex<X>& other)
    {
        real_ = other.real();
        imag_ = other.imag();
        return *this;
    }

    FEXPR_INLINE T& real() { return real_; }
    FEXPR_INLINE T& imag() { return imag_; }
    FEXPR_INLINE const T& real() const { return real_; }
    FEXPR_INLINE const T& imag() const { return imag_; }

    FEXPR_INLINE void real(const T& value) { real_ = value; }
    FEXPR_INLINE void imag(const T& value) { imag_ = value; }

    void setDerivative(
        typename ExprTraits<T>::nested_type rd,
        typename ExprTraits<T>::nested_type id = typename ExprTraits<T>::nested_type())
    {
        derivative(real_) = rd;
        derivative(imag_) = id;
    }

    FEXPR_INLINE void setAdjoint(
        typename ExprTraits<T>::nested_type rd,
        typename ExprTraits<T>::nested_type id = typename ExprTraits<T>::nested_type())
    {
        this->setDerivative(rd, id);
    }

    FEXPR_INLINE std::complex<typename ExprTraits<T>::nested_type> getDerivative() const
    {
        return std::complex<typename ExprTraits<T>::nested_type>(derivative(real_),
                                                                 derivative(imag_));
    }

    FEXPR_INLINE std::complex<typename ExprTraits<T>::nested_type> getAdjoint() const
    {
        return this->getDerivative();
    }

  private:
    T real_, imag_;
};

}  // namespace detail

}}  // namespace forge::expr

namespace std
{

template <class Scalar, class T, class Deriv>
class complex<forge::expr::ADTypeBase<Scalar, T, Deriv>> : public forge::expr::detail::complex_impl<T>
{
  public:
    typedef forge::expr::detail::complex_impl<T> base;
    typedef T value_type;

    complex(const T& areal = T(), const T& aimag = T()) : base(areal, aimag) {}

    FEXPR_INLINE complex<T>& derived() { return static_cast<complex<T>&>(*this); }

    FEXPR_INLINE complex<T>& operator+=(const T& other)
    {
        base::real() += other;
        return derived();
    }

    template <class X>
    FEXPR_INLINE complex<T>& operator+=(const std::complex<X>& other)
    {
        base::real() += other.real();
        base::imag() += other.imag();
        return derived();
    }

    FEXPR_INLINE complex<T>& operator-=(const T& other)
    {
        base::real() -= other;
        return derived();
    }

    template <class X>
    FEXPR_INLINE complex<T>& operator-=(const std::complex<X>& other)
    {
        base::real() -= other.real();
        base::imag() -= other.imag();
        return derived();
    }

    FEXPR_INLINE complex<T>& operator*=(const T& other)
    {
        base::real() *= other;
        base::imag() *= other;
        return derived();
    }

    template <class X>
    complex<T>& operator*=(const std::complex<X>& other)
    {
        T real_t = base::real() * other.real() - base::imag() * other.imag();
        base::imag(base::real() * other.imag() + other.real() * base::imag());
        base::real(real_t);
        return derived();
    }

    FEXPR_INLINE complex<T>& operator/=(const T& other)
    {
        base::real() /= other;
        base::imag() /= other;
        return derived();
    }

    template <class X>
    complex<T>& operator/=(const std::complex<X>& other)
    {
        T den = T(other.real()) * T(other.real()) + T(other.imag()) * T(other.imag());
        T real_t = ((base::real() * T(other.real())) + (base::imag() * T(other.imag()))) / den;
        base::imag((base::imag() * T(other.real()) - base::real() * T(other.imag())) / den);
        base::real(real_t);
        return derived();
    }
};

template <class T, std::size_t N>
class complex<forge::expr::AReal<T, N>>
    : public complex<
          forge::expr::ADTypeBase<T, forge::expr::AReal<T, N>, typename forge::expr::DerivativesTraits<T, N>::type>>
{
  public:
    typedef complex<
        forge::expr::ADTypeBase<T, forge::expr::AReal<T, N>, typename forge::expr::DerivativesTraits<T, N>::type>>
        base;

    // inheriting template constructors doesn't work in all compilers

    FEXPR_INLINE complex(const forge::expr::AReal<T, N>& areal = forge::expr::AReal<T, N>(),
                       const forge::expr::AReal<T, N>& aimag = forge::expr::AReal<T, N>())
        : base(areal, aimag)
    {
    }

    template <class X>
    FEXPR_INLINE complex(const X& areal,
                       typename std::enable_if<!forge::expr::ExprTraits<X>::isExpr>::type* = nullptr)
        : base(forge::expr::AReal<T, N>(areal), forge::expr::AReal<T, N>())
    {
    }

    template <class X>
    FEXPR_INLINE complex(  // cppcheck-suppress noExplicitConstructor
        const X& areal,
        typename std::enable_if<forge::expr::ExprTraits<X>::isExpr &&
                                forge::expr::ExprTraits<X>::direction == forge::expr::DIR_REVERSE>::type* = nullptr)
        : base(forge::expr::AReal<T, N>(areal), forge::expr::AReal<T, N>())
    {
    }

    template <class X>
    FEXPR_INLINE complex(const complex<X>& o)
        : base(forge::expr::AReal<T, N>(o.real()), forge::expr::AReal<T, N>(o.imag()))
    {
    }

    using base::operator+=;
    using base::operator-=;
    using base::operator*=;
    using base::operator/=;
};

// FReal complex specialization removed - not used by QuantLib-Risks-Cpp

}  // namespace std

namespace forge { namespace expr {

// read access to value and derivatives, only for scalars
template <class T, std::size_t N>
FEXPR_INLINE std::complex<T> derivative(const std::complex<AReal<T, N>>& z)
{
    static_assert(N == 1,
                  "Global derivative function is only defined for scalar derivatives - use "
                  "derivative(z.real()) instead");
    return std::complex<T>(derivative(z.real()), derivative(z.imag()));
}

// FReal complex derivative function removed - not used by QuantLib-Risks-Cpp

template <class T, std::size_t N>
FEXPR_INLINE std::complex<T> value(const std::complex<AReal<T, N>>& z)
{
    return std::complex<T>(value(z.real()), value(z.imag()));
}

// FReal complex value function removed - not used by QuantLib-Risks-Cpp

template <class T>
FEXPR_INLINE std::complex<T> value(const std::complex<T>& z)
{
    return z;
}

namespace detail
{

// declare first, implementation at the bottom of this file,
// to avoid issues with order of declaration (functions called in bodies
// not defined yet)

template <class T>
FEXPR_INLINE T abs_impl(const std::complex<T>& x);

template <class T>
FEXPR_INLINE std::complex<T> exp_impl(const std::complex<T>& z);

template <class T1, class T2>
FEXPR_INLINE std::complex<typename forge::expr::ExprTraits<T1>::value_type> polar_impl(const T1& r,
                                                                             const T2& theta);

template <class T>
FEXPR_INLINE std::complex<T> sqrt_impl(const std::complex<T>& z);

template <class T>
FEXPR_INLINE std::complex<T> sinh_impl(const std::complex<T>& z);

template <class T>
FEXPR_INLINE std::complex<T> cosh_impl(const std::complex<T>& z);

template <class T>
FEXPR_INLINE std::complex<T> tanh_impl(const std::complex<T>& z);

template <class T>
FEXPR_INLINE std::complex<T> asinh_impl(const std::complex<T>& z);

template <class T>
FEXPR_INLINE std::complex<T> acosh_impl(const std::complex<T>& z);

template <class T>
FEXPR_INLINE std::complex<T> atanh_impl(const std::complex<T>& z);

template <class T>
FEXPR_INLINE std::complex<T> sin_impl(const std::complex<T>& z);

template <class T>
FEXPR_INLINE std::complex<T> cos_impl(const std::complex<T>& z);

template <class T>
FEXPR_INLINE std::complex<T> tan_impl(const std::complex<T>& z);

template <class T>
FEXPR_INLINE std::complex<T> asin_impl(const std::complex<T>& z);

template <class T>
FEXPR_INLINE std::complex<T> acos_impl(const std::complex<T>& z);

template <class T>
FEXPR_INLINE std::complex<T> atan_impl(const std::complex<T>& z);

// note that this captures the AReal and FReal base types too
template <class Scalar, class Derived, class Deriv>
FEXPR_INLINE typename forge::expr::ExprTraits<Derived>::value_type arg_impl(
    const forge::expr::Expression<Scalar, Derived, Deriv>& x);

template <class T>
FEXPR_INLINE T arg_impl(const std::complex<T>& z);

#if (defined(_MSC_VER) && (_MSC_VER < 1920) || (defined(__GNUC__) && __GNUC__ < 7)) &&             \
    !defined(__clang__)
template <class Scalar, class Derived, class Deriv>
FEXPR_INLINE typename forge::expr::ExprTraits<Derived>::value_type proj_impl(
    const forge::expr::Expression<Scalar, Derived, Deriv>& x);
#else
template <class Scalar, class Derived, class Deriv>
FEXPR_INLINE std::complex<typename forge::expr::ExprTraits<Derived>::value_type> proj_impl(
    const forge::expr::Expression<Scalar, Derived, Deriv>& x);
#endif

template <class T>
FEXPR_INLINE T norm_impl(const std::complex<T>& x);

}  // namespace detail

}}  // namespace forge::expr

namespace std
{

// access to real / imag
template <class Scalar, class T, class Deriv>
FEXPR_INLINE T real(const std::complex<forge::expr::ADTypeBase<Scalar, T, Deriv>>& z)
{
    return z.derived().real();
}

template <class Scalar, class T, class Deriv>
FEXPR_INLINE T& real(std::complex<forge::expr::ADTypeBase<Scalar, T, Deriv>>& z)
{
    return z.derived().real();
}

template <class Scalar, class Expr>
FEXPR_INLINE typename forge::expr::ExprTraits<Expr>::value_type real(
    const forge::expr::Expression<Scalar, Expr>& other)
{
    return other.derived();
}

template <class Scalar, class T, class Deriv>
FEXPR_INLINE T imag(const std::complex<forge::expr::ADTypeBase<Scalar, T, Deriv>>& z)
{
    return z.derived().imag();
}

template <class Scalar, class T, class Deriv>
FEXPR_INLINE T& imag(std::complex<forge::expr::ADTypeBase<Scalar, T, Deriv>>& z)
{
    return z.derived().imag();
}

template <class Scalar, class Expr, class Deriv>
FEXPR_INLINE typename forge::expr::ExprTraits<Expr>::value_type imag(
    const forge::expr::Expression<Scalar, Expr, Deriv>&)
{
    return typename forge::expr::ExprTraits<Expr>::value_type(0);
}

///////////////////////// operators
template <class T, std::size_t N>
FEXPR_INLINE const std::complex<forge::expr::AReal<T, N>>& operator+(const std::complex<forge::expr::AReal<T, N>>& x)
{
    return x;
}

// FReal unary operator+ removed - not used by QuantLib-Risks-Cpp

template <class T, std::size_t N>
FEXPR_INLINE std::complex<forge::expr::AReal<T, N>> operator-(const std::complex<forge::expr::AReal<T, N>>& x)
{
    return std::complex<forge::expr::AReal<T, N>>(-x.real(), -x.imag());
}

// FReal unary operator- removed - not used by QuantLib-Risks-Cpp

// operator== - lots of variants here, I'm sure this could be done cleaner...

template <class T, std::size_t N>
FEXPR_INLINE bool operator==(const std::complex<forge::expr::AReal<T, N>>& lhs,
                           const std::complex<forge::expr::AReal<T, N>>& rhs)
{
    return (lhs.real() == rhs.real()) && (lhs.imag() == rhs.imag());
}

// FReal operator== (FReal, FReal) removed - not used by QuantLib-Risks-Cpp

template <class T, class Expr, std::size_t N, class Deriv>
FEXPR_INLINE bool operator==(const std::complex<forge::expr::AReal<T, N>>& lhs,
                           const forge::expr::Expression<T, Expr, Deriv>& rhs)
{
    return (lhs.real() == rhs) && (lhs.imag() == T());
}

// FReal operator== (FReal, Expression) removed - not used by QuantLib-Risks-Cpp

template <class T, std::size_t N>
FEXPR_INLINE bool operator==(const std::complex<forge::expr::AReal<T, N>>& lhs, const T& rhs)
{
    return (lhs.real() == rhs) && (lhs.imag() == T());
}

// FReal operator== (FReal, T) removed - not used by QuantLib-Risks-Cpp

template <class T, std::size_t N, class Expr, class Deriv>
FEXPR_INLINE bool operator==(const forge::expr::Expression<T, Expr, Deriv>& rhs,
                           const std::complex<forge::expr::AReal<T, N>>& lhs)
{
    return (lhs.real() == rhs) && (lhs.imag() == T());
}

// FReal operator== (Expression, FReal) removed - not used by QuantLib-Risks-Cpp

template <class T, std::size_t N>
FEXPR_INLINE bool operator==(const T& rhs, const std::complex<forge::expr::AReal<T, N>>& lhs)
{
    return (lhs.real() == rhs) && (lhs.imag() == T());
}

// FReal operator== (T, FReal) removed - not used by QuantLib-Risks-Cpp

// operator !=

template <class T, std::size_t N>
FEXPR_INLINE bool operator!=(const std::complex<forge::expr::AReal<T, N>>& lhs,
                           const std::complex<forge::expr::AReal<T, N>>& rhs)
{
    return !(lhs == rhs);
}

// FReal operator!= (FReal, FReal) removed - not used by QuantLib-Risks-Cpp

template <class T, std::size_t N, class Expr, class Deriv>
FEXPR_INLINE bool operator!=(const std::complex<forge::expr::AReal<T, N>>& lhs,
                           const forge::expr::Expression<T, Expr, Deriv>& rhs)
{
    return !(lhs == rhs);
}

// FReal operator!= (FReal, Expression) removed - not used by QuantLib-Risks-Cpp

template <class T, std::size_t N>
FEXPR_INLINE bool operator!=(const std::complex<forge::expr::AReal<T, N>>& lhs, const T& rhs)
{
    return !(lhs == rhs);
}

// FReal operator!= (FReal, T) removed - not used by QuantLib-Risks-Cpp

template <class T, std::size_t N, class Expr, class Deriv>
FEXPR_INLINE bool operator!=(const forge::expr::Expression<T, Expr, Deriv>& rhs,
                           const std::complex<forge::expr::AReal<T, N>>& lhs)
{
    return !(lhs == rhs);
}

// FReal operator!= (Expression, FReal) removed - not used by QuantLib-Risks-Cpp

template <class T, std::size_t N>
FEXPR_INLINE bool operator!=(const T& rhs, const std::complex<forge::expr::AReal<T, N>>& lhs)
{
    return !(lhs == rhs);
}

// FReal operator!= (T, FReal) removed - not used by QuantLib-Risks-Cpp

// operator+

template <class T, std::size_t N>
FEXPR_INLINE std::complex<forge::expr::AReal<T, N>> operator+(std::complex<forge::expr::AReal<T, N>> lhs,
                                                    const std::complex<forge::expr::AReal<T, N>>& rhs)
{
    lhs += rhs;
    return lhs;
}

template <class T, std::size_t N>
FEXPR_INLINE std::complex<forge::expr::AReal<T, N>> operator+(std::complex<forge::expr::AReal<T, N>> lhs,
                                                    const std::complex<T>& rhs)
{
    lhs += rhs;
    return lhs;
}

template <class T, std::size_t N>
FEXPR_INLINE std::complex<forge::expr::AReal<T, N>> operator+(std::complex<forge::expr::AReal<T, N>> lhs,
                                                    const forge::expr::AReal<T, N>& rhs)
{
    lhs += rhs;
    return lhs;
}

template <class T, std::size_t N>
FEXPR_INLINE std::complex<forge::expr::AReal<T, N>> operator+(std::complex<T> lhs,
                                                    const forge::expr::AReal<T, N>& rhs)
{
    std::complex<forge::expr::AReal<T, N>> z = lhs;
    z += rhs;
    return z;
}

template <class T, std::size_t N>
FEXPR_INLINE std::complex<forge::expr::AReal<T, N>> operator+(std::complex<forge::expr::AReal<T, N>> lhs,
                                                    const T& rhs)
{
    lhs += rhs;
    return lhs;
}

template <class T, std::size_t N, class Expr, class Deriv>
FEXPR_INLINE std::complex<forge::expr::AReal<T, N>> operator+(std::complex<forge::expr::AReal<T, N>> lhs,
                                                    const forge::expr::Expression<T, Expr, Deriv>& rhs)
{
    lhs += rhs;
    return lhs;
}

template <class T, class Deriv, class Expr>
FEXPR_INLINE std::complex<typename forge::expr::ExprTraits<Expr>::value_type> operator+(
    const std::complex<T>& lhs, const forge::expr::Expression<T, Expr, Deriv>& rhs)
{
    std::complex<typename forge::expr::ExprTraits<Expr>::value_type> z = lhs;
    z += rhs;
    return z;
}

template <class T, std::size_t N>
FEXPR_INLINE std::complex<forge::expr::AReal<T, N>> operator+(const std::complex<T>& rhs,
                                                    std::complex<forge::expr::AReal<T, N>> lhs)
{
    lhs += rhs;
    return lhs;
}

template <class T, std::size_t N>
FEXPR_INLINE std::complex<forge::expr::AReal<T, N>> operator+(const forge::expr::AReal<T, N>& rhs,
                                                    std::complex<forge::expr::AReal<T, N>> lhs)
{
    lhs += rhs;
    return lhs;
}

template <class T, std::size_t N>
FEXPR_INLINE std::complex<forge::expr::AReal<T, N>> operator+(const T& rhs,
                                                    std::complex<forge::expr::AReal<T, N>> lhs)
{
    lhs += rhs;
    return lhs;
}

template <class T, std::size_t N, class Expr, class Deriv>
FEXPR_INLINE std::complex<forge::expr::AReal<T, N>> operator+(const forge::expr::Expression<T, Expr, Deriv>& rhs,
                                                    std::complex<forge::expr::AReal<T, N>> lhs)
{
    lhs += rhs;
    return lhs;
}

template <class T, class Expr, class Deriv>
FEXPR_INLINE std::complex<typename forge::expr::ExprTraits<Expr>::value_type> operator+(
    const forge::expr::Expression<T, Expr, Deriv>& rhs, const std::complex<T>& lhs)
{
    std::complex<typename forge::expr::ExprTraits<Expr>::value_type> z = lhs;
    z += rhs;
    return z;
}

// All FReal operator+ overloads removed - not used by QuantLib-Risks-Cpp

// operator-

template <class T, std::size_t N>
FEXPR_INLINE std::complex<forge::expr::AReal<T, N>> operator-(std::complex<forge::expr::AReal<T, N>> lhs,
                                                    const std::complex<forge::expr::AReal<T, N>>& rhs)
{
    lhs -= rhs;
    return lhs;
}

template <class T, std::size_t N>
FEXPR_INLINE std::complex<forge::expr::AReal<T, N>> operator-(std::complex<forge::expr::AReal<T, N>> lhs,
                                                    const std::complex<T>& rhs)
{
    lhs -= rhs;
    return lhs;
}

template <class T, std::size_t N>
FEXPR_INLINE std::complex<forge::expr::AReal<T, N>> operator-(const std::complex<T>& lhs,
                                                    std::complex<forge::expr::AReal<T, N>> rhs)
{
    std::complex<forge::expr::AReal<T, N>> z = lhs;
    z -= rhs;
    return z;
}

template <class T, std::size_t N>
FEXPR_INLINE std::complex<forge::expr::AReal<T, N>> operator-(std::complex<forge::expr::AReal<T, N>> lhs,
                                                    const forge::expr::AReal<T, N>& rhs)
{
    lhs -= rhs;
    return lhs;
}

template <class T, std::size_t N>
FEXPR_INLINE std::complex<forge::expr::AReal<T, N>> operator-(std::complex<T> lhs,
                                                    const forge::expr::AReal<T, N>& rhs)
{
    std::complex<forge::expr::AReal<T, N>> z = lhs;
    z -= rhs;
    return z;
}

template <class T, std::size_t N>
FEXPR_INLINE std::complex<forge::expr::AReal<T, N>> operator-(std::complex<forge::expr::AReal<T, N>> lhs,
                                                    const T& rhs)
{
    lhs -= rhs;
    return lhs;
}

template <class T, std::size_t N, class Expr, class Deriv>
FEXPR_INLINE std::complex<forge::expr::AReal<T, N>> operator-(std::complex<forge::expr::AReal<T, N>> lhs,
                                                    const forge::expr::Expression<T, Expr, Deriv>& rhs)
{
    lhs -= rhs;
    return lhs;
}

template <class T, class Expr, class Deriv>
FEXPR_INLINE std::complex<typename forge::expr::ExprTraits<Expr>::value_type> operator-(
    const std::complex<T>& lhs, const forge::expr::Expression<T, Expr, Deriv>& rhs)
{
    std::complex<typename forge::expr::ExprTraits<Expr>::value_type> z = lhs;
    z -= rhs;
    return z;
}

template <class T, class Expr, class Deriv>
FEXPR_INLINE std::complex<typename forge::expr::ExprTraits<Expr>::value_type> operator-(
    const forge::expr::Expression<T, Expr, Deriv>& lhs, const std::complex<T>& rhs)
{
    std::complex<typename forge::expr::ExprTraits<Expr>::value_type> z = lhs;
    z -= rhs;
    return z;
}

template <class T, std::size_t N>
FEXPR_INLINE std::complex<forge::expr::AReal<T, N>> operator-(const forge::expr::AReal<T, N>& rhs,
                                                    std::complex<forge::expr::AReal<T, N>> lhs)
{
    return std::complex<forge::expr::AReal<T, N>>(rhs - lhs.real(), -lhs.imag());
}

template <class T, std::size_t N>
FEXPR_INLINE std::complex<forge::expr::AReal<T, N>> operator-(const T& rhs,
                                                    std::complex<forge::expr::AReal<T, N>> lhs)
{
    return std::complex<forge::expr::AReal<T, N>>(rhs - lhs.real(), -lhs.imag());
}

template <class T, std::size_t N, class Expr, class Deriv>
FEXPR_INLINE std::complex<forge::expr::AReal<T, N>> operator-(const forge::expr::Expression<T, Expr, Deriv>& rhs,
                                                    std::complex<forge::expr::AReal<T, N>> lhs)
{
    return std::complex<forge::expr::AReal<T, N>>(rhs - lhs.real(), -lhs.imag());
}

// All FReal operator- overloads removed - not used by QuantLib-Risks-Cpp

// operator*
template <class T, std::size_t N>
FEXPR_INLINE std::complex<forge::expr::AReal<T, N>> operator*(std::complex<forge::expr::AReal<T, N>> lhs,
                                                    const std::complex<forge::expr::AReal<T, N>>& rhs)
{
    lhs *= rhs;
    return lhs;
}

template <class T, std::size_t N>
FEXPR_INLINE std::complex<forge::expr::AReal<T, N>> operator*(std::complex<forge::expr::AReal<T, N>> lhs,
                                                    const std::complex<T>& rhs)
{
    lhs *= rhs;
    return lhs;
}

template <class T, std::size_t N>
FEXPR_INLINE std::complex<forge::expr::AReal<T, N>> operator*(const std::complex<T>& lhs,
                                                    const std::complex<forge::expr::AReal<T, N>>& rhs)
{
    return rhs * lhs;
}

template <class T, std::size_t N>
FEXPR_INLINE std::complex<forge::expr::AReal<T, N>> operator*(std::complex<forge::expr::AReal<T, N>> lhs,
                                                    const forge::expr::AReal<T, N>& rhs)
{
    lhs *= rhs;
    return lhs;
}

template <class T, std::size_t N>
FEXPR_INLINE std::complex<forge::expr::AReal<T, N>> operator*(const std::complex<T>& lhs,
                                                    const forge::expr::AReal<T, N>& rhs)
{
    std::complex<forge::expr::AReal<T, N>> z = lhs;
    z *= rhs;
    return z;
}

template <class T, std::size_t N>
FEXPR_INLINE std::complex<forge::expr::AReal<T, N>> operator*(std::complex<forge::expr::AReal<T, N>> lhs,
                                                    const T& rhs)
{
    lhs *= rhs;
    return lhs;
}

template <class T, std::size_t N, class Expr, class Deriv>
FEXPR_INLINE std::complex<forge::expr::AReal<T, N>> operator*(std::complex<forge::expr::AReal<T, N>> lhs,
                                                    const forge::expr::Expression<T, Expr, Deriv>& rhs)
{
    lhs *= rhs;
    return lhs;
}

template <class T, class Expr, class Deriv>
FEXPR_INLINE std::complex<typename forge::expr::ExprTraits<Expr>::value_type> operator*(
    const std::complex<T>& lhs, const forge::expr::Expression<T, Expr, Deriv>& rhs)
{
    std::complex<typename forge::expr::ExprTraits<Expr>::value_type> z = lhs;
    z *= rhs;
    return z;
}

template <class T, class Expr, class Deriv>
FEXPR_INLINE std::complex<typename forge::expr::ExprTraits<Expr>::value_type> operator*(
    const forge::expr::Expression<T, Expr, Deriv>& lhs, const std::complex<T>& rhs)
{
    std::complex<typename forge::expr::ExprTraits<Expr>::value_type> z = lhs;
    z *= rhs;
    return z;
}

template <class T, std::size_t N>
FEXPR_INLINE std::complex<forge::expr::AReal<T, N>> operator*(const forge::expr::AReal<T, N>& rhs,
                                                    std::complex<forge::expr::AReal<T, N>> lhs)
{
    lhs *= rhs;
    return lhs;
}

template <class T, std::size_t N>
FEXPR_INLINE std::complex<forge::expr::AReal<T, N>> operator*(const T& rhs,
                                                    std::complex<forge::expr::AReal<T, N>> lhs)
{
    lhs *= rhs;
    return lhs;
}

template <class T, std::size_t N, class Expr, class Deriv>
FEXPR_INLINE std::complex<forge::expr::AReal<T, N>> operator*(const forge::expr::Expression<T, Expr, Deriv>& rhs,
                                                    std::complex<forge::expr::AReal<T, N>> lhs)
{
    lhs *= rhs;
    return lhs;
}

// All FReal operator* overloads removed - not used by QuantLib-Risks-Cpp

// operator/

template <class T, std::size_t N>
FEXPR_INLINE std::complex<forge::expr::AReal<T, N>> operator/(std::complex<forge::expr::AReal<T, N>> lhs,
                                                    const std::complex<forge::expr::AReal<T, N>>& rhs)
{
    lhs /= rhs;
    return lhs;
}

template <class T, std::size_t N>
FEXPR_INLINE std::complex<forge::expr::AReal<T, N>> operator/(std::complex<forge::expr::AReal<T, N>> lhs,
                                                    const std::complex<T>& rhs)
{
    lhs /= rhs;
    return lhs;
}

template <class T, std::size_t N>
FEXPR_INLINE std::complex<forge::expr::AReal<T, N>> operator/(const std::complex<T>& lhs,
                                                    std::complex<forge::expr::AReal<T, N>> rhs)
{
    std::complex<forge::expr::AReal<T, N>> z = lhs;
    z /= rhs;
    return z;
}

template <class T, std::size_t N>
FEXPR_INLINE std::complex<forge::expr::AReal<T, N>> operator/(std::complex<forge::expr::AReal<T, N>> lhs,
                                                    const forge::expr::AReal<T, N>& rhs)
{
    lhs /= rhs;
    return lhs;
}

template <class T, std::size_t N>
FEXPR_INLINE std::complex<forge::expr::AReal<T, N>> operator/(std::complex<T> lhs,
                                                    const forge::expr::AReal<T, N>& rhs)
{
    std::complex<forge::expr::AReal<T, N>> z = lhs;
    z /= rhs;
    return z;
}

template <class T, std::size_t N>
FEXPR_INLINE std::complex<forge::expr::AReal<T, N>> operator/(std::complex<forge::expr::AReal<T, N>> lhs,
                                                    const T& rhs)
{
    lhs /= rhs;
    return lhs;
}

template <class T, std::size_t N, class Expr, class Deriv>
FEXPR_INLINE std::complex<forge::expr::AReal<T, N>> operator/(std::complex<forge::expr::AReal<T, N>> lhs,
                                                    const forge::expr::Expression<T, Expr, Deriv>& rhs)
{
    lhs /= rhs;
    return lhs;
}

template <class T, class Expr, class Deriv>
FEXPR_INLINE std::complex<typename forge::expr::ExprTraits<Expr>::value_type> operator/(
    const std::complex<T>& lhs, const forge::expr::Expression<T, Expr, Deriv>& rhs)
{
    std::complex<typename forge::expr::ExprTraits<Expr>::value_type> z = lhs;
    z /= rhs;
    return z;
}

template <class T, class Expr, class Deriv>
FEXPR_INLINE std::complex<typename forge::expr::ExprTraits<Expr>::value_type> operator/(
    const forge::expr::Expression<T, Expr, Deriv>& lhs, const std::complex<T>& rhs)
{
    std::complex<typename forge::expr::ExprTraits<Expr>::value_type> z = lhs;
    z /= rhs;
    return z;
}

template <class T, std::size_t N>
FEXPR_INLINE std::complex<forge::expr::AReal<T, N>> operator/(const forge::expr::AReal<T, N>& rhs,
                                                    std::complex<forge::expr::AReal<T, N>> lhs)
{
    return std::complex<forge::expr::AReal<T, N>>(rhs) / lhs;
}

template <class T, std::size_t N>
FEXPR_INLINE std::complex<forge::expr::AReal<T, N>> operator/(const T& rhs,
                                                    std::complex<forge::expr::AReal<T, N>> lhs)
{
    return std::complex<forge::expr::AReal<T, N>>(rhs) / lhs;
}

template <class T, std::size_t N, class Expr, class Deriv>
FEXPR_INLINE std::complex<forge::expr::AReal<T, N>> operator/(const forge::expr::Expression<T, Expr, Deriv>& rhs,
                                                    std::complex<forge::expr::AReal<T, N>> lhs)
{
    return std::complex<forge::expr::AReal<T, N>>(rhs) / lhs;
}

// All FReal operator/ overloads removed - not used by QuantLib-Risks-Cpp
/////////////////////// math functions

template <class T, std::size_t N>
FEXPR_INLINE forge::expr::AReal<T, N> arg(const complex<forge::expr::AReal<T, N>>& x)
{
    return forge::expr::detail::arg_impl(x);
}

// FReal arg functions removed - not used by QuantLib-Risks-Cpp

template <class Scalar, class Derived, class Deriv>
typename forge::expr::ExprTraits<Derived>::value_type arg(const forge::expr::Expression<Scalar, Derived, Deriv>& x)
{
    return ::forge::expr::detail::arg_impl(x);
}

template <class T, std::size_t N>
typename forge::expr::AReal<T, N> arg(const forge::expr::AReal<T, N>& x)
{
    return ::forge::expr::detail::arg_impl(x);
}

template <class T, class Scalar, class Deriv>
FEXPR_INLINE T norm(const complex<forge::expr::ADTypeBase<Scalar, T, Deriv>>& x)
{
    return ::forge::expr::detail::norm_impl(x);
}

template <class T, std::size_t N>
FEXPR_INLINE forge::expr::AReal<T, N> norm(const complex<forge::expr::AReal<T, N>>& x)
{
    return ::forge::expr::detail::norm_impl(x);
}

// FReal norm function removed - not used by QuantLib-Risks-Cpp

// appleclang15 needs this overload for type paramed norm
#if defined(__APPLE__) && defined(__clang__) && defined(__apple_build_version__) &&                \
    (__apple_build_version__ >= 15000000)
template <class T>
FEXPR_INLINE typename std::enable_if<forge::expr::ExprTraits<T>::isExpr, T>::type norm(complex<T>& x)
{
    return ::forge::expr::detail::norm_impl(x);
}
#endif

// return the expression type from multiplying x*x without actually evaluating it
template <class Scalar, class Derived, class Deriv>
FEXPR_INLINE auto norm(const forge::expr::Expression<Scalar, Derived, Deriv>& x) -> decltype(x * x)
{
    return x * x;
}

template <class T, std::size_t N>
FEXPR_INLINE forge::expr::AReal<T, N> abs(const complex<forge::expr::AReal<T, N>>& x)
{
    return forge::expr::detail::abs_impl(x);
}

// FReal abs function removed - not used by QuantLib-Risks-Cpp

template <class T, std::size_t N>
FEXPR_INLINE complex<forge::expr::AReal<T, N>> conj(const complex<forge::expr::AReal<T, N>>& z)
{
    complex<forge::expr::AReal<T, N>> ret(z.real(), -z.imag());
    return ret;
}

// FReal conj function removed - not used by QuantLib-Risks-Cpp

#if ((defined(_MSC_VER) && (_MSC_VER < 1920)) || (defined(__GNUC__) && __GNUC__ < 7)) &&           \
    !defined(__clang__)
template <class Scalar, class Derived, class Deriv>
FEXPR_INLINE typename forge::expr::ExprTraits<Derived>::value_type conj(
    const forge::expr::Expression<Scalar, Derived, Deriv>& x)
{
    return typename forge::expr::ExprTraits<Derived>::value_type(x);
}
#else
template <class Scalar, class Derived, class Deriv>
FEXPR_INLINE complex<typename forge::expr::ExprTraits<Derived>::value_type> conj(
    const forge::expr::Expression<Scalar, Derived, Deriv>& x)
{
    return complex<typename forge::expr::ExprTraits<Derived>::value_type>(x);
}
#endif

template <class T, std::size_t N>
FEXPR_INLINE complex<forge::expr::AReal<T, N>> proj(const std::complex<forge::expr::AReal<T, N>>& z)
{
    if (forge::expr::isinf(z.real()) || forge::expr::isinf(z.imag()))
    {
        typedef typename forge::expr::ExprTraits<T>::nested_type type;
        const type infty = std::numeric_limits<type>::infinity();
        if (forge::expr::signbit(z.imag()))
            return complex<forge::expr::AReal<T, N>>(infty, -0.0);
        else
            return complex<forge::expr::AReal<T, N>>(infty, 0.0);
    }
    else
        return z;
}

// FReal proj function removed - not used by QuantLib-Risks-Cpp

template <class Scalar, class Derived, class Deriv>
FEXPR_INLINE auto proj(const forge::expr::Expression<Scalar, Derived, Deriv>& x)
    -> decltype(::forge::expr::detail::proj_impl(x))
{
    return ::forge::expr::detail::proj_impl(x);
}

template <class T, std::size_t N>
FEXPR_INLINE auto proj(const forge::expr::AReal<T, N>& x) -> decltype(::forge::expr::detail::proj_impl(x))
{
    return ::forge::expr::detail::proj_impl(x);
}

// FReal proj (scalar) function removed - not used by QuantLib-Risks-Cpp

// T and expr
// expr and T
// different expr (derived1, derived2 - returns scalar)

template <class T, std::size_t N = 1>
FEXPR_INLINE complex<forge::expr::AReal<T, N>> polar(const forge::expr::AReal<T, N>& r,
                                           const forge::expr::AReal<T, N>& theta = forge::expr::AReal<T, N>())
{
    return forge::expr::detail::polar_impl(r, theta);
}

// FReal polar function removed - not used by QuantLib-Risks-Cpp

template <class Scalar, class Expr, class Deriv>
FEXPR_INLINE complex<typename forge::expr::ExprTraits<Expr>::value_type> polar(
    const forge::expr::Expression<Scalar, Expr, Deriv>& r,
    const forge::expr::Expression<Scalar, Expr, Deriv>& theta)
{
    typedef typename forge::expr::ExprTraits<Expr>::value_type type;
    return forge::expr::detail::polar_impl(type(r), type(theta));
}

#if defined(_MSC_VER) && _MSC_VER < 1920
// VS 2017 needs loads of specialisations to resolve the right overload and avoid calling the
// std::version

template <class Scalar, class Op, class Expr, class Deriv, std::size_t N = 1>
FEXPR_INLINE complex<forge::expr::AReal<Scalar, N>> polar(const forge::expr::UnaryExpr<Scalar, Op, Expr, Deriv>& r,
                                                const forge::expr::AReal<Scalar, N>& theta)
{
    return forge::expr::detail::polar_impl(forge::expr::AReal<Scalar, N>(r), theta);
}

template <class Scalar, class Op, class Expr, class Deriv, std::size_t N = 1>
FEXPR_INLINE complex<forge::expr::AReal<Scalar, N>> polar(
    const forge::expr::AReal<Scalar, N>& r, const forge::expr::UnaryExpr<Scalar, Op, Expr, Deriv>& theta)
{
    return forge::expr::detail::polar_impl(r, forge::expr::AReal<Scalar, N>(theta));
}

// FReal polar (UnaryExpr) functions removed - not used by QuantLib-Risks-Cpp

template <class Scalar, class Op, class Expr1, class Expr2, std::size_t M = 1>
FEXPR_INLINE complex<forge::expr::AReal<Scalar, M>> polar(
    const forge::expr::BinaryExpr<Scalar, Op, Expr1, Expr2, typename forge::expr::DerivativesTraits<Scalar, M>::type>& r,
    const forge::expr::AReal<Scalar, M>& theta)
{
    return forge::expr::detail::polar_impl(forge::expr::AReal<Scalar, M>(r), theta);
}

template <class Scalar, class Op, class Expr1, class Expr2, std::size_t M = 1>
FEXPR_INLINE complex<forge::expr::AReal<Scalar, M>> polar(
    const forge::expr::AReal<Scalar, M>& r,
    const forge::expr::BinaryExpr<Scalar, Op, Expr1, Expr2, typename forge::expr::DerivativesTraits<Scalar, M>::type>&
        theta)
{
    return forge::expr::detail::polar_impl(r, forge::expr::AReal<Scalar, M>(theta));
}

// FReal polar (BinaryExpr) functions removed - not used by QuantLib-Risks-Cpp

template <class Scalar, class Op1, class Expr1, class Expr2, class Op3, class Expr3, class Deriv>
FEXPR_INLINE complex<typename forge::expr::ExprTraits<Expr1>::value_type> polar(
    const forge::expr::UnaryExpr<Scalar, Op3, Expr3, Deriv>& r,
    const forge::expr::BinaryExpr<Scalar, Op1, Expr1, Expr2, Deriv>& theta)
{
    typedef typename forge::expr::ExprTraits<Expr1>::value_type type;
    return forge::expr::detail::polar_impl(type(r), type(theta));
}

template <class Scalar, class Op1, class Expr1, class Expr2, class Op3, class Expr3, class Deriv>
FEXPR_INLINE complex<typename forge::expr::ExprTraits<Expr1>::value_type> polar(
    const forge::expr::BinaryExpr<Scalar, Op1, Expr1, Expr2, Deriv>& r,
    const forge::expr::UnaryExpr<Scalar, Op3, Expr3, Deriv>& theta)
{
    typedef typename forge::expr::ExprTraits<Expr1>::value_type type;
    return forge::expr::detail::polar_impl(type(r), type(theta));
}

template <class Scalar, class Op1, class Expr1, class Op2, class Expr2, class Deriv>
FEXPR_INLINE complex<typename forge::expr::ExprTraits<Expr1>::value_type> polar(
    const forge::expr::UnaryExpr<Scalar, Op1, Expr1, Deriv>& r,
    const forge::expr::UnaryExpr<Scalar, Op2, Expr2, Deriv>& theta)
{
    typedef typename forge::expr::ExprTraits<Expr1>::value_type type;
    return forge::expr::detail::polar_impl(type(r), type(theta));
}

template <class Scalar, class Op1, class Expr1, class Expr2, class Op3, class Expr3, class Expr4,
          class Deriv>
FEXPR_INLINE complex<typename forge::expr::ExprTraits<Expr1>::value_type> polar(
    const forge::expr::BinaryExpr<Scalar, Op3, Expr3, Expr4, Deriv>& r,
    const forge::expr::BinaryExpr<Scalar, Op1, Expr1, Expr2, Deriv>& theta)
{
    typedef typename forge::expr::ExprTraits<Expr1>::value_type type;
    return forge::expr::detail::polar_impl(type(r), type(theta));
}

template <class Scalar, class Op, class Expr1, class Expr2, class Deriv>
FEXPR_INLINE complex<typename forge::expr::ExprTraits<Expr1>::value_type> polar(
    double r, const forge::expr::BinaryExpr<Scalar, Op, Expr1, Expr2, Deriv>& theta)
{
    return forge::expr::detail::polar_impl(typename forge::expr::ExprTraits<Expr1>::value_type(r),
                                   typename forge::expr::ExprTraits<Expr1>::value_type(theta));
}

template <class Scalar, class Op, class Expr1, class Expr2, class Deriv>
FEXPR_INLINE complex<typename forge::expr::ExprTraits<Expr1>::value_type> polar(
    const forge::expr::BinaryExpr<Scalar, Op, Expr1, Expr2, Deriv>& r, double theta)
{
    return forge::expr::detail::polar_impl(typename forge::expr::ExprTraits<Expr1>::value_type(r),
                                   typename forge::expr::ExprTraits<Expr1>::value_type(theta));
}

template <class Scalar, class Op, class Expr, class Deriv>
FEXPR_INLINE complex<typename forge::expr::ExprTraits<Expr>::value_type> polar(
    double r, const forge::expr::UnaryExpr<Scalar, Op, Expr, Deriv>& theta)
{
    return forge::expr::detail::polar_impl(typename forge::expr::ExprTraits<Expr>::value_type(r),
                                   typename forge::expr::ExprTraits<Expr>::value_type(theta));
}

template <class Scalar, class Op, class Expr, class Deriv>
FEXPR_INLINE complex<typename forge::expr::ExprTraits<Expr>::value_type> polar(
    const forge::expr::UnaryExpr<Scalar, Op, Expr, Deriv>& r, double theta)
{
    return forge::expr::detail::polar_impl(typename forge::expr::ExprTraits<Expr>::value_type(r),
                                   typename forge::expr::ExprTraits<Expr>::value_type(theta));
}

template <class Scalar, std::size_t M = 1>
FEXPR_INLINE complex<forge::expr::AReal<Scalar, M>> polar(const forge::expr::AReal<Scalar, M>& r, double theta)
{
    return forge::expr::detail::polar_impl(r, forge::expr::AReal<Scalar, M>(theta));
}

// FReal polar (with double) functions removed - not used by QuantLib-Risks-Cpp

template <class Scalar, std::size_t M = 1>
FEXPR_INLINE complex<forge::expr::AReal<Scalar, M>> polar(double r, const forge::expr::AReal<Scalar, M>& theta)
{
    return forge::expr::detail::polar_impl(forge::expr::AReal<Scalar, M>(r), theta);
}

#endif

// 2 different expression types passed:
// we only enable this function if the underlying value_type of both expressions
// is the same
template <class Scalar, class Expr1, class Expr2, class Deriv>
FEXPR_INLINE typename std::enable_if<std::is_same<typename forge::expr::ExprTraits<Expr1>::value_type,
                                                typename forge::expr::ExprTraits<Expr2>::value_type>::value,
                                   complex<typename forge::expr::ExprTraits<Expr1>::value_type>>::type
polar(const forge::expr::Expression<Scalar, Expr1, Deriv>& r,
      const forge::expr::Expression<Scalar, Expr2, Deriv>& theta = 0)
{
    typedef typename forge::expr::ExprTraits<Expr1>::value_type type;
    return forge::expr::detail::polar_impl(type(r), type(theta));
}

// T, expr - only enabled if T is scalar
template <class Scalar, class Expr, class Deriv>
FEXPR_INLINE std::complex<typename forge::expr::ExprTraits<Expr>::value_type> polar(
    Scalar r, const forge::expr::Expression<Scalar, Expr, Deriv>& theta = 0)
{
    return forge::expr::detail::polar_impl(typename forge::expr::ExprTraits<Expr>::value_type(r), theta.derived());
}

// expr, T - only enabled if T is scalar
template <class Scalar, class Expr, class Deriv>
FEXPR_INLINE std::complex<typename forge::expr::ExprTraits<Expr>::value_type> polar(
    const forge::expr::Expression<Scalar, Expr, Deriv>& r, Scalar theta = Scalar())
{
    return forge::expr::detail::polar_impl(r.derived(), typename forge::expr::ExprTraits<Expr>::value_type(theta));
}

// just one expr, as second parameter is optional
template <class Scalar, class Expr, class Deriv>
FEXPR_INLINE complex<typename forge::expr::ExprTraits<Expr>::value_type> polar(
    const forge::expr::Expression<Scalar, Expr, Deriv>& r)
{
    return complex<typename forge::expr::ExprTraits<Expr>::value_type>(r);
}

template <class T, std::size_t N>
FEXPR_INLINE complex<forge::expr::AReal<T, N>> exp(const complex<forge::expr::AReal<T, N>>& z)
{
    return forge::expr::detail::exp_impl(z);
}

// FReal exp function removed - not used by QuantLib-Risks-Cpp

template <class T, std::size_t N = 1>
FEXPR_INLINE complex<forge::expr::AReal<T, N>> log(const complex<forge::expr::AReal<T, N>>& z)
{
    return complex<forge::expr::AReal<T, N>>(log(forge::expr::detail::abs_impl(z)), forge::expr::detail::arg_impl(z));
}

// FReal log function removed - not used by QuantLib-Risks-Cpp

template <class T, std::size_t N = 1>
FEXPR_INLINE complex<forge::expr::AReal<T, N>> log10(const complex<forge::expr::AReal<T, N>>& z)
{
    // log(z) * 1/log(10)
    return log(z) * T(0.43429448190325182765112891891661);
}

// FReal log10 function removed - not used by QuantLib-Risks-Cpp

template <class T, std::size_t N>
FEXPR_INLINE complex<forge::expr::AReal<T, N>> pow(const complex<forge::expr::AReal<T, N>>& x,
                                         const complex<forge::expr::AReal<T, N>>& y)
{
    return forge::expr::detail::exp_impl(log(x) * y);
}

template <class T, std::size_t N>
FEXPR_INLINE complex<forge::expr::AReal<T, N>> pow(const complex<forge::expr::AReal<T, N>>& x,
                                         const forge::expr::AReal<T, N>& y)
{
    return forge::expr::detail::exp_impl(log(x) * y);
}

template <class T, std::size_t N>
FEXPR_INLINE complex<forge::expr::AReal<T, N>> pow(const complex<forge::expr::AReal<T, N>>& x, const complex<T>& y)
{
    return pow(x, complex<forge::expr::AReal<T, N>>(y));
}

template <class T, std::size_t N>
FEXPR_INLINE complex<forge::expr::AReal<T, N>> pow(const complex<T>& x, const complex<forge::expr::AReal<T, N>>& y)
{
    return pow(complex<forge::expr::AReal<T, N>>(x), y);
}

template <class T, std::size_t N>
FEXPR_INLINE complex<forge::expr::AReal<T, N>> pow(const complex<forge::expr::AReal<T, N>>& x, const T& y)
{
    return forge::expr::detail::exp_impl(log(x) * y);
}

template <class T, class T2, std::size_t N>
FEXPR_INLINE typename std::enable_if<forge::expr::ExprTraits<T2>::isExpr, complex<forge::expr::AReal<T, N>>>::type
pow(const complex<forge::expr::AReal<T, N>>& x, const T2& y)
{
    return forge::expr::detail::exp_impl(log(x) * y);
}

template <class T, std::size_t N>
FEXPR_INLINE complex<forge::expr::AReal<T, N>> pow(const complex<forge::expr::AReal<T, N>>& x, int y)
{
    return forge::expr::detail::exp_impl(log(x) * T(y));
}

template <class T, std::size_t N>
FEXPR_INLINE complex<forge::expr::AReal<T, N>> pow(const complex<forge::expr::AReal<T, N>>& x, short y)
{
    return forge::expr::detail::exp_impl(log(x) * T(y));
}

template <class T, std::size_t N>
FEXPR_INLINE complex<forge::expr::AReal<T, N>> pow(const complex<forge::expr::AReal<T, N>>& x, long y)
{
    return forge::expr::detail::exp_impl(log(x) * T(y));
}

template <class T, std::size_t N>
FEXPR_INLINE complex<forge::expr::AReal<T, N>> pow(const complex<forge::expr::AReal<T, N>>& x, long long y)
{
    return forge::expr::detail::exp_impl(log(x) * T(y));
}

template <class T, std::size_t N>
FEXPR_INLINE complex<forge::expr::AReal<T, N>> pow(const complex<forge::expr::AReal<T, N>>& x, unsigned y)
{
    return forge::expr::detail::exp_impl(log(x) * T(y));
}

template <class T, std::size_t N>
FEXPR_INLINE complex<forge::expr::AReal<T, N>> pow(const complex<forge::expr::AReal<T, N>>& x, unsigned short y)
{
    return forge::expr::detail::exp_impl(log(x) * T(y));
}

template <class T, std::size_t N>
FEXPR_INLINE complex<forge::expr::AReal<T, N>> pow(const complex<forge::expr::AReal<T, N>>& x, unsigned long y)
{
    return forge::expr::detail::exp_impl(log(x) * T(y));
}

template <class T, std::size_t N>
FEXPR_INLINE complex<forge::expr::AReal<T, N>> pow(const complex<forge::expr::AReal<T, N>>& x, unsigned long long y)
{
    return forge::expr::detail::exp_impl(log(x) * T(y));
}

template <class T, std::size_t N>
FEXPR_INLINE complex<forge::expr::AReal<T, N>> pow(const forge::expr::AReal<T, N>& x,
                                         const complex<forge::expr::AReal<T, N>>& y)
{
    return forge::expr::detail::exp_impl(log(x) * y);
}

template <class T, std::size_t N>
FEXPR_INLINE complex<forge::expr::AReal<T, N>> pow(const T& x, const complex<forge::expr::AReal<T, N>>& y)
{
    return forge::expr::detail::exp_impl(log(x) * y);
}

template <class T, std::size_t N>
FEXPR_INLINE complex<forge::expr::AReal<T, N>> pow(int x, const complex<forge::expr::AReal<T, N>>& y)
{
    return forge::expr::detail::exp_impl(log(T(x)) * y);
}

template <class T, std::size_t N>
FEXPR_INLINE complex<forge::expr::AReal<T, N>> pow(short x, const complex<forge::expr::AReal<T, N>>& y)
{
    return forge::expr::detail::exp_impl(log(T(x)) * y);
}

template <class T, std::size_t N>
FEXPR_INLINE complex<forge::expr::AReal<T, N>> pow(long x, const complex<forge::expr::AReal<T, N>>& y)
{
    return forge::expr::detail::exp_impl(log(T(x)) * y);
}

template <class T, std::size_t N>
FEXPR_INLINE complex<forge::expr::AReal<T, N>> pow(long long x, const complex<forge::expr::AReal<T, N>>& y)
{
    return forge::expr::detail::exp_impl(log(T(x)) * y);
}

template <class T, std::size_t N>
FEXPR_INLINE complex<forge::expr::AReal<T, N>> pow(unsigned x, const complex<forge::expr::AReal<T, N>>& y)
{
    return forge::expr::detail::exp_impl(log(T(x)) * y);
}

template <class T, std::size_t N>
FEXPR_INLINE complex<forge::expr::AReal<T, N>> pow(unsigned short x, const complex<forge::expr::AReal<T, N>>& y)
{
    return forge::expr::detail::exp_impl(log(T(x)) * y);
}

template <class T, std::size_t N>
FEXPR_INLINE complex<forge::expr::AReal<T, N>> pow(unsigned long x, const complex<forge::expr::AReal<T, N>>& y)
{
    return forge::expr::detail::exp_impl(log(T(x)) * y);
}

template <class T, std::size_t N>
FEXPR_INLINE complex<forge::expr::AReal<T, N>> pow(unsigned long long x, const complex<forge::expr::AReal<T, N>>& y)
{
    return forge::expr::detail::exp_impl(log(T(x)) * y);
}

template <class T, class T2, std::size_t N>
FEXPR_INLINE typename std::enable_if<forge::expr::ExprTraits<T2>::isExpr, complex<forge::expr::AReal<T, N>>>::type
pow(const T2& x, const complex<forge::expr::AReal<T, N>>& y)
{
    return forge::expr::detail::exp_impl(log(x) * y);
}

// All FReal pow overloads removed - not used by QuantLib-Risks-Cpp

template <class T, std::size_t N>
FEXPR_INLINE std::complex<forge::expr::AReal<T, N>> sqrt(const std::complex<forge::expr::AReal<T, N>>& z)
{
    return ::forge::expr::detail::sqrt_impl(z);
}

// FReal sqrt function removed - not used by QuantLib-Risks-Cpp

template <class T, std::size_t N>
FEXPR_INLINE std::complex<forge::expr::AReal<T, N>> sin(const std::complex<forge::expr::AReal<T, N>>& z)
{
    return ::forge::expr::detail::sin_impl(z);
}

// FReal sin function removed - not used by QuantLib-Risks-Cpp

template <class T, std::size_t N>
FEXPR_INLINE std::complex<forge::expr::AReal<T, N>> cos(const std::complex<forge::expr::AReal<T, N>>& z)
{
    return ::forge::expr::detail::cos_impl(z);
}

// FReal cos function removed - not used by QuantLib-Risks-Cpp

template <class T, std::size_t N>
FEXPR_INLINE std::complex<forge::expr::AReal<T, N>> tan(const std::complex<forge::expr::AReal<T, N>>& z)
{
    return ::forge::expr::detail::tan_impl(z);
}

// FReal tan function removed - not used by QuantLib-Risks-Cpp

template <class T, std::size_t N>
FEXPR_INLINE std::complex<forge::expr::AReal<T, N>> asin(const std::complex<forge::expr::AReal<T, N>>& z)
{
    return ::forge::expr::detail::asin_impl(z);
}

// FReal asin function removed - not used by QuantLib-Risks-Cpp

template <class T, std::size_t N>
FEXPR_INLINE std::complex<forge::expr::AReal<T, N>> acos(const std::complex<forge::expr::AReal<T, N>>& z)
{
    return ::forge::expr::detail::acos_impl(z);
}

// FReal acos function removed - not used by QuantLib-Risks-Cpp

template <class T, std::size_t N>
FEXPR_INLINE std::complex<forge::expr::AReal<T, N>> atan(const std::complex<forge::expr::AReal<T, N>>& z)
{
    return ::forge::expr::detail::atan_impl(z);
}

// FReal atan function removed - not used by QuantLib-Risks-Cpp

template <class T, std::size_t N>
FEXPR_INLINE std::complex<forge::expr::AReal<T, N>> sinh(const std::complex<forge::expr::AReal<T, N>>& z)
{
    return ::forge::expr::detail::sinh_impl(z);
}

// FReal sinh function removed - not used by QuantLib-Risks-Cpp

template <class T, std::size_t N>
FEXPR_INLINE std::complex<forge::expr::AReal<T, N>> cosh(const std::complex<forge::expr::AReal<T, N>>& z)
{
    return ::forge::expr::detail::cosh_impl(z);
}

// FReal cosh function removed - not used by QuantLib-Risks-Cpp

template <class T, std::size_t N>
FEXPR_INLINE std::complex<forge::expr::AReal<T, N>> tanh(const std::complex<forge::expr::AReal<T, N>>& z)
{
    return ::forge::expr::detail::tanh_impl(z);
}

// FReal tanh function removed - not used by QuantLib-Risks-Cpp

template <class T, std::size_t N>
FEXPR_INLINE std::complex<forge::expr::AReal<T, N>> asinh(const std::complex<forge::expr::AReal<T, N>>& z)
{
    return ::forge::expr::detail::asinh_impl(z);
}

// FReal asinh function removed - not used by QuantLib-Risks-Cpp

template <class T, std::size_t N>
FEXPR_INLINE std::complex<forge::expr::AReal<T, N>> acosh(const std::complex<forge::expr::AReal<T, N>>& z)
{
    return ::forge::expr::detail::acosh_impl(z);
}

// FReal acosh function removed - not used by QuantLib-Risks-Cpp

template <class T, std::size_t N>
FEXPR_INLINE std::complex<forge::expr::AReal<T, N>> atanh(const std::complex<forge::expr::AReal<T, N>>& z)
{
    return ::forge::expr::detail::atanh_impl(z);
}

// FReal atanh function removed - not used by QuantLib-Risks-Cpp

}  // namespace std

namespace forge { namespace expr {
namespace detail
{

template <class T>
FEXPR_INLINE T norm_impl(const std::complex<T>& x)
{
    return x.real() * x.real() + x.imag() * x.imag();
}

template <class T>
FEXPR_INLINE T abs_impl(const std::complex<T>& x)
{
    using std::sqrt;
    if (forge::expr::isinf(x.real()) || forge::expr::isinf(x.imag()))
        return std::numeric_limits<double>::infinity();
    return forge::expr::hypot(x.real(), x.imag());
}

template <class T>
FEXPR_INLINE std::complex<T> exp_impl(const std::complex<T>& z)
{
    using std::cos;
    using std::exp;
    using std::sin;
    typedef typename forge::expr::ExprTraits<T>::nested_type nested;
    if (forge::expr::isinf(z.real()))
    {
        if (z.real() > 0.0)
        {
            if (z.imag() == 0.0)
                return std::complex<T>(std::numeric_limits<nested>::infinity(), 0.0);
            if (forge::expr::isinf(z.imag()) && z.imag() > 0.0)
                return std::complex<T>(std::numeric_limits<nested>::infinity(),
                                       std::numeric_limits<nested>::quiet_NaN());
            if (forge::expr::isnan(z.imag()))
                return std::complex<T>(std::numeric_limits<nested>::infinity(),
                                       std::numeric_limits<nested>::quiet_NaN());
        }
        else
        {
            if (forge::expr::isinf(z.imag()) && z.imag() > 0.0)
                return std::complex<T>(0.0, 0.0);
            if (forge::expr::isnan(z.imag()))
                return std::complex<T>(0.0, 0.0);
        }
    }
    else if (forge::expr::isnan(z.real()))
    {
        if (z.imag() == 0.0 && !forge::expr::signbit(z.imag()))
            return std::complex<T>(std::numeric_limits<nested>::quiet_NaN(), 0.0);
        else
            return std::complex<T>(std::numeric_limits<nested>::quiet_NaN(),
                                   std::numeric_limits<nested>::quiet_NaN());
    }
    T e = ::forge::expr::exp(z.real());
    return std::complex<T>(e * cos(z.imag()), e * sin(z.imag()));
}

template <class T1, class T2>
FEXPR_INLINE std::complex<typename forge::expr::ExprTraits<T1>::value_type> polar_impl(const T1& r,
                                                                             const T2& theta)
{
    using std::cos;
    using std::sin;
    typedef typename forge::expr::ExprTraits<T1>::value_type base_type;
    return std::complex<base_type>(base_type(r * cos(theta)), base_type(r * sin(theta)));
}

template <class T>
FEXPR_INLINE std::complex<T> sqrt_impl(const std::complex<T>& z)
{
    typedef typename forge::expr::ExprTraits<T>::nested_type nested;
    if (forge::expr::isinf(z.real()) && z.real() < 0.0)
    {
        if (forge::expr::isfinite(z.imag()) && z.imag() > 0.0)
            return std::complex<T>(0.0, std::numeric_limits<nested>::infinity());
        if (forge::expr::isnan(z.imag()))
            return std::complex<T>(std::numeric_limits<nested>::quiet_NaN(),
                                   std::numeric_limits<nested>::infinity());
    }
    if (forge::expr::isinf(z.real()) && z.real() > 0.0)
    {
        if (forge::expr::isfinite(z.imag()) && z.imag() > 0.0)
            return std::complex<T>(std::numeric_limits<nested>::infinity(), 0.0);

        if (forge::expr::isnan(z.imag()))
            return std::complex<T>(std::numeric_limits<nested>::infinity(),
                                   std::numeric_limits<nested>::quiet_NaN());
    }
    if (forge::expr::isinf(z.imag()) && z.imag() > 0.0)
        return std::complex<T>(std::numeric_limits<nested>::infinity(),
                               std::numeric_limits<nested>::infinity());

    return ::forge::expr::detail::polar_impl(sqrt(abs(z)), arg(z) * T(0.5));
}

template <class T>
FEXPR_INLINE std::complex<T> sinh_impl(const std::complex<T>& z)
{
    typedef typename forge::expr::ExprTraits<T>::nested_type nested;
    auto cls = forge::expr::fpclassify(z.real());
    if (cls == FP_INFINITE && forge::expr::isinf(z.imag()) && z.real() > 0.0 && z.imag() > 0.0)
        return std::complex<T>(std::numeric_limits<nested>::infinity(),
                               std::numeric_limits<nested>::quiet_NaN());
    if (cls == FP_NAN && z.imag() == 0.0 && !forge::expr::signbit(z.imag()))
        return std::complex<T>(std::numeric_limits<nested>::quiet_NaN(), 0.0);
    if (cls == FP_ZERO && !forge::expr::signbit(z.real()))
    {
        if ((forge::expr::isinf(z.imag()) && z.imag() > 0.0) || forge::expr::isnan(z.imag()))
            return std::complex<T>(0.0, std::numeric_limits<nested>::quiet_NaN());
    }
    return (exp(z) - exp(-z)) / T(2.0);
}

template <class T>
FEXPR_INLINE std::complex<T> cosh_impl(const std::complex<T>& z)
{
    typedef typename forge::expr::ExprTraits<T>::nested_type nested;
    auto cls = forge::expr::fpclassify(z.real());
    if (cls == FP_INFINITE && forge::expr::isinf(z.imag()) && z.real() > 0.0 && z.imag() > 0.0)
        return std::complex<T>(std::numeric_limits<nested>::infinity(),
                               std::numeric_limits<nested>::quiet_NaN());
    if (cls == FP_NAN && z.imag() == 0.0 && !forge::expr::signbit(z.imag()))
        return std::complex<T>(std::numeric_limits<nested>::quiet_NaN(), 0.0);
    if (cls == FP_ZERO && !forge::expr::signbit(z.real()))
    {
        if ((forge::expr::isinf(z.imag()) && z.imag() > 0.0) || forge::expr::isnan(z.imag()))
            return std::complex<T>(std::numeric_limits<nested>::quiet_NaN(), 0.0);
    }
    return (exp(z) + exp(-z)) / T(2.0);
}

template <class T>
FEXPR_INLINE std::complex<T> tanh_impl(const std::complex<T>& z)
{
    typedef typename forge::expr::ExprTraits<T>::nested_type nested;
    if (z.real() == 0.0)
    {
        if (forge::expr::isinf(z.imag()) && z.imag() > 0.0)
        {
#if defined(__APPLE__) || (defined(__GLIBC__) && __GLIBC__ == 2 && __GLIBC_MINOR__ < 27)
            return std::complex<T>(std::numeric_limits<nested>::quiet_NaN(),
                                   std::numeric_limits<nested>::quiet_NaN());
#else
            return std::complex<T>(0.0, std::numeric_limits<nested>::quiet_NaN());
#endif
        }
        if (forge::expr::isnan(z.imag()))
        {
#if defined(__APPLE__) | (defined(__GLIBC__) && __GLIBC__ == 2 && __GLIBC_MINOR__ < 27)
            return std::complex<T>(std::numeric_limits<nested>::quiet_NaN(),
                                   std::numeric_limits<nested>::quiet_NaN());
#else
            return std::complex<T>(0.0, std::numeric_limits<nested>::quiet_NaN());
#endif
        }
    }
    if (forge::expr::isinf(z.real()) && z.real() > 0.0 && (z.imag() > 0.0 || forge::expr::isnan(z.imag())))
        return std::complex<T>(1.0, 0.0);
    if (forge::expr::isnan(z.real()) && z.imag() == 0.0)
        return std::complex<T>(std::numeric_limits<nested>::quiet_NaN(), 0.0);
    return sinh(z) / cosh(z);
}

template <class T>
FEXPR_INLINE std::complex<T> asinh_impl(const std::complex<T>& z)
{
    typedef typename forge::expr::ExprTraits<T>::nested_type nested;
    if (forge::expr::isinf(z.real()) && z.real() > 0.0)
    {
        if (forge::expr::isinf(z.imag()) && z.imag() > 0.0)
            return std::complex<T>(std::numeric_limits<nested>::infinity(),
                                   3.141592653589793238462643383279502884197169399 * 0.25);
        if (forge::expr::isnan(z.imag()))
            return std::complex<T>(std::numeric_limits<nested>::infinity(),
                                   std::numeric_limits<nested>::quiet_NaN());
        if (z.imag() > 0.0)
            return std::complex<T>(std::numeric_limits<nested>::infinity(), 0.0);
    }
    if (forge::expr::isnan(z.real()))
    {
        if (forge::expr::isinf(z.imag()) && z.imag() > 0.0)
            return std::complex<T>(std::numeric_limits<nested>::infinity(),
                                   std::numeric_limits<nested>::quiet_NaN());
        if (z.imag() == 0.0 && !forge::expr::signbit(z.imag()))
            return std::complex<T>(std::numeric_limits<nested>::quiet_NaN(), 0.0);
    }
    if (forge::expr::isinf(z.imag()) && z.imag() > 0.0 && forge::expr::isfinite(z.real()) && z.real() > 0.0)
        return std::complex<T>(std::numeric_limits<nested>::infinity(),
                               3.141592653589793238462643383279502884197169399 * 0.5);
    return log(z + sqrt(T(1.0) + (z * z)));
}

template <class T>
FEXPR_INLINE std::complex<T> acosh_impl(const std::complex<T>& z)
{
    typedef typename forge::expr::ExprTraits<T>::nested_type nested;
    if (forge::expr::isinf(z.imag()) && z.imag() > 0.0)
    {
        if (forge::expr::isfinite(z.real()))
            return std::complex<T>(std::numeric_limits<nested>::infinity(),
                                   3.141592653589793238462643383279502884197169399 * 0.5);
        if (forge::expr::isinf(z.real()) && z.real() < 0.0)
            return std::complex<T>(std::numeric_limits<nested>::infinity(),
                                   3.141592653589793238462643383279502884197169399 * 0.75);
    }
    if (forge::expr::isnan(z.imag()))
    {
        if (z.real() == 0.0)
#if defined(__APPLE__) | (defined(__GLIBC__) && __GLIBC__ == 2 && __GLIBC_MINOR__ < 27)
            return std::complex<T>(std::numeric_limits<nested>::quiet_NaN(),
                                   std::numeric_limits<nested>::quiet_NaN());
#else
            return std::complex<T>(std::numeric_limits<nested>::quiet_NaN(),
                                   3.141592653589793238462643383279502884197169399 * 0.5);
#endif
        else if (forge::expr::isinf(z.real()))
            return std::complex<T>(std::numeric_limits<nested>::infinity(),
                                   std::numeric_limits<nested>::quiet_NaN());
        else
            return std::complex<T>(std::numeric_limits<nested>::quiet_NaN(),
                                   std::numeric_limits<nested>::quiet_NaN());
    }
    if (forge::expr::isinf(z.real()) && forge::expr::isfinite(z.imag()) && z.imag() > 0.0)
    {
        if (z.real() < 0.0)
            return std::complex<T>(std::numeric_limits<nested>::infinity(),
                                   3.141592653589793238462643383279502884197169399);
        else
            return std::complex<T>(std::numeric_limits<nested>::infinity(), +0.0);
    }
    if (forge::expr::isnan(z.real()) && forge::expr::isinf(z.imag()) && z.imag() > 0.0)
        return std::complex<T>(std::numeric_limits<nested>::infinity(),
                               std::numeric_limits<nested>::quiet_NaN());

    return log(z + sqrt(z + T(1.0)) * sqrt(z - T(1.0)));
}

template <class T>
FEXPR_INLINE std::complex<T> atanh_impl(const std::complex<T>& z)
{
    typedef typename forge::expr::ExprTraits<T>::nested_type nested;
    if (forge::expr::isinf(z.real()) && z.real() > 0.0)
    {
        if (forge::expr::isinf(z.imag()) && z.imag() > 0.0)
            return std::complex<T>(0.0, 3.141592653589793238462643383279502884197169399 * 0.5);
        if (forge::expr::isnan(z.imag()))
            return std::complex<T>(0.0, std::numeric_limits<nested>::quiet_NaN());
        if (forge::expr::isfinite(z.imag()) && z.imag() > 0.0)
            return std::complex<T>(0.0, 3.141592653589793238462643383279502884197169399 * 0.5);
    }
    if (forge::expr::isnan(z.real()) && forge::expr::isinf(z.imag()) && z.imag() > 0.0)
        return std::complex<T>(0.0, 3.141592653589793238462643383279502884197169399 * 0.5);
    if (z.real() == 1.0 && z.imag() == 0.0)
        return std::complex<T>(std::numeric_limits<nested>::infinity(), 0.0);
    if (z.real() > 0.0 && forge::expr::isfinite(z.real()) && forge::expr::isinf(z.imag()) && z.imag() > 0.0)
        return std::complex<T>(0.0, 3.141592653589793238462643383279502884197169399 * 0.5);
    if (z.real() == 0.0)
    {
        if (z.imag() == 0.0)
            return std::complex<T>(0.0, 0.0);
        if (forge::expr::isnan(z.imag()))
            return std::complex<T>(0.0, std::numeric_limits<nested>::quiet_NaN());
    }
    return (log(T(1.0) + z) - log(T(1.0) - z)) / T(2.0);
}

template <class T>
FEXPR_INLINE std::complex<T> sin_impl(const std::complex<T>& z)
{
    // -i * sinh(i*z)
    std::complex<T> iz(-z.imag(), z.real());
    std::complex<T> sinhiz = sinh(iz);
    return std::complex<T>(sinhiz.imag(), -sinhiz.real());
}

template <class T>
FEXPR_INLINE std::complex<T> cos_impl(const std::complex<T>& z)
{
    // cosh(i*z)
    std::complex<T> iz(-z.imag(), z.real());
    return cosh(iz);
}

template <class T>
FEXPR_INLINE std::complex<T> tan_impl(const std::complex<T>& z)
{
    // -i * tanh(i*z)
    std::complex<T> iz(-z.imag(), z.real());
    std::complex<T> tanhiz = tanh(iz);
    return std::complex<T>(tanhiz.imag(), -tanhiz.real());
}

template <class T>
FEXPR_INLINE std::complex<T> asin_impl(const std::complex<T>& z)
{
    // -i * asinh(i*z);
    std::complex<T> iz(-z.imag(), z.real());
    std::complex<T> asinhiz = asinh(iz);
    return std::complex<T>(asinhiz.imag(), -asinhiz.real());
}

template <class T>
FEXPR_INLINE std::complex<T> acos_impl(const std::complex<T>& z)
{
    typedef typename forge::expr::ExprTraits<T>::nested_type nested;
    if (z.real() == 0.0)
    {
        if (z.imag() == 0.0 && !forge::expr::signbit(z.imag()))
            return std::complex<T>(3.141592653589793238462643383279502884197169399 * 0.5, -0.0);
        if (forge::expr::isnan(z.imag()))
            return std::complex<T>(3.141592653589793238462643383279502884197169399 * 0.5,
                                   -std::numeric_limits<nested>::quiet_NaN());
    }
    if (forge::expr::isfinite(z.real()) && forge::expr::isinf(z.imag()) && z.imag() > 0.0)
        return std::complex<T>(3.141592653589793238462643383279502884197169399 * 0.5,
                               -std::numeric_limits<nested>::infinity());
    if (forge::expr::isinf(z.real()))
    {
        if (z.real() < 0.0)
        {

            if (forge::expr::isfinite(z.imag()) && z.imag() >= 0.0)
                return std::complex<T>(3.141592653589793238462643383279502884197169399,
                                       -std::numeric_limits<nested>::infinity());
            if (forge::expr::isinf(z.imag()) && z.imag() > 0.0)
                return std::complex<T>(3.141592653589793238462643383279502884197169399 * 0.75,
                                       -std::numeric_limits<nested>::infinity());
        }
        else
        {
            if (forge::expr::isfinite(z.imag()) && z.imag() >= 0.0)
                return std::complex<T>(+0.0, -std::numeric_limits<nested>::infinity());
            if (forge::expr::isinf(z.imag()) && z.imag() > 0.0)
                return std::complex<T>(3.141592653589793238462643383279502884197169399 * 0.25,
                                       -std::numeric_limits<nested>::infinity());
        }
        if (forge::expr::isnan(z.imag()))
            return std::complex<T>(std::numeric_limits<nested>::quiet_NaN(),
                                   std::numeric_limits<nested>::infinity());
    }
    if (forge::expr::isnan(z.real()))
    {
        if (forge::expr::isfinite(z.imag()))
            return std::complex<T>(std::numeric_limits<nested>::quiet_NaN(),
                                   std::numeric_limits<nested>::quiet_NaN());
        else if (forge::expr::isinf(z.imag()) && z.imag() > 0.0)
            return std::complex<T>(std::numeric_limits<nested>::quiet_NaN(),
                                   -std::numeric_limits<nested>::infinity());
    }

    std::complex<T> iz(-z.imag(), z.real());
    std::complex<T> lnizsqrt = log(iz + sqrt(T(1.0) - (z * z)));
    std::complex<T> ilnizsqrt(-lnizsqrt.imag(), lnizsqrt.real());
    return T(3.141592653589793238462643383279502884197169399 * 0.5) + ilnizsqrt;
}

template <class T>
FEXPR_INLINE std::complex<T> atan_impl(const std::complex<T>& z)
{
    // -i * atanh(i*z)
    std::complex<T> iz(-z.imag(), z.real());
    std::complex<T> atanhiz = atanh(iz);
    return std::complex<T>(atanhiz.imag(), -atanhiz.real());
}

template <class T>
FEXPR_INLINE T arg_impl(const std::complex<T>& z)
{
    using std::atan2;
    return atan2(z.imag(), z.real());
}

template <class Scalar, class Derived, class Deriv>
FEXPR_INLINE typename forge::expr::ExprTraits<Derived>::value_type arg_impl(
    const forge::expr::Expression<Scalar, Derived, Deriv>& x)
{
    using std::atan2;

    // as this function returns constants only depending on > or < 0,
    // where derivatives are 0 anyway, we can return scalars converted to the
    // underlying expression type
    typedef typename forge::expr::ExprTraits<Derived>::value_type ret_type;
#if defined(_MSC_VER) && _MSC_VER < 1920
    // VS 2017 evaluates this differently
    (void)x;  // silence unused warning
    return ret_type();
#else
    if (x > 0.0)
        return ret_type();
    else if (x < 0.0)
        return ret_type(3.141592653589793238462643383279502884197169399);  // PI
    else
        return atan2(ret_type(), ret_type(x));  // for correct handling of +/- zero
#endif
}

#if (defined(_MSC_VER) && (_MSC_VER < 1920) || (defined(__GNUC__) && __GNUC__ < 7)) &&             \
    !defined(__clang__)
template <class Scalar, class Derived, class Deriv>
FEXPR_INLINE typename forge::expr::ExprTraits<Derived>::value_type proj_impl(
    const forge::expr::Expression<Scalar, Derived, Deriv>& x)
{
    return typename forge::expr::ExprTraits<Derived>::value_type(x);
}
#else
template <class Scalar, class Derived, class Deriv>
FEXPR_INLINE std::complex<typename forge::expr::ExprTraits<Derived>::value_type> proj_impl(
    const forge::expr::Expression<Scalar, Derived, Deriv>& x)
{
    if (forge::expr::isinf(x))
        return std::complex<typename forge::expr::ExprTraits<Derived>::value_type>(
            std::numeric_limits<typename forge::expr::ExprTraits<Derived>::nested_type>::infinity());
    else
        return std::complex<typename forge::expr::ExprTraits<Derived>::value_type>(x);
}
#endif

}  // namespace detail
}}  // namespace forge::expr

