/*******************************************************************************

   QuantLib-Forge header: Enables using forge::expr::AReal with QuantLib and Boost.

   This file is part of QuantLib-Forge, a Forge integration layer for QuantLib.

   Copyright (C) 2025 The QuantLib-Forge Authors

   Derived from QuantLib-Risks-Cpp / XAD (https://github.com/auto-differentiation/XAD)
   Original XAD code: Copyright (C) 2010-2024 Xcelerit Computing Ltd.

   This program is free software: you can redistribute it and/or modify
   it under the terms of the GNU Affero General Public License as published
   by the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU Affero General Public License for more details.

   You should have received a copy of the GNU Affero General Public License
   along with this program.  If not, see <http://www.gnu.org/licenses/>.

******************************************************************************/

#pragma once

#include <boost/accumulators/numeric/functional.hpp>
#include <boost/accumulators/statistics/tail_quantile.hpp>
#include <boost/math/special_functions/math_fwd.hpp>
#include <boost/math/tools/promotion.hpp>
#include <boost/math/tools/rational.hpp>
#include <boost/numeric/ublas/operations.hpp>
#include <boost/type_traits/is_arithmetic.hpp>
#include <boost/type_traits/is_floating_point.hpp>
#include <boost/type_traits/is_pod.hpp>
#include <boost/numeric/conversion/cast.hpp>
#include <expressions/Literals.hpp>
#include <expressions/ExpressionTemplates/BinaryExpr.hpp>
#include <expressions/ExpressionTemplates/UnaryExpr.hpp>
#include <expressions/Compatibility/Complex.hpp>
#include <expressions/Compatibility/StdCompatibility.hpp>
#include <limits>
#include <type_traits>

#define QL_REAL forge::expr::AReal<double>
#define QL_FORGE 1

// QuantLib specialisations to work with expressions
namespace QuantLib {

    // from ql/functional.hpp
    template <class T>
    T squared(T);

    // for binary expressions
    template <class Op, class Expr1, class Expr2>
    typename forge::expr::AReal<double> squared(const forge::expr::BinaryExpr<double, Op, Expr1, Expr2>& x) {
        return squared(forge::expr::AReal<double>(x));
    }

    // for unary expressions
    template <class Op, class Expr>
    typename forge::expr::AReal<double> squared(const forge::expr::UnaryExpr<double, Op, Expr>& x) {
        return squared(forge::expr::AReal<double>(x));
    }
}

// Boost specializations
namespace boost {

    template <class Target, class Op, class Expr>
    inline Target numeric_cast(const forge::expr::UnaryExpr<double, Op, Expr>& arg) {
        return numeric_cast<Target>(value(arg));
    }

    template <class Target, class Op, class Expr1, class Expr2>
    inline Target numeric_cast(const forge::expr::BinaryExpr<double, Op, Expr1, Expr2>& arg) {
        return numeric_cast<Target>(value(arg));
    }

    namespace math {

        // full specialisations for promoting 2 types where one of them is AReal<double>,
        // used by boost math functions a lot
        namespace tools {
            template <>
            struct promote_args_permissive<forge::expr::AReal<double>, forge::expr::AReal<double>> {
                typedef forge::expr::AReal<double> type;
            };

            template <class T>
            struct promote_args_permissive<forge::expr::AReal<double>, T> {
                typedef forge::expr::AReal<double> type;
            };

            template <class T>
            struct promote_args_permissive<T, forge::expr::AReal<double>> {
                typedef forge::expr::AReal<double> type;
            };
        }

        // propagating policies for boost math involving AReal
        namespace policies {
            template <class Policy>
            struct evaluation<forge::expr::AReal<double>, Policy> {
                using type = forge::expr::AReal<double>;
            };

            template <class Policy, class Op, class Expr1, class Expr2>
            struct evaluation<forge::expr::BinaryExpr<double, Op, Expr1, Expr2>, Policy> {
                using type = typename evaluation<forge::expr::AReal<double>, Policy>::type;
            };

            template <class Policy, class Op, class Expr>
            struct evaluation<forge::expr::UnaryExpr<double, Op, Expr>, Policy> {
                using type = typename evaluation<forge::expr::AReal<double>, Policy>::type;
            };
        }

        /* specialised version of boost/math/special_functions/erfc for forge::expr expressions,
         *  casting the argument type to the underlying value-type and calling the boost original.
         */
        template <class Op, class Expr, class Policy>
        inline forge::expr::AReal<double> erfc(forge::expr::UnaryExpr<double, Op, Expr> z, const Policy& pol) {
            return boost::math::erfc(forge::expr::AReal<double>(z), pol);
        }

        template <class Op, class Expr, class Policy>
        forge::expr::AReal<double> erfc_inv(forge::expr::UnaryExpr<double, Op, Expr> z, const Policy& pol) {
            return boost::math::erfc_inv(forge::expr::AReal<double>(z), pol);
        }

        namespace tools {
            // boost/tools/rational.hpp, as it's called from erfc with expressions since boost 1.83

            template <std::size_t N, class T, class Op, class Expr1, class Expr2>
            forge::expr::AReal<double> evaluate_polynomial(const T (&a)[N],
                                                   forge::expr::BinaryExpr<double, Op, Expr1, Expr2> val) {
                return evaluate_polynomial(a, forge::expr::AReal<double>(val));
            }
        }


        template <class RT1, class Op, class Expr, class RT3, class Policy>
        inline forge::expr::AReal<double>
        ibetac(RT1 a, forge::expr::UnaryExpr<double, Op, Expr> b, RT3 x, const Policy& pol) {
            return boost::math::ibetac(forge::expr::AReal<double>(a), forge::expr::AReal<double>(b),
                                       forge::expr::AReal<double>(x), pol);
        }

        template <class RT1, class RT2, class Op, class Expr1, class Expr2, class Policy>
        inline forge::expr::AReal<double> ibeta_derivative(RT1 a,
                                                   RT2 b,
                                                   forge::expr::BinaryExpr<double, Op, Expr1, Expr2> x,
                                                   const Policy& pol) {
            return boost::math::ibeta_derivative(forge::expr::AReal<double>(a), forge::expr::AReal<double>(b),
                                                 forge::expr::AReal<double>(x), pol);
        }

        template <class Op, class Expr, class RT2, class RT3, class Policy>
        inline forge::expr::AReal<double>
        ibeta(forge::expr::UnaryExpr<double, Op, Expr> a, RT2 b, RT3 x, const Policy& pol) {
            return boost::math::ibeta(forge::expr::AReal<double>(a), forge::expr::AReal<double>(b),
                                      forge::expr::AReal<double>(x), pol);
        }

        template <class Op, class Expr, class T2, class Op2, class Expr2, class T4, class Policy>
        inline forge::expr::AReal<double> ibeta_inv(forge::expr::UnaryExpr<double, Op, Expr> a,
                                            T2 b,
                                            forge::expr::UnaryExpr<double, Op2, Expr2> p,
                                            T4* py,
                                            const Policy& pol) {
            return boost::math::ibeta_inv(forge::expr::AReal<double>(a), forge::expr::AReal<double>(b),
                                          forge::expr::AReal<double>(p), py, pol);
        }

        template <class Op, class Expr, class RT2, class A>
        inline forge::expr::AReal<double> beta(forge::expr::UnaryExpr<double, Op, Expr> a, RT2 b, A arg) {
            return boost::math::beta(forge::expr::AReal<double>(a), forge::expr::AReal<double>(b), arg);
        }

        template <class Op, class Expr1, class Expr2, class Policy>
        inline forge::expr::AReal<double> log1p(forge::expr::BinaryExpr<double, Op, Expr1, Expr2> x,
                                        const Policy& pol) {
            return boost::math::log1p(forge::expr::AReal<double>(x), pol);
        }

        template <class Op, class Expr, class Policy>
        inline forge::expr::AReal<double> log1p(forge::expr::UnaryExpr<double, Op, Expr> x, const Policy& pol) {
            return boost::math::log1p(forge::expr::AReal<double>(x), pol);
        }

        template <class Op, class Expr1, class Expr2, class Policy>
        inline forge::expr::AReal<double> tgamma(forge::expr::BinaryExpr<double, Op, Expr1, Expr2> z,
                                         const Policy& pol) {
            return boost::math::tgamma(forge::expr::AReal<double>(z), pol);
        }

        template <class Op, class Expr, class Policy>
        inline forge::expr::AReal<double> tgamma(forge::expr::UnaryExpr<double, Op, Expr> z, const Policy& pol) {
            return boost::math::tgamma(forge::expr::AReal<double>(z), pol);
        }

        template <class Op, class Expr, class T2, class Policy>
        inline forge::expr::AReal<double>
        tgamma_delta_ratio(forge::expr::UnaryExpr<double, Op, Expr> z, T2 delta, const Policy& pol) {
            return boost::math::tgamma_delta_ratio(forge::expr::AReal<double>(z), forge::expr::AReal<double>(delta),
                                                   pol);
        }

        template <class Op, class Expr, class T2, class Policy>
        inline forge::expr::AReal<double>
        gamma_q_inv(forge::expr::UnaryExpr<double, Op, Expr> a, T2 p, const Policy& pol) {
            return boost::math::gamma_q_inv(forge::expr::AReal<double>(a), forge::expr::AReal<double>(p), pol);
        }

        template <class Op, class Expr, class T2, class Policy>
        inline forge::expr::AReal<double>
        gamma_p_inv(forge::expr::UnaryExpr<double, Op, Expr> a, T2 p, const Policy& pol) {
            return boost::math::gamma_p_inv(forge::expr::AReal<double>(a), forge::expr::AReal<double>(p), pol);
        }

        template <class Op, class Expr>
        inline forge::expr::AReal<double> trunc(const forge::expr::UnaryExpr<double, Op, Expr>& v) {
            return boost::math::trunc(forge::expr::AReal<double>(v));
        }

        template <class Op, class Expr1, class Expr2>
        inline forge::expr::AReal<double> trunc(const forge::expr::BinaryExpr<double, Op, Expr1, Expr2>& v) {
            return boost::math::trunc(forge::expr::AReal<double>(v));
        }

        template <class Op, class Expr>
        inline long_long_type lltrunc(const forge::expr::UnaryExpr<double, Op, Expr>& v) {
            return boost::math::lltrunc(forge::expr::value(v));
        }

        template <class Op, class Expr1, class Expr2>
        inline long_long_type lltrunc(const forge::expr::BinaryExpr<double, Op, Expr1, Expr2>& v) {
            return boost::math::lltrunc(forge::expr::value(v));
        }

        inline long_long_type lltrunc(const forge::expr::AReal<double>& v) {
            return boost::math::lltrunc(forge::expr::value(v));
        }

        template <class Policy>
        inline long_long_type llround(const forge::expr::AReal<double>& v, const Policy& p) {
            return boost::math::llround(forge::expr::value(v), p);
        }

        template <class Op, class Expr1, class Expr2>
        inline int itrunc(const forge::expr::BinaryExpr<double, Op, Expr1, Expr2>& v) {
            return itrunc(forge::expr::value(v), policies::policy<>());
        }

        template <class Op, class Expr>
        inline int itrunc(const forge::expr::UnaryExpr<double, Op, Expr>& v) {
            return itrunc(forge::expr::value(v), policies::policy<>());
        }

        template <class Op, class Expr1, class Expr2, class Policy>
        inline forge::expr::AReal<double> expm1(forge::expr::BinaryExpr<double, Op, Expr1, Expr2> x,
                                        const Policy& pol) {
            return boost::math::expm1(forge::expr::AReal<double>(x), pol);
        }

        template <class Op1, class Op2, class Expr1, class Expr2, class Policy>
        inline forge::expr::AReal<double> gamma_p(forge::expr::UnaryExpr<double, Op1, Expr1> a,
                                          forge::expr::UnaryExpr<double, Op2, Expr2> z,
                                          const Policy& pol) {
            return boost::math::gamma_p(forge::expr::AReal<double>(a), forge::expr::AReal<double>(z), pol);
        }

        template <class Op1, class Expr1, class Op2, class Expr2, class Expr3>
        inline forge::expr::AReal<double> gamma_p(forge::expr::UnaryExpr<double, Op1, Expr1> a,
                                          forge::expr::BinaryExpr<double, Op2, Expr2, Expr3> z) {
            return boost::math::gamma_p(forge::expr::AReal<double>(a), forge::expr::AReal<double>(z),
                                        policies::policy<>());
        }

        template <class Op1, class Op2, class Expr, class Expr1, class Expr2, class Policy>
        inline forge::expr::AReal<double> gamma_p(forge::expr::UnaryExpr<double, Op1, Expr> a,
                                          forge::expr::BinaryExpr<double, Op2, Expr1, Expr2> z,
                                          const Policy& pol) {
            return boost::math::gamma_p(forge::expr::AReal<double>(a), forge::expr::AReal<double>(z), pol);
        }

        inline int fpclassify BOOST_NO_MACRO_EXPAND(const forge::expr::AReal<double>& t) {
            return (boost::math::fpclassify)(forge::expr::value(t));
        }

        template <class Op1, class Op2, class Expr1, class Expr2, class Policy>
        inline forge::expr::AReal<double> gamma_p_derivative(forge::expr::UnaryExpr<double, Op1, Expr1> a,
                                                     forge::expr::UnaryExpr<double, Op2, Expr2> x,
                                                     const Policy&) {
            return boost::math::gamma_p_derivative(forge::expr::AReal<double>(a), forge::expr::AReal<double>(x),
                                                   policies::policy<>());
        }

        template <class Op1, class Op2, class Expr1, class Expr2, class Policy>
        inline forge::expr::AReal<double> gamma_q(forge::expr::UnaryExpr<double, Op1, Expr1> a,
                                          forge::expr::UnaryExpr<double, Op2, Expr2> x,
                                          const Policy& pol) {
            return boost::math::gamma_q(forge::expr::AReal<double>(a), forge::expr::AReal<double>(x), pol);
        }

        template <class Op, class Expr, class T2, class Policy>
        inline forge::expr::AReal<double>
        gamma_q(forge::expr::UnaryExpr<double, Op, Expr> a, T2 z, const Policy& pol) {
            return boost::math::gamma_q(forge::expr::AReal<double>(a), z, pol);
        }

        template <class Op, class Expr, class T2, class Policy>
        inline forge::expr::AReal<double>
        gamma_p_derivative(forge::expr::UnaryExpr<double, Op, Expr> a, T2 x, const Policy& pol) {
            return boost::math::gamma_p_derivative(forge::expr::AReal<double>(a), forge::expr::AReal<double>(x),
                                                   pol);
        }

        template <class Policy>
        inline int itrunc(const forge::expr::AReal<double>& v, const Policy& pol) {
            return boost::math::itrunc(forge::expr::value(v), pol);
        }

        template <class Policy>
        inline int iround(const forge::expr::AReal<double>& v, const Policy& pol) {
            return boost::math::iround(forge::expr::value(v), pol);
        }

        template <class Op, class Expr1, class Expr2>
        inline forge::expr::AReal<double> expm1(forge::expr::BinaryExpr<double, Op, Expr1, Expr2> x) {
            return expm1(forge::expr::AReal<double>(x), policies::policy<>());
        }

        template <class Op1, class Op2, class Expr1, class Expr2, class Policy>
        inline typename detail::bessel_traits<forge::expr::AReal<double>, forge::expr::AReal<double>, Policy>::
            result_type
            cyl_bessel_i(forge::expr::UnaryExpr<double, Op1, Expr1> v,
                         forge::expr::UnaryExpr<double, Op2, Expr2> x,
                         const Policy&) {
            return boost::math::cyl_bessel_i(forge::expr::AReal<double>(v), forge::expr::AReal<double>(x),
                                             policies::policy<>());
        }

        template <class Op, class Expr>
        inline forge::expr::AReal<double> lgamma(forge::expr::UnaryExpr<double, Op, Expr> z, int* sign) {
            return boost::math::lgamma(forge::expr::AReal<double>(z), sign);
        }

        template <class Op, class Expr1, class Expr2>
        inline forge::expr::AReal<double> lgamma(forge::expr::BinaryExpr<double, Op, Expr1, Expr2> z, int* sign) {
            return boost::math::lgamma(forge::expr::AReal<double>(z), sign);
        }

        template <class Op, class Expr, class Policy>
        inline forge::expr::AReal<double> lgamma(forge::expr::UnaryExpr<double, Op, Expr> x, const Policy& pol) {
            return boost::math::lgamma(forge::expr::AReal<double>(x), pol);
        }
        template <class Op, class Expr1, class Expr2, class Policy>
        inline forge::expr::AReal<double> lgamma(forge::expr::BinaryExpr<double, Op, Expr1, Expr2> x,
                                         const Policy& pol) {
            return boost::math::lgamma(forge::expr::AReal<double>(x), pol);
        }

        template <class Op, class Expr, class Policy>
        inline forge::expr::AReal<double> tgamma1pm1(forge::expr::UnaryExpr<double, Op, Expr> z,
                                             const Policy& pol) {
            return boost::math::tgamma1pm1(forge::expr::AReal<double>(z), pol);
        }

        inline bool(isfinite)(const forge::expr::AReal<double>& x) {
            return (boost::math::isfinite)(forge::expr::value(x));
        }
        inline bool(isinf)(const forge::expr::AReal<double>& x) {
            return (boost::math::isinf)(forge::expr::value(x));
        }

        template <class Op, class Expr1, class Expr2, class Policy>
        inline forge::expr::AReal<double> powm1(forge::expr::BinaryExpr<double, Op, Expr1, Expr2> a,
                                        const forge::expr::AReal<double>& z,
                                        const Policy& pol) {
            return boost::math::powm1(forge::expr::AReal<double>(a), z, pol);
        }

                // overrides for special functions that break with expression templates
        namespace detail {
            #if BOOST_VERSION >= 108900
                // we needed to copy this from here and make adjustments to the return statement, to ensure that it works with expression templates:
                // https://github.com/boostorg/math/blob/develop/include/boost/math/special_functions/beta.hpp#L1136
                template <class Policy>
                BOOST_MATH_GPU_ENABLED forge::expr::AReal<double> ibeta_large_ab(forge::expr::AReal<double> a, forge::expr::AReal<double> b, forge::expr::AReal<double> x, forge::expr::AReal<double> y, bool invert, bool normalised, const Policy& pol)
                {
                    BOOST_MATH_STD_USING

                    forge::expr::AReal<double> x0 = a / (a + b);
                    forge::expr::AReal<double> y0 = b / (a + b);
                    forge::expr::AReal<double> nu = x0 * log(x / x0) + y0 * log(y / y0);
                    if ((nu > 0) || (x == x0) || (y == y0))
                        nu = 0;
                    nu = sqrt(-2 * nu);

                    if ((nu != 0) && (nu / (x - x0) < 0))
                        nu = -nu;

                    forge::expr::AReal<double> mul = 1;
                    if (!normalised)
                        mul = boost::math::beta(a, b, pol);

                    return mul * ((invert ? forge::expr::AReal<double>((1 + boost::math::erf(forge::expr::AReal<double>(-nu * sqrt((a + b) / 2)), pol)) / 2) : forge::expr::AReal<double>(boost::math::erfc(forge::expr::AReal<double>(-nu * sqrt((a + b) / 2)), pol) / 2)));
                }
            #endif
        }
    }

    namespace numeric {

        // static integer power implementations with forge::expr expressions - evaluate first and call
        // underlying
        template <class Op, class Expr, int N>
        forge::expr::AReal<double> pow(forge::expr::UnaryExpr<double, Op, Expr> const& x, mpl::int_<N>) {
            return pow(forge::expr::AReal<double>(x), N);
        }

        template <class Op, class Expr1, class Expr2, int N>
        forge::expr::AReal<double> pow(forge::expr::BinaryExpr<double, Op, Expr1, Expr2> const& x, mpl::int_<N>) {
            return pow(forge::expr::AReal<double>(x), N);
        }

        // override boost accumulators traits to determine result types for AReal
        // (only divides and multiplies are used from QuantLib)
        namespace functional {

            template <>
            struct result_of_divides<forge::expr::AReal<double>, forge::expr::AReal<double> > {
                typedef forge::expr::AReal<double> type;
            };

            template <class T>
            struct result_of_divides<forge::expr::AReal<double>, T> {
                typedef forge::expr::AReal<double> type;
            };

            template <class T>
            struct result_of_divides<T, forge::expr::AReal<double> > {
                typedef forge::expr::AReal<double> type;
            };

            template <>
            struct result_of_multiplies<forge::expr::AReal<double>, forge::expr::AReal<double> > {
                typedef forge::expr::AReal<double> type;
            };

            template <class T>
            struct result_of_multiplies<forge::expr::AReal<double>, T> {
                typedef forge::expr::AReal<double> type;
            };

            template <class T>
            struct result_of_multiplies<T, forge::expr::AReal<double> > {
                typedef forge::expr::AReal<double> type;
            };

        }

        // traits for ublas type promotion for operands of forge::expr type
        namespace ublas {
            // AReal x AReal
            template <>
            struct promote_traits<forge::expr::AReal<double>, forge::expr::AReal<double> > {
                typedef forge::expr::AReal<double> promote_type;
            };

            // AReal x T
            template <class T>
            struct promote_traits<forge::expr::AReal<double>, T> {
                typedef forge::expr::AReal<double> promote_type;
            };

            // T x AReal
            template <class T>
            struct promote_traits<T, forge::expr::AReal<double> > {
                typedef forge::expr::AReal<double> promote_type;
            };

        }
    }

    // AReal behaves like a floating point number
    template <>
    struct is_floating_point<forge::expr::AReal<double> > : public true_type {};

    // AReal is arithmetic
    template <>
    struct is_arithmetic<forge::expr::AReal<double> > : public true_type {};

    // AReal is not a POD type though
    template <>
    struct is_pod<forge::expr::AReal<double> > : public false_type {};

    // AReal is only convertible to itself, not to another type
    template <class To>
    struct is_convertible<forge::expr::AReal<double>, To> : public false_type {};

    template <>
    struct is_convertible<forge::expr::AReal<double>, forge::expr::AReal<double> > : public true_type {};
}


// MSVC specialisations / fixes
#ifdef _MSC_VER

// required for random - as it calls ::sqrt on arguments
using forge::expr::sqrt;
using forge::expr::pow;
using forge::expr::exp;
using forge::expr::log;
using forge::expr::tan;

#include <random>

namespace std {

    // this is needed to avoid std::random to revert to a binary / constexpr
    // implementation for Real in the random generator
    template <>
    struct _Has_static_min_max<std::mt19937, void> : false_type {};
}

#endif

// Mac specialisations / fixes
#ifdef __APPLE__

// Mac uses an internal namespace _VSTD for its math functions, which are called from
// random header with full namespace qualification.
// We therefore need to import the forge::expr math functions into that to make it work

namespace std {
    inline namespace _LIBCPP_ABI_NAMESPACE {
        using forge::expr::sqrt;
        using forge::expr::pow;
        using forge::expr::log;
        using forge::expr::tan;
    }
}

// have to include this last, to make sure functions are in the right namespace before
#include <random>

#endif

