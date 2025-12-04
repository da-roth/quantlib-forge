/* -*- mode: c++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*- */

/*******************************************************************************

   Error Function - Forge-Aware Implementation

   This version uses ABool::If for all conditional branches, allowing
   Forge to properly record both branches in the computation graph.
   This enables kernel reuse when inputs change and different branches
   become active.

   Based on QuantLib's ErrorFunction implementation.

   This file is part of QuantLib-Forge, a Forge integration layer for QuantLib.

   Copyright (C) 2025 The QuantLib-Forge Authors

   SPDX-License-Identifier: AGPL-3.0-or-later

******************************************************************************/

#ifndef QUANTLIB_FORGE_TEST_ERROR_FUNCTION_FORGE_HPP
#define QUANTLIB_FORGE_TEST_ERROR_FUNCTION_FORGE_HPP

#include <expressions/abool.hpp>
#include <expressions/abool_helpers.hpp>
#include <expressions/Literals.hpp>
#include <expressions/ExpressionTemplates/BinaryOperators.hpp>
#include <expressions/ExpressionTemplates/UnaryOperators.hpp>
#include <cmath>
#include <cfloat>
#include <limits>

namespace qlforge_test {

    // Helper to compute absolute value for Real type using ABool::If
    template<typename T>
    inline T forgeAbs(const T& x) {
        forge::ABool cond = forge::less(x.forgeValue(), T(0.0).forgeValue());
        T neg_x = -x;
        T pos_x = x;
        return forge::ABool::If(cond, neg_x, pos_x);
    }

    // Forge-aware ErrorFunction implementation
    // Uses ABool::If for all conditional branches
    template<typename Real>
    class ErrorFunctionForge {
      public:
        Real operator()(Real x) const {
            Real ax = forgeAbs(x);

            // Compute all possible branch results
            // Branch 1: |x| < 0.84375 (small x)
            Real result_small = computeSmallX(x, ax);

            // Branch 2: 0.84375 <= |x| < 1.25 (medium x)
            Real result_medium = computeMediumX(x, ax);

            // Branch 3: |x| >= 6 (very large x)
            Real result_large = computeLargeX(x);

            // Branch 4: 1.25 <= |x| < 6 (intermediate x)
            Real result_intermediate = computeIntermediateX(x, ax);

            // Use ABool::If to select the correct result
            // Build conditions
            forge::ABool cond_small = forge::less(ax.forgeValue(), Real(0.84375).forgeValue());
            forge::ABool cond_medium = forge::less(ax.forgeValue(), Real(1.25).forgeValue());
            forge::ABool cond_large = forge::greaterEqual(ax.forgeValue(), Real(6.0).forgeValue());

            // Nested selection: if small, else if medium, else if large, else intermediate
            // Start from innermost: large vs intermediate
            Real result_not_medium = forge::ABool::If(cond_large, result_large, result_intermediate);

            // medium vs (large or intermediate)
            Real result_not_small = forge::ABool::If(cond_medium, result_medium, result_not_medium);

            // small vs everything else
            Real result = forge::ABool::If(cond_small, result_small, result_not_small);

            return result;
        }

      private:
        // Branch 1: |x| < 0.84375
        Real computeSmallX(Real x, Real /*ax*/) const {
            // For very small x, use simplified form
            // We'll use the polynomial form for all small x to keep it simple
            Real z = x * x;
            Real r = Real(pp0) + z * (Real(pp1) + z * (Real(pp2) + z * (Real(pp3) + z * Real(pp4))));
            Real s = Real(one) + z * (Real(qq1) + z * (Real(qq2) + z * (Real(qq3) + z * (Real(qq4) + z * Real(qq5)))));
            Real y = r / s;
            return x + x * y;
        }

        // Branch 2: 0.84375 <= |x| < 1.25
        Real computeMediumX(Real x, Real ax) const {
            Real s = ax - Real(one);
            Real P = Real(pa0) + s * (Real(pa1) + s * (Real(pa2) + s * (Real(pa3) + s * (Real(pa4) + s * (Real(pa5) + s * Real(pa6))))));
            Real Q = Real(one) + s * (Real(qa1) + s * (Real(qa2) + s * (Real(qa3) + s * (Real(qa4) + s * (Real(qa5) + s * Real(qa6))))));
            Real ratio = P / Q;

            // Use ABool::If for sign selection - force both branches to Real type
            forge::ABool cond_positive = forge::greaterEqual(x.forgeValue(), Real(0.0).forgeValue());
            Real pos_result = Real(erx) + ratio;
            Real neg_result = Real(-erx) - ratio;
            return forge::ABool::If(cond_positive, pos_result, neg_result);
        }

        // Branch 3: |x| >= 6
        Real computeLargeX(Real x) const {
            // Returns +/- (1 - tiny)
            forge::ABool cond_positive = forge::greaterEqual(x.forgeValue(), Real(0.0).forgeValue());
            Real pos_result = Real(one - tiny);
            Real neg_result = Real(tiny - one);
            return forge::ABool::If(cond_positive, pos_result, neg_result);
        }

        // Branch 4: 1.25 <= |x| < 6
        Real computeIntermediateX(Real x, Real ax) const {
            Real s = Real(one) / (ax * ax);

            // Sub-branch: |x| < 2.857... vs |x| >= 2.857...
            Real R_low = Real(ra0) + s * (Real(ra1) + s * (Real(ra2) + s * (Real(ra3) + s * (Real(ra4) + s * (Real(ra5) + s * (Real(ra6) + s * Real(ra7)))))));
            Real S_low = Real(one) + s * (Real(sa1) + s * (Real(sa2) + s * (Real(sa3) + s * (Real(sa4) + s * (Real(sa5) + s * (Real(sa6) + s * (Real(sa7) + s * Real(sa8))))))));

            Real R_high = Real(rb0) + s * (Real(rb1) + s * (Real(rb2) + s * (Real(rb3) + s * (Real(rb4) + s * (Real(rb5) + s * Real(rb6))))));
            Real S_high = Real(one) + s * (Real(sb1) + s * (Real(sb2) + s * (Real(sb3) + s * (Real(sb4) + s * (Real(sb5) + s * (Real(sb6) + s * Real(sb7)))))));

            forge::ABool cond_low = forge::less(ax.forgeValue(), Real(2.85714285714285).forgeValue());
            Real R = forge::ABool::If(cond_low, R_low, R_high);
            Real S = forge::ABool::If(cond_low, S_low, S_high);

            Real exp_arg = -ax * ax - Real(0.5625) + R / S;
            Real r = forge::expr::exp(exp_arg);

            forge::ABool cond_positive = forge::greaterEqual(x.forgeValue(), Real(0.0).forgeValue());
            Real pos_result = Real(one) - r / ax;
            Real neg_result = r / ax - Real(one);
            return forge::ABool::If(cond_positive, pos_result, neg_result);
        }

        // Constants
        static constexpr double tiny = std::numeric_limits<double>::epsilon();
        static constexpr double one = 1.00000000000000000000e+00;
        static constexpr double erx = 8.45062911510467529297e-01;

        // Coefficients for approximation to erf on [0,0.84375]
        static constexpr double efx = 1.28379167095512586316e-01;
        static constexpr double efx8 = 1.02703333676410069053e+00;
        static constexpr double pp0 = 1.28379167095512558561e-01;
        static constexpr double pp1 = -3.25042107247001499370e-01;
        static constexpr double pp2 = -2.84817495755985104766e-02;
        static constexpr double pp3 = -5.77027029648944159157e-03;
        static constexpr double pp4 = -2.37630166566501626084e-05;
        static constexpr double qq1 = 3.97917223959155352819e-01;
        static constexpr double qq2 = 6.50222499887672944485e-02;
        static constexpr double qq3 = 5.08130628187576562776e-03;
        static constexpr double qq4 = 1.32494738004321644526e-04;
        static constexpr double qq5 = -3.96022827877536812320e-06;

        // Coefficients for approximation to erf in [0.84375,1.25]
        static constexpr double pa0 = -2.36211856075265944077e-03;
        static constexpr double pa1 = 4.14856118683748331666e-01;
        static constexpr double pa2 = -3.72207876035701323847e-01;
        static constexpr double pa3 = 3.18346619901161753674e-01;
        static constexpr double pa4 = -1.10894694282396677476e-01;
        static constexpr double pa5 = 3.54783043256182359371e-02;
        static constexpr double pa6 = -2.16637559486879084300e-03;
        static constexpr double qa1 = 1.06420880400844228286e-01;
        static constexpr double qa2 = 5.40397917702171048937e-01;
        static constexpr double qa3 = 7.18286544141962662868e-02;
        static constexpr double qa4 = 1.26171219808761642112e-01;
        static constexpr double qa5 = 1.36370839120290507362e-02;
        static constexpr double qa6 = 1.19844998467991074170e-02;

        // Coefficients for approximation to erfc in [1.25,1/0.35]
        static constexpr double ra0 = -9.86494403484714822705e-03;
        static constexpr double ra1 = -6.93858572707181764372e-01;
        static constexpr double ra2 = -1.05586262253232909814e+01;
        static constexpr double ra3 = -6.23753324503260060396e+01;
        static constexpr double ra4 = -1.62396669462573470355e+02;
        static constexpr double ra5 = -1.84605092906711035994e+02;
        static constexpr double ra6 = -8.12874355063065934246e+01;
        static constexpr double ra7 = -9.81432934416914548592e+00;
        static constexpr double sa1 = 1.96512716674392571292e+01;
        static constexpr double sa2 = 1.37657754143519042600e+02;
        static constexpr double sa3 = 4.34565877475229228821e+02;
        static constexpr double sa4 = 6.45387271733267880336e+02;
        static constexpr double sa5 = 4.29008140027567833386e+02;
        static constexpr double sa6 = 1.08635005541779435134e+02;
        static constexpr double sa7 = 6.57024977031928170135e+00;
        static constexpr double sa8 = -6.04244152148580987438e-02;

        // Coefficients for approximation to erfc in [1/.35,28]
        static constexpr double rb0 = -9.86494292470009928597e-03;
        static constexpr double rb1 = -7.99283237680523006574e-01;
        static constexpr double rb2 = -1.77579549177547519889e+01;
        static constexpr double rb3 = -1.60636384855821916062e+02;
        static constexpr double rb4 = -6.37566443368389627722e+02;
        static constexpr double rb5 = -1.02509513161107724954e+03;
        static constexpr double rb6 = -4.83519191608651397019e+02;
        static constexpr double sb1 = 3.03380607434824582924e+01;
        static constexpr double sb2 = 3.25792512996573918826e+02;
        static constexpr double sb3 = 1.53672958608443695994e+03;
        static constexpr double sb4 = 3.19985821950859553908e+03;
        static constexpr double sb5 = 2.55305040643316442583e+03;
        static constexpr double sb6 = 4.74528541206955367215e+02;
        static constexpr double sb7 = -2.24409524465858183362e+01;
    };

}  // namespace qlforge_test

#endif
