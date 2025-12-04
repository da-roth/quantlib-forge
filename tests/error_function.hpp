/* -*- mode: c++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*- */

/*******************************************************************************

   Error Function - Original QuantLib Implementation

   This is a standalone copy of QuantLib's ErrorFunction for testing
   Forge kernel reuse without needing to build full QuantLib.

   Based on code from the GNU C library, originally written by Sun.
   Copyright (C) 1993 by Sun Microsystems, Inc. All rights reserved.

   This file is part of QuantLib-Forge, a Forge integration layer for QuantLib.

   Copyright (C) 2025 The QuantLib-Forge Authors

   SPDX-License-Identifier: AGPL-3.0-or-later

******************************************************************************/

#ifndef QUANTLIB_FORGE_TEST_ERROR_FUNCTION_HPP
#define QUANTLIB_FORGE_TEST_ERROR_FUNCTION_HPP

#include <cmath>
#include <cfloat>
#include <limits>

namespace qlforge_test {

    // Original QuantLib ErrorFunction implementation
    // Has multiple conditional branches based on input magnitude
    template<typename Real>
    class ErrorFunction {
      public:
        Real operator()(Real x) const {
            Real R, S, P, Q, s, y, z, r, ax;

            // Handle non-finite values
            double xVal = static_cast<double>(x);
            if (!std::isfinite(xVal)) {
                if (std::isnan(xVal))
                    return x;
                else
                    return (xVal > 0 ? Real(1.0) : Real(-1.0));
            }

            ax = std::fabs(x);

            if (ax < 0.84375) {      /* |x|<0.84375 */
                if (ax < 3.7252902984e-09) { /* |x|<2**-28 */
                    if (ax < DBL_MIN * 16)
                        return Real(0.125) * (Real(8.0) * x + efx8 * x);  /*avoid underflow */
                    return x + efx * x;
                }
                z = x * x;
                r = pp0 + z * (pp1 + z * (pp2 + z * (pp3 + z * pp4)));
                s = one + z * (qq1 + z * (qq2 + z * (qq3 + z * (qq4 + z * qq5))));
                y = r / s;
                return x + x * y;
            }
            if (ax < 1.25) {      /* 0.84375 <= |x| < 1.25 */
                s = ax - one;
                P = pa0 + s * (pa1 + s * (pa2 + s * (pa3 + s * (pa4 + s * (pa5 + s * pa6)))));
                Q = one + s * (qa1 + s * (qa2 + s * (qa3 + s * (qa4 + s * (qa5 + s * qa6)))));
                if (x >= Real(0.0)) return erx + P / Q; else return -erx - P / Q;
            }
            if (ax >= 6.0) {      /* inf>|x|>=6 */
                if (x >= Real(0.0)) return one - tiny; else return tiny - one;
            }

            /* Starts to lose accuracy when ax~5 */
            s = one / (ax * ax);

            if (ax < 2.85714285714285) { /* |x| < 1/0.35 */
                R = ra0 + s * (ra1 + s * (ra2 + s * (ra3 + s * (ra4 + s * (ra5 + s * (ra6 + s * ra7))))));
                S = one + s * (sa1 + s * (sa2 + s * (sa3 + s * (sa4 + s * (sa5 + s * (sa6 + s * (sa7 + s * sa8)))))));
            } else {    /* |x| >= 1/0.35 */
                R = rb0 + s * (rb1 + s * (rb2 + s * (rb3 + s * (rb4 + s * (rb5 + s * rb6)))));
                S = one + s * (sb1 + s * (sb2 + s * (sb3 + s * (sb4 + s * (sb5 + s * (sb6 + s * sb7))))));
            }
            r = std::exp(-ax * ax - 0.5625 + R / S);
            if (x >= Real(0.0)) return one - r / ax; else return r / ax - one;
        }

      private:
        // Constants from QuantLib
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
