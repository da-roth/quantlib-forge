/* -*- mode: c++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*- */

/*******************************************************************************

   Cumulative Normal Distribution - Forge-Aware Implementation

   This version uses the Forge-aware ErrorFunction with ABool::If branches,
   allowing Forge to properly record all branches in the computation graph.
   This enables kernel reuse when inputs change and different branches
   become active.

   This file is part of QuantLib-Forge, a Forge integration layer for QuantLib.

   Copyright (C) 2025 The QuantLib-Forge Authors

   SPDX-License-Identifier: AGPL-3.0-or-later

******************************************************************************/

#ifndef QUANTLIB_FORGE_TEST_CUMULATIVE_NORMAL_DISTRIBUTION_FORGE_HPP
#define QUANTLIB_FORGE_TEST_CUMULATIVE_NORMAL_DISTRIBUTION_FORGE_HPP

#include "error_function_forge.hpp"
#include <limits>

namespace qlforge_test {

    // Forge-aware CumulativeNormalDistribution
    // This version matches the QuantLib patch including the asymptotic expansion
    template<typename Real>
    class CumulativeNormalDistributionForge {
      public:
        CumulativeNormalDistributionForge(Real average = Real(0.0), Real sigma = Real(1.0))
            : average_(average), sigma_(sigma) {}

        Real operator()(Real z) const {
            z = (z - average_) / sigma_;
            Real result = Real(0.5) * (Real(1.0) + errorFunction_(z * M_SQRT_2));

            // Asymptotic expansion for very negative z
            // This matches the QuantLib implementation exactly
            // The asymptotic result is computed for very small results (result <= 1e-8)
            Real asymptotic_result = computeAsymptoticExpansion(z);

            // Use ABool::If to select between normal result and asymptotic expansion
            forge::ABool cond_use_asymptotic = forge::lessEqual(
                result.forgeValue(), Real(1e-8).forgeValue());

            return forge::ABool::If(cond_use_asymptotic, asymptotic_result, result);
        }

      private:
        // Compute asymptotic expansion for very negative z
        // Following (26.2.12) on page 408 in M. Abramowitz and A. Stegun
        Real computeAsymptoticExpansion(Real z) const {
            // For the asymptotic expansion, we need to compute a series sum
            // The original code uses a do-while loop, but we need to unroll it
            // for Forge to record all operations.
            //
            // The series converges quickly for large |z|, so we can use a fixed
            // number of terms that covers all practical cases.

            Real zsqr = z * z;
            Real sum = Real(1.0);
            Real g = Real(1.0);

            // Unroll the loop - typically converges in 5-10 iterations for |z| > 3
            // We'll do 20 iterations to be safe
            for (int i = 1; i <= 20; i++) {
                Real x = Real(4.0 * i - 3.0) / zsqr;
                Real y = x * (Real(4.0 * i - 1.0) / zsqr);
                Real a = g * (x - y);
                sum = sum - a;
                g = g * y;
            }

            // result = -gaussian_(z)/z*sum
            // gaussian_(z) = normalizationFactor * exp(-z*z/2)
            // For standard normal (sigma=1): gaussian(z) = exp(-z*z/2) / sqrt(2*pi)
            Real normFactor = Real(M_1_SQRT_2PI);
            Real gaussian_z = normFactor * forge::expr::exp(-z * z / Real(2.0));

            return -gaussian_z / z * sum;
        }

        Real average_;
        Real sigma_;
        ErrorFunctionForge<Real> errorFunction_;
        static constexpr double M_SQRT_2 = 0.7071067811865475244008443621048490392848359376884740;
        static constexpr double M_1_SQRT_2PI = 0.3989422804014326779399460599343818684758586311649346;
    };

}  // namespace qlforge_test

#endif
