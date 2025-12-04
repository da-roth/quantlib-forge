/* -*- mode: c++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*- */

/*******************************************************************************

   Cumulative Normal Distribution - Original QuantLib Implementation

   This is a standalone copy of QuantLib's CumulativeNormalDistribution
   for testing Forge kernel reuse without needing to build full QuantLib.

   This file is part of QuantLib-Forge, a Forge integration layer for QuantLib.

   Copyright (C) 2025 The QuantLib-Forge Authors

   SPDX-License-Identifier: AGPL-3.0-or-later

******************************************************************************/

#ifndef QUANTLIB_FORGE_TEST_CUMULATIVE_NORMAL_DISTRIBUTION_HPP
#define QUANTLIB_FORGE_TEST_CUMULATIVE_NORMAL_DISTRIBUTION_HPP

#include "error_function.hpp"

namespace qlforge_test {

    // Original CumulativeNormalDistribution using ErrorFunction
    template<typename Real>
    class CumulativeNormalDistribution {
      public:
        CumulativeNormalDistribution(Real average = Real(0.0), Real sigma = Real(1.0))
            : average_(average), sigma_(sigma) {}

        Real operator()(Real z) const {
            z = (z - average_) / sigma_;
            Real result = Real(0.5) * (Real(1.0) + errorFunction_(z * M_SQRT_2));
            return result;
        }

      private:
        Real average_;
        Real sigma_;
        ErrorFunction<Real> errorFunction_;
        static constexpr double M_SQRT_2 = 0.7071067811865475244008443621048490392848359376884740;
    };

}  // namespace qlforge_test

#endif
