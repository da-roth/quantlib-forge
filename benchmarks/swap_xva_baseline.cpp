/*******************************************************************************

   XVA Performance Benchmark - Baseline Version (Plain QuantLib, Finite Differences)

   This file is part of QuantLib-Forge, a Forge integration layer for QuantLib.

   Copyright (C) 2025 The QuantLib-Forge Authors

   Derived from QuantLib-Risks-Cpp / XAD (https://github.com/auto-differentiation/XAD)
   Original XAD code: Copyright (C) 2010-2024 Xcelerit Computing Ltd.
   Original QuantLib code: Copyright (C) 2003-2007 Ferdinando Ametrano,
                           StatPro Italia srl, Joseph Wang

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

// =============================================================================
// XVA PERFORMANCE TEST - BASELINE VERSION (Plain QuantLib, Finite Differences)
// =============================================================================
// This test runs ONLY the Bump-Reval approach:
//   - Direct QuantLib evaluation with plain double
//   - Greeks computed via finite differences (bump each input, recompute NPV)
//
// Test Cases:
//   1. 1 swap, 1 step, 1 path, 10 risk factors (EUR curve only)
//   2. 1 swap, 1 step, 1 path, 100 risk factors (full XVA)
//   3. 1 swap, 3 steps, 1 path, 100 risk factors
//   4. 1 swap, 3 steps, 10 paths, 100 risk factors
//   5. 1 swap, 3 steps, 100 paths, 100 risk factors
//   6. 1 swap, 3 steps, 1000 paths, 100 risk factors
//
// This serves as the baseline for comparing AAD approaches (XAD, Forge).
// =============================================================================

#include <ql/qldefines.hpp>
#include <ql/cashflows/couponpricer.hpp>
#include <ql/cashflows/fixedratecoupon.hpp>
#include <ql/cashflows/iborcoupon.hpp>
#include <ql/currencies/europe.hpp>
#include <ql/indexes/ibor/euribor.hpp>
#include <ql/instruments/vanillaswap.hpp>
#include <ql/pricingengines/swap/discountingswapengine.hpp>
#include <ql/termstructures/volatility/optionlet/constantoptionletvol.hpp>
#include <ql/termstructures/yield/flatforward.hpp>
#include <ql/termstructures/yield/zerocurve.hpp>
#include <ql/time/calendars/target.hpp>
#include <ql/time/daycounters/actual365fixed.hpp>
#include <ql/settings.hpp>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>
#include <algorithm>
#include <cmath>

using namespace QuantLib;

namespace {

    //=========================================================================
    // IR Curve Pillar Definition
    //=========================================================================
    struct IRPillar {
        std::string name;
        Period tenor;
        double baseRate;
        double volatility;
    };

    //=========================================================================
    // Standard IR curve pillars (typical XVA setup)
    //=========================================================================
    std::vector<IRPillar> createEURCurvePillars() {
        return {
            {"EUR_6M",  Period(6, Months),  0.0320, 0.0060},
            {"EUR_1Y",  Period(1, Years),   0.0335, 0.0055},
            {"EUR_2Y",  Period(2, Years),   0.0348, 0.0050},
            {"EUR_3Y",  Period(3, Years),   0.0358, 0.0048},
            {"EUR_5Y",  Period(5, Years),   0.0375, 0.0045},
            {"EUR_7Y",  Period(7, Years),   0.0388, 0.0043},
            {"EUR_10Y", Period(10, Years),  0.0402, 0.0040},
            {"EUR_15Y", Period(15, Years),  0.0415, 0.0038},
            {"EUR_20Y", Period(20, Years),  0.0422, 0.0035},
            {"EUR_30Y", Period(30, Years),  0.0428, 0.0032}
        };
    }

    //=========================================================================
    // Additional risk factors for XVA (FX, credit, other curves)
    //=========================================================================
    struct AdditionalRiskFactors {
        std::vector<double> usdRates;
        std::vector<double> gbpRates;
        std::vector<double> jpyRates;
        std::vector<double> chfRates;
        std::vector<double> fxRates;
        std::vector<double> counterpartySpreads;
        std::vector<double> ownSpreads;
        std::vector<double> volSurface;
    };

    AdditionalRiskFactors createBaseRiskFactors() {
        AdditionalRiskFactors rf;
        rf.usdRates = {0.0480, 0.0495, 0.0505, 0.0512, 0.0525, 0.0535, 0.0545, 0.0555, 0.0560, 0.0565};
        rf.gbpRates = {0.0420, 0.0435, 0.0448, 0.0458, 0.0470, 0.0480, 0.0490, 0.0500, 0.0505, 0.0510};
        rf.jpyRates = {-0.001, 0.000, 0.002, 0.004, 0.008, 0.012, 0.018, 0.022, 0.025, 0.028};
        rf.chfRates = {0.010, 0.012, 0.015, 0.018, 0.022, 0.026, 0.030, 0.034, 0.037, 0.040};
        rf.fxRates = {1.08, 1.27, 149.5, 0.88, 0.85};
        rf.counterpartySpreads = {0.0050, 0.0055, 0.0062, 0.0070, 0.0080, 0.0092, 0.0105, 0.0120, 0.0135, 0.0150};
        rf.ownSpreads = {0.0030, 0.0033, 0.0038, 0.0044, 0.0052, 0.0060, 0.0070, 0.0082, 0.0095, 0.0110};
        rf.volSurface = {
            0.20, 0.19, 0.18, 0.19, 0.21,
            0.19, 0.18, 0.17, 0.18, 0.20,
            0.18, 0.17, 0.16, 0.17, 0.19,
            0.17, 0.16, 0.15, 0.16, 0.18,
            0.16, 0.15, 0.14, 0.15, 0.17
        };
        return rf;
    }

    //=========================================================================
    // Configuration
    //=========================================================================
    struct XvaConfig {
        std::string name;
        Size numSwaps = 1;
        Size numTimeSteps = 1;
        Size numPaths = 1;
        Size numRiskFactors = 10;
        Size warmupRuns = 2;
        Size timedRuns = 5;
        double bumpSize = 1e-4;
    };

    //=========================================================================
    // Swap Definition
    //=========================================================================
    struct SwapDefinition {
        Integer tenorYears;
        Period fixedFreq;
        Period floatFreq;
        double notional;
        double fixedRate;
        double spread;
    };

    //=========================================================================
    // Market Scenario
    //=========================================================================
    struct MarketScenario {
        std::vector<double> flatData;

        const std::vector<double>& flatten() const { return flatData; }
    };

    //=========================================================================
    // Results
    //=========================================================================
    struct XvaResults {
        std::vector<std::vector<std::vector<double>>> exposures;
        std::vector<std::vector<std::vector<std::vector<double>>>> sensitivities;
        double expectedExposure = 0.0;
        double cva = 0.0;
    };

    //=========================================================================
    // Timing Results
    //=========================================================================
    struct TimingResults {
        double totalTimeMs = 0.0;
        double avgTimePerIterationMs = 0.0;
        double evaluationTimeMs = 0.0;
        double singleScenarioTimeUs = 0.0;
        Size numScenarios = 0;
        Size numEvaluations = 0;
    };

    //=========================================================================
    // Generate scenarios with specified number of risk factors
    //=========================================================================
    std::vector<std::vector<MarketScenario>> generateScenarios(
        const XvaConfig& config,
        const std::vector<IRPillar>& eurPillars,
        const AdditionalRiskFactors& baseFactors,
        unsigned int seed = 42) {

        std::mt19937 gen(seed);
        std::normal_distribution<> dist(0.0, 1.0);

        std::vector<std::vector<MarketScenario>> scenarios(config.numTimeSteps);

        double rateVol = 0.005;
        double fxVol = 0.10;
        double creditVol = 0.20;
        double volVol = 0.30;

        for (Size t = 0; t < config.numTimeSteps; ++t) {
            scenarios[t].resize(config.numPaths);
            double timeYears = config.numTimeSteps > 1 ? double(t + 1) / config.numTimeSteps * 5.0 : 0.0;
            double sqrtTime = std::sqrt(std::max(timeYears, 0.01));

            for (Size p = 0; p < config.numPaths; ++p) {
                MarketScenario& sc = scenarios[t][p];
                sc.flatData.resize(config.numRiskFactors);
                Size idx = 0;

                // EUR curve (first 10 risk factors)
                double eurParallel = dist(gen);
                for (Size i = 0; i < std::min(Size(10), config.numRiskFactors); ++i) {
                    sc.flatData[idx++] = eurPillars[i].baseRate + eurPillars[i].volatility * eurParallel * sqrtTime;
                }

                if (config.numRiskFactors <= 10) continue;

                // USD curve (10-19)
                double usdParallel = dist(gen);
                for (Size i = 0; i < 10 && idx < config.numRiskFactors; ++i) {
                    sc.flatData[idx++] = baseFactors.usdRates[i] + rateVol * usdParallel * sqrtTime;
                }

                // GBP curve (20-29)
                double gbpParallel = dist(gen);
                for (Size i = 0; i < 10 && idx < config.numRiskFactors; ++i) {
                    sc.flatData[idx++] = baseFactors.gbpRates[i] + rateVol * gbpParallel * sqrtTime;
                }

                // JPY curve (30-39)
                double jpyParallel = dist(gen);
                for (Size i = 0; i < 10 && idx < config.numRiskFactors; ++i) {
                    sc.flatData[idx++] = baseFactors.jpyRates[i] + rateVol * jpyParallel * sqrtTime;
                }

                // CHF curve (40-49)
                double chfParallel = dist(gen);
                for (Size i = 0; i < 10 && idx < config.numRiskFactors; ++i) {
                    sc.flatData[idx++] = baseFactors.chfRates[i] + rateVol * chfParallel * sqrtTime;
                }

                // FX rates (50-54)
                for (Size i = 0; i < 5 && idx < config.numRiskFactors; ++i) {
                    double fxShock = dist(gen);
                    sc.flatData[idx++] = baseFactors.fxRates[i] * std::exp(fxVol * fxShock * sqrtTime - 0.5 * fxVol * fxVol * timeYears);
                }

                // Counterparty credit spreads (55-64)
                double cptyShock = dist(gen);
                for (Size i = 0; i < 10 && idx < config.numRiskFactors; ++i) {
                    sc.flatData[idx++] = baseFactors.counterpartySpreads[i] * std::exp(creditVol * cptyShock * sqrtTime);
                }

                // Own credit spreads (65-74)
                double ownShock = dist(gen);
                for (Size i = 0; i < 10 && idx < config.numRiskFactors; ++i) {
                    sc.flatData[idx++] = baseFactors.ownSpreads[i] * std::exp(creditVol * ownShock * sqrtTime);
                }

                // Vol surface (75-99)
                double volShock = dist(gen);
                for (Size i = 0; i < 25 && idx < config.numRiskFactors; ++i) {
                    sc.flatData[idx++] = baseFactors.volSurface[i] * std::exp(volVol * volShock * sqrtTime);
                }
            }
        }
        return scenarios;
    }

    //=========================================================================
    // Create swap definitions
    //=========================================================================
    std::vector<SwapDefinition> createSwapDefinitions(Size numSwaps) {
        std::vector<SwapDefinition> swaps;
        swaps.push_back({5, Period(1, Years), Period(6, Months), 1000000.0, 0.03, 0.001});
        if (numSwaps > 1) {
            swaps.push_back({10, Period(6, Months), Period(3, Months), 2000000.0, 0.035, 0.0015});
        }
        return swaps;
    }

    //=========================================================================
    // Price swap - 10 risk factors (EUR curve only)
    //=========================================================================
    double priceSwap10RF(
        const SwapDefinition& swapDef,
        Size timeStep,
        Size totalTimeSteps,
        const std::vector<double>& allInputs,
        const std::vector<IRPillar>& eurPillarDefs,
        const Date& today,
        const Calendar& calendar,
        const DayCounter& dayCounter) {

        double timeStepFraction = totalTimeSteps > 1 ? double(timeStep) / totalTimeSteps : 0.0;
        Integer elapsedYears = Integer(timeStepFraction * swapDef.tenorYears);
        Integer remainingYears = swapDef.tenorYears - elapsedYears;

        if (remainingYears <= 0) {
            return 0.0;
        }

        // Build EUR discount curve
        std::vector<Date> curveDates;
        std::vector<Real> curveRates;
        curveDates.push_back(today);
        curveRates.push_back(allInputs[0]);

        for (Size i = 0; i < 10 && i < eurPillarDefs.size(); ++i) {
            curveDates.push_back(calendar.advance(today, eurPillarDefs[i].tenor));
            curveRates.push_back(allInputs[i]);
        }

        RelinkableHandle<YieldTermStructure> termStructure;
        auto zeroCurve = ext::make_shared<ZeroCurve>(curveDates, curveRates, dayCounter);
        zeroCurve->enableExtrapolation();
        termStructure.linkTo(zeroCurve);

        auto index = ext::make_shared<Euribor6M>(termStructure);
        Date start = calendar.advance(today, index->fixingDays(), Days);
        Date maturity = calendar.advance(start, remainingYears, Years);

        Schedule fixedSchedule(start, maturity, swapDef.fixedFreq, calendar,
                               ModifiedFollowing, ModifiedFollowing,
                               DateGeneration::Forward, false);
        Schedule floatSchedule(start, maturity, swapDef.floatFreq, calendar,
                               ModifiedFollowing, ModifiedFollowing,
                               DateGeneration::Forward, false);

        auto swap = ext::make_shared<VanillaSwap>(
            VanillaSwap::Payer, swapDef.notional, fixedSchedule, swapDef.fixedRate,
            dayCounter, floatSchedule, index, swapDef.spread, dayCounter);

        swap->setPricingEngine(ext::make_shared<DiscountingSwapEngine>(termStructure));
        return swap->NPV();
    }

    //=========================================================================
    // Price swap - 100 risk factors (full XVA)
    //=========================================================================
    double priceSwap100RF(
        const SwapDefinition& swapDef,
        Size timeStep,
        Size totalTimeSteps,
        const std::vector<double>& allInputs,
        const std::vector<IRPillar>& eurPillarDefs,
        const Date& today,
        const Calendar& calendar,
        const DayCounter& dayCounter) {

        double timeStepFraction = totalTimeSteps > 1 ? double(timeStep) / totalTimeSteps : 0.0;
        Integer elapsedYears = Integer(timeStepFraction * swapDef.tenorYears);
        Integer remainingYears = swapDef.tenorYears - elapsedYears;

        if (remainingYears <= 0) {
            return 0.0;
        }

        // Build EUR discount curve
        std::vector<Date> curveDates;
        std::vector<Real> curveRates;
        curveDates.push_back(today);
        curveRates.push_back(allInputs[0]);

        for (Size i = 0; i < 10 && i < eurPillarDefs.size(); ++i) {
            curveDates.push_back(calendar.advance(today, eurPillarDefs[i].tenor));
            curveRates.push_back(allInputs[i]);
        }

        RelinkableHandle<YieldTermStructure> termStructure;
        auto zeroCurve = ext::make_shared<ZeroCurve>(curveDates, curveRates, dayCounter);
        zeroCurve->enableExtrapolation();
        termStructure.linkTo(zeroCurve);

        auto index = ext::make_shared<Euribor6M>(termStructure);
        Date start = calendar.advance(today, index->fixingDays(), Days);
        Date maturity = calendar.advance(start, remainingYears, Years);

        Schedule fixedSchedule(start, maturity, swapDef.fixedFreq, calendar,
                               ModifiedFollowing, ModifiedFollowing,
                               DateGeneration::Forward, false);
        Schedule floatSchedule(start, maturity, swapDef.floatFreq, calendar,
                               ModifiedFollowing, ModifiedFollowing,
                               DateGeneration::Forward, false);

        auto swap = ext::make_shared<VanillaSwap>(
            VanillaSwap::Payer, swapDef.notional, fixedSchedule, swapDef.fixedRate,
            dayCounter, floatSchedule, index, swapDef.spread, dayCounter);

        swap->setPricingEngine(ext::make_shared<DiscountingSwapEngine>(termStructure));
        double baseNpv = swap->NPV();

        // XVA adjustments using remaining 90 risk factors
        std::vector<double> eurRates(allInputs.begin(), allInputs.begin() + 10);
        std::vector<double> usdRates(allInputs.begin() + 10, allInputs.begin() + 20);
        std::vector<double> gbpRates(allInputs.begin() + 20, allInputs.begin() + 30);
        std::vector<double> jpyRates(allInputs.begin() + 30, allInputs.begin() + 40);
        std::vector<double> chfRates(allInputs.begin() + 40, allInputs.begin() + 50);
        std::vector<double> fxRates(allInputs.begin() + 50, allInputs.begin() + 55);
        std::vector<double> counterpartySpreads(allInputs.begin() + 55, allInputs.begin() + 65);
        std::vector<double> ownSpreads(allInputs.begin() + 65, allInputs.begin() + 75);
        std::vector<double> volSurface(allInputs.begin() + 75, allInputs.begin() + 100);

        // Cross-currency basis adjustment
        double ccyBasisAdj = 0.0;
        double eurusd = fxRates[0];
        double eurgbp = fxRates[4];
        for (Size i = 0; i < 10; ++i) {
            ccyBasisAdj += (usdRates[i] - eurRates[i]) * eurusd * 0.0001 * swapDef.notional;
            ccyBasisAdj += (gbpRates[i] - eurRates[i]) * eurgbp * 0.00005 * swapDef.notional;
            ccyBasisAdj += jpyRates[i] * 0.00001 * swapDef.notional;
            ccyBasisAdj += chfRates[i] * 0.00002 * swapDef.notional;
        }

        // CVA/DVA adjustment
        double lgd = 0.4;
        double exposure = baseNpv > 0.0 ? baseNpv : 0.0;
        double negExposure = baseNpv < 0.0 ? -baseNpv : 0.0;

        double cvaAdj = 0.0;
        double dvaAdj = 0.0;
        for (Size i = 0; i < 10; ++i) {
            cvaAdj -= lgd * exposure * counterpartySpreads[i] * 0.1;
            dvaAdj += lgd * negExposure * ownSpreads[i] * 0.1;
        }

        // Volatility adjustment
        double volAdj = 0.0;
        for (Size i = 0; i < 25; ++i) {
            volAdj += volSurface[i] * 0.001 * swapDef.notional;
        }

        return baseNpv + ccyBasisAdj + cvaAdj + dvaAdj + volAdj;
    }

    //=========================================================================
    // Price swap dispatcher based on number of risk factors
    //=========================================================================
    double priceSwap(
        const SwapDefinition& swapDef,
        Size timeStep,
        Size totalTimeSteps,
        const std::vector<double>& allInputs,
        const std::vector<IRPillar>& eurPillarDefs,
        const Date& today,
        const Calendar& calendar,
        const DayCounter& dayCounter,
        Size numRiskFactors) {

        if (numRiskFactors <= 10) {
            return priceSwap10RF(swapDef, timeStep, totalTimeSteps, allInputs, eurPillarDefs, today, calendar, dayCounter);
        } else {
            return priceSwap100RF(swapDef, timeStep, totalTimeSteps, allInputs, eurPillarDefs, today, calendar, dayCounter);
        }
    }

    //=========================================================================
    // BUMP-REVAL COMPUTATION
    //=========================================================================
    XvaResults computeBumpRevalXva(
        const XvaConfig& config,
        const std::vector<SwapDefinition>& swaps,
        const std::vector<std::vector<MarketScenario>>& scenarios,
        const std::vector<IRPillar>& pillars,
        const Date& today,
        const Calendar& calendar,
        const DayCounter& dayCounter,
        TimingResults& timing) {

        XvaResults results;
        results.exposures.resize(config.numSwaps);
        results.sensitivities.resize(config.numSwaps);

        double totalExposure = 0.0;
        Size totalEvaluations = 0;
        Size numScenarios = 0;

        auto startTime = std::chrono::high_resolution_clock::now();

        for (Size s = 0; s < config.numSwaps; ++s) {
            results.exposures[s].resize(config.numTimeSteps);
            results.sensitivities[s].resize(config.numTimeSteps);

            for (Size t = 0; t < config.numTimeSteps; ++t) {
                results.exposures[s][t].resize(config.numPaths);
                results.sensitivities[s][t].resize(config.numPaths);

                for (Size p = 0; p < config.numPaths; ++p) {
                    const auto& scenario = scenarios[t][p];
                    std::vector<double> flatInputs = scenario.flatten();

                    // Base pricing
                    double baseNpv = priceSwap(
                        swaps[s], t, config.numTimeSteps, flatInputs, pillars, today, calendar, dayCounter, config.numRiskFactors);
                    totalEvaluations++;

                    results.exposures[s][t][p] = std::max(0.0, baseNpv);
                    totalExposure += results.exposures[s][t][p];

                    // Bump each input (finite differences)
                    results.sensitivities[s][t][p].resize(config.numRiskFactors);
                    for (Size i = 0; i < config.numRiskFactors; ++i) {
                        std::vector<double> bumpedInputs = flatInputs;
                        bumpedInputs[i] += config.bumpSize;
                        double bumpedNpv = priceSwap(
                            swaps[s], t, config.numTimeSteps, bumpedInputs, pillars, today, calendar, dayCounter, config.numRiskFactors);
                        totalEvaluations++;
                        results.sensitivities[s][t][p][i] = (bumpedNpv - baseNpv) / config.bumpSize;
                    }
                    numScenarios++;
                }
            }
        }

        auto endTime = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime);

        timing.numScenarios = numScenarios;
        timing.numEvaluations = totalEvaluations;
        timing.evaluationTimeMs = duration.count() / 1000.0;
        timing.singleScenarioTimeUs = numScenarios > 0 ? double(duration.count()) / numScenarios : 0.0;

        Size totalScenarios = config.numSwaps * config.numTimeSteps * config.numPaths;
        results.expectedExposure = totalScenarios > 0 ? totalExposure / totalScenarios : 0.0;
        results.cva = results.expectedExposure * 0.4 * 0.02;

        return results;
    }

    //=========================================================================
    // Run with timing
    //=========================================================================
    TimingResults runWithTiming(
        const XvaConfig& config,
        const std::vector<SwapDefinition>& swaps,
        const std::vector<std::vector<MarketScenario>>& scenarios,
        const std::vector<IRPillar>& pillars,
        const Date& today,
        const Calendar& calendar,
        const DayCounter& dayCounter,
        XvaResults& lastResults) {

        TimingResults timing;
        TimingResults warmupTiming;

        for (Size w = 0; w < config.warmupRuns; ++w) {
            computeBumpRevalXva(config, swaps, scenarios, pillars, today, calendar, dayCounter, warmupTiming);
        }

        TimingResults accumulatedTiming;
        auto startTime = std::chrono::high_resolution_clock::now();

        for (Size r = 0; r < config.timedRuns; ++r) {
            TimingResults iterTiming;
            lastResults = computeBumpRevalXva(config, swaps, scenarios, pillars, today, calendar, dayCounter, iterTiming);
            accumulatedTiming.evaluationTimeMs += iterTiming.evaluationTimeMs;
            accumulatedTiming.numScenarios += iterTiming.numScenarios;
            accumulatedTiming.numEvaluations += iterTiming.numEvaluations;
        }

        auto endTime = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime);

        timing.totalTimeMs = duration.count() / 1000.0;
        timing.avgTimePerIterationMs = timing.totalTimeMs / config.timedRuns;
        timing.evaluationTimeMs = accumulatedTiming.evaluationTimeMs / config.timedRuns;
        timing.numScenarios = accumulatedTiming.numScenarios / config.timedRuns;
        timing.numEvaluations = accumulatedTiming.numEvaluations / config.timedRuns;
        timing.singleScenarioTimeUs = accumulatedTiming.numScenarios > 0
            ? (accumulatedTiming.evaluationTimeMs * 1000.0) / accumulatedTiming.numScenarios
            : 0.0;

        return timing;
    }

    //=========================================================================
    // Print results table
    //=========================================================================
    void printResultsTable(
        const std::string& testName,
        const TimingResults& timing,
        const XvaResults& results,
        const std::vector<IRPillar>& eurPillars,
        const XvaConfig& config) {

        const int col1 = 28;
        const int col2 = 20;

        auto line = [&]() {
            std::cout << "+" << std::string(col1, '-') << "+"
                      << std::string(col2, '-') << "+\n";
        };

        std::cout << "\n";
        line();
        std::cout << "|" << std::setw(col1 + col2 + 1) << std::left
                  << (" " + testName) << "|\n";
        line();
        std::cout << "| Config: " << config.numSwaps << " swap, "
                  << config.numTimeSteps << " step" << (config.numTimeSteps > 1 ? "s" : "") << ", "
                  << config.numPaths << " path" << (config.numPaths > 1 ? "s" : "") << ", "
                  << config.numRiskFactors << " RF";
        int padding = col1 + col2 - 24 - std::to_string(config.numSwaps).length()
                      - std::to_string(config.numTimeSteps).length()
                      - std::to_string(config.numPaths).length()
                      - std::to_string(config.numRiskFactors).length();
        std::cout << std::string(std::max(1, padding), ' ') << "|\n";
        line();

        std::cout << "|" << std::setw(col1) << std::left << " Metric"
                  << "|" << std::setw(col2) << std::right << "Value" << "|\n";
        line();

        std::cout << std::fixed << std::setprecision(2);

        std::cout << "|" << std::setw(col1) << std::left << " Total Time (ms)"
                  << "|" << std::setw(col2) << std::right << timing.totalTimeMs << "|\n";

        std::cout << "|" << std::setw(col1) << std::left << " Avg Time/Iteration (ms)"
                  << "|" << std::setw(col2) << std::right << timing.avgTimePerIterationMs << "|\n";

        std::cout << "|" << std::setw(col1) << std::left << " # Scenarios"
                  << "|" << std::setw(col2) << std::right << timing.numScenarios << "|\n";

        std::cout << "|" << std::setw(col1) << std::left << " # Evaluations"
                  << "|" << std::setw(col2) << std::right << timing.numEvaluations << "|\n";

        std::cout << "|" << std::setw(col1) << std::left << " Evals/Scenario"
                  << "|" << std::setw(col2) << std::right << (1 + config.numRiskFactors) << "|\n";

        std::cout << "|" << std::setw(col1) << std::left << " Time/Scenario (us)"
                  << "|" << std::setw(col2) << std::right << timing.singleScenarioTimeUs << "|\n";
        line();

        std::cout << std::scientific << std::setprecision(4);
        std::cout << "|" << std::setw(col1) << std::left << " Expected Exposure"
                  << "|" << std::setw(col2) << std::right << results.expectedExposure << "|\n";
        std::cout << "|" << std::setw(col1) << std::left << " CVA"
                  << "|" << std::setw(col2) << std::right << results.cva << "|\n";
        line();

        // Sample sensitivities
        Size numSensToShow = std::min(Size(5), config.numRiskFactors);
        if (numSensToShow > 0 && !results.sensitivities.empty() &&
            !results.sensitivities[0].empty() &&
            !results.sensitivities[0][0].empty()) {
            std::cout << "| Sample Sensitivities (Swap 0, Time 0, Path 0):"
                      << std::string(col1 + col2 - 45, ' ') << "|\n";
            line();

            for (Size i = 0; i < numSensToShow && i < eurPillars.size(); ++i) {
                std::string label = " dNPV/d(" + eurPillars[i].name + ")";
                std::cout << "|" << std::setw(col1) << std::left << label
                          << "|" << std::setw(col2) << std::right << results.sensitivities[0][0][0][i] << "|\n";
            }
            if (config.numRiskFactors > numSensToShow) {
                std::string moreLabel = " ... (" + std::to_string(config.numRiskFactors - numSensToShow) + " more)";
                std::cout << "|" << std::setw(col1) << std::left << moreLabel
                          << "|" << std::setw(col2) << " " << "|\n";
            }
            line();
        }
        std::cout << "\n";
    }

    //=========================================================================
    // Create test case configurations
    //=========================================================================
    std::vector<XvaConfig> createTestCases() {
        std::vector<XvaConfig> configs;

        // Test 1: Simple pricing - 10 risk factors (EUR curve only)
        configs.push_back({"Test 1: Simple (10 RF)", 1, 1, 1, 10, 2, 5, 1e-4});

        // Test 2: Full XVA risk factors - 100 RF, single scenario
        configs.push_back({"Test 2: Full RF (100 RF)", 1, 1, 1, 100, 2, 5, 1e-4});

        // Test 3: Add time steps
        configs.push_back({"Test 3: Time Steps (3 steps)", 1, 3, 1, 100, 2, 5, 1e-4});

        // Test 4: Add MC paths (10)
        configs.push_back({"Test 4: MC 10 paths", 1, 3, 10, 100, 2, 5, 1e-4});

        // Test 5: Add MC paths (100)
        configs.push_back({"Test 5: MC 100 paths", 1, 3, 100, 100, 2, 5, 1e-4});

        // Test 6: Full scale (1000 paths)
        configs.push_back({"Test 6: Full Scale (1000 paths)", 1, 3, 1000, 100, 2, 5, 1e-4});

        return configs;
    }

}  // namespace

//=============================================================================
// MAIN: XVA Baseline Benchmark (Bump-Reval with Finite Differences)
//=============================================================================
int main(int argc, char* argv[]) {
    std::cout << "=============================================================\n";
    std::cout << "  XVA Baseline Benchmark - Plain QuantLib (Finite Differences)\n";
    std::cout << "=============================================================\n";

    try {
        Calendar calendar = TARGET();
        Date today = Date(15, January, 2024);
        Settings::instance().evaluationDate() = today;
        DayCounter dayCounter = Actual365Fixed();

        auto eurPillars = createEURCurvePillars();
        auto baseRiskFactors = createBaseRiskFactors();
        auto testCases = createTestCases();

        for (const auto& config : testCases) {
            std::cout << "\nRunning " << config.name << "...\n";

            auto swaps = createSwapDefinitions(config.numSwaps);
            auto scenarios = generateScenarios(config, eurPillars, baseRiskFactors);

            XvaResults results;
            auto timing = runWithTiming(config, swaps, scenarios, eurPillars, today, calendar, dayCounter, results);

            printResultsTable(config.name, timing, results, eurPillars, config);
        }

        std::cout << "=============================================================\n";
        std::cout << "  Baseline benchmark completed successfully.\n";
        std::cout << "=============================================================\n";
        return 0;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
