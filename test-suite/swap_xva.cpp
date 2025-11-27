/* -*- mode: c++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*- */

/*******************************************************************************

   XVA Performance Test - Forge Version

   This file is part of QuantLib-Forge, a Forge integration layer for QuantLib.

   Copyright (C) 2025 The QuantLib-Forge Authors
   Copyright (C) 2003, 2004 Ferdinando Ametrano
   Copyright (C) 2005, 2007 StatPro Italia srl
   Copyright (C) 2005 Joseph Wang

   Derived from QuantLib-Risks-Cpp / XAD (https://github.com/auto-differentiation/XAD)
   Original XAD code: Copyright (C) 2010-2024 Xcelerit Computing Ltd.

   SPDX-License-Identifier: AGPL-3.0-or-later

******************************************************************************/

// =============================================================================
// XVA PERFORMANCE TEST - FORGE VERSION
// =============================================================================
// This test compares THREE approaches:
//   1. Bump-Reval (baseline)    - Direct QuantLib evaluation with finite differences
//   2. Forge Kernel Bump-Reval  - JIT-compiled kernel, still using bump-reval for sensitivities
//   3. Forge AAD                - JIT-compiled kernel with automatic differentiation (no bumping)
//
// Forge provides two distinct benefits:
//   A) JIT Compilation: Kernel can be re-evaluated with different inputs much faster
//   B) AAD: Gradients computed automatically in single backward pass (no bumping needed)
// =============================================================================

#include "toplevelfixture.hpp"
#include "utilities_forge.hpp"
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
#include <chrono>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

// Forge integration headers for recording + compilation
#include <graph/graph_recorder.hpp>
#include <compiler/forge_engine.hpp>
#include <compiler/node_value_buffers/node_value_buffer.hpp>

using namespace QuantLib;
using namespace boost::unit_test_framework;

BOOST_FIXTURE_TEST_SUITE(QuantLibForgeTests, TopLevelFixture)

BOOST_AUTO_TEST_SUITE(SwapXvaTests)

namespace {

    //=========================================================================
    // IR Curve Pillar Definition
    //=========================================================================
    struct IRPillar {
        std::string name;      // e.g., "6M", "1Y", "10Y"
        Period tenor;          // QuantLib period
        double baseRate;       // Base zero rate
        double volatility;     // Annual volatility for simulation
    };

    //=========================================================================
    // Standard IR curve pillars (typical XVA setup)
    // We create pillars for multiple currencies to get realistic derivative count
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
    // These are "dummy" inputs that affect pricing to demonstrate AAD scaling
    //=========================================================================
    struct AdditionalRiskFactors {
        // USD curve (10 pillars)
        std::vector<double> usdRates;
        // GBP curve (10 pillars)
        std::vector<double> gbpRates;
        // JPY curve (10 pillars)
        std::vector<double> jpyRates;
        // CHF curve (10 pillars)
        std::vector<double> chfRates;
        // FX rates (5 rates: EURUSD, GBPUSD, USDJPY, USDCHF, EURGBP)
        std::vector<double> fxRates;
        // Credit spreads (10 pillars for counterparty)
        std::vector<double> counterpartySpreads;
        // Credit spreads (10 pillars for own credit - DVA)
        std::vector<double> ownSpreads;
        // Volatility surface points (25 points: 5 expiries x 5 strikes)
        std::vector<double> volSurface;
    };

    AdditionalRiskFactors createBaseRiskFactors() {
        AdditionalRiskFactors rf;
        // USD curve - slightly higher rates
        rf.usdRates = {0.0480, 0.0495, 0.0505, 0.0512, 0.0525, 0.0535, 0.0545, 0.0555, 0.0560, 0.0565};
        // GBP curve
        rf.gbpRates = {0.0420, 0.0435, 0.0448, 0.0458, 0.0470, 0.0480, 0.0490, 0.0500, 0.0505, 0.0510};
        // JPY curve - low rates
        rf.jpyRates = {-0.001, 0.000, 0.002, 0.004, 0.008, 0.012, 0.018, 0.022, 0.025, 0.028};
        // CHF curve - low rates
        rf.chfRates = {0.010, 0.012, 0.015, 0.018, 0.022, 0.026, 0.030, 0.034, 0.037, 0.040};
        // FX rates
        rf.fxRates = {1.08, 1.27, 149.5, 0.88, 0.85};  // EURUSD, GBPUSD, USDJPY, USDCHF, EURGBP
        // Counterparty credit spreads (bps converted to decimal)
        rf.counterpartySpreads = {0.0050, 0.0055, 0.0062, 0.0070, 0.0080, 0.0092, 0.0105, 0.0120, 0.0135, 0.0150};
        // Own credit spreads
        rf.ownSpreads = {0.0030, 0.0033, 0.0038, 0.0044, 0.0052, 0.0060, 0.0070, 0.0082, 0.0095, 0.0110};
        // Vol surface (5x5 = 25 points)
        rf.volSurface = {
            0.20, 0.19, 0.18, 0.19, 0.21,  // 1M expiry
            0.19, 0.18, 0.17, 0.18, 0.20,  // 3M expiry
            0.18, 0.17, 0.16, 0.17, 0.19,  // 6M expiry
            0.17, 0.16, 0.15, 0.16, 0.18,  // 1Y expiry
            0.16, 0.15, 0.14, 0.15, 0.17   // 2Y expiry
        };
        return rf;
    }

    // Total risk factors: 10 EUR + 10 USD + 10 GBP + 10 JPY + 10 CHF + 5 FX + 10 CptyCredit + 10 OwnCredit + 25 Vol = 100

    //=========================================================================
    // XVA Test Configuration
    //=========================================================================
    struct XvaConfig {
        Size numSwaps = 1;           // Number of different swap structures
        Size numTimeSteps = 3;       // Time steps in simulation (reduced for testing)
        Size numPaths = 1000;          // MC paths per time step (reduced for testing)
        Size warmupRuns = 2;         // Warmup iterations (not timed)
        Size timedRuns = 5;          // Timed iterations
        double bumpSize = 1e-4;      // Bump size for finite differences (1bp)
    };

    //=========================================================================
    // Swap Definition - defines the structure of a swap
    //=========================================================================
    struct SwapDefinition {
        Integer tenorYears;          // Total tenor in years
        Period fixedFreq;            // Fixed leg frequency
        Period floatFreq;            // Floating leg frequency
        Real notional;               // Notional amount
        Rate fixedRate;              // Fixed rate
        Real spread;                 // Floating spread
    };

    //=========================================================================
    // Market Scenario - all risk factors for a single MC path/time
    // Total: 100 risk factors
    // Data stored pre-flattened for performance (avoid allocations in tight loops)
    //=========================================================================
    struct MarketScenario {
        std::vector<double> flatData;  // Pre-flattened: 100 doubles

        // Access the pre-flattened data (no allocation, just returns reference)
        const std::vector<double>& flatten() const { return flatData; }

        static constexpr Size totalRiskFactors() { return 100; }
    };

    //=========================================================================
    // XVA Results - stores results for comparison
    //=========================================================================
    struct XvaResults {
        std::vector<std::vector<std::vector<double>>> exposures;  // [swap][time][path]
        std::vector<std::vector<std::vector<std::vector<double>>>> sensitivities;  // [swap][time][path][pillar]
        double expectedExposure = 0.0;  // Aggregated EE
        double cva = 0.0;               // Simple CVA approximation
    };

    //=========================================================================
    // Timing Results
    //=========================================================================
    struct TimingResults {
        double totalTimeMs = 0.0;
        double avgTimePerIterationMs = 0.0;
        Size totalPricings = 0;
        Size totalSensitivities = 0;

        // Detailed breakdown (per iteration averages)
        double kernelCreationTimeMs = 0.0;    // Forge: recording + compilation time
        double evaluationTimeMs = 0.0;        // Time for all evaluations (excl. kernel creation)
        double singleEvalTimeUs = 0.0;        // Average time per single evaluation (microseconds)
        Size numKernelsCreated = 0;           // Number of kernels compiled (Forge only)
        Size numEvaluations = 0;              // Number of kernel executions / pricings
    };

    //=========================================================================
    // Generate simulated scenarios for all 100 risk factors
    //=========================================================================
    std::vector<std::vector<MarketScenario>> generateScenarios(
        const XvaConfig& config,
        const std::vector<IRPillar>& eurPillars,
        const AdditionalRiskFactors& baseRf,
        unsigned int seed = 42) {

        std::mt19937 gen(seed);
        std::normal_distribution<> dist(0.0, 1.0);

        std::vector<std::vector<MarketScenario>> scenarios(config.numTimeSteps);

        // Volatilities for simulation
        double rateVol = 0.005;   // 50bp annual vol for rates
        double fxVol = 0.10;      // 10% annual vol for FX
        double creditVol = 0.20;  // 20% relative vol for credit spreads
        double volVol = 0.30;     // 30% relative vol for vol surface

        for (Size t = 0; t < config.numTimeSteps; ++t) {
            scenarios[t].resize(config.numPaths);
            double timeYears = double(t + 1) / config.numTimeSteps * 5.0;
            double sqrtTime = std::sqrt(timeYears);

            for (Size p = 0; p < config.numPaths; ++p) {
                MarketScenario& sc = scenarios[t][p];

                // Pre-allocate flatData with all 100 risk factors
                sc.flatData.resize(100);
                Size idx = 0;

                // EUR curve (indices 0-9)
                double eurParallel = dist(gen);
                for (Size i = 0; i < eurPillars.size(); ++i) {
                    sc.flatData[idx++] = eurPillars[i].baseRate + eurPillars[i].volatility * eurParallel * sqrtTime;
                }

                // USD curve (indices 10-19)
                double usdParallel = dist(gen);
                for (Size i = 0; i < 10; ++i) {
                    sc.flatData[idx++] = baseRf.usdRates[i] + rateVol * usdParallel * sqrtTime;
                }

                // GBP curve (indices 20-29)
                double gbpParallel = dist(gen);
                for (Size i = 0; i < 10; ++i) {
                    sc.flatData[idx++] = baseRf.gbpRates[i] + rateVol * gbpParallel * sqrtTime;
                }

                // JPY curve (indices 30-39)
                double jpyParallel = dist(gen);
                for (Size i = 0; i < 10; ++i) {
                    sc.flatData[idx++] = baseRf.jpyRates[i] + rateVol * jpyParallel * sqrtTime;
                }

                // CHF curve (indices 40-49)
                double chfParallel = dist(gen);
                for (Size i = 0; i < 10; ++i) {
                    sc.flatData[idx++] = baseRf.chfRates[i] + rateVol * chfParallel * sqrtTime;
                }

                // FX rates (indices 50-54)
                for (Size i = 0; i < 5; ++i) {
                    double fxShock = dist(gen);
                    sc.flatData[idx++] = baseRf.fxRates[i] * std::exp(fxVol * fxShock * sqrtTime - 0.5 * fxVol * fxVol * timeYears);
                }

                // Counterparty credit spreads (indices 55-64)
                double cptyShock = dist(gen);
                for (Size i = 0; i < 10; ++i) {
                    sc.flatData[idx++] = baseRf.counterpartySpreads[i] * std::exp(creditVol * cptyShock * sqrtTime);
                }

                // Own credit spreads (indices 65-74)
                double ownShock = dist(gen);
                for (Size i = 0; i < 10; ++i) {
                    sc.flatData[idx++] = baseRf.ownSpreads[i] * std::exp(creditVol * ownShock * sqrtTime);
                }

                // Vol surface (indices 75-99)
                double volShock = dist(gen);
                for (Size i = 0; i < 25; ++i) {
                    sc.flatData[idx++] = baseRf.volSurface[i] * std::exp(volVol * volShock * sqrtTime);
                }
            }
        }
        return scenarios;
    }

    //=========================================================================
    // Create swap definitions with different structures
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
    // Build a swap as of a future time step using all 100 risk factors
    // The NPV depends on: EUR curve (primary), other curves and FX (cross-currency effects),
    // credit spreads (CVA/DVA adjustment), vol surface (optionality adjustment)
    //=========================================================================
    Real priceSwapAtTimeStep(
        const SwapDefinition& swapDef,
        Size timeStep,
        Size totalTimeSteps,
        const std::vector<Real>& allInputs,  // All 100 risk factors flattened
        const std::vector<IRPillar>& eurPillarDefs,
        const Date& today,
        const Calendar& calendar,
        const DayCounter& dayCounter) {

        double timeStepFraction = double(timeStep) / totalTimeSteps;
        Integer elapsedYears = Integer(timeStepFraction * swapDef.tenorYears);
        Integer remainingYears = swapDef.tenorYears - elapsedYears;

        if (remainingYears <= 0) {
            return Real(0.0);
        }

        // Extract inputs from flattened vector
        // Layout: EUR(10) + USD(10) + GBP(10) + JPY(10) + CHF(10) + FX(5) + CptyCredit(10) + OwnCredit(10) + Vol(25) = 100
        Size idx = 0;
        std::vector<Real> eurRates(allInputs.begin() + idx, allInputs.begin() + idx + 10); idx += 10;
        std::vector<Real> usdRates(allInputs.begin() + idx, allInputs.begin() + idx + 10); idx += 10;
        std::vector<Real> gbpRates(allInputs.begin() + idx, allInputs.begin() + idx + 10); idx += 10;
        std::vector<Real> jpyRates(allInputs.begin() + idx, allInputs.begin() + idx + 10); idx += 10;
        std::vector<Real> chfRates(allInputs.begin() + idx, allInputs.begin() + idx + 10); idx += 10;
        std::vector<Real> fxRates(allInputs.begin() + idx, allInputs.begin() + idx + 5); idx += 5;
        std::vector<Real> cptySpreads(allInputs.begin() + idx, allInputs.begin() + idx + 10); idx += 10;
        std::vector<Real> ownSpreads(allInputs.begin() + idx, allInputs.begin() + idx + 10); idx += 10;
        std::vector<Real> volSurface(allInputs.begin() + idx, allInputs.begin() + idx + 25); idx += 25;

        // Build EUR discount curve
        std::vector<Date> curveDates;
        std::vector<Real> curveRates;
        curveDates.push_back(today);
        curveRates.push_back(eurRates[0]);

        for (Size i = 0; i < eurRates.size() && i < eurPillarDefs.size(); ++i) {
            curveDates.push_back(calendar.advance(today, eurPillarDefs[i].tenor));
            curveRates.push_back(eurRates[i]);
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
        Real baseNpv = swap->NPV();

        // Add cross-currency basis effect (small adjustment based on other curves)
        // This ensures sensitivities to USD, GBP, JPY, CHF curves are non-zero
        Real ccyBasisAdj = Real(0.0);
        Real eurusd = fxRates[0];
        Real eurgbp = fxRates[4];
        for (Size i = 0; i < 10; ++i) {
            // Cross-currency basis spread effect (simplified)
            ccyBasisAdj += (usdRates[i] - eurRates[i]) * eurusd * Real(0.0001) * swapDef.notional;
            ccyBasisAdj += (gbpRates[i] - eurRates[i]) * eurgbp * Real(0.00005) * swapDef.notional;
            ccyBasisAdj += jpyRates[i] * Real(0.00001) * swapDef.notional;
            ccyBasisAdj += chfRates[i] * Real(0.00002) * swapDef.notional;
        }

        // CVA/DVA adjustment (simplified)
        // CVA = -LGD * integral(EE * dPD_cpty)
        // DVA = +LGD * integral(NEE * dPD_own)
        Real lgd = Real(0.4);
        Real exposure = baseNpv > Real(0.0) ? baseNpv : Real(0.0);
        Real negExposure = baseNpv < Real(0.0) ? -baseNpv : Real(0.0);

        Real cvaAdj = Real(0.0);
        Real dvaAdj = Real(0.0);
        for (Size i = 0; i < 10; ++i) {
            cvaAdj -= lgd * exposure * cptySpreads[i] * Real(0.1);  // Simplified
            dvaAdj += lgd * negExposure * ownSpreads[i] * Real(0.1);
        }

        // Volatility adjustment (for potential swaption-like optionality in XVA)
        Real volAdj = Real(0.0);
        for (Size i = 0; i < 25; ++i) {
            volAdj += volSurface[i] * Real(0.001) * swapDef.notional;
        }

        return baseNpv + ccyBasisAdj + cvaAdj + dvaAdj + volAdj;
    }

    //=========================================================================
    // PRICER INTERFACE
    //=========================================================================
    class IXvaPricer {
    public:
        virtual ~IXvaPricer() = default;
        virtual std::string name() const = 0;
        virtual XvaResults computeXva(
            const XvaConfig& config,
            const std::vector<SwapDefinition>& swaps,
            const std::vector<std::vector<MarketScenario>>& scenarios,
            const std::vector<IRPillar>& pillars,
            const Date& today,
            const Calendar& calendar,
            const DayCounter& dayCounter,
            TimingResults& timing) = 0;
    };

    //=========================================================================
    // 1. BUMP-REVAL PRICER (Baseline - no Forge)
    //=========================================================================
    class BumpRevalPricer : public IXvaPricer {
    public:
        std::string name() const override { return "Bump-Reval"; }

        XvaResults computeXva(
            const XvaConfig& config,
            const std::vector<SwapDefinition>& swaps,
            const std::vector<std::vector<MarketScenario>>& scenarios,
            const std::vector<IRPillar>& pillars,
            const Date& today,
            const Calendar& calendar,
            const DayCounter& dayCounter,
            TimingResults& timing) override {

            XvaResults results;
            results.exposures.resize(config.numSwaps);
            results.sensitivities.resize(config.numSwaps);

            double totalExposure = 0.0;
            Size totalEvaluations = 0;
            constexpr Size numInputs = MarketScenario::totalRiskFactors();  // 100

            auto evalStartTime = std::chrono::high_resolution_clock::now();

            for (Size s = 0; s < config.numSwaps; ++s) {
                results.exposures[s].resize(config.numTimeSteps);
                results.sensitivities[s].resize(config.numTimeSteps);

                for (Size t = 0; t < config.numTimeSteps; ++t) {
                    results.exposures[s][t].resize(config.numPaths);
                    results.sensitivities[s][t].resize(config.numPaths);

                    for (Size p = 0; p < config.numPaths; ++p) {
                        const auto& scenario = scenarios[t][p];
                        const std::vector<double>& flatInputs = scenario.flatten();
                        std::vector<Real> inputs(flatInputs.begin(), flatInputs.end());

                        // Base pricing
                        double baseNpv = value(priceSwapAtTimeStep(
                            swaps[s], t, config.numTimeSteps, inputs, pillars, today, calendar, dayCounter));
                        totalEvaluations++;

                        results.exposures[s][t][p] = std::max(0.0, baseNpv);
                        totalExposure += results.exposures[s][t][p];

                        // Bump each risk factor for sensitivities
                        results.sensitivities[s][t][p].resize(numInputs);
                        for (Size i = 0; i < numInputs; ++i) {
                            std::vector<Real> bumpedInputs = inputs;
                            bumpedInputs[i] += config.bumpSize;
                            double bumpedNpv = value(priceSwapAtTimeStep(
                                swaps[s], t, config.numTimeSteps, bumpedInputs, pillars, today, calendar, dayCounter));
                            totalEvaluations++;
                            results.sensitivities[s][t][p][i] = (bumpedNpv - baseNpv) / config.bumpSize;
                        }
                    }
                }
            }

            auto evalEndTime = std::chrono::high_resolution_clock::now();
            auto evalDuration = std::chrono::duration_cast<std::chrono::microseconds>(evalEndTime - evalStartTime);

            timing.numKernelsCreated = 0;
            timing.numEvaluations = totalEvaluations;
            timing.evaluationTimeMs = evalDuration.count() / 1000.0;
            timing.kernelCreationTimeMs = 0.0;
            timing.singleEvalTimeUs = double(evalDuration.count()) / totalEvaluations;

            Size totalScenarios = config.numSwaps * config.numTimeSteps * config.numPaths;
            results.expectedExposure = totalExposure / totalScenarios;
            results.cva = results.expectedExposure * 0.4 * 0.02;

            return results;
        }
    };

    //=========================================================================
    // 2. FORGE KERNEL BUMP-REVAL PRICER (JIT speedup, still uses bumping)
    //=========================================================================
    class ForgeKernelBumpRevalPricer : public IXvaPricer {
    public:
        std::string name() const override { return "Forge-Bump"; }

        XvaResults computeXva(
            const XvaConfig& config,
            const std::vector<SwapDefinition>& swaps,
            const std::vector<std::vector<MarketScenario>>& scenarios,
            const std::vector<IRPillar>& pillars,
            const Date& today,
            const Calendar& calendar,
            const DayCounter& dayCounter,
            TimingResults& timing) override {

            XvaResults results;
            results.exposures.resize(config.numSwaps);
            results.sensitivities.resize(config.numSwaps);

            double totalExposure = 0.0;
            Size numKernels = 0;
            Size numEvaluations = 0;
            double totalKernelCreationUs = 0.0;
            double totalEvalUs = 0.0;
            constexpr Size numInputs = MarketScenario::totalRiskFactors();  // 100

            for (Size s = 0; s < config.numSwaps; ++s) {
                results.exposures[s].resize(config.numTimeSteps);
                results.sensitivities[s].resize(config.numTimeSteps);

                for (Size t = 0; t < config.numTimeSteps; ++t) {
                    results.exposures[s][t].resize(config.numPaths);
                    results.sensitivities[s][t].resize(config.numPaths);

                    // --- KERNEL CREATION ---
                    auto kernelStartTime = std::chrono::high_resolution_clock::now();

                    forge::GraphRecorder recorder;
                    recorder.start();

                    // Create inputs for all 100 risk factors
                    std::vector<double> flatInputs = scenarios[t][0].flatten();
                    std::vector<Real> rateInputs(numInputs);
                    std::vector<forge::NodeId> rateNodeIds(numInputs);
                    for (Size i = 0; i < numInputs; ++i) {
                        rateInputs[i] = flatInputs[i];
                        rateInputs[i].markForgeInput();
                        rateNodeIds[i] = rateInputs[i].forgeNodeId();
                    }

                    Real npv = priceSwapAtTimeStep(swaps[s], t, config.numTimeSteps, rateInputs, pillars, today, calendar, dayCounter);
                    npv.markForgeOutput();
                    forge::NodeId npvNodeId = npv.forgeNodeId();

                    recorder.stop();
                    forge::Graph graph = recorder.graph();

                    forge::ForgeEngine compiler;
                    auto kernel = compiler.compile(graph);
                    auto buffer = forge::NodeValueBufferFactory::create(graph, *kernel);

                    auto kernelEndTime = std::chrono::high_resolution_clock::now();
                    totalKernelCreationUs += std::chrono::duration_cast<std::chrono::microseconds>(kernelEndTime - kernelStartTime).count();
                    numKernels++;

                    // --- EVALUATION (using kernel for bump-reval) ---
                    auto evalStartTime = std::chrono::high_resolution_clock::now();

                    for (Size p = 0; p < config.numPaths; ++p) {
                        const auto& scenario = scenarios[t][p];
                        std::vector<double> scenarioInputs = scenario.flatten();

                        // Base pricing using kernel
                        for (Size i = 0; i < numInputs; ++i) {
                            buffer->setValue(rateNodeIds[i], scenarioInputs[i]);
                        }
                        kernel->execute(*buffer);
                        numEvaluations++;

                        double baseNpv = buffer->getValue(npvNodeId);
                        results.exposures[s][t][p] = std::max(0.0, baseNpv);
                        totalExposure += results.exposures[s][t][p];

                        // Bump each risk factor and re-evaluate kernel
                        results.sensitivities[s][t][p].resize(numInputs);
                        for (Size i = 0; i < numInputs; ++i) {
                            // Set bumped value
                            buffer->setValue(rateNodeIds[i], scenarioInputs[i] + config.bumpSize);
                            kernel->execute(*buffer);
                            numEvaluations++;

                            double bumpedNpv = buffer->getValue(npvNodeId);
                            results.sensitivities[s][t][p][i] = (bumpedNpv - baseNpv) / config.bumpSize;

                            // Reset to original
                            buffer->setValue(rateNodeIds[i], scenarioInputs[i]);
                        }
                    }

                    auto evalEndTime = std::chrono::high_resolution_clock::now();
                    totalEvalUs += std::chrono::duration_cast<std::chrono::microseconds>(evalEndTime - evalStartTime).count();
                }
            }

            timing.numKernelsCreated = numKernels;
            timing.numEvaluations = numEvaluations;
            timing.kernelCreationTimeMs = totalKernelCreationUs / 1000.0;
            timing.evaluationTimeMs = totalEvalUs / 1000.0;
            timing.singleEvalTimeUs = totalEvalUs / numEvaluations;

            Size totalScenarios = config.numSwaps * config.numTimeSteps * config.numPaths;
            results.expectedExposure = totalExposure / totalScenarios;
            results.cva = results.expectedExposure * 0.4 * 0.02;

            return results;
        }
    };

    //=========================================================================
    // 3. FORGE AAD PRICER (JIT + AAD - no bumping needed)
    //=========================================================================
    class ForgeAadPricer : public IXvaPricer {
    public:
        std::string name() const override { return "Forge-AAD"; }

        XvaResults computeXva(
            const XvaConfig& config,
            const std::vector<SwapDefinition>& swaps,
            const std::vector<std::vector<MarketScenario>>& scenarios,
            const std::vector<IRPillar>& pillars,
            const Date& today,
            const Calendar& calendar,
            const DayCounter& dayCounter,
            TimingResults& timing) override {

            XvaResults results;
            results.exposures.resize(config.numSwaps);
            results.sensitivities.resize(config.numSwaps);

            double totalExposure = 0.0;
            Size numKernels = 0;
            Size numEvaluations = 0;
            double totalKernelCreationUs = 0.0;
            double totalEvalUs = 0.0;
            constexpr Size numInputs = MarketScenario::totalRiskFactors();  // 100

            // Detailed timing breakdown (in nanoseconds for precision)
            double totalFlattenUs = 0.0;
            double totalSetValueUs = 0.0;
            double totalExecuteUs = 0.0;
            double totalGetValueUs = 0.0;
            double totalGetGradUs = 0.0;

            for (Size s = 0; s < config.numSwaps; ++s) {
                results.exposures[s].resize(config.numTimeSteps);
                results.sensitivities[s].resize(config.numTimeSteps);

                for (Size t = 0; t < config.numTimeSteps; ++t) {
                    results.exposures[s][t].resize(config.numPaths);
                    results.sensitivities[s][t].resize(config.numPaths);

                    // --- KERNEL CREATION ---
                    auto kernelStartTime = std::chrono::high_resolution_clock::now();

                    forge::GraphRecorder recorder;
                    recorder.start();

                    // Create inputs for all 100 risk factors
                    std::vector<double> flatInputs = scenarios[t][0].flatten();
                    std::vector<Real> rateInputs(numInputs);
                    std::vector<forge::NodeId> rateNodeIds(numInputs);
                    for (Size i = 0; i < numInputs; ++i) {
                        rateInputs[i] = flatInputs[i];
                        rateInputs[i].markForgeInput();
                        rateNodeIds[i] = rateInputs[i].forgeNodeId();
                    }

                    Real npv = priceSwapAtTimeStep(swaps[s], t, config.numTimeSteps, rateInputs, pillars, today, calendar, dayCounter);
                    npv.markForgeOutput();
                    forge::NodeId npvNodeId = npv.forgeNodeId();

                    recorder.stop();
                    forge::Graph graph = recorder.graph();

                    forge::ForgeEngine compiler;
                    auto kernel = compiler.compile(graph);
                    auto buffer = forge::NodeValueBufferFactory::create(graph, *kernel);

                    // Pre-compute gradient buffer indices (do mapping once, not per-evaluation)
                    int vectorWidth = buffer->getVectorWidth();
                    std::vector<size_t> gradientIndices(numInputs);
                    for (Size i = 0; i < numInputs; ++i) {
                        // The buffer stores gradients at nodeId * vectorWidth
                        // For inputs, the node IDs are typically already the buffer indices
                        gradientIndices[i] = static_cast<size_t>(rateNodeIds[i]) * vectorWidth;
                    }

                    auto kernelEndTime = std::chrono::high_resolution_clock::now();
                    totalKernelCreationUs += std::chrono::duration_cast<std::chrono::microseconds>(kernelEndTime - kernelStartTime).count();
                    numKernels++;

                    // --- EVALUATION (single pass gets value + all gradients) ---
                    auto evalStartTime = std::chrono::high_resolution_clock::now();

                    // Pre-allocate output buffer for gradients (reused across paths)
                    std::vector<double> gradOutput(numInputs);

                    for (Size p = 0; p < config.numPaths; ++p) {
                        const auto& scenario = scenarios[t][p];

                        auto flattenStart = std::chrono::high_resolution_clock::now();
                        std::vector<double> scenarioInputs = scenario.flatten();
                        auto flattenEnd = std::chrono::high_resolution_clock::now();
                        totalFlattenUs += std::chrono::duration_cast<std::chrono::nanoseconds>(flattenEnd - flattenStart).count();

                        auto setValueStart = std::chrono::high_resolution_clock::now();
                        for (Size i = 0; i < numInputs; ++i) {
                            buffer->setValue(rateNodeIds[i], scenarioInputs[i]);
                        }
                        auto setValueEnd = std::chrono::high_resolution_clock::now();
                        totalSetValueUs += std::chrono::duration_cast<std::chrono::nanoseconds>(setValueEnd - setValueStart).count();

                        // Note: clearGradients() not needed - kernel overwrites gradients each execution
                        auto executeStart = std::chrono::high_resolution_clock::now();
                        kernel->execute(*buffer);
                        auto executeEnd = std::chrono::high_resolution_clock::now();
                        totalExecuteUs += std::chrono::duration_cast<std::chrono::nanoseconds>(executeEnd - executeStart).count();
                        numEvaluations++;

                        auto getValueStart = std::chrono::high_resolution_clock::now();
                        double npvValue = buffer->getValue(npvNodeId);
                        auto getValueEnd = std::chrono::high_resolution_clock::now();
                        totalGetValueUs += std::chrono::duration_cast<std::chrono::nanoseconds>(getValueEnd - getValueStart).count();

                        results.exposures[s][t][p] = std::max(0.0, npvValue);
                        totalExposure += results.exposures[s][t][p];

                        // Extract sensitivities using ultra-fast direct method (no mapping, no allocation)
                        auto getGradStart = std::chrono::high_resolution_clock::now();
                        buffer->getGradientsDirect(gradientIndices, gradOutput.data());
                        results.sensitivities[s][t][p] = gradOutput;  // Copy to results
                        auto getGradEnd = std::chrono::high_resolution_clock::now();
                        totalGetGradUs += std::chrono::duration_cast<std::chrono::nanoseconds>(getGradEnd - getGradStart).count();
                    }

                    auto evalEndTime = std::chrono::high_resolution_clock::now();
                    totalEvalUs += std::chrono::duration_cast<std::chrono::microseconds>(evalEndTime - evalStartTime).count();
                }
            }

            timing.numKernelsCreated = numKernels;
            timing.numEvaluations = numEvaluations;
            timing.kernelCreationTimeMs = totalKernelCreationUs / 1000.0;
            timing.evaluationTimeMs = totalEvalUs / 1000.0;
            timing.singleEvalTimeUs = totalEvalUs / numEvaluations;

            // Print detailed AAD timing breakdown
            std::cout << "\n  [Forge-AAD Timing Breakdown per " << numEvaluations << " evaluations]\n";
            std::cout << "    flatten():     " << std::fixed << std::setprecision(2) << (totalFlattenUs / 1000.0) << " us ("
                      << (totalFlattenUs / numEvaluations / 1000.0) << " us/eval)\n";
            std::cout << "    setValue():    " << (totalSetValueUs / 1000.0) << " us ("
                      << (totalSetValueUs / numEvaluations / 1000.0) << " us/eval)\n";
            std::cout << "    execute():     " << (totalExecuteUs / 1000.0) << " us ("
                      << (totalExecuteUs / numEvaluations / 1000.0) << " us/eval)\n";
            std::cout << "    getValue():    " << (totalGetValueUs / 1000.0) << " us ("
                      << (totalGetValueUs / numEvaluations / 1000.0) << " us/eval)\n";
            std::cout << "    getGradient(): " << (totalGetGradUs / 1000.0) << " us ("
                      << (totalGetGradUs / numEvaluations / 1000.0) << " us/eval, "
                      << (totalGetGradUs / numEvaluations / numInputs) << " ns per grad)\n";
            double totalAccounted = totalFlattenUs + totalSetValueUs + totalExecuteUs + totalGetValueUs + totalGetGradUs;
            std::cout << "    TOTAL:         " << (totalAccounted / 1000.0) << " us ("
                      << (totalAccounted / numEvaluations / 1000.0) << " us/eval)\n";

            Size totalScenarios = config.numSwaps * config.numTimeSteps * config.numPaths;
            results.expectedExposure = totalExposure / totalScenarios;
            results.cva = results.expectedExposure * 0.4 * 0.02;

            return results;
        }
    };

    //=========================================================================
    // Run XVA with timing
    //=========================================================================
    TimingResults runWithTiming(
        IXvaPricer& pricer,
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
            pricer.computeXva(config, swaps, scenarios, pillars, today, calendar, dayCounter, warmupTiming);
        }

        TimingResults accumulatedTiming;
        auto startTime = std::chrono::high_resolution_clock::now();

        for (Size r = 0; r < config.timedRuns; ++r) {
            TimingResults iterTiming;
            lastResults = pricer.computeXva(config, swaps, scenarios, pillars, today, calendar, dayCounter, iterTiming);
            accumulatedTiming.kernelCreationTimeMs += iterTiming.kernelCreationTimeMs;
            accumulatedTiming.evaluationTimeMs += iterTiming.evaluationTimeMs;
            accumulatedTiming.numKernelsCreated += iterTiming.numKernelsCreated;
            accumulatedTiming.numEvaluations += iterTiming.numEvaluations;
        }

        auto endTime = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime);

        timing.totalTimeMs = duration.count() / 1000.0;
        timing.avgTimePerIterationMs = timing.totalTimeMs / config.timedRuns;
        timing.kernelCreationTimeMs = accumulatedTiming.kernelCreationTimeMs / config.timedRuns;
        timing.evaluationTimeMs = accumulatedTiming.evaluationTimeMs / config.timedRuns;
        timing.numKernelsCreated = accumulatedTiming.numKernelsCreated / config.timedRuns;
        timing.numEvaluations = accumulatedTiming.numEvaluations / config.timedRuns;
        timing.singleEvalTimeUs = (accumulatedTiming.evaluationTimeMs * 1000.0) / accumulatedTiming.numEvaluations;

        Size basePricings = config.numSwaps * config.numTimeSteps * config.numPaths;
        timing.totalPricings = basePricings * config.timedRuns;
        timing.totalSensitivities = timing.totalPricings * pillars.size();

        return timing;
    }

    //=========================================================================
    // Risk factor names for display
    //=========================================================================
    std::vector<std::string> getRiskFactorNames() {
        std::vector<std::string> names;
        // EUR curve (10)
        for (const auto& p : std::vector<std::string>{"6M","1Y","2Y","3Y","5Y","7Y","10Y","15Y","20Y","30Y"})
            names.push_back("EUR_" + p);
        // USD curve (10)
        for (const auto& p : std::vector<std::string>{"6M","1Y","2Y","3Y","5Y","7Y","10Y","15Y","20Y","30Y"})
            names.push_back("USD_" + p);
        // GBP curve (10)
        for (const auto& p : std::vector<std::string>{"6M","1Y","2Y","3Y","5Y","7Y","10Y","15Y","20Y","30Y"})
            names.push_back("GBP_" + p);
        // JPY curve (10)
        for (const auto& p : std::vector<std::string>{"6M","1Y","2Y","3Y","5Y","7Y","10Y","15Y","20Y","30Y"})
            names.push_back("JPY_" + p);
        // CHF curve (10)
        for (const auto& p : std::vector<std::string>{"6M","1Y","2Y","3Y","5Y","7Y","10Y","15Y","20Y","30Y"})
            names.push_back("CHF_" + p);
        // FX (5)
        names.push_back("EURUSD"); names.push_back("GBPUSD"); names.push_back("USDJPY");
        names.push_back("USDCHF"); names.push_back("EURGBP");
        // Counterparty credit (10)
        for (int i = 1; i <= 10; ++i) names.push_back("CptyCr_" + std::to_string(i) + "Y");
        // Own credit (10)
        for (int i = 1; i <= 10; ++i) names.push_back("OwnCr_" + std::to_string(i) + "Y");
        // Vol surface (25)
        for (int i = 0; i < 25; ++i) names.push_back("Vol_" + std::to_string(i));
        return names;
    }

    //=========================================================================
    // Print results table (3 columns)
    //=========================================================================
    void printResultsTable(
        const TimingResults& bumpTiming,
        const XvaResults& bumpResults,
        const TimingResults& forgeBumpTiming,
        const XvaResults& forgeBumpResults,
        const TimingResults& forgeAadTiming,
        const XvaResults& forgeAadResults,
        const XvaConfig& config) {

        const int col1 = 22;
        const int col2 = 14;
        const int col3 = 14;
        const int col4 = 14;
        constexpr Size numInputs = MarketScenario::totalRiskFactors();  // 100

        auto line = [&]() {
            std::cout << "+" << std::string(col1, '-') << "+"
                      << std::string(col2, '-') << "+"
                      << std::string(col3, '-') << "+"
                      << std::string(col4, '-') << "+\n";
        };

        std::cout << "\n";
        line();
        std::cout << "|                  XVA PERFORMANCE COMPARISON (FORGE)                  |\n";
        line();
        std::cout << "| Config: " << config.numSwaps << " swap, "
                  << config.numTimeSteps << " steps, "
                  << config.numPaths << " paths, "
                  << numInputs << " risk factors"
                  << std::string(6, ' ') << "|\n";
        line();

        // Header
        std::cout << "|" << std::setw(col1) << std::left << " Method"
                  << "|" << std::setw(col2) << std::right << "Bump-Reval"
                  << "|" << std::setw(col3) << std::right << "Forge-Bump"
                  << "|" << std::setw(col4) << std::right << "Forge-AAD" << "|\n";
        line();

        std::cout << std::fixed << std::setprecision(2);

        // Key metric: time per iteration (what you'd see in production)
        std::cout << "|" << std::setw(col1) << std::left << " Time/Iteration (ms)"
                  << "|" << std::setw(col2) << std::right << bumpTiming.avgTimePerIterationMs
                  << "|" << std::setw(col3) << std::right << forgeBumpTiming.avgTimePerIterationMs
                  << "|" << std::setw(col4) << std::right << forgeAadTiming.avgTimePerIterationMs << "|\n";

        // Number of kernel executions per iteration
        std::cout << "|" << std::setw(col1) << std::left << " Kernel Execs/Iter"
                  << "|" << std::setw(col2) << std::right << bumpTiming.numEvaluations
                  << "|" << std::setw(col3) << std::right << forgeBumpTiming.numEvaluations
                  << "|" << std::setw(col4) << std::right << forgeAadTiming.numEvaluations << "|\n";

        // Time per single kernel execution
        std::cout << "|" << std::setw(col1) << std::left << " Time/Kernel (us)"
                  << "|" << std::setw(col2) << std::right << bumpTiming.singleEvalTimeUs
                  << "|" << std::setw(col3) << std::right << forgeBumpTiming.singleEvalTimeUs
                  << "|" << std::setw(col4) << std::right << forgeAadTiming.singleEvalTimeUs << "|\n";

        // Time breakdown for Forge methods
        std::cout << "|" << std::setw(col1) << std::left << " Kernel Create (ms)"
                  << "|" << std::setw(col2) << std::right << "-"
                  << "|" << std::setw(col3) << std::right << forgeBumpTiming.kernelCreationTimeMs
                  << "|" << std::setw(col4) << std::right << forgeAadTiming.kernelCreationTimeMs << "|\n";

        // Pure evaluation time (excluding kernel creation)
        std::cout << "|" << std::setw(col1) << std::left << " Pure Eval (ms)"
                  << "|" << std::setw(col2) << std::right << bumpTiming.evaluationTimeMs
                  << "|" << std::setw(col3) << std::right << forgeBumpTiming.evaluationTimeMs
                  << "|" << std::setw(col4) << std::right << forgeAadTiming.evaluationTimeMs << "|\n";
        line();

        // Speedups (based on per-iteration time - the meaningful comparison)
        double speedupForgeBump = bumpTiming.avgTimePerIterationMs / forgeBumpTiming.avgTimePerIterationMs;
        double speedupForgeAad = bumpTiming.avgTimePerIterationMs / forgeAadTiming.avgTimePerIterationMs;

        std::cout << "|" << std::setw(col1) << std::left << " Speedup vs Baseline"
                  << "|" << std::setw(col2-1) << std::right << "1.00" << "x"
                  << "|" << std::setw(col3-1) << std::right << speedupForgeBump << "x"
                  << "|" << std::setw(col4-1) << std::right << speedupForgeAad << "x|\n";

        // Kernel efficiency: AAD vs Bump (same kernel, different # executions)
        double aadEfficiency = forgeBumpTiming.avgTimePerIterationMs / forgeAadTiming.avgTimePerIterationMs;
        Size evalsPerScenarioBump = 1 + numInputs;  // 101 for 100 risk factors
        std::cout << "|" << std::setw(col1) << std::left << " AAD vs Forge-Bump"
                  << "|" << std::setw(col2) << std::right << "-"
                  << "|" << std::setw(col2-1) << std::right << "1.00" << "x"
                  << "|" << std::setw(col4-1) << std::right << aadEfficiency << "x|\n";

        std::cout << "|" << std::setw(col1) << std::left << " AAD Theoretical"
                  << "|" << std::setw(col2) << std::right << "-"
                  << "|" << std::setw(col3) << std::right << "-"
                  << "|" << std::setw(col4-1) << std::right << evalsPerScenarioBump << "x|\n";
        line();

        // Value Results
        std::cout << std::scientific << std::setprecision(4);
        std::cout << "|" << std::setw(col1) << std::left << " Expected Exposure"
                  << "|" << std::setw(col2) << std::right << bumpResults.expectedExposure
                  << "|" << std::setw(col3) << std::right << forgeBumpResults.expectedExposure
                  << "|" << std::setw(col4) << std::right << forgeAadResults.expectedExposure << "|\n";

        std::cout << "|" << std::setw(col1) << std::left << " CVA"
                  << "|" << std::setw(col2) << std::right << bumpResults.cva
                  << "|" << std::setw(col3) << std::right << forgeBumpResults.cva
                  << "|" << std::setw(col4) << std::right << forgeAadResults.cva << "|\n";
        line();

        // Sample sensitivities
        auto rfNames = getRiskFactorNames();
        std::cout << "| Sample Sensitivities (Swap 0, Time 0, Path 0):" << std::string(19, ' ') << "|\n";
        line();

        // Show a few sensitivities from different categories
        std::vector<Size> sampleIndices = {0, 4, 10, 50, 55, 65, 75, 85};  // EUR_6M, EUR_5Y, USD_6M, FX, CptyCr, OwnCr, Vol
        for (Size i : sampleIndices) {
            if (i < numInputs && i < rfNames.size()) {
                std::string label = " d/" + rfNames[i];
                if (label.length() > 20) label = label.substr(0, 20);
                std::cout << "|" << std::setw(col1) << std::left << label
                          << "|" << std::setw(col2) << std::right << bumpResults.sensitivities[0][0][0][i]
                          << "|" << std::setw(col3) << std::right << forgeBumpResults.sensitivities[0][0][0][i]
                          << "|" << std::setw(col4) << std::right << forgeAadResults.sensitivities[0][0][0][i] << "|\n";
            }
        }
        std::cout << "|" << std::setw(col1) << std::left << " ... (100 total)"
                  << "|" << std::setw(col2) << " "
                  << "|" << std::setw(col3) << " "
                  << "|" << std::setw(col4) << " " << "|\n";
        line();
        std::cout << "\n";
    }

}  // namespace

//=============================================================================
// TEST CASE: XVA Performance Comparison (3 methods)
//=============================================================================
BOOST_AUTO_TEST_CASE(testSwapXvaPerformance) {
    SavedSettings save;

    BOOST_TEST_MESSAGE("Testing XVA: Bump-Reval vs Forge-Bump vs Forge-AAD...");
    BOOST_TEST_MESSAGE("  100 risk factors: EUR/USD/GBP/JPY/CHF curves, FX, credit spreads, vol surface");

    XvaConfig config;
    // Use default values from struct: 1 swap, 3 steps, 10 paths, 2 warmup, 5 timed

    Calendar calendar = TARGET();
    Date today = Date(15, January, 2024);
    Settings::instance().evaluationDate() = today;
    DayCounter dayCounter = Actual365Fixed();

    auto eurPillars = createEURCurvePillars();
    auto baseRf = createBaseRiskFactors();
    auto swaps = createSwapDefinitions(config.numSwaps);
    auto scenarios = generateScenarios(config, eurPillars, baseRf);

    // 1. Bump-Reval (baseline)
    BumpRevalPricer bumpPricer;
    BOOST_TEST_MESSAGE("  Running Bump-Reval...");
    XvaResults bumpResults;
    auto bumpTiming = runWithTiming(bumpPricer, config, swaps, scenarios, eurPillars, today, calendar, dayCounter, bumpResults);

    // 2. Forge Kernel Bump-Reval (JIT speedup only)
    ForgeKernelBumpRevalPricer forgeBumpPricer;
    BOOST_TEST_MESSAGE("  Running Forge-Bump...");
    XvaResults forgeBumpResults;
    auto forgeBumpTiming = runWithTiming(forgeBumpPricer, config, swaps, scenarios, eurPillars, today, calendar, dayCounter, forgeBumpResults);

    // 3. Forge AAD (JIT + AAD)
    ForgeAadPricer forgeAadPricer;
    BOOST_TEST_MESSAGE("  Running Forge-AAD...");
    XvaResults forgeAadResults;
    auto forgeAadTiming = runWithTiming(forgeAadPricer, config, swaps, scenarios, eurPillars, today, calendar, dayCounter, forgeAadResults);

    // Print comparison
    printResultsTable(bumpTiming, bumpResults,
                      forgeBumpTiming, forgeBumpResults,
                      forgeAadTiming, forgeAadResults,
                      config);

    // Verify results match
    BOOST_CHECK_CLOSE(bumpResults.expectedExposure, forgeBumpResults.expectedExposure, 0.1);
    BOOST_CHECK_CLOSE(bumpResults.expectedExposure, forgeAadResults.expectedExposure, 0.1);

    constexpr Size numInputs = MarketScenario::totalRiskFactors();
    for (Size i = 0; i < numInputs; ++i) {
        double bumpSens = bumpResults.sensitivities[0][0][0][i];
        if (std::abs(bumpSens) > 1e-10) {
            QL_CHECK_CLOSE(bumpSens, forgeBumpResults.sensitivities[0][0][0][i], 1.0);
            QL_CHECK_CLOSE(bumpSens, forgeAadResults.sensitivities[0][0][0][i], 1.0);
        }
    }

    BOOST_TEST_MESSAGE("  XVA test completed successfully.");
}

BOOST_AUTO_TEST_SUITE_END()

BOOST_AUTO_TEST_SUITE_END()
