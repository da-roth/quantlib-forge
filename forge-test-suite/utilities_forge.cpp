/* -*- mode: c++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*- */

/*******************************************************************************

   Test utilities implementation for QuantLib-Forge Risks test suite.

   This file is part of QuantLib-Forge, a Forge integration layer for QuantLib.

   Copyright (C) 2025 The QuantLib-Forge Authors
   Copyright (C) 2003, 2004 StatPro Italia srl

   Derived from QuantLib-Risks-Cpp / XAD (https://github.com/auto-differentiation/XAD)
   Original XAD code: Copyright (C) 2010-2024 Xcelerit Computing Ltd.

   SPDX-License-Identifier: AGPL-3.0-or-later

******************************************************************************/

#include "utilities_forge.hpp"
#include <ql/indexes/indexmanager.hpp>
#include <ql/instruments/payoffs.hpp>
#include <ql/termstructures/volatility/equityfx/blackconstantvol.hpp>
#include <ql/termstructures/yield/flatforward.hpp>
#include <ql/time/calendars/nullcalendar.hpp>

#define CHECK_DOWNCAST(Derived, Description)                                 \
    {                                                                        \
        ext::shared_ptr<Derived> hd = ext::dynamic_pointer_cast<Derived>(h); \
        if (hd)                                                              \
            return Description;                                              \
    }

namespace QuantLib {

    std::string payoffTypeToString(const ext::shared_ptr<Payoff>& h) {

        CHECK_DOWNCAST(PlainVanillaPayoff, "plain-vanilla");
        CHECK_DOWNCAST(CashOrNothingPayoff, "cash-or-nothing");
        CHECK_DOWNCAST(AssetOrNothingPayoff, "asset-or-nothing");
        CHECK_DOWNCAST(SuperSharePayoff, "super-share");
        CHECK_DOWNCAST(SuperFundPayoff, "super-fund");
        CHECK_DOWNCAST(PercentageStrikePayoff, "percentage-strike");
        CHECK_DOWNCAST(GapPayoff, "gap");
        CHECK_DOWNCAST(FloatingTypePayoff, "floating-type");

        QL_FAIL("unknown payoff type");
    }


    std::string exerciseTypeToString(const ext::shared_ptr<Exercise>& h) {

        CHECK_DOWNCAST(EuropeanExercise, "European");
        CHECK_DOWNCAST(AmericanExercise, "American");
        CHECK_DOWNCAST(BermudanExercise, "Bermudan");

        QL_FAIL("unknown exercise type");
    }


    ext::shared_ptr<YieldTermStructure>
    flatRate(const Date& today, const ext::shared_ptr<Quote>& forward, const DayCounter& dc) {
        return ext::shared_ptr<YieldTermStructure>(
            new FlatForward(today, Handle<Quote>(forward), dc));
    }

    ext::shared_ptr<YieldTermStructure>
    flatRate(const Date& today, Rate forward, const DayCounter& dc) {
        return flatRate(today, ext::shared_ptr<Quote>(new SimpleQuote(forward)), dc);
    }

    ext::shared_ptr<YieldTermStructure> flatRate(const ext::shared_ptr<Quote>& forward,
                                                 const DayCounter& dc) {
        return ext::shared_ptr<YieldTermStructure>(
            new FlatForward(0, NullCalendar(), Handle<Quote>(forward), dc));
    }

    ext::shared_ptr<YieldTermStructure> flatRate(Rate forward, const DayCounter& dc) {
        return flatRate(ext::shared_ptr<Quote>(new SimpleQuote(forward)), dc);
    }


    ext::shared_ptr<BlackVolTermStructure>
    flatVol(const Date& today, const ext::shared_ptr<Quote>& vol, const DayCounter& dc) {
        return ext::shared_ptr<BlackVolTermStructure>(
            new BlackConstantVol(today, NullCalendar(), Handle<Quote>(vol), dc));
    }

    ext::shared_ptr<BlackVolTermStructure>
    flatVol(const Date& today, Volatility vol, const DayCounter& dc) {
        return flatVol(today, ext::shared_ptr<Quote>(new SimpleQuote(vol)), dc);
    }

    ext::shared_ptr<BlackVolTermStructure> flatVol(const ext::shared_ptr<Quote>& vol,
                                                   const DayCounter& dc) {
        return ext::shared_ptr<BlackVolTermStructure>(
            new BlackConstantVol(0, NullCalendar(), Handle<Quote>(vol), dc));
    }

    ext::shared_ptr<BlackVolTermStructure> flatVol(Volatility vol, const DayCounter& dc) {
        return flatVol(ext::shared_ptr<Quote>(new SimpleQuote(vol)), dc);
    }


    Real relativeError(Real x1, Real x2, Real reference) {
        if (reference != 0.0)
            return std::fabs(x1 - x2) / reference;
        else
            // fall back to absolute error
            return std::fabs(x1 - x2);
    }


    IndexHistoryCleaner::~IndexHistoryCleaner() {
        IndexManager::instance().clearHistories();
    }

}
