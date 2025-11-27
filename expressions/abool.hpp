/*******************************************************************************

   ABool: Integration layer between Forge's fbool and AReal.

   This file is part of QuantLib-Forge, a Forge integration layer for QuantLib.

   Copyright (C) 2025 The QuantLib-Forge Authors

   SPDX-License-Identifier: Zlib

******************************************************************************/

#pragma once

#include <types/fbool.hpp>
#include <types/fdouble.hpp>
#include <iostream>

namespace forge {

/**
 * ABool: Trackable boolean for conditional graph recording.
 *
 * Stores both:
 *   - passive_ : normal C++ boolean (used when not recording)
 *   - active_  : graph-side fbool (used during recording)
 *
 * This type is intentionally generic w.r.t. the value type T. T is expected
 * to provide:
 *   - forge::fdouble forgeValue() const;
 *   - void setForgeValue(const forge::fdouble&);
 *
 * In practice, this matches forge::expr::AReal<double,N> in our integration layer.
 */
class ABool {
public:
    bool  passive_;   // C++ truth value
    fbool active_;    // Graph-side boolean
    bool  hasActive_; // Whether active_ has a valid graph node

    // Constructor: from plain bool (no graph tracking)
    explicit ABool(bool b = false)
        : passive_(b), active_(fbool(b)), hasActive_(false) {}

    // Constructor: from fbool + passive value (graph tracking enabled)
    ABool(const fbool& fb, bool passive)
        : passive_(passive), active_(fb), hasActive_(true) {}

    // Accessors
    bool passive() const { return passive_; }
    const fbool& active() const { return active_; }
    bool isActive() const { return hasActive_ && active_.isActive(); }

    // Allow seamless use in existing bool contexts (if, while, etc.).
    // NOTE: this only exposes the passive value; recording of dynamic
    // branches still requires using ABool::If(...) explicitly.
    //
    // When Forge recording is active and this ABool is active, using it
    // as a plain bool means we're about to drop out of the Forge graph
    // (e.g. by doing a normal C++ if/else instead of ABool::If). We
    // emit a warning (but do not throw) to help locate missing Forge wiring.
    operator bool() const {
        if (GraphRecorder::isAnyRecording() && isActive()) {
            static std::size_t warnCount = 0;
            if (warnCount < 10) {
                ++warnCount;
                std::cerr
                    << "[Forge][Warning] ABool::operator bool() called while Forge "
                    << "recording is active on an active condition – this drops you "
                    << "out of the Forge graph; use ABool::If(...) instead "
                    << "(occurrence " << warnCount << ")\n";
            }
        }
        return passive_;
    }

    // Core API: Conditional selection
    // Returns: trueVal if condition is true, falseVal otherwise.
    // During recording: creates OpCode::If node in the graph.
    template<class T>
    T If(const T& trueVal, const T& falseVal) const
    {
        // 1. Fallback to passive behavior when not recording or no active condition
        if (!hasActive_ || !GraphRecorder::isAnyRecording()) {
            return passive_ ? trueVal : falseVal;
        }

        // 2. Extract Forge numeric side from the value type
        forge::fdouble forgeTrue  = trueVal.forgeValue();
        forge::fdouble forgeFalse = falseVal.forgeValue();

        // 3. Graph-level If via fbool
        forge::fdouble forgeResult = active_.If(forgeTrue, forgeFalse);

        // 4. Build result value
        T result = passive_ ? trueVal : falseVal;
        // Sync Forge side to the graph result (T must provide setForgeValue)
        result.setForgeValue(forgeResult);
        return result;
    }

    // Helper: static If (alternative call style)
    template<class T>
    static T If(const ABool& cond, const T& trueVal, const T& falseVal)
    {
        return cond.If(trueVal, falseVal);
    }
};

} // namespace forge

