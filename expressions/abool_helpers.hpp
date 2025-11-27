/*******************************************************************************

   Helper functions to build ABool from fdouble comparisons.

   This file is part of QuantLib-Forge, a Forge integration layer for QuantLib.

   Copyright (C) 2025 The QuantLib-Forge Authors

   SPDX-License-Identifier: Zlib

******************************************************************************/

#pragma once

#include "abool.hpp"
#include <types/fdouble.hpp>

namespace forge {

inline ABool greater(const fdouble& a, const fdouble& b) {
    bool  passive = (a.value() > b.value());
    fbool active  = (a > b);
    return ABool(active, passive);
}

inline ABool less(const fdouble& a, const fdouble& b) {
    bool  passive = (a.value() < b.value());
    fbool active  = (a < b);
    return ABool(active, passive);
}

inline ABool greaterEqual(const fdouble& a, const fdouble& b) {
    bool  passive = (a.value() >= b.value());
    fbool active  = (a >= b);
    return ABool(active, passive);
}

inline ABool lessEqual(const fdouble& a, const fdouble& b) {
    bool  passive = (a.value() <= b.value());
    fbool active  = (a <= b);
    return ABool(active, passive);
}

inline ABool equal(const fdouble& a, const fdouble& b) {
    bool  passive = (a.value() == b.value());
    fbool active  = (a == b);
    return ABool(active, passive);
}

inline ABool notEqual(const fdouble& a, const fdouble& b) {
    bool  passive = (a.value() != b.value());
    fbool active  = (a != b);
    return ABool(active, passive);
}

} // namespace forge

