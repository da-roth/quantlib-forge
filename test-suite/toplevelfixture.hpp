/* -*- mode: c++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*- */

/*******************************************************************************

   Top-level test fixture for QuantLib-Forge tests.

   This file is part of QuantLib-Forge, a Forge integration layer for QuantLib.

   Copyright (C) 2025 The QuantLib-Forge Authors
   Copyright (C) 2004, 2005, 2006, 2007 Ferdinando Ametrano
   Copyright (C) 2004, 2005, 2006, 2007, 2008 StatPro Italia srl

   Derived from QuantLib-Risks-Cpp / XAD (https://github.com/auto-differentiation/XAD)
   Original XAD code: Copyright (C) 2010-2024 Xcelerit Computing Ltd.

   SPDX-License-Identifier: AGPL-3.0-or-later

******************************************************************************/

#ifndef quantlib_forge_top_level_fixture_hpp
#define quantlib_forge_top_level_fixture_hpp

#include <boost/test/unit_test.hpp>
#include <ql/indexes/indexmanager.hpp>
#include <ql/settings.hpp>

namespace QuantLib {

    using QuantLib::SavedSettings;
    using QuantLib::IndexManager;

    class TopLevelFixture {  // NOLINT(cppcoreguidelines-special-member-functions)
      public:
        // Restore settings after each test.
        SavedSettings restore;

        TopLevelFixture() = default;

        ~TopLevelFixture() {
            IndexManager::instance().clearHistories();
            BOOST_CHECK(true);
        }

#if BOOST_VERSION <= 105300
        // defined to avoid unused-variable warnings. It doesn't
        // work after Boost 1.53 because the functions were
        // overloaded and the address can't be resolved.
        void _use_check(const void* = &boost::test_tools::check_is_close,
                        const void* = &boost::test_tools::check_is_small) const {}
#endif
    };
}

#endif
