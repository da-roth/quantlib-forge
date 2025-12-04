/* -*- mode: c++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*- */

/*******************************************************************************

   Test suite main entry point for QuantLib-Forge Risks tests.

   This file is part of QuantLib-Forge, a Forge integration layer for QuantLib.

   Copyright (C) 2025 The QuantLib-Forge Authors
   Copyright (C) 2004, 2005, 2006, 2007 Ferdinando Ametrano
   Copyright (C) 2004, 2005, 2006, 2007, 2008 StatPro Italia srl

   Derived from QuantLib-Risks-Cpp / XAD (https://github.com/auto-differentiation/XAD)
   Original XAD code: Copyright (C) 2010-2024 Xcelerit Computing Ltd.

   SPDX-License-Identifier: AGPL-3.0-or-later

******************************************************************************/

#define BOOST_TEST_MODULE QuantLibForgeRisksTests

#include <boost/test/included/unit_test.hpp>

/* Use BOOST_MSVC instead of _MSC_VER since some other vendors (Metrowerks,
   for example) also #define _MSC_VER
*/
#if !defined(BOOST_ALL_NO_LIB) && defined(BOOST_MSVC)
#    include <ql/auto_link.hpp>
#endif
