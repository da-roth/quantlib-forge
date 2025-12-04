##############################################################################
#
#  CMake support for QuantLib-Forge test suite
#
#  This file is part of QuantLib-Forge, a Forge integration layer for QuantLib.
#
#  Copyright (C) 2025 The QuantLib-Forge Authors
#
#  Derived from QuantLib-Risks-Cpp / XAD (https://github.com/auto-differentiation/XAD)
#  Original XAD code: Copyright (C) 2010-2024 Xcelerit Computing Ltd.
#
#  SPDX-License-Identifier: AGPL-3.0-or-later
#
##############################################################################

set(QL_FORGE_TEST_SOURCES
    # bonds_forge.cpp first for debugging - to see warnings before counter fills up
    bonds_forge.cpp
    americanoption_forge.cpp
    barrieroption_forge_abool.cpp
    batesmodel_forge.cpp
    bermudanswaption_forge.cpp
    creditdefaultswap_forge.cpp
    europeanoption_forge.cpp
    forwardrateagreement_forge.cpp
    hestonmodel_forge.cpp
    swap_forge.cpp

    utilities_forge.cpp
    quantlibtestsuite_forge.cpp
)

set(QL_FORGE_TEST_HEADERS
    utilities_forge.hpp
    toplevelfixture.hpp
)

add_executable(ql_forge_test_suite ${QL_FORGE_TEST_SOURCES} ${QL_FORGE_TEST_HEADERS})
set_target_properties(ql_forge_test_suite PROPERTIES OUTPUT_NAME "forge-test-suite")
if (NOT Boost_USE_STATIC_LIBS)
    target_compile_definitions(ql_forge_test_suite PRIVATE BOOST_ALL_DYN_LINK)
endif()
target_link_libraries(ql_forge_test_suite PRIVATE
    ql_library
    Forge::forge
    ${QL_THREAD_LIBRARIES})
if (QL_INSTALL_TEST_SUITE)
    install(TARGETS ql_forge_test_suite RUNTIME DESTINATION ${QL_INSTALL_BINDIR})
endif()
add_test(NAME ql_forge_test_suite COMMAND ql_forge_test_suite --log_level=message)
