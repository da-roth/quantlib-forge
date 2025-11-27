##############################################################################
#
#  CMake support for expressions (forge::expr)
#  -------------------------------------------
#  Header-only expression template layer used by QuantLib-Forge.
#  This file deliberately contains NO AD/tape implementation; users
#  are expected to plug in their own AD backend.
#
#  Copyright (C) 2025 The QuantLib-Forge Authors
#
#  Derived from QuantLib-Risks-Cpp / XAD (https://github.com/auto-differentiation/XAD)
#  Original XAD code: Copyright (C) 2010-2024 Xcelerit Computing Ltd.
#
#  SPDX-License-Identifier: AGPL-3.0-or-later
#
##############################################################################

include(GNUInstallDirs)

# Match original XAD behaviour: allow integer conversions by default.
# This controls FEXPR_ALLOW_INT_CONVERSION in Config.hpp, which in turn
# enables implicit integer conversion operators in Expression.hpp.
set(FEXPR_ALLOW_INT_CONVERSION ON CACHE BOOL "Allow conversion from active type to integers")

# Generate Config.hpp from Config.hpp.in into an expressions subdir
# so that includes of <expressions/Config.hpp> work both in-build
# and after install.
file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/expressions)
configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/Config.hpp.in
    ${CMAKE_CURRENT_BINARY_DIR}/expressions/Config.hpp
    @ONLY
)

# Header-only interface target
add_library(forge-expressions INTERFACE)

# ========== Forge Integration: Link to installed Forge package ==========
# Forge::forge provides:
#   - Core Forge library (types/fdouble, graph recording, JIT compiler)
#   - All necessary include directories (src/, tools/)
#   - All necessary dependencies (asmjit, sleef, nlohmann_json)
if(TARGET Forge::forge)
    message(STATUS "Linking forge-expressions to Forge::forge")
    target_link_libraries(forge-expressions INTERFACE Forge::forge)
else()
    message(FATAL_ERROR "Forge::forge target not found - ensure find_package(Forge) was called before adding expressions")
endif()
# ========================================================================

# For builds:
#  - Parent directory (quantlib-forge) is on the include path so that
#    <expressions/...> resolves correctly
#  - Binary dir is on the include path so generated Config.hpp is visible
#  - Forge include paths come from the Forge::forge target automatically
get_filename_component(EXPRESSIONS_PARENT_DIR
    "${CMAKE_CURRENT_SOURCE_DIR}/.." ABSOLUTE)

target_include_directories(forge-expressions INTERFACE
    $<BUILD_INTERFACE:${EXPRESSIONS_PARENT_DIR}>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

# Install root headers
install(FILES
    Expression.hpp
    Traits.hpp
    Macros.hpp
    Exceptions.hpp
    Literals.hpp
    abool.hpp
    abool_helpers.hpp
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/expressions
)

# Install subdirectories
install(DIRECTORY ExpressionTemplates/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/expressions/ExpressionTemplates
    FILES_MATCHING PATTERN "*.hpp"
)

install(DIRECTORY Compatibility/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/expressions/Compatibility
    FILES_MATCHING PATTERN "*.hpp"
)

# Install generated Config.hpp
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/expressions/Config.hpp
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/expressions
)

# Add forge-expressions to the QuantLibTargets export set
# This is required because QuantLib-Forge links to forge-expressions
install(TARGETS forge-expressions EXPORT QuantLibTargets)

