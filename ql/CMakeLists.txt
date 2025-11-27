##############################################################################
#
#  CMake support for QuantLib-Forge interface library
#
#  Copyright (C) 2025 The QuantLib-Forge Authors
#
#  Derived from QuantLib-Risks-Cpp / XAD (https://github.com/auto-differentiation/XAD)
#  Original XAD code: Copyright (C) 2010-2024 Xcelerit Computing Ltd.
#
#  SPDX-License-Identifier: AGPL-3.0-or-later
#
##############################################################################

set(QLFORGE_HEADERS
    qlforge.hpp
)

add_library(QuantLib-Forge INTERFACE)

target_include_directories(QuantLib-Forge INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>
    $<INSTALL_INTERFACE:${QL_INSTALL_INCLUDEDIR}>
)

target_compile_features(QuantLib-Forge INTERFACE cxx_std_17)

if(NOT QLFORGE_DISABLE_AAD)
    target_compile_definitions(QuantLib-Forge INTERFACE QL_INCLUDE_FIRST=ql/qlforge.hpp)
else()
    target_compile_definitions(QuantLib-Forge INTERFACE QLFORGE_DISABLE_AAD=1)
endif()

if(MSVC)
    target_compile_options(QuantLib-Forge INTERFACE /bigobj)
endif()

# Link to:
#   - Forge::forge: Core Forge library (fdouble, graph recording, JIT compiler)
#   - forge-expressions: Expression templates (in quantlib-forge/expressions/)
target_link_libraries(QuantLib-Forge INTERFACE Forge::forge forge-expressions)

set_target_properties(QuantLib-Forge PROPERTIES
    EXPORT_NAME QuantLib-Forge
)

install(TARGETS QuantLib-Forge EXPORT QuantLibTargets)

foreach(file ${QLFORGE_HEADERS})
    install(FILES ${file} DESTINATION "${QL_INSTALL_INCLUDEDIR}/ql")
endforeach()


# Install a convenience config script for QuantLib-Forge, which allows users to
#
# find_package(QuantLib-Forge REQUIRED)
# target_link_libraries(myapp PRIVATE QuantLib::QuantLib)
#
# That is, it looks for the Forge dependency before it includes the QuantLibConfig file
#
# Without that, users have to first find_package(Forge) before find_package(QuantLib)
# to achieve the same.
#

set(QLFORGE_INSTALL_CMAKEDIR "lib/cmake/QuantLib-Forge" CACHE STRING
    "Installation directory for CMake scripts for QuantLib-Forge")

include(CMakePackageConfigHelpers)
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/../cmake/QuantLib-ForgeConfig.cmake.in"
    "${PROJECT_BINARY_DIR}/cmake/QuantLib-ForgeConfig.cmake"
    COPYONLY
)
configure_package_config_file("${CMAKE_CURRENT_SOURCE_DIR}/../cmake/QuantLib-ForgeConfig.cmake.in"
    "${PROJECT_BINARY_DIR}/cmake/QuantLib-ForgeConfig.cmake"
    INSTALL_DESTINATION "${QLFORGE_INSTALL_CMAKEDIR}"
)
install(FILES "${PROJECT_BINARY_DIR}/cmake/QuantLib-ForgeConfig.cmake"
    DESTINATION "${QLFORGE_INSTALL_CMAKEDIR}"
)
