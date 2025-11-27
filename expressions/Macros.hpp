/*******************************************************************************

   Utility macro declarations.

   This file is part of QuantLib-Forge, a Forge integration layer for QuantLib.

   Copyright (C) 2025 The QuantLib-Forge Authors

   Derived from XAD (https://github.com/auto-differentiation/XAD)
   Original code: Copyright (C) 2010-2024 Xcelerit Computing Ltd.

   SPDX-License-Identifier: AGPL-3.0-or-later

******************************************************************************/

#pragma once

#include <expressions/Config.hpp>

namespace forge { namespace expr {
namespace detail
{
template <class T>
void ignore_unused_variable(const T&)
{
}
}  // namespace detail
}}  // namespace forge::expr

#define FEXPR_UNUSED_VARIABLE(x) ::forge::expr::detail::ignore_unused_variable(x)

#ifdef _WIN32
#define FEXPR_FORCE_INLINE __forceinline
#define FEXPR_NEVER_INLINE __declspec(noinline)
#else
#define FEXPR_FORCE_INLINE __attribute__((always_inline)) inline
#define FEXPR_NEVER_INLINE __attribute__((noinline))
#endif

#if defined(__GNUC__) || defined(__clang__)
#define FEXPR_LIKELY(x) __builtin_expect(!!(x), 1)
#define FEXPR_UNLIKELY(x) __builtin_expect(!!(x), 0)
#if defined(__GNUC__) && __GNUC__ >= 9
#define FEXPR_VERY_LIKELY(x) __builtin_expect_with_probability(!!(x), 1, 0.999)
#define FEXPR_VERY_UNLIKELY(x) __builtin_expect_with_probability(!!(x), 0, 0.999)
#else
#define FEXPR_VERY_LIKELY(x) FEXPR_LIKELY(x)
#define FEXPR_VERY_UNLIKELY(x) FEXPR_UNLIKELY(x)
#endif
#else
#define FEXPR_LIKELY(x) (x)
#define FEXPR_UNLIKELY(x) (x)
#define FEXPR_VERY_LIKELY(x) (x)
#define FEXPR_VERY_UNLIKELY(x) (x)
#endif

#ifdef FEXPR_USE_STRONG_INLINE
#define FEXPR_INLINE FEXPR_FORCE_INLINE
#else
#define FEXPR_INLINE inline
#endif

#ifdef FEXPR_NO_THREADLOCAL
#define FEXPR_THREAD_LOCAL
#else
// we can't use thread_local here, as MacOS has an issue with that
#ifdef _WIN32
#define FEXPR_THREAD_LOCAL __declspec(thread)
#else
#define FEXPR_THREAD_LOCAL __thread
#endif
#endif

