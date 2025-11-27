/*******************************************************************************

   Declaration of exceptions.

   This file is part of QuantLib-Forge, a Forge integration layer for QuantLib.

   Copyright (C) 2025 The QuantLib-Forge Authors

   Derived from XAD (https://github.com/auto-differentiation/XAD)
   Original code: Copyright (C) 2010-2024 Xcelerit Computing Ltd.

   SPDX-License-Identifier: AGPL-3.0-or-later

******************************************************************************/

#pragma once

#include <stdexcept>
#include <string>

namespace forge { namespace expr {

class Exception : public std::runtime_error
{
  public:
    explicit Exception(const std::string& msg) : std::runtime_error(msg) {}
};

class TapeAlreadyActive : public Exception
{
  public:
    TapeAlreadyActive() : Exception("A tape is already active for the current thread") {}
};

class OutOfRange : public Exception
{
  public:
    explicit OutOfRange(const std::string& msg) : Exception(msg) {}
};

class DerivativesNotInitialized : public Exception
{
  public:
    explicit DerivativesNotInitialized(
        const std::string& msg = "At least one derivative must be set before computing adjoints")
        : Exception(msg)
    {
    }
};

class NoTapeException : public Exception
{
  public:
    explicit NoTapeException(const std::string& msg = "No active tape for the current thread")
        : Exception(msg)
    {
    }
};
#define FEXPR_NO_TAPE_EXCEPTION_DEFINED

}}  // namespace forge::expr

