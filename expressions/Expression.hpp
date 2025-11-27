/*******************************************************************************

   Declare expression types.

   This file is part of QuantLib-Forge, a Forge integration layer for QuantLib.

   Copyright (C) 2025 The QuantLib-Forge Authors

   Derived from XAD (https://github.com/auto-differentiation/XAD)
   Original code: Copyright (C) 2010-2024 Xcelerit Computing Ltd.

   SPDX-License-Identifier: AGPL-3.0-or-later

******************************************************************************/

#pragma once
#include <expressions/Macros.hpp>

namespace forge { namespace expr {

// keeps information about multipliers and slots for an expression
// locally on stack
template <typename TapeType, int N>
struct DerivInfo
{
    unsigned index = 0;
    typename TapeType::value_type multipliers[N];
    typename TapeType::slot_type slots[N];
};

/// Represents a generic expression, for the Scalar base type.
///
/// It uses the CTRP pattern, where derived classes register themselves with
/// the base class in the second template parameter
template <class Scalar, class Derived, class DerivativeType = Scalar>
struct Expression
{
    /// get a reference to the derived object
    FEXPR_INLINE const Derived& derived() const { return static_cast<const Derived&>(*this); }

    /// get the value
    FEXPR_INLINE Scalar value() const { return derived().value(); }

    FEXPR_INLINE Scalar getValue() const { return value(); }

#ifdef FEXPR_ALLOW_INT_CONVERSION
    FEXPR_INLINE explicit operator char() const { return static_cast<char>(getValue()); }
    FEXPR_INLINE explicit operator unsigned char() const
    {
        return static_cast<unsigned char>(getValue());
    }
    FEXPR_INLINE explicit operator signed char() const
    {
        return static_cast<signed char>(getValue());
    }
    FEXPR_INLINE explicit operator short() const { return static_cast<short>(getValue()); }
    FEXPR_INLINE explicit operator unsigned short() const
    {
        return static_cast<unsigned short>(getValue());
    }
    FEXPR_INLINE explicit operator int() const { return static_cast<int>(getValue()); }
    FEXPR_INLINE explicit operator unsigned int() const
    {
        return static_cast<unsigned int>(getValue());
    }
    FEXPR_INLINE explicit operator long() const { return static_cast<long>(getValue()); }
    FEXPR_INLINE explicit operator unsigned long() const
    {
        return static_cast<unsigned long>(getValue());
    }
    FEXPR_INLINE explicit operator long long() const { return static_cast<long long>(getValue()); }
    FEXPR_INLINE explicit operator unsigned long long() const
    {
        return static_cast<unsigned long long>(getValue());
    }
#endif

    // convert to boolean
    FEXPR_INLINE explicit operator bool() const { return value() != Scalar(0); }

    /// calculate the derivatives, given a tape object
    template <class Tape, int Size>
    FEXPR_INLINE void calc_derivatives(DerivInfo<Tape, Size>& info, Tape& s) const
    {
        derived().calc_derivatives(info, s, Scalar(1));
    }

    /// calculate the derivatives, given tape and multiplier
    template <class Tape, int Size>
    FEXPR_INLINE void calc_derivatives(DerivInfo<Tape, Size>& info, Tape& s,
                                     const Scalar& multiplier) const
    {
        derived().calc_derivatives(info, s, multiplier);
    }

    FEXPR_INLINE bool shouldRecord() const { return derived().shouldRecord(); }

    FEXPR_INLINE DerivativeType derivative() const { return derived().derivative(); }
};

template <class Scalar, class Expr, class DerivativeType>
FEXPR_INLINE Scalar value(const Expression<Scalar, Expr, DerivativeType>& expr)
{
    return expr.value();
}

template <class Scalar, class Expr, class DerivativeType>
FEXPR_INLINE DerivativeType derivative(const Expression<Scalar, Expr, DerivativeType>& expr)
{
    return expr.derivative();
}

}}  // namespace forge::expr

