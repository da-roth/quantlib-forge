/*******************************************************************************

   Literal AD types for all modes.

   This file is part of QuantLib-Forge, a Forge integration layer for QuantLib.

   Copyright (C) 2025 The QuantLib-Forge Authors

   Derived from XAD (https://github.com/auto-differentiation/XAD)
   Original code: Copyright (C) 2010-2024 Xcelerit Computing Ltd.

   SPDX-License-Identifier: AGPL-3.0-or-later

******************************************************************************/

#pragma once

#include <expressions/Expression.hpp>
#include <expressions/Exceptions.hpp>
#include <expressions/Traits.hpp>

// ========== Forge Integration: Add Forge includes ==========
#include <types/fdouble.hpp>
#include <graph/handles.hpp>  // For NodeId type
// ===========================================================

#include <algorithm>
#include <complex>
#include <iosfwd>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>

namespace forge { namespace expr {

// Minimal inline macros (replaces Macros.hpp dependency)
#ifdef _WIN32
#define FEXPR_FORCE_INLINE __forceinline
#else
#define FEXPR_FORCE_INLINE __attribute__((always_inline)) inline
#endif

#ifdef FEXPR_USE_STRONG_INLINE
#define FEXPR_INLINE FEXPR_FORCE_INLINE
#else
#define FEXPR_INLINE inline
#endif
// Forward declaration for AReal (needed by TapeStub)
template <class, std::size_t>
struct AReal;

// Minimal TapeStub class definition for template file
// Provides complete interface matching Tape class for template compatibility
// Users will implement their own Tape class to replace this stub
// All methods throw runtime errors to indicate they need implementation
template <class Scalar, std::size_t N>
class TapeStub {
public:
    typedef unsigned int slot_type;
    typedef Scalar value_type;
    typedef AReal<Scalar, N> active_type;
    typedef typename DerivativesTraits<Scalar, N>::type derivative_type;
    static constexpr slot_type INVALID_SLOT = slot_type(-1);

    // Static method needed for compilation (called via tape_type::getActive() throughout code)
    static TapeStub* getActive() { return nullptr; }

    // Tape interface methods - all throw runtime errors in template
    // Users implementing their own tape backend will provide the full implementation

    void activate() {
        throw NoTapeException("Template file: Tape backend not implemented. Users must provide their own Tape implementation.");
    }

    void deactivate() {
        throw NoTapeException("Template file: Tape backend not implemented. Users must provide their own Tape implementation.");
    }

    bool isActive() const { return false; }

    static void setActive(TapeStub* t) {
        (void)t;  // Suppress unused parameter
        throw NoTapeException("Template file: Tape backend not implemented. Users must provide their own Tape implementation.");
    }

    static void deactivateAll() {
        // No-op for template
    }

    void registerInput(active_type& inp) {
        (void)inp;  // Suppress unused parameter
        throw NoTapeException("Template file: Tape backend not implemented. Users must provide their own Tape implementation.");
    }

    void registerInput(std::complex<active_type>& inp) {
        (void)inp;  // Suppress unused parameter
        throw NoTapeException("Template file: Tape backend not implemented. Users must provide their own Tape implementation.");
    }

    template <class It>
    void registerInputs(It first, It last) {
        (void)first;
        (void)last;
        throw NoTapeException("Template file: Tape backend not implemented. Users must provide their own Tape implementation.");
    }

    template <class Container>
    void registerInputs(Container& v) {
        (void)v;
        throw NoTapeException("Template file: Tape backend not implemented. Users must provide their own Tape implementation.");
    }

    void registerOutput(active_type& outp) {
        (void)outp;  // Suppress unused parameter
        throw NoTapeException("Template file: Tape backend not implemented. Users must provide their own Tape implementation.");
    }

    void registerOutput(std::complex<active_type>& outp) {
        (void)outp;  // Suppress unused parameter
        throw NoTapeException("Template file: Tape backend not implemented. Users must provide their own Tape implementation.");
    }

    template <class It>
    void registerOutputs(It first, It last) {
        (void)first;
        (void)last;
        throw NoTapeException("Template file: Tape backend not implemented. Users must provide their own Tape implementation.");
    }

    template <class Container>
    void registerOutputs(Container& v) {
        (void)v;
        throw NoTapeException("Template file: Tape backend not implemented. Users must provide their own Tape implementation.");
    }

    void newRecording() {
        throw NoTapeException("Template file: Tape backend not implemented. Users must provide their own Tape implementation.");
    }

    void computeAdjoints() {
        throw NoTapeException("Template file: Tape backend not implemented. Users must provide their own Tape implementation.");
    }

    void clearAll() {
        throw NoTapeException("Template file: Tape backend not implemented. Users must provide their own Tape implementation.");
    }

    void clearDerivatives() {
        throw NoTapeException("Template file: Tape backend not implemented. Users must provide their own Tape implementation.");
    }

    const derivative_type& derivative(slot_type s) const {
        (void)s;  // Suppress unused parameter
        throw NoTapeException("Template file: Tape backend not implemented. Users must provide their own Tape implementation.");
    }

    derivative_type& derivative(slot_type s) {
        (void)s;  // Suppress unused parameter
        throw NoTapeException("Template file: Tape backend not implemented. Users must provide their own Tape implementation.");
    }

    derivative_type getDerivative(slot_type s) const {
        return derivative(s);
    }

    void setDerivative(slot_type s, const derivative_type& d) {
        (void)s;
        (void)d;
        throw NoTapeException("Template file: Tape backend not implemented. Users must provide their own Tape implementation.");
    }

    void setDerivative(slot_type s, derivative_type&& d) {
        (void)s;
        (void)d;
        throw NoTapeException("Template file: Tape backend not implemented. Users must provide their own Tape implementation.");
    }
};

// Helper types and constants needed by Literals.hpp
namespace detail {
    typedef unsigned int slot_type;
    static constexpr slot_type INVALID_SLOT_VALUE = slot_type(-1);  // Match Tape.hpp style
}
// Forward declarations for FReal (not used by QuantLib-Risks-Cpp, but needed for template compilation)
template <class, std::size_t>
struct FReal;
template <class Scalar, std::size_t N>
struct FRealTraits
{
    using type = FReal<Scalar, N>;
    using derivative_type = Vec<Scalar, N>;
};
template <class Scalar>
struct FRealTraits<Scalar, 1>
{
    using type = FReal<Scalar, 1>;
    using derivative_type = Scalar;
};

template <class Scalar, class Derived, class DerivativeType = Scalar>
struct ADTypeBase : public Expression<Scalar, Derived, DerivativeType>
{
    typedef typename ExprTraits<Derived>::value_type value_type;
    typedef typename ExprTraits<Derived>::nested_type nested_type;

    static_assert(std::is_floating_point<nested_type>::value,
                  "Active AD types only work with floating point");

    constexpr explicit FEXPR_INLINE ADTypeBase(Scalar val = Scalar()) : a_(val) {}
    constexpr FEXPR_INLINE ADTypeBase(ADTypeBase&& o) noexcept = default;
    constexpr FEXPR_INLINE ADTypeBase(const ADTypeBase& o) = default;
    FEXPR_INLINE ADTypeBase& operator=(ADTypeBase&& o) noexcept = default;
    FEXPR_INLINE ADTypeBase& operator=(const ADTypeBase& o) = default;
    FEXPR_INLINE ~ADTypeBase() = default;

    constexpr FEXPR_INLINE Scalar getValue() const { return value(); }
    FEXPR_INLINE const Scalar& value() const { return a_; }
    FEXPR_INLINE Scalar& value() { return a_; }

    template <class E>
    FEXPR_INLINE Derived& operator+=(const Expression<Scalar, E, DerivativeType>& x)
    {
        return derived() = (derived() + x);
    }
    template <class E>
    FEXPR_INLINE Derived& operator-=(const Expression<Scalar, E, DerivativeType>& x)
    {
        return derived() = (derived() - x);
    }
    template <class E>
    FEXPR_INLINE Derived& operator*=(const Expression<Scalar, E, DerivativeType>& x)
    {
        return derived() = (derived() * x);
    }
    template <class E>
    FEXPR_INLINE Derived& operator/=(const Expression<Scalar, E, DerivativeType>& x)
    {
        return derived() = (derived() / x);
    }

    FEXPR_INLINE Derived& operator+=(Scalar x)
    {
        a_ += x;
        return derived();
    }
    template <class I>
    FEXPR_INLINE typename std::enable_if<std::is_integral<I>::value, Derived>::type& operator+=(I x)
    {
        return *this += Scalar(x);
    }
    FEXPR_INLINE Derived& operator-=(Scalar rhs)
    {
        a_ -= rhs;
        return derived();
    }
    template <class I>
    FEXPR_INLINE typename std::enable_if<std::is_integral<I>::value, Derived>::type& operator-=(I x)
    {
        return *this -= Scalar(x);
    }
    FEXPR_INLINE Derived& operator*=(Scalar x) { return derived() = (derived() * x); }
    template <class I>
    FEXPR_INLINE typename std::enable_if<std::is_integral<I>::value, Derived>::type& operator*=(I x)
    {
        return *this *= Scalar(x);
    }
    FEXPR_INLINE Derived& operator/=(Scalar x) { return derived() = (derived() / x); }
    template <class I>
    FEXPR_INLINE typename std::enable_if<std::is_integral<I>::value, Derived>::type& operator/=(I x)
    {
        return *this /= Scalar(x);
    }
    FEXPR_INLINE Derived& operator+=(const value_type& x) { return derived() = derived() + x; }
    FEXPR_INLINE Derived& operator-=(const value_type& x) { return derived() = derived() - x; }
    FEXPR_INLINE Derived& operator*=(const value_type& x) { return derived() = derived() * x; }
    FEXPR_INLINE Derived& operator/=(const value_type& x) { return derived() = derived() / x; }
    FEXPR_INLINE Derived& operator++() { return derived() = (derived() + Scalar(1)); }
    FEXPR_INLINE Derived operator++(int)
    {
        auto tmp = derived();
        derived() = (derived() + Scalar(1));
        return tmp;
    }
    FEXPR_INLINE Derived& operator--() { return derived() = (derived() - Scalar(1)); }
    FEXPR_INLINE Derived operator--(int)
    {
        auto tmp = derived();
        derived() = (derived() - Scalar(1));
        return tmp;
    }

  private:
    FEXPR_INLINE Derived& derived() { return static_cast<Derived&>(*this); }
    FEXPR_INLINE const Derived& derived() const { return static_cast<const Derived&>(*this); }

  protected:
    Scalar a_;
};

template <class, std::size_t>
struct AReal;
template <class, std::size_t>
struct ADVar;

template <class Scalar, std::size_t M>
struct ExprTraits<AReal<Scalar, M>>
{
    static const bool isExpr = true;
    static const int numVariables = 1;
    static const bool isForward = false;
    static const bool isReverse = true;
    static const bool isLiteral = true;
    static const Direction direction = Direction::DIR_REVERSE;
    static const std::size_t vector_size = M;

    typedef typename ExprTraits<Scalar>::nested_type nested_type;
    typedef AReal<Scalar, M> value_type;
    typedef Scalar scalar_type;
};

template <class Scalar, std::size_t M>
struct ExprTraits<ADVar<Scalar, M>> : public ExprTraits<AReal<Scalar, M>>
{
};

template <class Scalar, std::size_t N = 1>
struct AReal
    : public ADTypeBase<Scalar, AReal<Scalar, N>, typename DerivativesTraits<Scalar, N>::type>
{
    typedef TapeStub<Scalar, N> tape_type;
    typedef ADTypeBase<Scalar, AReal<Scalar, N>, typename DerivativesTraits<Scalar, N>::type>
        base_type;
    typedef detail::slot_type slot_type;
    typedef Scalar value_type;
    typedef typename ExprTraits<Scalar>::nested_type nested_type;
    typedef typename DerivativesTraits<Scalar, N>::type derivative_type;

    // Make sure we inherit all ADTypeBase operator overloads; we will provide
    // a few AReal-specific overrides for scalar operations, but we still want
    // the expression-based operators (+== with Expression, etc.) to remain
    // visible.
    using base_type::operator+=;
    using base_type::operator-=;

    // ========== Forge Integration: Add fdouble member ==========
    ::forge::fdouble forge_value_;
    // ===========================================================

    FEXPR_INLINE AReal(nested_type val = nested_type())
        : base_type(val), slot_(detail::INVALID_SLOT_VALUE), forge_value_(val) {}

    // explicit conversion from int (also used by static_cast) to avoid warnings
    template <class U>
    explicit AReal(U val, typename std::enable_if<std::is_integral<U>::value>::type* = 0)
        : base_type(static_cast<nested_type>(val)), slot_(detail::INVALID_SLOT_VALUE),
          forge_value_(static_cast<double>(val))
    {
    }

    FEXPR_INLINE AReal(const AReal& o) : base_type(), slot_(detail::INVALID_SLOT_VALUE),
        forge_value_(o.forge_value_)
    {
        // Template: No tape recording - users implement their own tape backend
        // Original logic: auto s = tape_type::getActive(); if (s && o.shouldRecord()) { ... }
        // For template: Just copy value, no recording
        this->a_ = o.getValue();
    }

    static constexpr slot_type INVALID_SLOT = detail::INVALID_SLOT_VALUE;
    FEXPR_INLINE slot_type getSlot() const { return slot_; }
    FEXPR_INLINE tape_type* getTape() const { return nullptr; }  // Template: no tape implementation

    FEXPR_INLINE AReal(AReal&& o) noexcept : base_type(static_cast<base_type&&>(o)), slot_(o.slot_),
        forge_value_(std::move(o.forge_value_))
    {
        o.slot_ = detail::INVALID_SLOT_VALUE;
    }

    FEXPR_INLINE AReal& operator=(AReal&& o) noexcept
    {
        // Move the scalar / base part
        static_cast<base_type&>(*this) = static_cast<base_type&&>(o);

        // Move Forge side as well so that the target keeps the same Forge
        // value / node as the source. Without this, move-assignment would
        // leave forge_value_ stale (e.g. still 0) even though a_ was updated.
        forge_value_ = std::move(o.forge_value_);

        // Object moved-from still gets destructor called, so this makes sure
        // that the old slot we had gets destructed (template stub semantics).
        std::swap(slot_, o.slot_);
        return *this;
    }

    FEXPR_INLINE ~AReal()
    {
        // Template: No tape cleanup - users implement their own tape backend
        // Original logic: if (auto tape = tape_type::getActive()) { ... }
        // No action needed in template version
    }

    FEXPR_INLINE AReal& operator=(const AReal& o);

    FEXPR_INLINE AReal& operator=(nested_type x)
    {
        this->a_ = x;
        // Template: No tape recording - users implement their own tape backend.
        // Forge integration: keep Forge passive side in sync and warn if we're
        // overwriting an active Forge value during recording (dropping out of
        // the Forge graph).
        /*if (forge_value_.isRecording() && forge_value_.isActive()) {
            static std::size_t warnCount = 0;
            if (warnCount < 100) {
                ++warnCount;
                std::cerr
                    << "[Forge][Warning] AReal::operator=(Scalar) called while Forge "
                    << "recording is active on an active value – this replaces the "
                    << "active Forge node with a passive constant (occurrence "
                    << warnCount << ")\n";
            }
        }*/
        // Reset Forge-side representation to a passive constant matching the
        // scalar value. This keeps forward values consistent even if the graph
        // dependency has been lost.
        forge_value_ = ::forge::fdouble(static_cast<double>(x));
        return *this;
    }

    // In-place scalar addition: keep AReal semantics (mutate this), but make
    // the Forge side visible and keep the passive value in sync.
    FEXPR_INLINE AReal& operator+=(Scalar x)
    {
        // Delegate to base class to update the scalar side.
        base_type::operator+=(x);

        // If this value was active during recording, using += drops the
        // dependency on inputs from the Forge graph. Warn so users can
        // refactor (e.g. use an expression-based update instead).
        /*if (forge_value_.isRecording() && forge_value_.isActive()) {
            static std::size_t warnCount = 0;
            if (warnCount < 100) {
                ++warnCount;
                std::cerr
                    << "[Forge][Warning] AReal::operator+=(Scalar) called while Forge "
                    << "recording is active on an active value – this performs an "
                    << "in-place scalar update and drops you out of the Forge graph "
                    << "(occurrence " << warnCount << ")\n";
            }
        }*/

        // Ensure Forge's passive value matches the new scalar, even if the
        // active node (if any) is no longer meaningful.
        forge_value_ = ::forge::fdouble(static_cast<double>(this->a_));
        return *this;
    }

    // In-place scalar subtraction: same pattern as operator+=.
    FEXPR_INLINE AReal& operator-=(Scalar rhs)
    {
        base_type::operator-=(rhs);
        /*
        if (forge_value_.isRecording() && forge_value_.isActive()) {
            static std::size_t warnCount = 0;
            if (warnCount < 100) {
                ++warnCount;
                std::cerr
                    << "[Forge][Warning] AReal::operator-=(Scalar) called while Forge "
                    << "recording is active on an active value – this performs an "
                    << "in-place scalar update and drops you out of the Forge graph "
                    << "(occurrence " << warnCount << ")\n";
            }
        }
        */

        forge_value_ = ::forge::fdouble(static_cast<double>(this->a_));
        return *this;
    }

    // Forge integration: detect when code is pulling a raw scalar out of an
    // AReal that is Forge-active while Forge is recording. This usually means
    // we're about to drop out of the Forge graph (e.g., by doing math on
    // doubles instead of on AReal / forge::fdouble). We emit a warning (but do
    // not throw) to help locate missing Forge wiring without breaking code.
    FEXPR_INLINE const Scalar& value() const
    {
        /*
        if (forge_value_.isRecording() && forge_value_.isActive()) {
            static std::size_t warnCount = 0;
            if (warnCount < 100) {
                ++warnCount;
                std::cerr
                    << "[Forge][Warning] AReal::value() called while Forge recording "
                    << "is active on an active value – this drops you out of the "
                    << "Forge graph (occurrence " << warnCount << ")\n";
            }
        }
        */
        return this->a_;
    }

    FEXPR_INLINE Scalar& value()
    {
        /*
        if (forge_value_.isRecording() && forge_value_.isActive()) {
            static std::size_t warnCount = 0;
            if (warnCount < 100) {
                ++warnCount;
                std::cerr
                    << "[Forge][Warning] AReal::value() called while Forge recording "
                    << "is active on an active value – this drops you out of the "
                    << "Forge graph (occurrence " << warnCount << ")\n";
            }
        }
        */
        return this->a_;
    }

    template <class Expr>
    FEXPR_INLINE AReal(const Expression<Scalar, Expr, derivative_type>&
                         expr);  // cppcheck-suppress noExplicitConstructor

    template <class Expr>
    FEXPR_INLINE AReal& operator=(const Expression<Scalar, Expr, derivative_type>& expr);

    FEXPR_INLINE void setDerivative(derivative_type a) { derivative() = a; }
    FEXPR_INLINE void setAdjoint(derivative_type a) { setDerivative(a); }
    FEXPR_INLINE derivative_type getAdjoint() const { return getDerivative(); }

    // ========== Forge Integration: Helper methods ==========
    /// Mark this variable as a differentiable input for Forge
    FEXPR_INLINE void markForgeInput() {
        forge_value_.markInputAndDiff();
    }

    /// Mark this variable as an output for Forge
    FEXPR_INLINE void markForgeOutput() {
        forge_value_.markOutput();
    }

    /// Get the Forge graph node ID for this variable
    FEXPR_INLINE ::forge::NodeId forgeNodeId() const {
        return forge_value_.node();
    }

    /// Get the underlying forge::fdouble (for advanced usage)
    FEXPR_INLINE const ::forge::fdouble& forgeValue() const {
        return forge_value_;
    }

    FEXPR_INLINE ::forge::fdouble& forgeValue() {
        return forge_value_;
    }

    /// Set the underlying forge::fdouble (used by ABool::If and similar helpers)
    FEXPR_INLINE void setForgeValue(const ::forge::fdouble& v) {
        forge_value_ = v;
    }
    // =======================================================

    template <int Size>
    FEXPR_FORCE_INLINE void pushRhs(DerivInfo<tape_type, Size>& info, const Scalar& mul,
                                  slot_type slot) const
    {
        // Template: No-op - users implement their own tape backend
        (void)info;
        (void)mul;
        (void)slot;
    }

    template <int Size>
    FEXPR_FORCE_INLINE void calc_derivatives(DerivInfo<tape_type, Size>& info, tape_type&,
                                           const Scalar& mul) const
    {
        // Template: No-op - users implement their own tape backend
        // Original logic: if (slot_ != INVALID_SLOT) pushRhs(info, mul, slot_);
        (void)info;  // Suppress unused parameter warning
        (void)mul;
    }

    template <int Size>
    FEXPR_FORCE_INLINE void calc_derivatives(DerivInfo<tape_type, Size>& info, tape_type&) const
    {
        // Template: No-op - users implement their own tape backend
        (void)info;  // Suppress unused parameter warning
    }

    FEXPR_INLINE derivative_type getDerivative() const { return derivative(); }

    FEXPR_INLINE const derivative_type& derivative() const
    {
        // Template: Throw runtime error - users must implement their own tape backend
        throw NoTapeException("Template file: Tape backend not implemented. Users must provide their own Tape implementation.");
    }

    FEXPR_INLINE derivative_type& derivative()
    {
        // Template: Throw runtime error - users must implement their own tape backend
        throw NoTapeException("Template file: Tape backend not implemented. Users must provide their own Tape implementation.");
    }
    FEXPR_INLINE bool shouldRecord() const { return slot_ != detail::INVALID_SLOT_VALUE; }

  private:
    template <int Size, typename Expr>
    FEXPR_FORCE_INLINE void pushAll(tape_type* t, const Expr& expr) const
    {
        // Template: No-op - users implement their own tape backend
        (void)t;
        (void)expr;
    }

    template <class T, std::size_t d__cnt>
    friend class Tape;
    detail::slot_type slot_;
};

// this class wraps AReal<T> and makes sure that no new copies are created on
// the Tape
// when this guy is copied (unlike the AReal<T> copy)
// therefore we can use auto = ... in expressions
template <class Scalar, std::size_t N = 1>
struct ADVar
    : public Expression<Scalar, ADVar<Scalar, N>, typename DerivativesTraits<Scalar, N>::type>
{
    typedef AReal<Scalar, N> areal_type;
    typedef typename areal_type::tape_type tape_type;

    FEXPR_INLINE explicit ADVar(const areal_type& a) : ar_(a), shouldRecord_(a.shouldRecord()) {}

    FEXPR_INLINE Scalar getValue() const { return ar_.getValue(); }

    FEXPR_INLINE const Scalar& value() const { return ar_.value(); }

    // ========== Forge Integration: expose Forge value for expressions ==========
    FEXPR_INLINE ::forge::fdouble forgeValue() const {
        return ar_.forgeValue();
    }
    // ===========================================================================

    template <int Size>
    FEXPR_INLINE void calc_derivatives(DerivInfo<tape_type, Size>& info, tape_type& s,
                                     const Scalar& mul) const
    {
        ar_.calc_derivatives(info, s, mul);
    }

    template <int Size>
    FEXPR_INLINE void calc_derivatives(DerivInfo<tape_type, Size>& info, tape_type& s) const
    {
        // Template: No-op (fixed typo from original: was calc_derivative, now calc_derivatives)
        (void)info;
        (void)s;
    }

    FEXPR_INLINE const typename areal_type::derivative_type& derivative() const
    {
        return ar_.derivative();
    }

    FEXPR_INLINE bool shouldRecord() const { return shouldRecord_; }

  private:
    areal_type const& ar_;
    bool shouldRecord_;
};

template <class Scalar, std::size_t M>
FEXPR_INLINE AReal<Scalar, M>& AReal<Scalar, M>::operator=(const AReal& o)
{
    // Template: No tape recording - users implement their own tape backend
    // Original logic: tape_type* s = tape_type::getActive(); if (s && ...) { ... }
    this->a_ = o.getValue();
    forge_value_ = o.forge_value_;  // ← Forge: Also copy forge value
    return *this;
}

template <class Scalar, std::size_t M>
template <class Expr>
FEXPR_INLINE AReal<Scalar, M>::AReal(
    const Expression<Scalar, Expr, typename DerivativesTraits<Scalar, M>::type>& expr)
    : base_type(expr.getValue()), slot_(detail::INVALID_SLOT_VALUE),
      forge_value_(expr.derived().forgeValue())  // ← Forge: Get forge result from expression
{
    // Template: No tape recording - users implement their own tape backend
    // Original logic: tape_type* s = tape_type::getActive(); if (s && expr.shouldRecord()) { ... }
}

template <class Scalar, std::size_t M>
template <class Expr>
FEXPR_INLINE AReal<Scalar, M>& AReal<Scalar, M>::operator=(
    const Expression<Scalar, Expr, typename DerivativesTraits<Scalar, M>::type>& expr)
{
    // Template: No tape recording - users implement their own tape backend
    // Original logic: tape_type* s = tape_type::getActive(); if (s && ...) { ... }
    this->a_ = expr.getValue();
    forge_value_ = expr.derived().forgeValue();  // ← Forge: Update forge value from expression
    return *this;
}

// FReal removed - not used by QuantLib-Risks-Cpp

template <class, std::size_t>
struct FRealDirect;

template <class Scalar, std::size_t N>
struct ExprTraits<FRealDirect<Scalar, N>>
{
    static const bool isExpr = false;
    static const int numVariables = 1;
    static const bool isForward = true;
    static const bool isReverse = false;
    static const bool isLiteral = true;
    static const Direction direction = Direction::DIR_FORWARD;
    static const std::size_t vector_size = N;

    typedef typename ExprTraits<Scalar>::nested_type nested_type;
    typedef FRealDirect<Scalar, N> value_type;
    typedef Scalar scalar_type;
};

template <class, std::size_t>
struct ARealDirect;

template <class Scalar, std::size_t N>
struct ExprTraits<ARealDirect<Scalar, N>>
{
    static const bool isExpr = false;
    static const int numVariables = 1;
    static const bool isForward = false;
    static const bool isReverse = true;
    static const bool isLiteral = true;
    static const Direction direction = Direction::DIR_REVERSE;
    static const std::size_t vector_size = N;

    typedef typename ExprTraits<Scalar>::nested_type nested_type;
    typedef ARealDirect<Scalar, N> value_type;
    typedef Scalar scalar_type;
};

// FReal removed - not used by QuantLib-Risks-Cpp

template <class Scalar, std::size_t M = 1>
FEXPR_INLINE const Scalar& value(const AReal<Scalar, M>& x)
{
    return x.value();
}

template <class Scalar, std::size_t M>
FEXPR_INLINE Scalar& value(AReal<Scalar, M>& x)
{
    return x.value();
}

// FReal value() functions removed - not used by QuantLib-Risks-Cpp

template <class T>
FEXPR_INLINE typename std::enable_if<!ExprTraits<T>::isExpr && std::is_arithmetic<T>::value, T>::type&
value(T& x)
{
    return x;
}

template <class T>
FEXPR_INLINE const typename std::enable_if<!ExprTraits<T>::isExpr && std::is_arithmetic<T>::value,
                                         T>::type&
value(const T& x)
{
    return x;
}

// FReal derivative() functions removed - not used by QuantLib-Risks-Cpp

template <class Scalar, std::size_t M = 1>
FEXPR_INLINE const typename AReal<Scalar, M>::derivative_type& derivative(const AReal<Scalar, M>& fr)
{
    return fr.derivative();
}

template <class Scalar, std::size_t M = 1>
FEXPR_INLINE typename AReal<Scalar, M>::derivative_type& derivative(AReal<Scalar, M>& fr)
{
    return fr.derivative();
}

template <class T>
FEXPR_INLINE typename std::enable_if<!ExprTraits<T>::isExpr && std::is_arithmetic<T>::value, T>::type
derivative(T&)
{
    return T();
}

template <class T>
FEXPR_INLINE typename std::enable_if<!ExprTraits<T>::isExpr && std::is_arithmetic<T>::value, T>::type
derivative(const T&)
{
    return T();
}

template <class C, class T, class Scalar, class Derived, class Deriv>
FEXPR_INLINE std::basic_ostream<C, T>& operator<<(std::basic_ostream<C, T>& os,
                                                const Expression<Scalar, Derived, Deriv>& x)
{
    return os << value(x);
}

template <class C, class T, class Scalar, std::size_t N>
FEXPR_INLINE std::basic_istream<C, T>& operator>>(std::basic_istream<C, T>& is, AReal<Scalar, N>& x)
{
    return is >> value(x);
}

// FReal stream operator removed - not used by QuantLib-Risks-Cpp

typedef AReal<double> AD;
typedef AReal<float> AF;

// FAD and FAF typedefs removed - FReal not used by QuantLib-Risks-Cpp

typedef ARealDirect<double, 1> ADD;
typedef ARealDirect<float, 1> AFD;

}}  // namespace forge::expr


#if __clang_major__ > 16 && defined(_LIBCPP_VERSION)

namespace std {

// to make libc++ happy when calling pow(AReal, int) and similar functions
template<class T, class T1, std::size_t N>
class __promote<forge::expr::AReal<T, N>, T1> {
public:
   using type = forge::expr::AReal<T, N>;
};

template<class T, class T1, std::size_t N>
class __promote<T1, forge::expr::AReal<T, N>> {
public:
   using type = forge::expr::AReal<T, N>;
};

template<class T, std::size_t N>
class __promote<forge::expr::AReal<T, N>, forge::expr::AReal<T, N>> {
public:
   using type = forge::expr::AReal<T, N>;
};

template<class T, class T1, class T2, std::size_t N>
class __promote<forge::expr::AReal<T, N>, T1, T2> {
public:
   using type = forge::expr::AReal<T, N>;
};

template<class T, class T1, class T2, std::size_t N>
class __promote<T1, forge::expr::AReal<T, N>, T2> {
public:
   using type = forge::expr::AReal<T, N>;
};

template<class T, class T1, class T2, std::size_t N>
class __promote<T1, T2, forge::expr::AReal<T, N>> {
public:
   using type = forge::expr::AReal<T, N>;
};

template<class T, class T2, std::size_t N>
class __promote<forge::expr::AReal<T, N>, forge::expr::AReal<T, N>, T2> {
public:
   using type = forge::expr::AReal<T, N>;
};

template<class T, class T2, std::size_t N>
class __promote<forge::expr::AReal<T, N>, T2, forge::expr::AReal<T, N>> {
public:
   using type = forge::expr::AReal<T, N>;
};

template<class T, class T2, std::size_t N>
class __promote<T2, forge::expr::AReal<T, N>, forge::expr::AReal<T, N>> {
public:
   using type = forge::expr::AReal<T, N>;
};

template<class T, std::size_t N>
class __promote<forge::expr::AReal<T, N>, forge::expr::AReal<T, N>, forge::expr::AReal<T, N>> {
public:
   using type = forge::expr::AReal<T, N>;
};

// FReal __promote specializations removed - not used by QuantLib-Risks-Cpp

}

#endif

