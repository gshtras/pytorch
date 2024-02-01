#pragma once

/// Defines the Float8_e4m3fnuz type (8-bit floating-point) including
/// conversions to standard C types and basic arithmetic operations. Note that
/// arithmetic operations are implemented by converting to floating point and
/// performing the operation in float32.
/// Binary configuration remains the same as Float8_e4m3fn:
/// s eeee mmm
/// 1 sign bit
/// 4 exponent bits
/// 3 mantissa bits
/// The key differences versus Float8_e4m3fn are:
/// bias = 8
/// no infinities or negative zero
/// NaN only when sign bit is 1, rest all 0s
///
/// Implementation based on the paper https://arxiv.org/pdf/2206.02915.pdf and
/// the existing Float8_e4m3fn implementation.

#include <c10/macros/Macros.h>
#include <c10/util/C++17.h>
#include <c10/util/Float8_fnuz_cvt.h>
#include <c10/util/TypeSafeSignMath.h>
#include <c10/util/floating_point_utils.h>
#include <type_traits>

#if defined(__cplusplus) && (__cplusplus >= 201103L)
#include <cstdint>
#elif !defined(__OPENCL_VERSION__)
#include <math.h>
#include <stdint.h>
#endif

#include <iosfwd>
#include <ostream>

#if (defined(__HIP__) || defined(__HIPCC__)) && \
    (defined(__gfx940__) || defined(__gfx941__) || defined(__gfx942__))
#define __HIP__MI300__
#endif

namespace c10 {

namespace detail {
#ifdef __HIP__MI300__
// device specific optimized F8 down-conversion code
template <bool stochastic_rounding = false>
static C10_DEVICE uint8_t
fp8e4m3fnuz_from_fp32_value(float v, uint32_t rng = 0) {
  uint8_t i8data;
  union {
    float fval;
    uint32_t i32val;
    uint8_t i8val[4]; // NOTE: not endian independent
  } val;

  uint32_t ival = 0;
  val.fval = v;

  if ((val.i32val & 0x7F800000) !=
      0x7F800000) { /// propagate NAN/INF, no clipping
    val.fval = __builtin_amdgcn_fmed3f(val.fval, 240.0, -240.0);
  }
  if (stochastic_rounding) {
    ival = __builtin_amdgcn_cvt_sr_fp8_f32(val.fval, rng, ival, 0); // 0 pos
    val.i32val = ival;
    i8data = val.i8val[0]; // little endian
  } else // RNE CVT
  {
    ival = __builtin_amdgcn_cvt_pk_fp8_f32(
        val.fval,
        val.fval,
        ival,
        false); // false -> WORD0
    val.i32val = ival;
    i8data = val.i8val[0];
  }
  return i8data;
}
#endif // __HIP__MI300__

} // namespace detail

struct alignas(1) Float8_e4m3fnuz {
  uint8_t x;
  enum class rounding_mode { standard, stochastic };

  struct from_bits_t {};
  C10_HOST_DEVICE static constexpr from_bits_t from_bits() {
    return from_bits_t();
  }

  Float8_e4m3fnuz() = default;
  C10_HOST_DEVICE constexpr Float8_e4m3fnuz(const Float8_e4m3fnuz&) = default;

  constexpr C10_HOST_DEVICE Float8_e4m3fnuz(uint8_t bits, from_bits_t)
      : x(bits) {}

#ifdef __HIP__MI300__
  // NOTE: ON-DEVICE... always optimal bias
  C10_DEVICE Float8_e4m3fnuz(
      float v,
      rounding_mode rm = rounding_mode::standard,
      uint32_t rng = 0) {
    // runtime branch, use fp8e4m3fnuz_from_fp32_value if want to avoid it
    if (rm == rounding_mode::stochastic) {
      x = detail::fp8e4m3fnuz_from_fp32_value<true>(v, rng);
    } else {
      x = detail::fp8e4m3fnuz_from_fp32_value<false>(v);
    }
  }

  // Host only implementation using s/w simulation
  C10_HOST
#else // __HIP__MI300__
  // both Host and DEVICE for non-MI300 using s/w simulation
  C10_HOST_DEVICE
#endif // __HIP__MI300__
  Float8_e4m3fnuz(
      float v,
      rounding_mode rm = rounding_mode::standard,
      uint32_t rng = 0) {
    x = detail::fp8_fnuz_from_fp32_value<
        4,
        3,
        float,
        true /*negative_zero_nan*/,
        true /*clip*/>(v, (rm == rounding_mode::stochastic), rng);
  }

#ifdef __HIP__MI300__
  // upcast using device specific intrinsic
  inline C10_DEVICE operator float() const {
    float fval;
    uint32_t i32val = static_cast<uint32_t>(x);

    // upcast
    asm volatile("v_cvt_f32_fp8 %0, %1 src0_sel:BYTE_0"
                 : "=v"(fval)
                 : "v"(i32val));

    return fval;
  }

  inline C10_HOST operator float() const
#else // __HIP__MI300__
  inline C10_HOST_DEVICE operator float() const
#endif // __HIP__MI300__
  {
    return detail::
        fp8_fnuz_to_fp32_value<4, 3, float, true /*negative_zero_nan*/>(x);
  }

  inline C10_HOST_DEVICE bool isnan() const;
};

C10_API std::ostream& operator<<(
    std::ostream& out,
    const Float8_e4m3fnuz& value);
} // namespace c10

#include <c10/util/Float8_e4m3fnuz-inl.h> // IWYU pragma: keep
