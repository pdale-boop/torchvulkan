module rounding;

/*
Float arithmetic that rounds the same way on every driver. Vulkan allows a float division to be off
by 2.5 ulp, lets fma run as a separate multiply and add, and leaves the rounding mode of a float32 to
float16 conversion to the driver (AMD's is not round-to-nearest-even). Kernels that must match the
CPU bit for bit use these instead.

A 16-bit float kind is 1 for float16 and 2 for bfloat16 (0 means float32).

Kernels using divRounded must be compiled with fp_mode='precise' (scripts/config.py): otherwise slangc
ignores `precise`, so the driver may fuse Dekker's steps, and turns a division by a constant into a
multiplication by its reciprocal.
*/

// splits x into a high part with 12 significant bits and the exact remainder (Veltkamp)
void split(precise float x, out precise float hi, out precise float lo)
{
    precise float c = 4097.0 * x;
    hi = c - (c - x);
    lo = x - hi;
}

// a / b rounded to the nearest float: one correction step with the exact remainder a - q * b,
// computed with Dekker's product instead of fma, which may not be fused
public float divRounded(precise float a, precise float b)
{
    precise float q = a / b;
    precise float p = q * b;
    precise float qh, ql, bh, bl;
    split(q, qh, ql);
    split(b, bh, bl);
    precise float e = ((qh * bh - p) + qh * bl + ql * bh) + ql * bl;  // q * b - p, exactly
    precise float r = (a - p) - e;
    return q + r / b;
}

// float32 -> float16 bits, round to nearest even (Fabian Giesen's float_to_half_fast3_rtne)
public uint16_t floatToHalfBits(float f)
{
    uint x = asuint(f);
    uint sign = (x >> 16) & 0x8000;
    uint ax = x & 0x7fffffff;
    if (ax >= 0x7f800000) return uint16_t(sign | (ax > 0x7f800000 ? 0x7e00 : 0x7c00));  // inf, nan
    if (ax >= 0x477ff000) return uint16_t(sign | 0x7c00);                                // rounds to inf
    if (ax < 0x38800000)
    {
        // a float16 subnormal or zero: adding 0.5 lines the float's last bit up with float16's
        // smallest step, and the (correctly rounded) float add does the rounding
        precise float t = asfloat(ax) + 0.5;
        return uint16_t(sign | (asuint(t) - asuint(0.5)));
    }
    uint odd = (ax >> 13) & 1;
    ax += 0xc8000fff + odd;  // rebias the exponent and round half to even
    return uint16_t(sign | (ax >> 13));
}

// float32 -> bfloat16 bits, round to nearest even
public uint16_t floatToBf16Bits(float f)
{
    uint x = asuint(f);
    if ((x & 0x7fffffff) > 0x7f800000) return uint16_t((x >> 16) | 0x40);  // quiet nan
    return uint16_t((x + 0x7fff + ((x >> 16) & 1)) >> 16);
}

public float halfBitsToFloat(uint16_t h) { return f16tof32(uint(h)); }
public float bf16BitsToFloat(uint16_t h) { return asfloat(uint(h) << 16); }

public uint16_t floatTo16Bits(uint kind, float f) { return kind == 1 ? floatToHalfBits(f) : floatToBf16Bits(f); }
public float bits16ToFloat(uint kind, uint16_t h) { return kind == 1 ? halfBitsToFloat(h) : bf16BitsToFloat(h); }

// f rounded to the given float kind, as a float
public float roundToKind(uint kind, float f)
{
    if (kind == 0) return f;
    return bits16ToFloat(kind, floatTo16Bits(kind, f));
}
