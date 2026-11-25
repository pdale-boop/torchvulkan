module float8;

import core.indexing;

// float8 codes -> a 2- or 4-byte dtype in one pass: each code indexes a 256-entry table of the destination's bit patterns,
// which the host builds with c10's own conversions, so the result matches the CPU bit for bit. T only sets the output width.

[[vk::constant_id(0)]] const uint WG_x = 256;
// source and destination are 16-byte aligned, so whole runs of 16 codes load and store as uint4 (loadAligned /
// storeAligned say so: a plain uint4* access only promises 4 bytes)
[[vk::constant_id(1)]] const uint IS_ALIGNED = 0;

struct Float8DecodePushConstants
{
    uint64_t src;
    uint64_t dst;
    uint64_t table;
    uint64_t numel;
}

static groupshared uint table_bits[256];

uint codeAt(uint4 codes, uint i)
{
    return (codes[i / 4] >> (8 * (i % 4))) & 0xffu;
}

[shader("compute")]
[numthreads(WG_x, 1, 1)]
void float8_decode_main<T>(
    uint3 gi : SV_DispatchThreadID,
    uint3 li : SV_GroupThreadID,
    uniform Float8DecodePushConstants pcs)
{
    for (uint i = li.x; i < 256; i += WG_x) table_bits[i] = ((uint*)pcs.table)[i];
    GroupMemoryBarrierWithGroupSync();

    uint64_t first = uint64_t(flat_thread(gi, WG_x)) * 16;
    if (first >= pcs.numel) return;

    if (IS_ALIGNED == 1 && first + 16 <= pcs.numel)
    {
        uint4 codes = loadAligned<16>((uint4*)(pcs.src + first));
        if (sizeof(T) == 2)
        {
            [unroll] for (uint q = 0; q < 2; q++)
            {
                uint4 out;
                [unroll] for (uint w = 0; w < 4; w++)
                {
                    uint i = q * 8 + w * 2;
                    out[w] = table_bits[codeAt(codes, i)] | (table_bits[codeAt(codes, i + 1)] << 16);
                }
                storeAligned<16>((uint4*)(pcs.dst + first * 2 + q * 16), out);
            }
        }
        else
        {
            [unroll] for (uint q = 0; q < 4; q++)
            {
                uint4 out;
                [unroll] for (uint w = 0; w < 4; w++) out[w] = table_bits[codeAt(codes, q * 4 + w)];
                storeAligned<16>((uint4*)(pcs.dst + first * 4 + q * 16), out);
            }
        }
        return;
    }

    for (uint64_t i = first; i < min(first + 16, pcs.numel); i++)
    {
        uint bits = table_bits[uint(((uint8_t*)pcs.src)[i])];
        if (sizeof(T) == 2) ((uint16_t*)pcs.dst)[i] = uint16_t(bits);
        else ((uint*)pcs.dst)[i] = bits;
    }
}

// float32, float16 or bfloat16 -> float8_e4m3fn codes in one pass, with c10's fp8e4m3fn_from_fp32_value bit for bit:
// round to nearest even, saturate finite overflow and infinities to +-448, NaN to 0x7f. T only sets the input width;
// IS_BF16 says a 2-byte input is bfloat16 rather than float16.
[[vk::constant_id(2)]] const uint IS_BF16 = 0;

uint encodeE4M3(float f)
{
    uint bits = asuint(f);
    uint sign = bits & 0x80000000u;
    bits ^= sign;
    // all three cases computed and selected, so the 16 conversions of a thread don't become 16 branchy regions that the
    // compiler issues the loads between
    uint overflow = select(bits > 0x7F800000u, 0x7fu, 0x7eu);
    // under 2^-6: let the float add round the mantissa into place (denorm_mask = 2^14)
    uint small = (asuint(asfloat(bits) + asfloat(141u << 23)) - (141u << 23)) & 0xffu;
    uint normal = ((bits + ((7u - 127u) << 23) + 0x7FFFFu + ((bits >> 20) & 1u)) >> 20) & 0xffu;
    normal = select(normal == 0x7fu, 0x7eu, normal);
    uint result = select(bits >= (1087u << 20), overflow, select(bits < (121u << 23), small, normal));
    return result | (sign >> 24);
}

float wideValue(uint bits16)
{
    if (IS_BF16 == 1) return asfloat(bits16 << 16);
    // a float16 NaN keeps its sign (NVIDIA's conversion drops it, and c10 encodes -NaN as 0xff)
    if ((bits16 & 0x7c00u) == 0x7c00u && (bits16 & 0x3ffu) != 0) return asfloat(((bits16 & 0x8000u) << 16) | 0x7fc00000u);
    return f16tof32(bits16);
}

[shader("compute")]
[numthreads(WG_x, 1, 1)]
void float8_encode_main<T>(
    uint3 gi : SV_DispatchThreadID,
    uniform Float8DecodePushConstants pcs) // src, dst (codes), unused table, numel
{
    uint64_t first = uint64_t(flat_thread(gi, WG_x)) * 16;
    if (first >= pcs.numel) return;

    if (IS_ALIGNED == 1 && first + 16 <= pcs.numel)
    {
        uint4 codes;
        [unroll] for (uint q = 0; q < 4; q++)
        {
            // codes[q] holds values 4q .. 4q + 3
            uint word = 0;
            if (sizeof(T) == 2)
            {
                uint4 in4 = loadAligned<16>((uint4*)(pcs.src + first * 2 + (q / 2) * 16));
                [unroll] for (uint b = 0; b < 4; b++)
                {
                    uint pair = in4[(q % 2) * 2 + b / 2];
                    word |= encodeE4M3(wideValue((b % 2) == 0 ? (pair & 0xffffu) : (pair >> 16))) << (8 * b);
                }
            }
            else
            {
                uint4 in4 = loadAligned<16>((uint4*)(pcs.src + first * 4 + q * 16));
                [unroll] for (uint b = 0; b < 4; b++) word |= encodeE4M3(asfloat(in4[b])) << (8 * b);
            }
            codes[q] = word;
        }
        storeAligned<16>((uint4*)(pcs.dst + first), codes);
        return;
    }

    for (uint64_t i = first; i < min(first + 16, pcs.numel); i++)
    {
        float v = sizeof(T) == 2 ? wideValue(uint(((uint16_t*)pcs.src)[i])) : asfloat(((uint*)pcs.src)[i]);
        ((uint8_t*)pcs.dst)[i] = uint8_t(encodeE4M3(v));
    }
}
