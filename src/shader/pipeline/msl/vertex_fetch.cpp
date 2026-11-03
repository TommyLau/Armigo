// ============================================================================
// Armigo Shader - Native Shader Translation
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include <armigo/shader/vertex_fetch.h>

namespace armigo::shader {

// The fetch reads one attribute's components with loads of the component's
// own size (the replay refuses an attribute not aligned to it), returns zeros
// for a disabled attribute or one that would read past its stream, computes
// the register words the format writes, and fills missing components with
// (0, 0, 0, 1) in the register's type.
// Normalized values are correctly rounded quotients, as Metal's vertex fetch
// computes them, also when the stage compiles with fast math. A stage uses
// only the functions of its input classes; the others are marked unused so
// the package build's warnings-as-errors compile accepts it.
std::string_view vertex_fetch_msl() {
    return R"(struct ArmigoVertexFetch {
    device const uchar* stream[16];
    uint stride[16];
    uint step[16];
    uint limit[16];
    uint attribute[16];
};

struct ArmigoVertexRaw {
    uint4 bits;
    uint kind;
    uint count;
    bool scaled;
};

[[maybe_unused]]
static inline uint armigo_vertex_component(device const uchar* base, uint size, uint component) {
    if (size == 1u) return uint(base[component]);
    if (size == 2u) return uint(reinterpret_cast<device const ushort*>(base)[component]);
    return reinterpret_cast<device const uint*>(base)[component];
}

[[maybe_unused]]
static inline ArmigoVertexRaw armigo_vertex_raw(constant ArmigoVertexFetch& fetch, uint location, uint vertex_id,
                                                uint instance_id, uint base_instance) {
    ArmigoVertexRaw raw{uint4(0u), 0u, 0u, false};
    const uint code = fetch.attribute[location];
    const uint kind = code & 31u;
    if (kind == 0u) return raw;
    const uint count = (code >> 5) & 7u;
    const uint stream = (code >> 8) & 15u;
    const uint step = fetch.step[stream];
    const uint index = step == 0u ? vertex_id : base_instance + (instance_id - base_instance) / step;
    // Component bytes: 8-bit kinds and BGRA8, 16-bit kinds, 32-bit kinds and
    // the packed kinds (one 32-bit word).
    const uint size = kind <= 4u || kind == 16u ? 1u : kind <= 9u ? 2u : 4u;
    const bool packed = kind >= 13u && kind <= 15u;
    const ulong offset = ulong(index) * fetch.stride[stream] + ((code >> 12) & 0x7ffffu);
    if (offset + (packed ? 4u : size * count) > fetch.limit[stream]) return raw;
    device const uchar* base = fetch.stream[stream] + offset;
    raw.kind = kind;
    raw.count = count;
    raw.scaled = (code >> 31) != 0u;
    raw.bits.x = armigo_vertex_component(base, size, 0u);
    if (packed) return raw;
    if (count > 1u) raw.bits.y = armigo_vertex_component(base, size, 1u);
    if (count > 2u) raw.bits.z = armigo_vertex_component(base, size, 2u);
    if (count > 3u) raw.bits.w = armigo_vertex_component(base, size, 3u);
    return raw;
}

[[maybe_unused]]
static inline int4 armigo_vertex_sext(uint4 bits, uint width) {
    const uint shift = 32u - width;
    return as_type<int4>(bits << shift) >> int(shift);
}

// Missing components read (0, 0, 0, one).
[[maybe_unused]]
static inline uint4 armigo_vertex_defaults(uint4 value, uint count, uint one) {
    const uint4 component = uint4(0u, 1u, 2u, 3u);
    return select(uint4(0u, 0u, 0u, one), value, component < uint4(count));
}

// The value of a kind stored as float, normalized or packed data.
[[maybe_unused]]
static inline float4 armigo_vertex_float(ArmigoVertexRaw raw) {
    const uint4 b = raw.bits;
    switch (raw.kind) {
    case 1u: return precise::divide(float4(b), 255.0f);
    case 2u: return max(precise::divide(float4(armigo_vertex_sext(b, 8u)), 127.0f), -1.0f);
    case 5u: return precise::divide(float4(b), 65535.0f);
    case 6u: return max(precise::divide(float4(armigo_vertex_sext(b, 16u)), 32767.0f), -1.0f);
    case 9u: return float4(as_type<half4>(ushort4(b)));
    case 10u: return as_type<float4>(b);
    case 13u: {
        const uint word = b.x;
        return precise::divide(float4(uint4(word, word >> 10, word >> 20, word >> 30) &
                                      uint4(1023u, 1023u, 1023u, 3u)),
                               float4(1023.0f, 1023.0f, 1023.0f, 3.0f));
    }
    case 14u: {
        const uint word = b.x;
        const int4 signed_value = as_type<int4>(uint4(word << 22, word << 12, word << 2, word)) >> int4(22, 22, 22, 30);
        // Metal scales this format by the rounded reciprocal of 511.
        return max(float4(signed_value) * float4(1.0f / 511.0f, 1.0f / 511.0f, 1.0f / 511.0f, 1.0f), -1.0f);
    }
    case 15u: {
        const uint word = b.x;
        return float4(float(as_type<half>(ushort((word & 0x7ffu) << 4))),
                      float(as_type<half>(ushort(((word >> 11) & 0x7ffu) << 4))),
                      float(as_type<half>(ushort(((word >> 22) & 0x3ffu) << 5))), 1.0f);
    }
    case 16u: return precise::divide(float4(b.zyxw), 255.0f);
    default: return float4(0.0f);
    }
}

// The register words the attribute's format writes: integer storage writes
// the zero- or sign-extended integer, or its float value when scaled; every
// other kind writes float bits.
[[maybe_unused]]
static inline uint4 armigo_vertex_register(ArmigoVertexRaw raw) {
    const uint4 b = raw.bits;
    constexpr uint float_one = 0x3f800000u;
    switch (raw.kind) {
    case 0u: return uint4(0u);
    case 3u: case 7u: case 11u:
        return raw.scaled ? armigo_vertex_defaults(as_type<uint4>(float4(b)), raw.count, float_one)
                          : armigo_vertex_defaults(b, raw.count, 1u);
    case 4u: case 8u: case 12u: {
        const int4 value = raw.kind == 4u ? armigo_vertex_sext(b, 8u)
            : raw.kind == 8u ? armigo_vertex_sext(b, 16u) : as_type<int4>(b);
        return raw.scaled ? armigo_vertex_defaults(as_type<uint4>(float4(value)), raw.count, float_one)
                          : armigo_vertex_defaults(as_type<uint4>(value), raw.count, 1u);
    }
    default: return armigo_vertex_defaults(as_type<uint4>(armigo_vertex_float(raw)), raw.count, float_one);
    }
}

[[maybe_unused]]
static inline float4 armigo_vertex_fetch_float(constant ArmigoVertexFetch& fetch, uint location, uint vertex_id,
                                               uint instance_id, uint base_instance) {
    return as_type<float4>(armigo_vertex_register(
        armigo_vertex_raw(fetch, location, vertex_id, instance_id, base_instance)));
}

[[maybe_unused]]
static inline uint4 armigo_vertex_fetch_uint(constant ArmigoVertexFetch& fetch, uint location, uint vertex_id,
                                             uint instance_id, uint base_instance) {
    return armigo_vertex_register(armigo_vertex_raw(fetch, location, vertex_id, instance_id, base_instance));
}

[[maybe_unused]]
static inline int4 armigo_vertex_fetch_int(constant ArmigoVertexFetch& fetch, uint location, uint vertex_id,
                                           uint instance_id, uint base_instance) {
    return as_type<int4>(armigo_vertex_register(
        armigo_vertex_raw(fetch, location, vertex_id, instance_id, base_instance)));
}
)";
}

} // namespace armigo::shader
