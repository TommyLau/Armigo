// ============================================================================
// Armigo Graphics - Vertex Input Fetch
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================
// Vertex registers and encoded formats are decoded into an owned portable plan.
// Unsupported state returns a structured error instead of a fallback attribute.

#include <armigo/graphics/vertex_input.h>
#include <limits>

namespace armigo::graphics {
namespace {

CommandResult fail(CommandStatus status, std::string_view detail,
                   std::uint64_t address = 0) noexcept {
    return {status, detail, address, 0, 0, 0, 0, engine_class::threed};
}

// Maxwell packs the attribute format as a six bit size and three bit type.
constexpr std::uint32_t encoded_format_mask = (63u << 21) | (7u << 27);

enum class ComponentWidth : std::uint8_t { none, x8, x16, x32 };

struct SizeShape {
    ComponentWidth width = ComponentWidth::none;
    std::uint8_t components = 0;
    std::uint8_t size_bytes = 0;
};

// Size codes of the Maxwell attribute format field that carry a packed or
// component layout Metal fetches natively.
std::optional<SizeShape> size_shape(std::uint32_t size) noexcept {
    switch (size) {
    case 0x1d: return SizeShape{ComponentWidth::x8, 1, 1};   // R8
    case 0x18: return SizeShape{ComponentWidth::x8, 2, 2};   // R8G8
    case 0x13: return SizeShape{ComponentWidth::x8, 3, 3};   // R8G8B8
    case 0x0a: return SizeShape{ComponentWidth::x8, 4, 4};   // R8G8B8A8
    case 0x1b: return SizeShape{ComponentWidth::x16, 1, 2};  // R16
    case 0x0f: return SizeShape{ComponentWidth::x16, 2, 4};  // R16G16
    case 0x05: return SizeShape{ComponentWidth::x16, 3, 6};  // R16G16B16
    case 0x03: return SizeShape{ComponentWidth::x16, 4, 8};  // R16G16B16A16
    case 0x12: return SizeShape{ComponentWidth::x32, 1, 4};  // R32
    case 0x04: return SizeShape{ComponentWidth::x32, 2, 8};  // R32G32
    case 0x02: return SizeShape{ComponentWidth::x32, 3, 12}; // R32G32B32
    case 0x01: return SizeShape{ComponentWidth::x32, 4, 16}; // R32G32B32A32
    case 0x30: return SizeShape{ComponentWidth::x32, 4, 4};  // A2B10G10R10
    case 0x31: return SizeShape{ComponentWidth::x32, 3, 4};  // B10G11R11
    default: return std::nullopt;
    }
}

VertexAttribFormatInfo info(VertexAttribFormat format, VertexFormatClass format_class,
                            const SizeShape& shape) noexcept {
    return {format, format_class, shape.components, shape.size_bytes};
}

// Builds the format for one component width and count, applying the Maxwell
// type. Normalized and float types convert to float; integer types keep raw
// bits, so only their class can feed an integer shader declaration. The enum
// groups the component formats first (1, 2, 3, 4 components), each with the
// same sub-order: 8-bit unorm/snorm/uint/sint, 16-bit float/unorm/snorm/
// uint/sint, 32-bit float/uint/sint.
std::optional<VertexAttribFormatInfo> typed_format(ComponentWidth width, std::uint8_t components,
                                                   std::uint32_t type) noexcept {
    const unsigned width_bytes = width == ComponentWidth::x8 ? 1 : width == ComponentWidth::x16 ? 2 : 4;
    const SizeShape shape{width, components,
                          static_cast<std::uint8_t>(components * width_bytes)};
    const auto entry = [&](unsigned count) {
        const unsigned block = (components - 1u) * 12u;
        return static_cast<VertexAttribFormat>(block + count);
    };
    switch (width) {
    case ComponentWidth::x8: {
        switch (type) {
        case 2: return info(entry(0), VertexFormatClass::float_class, shape);
        case 1: return info(entry(1), VertexFormatClass::float_class, shape);
        case 4: return info(entry(2), VertexFormatClass::uint_class, shape);
        case 3: return info(entry(3), VertexFormatClass::sint_class, shape);
        default: return std::nullopt;
        }
    }
    case ComponentWidth::x16: {
        switch (type) {
        case 7: return info(entry(4), VertexFormatClass::float_class, shape);
        case 2: return info(entry(5), VertexFormatClass::float_class, shape);
        case 1: return info(entry(6), VertexFormatClass::float_class, shape);
        case 4: return info(entry(7), VertexFormatClass::uint_class, shape);
        case 3: return info(entry(8), VertexFormatClass::sint_class, shape);
        default: return std::nullopt;
        }
    }
    case ComponentWidth::x32: {
        switch (type) {
        case 7: return info(entry(9), VertexFormatClass::float_class, shape);
        case 4: return info(entry(10), VertexFormatClass::uint_class, shape);
        case 3: return info(entry(11), VertexFormatClass::sint_class, shape);
        default: return std::nullopt;
        }
    }
    default: return std::nullopt;
    }
}

}  // namespace

std::optional<VertexAttribFormatInfo> vertex_format_as_class(
    const VertexAttribFormatInfo& format, VertexFormatClass format_class) noexcept {
    if (format.format_class == format_class) return format;
    // Each component-count block of twelve ends with 32-bit float, uint, sint,
    // in VertexFormatClass order; the packed formats follow the four blocks.
    const auto index = static_cast<unsigned>(format.format);
    if (index >= 48 || index % 12 < 9) return std::nullopt;
    auto result = format;
    result.format = static_cast<VertexAttribFormat>(index - index % 12 + 9 +
                                                     static_cast<unsigned>(format_class));
    result.format_class = format_class;
    return result;
}

std::optional<VertexAttribFormatInfo> decode_vertex_attrib_format(
    std::uint32_t encoded) noexcept {
    if (encoded & ~encoded_format_mask) return std::nullopt;
    const auto size = (encoded >> 21) & 63u;
    const auto type = (encoded >> 27) & 7u;
    const auto shape = size_shape(size);
    if (!shape) return std::nullopt;
    // Packed formats: 1010 10 10 2 normalized and 11 10 11 float. Metal has
    // no raw integer 1010102 fetch, so those encodings stay unsupported.
    if (size == 0x30) {
        switch (type) {
        case 2: return info(VertexAttribFormat::A2B10G10R10Unorm, VertexFormatClass::float_class, *shape);
        case 1: return info(VertexAttribFormat::A2B10G10R10Snorm, VertexFormatClass::float_class, *shape);
        default: return std::nullopt;
        }
    }
    if (size == 0x31) {
        if (type != 7) return std::nullopt;
        return info(VertexAttribFormat::B10G11R11Float, VertexFormatClass::float_class, *shape);
    }
    return typed_format(shape->width, shape->components, type);
}

VertexAttributePlanResult plan_vertex_attribute(const RenderStop& snapshot,
                                                std::uint32_t location) noexcept {
    if (snapshot.kind != RenderKind::Draw || snapshot.engine != engine_class::threed) {
        return fail(CommandStatus::Unsupported, "Vertex input requires a 3D draw snapshot");
    }
    if (location >= 32) {
        return fail(CommandStatus::Unsupported, "Vertex attribute index exceeds the register table");
    }
    const auto register_value = snapshot.registers[0x458 + location];
    // The guest interface exposes the first sixteen stream bindings. Higher
    // encodings remain unsupported until their state layout is required.
    const auto buffer = register_value & 31u;
    if (buffer >= 16) {
        return fail(CommandStatus::Unsupported, "Vertex stream binding exceeds the supported table");
    }
    const auto decoded = decode_vertex_attrib_format(register_value & encoded_format_mask);
    if (!decoded) {
        return fail(CommandStatus::Unsupported,
                    "Vertex attribute format is unsupported on this host");
    }
    return VertexAttributePlan{location, buffer, (register_value >> 7) & 0x3fffu,
                               (register_value & (1u << 6)) != 0, *decoded};
}

VertexStreamPlanResult plan_vertex_stream(const RenderStop& snapshot,
                                          std::uint32_t stream) noexcept {
    if (snapshot.kind != RenderKind::Draw || snapshot.engine != engine_class::threed) {
        return fail(CommandStatus::Unsupported, "Vertex input requires a 3D draw snapshot");
    }
    if (stream >= 16) {
        return fail(CommandStatus::Unsupported, "Vertex stream binding exceeds the supported table");
    }
    const auto control = snapshot.registers[0x700 + stream * 4];
    if ((control & ~0x1fffu) != 0 || (control & 0x1000u) == 0) {
        return fail(CommandStatus::Unsupported,
                    "Vertex stream is disabled or uses unsupported control state");
    }
    const auto limit_register = 0x7c0 + stream * 2;
    // The divisor register counts only while the stream's instancing flag is
    // set; a flagged stream with divisor zero steps per vertex, matching the
    // host divisor mapping.
    const bool instanced = snapshot.registers[0x620 + stream] != 0;
    VertexStreamPlan plan{
        stream,
        (std::uint64_t{snapshot.registers[0x700 + stream * 4 + 1]} << 32) |
            snapshot.registers[0x700 + stream * 4 + 2],
        (std::uint64_t{snapshot.registers[limit_register]} << 32) |
            snapshot.registers[limit_register + 1],
        control & 0xfffu,
        instanced ? snapshot.registers[0x703 + stream * 4] : 0};
    if (plan.address == 0 || plan.address > plan.inclusive_limit) {
        return fail(CommandStatus::MemoryFault, "Vertex stream address range is invalid", plan.address);
    }
    return plan;
}

VertexFloat3PlanResult plan_vertex_float3(const RenderStop& snapshot,
                                         std::uint32_t attribute) noexcept {
    if (snapshot.kind != RenderKind::Draw || snapshot.engine != engine_class::threed) {
        return fail(CommandStatus::Unsupported, "Vertex input requires a 3D draw snapshot");
    }
    if (attribute >= 32) {
        return fail(CommandStatus::Unsupported, "Vertex attribute index exceeds the register table");
    }
    const auto format = snapshot.registers[0x458 + attribute];
    const auto buffer = format & 31u;
    if ((format & ((1u << 5) | (1u << 30))) != 0 ||
        ((format >> 21) & 63u) != 2 || ((format >> 27) & 7u) != 7 ||
        (format & ((1u << 6) | (1u << 31))) != 0) {
        return fail(CommandStatus::Unsupported,
                    "Vertex attribute requires direct float32 XYZ without channel reordering");
    }
    // The guest interface exposes the first sixteen stream bindings. Higher
    // encodings remain unsupported until their state layout is required.
    if (buffer >= 16) {
        return fail(CommandStatus::Unsupported, "Vertex stream binding exceeds the supported table");
    }
    const auto stream = 0x700 + buffer * 4;
    const auto control = snapshot.registers[stream];
    if ((control & ~0x1fffu) != 0 || (control & 0x1000u) == 0 ||
        snapshot.registers[0x620 + buffer] != 0) {
        return fail(CommandStatus::Unsupported,
                    "Vertex stream is disabled or uses unsupported instancing or control state");
    }
    const auto limit_register = 0x7c0 + buffer * 2;
    VertexFloat3Plan plan{
        attribute, buffer,
        (std::uint64_t{snapshot.registers[stream + 1]} << 32) |
            snapshot.registers[stream + 2],
        (std::uint64_t{snapshot.registers[limit_register]} << 32) |
            snapshot.registers[limit_register + 1],
        control & 0xfffu, (format >> 7) & 0x3fffu};
    if (plan.address == 0 || plan.address > plan.inclusive_limit) {
        return fail(CommandStatus::MemoryFault, "Vertex stream address range is invalid", plan.address);
    }
    return plan;
}

VertexFloat3Result fetch_vertex_float3(const VertexFloat3Plan& plan,
                                      std::uint64_t vertex_index,
                                      const GpuAddressSpace& memory) noexcept {
    constexpr auto maximum = std::numeric_limits<std::uint64_t>::max();
    if (plan.stride != 0 && vertex_index > maximum / plan.stride) {
        return fail(CommandStatus::MemoryFault, "Vertex stride multiplication overflows");
    }
    auto offset = vertex_index * plan.stride;
    if (offset > maximum - plan.offset) {
        return fail(CommandStatus::MemoryFault, "Vertex attribute offset overflows");
    }
    offset += plan.offset;
    if (offset > maximum - plan.address) {
        return fail(CommandStatus::MemoryFault, "Vertex address overflows");
    }
    const auto address = plan.address + offset;
    if (address > plan.inclusive_limit || plan.inclusive_limit - address < 11) {
        return fail(CommandStatus::MemoryFault, "Vertex attribute exceeds the inclusive stream limit", address);
    }
    std::array<std::byte, 12> bytes{};
    if (const auto read = memory.read(address, bytes); !read) {
        return fail(CommandStatus::MemoryFault, read.detail, address);
    }
    std::array<std::uint32_t, 3> result{};
    for (std::size_t component = 0; component != result.size(); ++component) {
        for (unsigned byte = 0; byte != 4; ++byte) {
            result[component] |= std::uint32_t{std::to_integer<unsigned>(bytes[component * 4 + byte])}
                                 << (byte * 8);
        }
    }
    return result;
}

VertexFloat2PlanResult plan_vertex_float2(const RenderStop& snapshot,
                                         std::uint32_t attribute) noexcept {
    if (snapshot.kind != RenderKind::Draw || snapshot.engine != engine_class::threed) {
        return fail(CommandStatus::Unsupported, "Vertex input requires a 3D draw snapshot");
    }
    if (attribute >= 32) {
        return fail(CommandStatus::Unsupported, "Vertex attribute index exceeds the register table");
    }
    const auto format = snapshot.registers[0x458 + attribute];
    const auto buffer = format & 31u;
    if ((format & ((1u << 5) | (1u << 30))) != 0 ||
        ((format >> 21) & 63u) != 4 || ((format >> 27) & 7u) != 7 ||
        (format & ((1u << 6) | (1u << 31))) != 0) {
        return fail(CommandStatus::Unsupported,
                    "Vertex attribute requires direct float32 XY without channel reordering");
    }
    // The guest interface exposes the first sixteen stream bindings. Higher
    // encodings remain unsupported until their state layout is required.
    if (buffer >= 16) {
        return fail(CommandStatus::Unsupported, "Vertex stream binding exceeds the supported table");
    }
    const auto stream = 0x700 + buffer * 4;
    const auto control = snapshot.registers[stream];
    if ((control & ~0x1fffu) != 0 || (control & 0x1000u) == 0 ||
        snapshot.registers[0x620 + buffer] != 0) {
        return fail(CommandStatus::Unsupported,
                    "Vertex stream is disabled or uses unsupported instancing or control state");
    }
    const auto limit_register = 0x7c0 + buffer * 2;
    VertexFloat2Plan plan{
        attribute, buffer,
        (std::uint64_t{snapshot.registers[stream + 1]} << 32) |
            snapshot.registers[stream + 2],
        (std::uint64_t{snapshot.registers[limit_register]} << 32) |
            snapshot.registers[limit_register + 1],
        control & 0xfffu, (format >> 7) & 0x3fffu};
    if (plan.address == 0 || plan.address > plan.inclusive_limit) {
        return fail(CommandStatus::MemoryFault, "Vertex stream address range is invalid", plan.address);
    }
    return plan;
}

VertexFloat2Result fetch_vertex_float2(const VertexFloat2Plan& plan,
                                      std::uint64_t vertex_index,
                                      const GpuAddressSpace& memory) noexcept {
    constexpr auto maximum = std::numeric_limits<std::uint64_t>::max();
    if (plan.stride != 0 && vertex_index > maximum / plan.stride) {
        return fail(CommandStatus::MemoryFault, "Vertex stride multiplication overflows");
    }
    auto offset = vertex_index * plan.stride;
    if (offset > maximum - plan.offset) {
        return fail(CommandStatus::MemoryFault, "Vertex attribute offset overflows");
    }
    offset += plan.offset;
    if (offset > maximum - plan.address) {
        return fail(CommandStatus::MemoryFault, "Vertex address overflows");
    }
    const auto address = plan.address + offset;
    if (address > plan.inclusive_limit || plan.inclusive_limit - address < 7) {
        return fail(CommandStatus::MemoryFault, "Vertex attribute exceeds the inclusive stream limit", address);
    }
    std::array<std::byte, 8> bytes{};
    if (const auto read = memory.read(address, bytes); !read) {
        return fail(CommandStatus::MemoryFault, read.detail, address);
    }
    std::array<std::uint32_t, 2> result{};
    for (std::size_t component = 0; component != result.size(); ++component) {
        for (unsigned byte = 0; byte != 4; ++byte) {
            result[component] |= std::uint32_t{std::to_integer<unsigned>(bytes[component * 4 + byte])}
                                 << (byte * 8);
        }
    }
    return result;
}

}  // namespace armigo::graphics
