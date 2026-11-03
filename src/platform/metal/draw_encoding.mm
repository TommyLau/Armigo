// ============================================================================
// Armigo Platform - Apple Platform Integration
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include "draw_encoding.h"

#include <algorithm>
#include <limits>
#include <new>

namespace armigo::platform::metal {
namespace {
using namespace graphics;
constexpr auto max_index = std::numeric_limits<std::uint32_t>::max();

RendererStatus fail(CommandStatus status, const char* detail) {
    return RendererError({status, detail});
}

std::size_t width(IndexFormat format) {
    return format == IndexFormat::U8 ? 1 : format == IndexFormat::U16 ? 2 : 4;
}

std::uint32_t load(const std::byte* p, std::size_t size) {
    std::uint32_t value = 0;
    for (std::size_t i = 0; i < size; ++i)
        value |= std::to_integer<std::uint32_t>(p[i]) << (i * 8);
    return value;
}

void store(std::byte* p, std::uint32_t value, std::size_t size) {
    for (std::size_t i = 0; i < size; ++i)
        p[i] = std::byte((value >> (i * 8)) & 255);
}

bool normalized_source(const DrawPlan& plan, const RenderStop& stop) {
    if (stop.registers[0x652] != 0) return stop.registers[0x65c] == 0x1015;
    const auto type = plan.source.topology & 0xffff;
    return type == 2 || (type >= 6 && type <= 9);
}

RendererStatus check_fetch(const DrawPlan& plan, std::uint32_t maximum) {
    for (const auto& attribute : plan.vertex.attributes) {
        if (attribute.constant) continue;
        const auto stream = std::find_if(plan.vertex.streams.begin(), plan.vertex.streams.end(),
            [&](const auto& item) { return item.stream == attribute.stream; });
        const auto step = std::find_if(plan.vertex.steps.begin(), plan.vertex.steps.end(),
            [&](const auto& item) { return item.stream == attribute.stream; });
        if (stream == plan.vertex.streams.end() || step == plan.vertex.steps.end())
            return fail(CommandStatus::Unsupported, "Native draw has an incomplete vertex stream plan");
        if (step->step != VertexStep::PerVertex) continue;
        if (stream->inclusive_limit < stream->address || attribute.format.size_bytes == 0)
            return fail(CommandStatus::MemoryFault, "Native draw vertex extent is invalid");
        const auto available = stream->inclusive_limit - stream->address;
        const auto tail = std::uint64_t(attribute.offset) + attribute.format.size_bytes - 1;
        if (tail > available || (stream->stride != 0 &&
            maximum > (available - tail) / stream->stride))
            return fail(CommandStatus::MemoryFault, "Native indexed fetch exceeds the inclusive vertex stream limit");
    }
    return renderer_success();
}

// Whether a live raw index equal to the format's maximum, which Metal always
// treats as a strip restart, can only name a vertex outside every per-vertex
// stream limit. Draws without per-vertex streams may consume it as a vertex
// ID, so they never qualify.
bool sentinel_unreachable(const DrawPlan& plan, std::uint32_t sentinel) {
    bool streamed = false;
    for (const auto& attribute : plan.vertex.attributes) {
        if (attribute.constant) continue;
        const auto stream = std::find_if(plan.vertex.streams.begin(), plan.vertex.streams.end(),
            [&](const auto& item) { return item.stream == attribute.stream; });
        const auto step = std::find_if(plan.vertex.steps.begin(), plan.vertex.steps.end(),
            [&](const auto& item) { return item.stream == attribute.stream; });
        if (stream == plan.vertex.streams.end() || step == plan.vertex.steps.end()) return false;
        if (step->step != VertexStep::PerVertex) continue;
        streamed = true;
        if (stream->stride == 0) return false;
        if (stream->inclusive_limit < stream->address) continue;
        const auto available = stream->inclusive_limit - stream->address;
        const auto tail = std::uint64_t(attribute.offset) + attribute.format.size_bytes - 1;
        if (tail > available) continue;
        const auto vertices = (available - tail) / stream->stride + 1;
        for (const auto& segment : plan.segments) {
            const auto vertex = std::int64_t{sentinel} + segment.base_vertex;
            if (vertex >= 0 && std::uint64_t(vertex) < vertices) return false;
        }
    }
    return streamed;
}

RendererStatus select(DrawPlan& plan, const RenderStop& stop, ResourceMemory& memory) {
    if (!plan.indices || plan.source.count == 0 || plan.source.instance_count == 0)
        return renderer_success();
    auto index = *plan.indices;
    auto segments = plan.segments;
    const bool normalized = normalized_source(plan, stop);
    // Guest U16/U32 indices that Metal consumes as stored stay in guest
    // memory and the signed base vertex goes to the native draw, with no CPU
    // snapshot or scan. Fetches are not bounded by index: Maxwell reads zero
    // beyond a stream limit while Metal reads the neighbouring guest bytes of
    // the bound pool. Metal restarts strips at the format's maximum value, so
    // that value must be the guest restart index or unreachable as a vertex.
    if (plan.source.indexed && index.inline_source.empty() && index.converted.empty() && !normalized &&
        index.encoded_format == index.source_format && index.source_format != IndexFormat::U8) {
        const auto sentinel = index.source_format == IndexFormat::U16 ? 0xffffu : max_index;
        if ((index.restart_enabled && index.restart_index == sentinel) ||
            sentinel_unreachable(plan, sentinel))
            return renderer_success();
    }
    // The CPU path reads the drawn elements from their start: rebase a
    // whole-buffer plan onto its first element.
    if (index.whole_buffer && segments.size() == 1) {
        const auto size = width(index.source_format);
        index.address += std::uint64_t(segments.front().first) * size;
        index.inclusive_limit = index.address + std::uint64_t(plan.source.count) * size - 1;
        segments.front().first = 0;
        index.whole_buffer = false;
    }
    const bool strip = plan.topology == PrimitiveTopology::LineStrip ||
                       plan.topology == PrimitiveTopology::TriangleStrip;
    std::vector<std::uint32_t> values;
    std::vector<bool> restart;
    std::int32_t native_base = 0;
    std::uint32_t maximum = 0;
    bool any_live = false;

    if (plan.source.indexed) {
        const auto size = width(index.source_format);
        const auto byte_count = std::uint64_t(plan.source.count) * size;
        if (!index.inline_source.empty()) {
            if (index.inline_source.size() != plan.source.count)
                return fail(CommandStatus::Unsupported,
                            "Native inline source does not cover the draw count");
            values = index.inline_source;
        } else {
            if (index.address == 0 || index.inclusive_limit < index.address ||
                byte_count - 1 > index.inclusive_limit - index.address)
                return fail(CommandStatus::MemoryFault, "Native index snapshot exceeds the checked source extent");
            auto retained = memory.retain(index.address, byte_count, ResourceAccess::Read);
            if (auto* error = std::get_if<RendererError>(&retained)) return *error;
            auto transaction = memory.prepare_cpu_read(std::get<RetainedRange>(retained).bytes,
                                                       ResolveReason::CpuRead);
            if (auto* error = std::get_if<RendererError>(&transaction)) return *error;
            std::vector<std::byte> bytes(byte_count);
            const auto read = memory.address_space().read(index.address, bytes);
            if (!read) return RendererError({CommandStatus::MemoryFault, read.detail});
            values.resize(plan.source.count);
            for (std::size_t i = 0; i < values.size(); ++i)
                values[i] = load(bytes.data() + i * size, size);
        }
        restart.resize(plan.source.count);
        for (std::size_t i = 0; i < values.size(); ++i) {
            const auto raw = values[i];
            values[i] = raw;
            restart[i] = index.restart_enabled && raw == index.restart_index;
            if (restart[i]) continue;
            const auto effective = std::int64_t(raw) + plan.source.base_vertex;
            if (effective < 0 || effective > max_index)
                return fail(CommandStatus::MemoryFault, "Native index plus signed base vertex exceeds the unsigned vertex ID range");
            maximum = std::max(maximum, std::uint32_t(effective));
            any_live = true;
        }
        if (!any_live) {
            plan.segments.clear();
            plan.indices.reset();
            return renderer_success();
        }
        if (auto status = check_fetch(plan, maximum); std::holds_alternative<RendererError>(status))
            return status;
        native_base = plan.source.base_vertex;
    }

    if (normalized || !plan.source.indexed) {
        // L8 already applied signed bias and topology expansion. Consume those
        // absolute IDs without adding the source base a second time.
        if (index.converted.empty() || index.converted.size() % width(index.encoded_format) != 0)
            return fail(CommandStatus::Unsupported, "Normalized native draw requires owned complete index elements");
        const auto size = width(index.encoded_format);
        values.resize(index.converted.size() / size);
        restart.assign(values.size(), false);
        for (std::size_t i = 0; i < values.size(); ++i)
            values[i] = load(index.converted.data() + i * size, size);
        native_base = 0;
    }

    for (const auto& segment : segments) {
        if (segment.first > values.size() || segment.count > values.size() - segment.first)
            return fail(CommandStatus::MemoryFault, "Native draw segment exceeds the owned index snapshot");
    }
    if (segments.empty()) return renderer_success();

    auto format = normalized || !plan.source.indexed ? index.encoded_format : index.source_format;
    if (format == IndexFormat::U8) format = IndexFormat::U16;
    bool collision = false;
    for (std::size_t i = 0; i < values.size(); ++i)
        collision |= !restart[i] && values[i] == (format == IndexFormat::U16 ? 65535u : max_index);
    if (collision && format == IndexFormat::U16) {
        format = IndexFormat::U32;
        collision = false;
    }
    if (collision) {
        // A nonwrapping affine rebase exists unless both endpoints of the U32
        // domain are live. Never infer permission for shader-side ID patching.
        std::uint32_t minimum = max_index;
        for (std::size_t i = 0; i < values.size(); ++i) {
            if (restart[i]) continue;
            const auto effective = std::int64_t(values[i]) + native_base;
            minimum = std::min(minimum, std::uint32_t(effective));
        }
        const auto rebase = std::min(minimum, std::uint32_t(std::numeric_limits<std::int32_t>::max()));
        for (std::size_t i = 0; i < values.size(); ++i) {
            if (restart[i]) continue;
            const auto effective = std::int64_t(values[i]) + native_base;
            const auto rebased = effective - rebase;
            if (rebased < 0 || rebased >= max_index)
                return fail(CommandStatus::Unsupported, "Native U32 sentinel preservation requires an admitted vertex ID emulation variant");
            values[i] = std::uint32_t(rebased);
        }
        native_base = std::int32_t(rebase);
    }

    const bool native_restart = strip && index.restart_enabled && !normalized;
    const auto size = width(format);
    index.converted.resize(values.size() * size);
    for (std::size_t i = 0; i < values.size(); ++i) {
        const auto value = restart[i] ? (format == IndexFormat::U16 ? 65535u : max_index) : values[i];
        store(index.converted.data() + i * size, value, size);
    }
    if (native_restart) {
        const auto native_instance = segments.front();
        segments = {{0, std::uint32_t(values.size()), native_base,
                     native_instance.base_instance, native_instance.instance_count,
                     native_instance.instance_index}};
    } else {
        for (auto& segment : segments) segment.base_vertex = native_base;
    }
    index.encoded_format = format;
    index.restart_enabled = native_restart;
    index.restart_index = format == IndexFormat::U16 ? 65535u : max_index;
    plan.indices = std::move(index);
    plan.segments = std::move(segments);
    return renderer_success();
}
} // namespace

graphics::RendererStatus select_native_draw(graphics::DrawPlan& plan,
    const graphics::RenderStop& stop, graphics::ResourceMemory& memory) {
    try {
        return select(plan, stop, memory);
    } catch (const std::bad_alloc&) {
        return fail(graphics::CommandStatus::BudgetExceeded, "Native index snapshot allocation failed");
    }
}
} // namespace armigo::platform::metal
