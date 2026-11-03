// ============================================================================
// Armigo Graphics - GPU Command Processing
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================
// Draw planning preserves the event-latched DrawEnd/DrawBegin flow, separate
// indexed and array parameters, restart-before-base-vertex ordering, Begin and
// topology-override encodings, and fan, polygon, and quad index expansion.
// Restart state, index type, and instancing divisors are decoded into owned
// portable structures. Unknown encodings stop with structured diagnostics.
// The portable plan widens byte indices, splits restart runs, and materializes
// signed bias; Metal may retain raw elements with signed baseVertex and fixed
// restart while preserving guest vertex and instance identifiers.

#include <armigo/graphics/draw_plan.h>

#include <armigo/graphics/vertex_input.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <new>
#include <optional>

namespace armigo::graphics {
namespace {

CommandResult fail(CommandStatus status, std::string_view detail,
                   std::uint64_t address = 0) noexcept {
    return {status, detail, address, 0, 0, 0, 0, engine_class::threed};
}

RendererError reject(CommandStatus status, std::string_view detail,
                     std::uint64_t address = 0) noexcept {
    return RendererError{fail(status, detail, address)};
}

// Maxwell draw registers consumed at assembly time. The register indices are
// 3D method offsets: the index buffer state block (start address, inclusive
// end address, element type), primitive restart with its array-draw control,
// and the separate topology override word. The latched DrawDescription in the
// snapshot already carries first/count/base vertex/base instance, so the
// corresponding registers are never re-read here.
constexpr std::uint32_t primitive_restart_draw_arrays = 0x37a;
constexpr std::uint32_t primitive_restart_enable = 0x591;
constexpr std::uint32_t primitive_restart_index = 0x592;
constexpr std::uint32_t index_buffer_start_high = 0x5f2;
constexpr std::uint32_t index_buffer_start_low = 0x5f3;
constexpr std::uint32_t index_buffer_end_high = 0x5f4;
constexpr std::uint32_t index_buffer_end_low = 0x5f5;
constexpr std::uint32_t index_buffer_type = 0x5f6;
constexpr std::uint32_t topology_override_enable = 0x652;
constexpr std::uint32_t topology_override = 0x65c;

void store16(std::byte* destination, std::uint32_t value) noexcept {
    destination[0] = std::byte(value & 255u);
    destination[1] = std::byte((value >> 8) & 255u);
}

void store32(std::byte* destination, std::uint32_t value) noexcept {
    store16(destination, value & 65535u);
    store16(destination + 2, (value >> 16) & 65535u);
}

enum class DecodeFailure : std::uint8_t { none, unknown, adjacency, patches };

struct DecodedTopology {
    PrimitiveTopology source = PrimitiveTopology::Triangles;
    DecodeFailure failure = DecodeFailure::none;
};

// The Begin argument's low sixteen bits select the guest primitive type; the
// upper bits are instance and split controls this planner never latches.
DecodedTopology decode_begin(std::uint32_t encoding) noexcept {
    switch (encoding & 0xffffu) {
    case 0: return {PrimitiveTopology::Points};
    case 1: return {PrimitiveTopology::Lines};
    case 2: return {PrimitiveTopology::LineLoop};
    case 3: return {PrimitiveTopology::LineStrip};
    case 4: return {PrimitiveTopology::Triangles};
    case 5: return {PrimitiveTopology::TriangleStrip};
    case 6: return {PrimitiveTopology::TriangleFan};
    case 7: return {PrimitiveTopology::Quads};
    case 8: return {PrimitiveTopology::QuadStrip};
    // A polygon is one fan over its vertices; Armigo plans it as a fan.
    case 9: return {PrimitiveTopology::TriangleFan};
    case 10:
    case 11:
    case 12:
    case 13: return {PrimitiveTopology::Triangles, DecodeFailure::adjacency};
    case 14: return {PrimitiveTopology::Triangles, DecodeFailure::patches};
    default: return {PrimitiveTopology::Triangles, DecodeFailure::unknown};
    }
}

// The override word uses its own sparse encoding with no loop, quad, quad
// strip or polygon entries.
DecodedTopology decode_override(std::uint32_t encoding) noexcept {
    switch (encoding) {
    case 1: return {PrimitiveTopology::Points};
    case 2: return {PrimitiveTopology::Lines};
    case 3: return {PrimitiveTopology::LineStrip};
    case 4: return {PrimitiveTopology::Triangles};
    case 5: return {PrimitiveTopology::TriangleStrip};
    case 0x1015: return {PrimitiveTopology::TriangleFan};
    case 10:
    case 11:
    case 12:
    case 13: return {PrimitiveTopology::Triangles, DecodeFailure::adjacency};
    case 14: return {PrimitiveTopology::Triangles, DecodeFailure::patches};
    default: return {PrimitiveTopology::Triangles, DecodeFailure::unknown};
    }
}

// What the backend can encode directly. Loop, fan, polygon, quad and quad
// strip draws normalize into lists through generated index bytes.
bool supported_directly(PrimitiveTopology topology) noexcept {
    return topology == PrimitiveTopology::Points ||
           topology == PrimitiveTopology::Lines ||
           topology == PrimitiveTopology::LineStrip ||
           topology == PrimitiveTopology::Triangles ||
           topology == PrimitiveTopology::TriangleStrip;
}

PrimitiveTopology host_topology(PrimitiveTopology source) noexcept {
    switch (source) {
    case PrimitiveTopology::LineLoop: return PrimitiveTopology::Lines;
    case PrimitiveTopology::TriangleFan:
    case PrimitiveTopology::Quads:
    case PrimitiveTopology::QuadStrip: return PrimitiveTopology::Triangles;
    default: return source;
    }
}

PrimitiveClass class_of(PrimitiveTopology host) noexcept {
    switch (host) {
    case PrimitiveTopology::Points: return PrimitiveClass::Point;
    case PrimitiveTopology::Lines:
    case PrimitiveTopology::LineStrip: return PrimitiveClass::Line;
    default: return PrimitiveClass::Triangle;
    }
}

std::size_t index_size(IndexFormat format) noexcept {
    switch (format) {
    case IndexFormat::U8: return 1;
    case IndexFormat::U16: return 2;
    case IndexFormat::U32: return 4;
    }
    return 1;
}

std::uint32_t read_element(const std::byte* bytes, IndexFormat format) noexcept {
    switch (format) {
    case IndexFormat::U8: return std::to_integer<std::uint32_t>(bytes[0]);
    case IndexFormat::U16:
        return std::uint32_t{std::to_integer<std::uint32_t>(bytes[0])} |
               (std::uint32_t{std::to_integer<std::uint32_t>(bytes[1])} << 8);
    case IndexFormat::U32:
        return std::uint32_t{std::to_integer<std::uint32_t>(bytes[0])} |
               (std::uint32_t{std::to_integer<std::uint32_t>(bytes[1])} << 8) |
               (std::uint32_t{std::to_integer<std::uint32_t>(bytes[2])} << 16) |
               (std::uint32_t{std::to_integer<std::uint32_t>(bytes[3])} << 24);
    }
    return 0;
}

// One run of consecutive live elements between restart boundaries, as element
// offsets relative to the first drawn element.
struct Run {
    std::uint32_t begin = 0;
    std::uint32_t length = 0;
};

struct Scan {
    // Raw element values of the drawn range, before base-vertex biasing.
    std::vector<std::uint32_t> values{};
    std::vector<Run> runs{};
    std::uint32_t max_value = 0;
};

// Reads and decodes the drawn index range once. Restart positions split the
// element sequence into runs before any base-vertex biasing is applied, so
// restart detection never depends on the signed offset, and an equal value
// stays an ordinary index when restart is disabled.
[[nodiscard]] bool scan_indices(std::uint64_t start, std::uint32_t count,
                                IndexFormat format, bool restart_enabled,
                                std::uint32_t restart_index, Scan& scan,
                                CommandResult& error,
                                const GpuAddressSpace& memory,
                                std::span<const std::uint32_t> inline_indices) noexcept {
    try {
        if (!inline_indices.empty()) {
            scan.values.assign(inline_indices.begin(), inline_indices.end());
        } else {
            const std::uint64_t element_size = index_size(format);
            std::vector<std::byte> bytes(element_size * count);
            if (const auto read = memory.read(start, bytes); !read) {
                error = fail(CommandStatus::MemoryFault, read.detail, start);
                return false;
            }
            scan.values.resize(count);
            for (std::uint32_t element = 0; element != count; ++element)
                scan.values[element] =
                    read_element(bytes.data() + element * element_size, format);
        }
    } catch (const std::bad_alloc&) {
        error = fail(CommandStatus::BudgetExceeded,
                     "Index source snapshot exceeds the supported size");
        return false;
    }
    scan.runs.clear();
    scan.max_value = 0;
    Run current{};
    try {
        for (std::uint32_t element = 0; element != count; ++element) {
            if (restart_enabled && scan.values[element] == restart_index) {
                if (current.length != 0) scan.runs.push_back(current);
                current = {};
                continue;
            }
            scan.max_value = std::max(scan.max_value, scan.values[element]);
            if (current.length == 0) current.begin = element;
            ++current.length;
        }
        if (current.length != 0) scan.runs.push_back(current);
    } catch (const std::bad_alloc&) {
        error = fail(CommandStatus::BudgetExceeded,
                     "Index buffer read exceeds the supported size");
        return false;
    }
    return true;
}

// Appends the expanded primitive set of one normalized topology. All emitted
// entries are absolute: index values for indexed draws, vertex numbers for
// arrays, so host builtin vertex IDs keep their guest values unchanged.
void append_normalized(PrimitiveTopology source, std::uint32_t first,
                       std::uint32_t count, const std::uint32_t* values,
                       std::int32_t base_vertex,
                       std::vector<std::uint32_t>& out) {
    const auto emit = [&](std::uint32_t value) {
        out.push_back(std::uint32_t(std::int64_t{value} + base_vertex));
    };
    switch (source) {
    case PrimitiveTopology::TriangleFan: {
        // Fan triangle i keeps the guest winding: anchor, i+1, i+2.
        for (std::uint32_t i = 0; i + 2 < count; ++i) {
            emit(values ? values[0] : first);
            emit(values ? values[i + 1] : first + i + 1);
            emit(values ? values[i + 2] : first + i + 2);
        }
        break;
    }
    case PrimitiveTopology::Quads: {
        // Quad i splits along its 0-2 diagonal: 0,1,2 then 0,2,3.
        for (std::uint32_t i = 0; i + 3 < count; i += 4) {
            emit(values ? values[i] : first + i);
            emit(values ? values[i + 1] : first + i + 1);
            emit(values ? values[i + 2] : first + i + 2);
            emit(values ? values[i] : first + i);
            emit(values ? values[i + 2] : first + i + 2);
            emit(values ? values[i + 3] : first + i + 3);
        }
        break;
    }
    case PrimitiveTopology::QuadStrip: {
        // Strip quad i spans vertices 2i..2i+3 in the guest cyclic order
        // 0,1,3,2, so each quad keeps its 1-3 diagonal: 0,1,3 then 0,3,2.
        for (std::uint32_t i = 0; i + 3 < count; i += 2) {
            emit(values ? values[i] : first + i);
            emit(values ? values[i + 1] : first + i + 1);
            emit(values ? values[i + 3] : first + i + 3);
            emit(values ? values[i] : first + i);
            emit(values ? values[i + 3] : first + i + 3);
            emit(values ? values[i + 2] : first + i + 2);
        }
        break;
    }
    case PrimitiveTopology::LineLoop: {
        // Line list pairs plus the closing wrap segment.
        for (std::uint32_t i = 0; i + 1 < count; ++i) {
            emit(values ? values[i] : first + i);
            emit(values ? values[i + 1] : first + i + 1);
        }
        emit(values ? values[count - 1] : first + count - 1);
        emit(values ? values[0] : first);
        break;
    }
    default: break;
    }
}

std::uint64_t normalized_count(PrimitiveTopology source, std::uint32_t count) noexcept {
    switch (source) {
    case PrimitiveTopology::TriangleFan:
        return count >= 3 ? std::uint64_t{count - 2} * 3 : 0;
    case PrimitiveTopology::Quads:
        return std::uint64_t{count / 4} * 6;
    case PrimitiveTopology::QuadStrip:
        return count >= 4 ? std::uint64_t{(count - 2) / 2} * 6 : 0;
    case PrimitiveTopology::LineLoop:
        return count >= 2 ? std::uint64_t{count} * 2 : 0;
    default: return 0;
    }
}

}  // namespace

RendererResult<DrawPlan> assemble_draw(const RenderStop& stop,
    std::span<const VertexClassBinding> inputs, const GpuAddressSpace& memory,
    std::span<const std::uint32_t> inline_indices, bool native_base_vertex, bool plan_vertex) {
    if (stop.kind != RenderKind::Draw || stop.engine != engine_class::threed) {
        return reject(CommandStatus::Unsupported,
                    "Draw assembly requires a 3D draw snapshot");
    }
    if (!stop.draw) {
        return reject(CommandStatus::Unsupported,
                    "Draw assembly requires an event-latched draw description");
    }
    const DrawDescription& draw = *stop.draw;
    if (draw.base_instance >
        std::numeric_limits<std::uint32_t>::max() - draw.instance_index) {
        return reject(CommandStatus::BudgetExceeded,
                      "Draw base instance and instance index overflowed");
    }
    const std::uint32_t native_base_instance =
        draw.base_instance + draw.instance_index;
    const bool inline_source = !inline_indices.empty();
    if (inline_source && (!draw.indexed || draw.first != 0 ||
                          inline_indices.size() != draw.count))
        return reject(CommandStatus::Unsupported,
                      "Inline indices require an indexed draw over the complete stream");


    // Decode the Begin encoding and the topology override separately: the
    // Begin word carries instance and split controls in its upper bits that
    // this planner never latches, and the override word uses its own table.
    const bool overridden = stop.registers[topology_override_enable] != 0;
    if (!overridden && (draw.topology & 0xffff0000u) != 0) {
        return reject(CommandStatus::Unsupported,
                    "Draw Begin instance and split controls are unsupported");
    }
    const auto decoded = overridden ? decode_override(stop.registers[topology_override])
                                    : decode_begin(draw.topology);
    switch (decoded.failure) {
    case DecodeFailure::none: break;
    case DecodeFailure::unknown:
        return reject(CommandStatus::Unsupported,
                    overridden ? "Topology override encoding is unknown"
                               : "Draw Begin primitive type is unknown");
    case DecodeFailure::adjacency:
        return reject(CommandStatus::Unsupported,
                    "Draw requests adjacency primitives which are unsupported");
    case DecodeFailure::patches:
        return reject(CommandStatus::Unsupported,
                    "Draw requests tessellation patches which are unsupported");
    }
    if (!draw.indexed && draw.base_vertex != 0) {
        return reject(CommandStatus::Unsupported,
                    "Signed base vertex applies only to indexed draws");
    }

    DrawPlan plan;
    plan.source = draw;
    plan.topology = host_topology(decoded.source);
    plan.primitive_class = class_of(plan.topology);

    // Plan the consumed vertex state first: typed formats, streams, divisors,
    // and the exact per-instance and array reach checks.
    VertexInputPlan& vertex = plan.vertex;
    for (const auto& binding : plan_vertex ? inputs : std::span<const VertexClassBinding>{}) {
        auto attribute = plan_vertex_attribute(stop, binding.location);
        auto* made = std::get_if<VertexAttributePlan>(&attribute);
        if (!made) return RendererError{std::get<CommandResult>(attribute)};
        if (made->format.format_class != binding.input_class) {
            const auto refetched = vertex_format_as_class(made->format, binding.input_class);
            if (!refetched)
                return reject(CommandStatus::Unsupported,
                            "Vertex attribute storage class does not match the translated shader declaration");
            made->format = *refetched;
        }
        const bool known_attribute = std::any_of(
            vertex.attributes.begin(), vertex.attributes.end(),
            [&](const VertexAttributePlan& existing) {
                return existing.location == made->location;
            });
        if (!known_attribute) vertex.attributes.push_back(*made);
        if (made->constant) continue;
        std::size_t slot = vertex.streams.size();
        const bool known_stream = std::any_of(
            vertex.streams.begin(), vertex.streams.end(),
            [&](const VertexStreamPlan& existing) {
                if (existing.stream == made->stream) {
                    slot = std::size_t(&existing - vertex.streams.data());
                    return true;
                }
                return false;
            });
        if (!known_stream) {
            const auto stream = plan_vertex_stream(stop, made->stream);
            auto* made_stream = std::get_if<VertexStreamPlan>(&stream);
            if (!made_stream) return RendererError{std::get<CommandResult>(stream)};
            vertex.streams.push_back(*made_stream);
            VertexStreamStep step;
            step.stream = made_stream->stream;
            if (made_stream->stride == 0) {
                // A zero stride reads one address for every fetch.
                step.step = VertexStep::Constant;
            } else if (made_stream->divisor != 0) {
                // The attribute index advances once per divisor instances,
                // counting from the base instance.
                step.step = VertexStep::PerInstance;
                step.rate = made_stream->divisor;
            } else {
                step.step = VertexStep::PerVertex;
            }
            vertex.steps.push_back(step);
            slot = vertex.streams.size() - 1;
        }
        const VertexStreamPlan& stream = vertex.streams[slot];
        const VertexStreamStep& step = vertex.steps[slot];
        const std::uint64_t available = stream.inclusive_limit - stream.address;
        const std::uint64_t attribute_span =
            std::uint64_t{made->offset} + made->format.size_bytes - 1;
        if (step.step == VertexStep::PerInstance && draw.instance_count != 0) {
            const std::uint64_t last_fetch =
                (std::uint64_t{native_base_instance} + draw.instance_count - 1) /
                step.rate;
            if (stream.stride != 0 &&
                last_fetch * stream.stride + attribute_span > available) {
                return reject(CommandStatus::MemoryFault,
                            "Per-instance vertex fetch exceeds the inclusive stream limit",
                            stream.address);
            }
        }
        if (step.step == VertexStep::PerVertex && !draw.indexed && draw.count != 0) {
            const std::uint64_t last_fetch =
                std::uint64_t{draw.first} + draw.count - 1;
            if (stream.stride != 0 &&
                last_fetch * stream.stride + attribute_span > available) {
                return reject(CommandStatus::MemoryFault,
                            "Vertex fetch exceeds the inclusive stream limit",
                            stream.address);
            }
        }
        if (attribute_span > available) {
            return reject(CommandStatus::MemoryFault,
                        "Vertex attribute exceeds the inclusive stream limit",
                        stream.address);
        }
    }

    const bool restart_state = stop.registers[primitive_restart_enable] != 0;
    const std::uint32_t restart_index = stop.registers[primitive_restart_index];

    // Zero count or zero instances is a real no-coverage plan: state stays
    // validated, no segments and no index plan are produced, and no GPU
    // completion is claimed by planning.
    if (draw.count == 0 || draw.instance_count == 0) {
        return plan;
    }

    if (!draw.indexed) {
        const bool array_restart =
            restart_state && stop.registers[primitive_restart_draw_arrays] != 0;
        if (supported_directly(decoded.source) && !array_restart) {
            plan.segments.push_back({draw.first, draw.count, 0,
                                     native_base_instance, draw.instance_count,
                                     draw.instance_index});
            return plan;
        }
        if (supported_directly(decoded.source)) {
            // Array restart splits the sequential vertex range on vertex
            // numbers equal to the restart index; no memory is read.
            Run current{};
            for (std::uint32_t element = 0; element != draw.count; ++element) {
                if (draw.first + element == restart_index) {
                    if (current.length != 0) {
                        plan.segments.push_back({draw.first + current.begin,
                            current.length, 0, native_base_instance,
                            draw.instance_count, draw.instance_index});
                    }
                    current = {};
                    continue;
                }
                if (current.length == 0) current.begin = element;
                ++current.length;
            }
            if (current.length != 0) {
                plan.segments.push_back({draw.first + current.begin,
                                         current.length, 0, native_base_instance,
                                         draw.instance_count, draw.instance_index});
            }
            return plan;
        }
        if (array_restart) {
            // Normalized array topologies also require index conversion. Stop
            // under the same documented limitation as indexed normalization
            // instead of expanding across a restart vertex.
            return reject(CommandStatus::Unsupported,
                        "Primitive restart with normalized topology is unsupported");
        }
        // Normalized topologies expand into owned little-endian U32 vertex
        // numbers, preserving guest vertex IDs and winding exactly.
        const std::uint64_t expanded = normalized_count(decoded.source, draw.count);
        if (expanded == 0) {
            // Fewer vertices than one complete primitive covers nothing.
            return plan;
        }
        if (expanded > std::numeric_limits<std::uint32_t>::max()) {
            return reject(CommandStatus::BudgetExceeded,
                        "Expanded index count exceeds the supported range");
        }
        try {
            plan.indices.emplace();
            plan.indices->source_format = IndexFormat::U32;
            plan.indices->encoded_format = IndexFormat::U32;
            plan.indices->restart_enabled = false;
            plan.indices->restart_index = restart_index;
            plan.indices->converted.resize(expanded * 4);
            std::vector<std::uint32_t> scratch;
            scratch.reserve(expanded);
            append_normalized(decoded.source, draw.first, draw.count, nullptr, 0,
                              scratch);
            for (std::size_t i = 0; i != scratch.size(); ++i) {
                store32(plan.indices->converted.data() + i * 4, scratch[i]);
            }
        } catch (const std::bad_alloc&) {
            return reject(CommandStatus::BudgetExceeded,
                        "Expanded index allocation failed");
        }
        plan.segments.push_back({0, std::uint32_t(expanded), 0,
                                 native_base_instance, draw.instance_count,
                                 draw.instance_index});
        return plan;
    }

    // Indexed path: decode the index buffer state block.
    const auto type = inline_source ? 2u : stop.registers[index_buffer_type];
    if (type > 2) {
        return reject(CommandStatus::Unsupported, "Index buffer type is unsupported");
    }
    const auto source_format = IndexFormat(type);
    const std::uint64_t element_size = index_size(source_format);
    const std::uint64_t address = inline_source ? 0 :
        (std::uint64_t{stop.registers[index_buffer_start_high]} << 32) |
        stop.registers[index_buffer_start_low];
    const std::uint64_t inclusive_limit = inline_source ? 0 :
        (std::uint64_t{stop.registers[index_buffer_end_high]} << 32) |
        stop.registers[index_buffer_end_low];
    if (!inline_source && (address == 0 || address > inclusive_limit)) {
        return reject(CommandStatus::MemoryFault,
                    "Index buffer address range is invalid", address);
    }
    const std::uint64_t first_byte = std::uint64_t{draw.first} * element_size;
    const std::uint64_t byte_count = std::uint64_t{draw.count} * element_size;
    if (!inline_source && (first_byte > inclusive_limit - address ||
        byte_count > inclusive_limit - address + 1 - first_byte)) {
        return reject(CommandStatus::MemoryFault,
                    "Index buffer element range exceeds the inclusive end address",
                    address);
    }

    const bool normalization = !supported_directly(decoded.source);
    if (normalization && restart_state) {
        // Restart during index conversion is unsupported;
        // Armigo stops with a structured diagnostic instead of dropping the
        // restart indices.
        return reject(CommandStatus::Unsupported,
                    "Primitive restart with normalized topology is unsupported");
    }

    // A scan is required whenever the encoded bytes cannot be the guest bytes:
    // widening for Metal, restart splitting, base-vertex biasing or topology
    // normalization. Otherwise the retained guest range is bound directly and
    // per-vertex reach is enforced at binding time through the retained-range
    // check rather than by reading indices here.
    const bool widening = source_format == IndexFormat::U8;
    // A native base vertex is applied by the draw call, except where the
    // indices are rewritten anyway (inline, widened or normalized streams).
    const bool native_bias = native_base_vertex && draw.base_vertex != 0 &&
        !inline_source && !widening && !normalization;
    const bool biasing = draw.base_vertex != 0 && !native_bias;
    const bool scanned = inline_source || widening || biasing || restart_state || normalization;
    // Settle the normalized expansion size before any plan state exists: a
    // range smaller than one primitive covers nothing, and an oversized range
    // exceeds the budget, so neither carries an IndexPlan.
    const std::uint64_t expanded =
        normalization ? normalized_count(decoded.source, draw.count) : 0;
    if (normalization && expanded == 0) {
        return plan;
    }
    if (normalization && expanded > std::numeric_limits<std::uint32_t>::max()) {
        return reject(CommandStatus::BudgetExceeded,
                      "Expanded index count exceeds the supported range");
    }
    const std::uint64_t start = address + first_byte;
    Scan scan;
    if (scanned) {
        CommandResult error{};
        if (!scan_indices(start, draw.count, source_format, restart_state,
                          restart_index, scan, error, memory, inline_indices)) {
            return RendererError{error};
        }
    }

    // Restart-only streams have no fetches or encoded index storage, even with
    // a negative base vertex. Complete them through the ordinary receipt path.
    if (scanned && scan.runs.empty()) return plan;

    // Validate the bias arithmetic once for every live element, before any
    // materialization, and pick the smallest encoded format that holds all
    // biased values. Unsigned byte sources always widen to at least U16
    // because Metal fetches no byte-sized index elements.
    IndexFormat encoded_format = source_format;
    if (widening && encoded_format == IndexFormat::U8) {
        encoded_format = IndexFormat::U16;
    }
    if (biasing) {
        const std::int64_t base = draw.base_vertex;
        bool needs_u32 = encoded_format == IndexFormat::U32;
        for (const auto& run : scan.runs) {
            for (std::uint32_t i = 0; i != run.length; ++i) {
                const std::int64_t biased =
                    std::int64_t{scan.values[run.begin + i]} + base;
                if (biased < 0) {
                    return reject(CommandStatus::MemoryFault,
                                "Signed base vertex makes an index negative",
                                address);
                }
                if (biased > 65535) needs_u32 = true;
            }
        }
        encoded_format = needs_u32 ? IndexFormat::U32 : IndexFormat::U16;
    }

    plan.indices.emplace();
    // Address and inclusive limit name the checked guest source extent that
    // fed this plan; segments address elements of the encoded buffer, which
    // spans exactly that extent before normalization.
    plan.indices->source_format = source_format;
    plan.indices->encoded_format = encoded_format;
    // Guest indices bound in place with a native base vertex name the whole
    // guest index buffer and draw from the first element: the binding then
    // depends only on index buffer registers, not on draw parameters.
    // Guest limits often reach far past the buffer: the whole-buffer range
    // ends at the mapping holding its start, and must still hold this draw.
    std::uint64_t whole_limit = inclusive_limit;
    bool whole_buffer = native_base_vertex && !scanned && !inline_source && !normalization;
    if (whole_buffer) {
        const auto mappings = memory.mappings();
        const auto after = std::upper_bound(mappings.begin(), mappings.end(), address,
            [](std::uint64_t value, const Mapping& mapping) { return value < mapping.gpu_base; });
        whole_buffer = after != mappings.begin() &&
            address - std::prev(after)->gpu_base < std::prev(after)->mapped_size;
        if (whole_buffer) {
            const auto& mapping = *std::prev(after);
            const auto backed = mapping.backing_size > mapping.buffer_offset
                ? mapping.backing_size - mapping.buffer_offset : 0;
            const auto extent = std::min<std::uint64_t>(mapping.mapped_size, backed);
            whole_buffer = extent > address - mapping.gpu_base;
            whole_limit = std::min(inclusive_limit, mapping.gpu_base + extent - 1);
            whole_buffer = byte_count == 0 || start + byte_count - 1 <= whole_limit;
        }
    }
    plan.indices->whole_buffer = whole_buffer;
    plan.indices->address = whole_buffer ? address : start;
    plan.indices->inclusive_limit = inline_source ? 0 : whole_buffer ? whole_limit : start + byte_count - 1;
    try {
        plan.indices->inline_source.assign(inline_indices.begin(), inline_indices.end());
    } catch (const std::bad_alloc&) {
        return reject(CommandStatus::BudgetExceeded, "Inline index ownership allocation failed");
    }
    plan.indices->restart_enabled = restart_state;
    plan.indices->restart_index = restart_index;

    if (!normalization) {
        if (inline_source || widening || biasing) {
            // Identity element mapping: one encoded element per source
            // element, so segment offsets are unchanged. Restart entries keep
            // their raw value; their segments never reference them.
            try {
                plan.indices->converted.resize(
                    std::size_t(draw.count) * index_size(encoded_format));
            } catch (const std::bad_alloc&) {
                return reject(CommandStatus::BudgetExceeded,
                            "Converted index allocation failed");
            }
            const std::size_t encoded_size = index_size(encoded_format);
            for (std::uint32_t element = 0; element != draw.count; ++element) {
                std::uint32_t value = scan.values[element];
                if (!(restart_state && value == restart_index) && biasing) {
                    value = std::uint32_t(std::int64_t{value} + draw.base_vertex);
                }
                std::byte* out =
                    plan.indices->converted.data() + std::size_t(element) * encoded_size;
                if (encoded_size == 2) {
                    store16(out, value);
                } else {
                    store32(out, value);
                }
            }
        }
        const std::int32_t segment_base = native_bias ? draw.base_vertex : 0;
        for (const auto& run : scan.runs) {
            // Segments address elements of the encoded buffer, whose element
            // zero is the first drawn element; the draw's first offset lives
            // in the bind address.
            plan.segments.push_back({run.begin, run.length, segment_base,
                                     native_base_instance, draw.instance_count,
                                     draw.instance_index});
        }
        if (!scanned) {
            plan.segments.push_back({whole_buffer ? draw.first : 0, draw.count, segment_base,
                                     native_base_instance, draw.instance_count,
                                     draw.instance_index});
        }
    } else {
        // Normalized indexed topology: expand the guest index values through
        // the guest fan and quad patterns into a U32 triangle or line list.
        // The size checks already ran before the scan.
        try {
            plan.indices->encoded_format = IndexFormat::U32;
            plan.indices->converted.resize(expanded * 4);
            std::vector<std::uint32_t> scratch;
            scratch.reserve(expanded);
            append_normalized(decoded.source, draw.first, draw.count,
                              scan.values.data(), draw.base_vertex, scratch);
            for (std::size_t i = 0; i != scratch.size(); ++i) {
                store32(plan.indices->converted.data() + i * 4, scratch[i]);
            }
        } catch (const std::bad_alloc&) {
            return reject(CommandStatus::BudgetExceeded,
                          "Expanded index allocation failed");
        }
        plan.segments.push_back({0, std::uint32_t(expanded), 0,
                                 native_base_instance, draw.instance_count,
                                 draw.instance_index});
    }
    if (!scanned) {
        return plan;
    }
    // Exact per-vertex reach: the maximum live index value is known from the
    // scan, bias included. Direct unscanned ranges delegate the per-vertex
    // check to binding time through the retained-range check.
    const std::int64_t biased_max = std::int64_t{scan.max_value} +
        (biasing || native_bias ? draw.base_vertex : 0);
    const std::uint64_t max_fetch = biased_max > 0 ? std::uint64_t(biased_max) : 0;
    for (const auto& attribute : vertex.attributes) {
        if (attribute.constant) continue;
        std::size_t slot = 0;
        for (std::size_t i = 0; i != vertex.streams.size(); ++i) {
            if (vertex.streams[i].stream == attribute.stream) slot = i;
        }
        if (vertex.steps[slot].step != VertexStep::PerVertex) continue;
        const VertexStreamPlan& stream = vertex.streams[slot];
        const std::uint64_t reach =
            max_fetch * stream.stride +
            (std::uint64_t{attribute.offset} + attribute.format.size_bytes - 1);
        if (stream.stride != 0 && reach > stream.inclusive_limit - stream.address) {
            return reject(CommandStatus::MemoryFault,
                          "Vertex fetch exceeds the inclusive stream limit",
                          stream.address);
        }
    }
    return plan;
}

}  // namespace armigo::graphics
