// ============================================================================
// Armigo Shader - Native Shader Translation
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include "mesh_stage.h"
#include <algorithm>
#include <array>
#include <string_view>
#include <vector>

namespace armigo::shader_native {
namespace {
std::string number(std::uint64_t value) { return std::to_string(value); }

// A parameter list split at its top-level commas.
std::vector<std::string> split_parameters(std::string_view text) {
    std::vector<std::string> out;
    std::string current;
    int depth = 0;
    for (const char c : text) {
        if (c == ',' && depth == 0) { out.push_back(current); current.clear(); continue; }
        if (c == '(' || c == '[' || c == '<') ++depth;
        if (c == ')' || c == ']' || c == '>') --depth;
        current += c;
    }
    if (current.find_first_not_of(" \n") != std::string::npos) out.push_back(current);
    for (auto& parameter : out) {
        const auto first = parameter.find_first_not_of(" \n"), last = parameter.find_last_not_of(" \n");
        parameter = parameter.substr(first, last - first + 1);
    }
    return out;
}
// A parameter without its [[...]] attribute, and its name.
std::string plain_parameter(const std::string& parameter) {
    const auto at = parameter.find(" [[");
    return at == std::string::npos ? parameter : parameter.substr(0, at);
}
std::string parameter_name(const std::string& plain) {
    const auto at = plain.find_last_of(" &*");
    return plain.substr(at + 1);
}

std::string word(const std::string& index) {
    return "atomic_load_explicit(&armigo_shared_words[" + index + "], memory_order_relaxed)";
}
std::string component_name(const StageComponent& c) {
    return "generic" + number(c.attribute) + "_" + number(c.component);
}
std::string component_location(const StageComponent& c) {
    return "[[user(locn" + number(c.attribute) + "_" + number(c.component) + ")]]";
}
} // namespace

std::array<std::uint8_t, 32> mesh_generic_outputs(const Program& program) {
    std::array<std::uint8_t, 32> outputs{};
    auto memory = stage_memory(program);
    if (const auto* layout = std::get_if<StageMemory>(&memory))
        for (const auto* list : {&layout->vertex, &layout->primitive})
            for (const auto& c : *list)
                if (c.kind == StageComponent::Kind::generic) outputs[c.attribute] |= static_cast<std::uint8_t>(1u << c.component);
    return outputs;
}

std::optional<std::string> wrap_workgroup_stage(ir::MslArtifact& artifact, const Program& program, const StageMemory& memory,
                                                std::optional<ir::DepthConvention> depth,
                                                const std::optional<std::array<std::uint8_t, 32>>& fragment_reads) {
    if (!program.mesh_launch) return "task or mesh launch facts are absent";
    const auto& launch = *program.mesh_launch;
    const bool task = program.stage == Stage::task;
    auto& source = artifact.source;
    const std::string head = "kernel void " + artifact.entry_point + "(";
    const auto start = source.find(head);
    if (start == std::string::npos || source.find(head, start + 1) != std::string::npos) return "the stage's kernel is not found";
    const auto close = source.find(") {\n", start);
    // The kernel is the source's last function: its body ends at the last brace.
    const auto end = source.rfind("}\n");
    if (close == std::string::npos || end == std::string::npos || end < close || end + 2 != source.size())
        return "the stage's kernel does not end the source";
    const auto parameters = split_parameters(std::string_view(source).substr(start + head.size(), close - start - head.size()));
    std::string body = source.substr(close + 4, end - close - 4);
    const std::string shared = "    threadgroup atomic_uint armigo_shared_words[" + number(memory.bytes / 4) + "];\n";
    const auto shared_at = body.find(shared);
    if (shared_at == std::string::npos) return "the stage's kernel does not hold its stage buffer";
    body.erase(shared_at, shared.size());
    if (body.find("threadgroup uint ") != std::string::npos) return "workgroup memory beside the stage buffer is not admitted";

    // The entry reuses the kernel's thread position and adds the threadgroup's.
    std::string thread;
    std::vector<std::string> plain, names, extras;
    for (const auto& parameter : parameters) {
        plain.push_back(plain_parameter(parameter));
        names.push_back(parameter_name(plain.back()));
        if (parameter.find("[[thread_position_in_threadgroup]]") != std::string::npos)
            thread = names.back() + (parameter.rfind("uint3 ", 0) == 0 ? ".x" : "");
        if (parameter.find("[[threadgroup_position_in_grid]]") != std::string::npos) return "the stage reads its group id";
    }
    if (thread.empty()) {
        thread = "armigo_stage_thread";
        extras.push_back("uint armigo_stage_thread [[thread_position_in_threadgroup]]");
    }
    extras.push_back("uint armigo_stage_group [[threadgroup_position_in_grid]]");
    // The draw's first workgroup: word 0 of the draw-parameter record, in the
    // first buffer index the stage leaves free.
    const bool first = task || !launch.task;
    if (first) {
        const auto& plan = artifact.reflection;
        std::array<bool, 31> occupied{};
        for (const auto& resource : plan.resources)
            if (const auto* range = std::get_if<ir::MetalRange>(&resource.location); range && range->space == ir::MetalNamespace::buffer)
                for (std::uint32_t i = 0; i < range->count && range->first + i < occupied.size(); ++i) occupied[range->first + i] = true;
        for (const auto index : {plan.argument_buffer ? std::optional<std::uint32_t>(plan.argument_buffer->buffer_index) : std::nullopt,
                                 plan.custom_border ? std::optional<std::uint32_t>(plan.custom_border->buffer_index) : std::nullopt,
                                 plan.draw_support_buffer_index})
            if (index && *index < occupied.size()) occupied[*index] = true;
        // A task stage's status words take the next index.
        std::uint32_t free = 0;
        while (free < occupied.size() && (occupied[free] || (task && (free + 1 == occupied.size() || occupied[free + 1])))) ++free;
        if (free == occupied.size()) return "no buffer index is free for the draw parameters";
        artifact.reflection.draw_support_buffer_index = free;
        extras.push_back("constant uint* armigo_stage_first [[buffer(" + number(free) + ")]]");
        if (task) extras.push_back("device atomic_uint* armigo_stage_status [[buffer(" + number(free + 1) + ")]]");
    }

    const auto threads = number(launch.threads) + "u", words = number(memory.bytes / 4) + "u";
    // The payload words the stage writes or reads; Metal's payload type holds at least one.
    const auto payload_words = (launch.payload_bytes + 3) / 4;
    const auto output = memory.output / 4, input = memory.input / 4;
    std::string out = source.substr(0, start);
    out += "static inline __attribute__((always_inline)) void armigo_stage_body(";
    for (const auto& parameter : plain) out += parameter + ", ";
    out += "threadgroup atomic_uint* armigo_shared_words) {\n" + body + "}\n\n";
    out += "struct ArmigoStagePayload { uint words[" + number(std::max<std::uint32_t>(payload_words, 1)) + "]; };\n";
    std::string call = "    armigo_stage_body(";
    for (const auto& name : names) call += name + ", ";
    call += "armigo_shared_words);\n";
    const std::string barrier = "    threadgroup_barrier(mem_flags::mem_threadgroup);\n";
    const auto store = [](const std::string& index, const std::string& value) {
        return "atomic_store_explicit(&armigo_shared_words[" + index + "], " + value + ", memory_order_relaxed);";
    };
    std::string signature;
    for (const auto& parameter : parameters) signature += parameter + ",\n    ";
    for (const auto& extra : extras) signature += extra + ",\n    ";
    const std::string prologue = "    threadgroup atomic_uint armigo_shared_words[" + number(memory.bytes / 4) + "];\n"
        "    for (uint i = " + thread + "; i < " + words + "; i += " + threads + ") " + store("i", "0u") + "\n" + barrier;

    if (task) {
        // A payload shorter than 8 bytes holds no launch count and launches
        // none. Metal launches at most 1,024 mesh workgroups from one object
        // threadgroup. A task workgroup asking for more launches none and
        // records the request and its program in the status words, which
        // the driver reports: no partial launch.
        const auto tag = program.stage_identity.size() >= 8 ? std::stoul(program.stage_identity.substr(0, 8), nullptr, 16) : 0ul;
        artifact.entry_point = "armigo_object_main";
        out += "[[object]] void armigo_object_main(" + signature +
               "object_data ArmigoStagePayload& armigo_payload [[payload]],\n    mesh_grid_properties armigo_grid) {\n" + prologue +
               "    if (" + thread + " == 0u) " + store("0", "armigo_stage_first[0] + armigo_stage_group") + "\n" + barrier + call + barrier +
               "    for (uint i = " + thread + "; i < " + number(payload_words) + "u; i += " + threads + ") armigo_payload.words[i] = " +
               word(number(output) + "u + i") + ";\n"
               "    if (" + thread + " == 0u) {\n"
               "        const uint armigo_launches = " + (launch.payload_bytes >= 8 ? word(number(output + 1) + "u") : std::string("0u")) + ";\n"
               "        if (armigo_launches > " + number(metal_mesh_grid) + "u) {\n"
               "            atomic_fetch_max_explicit(&armigo_stage_status[0], armigo_launches, memory_order_relaxed);\n"
               "            atomic_fetch_max_explicit(&armigo_stage_status[1], " + number(tag) + "u, memory_order_relaxed);\n"
               "        }\n"
               "        armigo_grid.set_threadgroups_per_grid(uint3(armigo_launches > " + number(metal_mesh_grid) + "u ? 0u : armigo_launches, 1u, 1u));\n"
               "    }\n}\n";
        source = std::move(out);
        return std::nullopt;
    }

    if (launch.max_vertices == 0 || launch.max_primitives == 0) return "a mesh without vertices or primitives is not admitted";
    if (!depth) return "the mesh stage's clip convention is unknown";
    static constexpr const char* topologies[]{"", "point", "line", "triangle"};
    const bool points = launch.primitive_vertices == 1;
    // Components in stage-buffer order: each one's row in its block.
    std::array<std::optional<std::size_t>, 4> position{};
    std::optional<std::size_t> point_size;
    std::string vertex_members, vertex_values, primitive_members, primitive_values;
    // With the fragment stage's reads, a generic component it does not read
    // is not output, and one it reads that no output map enables is output
    // with its default.
    const auto read = [&](unsigned attribute, unsigned component) { return !fragment_reads || ((*fragment_reads)[attribute] >> component & 1u); };
    std::array<std::uint8_t, 32> written{};
    std::size_t vertex_scalars = points ? 1 : 0, primitive_scalars = 0;
    for (std::size_t row = 0; row < memory.vertex.size(); ++row) {
        const auto& c = memory.vertex[row];
        const auto value = "as_type<float>(" + word("b + " + number(row * stage_block) + "u") + ")";
        if (c.kind == StageComponent::Kind::position) position[c.component] = row;
        else if (c.kind == StageComponent::Kind::point_size) point_size = row;
        else {
            written[c.attribute] |= static_cast<std::uint8_t>(1u << c.component);
            if (!read(c.attribute, c.component)) continue;
            vertex_members += "    float " + component_name(c) + " " + component_location(c) + ";\n";
            vertex_values += "        v." + component_name(c) + " = " + value + ";\n";
            ++vertex_scalars;
        }
    }
    if (std::none_of(position.begin(), position.end(), [](const auto& row) { return row.has_value(); }))
        return "the mesh stage's output map has no position";
    for (std::size_t row = 0; row < memory.primitive.size(); ++row) {
        const auto& c = memory.primitive[row];
        written[c.attribute] |= static_cast<std::uint8_t>(1u << c.component);
        if (!read(c.attribute, c.component)) continue;
        primitive_members += "    float " + component_name(c) + " " + component_location(c) + ";\n";
        primitive_values += "        p." + component_name(c) + " = as_type<float>(" + word("b + " + number(row * stage_block) + "u") + ");\n";
        ++primitive_scalars;
    }
    if (fragment_reads)
        for (std::uint8_t attribute = 0; attribute < 32; ++attribute)
            for (std::uint8_t component = 0; component < 4; ++component) {
                if (!((*fragment_reads)[attribute] >> component & 1u) || (written[attribute] >> component & 1u)) continue;
                const StageComponent c{StageComponent::Kind::generic, attribute, component};
                vertex_members += "    float " + component_name(c) + " " + component_location(c) + ";\n";
                vertex_values += "        v." + component_name(c) + " = " + std::string(component == 3 ? "1.0f" : "0.0f") + ";\n";
                ++vertex_scalars;
            }
    // Metal counts a mesh's declared outputs against 32 KB: each vertex its
    // position and scalar members, each primitive its members; index bytes
    // are not counted. A point mesh always has a point size.
    const auto output_bytes = std::uint64_t{launch.max_vertices} * (16 + 4 * vertex_scalars) +
                              std::uint64_t{launch.max_primitives} * 4 * primitive_scalars;
    if (output_bytes > metal_mesh_output_bytes) return "mesh outputs exceed Metal's 32 KB mesh output";
    std::string position_value = "float4(";
    for (unsigned c = 0; c < 4; ++c) {
        if (c) position_value += ", ";
        position_value += position[c] ? "as_type<float>(" + word("b + " + number(*position[c] * stage_block) + "u") + ")"
                                      : std::string(c == 3 ? "1.0f" : "0.0f");
    }
    position_value += ")";
    const bool primitives = primitive_scalars != 0;
    out += "struct ArmigoMeshVertex {\n    float4 position [[position, invariant]];\n" +
           std::string(points ? "    float point_size [[point_size]];\n" : "") + vertex_members + "};\n";
    if (primitives) out += "struct ArmigoMeshPrimitive {\n" + primitive_members + "};\n";
    out += "using ArmigoMesh = metal::mesh<ArmigoMeshVertex, " + std::string(primitives ? "ArmigoMeshPrimitive" : "void") + ", " +
           number(launch.max_vertices) + ", " + number(launch.max_primitives) + ", metal::topology::" +
           topologies[launch.primitive_vertices] + ">;\n";
    const auto blocks = [](std::uint32_t count) { return (count + stage_block - 1) / stage_block; };
    const auto vertex_stride = static_cast<std::uint32_t>(memory.vertex.size()) * stage_block;
    const auto primitive_stride = static_cast<std::uint32_t>(memory.primitive.size()) * stage_block;
    const auto primitive_output = output + blocks(launch.max_vertices) * vertex_stride;
    const auto count_word = memory.primitives / 4;
    artifact.entry_point = "armigo_mesh_main";
    out += "[[mesh]] void armigo_mesh_main(" + signature + "ArmigoMesh armigo_mesh" +
           (launch.task ? ",\n    const object_data ArmigoStagePayload& armigo_payload [[payload]]" : "") + ") {\n" + prologue;
    if (launch.task && payload_words)
        out += "    for (uint i = " + thread + "; i < " + number(payload_words) + "u; i += " + threads + ") " +
               store(number(input) + "u + i", "armigo_payload.words[i]") + "\n" + barrier;
    if (launch.task)
        out += std::string() +
               "    if (" + thread + " == 0u) { " + store("0", "armigo_stage_group") + " " + store(number(input), "armigo_stage_group") + " }\n";
    else
        out += "    if (" + thread + " == 0u) " + store("0", "armigo_stage_first[0] + armigo_stage_group") + "\n";
    out += barrier + call + barrier +
           "    const uint armigo_count = min(" + word(number(count_word) + "u") + " >> 24u, " + number(launch.max_primitives) + "u);\n"
           "    if (" + thread + " == 0u) armigo_mesh.set_primitive_count(armigo_count);\n"
           "    for (uint n = " + thread + "; n < " + number(launch.max_vertices) + "u; n += " + threads + ") {\n"
           "        const uint b = " + number(output) + "u + (n / 32u) * " + number(vertex_stride) + "u + n % 32u;\n"
           "        ArmigoMeshVertex v;\n"
           "        v.position = " + position_value + ";\n" +
           (points ? "        v.point_size = " + (point_size ? "as_type<float>(" + word("b + " + number(*point_size * stage_block) + "u") + ")" : std::string("1.0f")) + ";\n" : "") +
           vertex_values +
           (*depth == ir::DepthConvention::negative_one_to_one
                ? "        const float armigo_depth_sum = v.position.z + v.position.w;\n        v.position.z = armigo_depth_sum * 0.5f;\n" : "") +
           "        armigo_mesh.set_vertex(n, v);\n    }\n";
    if (primitives)
        out += "    for (uint n = " + thread + "; n < armigo_count; n += " + threads + ") {\n"
               "        const uint b = " + number(primitive_output) + "u + (n / 32u) * " + number(primitive_stride) + "u + n % 32u;\n"
               "        ArmigoMeshPrimitive p;\n" + primitive_values +
               "        armigo_mesh.set_primitive(n, p);\n    }\n";
    out += "    for (uint n = " + thread + "; n < armigo_count * " + number(launch.primitive_vertices) + "u; n += " + threads + ") {\n"
           "        const uint at = " + number(memory.primitives + 4) + "u + n;\n"
           "        armigo_mesh.set_index(n, (" + word("at / 4u") + " >> ((at & 3u) * 8u)) & 255u);\n    }\n}\n";
    source = std::move(out);
    return std::nullopt;
}
} // namespace armigo::shader_native
