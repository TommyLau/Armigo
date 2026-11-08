// ============================================================================
// Armigo Shader - Native Shader Translation
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include "stage_internal.h"
#include "../dispatch.h"
#include <map>
#include <set>

namespace armigo::shader::pipeline {
namespace {
using namespace msl;
unsigned dimensions(TextureDimension dimension) {
    switch (dimension) {
    case TextureDimension::d1: case TextureDimension::buffer: return 1;
    case TextureDimension::d2: return 2;
    case TextureDimension::d3: case TextureDimension::cube: return 3;
    } fail("Invalid texture dimension");
}
std::string vector(MslContext& c, const Operation& op, const std::vector<InputIndex>& inputs, ScalarType type, unsigned count) {
    require(inputs.size() == count, "Texture vector does not match its dimension");
    std::string result;
    for (const auto index : inputs) { if (!result.empty()) result += ", "; result += input(c, op, index, type); }
    return count == 1 ? result : scalar(type) + number(count) + "(" + result + ")";
}
}
Status generate_stage_texture(MslContext& c, OperationId id) {
    return callback(c, id, [&] {
        const auto& op = c.operation(id); const auto* info = std::get_if<TextureInfo>(&op.info);
        require(info && info->resource && info->descriptor, "Texture resource or descriptor is unresolved", DiagnosticCode::unresolved_resource);
        const auto& t = *info; const auto& d = *t.descriptor; const auto shape = d.shape;
        plain_numeric(op);
        require(!t.coherent && !t.lod_clamp,
                "Texture coherency and LOD clamps require explicit lowering", DiagnosticCode::unsupported_modifier);
        require(!t.atomic || op.opcode == Opcode::image_atomic,
                "Atomic metadata is only valid on an image atomic", DiagnosticCode::invalid_ir);
        const auto& binding = planned(c.resources(), *t.resource);
        const auto* use = std::get_if<TextureUse>(&binding.use);
        require(use && use->descriptor == d && std::holds_alternative<TextureKey>(t.texture) && std::get<TextureKey>(t.texture) == use->key,
                "Texture operation disagrees with its resolved resource");
        require(std::find(use->modes.begin(), use->modes.end(), t.mode) != use->modes.end(), "Texture mode is absent from resource requirements");
        const bool image = op.opcode == Opcode::image_load || op.opcode == Opcode::image_store ||
            op.opcode == Opcode::image_atomic;
        const bool atomic = op.opcode == Opcode::image_atomic;
        require((use->kind == ResourceTextureKind::storage_image) == image, "Texture/image resource kind mismatch");
        const bool write = op.opcode == Opcode::image_store;
        require(write ? use->access != Access::read : use->access != Access::write, "Texture access direction mismatch");
        const bool query = op.opcode == Opcode::texture_query;
        const bool fetch = t.mode == TextureMode::fetch;
        const bool lod_query = t.mode == TextureMode::query_lod;
        const bool sampling = !query && !fetch;
        const bool gather = t.mode == TextureMode::gather;
        require(t.component_mask && t.component_mask <= 15 && t.gather_component < 4, "Invalid texture component mask");
        require((write ? t.data.size() : op.outputs.size()) == static_cast<std::size_t>(std::popcount(t.component_mask)), "Texture component count mismatch");
        require(t.array_index.has_value() == (use->array_length > 1), "Texture array index disagrees with resource count");
        require(t.array_index || !t.array_index_upper_bound, "Array bound without an array index");
        std::string tex = take(c.resource(*t.resource));
        require(use->pool == t.pool_texture.has_value(), "Texture pool role disagrees with its resource");
        if (use->pool) {
            // The pool table's entry at the handle's texture ID, marked read
            // (the project documentation).
            const auto id = "((" + input(c, op, *t.pool_texture, ScalarType::u32) + ") & 0xfffffu)";
            tex = "(&" + tex + ".e[0])[armigo_pool_read(" + tex + ".seen, " + id + ", " + tex + ".n)]";
        }
        if (t.array_index) {
            require(t.array_index->value < op.inputs.size(), "Resource-array index role is out of range");
            if (const auto* literal = std::get_if<LiteralBits>(&op.inputs[t.array_index->value])) {
                require(literal->bits < use->array_length && (!t.array_index_upper_bound || literal->bits <= *t.array_index_upper_bound), "Texture array index exceeds its binding/proven bound");
            } else {
                require(use->dynamically_indexed && t.array_index_upper_bound, "Dynamic texture array index has no proven bound", DiagnosticCode::missing_fact);
                require(*t.array_index_upper_bound < use->array_length, "Texture index bound exceeds binding range");
            }
            tex += "[" + input(c, op, *t.array_index, ScalarType::u32) + "]";
        }
        const bool needs_sampler = sampling || lod_query;
        require(t.sampler_resource.has_value() == needs_sampler, "Sampler presence disagrees with texture operation");
        std::string samp;
        if (needs_sampler) {
            const auto& sampler_binding = planned(c.resources(), *t.sampler_resource);
            const auto* sampler = std::get_if<SamplerUse>(&sampler_binding.use);
            // The stage's one sampler pool is not typed by the texture's
            // comparison. Pooled sampling still requires normalized
            // coordinates, which check_texture enforces on the descriptor.
            require(sampler && (sampler->pool || (sampler->comparison == d.comparison &&
                                                  sampler->normalized_coordinates == d.normalized_coordinates)),
                    "Sampler/texture descriptor mismatch");
            require(!t.sampler || (std::holds_alternative<TextureKey>(*t.sampler) && std::get<TextureKey>(*t.sampler) == sampler->key), "Unresolved or mismatched sampler handle");
            require(sampler->array_length == 1 || sampler->array_length == use->array_length, "Sampler array index cannot be resolved");
            samp = take(c.resource(*t.sampler_resource));
            require(sampler->pool == t.pool_sampler.has_value(), "Sampler pool role disagrees with its resource");
            if (sampler->pool)
                samp = "(&" + samp + ".e[0])[min((" + input(c, op, *t.pool_sampler, ScalarType::u32) + ") >> 20u, " + samp + ".n)]";
            if (sampler->array_length > 1) samp += "[" + input(c, op, *t.array_index, ScalarType::u32) + "]";
        } else require(!t.sampler, "Unused sampler handle on texture read/query");
        const auto count = dimensions(shape.dimension);
        if (shape.dimension == TextureDimension::d1) {
            require(!sampling || (t.mode == TextureMode::sample_implicit && t.offsets.empty()),
                    "Metal 1D sampling cannot preserve explicit LOD, gradients or offsets", DiagnosticCode::unsupported_capability);
            if (t.level_or_bias) {
                require(t.level_or_bias->value < op.inputs.size(), "Level role is out of range");
                const auto* level = std::get_if<LiteralBits>(&op.inputs[t.level_or_bias->value]);
                require(level && level->bits == 0, "Metal 1D reads/size queries require constant level zero", DiagnosticCode::unsupported_capability);
            }
        }
        require(!t.compare || sampling, "Comparison reference is only valid for sampling");
        require(!sampling || t.compare.has_value() == d.comparison, "Comparison reference disagrees with depth sampler");
        require(!sampling || (!shape.multisample && shape.dimension != TextureDimension::buffer), "This texture shape cannot be sampled");
        require(!sampling || d.scalar == ScalarType::f32 || d.scalar == ScalarType::f16 || gather,
                "Integer textures require fetch/gather", DiagnosticCode::unsupported_capability);
        require(!lod_query || (c.program().ir.stage == Stage::fragment && c.resources().profile.supports_query_lod &&
                shape.dimension != TextureDimension::buffer && shape.dimension != TextureDimension::d1 && !shape.multisample),
                "LOD query is unavailable for this stage/profile/shape", DiagnosticCode::unsupported_capability);
        // Outside the fragment stage, legalization has rewritten implicit and
        // bias sampling into explicit-level sampling, and a Metal gather always
        // reads the base level, so only those two unlowered modes stay
        // fragment-exclusive here.
        if (t.mode == TextureMode::sample_implicit || t.mode == TextureMode::sample_bias)
            require(c.program().ir.stage == Stage::fragment, "Implicit derivative texture mode requires fragment stage", DiagnosticCode::unsupported_capability);
        if (!query) require(op.numeric.scalar == d.scalar, "Texture component conversion must be explicit", DiagnosticCode::unsupported_modifier);
        if (query) require(op.numeric.scalar == (lod_query ? ScalarType::f32 : ScalarType::u32), "Query result interpretation must be explicit");
        require(t.gradients_x.empty() == (t.mode != TextureMode::sample_gradients) && t.gradients_y.empty() == (t.mode != TextureMode::sample_gradients), "Unexpected or missing texture gradients");
        const bool needs_level = t.mode == TextureMode::sample_level || t.mode == TextureMode::sample_bias ||
            (fetch && !image && !shape.multisample && shape.dimension != TextureDimension::buffer) ||
            (t.mode == TextureMode::query_size && !shape.multisample && shape.dimension != TextureDimension::buffer);
        require(t.level_or_bias.has_value() == needs_level, "Unexpected or missing texture level/bias");
        require(t.sample_index.has_value() == (fetch && shape.multisample), "Unexpected or missing texture sample index");
        require(t.array_layer.has_value() == (shape.array && (!query || lod_query)), "Unexpected or missing texture array layer");
        require(!query || (t.offsets.empty() && t.gradients_x.empty() && t.gradients_y.empty() && t.data.empty() && !t.compare), "Query carries sampling/store metadata");
        if (query && !lod_query) {
            require(t.coordinates.empty() && op.outputs.size() == 1 && t.component_mask == 1, "Dimension query must select one scalar component");
            std::string call;
            if (t.mode == TextureMode::query_levels) {
                require(!shape.multisample && shape.dimension != TextureDimension::buffer && t.query_component == 0, "Invalid mip-count query");
                call = ".get_num_mip_levels()";
            } else if (t.mode == TextureMode::query_samples) {
                require(shape.multisample && t.query_component == 0, "Sample-count query requires a multisample texture");
                call = ".get_num_samples()";
            } else {
                require(t.mode == TextureMode::query_size, "Invalid query mode");
                unsigned size_dimensions = shape.dimension == TextureDimension::cube ? 2 : count;
                require(t.query_component < size_dimensions + unsigned(shape.array), "Dimension query component is out of range");
                if (t.query_component == size_dimensions) call = ".get_array_size()";
                else {
                    const char* names[] = {"get_width", "get_height", "get_depth"};
                    call = "." + std::string(names[t.query_component]) + "(";
                    if (t.level_or_bias) call += input(c, op, *t.level_or_bias, ScalarType::u32);
                    call += ")";
                }
            }
            take(c.assign(output(op, 0), tex + call, ScalarType::u32)); return;
        }
        const auto coords = vector(c, op, t.coordinates, fetch ? ScalarType::s32 : ScalarType::f32, count);
        if (atomic) {
            // A read-write 2D texture provides the atomic methods. Each call
            // takes and returns four components, and an R32 image uses the
            // first. Increment and decrement have no wrapping method, so they
            // add one. Compare-and-swap writes the observed vector back
            // through its expected argument.
            require(op.outputs.size() == 1 && t.component_mask == 1 &&
                    t.data.size() == (*t.atomic == AtomicKind::compare_exchange ? 2U : 1U),
                    "Invalid image atomic result shape");
            require(d.scalar == ScalarType::u32 || d.scalar == ScalarType::s32,
                    "Image atomics require integer components");
            require(count == 2 && !shape.array, "Image atomics require a non-array 2D image");
            const auto location = "uint2(" + coords + ")";
            const auto vector4 = [&](const std::string& component) {
                return scalar(d.scalar) + "4(" + component + ", 0, 0, 0)";
            };
            if (*t.atomic == AtomicKind::compare_exchange) {
                const auto observed = "atomic_observed" + number(id.value);
                take(c.statement(scalar(d.scalar) + "4 " + observed + " = " +
                    vector4(input(c, op, t.data[0], d.scalar)) + ";"));
                take(c.statement("(void)" + tex + ".atomic_compare_exchange_weak(" + location +
                    ", &" + observed + ", " + vector4(input(c, op, t.data[1], d.scalar)) + ");"));
                take(c.assign(output(op, 0), observed + ".x", d.scalar));
            } else {
                std::string call;
                std::string value = input(c, op, t.data.back(), d.scalar);
                switch (*t.atomic) {
                case AtomicKind::add: call = "atomic_fetch_add"; break;
                case AtomicKind::minimum: call = "atomic_fetch_min"; break;
                case AtomicKind::maximum: call = "atomic_fetch_max"; break;
                case AtomicKind::increment:
                    call = "atomic_fetch_add";
                    value = scalar(d.scalar) + "(1)";
                    break;
                case AtomicKind::decrement:
                    call = "atomic_fetch_add";
                    value = scalar(d.scalar) + "(-1)";
                    break;
                case AtomicKind::bit_and: call = "atomic_fetch_and"; break;
                case AtomicKind::bit_or: call = "atomic_fetch_or"; break;
                case AtomicKind::bit_xor: call = "atomic_fetch_xor"; break;
                case AtomicKind::exchange: call = "atomic_exchange"; break;
                default: fail("Image atomic operation is unsupported", DiagnosticCode::unsupported_capability);
                }
                take(c.assign(output(op, 0), tex + "." + call + "(" + location + ", " +
                    vector4(value) + ").x", d.scalar));
            }
            return;
        }
        if (lod_query) {
            require(t.component_mask == 1 && op.outputs.size() == 1 && t.query_component < 2,
                    "LOD query requires one scalar selection from the host pair");
            // Host X is clamped and Y is unclamped. TMML's reversal and signed
            // fixed-point conversion are explicit operations in the guest IR.
            const auto method = t.query_component ? ".calculate_unclamped_lod(" : ".calculate_clamped_lod(";
            take(c.assign(output(op, 0), tex + method + samp + ", " + coords + ")", ScalarType::f32));
            return;
        }
        std::string expression;
        if (fetch) {
            require(shape.dimension != TextureDimension::cube, "Cube texel reads require explicit face coordinates", DiagnosticCode::unsupported_capability);
            std::string location = coords;
            if (!t.offsets.empty()) location = "(" + coords + " + " + vector(c, op, t.offsets, ScalarType::s32, count) + ")";
            location = "uint" + (count == 1 ? std::string{} : number(count)) + "(" + location + ")";
            expression = tex + (write ? ".write(" : ".read(");
            if (write) {
                require(t.component_mask == 15, "Partial image stores require explicit read/modify/write semantics", DiagnosticCode::unsupported_capability);
                expression += vector(c, op, t.data, d.scalar, 4) + ", ";
            }
            expression += location;
            if (t.array_layer) expression += ", " + input(c, op, *t.array_layer, ScalarType::u32);
            if (t.sample_index) expression += ", " + input(c, op, *t.sample_index, ScalarType::u32);
            if (t.level_or_bias) expression += ", " + input(c, op, *t.level_or_bias, ScalarType::u32);
            expression += ")";
        } else {
            require(!gather || shape.dimension == TextureDimension::d2 || shape.dimension == TextureDimension::cube,
                    "Gather requires a 2D or cube texture", DiagnosticCode::unsupported_capability);
            std::string method = gather ? "gather" : "sample";
            if (t.compare) method += "_compare";
            // A Metal gather takes no level argument in any stage: it always
            // reads the base level, like the guest gather, which encodes none.
            expression = tex + "." + method + "(" + samp + ", " + coords;
            if (t.array_layer) expression += ", " + input(c, op, *t.array_layer, ScalarType::u32);
            if (t.compare) expression += ", " + input(c, op, *t.compare, ScalarType::f32);
            if (t.mode == TextureMode::sample_level || t.mode == TextureMode::sample_bias)
                expression += std::string(t.mode == TextureMode::sample_level ? ", level(" : ", bias(") + input(c, op, *t.level_or_bias, ScalarType::f32) + ")";
            if (t.mode == TextureMode::sample_gradients) {
                require(shape.dimension != TextureDimension::d1, "1D gradient sampling needs explicit lowering", DiagnosticCode::unsupported_capability);
                const auto gradient = shape.dimension == TextureDimension::cube ? "gradientcube" : count == 3 ? "gradient3d" : "gradient2d";
                expression += ", " + std::string(gradient) + "(" + vector(c, op, t.gradients_x, ScalarType::f32, count) + ", " + vector(c, op, t.gradients_y, ScalarType::f32, count) + ")";
            }
            if (gather && !t.compare && !t.offsets.empty() && t.offsets.size() == 4 * count) {
                // A per-texel offset set is four gathers. Texel k is the
                // (i0, j0) texel of the footprint at offsets[k], which one
                // gather returns as its w component. Literal offsets stay
                // integer literals. A non-literal set passes each texel's
                // axis pair as a run-time int2.
                require(shape.dimension != TextureDimension::cube && count == 2,
                        "Per-texel gather offsets require a 2D texture", DiagnosticCode::unsupported_capability);
                bool literal_offsets = true;
                for (auto index : t.offsets)
                    literal_offsets = literal_offsets && index.value < op.inputs.size() &&
                        std::holds_alternative<LiteralBits>(op.inputs[index.value]);
                std::string call = tex + ".gather(" + samp + ", " + coords;
                if (t.array_layer) call += ", " + input(c, op, *t.array_layer, ScalarType::u32);
                const auto component = ", component::" + std::string(1, "xyzw"[t.gather_component]);
                std::string combined;
                if (literal_offsets) {
                    for (auto index : t.offsets) {
                        require(index.value < op.inputs.size() && std::holds_alternative<LiteralBits>(op.inputs[index.value]), "Sample offset must be a constant", DiagnosticCode::unsupported_modifier);
                        auto value = static_cast<std::int32_t>(std::get<LiteralBits>(op.inputs[index.value]).bits);
                        require(value >= -8 && value <= 7, "Sample offset exceeds Metal range", DiagnosticCode::unsupported_modifier);
                    }
                    const auto offset_word = [&](InputIndex index) {
                        return std::to_string(static_cast<std::int32_t>(std::get<LiteralBits>(op.inputs[index.value]).bits));
                    };
                    for (unsigned texel = 0; texel < 4; ++texel) {
                        if (!combined.empty()) combined += ", ";
                        combined += call + ", int2(" + offset_word(t.offsets[2 * texel]) + ", " +
                            offset_word(t.offsets[2 * texel + 1]) + ")" + component + ").w";
                    }
                } else {
                    for (auto index : t.offsets) {
                        if (index.value < op.inputs.size() && std::holds_alternative<LiteralBits>(op.inputs[index.value])) {
                            auto value = static_cast<std::int32_t>(std::get<LiteralBits>(op.inputs[index.value]).bits);
                            require(value >= -8 && value <= 7, "Sample offset exceeds Metal range", DiagnosticCode::unsupported_modifier);
                        }
                    }
                    for (unsigned texel = 0; texel < 4; ++texel) {
                        if (!combined.empty()) combined += ", ";
                        combined += call + ", int2(" + input(c, op, t.offsets[2 * texel], ScalarType::s32) + ", " +
                            input(c, op, t.offsets[2 * texel + 1], ScalarType::s32) + ")" + component + ").w";
                    }
                }
                expression = scalar(d.scalar) + "4(" + combined + ")";
            } else if (!t.offsets.empty()) {
                require(shape.dimension != TextureDimension::cube && t.offsets.size() == count,
                        "Multiple gather offsets and cube offsets require explicit lowering", DiagnosticCode::unsupported_capability);
                for (auto index : t.offsets) {
                    require(index.value < op.inputs.size() && std::holds_alternative<LiteralBits>(op.inputs[index.value]), "Sample offset must be a constant", DiagnosticCode::unsupported_modifier);
                    auto value = static_cast<std::int32_t>(std::get<LiteralBits>(op.inputs[index.value]).bits);
                    require(value >= -8 && value <= 7, "Sample offset exceeds Metal range", DiagnosticCode::unsupported_modifier);
                }
                expression += ", " + vector(c, op, t.offsets, ScalarType::s32, count);
                if (gather && !t.compare) expression += ", component::" + std::string(1, "xyzw"[t.gather_component]);
                expression += ")";
            } else if (gather && shape.dimension == TextureDimension::d2 && !t.compare) {
                expression += ", int2(0), component::" + std::string(1, "xyzw"[t.gather_component]) + ")";
            } else {
                if (gather && !t.compare) expression += ", component::" + std::string(1, "xyzw"[t.gather_component]);
                expression += ")";
            }
        }
        if (write) { take(c.statement(expression + ";")); return; }
        const bool scalar_result = (t.compare && !gather) || (!gather && d.format == ComponentFormat::depth);
        // A comparison sample supplies the same scalar for each selected guest
        // component. Comparison gathers retain their separate texel results.
        require(!scalar_result || t.compare || t.component_mask == 1, "Raw scalar depth result cannot select vector components");
        auto temp = "texture_result" + number(id.value);
        std::optional<std::size_t> border;
        if (sampling && c.resources().custom_border) {
            const auto& pairs = c.resources().custom_border->pairs;
            for (std::size_t i = 0; i < pairs.size(); ++i)
                if (pairs[i].texture == *t.resource && t.sampler_resource == pairs[i].sampler)
                    border = i;
        }
        take(c.statement(std::string(border ? "" : "const ") + scalar(d.scalar) +
            (scalar_result ? "" : "4") + " " + temp + " = " + expression + ";"));
        if (border) {
            const auto support = "armigo_border[" + number(*border) + "]";
            // Reuse the complete call suffix, including sampler, gradients,
            // explicit LOD/bias and texel offsets. The coverage image has the
            // same dimensions and mip chain, so the native footprint is equal.
            const auto coverage = support + ".coverage" + expression.substr(tex.size());
            const auto records = c.fragment_output_types_from_render_targets()
                ? std::string("armigo_border_records && ") : std::string();
            take(c.statement("if (" + records + support + ".enabled != 0u) {"));
            take(c.statement(temp + " = armigo_custom_border(" + temp + ", " +
                coverage + ".x, " + support + ");"));
            take(c.statement("}"));
        }
        std::size_t result = 0;
        for (unsigned component = 0; component < 4; ++component) if (t.component_mask & (1u << component))
            take(c.assign(output(op, result++), temp + (scalar_result ? "" : "." + std::string(1, "xyzw"[component])), d.scalar));
    });
}

namespace msl {
ResourcePlan plan_custom_border(const StructuredProgram& program, ResourcePlan plan,
                               const EmitOptions& options) {
    plan.custom_border.reset();
    // Compute stages have no sampling pair ABI. Vertex stages plan the same
    // pairs as fragments, restricted to the sample modes Metal admits there
    // (the per-operation gates already reject implicit-LOD/bias outside
    // fragment stages).
    if (plan.stage != Stage::fragment && plan.stage != Stage::vertex) return plan;
    std::map<std::uint32_t, std::set<std::uint32_t>> texture_samplers;
    std::set<std::uint32_t> unsupported_textures, unsupported_samplers;
    for (const auto& op : program.ir.operations) {
        const auto* t = std::get_if<TextureInfo>(&op.info);
        if (!t || !t->resource) continue;
        const auto& binding = planned(plan, *t->resource);
        const auto* use = std::get_if<TextureUse>(&binding.use);
        const bool sample = t->mode == TextureMode::sample_implicit ||
            t->mode == TextureMode::sample_level || t->mode == TextureMode::sample_bias ||
            t->mode == TextureMode::sample_gradients;
        const bool eligible = use && !use->pool && use->kind == ResourceTextureKind::sampled &&
            use->array_length == 1 && use->descriptor.shape.dimension == TextureDimension::d2 &&
            !use->descriptor.shape.array && !use->descriptor.shape.multisample &&
            use->descriptor.scalar == ScalarType::f32 &&
            use->descriptor.format != ComponentFormat::depth &&
            use->descriptor.normalized_coordinates && !use->descriptor.comparison && sample;
        if (!eligible) unsupported_textures.insert(t->resource->value);
        if (!t->sampler_resource) continue;
        texture_samplers[t->resource->value].insert(t->sampler_resource->value);
        const auto* sampler = std::get_if<SamplerUse>(&planned(plan, *t->sampler_resource).use);
        if (!eligible || !sampler || sampler->pool || sampler->array_length != 1 || sampler->key != use->key)
            unsupported_samplers.insert(t->sampler_resource->value);
    }
    CustomBorderSupport support;
    for (const auto& [texture, samplers] : texture_samplers)
        if (samplers.size() != 1 || unsupported_textures.contains(texture))
            unsupported_samplers.insert(samplers.begin(), samplers.end());
    for (const auto& [texture, samplers] : texture_samplers) {
        if (samplers.size() != 1 || unsupported_textures.contains(texture) ||
            unsupported_samplers.contains(*samplers.begin())) continue;
        support.pairs.push_back({ResourceId{texture}, ResourceId{*samplers.begin()}});
    }
    if (support.pairs.empty()) return plan;
    std::set<std::uint32_t> occupied;
    for (const auto& binding : plan.resources)
        if (const auto* range = std::get_if<MetalRange>(&binding.location);
            range && range->space == MetalNamespace::buffer)
            for (std::uint32_t i = 0; i < range->count; ++i) occupied.insert(range->first + i);
    if (plan.argument_buffer) occupied.insert(plan.argument_buffer->buffer_index);
    if (options.emulate_depth_bias) occupied.insert(30);
    if (plan.draw_support_buffer_index) occupied.insert(*plan.draw_support_buffer_index);
    if (plan.vertex_fetch_buffer_index) occupied.insert(*plan.vertex_fetch_buffer_index);
    if (plan.stage == Stage::vertex) {
        // Native vertex-format inputs claim buffers 18 + ordinal without a
        // planned resource entry; avoid the same window the package tool and
        // the runtime reserve, exactly as ordinary buffer allocation does.
        const auto native_streams = std::count_if(plan.io.begin(), plan.io.end(), [](const auto& io) {
            return io.semantic == IoSemantic::generic && io.access == Access::read &&
                   io.vertex_format && !io.vertex_stream;
        });
        for (std::uint32_t ordinal = 0; ordinal < native_streams; ++ordinal)
            occupied.insert(constant_buffer_slot_count + ordinal);
    }
    while (occupied.contains(support.buffer_index)) ++support.buffer_index;
    if (support.buffer_index >= plan.profile.buffer_slots) return plan;
    plan.custom_border = std::move(support);
    return plan;
}
} // namespace msl
} // namespace armigo::shader::pipeline
