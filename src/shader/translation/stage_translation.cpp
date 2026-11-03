// ============================================================================
// Armigo Shader - Stage Translation
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================
// One stage through the ported pipeline: the offline translator and the
// run-time preparation share this code, so both produce the same source,
// reflection and program for the same stage and options. Storage-buffer
// reflection records descriptor origins and access metadata. The
// derived-descriptor pre-scan applies the per-encoding texture shape and
// depth-compare checks implemented by the texture emitter.

#include <armigo/shader_translation/stage_translation.h>

#include <armigo/shader/source_version.h>
#include <armigo/shader/vertex_fetch.h>
#include "pipeline/facts.h"
#include "pipeline/phases.h"
#include "sha256.h"

#include <algorithm>
#include <array>
#include <map>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace armigo::shader::translation {
namespace {
namespace si = armigo::shader_input;
namespace gx = armigo::shader::graphics;
namespace gm = armigo::shader::graphics_msl;
namespace pl = armigo::shader::pipeline;
using Bytes = std::span<const std::byte>;

si::ByteSpan view(Bytes bytes) {
    return {reinterpret_cast<const std::uint8_t*>(bytes.data()), bytes.size()};
}

std::string quote(std::string_view text) {
    constexpr char hex[] = "0123456789abcdef";
    std::string result = "\"";
    for (const unsigned char c : text) {
        if (c == '"' || c == '\\') {
            result += '\\';
            result += static_cast<char>(c);
        } else if (c < 0x20 || c >= 0x7f) {
            result += "\\u00";
            result += hex[c >> 4];
            result += hex[c & 15];
        } else {
            result += static_cast<char>(c);
        }
    }
    return result + '"';
}

Bytes text_bytes(std::string_view text) {
    return {reinterpret_cast<const std::byte*>(text.data()), text.size()};
}

std::string digest(Bytes bytes) {
    const auto hash = armigo::loader::detail::sha256(bytes);
    if (!hash) throw std::runtime_error("System SHA-256 failed");
    constexpr char hex[] = "0123456789abcdef";
    std::string result;
    for (const auto byte : *hash) {
        const auto value = std::to_integer<unsigned>(byte);
        result += hex[value >> 4];
        result += hex[value & 15];
    }
    return result;
}

const char* depth_name(gm::VertexDepthConvention depth) {
    switch (depth) {
    case gm::VertexDepthConvention::NegativeOneToOne: return "minus-one-to-one";
    case gm::VertexDepthConvention::ZeroToOne: return "zero-to-one";
    case gm::VertexDepthConvention::Unspecified: return "unspecified";
    }
    throw std::runtime_error("Invalid emitted depth convention");
}

// --- Ported pipeline path (--translator ported) -----------------------------

// The host policy the ported phases legalize against. Its identity is recorded
// in v3 reflection; older v2 packages retain their original host policy.
// Metal 4 has native binary16 conversion and packed half vectors, so fp16
// boundaries are admitted; FP64 stays unavailable.
pl::MetalProfile ported_profile() {
    // Version 3 adds the armigo_y_negate function constant for stages that
    // read S2R SR_YDIRECTION; older packages cannot be specialized for it.
    // Apple-silicon SIMD groups are 32 lanes wide in every stage, so warp
    // shuffles (SHFL) lower to native simd_shuffle; enabling it only admits
    // stages that were previously rejected.
    return {"armigo-macos-metal4", 3, 31, 128, 16,
            true, false, true, false, false, true};
}

// --- Derived texture descriptor facts (--texture-descriptors derived) --------

// What one texture instruction's encoding proves about the descriptor of the
// bound handle it uses. The sampled shape restates the texture emitter's
// per-encoding shape checks; depth-compare evidence exists only where the
// emitter actually checks the descriptor's comparison flag against the
// encoding (fetches never do). A use that leaves either open proves nothing.
struct TextureUseEvidence {
    pl::TextureShape shape{};
    bool shape_proven{};
    bool comparison{};
    bool comparison_proven{};
    // A surface atomic proves an integer component: no float format can back
    // an encoded integer atomic operation, so the component cannot stay the
    // assumed float. The runtime verifier still compares the recorded scalar
    // against the live TIC and stops the draw on any mismatch.
    bool integer{};
};

// Per bound handle: the shape and comparison its uses jointly prove. Uses
// must agree on every proven property; disagreement stays Unknown. A handle
// whose encodings leave shape or comparison open receives the documented
// assumed 2D descriptor instead, which the runtime verifier re-checks
// against the live texture and sampler before each draw.
using DerivedTextureFacts = std::map<pl::TextureKey, pl::Knowledge<pl::TextureDescriptor>>;

[[nodiscard]] constexpr pl::TextureDimension long_dimension(unsigned selector) {
    // TexDim folds dimension into bits 1+ and the array flag into bit 0.
    return selector / 2 == 0 ? pl::TextureDimension::d1 :
           selector / 2 == 1 ? pl::TextureDimension::d2 :
           selector / 2 == 2 ? pl::TextureDimension::d3 : pl::TextureDimension::cube;
}

[[nodiscard]] std::optional<std::uint64_t> scan_field(
    const pl::DecodedInstruction& inst, pl::DecodedField field) {
    for (const auto& item : inst.fields) if (item.field == field) return item.bits;
    return std::nullopt;
}

[[nodiscard]] std::optional<bool> scan_flag(const pl::DecodedInstruction& inst,
                                            pl::DecodedField field) {
    const auto bits = scan_field(inst, field);
    if (!bits || *bits > 1) return std::nullopt;
    return *bits != 0;
}

// The descriptor a handle's instructions cannot fully prove but the draw
// still needs. GPU state supplies it at translation time; offline translation
// instead records a plain float-2D
// assumption with its own evidence label, and the runtime verifier compares
// it against the live texture and sampler before every draw.
[[nodiscard]] pl::Known<pl::TextureDescriptor> assumed_descriptor(
    pl::TextureShape shape, std::string_view identity, bool integer = false) {
    pl::TextureDescriptor descriptor;
    descriptor.shape = shape;
    descriptor.scalar = integer ? pl::ScalarType::u32 : pl::ScalarType::f32;
    descriptor.format = integer ? pl::ComponentFormat::unsigned_integer : pl::ComponentFormat::floating;
    descriptor.normalized_coordinates = true;
    descriptor.comparison = false;
    descriptor.format_code = 0;
    descriptor.format_namespace = "armigo-derived-v1";
    return pl::Known<pl::TextureDescriptor>{std::move(descriptor),
                                            {pl::OriginKind::derived, std::string(identity)}};
}

[[nodiscard]] std::optional<TextureUseEvidence> texture_use_evidence(
    const pl::DecodedInstruction& inst) {
    using pl::DecodedField;
    using pl::InstructionName;
    using pl::TextureDimension;
    const auto dimension = scan_field(inst, DecodedField::texture_dimension);
    const bool long_form = inst.name == InstructionName::tex ||
                           inst.name == InstructionName::tld ||
                           inst.name == InstructionName::tld4 ||
                           inst.name == InstructionName::txd;
    if (long_form) {
        if (!dimension || *dimension > 7) return std::nullopt;
        const pl::TextureShape shape{long_dimension(static_cast<unsigned>(*dimension)),
                                     (*dimension & 1) != 0, false};
        if (inst.name == InstructionName::tld) {
            // A fetch checks no comparison flag. Its level-zero, un-offset,
            // non-multisample 1D form is also the only one whose descriptor
            // may be a buffer, so the shape stays open there.
            const auto lod = scan_field(inst, DecodedField::texture_mode);
            const auto multisample = scan_flag(inst, DecodedField::texture_multisample);
            const auto offset = scan_flag(inst, DecodedField::texture_offset);
            if (!lod || *lod > 1 || !multisample || !offset) return std::nullopt;
            if (*lod == 0 && *dimension == 0 && !*multisample && !*offset)
                return std::nullopt;
            return TextureUseEvidence{pl::TextureShape{shape.dimension, shape.array,
                                                       *multisample}, true, false, false};
        }
        if (inst.name == InstructionName::txd)
            return TextureUseEvidence{shape, true, false, true};
        const auto shadow = scan_flag(inst, DecodedField::texture_compare);
        if (!shadow) return std::nullopt;
        return TextureUseEvidence{shape, true, *shadow, true};
    }
    if (inst.name == InstructionName::texs || inst.name == InstructionName::texs_f16) {
        const auto target = scan_field(inst, DecodedField::texture_target);
        if (!target || *target > 13) return std::nullopt;
        const unsigned encoded = static_cast<unsigned>(*target);
        // TEXS target table: the dimension fold, the 2D-array band and the
        // depth-compare targets mirror the emitter's selection.
        const pl::TextureShape shape{
            encoded == 0 ? TextureDimension::d1 :
            encoded < 10 ? TextureDimension::d2 :
            encoded < 12 ? TextureDimension::d3 : TextureDimension::cube,
            encoded >= 7 && encoded <= 9, false};
        const bool shadow = encoded == 4 || encoded == 5 ||
                            encoded == 6 || encoded == 9;
        return TextureUseEvidence{shape, true, shadow, true};
    }
    if (inst.name == InstructionName::tlds || inst.name == InstructionName::tlds_f16) {
        const auto target = scan_field(inst, DecodedField::texture_target);
        if (!target || *target > 15) return std::nullopt;
        // Fetches prove no comparison. Target 0 is the level-zero 1D fetch
        // whose descriptor may also be a buffer, so it proves no shape.
        switch (*target) {
        case 1: return TextureUseEvidence{pl::TextureShape{TextureDimension::d1, false, false}, true, false, false};
        case 2: case 4: case 5: case 12:
            return TextureUseEvidence{pl::TextureShape{TextureDimension::d2, false, false}, true, false, false};
        case 6: return TextureUseEvidence{pl::TextureShape{TextureDimension::d2, false, true}, true, false, false};
        case 7: return TextureUseEvidence{pl::TextureShape{TextureDimension::d3, false, false}, true, false, false};
        case 8: return TextureUseEvidence{pl::TextureShape{TextureDimension::d2, true, false}, true, false, false};
        default: return std::nullopt;
        }
    }
    if (inst.name == InstructionName::tld4s || inst.name == InstructionName::tld4s_f16) {
        const auto shadow = scan_flag(inst, DecodedField::texture_compare);
        if (!shadow) return std::nullopt;
        return TextureUseEvidence{pl::TextureShape{TextureDimension::d2, false, false},
                                  true, *shadow, true};
    }
    if (inst.name == InstructionName::suatom || inst.name == InstructionName::suatom_b ||
        inst.name == InstructionName::suatom_b2 || inst.name == InstructionName::suatom_cas ||
        inst.name == InstructionName::suatom_cas_b)
        return TextureUseEvidence{{}, false, false, false, true};
    // TXQ, TMML and TXA either select no sampled shape or take it from the
    // descriptor itself; they register the handle but prove nothing.
    return std::nullopt;
}

[[nodiscard]] std::optional<pl::TextureKey> scanned_handle_key(
    const pl::DecodedInstruction& inst, std::optional<std::uint8_t> handle_cbuf) {
    if (inst.encoding != pl::Encoding::texture_bound) return std::nullopt;
    const auto byte_offset = scan_field(inst, pl::DecodedField::texture_handle_byte_offset);
    if (!byte_offset || *byte_offset % 4) return std::nullopt;
    // Decoders never emit an explicit texture_cbuf_slot; the handle buffer is
    // proven through the option, matching the emitter's slot selection.
    if (!handle_cbuf) return std::nullopt;
    return pl::TextureKey{*handle_cbuf, static_cast<std::uint32_t>(*byte_offset)};
}

// Pre-scan the program with the public decode entry point and derive, for
// every bound texture handle, the descriptor its texture instructions jointly
// prove. A decode failure yields no facts: the pipeline re-decodes with the
// real fact source and reports the same diagnostic itself.
[[nodiscard]] DerivedTextureFacts scan_derived_texture_facts(
    const pl::StageImage& image, std::optional<std::uint8_t> handle_cbuf) {
    struct SilentFacts final : public pl::FactSource {
        [[nodiscard]] pl::Knowledge<pl::FactValue> lookup(pl::FactKey) const override {
            return pl::Unknown{"The derived-descriptor pre-scan answers no facts"};
        }
    };
    SilentFacts silent;
    pl::FactReader reader{silent, pl::Limits{}};
    auto decoded = pl::decode(image, reader, pl::Limits{});
    if (const auto* error = std::get_if<pl::Diagnostic>(&decoded)) {
        static_cast<void>(error);
        return {};
    }
    const auto& program = std::get<pl::DecodedProgram>(decoded);
    struct Observation {
        std::vector<pl::TextureShape> shapes;
        std::vector<bool> comparisons;
        std::vector<bool> integers;
    };
    std::map<pl::TextureKey, Observation> observed;
    for (const auto& block : program.blocks)
        for (const auto& inst : block.instructions) {
            const auto key = scanned_handle_key(inst, handle_cbuf);
            if (!key) continue;
            auto& observation = observed[*key];
            if (auto evidence = texture_use_evidence(inst)) {
                if (evidence->shape_proven) observation.shapes.push_back(evidence->shape);
                if (evidence->comparison_proven) observation.comparisons.push_back(evidence->comparison);
                observation.integers.push_back(evidence->integer);
            }
        }
    DerivedTextureFacts facts;
    for (const auto& [key, observation] : observed) {
        std::optional<pl::TextureShape> shape;
        std::optional<bool> comparison;
        bool integer = false;
        bool disagreed = false;
        for (const auto& candidate : observation.shapes) {
            if (shape && *shape != candidate) disagreed = true;
            shape = candidate;
        }
        for (const bool candidate : observation.comparisons) {
            if (comparison && *comparison != candidate) disagreed = true;
            comparison = candidate;
        }
        for (const bool candidate : observation.integers) {
            if (candidate && integer) disagreed = true;
            integer = integer || candidate;
        }
        if (disagreed) {
            facts.emplace(key, pl::Unknown{
                "Texture instructions disagree on the derived descriptor"});
        } else if (!shape) {
            // Query-only handles (TXQ/TMML/TXA and surface accesses) prove no
            // sampled shape: assume the plain 2D form the runtime verifier
            // admits and let a live mismatch stop the draw. Size queries a
            // 2D shape cannot answer still fail their own diagnostics.
            facts.emplace(key, assumed_descriptor({pl::TextureDimension::d2, false, false},
                                                  "code:assumed-query-only-2d", integer));
        } else if (!comparison) {
            // Shape evidence exists only where no instruction constrains the
            // sampler compare mode (fetches ignore it). Keep the proven shape
            // and assume no comparison. Fetch-only reflection has no sampler
            // resource, so runtime verification checks the live TIC and does
            // not read or constrain the unused TSC state.
            facts.emplace(key, assumed_descriptor(*shape,
                                                  "code:assumed-fetch-no-compare", integer));
        } else {
            if (integer) disagreed = true;
            pl::TextureDescriptor descriptor;
            descriptor.shape = *shape;
            descriptor.scalar = pl::ScalarType::f32;
            descriptor.format = *comparison ? pl::ComponentFormat::depth : pl::ComponentFormat::floating;
            descriptor.normalized_coordinates = true;
            descriptor.comparison = *comparison;
            descriptor.format_code = 0;
            descriptor.format_namespace = "armigo-derived-v1";
            facts.emplace(key, pl::Known<pl::TextureDescriptor>{
                std::move(descriptor),
                {pl::OriginKind::derived, "code:texture-instruction-shape"}});
        }
    }
    return facts;
}

// The CLI's explicit options and the validated SPH are the only evidence this
// offline translation has. Every other fact stays Unknown, which the phases
// turn into missing-fact diagnostics; nothing defaults. With
// --texture-descriptors derived, the texture_descriptor facts the program's
// own texture instructions prove are answered from the pre-scan table.
class OptionFacts final : public pl::FactSource {
public:
    OptionFacts(const gm::EmitOptions& options, std::span<const std::byte> sph,
                std::optional<std::uint8_t> texture_handle_cbuf,
                const DerivedTextureFacts* derived_texture_facts,
                std::optional<pl::ComputeLocalSize> compute_local_size,
                std::optional<std::uint32_t> compute_shared_memory_bytes)
        : options_(options), sph_(sph), texture_handle_cbuf_(texture_handle_cbuf),
          derived_texture_facts_(derived_texture_facts), compute_local_size_(compute_local_size),
          compute_shared_memory_bytes_(compute_shared_memory_bytes) {}

    [[nodiscard]] pl::Knowledge<pl::FactValue> lookup(pl::FactKey key) const override {
        if (!pl::valid_fact_key(key)) return pl::Unknown{"Fact key is malformed"};
        switch (key.kind) {
        case pl::FactKind::compute_local_size:
            if (!compute_local_size_) return pl::Unknown{"No --compute-local-size launch evidence was supplied"};
            return pl::Known<pl::FactValue>{*compute_local_size_, {pl::OriginKind::explicit_option, "option:compute-local-size"}};
        case pl::FactKind::compute_shared_memory_bytes:
            if (!compute_shared_memory_bytes_) return pl::Unknown{"No --compute-shared-memory allocation evidence was supplied"};
            return pl::Known<pl::FactValue>{*compute_shared_memory_bytes_, {pl::OriginKind::explicit_option, "option:compute-shared-memory"}};
        case pl::FactKind::local_memory_bytes:
            // Graphics SPHs store local-memory components in common words 1--3.
            // CRS storage is warp-wide in the guest header, so divide it by
            // the fixed 32-thread warp width before adding it per thread.
            return pl::Known<pl::FactValue>{local_memory_bytes(),
                                            {pl::OriginKind::validated_header,
                                             "sph:local-memory"}};
        case pl::FactKind::texture_handle_cbuf:
            if (!texture_handle_cbuf_)
                return pl::Unknown{"No --texture-handle-cbuf option was supplied"};
            return pl::Known<pl::FactValue>{
                std::uint32_t{*texture_handle_cbuf_},
                {pl::OriginKind::explicit_option, "option:texture-handle-cbuf"}};
        case pl::FactKind::texture_descriptor:
            if (!derived_texture_facts_)
                return pl::Unknown{"No --texture-descriptors source was supplied"};
            return lookup_derived_descriptor(pl::TextureKey{
                static_cast<std::uint8_t>(key.index), key.byte_offset});
        case pl::FactKind::depth_convention:
            if (options_.vertex_depth == gm::VertexDepthConvention::Unspecified)
                return pl::Unknown{"No --vertex-depth convention was supplied"};
            return pl::Known<pl::FactValue>{
                options_.vertex_depth == gm::VertexDepthConvention::ZeroToOne
                    ? pl::DepthConvention::zero_to_one
                    : pl::DepthConvention::negative_one_to_one,
                {pl::OriginKind::explicit_option, "option:vertex-depth"}};
        case pl::FactKind::vertex_attribute_format:
            for (const auto& item : options_.vertex_attributes) {
                if (item.generic_slot == key.index)
                    return pl::Known<pl::FactValue>{
                        pl::AttributeFormat{pl::VertexFormat::f32, 3},
                        {pl::OriginKind::explicit_option, "option:vertex-float3"}};
            }
            return pl::Unknown{"No --vertex-float3 option names this slot"};
        case pl::FactKind::input_interpolation: {
            // The SPH input map holds one byte per generic attribute, with two
            // bits per component (x in the low bits). An unused component that
            // is still read takes the attribute's first used mode, which is how
            // one vector interface declaration covers the whole location. Zero
            // for every component means the header declares the attribute
            // unused.
            if (key.index >= 32 || key.component >= 4)
                return pl::Unknown{"Interpolation map covers generic slots only"};
            const unsigned byte_value = std::to_integer<unsigned>(sph_[0x18 + key.index]);
            unsigned mode = (byte_value >> (key.component * 2)) & 3;
            for (unsigned component = 0; mode == 0 && component < 4; ++component)
                mode = (byte_value >> (component * 2)) & 3;
            switch (mode) {
            case 1: return pl::Known<pl::FactValue>{
                pl::Interpolation{pl::InterpolationMode::flat, pl::InterpolationLocation::center},
                {pl::OriginKind::validated_header, "sph:interpolation"}};
            case 2: return pl::Known<pl::FactValue>{
                pl::Interpolation{pl::InterpolationMode::perspective,
                                  pl::InterpolationLocation::center},
                {pl::OriginKind::validated_header, "sph:interpolation"}};
            case 3: return pl::Known<pl::FactValue>{
                pl::Interpolation{pl::InterpolationMode::linear, pl::InterpolationLocation::center},
                {pl::OriginKind::validated_header, "sph:interpolation"}};
            default: return pl::Unknown{"SPH declares no interpolation mode for this slot"};
            }
        }
        case pl::FactKind::fragment_output_type:
            // Ported fragments retain the four raw guest output registers.
            // The render pipeline specializes their final Metal scalar class.
            if (key.index < 8)
                return pl::Known<pl::FactValue>{pl::IoType{pl::ScalarType::u32, 4},
                                                {pl::OriginKind::derived,
                                                 "raw-fragment-output-registers"}};
            return pl::Unknown{"No render target exists at this colour location"};
        default: return pl::Unknown{"No CLI evidence is available for this fact"};
        }
    }

private:
    [[nodiscard]] std::uint32_t local_memory_bytes() const {
        const auto common_word = [&](std::size_t offset) {
            std::uint32_t value = 0;
            for (unsigned byte = 0; byte != 4; ++byte) {
                value |= std::uint32_t{std::to_integer<unsigned char>(sph_[offset + byte])}
                    << (byte * 8);
            }
            return value;
        };
        constexpr std::uint32_t field_mask = (std::uint32_t{1} << 24) - 1;
        const std::uint32_t low = common_word(4) & field_mask;
        const std::uint32_t high = common_word(8) & field_mask;
        const std::uint32_t crs = common_word(12) & field_mask;
        return low + high + crs / 32;
    }

    [[nodiscard]] pl::Knowledge<pl::FactValue> lookup_derived_descriptor(
        pl::TextureKey key) const {
        const auto found = derived_texture_facts_->find(key);
        if (found == derived_texture_facts_->end())
            return pl::Unknown{
                "No texture instruction uses this handle, so the code proves "
                "no descriptor for it"};
        if (const auto* unknown = std::get_if<pl::Unknown>(&found->second)) return *unknown;
        const auto& known = std::get<pl::Known<pl::TextureDescriptor>>(found->second);
        return pl::Known<pl::FactValue>{known.value, known.origin};
    }

    const gm::EmitOptions& options_;
    std::span<const std::byte> sph_;
    std::optional<std::uint8_t> texture_handle_cbuf_;
    const DerivedTextureFacts* derived_texture_facts_;
    std::optional<pl::ComputeLocalSize> compute_local_size_;
    std::optional<std::uint32_t> compute_shared_memory_bytes_;
};

std::string_view phase_name(pl::Phase phase) {
    switch (phase) {
    case pl::Phase::decode: return "decode";
    case pl::Phase::emit_ir: return "emit_ir";
    case pl::Phase::construct_ssa: return "construct_ssa";
    case pl::Phase::legalize: return "legalize";
    case pl::Phase::structure: return "structure";
    case pl::Phase::plan_resources: return "plan_resources";
    case pl::Phase::emit_msl: return "emit_msl";
    }
    return "unknown";
}

[[noreturn]] void pipeline_failure(const pl::Diagnostic& error) {
    std::string text = std::string(phase_name(error.phase)) +
        " code=" + std::to_string(unsigned(error.code)) +
        " pc=" + std::to_string(error.pc.value_or(0));
    if (error.instruction)
        text += " instruction=" + std::to_string(unsigned(*error.instruction));
    if (!error.modifier.empty()) text += " modifier=" + error.modifier;
    if (error.missing_fact)
        text += " fact=" + std::to_string(unsigned(error.missing_fact->kind)) + ':' +
                std::to_string(error.missing_fact->index) + ':' +
                std::to_string(error.missing_fact->byte_offset) + ':' +
                std::to_string(error.missing_fact->component);
    text += ": " + error.message;
    throw std::runtime_error(text);
}

const char* scalar_name(pl::ScalarType type) {
    switch (type) {
    case pl::ScalarType::u8: return "u8";
    case pl::ScalarType::s8: return "s8";
    case pl::ScalarType::u16: return "u16";
    case pl::ScalarType::s16: return "s16";
    case pl::ScalarType::u32: return "u32";
    case pl::ScalarType::s32: return "s32";
    case pl::ScalarType::u64: return "u64";
    case pl::ScalarType::s64: return "s64";
    case pl::ScalarType::f16: return "f16";
    case pl::ScalarType::f32: return "f32";
    case pl::ScalarType::f64: return "f64";
    case pl::ScalarType::boolean: return "boolean";
    }
    return "unknown";
}

std::string fact_value_json(const pl::Dependency& dependency) {
    const auto& value = dependency.value;
    std::ostringstream out;
    if (const auto* number = std::get_if<std::uint32_t>(&value)) {
        out << *number;
    } else if (const auto* size = std::get_if<pl::ComputeLocalSize>(&value)) {
        out << '[' << size->dimensions[0] << ',' << size->dimensions[1] << ',' << size->dimensions[2] << ']';
    } else if (const auto* flag = std::get_if<bool>(&value)) {
        out << (*flag ? "true" : "false");
    } else if (const auto* format = std::get_if<pl::AttributeFormat>(&value)) {
        static constexpr const char* names[] = {"u8", "s8", "unorm8", "snorm8", "u16",
            "s16", "unorm16", "snorm16", "f16", "u32", "s32", "f32"};
        out << "{\"format\":" << quote(names[unsigned(format->format)])
            << ",\"components\":" << unsigned(format->components) << '}';
    } else if (const auto* type = std::get_if<pl::IoType>(&value)) {
        out << "{\"scalar\":" << quote(scalar_name(type->scalar))
            << ",\"components\":" << unsigned(type->components) << '}';
    } else if (const auto* interpolation = std::get_if<pl::Interpolation>(&value)) {
        static constexpr const char* modes[] = {"flat", "perspective", "linear"};
        static constexpr const char* locations[] = {"center", "centroid", "sample", "offset"};
        out << "{\"mode\":" << quote(modes[unsigned(interpolation->mode)])
            << ",\"location\":" << quote(locations[unsigned(interpolation->location)]) << '}';
    } else if (const auto* descriptor = std::get_if<pl::TextureDescriptor>(&value)) {
        static constexpr const char* dimensions[] = {"d1", "d2", "d3", "cube", "buffer"};
        static constexpr const char* formats[] = {"unorm", "snorm", "unsigned_integer",
            "signed_integer", "floating", "depth"};
        out << "{\"dimension\":" << quote(dimensions[unsigned(descriptor->shape.dimension)])
            << ",\"array\":" << (descriptor->shape.array ? "true" : "false")
            << ",\"multisample\":" << (descriptor->shape.multisample ? "true" : "false")
            << ",\"scalar\":" << quote(scalar_name(descriptor->scalar))
            << ",\"component_format\":" << quote(formats[unsigned(descriptor->format)])
            << ",\"normalized_coordinates\":" << (descriptor->normalized_coordinates ? "true" : "false")
            << ",\"comparison\":" << (descriptor->comparison ? "true" : "false") << '}';
    } else if (const auto* topology = std::get_if<pl::Topology>(&value)) {
        static constexpr const char* names[] = {"points", "lines", "triangles"};
        out << quote(names[unsigned(*topology)]);
    } else if (const auto* depth = std::get_if<pl::DepthConvention>(&value)) {
        out << quote(*depth == pl::DepthConvention::zero_to_one ? "zero-to-one"
                                                                : "minus-one-to-one");
    } else if (const auto* origin = std::get_if<pl::OriginConvention>(&value)) {
        out << quote(*origin == pl::OriginConvention::upper_left ? "upper-left" : "lower-left");
    } else if (const auto* alpha = std::get_if<pl::AlphaTest>(&value)) {
        static constexpr const char* compares[] = {"never", "less", "equal", "less_equal",
            "greater", "not_equal", "greater_equal", "always"};
        out << "{\"comparison\":" << quote(compares[unsigned(alpha->comparison)])
            << ",\"reference_bits\":" << alpha->reference_bits << '}';
    } else {
        throw std::runtime_error("Ported reflection cannot serialize this fact value");
    }
    return out.str();
}

const char* fact_kind_name(pl::FactKind kind) {
    switch (kind) {
    case pl::FactKind::local_memory_bytes: return "local_memory_bytes";
    case pl::FactKind::compute_local_size: return "compute_local_size";
    case pl::FactKind::compute_shared_memory_bytes: return "compute_shared_memory_bytes";
    case pl::FactKind::constant_buffer_bound_mask: return "constant_buffer_bound_mask";
    case pl::FactKind::constant_buffer_size: return "constant_buffer_size";
    case pl::FactKind::constant_buffer_word: return "constant_buffer_word";
    case pl::FactKind::texture_handle_cbuf: return "texture_handle_cbuf";
    case pl::FactKind::texture_descriptor: return "texture_descriptor";
    case pl::FactKind::texture_array_length: return "texture_array_length";
    case pl::FactKind::vertex_attribute_format: return "vertex_attribute_format";
    case pl::FactKind::fragment_output_type: return "fragment_output_type";
    case pl::FactKind::input_interpolation: return "input_interpolation";
    case pl::FactKind::topology: return "topology";
    case pl::FactKind::depth_convention: return "depth_convention";
    case pl::FactKind::origin_convention: return "origin_convention";
    case pl::FactKind::negate_y: return "negate_y";
    case pl::FactKind::alpha_test: return "alpha_test";
    case pl::FactKind::early_z: return "early_z";
    case pl::FactKind::alpha_to_coverage: return "alpha_to_coverage";
    case pl::FactKind::dual_source_blend: return "dual_source_blend";
    case pl::FactKind::draw_parameters: return "draw_parameters";
    case pl::FactKind::count: break;
    }
    throw std::runtime_error("Ported reflection cannot name this fact kind");
}

void write_resource_location(std::ostream& out, const pl::BindingLocation& location) {
    if (const auto* direct = std::get_if<pl::MetalRange>(&location)) {
        static constexpr const char* namespaces[] = {"buffer", "texture", "sampler"};
        out << "{\"kind\":\"direct\",\"namespace\":" << quote(namespaces[unsigned(direct->space)])
            << ",\"first\":" << direct->first << ",\"count\":" << direct->count << '}';
    } else {
        const auto& member = std::get<pl::ArgumentBufferMember>(location);
        out << "{\"kind\":\"argument_buffer_member\",\"byte_offset\":" << member.byte_offset
            << ",\"count\":" << member.count << ",\"stride\":" << member.stride << '}';
    }
}

std::string ported_reflection(const pl::MslArtifact& artifact, const gm::EmitOptions& options,
                              const pl::Stage stage, Bytes identity) {
    const auto& plan = artifact.reflection;
    std::ostringstream out;
    out << "{\"schema\":\"armigo-shader-stage-v3\",\"source_version\":"
        << quote(armigo::shader::source_version)
        << ",\"program_key\":" << quote(digest(identity))
        << ",\"program_size\":" << identity.size()
        << ",\"program_file\":\"program.bin\",\"source_file\":\"source.metal\""
        << ",\"source_digest\":" << quote(digest(text_bytes(artifact.source)))
        << ",\"stage\":" << quote(stage == pl::Stage::vertex ? "vertex" : stage == pl::Stage::compute ? "compute" : "fragment")
        << ",\"entry_point\":" << quote(artifact.entry_point)
        << ",\"vertex_depth\":" << quote(depth_name(options.vertex_depth))
        << ",\"fragment_position\":"
        << quote(options.fragment_position == gm::FragmentPositionConvention::Unspecified
                     ? "unspecified" : "reciprocal-clip-w")
        << (stage == pl::Stage::fragment
                ? ",\"fragment_output_mode\":\"render-target-v1\""
                : "")
        // The ported pipeline reads only the SPH and program words; the trailing
        // C1 bytes are bound at run time, so they never select a variant.
        << ",\"c1_digest\":null";
    const bool consumes_y_direction = std::any_of(plan.io.begin(), plan.io.end(),
        [](const pl::IoUse& io) {
            return io.access == pl::Access::read && io.semantic == pl::IoSemantic::y_direction;
        });
    if (consumes_y_direction) out << ",\"y_direction\":\"function-constant-v1\"";
    // A vertex stage without its own point size writes the fixed size 1.0.
    if (plan.fixed_point_size) out << ",\"point_size\":\"fixed-one-v1\"";
    else if (std::any_of(plan.io.begin(), plan.io.end(), [](const pl::IoUse& io) {
                 return io.access == pl::Access::write && io.semantic == pl::IoSemantic::point_size;
             }))
        out << ",\"point_size\":\"program-v1\"";
    if (plan.reads_point_coord) out << ",\"point_coord\":\"attribute-v1\"";
    if (plan.emulate_depth_bias) out << ",\"depth_bias_mode\":\"shader-v1\"";
    if (plan.custom_border) {
        out << ",\"custom_border\":{\"abi\":\"coverage-v1\",\"buffer_index\":"
            << plan.custom_border->buffer_index << ",\"pairs\":[";
        bool first_pair = true;
        for (const auto& pair : plan.custom_border->pairs) {
            if (!first_pair) out << ',';
            first_pair = false;
            out << "{\"texture\":" << pair.texture.value
                << ",\"sampler\":" << pair.sampler.value << '}';
        }
        out << "]}";
    }
    if (plan.draw_support_buffer_index) {
        out << ",\"draw_support\":{\"schema\":" << quote(pl::draw_support_schema)
            << ",\"version\":" << pl::draw_support_version
            << ",\"byte_extent\":" << pl::draw_support_bytes
            << ",\"alignment\":4,\"metal_buffer_index\":"
            << *plan.draw_support_buffer_index << '}';
    }
    if (plan.vertex_fetch_buffer_index) {
        out << ",\"vertex_fetch\":{\"schema\":" << quote(vertex_fetch_schema)
            << ",\"version\":" << vertex_fetch_version
            << ",\"byte_extent\":" << vertex_fetch_bytes
            << ",\"alignment\":8,\"metal_buffer_index\":"
            << *plan.vertex_fetch_buffer_index << '}';
    }
    out << ",\"profile\":{\"identity\":" << quote(plan.profile.identity)
        << ",\"version\":" << plan.profile.version << '}';
    out << ",\"argument_buffer\":";
    if (plan.argument_buffer) {
        const auto& layout = *plan.argument_buffer;
        out << "{\"abi_version\":" << layout.abi_version
            << ",\"buffer_index\":" << layout.buffer_index
            << ",\"byte_extent\":" << layout.byte_extent
            << ",\"alignment\":" << layout.alignment << '}';
    } else out << "null";
    if (plan.compute) {
        const auto& size = plan.compute->local_size.dimensions;
        out << ",\"compute\":{\"local_size\":[" << size[0] << ',' << size[1] << ',' << size[2]
            << "],\"shared_memory_bytes\":" << plan.compute->shared_memory_bytes << '}';
    }
    out << ",\"constant_buffers\":[";
    bool first = true;
    for (const auto& planned : plan.resources) {
        const auto* buffer = std::get_if<pl::BufferUse>(&planned.use);
        if (!buffer || !std::holds_alternative<pl::ConstantBufferSource>(buffer->source)) continue;
        if (buffer->size.kind != pl::SizeKind::constant_buffer_window ||
            buffer->size.constant_buffer_policy != pl::ConstantBufferPolicy::runtime_zero_words_v1)
            throw std::runtime_error("Ported reflection requires a runtime constant-buffer window");
        if (!first) out << ',';
        first = false;
        out << "{\"guest_binding\":"
            << unsigned(std::get<pl::ConstantBufferSource>(buffer->source).slot)
            << ",\"size_kind\":\"constant_buffer_window\""
            << ",\"window_bytes\":" << buffer->size.bytes
            << ",\"alignment\":" << buffer->size.alignment
            << ",\"policy\":\"runtime_zero_words_v1\""
            << ",\"metal_buffer_index\":" << std::get<pl::MetalRange>(planned.location).first << '}';
    }
    out << "],\"support_buffers\":[";
    first = true;
    for (const auto& planned : plan.resources) {
        const auto* buffer = std::get_if<pl::BufferUse>(&planned.use);
        if (!buffer) continue;
        const auto* support = std::get_if<pl::SupportBufferSource>(&buffer->source);
        if (!support) continue;
        if (support->schema != pl::constant_buffer_support_schema ||
            support->version != pl::constant_buffer_support_version)
            throw std::runtime_error("Ported reflection has an unsupported support-buffer schema");
        if (!first) out << ',';
        first = false;
        out << "{\"schema\":" << quote(support->schema)
            << ",\"version\":" << support->version
            << ",\"byte_extent\":" << buffer->size.bytes
            << ",\"alignment\":" << buffer->size.alignment
            << ",\"metal_buffer_index\":" << std::get<pl::MetalRange>(planned.location).first << '}';
    }
    out << "],\"storage_buffers\":[";
    first = true;
    for (const auto& planned : plan.resources) {
        const auto* buffer = std::get_if<pl::BufferUse>(&planned.use);
        if (!buffer) continue;
        const auto* storage = std::get_if<pl::StorageBufferSource>(&buffer->source);
        if (!storage) continue;
        if (buffer->size.kind != pl::SizeKind::storage_descriptor ||
            buffer->size.bytes != 4 || buffer->size.alignment != 4)
            throw std::runtime_error("Ported reflection has an unsupported storage-buffer size contract");
        const auto* location = std::get_if<pl::MetalRange>(&planned.location);
        if (!location || location->space != pl::MetalNamespace::buffer || location->count != 1)
            throw std::runtime_error("Ported reflection requires a direct storage-buffer binding");
        if (!first) out << ',';
        first = false;
        out << "{\"cbuf_slot\":" << unsigned(storage->descriptor.cbuf_slot)
            << ",\"descriptor_byte_offset\":" << storage->descriptor.handle_byte_offset
            << ",\"metal_buffer_index\":" << location->first
            << ",\"access\":" << quote(buffer->access == pl::Access::read ? "read" :
                buffer->access == pl::Access::write ? "write" : "read_write")
            << ",\"alignment\":" << buffer->size.alignment << '}';
    }
    out << "],\"textures\":[";
    first = true;
    for (const auto& planned : plan.resources) {
        const auto* texture = std::get_if<pl::TextureUse>(&planned.use);
        const auto* sampler = std::get_if<pl::SamplerUse>(&planned.use);
        if (!texture && !sampler) continue;
        if (!first) out << ',';
        first = false;
        if (texture) {
            static constexpr const char* dimensions[] = {"d1", "d2", "d3", "cube", "buffer"};
            static constexpr const char* formats[] = {"unorm", "snorm", "unsigned_integer",
                "signed_integer", "floating", "depth"};
            out << "{\"cbuf_slot\":" << unsigned(texture->key.cbuf_slot)
                << ",\"handle_byte_offset\":" << texture->key.handle_byte_offset
                << ",\"kind\":\"texture\""
                << ",\"dimension\":" << quote(dimensions[unsigned(texture->descriptor.shape.dimension)])
                << ",\"array\":" << (texture->descriptor.shape.array ? "true" : "false")
                << ",\"multisample\":" << (texture->descriptor.shape.multisample ? "true" : "false")
                << ",\"scalar\":" << quote(scalar_name(texture->descriptor.scalar))
                << ",\"component_format\":" << quote(formats[unsigned(texture->descriptor.format)])
                << ",\"normalized_coordinates\":"
                << (texture->descriptor.normalized_coordinates ? "true" : "false")
                << ",\"comparison\":" << (texture->descriptor.comparison ? "true" : "false")
                << ",\"resource_kind\":" << quote(texture->kind == pl::ResourceTextureKind::sampled
                    ? "sampled" : "storage_image")
                << ",\"access\":" << quote(texture->access == pl::Access::read ? "read" :
                    texture->access == pl::Access::write ? "write" : "read_write");
        } else {
            out << "{\"cbuf_slot\":" << unsigned(sampler->key.cbuf_slot)
                << ",\"handle_byte_offset\":" << sampler->key.handle_byte_offset
                << ",\"kind\":\"sampler\""
                << ",\"comparison\":" << (sampler->comparison ? "true" : "false")
                << ",\"normalized_coordinates\":"
                << (sampler->normalized_coordinates ? "true" : "false");
        }
        out << ",\"resource_id\":" << planned.resource.value
            << ",\"array_length\":" << (texture ? texture->array_length : sampler->array_length)
            << ",\"location\":";
        write_resource_location(out, planned.location);
        out << '}';
    }
    out << "],\"vertex_streams\":[";
    first = true;
    for (const auto& use : plan.io) {
        if (use.semantic != pl::IoSemantic::generic || use.access != pl::Access::read ||
            !use.vertex_format || stage != pl::Stage::vertex)
            continue;
        if (!first) out << ',';
        first = false;
        static constexpr const char* formats[] = {"u8", "s8", "unorm8", "snorm8", "u16",
            "s16", "unorm16", "snorm16", "f16", "u32", "s32", "f32"};
        out << "{\"slot\":" << use.location
            << ",\"attribute_index\":"
            << [&] {
                   for (const auto& item : options.vertex_attributes)
                       if (item.generic_slot == use.location) return unsigned(item.attribute_index);
                   throw std::runtime_error("Ported plan reads an undeclared vertex slot");
               }()
            << ",\"format\":" << quote(formats[unsigned(use.vertex_format->format)])
            << ",\"components\":" << unsigned(use.vertex_format->components)
            << ",\"byte_stride\":" << unsigned(use.vertex_format->components) * 4 << '}';
    }
    // Inputs classified by the post-SSA use analysis instead of a recorded
    // guest format: the runtime builds the vertex descriptor from the guest
    // attribute registers and admits only formats of this class.
    out << "],\"vertex_inputs\":[";
    first = true;
    for (const auto& use : plan.io) {
        if (use.semantic != pl::IoSemantic::generic || use.access != pl::Access::read ||
            !use.vertex_class || stage != pl::Stage::vertex)
            continue;
        if (!first) out << ',';
        first = false;
        static constexpr const char* classes[] = {"float", "uint", "sint"};
        out << "{\"location\":" << use.location
            << ",\"class\":" << quote(classes[unsigned(*use.vertex_class)]) << '}';
    }
    out << "],\"interface\":{\"position_input_mask\":";
    // The summary mask packs each target's component mask into its own four-bit
    // nibble, matching the SPH export-mask layout the fragment exporter reads.
    unsigned position_input = 0, position_output = 0;
    std::uint32_t color_output = 0;
    // One typed record per written render target, in first-written order. The
    // fragment exporter stores each component separately, so the writer merges
    // them back into one record per location; every component of a location
    // was written from the same fact-proven type, and disagreement here would
    // mean the plan itself is inconsistent.
    struct ColorOutputRecord {
        std::uint16_t location;
        std::uint8_t component_mask;
        pl::IoType type;
    };
    std::vector<ColorOutputRecord> color_outputs;
    bool writes_fragment_depth = false, writes_sample_mask = false;
    for (const auto& use : plan.io) {
        if (use.semantic == pl::IoSemantic::position && use.access != pl::Access::write)
            position_input |= use.mask & 15u;
        if (use.semantic == pl::IoSemantic::position && use.access == pl::Access::write)
            position_output |= use.mask & 15u;
        if (use.semantic == pl::IoSemantic::color && use.access == pl::Access::write) {
            if (use.location >= 8)
                throw std::runtime_error("Fragment colour output exceeds the eight render targets");
            color_output |= static_cast<std::uint32_t>(use.mask) << (4 * use.location);
            const auto prior = std::find_if(color_outputs.begin(), color_outputs.end(),
                [&](const ColorOutputRecord& record) { return record.location == use.location; });
            if (prior == color_outputs.end()) {
                color_outputs.push_back({use.location, use.mask, use.type});
            } else {
                if (prior->type != use.type)
                    throw std::runtime_error("Fragment colour output components disagree on their type");
                prior->component_mask |= use.mask;
            }
        }
        if (use.semantic == pl::IoSemantic::fragment_depth && use.access != pl::Access::read)
            writes_fragment_depth = true;
        if (use.semantic == pl::IoSemantic::sample_mask && use.access != pl::Access::read)
            writes_sample_mask = true;
    }
    out << position_input << ",\"position_output_mask\":" << position_output
        << ",\"color_output_mask\":" << color_output << ",\"color_outputs\":[";
    for (std::size_t index = 0; index < color_outputs.size(); ++index) {
        if (index) out << ',';
        const auto& record = color_outputs[index];
        out << "{\"location\":" << record.location
            << ",\"component_mask\":" << unsigned(record.component_mask)
            << ",\"scalar\":" << quote(scalar_name(record.type.scalar))
            << ",\"components\":" << unsigned(record.type.components) << '}';
    }
    out << "],\"writes_fragment_depth\":" << (writes_fragment_depth ? "true" : "false")
        << ",\"writes_sample_mask\":" << (writes_sample_mask ? "true" : "false");
    const auto io_list = [&](std::string_view name, bool inputs) {
        out << ',' << quote(name) << ":[";
        bool separator = false;
        for (const auto& use : plan.io) {
            if (use.semantic != pl::IoSemantic::generic) continue;
            if (inputs != (use.access != pl::Access::write)) continue;
            if (inputs && stage != pl::Stage::fragment && !(use.vertex_format || use.vertex_class))
                continue;
            static constexpr const char* modes[] = {"flat", "perspective", "linear"};
            if (separator) out << ',';
            separator = true;
            out << "{\"slot\":" << use.location
                << ",\"component_mask\":" << unsigned(use.mask)
                << ",\"scalar\":" << quote(scalar_name(use.type.scalar))
                << ",\"components\":" << unsigned(use.type.components)
                << ",\"interpolation\":" << quote(modes[unsigned(use.interpolation.mode)]) << '}';
        }
        out << ']';
    };
    io_list("generic_inputs", true);
    io_list("generic_outputs", false);
    out << "},\"dependencies\":[";
    first = true;
    for (const auto& dependency : plan.dependencies) {
        if (!first) out << ',';
        first = false;
        static constexpr const char* origins[] = {"validated_header", "captured_state",
            "explicit_option", "derived"};
        out << "{\"kind\":" << quote(fact_kind_name(dependency.key.kind))
            << ",\"index\":" << dependency.key.index
            << ",\"byte_offset\":" << dependency.key.byte_offset
            << ",\"component\":" << unsigned(dependency.key.component)
            << ",\"origin\":" << quote(origins[unsigned(dependency.origin.kind)])
            << ",\"value\":" << fact_value_json(dependency) << '}';
    }
    out << "]}\n";
    return out.str();
}

// Shader-side vertex fetch takes the lowest buffer index no planned resource
// uses for its per-draw table. Only a stage whose generic inputs all carry a
// code-derived class fetches: an input with a pipeline-state format fact
// reads through the vertex descriptor it was translated for.
void plan_vertex_fetch(pl::ResourcePlan& plan) {
    bool inputs = false;
    for (const auto& io : plan.io) {
        if (io.access != pl::Access::read || io.semantic != pl::IoSemantic::generic) continue;
        if (io.vertex_format || !io.vertex_class) return;
        if (io.location >= vertex_fetch_attributes)
            throw std::runtime_error("Vertex fetch input location exceeds the fetch table");
        inputs = true;
    }
    if (!inputs) return;
    std::array<bool, 31> occupied{};
    for (const auto& resource : plan.resources)
        if (const auto* direct = std::get_if<pl::MetalRange>(&resource.location);
            direct && direct->space == pl::MetalNamespace::buffer)
            for (std::uint32_t i = 0; i < direct->count; ++i) occupied.at(direct->first + i) = true;
    if (plan.argument_buffer) occupied.at(plan.argument_buffer->buffer_index) = true;
    if (plan.draw_support_buffer_index) occupied.at(*plan.draw_support_buffer_index) = true;
    const auto free = std::find(occupied.begin(), occupied.end(), false);
    if (free == occupied.end()) throw std::runtime_error("Vertex fetch has no free Metal buffer slot");
    plan.vertex_fetch_buffer_index = static_cast<std::uint32_t>(free - occupied.begin());
}

// The translation driver decodes, registers IR, builds SSA, legalizes,
// structures, plans resources and emits MSL. Every phase reports a structured
// diagnostic on failure.
pl::MslArtifact run_ported(const si::ParsedShader& shader, Bytes code,
                           const gm::EmitOptions& options,
                           std::optional<std::uint8_t> texture_handle_cbuf,
                           bool derived_descriptors,
                           std::optional<pl::ComputeLocalSize> compute_local_size,
                           std::optional<std::uint32_t> compute_shared_memory_bytes) {
    pl::StageImage image;
    image.stage = shader.stage == si::Stage::Vertex ? pl::Stage::vertex :
        shader.stage == si::Stage::Compute ? pl::Stage::compute : pl::Stage::fragment;
    if (shader.stage != si::Stage::Compute)
        std::copy_n(code.data() + shader.code_offset, image.sph.size(), image.sph.begin());
    const Bytes program = code.subspan(shader.program_offset, shader.program_size);
    image.bytes.assign(program.begin(), program.end());
    image.entry_pc = 8;
    pl::Limits limits{};
    const auto profile = ported_profile();
    DerivedTextureFacts derived;
    if (derived_descriptors)
        derived = scan_derived_texture_facts(image, texture_handle_cbuf);
    OptionFacts facts{options, image.sph, texture_handle_cbuf,
                      derived_descriptors ? &derived : nullptr, compute_local_size, compute_shared_memory_bytes};
    pl::FactReader reader{facts, limits};
    auto step = [](auto result) {
        if (const auto* error = std::get_if<pl::Diagnostic>(&result)) pipeline_failure(*error);
        return std::move(std::get<0>(std::move(result)));
    };
    auto decoded = step(pl::decode(image, reader, limits));
    auto register_ir = step(pl::emit_ir(decoded, reader, limits));
    auto ssa = step(pl::construct_ssa(std::move(register_ir), limits));
    auto legal = step(pl::legalize(std::move(ssa), reader, profile));
    auto structured = step(pl::structure(std::move(legal), limits));
    auto plan = step(pl::plan_resources(structured, profile));
    if (options.vertex_fetch && image.stage == pl::Stage::vertex) plan_vertex_fetch(plan);
    pl::EmitOptions emit_options;
    emit_options.fragment_output_types_from_render_targets = image.stage == pl::Stage::fragment;
    emit_options.emulate_depth_bias = image.stage == pl::Stage::fragment &&
        std::none_of(plan.io.begin(), plan.io.end(), [](const auto& io) {
            return io.semantic == pl::IoSemantic::fragment_depth;
        });
    if (emit_options.emulate_depth_bias) {
        // Reserve the shader-v1 binding even if a resource plan placed an
        // existing buffer there; reflection records any relocated binding.
        std::array<bool, 31> occupied{};
        for (const auto& resource : plan.resources) {
            const auto* direct = std::get_if<pl::MetalRange>(&resource.location);
            if (direct && direct->space == pl::MetalNamespace::buffer)
                for (std::uint32_t i = 0; i < direct->count; ++i)
                    occupied.at(direct->first + i) = true;
        }
        if (plan.argument_buffer) occupied.at(plan.argument_buffer->buffer_index) = true;
        auto relocate = [&](std::uint32_t& index, std::uint32_t count) {
            if (index != 30) return;
            const auto free = std::find(occupied.begin(), occupied.begin() + 30, false);
            if (count != 1 || free == occupied.begin() + 30)
                throw std::runtime_error("Shader depth bias has no free binding for the displaced buffer");
            index = static_cast<std::uint32_t>(free - occupied.begin());
            *free = true;
        };
        for (auto& resource : plan.resources) {
            auto* direct = std::get_if<pl::MetalRange>(&resource.location);
            if (direct && direct->space == pl::MetalNamespace::buffer)
                relocate(direct->first, direct->count);
        }
        if (plan.argument_buffer) relocate(plan.argument_buffer->buffer_index, 1);
    }
    emit_options.fragment_position = options.fragment_position ==
            gm::FragmentPositionConvention::MetalPositionWMatchesGuestPositionW
        ? pl::FragmentPositionConvention::metal_position_w_matches_guest_position_w
        : pl::FragmentPositionConvention::unspecified;
    return step(pl::emit_msl(structured, plan, emit_options));
}
} // namespace

TranslatedStage translate_stage(std::span<const std::byte> control, std::span<const std::byte> code,
                                const StageOptions& options) {
    auto parsed = si::parse_shader({view(control), view(code), {}});
    if (const auto* error = std::get_if<si::ParseError>(&parsed))
        throw std::runtime_error("parse code=" + std::to_string(unsigned(error->code)) +
            " pc=0 field=container value=" + std::to_string(error->value) +
            " bound=" + std::to_string(error->bound) + ": " + error->message());
    const auto& shader = std::get<si::ParsedShader>(parsed);
    if (shader.stage != si::Stage::Vertex && shader.stage != si::Stage::Fragment && shader.stage != si::Stage::Compute)
        throw std::runtime_error("stage code=unsupported pc=0 field=stage");
    if (shader.stage != si::Stage::Compute && (options.compute_local_size || options.compute_shared_memory_bytes))
        throw std::runtime_error("Compute launch options require a compute stage");
    if (shader.stage == si::Stage::Compute &&
        (!options.emission.vertex_attributes.empty() || options.emission.vertex_depth != gm::VertexDepthConvention::Unspecified ||
         options.emission.fragment_position != gm::FragmentPositionConvention::Unspecified))
        throw std::runtime_error("Graphics options are not compute-stage options");
    if (shader.stage == si::Stage::Vertex &&
        options.emission.fragment_position != gm::FragmentPositionConvention::Unspecified)
        throw std::runtime_error("--fragment-position is not a vertex-stage option");
    if (shader.stage == si::Stage::Fragment &&
        (!options.emission.vertex_attributes.empty() ||
         options.emission.vertex_depth != gm::VertexDepthConvention::Unspecified))
        throw std::runtime_error("Vertex options are not fragment-stage options");
    std::optional<pl::ComputeLocalSize> local_size;
    if (options.compute_local_size) local_size = pl::ComputeLocalSize{*options.compute_local_size};
    const Bytes bytes{code};
    const auto artifact = run_ported(shader, bytes, options.emission, options.texture_handle_cbuf,
                                     options.texture_descriptors_derived, local_size,
                                     options.compute_shared_memory_bytes);
    const Bytes identity = bytes.subspan(shader.code_offset, shader.c1_offset - shader.code_offset);
    const auto stage = shader.stage == si::Stage::Vertex ? pl::Stage::vertex :
        shader.stage == si::Stage::Compute ? pl::Stage::compute : pl::Stage::fragment;
    TranslatedStage result;
    result.stage = shader.stage;
    result.reflection = ported_reflection(artifact, options.emission, stage, identity);
    result.source = artifact.source;
    result.program.assign(identity.begin(), identity.end());
    return result;
}

} // namespace armigo::shader::translation
