// ============================================================================
// Armigo Tests - NVN API 231 Native Shader Stages
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

// The reflection written for native stages repeats the ported translator's.
// Both writers run on one synthetic fragment stage each; their documents must
// name the same fields in the same order, apart from the fields either writes
// only for a feature its stage uses, and the native one must make a stage.
// SetShaders through the native resolver records well-formed stages and
// translates them at first use.

#include "nvn_native_stages.h"
#include "entry_table.h"
#include "fixture.h"
#include "objects.h"
#include "programs.h"
#include "stage_fixture.h"

#include <armigo/platform/metal_shader_package.h>
#include <armigo/shader_translation/stage_translation.h>
#include "pipeline/phases.h"

#include <algorithm>
#include <unistd.h>
#include <iterator>
#include <fstream>
#include <filesystem>
#include <cstdio>
#include <optional>
#include <set>
#include <string>
#include <string_view>
#include <vector>

namespace {
namespace sn = armigo::shader_native;
namespace pl = armigo::shader::pipeline;
namespace gm = armigo::shader::graphics_msl;
namespace tr = armigo::shader::translation;

unsigned failures = 0;
#define CHECK(condition)                                                  \
    do {                                                                  \
        if (!(condition)) {                                               \
            std::fprintf(stderr, "FAIL %d: %s\n", __LINE__, #condition); \
            ++failures;                                                   \
        }                                                                 \
    } while (false)

// The keys of the object at `path`, in document order. A path starts with
// "{" for the root and names the key whose value each nested object is; array
// elements are "[]". String contents never count.
std::vector<std::string> keys(std::string_view json, const std::vector<std::string>& path) {
    std::vector<std::string> found, current;
    std::string value_key;
    bool value_pending = false;
    for (std::size_t i = 0; i < json.size(); ++i) {
        const char c = json[i];
        if (c == '"') {
            std::size_t end = i + 1;
            while (end < json.size() && json[end] != '"') end += json[end] == '\\' ? 2 : 1;
            std::size_t next = end + 1;
            while (next < json.size() && json[next] == ' ') ++next;
            if (next < json.size() && json[next] == ':') {
                value_key = std::string(json.substr(i + 1, end - i - 1));
                if (current == path) found.push_back(value_key);
                value_pending = true;
                i = next;
            } else {
                value_pending = false;
                i = end;
            }
        } else if (c == '{' || c == '[') {
            current.push_back(current.empty() ? "{" : value_pending ? value_key : "[]");
            value_pending = false;
        } else if (c == '}' || c == ']') {
            if (!current.empty()) current.pop_back();
            value_pending = false;
        } else if (c == ',') {
            value_pending = false;
        }
    }
    return found;
}

template <armigo::nvn::Entry E>
auto entry_function() {
    return reinterpret_cast<typename armigo::nvn::EntrySignature<E>::type>(armigo::nvn::installed_entry(E));
}

std::vector<std::string> without(std::vector<std::string> names, const std::set<std::string>& optional) {
    std::erase_if(names, [&](const std::string& name) { return optional.contains(name); });
    return names;
}

// A one-block register-IR fragment program for the shared phases: the
// production writer and stage maker then see what a front end emits for a
// render-target height read or a guest-memory load.
struct IrFragment {
    pl::Program p;
    IrFragment() {
        p.stage = pl::Stage::fragment; p.entry = pl::FunctionId{0};
        p.functions.push_back({0, pl::BlockId{0}, {pl::BlockId{0}}, {}, {}});
        p.blocks.push_back({pl::FunctionId{0}, {}, {}, {}, {}, pl::Return{}});
    }
    pl::ValueId value() {
        p.values.push_back({pl::ValueType::word32, pl::ValueKind::temporary});
        return pl::ValueId{static_cast<std::uint32_t>(p.values.size() - 1)};
    }
    void add(pl::Operation op) {
        op.source = {static_cast<pl::Pc>(p.operations.size() * 16), {}};
        op.effects = pl::required_effects(op);
        p.blocks[0].operations.push_back(pl::OperationId{static_cast<std::uint32_t>(p.operations.size())});
        p.operations.push_back(std::move(op));
    }
    pl::ValueId height() {
        pl::IoInfo io; io.semantic = pl::IoSemantic::render_target_height; io.type = {pl::ScalarType::u32, 1};
        pl::Operation op; op.opcode = pl::Opcode::load_io; op.numeric = {pl::Domain::unsigned_integer, pl::ScalarType::u32};
        const auto v = value(); op.outputs.push_back(v); op.info = io; add(std::move(op)); return v;
    }
    pl::ValueId guest_load(std::uint64_t address) {
        pl::MemoryInfo m; m.space = pl::AddressSpace::guest; m.access = pl::Access::read;
        m.element_bytes = 4; m.alignment = 4;
        m.dynamic_byte_offset = pl::InputIndex{0}; m.global_address_high = pl::InputIndex{1};
        pl::Operation op; op.opcode = pl::Opcode::load; op.numeric = {pl::Domain::unsigned_integer, pl::ScalarType::u32};
        op.inputs = {pl::LiteralBits{std::uint32_t(address), pl::ValueType::word32},
                     pl::LiteralBits{std::uint32_t(address >> 32), pl::ValueType::word32}};
        const auto v = value(); op.outputs.push_back(v); op.info = m; add(std::move(op)); return v;
    }
    void export_word(pl::Operand word) {
        for (unsigned c = 0; c < 4; ++c) {
            pl::IoInfo io; io.semantic = pl::IoSemantic::color; io.component_mask = 1u << c;
            io.type = {pl::ScalarType::u32, 4}; io.output_storage = true; io.data = {pl::InputIndex{0}};
            pl::Operation op; op.opcode = pl::Opcode::store_io; op.numeric = {pl::Domain::unsigned_integer, pl::ScalarType::u32};
            op.inputs = {c == 0 ? word : pl::Operand{pl::LiteralBits{0, pl::ValueType::word32}}}; op.info = io; add(std::move(op));
        }
    }
};
struct RawOutputs final : pl::FactSource {
    pl::Knowledge<pl::FactValue> lookup(pl::FactKey key) const override {
        if (key.kind == pl::FactKind::fragment_output_type)
            return pl::Known<pl::FactValue>{pl::IoType{pl::ScalarType::u32, 4}, {pl::OriginKind::explicit_option, "native-stage-test"}};
        return pl::Unknown{"No fact in this test"};
    }
};
std::optional<pl::MslArtifact> emit(pl::Program program, std::uint32_t tag) {
    pl::Limits limits; RawOutputs source; pl::FactReader facts(source, limits);
    const pl::MetalProfile profile{"armigo-native-metal4", 3, 31, 128, 16, true, false, true, false, false, true};
    auto ssa = pl::construct_ssa(pl::RegisterProgram{std::move(program)}, limits);
    if (!std::holds_alternative<pl::SsaProgram>(ssa)) return std::nullopt;
    auto legal = pl::legalize(std::get<pl::SsaProgram>(std::move(ssa)), facts, profile);
    if (!std::holds_alternative<pl::LegalProgram>(legal)) return std::nullopt;
    auto structured = pl::structure(std::get<pl::LegalProgram>(std::move(legal)), limits);
    if (!std::holds_alternative<pl::StructuredProgram>(structured)) return std::nullopt;
    const auto& tree = std::get<pl::StructuredProgram>(structured);
    auto plan = pl::plan_resources(tree, profile);
    if (!std::holds_alternative<pl::ResourcePlan>(plan)) return std::nullopt;
    pl::EmitOptions options; options.fragment_output_types_from_render_targets = true; options.guest_memory_tag = tag;
    auto artifact = pl::emit_msl(tree, std::get<pl::ResourcePlan>(plan), options);
    if (!std::holds_alternative<pl::MslArtifact>(artifact)) return std::nullopt;
    return std::get<pl::MslArtifact>(std::move(artifact));
}

} // namespace

// Refused stages are recorded by content: two stages with the same selector
// get two directories, the same bytes again reuse theirs, and a directory
// that cannot be created reports failure without a record.
void check_refused_stage_records() {
    namespace fs = std::filesystem;
    auto pattern = (fs::temp_directory_path() / "armigo-refused-XXXXXX").string();
    std::vector<char> path(pattern.begin(), pattern.end());
    path.push_back('\0');
    if (!mkdtemp(path.data())) {
        CHECK(false);
        return;
    }
    const fs::path root(path.data());
    const std::vector<std::byte> control(0xe0, std::byte{1});
    const std::vector<std::byte> first(64, std::byte{2}), second(64, std::byte{3});
    const auto a = armigo::platform::metal::record_refused_stage(root, 5, control, first);
    const auto b = armigo::platform::metal::record_refused_stage(root, 5, control, second);
    CHECK(!a.empty() && !b.empty() && a != b && a.starts_with("231-5-") && a.size() == 6 + 64);
    CHECK(armigo::platform::metal::record_refused_stage(root, 5, control, first) == a);
    const auto read = [](const fs::path& file) {
        std::ifstream input(file, std::ios::binary);
        const std::string text{std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
        return std::vector<std::byte>(reinterpret_cast<const std::byte*>(text.data()),
                                      reinterpret_cast<const std::byte*>(text.data()) + text.size());
    };
    CHECK(read(root / a / "control.bin") == control && read(root / a / "code.bin") == first);
    CHECK(read(root / b / "code.bin") == second);
    std::size_t entries = 0;
    for ([[maybe_unused]] const auto& entry : fs::directory_iterator(root)) ++entries;
    CHECK(entries == 2);
    // A regular file where the directory should be.
    { std::ofstream(root / "blocked") << "x"; }
    CHECK(armigo::platform::metal::record_refused_stage(root / "blocked", 5, control, first).empty());
    std::error_code ignored;
    fs::remove_all(root, ignored);
}

int main() {
    using namespace sn::test;
    // The ported writer on its own synthetic fragment stage.
    const auto maxwell = armigo::test::stage_fixture::fragment_stage();
    const auto ported = tr::translate_stage(maxwell.control, maxwell.code, {});

    // The native writer on a native fragment stage that exports a constant colour.
    const Fixture fragment({move(0, 0x3f800000), move(1, 0), move(2, 0), move(3, 0x3f800000), exit_instruction()});
    const auto program = fragment.program();
    sn::FrontendOptions options;
    options.vertex_fetch = false;
    options.fragment_position = pl::FragmentPositionConvention::metal_position_w_matches_guest_position_w;
    auto translated = sn::translate_native(program, options);
    CHECK(std::holds_alternative<pl::MslArtifact>(translated));
    if (!std::holds_alternative<pl::MslArtifact>(translated)) {
        std::fprintf(stderr, "native translation: %s\n", std::get<pl::Diagnostic>(translated).message.c_str());
        return 1;
    }
    const auto& artifact = std::get<pl::MslArtifact>(translated);
    const auto native = armigo::platform::metal::native_stage_reflection(artifact, gm::EmitOptions{},
                                                                          pl::Stage::fragment, program.image);

    // Fields written only when a stage uses their feature.
    const std::set<std::string> optional{"y_direction", "point_size", "point_coord", "depth_bias_mode",
                                         "custom_border", "draw_support", "guest_memory", "rasterizer_discard",
                                         "vertex_fetch", "compute", "handle_programs"};
    const auto ported_top = without(keys(ported.reflection, {"{"}), optional);
    const auto native_top = without(keys(native, {"{"}), optional);
    CHECK(!ported_top.empty() && ported_top == native_top);
    CHECK(std::find(native_top.begin(), native_top.end(), "interface") != native_top.end());
    CHECK(keys(ported.reflection, {"{", "interface"}) == keys(native, {"{", "interface"}));
    CHECK(keys(ported.reflection, {"{", "profile"}) == keys(native, {"{", "profile"}));
    CHECK(native.find("\"schema\":\"armigo-shader-stage-v3\"") == 1);
    CHECK(native.find("\"stage\":\"fragment\"") != std::string::npos);

    // The stage maker accepts the native document with its source.
    const auto made = armigo::platform::make_source_stage(
        armigo::shader_input::Stage::Fragment, program.image,
        {reinterpret_cast<const std::byte*>(native.data()), native.size()}, artifact.source);
    CHECK(std::holds_alternative<std::shared_ptr<const armigo::platform::CompiledGraphicsStage>>(made));
    if (const auto* failure = std::get_if<armigo::platform::MetalShaderPackageError>(&made))
        std::fprintf(stderr, "stage: %s\n", failure->detail.c_str());

    // A texture keyed at slot 31 round-trips with its handle program. An index
    // past the program list is refused.
    {
        auto with_handles = artifact;
        pl::TextureDescriptor descriptor;
        descriptor.shape.dimension = pl::TextureDimension::d2;
        descriptor.scalar = pl::ScalarType::f32;
        descriptor.format = pl::ComponentFormat::floating;
        descriptor.normalized_coordinates = true;
        pl::TextureUse texture;
        texture.kind = pl::ResourceTextureKind::sampled;
        texture.key = {pl::handle_program_slot, 0};
        texture.descriptor = descriptor;
        texture.access = pl::Access::read;
        texture.array_length = 1;
        std::uint32_t next_id = 0, next_texture = 0;
        for (const auto& planned : with_handles.reflection.resources) {
            if (planned.resource.value != pl::invalid_index)
                next_id = std::max(next_id, planned.resource.value + 1);
            if (const auto* range = std::get_if<pl::MetalRange>(&planned.location))
                if (range->space == pl::MetalNamespace::texture)
                    next_texture = std::max(next_texture, range->first + range->count);
        }
        pl::BindingLocation location = pl::MetalRange{pl::MetalNamespace::texture, next_texture, 1};
        if (with_handles.reflection.argument_buffer) {
            const auto member = with_handles.reflection.argument_buffer->byte_extent;
            location = pl::ArgumentBufferMember{member, 1, 8};
            with_handles.reflection.argument_buffer->byte_extent = member + 8;
        }
        with_handles.reflection.resources.push_back({pl::ResourceId{next_id}, texture, location});
        with_handles.reflection.handle_programs = {{1u, 3u, 0x10u, 1u, 3u, 0x14u, 3u, 0x20u}};
        with_handles.reflection.dependencies.push_back(
            {{pl::FactKind::texture_descriptor, pl::handle_program_slot, 0, 0},
             descriptor, {pl::OriginKind::derived, "handle-program"}});
        const auto document = armigo::platform::metal::native_stage_reflection(
            with_handles, gm::EmitOptions{}, pl::Stage::fragment, program.image);
        CHECK(document.find("\"handle_programs\":[[1,3,16,1,3,20,3,32]]") != std::string::npos);
        const auto stage = armigo::platform::make_source_stage(
            armigo::shader_input::Stage::Fragment, program.image,
            {reinterpret_cast<const std::byte*>(document.data()), document.size()}, with_handles.source);
        const auto* loaded = std::get_if<std::shared_ptr<const armigo::platform::CompiledGraphicsStage>>(&stage);
        CHECK(loaded != nullptr);
        if (const auto* failure = std::get_if<armigo::platform::MetalShaderPackageError>(&stage))
            std::fprintf(stderr, "handle program stage: %s\n", failure->detail.c_str());
        if (loaded) {
            CHECK((*loaded)->handle_programs == with_handles.reflection.handle_programs);
            const auto bound = std::find_if((*loaded)->resources.begin(), (*loaded)->resources.end(),
                [](const auto& resource) { return resource.cbuf_slot == pl::handle_program_slot; });
            CHECK(bound != (*loaded)->resources.end() && bound->handle_byte_offset == 0 &&
                  bound->array_length == 1);
        }
        auto out_of_range = with_handles;
        for (auto& planned : out_of_range.reflection.resources)
            if (auto* use = std::get_if<pl::TextureUse>(&planned.use))
                if (use->key.cbuf_slot == pl::handle_program_slot) use->key.handle_byte_offset = 4;
        const auto refused_document = armigo::platform::metal::native_stage_reflection(
            out_of_range, gm::EmitOptions{}, pl::Stage::fragment, program.image);
        const auto refused = armigo::platform::make_source_stage(
            armigo::shader_input::Stage::Fragment, program.image,
            {reinterpret_cast<const std::byte*>(refused_document.data()), refused_document.size()},
            out_of_range.source);
        CHECK(std::holds_alternative<armigo::platform::MetalShaderPackageError>(refused));
        if (const auto* failure = std::get_if<armigo::platform::MetalShaderPackageError>(&refused))
            CHECK(failure->detail.find("Handle program index") != std::string::npos);
    }

    // A fragment reading the render-target height takes a version-2
    // draw-parameter record, and one reading guest memory a guest_memory
    // record with its tag: both pass the production writer and stage maker.
    {
        IrFragment height;
        height.export_word(height.height());
        const auto with_height = emit(std::move(height.p), 0);
        CHECK(with_height && with_height->reflection.reads_render_target_height);
        IrFragment guest;
        guest.export_word(guest.guest_load(0x7db6000040ull));
        const auto with_guest = emit(std::move(guest.p), 0x5eed1234u);
        CHECK(with_guest.has_value());
        for (const auto* artifact : {with_height ? &*with_height : nullptr, with_guest ? &*with_guest : nullptr}) {
            if (!artifact) continue;
            const auto document = armigo::platform::metal::native_stage_reflection(*artifact, gm::EmitOptions{},
                                                                                     pl::Stage::fragment, program.image);
            const bool reads_height = artifact == &*with_height;
            CHECK(document.find(reads_height ? "\"draw_support\":{\"schema\":\"armigo-draw-parameters\",\"version\":2,\"byte_extent\":8"
                                             : "\"guest_memory\":{\"schema\":\"armigo-guest-memory\",\"version\":3,\"byte_extent\":263200") !=
                  std::string::npos);
            const auto stage = armigo::platform::make_source_stage(
                armigo::shader_input::Stage::Fragment, program.image,
                {reinterpret_cast<const std::byte*>(document.data()), document.size()}, artifact->source);
            const auto* made = std::get_if<std::shared_ptr<const armigo::platform::CompiledGraphicsStage>>(&stage);
            CHECK(made != nullptr);
            if (const auto* failure = std::get_if<armigo::platform::MetalShaderPackageError>(&stage))
                std::fprintf(stderr, "stage: %s\n", failure->detail.c_str());
            if (!made) continue;
            if (reads_height) {
                CHECK((*made)->draw_support_buffer_index == artifact->reflection.draw_support_buffer_index);
                CHECK(!(*made)->guest_memory);
            } else {
                CHECK((*made)->guest_memory && (*made)->guest_memory->tag == 0x5eed1234u);
                CHECK(!(*made)->draw_support_buffer_index);
            }
        }
        // A draw-parameter record of the wrong version for its stage is refused.
        if (with_height) {
            auto document = armigo::platform::metal::native_stage_reflection(*with_height, gm::EmitOptions{},
                                                                               pl::Stage::fragment, program.image);
            const auto at = document.find("\"version\":2,\"byte_extent\":8");
            if (at != std::string::npos) document.replace(at, 27, "\"version\":1,\"byte_extent\":4");
            const auto stage = armigo::platform::make_source_stage(
                armigo::shader_input::Stage::Fragment, program.image,
                {reinterpret_cast<const std::byte*>(document.data()), document.size()}, with_height->source);
            CHECK(std::holds_alternative<armigo::platform::MetalShaderPackageError>(stage));
        }
    }

    // An output-less vertex stage reflects its rasterizer-discard requirement.
    {
        Fixture empty({exit_instruction()}, sn::Stage::vertex);
        empty.code[0x35] = std::byte{};
        sn::FrontendOptions vertex_options; vertex_options.vertex_depth = pl::DepthConvention::negative_one_to_one;
        vertex_options.vertex_fetch = false;
        const auto empty_program = empty.program();
        auto result = sn::translate_native(empty_program, vertex_options);
        CHECK(std::holds_alternative<pl::MslArtifact>(result));
        if (const auto* artifact = std::get_if<pl::MslArtifact>(&result)) {
            const auto document = armigo::platform::metal::native_stage_reflection(*artifact, gm::EmitOptions{},
                                                                                     pl::Stage::vertex, empty_program.image);
            CHECK(document.find("\"rasterizer_discard\":\"required-v1\"") != std::string::npos);
            const auto stage = armigo::platform::make_source_stage(
                armigo::shader_input::Stage::Vertex, empty_program.image,
                {reinterpret_cast<const std::byte*>(document.data()), document.size()}, artifact->source);
            const auto* made = std::get_if<std::shared_ptr<const armigo::platform::CompiledGraphicsStage>>(&stage);
            CHECK(made && (*made)->requires_rasterizer_discard);
            if (const auto* failure = std::get_if<armigo::platform::MetalShaderPackageError>(&stage))
                std::fprintf(stderr, "vertex stage: %s\n", failure->detail.c_str());
        }
    }

    // SetShaders as the console driver takes it, with controls as a title
    // passes them (zero container extents). A well-formed stage is recorded
    // and SetShaders succeeds; a stage the front end refuses fails at first
    // use, which marks its program's stage failed; malformed input fails.
    {
        namespace nvn = armigo::nvn;
        nvn::select_api_version({231, 40});
        nvn::register_program_entries();
        nvn::set_native_program_resolver();
        const auto initialize = entry_function<nvn::Entry::nvnProgramInitialize>();
        const auto finalize = entry_function<nvn::Entry::nvnProgramFinalize>();
        const auto set = entry_function<nvn::Entry::nvnProgramSetShaders>();
        const auto runtime = [](Fixture stage) {
            put(stage.control, 0x30, 0);
            return stage;
        };
        const auto descriptor = [](const Fixture& stage) {
            return nvn::NVNshaderData{reinterpret_cast<std::uint64_t>(stage.code.data()), stage.control.data()};
        };
        const auto translatable = runtime(Fixture({move(0, 0x3f800000), move(1, 0), move(2, 0), move(3, 0x3f800000),
                                                   exit_instruction()}));
        const auto refused = runtime(Fixture({raw(0xfff), exit_instruction()}));
        auto malformed = translatable;
        put(malformed.control, 0, 0x34127698);
        nvn::NVNprogram program{};
        CHECK(initialize(&program, nullptr));
        const auto& live = *reinterpret_cast<const nvn::ProgramState*>(&program);
        auto shader = descriptor(translatable);
        CHECK(set(&program, 1, &shader) == 1);
        auto resolved = nvn::retain_program(live);
        CHECK(resolved && resolved->stages[1] && !resolved->failed_stages);
        shader = descriptor(refused);
        CHECK(set(&program, 1, &shader) == 1);
        resolved = nvn::retain_program(live);
        CHECK(resolved && !resolved->stages[1] && resolved->failed_stages == 2u);
        shader = descriptor(malformed);
        CHECK(set(&program, 1, &shader) == 0);
        // Control version 17, which the front end reads with the version-20
        // layout, translates; a version the driver does not read fails
        // SetShaders.
        auto older = translatable;
        put(older.control, 8, 17);
        shader = descriptor(older);
        CHECK(set(&program, 1, &shader) == 1);
        resolved = nvn::retain_program(live);
        CHECK(resolved && resolved->stages[1] && !resolved->failed_stages);
        auto newer = translatable;
        put(newer.control, 8, 21);
        shader = descriptor(newer);
        CHECK(set(&program, 1, &shader) == 0);
        // A task or mesh stage takes a tessellation slot (mesh: 4); the front
        // end does not translate it yet, so it is refused at first use.
        auto mesh = translatable;
        mesh.control[0x56] = std::byte{7};
        put(mesh.control, 0x58, 32);  // a mesh workgroup of 32 threads
        const nvn::NVNshaderData both[2]{descriptor(mesh), descriptor(translatable)};
        CHECK(set(&program, 2, both) == 1);
        resolved = nvn::retain_program(live);
        CHECK(resolved && resolved->stages[1] && !resolved->stages[4] && resolved->failed_stages == 0x10u &&
              resolved->state.stage_count == 2);
        const nvn::NVNshaderData twice[2]{descriptor(mesh), descriptor(mesh)};
        CHECK(set(&program, 2, twice) == 0);
        finalize(&program);
    }

    // A translated task and mesh stage pass the native writer and the stage
    // maker as the tessellation stage kinds, launched as their controls say.
    {
        // ISBEWR data, [RZ + offset], mode: a mesh stage's attribute and index stores.
        const auto store = [](unsigned data, std::uint32_t offset, unsigned mode) {
            auto i = raw(0x927);
            field(i, 32, 8, data);
            field(i, 24, 8, 255);
            field(i, 40, 24, offset);
            field(i, 72, 8, mode);
            return i;
        };
        std::vector<sn::Instruction> code{move(1, 0xbf800000u), move(2, 0x3f800000u), move(6, 1), move(7, 2), move(8, 3)};
        for (unsigned vertex = 0; vertex < 4; ++vertex) {
            code.push_back(store(vertex & 1 ? 2 : 1, 0 * 128 + vertex * 4, 0xf8));
            code.push_back(store(vertex & 2 ? 2 : 1, 1 * 128 + vertex * 4, 0xf8));
            code.push_back(store(255, 2 * 128 + vertex * 4, 0xf8));
            code.push_back(store(2, 3 * 128 + vertex * 4, 0xf8));
        }
        for (unsigned at = 0; const unsigned index : {255u, 6u, 7u, 7u, 6u, 8u}) code.push_back(store(index, 4 + at++, 0x80));
        code.push_back(exit_instruction());
        Fixture mesh(std::move(code), sn::Stage::mesh);
        put(mesh.control, 0x58, 32);
        mesh.code[0x36] = std::byte{1};
        put(mesh.control, 0x5c, 0x03000000u);
        put(mesh.control, 0x60, 4 | (2u << 16));
        Fixture task({move(1, 1), store(1, 4, 0xb8), exit_instruction()}, sn::Stage::task);
        put(task.control, 0x58, 32 | (0x20u << 16));
        for (const auto* fixture : {&mesh, &task}) {
            const bool is_mesh = fixture == &mesh;
            const auto native_program = fixture->program();
            CHECK(native_program.mesh_launch.has_value());
            if (!native_program.mesh_launch) continue;
            sn::FrontendOptions launch_options;
            launch_options.vertex_fetch = false;
            gm::EmitOptions emission;
            if (is_mesh) {
                launch_options.vertex_depth = pl::DepthConvention::negative_one_to_one;
                emission.vertex_depth = gm::VertexDepthConvention::NegativeOneToOne;
            }
            auto translation = sn::translate_native(native_program, launch_options);
            if (const auto* refused = std::get_if<pl::Diagnostic>(&translation)) {
                std::fprintf(stderr, "%s translation: %s\n", is_mesh ? "mesh" : "task", refused->message.c_str());
                CHECK(false);
                continue;
            }
            const auto& launched = std::get<pl::MslArtifact>(translation);
            const auto& facts = *native_program.mesh_launch;
            const armigo::platform::metal::NativeMeshLaunch launch{!is_mesh, facts.threads, facts.payload_bytes,
                facts.task, facts.primitive_vertices, facts.max_vertices, facts.max_primitives};
            const auto document = armigo::platform::metal::native_stage_reflection(
                launched, emission, is_mesh ? pl::Stage::vertex : pl::Stage::compute, native_program.image, &launch);
            CHECK(document.find(is_mesh ? "\"stage\":\"mesh\"" : "\"stage\":\"task\"") != std::string::npos);
            CHECK(document.find("\"compute\":{") == std::string::npos);
            const auto kind = is_mesh ? armigo::shader_input::Stage::TessellationEvaluation
                                      : armigo::shader_input::Stage::TessellationControl;
            const auto stage = armigo::platform::make_source_stage(kind, native_program.image,
                {reinterpret_cast<const std::byte*>(document.data()), document.size()}, launched.source);
            if (const auto* failure = std::get_if<armigo::platform::MetalShaderPackageError>(&stage))
                std::fprintf(stderr, "%s stage: %s\n", is_mesh ? "mesh" : "task", failure->detail.c_str());
            const auto* made = std::get_if<std::shared_ptr<const armigo::platform::CompiledGraphicsStage>>(&stage);
            CHECK(made != nullptr);
            if (!made) continue;
            CHECK((*made)->stage == kind);
            CHECK((*made)->entry_point == (is_mesh ? "armigo_mesh_main" : "armigo_object_main"));
            const armigo::platform::CompiledMeshLaunch expected{{32, 1, 1}, is_mesh ? 0u : 32u};
            CHECK(is_mesh ? (*made)->mesh == expected && !(*made)->task : (*made)->task == expected && !(*made)->mesh);
            CHECK((*made)->interface.generic_inputs.empty() && (*made)->color_outputs.empty());
        }
    }

    // A pool record names its kind and buffer location. A texture pool also
    // names its element type, including comparison. Neither carries a
    // constant-buffer key, and a sampler pool carries no sampler-state flags.
    {
        using namespace sn::test;
        auto sample = raw(0x361);
        field(sample, 16, 8, 0); field(sample, 24, 8, 8); field(sample, 32, 8, 12); field(sample, 59, 1, 1);
        field(sample, 60, 1, 1); field(sample, 61, 3, 1); field(sample, 64, 8, 255); field(sample, 72, 4, 1);
        field(sample, 81, 3, 7); field(sample, 87, 3, 1);
        std::vector<sn::Instruction> code;
        for (unsigned r = 0; r < 4; ++r) code.push_back(move(r, 0));
        code.push_back(move(8, 0x3f000000u));
        code.push_back(move(9, 0x3f000000u));
        code.push_back(move(12, 1u | (1u << 20)));
        code.push_back(sample);
        code.push_back(exit_instruction());
        Fixture fixture(code);
        put(fixture.code, 0x48, 0xf);
        sn::FrontendOptions pool_options;
        pool_options.texture_pools = true;
        const auto program = fixture.program();
        auto translated = sn::translate_native(program, pool_options);
        CHECK(std::holds_alternative<pl::MslArtifact>(translated));
        if (const auto* error = std::get_if<pl::Diagnostic>(&translated))
            std::fprintf(stderr, "pool reflection: %s\n", error->message.c_str());
        if (const auto* artifact = std::get_if<pl::MslArtifact>(&translated)) {
            const auto document = armigo::platform::metal::native_stage_reflection(
                *artifact, gm::EmitOptions{}, pl::Stage::fragment, program.image);
            const auto at = document.find("\"textures\":[");
            std::vector<std::string> objects;
            if (at != std::string::npos) {
                int depth = 0;
                std::size_t start = std::string::npos;
                for (std::size_t i = at + std::char_traits<char>::length("\"textures\":["); i < document.size(); ++i) {
                    if (document[i] == '{') {
                        if (depth == 0) start = i;
                        ++depth;
                    } else if (document[i] == '}') {
                        --depth;
                        if (depth == 0 && start != std::string::npos) {
                            objects.push_back(document.substr(start, i - start + 1));
                            start = std::string::npos;
                        }
                    } else if (document[i] == ']' && depth == 0) break;
                }
            }
            const auto has = [](const std::string& object, const char* key) {
                return object.find(std::string("\"") + key + "\":") != std::string::npos;
            };
            CHECK(objects.size() == 2);
            if (objects.size() == 2) {
                CHECK(objects[0].rfind("{\"kind\":\"texture_pool\"", 0) == 0);
                CHECK(has(objects[0], "dimension") && has(objects[0], "array") && has(objects[0], "multisample"));
                CHECK(has(objects[0], "scalar") && has(objects[0], "component_format") && has(objects[0], "comparison"));
                CHECK(!has(objects[0], "cbuf_slot") && !has(objects[0], "handle_byte_offset"));
                CHECK(!has(objects[0], "normalized_coordinates"));
                CHECK(objects[0].find("\"namespace\":\"buffer\",\"first\":0,\"count\":1") != std::string::npos);
                CHECK(objects[1].rfind("{\"kind\":\"sampler_pool\"", 0) == 0);
                CHECK(!has(objects[1], "cbuf_slot") && !has(objects[1], "handle_byte_offset"));
                CHECK(!has(objects[1], "comparison") && !has(objects[1], "normalized_coordinates"));
                CHECK(objects[1].find("\"namespace\":\"buffer\",\"first\":1,\"count\":1") != std::string::npos);
            } else std::fprintf(stderr, "pool reflection:\n%s\n", document.c_str());
        }
    }

    check_refused_stage_records();
    std::printf("NVN native stage failures: %u\n", failures);
    return failures ? 1 : 0;
}
