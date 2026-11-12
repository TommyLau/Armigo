// ============================================================================
// Armigo Tests - Native Runtime Verification
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================
// Independently authored synthetic tests for resource planning and bindless
// handle legalization.

#include "pipeline/phases.h"
#include <algorithm>
#include <iostream>
#include <map>
#include <string_view>

namespace {
using namespace armigo::shader::pipeline;
int failures{};
int cleanup_calls{}, nontexture_calls{};
bool reject_pass{};
void check(bool value, std::string_view label) {
    if (!value) { std::cerr << "FAIL: " << label << '\n'; ++failures; }
}
template <class T> const T* success(const Result<T>& result, std::string_view label) {
    const auto* value = std::get_if<T>(&result);
    check(value != nullptr, label);
    if (const auto* error = std::get_if<Diagnostic>(&result)) std::cerr << "  " << error->message << '\n';
    return value;
}
template <class T> const Diagnostic* rejected(const Result<T>& result, DiagnosticCode code, std::string_view label) {
    const auto* error = std::get_if<Diagnostic>(&result);
    check(error && error->code == code, label);
    if (error && error->code != code) std::cerr << "  unexpected error: " << error->message << '\n';
    return error;
}
class Facts final : public FactSource {
public:
    std::map<FactKey, FactValue> values;
    Knowledge<FactValue> lookup(FactKey key) const override {
        auto it = values.find(key);
        if (it == values.end()) return Unknown{"Synthetic fact is unavailable"};
        return Known<FactValue>{it->second, {OriginKind::explicit_option, "resources-synthetic"}};
    }
    template <FactKind K> void set(FactQuery<K> key, typename FactTraits<K>::value_type value) { values[key.key()] = std::move(value); }
};
MetalProfile profile() { return {"synthetic-metal", 1, 31, 32, 16, true, false, true, false, false, true}; }
TextureDescriptor descriptor() { return {{TextureDimension::d2, false, false}, ScalarType::f32,
    ComponentFormat::floating, true, false, 1, "synthetic-format-v1"}; }
Facts facts() {
    Facts f;
    f.set(fact_query::texture({2, 16}), descriptor());
    f.set(fact_query::texture({3, 32}), descriptor());
    return f;
}
Program empty(Stage stage = Stage::fragment) {
    Program p;
    p.stage = stage; p.entry = FunctionId{0};
    p.functions.push_back({8, BlockId{0}, {BlockId{0}}, {}, {}});
    p.blocks.push_back({FunctionId{0}, {8, {}}, {}, {}, {}, Return{}});
    return p;
}
Operand word(std::uint32_t value) { return LiteralBits{value, ValueType::word32}; }
ValueId add(Program& p, Operation operation) {
    ValueId result{static_cast<std::uint32_t>(p.values.size())};
    p.values.push_back({storage_type(operation.numeric.scalar)});
    operation.outputs = {result};
    p.blocks[0].operations.push_back(OperationId{static_cast<std::uint32_t>(p.operations.size())});
    p.operations.push_back(std::move(operation));
    return result;
}
ValueId copy(Program& p, Operand input) { return add(p, {{8, InstructionName::mov}, Opcode::copy, Effects::none, {}, {input}, {}, {}}); }
ValueId binary(Program& p, Opcode opcode, Operand a, Operand b) {
    return add(p, {{8, {}}, opcode, Effects::none, {}, {a, b}, {}, {}});
}
Operation sample(HandleSource handle = TextureKey{2, 16}) {
    TextureInfo info;
    info.texture = handle; info.coordinates = {InputIndex{0}, InputIndex{1}};
    Operation op{{40, InstructionName::tex}, Opcode::texture_sample, Effects::texture | Effects::derivative,
        {Domain::floating, ScalarType::f32, Rounding::nearest_even}, {word(0), word(0)}, {}, info};
    return op;
}
Program sampled() { auto p = empty(); add(p, sample()); return p; }
Program bindless(Operand input) {
    auto p = empty();
    auto op = sample(DynamicHandle{InputIndex{0}});
    op.inputs.insert(op.inputs.begin(), input);
    auto& t = std::get<TextureInfo>(op.info);
    t.coordinates = {InputIndex{1}, InputIndex{2}};
    add(p, op);
    return p;
}
StructuredProgram structured(const LegalProgram& legal) {
    return {legal.ir, legal.requirements, legal.profile, {BlockRegion{BlockId{0}}}, {{FunctionId{0}, RegionId{0}}}};
}
Result<LegalProgram> run(Program p, Facts& f, const MetalProfile& metal) {
    FactReader reader(f); return legalize({std::move(p)}, reader, metal);
}
Result<LegalProgram> run(Program p, Facts& f) { return run(std::move(p), f, profile()); }
const BufferUse* constant_buffer(const ResourceRequirements& requirements, std::uint8_t slot) {
    for (const auto& use : requirements.uses) {
        const auto* buffer = std::get_if<BufferUse>(&use);
        const auto* source = buffer ? std::get_if<ConstantBufferSource>(&buffer->source) : nullptr;
        if (source && source->slot == slot) return buffer;
    }
    return nullptr;
}

void vertex_depth_dependency() {
    auto p = empty(Stage::vertex);
    IoInfo io;
    io.semantic = IoSemantic::position;
    io.type = {ScalarType::f32, 4};
    io.component_mask = 15;
    io.data = {InputIndex{0}, InputIndex{1}, InputIndex{2}, InputIndex{3}};
    p.operations.push_back({{8, InstructionName::ast}, Opcode::store_io, Effects::write_io, {},
                            {word(0), word(0), word(0), word(0x3f800000)}, {}, io});
    p.blocks[0].operations.push_back(OperationId{0});
    auto f = facts();
    auto missing = run(p, f);
    rejected(missing, DiagnosticCode::missing_fact, "Position export requires a depth convention");
    if (const auto* error = std::get_if<Diagnostic>(&missing))
        check(error->missing_fact == FactKey{FactKind::depth_convention} && error->pc == 8,
              "Missing depth diagnostic identifies the fact and position export");
    f.set(FactQuery<FactKind::depth_convention>{}, DepthConvention::negative_one_to_one);
    const auto result = run(p, f);
    if (const auto* legal = success(result, "Position export consumes depth convention")) {
        check(legal->ir.dependencies.size() == 1 &&
              legal->ir.dependencies[0].key == FactKey{FactKind::depth_convention} &&
              std::get<DepthConvention>(legal->ir.dependencies[0].value) == DepthConvention::negative_one_to_one,
              "Depth is recorded by legalization without caller injection");
        const auto plan = plan_resources(structured(*legal), profile());
        if (const auto* value = success(plan, "Position resource plan succeeds"))
            check(value->dependencies == legal->ir.dependencies, "Plan carries the consumed depth dependency");
    }
    f.set(FactQuery<FactKind::depth_convention>{}, static_cast<DepthConvention>(2));
    rejected(run(p, f), DiagnosticCode::invalid_fact, "Invalid depth convention fails explicitly");
    f.set(FactQuery<FactKind::depth_convention>{}, DepthConvention::zero_to_one);
    p.dependencies.push_back({{FactKind::depth_convention}, DepthConvention::negative_one_to_one,
                              {OriginKind::explicit_option, "resources-synthetic"}});
    rejected(run(p, f), DiagnosticCode::conflicting_fact, "Inherited depth cannot change during legalization");
}

void basic_resolution() {
    auto f = facts();
    auto p = sampled();
    auto result = run(p, f);
    const auto* legal = success(result, "Bound sample resolves");
    if (!legal) return;
#if !__has_include("pipeline/passes/passes.h")
    check(cleanup_calls > 0 && nontexture_calls > 0, "Mandatory passes are called");
#endif
    check(legal->requirements.uses.size() == 2, "Sample allocates texture and sampler only");
    const auto& t = std::get<TextureInfo>(legal->ir.operations[0].info);
    check(t.descriptor == descriptor() && t.resource && t.sampler_resource && t.resource != t.sampler_resource, "Distinct resolved resource IDs");
    check(legal->ir.dependencies.size() == 1 && legal->ir.dependencies[0].key == fact_query::texture({2, 16}).key(),
        "Texture resolution consumes only descriptor semantics");
    auto planned = plan_resources(structured(*legal), profile());
    if (const auto* plan = success(planned, "Resolved sample plans")) {
        check(std::get<MetalRange>(plan->resources[0].location).space == MetalNamespace::texture && std::get<MetalRange>(plan->resources[0].location).first == 0, "Texture starts at zero");
        check(std::get<MetalRange>(plan->resources[1].location).space == MetalNamespace::sampler && std::get<MetalRange>(plan->resources[1].location).first == 0, "Independent sampler starts at zero");
        check(plan->dependencies == legal->ir.dependencies, "Plan retains exact dependencies");
    }
    add(p, sample());
    auto repeated = run(p, f);
    if (const auto* value = success(repeated, "Repeated texture resolves")) check(value->requirements.uses.size() == 2, "Resource uses deduplicate");
    f.values.erase(fact_query::texture({2, 16}).key());
    auto missing = run(sampled(), f);
    if (const auto* error = rejected(missing, DiagnosticCode::missing_fact, "Unknown descriptor fails")) {
        check(error->missing_fact == fact_query::texture({2, 16}).key() && error->pc == 40, "Missing descriptor names key and source PC");
    }
    f = facts(); f.set(FactQuery<FactKind::constant_buffer_bound_mask>{}, std::uint32_t{0});
    success(run(sampled(), f), "Deprecated binding state does not reject texture provenance");
    f = facts(); f.set(fact_query::texture({2, 16}), TextureDescriptor{});
    rejected(run(sampled(), f), DiagnosticCode::invalid_fact, "Empty descriptor namespace fails");
}
void bindless_resolution() {
    auto f = facts();
    auto result = run(bindless(ConstantWord{2, 16}), f);
    if (const auto* legal = success(result, "Direct CB handle resolves")) {
        const auto& op = legal->ir.operations[0];
        const auto& t = std::get<TextureInfo>(op.info);
        check(op.inputs.size() == 2 && t.coordinates[0].value == 0 && t.coordinates[1].value == 1, "Handle removal remaps every coordinate");
    }
    rejected(run(bindless(word(7)), f), DiagnosticCode::unresolved_resource, "Raw handle bits are not descriptor provenance");
    f.set(fact_query::texture({2, 4}), descriptor());
    auto unaligned_program = empty();
    MemoryInfo handle_load; handle_load.constant_buffer_slot = 2; handle_load.byte_offset = 6;
    auto handle_value = add(unaligned_program,
        {{8, {}}, Opcode::load, Effects::read_memory, {}, {}, {}, handle_load});
    auto unaligned_sample = sample(DynamicHandle{InputIndex{2}});
    unaligned_sample.inputs.push_back(handle_value);
    add(unaligned_program, std::move(unaligned_sample));
    auto unaligned = run(unaligned_program, f);
    if (const auto* legal = success(unaligned, "Unaligned U32 handle address resolves by word provenance"))
        check(std::get<TextureKey>(std::get<TextureInfo>(legal->ir.operations.back().info).texture) == TextureKey{2, 4},
            "Handle provenance aligns a U32 address down to its source word");
    rejected(run(bindless(ConstantWord{2, constant_buffer_window_bytes}), f), DiagnosticCode::unresolved_resource,
        "Handle provenance outside the CB window is rejected");
    auto p = empty();
    auto h = copy(p, ConstantWord{2, 16});
    auto op = sample(DynamicHandle{InputIndex{2}});
    op.inputs.push_back(h);
    add(p, op);
    auto copied = run(p, f);
    if (const auto* legal = success(copied, "Copied CB handle resolves")) {
        check(legal->ir.operations.size() == 1 && legal->ir.operations[0].opcode == Opcode::texture_sample,
            "Copied handle propagates without an unused CB load");
        check(std::none_of(legal->ir.dependencies.begin(), legal->ir.dependencies.end(), [](const auto& d) { return d.key.kind == FactKind::constant_buffer_word; }), "Provenance alone does not specialize raw handle words");
    }
    p = empty();
    auto tex = binary(p, Opcode::bit_and, ConstantWord{2, 16}, word(0x000fffff));
    auto sam = binary(p, Opcode::bit_and, word(0xfff00000), ConstantWord{3, 32});
    auto combined = binary(p, Opcode::bit_or, sam, tex);
    op = sample(DynamicHandle{InputIndex{2}}); op.inputs.push_back(combined); add(p, op);
    auto split = run(p, f);
    if (const auto* legal = success(split, "Masked separate handles resolve in either OR order")) {
        const auto& t = std::get<TextureInfo>(legal->ir.operations[3].info);
        check(std::get<TextureKey>(t.texture) == TextureKey{2, 16} &&
            std::get<TextureKey>(*t.sampler) == TextureKey{3, 32}, "Texture and sampler retain distinct keys");
    }
    std::get<LiteralBits>(p.operations[0].inputs[1]).bits = 0xff;
    rejected(run(p, f), DiagnosticCode::unresolved_resource, "Incorrect texture mask is not guessed");
    p.operations[0].inputs[1] = ConstantWord{1, 4};
    f.set(fact_query::constant_word(1, 4), std::uint32_t{0xfffff});
    auto specialized = run(p, f);
    if (const auto* legal = success(specialized, "Known CB mask resolves"))
        check(std::any_of(legal->ir.dependencies.begin(), legal->ir.dependencies.end(), [](const auto& d) { return d.key == fact_query::constant_word(1, 4).key(); }), "Consumed mask word is a dependency");
    f.values.erase(fact_query::constant_word(1, 4).key());
    rejected(run(p, f), DiagnosticCode::unresolved_resource, "Unknown optional mask word leaves handle unresolved");
}
void phi_and_io_resolution() {
    auto f = facts();
    auto p = empty();
    auto left = copy(p, ConstantWord{2, 16});
    auto right = copy(p, ConstantWord{2, 16});
    ValueId merged{static_cast<std::uint32_t>(p.values.size())};
    p.values.push_back({ValueType::word32});
    auto op = sample(DynamicHandle{InputIndex{2}}); op.inputs.push_back(merged); add(p, op);
    p.blocks.resize(4);
    p.functions[0].blocks = {BlockId{0}, BlockId{1}, BlockId{2}, BlockId{3}};
    for (auto& block : p.blocks) block.function = FunctionId{0};
    p.blocks[0].operations.clear();
    const ValueId condition{static_cast<std::uint32_t>(p.values.size())};
    p.values.push_back({ValueType::predicate}); p.functions[0].parameters.push_back(condition);
    p.blocks[0].terminator = Branch{condition, BlockId{1}, BlockId{2}};
    for (std::uint32_t id = 1; id <= 2; ++id) {
        p.blocks[id].predecessors = {BlockId{0}};
        p.blocks[id].operations = {OperationId{id - 1}};
        p.blocks[id].terminator = Jump{BlockId{3}};
    }
    p.blocks[3].predecessors = {BlockId{1}, BlockId{2}};
    p.blocks[3].phis = {{{40, {}}, merged, {{BlockId{1}, left}, {BlockId{2}, right}}}};
    p.blocks[3].operations = {OperationId{2}}; p.blocks[3].terminator = Return{};
    success(run(p, f), "Equivalent phi handle provenance resolves");
    p.operations[1].inputs[0] = ConstantWord{3, 32};
    rejected(run(p, f), DiagnosticCode::unresolved_resource, "Different phi handle provenance is rejected");

    p = empty(Stage::vertex);
    IoInfo input; input.type = {ScalarType::f32, 1}; input.location = 5;
    add(p, {{8, {}}, Opcode::load_io, Effects::read_io, {Domain::floating, ScalarType::f32, Rounding::nearest_even}, {}, {}, input});
    auto unformatted = run(p, f);
    if (const auto* legal = success(unformatted, "Vertex input without a format fact derives its class")) {
        check(legal->requirements.io.size() == 1 && legal->requirements.io[0].vertex_class == VertexAttributeClass::float_class,
            "Unused vertex input defaults to the float class");
        check(!legal->requirements.io[0].vertex_format.has_value(),
            "No vertex format is invented for a fact-less input");
        check(std::none_of(legal->ir.dependencies.begin(), legal->ir.dependencies.end(),
                [](const auto& d) { return d.key.kind == FactKind::vertex_attribute_format; }),
            "Fact-less vertex inputs record no format dependency");
        success(plan_resources(structured(*legal), profile()), "Class-based vertex input plans");
    }
    f.set(FactQuery<FactKind::vertex_attribute_format>{5}, AttributeFormat{VertexFormat::f32, 1});
    auto result = run(p, f);
    if (const auto* legal = success(result, "Vertex IO format resolves")) {
        check(legal->requirements.io.size() == 1 && legal->requirements.io[0].vertex_format.has_value(), "Vertex format is reflected");
        check(!legal->requirements.io[0].vertex_stream && legal->requirements.io[0].byte_stride == 0, "Vertex layout is never invented");
        success(plan_resources(structured(*legal), profile()), "Native vertex attribute reflection plans");
    }
    f.set(FactQuery<FactKind::vertex_attribute_format>{5}, AttributeFormat{VertexFormat::u32, 1});
    rejected(run(p, f), DiagnosticCode::unsupported_modifier, "Integer vertex format is not silently treated as float");
    p.stage = Stage::fragment;
    rejected(run(p, f), DiagnosticCode::missing_fact, "Fragment interpolation fact is required per component");
    f.set(fact_query::interpolation(5, 0), Interpolation{});
    success(run(p, f), "Fragment interpolation is consumed");
}
void memory_and_limits() {
    auto f = facts();
    auto p = empty();
    const auto first = copy(p, ConstantWord{4, 252});
    const auto second = copy(p, ConstantWord{4, 12});
    p.functions[0].results = {ValueType::word32, ValueType::word32};
    p.blocks[0].terminator = Return{{first, second}};
    auto result = run(p, f);
    if (const auto* legal = success(result, "CB word loads legalize")) {
        check(legal->requirements.uses.size() == 2, "CB loads share one resource plus one support table");
        const auto& b = std::get<BufferUse>(legal->requirements.uses[0]);
        check(b.size.bytes == 256 && b.size.kind == SizeKind::constant_buffer_window &&
            b.size.constant_buffer_policy == ConstantBufferPolicy::proven_zero_words_v2,
            "CB window is the maximum normalized word end");
        const auto& support = std::get<BufferUse>(legal->requirements.uses[1]);
        check(std::get<SupportBufferSource>(support.source).schema == constant_buffer_support_schema &&
            support.size.bytes == constant_buffer_support_bytes, "CB legalization appends the versioned size table");
        success(validate(legal->ir, IrForm::legal, Phase::legalize), "Explicit loads form valid legal SSA");
        auto mutated = structured(*legal);
        std::get<MemoryInfo>(mutated.ir.operations[0].info).data = {InputIndex{99}};
        rejected(plan_resources(mutated, profile()), DiagnosticCode::invalid_ir, "Planner checks memory role indices");
        mutated = structured(*legal);
        std::get<MemoryInfo>(mutated.ir.operations[0].info).element_bytes = 0;
        rejected(plan_resources(mutated, profile()), DiagnosticCode::invalid_ir, "Planner rejects zero memory width");
    }
    p.operations[0].inputs[0] = ConstantWord{4, 256};
    auto static_result = run(p, f);
    if (const auto* legal = success(static_result, "Static CB access does not specialize on a draw binding")) {
        const auto* buffer = constant_buffer(legal->requirements, 4);
        check(buffer && buffer->size.bytes == 260, "Static CB window retains its normalized accessed extent");
    }
    p = empty(); const auto demanded = copy(p, ConstantWord{0, 0});
    p.functions[0].results = {ValueType::word32}; p.blocks[0].terminator = Return{{demanded}};
    p.limits.values = 1;
    rejected(run(p, f), DiagnosticCode::limit_exceeded, "Load insertion respects value limit");
    p = sampled(); p.limits.resources = 1;
    rejected(run(p, f), DiagnosticCode::limit_exceeded, "Sampler counts against resource limit");
    p = sampled(); p.limits.analysis_steps = 0;
    rejected(run(p, f), DiagnosticCode::limit_exceeded, "Zero work budget fails");
    p = empty();
    MemoryInfo m; m.space = AddressSpace::storage; m.resource = ResourceId{0};
    add(p, {{8, {}}, Opcode::load, Effects::read_memory, {}, {}, {}, m});
    rejected(run(p, f), DiagnosticCode::invalid_ir, "SSA storage rejects a preallocated ResourceId");
    p = empty(); m = {}; m.constant_buffer_slot = 2; m.dynamic_byte_offset = InputIndex{0};
    auto index = copy(p, word(16));
    add(p, {{8, {}}, Opcode::load, Effects::read_memory, {}, {index}, {}, m});
    auto dynamic = run(p, f);
    if (const auto* legal = success(dynamic, "Bound dynamic CB offset resolves")) {
        const auto& b = std::get<BufferUse>(legal->requirements.uses[0]);
        check(b.size.kind == SizeKind::constant_buffer_window && b.size.bytes == 20,
            "Bounded dynamic CB reflects its normalized accessed window");
        auto mutated = structured(*legal);
        mutated.ir.operations[0].inputs[0] = word(256);
        rejected(plan_resources(mutated, profile()), DiagnosticCode::invalid_ir, "Planner rechecks dynamic CB extent");
    }
    p.operations[0].inputs[0] = word(256);
    auto large_result = run(p, f);
    if (const auto* legal = success(large_result, "Large proven CB offset remains runtime guarded")) {
        const auto* buffer = constant_buffer(legal->requirements, 2);
        check(buffer && buffer->size.bytes == 260, "Large proven CB offset does not use captured binding size");
    }

    p = empty();
    ValueId unknown{static_cast<std::uint32_t>(p.values.size())};
    p.values.push_back({ValueType::word32});
    p.functions[0].parameters.push_back(unknown);
    m = {}; m.constant_buffer_slot = 2; m.dynamic_byte_offset = InputIndex{0};
    add(p, {{8, {}}, Opcode::load, Effects::read_memory, {}, {unknown}, {}, m});
    auto unknown_result = run(p, f);
    if (const auto* legal = success(unknown_result, "Unknown dynamic CB offset uses a guarded window")) {
        const auto load = std::find_if(legal->ir.operations.begin(), legal->ir.operations.end(), [](const auto& op) {
            return op.opcode == Opcode::load;
        });
        const auto* buffer = constant_buffer(legal->requirements, 2);
        check(load != legal->ir.operations.end() &&
            !std::get<MemoryInfo>(load->info).dynamic_byte_offset_upper_bound && buffer &&
            buffer->size.bytes == constant_buffer_window_bytes,
            "Unknown dynamic CB offset retains no invented bound and uses the full window");
        check(buffer && buffer->size.constant_buffer_policy == ConstantBufferPolicy::padded_zero_words_v2,
            "Unknown offsets select the padded contract at translation time");
    }
    const auto late_static = copy(p, ConstantWord{2, 8});
    const auto other_static = copy(p, ConstantWord{3, 8});
    p.functions[0].results = {ValueType::word32, ValueType::word32};
    p.blocks[0].terminator = Return{{late_static, other_static}};
    const auto mixed_result = run(p, f);
    if (const auto* mixed = success(mixed_result, "Mixed bounded and unbounded windows legalize")) {
        const auto* padded = constant_buffer(mixed->requirements, 2);
        const auto* proven = constant_buffer(mixed->requirements, 3);
        check(padded && padded->size.constant_buffer_policy == ConstantBufferPolicy::padded_zero_words_v2 &&
              proven && proven->size.constant_buffer_policy == ConstantBufferPolicy::proven_zero_words_v2,
            "One unknown read keeps only its own buffer padded after later static loads");
        if (!padded || !proven || padded->size.constant_buffer_policy != ConstantBufferPolicy::padded_zero_words_v2 ||
            proven->size.constant_buffer_policy != ConstantBufferPolicy::proven_zero_words_v2) {
            for (const auto& use : mixed->requirements.uses) if (const auto* buffer = std::get_if<BufferUse>(&use))
                if (const auto* slot = std::get_if<ConstantBufferSource>(&buffer->source))
                    std::cerr << "  slot=" << unsigned(slot->slot) << " policy=" << unsigned(buffer->size.constant_buffer_policy)
                              << " bytes=" << buffer->size.bytes << '\n';
        }
    }

    p = empty();
    const auto ordinary = copy(p, ConstantWord{2, 4});
    MemoryInfo boundary; boundary.constant_buffer_slot = 2; boundary.byte_offset = 65532;
    boundary.elements = 2;
    const ValueId low{static_cast<std::uint32_t>(p.values.size())}; p.values.push_back({ValueType::word32});
    const ValueId high{static_cast<std::uint32_t>(p.values.size())}; p.values.push_back({ValueType::word32});
    p.blocks[0].operations.push_back(OperationId{static_cast<std::uint32_t>(p.operations.size())});
    p.operations.push_back({{8,{}},Opcode::load,Effects::read_memory,{}, {},{low,high},boundary});
    p.functions[0].results = {ValueType::word32,ValueType::word32,ValueType::word32};
    p.blocks[0].terminator = Return{{ordinary,low,high}};
    const auto mixed_boundary = run(p,f);
    if (const auto* legal = success(mixed_boundary, "Mixed ordinary and crossing reads legalize")) {
        const auto* buffer = constant_buffer(legal->requirements,2);
        check(buffer && buffer->size.bytes == constant_buffer_window_bytes &&
              buffer->size.constant_buffer_policy == ConstantBufferPolicy::padded_zero_words_v2,
            "One crossing vector component keeps the whole shared buffer padded");
    }

    p = empty();
    MemoryInfo bound_source; bound_source.constant_buffer_slot = 1; bound_source.byte_offset = 6;
    auto bound_value = add(p, {{8, {}}, Opcode::load, Effects::read_memory, {}, {}, {}, bound_source});
    m = {}; m.constant_buffer_slot = 2; m.dynamic_byte_offset = InputIndex{0};
    add(p, {{8, {}}, Opcode::load, Effects::read_memory, {}, {bound_value}, {}, m});
    auto optional = run(p, f);
    if (const auto* legal = success(optional, "Unknown optional CB word does not reject a guarded offset")) {
        const auto* buffer = constant_buffer(legal->requirements, 2);
        check(buffer && buffer->size.bytes == constant_buffer_window_bytes,
            "Unknown optional CB word leaves the dynamic access unbounded");
        check(std::none_of(legal->ir.dependencies.begin(), legal->ir.dependencies.end(), [](const auto& dependency) {
            return dependency.key.kind == FactKind::constant_buffer_word;
        }), "Unknown optional CB word is not a specialization dependency");
    }
    f.set(fact_query::constant_word(1, 4), std::uint32_t{8});
    auto known_result = run(p, f);
    if (const auto* legal = success(known_result, "Known aligned CB word bounds a guarded offset")) {
        const auto* buffer = constant_buffer(legal->requirements, 2);
        check(buffer && buffer->size.bytes == 12,
            "Known CB word narrows the dynamic read window using aligned provenance");
        check(std::any_of(legal->ir.dependencies.begin(), legal->ir.dependencies.end(), [](const auto& dependency) {
            return dependency.key == fact_query::constant_word(1, 4).key();
        }), "Known CB word remains a specialization dependency");
    }
}
void arrays_and_numeric() {
    auto f = facts();
    auto p = sampled();
    auto& t = std::get<TextureInfo>(p.operations[0].info);
    t.array_index = InputIndex{2}; p.operations[0].inputs.push_back(word(2));
    f.set(FactQuery<FactKind::texture_array_length>{2, 16}, std::uint32_t{4});
    auto result = run(p, f);
    if (const auto* legal = success(result, "Proven resource array index resolves")) {
        const auto& texture = std::get<TextureUse>(legal->requirements.uses[0]);
        check(texture.array_length == 4 && texture.dynamically_indexed, "Array extent and indexing are explicit");
        auto mutated = structured(*legal);
        mutated.ir.operations[0].inputs[2] = word(4);
        rejected(plan_resources(mutated, profile()), DiagnosticCode::unresolved_resource, "Planner rechecks texture array bound");
        auto plan = plan_resources(structured(*legal), profile());
        if (const auto* value = success(plan, "Array plans")) check(std::get<MetalRange>(value->resources[0].location).count == 4 && std::get<MetalRange>(value->resources[1].location).count == 4, "Full texture and sampler ranges allocated");
    }
    p.operations[0].inputs[2] = word(4);
    rejected(run(p, f), DiagnosticCode::unresolved_resource, "Array index equal to length fails");
    f.set(FactQuery<FactKind::texture_array_length>{2, 16}, std::uint32_t{0});
    rejected(run(p, f), DiagnosticCode::invalid_fact, "Zero array length fails");
    p = sampled(); auto original = p.operations[0].numeric;
    original.flush_inputs = true; original.flush_output = true; original.saturate = true; original.precise = true;
    p.operations[0].numeric = original;
    result = run(p, f);
    if (const auto* legal = success(result, "Numeric modes are retained")) check(legal->ir.operations[0].numeric == original, "No FTZ/saturation/precision loss");
    p = empty();
    const auto fp64 = add(p, {{8, {}}, Opcode::add, Effects::none, {Domain::floating, ScalarType::f64, Rounding::nearest_even},
        {LiteralBits{0, ValueType::word64}, LiteralBits{0, ValueType::word64}}, {}, {}});
    p.functions[0].results = {ValueType::word64}; p.blocks[0].terminator = Return{{fp64}};
    rejected(run(p, f), DiagnosticCode::unsupported_capability, "FP64 never narrows to FP32");
    p = sampled(); auto d = descriptor(); d.normalized_coordinates = false; f.set(fact_query::texture({2, 16}), d);
    rejected(run(p, f), DiagnosticCode::unsupported_modifier, "Unnormalized coordinates demand explicit lowering");
    p = sampled(); p.operations[0].inputs[0] = LiteralBits{1, ValueType::predicate};
    f = facts();
    rejected(run(p, f), DiagnosticCode::invalid_ir, "Predicate texture coordinate fails storage validation");
    p = sampled(); p.stage = Stage::vertex;
    f = facts();
    result = run(p, f);
    // Implicit-LOD sampling has no host derivatives outside the fragment
    // stage; legalization rewrites it to an explicit level, reusing
    // the bias operand when encoded, so a vertex implicit sample legalizes as
    // an explicit level sample with a literal level-zero input.
    if (const auto* legal = success(result, "Implicit derivatives in vertex stage become explicit")) {
        const auto& t = std::get<TextureInfo>(legal->ir.operations[0].info);
        const auto* level = t.level_or_bias ?
            std::get_if<LiteralBits>(&legal->ir.operations[0].inputs[t.level_or_bias->value]) : nullptr;
        check(t.mode == TextureMode::sample_level && level && level->bits == 0,
            "Vertex implicit sample rewrites to level zero");
    }
    p = sampled(); p.stage = Stage::vertex;
    auto& vertex_info = std::get<TextureInfo>(p.operations[0].info);
    vertex_info.mode = TextureMode::sample_bias;
    vertex_info.level_or_bias = InputIndex{2};
    p.operations[0].inputs = {word(0), word(0), word(1)};
    f = facts();
    result = run(p, f);
    if (const auto* legal = success(result, "Vertex bias sample legalizes")) {
        const auto& t = std::get<TextureInfo>(legal->ir.operations[0].info);
        const auto* level = t.level_or_bias ?
            std::get_if<LiteralBits>(&legal->ir.operations[0].inputs[t.level_or_bias->value]) : nullptr;
        check(t.mode == TextureMode::sample_level && level && level->bits == 1,
            "Vertex bias operand becomes the explicit level");
    }
}
void dependency_and_pass_contracts() {
    auto f = facts();
    auto p = sampled();
    p.dependencies.push_back({FactQuery<FactKind::negate_y>{}.key(), true, {OriginKind::derived, "prior-phase"}});
    FactReader reader(f);
    auto result = legalize({p}, reader, profile());
    if (const auto* legal = success(result, "Fresh reader preserves previous dependencies")) check(legal->ir.dependencies.size() == 2, "Canonical dependency union retained");
    p.dependencies.push_back({fact_query::texture({2, 16}).key(), descriptor(), {OriginKind::derived, "changed-origin"}});
    std::sort(p.dependencies.begin(), p.dependencies.end(), [](const auto& a, const auto& b) { return a.key < b.key; });
    rejected(legalize({p}, reader, profile()), DiagnosticCode::conflicting_fact, "Evidence-origin mismatch fails across phase");
    auto bad_profile = profile(); bad_profile.identity.clear();
    rejected(legalize({sampled()}, reader, bad_profile), DiagnosticCode::unsupported_capability, "Empty profile fails");
#if !__has_include("pipeline/passes/passes.h")
    reject_pass = true;
    rejected(run(sampled(), f), DiagnosticCode::unsupported_modifier, "Mandatory pass error propagates");
    reject_pass = false;
#else
    p = sampled();
    MemoryInfo barrier; barrier.space = AddressSpace::local;
    p.blocks[0].operations.push_back(OperationId{static_cast<std::uint32_t>(p.operations.size())});
    p.operations.push_back({{40, {}}, Opcode::memory_barrier, Effects::barrier, {}, {}, {}, barrier});
    rejected(run(p, f), DiagnosticCode::unsupported_capability, "Real mandatory pass error propagates");
#endif
}
void planning() {
    StructuredProgram p; p.ir = empty(Stage::vertex); p.profile = profile();
    auto buffer = [](BufferSource source) -> ResourceUse { return BufferUse{std::move(source), Access::read, {SizeKind::full_binding, 256, 4}}; };
    auto cb = [](std::uint8_t slot) -> ResourceUse { return BufferUse{ConstantBufferSource{slot}, Access::read,
        {SizeKind::constant_buffer_window, 256, 4, ConstantBufferPolicy::padded_zero_words_v2}}; };
    auto support = []() -> ResourceUse { return BufferUse{SupportBufferSource{std::string(constant_buffer_support_schema),
        constant_buffer_support_version}, Access::read, {SizeKind::accessed_extent, constant_buffer_support_bytes, 4}}; };
    p.requirements.uses = {cb(7), buffer(VertexStreamSource{9}), support(), cb(1), buffer(VertexStreamSource{2})};
    IoUse io; io.semantic = IoSemantic::generic; io.type = {ScalarType::f32, 3}; io.mask = 7;
    io.vertex_format = AttributeFormat{VertexFormat::f32, 3}; io.vertex_stream = 2; io.byte_stride = 12;
    p.requirements.io.push_back(io); io.vertex_stream = 9; io.location = 1; p.requirements.io.push_back(io);
    auto result = plan_resources(p, profile());
    if (const auto* plan = success(result, "Stage-local buffer allocator plans")) {
        check(plan->reflection_version == 3, "Resource reflection version is three");
        check(std::get<MetalRange>(plan->resources[4].location).first == 18 && std::get<MetalRange>(plan->resources[1].location).first == 19, "Streams follow reflected IO ordinal, not guest slot");
        check(std::get<MetalRange>(plan->resources[3].location).first == 0 && std::get<MetalRange>(plan->resources[0].location).first == 1 && std::get<MetalRange>(plan->resources[2].location).first == 2, "CBs and support share allocator");
        const auto& planned_support = std::get<BufferUse>(plan->resources[2].use);
        check(std::get<SupportBufferSource>(planned_support.source).version == constant_buffer_support_version &&
            planned_support.size.bytes == constant_buffer_support_bytes && std::get<MetalRange>(plan->resources[2].location).space == MetalNamespace::buffer,
            "Plan retains support schema, size, version, and buffer index");
        for (std::uint32_t id = 0; id < plan->resources.size(); ++id) check(plan->resources[id].resource.value == id, "ResourceId survives allocation sorting");
    }
    auto changed = profile(); ++changed.version;
    rejected(plan_resources(p, changed), DiagnosticCode::unsupported_capability, "Planner rejects profile mismatch");
    p.profile.buffer_slots = 19;
    rejected(plan_resources(p, p.profile), DiagnosticCode::limit_exceeded, "Vertex stream capacity checked");
    p.profile = profile(); p.requirements.io[0].byte_stride = 4;
    rejected(plan_resources(p, profile()), DiagnosticCode::invalid_ir, "Invalid stream stride rejected");
    p = {}; p.ir = empty(Stage::vertex); p.profile = profile();
    for (std::uint8_t slot = 0; slot < constant_buffer_slot_count; ++slot) p.requirements.uses.push_back(cb(slot));
    p.requirements.uses.push_back(support());
    for (std::uint16_t slot = 0; slot < 13; ++slot) p.requirements.uses.push_back(buffer(VertexStreamSource{slot}));
    rejected(plan_resources(p, profile()), DiagnosticCode::limit_exceeded,
        "Eighteen CBs, support, and thirteen streams diagnose the concrete 31-slot collision");
    p.requirements.uses.pop_back();
    success(plan_resources(p, profile()), "Eighteen CBs, support, and twelve streams fill 31 buffer slots exactly");
    p.requirements.uses.erase(p.requirements.uses.begin() + constant_buffer_slot_count);
    rejected(plan_resources(p, profile()), DiagnosticCode::invalid_ir, "CB resources require exactly one size table");
    p = {}; p.ir = empty(); p.profile = profile();
    p.requirements.uses = {TextureUse{ResourceTextureKind::sampled, false, false, {2, 16}, descriptor(), Access::read, 33, false, {TextureMode::sample_level}}};
    success(plan_resources(p, profile()), "Texture array above direct capacity uses an argument buffer");
    std::get<TextureUse>(p.requirements.uses[0]).array_length = UINT32_MAX;
    rejected(plan_resources(p, profile()), DiagnosticCode::limit_exceeded, "Texture range integer overflow rejected");
    p.requirements.uses = {SamplerUse{{2, 16}, false, true, 17}};
    success(plan_resources(p, profile()), "Sampler overflow uses an argument buffer");
    auto f = facts(); auto legal_result = run(sampled(), f);
    if (const auto* legal = success(legal_result, "Resource cross-check fixture legalizes")) {
        p = structured(*legal); std::get<TextureInfo>(p.ir.operations[0].info).resource = ResourceId{99};
        rejected(plan_resources(p, profile()), DiagnosticCode::invalid_ir, "Out-of-range resource ID rejected");
        p = structured(*legal); std::get<TextureInfo>(p.ir.operations[0].info).resource = ResourceId{1};
        rejected(plan_resources(p, profile()), DiagnosticCode::invalid_ir, "Wrong resource variant rejected");
        p = structured(*legal); std::get<TextureInfo>(p.ir.operations[0].info).coordinates[0] = InputIndex{99};
        rejected(plan_resources(p, profile()), DiagnosticCode::invalid_ir, "Planner checks role index bounds");
    }
}
void argument_buffer_planning() {
    StructuredProgram p; p.ir = empty(); p.profile = profile();
    p.profile.texture_slots = 128; p.profile.sampler_slots = 16;
    auto texture = [](std::uint32_t key, std::uint32_t count = 1) -> ResourceUse {
        return TextureUse{ResourceTextureKind::sampled, false, false, {2, key}, descriptor(),
            Access::read, count, false, {TextureMode::sample_level}};
    };
    for (const bool textures : {false, true}) {
        const unsigned capacity = textures ? 128 : 16;
        for (const unsigned count : {capacity, capacity + 1}) {
            p.requirements.uses.clear();
            for (unsigned i = 0; i < count; ++i)
                p.requirements.uses.push_back(textures ? texture(4 * (count - i)) :
                    ResourceUse{SamplerUse{{2, 4 * (count - i)}, false, true, 1}});
            // The other namespace also moves when either threshold is exceeded.
            p.requirements.uses.push_back(textures ? ResourceUse{SamplerUse{{2, 0}, false, true, 1}} : texture(0));
            const auto result = plan_resources(p, p.profile);
            if (const auto* plan = success(result, "Direct/indirect resource threshold plans")) {
                check(plan->argument_buffer.has_value() == (count > capacity), "Exact capacity stays direct; one extra is indirect");
                for (const auto& resource : plan->resources)
                    check(std::holds_alternative<ArgumentBufferMember>(resource.location) == (count > capacity),
                        "All texture/sampler locations switch together");
                if (plan->argument_buffer) {
                    const auto& layout = *plan->argument_buffer;
                    check(layout.abi_version == 1 && layout.alignment == 8 && layout.buffer_index == 0 &&
                        layout.byte_extent == (count + 1) * 8, "Container has the exact checked eight-byte layout");
                    const auto first = std::get<ArgumentBufferMember>(plan->resources[count - 1].location);
                    check(first.byte_offset == (textures ? 0u : 8u) && first.count == 1 && first.stride == 8,
                        "Member order follows provenance sorting rather than input order");
                }
            }
        }
    }
    p.requirements.uses = {SamplerUse{{2, 16}, false, true, 17}, texture(16, 129)};
    auto result = plan_resources(p, p.profile);
    if (const auto* plan = success(result, "Resource arrays retain their full extent")) {
        const auto sampler = std::get<ArgumentBufferMember>(plan->resources[0].location);
        const auto texture_member = std::get<ArgumentBufferMember>(plan->resources[1].location);
        check(texture_member.byte_offset == 0 && texture_member.count == 129 &&
            sampler.byte_offset == 129 * 8 && sampler.count == 17 && plan->argument_buffer->byte_extent == 146 * 8,
            "Arrays occupy consecutive handles in texture-before-sampler order");
    }
    std::get<TextureUse>(p.requirements.uses[1]).array_length = UINT32_MAX / 8;
    rejected(plan_resources(p, p.profile), DiagnosticCode::limit_exceeded, "Combined array extent overflow is rejected");
    p.requirements.uses = {texture(16, 129)};
    std::get<TextureUse>(p.requirements.uses[0]).array_length = UINT32_MAX / 8;
    result = plan_resources(p, p.profile);
    if (const auto* plan = success(result, "Maximum representable layout needs no per-element allocation"))
        check(plan->argument_buffer->byte_extent == UINT32_MAX - 7u, "Largest eight-byte extent is exact");
    p.ir = empty(Stage::vertex);
    p.requirements.uses = {SamplerUse{{2, 16}, false, true, 17}};
    for (unsigned i = 0; i < 18; ++i)
        p.requirements.uses.push_back(BufferUse{StorageBufferSource{{2, 4 * i}}, Access::read, {SizeKind::full_binding, 16, 4}});
    p.requirements.uses.push_back(BufferUse{VertexStreamSource{7}, Access::read, {SizeKind::full_binding, 16, 4}});
    result = plan_resources(p, p.profile);
    if (const auto* plan = success(result, "Container shares stream-aware buffer allocator"))
        check(plan->argument_buffer->buffer_index == 19 &&
            std::get<MetalRange>(plan->resources.back().location).first == 18,
            "Container skips the reserved vertex stream slot");
    // CLI native vertex inputs do not create an explicit stream resource.
    auto native = p;
    native.requirements.uses.pop_back();
    IoUse input; input.semantic = IoSemantic::generic; input.type = {ScalarType::f32, 3};
    input.mask = 7; input.vertex_format = AttributeFormat{VertexFormat::f32, 3};
    native.requirements.io.push_back(input);
    const auto native_result = plan_resources(native, native.profile);
    if (const auto* plan = success(native_result, "Native vertex format reserves its runtime stream"))
        check(plan->argument_buffer->buffer_index == 19,
            "Container skips native stream 18 without an explicit buffer requirement");
    for (unsigned i = 18; i < 30; ++i)
        p.requirements.uses.push_back(BufferUse{StorageBufferSource{{2, 4 * i}}, Access::read, {SizeKind::full_binding, 16, 4}});
    rejected(plan_resources(p, p.profile), DiagnosticCode::limit_exceeded, "Container cannot overlap a full buffer namespace");
}
}

#if !__has_include("pipeline/passes/passes.h")
namespace armigo::shader::pipeline::passes {
// These stand-ins check orchestration only; they are never linked into production.
Status cleanup(Program&) {
    ++cleanup_calls;
    if (reject_pass) return diagnose(Phase::legalize, DiagnosticCode::unsupported_modifier, {}, "Synthetic mandatory pass rejection");
    return std::monostate{};
}
Status legalize_nontexture(Program&, FactReader&, const MetalProfile&) { ++nontexture_calls; return std::monostate{}; }
}
#endif
// Per-texel gather offset sets (TLD4 Ptp): eight literals lower as four
// gather-offset pairs; everything else keeps its structured rejection.
void per_texel_gather_offsets() {
    auto f = facts();
    Program p = empty();
    auto op = sample();
    auto& t = std::get<TextureInfo>(op.info);
    t.mode = TextureMode::gather;
    for (unsigned texel = 0; texel < 4; ++texel) {
        t.offsets.push_back(InputIndex{static_cast<std::uint16_t>(2 + 2 * texel)});
        t.offsets.push_back(InputIndex{static_cast<std::uint16_t>(3 + 2 * texel)});
        op.inputs.push_back(word(static_cast<std::uint32_t>(texel - 2)));
        op.inputs.push_back(word(static_cast<std::uint32_t>(texel == 3 ? 7 : -8)));
    }
    add(p, op);
    auto result = run(p, f);
    if (const auto* legal = success(result, "Per-texel gather offsets resolve")) {
        const auto& use = std::get<TextureUse>(legal->requirements.uses[0]);
        check(use.modes == std::vector<TextureMode>{TextureMode::gather},
            "Per-texel gather records its mode");
    }
    // A depth-comparison gather has no per-texel offset lowering.
    auto shadow = empty();
    auto shadow_op = sample();
    auto& shadow_t = std::get<TextureInfo>(shadow_op.info);
    shadow_t.mode = TextureMode::gather;
    shadow_t.compare = InputIndex{2};
    shadow_op.inputs.push_back(word(0));
    for (unsigned texel = 0; texel < 4; ++texel) {
        shadow_t.offsets.push_back(InputIndex{static_cast<std::uint16_t>(3 + texel)});
        shadow_op.inputs.push_back(word(0));
    }
    auto shadow_descriptor = descriptor();
    shadow_descriptor.format = ComponentFormat::depth;
    shadow_descriptor.comparison = true;
    f.set(fact_query::texture({2, 16}), shadow_descriptor);
    add(shadow, shadow_op);
    rejected(run(shadow, f), DiagnosticCode::unsupported_modifier, "Shadow per-texel gather has no lowering");
    // Neither one-offset-per-axis nor a per-texel set.
    f.set(fact_query::texture({2, 16}), descriptor());
    auto short_set = empty();
    auto short_op = sample();
    auto& short_t = std::get<TextureInfo>(short_op.info);
    short_t.mode = TextureMode::gather;
    for (unsigned n = 0; n < 3; ++n) {
        short_t.offsets.push_back(InputIndex{static_cast<std::uint16_t>(2 + n)});
        short_op.inputs.push_back(word(0));
    }
    add(short_set, short_op);
    rejected(run(short_set, f), DiagnosticCode::unsupported_modifier, "Incomplete offset set has no lowering");
    // The literal range stays the Metal limit even inside a full set.
    auto out_of_range = empty();
    auto range_op = sample();
    auto& range_t = std::get<TextureInfo>(range_op.info);
    range_t.mode = TextureMode::gather;
    for (unsigned texel = 0; texel < 4; ++texel) {
        range_t.offsets.push_back(InputIndex{static_cast<std::uint16_t>(2 + 2 * texel)});
        range_t.offsets.push_back(InputIndex{static_cast<std::uint16_t>(3 + 2 * texel)});
        range_op.inputs.push_back(word(0));
        range_op.inputs.push_back(word(8));
    }
    add(out_of_range, range_op);
    rejected(run(out_of_range, f), DiagnosticCode::unsupported_modifier, "Offset literals stay inside the Metal range");
}

// Texture and sampler handles resolve independently. A constant-buffer key
// or a handle program is never rewritten into a pool access. Pooled samplers
// of one stage share one sampler pool.
void independent_pool_resolution() {
    auto pools = profile();
    pools.texture_pools = true;
    auto f = facts();
    auto handle_descriptor = descriptor();
    f.set(fact_query::texture({handle_program_slot, 0}), handle_descriptor);
    const auto says = [](const Diagnostic* error, std::string_view text) {
        const auto matches = error && error->message.find(text) != std::string::npos;
        if (error && !matches) std::cerr << "  message: " << error->message << '\n';
        return matches;
    };
    auto sample_at = [](HandleSource texture, std::optional<HandleSource> sampler, std::vector<Operand> inputs,
                        TextureDescriptor desc) {
        TextureInfo info;
        info.texture = texture;
        info.sampler = std::move(sampler);
        info.descriptor = desc;
        const auto coordinates = static_cast<std::uint16_t>(inputs.size() - 2);
        info.coordinates = {InputIndex{coordinates}, InputIndex{static_cast<std::uint16_t>(coordinates + 1)}};
        Operation op{{40, InstructionName::tex}, Opcode::texture_sample, Effects::texture | Effects::derivative,
            {Domain::floating, ScalarType::f32, Rounding::nearest_even}, std::move(inputs), {}, info};
        return op;
    };
    auto one = [&](Operation op, const MetalProfile& metal) {
        auto p = empty();
        add(p, std::move(op));
        return run(std::move(p), f, metal);
    };
    const auto sampled = [&](const LegalProgram& legal) -> const TextureInfo* {
        const TextureInfo* found = nullptr;
        for (const auto& op : legal.ir.operations) if (op.opcode == Opcode::texture_sample) {
            check(found == nullptr, "one sample in the stage");
            found = &std::get<TextureInfo>(op.info);
        }
        check(found != nullptr, "the sample survives legalization");
        return found;
    };
    const auto sampler_use = [](const LegalProgram& legal, ResourceId id) -> const SamplerUse* {
        return id.value < legal.requirements.uses.size() ? std::get_if<SamplerUse>(&legal.requirements.uses[id.value]) : nullptr;
    };
    const auto texture_use = [](const LegalProgram& legal, ResourceId id) -> const TextureUse* {
        return id.value < legal.requirements.uses.size() ? std::get_if<TextureUse>(&legal.requirements.uses[id.value]) : nullptr;
    };
    auto plain = descriptor();
    // Pool texture, constant-buffer sampler key.
    {
        auto result = one(sample_at(DynamicHandle{InputIndex{0}}, TextureKey{2, 16}, {word(1), word(0), word(0)}, plain), pools);
        if (const auto* legal = success(result, "Pool texture with a constant-buffer sampler")) {
            const auto* t = sampled(*legal);
            check(t && std::get<TextureKey>(t->texture) == TextureKey{texture_pool_slot, 0} && t->pool_texture &&
                  t->sampler && std::get<TextureKey>(*t->sampler) == TextureKey{2, 16} && !t->pool_sampler,
                  "the texture is pooled and the sampler keeps its key");
            if (t && t->sampler_resource && t->resource) {
                const auto* sampler = sampler_use(*legal, *t->sampler_resource);
                const auto* texture = texture_use(*legal, *t->resource);
                check(sampler && !sampler->pool && sampler->key == TextureKey{2, 16} && sampler->comparison == plain.comparison &&
                      sampler->normalized_coordinates == plain.normalized_coordinates, "keyed sampler binding");
                check(texture && texture->pool && texture->descriptor == plain, "texture pool binding");
            }
            success(plan_resources(structured(*legal), legal->profile), "Pool texture with a keyed sampler plans");
        }
    }
    // Pool texture, sampler proven by a constant-buffer word.
    {
        auto result = one(sample_at(DynamicHandle{InputIndex{0}}, DynamicHandle{InputIndex{1}},
                                    {word(1), ConstantWord{2, 16}, word(0), word(0)}, plain), pools);
        if (const auto* legal = success(result, "Pool texture with a constant-buffer sampler word")) {
            const auto* t = sampled(*legal);
            check(t && t->pool_texture && !t->pool_sampler && t->sampler &&
                  std::get<TextureKey>(*t->sampler) == TextureKey{2, 16}, "a constant-buffer sampler word stays keyed");
        }
    }
    // Pool texture, handle-program sampler.
    {
        auto result = one(sample_at(DynamicHandle{InputIndex{0}}, TextureKey{handle_program_slot, 0},
                                    {word(1), word(0), word(0)}, plain), pools);
        if (const auto* legal = success(result, "Pool texture with a handle-program sampler")) {
            const auto* t = sampled(*legal);
            check(t && t->pool_texture && !t->pool_sampler && t->sampler &&
                  std::get<TextureKey>(*t->sampler) == TextureKey{handle_program_slot, 0},
                  "a handle-program sampler stays keyed");
        }
    }
    // Constant-buffer texture, unresolved separate sampler: the sampler pools.
    {
        auto result = one(sample_at(TextureKey{2, 16}, DynamicHandle{InputIndex{0}},
                                    {word(0x000abcde), word(0), word(0)}, plain), pools);
        if (const auto* legal = success(result, "Constant-buffer texture with a pool sampler")) {
            const auto* t = sampled(*legal);
            check(t && !t->pool_texture && std::get<TextureKey>(t->texture) == TextureKey{2, 16} && t->pool_sampler &&
                  t->sampler && std::get<TextureKey>(*t->sampler) == TextureKey{texture_pool_slot, 0},
                  "the texture stays keyed and the sampler is pooled");
            if (t && t->sampler_resource) {
                const auto* sampler = sampler_use(*legal, *t->sampler_resource);
                check(sampler && sampler->pool && !sampler->comparison && sampler->normalized_coordinates &&
                      sampler->key == TextureKey{texture_pool_slot, 0}, "the stage sampler pool is untyped by comparison");
            }
            success(plan_resources(structured(*legal), legal->profile), "A pool sampler without a pool texture plans");
        }
    }
    // The same split when the texture word is a constant-buffer load, not a key literal.
    {
        auto result = one(sample_at(DynamicHandle{InputIndex{0}}, DynamicHandle{InputIndex{1}},
                                    {ConstantWord{2, 16}, word(0x000abcde), word(0), word(0)}, plain), pools);
        if (const auto* legal = success(result, "Constant-buffer texture word with a pool sampler")) {
            const auto* t = sampled(*legal);
            check(t && !t->pool_texture && std::get<TextureKey>(t->texture) == TextureKey{2, 16} && t->pool_sampler,
                  "a constant-buffer texture word is never a pool");
        }
    }
    // Handle-program texture, pool sampler.
    {
        auto result = one(sample_at(TextureKey{handle_program_slot, 0}, DynamicHandle{InputIndex{0}},
                                    {word(0x000abcde), word(0), word(0)}, plain), pools);
        if (const auto* legal = success(result, "Handle-program texture with a pool sampler")) {
            const auto* t = sampled(*legal);
            check(t && !t->pool_texture && t->pool_sampler &&
                  std::get<TextureKey>(t->texture) == TextureKey{handle_program_slot, 0},
                  "a handle program is never a pool");
        }
    }
    // Both halves already resolved: pools stay unused.
    {
        auto result = one(sample_at(TextureKey{2, 16}, TextureKey{3, 32}, {word(0), word(0)}, plain), pools);
        if (const auto* legal = success(result, "Resolved texture and sampler ignore pools")) {
            unsigned pooled = 0;
            for (const auto& use : legal->requirements.uses) {
                if (const auto* texture = std::get_if<TextureUse>(&use)) pooled += texture->pool;
                if (const auto* sampler = std::get_if<SamplerUse>(&use)) pooled += sampler->pool;
            }
            check(pooled == 0, "a constant-buffer key is never rewritten into a pool");
        }
    }
    // Ordinary and depth-comparison pooled samples share one sampler pool.
    {
        auto depth = descriptor();
        depth.format = ComponentFormat::depth;
        depth.comparison = true;
        auto p = empty();
        add(p, sample_at(DynamicHandle{InputIndex{0}}, std::nullopt, {word(1), word(0), word(0)}, plain));
        auto compared = sample_at(DynamicHandle{InputIndex{0}}, std::nullopt, {word(2), word(0), word(0), word(0)}, depth);
        auto& compared_info = std::get<TextureInfo>(compared.info);
        compared_info.coordinates = {InputIndex{1}, InputIndex{2}};
        compared_info.compare = InputIndex{3};
        add(p, std::move(compared));
        auto result = run(std::move(p), f, pools);
        if (const auto* legal = success(result, "Ordinary and depth-comparison samples share a sampler pool")) {
            unsigned texture_pools = 0, sampler_pools = 0;
            const SamplerUse* sampler = nullptr;
            for (const auto& use : legal->requirements.uses) {
                if (const auto* texture = std::get_if<TextureUse>(&use); texture && texture->pool) ++texture_pools;
                if (const auto* use_sampler = std::get_if<SamplerUse>(&use); use_sampler && use_sampler->pool) {
                    ++sampler_pools; sampler = use_sampler;
                }
            }
            check(texture_pools == 2 && sampler_pools == 1, "one sampler pool and one texture pool per element type");
            check(sampler && !sampler->comparison && sampler->normalized_coordinates &&
                  sampler->key == TextureKey{texture_pool_slot, 0} && sampler->array_length == 1,
                  "the shared sampler pool is not typed by comparison");
            std::optional<ResourceId> shared;
            unsigned samples = 0;
            for (const auto& op : legal->ir.operations) if (op.opcode == Opcode::texture_sample) {
                ++samples;
                const auto& t = std::get<TextureInfo>(op.info);
                check(t.pool_texture && t.pool_sampler && t.sampler_resource, "both samples read the pools");
                if (!shared) shared = t.sampler_resource;
                else check(shared == t.sampler_resource, "both samples bind the same sampler pool");
            }
            check(samples == 2, "both samples survive");
            success(plan_resources(structured(*legal), legal->profile), "One sampler pool plans");
        }
    }
    // Pools off keeps the previous refusals.
    {
        auto missing = one(sample_at(DynamicHandle{InputIndex{0}}, std::nullopt, {word(1), word(0), word(0)}, plain), profile());
        auto* error = rejected(missing, DiagnosticCode::unresolved_resource, "Pools off refuse an unresolved texture");
        check(says(error, "Bindless texture handle has no proven constant-buffer provenance"), "unresolved texture message");
        auto sampler = one(sample_at(TextureKey{2, 16}, DynamicHandle{InputIndex{0}}, {word(1), word(0), word(0)}, plain), profile());
        error = rejected(sampler, DiagnosticCode::unresolved_resource, "Pools off refuse an unresolved separate sampler");
        check(says(error, "Separate sampler handle has no unambiguous provenance"), "unresolved sampler message");
    }
    // A separate sampler that is itself a combined handle stays refused.
    {
        auto p = empty();
        auto tex = binary(p, Opcode::bit_and, ConstantWord{2, 16}, word(0x000fffff));
        auto sam = binary(p, Opcode::bit_and, word(0xfff00000), ConstantWord{3, 32});
        auto combined = binary(p, Opcode::bit_or, sam, tex);
        auto op = sample_at(DynamicHandle{InputIndex{0}}, DynamicHandle{InputIndex{1}},
                            {word(1), combined, word(0), word(0)}, plain);
        add(p, std::move(op));
        auto combined_result = run(std::move(p), f, pools);
        auto* error = rejected(combined_result, DiagnosticCode::unresolved_resource,
                               "A combined separate sampler stays refused");
        check(says(error, "Separate sampler handle has no unambiguous provenance"), "combined sampler message");
    }
    // Storage images, image atomics and buffer textures stay outside pool admission.
    {
        TextureInfo image;
        image.texture = DynamicHandle{InputIndex{0}};
        image.mode = TextureMode::fetch;
        image.descriptor = plain;
        image.coordinates = {InputIndex{1}, InputIndex{2}};
        auto load = one({{40, {}}, Opcode::image_load, Effects::read_memory | Effects::texture,
                         {Domain::floating, ScalarType::f32, Rounding::nearest_even}, {word(1), word(0), word(0)}, {}, image}, pools);
        auto* error = rejected(load, DiagnosticCode::unresolved_resource, "A pooled storage image is refused");
        check(says(error, "Bindless texture handle has no proven constant-buffer provenance"), "storage image message");
        image.atomic = AtomicKind::add;
        image.data = {InputIndex{3}};
        auto atomic = one({{40, {}}, Opcode::image_atomic,
                           Effects::atomic | Effects::read_memory | Effects::write_memory | Effects::texture,
                           {Domain::unsigned_integer, ScalarType::u32}, {word(1), word(0), word(0), word(1)}, {}, image}, pools);
        error = rejected(atomic, DiagnosticCode::unresolved_resource, "A pooled image atomic is refused");
        check(says(error, "Bindless texture handle has no proven constant-buffer provenance"), "image atomic message");
        auto buffer_desc = plain;
        buffer_desc.shape.dimension = TextureDimension::buffer;
        auto buffer = sample_at(DynamicHandle{InputIndex{0}}, std::nullopt, {word(1), word(0)}, buffer_desc);
        std::get<TextureInfo>(buffer.info).coordinates = {InputIndex{1}};
        auto buffer_result = one(std::move(buffer), pools);
        error = rejected(buffer_result, DiagnosticCode::missing_fact, "A pooled buffer texture is refused");
        check(says(error, "Pool texture has no admitted descriptor from its instructions"), "buffer texture message");
    }
}

void direct_storage_resolution() {
    auto p = empty();
    const auto low = copy(p, ConstantWord{0, 0x510});
    const auto high = copy(p, ConstantWord{0, 0x514});
    MemoryInfo memory;
    memory.space = AddressSpace::guest; memory.access = Access::read;
    memory.element_bytes = 4; memory.elements = 1; memory.alignment = 4;
    memory.scope = MemoryScope::workgroup;
    memory.dynamic_byte_offset = InputIndex{0}; memory.global_address_high = InputIndex{1};
    Operation op{{80, {}}, Opcode::load, Effects::read_memory, {}, {low, high}, {}, memory};
    const auto loaded = add(p, std::move(op));
    p.functions[0].results = {ValueType::word32}; p.blocks[0].terminator = Return{{loaded}};
    auto f = facts();
    auto unresolved = run(p, f);
    if (const auto* legal = success(unresolved, "Unknown driver binding keeps a guest-only access")) {
        check(std::none_of(legal->ir.operations.begin(), legal->ir.operations.end(), [](const auto& operation) {
            const auto* info = std::get_if<MemoryInfo>(&operation.info);
            return info && info->direct_storage.has_value();
        }), "Address provenance alone cannot invent a host buffer binding");
    }
    f.set(FactQuery<FactKind::bound_storage_descriptor>{0, 0x510}, true);
    auto resolved = run(p, f);
    if (const auto* legal = success(resolved, "Known driver binding resolves the descriptor-relative load")) {
        success(plan_resources(structured(*legal), profile()), "Direct storage plans without a pool table");
        check(std::none_of(legal->requirements.uses.begin(), legal->requirements.uses.end(), [](const auto& use) {
            const auto* buffer = std::get_if<BufferUse>(&use);
            const auto* support = buffer ? std::get_if<SupportBufferSource>(&buffer->source) : nullptr;
            return support && support->schema == guest_memory_schema;
        }), "A proven direct load needs no guest-pool table");
        const auto access = std::find_if(legal->ir.operations.begin(), legal->ir.operations.end(), [](const auto& operation) {
            const auto* info = std::get_if<MemoryInfo>(&operation.info);
            return info && info->direct_storage.has_value();
        });
        check(access != legal->ir.operations.end(), "Known driver descriptor is reflected on the load");
        if (access == legal->ir.operations.end()) return;
        const auto index = static_cast<std::size_t>(access - legal->ir.operations.begin());
        auto mutated = structured(*legal);
        std::erase_if(mutated.ir.dependencies, [](const auto& dependency) {
            return dependency.key.kind == FactKind::bound_storage_descriptor;
        });
        rejected(plan_resources(mutated, profile()), DiagnosticCode::invalid_ir, "Bound guest storage requires its driver fact");
        mutated = structured(*legal);
        std::get<MemoryInfo>(mutated.ir.operations[index].info).resource = ResourceId{9999};
        rejected(plan_resources(mutated, profile()), DiagnosticCode::invalid_ir, "Bound guest storage cannot name an invalid resource");
        mutated = structured(*legal);
        std::get<MemoryInfo>(mutated.ir.operations[index].info).direct_storage->handle_byte_offset += 16;
        rejected(plan_resources(mutated, profile()), DiagnosticCode::invalid_ir, "Bound guest storage rechecks descriptor identity");
        mutated = structured(*legal);
        std::get<MemoryInfo>(mutated.ir.operations[index].info).dynamic_byte_offset = InputIndex{9999};
        rejected(plan_resources(mutated, profile()), DiagnosticCode::invalid_ir, "Bound guest storage rechecks descriptor input roles");
    }
    for (const auto scope : {MemoryScope::workgroup, MemoryScope::device}) {
        auto native = p;
        auto& load = std::get<MemoryInfo>(native.operations.back().info);
        load.scope = scope;
        load.native_load_policy = NativeLoadPolicy::aligned_v1;
        auto native_result = run(std::move(native), f);
        if (const auto* legal = success(native_result, "Native aligned LDG can resolve to a driver storage binding")) {
            const auto access = std::find_if(legal->ir.operations.begin(), legal->ir.operations.end(), [](const auto& operation) {
                const auto* info = std::get_if<MemoryInfo>(&operation.info);
                return info && info->direct_storage.has_value();
            });
            check(access != legal->ir.operations.end(), "Native aligned LDG retains descriptor resolution");
            if (access != legal->ir.operations.end()) {
                const auto& info = std::get<MemoryInfo>(access->info);
                check(info.native_load_policy == NativeLoadPolicy::aligned_v1 && info.scope == scope,
                      "Descriptor resolution preserves native load policy and decoded scope");
            }
            success(plan_resources(structured(*legal), profile()), "Resolved native LDG policy passes independent planning");
        }
    }
}

int main() {
    for (auto test : {vertex_depth_dependency, basic_resolution, bindless_resolution, phi_and_io_resolution, memory_and_limits,
            arrays_and_numeric, per_texel_gather_offsets, dependency_and_pass_contracts, planning, argument_buffer_planning,
            independent_pool_resolution, direct_storage_resolution}) {
        try { test(); } catch (const std::exception& error) {
            ++failures; std::cerr << "Test threw: " << error.what() << '\n';
        }
    }
    if (failures) std::cerr << failures << " resource checks failed\n";
    else std::cout << "All pipeline resource checks passed\n";
    return failures ? 1 : 0;
}
