// ============================================================================
// Armigo Tests - Native Runtime Verification
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include "global_memory_fixture.h"

#include <iostream>

namespace {
using namespace armigo::shader::test::global_memory;

void check(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

void decoded_addresses(const Configuration& config) {
    Facts source(config);
    FactReader facts(source);
    const auto decoded = take(decode(image(config), facts, {}));
    unsigned memory_instructions{};
    for (const auto& block : decoded.blocks) {
        for (const auto& instruction : block.instructions) {
            if (instruction.name != InstructionName::ldg && instruction.name != InstructionName::stg) continue;
            ++memory_instructions;
            const bool store = instruction.name == InstructionName::stg;
            const unsigned base = store ? 2 : config.shared_table ? 4 : config.copy_pair ? 8 : 0;
            unsigned words{};
            for (const auto& access : instruction.registers) {
                if (access.role != RegisterRole::source_a) continue;
                check(access.access == Access::read && access.lane == words &&
                    access.reg.index == base + words,
                    "Decoded global address reads the consecutive low/high register pair");
                ++words;
            }
            check(words == (config.extended ? 2u : 1u),
                "Extended bit controls the decoded address register count");
            const auto displacement = static_cast<std::uint64_t>(static_cast<std::int64_t>(
                store ? config.store_displacement : config.load_displacement));
            check(take(decoded_field(instruction, DecodedField::memory_byte_offset)) == displacement,
                "Decoder sign-extends the global signed24 displacement");
        }
    }
    check(memory_instructions == 2, "Synthetic program decodes one global load and one store");
}

void accepts(const Configuration& config, const char* label) {
    decoded_addresses(config);
    const auto result = artifact(config);
    unsigned storage_count{};
    bool input{}, output{}, constants{}, support{};
    for (const auto& resource : result.reflection.resources) {
        const auto* buffer = std::get_if<BufferUse>(&resource.use);
        if (!buffer) continue;
        if (const auto* storage = std::get_if<StorageBufferSource>(&buffer->source)) {
            ++storage_count;
            if (storage->descriptor == TextureKey{descriptor_slot, input_descriptor_offset}) {
                check(buffer->access == Access::read, "LDG.E input reflects read access");
                input = true;
            }
            if (storage->descriptor == TextureKey{descriptor_slot, output_descriptor_offset}) {
                check(buffer->access == Access::write, "STG.E output reflects write access");
                output = true;
            }
        }
        if (const auto* cbuf = std::get_if<ConstantBufferSource>(&buffer->source))
            constants |= cbuf->slot == descriptor_slot;
        if (const auto* table = std::get_if<SupportBufferSource>(&buffer->source))
            support |= table->schema == constant_buffer_support_schema &&
                table->version == constant_buffer_support_version;
    }
    check(storage_count == 2 && input && output, "Descriptor provenance reflects two distinct storage resources");
    check(constants && support, "Runtime descriptors retain their constant-buffer bindings and size table");
    check(!result.source.empty(), "End-to-end shader emits MSL");
    if (config.extended) {
        check(result.source.find("const uint global_low") != std::string::npos &&
            result.source.find("const uint global_high") != std::string::npos &&
            result.source.find(" + uint(global_low") != std::string::npos,
            "MSL forms the 64-bit guest address in 32-bit halves with a carry");
        check(result.source.find("storage_base_low") != std::string::npos &&
            result.source.find(" == 0u && ") != std::string::npos &&
            result.source.find(" >= storage_base_high") != std::string::npos,
            "MSL computes a descriptor-relative offset and rejects addresses below its base");
        if (config.load_displacement < 0)
            check(result.source.find(" + " + std::to_string(static_cast<std::uint32_t>(config.load_displacement)) +
                "u;") != std::string::npos && result.source.find(" + 4294967295u") != std::string::npos,
                "MSL applies the signed immediate to both address halves");
    } else {
        check(result.source.find("const uint global_low") == std::string::npos &&
            result.source.find("const uint storage_offset") != std::string::npos,
            "Nonextended addressing preserves modulo-32-bit relative subtraction");
    }
    check(result.source.find("<= storage_size") != std::string::npos &&
        result.source.find(" ? ") != std::string::npos && result.source.find(" : 0u") != std::string::npos &&
        result.source.find("if (") != std::string::npos,
        "MSL guards runtime descriptor extents for both zero-returning loads and skipped stores");
    // Legalization proves the load's descriptor-relative address word aligned
    // unless a displacement breaks the alignment.
    const bool word = config.load_displacement % 4 == 0 && config.low_addend % 4 == 0;
    check((result.source.find(" / 4u] : ") != std::string::npos) == word &&
        (result.source.find(" + 1u) / 4u] >> uint(") == std::string::npos) == word,
        word ? "A 32-bit global load at a proven word-aligned address reads one word"
             : "A 32-bit global load without a word-alignment proof reads its bytes");
    for (const auto& dependency : result.reflection.dependencies)
        check(dependency.key.kind != FactKind::constant_buffer_word,
            "Descriptor identity must not require runtime address words as translation facts");
    std::cout << "PASS: " << label << '\n';
}

void rejects(Configuration config, const char* label, InstructionName instruction = InstructionName::ldg,
             DiagnosticCode code = DiagnosticCode::unresolved_resource) {
    const auto result = legalize_image(config);
    const auto* error = std::get_if<Diagnostic>(&result);
    check(error && error->phase == Phase::legalize &&
        error->code == code && error->pc.has_value() &&
        error->instruction == instruction, label);
    std::cout << "PASS: " << label << '\n';
}

void rejects_malformed_legal(bool excess_inputs) {
    auto program = take(legalize_image());
    bool changed = false;
    for (auto& operation : program.ir.operations) {
        auto* memory = std::get_if<MemoryInfo>(&operation.info);
        if (!memory || memory->space != AddressSpace::storage || operation.opcode != Opcode::load) continue;
        check(memory->storage_address.has_value(), "Legal storage load carries descriptor input roles");
        if (excess_inputs) {
            while (operation.inputs.size() < 20)
                operation.inputs.push_back(LiteralBits{0, ValueType::word32});
        } else {
            const auto& address = *memory->storage_address;
            operation.inputs[address.base_low.value] = operation.inputs[address.base_high.value];
        }
        changed = true;
        break;
    }
    check(changed, "Malformed fixture changes an actual legalized global load");
    auto structured = structure(std::move(program), {});
    std::optional<Diagnostic> error;
    if (const auto* diagnostic = std::get_if<Diagnostic>(&structured)) error = *diagnostic;
    else {
        auto planned = plan_resources(std::get<StructuredProgram>(structured), profile());
        if (const auto* diagnostic = std::get_if<Diagnostic>(&planned)) error = *diagnostic;
    }
    check(error && error->code == DiagnosticCode::invalid_ir &&
        (error->phase == Phase::structure || error->phase == Phase::plan_resources),
        excess_inputs ? "A legal global load with excess operands fails a phase boundary" :
            "Descriptor role loads must match the reflected descriptor words at a phase boundary");
}
} // namespace

int main() {
    try {
        accepts({}, "Direct runtime descriptor pair");
        Configuration config;
        config.copy_pair = true;
        accepts(config, "Copied descriptor pair retains provenance");
        config = {};
        config.load_displacement = 4;
        config.store_displacement = 4;
        accepts(config, "Positive signed displacements");
        config = {};
        config.load_displacement = 1;
        accepts(config, "A load off word alignment keeps byte extraction");
        config = {};
        config.low_addend = 8;
        config.load_displacement = -4;
        accepts(config, "Low address arithmetic and negative displacement");
        config = {};
        config.low_addend = 4;
        config.high_addend = 1;
        config.load_displacement = -4;
        accepts(config, "High word arithmetic and signed displacement retain descriptor identity");
        config = {};
        config.extended = false;
        config.high_source = AddressSource::literal_nonzero;
        config.forge_store_high = true;
        accepts(config, "Nonextended addressing reads only the low register");
        for (const auto source : {AddressSource::literal_zero, AddressSource::literal_nonzero,
                                  AddressSource::other_descriptor}) {
            config = {};
            config.high_source = source;
            rejects(config, "High word without matching descriptor provenance is rejected");
        }
        config = {};
        config.low_source = AddressSource::literal_zero;
        rejects(config, "Literal low address is not a descriptor");
        config = {};
        config.low_source = AddressSource::other_descriptor;
        rejects(config, "Low and high words from different descriptors are rejected");
        config = {};
        config.forge_store_high = true;
        rejects(config, "STG.E cannot ignore an unproven high word", InstructionName::stg);
        config = {};
        config.mixed_high = true;
        accepts(config, "A conditional high word keeps the low word's descriptor and a runtime high check");
        config = {};
        config.branch_pair = true;
        {
            const auto legal = take(legalize_image(config));
            bool multi = false;
            for (const auto& op : legal.ir.operations)
                if (const auto* m = std::get_if<MemoryInfo>(&op.info); m && op.opcode == Opcode::load &&
                    m->space == AddressSpace::storage)
                    multi = m->storage_alternatives.size() == 1 &&
                        m->storage_address->descriptor == TextureKey{descriptor_slot, input_descriptor_offset} &&
                        m->storage_alternatives[0].address.descriptor ==
                            TextureKey{descriptor_slot, output_descriptor_offset};
            check(multi, "A load with two descriptor origins keeps both in guest order");
            const auto source = artifact(config).source;
            check(source.find("storage_target") != std::string::npos,
                "Multi-target load selects its descriptor at run time");
            std::cout << "PASS: Multi-target load\n";
        }
        config = {};
        config.ambiguous_pair = true;
        rejects(config, "Two additive descriptor pairs do not identify a unique storage binding");
        config = {};
        config.store_displacement = 1;
        rejects(config, "Unaligned global storage writes require explicit lowering", InstructionName::stg,
            DiagnosticCode::unsupported_capability);
        config = {};
        config.shared_table = true;
        accepts(config, "Descriptor words staged through shared memory keep their storage binding");
        {
            const auto direct = artifact({});
            const auto staged = artifact(config);
            unsigned direct_bindings{}, staged_bindings{};
            auto count = [](const MslArtifact& result) {
                unsigned bindings{};
                for (const auto& resource : result.reflection.resources) {
                    const auto* buffer = std::get_if<BufferUse>(&resource.use);
                    if (buffer && std::holds_alternative<StorageBufferSource>(buffer->source)) ++bindings;
                }
                return bindings;
            };
            direct_bindings = count(direct);
            staged_bindings = count(staged);
            check(staged_bindings == direct_bindings,
                "Shared-memory descriptor staging resolves to the same storage bindings");
            check(staged.source.find("armigo_shared_words") != std::string::npos &&
                staged.source.find("storage_base") != std::string::npos,
                "Staged table MSL keeps the shared array and the runtime descriptor check");
        }
        rejects_malformed_legal(true);
        rejects_malformed_legal(false);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
