// ============================================================================
// Armigo Tests - Native Runtime Verification
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include "preparation_analysis.h"

#include <armigo/aarch64/instruction.h>
#include <armigo/execution/guest_entry.h>
#include <armigo/execution/prepared_image.h>
#include <armigo/platform/host_module.h>
#include <armigo/platform/thread_slot.h>

#include <array>
#include <atomic>
#include <barrier>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <initializer_list>
#include <memory>
#include <string>
#include <thread>
#include <tuple>
#include <vector>

namespace {
using namespace armigo;
using namespace runtime::preparation;
namespace a64 = aarch64;

struct Fixture {
    loader::NsoModuleImage image;
    linker::DynamicModule metadata;
    std::vector<RelocatedPointer> pointers;
    std::uint64_t base;

    explicit Fixture(std::uint64_t address = 0x100000) : base(address) {
        image.bytes.resize(0x3000);
        const std::array permissions{
            loader::SegmentPermission::read | loader::SegmentPermission::execute,
            loader::SegmentPermission::read,
            loader::SegmentPermission::read | loader::SegmentPermission::write};
        for (std::size_t i = 0; i < 3; ++i) {
            auto& segment = image.segments[i];
            segment.kind = static_cast<loader::NsoSegmentKind>(i);
            segment.image_offset = i * 0x1000;
            segment.content_size = i == 0 ? 0x100 : 0x1000;
            segment.mapped_size = 0x1000;
            segment.permission = permissions[i];
        }
        root(0);
    }

    void root(std::uint64_t offset) {
        linker::Symbol symbol;
        symbol.name = "authored_entry";
        symbol.value = offset;
        symbol.section = 1;
        symbol.info = 0x12;
        metadata.symbols.push_back(symbol);
    }

    void put(std::uint64_t offset, std::uint64_t value, unsigned width = 4) {
        for (unsigned i = 0; i < width; ++i)
            image.bytes.at(offset + i) = std::byte((value >> (i * 8)) & 255);
    }

    void put(std::uint64_t offset, a64::Encoded encoded) {
        if (!encoded.ok()) std::abort();
        put(offset, encoded.word);
    }

    void unwind(std::span<const std::uint64_t> entries, std::uint64_t frame = 0x1200) {
        constexpr std::uint64_t header = 0x1100;
        image.module_header = loader::ModuleHeaderInfo{};
        image.module_header->eh_frame_hdr_start_offset = header;
        image.module_header->eh_frame_hdr_end_offset = header + 12 + entries.size() * 8;
        put(header, 0x3b031b01u);
        put(header + 4, static_cast<std::uint32_t>(frame - (header + 4)));
        put(header + 8, entries.size());
        // Authored DWARF32 CIE: version 1, no augmentation, alignment 1/-8,
        // return register 30, and zero padding. No CFI program is interpreted.
        put(frame, 12);
        put(frame + 4, 0);
        put(frame + 8, 1, 1);
        put(frame + 9, 0, 1);
        put(frame + 10, 1, 1);
        put(frame + 11, 0x78, 1);
        put(frame + 12, 30, 1);
        for (std::size_t i = 0; i < entries.size(); ++i) {
            const auto fde = frame + 16 + i * 24;
            put(header + 12 + i * 8, static_cast<std::uint32_t>(entries[i] - header));
            put(header + 16 + i * 8, static_cast<std::uint32_t>(fde - header));
            put(fde, 20);
            put(fde + 4, fde + 4 - frame);
            put(fde + 8, base + entries[i], 8);
            put(fde + 16, 4, 8);
            put(entries[i], a64::encode_ret());
        }
    }

    AnalysisModule view() const { return {&image, &metadata, image.bytes, base, pointers}; }
};

bool check(bool success, const char* message) {
    if (!success) std::fprintf(stderr, "%s\n", message);
    return success;
}

AnalysisResult analyze(const Fixture& fixture, const AnalysisLimits& limits = {}) {
    const std::array modules{fixture.view()};
    return analyze_modules(modules, limits);
}

bool rejects(const Fixture& fixture, const char* expected) {
    const auto result = analyze(fixture);
    return check(!result.ok() && result.modules.empty() && result.roots.empty() &&
                 result.error.find(expected) != std::string::npos, expected);
}

bool coverage() {
    Fixture fixture;
    fixture.put(0, a64::encode_ret());
    fixture.put(0x40, a64::encode_svc(17));
    const auto result = analyze(fixture);
    bool ok = check(result.ok() && result.modules.size() == 1 &&
                    result.modules[0].code.size() == 1 && result.roots.size() == 1,
                    "Unreached bytes must remain outside instruction coverage.");
    fixture.put(0, a64::encode_load_literal(a64::Opcode::ldr_literal_x, 0,
                                           fixture.base, fixture.base + 0x40));
    fixture.put(4, a64::encode_ret());
    ok &= check(analyze(fixture).ok(), "Unreached text literals are preserved.");
    fixture.root(0x40);
    fixture.put(0x44, a64::encode_ret());
    ok &= rejects(fixture, "overlaps a literal");
    return ok;
}

bool execute_only_text_admission() {
    Fixture fixture;
    fixture.image.segments[0].permission = loader::SegmentPermission::execute;
    fixture.put(0, a64::encode_ret());
    const auto result = analyze(fixture);
    return check(result.ok() && result.modules.size() == 1 && result.modules[0].code.size() == 1,
                 "Owned instruction bytes must remain analyzable for execute-only guest text.");
}

bool authenticated_return_provenance() {
    bool ok = true;
    for (const auto first : {0xd503237fu, 0xaa0003feu}) { // PACIBSP; MOV LR,X0
        Fixture fixture;
        fixture.put(0, first);
        fixture.put(4, 0xd65f0fffu); // RETAB
        const auto result = analyze(fixture);
        ok &= check(!result.ok() && result.modules.empty() && result.roots.empty() &&
            !result.exclusive_proof_complete && result.return_diagnostics.size() == 1 &&
            result.return_diagnostics[0].module_index == 0 &&
            result.return_diagnostics[0].instruction_offset == 0 &&
            result.return_diagnostics[0].reason == ReturnError::unproven_link_write,
            "RETAB must not turn signed or arbitrary LR writes into proven return targets.");
    }
    Fixture terminal;
    terminal.put(0, 0xd65f0fffu);
    terminal.put(4, 0xaa0003feu);
    const auto result = analyze(terminal);
    ok &= check(result.ok() && result.modules[0].code.size() == 1,
        "RETAB ends offline discovery before an unreachable LR write.");
    return ok;
}

bool register_addressed_code_reads() {
    bool ok = true;
    for (const bool page_relative : {false, true}) {
        Fixture fixture;
        fixture.root(0x40);
        fixture.put(0x40, 0x91001652u); // ADD X18, X18, #5: requires a patch.
        fixture.put(0x44, a64::encode_ret());
        if (page_relative) {
            fixture.put(0, a64::encode_adrp(0, fixture.base, fixture.base));
            fixture.put(4, 0x91010000u); // ADD X0, X0, #64.
            fixture.put(8, 0xb9400000u); // LDR W0, [X0].
            fixture.put(12, a64::encode_ret());
        } else {
            fixture.put(0, a64::encode_adr(0, fixture.base, fixture.base + 0x40));
            fixture.put(4, 0xb9400000u); // LDR W0, [X0].
            fixture.put(8, a64::encode_ret());
        }
        ok &= rejects(fixture, "data reference");
    }
    return ok;
}

bool guest_relocation_store() {
    Fixture fixture;
    fixture.put(0, a64::encode_adr(9, fixture.base, fixture.base + 0x2000));
    fixture.put(4, a64::encode_adr(10, fixture.base + 4, fixture.base + 0x2010));
    fixture.put(8, 0xf900012au); // STR X10, [X9].
    fixture.put(12, a64::encode_ret());
    fixture.put(0x2000, fixture.base + 0x2010, 8);
    fixture.pointers.push_back({0x2000, {0, 0x2010}});
    auto view = fixture.view();
    bool ok = check(!analyze_modules({&view, 1}).ok(),
                    "Synthetic code must retain the pointer-store refusal.");
    view.guest_relocations = true;
    const auto admitted = analyze_modules({&view, 1});
    if (!admitted.ok()) std::fprintf(stderr, "Relocation store: %s\n", admitted.error.c_str());
    ok &= check(admitted.ok(),
                "Guest bootstrap may materialize a proven writable relocation.");
    fixture.put(4, a64::encode_adr(10, fixture.base + 4, fixture.base + 0x2018));
    ok &= check(!analyze_modules({&view, 1}).ok(),
                "A different pointer value must not inherit relocation provenance.");
    fixture.put(4, a64::encode_adr(10, fixture.base + 4, fixture.base + 0x2010));
    view.relocated_pointers = {};
    ok &= check(!analyze_modules({&view, 1}).ok(),
                "A writable slot alone is not proof of a relocated pointer.");
    view.relocated_pointers = fixture.pointers;
    fixture.put(8, 0xb900012au); // STR W10, [X9].
    ok &= check(!analyze_modules({&view, 1}).ok(),
                "A partial relocation-pointer store must remain unsupported.");
    fixture.put(8, 0xa900292au); // STP X10, X10, [X9].
    ok &= check(!analyze_modules({&view, 1}).ok(),
                "A pair store cannot use a single-slot relocation proof.");
    fixture.put(8, 0xf900012au); // STR X10, [X9].
    fixture.put(12, 0xf940012au); // LDR X10, [X9].
    fixture.put(16, 0xb940014bu); // LDR W11, [X10].
    fixture.put(20, a64::encode_ret());
    ok &= check(!analyze_modules({&view, 1}).ok(),
                "A mutable relocation reload must retain address taint.");
    fixture.put(4, a64::encode_b(fixture.base + 4, fixture.base + 8));
    fixture.put(8, a64::encode_adr(10, fixture.base + 8, fixture.base + 0x2010));
    fixture.put(12, 0xf900012au); // STR X10, [X9] across a block boundary.
    fixture.put(16, a64::encode_ret());
    ok &= check(!analyze_modules({&view, 1}).ok(),
                "A bootstrap value proof must not cross a branch leader.");
    return ok;
}

bool address_provenance_escapes() {
    bool ok = true;
    const auto base_fixture = [] {
        Fixture fixture;
        fixture.root(0x40);
        fixture.put(0, a64::encode_adr(0, fixture.base, fixture.base + 0x40));
        fixture.put(0x40, 0x91001652u); // ADD X18, X18, #5.
        fixture.put(0x44, a64::encode_ret());
        return fixture;
    };
    for (const auto intervening : {0xd503201fu, 0xaa0003e1u}) { // NOP; MOV X1, X0.
        auto fixture = base_fixture();
        fixture.put(4, intervening);
        fixture.put(8, intervening == 0xd503201fu ? 0xb9400002u : 0xb9400022u);
        fixture.put(12, a64::encode_ret());
        ok &= rejects(fixture, "data reference");
    }
    auto fixture = base_fixture();
    fixture.put(4, 0xf90003e0u); // STR X0, [SP]: pointer spill.
    fixture.put(8, a64::encode_ret());
    ok &= rejects(fixture, "stored data reference");

    fixture = base_fixture();
    fixture.put(4, a64::encode_br(0));
    fixture.put(0x40, 0xb9400001u); // LDR W1, [X0] in the indirect target.
    ok &= rejects(fixture, "data reference");

    fixture = base_fixture();
    fixture.put(4, a64::encode_blr(0));
    fixture.put(8, 0xb9400001u); // LDR W1, [X0] after returning from BLR.
    fixture.put(12, a64::encode_ret());
    ok &= rejects(fixture, "data reference");

    fixture = base_fixture();
    fixture.put(4, a64::encode_b(fixture.base + 4, fixture.base + 0x20));
    fixture.put(0x20, 0xb9400001u); // LDR W1, [X0] across a block leader.
    fixture.put(0x24, a64::encode_ret());
    ok &= rejects(fixture, "data reference");

    fixture = base_fixture();
    fixture.root(0x80);
    fixture.put(4, a64::encode_adr(30, fixture.base + 4, fixture.base + 0x80));
    fixture.put(8, a64::encode_ret());
    fixture.put(0x80, 0xb9400001u); // LDR W1, [X0] reached through an authored RET target.
    fixture.put(0x84, a64::encode_ret());
    ok &= rejects(fixture, "RET X30");

    fixture = base_fixture();
    fixture.put(4, 0xd2800000u); // MOV X0, #0 independently kills provenance.
    fixture.put(8, 0xb9400001u); // Incoming/unproven integer addresses are outside this proof.
    fixture.put(12, a64::encode_ret());
    ok &= check(analyze(fixture).ok(), "An independent overwrite kills address provenance.");
    fixture.put(4, 0x91001000u); // ADD X0, X0, #4 retains provenance.
    ok &= rejects(fixture, "data reference");

    for (const auto load : {0xb9400001u, 0x3dc00000u, 0xa9400801u}) {
        fixture = base_fixture();
        fixture.put(0, a64::encode_adr(0, fixture.base, fixture.base + 0x3f));
        fixture.put(4, load); // Scalar, vector and pair reads overlap code from before its first byte.
        fixture.put(8, a64::encode_ret());
        ok &= rejects(fixture, "data reference");
    }
    fixture = base_fixture();
    fixture.put(0, a64::encode_adr(0, fixture.base, fixture.base + 0x44));
    fixture.put(4, 0xb85fc001u); // LDUR W1, [X0, #-4].
    fixture.put(8, a64::encode_ret());
    ok &= rejects(fixture, "data reference");

    fixture = base_fixture();
    fixture.put(0, a64::encode_adrp(0, fixture.base, fixture.base));
    fixture.put(4, 0x91010001u); // ADD X1, X0, #64.
    fixture.put(8, 0xb9400020u); // LDR W0, [X1].
    fixture.put(12, a64::encode_ret());
    ok &= rejects(fixture, "data reference");

    fixture = base_fixture();
    fixture.put(0, a64::encode_load_literal(a64::Opcode::ldr_literal_x, 0,
                                           fixture.base, fixture.base + 0x1000));
    fixture.put(0x1000, fixture.base + 0x40, 8);
    fixture.pointers.push_back({0x1000, {0, 0x40}});
    fixture.put(4, 0xb9400001u); // LDR W1, [X0] through a relocated pointer.
    fixture.put(8, a64::encode_ret());
    ok &= rejects(fixture, "data reference");

    fixture = base_fixture();
    fixture.put(0, a64::encode_adr(0, fixture.base, fixture.base + 0x1000));
    fixture.put(0x1000, fixture.base + 0x40, 8);
    fixture.pointers.push_back({0x1000, {0, 0x40}});
    fixture.put(4, 0xfd400000u); // LDR D0, [X0].
    fixture.put(8, 0x9e660000u); // FMOV X0, D0 must not launder pointer provenance.
    fixture.put(12, 0xb9400001u); // LDR W1, [X0].
    fixture.put(16, a64::encode_ret());
    ok &= rejects(fixture, "data reference");

    fixture = base_fixture();
    fixture.put(0, a64::encode_adr(0, fixture.base, fixture.base + 0x2000));
    fixture.put(0x2000, fixture.base + 0x40, 8);
    fixture.pointers.push_back({0x2000, {0, 0x40}});
    fixture.put(4, 0xf9400000u); // LDR X0, [X0] from a mutable relocated slot.
    fixture.put(8, 0xb9400001u); // LDR W1, [X0].
    fixture.put(12, a64::encode_ret());
    ok &= rejects(fixture, "data reference");

    fixture = base_fixture();
    fixture.put(0, a64::encode_adr(0, fixture.base, fixture.base + 0x80));
    fixture.put(4, 0xb9400000u); // LDR W0, [X0] reads only unreached text data.
    fixture.put(8, a64::encode_ret());
    fixture.put(0x80, 0x91001652u);
    const auto original = fixture.image.bytes;
    ok &= check(analyze(fixture).ok() && fixture.image.bytes == original,
                "Proven register-addressed data outside coverage remains intact.");
    AnalysisLimits limits;
    limits.max_address_steps = 0;
    ok &= check(!analyze(fixture, limits).ok(), "Address provenance has a bounded work budget.");
    return ok;
}

// Atomic read-modify-writes are data references of their full width, store
// only their stored operands, and load only the old memory value.
bool atomic_data_references() {
    bool ok = true;
    const auto program = [](std::uint64_t data, std::initializer_list<std::uint32_t> words) {
        Fixture fixture;
        fixture.root(0x40);
        fixture.put(0x40, 0x91001652u); // ADD X18, X18, #5.
        fixture.put(0x44, a64::encode_ret());
        fixture.put(0, a64::encode_adr(0, fixture.base, fixture.base + data));
        std::uint64_t offset = 4;
        for (const auto word : words) {
            fixture.put(offset, word);
            offset += 4;
        }
        fixture.put(offset, a64::encode_ret());
        return fixture;
    };
    // An atomic on reached code is refused like a load or store.
    ok &= rejects(program(0x40, {0xb8210002u}), "data reference"); // LDADD W1, W2, [X0].
    // CASP covers both halves: 16 bytes from 0x38 reach code at 0x40, 8 do not.
    ok &= rejects(program(0x38, {0x48227c04u}), "data reference"); // CASP X2, X3, X4, X5, [X0].
    ok &= check(analyze(program(0x38, {0x08227c04u})).ok(), // CASP W2, W3, W4, W5, [X0].
                "An 8-byte CASP ending before reached code is a proven data reference.");

    // A proven data atomic gives its result no address provenance, and an ST
    // alias (no destination) is an ordinary data reference.
    ok &= check(analyze(program(0x2000, {0xb8210002u, 0xb9400043u})).ok(), // LDADD; LDR W3, [X2].
                "An atomic's old value must not inherit its base's address provenance.");
    ok &= check(analyze(program(0x2000, {0xb821001fu})).ok(), // STADD W1, [X0].
                "An ST alias on proven data is a data reference.");
    // The old value of a relocated pointer is an address, in either half.
    auto relocated = program(0x2000, {0x48227c04u, 0xb9400043u}); // CASP X2, X3, ...; LDR W3, [X2].
    relocated.put(0x2000, relocated.base + 0x40, 8);
    relocated.pointers.push_back({0x2000, {0, 0x40}});
    ok &= rejects(relocated, "data reference");
    relocated = program(0x2000, {0x48227c04u, 0xb9400063u}); // ...; LDR W3, [X3].
    relocated.put(0x2008, relocated.base + 0x40, 8);
    relocated.pointers.push_back({0x2008, {0, 0x40}});
    ok &= rejects(relocated, "data reference");

    // A stored module address escapes; a compared one does not.
    for (const auto& [word, stored_register, stored] : {
             std::tuple{0xf8218002u, 1u, true},   // SWP X1, X2, [X0]: X1 stored.
             std::tuple{0xb8210002u, 1u, true},   // LDADD W1, W2, [X0]: X1 combined.
             std::tuple{0xc8a27c01u, 1u, true},   // CAS X2, X1, [X0]: X1 desired.
             std::tuple{0xc8a17c02u, 1u, false},  // CAS X1, X2, [X0]: X1 compared.
             std::tuple{0x48247c02u, 3u, true},   // CASP X4, X5, X2, X3, [X0]: X3 desired.
             std::tuple{0x48227c04u, 3u, false},  // CASP X2, X3, X4, X5, [X0]: X3 compared.
         }) {
        // The base's producer must precede the access directly.
        auto fixture = program(0x2008, {0xd503201fu, word});
        fixture.put(0, a64::encode_adr(stored_register, fixture.base, fixture.base + 0x2000));
        fixture.put(4, a64::encode_adr(0, fixture.base + 4, fixture.base + 0x2008));
        if (stored) {
            ok &= rejects(fixture, "stored data reference");
            continue;
        }
        const auto admitted = analyze(fixture);
        if (!admitted.ok()) std::fprintf(stderr, "Compared atomic: %s\n", admitted.error.c_str());
        ok &= check(admitted.ok(), "A compared module address does not escape.");
    }
    // The escape does not depend on the base being a module address.
    auto escape = program(0x2008, {0xb8210002u});
    escape.put(0, a64::encode_adr(1, escape.base, escape.base + 0x2000));
    ok &= rejects(escape, "stored data reference");
    return ok;
}

bool targets() {
    Fixture consumer, provider(0x200000);
    provider.put(0, a64::encode_ret());
    consumer.put(0, a64::encode_load_literal(a64::Opcode::ldr_literal_x, 18,
                                            consumer.base, consumer.base + 0x1000));
    consumer.put(4, a64::encode_blr(18));
    consumer.put(8, a64::encode_ret());
    consumer.put(0x1000, provider.base, 8);
    consumer.pointers.push_back({0x1000, {1, 0}});
    const auto original = consumer.image.bytes;
    const std::array modules{consumer.view(), provider.view()};
    const auto first = analyze_modules(modules);
    const auto second = analyze_modules(modules);
    bool ok = check(first.ok() && second.ok() && first.modules[0].code.size() == 3 &&
                    first.modules[0].indirect_sites.size() == 1 &&
                    first.modules[0].indirect_sites[0].offset == 4 &&
                    first.modules[0].indirect_sites[0].targets == std::vector<Target>{{1, 0}} &&
                    first.modules[0].indirect_sites[0].targets ==
                        second.modules[0].indirect_sites[0].targets &&
                    consumer.image.bytes == original,
                    "Immutable relocated pointers must resolve cross-module BLR without mutations.");
    consumer.put(0, a64::encode_bl(consumer.base, provider.base));
    consumer.put(4, a64::encode_ret());
    const auto direct = analyze_modules(modules);
    ok &= check(direct.ok() && direct.modules[0].code.size() == 2 &&
                direct.modules[0].indirect_sites.empty(), "Direct cross-module calls retain continuation.");

    Fixture local;
    local.put(0, a64::encode_adr(18, local.base, local.base + 0x40));
    local.put(4, a64::encode_br(18));
    local.put(0x40, a64::encode_ret());
    const auto adr = analyze(local);
    ok &= check(adr.ok() && adr.modules[0].code.size() == 3 &&
                adr.modules[0].indirect_sites[0].targets == std::vector<Target>{{0, 0x40}},
                "ADR proves a local BR X18 target.");
    local.put(0, a64::encode_adrp(9, local.base, local.base + 0x1000));
    local.put(4, 0x91002129u); // ADD X9, X9, #8.
    local.put(8, 0xf9400132u); // LDR X18, [X9].
    local.put(12, a64::encode_br(18));
    local.put(0x1008, local.base + 0x40, 8);
    local.pointers.push_back({0x1008, {0, 0x40}});
    ok &= check(analyze(local).ok(), "ADRP/ADD/LDR proves a three-instruction pointer producer.");
    local.put(0, a64::encode_adrp(18, local.base, local.base));
    local.put(4, 0x91010252u); // ADD X18, X18, #64.
    local.put(8, a64::encode_br(18));
    ok &= check(analyze(local).ok(), "ADRP/ADD proves a direct executable pointer.");
    return ok;
}

bool cross_module_exclusive_x18_ingress() {
    enum class IngressKind {
        indirect,
        direct,
        b_cond,
        bc_cond,
        cbz,
        cbnz,
        tbz,
        tbnz,
    };
    const auto analyze_ingress = [](bool x18, IngressKind kind, std::uint64_t target) {
        Fixture consumer, provider(0x104000);
        provider.put(0, 0xd503201fu); // NOP: ordinary metadata root before the window.
        provider.put(4, x18 ? 0xc85f7c32u : 0xc85f7c22u); // LDXR X18/X2, [X1].
        provider.put(8, 0xd5033f5fu); // CLREX.
        provider.put(12, 0xd2800009u); // MOV X9, #0: window exit boundary.
        provider.put(16, a64::encode_ret());
        if (kind == IngressKind::indirect) {
            consumer.put(0, a64::encode_load_literal(a64::Opcode::ldr_literal_x, 5,
                                                    consumer.base, consumer.base + 0x1000));
            consumer.put(4, a64::encode_br(5));
            consumer.put(0x1000, provider.base + target, 8);
            consumer.pointers.push_back({0x1000, {1, target}});
        } else {
            const auto destination = provider.base + target;
            switch (kind) {
            case IngressKind::direct:
                consumer.put(0, a64::encode_b(consumer.base, destination));
                break;
            case IngressKind::b_cond:
                consumer.put(0, a64::encode_b_cond(a64::Condition::ne, consumer.base,
                                                   destination));
                break;
            case IngressKind::bc_cond:
                consumer.put(0, a64::encode_b_cond(a64::Condition::ne, consumer.base,
                                                   destination, true));
                break;
            case IngressKind::cbz:
            case IngressKind::cbnz:
                consumer.put(0, a64::encode_cbz(kind == IngressKind::cbnz, true, 0,
                                                consumer.base, destination));
                break;
            case IngressKind::tbz:
            case IngressKind::tbnz:
                consumer.put(0, a64::encode_tbz(kind == IngressKind::tbnz, 0, 0,
                                                consumer.base, destination));
                break;
            case IngressKind::indirect:
                std::abort();
            }
            if (kind != IngressKind::direct) consumer.put(4, a64::encode_ret());
        }
        const std::array modules{consumer.view(), provider.view()};
        return analyze_modules(modules);
    };

    struct IngressCase {
        std::uint64_t target;
        execution::ExclusiveWindowError reason;
    };
    constexpr std::array cases{
        IngressCase{4, execution::ExclusiveWindowError::entry_transfer},
        IngressCase{12, execution::ExclusiveWindowError::exit_transfer},
    };
    constexpr std::array ingress_kinds{
        IngressKind::indirect, IngressKind::direct, IngressKind::b_cond,
        IngressKind::bc_cond, IngressKind::cbz, IngressKind::cbnz,
        IngressKind::tbz, IngressKind::tbnz,
    };
    bool ok = true;
    for (const auto kind : ingress_kinds) {
        for (const auto& ingress : cases) {
            const auto refused = analyze_ingress(true, kind, ingress.target);
            ok &= check(!refused.ok() && refused.modules.empty() && refused.roots.empty() &&
                            !refused.exclusive_proof_complete &&
                            refused.exclusive_diagnostics.size() == 1 &&
                            refused.exclusive_diagnostics[0].module_index == 1 &&
                            refused.exclusive_diagnostics[0].window.window_offset == 4 &&
                            refused.exclusive_diagnostics[0].window.reason == ingress.reason,
                        "Cross-module ingress bypassed an x18 window transfer boundary.");
            const auto native = analyze_ingress(false, kind, ingress.target);
            ok &= check(native.ok() && native.exclusive_proof_complete &&
                            native.modules[1].external_entries ==
                                std::vector<std::uint64_t>{ingress.target},
                        "Cross-module ingress into an unchanged native window was refused.");
        }
        const auto entry = analyze_ingress(true, kind, 0);
        ok &= check(entry.ok() && entry.exclusive_proof_complete &&
                        entry.modules[1].external_entries == std::vector<std::uint64_t>{0},
                    "Safe cross-module ingress lost its window-entry ownership.");
    }
    return ok;
}

bool return_target_contract() {
    // The adjusted link skips the discovered continuation at 8 and opens an
    // exclusive window at 12 before entering the rewrite at 20.
    Fixture fixture;
    fixture.put(0, 0xaa1e03f3u); // MOV X19, X30.
    fixture.put(4, a64::encode_bl(fixture.base + 4, fixture.base + 64));
    fixture.put(8, a64::encode_b(fixture.base + 8, fixture.base + 20));
    fixture.put(12, 0xc85f7c02u); // LDXR X2, [X0].
    fixture.put(16, a64::encode_b(fixture.base + 16, fixture.base + 20));
    fixture.put(20, 0x91000652u); // ADD X18, X18, #1.
    fixture.put(24, 0xc8037c02u); // STXR W3, X2, [X0].
    fixture.put(28, 0xaa1303feu); // MOV X30, X19.
    fixture.put(32, a64::encode_ret());
    fixture.put(64, 0x910013deu); // ADD X30, X30, #4.
    fixture.put(68, a64::encode_ret());
    const auto original = fixture.image.bytes;
    auto result = analyze(fixture);
    bool ok = check(!result.ok() && result.modules.empty() && result.roots.empty() &&
                    !result.exclusive_proof_complete &&
                    result.error.find("proven RET X30 target") != std::string::npos &&
                    fixture.image.bytes == original,
                    "An adjusted return must not certify coverage that omits an exclusive opener.");

    // Remove the caller's unproven restore to isolate the callee's adjustment.
    fixture.put(28, 0xd503201fu); // NOP.
    result = analyze(fixture);
    ok &= check(!result.ok() && !result.exclusive_proof_complete &&
                result.error.find("module=0 site=64") != std::string::npos,
                "BL's link value must not hide arithmetic on X30 in the callee.");

    for (const auto word : {
            0x910013deu, // ADD X30, X30, #4.
            0xd10013deu, // SUB X30, X30, #4.
            0x8b0003deu, // ADD X30, X30, X0.
            0x110013deu, // ADD W30, W30, #4.
            0xaa0003feu, // MOV X30, X0.
            0x2a0003feu, // MOV W30, W0.
            0xd280019eu, // MOV X30, #12.
            0xf940001eu, // LDR X30, [X0].
            0xa8c17bfdu, // LDP X29, X30, [SP], #16.
        }) {
        fixture.put(64, word);
        result = analyze(fixture);
        ok &= check(!result.ok() && result.modules.empty() && !result.exclusive_proof_complete &&
                    result.error.find("module=0 site=64") != std::string::npos,
                    "Arithmetic, moves and loads must not establish unproven return targets.");
    }
    fixture.put(64, a64::encode_load_literal(a64::Opcode::ldr_literal_x, 30,
                                            fixture.base + 64, fixture.base + 0x1000));
    fixture.put(0x1000, fixture.base + 12, 8);
    ok &= rejects(fixture, "proven RET X30 target");

    // Unmodified X30 and both kinds of call still use discovered continuations.
    Fixture ordinary;
    ordinary.put(0, a64::encode_ret());
    ok &= check(analyze(ordinary).ok(), "An ordinary leaf RET X30 remains supported.");
    ordinary.put(0, a64::encode_bl(ordinary.base, ordinary.base + 64));
    ordinary.put(4, a64::encode_ret());
    ordinary.put(64, 0x91000400u); // ADD X0, X0, #1.
    ordinary.put(68, a64::encode_ret());
    result = analyze(ordinary);
    ok &= check(result.ok() && result.exclusive_proof_complete && result.modules[0].code.size() == 4,
                "BL with an unchanged callee link retains ordinary return coverage.");
    ordinary.put(0, a64::encode_adr(9, ordinary.base, ordinary.base + 64));
    ordinary.put(4, a64::encode_blr(9));
    ordinary.put(8, a64::encode_ret());
    result = analyze(ordinary);
    ok &= check(result.ok() && result.exclusive_proof_complete && result.modules[0].code.size() == 5,
                "Proven BLR with an unchanged callee link retains ordinary return coverage.");

    ordinary.put(64, 0xaa0003feu); // MOV X30, X0, including through a resolved BLR.
    ok &= rejects(ordinary, "proven RET X30 target");
    return ok;
}

bool saved_return_provenance() {
    const auto refused = [](const Fixture& fixture, std::uint64_t site,
                            ReturnError reason = ReturnError::unproven_link_write,
                            const AnalysisLimits& limits = AnalysisLimits{}) {
        const auto result = analyze(fixture, limits);
        return check(!result.ok() && result.modules.empty() && result.roots.empty() &&
            !result.exclusive_proof_complete && result.return_diagnostics.size() == 1 &&
            result.return_diagnostics[0].module_index == 0 &&
            result.return_diagnostics[0].instruction_offset == site &&
            result.return_diagnostics[0].reason == reason,
            "Return refusals must retain structured module, site and reason evidence.");
    };
    Fixture frame;
    frame.put(0, 0xa9bf7bfdu); // STP X29, X30, [SP, #-16]!.
    frame.put(4, a64::encode_bl(frame.base + 4, frame.base + 64));
    frame.put(8, 0xa8c17bfdu); // LDP X29, X30, [SP], #16.
    frame.put(12, a64::encode_ret());
    frame.put(64, a64::encode_ret());
    bool ok = check(analyze(frame).ok(), "A matching pre/post-index saved LR frame must be proven.");

    // An unknown slot, or a different slot in an otherwise familiar epilogue,
    // never becomes a proof merely because the instruction is LDP/RET.
    frame.put(0, 0xd10043ffu); // SUB SP, SP, #16, without saving LR.
    ok &= refused(frame, 8);
    frame.put(0, 0xa9bf7bfdu);
    frame.put(8, 0xa9407bfdu); // LDP X29, X30, [SP], without writeback.
    frame.put(12, 0x910043ffu); // ADD SP, SP, #16.
    frame.put(16, a64::encode_ret());
    ok &= check(analyze(frame).ok(), "An offset pair load may restore the exact saved LR slot.");
    frame.put(8, 0xa940fbfdu); // LDP X29, X30, [SP, #8], wrong LR slot.
    ok &= refused(frame, 8);
    frame.put(8, 0xa9407bfdu);

    for (const auto overwrite : {
            0xf90007ffu, // STR XZR, [SP, #8].
            0xb9000bffu, // STR WZR, [SP, #8], partial overwrite.
            0x390033ffu, // STRB WZR, [SP, #12], interior byte overwrite.
            0x3d8003e0u, // STR Q0, [SP], overlapping both saved values.
        }) {
        frame.put(64, overwrite);
        frame.put(68, a64::encode_ret());
        ok &= refused(frame, 8);
    }
    frame.put(64, 0xf90003ffu); // STR XZR, [SP], disjoint from LR at SP+8.
    ok &= check(analyze(frame).ok(), "A disjoint slot write must preserve the saved LR proof.");
    frame.put(64, 0x910003e9u); // MOV X9, SP.
    frame.put(68, 0xf900053fu); // STR XZR, [X9, #8].
    frame.put(72, a64::encode_ret());
    ok &= refused(frame, 8);
    frame.put(68, 0xf82a693fu); // STR XZR, [X9, X10], unknown stack slot.
    ok &= refused(frame, 8);
    frame.put(64, 0xf94003e9u); // LDR X9, [SP], the saved external X29 value.
    frame.put(68, 0xf900013fu); // STR XZR, [X9], disjoint under the entry contract.
    ok &= check(analyze(frame).ok(), "A matching saved external value retains its disjoint provenance.");
    frame.put(64, 0xf9400be9u); // LDR X9, [SP, #16], unproven contents may alias the frame.
    ok &= refused(frame, 8);
    frame.put(64, a64::encode_ret());

    Fixture alias;
    alias.put(0, 0xa9bf7bfdu); // STP X29, X30, [SP, #-16]!.
    alias.put(4, 0x910003fdu); // MOV X29, SP.
    alias.put(8, a64::encode_bl(alias.base + 8, alias.base + 64));
    alias.put(12, 0xf94007beu); // LDR X30, [X29, #8].
    alias.put(16, 0x910043ffu); // ADD SP, SP, #16.
    alias.put(20, a64::encode_ret());
    alias.put(64, 0xf81f0ffeu); // STR X30, [SP, #-16]!.
    alias.put(68, a64::encode_bl(alias.base + 68, alias.base + 96));
    alias.put(72, 0xf84107feu); // LDR X30, [SP], #16.
    alias.put(76, a64::encode_ret());
    alias.put(96, a64::encode_ret());
    ok &= check(analyze(alias).ok(),
        "Nested single-register frames and an exact frame-pointer alias preserve saved links.");
    alias.put(96, 0xf90003bfu); // STR XZR, [X29], disjoint from caller's LR.
    alias.put(100, a64::encode_ret());
    ok &= check(analyze(alias).ok(), "A callee's proven disjoint stack alias preserves the LR slot.");
    alias.put(96, 0xf90007bfu); // STR XZR, [X29, #8], caller's LR slot.
    ok &= refused(alias, 12);
    alias.put(96, 0xf900001du); // STR X29, [X0], escaping the private frame.
    ok &= refused(alias, 96, ReturnError::stack_pointer_escape);

    Fixture supervisor;
    supervisor.put(0, 0xd10043ffu); // SUB SP, SP, #16.
    supervisor.put(4, 0xf90007feu); // STR X30, [SP, #8].
    supervisor.put(8, 0x910003f3u); // MOV X19, SP.
    supervisor.put(12, a64::encode_svc(0));
    supervisor.put(16, 0xf9000260u); // STR X0, [X19].
    supervisor.put(20, 0xf94007feu); // LDR X30, [SP, #8].
    supervisor.put(24, 0x910043ffu); // ADD SP, SP, #16.
    supervisor.put(28, a64::encode_ret());
    // A callback may change X19 to SP+8. Its old, disjoint stack offset must
    // not survive SVC and make the subsequent overwrite appear harmless.
    ok &= refused(supervisor, 20);
    supervisor.put(16, 0xd503201fu); // NOP: no post-SVC store aliases the saved LR.
    ok &= check(analyze(supervisor).ok(), "SVC may preserve a private slot without preserving register values.");
    supervisor.put(0, 0xaa1e03f3u); // MOV X19, X30.
    supervisor.put(4, a64::encode_svc(0));
    supervisor.put(8, 0xaa1303feu); // MOV X30, X19, possibly changed by the callback.
    supervisor.put(12, a64::encode_ret());
    ok &= refused(supervisor, 8);

    Fixture registers;
    registers.put(0, 0xaa1e03f3u); // MOV X19, X30.
    registers.put(4, a64::encode_bl(registers.base + 4, registers.base + 64));
    registers.put(8, 0xaa1303feu); // MOV X30, X19.
    registers.put(12, a64::encode_ret());
    registers.put(64, a64::encode_ret());
    ok &= check(analyze(registers).ok(), "A preserved register copy may restore the exact entry link.");
    registers.put(64, 0xaa0003f3u); // MOV X19, X0: no assumed callee preservation.
    registers.put(68, a64::encode_ret());
    ok &= refused(registers, 8);
    registers.put(64, 0x910013deu); // ADD X30, X30, #4, despite the caller restore.
    ok &= refused(registers, 64);

    Fixture branches;
    branches.put(0, 0xd10043ffu); // SUB SP, SP, #16.
    branches.put(4, a64::encode_cbz(false, true, 0, branches.base + 4, branches.base + 16));
    branches.put(8, 0xf90007feu); // STR X30, [SP, #8].
    branches.put(12, a64::encode_b(branches.base + 12, branches.base + 20));
    branches.put(16, 0xf90007feu); // Same saved link on both paths.
    branches.put(20, a64::encode_bl(branches.base + 20, branches.base + 64));
    branches.put(24, 0xf94007feu); // LDR X30, [SP, #8].
    branches.put(28, 0x910043ffu); // ADD SP, SP, #16.
    branches.put(32, a64::encode_ret());
    branches.put(64, a64::encode_ret());
    ok &= check(analyze(branches).ok(), "Identical saved slots on every incoming path remain proven.");
    branches.put(16, 0xd503201fu); // NOP: one path has no saved value.
    ok &= refused(branches, 24);
    branches.put(16, 0xf90007ffu); // STR XZR, [SP, #8]: conflicting value.
    ok &= refused(branches, 24);
    branches.put(16, 0xf90003feu); // STR X30, [SP]: conflicting slot.
    ok &= refused(branches, 24);

    // Releasing and then reacquiring a frame cannot revive stale slot evidence.
    frame.put(4, 0x910043ffu); // ADD SP, SP, #16.
    frame.put(8, 0xd10043ffu); // SUB SP, SP, #16.
    frame.put(12, 0xf94007feu); // LDR X30, [SP, #8].
    ok &= refused(frame, 12);

    AnalysisLimits limits;
    registers.put(64, a64::encode_ret());
    limits.max_return_steps = 0;
    ok &= refused(registers, 0, ReturnError::work_limit, limits);
    limits = {};
    limits.max_return_states = 1;
    ok &= refused(registers, 4, ReturnError::state_limit, limits);
    limits = {};
    limits.max_return_stack_slots = 0;
    ok &= refused(frame, 0, ReturnError::stack_slot_limit, limits);

    // CAS and CASP only compare Rs (Rs+1); a frame address there does not
    // escape. A desired Rt (Rt+1) or an LSE source does.
    for (const auto& [word, alias, escapes] : {
             std::tuple{0xc8a07c41u, 0u, false}, // CAS X0, X1, [X2].
             std::tuple{0x88a07c41u, 0u, false}, // CAS W0, W1, [X2].
             std::tuple{0x48207c44u, 1u, false}, // CASP X0, X1, X4, X5, [X2].
             std::tuple{0x08207c44u, 1u, false}, // CASP W0, W1, W4, W5, [X2].
             std::tuple{0xc8a17c40u, 0u, true},  // CAS X1, X0, [X2].
             std::tuple{0x48247c40u, 1u, true},  // CASP X4, X5, X0, X1, [X2].
             std::tuple{0xf8200043u, 0u, true},  // LDADD X0, X3, [X2].
             std::tuple{0xf8208043u, 0u, true},  // SWP X0, X3, [X2].
         }) {
        Fixture atomic;
        atomic.put(0, 0xa9bf7bfdu); // STP X29, X30, [SP, #-16]!.
        atomic.put(4, 0x910003e0u | alias); // MOV Xalias, SP.
        atomic.put(8, word);
        atomic.put(12, 0xa8c17bfdu); // LDP X29, X30, [SP], #16.
        atomic.put(16, a64::encode_ret());
        ok &= escapes ? refused(atomic, 8, ReturnError::stack_pointer_escape)
                      : check(analyze(atomic).ok(), "A compared frame address does not escape.");
    }

    Fixture second(0x200000);
    second.put(0, 0x910013deu); // ADD X30, X30, #4.
    second.put(4, a64::encode_ret());
    const std::array modules{registers.view(), second.view()};
    const auto result = analyze_modules(modules);
    ok &= check(!result.ok() && result.return_diagnostics.size() == 1 &&
        result.return_diagnostics[0].module_index == 1 &&
        result.return_diagnostics[0].instruction_offset == 0 &&
        result.return_diagnostics[0].reason == ReturnError::unproven_link_write,
        "Structured return diagnostics must preserve the actual module index.");
    return ok;
}

bool exclusive_analysis() {
    Fixture fixture;
    fixture.put(0, 0xc85ffc20u); // LDAXR X0, [X1].
    fixture.put(4, a64::encode_cbz(false, true, 4, fixture.base + 4, fixture.base + 16));
    fixture.put(8, 0xc803fc20u); // STLXR W3, X0, [X1].
    fixture.put(12, a64::encode_b(fixture.base + 12, fixture.base + 20));
    fixture.put(16, 0xd5033f5fu); // CLREX.
    fixture.put(20, 0x91000652u); // ADD X18, X18, #1, after both closures.
    fixture.put(24, a64::encode_ret());
    auto result = analyze(fixture);
    bool ok = check(result.ok() && result.exclusive_proof_complete &&
        result.exclusive_windows_found == 1 && result.exclusive_windows_proved == 1 &&
        !result.modules[0].code[1].exclusive_monitor_clear &&
        result.modules[0].code[5].exclusive_monitor_clear,
        "The reached CFG must prove both exclusive-window exits and per-site clearance.");
    fixture.put(16, 0x91000652u);
    result = analyze(fixture);
    ok &= check(!result.ok() && result.modules.empty() && result.exclusive_diagnostics.size() == 1 &&
        result.exclusive_diagnostics[0].window.window_offset == 0 &&
        result.exclusive_diagnostics[0].window.instruction_offset == 24 &&
        result.exclusive_diagnostics[0].window.reason == execution::ExclusiveWindowError::unclosed_exit,
        "A path without STXR or CLREX must have a structured refusal.");
    fixture.put(16, 0xd5033f5fu);
    AnalysisLimits limits;
    limits.exclusive_windows.max_words_per_window = 2;
    result = analyze(fixture, limits);
    ok &= check(!result.ok() && !result.exclusive_proof_complete && result.exclusive_diagnostics[0].window.reason ==
        execution::ExclusiveWindowError::word_limit, "The runtime classifier must enforce the window bound.");

    fixture.put(4, 0xc85f7c22u); // Nested LDXR X2, [X1].
    result = analyze(fixture);
    ok &= check(result.ok() && result.exclusive_windows_found == 2 && result.exclusive_windows_proved == 2,
                "A nested load must start a separate proof and not close the older window.");
    fixture.put(4, a64::encode_svc(1));
    result = analyze(fixture);
    ok &= check(!result.ok() && result.exclusive_diagnostics[0].window.reason ==
        execution::ExclusiveWindowError::prepared_rewrite, "SVC gates cannot split an exclusive window.");
    fixture.put(4, a64::encode_ret());
    result = analyze(fixture);
    ok &= check(!result.ok() && result.exclusive_diagnostics[0].window.reason ==
        execution::ExclusiveWindowError::unclosed_exit, "A function cannot export an unclosed reservation.");

    fixture.put(0, a64::encode_bl(fixture.base, fixture.base + 32));
    fixture.put(4, a64::encode_ret());
    fixture.put(32, 0xc85f7c20u);
    fixture.put(36, a64::encode_ret());
    result = analyze(fixture);
    ok &= check(!result.ok() && result.exclusive_diagnostics[0].window.window_offset == 32,
                "An active return from a called function must not escape the proof.");
    fixture.put(36, 0xd5033f5fu);
    fixture.put(40, a64::encode_ret());
    ok &= check(analyze(fixture).ok(), "A called function may close its own native exclusive window.");
    return ok;
}

bool native_exclusive_contention() {
    const std::array<std::uint32_t, 12> code{
        0xf9400002u, // LDR X2, [X0]: expected value for compare-and-exchange.
        0xc85ffc03u, // LDAXR X3, [X0].
        0xeb02007fu, // CMP X3, X2.
        a64::encode_b_cond(a64::Condition::ne, 12, 40).word,
        0x91000463u, // ADD X3, X3, #1.
        0xc804fc03u, // STLXR W4, X3, [X0].
        a64::encode_cbz(true, false, 4, 24, 0).word,
        0xf1000421u, // SUBS X1, X1, #1.
        a64::encode_b_cond(a64::Condition::ne, 32, 0).word,
        0xd65f03c0u, // RET.
        0xd5033f5fu, // CLREX on comparison mismatch before retrying.
        a64::encode_b(44, 0).word,
    };
    Fixture fixture;
    for (std::size_t index = 0; index < code.size(); ++index)
        fixture.put(index * sizeof(std::uint32_t), code[index]);
    const auto source = fixture.image.bytes;

    const auto analysis = analyze(fixture);
    if (!check(analysis.ok() && analysis.modules.size() == 1 &&
                   analysis.modules[0].code.size() == code.size(),
               "A bounded exclusive retry loop was not admitted by analysis.")) {
        if (!analysis.error.empty()) std::fprintf(stderr, "%s\n", analysis.error.c_str());
        return false;
    }
    std::vector<execution::CoveredInstruction> covered;
    covered.reserve(analysis.modules[0].code.size());
    for (const auto& instruction : analysis.modules[0].code)
        covered.push_back({instruction.offset, instruction.word, std::nullopt, {}});

    const auto planned = execution::prepare_image(
        fixture.image.bytes.size(), covered,
        {.module_index = 0,
         .module_count = 1,
         .scalar_x18_policy = {.exclusive_monitor_clear = false,
                               .stack_scratch_available = true,
                               .runtime_storage_disjoint = true}});
    if (!check(planned.ok(), "A proven exclusive retry loop was not prepared.")) return false;
    const auto& prepared = *planned.image;
    const bool native_only = fixture.image.bytes == source && prepared.appendix.empty() &&
                             prepared.replacements.empty() &&
                             prepared.image_size() == source.size() &&
                             prepared.coverage.size() == code.size();
    if (!check(native_only,
               "Exclusive source words changed or preparation injected code into the window."))
        return false;
    bool unchanged = true;
    for (std::size_t index = 0; index < prepared.coverage.size(); ++index) {
        const auto& word = prepared.coverage[index];
        unchanged &= word.image_offset == index * sizeof(std::uint32_t) &&
                     word.expected_word == code[index] &&
                     word.replacement_word == word.expected_word;
    }
    if (!check(unchanged,
               "Exclusive source words changed or preparation injected code into the window."))
        return false;

    const std::array regions = {platform::ModuleRegion{
        0, source.size(), memory::Permission::read | memory::Permission::execute}};
    auto mapped = platform::HostModule::create(source, regions);
    if (!check(mapped.ok(), "The exclusive contention fixture could not be mapped.")) return false;
    const std::array module_bases = {
        static_cast<std::uint64_t>(mapped.module->base_address())};
    {
        auto bytes = mapped.module->preparation_bytes();
        const auto applied = execution::apply_prepared_image(
            prepared, bytes, {.module_bases = module_bases});
        if (!check(applied.ok() &&
                       std::memcmp(bytes.data(), source.data(), code.size() * sizeof(code[0])) == 0,
                   "Applying the prepared image changed the native exclusive window."))
            return false;
    }
    if (!check(mapped.module->seal().ok(), "The exclusive contention fixture could not be sealed."))
        return false;
    const auto entry = mapped.module->executable_address(0);
    if (!check(entry.has_value(), "The exclusive contention entry is not executable.")) return false;

    auto first_stack = platform::GuardedStack::create({.usable_size = 1u << 20});
    auto second_stack = platform::GuardedStack::create({.usable_size = 1u << 20});
    const auto table = execution::GuestSiteTable::build(
        std::span<const execution::GuestGateSite>{});
    if (!check(first_stack.ok() && second_stack.ok() && table.ok(),
               "Concurrent guest execution resources could not be created."))
        return false;

    constexpr std::uint64_t iterations = 250'000;
    alignas(64) std::uint64_t counter = 0;
    std::array<execution::GuestThreadContext, 2> contexts{};
    contexts[0].set_thread_pointer(1);
    contexts[1].set_thread_pointer(2);
    std::array<execution::GuestEntryResult, 2> results{};
    std::array<bool, 2> bound{};
    std::barrier start{2};
    const std::array stacks = {first_stack.stack.get(), second_stack.stack.get()};
    const auto run = [&](std::size_t index) {
        execution::GuestDispatchScope dispatch(table.table, contexts[index]);
        bound[index] = dispatch.setup_error() == execution::GuestDispatchFailureKind::none;
        start.arrive_and_wait();
        if (bound[index]) {
            results[index] = execution::invoke_guest_entry(
                *stacks[index], {.function = *entry,
                                 .x0 = reinterpret_cast<std::uintptr_t>(&counter),
                                 .x1 = iterations});
        }
    };
    std::thread first(run, 0);
    std::thread second(run, 1);
    first.join();
    second.join();

    return check(bound[0] && bound[1] && results[0].returned() && results[1].returned() &&
                     results[0].x1 == 0 && results[1].x1 == 0 &&
                     counter == iterations * 2 && !contexts[0].bound() && !contexts[1].bound(),
                 "Concurrent native exclusive loops lost an increment or execution context.");
}

enum class ExclusiveX18Role {
    base,
    transfer,
    status,
};

template <std::size_t N, typename Execute>
bool with_prepared_x18_oracle(const std::array<std::uint32_t, N>& code, Execute execute) {
    Fixture fixture;
    for (std::size_t index = 0; index < code.size(); ++index)
        fixture.put(index * sizeof(std::uint32_t), code[index]);

    const auto analysis = analyze(fixture);
    if (!check(analysis.ok() && analysis.modules.size() == 1 &&
                   analysis.modules[0].code.size() == code.size() &&
                   analysis.exclusive_windows_found == 1 &&
                   analysis.exclusive_windows_proved == 1,
               "A deterministic x18 exclusive oracle was not admitted by analysis.")) {
        if (!analysis.error.empty()) std::fprintf(stderr, "%s\n", analysis.error.c_str());
        return false;
    }
    std::vector<execution::CoveredInstruction> covered;
    covered.reserve(analysis.modules[0].code.size());
    for (const auto& instruction : analysis.modules[0].code)
        covered.push_back({instruction.offset, instruction.word, std::nullopt, {}});

    const auto planned = execution::prepare_image(
        fixture.image.bytes.size(), covered,
        {.module_index = 0,
         .module_count = 1,
         .scalar_x18_policy = {.exclusive_monitor_clear = false,
                               .stack_scratch_available = true,
                               .runtime_storage_disjoint = true}});
    if (!check(planned.ok() && planned.image->image_size() > fixture.image.bytes.size() &&
                   !planned.image->appendix.empty() && !planned.image->replacements.empty(),
               "A deterministic x18 exclusive oracle was not prepared as a whole window."))
        return false;

    platform::ThreadSlotKey thread_slot;
    const auto native_slot = thread_slot.native_access();
    if (!check(thread_slot.valid() && native_slot.has_value(),
               "A deterministic x18 exclusive oracle could not probe a native thread slot."))
        return false;
    std::vector<std::byte> source(planned.image->image_size());
    std::memcpy(source.data(), fixture.image.bytes.data(), fixture.image.bytes.size());
    const std::array regions = {platform::ModuleRegion{
        0, source.size(), memory::Permission::read | memory::Permission::execute}};
    auto mapped = platform::HostModule::create(source, regions);
    if (!check(mapped.ok(), "A deterministic x18 exclusive oracle could not be mapped."))
        return false;
    const std::array module_bases = {
        static_cast<std::uint64_t>(mapped.module->base_address())};
    const auto applied = execution::apply_prepared_image(
        *planned.image, mapped.module->preparation_bytes(),
        {.module_bases = module_bases,
         .thread_slot_byte_offset = native_slot->tsd_byte_offset});
    if (!check(applied.ok() && mapped.module->seal().ok(),
               "A deterministic x18 exclusive oracle could not be applied and sealed."))
        return false;
    const auto entry = mapped.module->executable_address(0);
    auto stack = platform::GuardedStack::create({.usable_size = 1u << 20});
    const auto table = execution::GuestSiteTable::build(
        std::span<const execution::GuestGateSite>{});
    if (!check(entry.has_value() && stack.ok() && table.ok(),
               "A deterministic x18 exclusive oracle could not allocate execution resources."))
        return false;
    return execute(*entry, thread_slot, *stack.stack, table.table);
}

bool deterministic_clrex_x18_exit(ExclusiveX18Role role) {
    const auto exclusive_load = role == ExclusiveX18Role::base ? 0xc85ffe43u :
                                role == ExclusiveX18Role::transfer ? 0xc85ffcb2u :
                                                                   0xc85ffca3u;
    const std::array<std::uint32_t, 7> code{
        0xf9400005u, // LDR X5, [X0]: counter address.
        0xf9400406u, // LDR X6, [X0, #8]: observation address and entry host.
        exclusive_load,
        0xd5033f5fu, // CLREX.
        0xd2800009u, // MOV X9, #0: scratch is dead on the CLREX exit.
        0xf90000d2u, // STR X18, [X6]: observe before any overwrite or retry.
        a64::encode_ret().word,
    };
    return with_prepared_x18_oracle(code,
        [role](std::uint64_t entry, platform::ThreadSlotKey& thread_slot,
               platform::GuardedStack& stack,
               const std::shared_ptr<const execution::GuestSiteTable>& table) {
            constexpr std::uint64_t value = 49;
            constexpr std::uint64_t initial_x18 = 0x1818'1234'5678'0000ull;
            std::uint64_t counter = value;
            std::uint64_t observation = 0xfeed'face'feed'faceull;
            const std::array arguments{
                reinterpret_cast<std::uintptr_t>(&counter),
                reinterpret_cast<std::uintptr_t>(&observation)};
            execution::GuestThreadContext context;
            const auto expected = role == ExclusiveX18Role::base
                ? reinterpret_cast<std::uintptr_t>(&counter) :
                role == ExclusiveX18Role::transfer ? value : initial_x18;
            context.set_guest_x18(role == ExclusiveX18Role::base
                                      ? reinterpret_cast<std::uintptr_t>(&counter)
                                      : initial_x18);
            execution::GuestDispatchScope dispatch(table, context);
            platform::ThreadSlotBinding binding(thread_slot, context.bound_register_cell());
            if (!check(dispatch.setup_error() == execution::GuestDispatchFailureKind::none &&
                           binding.native_access().has_value(),
                       "The deterministic CLREX oracle could not bind guest state."))
                return false;
            const auto result = execution::invoke_guest_entry(
                stack, {.function = entry,
                        .x0 = reinterpret_cast<std::uintptr_t>(arguments.data())});
            return check(result.returned() && counter == value && observation == expected &&
                             context.guest_x18() == expected,
                         "CLREX did not publish x18 before its immediate exit observation.");
        });
}

bool deterministic_failed_stxr_x18_exit(ExclusiveX18Role role) {
    const auto exclusive_load = role == ExclusiveX18Role::base ? 0xc85ffe43u :
                                role == ExclusiveX18Role::transfer ? 0xc85ffcb2u :
                                                                   0xc85ffca3u;
    const auto increment = role == ExclusiveX18Role::transfer ? 0x91000652u : 0x91000463u;
    const auto exclusive_store = role == ExclusiveX18Role::base ? 0xc804fe43u :
                                 role == ExclusiveX18Role::transfer ? 0xc804fcb2u :
                                                                    0xc812fca3u;
    const std::array<std::uint32_t, 19> code{
        0xf9400005u, // LDR X5, [X0]: counter address.
        0xf9400406u, // LDR X6, [X0, #8]: observation address.
        0xf9400807u, // LDR X7, [X0, #16]: ready address.
        0xf9400c08u, // LDR X8, [X0, #24]: release address.
        0xf940100au, // LDR X10, [X0, #32]: status address.
        0xd503201fu, // NOP: fallthrough host for the window-entry transfer.
        exclusive_load,
        0xd5033bbfu, // DMB ISH: publish readiness only after the exclusive load.
        0x5280002bu, // MOV W11, #1.
        0xb90000ebu, // STR W11, [X7].
        0xb940010bu, // LDR W11, [X8].
        a64::encode_cbz(false, false, 11, 44, 40).word,
        0xd5033bbfu, // DMB ISH: observe the conflicting host store before STXR.
        increment,
        exclusive_store,
        0xd2800009u, // MOV X9, #0: scratch is dead on the STXR exit.
        0xf90000d2u, // STR X18, [X6]: observe before any overwrite or retry.
        role == ExclusiveX18Role::status ? 0xb9000152u : 0xb9000144u,
        a64::encode_ret().word,
    };
    return with_prepared_x18_oracle(code,
        [role](std::uint64_t entry, platform::ThreadSlotKey& thread_slot,
               platform::GuardedStack& stack,
               const std::shared_ptr<const execution::GuestSiteTable>& table) {
            constexpr std::uint64_t initial = 41;
            constexpr std::uint64_t conflict = 99;
            constexpr std::uint64_t initial_x18 = 0x1818'9876'5432'0000ull;
            alignas(64) std::atomic<std::uint64_t> counter{initial};
            std::uint64_t observation = 0xfeed'face'feed'faceull;
            std::atomic<std::uint32_t> ready{0};
            std::atomic<std::uint32_t> release{0};
            std::uint32_t status = ~std::uint32_t{};
            const std::array arguments{
                reinterpret_cast<std::uintptr_t>(&counter),
                reinterpret_cast<std::uintptr_t>(&observation),
                reinterpret_cast<std::uintptr_t>(&ready),
                reinterpret_cast<std::uintptr_t>(&release),
                reinterpret_cast<std::uintptr_t>(&status)};
            execution::GuestThreadContext context;
            context.set_guest_x18(role == ExclusiveX18Role::base
                                      ? reinterpret_cast<std::uintptr_t>(&counter)
                                      : initial_x18);
            execution::GuestEntryResult result;
            bool bound = false;
            std::atomic<bool> started{false};
            std::thread guest([&] {
                execution::GuestDispatchScope dispatch(table, context);
                platform::ThreadSlotBinding binding(thread_slot, context.bound_register_cell());
                bound = dispatch.setup_error() == execution::GuestDispatchFailureKind::none &&
                        binding.native_access().has_value();
                started.store(true, std::memory_order_release);
                if (bound) {
                    result = execution::invoke_guest_entry(
                        stack, {.function = entry,
                                .x0 = reinterpret_cast<std::uintptr_t>(arguments.data())});
                }
            });
            while (!started.load(std::memory_order_acquire)) std::this_thread::yield();
            if (bound) {
                while (!ready.load(std::memory_order_acquire)) std::this_thread::yield();
                counter.store(conflict, std::memory_order_seq_cst);
                release.store(1, std::memory_order_release);
            }
            guest.join();
            const auto expected = role == ExclusiveX18Role::base
                ? reinterpret_cast<std::uintptr_t>(&counter) :
                role == ExclusiveX18Role::transfer ? initial + 1 : 1;
            return check(bound && result.returned() && status == 1 &&
                             counter.load(std::memory_order_relaxed) == conflict &&
                             observation == expected && context.guest_x18() == expected,
                         "A forced failed STXR did not run and publish x18 before observation.");
        });
}

bool native_exclusive_x18_contention(ExclusiveX18Role role) {
    const auto exclusive_load = role == ExclusiveX18Role::base ? 0xc85ffe43u :
                                role == ExclusiveX18Role::transfer ? 0xc85ffcb2u :
                                                                   0xc85ffca3u;
    const auto compare = role == ExclusiveX18Role::transfer ? 0xeb02025fu : 0xeb02007fu;
    const auto increment = role == ExclusiveX18Role::transfer ? 0x91000652u : 0x91000463u;
    const auto exclusive_store = role == ExclusiveX18Role::base ? 0xc804fe43u :
                                 role == ExclusiveX18Role::transfer ? 0xc804fcb2u :
                                                                    0xc812fca3u;
    const std::array<std::uint32_t, 31> code{
        0xf9400005u, // LDR X5, [X0]: shared counter address.
        0xf9400406u, // LDR X6, [X0, #8]: per-thread observations.
        0xf9400807u, // LDR X7, [X0, #16]: latest CLREX-exit observation.
        0xf9400c08u, // LDR X8, [X0, #24]: latest failed-STXR observation.
        role == ExclusiveX18Role::base ? 0xf9400242u : 0xf94000a2u,
        0xd503201fu, // NOP: fallthrough host for the window-entry transfer.
        exclusive_load,
        0xaa1203eau, // MOV X10, X18: preserve the live value independently of publication.
        compare,
        a64::encode_b_cond(a64::Condition::ne, 36, 80).word,
        increment,
        role == ExclusiveX18Role::transfer ? 0x9100054au : 0xd503201fu,
        exclusive_store,
        0xd2800009u, // MOV X9, #0: scratch is dead on the STXR exit.
        a64::encode_cbz(true, false, role == ExclusiveX18Role::status ? 18 : 4, 56, 104).word,
        0xf90000d2u, // STR X18, [X6]: observe the virtual value after the window.
        0x910020c6u, // ADD X6, X6, #8.
        0xf1000421u, // SUBS X1, X1, #1.
        a64::encode_b_cond(a64::Condition::ne, 72, 16).word,
        a64::encode_ret().word,
        0xd5033f5fu, // CLREX on comparison mismatch.
        0xd2800009u, // MOV X9, #0: scratch is dead on the CLREX exit.
        0xeb0a025fu, // CMP X18, X10: check every abort before its value can be overwritten.
        a64::encode_b_cond(a64::Condition::ne, 92, 76).word,
        0xf90000f2u, // STR X18, [X7]: observe before retrying the CLREX path.
        a64::encode_b(100, 16).word,
        role == ExclusiveX18Role::status ? 0xd280002au : 0xd503201fu, // MOV X10, #1 / NOP.
        0xeb0a025fu, // CMP X18, X10: a mismatch returns with iterations still remaining.
        a64::encode_b_cond(a64::Condition::ne, 112, 76).word,
        0xf9000112u, // STR X18, [X8]: observe before retrying the failed STXR path.
        a64::encode_b(120, 16).word,
    };
    Fixture fixture;
    for (std::size_t index = 0; index < code.size(); ++index)
        fixture.put(index * sizeof(std::uint32_t), code[index]);

    const auto analysis = analyze(fixture);
    if (!check(analysis.ok() && analysis.modules.size() == 1 &&
                   analysis.modules[0].code.size() == code.size() &&
                   analysis.exclusive_windows_found == 1 &&
                   analysis.exclusive_windows_proved == 1,
               "An x18 exclusive retry loop was not admitted by analysis.")) {
        if (!analysis.error.empty()) std::fprintf(stderr, "%s\n", analysis.error.c_str());
        return false;
    }
    std::vector<execution::CoveredInstruction> covered;
    covered.reserve(analysis.modules[0].code.size());
    for (const auto& instruction : analysis.modules[0].code)
        covered.push_back({instruction.offset, instruction.word, std::nullopt, {}});

    const auto planned = execution::prepare_image(
        fixture.image.bytes.size(), covered,
        {.module_index = 0,
         .module_count = 1,
         .scalar_x18_policy = {.exclusive_monitor_clear = false,
                               .stack_scratch_available = true,
                               .runtime_storage_disjoint = true}});
    if (!check(planned.ok() && planned.image->image_size() > fixture.image.bytes.size() &&
                   !planned.image->appendix.empty() && !planned.image->replacements.empty(),
               "An x18 exclusive retry loop was not prepared as a whole window."))
        return false;
    const auto& prepared = *planned.image;

    platform::ThreadSlotKey thread_slot;
    const auto native_slot = thread_slot.native_access();
    if (!check(thread_slot.valid() && native_slot.has_value(),
               "The x18 contention fixture could not probe a native thread slot."))
        return false;
    std::vector<std::byte> source(prepared.image_size());
    std::memcpy(source.data(), fixture.image.bytes.data(), fixture.image.bytes.size());
    const std::array regions = {platform::ModuleRegion{
        0, source.size(), memory::Permission::read | memory::Permission::execute}};
    auto mapped = platform::HostModule::create(source, regions);
    if (!check(mapped.ok(), "The x18 contention fixture could not be mapped.")) return false;
    const std::array module_bases = {
        static_cast<std::uint64_t>(mapped.module->base_address())};
    const auto applied = execution::apply_prepared_image(
        prepared, mapped.module->preparation_bytes(),
        {.module_bases = module_bases,
         .thread_slot_byte_offset = native_slot->tsd_byte_offset});
    if (!check(applied.ok() && mapped.module->seal().ok(),
               "The x18 contention fixture could not be applied and sealed."))
        return false;
    const auto entry = mapped.module->executable_address(0);
    if (!check(entry.has_value(), "The x18 contention entry is not executable.")) return false;

    auto first_stack = platform::GuardedStack::create({.usable_size = 1u << 20});
    auto second_stack = platform::GuardedStack::create({.usable_size = 1u << 20});
    const auto table = execution::GuestSiteTable::build(
        std::span<const execution::GuestGateSite>{});
    if (!check(first_stack.ok() && second_stack.ok() && table.ok(),
               "Concurrent x18 guest execution resources could not be created."))
        return false;

    constexpr std::uint64_t iterations = 25'000;
    alignas(64) std::uint64_t counter = 0;
    std::array<std::vector<std::uint64_t>, 2> observations{
        std::vector<std::uint64_t>(iterations), std::vector<std::uint64_t>(iterations)};
    struct Arguments {
        std::uint64_t counter;
        std::uint64_t observations;
        std::uint64_t clrex_observation;
        std::uint64_t stxr_observation;
    };
    std::array<Arguments, 2> arguments{};
    constexpr std::uint64_t unobserved = 0xfeed'face'feed'faceull;
    std::array<std::uint64_t, 2> clrex_observations{unobserved, unobserved};
    std::array<std::uint64_t, 2> stxr_observations{unobserved, unobserved};
    std::array<execution::GuestThreadContext, 2> contexts{};
    for (std::size_t index = 0; index < contexts.size(); ++index) {
        arguments[index] = {
            reinterpret_cast<std::uintptr_t>(&counter),
            reinterpret_cast<std::uintptr_t>(observations[index].data()),
            reinterpret_cast<std::uintptr_t>(&clrex_observations[index]),
            reinterpret_cast<std::uintptr_t>(&stxr_observations[index])};
        contexts[index].set_guest_x18(
            role == ExclusiveX18Role::base ? reinterpret_cast<std::uintptr_t>(&counter) :
                                             0x1818'0000'0000'0000ull + index);
    }
    std::array<execution::GuestEntryResult, 2> results{};
    std::array<bool, 2> bound{};
    std::barrier start{2};
    const std::array stacks = {first_stack.stack.get(), second_stack.stack.get()};
    const auto run = [&](std::size_t index) {
        execution::GuestDispatchScope dispatch(table.table, contexts[index]);
        platform::ThreadSlotBinding binding(thread_slot, contexts[index].bound_register_cell());
        bound[index] = dispatch.setup_error() == execution::GuestDispatchFailureKind::none &&
                       binding.native_access().has_value();
        start.arrive_and_wait();
        if (bound[index]) {
            results[index] = execution::invoke_guest_entry(
                *stacks[index], {.function = *entry,
                                 .x0 = reinterpret_cast<std::uintptr_t>(&arguments[index]),
                                 .x1 = iterations});
        }
    };
    std::thread first(run, 0);
    std::thread second(run, 1);
    first.join();
    second.join();

    bool observed = true;
    if (role == ExclusiveX18Role::transfer) {
        std::vector<bool> seen(iterations * 2 + 1);
        for (const auto& thread : observations) {
            for (const auto value : thread) {
                if (value == 0 || value > iterations * 2 || seen[value]) {
                    observed = false;
                } else {
                    seen[value] = true;
                }
            }
        }
    } else {
        const auto expected = role == ExclusiveX18Role::base
            ? reinterpret_cast<std::uintptr_t>(&counter) : 0;
        for (const auto& thread : observations) {
            for (const auto value : thread) observed &= value == expected;
        }
    }
    for (std::size_t index = 0; index < contexts.size(); ++index) {
        const auto valid_failure_observation = [&](std::uint64_t value, bool stxr) {
            if (value == unobserved) return true;
            if (role == ExclusiveX18Role::base)
                return value == reinterpret_cast<std::uintptr_t>(&counter);
            if (role == ExclusiveX18Role::status)
                return stxr ? value == 1 :
                    value <= 1 || value == 0x1818'0000'0000'0000ull + index;
            return value > 0 && value <= iterations * 2;
        };
        observed &= valid_failure_observation(clrex_observations[index], false) &&
                    valid_failure_observation(stxr_observations[index], true);
    }
    for (std::size_t index = 0; index < contexts.size(); ++index) {
        const auto expected = role == ExclusiveX18Role::base
            ? reinterpret_cast<std::uintptr_t>(&counter) :
            role == ExclusiveX18Role::transfer ? observations[index].back() : 0;
        observed &= contexts[index].guest_x18() == expected;
    }
    return check(bound[0] && bound[1] && results[0].returned() && results[1].returned() &&
                     results[0].x1 == 0 && results[1].x1 == 0 &&
                     counter == iterations * 2 && observed &&
                     !contexts[0].bound() && !contexts[1].bound(),
                 "A prepared x18 exclusive loop lost contention or virtual-register state.");
}

bool native_exclusive_x18_contention() {
    bool ok = true;
    for (const auto role : {ExclusiveX18Role::base, ExclusiveX18Role::transfer,
                            ExclusiveX18Role::status}) {
        ok &= deterministic_clrex_x18_exit(role);
        ok &= deterministic_failed_stxr_x18_exit(role);
        ok &= native_exclusive_x18_contention(role);
    }
    return ok;
}

bool refusal() {
    bool ok = true;
    Fixture fixture;
    fixture.put(0, a64::encode_br(18));
    ok &= rejects(fixture, "singleton target proof");
    fixture.put(0, a64::encode_ret(18));
    ok &= rejects(fixture, "RET X30");
    fixture.put(0, 0xffffffffu);
    ok &= rejects(fixture, "register effects");
    fixture.put(0, a64::encode_brk(0));
    ok &= rejects(fixture, "source trap");
    fixture.put(0, 0xc85f7c32u); // LDXR X18, [X1].
    fixture.put(4, a64::encode_ret());
    ok &= rejects(fixture, "Exclusive window refusal");
    fixture.put(0, a64::encode_load_literal(a64::Opcode::ldr_literal_x, 18,
                                           fixture.base, fixture.base + 0x2000));
    fixture.put(4, a64::encode_br(18));
    fixture.put(0x2000, fixture.base + 0x40, 8);
    fixture.put(0x40, a64::encode_ret());
    fixture.pointers.push_back({0x2000, {0, 0x40}});
    ok &= rejects(fixture, "singleton target proof");
    fixture.pointers.clear();
    fixture.put(0, a64::encode_load_literal(a64::Opcode::ldr_literal_x, 18,
                                           fixture.base, fixture.base + 0x1000));
    fixture.put(0x1000, fixture.base + 0x40, 8);
    ok &= rejects(fixture, "singleton target proof");
    fixture.put(0, a64::encode_adr(18, fixture.base, fixture.base + 0x40));
    fixture.root(4);
    ok &= rejects(fixture, "singleton target proof");
    fixture.metadata.symbols.resize(1);
    fixture.put(0, a64::encode_adr(18, fixture.base, fixture.base + 0x40));
    fixture.put(4, a64::encode_bl(fixture.base + 4, fixture.base + 0x80));
    fixture.put(8, a64::encode_br(18));
    fixture.put(0x80, a64::encode_ret());
    ok &= rejects(fixture, "singleton target proof");
    fixture.put(0, a64::encode_ret());
    fixture.image.module_header = loader::ModuleHeaderInfo{};
    fixture.image.module_header->eh_frame_hdr_end_offset = 4;
    ok &= rejects(fixture, "unwind metadata");
    fixture.image.module_header.reset();
    AnalysisLimits limits;
    limits.max_code_words = 0;
    ok &= check(!analyze(fixture, limits).ok(), "Zero code budget refuses before discovery.");
    return ok;
}

bool later_leader() {
    Fixture fixture;
    fixture.put(0, a64::encode_adr(18, fixture.base, fixture.base + 0x40));
    fixture.put(4, a64::encode_br(18));
    fixture.put(0x40, a64::encode_adr(18, fixture.base + 0x40, fixture.base + 4));
    fixture.put(0x44, a64::encode_br(18));
    bool ok = rejects(fixture, "singleton target proof");
    fixture.put(0, a64::encode_adrp(18, fixture.base, fixture.base));
    fixture.put(4, 0x91001252u); // ADD X18, X18, #4.
    fixture.put(8, a64::encode_br(18));
    ok &= rejects(fixture, "singleton target proof");
    return ok;
}

bool metadata() {
    Fixture fixture;
    fixture.metadata.symbols.clear();
    fixture.metadata.init_array = 0x2000;
    fixture.metadata.init_array_size = 8;
    fixture.put(0x2000, fixture.base + 0x40, 8);
    fixture.pointers.push_back({0x2000, {0, 0x40}});
    fixture.put(0x40, a64::encode_ret());
    const auto result = analyze(fixture);
    bool ok = check(result.ok() && result.roots == std::vector<Target>{{0, 0x40}},
                    "Linked initializer metadata supplies roots.");
    fixture.pointers.clear();
    ok &= rejects(fixture, "relocation provenance");
    fixture.pointers.push_back({0x2000, {0, 0x44}});
    ok &= rejects(fixture, "relocation provenance");
    fixture.pointers.front().target.image_offset = 0x40;
    fixture.metadata.init_array_size = 0;
    ok &= rejects(fixture, "No supported executable metadata roots");
    fixture.root(0);
    fixture.metadata.symbols.front().size = 0x101;
    ok &= rejects(fixture, "executable extent");
    return ok;
}

bool invalid_inputs() {
    Fixture fixture;
    fixture.put(0, a64::encode_ret());
    fixture.image.segments[1].image_offset = 0;
    bool ok = rejects(fixture, "segment mappings overlap");
    fixture.image.segments[1].image_offset = 0x1000;
    const std::array overlap{fixture.view(), fixture.view()};
    ok &= check(!analyze_modules(overlap).ok(), "Overlapping synthetic images are rejected.");
    fixture.put(0, a64::encode_b(fixture.base, fixture.base + 0x100));
    ok &= rejects(fixture, "outside executable text content");
    fixture.put(0, a64::encode_load_literal(a64::Opcode::ldr_literal_q, 0,
                                           fixture.base, fixture.base + 0x2ffc));
    fixture.put(4, a64::encode_ret());
    ok &= rejects(fixture, "literal reference is outside");
    fixture.put(0, a64::encode_ret());
    fixture.pointers = {{0x1000, {1, 0}}};
    ok &= rejects(fixture, "sorted unique module-bounded");
    fixture.pointers = {{0x1000, {0, 0}}, {0x1000, {0, 4}}};
    ok &= rejects(fixture, "sorted unique module-bounded");
    return ok;
}

bool unwind_roots() {
    Fixture fixture;
    fixture.metadata.symbols.clear();
    constexpr std::array<std::uint64_t, 2> entries{0x40, 0x80};
    fixture.unwind(entries);
    const auto result = analyze(fixture);
    bool ok = check(result.ok() && result.roots == std::vector<Target>{{0, 0x40}, {0, 0x80}} &&
                    result.modules[0].code.size() == 2,
                    "Unwind search-table entries independently root function instructions.");
    fixture.unwind(entries, 0x1040);
    ok &= check(analyze(fixture).ok(), "Negative frame and FDE displacements use their specified bases.");

    fixture.unwind(entries);
    fixture.put(0x1100, 2, 1);
    ok &= rejects(fixture, "version or encodings");
    fixture.unwind(entries);
    fixture.put(0x1101, 0x1c, 1);
    ok &= rejects(fixture, "version or encodings");
    fixture.unwind(entries);
    fixture.put(0x1103, 0x1b, 1);
    ok &= rejects(fixture, "version or encodings");
    fixture.unwind(entries);
    fixture.put(0x1108, 1);
    ok &= rejects(fixture, "table extent");
    fixture.unwind(entries);
    fixture.image.module_header->eh_frame_hdr_end_offset -= 1;
    ok &= rejects(fixture, "table extent");
    fixture.unwind(entries);
    fixture.image.segments[1].permission = loader::SegmentPermission::read | loader::SegmentPermission::write;
    ok &= rejects(fixture, "bounds or permissions");
    fixture.image.segments[1].permission = loader::SegmentPermission::read;

    fixture.unwind(entries);
    fixture.put(0x1104, 0x80000000u);
    ok &= rejects(fixture, "frame pointer");
    fixture.unwind(entries);
    fixture.put(0x1104, 0x7fffffffu);
    ok &= rejects(fixture, "frame pointer");
    fixture.unwind(entries);
    fixture.put(0x110c, 0x80000000u);
    ok &= rejects(fixture, "strictly sorted executable");
    fixture.unwind(entries);
    fixture.put(0x110c, static_cast<std::uint32_t>(0x1000 - 0x1100));
    ok &= rejects(fixture, "strictly sorted executable");
    fixture.unwind(entries);
    fixture.put(0x1114, static_cast<std::uint32_t>(0x40 - 0x1100));
    ok &= rejects(fixture, "strictly sorted executable");
    fixture.unwind(entries);
    fixture.put(0x1118, 0x110);
    ok &= rejects(fixture, "duplicate FDE pointer");
    fixture.unwind(entries);
    fixture.put(0x1110, 0x114);
    ok &= rejects(fixture, "record boundary");
    fixture.unwind(entries);
    fixture.put(0x1110, 0x100);
    ok &= rejects(fixture, "indexes a CIE");
    fixture.unwind(entries);
    fixture.put(0x1110, 0x7fffffffu);
    ok &= rejects(fixture, "FDE pointer");

    fixture.unwind(entries);
    fixture.put(0x1200, 0);
    ok &= rejects(fixture, "invalid framing");
    fixture.unwind(entries);
    fixture.put(0x1210, 0xffffffffu);
    ok &= rejects(fixture, "invalid framing");
    fixture.unwind(entries);
    fixture.put(0x1210, 0x1000);
    ok &= rejects(fixture, "invalid framing");
    fixture.unwind(entries);
    fixture.put(0x1214, 0xffffffffu);
    ok &= rejects(fixture, "backward CIE reference");
    fixture.unwind(entries);
    fixture.put(0x1214, 0x10);
    ok &= rejects(fixture, "backward CIE reference");
    fixture.unwind(entries);
    fixture.put(0x1210, 4);
    ok &= rejects(fixture, "backward CIE reference");
    fixture.unwind(entries);
    fixture.put(0x1104, static_cast<std::uint32_t>(0x10e0 - 0x1104));
    fixture.put(0x10e0, 12);
    fixture.put(0x10e4, 0);
    fixture.put(0x10f0, 20);
    fixture.put(0x10f4, 20);
    ok &= rejects(fixture, "overlap the search header");

    fixture.unwind(entries);
    AnalysisLimits limits;
    limits.max_roots = 1;
    ok &= check(!analyze(fixture, limits).ok(), "Unwind roots consume the metadata budget.");
    limits = {};
    limits.max_unwind_records = 2;
    ok &= check(!analyze(fixture, limits).ok(), "Unwind record traversal has a separate work budget.");
    fixture.unwind({});
    fixture.root(0);
    fixture.put(0, a64::encode_ret());
    ok &= check(analyze(fixture).ok(), "An empty unwind table contributes no roots.");
    return ok;
}
} // namespace

int main() {
    bool ok = coverage();
    ok &= execute_only_text_admission();
    ok &= authenticated_return_provenance();
    ok &= register_addressed_code_reads();
    ok &= address_provenance_escapes();
    ok &= atomic_data_references();
    ok &= guest_relocation_store();
    ok &= targets();
    ok &= cross_module_exclusive_x18_ingress();
    ok &= return_target_contract();
    ok &= saved_return_provenance();
    ok &= native_exclusive_contention();
    ok &= native_exclusive_x18_contention();
    ok &= exclusive_analysis();
    ok &= refusal();
    ok &= later_leader();
    ok &= metadata();
    ok &= invalid_inputs();
    ok &= unwind_roots();
    return ok ? 0 : 1;
}
