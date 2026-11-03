// ============================================================================
// Armigo Tests - Native Runtime Verification
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include <armigo/linker/link_plan.h>

#include <atomic>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <new>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

// Count global allocations so tests can prove that application never allocates.
namespace {
std::atomic<std::size_t> g_allocations{0};
}

void* operator new(std::size_t size) {
    ++g_allocations;
    if (void* memory = std::malloc(size == 0 ? 1 : size)) {
        return memory;
    }
    throw std::bad_alloc();
}
void* operator new[](std::size_t size) {
    ++g_allocations;
    if (void* memory = std::malloc(size == 0 ? 1 : size)) {
        return memory;
    }
    throw std::bad_alloc();
}
void operator delete(void* memory) noexcept { std::free(memory); }
void operator delete[](void* memory) noexcept { std::free(memory); }
void operator delete(void* memory, std::size_t) noexcept { std::free(memory); }
void operator delete[](void* memory, std::size_t) noexcept { std::free(memory); }

namespace {
using namespace armigo::linker;
using Image = std::vector<std::byte>;
static_assert(!std::is_copy_constructible_v<LinkPlan>);
static_assert(!std::is_copy_assignable_v<LinkPlan>);
static_assert(std::is_nothrow_move_constructible_v<LinkPlan>);
static_assert(std::is_nothrow_move_assignable_v<LinkPlan>);

int g_failures = 0;

void check(bool condition, const char* text, int line) {
    if (!condition) {
        std::fprintf(stderr, "link_plan_tests.cpp:%d: check failed: %s\n", line, text);
        ++g_failures;
    }
}
#define CHECK(condition) check((condition), #condition, __LINE__)

// Symbol encodings written from the ELF64 specification.
constexpr std::uint8_t local = 0, global = 1, weak = 2;
constexpr std::uint8_t notype = 0, object = 1, func = 2, section_type = 3, tls = 6, ifunc = 10;
constexpr std::uint8_t internal_vis = 1, hidden_vis = 2, protected_vis = 3;
constexpr std::uint16_t undef = 0, text = 1, abs_section = 0xfff1, common_section = 0xfff2;

Symbol sym(std::string name, std::uint8_t binding, std::uint8_t type, std::uint16_t section,
           std::uint64_t value, std::uint8_t visibility = 0, std::uint64_t size = 0) {
    Symbol symbol;
    symbol.name = std::move(name);
    symbol.info = static_cast<std::uint8_t>((binding << 4) | type);
    symbol.visibility = visibility;
    symbol.section = section;
    symbol.value = value;
    symbol.size = size;
    return symbol;
}

Relocation rel(std::uint64_t target, RelocationKind kind, std::uint32_t symbol,
               std::int64_t addend = 0) {
    return {target, kind, symbol, addend};
}

DynamicModule meta(std::vector<Symbol> symbols, std::vector<Relocation> relocations,
                   std::vector<std::uint64_t> relr = {}) {
    DynamicModule module;
    module.symbols.push_back({});
    for (auto& symbol : symbols) {
        module.symbols.push_back(std::move(symbol));
    }
    module.relocations = std::move(relocations);
    module.packed_relative_targets = std::move(relr);
    return module;
}

constexpr auto abs64 = RelocationKind::absolute64;
constexpr auto glob = RelocationKind::global_data;
constexpr auto jump = RelocationKind::jump_slot;
constexpr auto relative = RelocationKind::relative;

struct Spec {
    std::string name;
    std::uint64_t base = 0;
    std::uint64_t span = 0x1000;
    Image image = Image(0x200);
    DynamicModule metadata;
    std::vector<ModuleRange> writable{{0x100, 0x100}};
};

Spec module(std::string name, std::uint64_t base, DynamicModule metadata) {
    Spec spec;
    spec.name = std::move(name);
    spec.base = base;
    spec.metadata = std::move(metadata);
    return spec;
}

// Little-endian access written independently of the implementation.
std::uint64_t read64(const Image& image, std::size_t offset) {
    std::uint64_t value = 0;
    for (int index = 7; index >= 0; --index) {
        value = (value << 8) | std::to_integer<std::uint64_t>(image.at(offset + index));
    }
    return value;
}

void write64(Image& image, std::size_t offset, std::uint64_t value) {
    for (std::size_t index = 0; index < 8; ++index) {
        image.at(offset + index) = std::byte(static_cast<unsigned char>(value >> (8 * index)));
    }
}

std::vector<LinkModule> view(std::vector<Spec>& specs) {
    std::vector<LinkModule> modules;
    for (auto& spec : specs) {
        modules.push_back({spec.name, spec.image, &spec.metadata, spec.base, spec.span, spec.writable});
    }
    return modules;
}

PlanResult plan(std::vector<Spec>& specs, const LinkOptions& options = {},
                const LinkLimits& limits = {}) {
    std::vector<Image> before;
    for (const auto& spec : specs) {
        before.push_back(spec.image);
    }
    const auto modules = view(specs);
    auto result = plan_link(modules, options, limits);
    for (std::size_t index = 0; index < specs.size(); ++index) {
        CHECK(specs[index].image == before[index]);
    }
    CHECK(result.ok() == (result.error == LinkError::none));
    return result;
}

std::vector<std::span<std::byte>> spans(std::vector<Spec>& specs) {
    std::vector<std::span<std::byte>> result;
    for (auto& spec : specs) {
        result.emplace_back(spec.image);
    }
    return result;
}

ApplyResult apply_plan(const LinkPlan& link_plan, std::vector<std::span<std::byte>> images) {
    const auto start = g_allocations.load();
    const auto result = link_plan.apply(images);
    CHECK(g_allocations.load() == start);
    return result;
}

bool link(std::vector<Spec>& specs, const LinkOptions& options = {}) {
    auto result = plan(specs, options);
    if (!result.ok()) {
        std::fprintf(stderr, "unexpected plan failure: %s\n", result.message.c_str());
        return false;
    }
    const auto applied = apply_plan(*result.plan, spans(specs));
    if (!applied.ok()) {
        std::fprintf(stderr, "unexpected apply failure: %s\n", applied.message);
    }
    return applied.ok();
}

LinkError plan_error(std::vector<Spec>& specs, const LinkOptions& options = {},
                     const LinkLimits& limits = {}) {
    return plan(specs, options, limits).error;
}

std::vector<Spec> three_modules() {
    std::vector<Spec> specs;
    specs.push_back(module("a", 0x10000000,
        meta({sym("shared", global, func, text, 0x10), sym("from_b", global, func, undef, 0),
              sym("from_c", global, func, undef, 0)},
             {rel(0x100, glob, 2), rel(0x108, jump, 3), rel(0x110, abs64, 1, 4)})));
    specs.push_back(module("b", 0x20000000,
        meta({sym("from_b", global, func, text, 0x20), sym("from_c", global, func, text, 0x28),
              sym("shared", global, func, text, 0x30)},
             {rel(0x100, abs64, 3)})));
    specs.push_back(module("c", 0x30000000,
        meta({sym("from_c", global, func, text, 0x40), sym("from_b", global, object, undef, 0),
              sym("shared", global, func, undef, 0)},
             {rel(0x100, glob, 2), rel(0x108, glob, 3)})));
    return specs;
}

void test_load_order() {
    auto specs = three_modules();
    CHECK(link(specs));
    CHECK(read64(specs[0].image, 0x100) == 0x20000020);
    CHECK(read64(specs[0].image, 0x108) == 0x20000028); // B precedes C.
    CHECK(read64(specs[0].image, 0x110) == 0x10000014);
    CHECK(read64(specs[1].image, 0x100) == 0x10000010); // Preempted by A.
    CHECK(read64(specs[2].image, 0x100) == 0x20000020);
    CHECK(read64(specs[2].image, 0x108) == 0x10000010);

    specs = three_modules();
    auto result = plan(specs);
    CHECK(result.ok());
    const auto writes = result.plan->writes(1);
    CHECK(writes.size() == 1 && writes[0].source == WriteSource::module_symbol &&
          writes[0].provider == 0 && writes[0].provider_symbol == 1 && writes[0].symbol == 3);

    specs = three_modules();
    LinkOptions options;
    options.preemption = PreemptionModel::own_definition_first;
    CHECK(link(specs, options));
    CHECK(read64(specs[1].image, 0x100) == 0x20000030); // Own definition first.
    CHECK(read64(specs[2].image, 0x108) == 0x10000010); // Undefined: still load order.
    CHECK(read64(specs[0].image, 0x108) == 0x20000028);
}

void test_untouched_bytes() {
    // Untouched bytes outside relocation targets must remain intact.
    std::vector<Spec> specs;
    specs.push_back(module("main", 0x40000000,
        meta({sym("import", global, func, undef, 0)}, {rel(0x180, jump, 1)})));
    specs.push_back(module("lib", 0x40100000,
        meta({sym("import", global, func, text, 0x1c0, 0, 8)}, {})));
    for (std::size_t index = 0; index < 0x200; ++index) {
        specs[0].image[index] = std::byte(index & 0xff);
    }
    const auto original = specs[0].image;
    CHECK(link(specs));
    CHECK(read64(specs[0].image, 0x180) == 0x401001c0);
    for (std::size_t index = 0; index < 0x200; ++index) {
        if (index < 0x180 || index >= 0x188) {
            CHECK(specs[0].image[index] == original[index]);
        }
    }
}

void test_weak_and_unresolved() {
    std::vector<Spec> specs;
    specs.push_back(module("a", 0x10000000,
        meta({sym("maybe", weak, notype, undef, 0), sym("missing", global, func, undef, 0),
              sym("missing2", global, object, undef, 0), sym("provided", weak, func, undef, 0)},
             {rel(0x100, glob, 1), rel(0x108, abs64, 1, 0x10), rel(0x110, glob, 2),
              rel(0x118, jump, 2), rel(0x120, jump, 3), rel(0x128, jump, 4)})));
    specs.push_back(module("b", 0x20000000,
        meta({sym("provided", weak, func, text, 0x50)}, {})));
    auto result = plan(specs);
    CHECK(result.error == LinkError::unresolved_symbol);
    CHECK(!result.plan.has_value());
    CHECK(result.unresolved_count == 2 && result.unresolved.size() == 2);
    CHECK(result.unresolved.size() == 2 && result.unresolved[0].name == "missing" &&
          result.unresolved[0].module == 0 && result.unresolved[0].symbol == 2 &&
          result.unresolved[1].name == "missing2");
    CHECK(result.message.find("missing") != std::string::npos);

    LinkLimits limits;
    limits.max_reported_unresolved = 1;
    result = plan(specs, {}, limits);
    CHECK(result.unresolved_count == 2 && result.unresolved.size() == 1);
    limits.max_reported_unresolved = 0;
    result = plan(specs, {}, limits);
    CHECK(result.error == LinkError::unresolved_symbol && result.unresolved_count == 2 &&
          result.unresolved.empty());

    const HostBinding host[] = {{"missing", 0x7000000000}, {"missing2", 0x7000000100}};
    LinkOptions options;
    options.fallback = host;
    CHECK(link(specs, options));
    CHECK(read64(specs[0].image, 0x100) == 0);
    CHECK(read64(specs[0].image, 0x108) == 0x10); // S is zero; the addend remains.
    CHECK(read64(specs[0].image, 0x110) == 0x7000000000);
    CHECK(read64(specs[0].image, 0x118) == 0x7000000000);
    CHECK(read64(specs[0].image, 0x120) == 0x7000000100);
    CHECK(read64(specs[0].image, 0x128) == 0x20000050); // A weak definition binds.
}

void test_visibility() {
    std::vector<Spec> specs;
    specs.push_back(module("a", 0x10000000,
        meta({sym("loc", global, object, text, 0x10), sym("hid", global, object, text, 0x18),
              sym("pro", global, object, text, 0x20), sym("int", global, object, text, 0x28)},
             {})));
    specs.push_back(module("b", 0x20000000,
        meta({sym("loc", local, object, text, 0x40), sym("hid", global, object, text, 0x48, hidden_vis),
              sym("pro", global, object, text, 0x50, protected_vis),
              sym("int", global, object, text, 0x58, internal_vis),
              sym("", local, section_type, text, 0x60),
              sym("hidden_only", global, func, text, 0x70, hidden_vis),
              sym("pro_b", global, func, text, 0x78, protected_vis),
              sym("local_only", local, func, text, 0x80)},
             {rel(0x100, abs64, 1), rel(0x108, abs64, 2), rel(0x110, abs64, 3),
              rel(0x118, abs64, 4), rel(0x120, abs64, 5, 8)})));
    specs.push_back(module("c", 0x30000000,
        meta({sym("pro_b", global, func, undef, 0), sym("hidden_only", weak, func, undef, 0),
              sym("local_only", weak, func, undef, 0)},
             {rel(0x100, glob, 1), rel(0x108, glob, 2), rel(0x110, glob, 3)})));
    const HostBinding forced[] = {{"hid", 0x7000000000}};
    LinkOptions options;
    options.forced = forced;
    auto result = plan(specs, options);
    CHECK(result.error == LinkError::unused_forced_binding);
    CHECK(result.unused_forced == std::vector<std::string>{"hid"});
    options.require_forced_bindings_used = false;
    result = plan(specs, options);
    CHECK(result.ok() && result.plan->forced_binding_uses().size() == 1 &&
          result.plan->forced_binding_uses()[0] == 0);
    CHECK(link(specs, options));
    CHECK(read64(specs[1].image, 0x100) == 0x20000040);
    CHECK(read64(specs[1].image, 0x108) == 0x20000048);
    CHECK(read64(specs[1].image, 0x110) == 0x20000050);
    CHECK(read64(specs[1].image, 0x118) == 0x20000058);
    CHECK(read64(specs[1].image, 0x120) == 0x20000068);
    CHECK(read64(specs[2].image, 0x100) == 0x20000078); // Protected is exported.
    CHECK(read64(specs[2].image, 0x108) == 0);          // Hidden is not exported.
    CHECK(read64(specs[2].image, 0x110) == 0);          // Local is not exported.
}

void test_absolute() {
    std::vector<Spec> specs;
    specs.push_back(module("a", 0x10000000,
        meta({sym("abs_g", global, notype, abs_section, 0x1234),
              sym("abs_l", local, notype, abs_section, 0xabcd0000)},
             {rel(0x100, abs64, 2, 1), rel(0x108, glob, 1)})));
    specs.push_back(module("b", 0x20000000,
        meta({sym("abs_g", global, notype, undef, 0)}, {rel(0x100, glob, 1, -4)})));
    auto result = plan(specs);
    CHECK(result.ok() && result.plan->writes(1)[0].source == WriteSource::absolute_symbol);
    CHECK(link(specs));
    CHECK(read64(specs[0].image, 0x100) == 0xabcd0001); // No base is added.
    CHECK(read64(specs[0].image, 0x108) == 0x1234);
    CHECK(read64(specs[1].image, 0x100) == 0x1230);
}

void test_host_precedence() {
    auto make = [] {
        std::vector<Spec> specs;
        specs.push_back(module("a", 0x10000000,
            meta({sym("f", global, func, text, 0x10), sym("g", global, func, undef, 0),
                  sym("h", global, func, text, 0x18)},
                 {rel(0x100, jump, 1), rel(0x108, jump, 2), rel(0x110, jump, 3)})));
        specs.push_back(module("b", 0x20000000,
            meta({sym("f", global, func, undef, 0), sym("h", global, func, undef, 0)},
                 {rel(0x100, jump, 1), rel(0x108, jump, 2)})));
        return specs;
    };
    const HostBinding forced[] = {{"f", 0x700000000000}};
    const HostBinding fallback[] = {{"g", 0x710000000000}, {"h", 0x720000000000}};
    LinkOptions options;
    options.forced = forced;
    options.fallback = fallback;
    auto specs = make();
    auto result = plan(specs, options);
    CHECK(result.ok() && result.plan->forced_binding_uses()[0] == 2);
    CHECK(result.ok() && result.plan->writes(0)[0].source == WriteSource::forced_host &&
          result.plan->writes(0)[1].source == WriteSource::host_fallback &&
          result.plan->writes(0)[1].provider == 0);
    CHECK(link(specs, options));
    CHECK(read64(specs[0].image, 0x100) == 0x700000000000); // Forced overrides definition.
    CHECK(read64(specs[0].image, 0x108) == 0x710000000000);
    CHECK(read64(specs[0].image, 0x110) == 0x10000018);     // Module beats fallback.
    CHECK(read64(specs[1].image, 0x100) == 0x700000000000);
    CHECK(read64(specs[1].image, 0x108) == 0x10000018);

    const HostBinding unused[] = {{"f", 0x700000000000}, {"zzz", 0x700000000100}};
    options.forced = unused;
    specs = make();
    result = plan(specs, options);
    CHECK(result.error == LinkError::unused_forced_binding &&
          result.unused_forced == std::vector<std::string>{"zzz"});

    const HostBinding duplicate[] = {{"f", 1}, {"f", 2}};
    options.forced = duplicate;
    CHECK(plan_error(specs, options) == LinkError::duplicate_host_binding);
    const HostBinding null_address[] = {{"g", 0}};
    options.forced = {};
    options.fallback = null_address;
    CHECK(plan_error(specs, options) == LinkError::invalid_argument);
    const HostBinding empty_name[] = {{"", 1}};
    options.fallback = empty_name;
    CHECK(plan_error(specs, options) == LinkError::invalid_argument);
    const std::string long_name(5000, 'x');
    const HostBinding too_long[] = {{long_name, 1}};
    options.fallback = too_long;
    CHECK(plan_error(specs, options) == LinkError::resource_limit);
}

void test_signed_addends() {
    std::vector<Spec> specs;
    specs.push_back(module("a", 0x10000000,
        meta({sym("obj", global, object, text, 0x40)},
             {rel(0x100, relative, 0, -0x10), rel(0x108, relative, 0, 0x7fff),
              rel(0x110, abs64, 1, -0x40), rel(0x118, jump, 1, 4)})));
    CHECK(link(specs));
    CHECK(read64(specs[0].image, 0x100) == 0x0ffffff0);
    CHECK(read64(specs[0].image, 0x108) == 0x10007fff);
    CHECK(read64(specs[0].image, 0x110) == 0x10000000);
    CHECK(read64(specs[0].image, 0x118) == 0x10000044); // JUMP_SLOT is S + A.

    for (const auto addend : {std::int64_t{-0x10000001}, std::numeric_limits<std::int64_t>::min()}) {
        std::vector<Spec> overflow;
        overflow.push_back(module("a", 0x10000000, meta({}, {rel(0x100, relative, 0, addend)})));
        CHECK(plan_error(overflow) == LinkError::address_overflow);
    }
    std::vector<Spec> overflow;
    overflow.push_back(module("a", 0x10000000,
        meta({sym("top", local, notype, abs_section, 0xfffffffffffffff0ull)},
             {rel(0x100, abs64, 1, 0x20)})));
    CHECK(plan_error(overflow) == LinkError::address_overflow);
    overflow.clear();
    overflow.push_back(module("a", 0x10000000,
        meta({sym("weak", weak, notype, undef, 0)}, {rel(0x100, abs64, 1, -1)})));
    CHECK(plan_error(overflow) == LinkError::address_overflow);
}

void test_relr() {
    std::vector<Spec> specs;
    specs.push_back(module("a", 0x10000000, meta({}, {rel(0x108, relative, 0, 0x20)},
                                                 {0x100, 0x110, 0x118})));
    write64(specs[0].image, 0x100, 0x40);
    write64(specs[0].image, 0x110, 0x7fffffff);
    write64(specs[0].image, 0x118, 0);
    auto result = plan(specs);
    CHECK(result.ok() && result.plan->writes(0).size() == 4 &&
          result.plan->writes(0)[0].source == WriteSource::packed_relative &&
          result.plan->writes(0)[0].expected == 0x40 &&
          result.plan->writes(0)[1].source == WriteSource::relative);
    CHECK(link(specs));
    CHECK(read64(specs[0].image, 0x100) == 0x10000040);
    CHECK(read64(specs[0].image, 0x108) == 0x10000020);
    CHECK(read64(specs[0].image, 0x110) == 0x8fffffff);
    CHECK(read64(specs[0].image, 0x118) == 0x10000000);

    std::vector<Spec> overflow;
    overflow.push_back(module("a", 0xfffffffffff00000ull, meta({}, {}, {0x100})));
    overflow[0].span = 0x1000;
    write64(overflow[0].image, 0x100, 0x200000);
    CHECK(plan_error(overflow) == LinkError::address_overflow);
}

void test_targets() {
    auto one = [](std::vector<Relocation> relocations, std::vector<std::uint64_t> relr = {}) {
        std::vector<Spec> specs;
        specs.push_back(module("a", 0x10000000,
                               meta({sym("x", global, object, text, 0x10)},
                                    std::move(relocations), std::move(relr))));
        return specs;
    };
    auto specs = one({rel(0x80, relative, 0)});
    CHECK(plan_error(specs) == LinkError::forbidden_target);
    specs = one({}, {0x80});
    CHECK(plan_error(specs) == LinkError::forbidden_target);
    specs = one({rel(0x104, relative, 0)});
    CHECK(plan_error(specs) == LinkError::invalid_metadata);
    specs = one({rel(0x200, abs64, 1)});
    CHECK(plan_error(specs) == LinkError::invalid_metadata);
    specs = one({rel(0x1f8, abs64, 1)});
    specs[0].writable = {{0x100, 0xfc}};
    CHECK(plan_error(specs) == LinkError::forbidden_target);
    specs = one({rel(0x100, abs64, 1)});
    specs[0].writable.clear();
    CHECK(plan_error(specs) == LinkError::forbidden_target);
    specs = one({rel(0x100, abs64, 1)});
    specs[0].writable = {{0x104, 0xc}, {0x100, 0x6}};
    CHECK(link(specs));
    CHECK(read64(specs[0].image, 0x100) == 0x10000010);
    specs = one({});
    specs[0].writable = {{0x100, 0x101}};
    CHECK(plan_error(specs) == LinkError::invalid_argument);
    specs = one({});
    specs[0].writable = {{0x100, 0}};
    CHECK(plan_error(specs) == LinkError::invalid_argument);
    specs = one({rel(0x100, abs64, 1), rel(0x100, relative, 0)});
    CHECK(plan_error(specs) == LinkError::conflicting_write);
    specs = one({rel(0x108, abs64, 1)}, {0x108});
    CHECK(plan_error(specs) == LinkError::conflicting_write);
    specs = one({}, {0x110, 0x110});
    CHECK(plan_error(specs) == LinkError::conflicting_write);
}

void test_malformed_metadata() {
    auto one = [](std::vector<Symbol> symbols, std::vector<Relocation> relocations) {
        std::vector<Spec> specs;
        specs.push_back(module("a", 0x10000000, meta(std::move(symbols), std::move(relocations))));
        return specs;
    };
    auto specs = one({}, {rel(0x100, abs64, 1)});
    CHECK(plan_error(specs) == LinkError::invalid_metadata);
    specs = one({sym("x", global, object, text, 0)}, {rel(0x100, relative, 1)});
    CHECK(plan_error(specs) == LinkError::invalid_metadata);
    specs = one({}, {rel(0x100, static_cast<RelocationKind>(1031), 0)});
    CHECK(plan_error(specs) == LinkError::unsupported_relocation);
    specs = one({sym("t", global, tls, text, 0)}, {rel(0x100, abs64, 1)});
    auto result = plan(specs);
    CHECK(result.error == LinkError::unsupported_symbol &&
          result.message.find("thread-local") != std::string::npos);
    specs = one({sym("i", global, ifunc, text, 0)}, {rel(0x100, jump, 1)});
    result = plan(specs);
    CHECK(result.error == LinkError::unsupported_symbol &&
          result.message.find("indirect function") != std::string::npos);
    specs = one({sym("c", global, object, common_section, 8)}, {rel(0x100, glob, 1)});
    CHECK(plan_error(specs) == LinkError::unsupported_symbol);
    specs = one({sym("u", 10, object, text, 8)}, {rel(0x100, glob, 1)});
    CHECK(plan_error(specs) == LinkError::unsupported_symbol);
    specs = one({sym("h", global, object, undef, 0, hidden_vis)}, {rel(0x100, glob, 1)});
    CHECK(plan_error(specs) == LinkError::invalid_symbol);
    specs = one({sym("l", local, object, undef, 0)}, {rel(0x100, glob, 1)});
    CHECK(plan_error(specs) == LinkError::invalid_symbol);
    specs = one({sym("", global, object, undef, 0)}, {rel(0x100, glob, 1)});
    CHECK(plan_error(specs) == LinkError::invalid_symbol);
    specs = one({sym("far", global, object, text, 0x1000, 0, 8)}, {rel(0x100, glob, 1)});
    CHECK(plan_error(specs) == LinkError::invalid_symbol);
    specs = one({sym("edge", global, object, text, 0xff8, 0, 8)}, {rel(0x100, glob, 1)});
    CHECK(plan_error(specs) == LinkError::none); // Ends exactly at the span.

    // A plain reference binding to a thread-local definition elsewhere.
    std::vector<Spec> pair;
    pair.push_back(module("a", 0x10000000,
        meta({sym("v", global, object, undef, 0)}, {rel(0x100, glob, 1)})));
    pair.push_back(module("b", 0x20000000, meta({sym("v", global, tls, text, 0)}, {})));
    CHECK(plan_error(pair) == LinkError::unsupported_symbol);

    specs = one({}, {});
    specs[0].metadata.symbols.clear();
    CHECK(plan_error(specs) == LinkError::invalid_metadata);
    specs = one({}, {});
    specs[0].metadata.symbols[0].name = "named";
    CHECK(plan_error(specs) == LinkError::invalid_metadata);
    specs = one({}, {});
    auto modules = view(specs);
    modules[0].metadata = nullptr;
    CHECK(plan_link(modules).error == LinkError::invalid_argument);
    CHECK(plan_link(std::span<const LinkModule>{}).error == LinkError::invalid_argument);
}

void test_layout() {
    auto two = [] {
        std::vector<Spec> specs;
        specs.push_back(module("a", 0x10000000, meta({}, {})));
        specs.push_back(module("b", 0x20000000, meta({}, {})));
        return specs;
    };
    auto specs = two();
    specs[0].span = 0x2000;
    specs[1].base = 0x10001000;
    CHECK(plan_error(specs) == LinkError::invalid_layout);
    specs = two();
    specs[1].base = 0x10001000; // Adjacent ranges are valid.
    CHECK(plan_error(specs) == LinkError::none);
    specs = two();
    specs[0].base = 0x10000010;
    CHECK(plan_error(specs) == LinkError::invalid_layout);
    specs = two();
    specs[0].span = 0x100;
    CHECK(plan_error(specs) == LinkError::invalid_layout);
    specs = two();
    specs[1].base = 0xfffffffffffff000ull;
    CHECK(plan_error(specs) == LinkError::address_overflow);
    specs = two();
    specs[0].image.clear();
    CHECK(plan_error(specs) == LinkError::invalid_argument);

    LinkLimits limits;
    limits.max_modules = 1;
    specs = two();
    CHECK(plan_error(specs, {}, limits) == LinkError::resource_limit);

    // Input images that alias each other are rejected; adjacent ones are not.
    Image shared(0x400);
    const auto empty = meta({}, {});
    const std::vector<ModuleRange> none;
    auto aliased = [&](std::size_t second_offset) {
        const LinkModule modules[] = {
            {"a", std::span<const std::byte>(shared).subspan(0, 0x200), &empty, 0x10000000, 0x1000, none},
            {"b", std::span<const std::byte>(shared).subspan(second_offset, 0x200), &empty, 0x20000000,
             0x1000, none}};
        return plan_link(modules).error;
    };
    CHECK(aliased(0x100) == LinkError::aliased_images);
    CHECK(aliased(0) == LinkError::aliased_images);
    CHECK(aliased(0x200) == LinkError::none);
}

void test_limits() {
    std::vector<Spec> specs;
    specs.push_back(module("a", 0x10000000,
        meta({}, {rel(0x100, relative, 0), rel(0x108, relative, 0)}, {0x110})));
    LinkLimits limits;
    limits.max_writes = 2;
    CHECK(plan_error(specs, {}, limits) == LinkError::resource_limit);
    limits.max_writes = 3;
    CHECK(plan_error(specs, {}, limits) == LinkError::none);
    limits = {};
    limits.max_total_symbols = 0;
    CHECK(plan_error(specs, {}, limits) == LinkError::resource_limit);
    limits = {};
    limits.max_writable_ranges = 0;
    CHECK(plan_error(specs, {}, limits) == LinkError::resource_limit);
    limits = {};
    limits.max_host_bindings = 0;
    const HostBinding fallback[] = {{"x", 1}};
    LinkOptions options;
    options.fallback = fallback;
    CHECK(plan_error(specs, options, limits) == LinkError::resource_limit);
}

void test_transactional_apply() {
    auto specs = three_modules();
    write64(specs[2].image, 0x1f0, 0x5555);
    auto result = plan(specs);
    CHECK(result.ok());
    if (!result.ok()) {
        return;
    }
    const auto& link_plan = *result.plan;
    const std::vector<Image> original = {specs[0].image, specs[1].image, specs[2].image};
    auto unchanged = [&] {
        return specs[0].image == original[0] && specs[1].image == original[1] &&
               specs[2].image == original[2];
    };

    auto images = spans(specs);
    CHECK(apply_plan(link_plan, {images[0], images[1]}).error == ApplyError::module_count_mismatch);
    CHECK(unchanged());
    CHECK(apply_plan(link_plan, {images[0], images[1], images[2].first(0x100)}).error ==
          ApplyError::image_size_mismatch);
    CHECK(unchanged());
    const auto aliased = apply_plan(link_plan, {images[0], images[1], images[1]});
    CHECK(aliased.error == ApplyError::aliased_images && aliased.module == 2);
    CHECK(unchanged());

    // A stale target in the last module must leave earlier modules untouched.
    write64(specs[2].image, 0x108, 0x99);
    const auto stale = apply_plan(link_plan, spans(specs));
    CHECK(stale.error == ApplyError::image_mismatch && stale.module == 2);
    write64(specs[2].image, 0x108, 0);
    CHECK(unchanged());

    // So must a change outside every relocation target.
    specs[2].image[0x1f0] = std::byte{0x56};
    CHECK(apply_plan(link_plan, spans(specs)).error == ApplyError::image_mismatch);
    specs[2].image[0x1f0] = std::byte{0x55};
    CHECK(unchanged());

    // Application to an equal copy in different storage succeeds.
    std::vector<Image> copies = original;
    std::vector<std::span<std::byte>> copy_spans(copies.begin(), copies.end());
    CHECK(apply_plan(link_plan, copy_spans).ok());
    CHECK(unchanged());
    CHECK(read64(copies[0], 0x100) == 0x20000020 && read64(copies[2], 0x108) == 0x10000010);
    CHECK(read64(copies[2], 0x1f0) == 0x5555);
    const auto applied = copies;
    CHECK(apply_plan(link_plan, copy_spans).error == ApplyError::image_mismatch);
    CHECK(copies == applied);
}

// The full path from a synthetic MOD0 image through the metadata reader.
struct ImageBuilder {
    Image image = Image(0x1000);
    void put(std::size_t offset, std::uint64_t value, unsigned width = 8) {
        for (unsigned index = 0; index < width; ++index) {
            image.at(offset + index) = std::byte((value >> (index * 8)) & 0xff);
        }
    }
    // Symbols are (name, info, section, value, size); names are packed at 0x400.
    void build(const std::vector<std::tuple<std::string, std::uint8_t, std::uint16_t, std::uint64_t,
                                            std::uint64_t>>& symbols,
               const std::vector<std::tuple<std::uint64_t, std::uint32_t, std::uint32_t, std::int64_t>>& rela,
               const std::vector<std::uint64_t>& relr) {
        put(4, 0x40, 4);
        put(0x40, 0x30444f4d, 4);
        put(0x44, 0x40, 4);
        const std::size_t count = symbols.size() + 1;
        std::size_t names = 1;
        for (std::size_t index = 0; index < symbols.size(); ++index) {
            const auto& [name, info, section, value, size] = symbols[index];
            const auto at = 0x500 + (index + 1) * 24;
            put(at, names, 4);
            put(at + 4, info, 1);
            put(at + 6, section, 2);
            put(at + 8, value);
            put(at + 16, size);
            for (const char character : name) {
                image.at(0x400 + names++) = std::byte(character);
            }
            ++names;
        }
        for (std::size_t index = 0; index < rela.size(); ++index) {
            const auto& [target, kind, symbol, addend] = rela[index];
            put(0x600 + index * 24, target);
            put(0x608 + index * 24, (std::uint64_t{symbol} << 32) | kind);
            put(0x610 + index * 24, static_cast<std::uint64_t>(addend));
        }
        for (std::size_t index = 0; index < relr.size(); ++index) {
            put(0x700 + index * 8, relr[index]);
        }
        put(0x300, 1, 4);
        put(0x304, count, 4);
        const std::pair<std::uint64_t, std::uint64_t> tags[] = {
            {4, 0x300}, {5, 0x400}, {10, names}, {6, 0x500}, {11, 24},
            {7, 0x600}, {8, rela.size() * 24}, {9, 24},
            {36, 0x700}, {35, relr.size() * 8}, {37, 8}};
        std::size_t at = 0x80;
        for (const auto& [tag, value] : tags) {
            put(at, tag);
            put(at + 8, value);
            at += 16;
        }
    }
};

void test_parsed_modules() {
    ImageBuilder main_image;
    main_image.build({{"service", 0x12, 0, 0, 0}},
                     {{0x800, 1025, 1, 0}, {0x808, 1027, 0, 0x810}}, {0x810});
    main_image.put(0x810, 0x900);
    ImageBuilder library;
    library.build({{"service", 0x12, 1, 0x900, 0x10}}, {}, {});
    const auto main_metadata = read_dynamic_module(main_image.image);
    const auto library_metadata = read_dynamic_module(library.image);
    CHECK(main_metadata.ok() && library_metadata.ok());
    if (!main_metadata.ok() || !library_metadata.ok()) {
        return;
    }
    const ModuleRange data[] = {{0x800, 0x800}};
    const LinkModule modules[] = {
        {"main", main_image.image, &*main_metadata.module, 0x8000000, 0x2000, data},
        {"library", library.image, &*library_metadata.module, 0x8004000, 0x1000, data}};
    const auto result = plan_link(modules);
    CHECK(result.ok());
    if (!result.ok()) {
        return;
    }
    std::span<std::byte> images[] = {main_image.image, library.image};
    CHECK(apply_plan(*result.plan, {images[0], images[1]}).ok());
    CHECK(read64(main_image.image, 0x800) == 0x8004900);
    CHECK(read64(main_image.image, 0x808) == 0x8000810);
    CHECK(read64(main_image.image, 0x810) == 0x8000900);
}

void test_plan_invariants() {
    std::vector<Spec> specs = {
        module("first", 0x10000, meta({}, {rel(0x100, relative, 0, 8)})),
        module("second", 0x20000, meta({}, {rel(0x100, relative, 0, 16)})),
    };
    LinkOptions invalid;
    invalid.preemption = static_cast<PreemptionModel>(255);
    CHECK(plan(specs, invalid).error == LinkError::invalid_argument);
    for (unsigned field = 0; field < 4; ++field) {
        auto malformed = specs;
        auto& zero = malformed[0].metadata.symbols[0];
        if (field == 0) zero.value = 8;
        if (field == 1) zero.size = 8;
        if (field == 2) zero.info = 0x10;
        if (field == 3) zero.visibility = 1;
        CHECK(plan(malformed).error == LinkError::invalid_metadata);
    }

    // Descriptors are ordinary typed objects deliberately sharing a test
    // image's storage. Applying relocations must reject this layout unchanged.
    using Descriptors = std::array<std::span<std::byte>, 2>;
    static_assert(sizeof(Descriptors) < 0x100);
    auto* descriptors = new (specs[0].image.data() + 0x100)
        Descriptors{specs[0].image, specs[1].image};
    const auto aliased = plan(specs);
    CHECK(aliased.ok());
    const auto before = specs[0].image;
    if (aliased.ok()) {
        CHECK(aliased.plan->apply(*descriptors).error == ApplyError::aliased_descriptors);
        CHECK(specs[0].image == before);
    }
    descriptors->~Descriptors();

    auto first = plan(specs);
    auto second = plan(specs);
    CHECK(first.ok() && second.ok());
    if (first.ok() && second.ok()) {
        *second.plan = std::move(*first.plan);
        CHECK(first.plan->module_count() == 0);
        CHECK(second.plan->module_count() == 2);
        auto* self = &*second.plan;
        *self = std::move(*self);
        CHECK(second.plan->module_count() == 2);
        CHECK(apply_plan(*second.plan, {specs[0].image, specs[1].image}).ok());
        CHECK(read64(specs[0].image, 0x100) == 0x10008);
        CHECK(read64(specs[1].image, 0x100) == 0x20010);
    }
}

} // namespace

int main() {
    test_load_order();
    test_untouched_bytes();
    test_weak_and_unresolved();
    test_visibility();
    test_absolute();
    test_host_precedence();
    test_signed_addends();
    test_relr();
    test_targets();
    test_malformed_metadata();
    test_layout();
    test_limits();
    test_transactional_apply();
    test_parsed_modules();
    test_plan_invariants();
    if (g_failures != 0) {
        std::fprintf(stderr, "%d link plan check(s) failed.\n", g_failures);
        return 1;
    }
    std::puts("Link plan binding, relocation, validation, and transaction checks passed.");
    return 0;
}
