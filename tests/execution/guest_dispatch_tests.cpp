// ============================================================================
// Armigo Tests - Native Runtime Verification
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

// Executable verification of prepared gate-site dispatch on Apple Silicon.
//
// Site tables are built from synthetic instruction words produced by the
// checked encoders in armigo/aarch64. Dispatch scenarios enter the real host
// gate from the authored fixtures in guest_dispatch_fixtures.S and compare
// the registers that come back against the exact changes each site may make.
// Process-fatal behaviors run in a child copy of this executable.

#include <armigo/execution/guest_dispatch.h>
#include <armigo/execution/gate_code.h>

#include <armigo/aarch64/instruction.h>

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <initializer_list>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <fcntl.h>
#include <mach-o/dyld.h>
#include <signal.h>
#include <spawn.h>
#include <sys/wait.h>
#include <unistd.h>

extern char** environ;

namespace {

using namespace armigo::execution;
namespace a64 = armigo::aarch64;

// ---------------------------------------------------------------------------
// Fixture interface (guest_dispatch_fixtures.S)
// ---------------------------------------------------------------------------

struct GuestImage {
    std::uint64_t x[31];
    std::uint64_t nzcv;
    std::uint64_t x18_before;
    std::uint64_t x18_after;
    std::uint64_t sp_before;
    std::uint64_t sp_after;
};

static_assert(offsetof(GuestImage, nzcv) == 248);
static_assert(offsetof(GuestImage, x18_before) == 256);
static_assert(offsetof(GuestImage, x18_after) == 264);
static_assert(offsetof(GuestImage, sp_before) == 272);
static_assert(offsetof(GuestImage, sp_after) == 280);

extern "C" {
void armigo_test_guest_run_site(const GuestImage* in, GuestImage* out, std::uint64_t site);
void armigo_test_guest_sites();
void armigo_test_guest_sequence(std::uint64_t value, std::uint64_t* out);
void armigo_test_guest_sequence_return_0();
void armigo_test_guest_sequence_return_1();
void armigo_test_guest_sequence_return_2();
void armigo_test_guest_sequence_return_3();
void armigo_test_guest_sequence_return_4();
void armigo_test_guest_sequence_return_5();
}

constexpr unsigned kFixtureSites = 16;

std::uint64_t site_entry(unsigned index) {
    return reinterpret_cast<std::uint64_t>(&armigo_test_guest_sites) + 8u * index;
}

std::uint64_t site_return(unsigned index) {
    return site_entry(index) + 4;
}

std::uint64_t address_of(void (*symbol)()) {
    return reinterpret_cast<std::uint64_t>(symbol);
}

// ---------------------------------------------------------------------------
// Minimal assertion harness
// ---------------------------------------------------------------------------

int g_checks_failed = 0;
int g_checks_total = 0;

void record_check(bool passed, const char* text, const char* file, int line) {
    ++g_checks_total;
    if (!passed) {
        ++g_checks_failed;
        std::printf("FAIL %s:%d: %s\n", file, line, text);
    }
}

void record_equal(std::uint64_t actual, std::uint64_t expected, const char* text, const char* file,
                  int line) {
    ++g_checks_total;
    if (actual != expected) {
        ++g_checks_failed;
        std::printf("FAIL %s:%d: %s: actual 0x%llx expected 0x%llx\n", file, line, text,
                    static_cast<unsigned long long>(actual),
                    static_cast<unsigned long long>(expected));
    }
}

#define CHECK(condition) record_check((condition), #condition, __FILE__, __LINE__)
#define CHECK_EQ(actual, expected) \
    record_equal(static_cast<std::uint64_t>(actual), static_cast<std::uint64_t>(expected), \
                 #actual " == " #expected, __FILE__, __LINE__)

// ---------------------------------------------------------------------------
// Instruction words
// ---------------------------------------------------------------------------

std::uint16_t sysreg(a64::SystemRegister reg) {
    return *a64::encoding_of(reg);
}

std::uint32_t mrs(unsigned rt, a64::SystemRegister reg) {
    const a64::Encoded encoded = a64::encode_mrs(rt, sysreg(reg));
    CHECK(encoded.ok());
    return encoded.word;
}

// DC CIVAC, Xt and IC IVAU, Xt: 1101010100 0 0 1 op1 CRn CRm op2 Rt.
std::uint32_t cache_instruction(unsigned op1, unsigned crn, unsigned crm, unsigned op2,
                                unsigned rt) {
    return 0xD5080000u | ((op1 & 7u) << 16) | ((crn & 0xfu) << 12) | ((crm & 0xfu) << 8) |
           ((op2 & 7u) << 5) | (rt & 31u);
}

std::uint32_t clean_invalidate_data(unsigned rt) {
    return cache_instruction(3, 7, 14, 1, rt);
}

std::uint32_t invalidate_instruction(unsigned rt) {
    return cache_instruction(3, 7, 5, 1, rt);
}

std::uint32_t msr(a64::SystemRegister reg, unsigned rt) {
    const a64::Encoded encoded = a64::encode_msr(sysreg(reg), rt);
    CHECK(encoded.ok());
    return encoded.word;
}

std::uint32_t svc(std::uint16_t immediate) {
    const a64::Encoded encoded = a64::encode_svc(immediate);
    CHECK(encoded.ok());
    return encoded.word;
}

using a64::SystemRegister;
constexpr unsigned kZr = 31;

// ---------------------------------------------------------------------------
// Register images
// ---------------------------------------------------------------------------

std::uint64_t mix(std::uint64_t seed, std::uint64_t salt) {
    std::uint64_t value = seed * 0x9E3779B97F4A7C15ull + salt * 0xBF58476D1CE4E5B9ull;
    value ^= value >> 29;
    value *= 0x94D049BB133111EBull;
    value ^= value >> 32;
    return value;
}

constexpr std::uint64_t kHostReservedSentinel = 0xDEAD18DEAD18DEADull;

GuestImage make_image(std::uint64_t seed) {
    GuestImage image{};
    for (unsigned i = 0; i < 31; ++i) {
        image.x[i] = mix(seed, i);
    }
    image.x[18] = kHostReservedSentinel;  // never loaded by the fixture
    image.x[30] = 0;
    image.nzcv = (seed & 0xF) << 28;
    return image;
}

GuestImage run_site(const GuestImage& in, unsigned index) {
    GuestImage out{};
    armigo_test_guest_run_site(&in, &out, site_entry(index));
    return out;
}

// Checks every register the fixture records against the expected image; x18
// is checked through the virtual value elsewhere. x30 must hold the site's
// return address, which proves the gate returned to the site.
void check_image(const GuestImage& out, const GuestImage& expected, unsigned index,
                 const char* what) {
    for (unsigned i = 0; i < 30; ++i) {
        if (i == 18) {
            continue;
        }
        ++g_checks_total;
        if (out.x[i] != expected.x[i]) {
            ++g_checks_failed;
            std::printf("FAIL %s: x%u actual 0x%llx expected 0x%llx\n", what, i,
                        static_cast<unsigned long long>(out.x[i]),
                        static_cast<unsigned long long>(expected.x[i]));
        }
    }
    record_equal(out.x[30], site_return(index), what, __FILE__, __LINE__);
    record_equal(out.nzcv, expected.nzcv, what, __FILE__, __LINE__);
    record_equal(out.x18_after, out.x18_before, what, __FILE__, __LINE__);
    record_equal(out.sp_after, out.sp_before, what, __FILE__, __LINE__);
}

// Host thread registers; dispatch must never write them.
std::uint64_t host_tpidrro_el0() {
    std::uint64_t value;
    asm volatile("mrs %0, tpidrro_el0" : "=r"(value));
    return value;
}

std::shared_ptr<const GuestSiteTable> build_table(
    std::initializer_list<std::pair<unsigned, std::uint32_t>> fixture_sites,
    const GuestSiteTableOptions& options = {}) {
    std::vector<GuestGateSite> sites;
    for (const auto& [index, word] : fixture_sites) {
        sites.push_back(GuestGateSite{site_return(index), word});
    }
    GuestSiteTableBuild built = GuestSiteTable::build(sites, options);
    CHECK(built.ok());
    if (!built.ok()) {
        std::printf("  table error %s at %zu\n", describe(built.error.kind), built.error.site_index);
    }
    return built.table;
}

// Scripted deterministic tick source.
struct ScriptedTicks {
    std::vector<std::uint64_t> values;
    std::size_t next = 0;
    static std::uint64_t read(void* raw) noexcept {
        auto& self = *static_cast<ScriptedTicks*>(raw);
        return self.next < self.values.size() ? self.values[self.next++] : self.values.back();
    }
};

constexpr std::uint64_t kFrequency = 19'200'000;

// ---------------------------------------------------------------------------
// Site table construction
// ---------------------------------------------------------------------------

GuestSiteTableError build_error(std::vector<GuestGateSite> sites,
                                const GuestSiteTableOptions& options = {}) {
    const GuestSiteTableBuild built = GuestSiteTable::build(sites, options);
    CHECK(!built.ok() == (built.table == nullptr));
    return built.error;
}

#define CHECK_KIND(actual, expected) \
    CHECK_EQ(static_cast<unsigned>(actual), static_cast<unsigned>(expected))

void test_table_validation() {
    using K = GuestSiteTableErrorKind;

    // A valid table owns a sorted copy; the source can disappear.
    std::shared_ptr<const GuestSiteTable> table;
    {
        auto source = std::make_unique<std::vector<GuestGateSite>>(std::vector<GuestGateSite>{
            {0x3008, mrs(5, SystemRegister::tpidr_el0)},
            {0x1004, svc(0x1234)},
            {0x2000, msr(SystemRegister::tpidr_el0, 18)},
            {0x4000, mrs(kZr, SystemRegister::cntvct_el0)},
            {0x5000, mrs(7, SystemRegister::cntfrq_el0)},
            {0x6000, mrs(0, SystemRegister::tpidrro_el0)},
            {0x7000, mrs(29, SystemRegister::cntpct_el0)},
        });
        GuestSiteTableBuild built = GuestSiteTable::build(*source);
        CHECK(built.ok());
        table = built.table;
        for (GuestGateSite& site : *source) {
            site = GuestGateSite{0xFFFF0000, 0};
        }
        source.reset();
    }
    CHECK(table != nullptr);
    if (table) {
        CHECK_EQ(table->sites().size(), 7);
        for (std::size_t i = 1; i < table->sites().size(); ++i) {
            CHECK(table->sites()[i - 1].return_address < table->sites()[i].return_address);
        }
        const GuestSite* site = table->find(0x1004);
        CHECK(site != nullptr && site->kind == GuestSiteKind::supervisor_call &&
              site->immediate == 0x1234 && site->instruction == svc(0x1234));
        site = table->find(0x3008);
        CHECK(site != nullptr && site->kind == GuestSiteKind::read_thread_pointer && site->reg == 5);
        site = table->find(0x2000);
        CHECK(site != nullptr && site->kind == GuestSiteKind::write_thread_pointer &&
              site->reg == 18);
        site = table->find(0x4000);
        CHECK(site != nullptr && site->kind == GuestSiteKind::read_counter && site->reg == kZr &&
              site->system_register == SystemRegister::cntvct_el0);
        site = table->find(0x5000);
        CHECK(site != nullptr && site->kind == GuestSiteKind::read_counter_frequency);
        site = table->find(0x6000);
        CHECK(site != nullptr && site->kind == GuestSiteKind::read_read_only_thread_pointer);
        CHECK(table->find(0x1000) == nullptr);
        CHECK(table->find(0x1008) == nullptr);
        CHECK(table->find(0xFFFF0000) == nullptr);
        CHECK(table->find(0) == nullptr);
    }

    // An empty table is valid and knows no site.
    {
        CHECK(!GuestSiteTableBuild{}.ok());
        GuestSiteTableBuild built = GuestSiteTable::build({});
        CHECK(built.ok() && built.table != nullptr && built.table->find(4) == nullptr);
        const auto moved = std::move(built);
        CHECK(moved.ok() && !built.ok());
    }

    // Addresses.
    GuestSiteTableError error = build_error({{0x1000, svc(1)}, {0, svc(2)}});
    CHECK_KIND(error.kind, K::null_return_address);
    CHECK_EQ(error.site_index, 1);
    CHECK_KIND(build_error({{2, svc(1)}}).kind, K::null_return_address);
    error = build_error({{0x1002, svc(1)}});
    CHECK_KIND(error.kind, K::misaligned_return_address);
    CHECK_EQ(error.return_address, 0x1002);
    error = build_error({{0x1000, svc(1)}, {0x2000, svc(2)}, {0x3000, svc(3)}, {0x2000, svc(2)},
                         {0x1000, mrs(1, SystemRegister::tpidr_el0)}});
    CHECK_KIND(error.kind, K::duplicate_return_address);
    CHECK_EQ(error.site_index, 3);
    CHECK_EQ(error.return_address, 0x2000);
    error = build_error({{0x1000, svc(1)}, {0x1000, svc(2)}, {0x2000, 0xD503201Fu}});
    CHECK_KIND(error.kind, K::duplicate_return_address);
    CHECK_EQ(error.site_index, 1);

    // Unsupported operations.
    error = build_error({{0x1000, svc(1)}, {0x2000, 0xD503201Fu /* nop */}});
    CHECK_KIND(error.kind, K::unsupported_instruction);
    CHECK_EQ(error.site_index, 1);
    CHECK_EQ(error.instruction, 0xD503201Fu);
    CHECK_KIND(build_error({{0x1000, a64::encode_brk(1).word}}).kind, K::unsupported_instruction);
    CHECK_KIND(build_error({{0x1000, 0xD4000002u /* hvc #0 */}}).kind,
               K::unsupported_instruction);
    CHECK_KIND(build_error({{0x1000, 0x14000000u /* b . */}}).kind, K::unsupported_instruction);
    CHECK_KIND(build_error({{0x1000, mrs(1, SystemRegister::nzcv)}}).kind,
               K::unsupported_system_register);
    // CTR_EL0 is served: the cache geometry the guest sees is the console's.
    {
        const std::array<GuestGateSite, 1> cache_sites{{
            GuestGateSite{0x1000, mrs(1, SystemRegister::ctr_el0)}}};
        const auto built = GuestSiteTable::build(cache_sites, {});
        CHECK(built.ok());
        const GuestSite* cache = built.table ? built.table->find(0x1000) : nullptr;
        CHECK(cache != nullptr && cache->kind == GuestSiteKind::read_cache_type &&
              cache->reg == 1);
    }
    CHECK_KIND(build_error({{0x1000, mrs(1, SystemRegister::dczid_el0)}}).kind,
               K::unsupported_system_register);
    CHECK_KIND(build_error({{0x1000, a64::encode_mrs(1, a64::system_register_encoding(3, 0, 0, 0, 0))
                                         .word /* MIDR_EL1 */}})
                   .kind,
               K::unsupported_system_register);
    CHECK_KIND(build_error({{0x1000, msr(SystemRegister::fpcr, 1)}}).kind,
               K::unsupported_system_register);
    CHECK_KIND(build_error({{0x1000, msr(SystemRegister::tpidrro_el0, 1)}}).kind,
               K::unsupported_register_write);
    CHECK_KIND(build_error({{0x1000, msr(SystemRegister::cntvct_el0, 1)}}).kind,
               K::unsupported_register_write);
    CHECK_KIND(build_error({{0x1000, msr(SystemRegister::cntfrq_el0, 1)}}).kind,
               K::unsupported_register_write);
    CHECK_KIND(build_error({{0x1000, mrs(30, SystemRegister::tpidr_el0)}}).kind,
               K::link_register_operand);
    CHECK_KIND(build_error({{0x1000, msr(SystemRegister::tpidr_el0, 30)}}).kind,
               K::link_register_operand);

    // Options.
    GuestSiteTableOptions options;
    options.max_sites = 0;
    CHECK_KIND(build_error({}, options).kind, K::invalid_options);
    options = {};
    options.supervisor_calls = options.thread_registers = options.counters = false;
    CHECK_KIND(build_error({}, options).kind, K::invalid_options);
    options = {};
    options.counters = false;
    options.self_synchronized_counters = true;
    CHECK_KIND(build_error({}, options).kind, K::invalid_options);
    options = {};
    options.max_sites = 2;
    error = build_error({{0x1000, svc(1)}, {0x2000, svc(2)}, {0x3000, svc(3)}}, options);
    CHECK_KIND(error.kind, K::too_many_sites);

    options = {};
    options.supervisor_calls = false;
    CHECK_KIND(build_error({{0x1000, svc(1)}}, options).kind, K::disabled_site_class);
    options = {};
    options.thread_registers = false;
    CHECK_KIND(build_error({{0x1000, mrs(1, SystemRegister::tpidrro_el0)}}, options).kind,
               K::disabled_site_class);
    options = {};
    options.counters = false;
    CHECK_KIND(build_error({{0x1000, mrs(1, SystemRegister::cntfrq_el0)}}, options).kind,
               K::disabled_site_class);
    CHECK_KIND(build_error({{0x1000, mrs(1, SystemRegister::cntvctss_el0)}}).kind,
               K::disabled_site_class);
    options = {};
    options.self_synchronized_counters = true;
    const GuestSiteTableBuild ss = GuestSiteTable::build(
        std::vector<GuestGateSite>{{0x1000, mrs(1, SystemRegister::cntpctss_el0)}}, options);
    CHECK(ss.ok() && ss.table->find(0x1000) != nullptr &&
          ss.table->find(0x1000)->kind == GuestSiteKind::read_counter);

    // Every enumerator has a description.
    for (unsigned k = 0; k <= static_cast<unsigned>(K::disabled_site_class); ++k) {
        CHECK(std::strcmp(describe(static_cast<K>(k)), "unknown error") != 0);
    }
    for (unsigned k = 0; k <= static_cast<unsigned>(GuestDispatchFailureKind::counter_regressed);
         ++k) {
        CHECK(std::strcmp(describe(static_cast<GuestDispatchFailureKind>(k)), "unknown failure") !=
              0);
    }
    for (unsigned k = 0;
         k <= static_cast<unsigned>(GuestSiteKind::move_wide); ++k) {
        CHECK(std::strcmp(describe(static_cast<GuestSiteKind>(k)), "unknown site kind") != 0);
    }
    for (unsigned k = 0; k <= static_cast<unsigned>(GuestDispatchFailureKind::cache_maintenance_unavailable); ++k) {
        CHECK(std::strcmp(describe(static_cast<GuestDispatchFailureKind>(k)), "unknown failure") !=
              0);
    }
}

// ---------------------------------------------------------------------------
// Cache type and cache maintenance through the gate
// ---------------------------------------------------------------------------

struct CacheScript {
    unsigned calls = 0;
    CacheMaintenanceOperation operation = CacheMaintenanceOperation::clean_invalidate_data;
    std::uint64_t address = 0;
    bool refuse = false;
};

bool apply_cache(void* raw, CacheMaintenanceOperation operation, std::uint64_t address) noexcept {
    auto& script = *static_cast<CacheScript*>(raw);
    ++script.calls;
    script.operation = operation;
    script.address = address;
    return !script.refuse;
}

void test_cache_sites() {
    using K = GuestSiteTableErrorKind;
    const auto table = build_table({
        {0, mrs(3, SystemRegister::ctr_el0)},
        {1, clean_invalidate_data(4)},
        {2, invalidate_instruction(5)},
        {3, mrs(kZr, SystemRegister::ctr_el0)},
    });
    CHECK(table != nullptr);
    if (table) {
        const GuestSite* site = table->find(site_return(1));
        CHECK(site != nullptr && site->kind == GuestSiteKind::clean_invalidate_data_cache &&
              site->reg == 4);
        site = table->find(site_return(2));
        CHECK(site != nullptr && site->kind == GuestSiteKind::invalidate_instruction_cache &&
              site->reg == 5);
    }
    // The class can be disabled, and a disabled site is rejected, never ignored.
    CHECK_KIND(build_error({{0x1000, mrs(1, SystemRegister::ctr_el0)}},
                           {.cache_maintenance = false})
                   .kind,
               K::disabled_site_class);
    CHECK_KIND(build_error({{0x1000, clean_invalidate_data(1)}}, {.cache_maintenance = false}).kind,
               K::disabled_site_class);
    // Other cache operations are not served.
    CHECK_KIND(build_error({{0x1000, cache_instruction(3, 7, 10, 1, 1) /* DC CVAC */}}).kind,
               K::unsupported_instruction);

    GuestThreadContext thread;
    GuestDispatchScope scope(table, thread, {}, GateFailurePolicy::resume_unchanged);
    set_thread_guest_x18(0x77);

    // Without a value and a handler the sites fail and change nothing.
    GuestImage in = make_image(0x300);
    check_image(run_site(in, 0), in, 0, "cache type unavailable");
    check_image(run_site(in, 1), in, 1, "cache maintenance unavailable");
    CHECK_EQ(scope.failure_count(), 2);
    CHECK(scope.first_failure() != nullptr &&
          scope.first_failure()->kind == GuestDispatchFailureKind::cache_type_unavailable);
    scope.clear_failures();
    scope.gate_scope().clear_failures();

    thread.set_cache_type(0x8444C004ull);
    CacheScript script;
    thread.set_cache_maintenance({&apply_cache, &script});
    GuestImage expected = in;
    expected.x[3] = 0x8444C004ull;
    check_image(run_site(in, 0), expected, 0, "mrs x3, ctr_el0");
    check_image(run_site(in, 3), in, 3, "mrs xzr, ctr_el0");

    // A cache site hands its address operand to the runtime and changes no register.
    in = make_image(0x301);
    in.x[4] = 0x7000001000ull;
    in.x[5] = 0x7000002000ull;
    check_image(run_site(in, 1), in, 1, "dc civac, x4");
    CHECK_EQ(script.calls, 1);
    CHECK(script.operation == CacheMaintenanceOperation::clean_invalidate_data);
    CHECK_EQ(script.address, 0x7000001000ull);
    check_image(run_site(in, 2), in, 2, "ic ivau, x5");
    CHECK_EQ(script.calls, 2);
    CHECK(script.operation == CacheMaintenanceOperation::invalidate_instruction);
    CHECK_EQ(script.address, 0x7000002000ull);

    // A refused operation fails the site instead of pretending it ran.
    script.refuse = true;
    scope.clear_failures();
    scope.gate_scope().clear_failures();
    check_image(run_site(in, 1), in, 1, "refused cache maintenance");
    CHECK(scope.first_failure() != nullptr &&
          scope.first_failure()->kind == GuestDispatchFailureKind::cache_maintenance_unavailable);
    script.refuse = false;
}

// ---------------------------------------------------------------------------
// Thread registers through the gate
// ---------------------------------------------------------------------------

// MOVZ (keep false) or MOVK (keep true) Rd, #imm16, LSL #(16 * hw).
std::uint32_t move_wide(bool sf, bool keep, unsigned hw, std::uint16_t immediate, unsigned rd) {
    return (sf ? 0x80000000u : 0u) | (keep ? 0x60000000u : 0x40000000u) | 0x12800000u | (hw << 21) |
           (std::uint32_t{immediate} << 5) | rd;
}

void test_move_wide_sites() {
    using K = GuestSiteTableErrorKind;
    const auto table = build_table({
        {0, move_wide(true, false, 2, 0x1234, 3)},
        {1, move_wide(true, true, 1, 0xbeef, 4)},
        {2, move_wide(false, false, 1, 0xabcd, 5)},
        {3, move_wide(false, true, 0, 0x5555, 6)},
        {4, move_wide(true, true, 3, 0x7fff, 7)},
        {5, move_wide(true, false, 0, 0x1, kZr)},
    }, {.move_wide = true});
    CHECK(table != nullptr);
    if (table) {
        const GuestSite* site = table->find(site_return(1));
        CHECK(site != nullptr && site->kind == GuestSiteKind::move_wide && site->reg == 4 &&
              site->immediate == 0xbeef);
    }
    CHECK_KIND(build_error({{0x1000, move_wide(true, true, 0, 1, 30)}}, {.move_wide = true}).kind,
               K::link_register_operand);
    CHECK_KIND(build_error({{0x1000, move_wide(false, false, 2, 1, 1)}}, {.move_wide = true}).kind,
               K::unsupported_instruction);
    CHECK_KIND(build_error({{0x1000, 0x92800021u /* MOVN X1, #1 */}}, {.move_wide = true}).kind,
               K::unsupported_instruction);
    // Tables admit them only on request.
    CHECK_KIND(build_error({{0x1000, move_wide(true, false, 0, 1, 1)}}).kind, K::disabled_site_class);

    GuestThreadContext thread;
    GuestDispatchScope scope(table, thread, {}, GateFailurePolicy::resume_unchanged);
    set_thread_guest_x18(0x77);
    const GuestImage in = make_image(0x400);
    GuestImage expected = in;
    expected.x[3] = 0x1234ull << 32;
    check_image(run_site(in, 0), expected, 0, "movz x3, #0x1234, lsl #32");
    expected = in;
    expected.x[4] = (in.x[4] & ~0xffff0000ull) | 0xbeef0000ull;
    check_image(run_site(in, 1), expected, 1, "movk x4, #0xbeef, lsl #16");
    expected = in;
    expected.x[5] = 0xabcd0000ull;
    check_image(run_site(in, 2), expected, 2, "movz w5, #0xabcd, lsl #16");
    expected = in;
    expected.x[6] = (in.x[6] & 0xffff0000ull) | 0x5555ull;
    check_image(run_site(in, 3), expected, 3, "movk w6, #0x5555");
    expected = in;
    expected.x[7] = (in.x[7] & 0x0000ffffffffffffull) | 0x7fff000000000000ull;
    check_image(run_site(in, 4), expected, 4, "movk x7, #0x7fff, lsl #48");
    check_image(run_site(in, 5), in, 5, "movz xzr, #1");
    CHECK_EQ(scope.failure_count(), 0);
}

void test_thread_registers() {
    const auto table = build_table({
        {0, mrs(5, SystemRegister::tpidr_el0)},
        {1, mrs(18, SystemRegister::tpidrro_el0)},
        {2, msr(SystemRegister::tpidr_el0, 18)},
        {3, mrs(kZr, SystemRegister::tpidr_el0)},
        {4, msr(SystemRegister::tpidr_el0, kZr)},
        {5, msr(SystemRegister::tpidr_el0, 9)},
        {6, mrs(0, SystemRegister::tpidrro_el0)},
        {7, mrs(29, SystemRegister::tpidr_el0)},
        {8, mrs(kZr, SystemRegister::tpidrro_el0)},
    });
    if (!table) {
        return;
    }
    const std::uint64_t host_tpidrro = host_tpidrro_el0();
    // XNU may update TPIDR_EL0 CPU/cluster metadata when this thread migrates.
    // Inspect the shared emitted gate body instead of assuming that host
    // register is constant across a sequence of host calls.
    const auto gate_code = make_local_gate_code();
    unsigned system_writes = 0;
    CHECK(gate_code.size() >= 8);
    for (std::size_t offset = 0; offset + 8 < gate_code.size(); offset += 4) {
        std::uint32_t word;
        std::memcpy(&word, gate_code.data() + offset, 4);
        const auto instruction = a64::decode(word);
        if (instruction.opcode != a64::Opcode::msr_register) continue;
        ++system_writes;
        const auto target = a64::identify_system_register(instruction.system_register);
        CHECK(target == SystemRegister::nzcv || target == SystemRegister::fpcr ||
              target == SystemRegister::fpsr);
    }
    CHECK_EQ(system_writes, 3);
    set_thread_guest_x18(0x0018001800180018ull);

    GuestThreadContext thread;
    thread.set_thread_pointer(0x1111000011110000ull);
    thread.set_read_only_thread_pointer(0x2222000022220000ull);
    {
        GuestDispatchScope scope(table, thread, {}, GateFailurePolicy::resume_unchanged);
        CHECK_KIND(scope.setup_error(), GuestDispatchFailureKind::none);
        CHECK(thread.bound());

        GuestImage in = make_image(0x100);
        GuestImage expected = in;
        expected.x[5] = 0x1111000011110000ull;
        check_image(run_site(in, 0), expected, 0, "mrs x5, tpidr_el0");

        // A write to x18 lands in the virtual guest x18, never in hardware.
        in = make_image(0x101);
        GuestImage out = run_site(in, 1);
        check_image(out, in, 1, "mrs x18, tpidrro_el0");
        CHECK_EQ(thread_guest_x18(), 0x2222000022220000ull);
        CHECK(out.x18_after != 0x2222000022220000ull);

        // A read of x18 comes from the virtual guest x18.
        set_thread_guest_x18(0x0A0B0C0D0E0F1011ull);
        in = make_image(0x102);
        check_image(run_site(in, 2), in, 2, "msr tpidr_el0, x18");
        CHECK_EQ(thread.thread_pointer(), 0x0A0B0C0D0E0F1011ull);

        // xzr: the read result is discarded; the write stores zero.
        in = make_image(0x103);
        check_image(run_site(in, 3), in, 3, "mrs xzr, tpidr_el0");
        CHECK_EQ(thread.thread_pointer(), 0x0A0B0C0D0E0F1011ull);
        check_image(run_site(in, 8), in, 8, "mrs xzr, tpidrro_el0");
        CHECK_EQ(thread_guest_x18(), 0x0A0B0C0D0E0F1011ull);
        check_image(run_site(in, 4), in, 4, "msr tpidr_el0, xzr");
        CHECK_EQ(thread.thread_pointer(), 0);

        in = make_image(0x104);
        check_image(run_site(in, 5), in, 5, "msr tpidr_el0, x9");
        CHECK_EQ(thread.thread_pointer(), in.x[9]);

        in = make_image(0x105);
        expected = in;
        expected.x[0] = 0x2222000022220000ull;
        check_image(run_site(in, 6), expected, 6, "mrs x0, tpidrro_el0");

        const std::uint64_t tls = thread.thread_pointer();
        in = make_image(0x106);
        expected = in;
        expected.x[29] = tls;
        check_image(run_site(in, 7), expected, 7, "mrs x29, tpidr_el0");

        CHECK_EQ(scope.dispatched(), 9);
        CHECK_EQ(scope.failure_count(), 0);
        CHECK_EQ(scope.gate_scope().failure_count(), 0);
        CHECK_EQ(thread.read_only_thread_pointer(), 0x2222000022220000ull);
    }
    CHECK(!thread.bound());
    CHECK_EQ(host_tpidrro_el0() & ~std::uint64_t{7}, host_tpidrro & ~std::uint64_t{7});
    set_thread_guest_x18(0);
}

// ---------------------------------------------------------------------------
// Counter
// ---------------------------------------------------------------------------

void test_counter() {
    GuestSiteTableOptions options;
    options.self_synchronized_counters = true;
    const auto table = build_table(
        {
            {0, mrs(3, SystemRegister::cntvct_el0)},
            {1, mrs(4, SystemRegister::cntpct_el0)},
            {2, mrs(5, SystemRegister::cntfrq_el0)},
            {3, mrs(kZr, SystemRegister::cntvct_el0)},
            {4, mrs(18, SystemRegister::cntvct_el0)},
            {5, mrs(6, SystemRegister::cntvctss_el0)},
        },
        options);
    if (!table) {
        return;
    }
    GuestThreadContext thread;
    ScriptedTicks ticks{{100, 100, 250, 300, 400, 50}};

    // Invalid counters are refused and leave no counter installed.
    CHECK_KIND(thread.set_counter({}), GuestCounterError::missing_tick_source);
    CHECK_KIND(thread.set_counter({&ScriptedTicks::read, &ticks, 0}),
               GuestCounterError::zero_frequency);
    CHECK_KIND(thread.set_counter({&ScriptedTicks::read, &ticks, 0x100000000ull}),
               GuestCounterError::frequency_out_of_range);
    CHECK(!thread.counter().has_value());

    GuestDispatchScope scope(table, thread, {}, GateFailurePolicy::resume_unchanged);
    set_thread_guest_x18(0x55);

    // Without a counter, counter sites fail and change nothing.
    GuestImage in = make_image(0x200);
    check_image(run_site(in, 0), in, 0, "counter unavailable");
    check_image(run_site(in, 2), in, 2, "frequency unavailable");
    CHECK_EQ(scope.failure_count(), 2);
    CHECK(scope.first_failure() != nullptr &&
          scope.first_failure()->kind == GuestDispatchFailureKind::counter_unavailable &&
          scope.first_failure()->return_address == site_return(0) &&
          scope.first_failure()->instruction == mrs(3, SystemRegister::cntvct_el0) &&
          scope.first_failure()->depth == 1);
    CHECK_EQ(scope.gate_scope().failure_count(), 2);
    CHECK(scope.gate_scope().first_failure() != nullptr &&
          scope.gate_scope().first_failure()->kind == GateFailureKind::handler_reported &&
          std::strcmp(scope.gate_scope().first_failure()->message,
                      describe(GuestDispatchFailureKind::counter_unavailable)) == 0);
    CHECK(!ticks.next);
    scope.clear_failures();
    scope.gate_scope().clear_failures();

    CHECK_KIND(thread.set_counter({&ScriptedTicks::read, &ticks, kFrequency}),
               GuestCounterError::none);

    in = make_image(0x201);
    GuestImage expected = in;
    expected.x[3] = 100;
    check_image(run_site(in, 0), expected, 0, "mrs x3, cntvct_el0");
    expected = in;
    expected.x[4] = 100;  // equal values are not a regression
    check_image(run_site(in, 1), expected, 1, "mrs x4, cntpct_el0");
    expected = in;
    expected.x[5] = kFrequency;
    check_image(run_site(in, 2), expected, 2, "mrs x5, cntfrq_el0");
    CHECK_EQ(ticks.next, 2);  // the frequency does not consume a tick
    check_image(run_site(in, 3), in, 3, "mrs xzr, cntvct_el0");
    CHECK(thread.last_counter_value() == std::optional<std::uint64_t>{250});
    check_image(run_site(in, 4), in, 4, "mrs x18, cntvct_el0");
    CHECK_EQ(thread_guest_x18(), 300);
    expected = in;
    expected.x[6] = 400;
    check_image(run_site(in, 5), expected, 5, "mrs x6, cntvctss_el0");
    CHECK_EQ(scope.failure_count(), 0);

    // A tick source that goes backwards is reported, not delivered.
    check_image(run_site(in, 0), in, 0, "counter regression");
    CHECK_EQ(scope.failure_count(), 1);
    CHECK(scope.first_failure() != nullptr &&
          scope.first_failure()->kind == GuestDispatchFailureKind::counter_regressed);
    CHECK(thread.last_counter_value() == std::optional<std::uint64_t>{400});
    CHECK_EQ(scope.dispatched(), 6);

    // Replacing the source keeps the baseline, so a lagging source is caught.
    ScriptedTicks lagging{{399, 401}};
    CHECK_KIND(thread.set_counter({&ScriptedTicks::read, &lagging, kFrequency}),
               GuestCounterError::none);
    check_image(run_site(in, 0), in, 0, "lagging replacement");
    expected = in;
    expected.x[3] = 401;
    check_image(run_site(in, 0), expected, 0, "replacement catches up");
    CHECK_EQ(scope.failure_count(), 2);

    thread.clear_counter();
    check_image(run_site(in, 1), in, 1, "counter cleared");
    CHECK_EQ(scope.failure_count(), 3);
    set_thread_guest_x18(0);
}

// ---------------------------------------------------------------------------
// Supervisor calls
// ---------------------------------------------------------------------------

struct SupervisorLog {
    unsigned calls = 0;
    std::uint16_t immediate = 0;
    std::uint64_t return_address = 0;
    std::uint32_t depth = 0;
    GuestThreadContext* thread = nullptr;
    bool omit_message = false;
};

GateStatus supervisor(std::uint16_t immediate, GateCall& call, GuestThreadContext& thread,
                      void* raw) {
    auto& log = *static_cast<SupervisorLog*>(raw);
    ++log.calls;
    log.immediate = immediate;
    log.return_address = call.return_address();
    log.depth = call.depth();
    log.thread = &thread;
    switch (immediate) {
    case 0x21:
        call.set_x(0, call.x(0) * 3 + immediate);
        call.set_x(1, thread.thread_pointer());
        thread.set_thread_pointer(0xABC0);
        call.set_guest_x18(0x1818);
        return GateStatus::resume;
    case 0xFFFF:
        call.set_x(2, 0xBAD);
        call.set_guest_x18(0xBAD);
        thread.set_thread_pointer(0xBAD0);
        thread.set_read_only_thread_pointer(0xBAD1);
        if (!log.omit_message) {
            call.set_failure_message("synthetic supervisor failure");
        }
        return GateStatus::failed;
    case 0:
        call.set_x(3, 0xBAD);
        thread.set_thread_pointer(0xBAD2);
        throw std::runtime_error("synthetic supervisor exception");
    case 0x70:
    case 0x71:
        thread.set_thread_pointer(0xBAD0);
        thread.set_read_only_thread_pointer(0xBAD1);
        (void)thread.set_counter({+[](void*) noexcept -> std::uint64_t { return 1; }, nullptr, 500});
        call.set_guest_x18(0xBAD);
        if (immediate == 0x70) call.frame().x[18] = 1;
        else call.frame().sp += 16;
        return GateStatus::resume;
    default:
        return GateStatus::resume;
    }
}

void test_supervisor_calls() {
    const auto table = build_table({{0, svc(0x21)}, {1, svc(0xFFFF)}, {2, svc(0)},
                                    {3, svc(7)}, {4, svc(0x70)}, {5, svc(0x71)}});
    if (!table) {
        return;
    }
    GuestThreadContext thread;
    thread.set_thread_pointer(0x7000);
    thread.set_read_only_thread_pointer(0x8000);
    set_thread_guest_x18(0x4242);

    // No handler: explicit failure, nothing changes.
    {
        GuestDispatchScope scope(table, thread, {}, GateFailurePolicy::resume_unchanged);
        const GuestImage in = make_image(0x300);
        check_image(run_site(in, 0), in, 0, "missing supervisor handler");
        CHECK_EQ(scope.failure_count(), 1);
        CHECK(scope.first_failure() != nullptr &&
              scope.first_failure()->kind == GuestDispatchFailureKind::missing_supervisor_handler &&
              scope.first_failure()->instruction == svc(0x21));
        CHECK_EQ(scope.dispatched(), 0);
    }

    SupervisorLog log;
    GuestDispatchScope scope(table, thread, {&supervisor, &log},
                             GateFailurePolicy::resume_unchanged);

    GuestImage in = make_image(0x301);
    GuestImage expected = in;
    expected.x[0] = in.x[0] * 3 + 0x21;
    expected.x[1] = 0x7000;
    check_image(run_site(in, 0), expected, 0, "svc #0x21");
    CHECK_EQ(log.calls, 1);
    CHECK_EQ(log.immediate, 0x21);
    CHECK_EQ(log.return_address, site_return(0));
    CHECK_EQ(log.depth, 1);
    CHECK(log.thread == &thread);
    CHECK_EQ(thread.thread_pointer(), 0xABC0);
    CHECK_EQ(thread_guest_x18(), 0x1818);

    expected = in;
    check_image(run_site(in, 3), expected, 3, "svc #7");
    CHECK_EQ(log.immediate, 7);

    // A failing handler is rolled back completely: frame, virtual x18 and the
    // context's thread pointers.
    in = make_image(0x302);
    check_image(run_site(in, 1), in, 1, "failing supervisor");
    CHECK_EQ(log.immediate, 0xFFFF);
    CHECK_EQ(thread.thread_pointer(), 0xABC0);
    CHECK_EQ(thread.read_only_thread_pointer(), 0x8000);
    CHECK_EQ(thread_guest_x18(), 0x1818);
    CHECK_EQ(scope.failure_count(), 1);
    CHECK(scope.first_failure() != nullptr &&
          scope.first_failure()->kind == GuestDispatchFailureKind::supervisor_failed &&
          std::strcmp(scope.first_failure()->message, "synthetic supervisor failure") == 0 &&
          scope.first_failure()->instruction == svc(0xFFFF));
    CHECK(scope.gate_scope().first_failure() != nullptr &&
          std::strcmp(scope.gate_scope().first_failure()->message,
                      "synthetic supervisor failure") == 0);

    scope.clear_failures();
    log.omit_message = true;
    check_image(run_site(in, 1), in, 1, "failing supervisor without message");
    CHECK(scope.first_failure() != nullptr &&
          std::strcmp(scope.first_failure()->message,
                      describe(GuestDispatchFailureKind::supervisor_failed)) == 0);

    // An escaping exception is rolled back, recorded here and by the gate.
    scope.clear_failures();
    scope.gate_scope().clear_failures();
    in = make_image(0x303);
    check_image(run_site(in, 2), in, 2, "throwing supervisor");
    CHECK_EQ(thread.thread_pointer(), 0xABC0);
    CHECK(scope.first_failure() != nullptr &&
          scope.first_failure()->kind == GuestDispatchFailureKind::supervisor_threw);
    CHECK(scope.gate_scope().first_failure() != nullptr &&
          scope.gate_scope().first_failure()->kind == GateFailureKind::handler_threw);
    CHECK_EQ(scope.dispatched(), 2);
    for (unsigned site : {4u, 5u}) {
        scope.clear_failures();
        scope.gate_scope().clear_failures();
        check_image(run_site(in, site), in, site, "invalid supervisor frame");
        CHECK_EQ(thread.thread_pointer(), 0xABC0);
        CHECK_EQ(thread.read_only_thread_pointer(), 0x8000);
        CHECK(!thread.counter());
        CHECK_EQ(thread_guest_x18(), 0x1818);
        CHECK_EQ(scope.dispatched(), 2);
        CHECK(scope.first_failure() != nullptr &&
              scope.first_failure()->kind == GuestDispatchFailureKind::invalid_supervisor_frame);
        CHECK(scope.gate_scope().first_failure() != nullptr &&
              scope.gate_scope().first_failure()->kind == GateFailureKind::handler_reported);
    }
    set_thread_guest_x18(0);
}

// ---------------------------------------------------------------------------
// Unknown sites and invalid scopes
// ---------------------------------------------------------------------------

void test_unknown_sites() {
    const auto table = build_table({{0, mrs(1, SystemRegister::tpidr_el0)}});
    if (!table) {
        return;
    }
    GuestThreadContext thread;
    thread.set_thread_pointer(0x1234);
    {
        GuestDispatchScope scope(table, thread, {}, GateFailurePolicy::resume_unchanged);
        for (unsigned index = 1; index < kFixtureSites; ++index) {
            const GuestImage in = make_image(0x400 + index);
            check_image(run_site(in, index), in, index, "unknown site");
        }
        CHECK_EQ(scope.failure_count(), kFixtureSites - 1);
        CHECK(scope.first_failure() != nullptr &&
              scope.first_failure()->kind == GuestDispatchFailureKind::unknown_site &&
              scope.first_failure()->return_address == site_return(1) &&
              scope.first_failure()->instruction == 0);
        CHECK(scope.gate_scope().first_failure() != nullptr &&
              scope.gate_scope().first_failure()->return_address == site_return(1));
        CHECK_EQ(scope.dispatched(), 0);
        // The registered site still works.
        const GuestImage in = make_image(0x420);
        GuestImage expected = in;
        expected.x[1] = 0x1234;
        check_image(run_site(in, 0), expected, 0, "known site after unknown ones");
    }

    // A scope without a table still captures entries and fails them.
    {
        GuestDispatchScope scope(nullptr, thread, {}, GateFailurePolicy::resume_unchanged);
        CHECK_KIND(scope.setup_error(), GuestDispatchFailureKind::missing_table);
        CHECK(!thread.bound());
        const GuestImage in = make_image(0x430);
        check_image(run_site(in, 0), in, 0, "missing table");
        CHECK(scope.first_failure() != nullptr &&
              scope.first_failure()->kind == GuestDispatchFailureKind::missing_table);
    }
}

// ---------------------------------------------------------------------------
// Nesting
// ---------------------------------------------------------------------------

struct NestingState {
    unsigned outer_calls = 0;
    std::uint64_t inner_x1 = 0;
    std::uint64_t inner_x7 = 0;
    std::uint32_t inner_depth_seen = 0;
    bool unknown_unchanged = false;
    bool shadow_ok = false;
    std::shared_ptr<const GuestSiteTable> table;
};

GateStatus nesting_supervisor(std::uint16_t immediate, GateCall& call, GuestThreadContext& thread,
                              void* raw) {
    auto& state = *static_cast<NestingState*>(raw);
    ++state.outer_calls;
    if (immediate != 0x40 || call.depth() != 1) {
        return GateStatus::failed;
    }
    // Re-enter guest-like code: depth-2 entries go to the same scope.
    thread.set_thread_pointer(0x6060);
    GuestImage in = make_image(0x500);
    GuestImage out = run_site(in, 1);  // mrs x1, tpidr_el0
    state.inner_x1 = out.x[1];
    state.inner_depth_seen = GateScope::thread_gate_depth();

    // A nested failure fails only the nested entry.
    in = make_image(0x501);
    out = run_site(in, 9);
    bool same = out.x18_after == out.x18_before;
    for (unsigned i = 0; i < 30; ++i) {
        same = same && (i == 18 || out.x[i] == in.x[i]);
    }
    state.unknown_unchanged = same;

    // A shadowing scope with its own context serves the next nested entry;
    // binding the outer context again on the same thread is permitted.
    {
        GuestThreadContext shadow;
        shadow.set_thread_pointer(0x7171);
        GuestDispatchScope inner(state.table, shadow, {}, GateFailurePolicy::resume_unchanged);
        GuestDispatchScope rebind(state.table, thread, {}, GateFailurePolicy::resume_unchanged);
        GuestImage image = make_image(0x502);
        GuestImage result = run_site(image, 2);  // mrs x7, tpidr_el0 via `rebind`
        state.inner_x7 = result.x[7];
        state.shadow_ok = inner.setup_error() == GuestDispatchFailureKind::none &&
                          rebind.setup_error() == GuestDispatchFailureKind::none &&
                          rebind.dispatched() == 1 && inner.dispatched() == 0;
    }
    {
        GuestThreadContext shadow;
        shadow.set_thread_pointer(0x7272);
        GuestDispatchScope inner(state.table, shadow, {}, GateFailurePolicy::resume_unchanged);
        GuestImage image = make_image(0x503);
        GuestImage result = run_site(image, 2);
        state.shadow_ok = state.shadow_ok && result.x[7] == 0x7272 && inner.dispatched() == 1;
    }
    state.shadow_ok = state.shadow_ok && thread.bound();
    call.set_x(0, 0x4040);
    return GateStatus::resume;
}

GateStatus failing_nested_supervisor(std::uint16_t, GateCall&, GuestThreadContext& thread,
                                    void* raw) {
    auto image = make_image(0x511);
    image.x[9] = 0x9999;
    (void)run_site(image, 1);
    const auto read = run_site(image, 2);
    *static_cast<bool*>(raw) = thread.thread_pointer() == 0x9999 && read.x[3] == 110;
    return GateStatus::failed;
}

void test_nested_failure_rollback() {
    const auto table = build_table({{0, svc(0x40)},
        {1, msr(SystemRegister::tpidr_el0, 9)}, {2, mrs(3, SystemRegister::cntpct_el0)}});
    if (!table) return;
    GuestThreadContext thread;
    thread.set_thread_pointer(0x7000);
    ScriptedTicks ticks{{110}};
    CHECK_KIND(thread.set_counter({&ScriptedTicks::read, &ticks, kFrequency}),
               GuestCounterError::none);
    bool inner_observed = false;
    GuestDispatchScope scope(table, thread, {failing_nested_supervisor, &inner_observed},
                             GateFailurePolicy::resume_unchanged);
    const auto image = make_image(0x512);
    check_image(run_site(image, 0), image, 0, "outer failure after nested write and read");
    CHECK(inner_observed);
    CHECK_EQ(thread.thread_pointer(), 0x7000);
    CHECK(thread.last_counter_value() == 110);
    CHECK_EQ(scope.dispatched(), 2);
    CHECK_EQ(scope.failure_count(), 1);
}

void test_nesting() {
    const auto table = build_table({
        {0, svc(0x40)},
        {1, mrs(1, SystemRegister::tpidr_el0)},
        {2, mrs(7, SystemRegister::tpidr_el0)},
    });
    if (!table) {
        return;
    }
    NestingState state;
    state.table = table;
    GuestThreadContext thread;
    GuestDispatchScope scope(table, thread, {&nesting_supervisor, &state},
                             GateFailurePolicy::resume_unchanged);
    const GuestImage in = make_image(0x510);
    GuestImage expected = in;
    expected.x[0] = 0x4040;
    check_image(run_site(in, 0), expected, 0, "nested outer");
    CHECK_EQ(state.outer_calls, 1);
    CHECK_EQ(state.inner_x1, 0x6060);
    CHECK_EQ(state.inner_depth_seen, 1);
    CHECK(state.unknown_unchanged);
    CHECK_EQ(state.inner_x7, 0x6060);
    CHECK(state.shadow_ok);
    CHECK_EQ(scope.failure_count(), 1);
    CHECK(scope.first_failure() != nullptr &&
          scope.first_failure()->kind == GuestDispatchFailureKind::unknown_site &&
          scope.first_failure()->depth == 2 &&
          scope.first_failure()->return_address == site_return(9));
    CHECK_EQ(scope.dispatched(), 2);  // the inner read and the outer call
    CHECK_EQ(thread.thread_pointer(), 0x6060);
    CHECK_EQ(GateScope::thread_gate_depth(), 0);
}

// ---------------------------------------------------------------------------
// Straight-line prepared sequence
// ---------------------------------------------------------------------------

std::vector<GuestGateSite> sequence_sites(std::uint16_t svc_immediate) {
    return {
        {address_of(&armigo_test_guest_sequence_return_0), msr(SystemRegister::tpidr_el0, 1)},
        {address_of(&armigo_test_guest_sequence_return_1), mrs(2, SystemRegister::tpidr_el0)},
        {address_of(&armigo_test_guest_sequence_return_2), mrs(3, SystemRegister::cntvct_el0)},
        {address_of(&armigo_test_guest_sequence_return_3), mrs(4, SystemRegister::cntpct_el0)},
        {address_of(&armigo_test_guest_sequence_return_4), mrs(5, SystemRegister::cntfrq_el0)},
        {address_of(&armigo_test_guest_sequence_return_5), svc(svc_immediate)},
    };
}

GateStatus sequence_supervisor(std::uint16_t immediate, GateCall& call, GuestThreadContext&,
                               void*) {
    call.set_x(0, call.x(0) * 6 + immediate);
    return GateStatus::resume;
}

void test_sequence() {
    const GuestSiteTableBuild built = GuestSiteTable::build(sequence_sites(0x21));
    CHECK(built.ok());
    if (!built.ok()) {
        return;
    }
    GuestThreadContext thread;
    ScriptedTicks ticks{{1000, 1003}};
    CHECK_KIND(thread.set_counter({&ScriptedTicks::read, &ticks, kFrequency}),
               GuestCounterError::none);
    GuestDispatchScope scope(built.table, thread, {&sequence_supervisor, nullptr});
    std::uint64_t out[5] = {};
    armigo_test_guest_sequence(0xFEEDF00D12345678ull, out);
    CHECK_EQ(out[0], 0xFEEDF00D12345678ull);
    CHECK_EQ(out[1], 1000);
    CHECK_EQ(out[2], 1003);
    CHECK_EQ(out[3], kFrequency);
    CHECK_EQ(out[4], 7 * 6 + 0x21);
    CHECK_EQ(thread.thread_pointer(), 0xFEEDF00D12345678ull);
    CHECK_EQ(scope.dispatched(), 6);
    CHECK_EQ(scope.failure_count(), 0);
}

// ---------------------------------------------------------------------------
// Threads
// ---------------------------------------------------------------------------

struct ThreadTicks {
    std::uint64_t value = 0;
    std::uint64_t step = 0;
    static std::uint64_t read(void* raw) noexcept {
        auto& self = *static_cast<ThreadTicks*>(raw);
        self.value += self.step;
        return self.value;
    }
};

void test_thread_isolation() {
    const GuestSiteTableBuild sequence = GuestSiteTable::build(sequence_sites(0x30));
    const auto sites = build_table({
        {0, mrs(18, SystemRegister::tpidrro_el0)},
        {1, mrs(9, SystemRegister::tpidr_el0)},
    });
    CHECK(sequence.ok());
    if (!sequence.ok() || !sites) {
        return;
    }
    constexpr unsigned kThreads = 8;
    constexpr std::uint64_t kIterations = 300;
    std::atomic<std::uint64_t> mismatches{0};
    std::vector<std::thread> threads;
    for (unsigned t = 0; t < kThreads; ++t) {
        threads.emplace_back([&, t] {
            const std::uint64_t tag = 0x100 + t;
            std::uint64_t local_mismatches = 0;
            GuestThreadContext thread;
            thread.set_read_only_thread_pointer(tag << 40);
            ThreadTicks ticks{tag << 20, tag};
            if (thread.set_counter({&ThreadTicks::read, &ticks, kFrequency + t}) !=
                GuestCounterError::none) {
                ++local_mismatches;
            }
            GuestDispatchScope scope(sequence.table, thread, {&sequence_supervisor, nullptr});
            for (std::uint64_t i = 0; i < kIterations; ++i) {
                const std::uint64_t value = (tag << 32) | i;
                std::uint64_t out[5] = {};
                armigo_test_guest_sequence(value, out);
                const std::uint64_t base = (tag << 20) + (2 * i + 1) * tag;
                if (out[0] != value || out[1] != base || out[2] != base + tag ||
                    out[3] != kFrequency + t || out[4] != 7 * 6 + 0x30 ||
                    thread.thread_pointer() != value) {
                    ++local_mismatches;
                }
                if ((i & 15) == 0) {
                    // Switch to the site table for a virtual-x18 round trip.
                    GuestDispatchScope inner(sites, thread);
                    const GuestImage in = make_image(tag * 1000 + i);
                    GuestImage out18 = run_site(in, 0);
                    if (thread_guest_x18() != (tag << 40) || out18.x18_after != out18.x18_before) {
                        ++local_mismatches;
                    }
                    GuestImage expected = in;
                    expected.x[9] = value;
                    out18 = run_site(in, 1);
                    for (unsigned r = 0; r < 30; ++r) {
                        if (r != 18 && out18.x[r] != expected.x[r]) {
                            ++local_mismatches;
                        }
                    }
                }
            }
            if (scope.failure_count() != 0 || scope.dispatched() != kIterations * 6) {
                ++local_mismatches;
            }
            mismatches.fetch_add(local_mismatches);
        });
    }
    for (std::thread& thread : threads) {
        thread.join();
    }
    CHECK_EQ(mismatches.load(), 0);

    // A context bound on one thread is refused on another.
    GuestThreadContext shared;
    shared.set_thread_pointer(0x5151);
    {
        GuestDispatchScope owner(sites, shared, {}, GateFailurePolicy::resume_unchanged);
        bool refused = false;
        bool unchanged = false;
        std::uint64_t failures = 0;
        std::thread([&] {
            GuestDispatchScope other(sites, shared, {}, GateFailurePolicy::resume_unchanged);
            refused = other.setup_error() == GuestDispatchFailureKind::context_bound_elsewhere;
            const GuestImage in = make_image(0x600);
            const GuestImage out = run_site(in, 1);
            unchanged = out.x[9] == in.x[9];
            failures = other.failure_count();
        }).join();
        CHECK(refused);
        CHECK(unchanged);
        CHECK_EQ(failures, 1);
        CHECK(shared.bound());
        CHECK_EQ(owner.failure_count(), 0);
    }
    CHECK(!shared.bound());
    // Once released, another thread may bind it.
    bool rebound = false;
    std::thread([&] {
        GuestDispatchScope other(sites, shared, {}, GateFailurePolicy::resume_unchanged);
        const GuestImage in = make_image(0x601);
        rebound = other.setup_error() == GuestDispatchFailureKind::none &&
                  run_site(in, 1).x[9] == 0x5151;
    }).join();
    CHECK(rebound);
    CHECK(!shared.bound());
}

// ---------------------------------------------------------------------------
// Guest register cell and per-context virtual x18
// ---------------------------------------------------------------------------

std::shared_ptr<const GuestSiteTable> register_cell_table() {
    return build_table({
        {0, mrs(5, SystemRegister::tpidr_el0)},
        {1, msr(SystemRegister::tpidr_el0, 18)},
        {2, mrs(18, SystemRegister::tpidrro_el0)},
        {3, svc(0x50)},  // nested distinct context, outer resumes
        {4, svc(0x51)},  // nested distinct context, outer fails
        {5, svc(0x52)},  // records the entry's x18 storage
        {8, svc(0x53)},  // leaves an inner context selected, then fails
    });
}

// Plain-GateScope handler: records what the entry sees and optionally writes.
struct X18Probe {
    const std::uint64_t* storage = nullptr;
    std::uint64_t value = 0;
    std::uint64_t write = 0;
    bool do_write = false;
    unsigned calls = 0;
};

GateStatus x18_probe(GateCall& call, void* raw) {
    auto& probe = *static_cast<X18Probe*>(raw);
    ++probe.calls;
    probe.storage = call.guest_x18_storage();
    probe.value = call.guest_x18();
    if (probe.do_write) {
        call.set_guest_x18(probe.write);
    }
    return GateStatus::resume;
}

struct CellNesting {
    std::shared_ptr<const GuestSiteTable> table;
    GuestThreadContext* inner = nullptr;
    const std::uint64_t* probe_storage = nullptr;
    std::uint64_t entry_x18 = 0;
    bool entry_storage_is_outer_cell = false;
    bool inner_bound = false;
    bool inner_selected = false;
    bool inner_write_isolated = false;
    bool outer_write_isolated = false;
    bool plain_inherits_inner = false;
    bool outer_selected_after = false;
    GuestDispatchScope* left_installed = nullptr;
};

GateStatus cell_supervisor(std::uint16_t immediate, GateCall& call, GuestThreadContext& thread,
                           void* raw) {
    auto& state = *static_cast<CellNesting*>(raw);
    if (immediate == 0x52) {
        state.probe_storage = call.guest_x18_storage();
        return GateStatus::resume;
    }
    if (immediate == 0x53) {
        // The handler returns while a scope selecting another context is
        // still installed; the outer entry's rollback must not follow it.
        call.set_guest_x18(0xA9);
        state.left_installed = new GuestDispatchScope(state.table, *state.inner, {},
                                                      GateFailurePolicy::resume_unchanged);
        set_thread_guest_x18(0xB9);
        call.set_guest_x18(0xAA);
        return GateStatus::failed;
    }
    GuestRegisterCell* const outer_cell = thread.bound_register_cell();
    state.entry_x18 = call.guest_x18();
    state.entry_storage_is_outer_cell =
        outer_cell != nullptr && call.guest_x18_storage() == &outer_cell->x18;
    call.set_guest_x18(0xA1);

    GuestThreadContext& inner_thread = *state.inner;
    {
        GuestDispatchScope inner(state.table, inner_thread, {},
                                 GateFailurePolicy::resume_unchanged);
        GuestRegisterCell* const inner_cell = inner_thread.bound_register_cell();
        state.inner_bound = inner.setup_error() == GuestDispatchFailureKind::none &&
                            inner_cell != nullptr && inner_cell != outer_cell;
        state.inner_selected = thread_guest_x18() == 0xB0 &&
                               inner.gate_scope().guest_x18_storage() == &inner_cell->x18;

        // A nested entry writes the inner context's x18 only.
        GuestImage image = make_image(0x900 + immediate);
        (void)run_site(image, 2);  // mrs x18, tpidrro_el0 (inner: 0xB2)
        state.inner_write_isolated = inner_thread.guest_x18() == 0xB2 &&
                                     thread.guest_x18() == 0xA1 && inner.dispatched() == 1;

        // The outer call keeps its own storage while the inner one is selected.
        call.set_guest_x18(0xA2);
        const bool via_set_x = call.set_x(18, 0xA3) && call.x(18) == 0xA3;
        (void)run_site(image, 1);  // msr tpidr_el0, x18 (inner reads 0xB2)
        state.outer_write_isolated = via_set_x && thread.guest_x18() == 0xA3 &&
                                     inner_thread.guest_x18() == 0xB2 &&
                                     inner_thread.thread_pointer() == 0xB2 &&
                                     thread_guest_x18() == 0xB2;

        // An ordinary scope installed now inherits the inner selection.
        X18Probe probe;
        probe.do_write = true;
        probe.write = 0xB3;
        {
            GateScope plain(&x18_probe, &probe, GateFailurePolicy::resume_unchanged);
            (void)run_site(image, 7);
        }
        state.plain_inherits_inner = probe.calls == 1 && probe.storage == &inner_cell->x18 &&
                                     probe.value == 0xB2 && inner_thread.guest_x18() == 0xB3;
    }
    state.outer_selected_after = thread_guest_x18() == 0xA3 && !inner_thread.bound();
    return immediate == 0x50 ? GateStatus::resume : GateStatus::failed;
}

void test_register_cell_binding() {
    const auto table = register_cell_table();
    if (!table) {
        return;
    }
    GuestThreadContext thread;
    CHECK(thread.bound_register_cell() == nullptr);

    // Accessors and the cell are one storage.
    thread.set_guest_x18(0x18);
    thread.set_thread_pointer(0x7700);
    thread.set_read_only_thread_pointer(0x7800);
    {
        GuestDispatchScope scope(table, thread, {}, GateFailurePolicy::resume_unchanged);
        GuestRegisterCell* const cell = thread.bound_register_cell();
        CHECK(cell != nullptr);
        if (cell == nullptr) {
            return;
        }
        CHECK_EQ(reinterpret_cast<std::uintptr_t>(cell) % kGuestRegisterCellAlignment, 0);
        const auto* bytes = reinterpret_cast<const unsigned char*>(cell);
        std::uint64_t word = 0;
        std::memcpy(&word, bytes + kGuestRegisterCellX18Offset, 8);
        CHECK_EQ(word, 0x18);
        std::memcpy(&word, bytes + kGuestRegisterCellThreadPointerOffset, 8);
        CHECK_EQ(word, 0x7700);
        std::memcpy(&word, bytes + kGuestRegisterCellReadOnlyThreadPointerOffset, 8);
        CHECK_EQ(word, 0x7800);
        CHECK(scope.gate_scope().guest_x18_storage() == &cell->x18);
        CHECK_EQ(thread_guest_x18(), 0x18);

        // Nested binding of the same context: same cell, still bound after.
        {
            GuestDispatchScope again(table, thread, {}, GateFailurePolicy::resume_unchanged);
            CHECK(thread.bound_register_cell() == cell);
        }
        CHECK(thread.bound_register_cell() == cell);

        // Another host thread never receives the cell.
        GuestRegisterCell* foreign = cell;
        std::thread([&] { foreign = thread.bound_register_cell(); }).join();
        CHECK(foreign == nullptr);

        // Direct cell writes are what gate entries observe.
        cell->thread_pointer = 0x1234'5678'0000ull;
        GuestImage in = make_image(0x880);
        GuestImage expected = in;
        expected.x[5] = 0x1234'5678'0000ull;
        check_image(run_site(in, 0), expected, 0, "mrs x5, tpidr_el0 after cell write");

        cell->x18 = 0x0F0E'0D0C'0B0A'0908ull;
        in = make_image(0x881);
        check_image(run_site(in, 1), in, 1, "msr tpidr_el0, x18 after cell write");
        CHECK_EQ(cell->thread_pointer, 0x0F0E'0D0C'0B0A'0908ull);
        CHECK_EQ(thread.thread_pointer(), 0x0F0E'0D0C'0B0A'0908ull);

        // Gate writes are what the cell holds.
        cell->read_only_thread_pointer = 0x4444'0000ull;
        in = make_image(0x882);
        check_image(run_site(in, 2), in, 2, "mrs x18, tpidrro_el0 into the cell");
        CHECK_EQ(cell->x18, 0x4444'0000ull);
        CHECK_EQ(thread.guest_x18(), 0x4444'0000ull);
        CHECK_EQ(thread_guest_x18(), 0x4444'0000ull);

        // Supervisor entries use the cell's x18 word.
        CellNesting state;
        GuestDispatchScope svc_scope(table, thread, {&cell_supervisor, &state},
                                     GateFailurePolicy::resume_unchanged);
        in = make_image(0x883);
        check_image(run_site(in, 5), in, 5, "svc storage probe");
        CHECK(state.probe_storage == &cell->x18);
        CHECK_EQ(svc_scope.failure_count(), 0);
        CHECK_EQ(scope.failure_count(), 0);
        CHECK_EQ(scope.gate_scope().failure_count(), 0);
    }
    CHECK(!thread.bound());
    CHECK(thread.bound_register_cell() == nullptr);
    CHECK_EQ(thread.guest_x18(), 0x4444'0000ull);

    // A scope whose setup failed exposes no cell and selects no storage.
    std::uint64_t before = 0x5150;
    set_thread_guest_x18(before);
    {
        GuestDispatchScope broken(nullptr, thread, {}, GateFailurePolicy::resume_unchanged);
        CHECK_KIND(broken.setup_error(), GuestDispatchFailureKind::missing_table);
        CHECK(thread.bound_register_cell() == nullptr);
        CHECK(broken.gate_scope().guest_x18_storage() != nullptr &&
              *broken.gate_scope().guest_x18_storage() == before);
        set_thread_guest_x18(0x5151);
        CHECK_EQ(thread.guest_x18(), 0x4444'0000ull);
    }
    CHECK_EQ(thread_guest_x18(), 0x5151);
    set_thread_guest_x18(0);
}

void test_register_cell_nesting() {
    const auto table = register_cell_table();
    if (!table) {
        return;
    }
    GuestThreadContext outer;
    GuestThreadContext inner;
    CellNesting state;
    state.table = table;
    state.inner = &inner;
    set_thread_guest_x18(0xD0);

    GuestDispatchScope scope(table, outer, {&cell_supervisor, &state},
                             GateFailurePolicy::resume_unchanged);
    for (std::uint16_t immediate : {std::uint16_t{0x50}, std::uint16_t{0x51}}) {
        outer.set_guest_x18(0xA0);
        inner.set_guest_x18(0xB0);
        inner.set_thread_pointer(0);
        inner.set_read_only_thread_pointer(0xB2);
        state = CellNesting{table, &inner};
        const unsigned site = immediate == 0x50 ? 3 : 4;
        const GuestImage in = make_image(0x890 + immediate);
        check_image(run_site(in, site), in, site, "nested distinct contexts");
        CHECK_EQ(state.entry_x18, 0xA0);
        CHECK(state.entry_storage_is_outer_cell);
        CHECK(state.inner_bound);
        CHECK(state.inner_selected);
        CHECK(state.inner_write_isolated);
        CHECK(state.outer_write_isolated);
        CHECK(state.plain_inherits_inner);
        CHECK(state.outer_selected_after);
        // The inner context's committed entries survive either outcome; the
        // outer entry's x18 is committed on resume and rolled back on failure.
        CHECK_EQ(inner.guest_x18(), 0xB3);
        CHECK_EQ(inner.thread_pointer(), 0xB2);
        CHECK_EQ(outer.guest_x18(), immediate == 0x50 ? 0xA3 : 0xA0);
        CHECK_EQ(thread_guest_x18(), outer.guest_x18());
    }
    CHECK_EQ(scope.failure_count(), 1);
    CHECK(scope.first_failure() != nullptr &&
          scope.first_failure()->kind == GuestDispatchFailureKind::supervisor_failed &&
          scope.first_failure()->return_address == site_return(4));
    CHECK_EQ(scope.gate_scope().failure_count(), 1);

    // Failure rollback while another context is still selected restores the
    // outer entry's own cell and leaves the selected one alone.
    outer.set_guest_x18(0xA0);
    inner.set_guest_x18(0xB0);
    state = CellNesting{table, &inner};
    const GuestImage in = make_image(0x8F0);
    check_image(run_site(in, 8), in, 8, "outer failure with inner context selected");
    CHECK(state.left_installed != nullptr);
    CHECK_EQ(outer.guest_x18(), 0xA0);
    CHECK_EQ(inner.guest_x18(), 0xB9);
    CHECK_EQ(thread_guest_x18(), 0xB9);
    CHECK(inner.bound());
    delete state.left_installed;
    CHECK(!inner.bound());
    CHECK_EQ(thread_guest_x18(), 0xA0);
    CHECK_EQ(scope.failure_count(), 2);
    CHECK_EQ(GateScope::thread_gate_depth(), 0);
}

void test_register_cell_selection() {
    const auto table = register_cell_table();
    if (!table) {
        return;
    }
    // Plain scopes use the thread default; explicit storage replaces it until
    // the scope ends; a dispatch scope selects its context and restores.
    set_thread_guest_x18(0xDEF0);
    X18Probe probe;
    GuestThreadContext thread;
    thread.set_guest_x18(0xC0);
    {
        GateScope plain(&x18_probe, &probe, GateFailurePolicy::resume_unchanged);
        (void)run_site(make_image(0x8A0), 6);
        CHECK_EQ(probe.value, 0xDEF0);
        const std::uint64_t* const thread_default = probe.storage;

        std::uint64_t explicit_word = 0xE0;
        {
            GateScope own(&x18_probe, &probe, GateFailurePolicy::resume_unchanged,
                          &explicit_word);
            CHECK(own.guest_x18_storage() == &explicit_word);
            CHECK_EQ(thread_guest_x18(), 0xE0);
            probe.do_write = true;
            probe.write = 0xE1;
            (void)run_site(make_image(0x8A1), 6);
            probe.do_write = false;
            CHECK(probe.storage == &explicit_word);
            CHECK_EQ(explicit_word, 0xE1);
            set_thread_guest_x18(0xE2);
            CHECK_EQ(explicit_word, 0xE2);
        }
        CHECK_EQ(thread_guest_x18(), 0xDEF0);

        // The same holds for ordinary scopes with explicit storage.
        {
            std::uint64_t outer_word = 0x0A;
            std::uint64_t inner_word = 0x0B;
            struct LeaveInstalled {
                std::uint64_t* inner_word;
                GateScope* installed = nullptr;
                static GateStatus handle(GateCall& call, void* raw) {
                    auto& self = *static_cast<LeaveInstalled*>(raw);
                    call.set_guest_x18(0x1A);
                    self.installed = new GateScope(&x18_probe, nullptr,
                                                   GateFailurePolicy::resume_unchanged,
                                                   self.inner_word);
                    set_thread_guest_x18(0x1B);
                    call.set_guest_x18(0x2A);
                    return GateStatus::failed;
                }
            } leave{&inner_word};
            GateScope own(&LeaveInstalled::handle, &leave, GateFailurePolicy::resume_unchanged,
                          &outer_word);
            (void)run_site(make_image(0x8A4), 6);
            CHECK_EQ(own.failure_count(), 1);
            CHECK_EQ(outer_word, 0x0A);
            CHECK_EQ(inner_word, 0x1B);
            CHECK(leave.installed != nullptr);
            delete leave.installed;
            CHECK_EQ(thread_guest_x18(), 0x0A);
        }

        {
            GuestDispatchScope dispatch(table, thread, {}, GateFailurePolicy::resume_unchanged);
            CHECK_EQ(thread_guest_x18(), 0xC0);
            set_thread_guest_x18(0xC1);
            CHECK_EQ(thread.guest_x18(), 0xC1);
            {
                GateScope nested(&x18_probe, &probe, GateFailurePolicy::resume_unchanged);
                probe.do_write = true;
                probe.write = 0xC2;
                (void)run_site(make_image(0x8A2), 6);
                probe.do_write = false;
                CHECK(probe.storage == &thread.bound_register_cell()->x18);
                CHECK_EQ(probe.value, 0xC1);
            }
            CHECK_EQ(thread.guest_x18(), 0xC2);
        }
        CHECK_EQ(thread_guest_x18(), 0xDEF0);
        (void)run_site(make_image(0x8A3), 6);
        CHECK(probe.storage == thread_default);
        CHECK_EQ(probe.value, 0xDEF0);
    }
    CHECK_EQ(thread_guest_x18(), 0xDEF0);
    CHECK_EQ(thread.guest_x18(), 0xC2);
    set_thread_guest_x18(0);
}

void test_register_cell_threads() {
    const auto table = register_cell_table();
    if (!table) {
        return;
    }
    set_thread_guest_x18(0x3A1);
    GuestThreadContext migrating;
    migrating.set_read_only_thread_pointer(0x1111'0000ull);

    // Sequential migration: the context's x18 follows it; each host thread's
    // own default is untouched and restored when the binding ends.
    struct Leg {
        GuestRegisterCell* cell = nullptr;
        bool ok = false;
    };
    Leg first;
    std::thread([&] {
        set_thread_guest_x18(0x7101);
        bool ok = true;
        {
            GuestDispatchScope scope(table, migrating, {}, GateFailurePolicy::resume_unchanged);
            first.cell = migrating.bound_register_cell();
            const GuestImage in = make_image(0x8B0);
            const GuestImage out = run_site(in, 2);  // mrs x18, tpidrro_el0
            ok = ok && out.x18_after == out.x18_before && thread_guest_x18() == 0x1111'0000ull &&
                 scope.dispatched() == 1;
        }
        ok = ok && thread_guest_x18() == 0x7101 && migrating.bound_register_cell() == nullptr;
        first.ok = ok;
    }).join();
    CHECK(first.ok);
    CHECK_EQ(migrating.guest_x18(), 0x1111'0000ull);

    Leg second;
    std::thread([&] {
        set_thread_guest_x18(0x7202);
        bool ok = true;
        {
            GuestDispatchScope scope(table, migrating, {}, GateFailurePolicy::resume_unchanged);
            second.cell = migrating.bound_register_cell();
            ok = ok && thread_guest_x18() == 0x1111'0000ull;
            const GuestImage in = make_image(0x8B1);
            (void)run_site(in, 1);  // msr tpidr_el0, x18
            ok = ok && migrating.thread_pointer() == 0x1111'0000ull && scope.dispatched() == 1;
            set_thread_guest_x18(0x2222'0000ull);
        }
        ok = ok && thread_guest_x18() == 0x7202;
        second.ok = ok;
    }).join();
    CHECK(second.ok);
    CHECK(first.cell != nullptr && first.cell == second.cell);
    CHECK_EQ(migrating.guest_x18(), 0x2222'0000ull);
    CHECK_EQ(thread_guest_x18(), 0x3A1);

    // A binding refused on another thread selects none of the context's
    // storage there; that thread keeps its own default.
    GuestThreadContext owned;
    owned.set_guest_x18(0x0E0E);
    owned.set_thread_pointer(0x0F0F);
    {
        GuestDispatchScope owner(table, owned, {}, GateFailurePolicy::resume_unchanged);
        GuestRegisterCell* const cell = owned.bound_register_cell();
        bool refused = false;
        bool isolated = false;
        std::thread([&] {
            set_thread_guest_x18(0x9090);
            GuestDispatchScope other(table, owned, {}, GateFailurePolicy::resume_unchanged);
            refused = other.setup_error() == GuestDispatchFailureKind::context_bound_elsewhere;
            const std::uint64_t* const storage = other.gate_scope().guest_x18_storage();
            bool ok = storage != nullptr && storage != &cell->x18 && *storage == 0x9090 &&
                      owned.bound_register_cell() == nullptr && thread_guest_x18() == 0x9090;
            set_thread_guest_x18(0x9191);
            const GuestImage in = make_image(0x8B2);
            const GuestImage out = run_site(in, 2);  // fails: setup error
            ok = ok && out.x[5] == in.x[5] && thread_guest_x18() == 0x9191 &&
                 other.failure_count() == 1;
            isolated = ok;
        }).join();
        CHECK(refused);
        CHECK(isolated);
        CHECK_EQ(owned.guest_x18(), 0x0E0E);
        CHECK_EQ(owned.thread_pointer(), 0x0F0F);
        CHECK(owned.bound_register_cell() == cell);
        CHECK_EQ(thread_guest_x18(), 0x0E0E);
        CHECK_EQ(owner.failure_count(), 0);
    }
    CHECK_EQ(thread_guest_x18(), 0x3A1);
    set_thread_guest_x18(0);
}

// ---------------------------------------------------------------------------
// Process-fatal scenarios, executed in a child process
// ---------------------------------------------------------------------------

constexpr int kChildHookExitBase = 100;

void child_trap_hook(const GateFailure& failure) noexcept {
    _exit(kChildHookExitBase + static_cast<int>(failure.kind));
}

[[noreturn]] void run_child(const std::string& mode) {
    set_gate_trap_hook(&child_trap_hook);
    const auto table = build_table({{0, svc(1)}});
    const GuestImage in = make_image(0x700);
    if (mode == "trap-unknown-site") {
        GuestThreadContext thread;
        GuestDispatchScope scope(table, thread);  // default trap policy
        run_site(in, 5);
    } else if (mode == "trap-missing-handler") {
        GuestThreadContext thread;
        GuestDispatchScope scope(table, thread);
        run_site(in, 0);
    } else if (mode == "context-destroyed-while-bound") {
        auto* thread = new GuestThreadContext;
        auto* scope = new GuestDispatchScope(table, *thread);
        delete thread;  // Must abort: the scope still binds it.
        delete scope;
    }
    _exit(1);
}

struct ChildOutcome {
    bool spawned = false;
    bool exited = false;
    int exit_code = -1;
    int signal = 0;
};

ChildOutcome spawn_child(const char* mode) {
    ChildOutcome outcome;
    char path[4096];
    std::uint32_t size = sizeof path;
    if (_NSGetExecutablePath(path, &size) != 0) {
        return outcome;
    }
    posix_spawn_file_actions_t actions;
    posix_spawn_file_actions_init(&actions);
    posix_spawn_file_actions_addopen(&actions, STDERR_FILENO, "/dev/null", O_WRONLY, 0);
    char arg_child[] = "--child";
    std::string mode_text = mode;
    char* argv[] = {path, arg_child, mode_text.data(), nullptr};
    pid_t pid = 0;
    const int error = posix_spawn(&pid, path, &actions, nullptr, argv, environ);
    posix_spawn_file_actions_destroy(&actions);
    if (error != 0) {
        return outcome;
    }
    outcome.spawned = true;
    int status = 0;
    if (waitpid(pid, &status, 0) != pid) {
        return outcome;
    }
    if (WIFEXITED(status)) {
        outcome.exited = true;
        outcome.exit_code = WEXITSTATUS(status);
    } else if (WIFSIGNALED(status)) {
        outcome.signal = WTERMSIG(status);
    }
    return outcome;
}

void test_fatal_scenarios_in_children() {
    const char* const trapped[] = {"trap-unknown-site", "trap-missing-handler"};
    for (const char* mode : trapped) {
        const ChildOutcome outcome = spawn_child(mode);
        CHECK(outcome.spawned);
        if (!outcome.exited ||
            outcome.exit_code !=
                kChildHookExitBase + static_cast<int>(GateFailureKind::handler_reported)) {
            std::printf("FAIL child mode %s: exited=%d code=%d signal=%d\n", mode, outcome.exited,
                        outcome.exit_code, outcome.signal);
            ++g_checks_failed;
        }
        ++g_checks_total;
    }
    const ChildOutcome aborted = spawn_child("context-destroyed-while-bound");
    CHECK(aborted.spawned);
    CHECK(!aborted.exited);
    CHECK_EQ(aborted.signal, SIGABRT);
}

}  // namespace

int main(int argc, char** argv) {
    if (argc == 3 && std::strcmp(argv[1], "--child") == 0) {
        run_child(argv[2]);
    }

    test_table_validation();
    test_thread_registers();
    test_cache_sites();
    test_move_wide_sites();
    test_counter();
    test_supervisor_calls();
    test_unknown_sites();
    test_nesting();
    test_nested_failure_rollback();
    test_sequence();
    test_thread_isolation();
    test_register_cell_binding();
    test_register_cell_nesting();
    test_register_cell_selection();
    test_register_cell_threads();
    test_fatal_scenarios_in_children();

    std::printf("guest dispatch tests: %d/%d checks passed\n", g_checks_total - g_checks_failed,
                g_checks_total);
    return g_checks_failed == 0 ? 0 : 1;
}
