// ============================================================================
// Armigo Tests - Native Runtime Verification
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

// Executable verification of the AArch64 host gate on Apple Silicon.
//
// Every scenario drives the gate through the synthetic fixture in
// fixtures.S, which loads a register image, enters the gate with a direct
// `bl` and records what the gate handed back. Checks compare that record
// against the image and against the explicit changes a handler made.
// Process-fatal behaviors run in a child copy of this executable.

#include <armigo/execution/gate.h>

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <thread>
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

// ---------------------------------------------------------------------------
// Fixture interface (fixtures.S)
// ---------------------------------------------------------------------------

struct alignas(16) FixtureImage {
    GateFrame frame;
    std::uint64_t x18_before;
    std::uint64_t sp_before;
    std::uint64_t sp_after;
    std::uint64_t return_label;
    std::uint64_t alternate_marker;
    std::uint64_t x18_after;
};

static_assert(offsetof(FixtureImage, frame) == 0);
static_assert(offsetof(FixtureImage, x18_before) == ARMIGO_GATE_FRAME_SIZE + 0);
static_assert(offsetof(FixtureImage, sp_before) == ARMIGO_GATE_FRAME_SIZE + 8);
static_assert(offsetof(FixtureImage, sp_after) == ARMIGO_GATE_FRAME_SIZE + 16);
static_assert(offsetof(FixtureImage, return_label) == ARMIGO_GATE_FRAME_SIZE + 24);
static_assert(offsetof(FixtureImage, alternate_marker) == ARMIGO_GATE_FRAME_SIZE + 32);
static_assert(offsetof(FixtureImage, x18_after) == ARMIGO_GATE_FRAME_SIZE + 40);
static_assert(sizeof(FixtureImage) == ARMIGO_GATE_FRAME_SIZE + 48);

extern "C" {
void armigo_test_fixture_run_gate(const FixtureImage* in, FixtureImage* out);
void armigo_test_fixture_gate_return();
void armigo_test_fixture_alternate_return();
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
// Register images
// ---------------------------------------------------------------------------

constexpr std::uint64_t kFpsrIoc = 1u << 0;
constexpr std::uint64_t kFpsrDzc = 1u << 1;
constexpr std::uint64_t kFpsrOfc = 1u << 2;
constexpr std::uint64_t kFpsrUfc = 1u << 3;
constexpr std::uint64_t kFpsrIxc = 1u << 4;
constexpr std::uint64_t kFpsrIdc = 1u << 7;
constexpr std::uint64_t kFpsrQc = 1u << 27;
constexpr std::uint64_t kFpcrRoundPlusInfinity = 1u << 22;  // RMode = 0b01
constexpr std::uint64_t kFpcrRoundMinusInfinity = 2u << 22;  // RMode = 0b10
constexpr std::uint64_t kFpcrFlushToZero = 1u << 24;
constexpr std::uint64_t kFpcrDefaultNan = 1u << 25;
constexpr std::uint64_t kFpcrAlternateHalf = 1u << 26;
constexpr std::uint64_t kNzcvMask = 0xFu << 28;

// The fixture ignores this slot and the gate stores zero there; the sentinel
// proves neither leaks the value anywhere.
constexpr std::uint64_t kHostReservedSentinel = 0xDEAD18DEAD18DEADull;

std::uint64_t mix(std::uint64_t seed, std::uint64_t salt) {
    std::uint64_t value = seed * 0x9E3779B97F4A7C15ull + salt * 0xBF58476D1CE4E5B9ull;
    value ^= value >> 29;
    value *= 0x94D049BB133111EBull;
    value ^= value >> 32;
    return value;
}

FixtureImage make_image(std::uint64_t seed, std::uint64_t nzcv, std::uint64_t fpsr,
                        std::uint64_t fpcr) {
    FixtureImage image{};
    for (unsigned i = 0; i < kGateGeneralRegisterCount; ++i) {
        image.frame.x[i] = mix(seed, i);
    }
    image.frame.x[kGateHostReservedRegister] = kHostReservedSentinel;
    for (unsigned i = 0; i < kGateVectorRegisterCount; ++i) {
        image.frame.v[i] = GateVector{{mix(seed, 100 + i), mix(seed, 200 + i)}};
    }
    image.frame.nzcv = nzcv & kNzcvMask;
    image.frame.fpsr = fpsr;
    image.frame.fpcr = fpcr;
    return image;
}

FixtureImage make_image(std::uint64_t seed) {
    return make_image(seed, (seed & 0xF) << 28, kFpsrIoc | kFpsrIxc | kFpsrQc,
                      kFpcrRoundPlusInfinity | kFpcrFlushToZero);
}

// Compares the registers the gate is contracted to carry, skipping x18 and
// x30, which the fixture handles separately.
void check_registers(const GateFrame& actual, const GateFrame& expected, const char* what) {
    for (unsigned i = 0; i < kGateGeneralRegisterCount; ++i) {
        if (i == kGateHostReservedRegister || i == kGateLinkRegister) {
            continue;
        }
        if (actual.x[i] != expected.x[i]) {
            std::printf("FAIL %s: x%u actual 0x%llx expected 0x%llx\n", what, i,
                        static_cast<unsigned long long>(actual.x[i]),
                        static_cast<unsigned long long>(expected.x[i]));
            ++g_checks_failed;
        }
        ++g_checks_total;
    }
    for (unsigned i = 0; i < kGateVectorRegisterCount; ++i) {
        if (!(actual.v[i] == expected.v[i])) {
            std::printf("FAIL %s: q%u actual {0x%llx,0x%llx} expected {0x%llx,0x%llx}\n", what, i,
                        static_cast<unsigned long long>(actual.v[i].lane[0]),
                        static_cast<unsigned long long>(actual.v[i].lane[1]),
                        static_cast<unsigned long long>(expected.v[i].lane[0]),
                        static_cast<unsigned long long>(expected.v[i].lane[1]));
            ++g_checks_failed;
        }
        ++g_checks_total;
    }
    record_equal(actual.nzcv & kNzcvMask, expected.nzcv & kNzcvMask, what, __FILE__, __LINE__);
    record_equal(actual.fpsr, expected.fpsr, what, __FILE__, __LINE__);
    record_equal(actual.fpcr, expected.fpcr, what, __FILE__, __LINE__);
}

// Checks the facts the fixture records about the host side of the gate.
void check_host_facts(const FixtureImage& out, bool expect_alternate_path = false) {
    CHECK_EQ(out.sp_after, out.sp_before);
    CHECK_EQ(out.x18_after, out.x18_before);
    CHECK_EQ(out.return_label, address_of(&armigo_test_fixture_gate_return));
    if (expect_alternate_path) {
        CHECK(out.alternate_marker != 0);
    } else {
        CHECK_EQ(out.alternate_marker, 0);
    }
}

// Context shared with handlers: the image that was loaded and the output
// image the fixture is writing (its pre-gate fields are already valid).
struct HandlerContext {
    const FixtureImage* in = nullptr;
    FixtureImage* out = nullptr;
    unsigned calls = 0;
    std::uint64_t observed_depth = 0;
    const GateCall* observed_outer = reinterpret_cast<const GateCall*>(1);
    bool frame_matched = true;
    int variant = 0;
    std::uint64_t value = 0;
    HandlerContext* inner = nullptr;
    unsigned inner_calls = 0;
};

// Verifies that the frame the dispatcher presents matches what the fixture
// loaded, including the host-only facts.
void check_frame_against_image(GateCall& call, HandlerContext& context) {
    const FixtureImage& in = *context.in;
    const GateFrame& frame = call.frame();
    bool matched = true;
    for (unsigned i = 0; i < kGateGeneralRegisterCount; ++i) {
        if (i == kGateHostReservedRegister || i == kGateLinkRegister) {
            continue;
        }
        matched = matched && frame.x[i] == in.frame.x[i] && call.x(i) == in.frame.x[i];
    }
    for (unsigned i = 0; i < kGateVectorRegisterCount; ++i) {
        matched = matched && frame.v[i] == in.frame.v[i] && call.v(i) == in.frame.v[i];
    }
    matched = matched && (frame.nzcv & kNzcvMask) == in.frame.nzcv;
    matched = matched && frame.fpsr == in.frame.fpsr && frame.fpcr == in.frame.fpcr;
    matched = matched && frame.x[kGateHostReservedRegister] == 0;
    matched = matched && frame.x[kGateLinkRegister] == address_of(&armigo_test_fixture_gate_return);
    matched = matched && call.return_address() == address_of(&armigo_test_fixture_gate_return);
    matched = matched && frame.sp == context.out->sp_before;
    matched = matched && frame.sp == reinterpret_cast<std::uint64_t>(&frame) + sizeof(GateFrame);
    matched = matched && frame.magic == kGateFrameMagic;
    matched = matched && frame.frame_record[0] == in.frame.x[kGateFramePointerRegister];
    matched = matched && frame.frame_record[1] == address_of(&armigo_test_fixture_gate_return);
    matched = matched && GateScope::thread_gate_depth() == call.depth();
    context.frame_matched = context.frame_matched && matched;
}

GateStatus observe_only(GateCall& call, void* raw) {
    auto& context = *static_cast<HandlerContext*>(raw);
    ++context.calls;
    context.observed_depth = call.depth();
    context.observed_outer = call.outer();
    check_frame_against_image(call, context);
    return GateStatus::resume;
}

// Applies a deterministic transformation to everything a handler may change.
void transform(GateCall& call, std::uint64_t salt) {
    for (unsigned i = 0; i < kGateGeneralRegisterCount; ++i) {
        if (i != kGateHostReservedRegister && i != kGateLinkRegister) {
            call.set_x(i, call.x(i) ^ mix(salt, i));
        }
    }
    for (unsigned i = 0; i < kGateVectorRegisterCount; ++i) {
        GateVector vector = call.v(i);
        vector.lane[0] ^= mix(salt, 300 + i);
        vector.lane[1] ^= mix(salt, 400 + i);
        call.set_v(i, vector);
    }
    call.set_nzcv((call.nzcv() ^ 0x90000000u) & kNzcvMask);
    call.set_fpsr(kFpsrDzc | kFpsrOfc | kFpsrUfc | kFpsrIdc);
    call.set_fpcr(kFpcrRoundMinusInfinity | kFpcrDefaultNan | kFpcrAlternateHalf);
}

FixtureImage expected_after_transform(const FixtureImage& in, std::uint64_t salt) {
    FixtureImage expected = in;
    for (unsigned i = 0; i < kGateGeneralRegisterCount; ++i) {
        if (i == kGateHostReservedRegister || i == kGateLinkRegister) {
            continue;
        }
        expected.frame.x[i] ^= mix(salt, i);
    }
    for (unsigned i = 0; i < kGateVectorRegisterCount; ++i) {
        expected.frame.v[i].lane[0] ^= mix(salt, 300 + i);
        expected.frame.v[i].lane[1] ^= mix(salt, 400 + i);
    }
    expected.frame.nzcv = (in.frame.nzcv ^ 0x90000000u) & kNzcvMask;
    expected.frame.fpsr = kFpsrDzc | kFpsrOfc | kFpsrUfc | kFpsrIdc;
    expected.frame.fpcr = kFpcrRoundMinusInfinity | kFpcrDefaultNan | kFpcrAlternateHalf;
    return expected;
}

GateStatus transform_handler(GateCall& call, void* raw) {
    auto& context = *static_cast<HandlerContext*>(raw);
    ++context.calls;
    check_frame_against_image(call, context);
    transform(call, context.value);
    return GateStatus::resume;
}

FixtureImage run_fixture(const FixtureImage& in, HandlerContext& context) {
    FixtureImage out{};
    context.in = &in;
    context.out = &out;
    armigo_test_fixture_run_gate(&in, &out);
    return out;
}

// ---------------------------------------------------------------------------
// Scenarios
// ---------------------------------------------------------------------------

void test_identity_roundtrip() {
    const std::uint64_t fpcrs[] = {0, kFpcrRoundPlusInfinity | kFpcrFlushToZero,
                                   kFpcrRoundMinusInfinity | kFpcrDefaultNan | kFpcrAlternateHalf};
    const std::uint64_t fpsrs[] = {0, kFpsrIoc | kFpsrDzc | kFpsrOfc | kFpsrUfc | kFpsrIxc | kFpsrIdc,
                                   kFpsrQc};
    for (std::uint64_t flags = 0; flags < 16; ++flags) {
        const FixtureImage in = make_image(0x1000 + flags, flags << 28, fpsrs[flags % 3],
                                           fpcrs[flags % 3]);
        HandlerContext context;
        GateScope scope(&observe_only, &context, GateFailurePolicy::resume_unchanged);
        const FixtureImage out = run_fixture(in, context);
        CHECK_EQ(context.calls, 1);
        CHECK_EQ(context.observed_depth, 1);
        CHECK(context.observed_outer == nullptr);
        CHECK(context.frame_matched);
        check_registers(out.frame, in.frame, "identity");
        CHECK_EQ(out.frame.x[kGateLinkRegister], address_of(&armigo_test_fixture_gate_return));
        CHECK_EQ(out.frame.x[kGateHostReservedRegister], 0);
        check_host_facts(out);
        CHECK_EQ(scope.failure_count(), 0);
        CHECK_EQ(GateScope::thread_gate_depth(), 0);
    }
}

void test_handler_changes_apply() {
    for (std::uint64_t seed = 1; seed <= 4; ++seed) {
        const FixtureImage in = make_image(0x2000 + seed);
        HandlerContext context;
        context.value = 0x77 + seed;
        GateScope scope(&transform_handler, &context, GateFailurePolicy::resume_unchanged);
        const FixtureImage out = run_fixture(in, context);
        CHECK_EQ(context.calls, 1);
        CHECK(context.frame_matched);
        check_registers(out.frame, expected_after_transform(in, context.value).frame,
                        "transform");
        CHECK_EQ(out.frame.x[kGateLinkRegister], address_of(&armigo_test_fixture_gate_return));
        check_host_facts(out);
        CHECK_EQ(scope.failure_count(), 0);
    }

    // Out-of-range register indices are rejected rather than aliased.
    const FixtureImage in = make_image(0x2100);
    HandlerContext context;
    GateScope scope(
        +[](GateCall& call, void* raw) {
            auto& inner = *static_cast<HandlerContext*>(raw);
            inner.frame_matched = !call.set_x(31, 1) && !call.set_v(32, GateVector{{1, 1}}) &&
                                  call.x(31) == 0 && call.v(32) == GateVector{{0, 0}};
            return GateStatus::resume;
        },
        &context, GateFailurePolicy::resume_unchanged);
    const FixtureImage out = run_fixture(in, context);
    CHECK(context.frame_matched);
    check_registers(out.frame, in.frame, "out-of-range");
}

void test_virtual_x18() {
    std::uint64_t main_thread_x18 = 0;
    set_thread_guest_x18(0x3333);
    std::thread worker([&] {
        CHECK_EQ(thread_guest_x18(), 0);
        const FixtureImage in = make_image(0x3000);
        HandlerContext context;
        context.value = 0xA5A5A5A5A5A5A5A5ull;
        GateScope scope(
            +[](GateCall& call, void* raw) {
                auto& inner = *static_cast<HandlerContext*>(raw);
                ++inner.calls;
                if (inner.calls == 1) {
                    inner.frame_matched = call.x(18) == 0 && call.guest_x18() == 0;
                    call.set_x(18, inner.value);
                    inner.frame_matched = inner.frame_matched && call.guest_x18() == inner.value &&
                                          call.frame().x[18] == 0;
                } else {
                    inner.frame_matched = inner.frame_matched && call.x(18) == inner.value &&
                                          call.frame().x[18] == 0;
                    call.set_guest_x18(inner.value + 1);
                }
                return GateStatus::resume;
            },
            &context, GateFailurePolicy::resume_unchanged);
        const FixtureImage first = run_fixture(in, context);
        CHECK_EQ(thread_guest_x18(), context.value);
        check_registers(first.frame, in.frame, "virtual x18 first");
        CHECK_EQ(first.frame.x[kGateHostReservedRegister], 0);
        check_host_facts(first);
        const FixtureImage second = run_fixture(in, context);
        CHECK_EQ(context.calls, 2);
        CHECK(context.frame_matched);
        CHECK_EQ(thread_guest_x18(), context.value + 1);
        check_host_facts(second);
        CHECK_EQ(scope.failure_count(), 0);
        // The virtual value never reaches the hardware register.
        CHECK(second.x18_after != context.value && second.x18_after != context.value + 1);
    });
    worker.join();
    main_thread_x18 = thread_guest_x18();
    CHECK_EQ(main_thread_x18, 0x3333);
    set_thread_guest_x18(0);
}

void test_redirected_return() {
    const FixtureImage in = make_image(0x4000);
    HandlerContext context;
    GateScope scope(
        +[](GateCall& call, void* raw) {
            auto& inner = *static_cast<HandlerContext*>(raw);
            ++inner.calls;
            inner.frame_matched = call.set_x(kGateLinkRegister,
                                             address_of(&armigo_test_fixture_alternate_return));
            call.set_x(0, 0x5EC0ull);
            return GateStatus::resume;
        },
        &context, GateFailurePolicy::resume_unchanged);
    const FixtureImage out = run_fixture(in, context);
    CHECK_EQ(context.calls, 1);
    CHECK(context.frame_matched);
    check_host_facts(out, /*expect_alternate_path=*/true);
    CHECK_EQ(out.frame.x[kGateLinkRegister], address_of(&armigo_test_fixture_alternate_return));
    CHECK_EQ(out.frame.x[0], 0x5EC0ull);
    FixtureImage expected = in;
    expected.frame.x[0] = 0x5EC0ull;
    check_registers(out.frame, expected.frame, "redirect");
}

GateStatus failing_handler(GateCall& call, void* raw) {
    auto& context = *static_cast<HandlerContext*>(raw);
    ++context.calls;
    transform(call, 0x99);
    call.set_guest_x18(0xBAD);
    switch (context.variant) {
    case 0:
        call.set_failure_message("synthetic failure");
        return GateStatus::failed;
    case 1:
        throw std::runtime_error("synthetic exception");
    case 2:
        throw 42;
    case 3:
        call.frame().x[kGateHostReservedRegister] = 1;
        return GateStatus::resume;
    case 4:
        call.frame().sp += 16;
        return GateStatus::resume;
    default:
        return GateStatus::resume;
    }
}

void test_failures_resume_unchanged() {
    struct Expectation {
        int variant;
        GateFailureKind kind;
        const char* message;
    };
    const Expectation expectations[] = {
        {0, GateFailureKind::handler_reported, "synthetic failure"},
        {1, GateFailureKind::handler_threw, nullptr},
        {2, GateFailureKind::handler_threw, nullptr},
        {3, GateFailureKind::host_reserved_slot_modified, nullptr},
        {4, GateFailureKind::stack_pointer_modified, nullptr},
    };
    set_thread_guest_x18(0x1234);
    HandlerContext context;
    GateScope scope(&failing_handler, &context, GateFailurePolicy::resume_unchanged);
    std::uint64_t expected_failures = 0;
    for (const Expectation& expectation : expectations) {
        const FixtureImage in = make_image(0x5000 + static_cast<std::uint64_t>(expectation.variant));
        context.variant = expectation.variant;
        scope.clear_failures();
        const FixtureImage out = run_fixture(in, context);
        ++expected_failures;
        CHECK_EQ(context.calls, expected_failures);
        check_registers(out.frame, in.frame, "failure restores the pre-gate state");
        CHECK_EQ(out.frame.x[kGateLinkRegister], address_of(&armigo_test_fixture_gate_return));
        check_host_facts(out);
        CHECK_EQ(thread_guest_x18(), 0x1234);
        CHECK_EQ(scope.failure_count(), 1);
        const GateFailure* failure = scope.first_failure();
        CHECK(failure != nullptr);
        if (failure != nullptr) {
            CHECK_EQ(static_cast<std::uint32_t>(failure->kind),
                     static_cast<std::uint32_t>(expectation.kind));
            CHECK_EQ(failure->return_address, address_of(&armigo_test_fixture_gate_return));
            CHECK_EQ(failure->depth, 1);
            CHECK(failure->message != nullptr);
            if (expectation.message != nullptr && failure->message != nullptr) {
                CHECK(std::strcmp(failure->message, expectation.message) == 0);
            }
        }
    }
    // A scope without a handler fails explicitly instead of calling null.
    {
        GateScope empty(nullptr, nullptr, GateFailurePolicy::resume_unchanged);
        const FixtureImage in = make_image(0x5200);
        FixtureImage out{};
        armigo_test_fixture_run_gate(&in, &out);
        check_registers(out.frame, in.frame, "no handler restores the pre-gate state");
        check_host_facts(out);
        CHECK_EQ(empty.failure_count(), 1);
        CHECK(empty.first_failure() != nullptr &&
              empty.first_failure()->kind == GateFailureKind::handler_reported &&
              empty.first_failure()->message != nullptr);
        CHECK_EQ(scope.failure_count(), 1);
    }

    // Repeated failures keep the first record and count the rest.
    scope.clear_failures();
    context.variant = 0;
    const FixtureImage in = make_image(0x5100);
    run_fixture(in, context);
    context.variant = 1;
    run_fixture(in, context);
    CHECK_EQ(scope.failure_count(), 2);
    CHECK(scope.first_failure() != nullptr &&
          scope.first_failure()->kind == GateFailureKind::handler_reported);
    scope.clear_failures();
    CHECK_EQ(scope.failure_count(), 0);
    CHECK(scope.first_failure() == nullptr);
    set_thread_guest_x18(0);
}

// Depth-1 handler enters the gate again through the fixture; depth-2 handler
// inspects the outer call.
GateStatus nesting_handler(GateCall& call, void* raw) {
    auto& context = *static_cast<HandlerContext*>(raw);
    if (call.depth() == 1) {
        ++context.calls;
        check_frame_against_image(call, context);
        HandlerContext& inner = *context.inner;
        inner.value = reinterpret_cast<std::uint64_t>(&call);
        const FixtureImage inner_in = make_image(0x6100);
        const FixtureImage inner_out = run_fixture(inner_in, inner);
        ++context.inner_calls;
        FixtureImage expected = inner_in;
        expected.frame.x[1] = 0x1111;
        check_registers(inner_out.frame, expected.frame, "nested inner");
        check_host_facts(inner_out);
        // The outer frame is intact after the inner gate completed.
        context.frame_matched = context.frame_matched && GateScope::thread_gate_depth() == 1;
        check_frame_against_image(call, context);
        call.set_x(0, 0x0000);
        return GateStatus::resume;
    }
    // Depth 2: the same scope context arrives; bookkeeping goes to the
    // inner context whose image the nested fixture run loaded.
    HandlerContext& inner = *context.inner;
    ++inner.calls;
    inner.observed_depth = call.depth();
    inner.observed_outer = call.outer();
    const GateCall* outer = call.outer();
    inner.frame_matched = inner.frame_matched && outer != nullptr &&
                          reinterpret_cast<std::uint64_t>(outer) == inner.value &&
                          outer->depth() == 1 &&
                          outer->return_address() == address_of(&armigo_test_fixture_gate_return) &&
                          outer->x(5) == make_image(0x6000).frame.x[5] &&
                          outer->sp() > call.sp() && GateScope::thread_gate_depth() == 2;
    check_frame_against_image(call, inner);
    call.set_x(1, 0x1111);
    return GateStatus::resume;
}

void test_nested_gate_entries() {
    {
        const FixtureImage in = make_image(0x6000);
        HandlerContext outer_context;
        HandlerContext inner_context;
        outer_context.inner = &inner_context;
        // Both depths dispatch to the same scope with the outer context; the
        // handler routes the depth-2 entry to the inner context itself.
        GateScope scope(&nesting_handler, &outer_context, GateFailurePolicy::resume_unchanged);
        outer_context.in = &in;
        FixtureImage out{};
        outer_context.out = &out;
        armigo_test_fixture_run_gate(&in, &out);
        CHECK_EQ(outer_context.calls, 1);
        CHECK_EQ(outer_context.inner_calls, 1);
        CHECK_EQ(inner_context.calls, 1);
        CHECK_EQ(inner_context.observed_depth, 2);
        CHECK(outer_context.frame_matched);
        CHECK(inner_context.frame_matched);
        FixtureImage expected = in;
        expected.frame.x[0] = 0;
        check_registers(out.frame, expected.frame, "nested outer");
        check_host_facts(out);
        CHECK_EQ(scope.failure_count(), 0);
        CHECK_EQ(GateScope::thread_gate_depth(), 0);
    }

    // An inner scope installed inside a handler shadows the outer one for
    // the nested entry and is removed again before the outer handler returns.
    {
        struct Shadow {
            HandlerContext outer;
            HandlerContext inner;
        } shadow;
        const FixtureImage in = make_image(0x6200);
        GateScope scope(
            +[](GateCall& call, void* raw) {
                auto& context = *static_cast<Shadow*>(raw);
                ++context.outer.calls;
                context.outer.frame_matched = call.depth() == 1;
                GateScope inner(&observe_only, &context.inner, GateFailurePolicy::resume_unchanged);
                const FixtureImage inner_in = make_image(0x6300);
                const FixtureImage inner_out = run_fixture(inner_in, context.inner);
                check_registers(inner_out.frame, inner_in.frame, "shadowed inner");
                check_host_facts(inner_out);
                return GateStatus::resume;
            },
            &shadow, GateFailurePolicy::resume_unchanged);
        FixtureImage out{};
        armigo_test_fixture_run_gate(&in, &out);
        CHECK_EQ(shadow.outer.calls, 1);
        CHECK_EQ(shadow.inner.calls, 1);
        CHECK_EQ(shadow.inner.observed_depth, 2);
        CHECK(shadow.inner.observed_outer != nullptr);
        CHECK(shadow.outer.frame_matched);
        CHECK(shadow.inner.frame_matched);
        check_registers(out.frame, in.frame, "shadowing outer");
        check_host_facts(out);
    }

    // Sequential scopes on one thread: the innermost receives the entry.
    {
        HandlerContext first;
        HandlerContext second;
        const FixtureImage in = make_image(0x6400);
        GateScope outer(&observe_only, &first, GateFailurePolicy::resume_unchanged);
        {
            GateScope inner(&observe_only, &second, GateFailurePolicy::resume_unchanged);
            run_fixture(in, second);
            CHECK_EQ(first.calls, 0);
            CHECK_EQ(second.calls, 1);
        }
        run_fixture(in, first);
        CHECK_EQ(first.calls, 1);
        CHECK_EQ(second.calls, 1);
        CHECK(first.frame_matched && second.frame_matched);
    }
}

struct ThreadResult {
    std::uint64_t tag = 0;
    std::uint64_t iterations = 0;
    std::uint64_t mismatches = 0;
    std::uint64_t failures = 0;
    bool x18_isolated = true;
};

GateStatus thread_handler(GateCall& call, void* raw) {
    auto& result = *static_cast<ThreadResult*>(raw);
    if (call.x(18) != result.tag || call.frame().x[kGateHostReservedRegister] != 0) {
        result.x18_isolated = false;
    }
    if (call.depth() == 1) {
        // Every 16th entry nests once so concurrent threads also exercise
        // nested dispatch.
        if ((call.x(1) & 15) == 0) {
            const FixtureImage inner_in = make_image(result.tag * 977 + call.x(1) + 1);
            FixtureImage inner_out{};
            armigo_test_fixture_run_gate(&inner_in, &inner_out);
            FixtureImage expected = inner_in;
            expected.frame.x[2] = result.tag;
            for (unsigned i = 0; i < kGateGeneralRegisterCount; ++i) {
                if (i != kGateHostReservedRegister && i != kGateLinkRegister &&
                    inner_out.frame.x[i] != expected.frame.x[i]) {
                    ++result.mismatches;
                }
            }
            if (inner_out.sp_after != inner_out.sp_before ||
                inner_out.x18_after != inner_out.x18_before) {
                ++result.mismatches;
            }
        }
        call.set_x(0, call.x(0) ^ result.tag);
        GateVector vector = call.v(3);
        vector.lane[1] ^= result.tag;
        call.set_v(3, vector);
    } else {
        call.set_x(2, result.tag);
    }
    return GateStatus::resume;
}

void test_multithreaded_isolation() {
    constexpr unsigned kThreads = 8;
    constexpr std::uint64_t kIterations = 500;
    std::vector<ThreadResult> results(kThreads);
    std::vector<std::thread> threads;
    for (unsigned t = 0; t < kThreads; ++t) {
        results[t].tag = 0x1800 + t;
        threads.emplace_back([&result = results[t]] {
            set_thread_guest_x18(result.tag);
            GateScope scope(&thread_handler, &result, GateFailurePolicy::resume_unchanged);
            for (std::uint64_t iteration = 0; iteration < kIterations; ++iteration) {
                FixtureImage in = make_image(result.tag * 7919 + iteration);
                in.frame.x[1] = iteration;
                FixtureImage out{};
                armigo_test_fixture_run_gate(&in, &out);
                FixtureImage expected = in;
                expected.frame.x[0] ^= result.tag;
                expected.frame.v[3].lane[1] ^= result.tag;
                for (unsigned i = 0; i < kGateGeneralRegisterCount; ++i) {
                    if (i != kGateHostReservedRegister && i != kGateLinkRegister &&
                        out.frame.x[i] != expected.frame.x[i]) {
                        ++result.mismatches;
                    }
                }
                for (unsigned i = 0; i < kGateVectorRegisterCount; ++i) {
                    if (!(out.frame.v[i] == expected.frame.v[i])) {
                        ++result.mismatches;
                    }
                }
                if ((out.frame.nzcv & kNzcvMask) != in.frame.nzcv || out.frame.fpsr != in.frame.fpsr ||
                    out.frame.fpcr != in.frame.fpcr || out.sp_after != out.sp_before ||
                    out.x18_after != out.x18_before ||
                    out.frame.x[kGateLinkRegister] != address_of(&armigo_test_fixture_gate_return)) {
                    ++result.mismatches;
                }
                if (thread_guest_x18() != result.tag) {
                    result.x18_isolated = false;
                }
                ++result.iterations;
            }
            result.failures = scope.failure_count();
        });
    }
    for (std::thread& thread : threads) {
        thread.join();
    }
    for (const ThreadResult& result : results) {
        CHECK_EQ(result.iterations, kIterations);
        CHECK_EQ(result.mismatches, 0);
        CHECK_EQ(result.failures, 0);
        CHECK(result.x18_isolated);
    }
    CHECK_EQ(thread_guest_x18(), 0);
    CHECK(!GateScope::thread_has_scope());
}

void test_scope_queries() {
    CHECK(!GateScope::thread_has_scope());
    {
        HandlerContext context;
        GateScope scope(&observe_only, &context);
        CHECK(GateScope::thread_has_scope());
        CHECK(scope.policy() == GateFailurePolicy::trap);
        CHECK(scope.handler() == &observe_only);
        CHECK(scope.context() == &context);
        CHECK(std::strcmp(describe(GateFailureKind::no_scope),
                          "gate entered without an active GateScope on this thread") == 0);
    }
    CHECK(!GateScope::thread_has_scope());
}

// ---------------------------------------------------------------------------
// Process-fatal scenarios, executed in a child process
// ---------------------------------------------------------------------------

constexpr int kChildHookExitBase = 100;

void child_trap_hook(const GateFailure& failure) noexcept {
    _exit(kChildHookExitBase + static_cast<int>(failure.kind));
}

[[noreturn]] void run_child(const std::string& mode) {
    if (mode != "trap-no-hook") {
        set_gate_trap_hook(&child_trap_hook);
    }
    const FixtureImage in = make_image(0x7000);
    FixtureImage out{};
    if (mode == "no-scope") {
        armigo_test_fixture_run_gate(&in, &out);
    } else if (mode == "trap-policy" || mode == "trap-no-hook") {
        HandlerContext context;
        context.variant = 0;
        GateScope scope(&failing_handler, &context, GateFailurePolicy::trap);
        armigo_test_fixture_run_gate(&in, &out);
    } else if (mode == "trap-threw") {
        HandlerContext context;
        context.variant = 1;
        GateScope scope(&failing_handler, &context, GateFailurePolicy::trap);
        armigo_test_fixture_run_gate(&in, &out);
    } else if (mode == "invalid-frame") {
        GateFrame bogus{};
        armigo_execution_gate_dispatch(&bogus);
    } else if (mode == "scope-order") {
        auto* outer = new GateScope(&observe_only, nullptr);
        auto* inner = new GateScope(&observe_only, nullptr);
        delete outer;  // Out of order: must trap before the chain is corrupted.
        delete inner;
    } else if (mode == "scope-thread") {
        GateScope* scope = nullptr;
        std::thread([&] { scope = new GateScope(&observe_only, nullptr); }).join();
        delete scope;
    } else if (mode == "scope-active") {
        GateScope* scope = nullptr;
        scope = new GateScope(
            [](GateCall&, void* context) -> GateStatus {
                delete *static_cast<GateScope**>(context);
                return GateStatus::failed;
            }, &scope, GateFailurePolicy::resume_unchanged);
        armigo_test_fixture_run_gate(&in, &out);
    } else if (mode == "invalid-policy") {
        GateScope scope(&observe_only, nullptr, static_cast<GateFailurePolicy>(255));
    }
    _exit(1);  // Every mode above must have trapped.
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
    struct Expectation {
        const char* mode;
        GateFailureKind kind;
    };
    const Expectation hooked[] = {
        {"no-scope", GateFailureKind::no_scope},
        {"trap-policy", GateFailureKind::handler_reported},
        {"trap-threw", GateFailureKind::handler_threw},
        {"invalid-frame", GateFailureKind::invalid_frame},
        {"scope-order", GateFailureKind::scope_misuse},
        {"scope-thread", GateFailureKind::scope_misuse},
        {"scope-active", GateFailureKind::scope_misuse},
        {"invalid-policy", GateFailureKind::invalid_policy},
    };
    for (const Expectation& expectation : hooked) {
        const ChildOutcome outcome = spawn_child(expectation.mode);
        CHECK(outcome.spawned);
        CHECK(outcome.exited);
        if (!outcome.exited || outcome.exit_code != kChildHookExitBase + static_cast<int>(expectation.kind)) {
            std::printf("FAIL child mode %s: exited=%d code=%d signal=%d\n", expectation.mode,
                        outcome.exited, outcome.exit_code, outcome.signal);
            ++g_checks_failed;
        }
        ++g_checks_total;
    }
    const ChildOutcome aborted = spawn_child("trap-no-hook");
    CHECK(aborted.spawned);
    CHECK(!aborted.exited);
    CHECK_EQ(aborted.signal, SIGABRT);
}

}  // namespace

int main(int argc, char** argv) {
    if (argc == 3 && std::strcmp(argv[1], "--child") == 0) {
        run_child(argv[2]);
    }

    test_identity_roundtrip();
    test_handler_changes_apply();
    test_virtual_x18();
    test_redirected_return();
    test_failures_resume_unchanged();
    test_nested_gate_entries();
    test_multithreaded_isolation();
    test_scope_queries();
    test_fatal_scenarios_in_children();

    std::printf("%d checks, %d failed\n", g_checks_total, g_checks_failed);
    return g_checks_failed == 0 ? 0 : 1;
}
