// ============================================================================
// Armigo Tests - Native Runtime Verification
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include <armigo/execution/gate_code.h>
#include <armigo/execution/gate_patch.h>
#include <armigo/platform/host_module.h>

#include <cstdio>
#include <cstring>
#include <limits>

extern "C" {
extern const std::byte armigo_test_patch_begin[], armigo_test_patch_end[];
extern const std::byte armigo_test_patch_write_tls[], armigo_test_patch_read_tls[];
extern const std::byte armigo_test_patch_read_frame_register[];
extern const std::byte armigo_test_patch_read_tlr[], armigo_test_patch_read_frequency[];
extern const std::byte armigo_test_patch_read_ticks[], armigo_test_patch_svc[];
}

namespace {
using namespace armigo;
using namespace armigo::execution;

bool check(bool value, const char* message) {
    if (!value) std::fprintf(stderr, "%s\n", message);
    return value;
}

std::size_t offset(const std::byte* label) {
    return reinterpret_cast<std::uintptr_t>(label) -
           reinterpret_cast<std::uintptr_t>(armigo_test_patch_begin);
}

std::uint64_t ticks(void*) noexcept { return 123456; }

struct Observation {
    unsigned count = 0;
    bool valid = true;
};

GateStatus supervisor(std::uint16_t immediate, GateCall& call,
                      GuestThreadContext& context, void* opaque) {
    auto& observed = *static_cast<Observation*>(opaque);
    ++observed.count;
    observed.valid &= immediate == 0x1e && call.x(0) == 0x1234 &&
                      context.thread_pointer() == 0x1234 &&
                      call.x(1) == 0x5678 && call.x(2) == 19200000 &&
                      call.x(3) == ticks(nullptr) && call.x(5) == 0x1234 &&
                      call.sp() + GateSitePatch::stack_adjustment == call.x(4);
    call.set_x(0, ticks(nullptr));
    return GateStatus::resume;
}

bool rejects_invalid_plans() {
    const auto word = aarch64::encode_svc(0x1e).word;
    constexpr auto maximum = std::numeric_limits<std::uint64_t>::max();
    const auto valid = make_gate_site_patch(word, 0x100, 0x200, {0x1000, 512});
    if (!check(valid.ok(), "A valid gate-site layout was rejected.")) return false;
    const auto& patch = *valid.patch;
    if (!check(aarch64::referenced_address(aarch64::decode(patch.replacement_instruction),
                                           0x100) == 0x200 &&
               aarch64::referenced_address(aarch64::decode(patch.trampoline[1]),
                                           0x204) == 0x1000 &&
               aarch64::referenced_address(aarch64::decode(patch.trampoline[3]),
                                           0x20c) == 0x104 &&
               patch.dispatch_site.return_address == 0x208,
               "Gate-site branches do not reach their declared destinations.")) return false;
    const std::array invalid = {
        make_gate_site_patch(word, 0x101, 0x200, {0x1000, 512}),
        make_gate_site_patch(word, 0x100, 0x202, {0x1000, 512}),
        make_gate_site_patch(word, 0x100, 0x200, {0x1004, 512}),
        make_gate_site_patch(word, maximum - 3, 0x200, {0x1000, 512}),
        make_gate_site_patch(word, 0x100, maximum - 3, {0x1000, 512}),
        make_gate_site_patch(word, 0x100, 0x200, {maximum - 7, 16}),
        make_gate_site_patch(word, 0x100, 0x100, {0x1000, 512}),
        make_gate_site_patch(word, 0x100, 0x104, {0x1000, 512}),
        make_gate_site_patch(word, 0x104, 0x1000, {0x108, 512}),
        make_gate_site_patch(word, 0x100, 0x200, {0x100, 512}),
        make_gate_site_patch(word, 0x100, 0x200, {0x200, 512}),
        make_gate_site_patch(word, 0x100, 0x10000000, {0x1000, 512}),
        make_gate_site_patch(0xd503201f, 0x100, 0x200, {0x1000, 512}),
        make_gate_site_patch(word, 0x100, 0x200, {0x1000, 512},
                             {.supervisor_calls = false}),
        make_gate_site_patch(aarch64::encode_mrs(30,
            *aarch64::encoding_of(aarch64::SystemRegister::tpidr_el0)).word,
            0x100, 0x200, {0x1000, 512}),
    };
    for (const auto& result : invalid) {
        if (!check(!result.ok() && result.error != GatePatchError::none,
                   "An invalid gate-site plan was accepted.")) return false;
    }
    return check(make_gate_site_patch(word, maximum - 0xfff, maximum - 15,
                                     {maximum - 0x1fff, 16}).ok(),
                 "A fully representable boundary layout was rejected.");
}

bool executes_prepared_sites() {
    const auto source_size = offset(armigo_test_patch_end);
    const auto gate = make_local_gate_code();
    const auto gate_offset = (source_size + 7) & ~std::size_t{7};
    const auto first_trampoline = gate_offset + gate.size();
    const std::array labels = {
        armigo_test_patch_write_tls, armigo_test_patch_read_tls,
        armigo_test_patch_read_frame_register,
        armigo_test_patch_read_tlr, armigo_test_patch_read_frequency,
        armigo_test_patch_read_ticks, armigo_test_patch_svc};
    std::vector<std::byte> image(first_trampoline + labels.size() * 16);
    std::memcpy(image.data(), armigo_test_patch_begin, source_size);
    std::memcpy(image.data() + gate_offset, gate.data(), gate.size());
    std::vector<GuestGateSite> sites;
    for (std::size_t index = 0; index < labels.size(); ++index) {
        const auto at = offset(labels[index]);
        std::uint32_t original;
        std::memcpy(&original, image.data() + at, 4);
        const auto plan = make_gate_site_patch(original, at, first_trampoline + index * 16,
                                               {gate_offset, gate.size()});
        if (!check(plan.ok(), "An authored system instruction could not be prepared.")) return false;
        const auto& patch = *plan.patch;
        if (patch.expected_instruction != original) return false;
        std::memcpy(image.data() + at, &patch.replacement_instruction, 4);
        std::memcpy(image.data() + patch.trampoline_address, patch.trampoline.data(), 16);
        sites.push_back(patch.dispatch_site);
    }
    const std::array regions = {platform::ModuleRegion{
        0, image.size(), memory::Permission::read | memory::Permission::execute}};
    auto mapped = platform::HostModule::create(image, regions);
    if (!check(mapped.ok(), "Prepared synthetic code could not be mapped.")) return false;
    for (auto& site : sites) site.return_address += mapped.module->base_address();
    const auto table = GuestSiteTable::build(sites);
    if (!check(table.ok(), "Prepared return addresses could not be registered.") ||
        !check(mapped.module->seal().ok(), "Prepared synthetic code could not be sealed.")) return false;
    const auto entry = mapped.module->executable_address(0);
    if (!entry) return false;
    GuestThreadContext thread;
    thread.set_read_only_thread_pointer(0x5678);
    if (thread.set_counter({ticks, nullptr, 19200000}) != GuestCounterError::none) return false;
    Observation observed;
    GuestDispatchScope scope(table.table, thread, {supervisor, &observed});
    const auto result = reinterpret_cast<std::uint64_t (*)(std::uint64_t)>(*entry)(0x1234);
    return check(result == ticks(nullptr) && observed.valid && observed.count == 1 &&
                 scope.dispatched() == labels.size() && scope.failure_count() == 0,
                 "Prepared native system sites did not preserve their execution contract.");
}
} // namespace

int main() {
    if (!rejects_invalid_plans() || !executes_prepared_sites()) return 1;
    std::puts("System-site patching, local gates, guest register dispatch, and native return passed.");
    return 0;
}
