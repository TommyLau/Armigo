// ============================================================================
// Armigo Tests - NVN Entry Call Counting
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include "entry_table.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <thread>
#include <vector>

extern "C" void armigo_nvn_guest_call_loop(armigo::nvn::GenericFunction, std::uint64_t);

namespace {

using namespace armigo::nvn;

unsigned failures = 0;
#define CHECK(condition)                                                        \
    do {                                                                        \
        if (!(condition)) {                                                     \
            std::fprintf(stderr, "FAIL %d: %s\n", __LINE__, #condition);       \
            ++failures;                                                         \
        }                                                                       \
    } while (false)

std::uint64_t first_calls = 0;
std::uint64_t second_calls = 0;
std::uint64_t finish_calls = 0;

void first_flush(NVNqueue*) { ++first_calls; }
void second_flush(NVNqueue*) { ++second_calls; }
void finish(NVNqueue*) { ++finish_calls; }
void concurrent_flush(NVNqueue*) {}

const CallCount* find_count(std::span<const CallCount> counts, std::string_view name) {
    const auto found = std::find_if(counts.begin(), counts.end(),
                                    [name](const auto& count) { return count.name == name; });
    return found == counts.end() ? nullptr : &*found;
}

} // namespace

int main() {
    setenv("ARMIGO_NVN_COUNT", "1", 1);
    select_api_version({55, 15});
    install<Entry::nvnQueueFlush>(&first_flush);
    install<Entry::nvnQueueFinish>(&finish);

    const auto generic_flush = armigo_nvn_bootstrap_loader("nvnQueueFlush");
    const auto generic_finish = armigo_nvn_bootstrap_loader("nvnQueueFinish");
    const auto flush = reinterpret_cast<void (*)(NVNqueue*)>(generic_flush);
    const auto queue_finish = reinterpret_cast<void (*)(NVNqueue*)>(generic_finish);
    CHECK(generic_flush != reinterpret_cast<GenericFunction>(&first_flush));

    for (int index = 0; index < 3; ++index) flush(nullptr);
    install<Entry::nvnQueueFlush>(&second_flush);
    for (int index = 0; index < 2; ++index) flush(nullptr);
    CHECK(first_calls == 3);
    CHECK(second_calls == 2);

    std::array<CallCount, entry_count> counts{};
    CHECK(call_counts({}) == entry_count);
    CHECK(call_counts(counts) == entry_count);
    const auto* flush_count = find_count(counts, "nvnQueueFlush");
    CHECK(flush_count && flush_count->calls == 5);

    CallCountFrameState frame{};
    std::array<CallCount, 2> top{};
    CHECK(sample_call_count_frame(frame, top, 2, nullptr, nullptr) == 1);
    CHECK(top[0].name == "nvnQueueFlush" && top[0].calls == 5);

    for (int index = 0; index < 7; ++index) flush(nullptr);
    for (int index = 0; index < 3; ++index) queue_finish(nullptr);
    CHECK(sample_call_count_frame(frame, top, 1, nullptr, nullptr) == 1);
    CHECK(top[0].name == "nvnQueueFlush" && top[0].calls == 7);
    CHECK(frame.total_calls == 10);
    CHECK(sample_call_count_frame(frame, top, 2, nullptr, nullptr) == 0);

    flush(nullptr);
    auto* output = std::tmpfile();
    CHECK(output != nullptr);
    if (output) {
        CHECK(sample_call_count_frame(frame, top, 2, &print_call_count_frame, output) == 1);
        CHECK(std::ftell(output) > 0);
        std::fclose(output);
    }

    setenv("ARMIGO_NVN_COUNT", "0", 1);
    CHECK(armigo_nvn_bootstrap_loader("nvnQueueFlush") ==
          reinterpret_cast<GenericFunction>(&second_flush));
    setenv("ARMIGO_NVN_COUNT", "1", 1);

    install<Entry::nvnQueueFlush>(&concurrent_flush);
    static_cast<void>(call_counts(counts));
    const auto before_concurrent = find_count(counts, "nvnQueueFlush")->calls;
    std::vector<std::thread> workers;
    for (unsigned i = 0; i < 4; ++i) workers.emplace_back([&] {
        armigo_nvn_guest_call_loop(generic_flush, 10000);
    });
    for (auto& worker : workers) worker.join();
    static_cast<void>(call_counts(counts));
    CHECK(find_count(counts, "nvnQueueFlush")->calls == before_concurrent + 40000);

    constexpr std::uint64_t iterations = 1'000'000;
    double best_nanoseconds = 1.0e30;
    for (int sample = 0; sample < 5; ++sample) {
        const auto begin = std::chrono::steady_clock::now();
        armigo_nvn_guest_call_loop(generic_flush, iterations);
        const auto end = std::chrono::steady_clock::now();
        const auto elapsed = std::chrono::duration<double, std::nano>(end - begin).count();
        best_nanoseconds = std::min(best_nanoseconds, elapsed / static_cast<double>(iterations));
    }
    std::printf("NVN counted guest BLR cost: %.2f ns/call\n", best_nanoseconds);
    CHECK(best_nanoseconds < 50.0);

    std::printf("NVN counting failures: %u\n", failures);
    return failures ? 1 : 0;
}
