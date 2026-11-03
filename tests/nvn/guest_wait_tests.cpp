// ============================================================================
// Armigo NVN - Entry Point Table
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include "guest_wait.h"
#include <armigo/horizon/startup_queries.h>
#include <cstdio>
#include <future>
#include <thread>

namespace {
unsigned failures = 0;
#define CHECK(value) do { if (!(value)) { std::fprintf(stderr, "line %d: %s\n", __LINE__, #value); ++failures; } } while (false)
std::stop_token token(void* context) noexcept { return static_cast<std::stop_source*>(context)->get_token(); }
}
int main() {
    using namespace armigo;
    horizon::ProcessConfiguration configuration;
    configuration.heap_address = 0x1000000000;
    configuration.alias_address = 0x2000000000;
    configuration.stack_address = 0x6000000000;
    configuration.allowed_core_mask = 1;
    const auto process = horizon::ProcessRecord::create(configuration, {});
    CHECK(process.ok());
    if (!process.ok()) return 1;
    horizon::ThreadRecord caller(process.process, 1, 0);
    horizon::HandleTable table(4);
    const auto event = std::make_shared<horizon::Event>();
    const auto handle = table.insert(event);
    CHECK(handle.has_value());
    if (!handle) return 1;
    CHECK(nvn::wait_guest_event(*handle, 0).disposition == horizon::EventDisposition::unsupported);
    std::stop_source stop;
    {
        nvn::GuestWaitScope scope(table, caller, token, &stop);
        CHECK(nvn::wait_guest_event(*handle, 0).result == 0xea01);
        event->signal();
        CHECK(nvn::wait_guest_event(*handle, 0).result == 0);
        static_cast<void>(event->reset());
        std::stop_source stopped;
        stopped.request_stop();
        {
            nvn::GuestWaitScope nested(table, caller, token, &stopped);
            CHECK(nvn::wait_guest_event(*handle, -1).disposition == horizon::EventDisposition::cancelled);
        }
        CHECK(nvn::wait_guest_event(*handle, 0).result == 0xea01);
        CHECK(horizon::cancel_synchronization(table, horizon::current_thread_handle, caller).result == 0);
        CHECK(nvn::wait_guest_event(*handle, -1).result == 0xec01);
    }
    auto waiting = std::async(std::launch::async, [&] {
        nvn::GuestWaitScope scope(table, caller, token, &stop);
        return nvn::wait_guest_event(*handle, -1);
    });
    stop.request_stop();
    CHECK(waiting.wait_for(std::chrono::seconds(5)) == std::future_status::ready);
    CHECK(waiting.get().disposition == horizon::EventDisposition::cancelled);
    CHECK(nvn::wait_guest_event(*handle, 0).disposition == horizon::EventDisposition::unsupported);
    std::printf("NVN guest wait failures: %u\n", failures);
    return failures ? 1 : 0;
}
