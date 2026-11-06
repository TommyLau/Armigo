// ============================================================================
// Armigo Tests - Native Runtime Verification
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include "../../src/runtime/service_registry.h"
#include "../loader/npdm_fixture.h"
#include "../../tools/headless/pacing_clock.h"
#include <sstream>

#include <armigo/romfs/romfs_virtual.h>
#include <armigo/runtime/preparation_format.h>

#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <optional>
#include <span>
#include <stop_token>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

namespace {
using namespace armigo;
using namespace armigo::runtime;
using namespace std::chrono_literals;

unsigned failures = 0;
#define CHECK(condition) do { if (!(condition)) { \
    std::fprintf(stderr, "Check failed at line %u: %s\n", __LINE__, #condition); \
    ++failures; \
} } while (false)

template<class T>
const T* value(const LaunchResult<T>& result) {
    return std::get_if<T>(&result);
}

template<class T>
const LaunchFailure* failure(const LaunchResult<T>& result) {
    return std::get_if<LaunchFailure>(&result);
}

void put(horizon::IpcMessage& message, std::size_t offset,
         std::uint64_t value, unsigned width = 4) {
    for (unsigned index = 0; index < width; ++index) {
        message[offset + index] =
            static_cast<std::byte>((value >> (8 * index)) & 0xff);
    }
}

std::uint64_t get(const horizon::IpcMessage& message, std::size_t offset,
                  unsigned width = 4) {
    std::uint64_t value = 0;
    for (unsigned index = 0; index < width; ++index) {
        value |= std::uint64_t{std::to_integer<unsigned char>(
                     message[offset + index])} << (8 * index);
    }
    return value;
}

horizon::IpcMessage register_client() {
    horizon::IpcMessage message{};
    put(message, 0, 0x8000000000000010ull, 8);
    put(message, 8, 1);
    return message;
}

horizon::IpcMessage get_service(std::string_view name) {
    horizon::IpcMessage message{};
    put(message, 0, 0x200000011ull, 8);
    for (std::size_t index = 0; index < name.size(); ++index) {
        message[8 + index] = static_cast<std::byte>(name[index]);
    }
    return message;
}

class TestPresenter final : public platform::Presenter {
public:
    platform::PresentationMode mode() const noexcept override {
        return platform::PresentationMode::headless;
    }
    platform::PresentResult present(const platform::PresentationFrame&,
                                    std::optional<ClockInstant>,
                                    std::stop_token) override {
        return platform::PresentationFailure{
            platform::PresentationError::host_failure,
            "The service bootstrap fixture does not submit frames."};
    }
    platform::PresenterStatus drain(std::optional<ClockInstant>) override {
        ++drains;
        return std::monostate{};
    }
    unsigned drains = 0;
};

constexpr std::array<std::string_view, 18> service_inventory{
    "lm", "acc:u0", "aoc:u", "ectx:aw", "pctl", "prepo:u", "time:u", "fsp-srv",
    "ro", "nvdrv", "nfp:user", "audren:u", "apm", "appletOE", "hid",
    "vi:u", "vi:m", "vi:s",
};
constexpr std::array<std::string_view, 15> service_inventory_without_presentation{
    "lm", "acc:u0", "aoc:u", "ectx:aw", "pctl", "prepo:u", "time:u", "fsp-srv",
    "ro", "nvdrv", "nfp:user", "audren:u", "apm", "appletOE", "hid",
};
constexpr std::array<std::string_view, 3> presentation_services{"vi:u", "vi:m", "vi:s"};

LaunchProfile profile_for(std::span<const std::string_view> services) {
    LaunchProfile profile;
    profile.origin = ProfileOrigin::synthetic_test;
    profile.process.allowed_core_mask = 1;
    profile.process.heap_size = 8ull << 20;
    profile.process.base_alias_size = 2ull << 20;
    profile.process.stack_size = 2ull << 20;
    profile.permissions.handle_table_capacity = 256;
    profile.permissions.highest_priority = 16;
    profile.permissions.lowest_priority = 63;
    profile.permissions.file_system_access_control =
        npdm_fixture::fs_access_block(1, 0x3fff'ffff'ffff'ffffull);
    for (const auto name : services) {
        profile.permissions.service_access_control.push_back(
            static_cast<std::byte>(name.size() - 1));
        for (const char character : name) {
            profile.permissions.service_access_control.push_back(
                static_cast<std::byte>(character));
        }
    }
    profile.main_thread.priority = 44;
    profile.main_thread.ideal_core = 0;
    profile.main_thread.stack_size = 64 * 1024;
    return profile;
}

struct RegistryFixture {
    LaunchProfile profile = profile_for(service_inventory);
    std::shared_ptr<horizon::HandleTable> handles =
        std::make_shared<horizon::HandleTable>(profile.permissions.handle_table_capacity);
    std::shared_ptr<GuestMemory> memory;
    std::shared_ptr<GuestClock> clock;
    std::shared_ptr<PresentationClock> presentation_clock;
    std::shared_ptr<TestPresenter> presenter = std::make_shared<TestPresenter>();
    std::shared_ptr<romfs::MemoryStorage> romfs =
        std::make_shared<romfs::MemoryStorage>(
            std::vector<std::byte>(romfs::header_length, std::byte{}));

    RegistryFixture() {
        auto made_memory = create_guest_memory(profile, {32ull << 20}, nullptr, handles);
        CHECK(value(made_memory) != nullptr);
        if (value(made_memory)) memory = *value(made_memory);
        auto made_clock = create_guest_clock();
        CHECK(value(made_clock) != nullptr);
        if (value(made_clock)) clock = *value(made_clock);
        if (clock) {
            const auto sample = clock->sample();
            CHECK(value(sample) != nullptr);
            if (value(sample)) {
                auto made_grid = create_presentation_clock(clock, {
                    .epoch = value(sample)->instant,
                    .refresh_period = 16'666'667ns,
                });
                CHECK(value(made_grid) != nullptr);
                if (value(made_grid)) presentation_clock = *value(made_grid);
            }
        }
    }

    ServiceRegistryDependencies dependencies() const {
        ServiceRegistryDependencies result;
        result.profile = profile;
        result.process_id = 71;
        result.handles = handles;
        result.memory = memory;
        result.clock = clock;
        result.presentation_clock = presentation_clock;
        result.presenter = presenter;
        result.romfs = romfs;
        result.observe_frame = [](const platform::PresentedFrame&) -> LaunchStatus {
            return std::monostate{};
        };
        result.acquire_graphics_access = []() -> LaunchResult<std::shared_ptr<void>> {
            return std::static_pointer_cast<void>(std::make_shared<unsigned>(1));
        };
        return result;
    }
};

std::shared_ptr<horizon::ClientSession> open_service_manager(
    const std::shared_ptr<ServiceRegistry>& registry) {
    auto connected = registry->connect_named_port("sm:");
    CHECK(connected.disposition == horizon::IpcDisposition::handled);
    CHECK(connected.result == 0 && connected.session != nullptr);
    return connected.session;
}

bool register_process(const std::shared_ptr<horizon::ClientSession>& sm) {
    auto message = register_client();
    auto reply = sm->request(message, 71);
    CHECK(reply.disposition == horizon::IpcDisposition::handled);
    CHECK(reply.transport_result == 0);
    CHECK(get(reply.bytes, 8) == 0);
    return reply.disposition == horizon::IpcDisposition::handled &&
           reply.transport_result == 0;
}

void test_inventory_is_reachable_through_sm() {
    RegistryFixture fixture;
    auto made = create_service_registry(fixture.dependencies());
    CHECK(value(made) != nullptr);
    if (!value(made)) return;
    auto registry = *value(made);
    auto sm = open_service_manager(registry);
    if (!sm || !register_process(sm)) return;

    const auto open = [&](std::string_view name) {
        const auto before = fixture.handles->size();
        auto reply = sm->request(get_service(name), 71);
        CHECK(reply.disposition == horizon::IpcDisposition::handled);
        CHECK(reply.transport_result == 0);
        CHECK(get(reply.bytes, 16) == 0);
        const auto handle = static_cast<horizon::Handle>(get(reply.bytes, 12));
        CHECK(handle != 0);
        CHECK(fixture.handles->lookup_as<horizon::ClientSession>(handle) != nullptr);
        CHECK(fixture.handles->size() == before + 1);
        CHECK(fixture.handles->close(handle));
    };
    for (const auto name : service_inventory_without_presentation) open(name);
    // The registry reports VI support according to whether runtime presentation
    // was linked into its own target. That feature define is intentionally
    // private, so observe the public IPC result instead of mirroring build flags.
    auto vi_reply = sm->request(get_service(presentation_services.front()), 71);
    const bool presentation_linked =
        vi_reply.disposition == horizon::IpcDisposition::handled;
    if (presentation_linked) {
        CHECK(vi_reply.transport_result == 0);
        CHECK(get(vi_reply.bytes, 16) == 0);
        const auto first_vi = static_cast<horizon::Handle>(get(vi_reply.bytes, 12));
        CHECK(first_vi != 0);
        CHECK(fixture.handles->lookup_as<horizon::ClientSession>(first_vi) != nullptr);
        CHECK(fixture.handles->close(first_vi));
        open(presentation_services[1]);
        open(presentation_services[2]);
    } else {
        CHECK(vi_reply.disposition == horizon::IpcDisposition::unsupported);
        const auto unavailable = registry->failure();
        CHECK(unavailable.has_value());
        if (unavailable) {
            CHECK(unavailable->stage == LaunchStage::presentation);
            CHECK(unavailable->code == LaunchError::unsupported);
        }
    }
    const auto statistics = registry->statistics();
    CHECK(statistics.named_port_connections == 1);
    CHECK(statistics.service_connections ==
          (presentation_linked ? service_inventory.size()
                               : service_inventory_without_presentation.size()));
    CHECK(presentation_linked == !registry->failure().has_value());
    CHECK(std::holds_alternative<std::monostate>(registry->drain(std::nullopt)));
    CHECK(fixture.presenter->drains == 1);
}

void test_named_port_and_service_acl_denial() {
    RegistryFixture fixture;
    const std::array<std::string_view, 1> allowed{"fsp-srv"};
    fixture.profile = profile_for(allowed);
    auto made = create_service_registry(fixture.dependencies());
    CHECK(value(made) != nullptr);
    if (!value(made)) return;
    auto registry = *value(made);

    const auto absent = registry->connect_named_port("missing");
    CHECK(absent.disposition == horizon::IpcDisposition::handled);
    CHECK(absent.result != 0 && !absent.session);
    auto sm = open_service_manager(registry);
    if (!sm || !register_process(sm)) return;
    auto denied = sm->request(get_service("lm"), 71);
    CHECK(denied.disposition == horizon::IpcDisposition::handled);
    CHECK(get(denied.bytes, 16) != 0);
    CHECK(registry->statistics().service_connections == 0);
    CHECK(!registry->failure().has_value());

    registry->request_stop();
    auto stopped = sm->request(get_service("fsp-srv"), 71);
    CHECK(stopped.disposition == horizon::IpcDisposition::cancelled);
    CHECK(!registry->failure().has_value());
}

void test_file_system_acl_is_fail_closed() {
    RegistryFixture fixture;
    auto malformed = fixture.dependencies();
    malformed.profile.permissions.file_system_access_control.resize(4);
    auto made = create_service_registry(std::move(malformed));
    CHECK(failure(made) != nullptr);
    if (failure(made)) {
        CHECK(failure(made)->stage == LaunchStage::services);
        CHECK(failure(made)->code == LaunchError::unsupported);
    }

    // A partial permission mask is admitted; it is not enforced per command.
    auto partial = fixture.dependencies();
    partial.profile.permissions.file_system_access_control =
        npdm_fixture::fs_access_block(1, 1);
    made = create_service_registry(std::move(partial));
    CHECK(failure(made) == nullptr);

    RegistryFixture no_fs;
    const std::array<std::string_view, 1> allowed{"lm"};
    no_fs.profile = profile_for(allowed);
    no_fs.profile.permissions.file_system_access_control.clear();
    made = create_service_registry(no_fs.dependencies());
    CHECK(value(made) != nullptr);
}

void test_factory_rejects_inconsistent_dependencies() {
    RegistryFixture fixture;
    auto missing = fixture.dependencies();
    missing.process_id = 0;
    CHECK(failure(create_service_registry(std::move(missing))) != nullptr);

    missing = fixture.dependencies();
    missing.memory.reset();
    CHECK(failure(create_service_registry(std::move(missing))) != nullptr);

    missing = fixture.dependencies();
    missing.presenter.reset();
    CHECK(failure(create_service_registry(std::move(missing))) != nullptr);

    auto wrong_handles = fixture.dependencies();
    wrong_handles.handles = std::make_shared<horizon::HandleTable>(4);
    CHECK(failure(create_service_registry(std::move(wrong_handles))) != nullptr);

    auto malformed_acl = fixture.dependencies();
    malformed_acl.profile.permissions.service_access_control = {
        std::byte{7}, std::byte{'f'}, std::byte{'s'}};
    auto malformed_result = create_service_registry(std::move(malformed_acl));
    CHECK(failure(malformed_result) != nullptr);
    if (failure(malformed_result)) {
        CHECK(failure(malformed_result)->stage == LaunchStage::services);
        CHECK(failure(malformed_result)->code == LaunchError::invalid_input);
    }

    auto other_clock = create_guest_clock();
    CHECK(value(other_clock) != nullptr);
    if (value(other_clock)) {
        auto mismatched = fixture.dependencies();
        mismatched.clock = *value(other_clock);
        CHECK(failure(create_service_registry(std::move(mismatched))) != nullptr);
    }
}

void test_pacing_log_failure_is_visible_to_runtime_polling() {
    struct FailingBuffer : std::stringbuf {
        bool fail = false;
        int sync() override { return fail ? -1 : std::stringbuf::sync(); }
    } output;
    RegistryFixture fixture;
    auto observed = std::make_shared<tools::PacingClock>(fixture.presentation_clock, std::nullopt,
        std::make_unique<std::ostream>(&output));
    fixture.presentation_clock = observed;
    auto made = create_service_registry(fixture.dependencies());
    CHECK(value(made) != nullptr);
    if (!value(made)) return;
    auto registry = *value(made);
    CHECK(!registry->failure());
    output.fail = true;
    auto status = observed->observe_present(1, 1, {});
    CHECK(failure(status) != nullptr);
    const auto polled = registry->failure();
    CHECK(polled && polled->stage == LaunchStage::presentation &&
          polled->code == LaunchError::host_failure && polled->operation == "headless.pacing");
    // Recovery of the stream must not erase the first causal runtime failure.
    output.fail = false;
    CHECK(failure(observed->observe_present(2, 1, {})) != nullptr);
    CHECK(registry->failure()->detail == polled->detail);
    registry->request_stop();
    CHECK(std::holds_alternative<std::monostate>(registry->drain(std::nullopt)));
}

} // namespace

int main() {
    test_pacing_log_failure_is_visible_to_runtime_polling();
    test_inventory_is_reachable_through_sm();
    test_named_port_and_service_acl_denial();
    test_file_system_acl_is_fail_closed();
    test_factory_rejects_inconsistent_dependencies();
    std::printf("Service-bootstrap failures: %u\n", failures);
    return failures ? 1 : 0;
}
