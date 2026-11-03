// ============================================================================
// Armigo Tests - Native Runtime Verification
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#pragma once

#include <armigo/aarch64/instruction.h>
#include <armigo/runtime/preparation_format.h>
#include <armigo/runtime/process_runtime.h>

#include "../../src/platform/process_memory.h"
#include "../loader/npdm_fixture.h"
#include "preparation_fixture.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <optional>
#include <span>
#include <stop_token>
#include <string>
#include <thread>
#include <utility>
#include <variant>
#include <vector>

#include <unistd.h>
#include <spawn.h>
#include <sys/wait.h>
extern char** environ;

#if defined(__APPLE__)
#include <mach/mach.h>
#include <mach/mach_vm.h>
#endif

extern "C" {
extern const std::byte armigo_process_fixture_begin[];
extern const std::byte armigo_process_fixture_end[];
extern const std::byte armigo_process_fixture_main[];
extern const std::byte armigo_process_fixture_main_end[];
extern const std::byte armigo_process_fixture_worker[];
extern const std::byte armigo_process_fixture_worker_end[];
extern const std::byte armigo_process_fixture_block[];
extern const std::byte armigo_process_fixture_block_end[];
extern const std::byte armigo_process_fixture_grant_stop[];
extern const std::byte armigo_process_fixture_grant_stop_end[];
extern const std::byte armigo_process_fixture_grant_heap[];
extern const std::byte armigo_process_fixture_grant_worker_stack[];
extern const std::byte armigo_process_fixture_worker_stack[];
extern const std::byte armigo_process_fixture_original[];
extern const std::byte armigo_process_fixture_original_end[];
extern const std::byte armigo_process_fixture_original_slot[];
extern const std::byte armigo_process_fixture_original_target[];
extern const std::byte armigo_process_fixture_original_relocate_slot[];
extern const std::byte armigo_process_fixture_original_worker[];
extern const std::byte armigo_process_fixture_original_worker_end[];
extern const std::byte armigo_process_fixture_original_worker_stack[];
}

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

const LaunchFailure* failure(const RunResult& result) {
    return std::get_if<LaunchFailure>(&result.outcome);
}

bool ok(const LaunchStatus& status) {
    return std::holds_alternative<std::monostate>(status);
}

struct TemporaryDirectory {
    std::filesystem::path path;

    TemporaryDirectory() {
        auto pattern = (std::filesystem::temp_directory_path() /
                        "armigo-process-runtime-XXXXXX").string();
        std::vector<char> writable(pattern.begin(), pattern.end());
        writable.push_back('\0');
        if (char* created = ::mkdtemp(writable.data())) path = created;
    }
    ~TemporaryDirectory() {
        std::error_code ignored;
        std::filesystem::remove_all(path, ignored);
    }
};

bool write_file(const std::filesystem::path& path, std::span<const std::byte> bytes) {
    std::ofstream stream(path, std::ios::binary);
    stream.write(reinterpret_cast<const char*>(bytes.data()),
                 static_cast<std::streamsize>(bytes.size()));
    return stream.good();
}

class TestPresenter final : public platform::Presenter {
public:
    explicit TestPresenter(platform::PresentationMode mode) : mode_(mode) {}

    platform::PresentationMode mode() const noexcept override { return mode_; }
    platform::PresentResult present(const platform::PresentationFrame&,
                                    std::optional<ClockInstant>,
                                    std::stop_token) override {
        ++present_calls;
        return platform::PresentationFailure{
            platform::PresentationError::host_failure,
            "The process integration fixture does not submit graphics work."};
    }
    platform::PresenterStatus drain(std::optional<ClockInstant>) override {
        ++drain_calls;
        if (fail_drains != 0) {
            --fail_drains;
            return platform::PresentationFailure{
                platform::PresentationError::deadline_expired,
                "Synthetic presenter drain deadline expired."};
        }
        return std::monostate{};
    }

    std::atomic<unsigned> present_calls{0};
    std::atomic<unsigned> drain_calls{0};
    std::atomic<unsigned> fail_drains{0};

private:
    platform::PresentationMode mode_;
};

std::size_t fixture_offset(const std::byte* symbol) {
    return 0x100 + static_cast<std::size_t>(symbol - armigo_process_fixture_begin);
}

enum class FixtureEntry { normal, blocking, grant_stop, original, original_rtld };

std::uint32_t move_wide(bool keep, unsigned destination, unsigned half,
                        std::uint16_t immediate) {
    return (keep ? 0xf2800000u : 0xd2800000u) |
           (half << 21) | (std::uint32_t{immediate} << 5) | destination;
}

loader::ExeFsDirectoryInput process_input(FixtureEntry entry_kind,
                                          std::uint64_t grant_heap = 0) {
    constexpr std::uint64_t worker_stack_base = 0x10000;
    constexpr std::uint64_t worker_stack_top = 0x20000;
    const bool original = entry_kind == FixtureEntry::original ||
                          entry_kind == FixtureEntry::original_rtld;
    runtime::test::ModuleSpec module;
    module.name = entry_kind == FixtureEntry::original_rtld ? "rtld" : "main";
    const auto fixture_size = static_cast<std::size_t>(
        armigo_process_fixture_end - armigo_process_fixture_begin);
    module.text.resize(std::max<std::size_t>(runtime::test::rodata_offset,
                                             fixture_offset(armigo_process_fixture_end)));
    std::memcpy(module.text.data() + 0x100, armigo_process_fixture_begin,
                fixture_size);
    module.data.resize(worker_stack_top - runtime::test::data_offset);

    const auto main_entry = fixture_offset(
        original ? armigo_process_fixture_original :
        entry_kind == FixtureEntry::blocking ? armigo_process_fixture_block :
        entry_kind == FixtureEntry::grant_stop ? armigo_process_fixture_grant_stop :
                                                 armigo_process_fixture_main);
    const auto main_end = fixture_offset(
        original ? armigo_process_fixture_original_end :
        entry_kind == FixtureEntry::blocking ? armigo_process_fixture_block_end :
        entry_kind == FixtureEntry::grant_stop ? armigo_process_fixture_grant_stop_end :
                                                 armigo_process_fixture_main_end);
    const auto worker_entry = fixture_offset(armigo_process_fixture_worker);
    const auto worker_end = fixture_offset(armigo_process_fixture_worker_end);
    module.symbols = {
        {original ? "authored_bootstrap" : "_start",
         main_entry, main_end - main_entry, true},
        {"synthetic_worker", worker_entry, worker_end - worker_entry, true},
    };
    if (original) {
        module.symbols.push_back({"authored_original_worker",
            fixture_offset(armigo_process_fixture_original_worker),
            static_cast<std::uint64_t>(armigo_process_fixture_original_worker_end -
                                       armigo_process_fixture_original_worker), true});
        runtime::test::put_word(module, 0, aarch64::encode_b(0, main_entry).word);
        const auto slot = fixture_offset(armigo_process_fixture_original_slot);
        const auto target = fixture_offset(armigo_process_fixture_original_target);
        const auto relocate_slot = fixture_offset(armigo_process_fixture_original_relocate_slot);
        const auto original_stack = fixture_offset(armigo_process_fixture_original_worker_stack);
        runtime::test::put_word(module, original_stack,
            aarch64::encode_adr(3, original_stack, worker_stack_base).word);
        runtime::test::put_word(module, slot,
            aarch64::encode_adr(9, slot, runtime::test::data_offset).word);
        runtime::test::put_word(module, relocate_slot,
            aarch64::encode_adr(9, relocate_slot, runtime::test::data_offset).word);
        runtime::test::put_word(module, target,
            aarch64::encode_adr(10, target, runtime::test::data_offset + 16).word);
        module.relocations = {{runtime::test::data_offset, 0, 1027,
                               static_cast<std::int64_t>(runtime::test::data_offset + 16)}};
    }
    const auto stack_instruction = fixture_offset(armigo_process_fixture_worker_stack);
    runtime::test::put_word(module, stack_instruction,
        aarch64::encode_adr(3, stack_instruction, worker_stack_base).word);
    if (entry_kind == FixtureEntry::grant_stop) {
        const auto heap_instruction = fixture_offset(armigo_process_fixture_grant_heap);
        for (unsigned half = 0; half != 4; ++half) {
            runtime::test::put_word(module, heap_instruction + half * 4,
                move_wide(half != 0, 19, half,
                          static_cast<std::uint16_t>(grant_heap >> (half * 16))));
        }
        const auto grant_stack = fixture_offset(
            armigo_process_fixture_grant_worker_stack);
        runtime::test::put_word(module, grant_stack,
            aarch64::encode_adr(3, grant_stack, worker_stack_base).word);
    }

    loader::ExeFsDirectoryInput input;
    const auto module_name = module.name;
    input.modules.push_back({module_name, runtime::test::build_module(std::move(module))});
    if (entry_kind == FixtureEntry::original_rtld) {
        runtime::test::ModuleSpec main;
        main.name = "main";
        main.text.resize(0x4000);
        main.data.resize(0x4000);
        main.symbols = {{"must_not_be_the_process_entry", 0x100, 8, true}};
        runtime::test::put_word(main, 0, aarch64::encode_b(0, 0x100).word);
        runtime::test::put_word(main, 0x100, aarch64::encode_svc(0x26).word);
        runtime::test::put_word(main, 0x104, aarch64::encode_b(0x104, 0x104).word);
        input.modules.push_back({"main", runtime::test::build_module(std::move(main))});
    }
    if (original) {
        npdm_fixture::Parts parts;
        parts.system_resource = 0;
        parts.main_core = 0;
        parts.stack_size = 0x40000;
        parts.aci0_fs_permissions = parts.acid_fs_permissions = 0x3fff'ffff'ffff'ffffull;
        parts.aci0_capabilities = {
            npdm_fixture::cap_core_priority(59, 28, 0, 2),
            npdm_fixture::cap_syscall_mask(0, (1u << 6) | (1u << 7) | (1u << 8) |
                                             (1u << 9) | (1u << 10) | (1u << 12)),
            npdm_fixture::cap_syscall_mask(1, 1u << (0x26 - 24)),
            npdm_fixture::cap_syscall_mask(2, (1u << (0x34 - 48)) | (1u << (0x35 - 48))),
            npdm_fixture::cap_handle_table(64),
            npdm_fixture::cap_kernel_version(20, 5),
            npdm_fixture::cap_program_type(1),
        };
        parts.acid_capabilities = parts.aci0_capabilities;
        input.process_metadata = npdm_fixture::build_npdm(parts).bytes;
    }
    std::sort(input.modules.begin(), input.modules.end(),
        [](const auto& left, const auto& right) { return left.name < right.name; });
    return input;
}

LaunchProfile synthetic_profile(std::uint32_t handles = 64) {
    LaunchProfile profile;
    profile.origin = ProfileOrigin::synthetic_test;
    profile.is_64_bit = true;
    profile.process.allowed_core_mask = 1;
    profile.process.heap_size = 8ull << 20;
    profile.process.base_alias_size = 2ull << 20;
    profile.process.stack_size = 2ull << 20;
    profile.permissions.supervisor_calls = {
        0x07, 0x08, 0x09, 0x0a, 0x18, 0x1f, 0x21, 0x34, 0x35};
    profile.permissions.service_access_control =
        npdm_fixture::service_access_control({{"fsp-srv", false}});
    profile.permissions.file_system_access_control =
        npdm_fixture::fs_access_block(1, 0x3fff'ffff'ffff'ffffull);
    profile.permissions.handle_table_capacity = handles;
    profile.permissions.highest_priority = 16;
    profile.permissions.lowest_priority = 63;
    profile.main_thread.priority = 44;
    profile.main_thread.ideal_core = 0;
    profile.main_thread.stack_size = 64 * 1024;
    return profile;
}

struct Fixture {
    TemporaryDirectory temporary;
    std::filesystem::path exefs = temporary.path / "exefs";
    std::filesystem::path romfs = temporary.path / "romfs";
    std::filesystem::path prepared = temporary.path / "prepared";
    loader::ExeFsDirectoryInput input;
    LaunchDescription description;

    explicit Fixture(FixtureEntry entry = FixtureEntry::normal,
                     std::uint64_t grant_heap = 0)
        : input(process_input(entry, grant_heap)) {
        std::filesystem::create_directories(exefs);
        std::filesystem::create_directories(romfs);
        CHECK(!temporary.path.empty());
        for (const auto& module : input.modules)
            CHECK(write_file(exefs / module.name, module.bytes));
        if (input.process_metadata)
            CHECK(write_file(exefs / "main.npdm", *input.process_metadata));
        const std::array marker{
            std::byte{0x73}, std::byte{0x79}, std::byte{0x6e}, std::byte{0x74},
            std::byte{0x68}, std::byte{0x65}, std::byte{0x74}, std::byte{0x69},
            std::byte{0x63}};
        CHECK(write_file(romfs / "marker.bin", marker));
        const PreparationOptions options{
            .required_preparation_identity = installed_preparation_identity(),
            .max_artifact_bytes = default_prepared_artifact_limit,
            .mode = input.process_metadata ? PreparationMode::automatic : PreparationMode::synthetic_test,
        };
        const auto artifacts = prepare_modules(input, options);
        CHECK(value(artifacts) != nullptr);
        if (const auto* error = failure(artifacts)) {
            std::fprintf(stderr, "Fixture preparation failed: %s: %s\n",
                         error->operation.c_str(), error->detail.c_str());
        } else {
            CHECK(ok(write_prepared_modules(prepared, *value(artifacts))));
        }

        description.exefs_directory = exefs;
        description.romfs_directory = romfs;
        description.prepared_directory = prepared;
        description.presentation.mode = platform::PresentationMode::headless;
        if (entry == FixtureEntry::original || entry == FixtureEntry::original_rtld) {
            description.npdm_trust = NpdmTrustPolicy::structurally_validated_extracted_input;
        } else {
            description.synthetic_profile = synthetic_profile();
        }
    }

    explicit Fixture(bool blocking)
        : Fixture(blocking ? FixtureEntry::blocking : FixtureEntry::normal) {}
};

RuntimeDependencies dependencies(std::shared_ptr<TestPresenter> presenter = {}) {
    RuntimeDependencies result;
    result.memory_options.resource_limit_bytes = 32ull << 20;
    result.preparation_options = {
        .required_preparation_identity = installed_preparation_identity(),
        .max_artifact_bytes = default_prepared_artifact_limit,
    };
    auto clock = create_guest_clock();
    CHECK(value(clock) != nullptr);
    if (!value(clock)) return result;
    result.clock = *value(clock);
    const auto sample = result.clock->sample();
    CHECK(value(sample) != nullptr);
    if (!value(sample)) return result;
    const PresentationClockConfiguration configuration{
        .epoch = value(sample)->instant,
        .refresh_period = 16'666'667ns,
    };
    auto presentation_clock = create_presentation_clock(result.clock, configuration);
    CHECK(value(presentation_clock) != nullptr);
    if (value(presentation_clock)) result.presentation_clock = *value(presentation_clock);
    result.presenter = presenter ? std::move(presenter)
                                 : std::make_shared<TestPresenter>(
                                       platform::PresentationMode::headless);
    return result;
}

std::optional<std::uint64_t> probe_heap_address() {
    // Probe a larger hole and use an aligned interior range. Module loading
    // occurs after this reservation is released; keeping 32 MiB of slack on
    // either side prevents its small anonymous mappings from repeatedly taking
    // the exact fixed heap candidate under sanitizer allocators.
    auto reserved = platform::ProcessMemoryRegion::reserve(64ull << 20);
    auto* region = std::get_if<platform::ProcessMemoryRegion>(&reserved);
    if (!region) return std::nullopt;
    return region->address() + (32ull << 20);
}

bool native_writable(std::uint64_t address) {
#if defined(__APPLE__)
    mach_vm_address_t region_address = address;
    mach_vm_size_t region_size = 0;
    vm_region_basic_info_data_64_t info{};
    mach_msg_type_number_t count = VM_REGION_BASIC_INFO_COUNT_64;
    mach_port_t object = MACH_PORT_NULL;
    const auto result = ::mach_vm_region(
        mach_task_self(), &region_address, &region_size, VM_REGION_BASIC_INFO_64,
        reinterpret_cast<vm_region_info_t>(&info), &count, &object);
    if (object != MACH_PORT_NULL) ::mach_port_deallocate(mach_task_self(), object);
    return result == KERN_SUCCESS && region_address <= address &&
           region_size > address - region_address &&
           (info.protection & (VM_PROT_READ | VM_PROT_WRITE)) ==
               (VM_PROT_READ | VM_PROT_WRITE);
#else
    (void)address;
    return false;
#endif
}

bool wait_for_native_word(std::uint64_t address, std::uint32_t expected,
                          std::chrono::milliseconds timeout) {
    const auto deadline = std::chrono::steady_clock::now() + timeout;
    do {
        if (native_writable(address) &&
            __atomic_load_n(reinterpret_cast<const std::uint32_t*>(address),
                            __ATOMIC_ACQUIRE) == expected) return true;
        std::this_thread::sleep_for(1ms);
    } while (std::chrono::steady_clock::now() < deadline);
    return false;
}

void store_native_word(std::uint64_t address, std::uint32_t value) {
    __atomic_store_n(reinterpret_cast<std::uint32_t*>(address), value,
                     __ATOMIC_RELEASE);
}

} // namespace
