// ============================================================================
// Armigo Tests - Native Runtime Verification
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include <armigo/runtime/guest_memory.h>
#include <armigo/runtime/module_set.h>

#include "nso_fixture.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <span>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

namespace {

using namespace armigo;
using namespace armigo::runtime;
using Image = std::vector<std::byte>;

constexpr std::uint64_t mib = 1024 * 1024;

unsigned failures = 0;
#define CHECK(condition) do { if (!(condition)) { \
    std::fprintf(stderr, "Failed at line %d: %s\n", __LINE__, #condition); \
    ++failures; \
} } while (false)

void put(Image& image, std::size_t offset, std::uint64_t value,
         unsigned width = 8) {
    for (unsigned index = 0; index < width; ++index) {
        image.at(offset + index) =
            static_cast<std::byte>((value >> (index * 8)) & 0xff);
    }
}

Image module_fixture(std::size_t text_size, std::size_t ro_size,
                     std::size_t data_size, std::uint32_t ro_offset,
                     std::uint32_t data_offset) {
    using namespace armigo::loader::test;
    FixtureSpec spec;
    spec.segments[0] = {Image(text_size, std::byte{0x11}), 0, false, true};
    spec.segments[1] = {Image(ro_size), ro_offset, false, true};
    spec.segments[2] = {Image(data_size, std::byte{0x33}), data_offset, false, true};
    auto& text = spec.segments[0].content;
    auto& ro = spec.segments[1].content;
    auto& data = spec.segments[2].content;

    // Independently authored minimal MOD0 and dynamic tables. The one packed
    // relative relocation points into data; no fixture code is executed.
    put(text, 4, ro_offset, 4);
    put(ro, 0, 0x30444f4d, 4);
    put(ro, 4, 0x40, 4);
    const auto image_end = data_offset + static_cast<std::uint32_t>(data_size);
    put(ro, 8, image_end - ro_offset, 4);
    put(ro, 12, image_end - ro_offset, 4);
    put(ro, 16, 0xf00, 4);
    put(ro, 20, 0xf00, 4);
    put(ro, 24, data_offset + 0x100 - ro_offset, 4);

    std::size_t dynamic = 0x40;
    const auto tag = [&](std::uint64_t key, std::uint64_t value) {
        put(ro, dynamic, key);
        put(ro, dynamic + 8, value);
        dynamic += 16;
    };
    tag(4, ro_offset + 0x180);
    tag(5, ro_offset + 0x200);
    tag(10, 7);
    tag(6, ro_offset + 0x300);
    tag(11, 24);
    tag(36, ro_offset + 0x500);
    tag(35, 8);
    tag(37, 8);
    tag(0, 0);

    put(ro, 0x180, 1, 4);
    put(ro, 0x184, 2, 4);
    put(ro, 0x188, 1, 4);
    constexpr char strings[] = "\0entry";
    static_assert(sizeof(strings) == 7);
    std::memcpy(ro.data() + 0x200, strings, sizeof(strings));
    put(ro, 0x300 + 24, 1, 4);
    put(ro, 0x300 + 24 + 4, 0x12, 1);
    put(ro, 0x300 + 24 + 6, 1, 2);
    put(ro, 0x300 + 24 + 8, 0x100);
    put(ro, 0x300 + 24 + 16, 4);
    put(ro, 0x500, data_offset);
    put(data, 8, data_offset + 0x10);
    put(data, 16, 0x1122334455667788ULL);
    return build_nso(spec);
}

template <class Mapping>
constexpr bool has_declared_regions = requires(const Mapping& mapping) {
    mapping.guest_regions();
};

LaunchProfile profile() {
    LaunchProfile result;
    result.origin = ProfileOrigin::synthetic_test;
    result.process.allowed_core_mask = 1;
    result.process.heap_address = 0;
    result.process.alias_address = 0;
    result.process.stack_address = 0;
    result.process.heap_size = 8 * mib;
    result.process.base_alias_size = 2 * mib;
    result.process.stack_size = 2 * mib;
    result.permissions.handle_table_capacity = 64;
    return result;
}

std::shared_ptr<ModuleSet> load(std::string_view name, const Image& image,
                                const ModuleLoadOptions& options = {}) {
    const std::array sources{NsoSource{name, image}};
    auto loaded = ModuleSet::load_nso(sources, options);
    CHECK(loaded.ok());
    if (!loaded.ok()) {
        std::fprintf(stderr, "module load failed: %s\n", loaded.message.c_str());
        return {};
    }
    return std::make_shared<ModuleSet>(std::move(*loaded.modules));
}

LaunchResult<std::shared_ptr<GuestMemory>> create(
    const std::shared_ptr<ModuleSet>& modules) {
    return create_guest_memory(profile(), {16 * mib}, modules,
                               std::make_shared<horizon::HandleTable>(64));
}

bool prepare_empty_append(const ModulePreparationView& view, void*) noexcept {
    for (const auto byte : view.appended_code) {
        if (byte != std::byte{0}) return false;
    }
    return true;
}

bool test_declared_region_requirement_or_integrated_modules() {
    const auto native = platform::HostModule::native_page_size();
    CHECK(native.has_value());
    if (!native) return false;
    const auto page = static_cast<std::uint32_t>(*native);
    const Image image = module_fixture(page, page, page, page, 2 * page);
    const std::array preparation{
        ModulePreparation{*native, prepare_empty_append, nullptr}};
    ModuleLoadOptions options;
    options.preparation = preparation;
    auto modules = load("exact-pages", image, options);
    if (!modules) return false;
    CHECK(modules->mapping(0).mapping_size() == 4 * *native);

    auto created = create(modules);
    if constexpr (!has_declared_regions<platform::HostModule>) {
        const auto* failure = std::get_if<LaunchFailure>(&created);
        CHECK(failure != nullptr && failure->code == LaunchError::unsupported &&
              failure->operation == "create_guest_memory modules" &&
              failure->detail.find("declared-region accessor") != std::string::npos);
        std::puts("Module-backed GuestMemory integration coverage skipped: "
                  "HostModule has no declared-region accessor.");
        return true;
    } else {
        CHECK(std::holds_alternative<std::shared_ptr<GuestMemory>>(created));
        if (!std::holds_alternative<std::shared_ptr<GuestMemory>>(created)) return false;
        auto memory = std::get<std::shared_ptr<GuestMemory>>(std::move(created));
        const auto base = modules->mapping(0).base_address();
        const auto process = memory->process();
        const auto text = process->query_memory(base);
        const auto ro = process->query_memory(base + *native);
        const auto data = process->query_memory(base + 2 * *native);
        const auto append = process->query_memory(base + 3 * *native);
        CHECK(text.type == horizon::MemoryType::code && text.permissions == 5 &&
              text.range.size == *native);
        CHECK(ro.type == horizon::MemoryType::code && ro.permissions == 1 &&
              ro.range.size == *native);
        CHECK(data.type == horizon::MemoryType::code_data && data.permissions == 3 &&
              data.range.size == *native);
        CHECK(append.type == horizon::MemoryType::code && append.permissions == 5 &&
              append.range.size == *native);
        CHECK(process->used_memory() == 3 * *native &&
              process->total_memory() == 16 * mib);

        horizon::QueryRegisters protection{};
        protection[0] = base + *native;
        protection[1] = *native;
        protection[2] = 3;
        const auto ro_result = memory->dispatch(2, protection);
        CHECK(std::holds_alternative<horizon::QueryDisposition>(ro_result) &&
              protection[0] == 0xd401);
        CHECK(process->query_memory(base + *native).permissions == 1);
        protection[0] = base + *native;
        protection[1] = *native;
        protection[2] = 1;
        const auto unchanged_ro = memory->dispatch(2, protection);
        CHECK(std::holds_alternative<horizon::QueryDisposition>(unchanged_ro) && protection[0] == 0xd401);
        protection[0] = base + 2 * *native;
        protection[1] = *native;
        protection[2] = 1;
        const auto data_result = memory->dispatch(2, protection);
        CHECK(std::holds_alternative<horizon::QueryDisposition>(data_result) && protection[0] == 0);
        CHECK(process->query_memory(base + 2 * *native).type == horizon::MemoryType::code_data &&
              process->query_memory(base + 2 * *native).permissions == 1);
        protection[0] = base + 2 * *native;
        protection[1] = *native;
        protection[2] = 3;
        const auto restored = memory->dispatch(2, protection);
        CHECK(std::holds_alternative<horizon::QueryDisposition>(restored) && protection[0] == 0);

        std::weak_ptr<ModuleSet> weak = modules;
        auto pinned = memory->pin({base + 2 * *native, *native},
                                  memory::Permission::read |
                                      memory::Permission::write);
        CHECK(std::holds_alternative<GuestMemoryPin>(pinned));
        auto pin = std::holds_alternative<GuestMemoryPin>(pinned)
                       ? std::get<GuestMemoryPin>(std::move(pinned))
                       : GuestMemoryPin{};
        modules.reset();
        CHECK(!weak.expired());
        memory.reset();
        CHECK(!weak.expired() && pin && pin->bytes()[0x100] == std::byte{0x33});
        pin.reset();
        CHECK(weak.expired());
        return true;
    }
}

bool test_mixed_native_subpage_permissions_take_the_union() {
    const auto native = platform::HostModule::native_page_size();
    CHECK(native.has_value());
    if (!native) return false;
    if (*native == horizon::guest_page_size) {
        std::puts("Mixed native-subpage module test skipped: native pages are 4096 bytes.");
        return true;
    }
    CHECK(*native > 2 * horizon::guest_page_size);
    const auto page = static_cast<std::uint32_t>(*native);
    const Image mixed = module_fixture(
        *native, horizon::guest_page_size, horizon::guest_page_size,
        page, page + static_cast<std::uint32_t>(horizon::guest_page_size));
    auto modules = load("mixed-subpages", mixed);
    if (!modules) return false;
    const auto& mapping = modules->mapping(0);
    CHECK(mapping.host_regions().size() == 2);
    CHECK(mapping.host_regions().back().permissions ==
          (memory::Permission::read | memory::Permission::write));

    // The host maps the union of a mixed native page's permissions; the guest
    // map keeps each segment's own.
    auto created = create(modules);
    const auto* failure = std::get_if<LaunchFailure>(&created);
    if constexpr (has_declared_regions<platform::HostModule>) {
        CHECK(failure == nullptr);
    } else {
        CHECK(failure != nullptr &&
              failure->detail.find("declared-region accessor") != std::string::npos);
        std::puts("Mixed native-subpage admission detail skipped: "
                  "HostModule has no declared-region accessor.");
    }
    return true;
}

// A module whose code the host never runs may share a native page between its
// code and its writable data; guest memory admits it with the declared
// permissions while no host page is executable.
bool test_host_unexecuted_module_is_admitted() {
    const auto native = platform::HostModule::native_page_size();
    if (!native || *native < 4 * horizon::guest_page_size) {
        std::puts("Host-unexecuted module test skipped: native pages hold fewer than four guest pages.");
        return true;
    }
    const auto guest = static_cast<std::uint32_t>(horizon::guest_page_size);
    const Image packed = module_fixture(guest, guest, guest, guest, 2 * guest);
    const std::array preparation{ModulePreparation{.host_executes = false}};
    ModuleLoadOptions options;
    options.preparation = preparation;
    auto modules = load("packed", packed, options);
    if (!modules) return false;
    const auto& mapping = modules->mapping(0);
    CHECK(!mapping.host_executes() && !mapping.executable_address(0));
    CHECK(mapping.guest_regions().front().permissions ==
          (memory::Permission::read | memory::Permission::execute));
    auto created = create(modules);
    CHECK(std::get_if<LaunchFailure>(&created) == nullptr);
    return true;
}

} // namespace

int main() {
    CHECK(test_declared_region_requirement_or_integrated_modules());
    CHECK(test_mixed_native_subpage_permissions_take_the_union());
    CHECK(test_host_unexecuted_module_is_admitted());
    return failures == 0 ? 0 : 1;
}
