// ============================================================================
// Armigo Tests - Native Runtime Verification
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include <armigo/platform/host_module.h>

#include <mach/mach.h>
#include <mach/mach_vm.h>
#include <unistd.h>

#include <array>
#include <cstdio>
#include <cstring>
#include <limits>
#include <utility>

extern "C" unsigned char armigo_test_mapping_code_begin[];
extern "C" unsigned char armigo_test_mapping_code_end[];

namespace {
using namespace armigo::platform;
using armigo::memory::Permission;
using Image = std::vector<std::byte>;

bool check(bool condition, const char* message) {
    if (!condition) std::fprintf(stderr, "%s\n", message);
    return condition;
}

std::optional<vm_prot_t> protection(std::uintptr_t pointer) {
    mach_vm_address_t address = pointer;
    mach_vm_size_t size = 0;
    vm_region_basic_info_data_64_t info{};
    mach_msg_type_number_t count = VM_REGION_BASIC_INFO_COUNT_64;
    mach_port_t object = MACH_PORT_NULL;
    const auto status = ::mach_vm_region(mach_task_self(), &address, &size,
        VM_REGION_BASIC_INFO_64, reinterpret_cast<vm_region_info_t>(&info), &count, &object);
    if (object != MACH_PORT_NULL) ::mach_port_deallocate(mach_task_self(), object);
    if (status != KERN_SUCCESS || pointer < address || pointer - address >= size) return {};
    return info.protection;
}

bool mapping_and_execution(std::size_t page) {
    Image image(page * 3, std::byte{0x3c});
    const auto begin = reinterpret_cast<std::uintptr_t>(armigo_test_mapping_code_begin);
    const auto end = reinterpret_cast<std::uintptr_t>(armigo_test_mapping_code_end);
    if (!check(end > begin && end - begin < page, "Invalid authored assembly fixture.")) return false;
    std::memcpy(image.data(), armigo_test_mapping_code_begin, end - begin);
    const std::array regions = {
        ModuleRegion{0, page, Permission::read | Permission::execute},
        ModuleRegion{page, page, Permission::read},
        ModuleRegion{2 * page, page, Permission::read | Permission::write},
    };
    auto result = HostModule::create(image, regions);
    if (!check(result.ok(), "Module mapping failed.")) return false;
    auto& module = *result.module;
    const auto base = module.base_address();
    if (!check(base % page == 0 && module.host_page_size() == page &&
               module.preparation_bytes().size() == image.size(),
               "Preparation mapping geometry is wrong.") ||
        !check(protection(base) == (VM_PROT_READ | VM_PROT_WRITE),
               "Preparation mapping unexpectedly executable.") ||
        !check(!module.executable_address(0), "An unsealed module exposed an entry.")) return false;
    module.preparation_bytes()[2 * page] = std::byte{0x7b};
    if (!check(module.seal().ok() && module.sealed() && module.preparation_bytes().empty(),
               "Sealing failed or kept a writable preparation view.") ||
        !check(module.seal().ok(), "Repeated sealing failed.") ||
        !check(protection(base) == (VM_PROT_READ | VM_PROT_EXECUTE),
               "Code pages are not read/execute.") ||
        !check(protection(base + page) == VM_PROT_READ, "Read-only page protection is wrong.") ||
        !check(protection(base + 2 * page) == (VM_PROT_READ | VM_PROT_WRITE),
               "Data page protection is wrong.") ||
        !check(!module.executable_address(page) && !module.executable_address(2) &&
               !module.executable_address(image.size()), "Invalid entry address was accepted.")) return false;
    const auto data = module.readable_bytes(2 * page, 1);
    if (!check(data && (*data)[0] == std::byte{0x7b}, "Prepared data changed during sealing.")) return false;
    const auto entry = module.executable_address(0);
    if (!check(entry.has_value(), "Sealed code entry is unavailable.")) return false;
    const auto function = reinterpret_cast<std::uint64_t (*)()>(*entry);
    if (!check(function() == 42, "Mapped ARM64 instructions did not execute correctly.")) return false;
    HostModule moved = std::move(module);
    if (!check(!module.valid() && module.preparation_bytes().empty() &&
               !module.executable_address(0) && moved.base_address() == base,
               "Mapping ownership did not move correctly.")) return false;
    const auto moved_entry = moved.executable_address(0);
    return check(moved_entry && reinterpret_cast<std::uint64_t (*)()>(*moved_entry)() == 42,
                 "Moving ownership changed mapped code behavior.");
}

bool shared_pages_and_gaps(std::size_t page) {
    Image image(page * 3);
    const std::array shared = {
        ModuleRegion{0, page / 2, Permission::read},
        ModuleRegion{page / 2, page / 2, Permission::read | Permission::write},
        ModuleRegion{2 * page, page, Permission::read},
    };
    auto result = HostModule::create(image, shared);
    if (!check(result.ok() && result.module->seal().ok(), "Shared host-page layout failed.")) return false;
    const auto base = result.module->base_address();
    if (!check(protection(base) == (VM_PROT_READ | VM_PROT_WRITE),
               "Shared host-page permissions were not combined.") ||
        !check(protection(base + page) == VM_PROT_NONE, "A gap page remained accessible.") ||
        !check(!result.module->readable_bytes(page - 1, 2),
               "A readable view crossed an unmapped guest gap.")) return false;
    const std::array conflict = {
        ModuleRegion{0, page / 2, Permission::read | Permission::execute},
        ModuleRegion{page / 2, page / 2, Permission::read | Permission::write},
    };
    return check(HostModule::create(image, conflict).status.error ==
                     HostMapError::write_execute_conflict,
                 "A shared writable/executable host page was accepted.");
}

bool minimum_address_mapping(std::size_t page) {
    Image image(page, std::byte{0x5a});
    const std::array regions = {
        ModuleRegion{0, page, Permission::read | Permission::write},
    };
    auto first = HostModule::create(image, regions);
    if (!check(first.ok(), "The initial minimum-address fixture mapping failed.")) return false;
    const auto minimum = first.module->base_address() + first.module->mapping_size();
    HostMapLimits limits;
    limits.max_image_bytes = page;
    limits.minimum_address = minimum;
    auto second = HostModule::create(image, regions, limits);
    if (!check(second.ok() && second.module->base_address() >= minimum,
               "A mapping did not honor its minimum address.")) return false;
    const auto first_byte = first.module->preparation_bytes().front();
    if (!check(first_byte == std::byte{0x5a},
               "Minimum-address placement clobbered an existing mapping.")) return false;

    limits.minimum_address = std::numeric_limits<std::uintptr_t>::max();
    return check(!HostModule::create(image, regions, limits).ok(),
                 "An impossible minimum mapping address was accepted.");
}

bool invalid_inputs(std::size_t page) {
    Image image(page);
    for (const auto& regions : {
        std::vector<ModuleRegion>{{0, 0, Permission::read}},
        std::vector<ModuleRegion>{{page, 1, Permission::read}},
        std::vector<ModuleRegion>{{1, std::numeric_limits<std::size_t>::max(), Permission::read}},
        std::vector<ModuleRegion>{{0, page, Permission::read}, {page / 2, 1, Permission::read}},
        std::vector<ModuleRegion>{{0, page, static_cast<Permission>(128)}},
        std::vector<ModuleRegion>{{0, page, Permission::write | Permission::execute}},
    }) {
        if (!check(!HostModule::create(image, regions).ok(), "Invalid mapping input was accepted.")) return false;
    }
    const std::array region = {ModuleRegion{0, page, Permission::read}};
    return check(!HostModule::create({}, region).ok() &&
                 !HostModule::create(image, {}).ok() &&
                 HostModule::create(image, region, {page - 1, 1}).status.error == HostMapError::resource_limit &&
                 HostModule::create(image, region, {page, 0}).status.error == HostMapError::resource_limit,
                 "An empty or oversized mapping was accepted.");
}
bool embedded_linked_text(std::size_t page) {
    const auto entry = reinterpret_cast<std::uintptr_t>(armigo_test_mapping_code_begin);
    const auto source = entry & ~(page - 1);
    const std::array regions = {ModuleRegion{0, page, Permission::read | Permission::execute}};
    const std::array pieces = {EmbeddedModulePiece{
        0, {reinterpret_cast<const std::byte*>(source), page}, true}};
    auto mapped = HostModule::create_embedded(page, regions, pieces);
    if (!check(mapped.ok() && mapped.module->seal().ok(),
               "Linked executable text could not be remapped.")) return false;
    const auto remapped_entry = mapped.module->executable_address(entry - source);
    return check(remapped_entry && reinterpret_cast<std::uint64_t(*)()>(*remapped_entry)() == 42,
                 "Remapped linked text did not execute.");
}

bool embedded_mapping(std::size_t page) {
    Image image(page * 3);
    const auto length = reinterpret_cast<std::uintptr_t>(armigo_test_mapping_code_end) -
                        reinterpret_cast<std::uintptr_t>(armigo_test_mapping_code_begin);
    std::memcpy(image.data(), armigo_test_mapping_code_begin, length);
    image[page] = std::byte{42};
    const auto rx = Permission::read | Permission::execute;
    const auto rw = Permission::read | Permission::write;
    const std::array regions = {ModuleRegion{0, page, rx},
        ModuleRegion{page, page, rw}, ModuleRegion{page * 2, page, Permission::read}};
    auto source = HostModule::create(image, regions);
    if (!source.ok() || !source.module->seal().ok()) return false;
    const auto base = source.module->base_address();
    std::array pieces = {
        EmbeddedModulePiece{0, {reinterpret_cast<const std::byte*>(base), page}, true},
        EmbeddedModulePiece{page, {reinterpret_cast<const std::byte*>(base + page), page * 2}, false}};
    auto mapped = HostModule::create_embedded(image.size(), regions, pieces);
    if (!check(mapped.ok(), "Embedded image remap failed.")) return false;
    const auto destination = mapped.module->base_address();
    if (!check(protection(destination) == (VM_PROT_READ | VM_PROT_EXECUTE) &&
               protection(destination + page) == (VM_PROT_READ | VM_PROT_WRITE) &&
               !mapped.module->executable_address(0),
               "Embedded image exposed an entry early or changed code protection.")) return false;
    mapped.module->preparation_bytes()[page] = std::byte{43};
    if (!check(*reinterpret_cast<const std::byte*>(base + page) == std::byte{42},
               "Embedded writable pages did not use copy-on-write.")) return false;
    if (!check(::mach_vm_protect(mach_task_self(), destination, page, FALSE,
                                VM_PROT_READ | VM_PROT_WRITE) != KERN_SUCCESS,
               "Embedded executable pages can become writable.")) return false;
    if (!mapped.module->seal().ok()) return false;
    source.module.reset();
    if (!check(protection(destination + 2 * page) == VM_PROT_READ &&
               *reinterpret_cast<const std::byte*>(destination + page) == std::byte{43} &&
               reinterpret_cast<std::uint64_t(*)()>(destination)() == 42,
               "Embedded mapping did not survive source destruction and sealing.")) return false;
    HostModule moved = std::move(*mapped.module);
    if (!check(moved.seal().ok() && moved.executable_address(0) &&
               reinterpret_cast<std::uint64_t(*)()>(*moved.executable_address(0))() == 42,
               "Moving an embedded mapping changed execution.")) return false;
    pieces[0].bytes = {reinterpret_cast<const std::byte*>(destination), page};
    pieces[1].bytes = {reinterpret_cast<const std::byte*>(destination + page), page * 2};
    if (!check(!HostModule::create_embedded(image.size(), regions, std::span(pieces).first(1)).ok(),
               "Incomplete embedded pieces were accepted.")) return false;
    pieces[0].executable = false;
    if (!check(!HostModule::create_embedded(image.size(), regions, pieces).ok(),
               "Nonexecutable backing was accepted for executable pages.")) return false;
    pieces[0].executable = true;
    pieces[0].bytes = pieces[0].bytes.subspan(1);
    return check(!HostModule::create_embedded(image.size(), regions, pieces).ok(),
                 "Unaligned embedded backing was accepted.");
}
} // namespace

int main() {
    const auto page = ::sysconf(_SC_PAGESIZE);
    if (!check(page > 0, "Host page size is unavailable.") ||
        !mapping_and_execution(static_cast<std::size_t>(page)) ||
        !embedded_mapping(static_cast<std::size_t>(page)) ||
        !embedded_linked_text(static_cast<std::size_t>(page)) ||
        !shared_pages_and_gaps(static_cast<std::size_t>(page)) ||
        !minimum_address_mapping(static_cast<std::size_t>(page)) ||
        !invalid_inputs(static_cast<std::size_t>(page))) return 1;
    std::puts("Host module permissions, ownership, and native execution checks passed.");
    return 0;
}
