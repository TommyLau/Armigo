// ============================================================================
// Armigo Tests - Native Runtime Verification
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include <armigo/runtime/module_set.h>
#include <armigo/aarch64/instruction.h>
#include "nso_fixture.h"

#include <mach/mach.h>
#include <mach/mach_vm.h>

#include <array>
#include <atomic>
#include <cstdio>
#include <cstring>
#include <string>

extern "C" unsigned char armigo_test_import_begin[], armigo_test_import_end[];
extern "C" unsigned char armigo_test_provider_begin[], armigo_test_provider_end[];
extern "C" unsigned char armigo_test_prepared_begin[], armigo_test_prepared_end[];

namespace {
using namespace armigo::runtime;
using Image = std::vector<std::byte>;

void put(Image& image, std::size_t offset, std::uint64_t value, unsigned width = 8) {
    for (unsigned index = 0; index < width; ++index) {
        image.at(offset + index) = std::byte((value >> (index * 8)) & 255u);
    }
}

Image fixture(bool consumer, bool compressed, bool hidden = false, bool outside_function = false,
              std::uint64_t rela_target = 0x8000, std::uint64_t relr_target = 0x8008,
              std::uint64_t ro_base = 0x4000, std::uint64_t data_base = 0x8000) {
    using namespace armigo::loader::test;
    FixtureSpec spec;
    spec.segments[0] = {Image(0x1000), 0, compressed, true};
    spec.segments[1] = {Image(0x1000), static_cast<std::uint32_t>(ro_base), compressed, true};
    spec.segments[2] = {Image(0x1000), static_cast<std::uint32_t>(data_base), compressed, true};
    auto& text = spec.segments[0].content;
    auto& ro = spec.segments[1].content;
    auto& data = spec.segments[2].content;
    const auto* begin = consumer ? armigo_test_import_begin : armigo_test_provider_begin;
    const auto* end = consumer ? armigo_test_import_end : armigo_test_provider_end;
    const auto code_size = reinterpret_cast<std::uintptr_t>(end) -
                           reinterpret_cast<std::uintptr_t>(begin);
    std::memcpy(text.data() + 0x100, begin, code_size);
    put(text, 4, ro_base, 4);
    put(ro, 0, 0x30444f4d, 4);
    put(ro, 4, 0x40, 4);
    put(ro, 8, data_base + 0x1000 - ro_base, 4);
    put(ro, 12, data_base + 0x1000 - ro_base, 4);
    put(ro, 16, 0xf00, 4);
    put(ro, 20, 0xf00, 4);
    put(ro, 24, data_base + 0x100 - ro_base, 4);

    std::size_t dynamic = 0x40;
    auto tag = [&](std::uint64_t key, std::uint64_t value) {
        put(ro, dynamic, key); put(ro, dynamic + 8, value); dynamic += 16;
    };
    tag(4, ro_base + 0x180); tag(5, ro_base + 0x200); tag(10, 15); tag(6, ro_base + 0x300); tag(11, 24);
    if (consumer) { tag(7, ro_base + 0x400); tag(8, 24); tag(9, 24); }
    tag(36, ro_base + 0x500); tag(35, 8); tag(37, 8); tag(0, 0);
    put(ro, 0x180, 1, 4);
    put(ro, 0x184, consumer ? 3 : 2, 4);
    put(ro, 0x188, 1, 4);
    constexpr char strings[] = "\0entry\0service";
    static_assert(sizeof(strings) == 15);
    std::memcpy(ro.data() + 0x200, strings, sizeof(strings));
    auto symbol = [&](std::size_t index, std::uint32_t name,
                      std::uint16_t section, std::uint64_t value, std::uint64_t size) {
        const auto at = 0x300 + index * 24;
        put(ro, at, name, 4); put(ro, at + 4, 0x12, 1);
        put(ro, at + 6, section, 2); put(ro, at + 8, value); put(ro, at + 16, size);
    };
    symbol(1, consumer ? 1 : 7, 1, outside_function ? 0xc000 : 0x100, code_size);
    if (hidden) put(ro, 0x300 + 24 + 5, 2, 1);
    if (consumer) {
        symbol(2, 7, 0, 0, 0);
        put(ro, 0x400, rela_target);
        put(ro, 0x408, (std::uint64_t{2} << 32) | 257);
    }
    put(ro, 0x500, relr_target);
    put(data, 8, data_base + 0x10);
    put(data, 16, 0x1122334455667788ull);
    return build_nso(spec);
}

bool check(bool condition, const char* message) {
    if (!condition) std::fprintf(stderr, "%s\n", message);
    return condition;
}

std::uint64_t word(std::span<const std::byte> bytes) {
    std::uint64_t result = 0;
    for (unsigned index = 0; index < 8; ++index) {
        result |= std::to_integer<std::uint64_t>(bytes[index]) << (index * 8);
    }
    return result;
}

std::uint64_t alternate_service() { return 95; }

// Read actual host permissions without touching the probed mapping.
std::optional<vm_prot_t> protection(std::uintptr_t address) {
    mach_vm_address_t region = address;
    mach_vm_size_t size = 0;
    vm_region_basic_info_data_64_t information{};
    mach_msg_type_number_t count = VM_REGION_BASIC_INFO_COUNT_64;
    mach_port_t object = MACH_PORT_NULL;
    const auto result = mach_vm_region(mach_task_self(), &region, &size, VM_REGION_BASIC_INFO_64,
        reinterpret_cast<vm_region_info_t>(&information), &count, &object);
    if (object != MACH_PORT_NULL) mach_port_deallocate(mach_task_self(), object);
    if (result != KERN_SUCCESS || region > address || address - region >= size) return {};
    return information.protection;
}

struct PreparationState {
    std::array<std::uintptr_t, 2> bases{};
    std::uintptr_t import = 0;
    std::size_t calls = 0;
    bool reject_provider = false;
    bool correct = true;
};

struct GuestPreparationState {
    std::size_t calls = 0;
    bool correct = true;
};

bool prepare_guest_relocations(const ModulePreparationView& view, void* raw) noexcept {
    auto& state = *static_cast<GuestPreparationState*>(raw);
    ++state.calls;
    if (view.module_index == 0) {
        state.correct &= word(view.image.subspan(0x8000, 8)) == 0 &&
            word(view.image.subspan(0x8008, 8)) == 0x8010;
    }
    return state.correct;
}

struct ConcurrentState {
    std::atomic<std::size_t> calls{0};
    std::atomic<unsigned> rejected{0};  // bit per module index
    std::array<std::atomic<std::uintptr_t>, 2> bases{};
};

// Touches nothing outside its own module, as a callback that runs beside others must.
bool prepare_concurrently(const ModulePreparationView& view, void* raw) noexcept {
    auto& state = *static_cast<ConcurrentState*>(raw);
    ++state.calls;
    state.bases[view.module_index] = view.base_address;
    return !(state.rejected & (1u << view.module_index));
}

bool prepare(const ModulePreparationView& view, void* raw) noexcept {
    auto& state = *static_cast<PreparationState*>(raw);
    if (view.module_index >= state.bases.size()) return false;
    state.bases[view.module_index] = view.base_address;
    ++state.calls;
    state.correct &= view.image.size() == 0x9000 && !view.metadata.symbols.empty() &&
        protection(view.base_address + 0x100) == (VM_PROT_READ | VM_PROT_WRITE);
    if (!state.correct) return false;
    if (view.module_index == 0) {
        state.import = word(view.image.subspan(0x8000, 8));
        state.correct &= state.import != 0 && view.appended_code.empty() &&
            view.appended_code_offset == 0 && view.module_name == "consumer" &&
            word(view.image.subspan(0x8008, 8)) == view.base_address + 0x8010;
        return state.correct;
    }
    const auto page = armigo::platform::HostModule::native_page_size();
    const auto size = reinterpret_cast<std::uintptr_t>(armigo_test_prepared_end) -
                      reinterpret_cast<std::uintptr_t>(armigo_test_prepared_begin);
    state.correct &= page && view.appended_code_offset % *page == 0 &&
        view.appended_code_offset >= view.image.size() && view.appended_code.size() == size &&
        view.module_name == "provider" && state.import == view.base_address + 0x100 &&
        protection(view.base_address + view.appended_code_offset) == (VM_PROT_READ | VM_PROT_WRITE);
    if (!state.correct) return false;
    for (auto byte : view.appended_code) state.correct &= byte == std::byte{0};
    if (!state.correct ||
        std::memcmp(view.image.data() + 0x100, armigo_test_provider_begin, 4) != 0) return false;
    const auto redirect = armigo::aarch64::encode_b(
        view.base_address + 0x100, view.base_address + view.appended_code_offset);
    if (!redirect.ok()) return false;
    std::memcpy(view.image.data() + 0x100, &redirect.word, 4);
    std::memcpy(view.appended_code.data(), armigo_test_prepared_begin, size);
    return !state.reject_provider;
}
bool prepare_with_data(const ModulePreparationView& view, void* raw) noexcept {
    if (view.module_index == 1) {
        const auto page = armigo::platform::HostModule::native_page_size();
        if (!page || view.appended_data.size() != *page || view.appended_data_offset % *page ||
            view.appended_data_offset < view.image.size() ||
            view.appended_data_offset + view.appended_data.size() != view.appended_code_offset)
            return false;
        for (const auto byte : view.appended_data) if (byte != std::byte{0}) return false;
        view.appended_data.front() = std::byte{42};
    }
    return prepare(view, raw);
}
bool embedded_code_page_relocation() {
    auto consumer = fixture(true, false, false, false, 0x1800, 0x8008, 0x1000, 0x8000);
    auto provider = fixture(false, false);
    const std::array sources = {NsoSource{"consumer", consumer}, NsoSource{"provider", provider}};
    auto original = ModuleSet::load_nso(sources);
    if (!check(original.ok(), "Synthetic shared-page relocation setup failed.")) return false;
    std::array<std::vector<armigo::platform::EmbeddedModulePiece>, 2> pieces;
    std::array<ModulePreparation, 2> preparation;
    for (std::size_t i = 0; i < 2; ++i) {
        const auto& mapped = original.modules->mapping(i);
        for (const auto& region : mapped.host_regions())
            pieces[i].push_back({region.offset,
                {reinterpret_cast<const std::byte*>(mapped.base_address() + region.offset), region.size},
                armigo::memory::has_permission(region.permissions, armigo::memory::Permission::execute)});
        preparation[i].callback = [](const ModulePreparationView&, void*) noexcept { return true; };
        preparation[i].embedded_pieces = pieces[i];
    }
    ModuleLoadOptions options;
    options.preparation = preparation;
    const auto loaded = ModuleSet::load_nso(sources, options);
    return check(!loaded.ok() && loaded.error == ModuleLoadError::linking_failed,
                 "An embedded relocation sharing an executable host page was not rejected before writing.");
}
} // namespace

// A module whose text, read-only data and writable data share one 16 KiB
// native page loads when the host never runs its code, and is refused
// otherwise (the union would be writable and executable).
bool packed_module_maps_without_execute() {
    const auto packed = fixture(false, false, false, false, 0x2000, 0x2008, 0x1000, 0x2000);
    const std::array sources = {NsoSource{"packed", packed}};
    const auto page = armigo::platform::HostModule::native_page_size();
    if (!page || *page < 0x4000) return true;  // a 16 KiB page holds all three segments
    auto executed = ModuleSet::load_nso(sources);
    if (!check(!executed.ok() && executed.error == ModuleLoadError::mapping_failed,
               "A writable executable native page was accepted.")) return false;
    const std::array preparation = {ModulePreparation{.host_executes = false}};
    ModuleLoadOptions options;
    options.preparation = preparation;
    auto loaded = ModuleSet::load_nso(sources, options);
    if (!check(loaded.ok(), loaded.message.c_str())) return false;
    const auto& mapping = loaded.modules->mapping(0);
    const auto relative = mapping.readable_bytes(0x2008, 8);
    const auto regions = mapping.guest_regions();
    if (!check(!regions.empty() && regions.front().offset == 0 &&
                   regions.front().permissions == (armigo::memory::Permission::read | armigo::memory::Permission::execute),
               "The guest view lost the text segment's permissions.")) return false;
    return check(mapping.sealed() && !mapping.executable_address(0x100) &&
                     protection(mapping.base_address() + 0x100) == (VM_PROT_READ | VM_PROT_WRITE),
                 "The unexecuted module's text is executable or not writable data.") &&
           check(relative && word(*relative) == mapping.base_address() + 0x2010,
                 "The unexecuted module was not relocated.");
}

int main() {
    if (!embedded_code_page_relocation()) return 1;
    if (!packed_module_maps_without_execute()) return 1;
    const auto consumer = fixture(true, true);
    const auto provider = fixture(false, false);
    const auto consumer_before = consumer;
    const auto provider_before = provider;
    const std::array sources = {NsoSource{"consumer", consumer}, NsoSource{"provider", provider}};
    auto result = ModuleSet::load_nso(sources);
    if (!check(result.ok(), result.message.c_str())) return 1;
    auto& set = *result.modules;
    if (!check(set.module_count() == 2 && set.relocation_count() == 3,
               "The complete module set was not linked.") ||
        !check(consumer == consumer_before && provider == provider_before,
               "Loading changed a source container.") ||
        !check(set.mapping(0).sealed() && set.mapping(1).sealed(), "A module remained unsealed.")) return 1;
    const auto entry = set.entry_address(0, "entry");
    const auto service = set.entry_address(1, "service");
    if (!check(entry && service, "A defined function was not found.")) return 1;
    const auto slot = set.mapping(0).readable_bytes(0x8000, 8);
    const auto relative = set.mapping(0).readable_bytes(0x8008, 8);
    if (!check(slot && word(*slot) == *service, "The cross-module import is wrong.") ||
        !check(relative && word(*relative) == set.mapping(0).base_address() + 0x8010,
               "The packed relative relocation is wrong.") ||
        !check(reinterpret_cast<std::uint64_t (*)()>(*entry)() == 42,
               "The linked native module call returned the wrong result.")) return 1;
    if (!check(!set.entry_address(0, "service") && !set.entry_address(99, "entry"),
               "An undefined or absent module entry was exposed.")) return 1;

    const std::array forced = {armigo::linker::HostBinding{
        "service", reinterpret_cast<std::uintptr_t>(&alternate_service)}};
    ModuleLoadOptions options;
    options.bindings.forced = forced;
    auto overridden = ModuleSet::load_nso(sources, options);
    if (!check(overridden.ok(), overridden.message.c_str())) return 1;
    const auto overridden_entry = overridden.modules->entry_address(0, "entry");
    if (!check(overridden_entry &&
               reinterpret_cast<std::uint64_t (*)()>(*overridden_entry)() == 100,
               "An explicit host override was not used.")) return 1;

    GuestPreparationState guest_state;
    const std::array guest_preparation = {
        ModulePreparation{0, prepare_guest_relocations, &guest_state},
        ModulePreparation{0, prepare_guest_relocations, &guest_state}};
    options = {};
    options.preparation = guest_preparation;
    options.relocation_mode = ModuleRelocationMode::guest_bootstrap;
    auto guest_loaded = ModuleSet::load_nso(sources, options);
    if (!check(guest_loaded.ok() && guest_state.correct && guest_state.calls == 2,
               "Guest relocation mode did not prepare the original images.") ||
        !check(guest_loaded.modules->mapping(0).sealed() &&
                   guest_loaded.modules->mapping(1).sealed(),
               "Guest relocation mode left a module unsealed.") ||
        !check(guest_loaded.modules->mapping(0).base_address() +
                   guest_loaded.modules->mapping(0).mapping_size() <=
                   guest_loaded.modules->mapping(1).base_address(),
               "Guest relocation mappings are not in ascending source order.")) return 1;
    const auto guest_slot = guest_loaded.modules->mapping(0).readable_bytes(0x8000, 8);
    const auto guest_relative = guest_loaded.modules->mapping(0).readable_bytes(0x8008, 8);
    if (!check(guest_slot && word(*guest_slot) == 0 && guest_relative &&
                   word(*guest_relative) == 0x8010,
               "Guest relocation mode modified a runtime relocation target.")) return 1;
    options.bindings.forced = forced;
    if (!check(ModuleSet::load_nso(sources, options).error == ModuleLoadError::invalid_sources,
               "Guest relocation mode accepted host symbol bindings.")) return 1;
    options = {};
    options.relocation_mode = ModuleRelocationMode::guest_bootstrap;
    const std::array incomplete_guest = {sources[0]};
    const auto unresolved_guest = ModuleSet::load_nso(incomplete_guest, options);
    if (!check(unresolved_guest.ok(),
               "Guest relocation mode attempted to resolve a guest-owned import.")) return 1;
    const auto rodata_rela = fixture(true, false, false, false, 0x4800, 0x8008);
    const std::array rodata_rela_source = {NsoSource{"consumer", rodata_rela}};
    const auto refused_rela = ModuleSet::load_nso(rodata_rela_source, options);
    if (!check(refused_rela.error == ModuleLoadError::invalid_dynamic_metadata &&
                   refused_rela.module_index == 0,
               "Guest relocation mode accepted a RELA target in read-only data.")) return 1;
    const auto rodata_relr = fixture(false, false, false, false, 0x8000, 0x4800);
    const std::array rodata_relr_source = {NsoSource{"provider", rodata_relr}};
    const auto refused_relr = ModuleSet::load_nso(rodata_relr_source, options);
    if (!check(refused_relr.error == ModuleLoadError::invalid_dynamic_metadata &&
                   refused_relr.module_index == 0,
               "Guest relocation mode accepted a RELR target in read-only data.")) return 1;

    const std::array incomplete = {sources[0]};
    auto missing = ModuleSet::load_nso(incomplete);
    if (!check(!missing.ok() && missing.error == ModuleLoadError::linking_failed &&
               missing.unresolved_count == 1 && missing.module_index == 0,
               "An unresolved import did not fail the load.")) return 1;
    const auto hidden_provider = fixture(false, false, true);
    const std::array hidden_sources = {
        NsoSource{"first", hidden_provider}, NsoSource{"second", consumer}};
    const auto missing_later = ModuleSet::load_nso(hidden_sources);
    if (!check(missing_later.error == ModuleLoadError::linking_failed &&
               missing_later.module_index == 1, "A later unresolved import was misattributed.")) return 1;
    const std::array duplicate = {sources[0], sources[0]};
    if (!check(ModuleSet::load_nso(duplicate).error == ModuleLoadError::invalid_sources,
               "Duplicate module names were accepted.")) return 1;
    options = {};
    options.max_total_image_bytes = 0x9000;
    if (!check(ModuleSet::load_nso(sources, options).error == ModuleLoadError::resource_limit,
               "The aggregate image limit was ignored.")) return 1;
    auto broken = provider;
    broken[0] = std::byte{0};
    const std::array invalid = {sources[0], NsoSource{"broken", broken}};
    const auto malformed = ModuleSet::load_nso(invalid);
    if (!check(malformed.error == ModuleLoadError::invalid_container && malformed.module_index == 1,
               "A malformed second module produced a usable set.")) return 1;

    PreparationState preparation_state;
    const auto prepared_size = reinterpret_cast<std::uintptr_t>(armigo_test_prepared_end) -
                               reinterpret_cast<std::uintptr_t>(armigo_test_prepared_begin);
    const std::array preparation = {
        ModulePreparation{0, prepare, &preparation_state},
        ModulePreparation{prepared_size, prepare, &preparation_state}};
    options = {};
    options.preparation = preparation;
    auto prepared = ModuleSet::load_nso(sources, options);
    if (!check(prepared.ok() && preparation_state.correct && preparation_state.calls == 2,
               "Post-link preparation failed.")) return 1;
    const auto prepared_entry = prepared.modules->entry_address(0, "entry");
    const auto added = prepared.modules->appended_code(1);
    if (!check(prepared_entry && added && !prepared.modules->appended_code(0) &&
               reinterpret_cast<std::uint64_t (*)()>(*prepared_entry)() == 77,
               "Linked code did not execute through its appended preparation area.") ||
        !check(protection(prepared.modules->mapping(1).base_address() + added->offset) ==
                   (VM_PROT_READ | VM_PROT_EXECUTE),
               "Appended code was not sealed read/execute.") ||
        !check(consumer == consumer_before && provider == provider_before,
               "Preparation changed a source container.")) return 1;
    {
        PreparationState data_state;
        auto with_data = preparation;
        for (auto& requested : with_data) {
            requested.callback = prepare_with_data;
            requested.context = &data_state;
        }
        const auto page = *armigo::platform::HostModule::native_page_size();
        with_data[1].appended_data_bytes = page;
        auto data_options = options;
        data_options.preparation = with_data;
        const auto loaded = ModuleSet::load_nso(sources, data_options);
        if (!check(loaded.ok() && data_state.correct, "Separate writable preparation storage failed.")) return 1;
        const auto code = loaded.modules->appended_code(1);
        const auto data_address = loaded.modules->mapping(1).base_address() + code->offset - page;
        if (!check(code->offset == added->offset + page &&
                   protection(data_address) == (VM_PROT_READ | VM_PROT_WRITE) &&
                   *reinterpret_cast<std::byte*>(data_address) == std::byte{42},
                   "Writable preparation storage was lost or sealed executable.")) return 1;
        *reinterpret_cast<std::byte*>(data_address) = std::byte{43};
        with_data[1].appended_data_bytes = page - 1;
        if (!check(ModuleSet::load_nso(sources, data_options).error == ModuleLoadError::invalid_sources,
                   "Unaligned writable preparation storage was accepted.")) return 1;
    }
    preparation_state = {};
    preparation_state.reject_provider = true;
    const auto refused = ModuleSet::load_nso(sources, options);
    if (!check(!refused.ok() && refused.error == ModuleLoadError::preparation_failed &&
                   refused.module_index == 1 && preparation_state.correct &&
                   preparation_state.calls == 2,
               "A failed preparation returned a module set.") ||
        !check(!protection(preparation_state.bases[0]) && !protection(preparation_state.bases[1]),
               "Failed preparation retained an earlier or partially changed mapping.")) return 1;

    // Callbacks that run at the same time give the module set a serial load would.
    ConcurrentState concurrent;
    const std::array concurrent_preparation = {
        ModulePreparation{0, prepare_concurrently, &concurrent},
        ModulePreparation{prepared_size, prepare_concurrently, &concurrent}};
    ModuleLoadOptions parallel_options;
    parallel_options.preparation = concurrent_preparation;
    parallel_options.parallel_preparation = true;
    const auto parallel = ModuleSet::load_nso(sources, parallel_options);
    if (!check(parallel.ok() && concurrent.calls == 2 &&
                   concurrent.bases[0] == parallel.modules->mapping(0).base_address() &&
                   concurrent.bases[1] == parallel.modules->mapping(1).base_address() &&
                   parallel.modules->appended_code(1) && !parallel.modules->appended_code(0) &&
                   protection(parallel.modules->mapping(1).base_address() +
                              parallel.modules->appended_code(1)->offset) == (VM_PROT_READ | VM_PROT_EXECUTE),
               "Parallel preparation did not prepare and seal every module.")) return 1;
    concurrent.calls = 0;
    concurrent.rejected = 0b10;
    auto parallel_refused = ModuleSet::load_nso(sources, parallel_options);
    if (!check(!parallel_refused.ok() && parallel_refused.error == ModuleLoadError::preparation_failed &&
                   parallel_refused.module_index == 1 && concurrent.calls == 2,
               "Parallel preparation hid a rejected module.")) return 1;
    concurrent.calls = 0;
    concurrent.rejected = 0b11;
    parallel_refused = ModuleSet::load_nso(sources, parallel_options);
    if (!check(parallel_refused.module_index == 0,
               "Parallel preparation did not report the first rejected module.")) return 1;

    // A module decoded by the caller loads as the file would.
    {
        auto consumer_image = armigo::loader::read_nso_module(consumer);
        auto provider_image = armigo::loader::read_nso_module(provider);
        if (!check(consumer_image.ok() && provider_image.ok(), "The fixture modules did not decode.")) return 1;
        const std::array decoded_sources = {
            NsoSource{"consumer", consumer, &*consumer_image.image},
            NsoSource{"provider", provider, &*provider_image.image}};
        ModuleLoadOptions decoded_options;
        decoded_options.preparation = preparation;
        preparation_state = {};
        const auto loaded = ModuleSet::load_nso(decoded_sources, decoded_options);
        const auto entry = loaded.ok() ? loaded.modules->entry_address(0, "entry") : std::nullopt;
        if (!check(loaded.ok() && preparation_state.correct && preparation_state.calls == 2 && entry &&
                       reinterpret_cast<std::uint64_t (*)()>(*entry)() == 77 &&
                       loaded.modules->mapping(0).mapping_size() == prepared.modules->mapping(0).mapping_size() &&
                       loaded.modules->mapping(1).mapping_size() == prepared.modules->mapping(1).mapping_size(),
                   "A pre-decoded module loaded differently from its file.")) return 1;
        auto oversized = armigo::loader::read_nso_module(consumer);
        const std::array oversized_sources = {NsoSource{"consumer", consumer, &*oversized.image}};
        decoded_options = {};
        decoded_options.max_total_image_bytes = 0x100;
        if (!check(ModuleSet::load_nso(oversized_sources, decoded_options).error == ModuleLoadError::resource_limit,
                   "A pre-decoded module bypassed the image budget.")) return 1;
    }
    options.preparation = std::span(preparation).first(1);
    if (!check(ModuleSet::load_nso(sources, options).error == ModuleLoadError::invalid_sources,
               "A mismatched preparation count was accepted.")) return 1;
    const std::array no_callback = {ModulePreparation{}, ModulePreparation{16}};
    options.preparation = no_callback;
    if (!check(ModuleSet::load_nso(sources, options).error == ModuleLoadError::invalid_sources,
               "An appended area without preparation was accepted.")) return 1;
    options.preparation = preparation;
    preparation_state = {};
    const auto outside_provider = fixture(false, false, false, true);
    const std::array outside_sources = {sources[0], NsoSource{"provider", outside_provider}};
    const auto outside = ModuleSet::load_nso(outside_sources, options);
    if (!check(!outside.ok() && outside.error == ModuleLoadError::invalid_dynamic_metadata &&
                   preparation_state.calls == 0,
               "Appended code made an out-of-image symbol valid.")) return 1;
    options.max_total_image_bytes = 2 * 0x9000;
    if (!check(ModuleSet::load_nso(sources, options).error == ModuleLoadError::resource_limit,
               "The preparation area bypassed the aggregate image budget.")) return 1;
    std::puts("NSO decoding, cross-module linking, sealing, and native calls passed.");
    return 0;
}
