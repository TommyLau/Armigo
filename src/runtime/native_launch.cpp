// ============================================================================
// Armigo Runtime - Native Launch Loading
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include <armigo/common/diagnostics.h>
#include <armigo/runtime/launch_plan.h>
#include <armigo/compat/title_plan.h>
#include <armigo/execution/gate_code.h>
#include <armigo/execution/embedded_patch.h>
#include <armigo/platform/host_module.h>

#include "native_preparation.h"
#include "prepared_artifact.h"
#include "sha256.h"
#include "../nvn/entry_table_231.h"
#include "../nvn/redirect.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <limits>
#include <system_error>
#include <thread>

namespace armigo::runtime {
namespace {

LaunchFailure fail(LaunchError code, std::string operation, std::string detail) {
    auto failure = preparation::failure(code, std::move(operation), std::move(detail));
    failure.stage = LaunchStage::modules;
    return failure;
}

struct NativeLoading {
    std::vector<native::ModuleCodePreparation> modules;
    std::vector<execution::CallHookSite> call_hooks;
    std::vector<std::byte> hook_gate;
    std::shared_ptr<compat::InstalledPatch> compatibility_patch;
    std::optional<std::size_t> compatibility_module;
    std::vector<compat::WordPatch> words;
    std::optional<std::size_t> word_module;
    // Embedded code: words whose immediate is chosen at launch. Each site is a
    // gate site holding expected_instruction; the site table gets the
    // replacement (the original word when the launch changes nothing).
    std::vector<compat::WordPatch> launch_words;
    std::optional<std::size_t> launch_module;
    std::optional<std::size_t> nvn_module;
    // The host bootstrap loader of the selected API version.
    nvn::BootstrapLoader nvn_loader = nullptr;
    std::optional<compat::CallPatch> gpu_budget_hook;
    EmbeddedNativeArtifacts* capture = nullptr;
    const EmbeddedNativeArtifacts* artifacts = nullptr;
};

std::array<std::byte, 32> code_settings_digest(const NativeLoading& loading) {
    std::vector<std::byte> bytes;
    const auto number = [&](std::uint64_t value) {
        for (unsigned i = 0; i < 8; ++i) bytes.push_back(std::byte(value >> (8 * i)));
    };
    number(loading.nvn_module ? *loading.nvn_module + 1 : 0);
    number(loading.word_module ? *loading.word_module + 1 : 0);
    number(loading.words.size());
    for (const auto& word : loading.words) {
        number(word.offset); number(word.expected_instruction); number(word.replacement_instruction);
    }
    number(loading.compatibility_module ? *loading.compatibility_module + 1 : 0);
    number(loading.compatibility_patch ? loading.compatibility_patch->description.calls.size() : 0);
    if (loading.compatibility_patch) for (const auto& call : loading.compatibility_patch->description.calls) {
        number(call.offset); number(call.expected_instruction);
        if (call.phases) {
            number(0x7068617365); // Phase-call metadata extends only the selected mode's identity.
            number(call.phases->original_callee);
            number(call.phases->counter);
            number(call.phases->phase_count);
            number(loading.compatibility_patch->description.layout.frame_step);
        }
        if (call.gpu_budget) {
            number(0x6770756275646765); // GPU budget instruction metadata.
            number(call.gpu_budget->target_rate);
        }
        if (call.frame_ratio) {
            number(0x726174696f); // Frame-ratio division metadata.
            number(call.frame_ratio->divisor_register);
        }
    }
    // Launch-time sites are code; the words they apply are not.
    if (!loading.launch_words.empty()) {
        number(*loading.launch_module + 1);
        number(loading.launch_words.size());
        for (const auto& word : loading.launch_words) { number(word.offset); number(word.expected_instruction); }
    }
    const auto digest = loader::detail::sha256(bytes);
    if (!digest) throw std::runtime_error("Embedded code settings could not be hashed.");
    return *digest;
}

void validate_native_artifact(const EmbeddedNativeModule& artifact,
    const loader::ExeFsFile& source, const loader::NsoModuleImage& original,
    const EmbeddedNativeBacking& backing, std::span<const native::ModuleCodePreparation> modules) {
    constexpr std::uint64_t page = execution::embedded_key_page_size;
    const auto& text = original.segment(loader::NsoSegmentKind::text);
    const auto digest = loader::detail::sha256(source.bytes);
    const auto key = (original.bytes.size() + page - 1) & ~(page - 1);
    if (!digest || artifact.name != source.name || artifact.source_sha256 != *digest ||
        artifact.original_image_size != original.bytes.size() || artifact.key_page_offset != key ||
        artifact.appended_code_offset != key + page || artifact.image.size() <= key + page ||
        artifact.image.size() % page || artifact.image.size() > (1ull << 30) ||
        artifact.text_offset != text.image_offset || artifact.text_size != text.content_size ||
        artifact.host_executes != (source.name != "rtld") ||
        artifact.code_bytes > artifact.image.size() - artifact.appended_code_offset)
        throw std::runtime_error("Embedded module identity or layout does not match the source.");
    if (!std::equal(original.bytes.begin(), original.bytes.begin() + text.image_offset, artifact.image.begin()) ||
        !std::equal(original.bytes.begin() + text.image_offset + text.content_size, original.bytes.end(),
                    artifact.image.begin() + text.image_offset + text.content_size) ||
        !std::all_of(artifact.image.begin() + original.bytes.size(),
                     artifact.image.begin() + artifact.appended_code_offset,
                     [](std::byte byte) { return byte == std::byte{0}; }))
        throw std::runtime_error("Embedded image contains altered data or persisted runtime bindings.");
    if (artifact.sites.size() > 1'000'000 || artifact.entry_offsets.size() > 1'000'000 ||
        artifact.foreign_entries.size() > 1'000'000 || artifact.stops.size() > 1'000'000 ||
        artifact.hooks.size() > 64 || artifact.scalar_sites > text.content_size / 4 ||
        artifact.thread_register_sites > text.content_size / 4)
        throw std::runtime_error("Embedded metadata exceeds its record limits.");
    const auto text_location = [&](std::uint64_t offset) {
        return offset % 4 == 0 && offset >= text.image_offset &&
            offset - text.image_offset < text.content_size;
    };
    const auto appendix_location = [&](std::uint64_t offset) {
        return offset % 4 == 0 && offset >= artifact.appended_code_offset && offset < artifact.image.size();
    };
    const auto foreign = [&](execution::PreparedLocation location) {
        if (location.module_index >= modules.size()) return false;
        const auto& target = modules[location.module_index];
        return location.image_offset % 4 == 0 && location.image_offset >= target.text_begin &&
            location.image_offset - target.text_begin < target.text_size;
    };
    std::optional<std::uint64_t> previous;
    for (const auto offset : artifact.entry_offsets) {
        if (!text_location(offset) || (previous && offset <= *previous))
            throw std::runtime_error("Embedded entries are not sorted unique text locations.");
        previous = offset;
    }
    for (const auto& site : artifact.sites) {
        if (!appendix_location(site.return_address_offset))
            throw std::runtime_error("Embedded gate identity is outside the appendix.");
        if (site.allowed_indirect_targets.size() > 1'000'000)
            throw std::runtime_error("Embedded gate target count exceeds its limit.");
        for (auto location : site.allowed_indirect_targets) if (!foreign(location))
            throw std::runtime_error("Embedded gate target is outside executable text.");
    }
    for (auto location : artifact.foreign_entries) if (!foreign(location))
        throw std::runtime_error("Embedded foreign entry is outside executable text.");
    for (const auto& stop : artifact.stops) if (!text_location(stop.image_offset) || stop.reason.size() > 4096)
        throw std::runtime_error("Embedded stop metadata is invalid.");
    for (const auto& hook : artifact.hooks) {
        if (!appendix_location(hook.return_offset) || hook.call_return_offset < 4 ||
            !text_location(hook.call_return_offset - 4) || !foreign(hook.target))
            throw std::runtime_error("Embedded call-hook metadata is invalid.");
    }
    std::size_t cursor = 0;
    for (const auto& piece : backing.pieces) {
        if (piece.offset != cursor || piece.bytes.empty() || piece.bytes.size() > artifact.image.size() - cursor ||
            !std::equal(piece.bytes.begin(), piece.bytes.end(), artifact.image.begin() + cursor))
            throw std::runtime_error("Embedded mapped backing differs from its captured image.");
        cursor += piece.bytes.size();
    }
    if (cursor != artifact.image.size()) throw std::runtime_error("Embedded mapped backing is incomplete.");
}

void restore_native_module(const ModulePreparationView& view, NativeLoading& loading) {
    auto& module = loading.modules[view.module_index];
    const auto& saved = loading.artifacts->modules[view.module_index];
    const auto address = [&](execution::PreparedLocation location) {
        return view.module_bases[location.module_index] + location.image_offset;
    };
    module.prepared_begin = view.base_address + module.text_begin;
    module.prepared_end = module.prepared_begin + module.text_size;
    module.code_bytes = saved.code_bytes;
    module.scalar_sites = saved.scalar_sites;
    module.thread_register_sites = saved.thread_register_sites;
    for (const auto& site : saved.sites) {
        execution::GuestGateSite restored{view.base_address + site.return_address_offset, site.instruction, {}};
        for (auto target : site.allowed_indirect_targets) restored.allowed_indirect_targets.push_back(address(target));
        module.sites.push_back(std::move(restored));
    }
    for (auto offset : saved.entry_offsets) module.prepared_entries.push_back(view.base_address + offset);
    for (auto location : saved.foreign_entries) module.required_foreign_entries.push_back(address(location));
    for (const auto& stop : saved.stops) module.stops.push_back({view.base_address + stop.image_offset, stop.instruction, stop.reason});
    if (loading.compatibility_module == view.module_index) {
        for (const auto& hook : saved.hooks) loading.call_hooks.push_back({view.base_address + hook.return_offset,
            view.base_address + hook.call_return_offset, address(hook.target)});
        loading.compatibility_patch->module_base = view.base_address;
    } else if (!saved.hooks.empty()) {
        throw std::runtime_error("Embedded hooks have no selected compatibility description.");
    }
    const std::uint64_t slot = module.slot;
    const auto dispatcher = execution::gate_dispatcher_address();
    std::memcpy(view.appended_data.data(), &slot, 8);
    std::memcpy(view.appended_data.data() + execution::embedded_gate_dispatcher_offset, &dispatcher, 8);
    if (loading.nvn_module == view.module_index) {
        const auto nvn = reinterpret_cast<std::uintptr_t>(
            loading.nvn_loader ? loading.nvn_loader : &nvn::armigo_nvn_bootstrap_loader);
        std::memcpy(view.appended_data.data() + execution::embedded_nvn_bootstrap_offset, &nvn, 8);
    }
    module.completed = true;
}

void capture_native_module(const ModulePreparationView& view, const NativeLoading& loading) {
    auto& artifact = loading.capture->modules.at(view.module_index);
    const auto& module = loading.modules.at(view.module_index);
    const auto local = [&](std::uint64_t address) {
        if (address < view.base_address || address - view.base_address >= artifact.image.size())
            throw std::runtime_error("Embedded metadata address is outside its module.");
        return address - view.base_address;
    };
    const auto location = [&](std::uint64_t address) -> execution::PreparedLocation {
        for (std::size_t i = 0; i < view.module_bases.size(); ++i) {
            const auto base = view.module_bases[i];
            const auto& target = loading.modules[i];
            if (address >= base && address - base >= target.text_begin &&
                address - base - target.text_begin < target.text_size)
                return {static_cast<std::uint32_t>(i), address - base};
        }
        throw std::runtime_error("Embedded metadata target has no executable module owner.");
    };
    artifact.key_page_offset = view.appended_data_offset;
    artifact.appended_code_offset = view.appended_code_offset;
    artifact.image.resize(view.appended_code_offset + view.appended_code.size());
    std::memcpy(artifact.image.data() + module.text_begin,
                view.image.data() + module.text_begin, module.text_size);
    std::memcpy(artifact.image.data() + view.appended_code_offset,
                view.appended_code.data(), view.appended_code.size());
    for (const auto& site : module.sites) {
        execution::PreparedGateSite saved{local(site.return_address), site.instruction, {}};
        for (auto address : site.allowed_indirect_targets) saved.allowed_indirect_targets.push_back(location(address));
        artifact.sites.push_back(std::move(saved));
    }
    for (auto address : module.prepared_entries) artifact.entry_offsets.push_back(local(address));
    for (auto address : module.required_foreign_entries) artifact.foreign_entries.push_back(location(address));
    for (const auto& stop : module.stops)
        artifact.stops.push_back({local(stop.address), stop.word, stop.reason});
    if (loading.compatibility_module == view.module_index) for (const auto& hook : loading.call_hooks)
        artifact.hooks.push_back({local(hook.return_address), local(hook.call_return_address), location(hook.target_address)});
    artifact.code_bytes = module.code_bytes;
    artifact.scalar_sites = module.scalar_sites;
    artifact.thread_register_sites = module.thread_register_sites;
}

// A launch-time site: its gate return address and the words it holds and gets.
struct LaunchSite {
    std::uint64_t return_address = 0;
    std::uint32_t expected_instruction = 0;
    std::uint32_t replacement_instruction = 0;
};

// Preparation callbacks of different modules run at the same time: a callback
// writes only its own module's entry in `loading.modules`, its own context, and
// the compatibility state of the one module that owns it.
struct ModuleContext {
    NativeLoading* loading = nullptr;
    std::size_t index = 0;
    std::string error;
    bool allocation_failed = false;
    std::vector<LaunchSite> launch_sites;
};

// Each launch-time word's site branches to its gate trampoline; the gate
// returns to the trampoline's third word. Reads the prepared or restored text.
bool find_launch_sites(const ModulePreparationView& view, ModuleContext& context) {
    const auto& loading = *context.loading;
    if (loading.launch_module != context.index) return true;
    for (const auto& word : loading.launch_words) {
        std::uint32_t branch = 0;
        std::memcpy(&branch, view.image.data() + word.offset, 4);
        if ((branch & 0xfc000000u) != 0x14000000u) {
            context.error = "A launch-time word is not a prepared gate site.";
            return false;
        }
        const auto delta = std::int64_t{static_cast<std::int32_t>(branch << 6) >> 6} * 4;
        context.launch_sites.push_back({view.base_address + word.offset + static_cast<std::uint64_t>(delta) + 8,
                                        word.expected_instruction, word.replacement_instruction});
    }
    return true;
}

bool prepare_native_module(const ModulePreparationView& view, void* raw) noexcept {
    auto& context = *static_cast<ModuleContext*>(raw);
    auto& loading = *context.loading;
    auto& module = loading.modules[context.index];
    try {
        if (loading.artifacts) {
            restore_native_module(view, loading);
            return find_launch_sites(view, context);
        }
        // The NVN bootstrap entry must leave preparation untouched before the
        // driver redirect replaces it.
        execution::PreparedImage entry_words;
        if (loading.nvn_module == context.index) {
            entry_words.original_image_size = view.image.size();
            for (const auto& symbol : view.metadata.symbols) {
                if (symbol.name != "nvnBootstrapLoader" || !symbol.defined() ||
                    symbol.value > view.image.size() || view.image.size() - symbol.value < 16) continue;
                for (std::uint64_t offset = symbol.value; offset < symbol.value + 16; offset += 4) {
                    std::uint32_t word;
                    std::memcpy(&word, view.image.data() + offset, 4);
                    entry_words.coverage.push_back({offset, word, word});
                }
            }
        }
        if (loading.launch_module == context.index)
            for (const auto& word : loading.launch_words) module.launch_words.push_back(word.offset);
        module.prepare(view);
        if (!find_launch_sites(view, context)) return false;
        if (module.embedded) {
            const std::uint64_t slot = module.slot;
            const auto dispatcher = execution::gate_dispatcher_address();
            std::memcpy(view.appended_data.data() + execution::embedded_thread_slot_offset, &slot, 8);
            std::memcpy(view.appended_data.data() + execution::embedded_gate_dispatcher_offset, &dispatcher, 8);
        }
        if (loading.nvn_module == context.index) {
            const auto stub = (module.code_bytes + 3) & ~std::size_t{3};
            const auto redirected = nvn::redirect_bootstrap_loader(view, entry_words,
                module.text_begin, module.text_size, module.embedded ? std::optional(stub) : std::nullopt,
                loading.nvn_loader);
            if (!redirected.ok()) {
                context.error = redirected.error;
                return false;
            }
            if (module.embedded) module.code_bytes = stub + 12;
        }
        if (loading.word_module == context.index) {
            for (const auto& word : loading.words) {
                std::uint32_t actual = 0;
                std::memcpy(&actual, view.image.data() + word.offset, 4);
                if (actual != word.expected_instruction) {
                    context.error = "Compatibility instruction word changed during preparation.";
                    return false;
                }
                std::memcpy(view.image.data() + word.offset, &word.replacement_instruction, 4);
            }
        }
        if (loading.compatibility_module == context.index) {
            const auto& description = loading.compatibility_patch->description;
            const auto gate_offset = (module.code_bytes + 15) & ~std::size_t{15};
            const auto gate_address = view.base_address + view.appended_code_offset + gate_offset;
            const auto embedded_gate = module.embedded ? execution::make_embedded_gate_code(gate_address,
                view.base_address + view.appended_data_offset + execution::embedded_gate_dispatcher_offset)
                : std::optional<std::vector<std::byte>>{};
            if (module.embedded && !embedded_gate) {
                context.error = "Compatibility gate cannot address its embedded key page.";
                return false;
            }
            const auto& hook_gate = module.embedded ? *embedded_gate : loading.hook_gate;
            const auto& observers = loading.compatibility_patch->observers;
            std::size_t call_bytes = 0;
            for (const auto& call : description.calls) call_bytes += compat::call_stub_bytes(call);
            const auto need = hook_gate.size() + call_bytes + observers.size() * compat::observe_stub_bytes;
            if (gate_offset > view.appended_code.size() || need > view.appended_code.size() - gate_offset) {
                context.error = "Compatibility hooks do not fit the preparation appendix.";
                return false;
            }
            std::memcpy(view.appended_code.data() + gate_offset, hook_gate.data(), hook_gate.size());
            auto stub_offset = gate_offset + hook_gate.size();
            for (std::size_t i = 0; i < description.calls.size(); ++i) {
                const auto& site = description.calls[i];
                std::uint32_t actual = 0;
                std::memcpy(&actual, view.image.data() + site.offset, 4);
                if (actual != site.expected_instruction) {
                    context.error = "Compatibility instruction changed during preparation.";
                    return false;
                }
                const auto stub = view.base_address + view.appended_code_offset + stub_offset;
                if (site.frame_ratio) {
                    // The gate runs before the division, which then runs as it is.
                    auto ratio = compat::make_observe_hook(actual, view.base_address + site.offset, stub,
                                                           {gate_address, hook_gate.size()});
                    if (const auto* failure = std::get_if<std::string>(&ratio)) {
                        context.error = "Compatibility frame-ratio hook: " + *failure;
                        return false;
                    }
                    const auto& made = std::get<compat::ObserveHook>(ratio);
                    std::memcpy(view.appended_code.data() + stub_offset, made.trampoline.data(), compat::observe_stub_bytes);
                    std::memcpy(view.image.data() + site.offset, &made.replacement_instruction, 4);
                    loading.call_hooks.push_back(made.site);
                    stub_offset += compat::observe_stub_bytes;
                    continue;
                }
                const auto hook = compat::make_patch_hook(site, actual, view.base_address + site.offset, stub,
                    {gate_address, hook_gate.size()});
                if (!hook.ok()) {
                    context.error = execution::describe(hook.error);
                    return false;
                }
                std::memcpy(view.appended_code.data() + stub_offset, hook.patch->trampoline.data(), 16);
                std::memcpy(view.image.data() + site.offset, &hook.patch->replacement_instruction, 4);
                loading.call_hooks.push_back(hook.patch->site);
                stub_offset += 16;
            }
            for (std::size_t i = 0; i < observers.size(); ++i) {
                const auto& point = observers[i];
                std::uint32_t actual = 0;
                std::memcpy(&actual, view.image.data() + point.offset, 4);
                if (actual != point.expected_instruction) {
                    context.error = "Observer '" + point.text + "' is an instruction that preparation rewrote.";
                    return false;
                }
                const auto stub_offset = gate_offset + hook_gate.size() + call_bytes + i * compat::observe_stub_bytes;
                auto hook = compat::make_observe_hook(actual, view.base_address + point.offset,
                    view.base_address + view.appended_code_offset + stub_offset, {gate_address, hook_gate.size()});
                if (const auto* failure = std::get_if<std::string>(&hook)) {
                    context.error = "Observer '" + point.text + "': " + *failure;
                    return false;
                }
                const auto& made = std::get<compat::ObserveHook>(hook);
                std::memcpy(view.appended_code.data() + stub_offset, made.trampoline.data(), compat::observe_stub_bytes);
                std::memcpy(view.image.data() + point.offset, &made.replacement_instruction, 4);
                loading.call_hooks.push_back(made.site);
                armigo::diag::write(armigo::diag::Level::info, "Armigo observe: main+%#llx (%#010x) observed\n",
                                    static_cast<unsigned long long>(point.offset), actual);
            }
            loading.compatibility_patch->module_base = view.base_address;
        }
        if (loading.capture) capture_native_module(view, loading);
        return true;
    } catch (const std::bad_alloc&) {
        context.allocation_failed = true;
        return false;
    } catch (const std::exception& error) {
        context.error = error.what();
        return false;
    } catch (...) {
        return false;
    }
}

// The one module that defines a startup function; imports never qualify.
std::optional<std::pair<std::size_t, std::uint64_t>> defined_function(const ModuleSet& modules,
                                                                      std::string_view name) {
    std::optional<std::pair<std::size_t, std::uint64_t>> found;
    for (std::size_t i = 0; i != modules.module_count(); ++i) {
        const auto address = modules.entry_address(i, name);
        if (!address) continue;
        if (found) return {};
        found = {i, *address};
    }
    return found;
}

} // namespace

std::string installed_embedded_preparation_identity() {
    const auto gate = execution::make_embedded_gate_code(0x8000, 0x4008);
    const auto digest = gate ? loader::detail::sha256(*gate) : std::nullopt;
    if (!digest) throw std::runtime_error("Embedded gate identity could not be computed.");
    std::string result = "armigo-native-embedded-v1-r3-";
    constexpr char hex[] = "0123456789abcdef";
    for (const auto byte : *digest) {
        const auto value = std::to_integer<unsigned>(byte);
        result.push_back(hex[value >> 4]); result.push_back(hex[value & 15]);
    }
    return result;
}

LaunchResult<PreparedLaunch> load_native_launch(const loader::ExeFsDirectoryInput& input,
    const LaunchProfile& profile, std::shared_ptr<platform::ThreadSlotKey> thread_slot,
    std::span<const compat::PatchDescription> patches, const NativeLaunchOptions& native_options) {
    try {
        if ((native_options.artifacts && (!native_options.embedded_preparation || native_options.capture)) ||
            (!native_options.artifacts && !native_options.backings.empty()))
            return fail(LaunchError::invalid_input, "load-native", "Invalid embedded artifact loading options.");
        if (native_options.capture && !native_options.embedded_preparation)
            return fail(LaunchError::invalid_input, "load-native", "Artifact capture requires embedded preparation.");
        if (profile.origin != ProfileOrigin::npdm || !profile.is_64_bit)
            return fail(LaunchError::unsupported, "load-native", "Native launch requires 64-bit NPDM-backed input.");
        if (!thread_slot || !thread_slot->valid() || !thread_slot->native_access())
            return fail(LaunchError::invalid_input, "load-native", "A valid probed native thread-slot key is required.");
        const auto order = preparation::original_module_order(input);
        if (!order)
            return fail(LaunchError::unsupported, "load-native",
                        "Executable modules do not match the fixed executable slot roster.");
        if (native_options.artifacts && (native_options.artifacts->version != 1 ||
            native_options.artifacts->title_id != profile.title_id ||
            native_options.artifacts->preparation_identity != installed_embedded_preparation_identity() ||
            native_options.artifacts->modules.size() != order->size() || native_options.backings.size() != order->size()))
            return fail(LaunchError::artifact_mismatch, "load-native", "Embedded artifact identity or module roster differs.");
        const auto page_size = platform::HostModule::native_page_size();
        if (!page_size)
            return fail(LaunchError::host_failure, "load-native", "The native module page size is unavailable.");

        EmbeddedNativeArtifacts capture;
        capture.title_id = profile.title_id;
        if (native_options.capture) {
            capture.preparation_identity = installed_embedded_preparation_identity();
            capture.modules.resize(order->size());
        }
        NativeLoading loading;
        loading.capture = native_options.capture ? &capture : nullptr;
        loading.artifacts = native_options.artifacts;
        loading.modules.resize(order->size());
        std::vector<loader::NsoModuleImage> images;
        // Each module decodes and scans its text on its own thread; the work
        // that follows needs every module, so it waits for all of them.
        struct Decoded {
            loader::NsoReadResult read;
            std::size_t appendix = 0;
            std::exception_ptr failure;
        };
        std::vector<Decoded> decoded(order->size());
        const auto embedded_gate = native_options.embedded_preparation && !native_options.artifacts
            ? execution::make_embedded_gate_code(0x8000, 0x4008) : std::optional<std::vector<std::byte>>{};
        if (native_options.embedded_preparation && !native_options.artifacts && !embedded_gate)
            return fail(LaunchError::host_failure, "load-native", "Embedded gate template is invalid.");
        const auto gate = native_options.artifacts ? std::vector<std::byte>{} :
            native_options.embedded_preparation ? *embedded_gate : execution::make_local_gate_code();
        const unsigned scan_threads = std::clamp(std::thread::hardware_concurrency(), 1u, 8u);
        const auto decode_module = [&](std::size_t position) noexcept {
            try {
                auto& entry = decoded[position];
                entry.read = loader::read_nso_module(input.modules[(*order)[position]].bytes);
                if (!entry.read.ok()) return;
                const auto& text = entry.read.image->segment(loader::NsoSegmentKind::text);
                auto& module = loading.modules[position];
                module.embedded = native_options.embedded_preparation;
                module.slot = thread_slot->native_access()->tsd_byte_offset;
                module.gate = gate;
                module.text_begin = text.image_offset;
                module.text_size = text.content_size;
                module.module_header = entry.read.image->module_header;
                if (!native_options.artifacts) entry.appendix = module.scan_text(
                    std::span(entry.read.image->bytes).subspan(text.image_offset, text.content_size), scan_threads,
                    entry.read.image->bytes);
            } catch (...) {
                decoded[position].failure = std::current_exception();
            }
        };
        {
            std::vector<std::thread> workers;
            std::size_t started = 1;
            try {
                for (; started < order->size(); ++started) workers.emplace_back(decode_module, started);
            } catch (const std::system_error&) {
                // The modules without a thread decode below, on this one.
            }
            decode_module(0);
            for (auto position = started; position < order->size(); ++position) decode_module(position);
            for (auto& worker : workers) worker.join();
        }
        for (std::size_t position = 0; position < order->size(); ++position) {
            if (decoded[position].failure) std::rethrow_exception(decoded[position].failure);
            if (!decoded[position].read.ok())
                return fail(LaunchError::invalid_input, "load-native",
                            "Module " + input.modules[(*order)[position]].name + " is not a valid NSO.");
            if (native_options.artifacts) {
                try {
                    validate_native_artifact(native_options.artifacts->modules[position], input.modules[(*order)[position]],
                        *decoded[position].read.image, native_options.backings[position], loading.modules);
                } catch (const std::runtime_error& error) {
                    return fail(LaunchError::artifact_mismatch, "load-native", error.what());
                }
                const auto& artifact = native_options.artifacts->modules[position];
                decoded[position].appendix = artifact.image.size() - artifact.appended_code_offset;
            }
            if (native_options.capture) {
                auto& artifact = capture.modules[position];
                const auto& source = input.modules[(*order)[position]];
                const auto digest = loader::detail::sha256(source.bytes);
                if (!digest) return fail(LaunchError::host_failure, "load-native", "Embedded source hashing failed.");
                artifact.name = source.name;
                artifact.source_sha256 = *digest;
                artifact.image = decoded[position].read.image->bytes;
                artifact.original_image_size = artifact.image.size();
                artifact.text_offset = loading.modules[position].text_begin;
                artifact.text_size = loading.modules[position].text_size;
            }
            images.push_back(std::move(*decoded[position].read.image));
        }
        const auto position_of = [&](std::string_view name) -> std::optional<std::size_t> {
            for (std::size_t position = 0; position < order->size(); ++position)
                if (input.modules[(*order)[position]].name == name) return position;
            return {};
        };

        bool title_nvn_enabled = false;
        std::string title_pso_manifest;
        bool title_startup_user = false;
        nvn::TitlePolicy title_nvn_policy;
        bool title_matched = false;
        const compat::PatchDescription* rate_patch = nullptr;
        std::string requested_title_rate;
        if (const auto main = position_of("main"); main && !patches.empty()) {
            const auto& source = images[*main];
            if (const auto* patch = compat::match_patch(patches, {profile.title_id, source.build_id})) {
                title_matched = true;
                title_nvn_enabled = patch->nvn_driver;
                title_pso_manifest = patch->pso_manifest;
                title_startup_user = patch->startup_user;
                title_nvn_policy.cpu_cached_texture_writes = patch->nvn_policy.cpu_cached_texture_writes;
                if (patch->nvn_policy.segment_timestamps)
                    title_nvn_policy.timestamps = nvn::TitlePolicy::Timestamps::segment;
                if (patch->nvn_policy.game_barriers)
                    title_nvn_policy.barriers = nvn::TitlePolicy::Barriers::game;
                title_nvn_policy.hoist_implicit_samples = patch->nvn_policy.hoist_implicit_samples;
                if (auto error = compat::validate_patch(*patch, source); !error.empty())
                    return fail(LaunchError::artifact_mismatch, "compatibility-preflight", std::move(error));
                // Embedded capture records the profile's default independently of
                // a developer's launch environment. Restoring another mode must
                // still satisfy the captured preparation fingerprint.
                const char* requested_rate = native_options.embedded_preparation && !native_options.artifacts
                    ? nullptr : std::getenv("ARMIGO_COMPAT_FRAME_RATE");
                if (patch->frame_rate) {
                    rate_patch = patch;
                    requested_title_rate = requested_rate ? requested_rate : "";
                }
                const bool ahead_of_time = native_options.embedded_preparation || native_options.artifacts;
                auto planned = compat::plan_title_words(*patch, source,
                    {requested_rate ? requested_rate : "", ahead_of_time});
                if (const auto* failure = std::get_if<compat::TitlePlanFailure>(&planned))
                    return fail(failure->kind == compat::TitlePlanFailure::Kind::invalid_input
                        ? LaunchError::invalid_input : LaunchError::artifact_mismatch, failure->stage, failure->detail);
                auto& plan = std::get<compat::TitlePlan>(planned);
                auto words = std::move(plan.words);
                auto launch_words = std::move(plan.launch_words);
                if (!plan.module_calls.empty()) {
                    loading.gpu_budget_hook = plan.module_calls.front();
                    if (patch->module != "frame-rate" || ahead_of_time)
                        return fail(LaunchError::unsupported, "compatibility-resolution",
                                    "GPU-work feedback requires load-time preparation and the title frame-rate module.");
                }
                if (!launch_words.empty()) {
                    loading.launch_module = *main;
                    loading.launch_words = std::move(launch_words);
                }
                if (!words.empty()) {
                    loading.word_module = *main;
                    loading.words = std::move(words);
                }
                if (!patch->calls.empty()) {
                    loading.compatibility_module = *main;
                    loading.compatibility_patch =
                        std::make_shared<compat::InstalledPatch>(compat::InstalledPatch{*patch, 0});
                    if (loading.gpu_budget_hook)
                        loading.compatibility_patch->description.calls.push_back(*loading.gpu_budget_hook);
                    if (auto error = compat::validate_patch(loading.compatibility_patch->description, source); !error.empty())
                        return fail(LaunchError::artifact_mismatch, "compatibility-preflight", std::move(error));
                    loading.hook_gate = gate;
                }
            }
        }
        if (!patches.empty() && !title_matched)
            armigo::diag::write(armigo::diag::Level::info,  "Armigo: the title settings describe no build matching this main module; running without them.\n");
        const char* override_environment = std::getenv("ARMIGO_NVN");
        const std::string_view override_value = override_environment ? override_environment : "";
        if (const auto selection = nvn::select_driver(false, {}, override_value); selection.error)
            return fail(LaunchError::invalid_input, "nvn-selection", selection.error);
        std::optional<nvn::ApiVersion> nvn_version;
        if (const auto sdk = position_of("sdk")) {
            const auto selection = nvn::select_driver(title_nvn_enabled, images[*sdk].bytes, override_value);
            if (selection.error) return fail(LaunchError::invalid_input, "nvn-selection", selection.error);
            if (selection.version) {
                loading.nvn_module = *sdk;
                loading.nvn_loader = nvn::bootstrap_loader_for(*selection.version);
                nvn_version = selection.version;
            } else if ((title_nvn_enabled || override_value == "driver") && override_value != "stream") {
                const auto package = nvn::driver_package(images[*sdk].bytes);
                armigo::diag::write(armigo::diag::Level::warning, 
                             "Armigo NVN: the driver is requested, but the NVN package %.*s is not one the driver "
                             "implements; using the command-stream renderer.\n",
                             package ? static_cast<int>(package->size()) : 7, package ? package->data() : "unknown");
            }
        }
        // Words meant for the NVN driver apply only when it replaces the game's own.
        std::uint32_t admitted_frame_rate = 0;
        if (rate_patch) {
            auto effective = compat::select_patch_words(*rate_patch, requested_title_rate,
                loading.nvn_module.has_value(), &admitted_frame_rate);
            if (const auto* failure = std::get_if<std::string>(&effective))
                return fail(LaunchError::invalid_input, "compatibility-frame-rate", *failure);
        }
        if (loading.compatibility_patch) {
            if (loading.gpu_budget_hook && !loading.nvn_module)
                return fail(LaunchError::unsupported, "compatibility-resolution", "GPU-work feedback requires the NVN driver.");
            auto& description = loading.compatibility_patch->description;
            auto selected = compat::select_patch_calls(description, requested_title_rate, loading.nvn_module.has_value());
            if (const auto* failure = std::get_if<std::string>(&selected))
                return fail(LaunchError::invalid_input, "compatibility-frame-rate", *failure);
            description.calls = std::get<std::vector<compat::CallPatch>>(std::move(selected));
            if (description.calls.empty()) {
                loading.compatibility_patch.reset();
                loading.compatibility_module.reset();
            } else {
                compat::set_present_link(description.present_link);
                compat::initialize_gpu_budget(*loading.compatibility_patch);
                compat::initialize_frame_ratio(*loading.compatibility_patch);
            }
        }
        // Debugging observers share the compatibility gate of the main module.
        if (const char* observe = std::getenv("ARMIGO_OBSERVE"); observe && *observe) {
            const auto main = position_of("main");
            if (native_options.embedded_preparation || native_options.artifacts || !main)
                return fail(LaunchError::unsupported, "observe", "Observers need a load-time prepared main module.");
            auto parsed = compat::parse_observe_points(observe);
            if (const auto* failure = std::get_if<std::string>(&parsed))
                return fail(LaunchError::invalid_input, "observe", *failure);
            auto points = std::get<std::vector<compat::ObservePoint>>(std::move(parsed));
            const auto& text = images[*main].segment(loader::NsoSegmentKind::text);
            for (auto& point : points) {
                if (point.offset < text.image_offset || point.offset - text.image_offset >= text.content_size)
                    return fail(LaunchError::invalid_input, "observe", "Observer '" + point.text + "' is outside main's text.");
                std::memcpy(&point.expected_instruction, images[*main].bytes.data() + point.offset, 4);
                if (loading.compatibility_patch)
                    for (const auto& call : loading.compatibility_patch->description.calls)
                        if (call.offset == point.offset)
                            return fail(LaunchError::invalid_input, "observe", "Observer '" + point.text + "' is a compatibility call.");
            }
            if (!loading.compatibility_patch) {
                loading.compatibility_patch = std::make_shared<compat::InstalledPatch>();
                loading.compatibility_patch->description.module = "observe";
            }
            loading.compatibility_module = *main;
            loading.hook_gate = gate;
            loading.compatibility_patch->observers = std::move(points);
        }
        if (!loading.nvn_module)
            std::erase_if(loading.words, [](const compat::WordPatch& word) { return word.nvn_driver_only; });
        if (loading.words.empty()) loading.word_module.reset();
        if (native_options.capture) capture.code_settings_sha256 = code_settings_digest(loading);
        if (native_options.artifacts && native_options.artifacts->code_settings_sha256 != code_settings_digest(loading))
            return fail(LaunchError::artifact_mismatch, "load-native",
                        "Embedded code settings differ from the selected title settings; pack the app again.");

        if (native_options.artifacts) for (std::size_t i = 0; i < order->size(); ++i) {
            const auto& hooks = native_options.artifacts->modules[i].hooks;
            const auto expected = loading.compatibility_module == i ? loading.compatibility_patch->description.calls.size() : 0;
            if (hooks.size() != expected)
                return fail(LaunchError::artifact_mismatch, "load-native", "Embedded hook count differs from title settings.");
            for (std::size_t j = 0; j < hooks.size(); ++j) {
                const auto& description = loading.compatibility_patch->description;
                if (hooks[j].call_return_offset != description.calls[j].offset + 4 ||
                    hooks[j].target.module_index != i ||
                    hooks[j].target.image_offset != compat::call_callee(description, description.calls[j]))
                    return fail(LaunchError::artifact_mismatch, "load-native", "Embedded hook target differs from title settings.");
            }
        }
        // The host links and initializes the modules, so rtld's own code never
        // runs (it defines no symbols for other modules). Its text stays off the
        // host's executable pages, since a 16 KiB native page can hold rtld's
        // code and its writable data together; nothing may enter it, and its
        // gate sites can never be reached.
        const auto host_executes = [&](std::size_t position) {
            return input.modules[(*order)[position]].name != "rtld";
        };
        std::vector<NsoSource> sources;
        std::vector<ModulePreparation> preparations;
        std::vector<ModuleContext> contexts(order->size());
        for (std::size_t position = 0; position < order->size(); ++position) {
            auto bytes = decoded[position].appendix;
            if (!native_options.artifacts && native_options.embedded_preparation && loading.nvn_module == position) bytes += 16;
            if (!native_options.artifacts && loading.compatibility_module == position) {
                bytes += 16 + loading.hook_gate.size() + loading.compatibility_patch->observers.size() * compat::observe_stub_bytes;
                for (const auto& call : loading.compatibility_patch->description.calls) bytes += compat::call_stub_bytes(call);
            }
            if (!native_options.artifacts && loading.launch_module == position) bytes += loading.launch_words.size() * 16;
            bytes = (bytes + *page_size - 1) & ~(*page_size - 1);
            contexts[position].loading = &loading;
            contexts[position].index = position;
            sources.push_back({input.modules[(*order)[position]].name, input.modules[(*order)[position]].bytes,
                               &images[position]});
            preparations.push_back({bytes, prepare_native_module, &contexts[position]});
            preparations.back().host_executes = host_executes(position);
            if (native_options.capture) capture.modules[position].host_executes = host_executes(position);
            if (native_options.embedded_preparation)
                preparations.back().appended_data_bytes = execution::embedded_key_page_size;
            if (native_options.artifacts) preparations.back().embedded_pieces = native_options.backings[position].pieces;
        }
        ModuleLoadOptions options;
        options.preparation = preparations;
        options.relocation_mode = ModuleRelocationMode::host_link;
        // The title's rtld binds a default-visibility reference to the first
        // module in load order that defines it (only other visibilities bind
        // to the referring module), so main's operator new preempts the SDK
        // module's weak one, as on the console.
        options.bindings.preemption = linker::PreemptionModel::load_order;
        options.parallel_preparation = true;
        auto loaded = ModuleSet::load_nso(sources, options);
        if (!loaded.ok()) {
            const bool allocation_failed = std::any_of(contexts.begin(), contexts.end(),
                [](const ModuleContext& context) { return context.allocation_failed; });
            auto failure = fail(allocation_failed || loaded.error == ModuleLoadError::resource_limit
                ? LaunchError::resource_limit : LaunchError::host_failure, "load-native", loaded.message);
            for (const auto& context : contexts) {
                if (context.error.empty()) continue;
                failure.detail += " " + context.error;
                break;
            }
            return failure;
        }
        for (std::size_t position = 0; position < order->size(); ++position)
            if (!host_executes(position))
                loading.modules[position].prepared_end = loading.modules[position].prepared_begin;
        if (!native::foreign_entries_prepared(loading.modules))
            return fail(LaunchError::host_failure, "load-native", "A linked entry has no prepared executable owner.");
        // The host address of each module lets a crash report's guest pc be
        // mapped back to a module offset.
        for (std::size_t position = 0; position < order->size(); ++position) {
            const auto& module = loading.modules[position];
            const auto& mapping = loaded.modules->mapping(position);
            armigo::diag::write(armigo::diag::Level::info, 
                         "Armigo preparation: %s base=0x%llx size=0x%zx gate_sites=%zu thread_register_sites=%zu "
                         "x18_sites=%zu stops=%zu bytes=%zu\n",
                         input.modules[(*order)[position]].name.c_str(),
                         static_cast<unsigned long long>(mapping.base_address()), mapping.mapping_size(),
                         module.sites.size(), module.thread_register_sites, module.scalar_sites, module.stops.size(),
                         module.code_bytes);
        }
        std::vector<execution::GuestGateSite> sites;
        for (std::size_t position = 0; position < loading.modules.size(); ++position)
            if (host_executes(position))
                sites.insert(sites.end(), loading.modules[position].sites.begin(), loading.modules[position].sites.end());
        for (const auto& context : contexts) for (const auto& launch : context.launch_sites) {
            const auto found = std::find_if(sites.begin(), sites.end(),
                [&](const auto& site) { return site.return_address == launch.return_address; });
            if (found == sites.end() || found->instruction != launch.expected_instruction)
                return fail(LaunchError::artifact_mismatch, "load-native", "A launch-time site is missing from the prepared code.");
            found->instruction = launch.replacement_instruction;
        }
        auto table = execution::GuestSiteTable::build(sites, {.move_wide = !loading.launch_words.empty()});
        if (!table.ok()) return fail(LaunchError::host_failure, "load-native", execution::describe(table.error.kind));

        PreparedLaunch result;
        result.title_frame_rate = admitted_frame_rate;
        result.modules = std::make_shared<ModuleSet>(std::move(*loaded.modules));
        const auto& modules = *result.modules;
        // The SDK's own startup sequence, in the order its loader runs it.
        const auto call = [&](std::string_view name, StartupCall::Argument argument = StartupCall::Argument::none,
                              std::uint64_t x0 = 0) -> bool {
            const auto found = defined_function(modules, name);
            if (!found) return false;
            result.startup_calls.push_back({std::string(name), found->second, x0, argument});
            return true;
        };
        if (!call("_init_libc0") || !call("nnosInitialize", StartupCall::Argument::main_thread_handle) ||
            !call("_init_libc1") || !call("nninitInitializeSdkModule") || !call("nninitStartup") ||
            !call("_init_libc2"))
            return fail(LaunchError::unsupported, "load-native", "SDK startup functions are missing or ambiguous.");
        for (std::size_t position = 0; position < order->size(); ++position) {
            if (input.modules[(*order)[position]].name == "rtld") continue;
            const auto offset = modules.metadata(position).init;
            if (!offset) continue;
            const auto address = modules.mapping(position).executable_address(offset);
            if (!address || offset % 4)
                return fail(LaunchError::invalid_input, "load-native", "DT_INIT is outside executable code.");
            result.startup_calls.push_back({"DT_INIT:" + input.modules[(*order)[position]].name, *address});
        }
        if (!call("_ZN2nn2os11SetHostArgcEi", StartupCall::Argument::none, 1) ||
            !call("_ZN2nn2os11SetHostArgvEPPc", StartupCall::Argument::host_arguments) || !call("nnMain"))
            return fail(LaunchError::unsupported, "load-native", "SDK main entry is missing or ambiguous.");

        if (loading.compatibility_patch) {
            auto hooks = execution::CallHookTable::build(loading.call_hooks);
            if (!hooks.ok()) return fail(LaunchError::artifact_mismatch, "compatibility-hooks", execution::describe(hooks.error.kind));
            result.call_hooks = std::move(hooks.table);
            result.compatibility_patch = std::move(loading.compatibility_patch);
            armigo::diag::write(armigo::diag::Level::info,  "Armigo compatibility: installed %s (%zu call hooks).\n",
                result.compatibility_patch->description.module.c_str(), loading.call_hooks.size());
        }
        result.sites = std::move(table.table);
        result.thread_slot = std::move(thread_slot);
        result.plan.module_order = *order;
        result.plan.entry_abi = LaunchEntryAbi::host_startup;
        result.plan.relocation_owner = LaunchOwnership::host_loader;
        result.plan.initialization_owner = LaunchOwnership::host_loader;
        result.entry_address = result.startup_calls.front().function;
        result.startup_user = title_startup_user;
        if (nvn_version) {
            result.nvn_driver = true;
            result.nvn_pso_manifest = title_pso_manifest;
            nvn::register_null_entries();
            if (nvn_version->major == 231) nvn::v231::register_entries();
            nvn::select_api_version(*nvn_version);
            nvn::set_title_policy(title_nvn_policy);
            armigo::diag::write(armigo::diag::Level::info,  "Armigo NVN: driver selected for API %d.%d.\n", nvn_version->major, nvn_version->minor);
        }
        if (native_options.capture) *native_options.capture = std::move(capture);
        if (rate_patch && requested_title_rate == "variable")
            armigo::diag::warning("%s\n", compat::variable_frame_rate_warning);
        return result;
    } catch (const std::bad_alloc&) {
        return fail(LaunchError::resource_limit, "load-native", "Native launch allocation failed.");
    } catch (const std::exception& error) {
        return fail(LaunchError::host_failure, "load-native", error.what());
    }
}

} // namespace armigo::runtime
