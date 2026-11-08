// ============================================================================
// Armigo Runtime - Native Module Runtime
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include <armigo/common/diagnostics.h>
#include <armigo/runtime/launch_plan.h>
#include <armigo/aarch64/instruction.h>
#include <armigo/compat/title_plan.h>
#include <armigo/execution/gate_code.h>

#include "prepared_artifact.h"
#include "../nvn/entry_table_231.h"
#include "../nvn/redirect.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <cstdlib>
#include <cstdio>
#include <limits>
#include <map>
#include <new>
#include <optional>
#include <stdexcept>

namespace armigo::runtime {
namespace {
using preparation::ModulePayload;

LaunchFailure fail(LaunchError code, std::string operation, std::string detail) {
    return preparation::failure(code, std::move(operation), std::move(detail));
}

LaunchResult<LaunchPlan> derive_plan(const loader::ExeFsDirectoryInput& input,
    const LaunchProfile& profile,
    const preparation::DecodedInput& decoded, std::span<const ModulePayload> payloads) {
    if (!profile.is_64_bit)
        return fail(LaunchError::unsupported, "plan-launch",
            "Only 64-bit launch profiles are supported.");
    LaunchPlan plan;
    for (std::size_t i = 0; i < decoded.images.size(); ++i) {
        const auto& payload = payloads[i];
        const auto& text = decoded.images[i].segment(loader::NsoSegmentKind::text);
        if (payload.original_process != decoded.original_process ||
            payload.image.original_image_size != decoded.images[i].bytes.size() ||
            payload.text_offset != text.image_offset || payload.text_size != text.content_size ||
            payload.load_rank >= decoded.order.size() || decoded.order[payload.load_rank] != i)
            return fail(LaunchError::artifact_mismatch, "plan-launch", "Prepared module layout or link order differs from executable metadata.");
    }
    if (profile.origin == ProfileOrigin::npdm) {
        if (!input.process_metadata)
            return fail(LaunchError::unsupported, "plan-launch",
                "Original process planning requires NPDM-backed executable input.");
        const auto order = preparation::original_module_order(input);
        if (!order)
            return fail(LaunchError::unsupported, "plan-launch",
                "Original process modules do not match the fixed executable slot roster.");
        for (const auto index : *order) {
            if (payloads[index].text_offset != 0)
                return fail(LaunchError::unsupported, "plan-launch",
                    "Original process modules require text at image offset zero.");
        }
        const auto entry_index = order->front();
        const auto& payload = payloads[entry_index];
        if (payload.text_offset != 0 || payload.text_size < 4 ||
            !std::binary_search(payload.roots.begin(), payload.roots.end(), 0) ||
            !preparation::covers(payload.image, 0))
            return fail(LaunchError::unsupported, "plan-launch",
                "Original process entry at image offset zero lacks prepared root coverage.");
        plan.module_order = *order;
        plan.entry = {entry_index, 0};
        plan.entry_abi = LaunchEntryAbi::original_process;
        plan.relocation_owner = LaunchOwnership::guest_bootstrap;
        plan.initialization_owner = LaunchOwnership::guest_bootstrap;
        return plan;
    }
    if (profile.origin != ProfileOrigin::synthetic_test)
        return fail(LaunchError::unsupported, "plan-launch", "Launch profile origin is unsupported.");
    plan.module_order = decoded.order;
    plan.entry_abi = LaunchEntryAbi::synthetic_exact_sp;
    plan.relocation_owner = LaunchOwnership::host_loader;
    plan.initialization_owner = LaunchOwnership::guest_bootstrap;
    bool entry_found = false;
    for (std::size_t i = 0; i < decoded.images.size(); ++i) {
        const auto& payload = payloads[i];
        for (const auto& symbol : decoded.metadata[i].symbols) {
            if (symbol.name != "_start" || !symbol.defined()) continue;
            if (entry_found || (symbol.info & 15u) != 2u || symbol.section >= 0xff00 ||
                !std::binary_search(payload.roots.begin(), payload.roots.end(), symbol.value) ||
                !preparation::covers(payload.image, symbol.value))
                return fail(LaunchError::unsupported, "plan-launch", "Synthetic _start is ambiguous or lacks prepared function coverage.");
            entry_found = true; plan.entry = {i, symbol.value};
        }
    }
    if (!entry_found) return fail(LaunchError::unsupported, "plan-launch", "Synthetic input requires one defined _start function.");
    return plan;
}

bool same_plan(const LaunchPlan& left, const LaunchPlan& right) {
    return left.module_order == right.module_order &&
        left.entry.module_index == right.entry.module_index &&
        left.entry.image_offset == right.entry.image_offset &&
        left.entry_abi == right.entry_abi && left.relocation_owner == right.relocation_owner &&
        left.initialization_owner == right.initialization_owner;
}

std::optional<std::size_t> appendix_mapping_size(std::size_t bytes, std::size_t page) {
    if (bytes > std::numeric_limits<std::size_t>::max() - (page - 1)) return {};
    return (bytes + page - 1) & ~(page - 1);
}

std::optional<std::size_t> landing_bytes(std::size_t roots) {
    if (roots > std::numeric_limits<std::size_t>::max() / 8) return {};
    return roots * 8;
}

struct Loading {
    std::span<const ModulePayload> payloads;
    std::span<const std::size_t> order;
    std::vector<std::uint64_t> manifest_bases;
    std::vector<execution::GuestGateSite> sites;
    std::vector<bool> prepared;
    std::uint32_t slot = 0;
    std::size_t page_size = 0;
    bool original_process = false;
    std::vector<PreparedEntryLanding> entry_landings;
    execution::PreparedImageError image_error = execution::PreparedImageError::none;
    bool allocation_failed = false;
    std::optional<std::size_t> compatibility_module;
    std::shared_ptr<compat::InstalledPatch> compatibility_patch;
    std::vector<std::byte> hook_gate;
    std::vector<execution::CallHookSite> call_hooks;
    std::string compatibility_error;
    std::optional<std::size_t> nvn_module;
    std::optional<nvn::ApiVersion> nvn_version;
    // Instruction words of the main module replaced by the title's patch.
    std::optional<std::size_t> word_module;
    std::vector<compat::WordPatch> words;
    std::optional<compat::CallPatch> gpu_budget_hook;
};

std::size_t compatibility_bytes(const Loading& loading, std::size_t index,
                                std::size_t preceding_bytes) {
    if (loading.compatibility_module != index) return 0;
    return (8 - preceding_bytes % 8) % 8 + loading.hook_gate.size() +
        loading.compatibility_patch->description.calls.size() * 16;
}

bool prepare_loaded_module(const ModulePreparationView& view, void* raw) noexcept {
    auto& loading = *static_cast<Loading*>(raw);
    try {
        if (view.module_index >= loading.order.size() || view.module_bases.size() != loading.order.size()) return false;
        const auto manifest_index = loading.order[view.module_index];
        if (manifest_index >= loading.payloads.size() || loading.prepared[manifest_index]) return false;
        for (std::size_t i = 0; i < loading.order.size(); ++i)
            loading.manifest_bases[loading.order[i]] = view.module_bases[i];
        const auto& image = loading.payloads[manifest_index].image;
        const auto veneers = landing_bytes(
            loading.original_process ? loading.payloads[manifest_index].roots.size() : 0);
        if (!veneers || image.appendix.size() >
                std::numeric_limits<std::size_t>::max() - *veneers) return false;
        const auto preceding_bytes = image.appendix.size() + *veneers;
        const auto requested_bytes = preceding_bytes + compatibility_bytes(loading, manifest_index, preceding_bytes);
        const auto appendix_bytes = appendix_mapping_size(requested_bytes, loading.page_size);
        if (view.base_address != loading.manifest_bases[manifest_index] ||
            view.image.size() != image.original_image_size ||
            view.appended_code_offset != (requested_bytes == 0 ? 0 : image.appendix_offset) ||
            !appendix_bytes || view.appended_code.size() != *appendix_bytes ||
            (requested_bytes != 0 && view.appended_code.data() != view.image.data() + image.appendix_offset) ||
            image.image_size() > std::numeric_limits<std::size_t>::max()) return false;
        const execution::PreparedImageBindings bindings{loading.manifest_bases, loading.slot,
            execution::gate_dispatcher_address()};
        // ModuleSet owns a single contiguous writable allocation spanning the
        // original image, alignment gap and appendix until every callback ends.
        // Apply only the artifact payload; the declared RX tail stays zero-filled
        // and never contributes roots, patches, fixups or gate sites.
        const auto applied = execution::apply_prepared_image(image,
            {view.image.data(), static_cast<std::size_t>(image.image_size())}, bindings);
        if (!applied.ok()) { loading.image_error = applied.error; return false; }
        if (loading.original_process) {
            const auto& roots = loading.payloads[manifest_index].roots;
            if (!std::is_sorted(roots.begin(), roots.end()) ||
                std::adjacent_find(roots.begin(), roots.end()) != roots.end() ||
                image.appendix.size() % 4 ||
                view.base_address > std::numeric_limits<std::uint64_t>::max() -
                    view.appended_code_offset)
                return false;
            const auto appendix_address = std::uint64_t{view.base_address} +
                view.appended_code_offset;
            constexpr std::uint32_t clear_x16 = 0xaa1f03f0u;
            for (std::size_t index = 0; index < roots.size(); ++index) {
                const auto root_offset = roots[index];
                if ((root_offset & 3) != 0 || !preparation::covers(image, root_offset) ||
                    root_offset > std::numeric_limits<std::uint64_t>::max() - view.base_address ||
                    index > (std::numeric_limits<std::size_t>::max() - image.appendix.size()) / 8)
                    return false;
                const auto stub_offset = image.appendix.size() + index * 8;
                if (stub_offset > view.appended_code.size() ||
                    view.appended_code.size() - stub_offset < 8 ||
                    stub_offset > std::numeric_limits<std::uint64_t>::max() - appendix_address)
                    return false;
                const auto landing = appendix_address + stub_offset;
                const auto root = std::uint64_t{view.base_address} + root_offset;
                if (landing > std::numeric_limits<std::uint64_t>::max() - 4) return false;
                const auto branch = aarch64::encode_b(landing + 4, root);
                if (!branch.ok()) return false;
                std::memcpy(view.appended_code.data() + stub_offset, &clear_x16, 4);
                std::memcpy(view.appended_code.data() + stub_offset + 4, &branch.word, 4);
                loading.entry_landings.push_back({root, landing});
            }
        }
        for (const auto& site : image.gate_sites) {
            if (site.return_address_offset > std::numeric_limits<std::uint64_t>::max() - view.base_address) return false;
            execution::GuestGateSite absolute{view.base_address + site.return_address_offset, site.instruction, {}};
            for (const auto target : site.allowed_indirect_targets) {
                if (target.module_index >= loading.manifest_bases.size() ||
                    target.image_offset > std::numeric_limits<std::uint64_t>::max() - loading.manifest_bases[target.module_index]) return false;
                absolute.allowed_indirect_targets.push_back(loading.manifest_bases[target.module_index] + target.image_offset);
            }
            loading.sites.push_back(std::move(absolute));
        }
        if (loading.compatibility_module == manifest_index) {
            const auto& description = loading.compatibility_patch->description;
            const auto gate_offset = (preceding_bytes + 7) & ~std::size_t{7};
            const auto gate_address = view.base_address + view.appended_code_offset + gate_offset;
            std::vector<execution::CallHookPatch> planned;
            for (std::size_t i = 0; i < description.calls.size(); ++i) {
                const auto& site = description.calls[i];
                std::uint32_t actual = 0;
                std::memcpy(&actual, view.image.data() + site.offset, 4);
                if (actual != site.expected_instruction) {
                    loading.compatibility_error = "Compatibility instruction changed during preparation.";
                    return false;
                }
                if (site.frame_ratio) {
                    loading.compatibility_error = "Prepared launches do not place frame-ratio hooks; prepare at load time.";
                    return false;
                }
                const auto stub = gate_address + loading.hook_gate.size() + i * 16;
                auto hook = compat::make_patch_hook(site, actual, view.base_address + site.offset,
                    stub, {gate_address, loading.hook_gate.size()});
                if (!hook.ok()) {
                    loading.compatibility_error = execution::describe(hook.error);
                    return false;
                }
                planned.push_back(*hook.patch);
            }
            std::memcpy(view.appended_code.data() + gate_offset, loading.hook_gate.data(), loading.hook_gate.size());
            for (std::size_t i = 0; i < planned.size(); ++i) {
                const auto& hook = planned[i];
                std::memcpy(view.appended_code.data() + gate_offset + loading.hook_gate.size() + i * 16,
                            hook.trampoline.data(), 16);
                std::memcpy(view.image.data() + description.calls[i].offset, &hook.replacement_instruction, 4);
                loading.call_hooks.push_back(hook.site);
            }
            loading.compatibility_patch->module_base = view.base_address;
        }
        if (loading.word_module == manifest_index) {
            for (const auto& word : loading.words) {
                std::uint32_t actual = 0;
                std::memcpy(&actual, view.image.data() + word.offset, 4);
                if (actual != word.expected_instruction) {
                    loading.compatibility_error = "Compatibility instruction word changed during preparation.";
                    return false;
                }
                std::memcpy(view.image.data() + word.offset, &word.replacement_instruction, 4);
            }
        }
        if (loading.nvn_module == manifest_index) {
            const auto& payload = loading.payloads[manifest_index];
            const auto redirected = nvn::redirect_bootstrap_loader(view, image,
                payload.text_offset, payload.text_size, std::nullopt,
                nvn::bootstrap_loader_for(*loading.nvn_version));
            if (!redirected.ok()) {
                loading.compatibility_error = redirected.error;
                return false;
            }
        }
        loading.prepared[manifest_index] = true;
        return true;
    } catch (const std::bad_alloc&) { loading.allocation_failed = true; return false; }
      catch (...) { return false; }
}
} // namespace

LaunchResult<LaunchPlan> plan_launch(const loader::ExeFsDirectoryInput& input,
    const LaunchProfile& profile, const PreparedModuleArtifacts& artifacts) {
    try {
        if (profile.origin == ProfileOrigin::npdm) {
            if (auto admitted = preparation::admitted_input(input, PreparationMode::original_process))
                return plan_launch(*admitted, profile, artifacts);
        }
        const auto mode = profile.origin == ProfileOrigin::synthetic_test
            ? PreparationMode::synthetic_test : PreparationMode::original_process;
        auto validated = preparation::validate_artifacts(input, artifacts, {.mode = mode});
        if (const auto* failed = std::get_if<LaunchFailure>(&validated)) return *failed;
        auto decoded = preparation::decode_input(input, mode);
        if (const auto* failed = std::get_if<LaunchFailure>(&decoded)) return *failed;
        return derive_plan(input, profile, std::get<preparation::DecodedInput>(decoded),
            std::get<std::vector<ModulePayload>>(validated));
    } catch (const std::bad_alloc&) {
        return fail(LaunchError::resource_limit, "plan-launch", "Launch planning allocation failed.");
    } catch (const std::exception& error) {
        return fail(LaunchError::host_failure, "plan-launch", error.what());
    }
}

LaunchResult<PreparedLaunch> load_prepared_launch(
    const loader::ExeFsDirectoryInput& input, const PreparedModuleArtifacts& artifacts,
    const LaunchProfile& profile, const LaunchPlan& plan,
    std::shared_ptr<platform::ThreadSlotKey> thread_slot, const ModuleLoadOptions& options,
    std::span<const compat::PatchDescription> patches) {
    try {
        if (profile.origin == ProfileOrigin::npdm) {
            if (auto admitted = preparation::admitted_input(input, PreparationMode::original_process))
                return load_prepared_launch(*admitted, artifacts, profile, plan, thread_slot, options, patches);
        }
        if (!thread_slot || !thread_slot->valid() || !thread_slot->native_access())
            return fail(LaunchError::invalid_input, "load-prepared", "A valid probed native thread-slot key is required.");
        if (!options.preparation.empty() || !options.bindings.forced.empty() || !options.bindings.fallback.empty() ||
            options.bindings.preemption != linker::PreemptionModel::own_definition_first)
            return fail(LaunchError::unsupported, "load-prepared", "Custom preparation or symbol bindings differ from the offline link contract.");
        const auto mode = profile.origin == ProfileOrigin::synthetic_test
            ? PreparationMode::synthetic_test : PreparationMode::original_process;
        auto validated = preparation::validate_artifacts(input, artifacts, {.mode = mode});
        if (const auto* failed = std::get_if<LaunchFailure>(&validated)) return *failed;
        auto payloads = std::get<std::vector<ModulePayload>>(std::move(validated));
        auto decoded = preparation::decode_input(input, mode);
        if (const auto* failed = std::get_if<LaunchFailure>(&decoded)) return *failed;
        if (plan.module_order.size() != payloads.size() || plan.entry.module_index >= payloads.size() ||
            plan.entry.image_offset % 4 ||
            !preparation::covers(payloads[plan.entry.module_index].image, plan.entry.image_offset))
            return fail(LaunchError::invalid_input, "load-prepared", "Launch plan has an invalid module permutation or entry location.");
        std::vector<bool> used(payloads.size());
        for (const auto index : plan.module_order) {
            if (index >= used.size() || used[index])
                return fail(LaunchError::invalid_input, "load-prepared", "Launch module order must be an exact manifest permutation.");
            used[index] = true;
        }
        auto expected = derive_plan(input, profile, std::get<preparation::DecodedInput>(decoded), payloads);
        if (const auto* failed = std::get_if<LaunchFailure>(&expected)) return *failed;
        if (!same_plan(plan, std::get<LaunchPlan>(expected)))
            return fail(LaunchError::unsupported, "load-prepared", "Launch order, entry ABI or relocation/initialization ownership differs from the admitted plan.");
        if (plan.relocation_owner == LaunchOwnership::host_loader &&
            options.relocation_mode != ModuleRelocationMode::host_link)
            return fail(LaunchError::unsupported, "load-prepared",
                "Synthetic loading cannot override the plan's host relocation ownership.");

        const auto page_size = platform::HostModule::native_page_size();
        if (!page_size)
            return fail(LaunchError::host_failure, "load-prepared", "The native module page size is unavailable.");
        Loading loading{payloads, plan.module_order,
            std::vector<std::uint64_t>(payloads.size()), {}, std::vector<bool>(payloads.size()),
            thread_slot->native_access()->tsd_byte_offset, *page_size,
            plan.entry_abi == LaunchEntryAbi::original_process, {},
            execution::PreparedImageError::none, false, {}, {}, {}, {}, {}, {}, {}};
        bool title_nvn_enabled = false;
        bool title_matched = false;
        const compat::PatchDescription* rate_patch = nullptr;
        std::string requested_title_rate;
        if (!patches.empty()) {
            const auto main = std::find_if(input.modules.begin(), input.modules.end(),
                [](const auto& module) { return module.name == "main"; });
            if (main != input.modules.end()) {
                const auto index = static_cast<std::size_t>(main - input.modules.begin());
                const auto& source = std::get<preparation::DecodedInput>(decoded).images[index];
                const auto* patch = compat::match_patch(patches, {profile.title_id, source.build_id});
                if (patch) {
                    title_matched = true;
                    title_nvn_enabled = patch->nvn_driver;
                    if (auto error = compat::validate_patch(*patch, source); !error.empty())
                        return fail(LaunchError::artifact_mismatch, "compatibility-preflight", std::move(error));
                    const char* requested_rate = std::getenv("ARMIGO_COMPAT_FRAME_RATE");
                    if (patch->frame_rate) {
                        rate_patch = patch;
                        requested_title_rate = requested_rate ? requested_rate : "";
                    }
                    auto planned = compat::plan_title_words(*patch, source,
                        {requested_rate ? requested_rate : "", false});
                    if (const auto* failure = std::get_if<compat::TitlePlanFailure>(&planned))
                        return fail(failure->kind == compat::TitlePlanFailure::Kind::invalid_input
                            ? LaunchError::invalid_input : LaunchError::artifact_mismatch, failure->stage, failure->detail);
                    auto& plan = std::get<compat::TitlePlan>(planned);
                    auto words = std::move(plan.words);
                    if (!plan.module_calls.empty()) {
                        loading.gpu_budget_hook = plan.module_calls.front();
                        if (patch->module != "frame-rate")
                            return fail(LaunchError::unsupported, "compatibility-resolution",
                                        "GPU-work feedback requires the title frame-rate module.");
                    }
                    if (!words.empty()) {
                        const auto& prepared = payloads[index].image;
                        for (const auto& word : words) {
                            if (!preparation::covers(prepared, word.offset) ||
                                std::any_of(prepared.replacements.begin(), prepared.replacements.end(),
                                    [&](const auto& item) { return item.image_offset == word.offset; }) ||
                                std::any_of(prepared.fixups.begin(), prepared.fixups.end(),
                                    [&](const auto& item) { return item.image_offset == word.offset; }))
                                return fail(LaunchError::artifact_mismatch, "compatibility-preflight",
                                    "Compatibility instruction word must be covered and unchanged by offline preparation.");
                        }
                        loading.word_module = index;
                        loading.words = std::move(words);
                    }
                    if (!patch->module.empty()) {
                        loading.compatibility_module = index;
                        loading.compatibility_patch = std::make_shared<compat::InstalledPatch>(compat::InstalledPatch{*patch, 0});
                        if (loading.gpu_budget_hook)
                            loading.compatibility_patch->description.calls.push_back(*loading.gpu_budget_hook);
                        if (auto error = compat::validate_patch(loading.compatibility_patch->description, source); !error.empty())
                            return fail(LaunchError::artifact_mismatch, "compatibility-preflight", std::move(error));
                        loading.hook_gate = execution::make_local_gate_code();
                    }
                }
            }
        }
        if (!patches.empty() && !title_matched)
            armigo::diag::write(armigo::diag::Level::info,  "Armigo: the title settings describe no build matching this main module; running without them.\n");
        const char* override_environment = std::getenv("ARMIGO_NVN");
        const std::string_view override_value = override_environment ? override_environment : "";
        // Validate overrides even if the module roster has no SDK slot.
        if (const auto selection = nvn::select_driver(false, {}, override_value); selection.error)
            return fail(LaunchError::invalid_input, "nvn-selection", selection.error);
        for (std::size_t index = 0; index < input.modules.size(); ++index) {
            if (input.modules[index].name != "sdk") continue;
            const auto selection = nvn::select_driver(title_nvn_enabled,
                std::get<preparation::DecodedInput>(decoded).images[index].bytes, override_value);
            if (selection.version) {
                if (loading.nvn_module)
                    return fail(LaunchError::invalid_input, "nvn-selection", "SDK module selection is ambiguous.");
                loading.nvn_module = index;
                loading.nvn_version = selection.version;
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
                const auto& prepared = payloads[*loading.compatibility_module].image;
                if (!preparation::covers(prepared, description.original_callee))
                    return fail(LaunchError::artifact_mismatch, "compatibility-preflight",
                                "Compatibility callee lacks prepared instruction coverage.");
                for (const auto& call : description.calls) {
                    if (!preparation::covers(prepared, compat::call_callee(description, call)))
                        return fail(LaunchError::artifact_mismatch, "compatibility-preflight",
                                    "Compatibility call callee lacks prepared instruction coverage.");
                    if (!preparation::covers(prepared, call.offset) ||
                        std::any_of(prepared.replacements.begin(), prepared.replacements.end(),
                            [&](const auto& item) { return item.image_offset == call.offset; }) ||
                        std::any_of(prepared.fixups.begin(), prepared.fixups.end(),
                            [&](const auto& item) { return item.image_offset == call.offset; }))
                        return fail(LaunchError::artifact_mismatch, "compatibility-preflight",
                            "Compatibility call must be covered and unchanged by offline preparation.");
                }
                compat::set_present_link(description.present_link);
                compat::initialize_gpu_budget(*loading.compatibility_patch);
            }
        }
        if (!loading.nvn_module)
            std::erase_if(loading.words, [](const compat::WordPatch& word) { return word.nvn_driver_only; });
        if (loading.words.empty()) loading.word_module.reset();
        std::vector<NsoSource> sources;
        std::vector<ModulePreparation> preparations;
        for (const auto index : plan.module_order) {
            const auto veneers = landing_bytes(plan.entry_abi == LaunchEntryAbi::original_process
                ? payloads[index].roots.size() : 0);
            if (!veneers || payloads[index].image.appendix.size() >
                    std::numeric_limits<std::size_t>::max() - *veneers)
                return fail(LaunchError::resource_limit, "load-prepared", "Original entry landing size overflows.");
            const auto preceding_bytes = payloads[index].image.appendix.size() + *veneers;
            const auto extra = compatibility_bytes(loading, index, preceding_bytes);
            if (extra > std::numeric_limits<std::size_t>::max() - preceding_bytes)
                return fail(LaunchError::resource_limit, "compatibility-layout", "Compatibility appendix size overflows.");
            const auto requested_bytes = preceding_bytes + extra;
            const auto appendix_bytes = appendix_mapping_size(requested_bytes, *page_size);
            if (!appendix_bytes)
                return fail(LaunchError::resource_limit, "load-prepared", "Prepared appendix page alignment overflows.");
            sources.push_back({input.modules[index].name, input.modules[index].bytes});
            preparations.push_back({*appendix_bytes, prepare_loaded_module, &loading,
                true, payloads[index].roots});
        }
        auto load_options = options;
        load_options.preparation = preparations;
        load_options.relocation_mode = plan.relocation_owner == LaunchOwnership::guest_bootstrap
            ? ModuleRelocationMode::guest_bootstrap : ModuleRelocationMode::host_link;
        auto loaded = ModuleSet::load_nso(sources, load_options);
        if (!loaded.ok()) {
            auto failure = fail(loading.allocation_failed ? LaunchError::resource_limit :
                loaded.error == ModuleLoadError::preparation_failed ? LaunchError::artifact_mismatch :
                loaded.error == ModuleLoadError::resource_limit ? LaunchError::resource_limit : LaunchError::host_failure,
                "load-prepared", loaded.message);
            if (loading.image_error != execution::PreparedImageError::none)
                failure.detail += " Prepared image status " + std::to_string(static_cast<unsigned>(loading.image_error)) + ".";
            if (!loading.compatibility_error.empty()) failure.detail += " " + loading.compatibility_error;
            failure.stage = LaunchStage::modules;
            return failure;
        }
        if (std::find(loading.prepared.begin(), loading.prepared.end(), false) != loading.prepared.end())
            return fail(LaunchError::host_failure, "load-prepared", "The module set was not prepared exactly once.");
        std::sort(loading.entry_landings.begin(), loading.entry_landings.end(),
            [](const auto& left, const auto& right) {
                return left.logical_root < right.logical_root;
            });
        if (plan.entry_abi == LaunchEntryAbi::original_process &&
            (loading.entry_landings.empty() ||
             std::adjacent_find(loading.entry_landings.begin(), loading.entry_landings.end(),
                 [](const auto& left, const auto& right) {
                     return left.logical_root == right.logical_root;
                 }) != loading.entry_landings.end()))
            return fail(LaunchError::host_failure, "load-prepared", "Original entry landings were not generated exactly once.");
        auto sites = execution::GuestSiteTable::build(loading.sites);
        if (!sites.ok()) return fail(LaunchError::artifact_mismatch, "load-prepared", execution::describe(sites.error.kind));
        const auto entry_index = static_cast<std::size_t>(std::find(plan.module_order.begin(), plan.module_order.end(),
            plan.entry.module_index) - plan.module_order.begin());
        const auto entry = loaded.modules->mapping(entry_index).executable_address(plan.entry.image_offset);
        if (!entry) return fail(LaunchError::artifact_mismatch, "load-prepared", "Prepared entry has no executable mapping.");
        PreparedLaunch result;
        result.title_frame_rate = admitted_frame_rate;
        result.modules = std::make_shared<ModuleSet>(std::move(*loaded.modules));
        if (loading.compatibility_patch) {
            auto hooks = execution::CallHookTable::build(loading.call_hooks);
            if (!hooks.ok()) return fail(LaunchError::artifact_mismatch, "compatibility-hooks", execution::describe(hooks.error.kind));
            result.call_hooks = std::move(hooks.table);
            result.compatibility_patch = std::move(loading.compatibility_patch);
            armigo::diag::write(armigo::diag::Level::info,  "Armigo compatibility: installed %s (%zu call hooks).\n",
                result.compatibility_patch->description.module.c_str(), loading.call_hooks.size());
        }
        result.sites = std::move(sites.table); result.thread_slot = std::move(thread_slot);
        result.plan = plan; result.entry_address = *entry;
        result.original_entry_landings = std::move(loading.entry_landings);
        if (plan.entry_abi == LaunchEntryAbi::original_process) {
            const auto found = std::lower_bound(result.original_entry_landings.begin(),
                result.original_entry_landings.end(), result.entry_address,
                [](const auto& candidate, std::uint64_t root) {
                    return candidate.logical_root < root;
                });
            if (found == result.original_entry_landings.end() ||
                found->logical_root != result.entry_address)
                return fail(LaunchError::host_failure, "load-prepared", "The process entry landing is missing.");
            result.process_entry_address = found->landing;
        }
        if (loading.nvn_version) {
            nvn::register_null_entries();
            if (loading.nvn_version->major == 231) nvn::v231::register_entries();
            nvn::select_api_version(*loading.nvn_version);
            armigo::diag::write(armigo::diag::Level::info,  "Armigo NVN: null device selected for API %d.%d.\n",
                loading.nvn_version->major, loading.nvn_version->minor);
        }
        // Synthetic loading applies one host LinkPlan. Original loading leaves
        // relocation targets untouched for the guest bootstrap. Neither path
        // invokes initializers.
        if (rate_patch && requested_title_rate == "variable")
            armigo::diag::warning("%s\n", compat::variable_frame_rate_warning);
        return result;
    } catch (const std::bad_alloc&) {
        return fail(LaunchError::resource_limit, "load-prepared", "Prepared launch allocation failed.");
    } catch (const std::exception& error) {
        return fail(LaunchError::host_failure, "load-prepared", error.what());
    }
}

} // namespace armigo::runtime
