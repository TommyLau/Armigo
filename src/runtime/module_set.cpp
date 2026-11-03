// ============================================================================
// Armigo Runtime - Native Module Runtime
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include <armigo/runtime/module_set.h>

#include <algorithm>
#include <limits>
#include <new>
#include <stdexcept>
#include <system_error>
#include <thread>
#include <utility>

namespace armigo::runtime {
namespace {

ModuleSetResult failure(ModuleLoadError error, std::optional<std::size_t> index,
                        std::string message) {
    ModuleSetResult result;
    result.error = error;
    result.module_index = index;
    result.message = std::move(message);
    return result;
}

memory::Permission permission(loader::SegmentPermission source) {
    const auto bits = static_cast<unsigned>(source);
    auto result = memory::Permission::none;
    if ((bits & 1u) != 0) result |= memory::Permission::read;
    if ((bits & 2u) != 0) result |= memory::Permission::write;
    if ((bits & 4u) != 0) result |= memory::Permission::execute;
    return result;
}

bool permits_guest_relocation(const loader::NsoSegment& data,
                              std::uint64_t target) noexcept {
    if (!memory::has_permission(permission(data.permission), memory::Permission::write) ||
        target < data.image_offset) {
        return false;
    }
    const auto relative = target - data.image_offset;
    return relative <= data.mapped_size &&
        data.mapped_size - relative >= sizeof(std::uint64_t);
}
} // namespace

ModuleSetResult ModuleSet::load_nso(std::span<const NsoSource> sources,
                                   const ModuleLoadOptions& options) {
    if (sources.empty() || sources.size() > options.link_limits.max_modules) {
        return failure(ModuleLoadError::invalid_sources, std::nullopt, "Invalid module source count.");
    }
    if (!options.preparation.empty() && options.preparation.size() != sources.size()) {
        return failure(ModuleLoadError::invalid_sources, std::nullopt,
                       "Preparation entries must match the module source count.");
    }
    if (options.relocation_mode != ModuleRelocationMode::host_link &&
        options.relocation_mode != ModuleRelocationMode::guest_bootstrap) {
        return failure(ModuleLoadError::invalid_sources, std::nullopt,
                       "Invalid module relocation mode.");
    }
    if (options.relocation_mode == ModuleRelocationMode::guest_bootstrap &&
        (!options.bindings.forced.empty() || !options.bindings.fallback.empty() ||
         options.bindings.preemption != linker::PreemptionModel::own_definition_first)) {
        return failure(ModuleLoadError::invalid_sources, std::nullopt,
                       "Guest bootstrap relocation does not accept host symbol bindings.");
    }
    for (std::size_t index = 0; index < options.preparation.size(); ++index) {
        const auto& requested = options.preparation[index];
        if ((requested.callback == nullptr && (requested.appended_code_bytes != 0 ||
             requested.appended_data_bytes != 0 || !requested.embedded_pieces.empty() ||
             requested.restrict_entry_points)) ||
            (!requested.restrict_entry_points && !requested.admitted_entry_offsets.empty())) {
            return failure(ModuleLoadError::invalid_sources, index,
                           "Appended code and restricted entries require a preparation callback.");
        }
    }
    const auto page_size = platform::HostModule::native_page_size();
    if (!page_size) {
        return failure(ModuleLoadError::mapping_failed, std::nullopt,
                       "The host page size is unavailable.");
    }
    try {
        ModuleSet set;
        set.modules_.reserve(sources.size());
        std::size_t total_bytes = 0;
        std::uintptr_t minimum_address = 0;
        for (std::size_t index = 0; index < sources.size(); ++index) {
            const auto& source = sources[index];
            if (source.name.empty() || source.name.size() > options.max_module_name_bytes ||
                source.name.find('\0') != std::string_view::npos) {
                return failure(ModuleLoadError::invalid_sources, index, "Invalid logical module name.");
            }
            for (const auto& loaded : set.modules_) {
                if (loaded.name == source.name) {
                    return failure(ModuleLoadError::invalid_sources, index, "Duplicate logical module name.");
                }
            }
            std::optional<loader::NsoModuleImage> image;
            const auto remaining = options.max_total_image_bytes - total_bytes;
            if (source.decoded) {
                if (source.decoded->bytes.size() > std::min<std::uint64_t>(options.reader.limits.max_image_bytes, remaining))
                    return failure(ModuleLoadError::resource_limit, index,
                                   loader::to_string(loader::NsoError::image_too_large));
                image = std::move(*source.decoded);
            } else {
                auto reader_options = options.reader;
                reader_options.limits.max_image_bytes =
                    std::min<std::uint64_t>(reader_options.limits.max_image_bytes, remaining);
                auto decoded = loader::read_nso_module(source.file, reader_options);
                if (!decoded.ok()) {
                    const bool limited = decoded.error.code == loader::NsoError::image_too_large ||
                                         decoded.error.code == loader::NsoError::input_too_large ||
                                         decoded.error.code == loader::NsoError::allocation_failed;
                    return failure(limited ? ModuleLoadError::resource_limit : ModuleLoadError::invalid_container,
                                   index, loader::to_string(decoded.error.code));
                }
                image = std::move(decoded.image);
            }
            if (!options.preparation.empty() && options.preparation[index].restrict_entry_points) {
                const auto entries = options.preparation[index].admitted_entry_offsets;
                const auto& text = image->segment(loader::NsoSegmentKind::text);
                for (std::size_t entry = 0; entry < entries.size(); ++entry) {
                    const auto offset = entries[entry];
                    if (offset % 4 || offset < text.image_offset || text.content_size < 4 ||
                        offset - text.image_offset > text.content_size - 4 ||
                        (entry && entries[entry - 1] >= offset))
                        return failure(ModuleLoadError::invalid_sources, index,
                                       "Prepared entry roots must be sorted unique text locations.");
                }
            }
            const auto original_size = image->bytes.size();
            if (original_size > std::numeric_limits<std::size_t>::max() - (*page_size - 1)) {
                return failure(ModuleLoadError::resource_limit, index, "Module size overflow.");
            }
            const auto original_mapping_size = (original_size + *page_size - 1) & ~(*page_size - 1);
            const auto extra_size = options.preparation.empty() ? 0 :
                                    options.preparation[index].appended_code_bytes;
            const auto data_size = options.preparation.empty() ? 0 :
                                   options.preparation[index].appended_data_bytes;
            if (data_size % *page_size != 0)
                return failure(ModuleLoadError::invalid_sources, index,
                               "Appended writable storage must cover whole native pages.");
            if ((extra_size != 0 || data_size != 0) &&
                (original_mapping_size > remaining || data_size > remaining - original_mapping_size ||
                 extra_size > remaining - original_mapping_size - data_size)) {
                return failure(ModuleLoadError::resource_limit, index,
                               "Appended preparation code exceeds the image budget.");
            }
            auto dynamic = linker::read_dynamic_module(image->bytes, options.dynamic_limits);
            if (!dynamic.ok()) {
                return failure(ModuleLoadError::invalid_dynamic_metadata, index, std::move(dynamic.message));
            }
            if (options.relocation_mode == ModuleRelocationMode::guest_bootstrap) {
                const auto& data = image->segment(loader::NsoSegmentKind::data);
                for (const auto& relocation : dynamic.module->relocations) {
                    if (!permits_guest_relocation(data, relocation.target)) {
                        return failure(ModuleLoadError::invalid_dynamic_metadata, index,
                            "Guest RELA target " + std::to_string(relocation.target) +
                            " is outside the original writable data segment.");
                    }
                }
                for (const auto target : dynamic.module->packed_relative_targets) {
                    if (!permits_guest_relocation(data, target)) {
                        return failure(ModuleLoadError::invalid_dynamic_metadata, index,
                            "Guest RELR target " + std::to_string(target) +
                            " is outside the original writable data segment.");
                    }
                }
            }
            std::vector<platform::ModuleRegion> regions;
            std::vector<linker::ModuleRange> relocatable;
            for (const auto& segment : image->segments) {
                if (segment.mapped_size == 0) continue;
                regions.push_back({static_cast<std::size_t>(segment.image_offset),
                                   static_cast<std::size_t>(segment.mapped_size),
                                   permission(segment.permission)});
                // Read-only data may contain relocations, but executable
                // instructions may not.
                if (segment.kind != loader::NsoSegmentKind::text) {
                    relocatable.push_back({segment.image_offset, segment.mapped_size});
                }
            }
            if (data_size != 0) {
                image->bytes.resize(original_mapping_size + data_size);
                regions.push_back({original_mapping_size, data_size,
                                   memory::Permission::read | memory::Permission::write});
            }
            if (extra_size != 0) {
                // Keep the extra RX pages disjoint from the original data
                // pages. Appending never shifts an existing module offset.
                image->bytes.resize(original_mapping_size + data_size + extra_size);
                regions.push_back({original_mapping_size + data_size, extra_size,
                                   memory::Permission::read | memory::Permission::execute});
            }
            total_bytes += image->bytes.size();
            platform::HostMapLimits map_limits;
            map_limits.max_image_bytes = image->bytes.size();
            map_limits.host_executes = options.preparation.empty() || options.preparation[index].host_executes;
            if (options.relocation_mode == ModuleRelocationMode::guest_bootstrap) {
                map_limits.minimum_address = minimum_address;
            }
            const auto pieces = options.preparation.empty() ? std::span<const platform::EmbeddedModulePiece>{}
                : options.preparation[index].embedded_pieces;
            auto mapped = pieces.empty() ? platform::HostModule::create(image->bytes, regions, map_limits)
                : platform::HostModule::create_embedded(image->bytes.size(), regions, pieces, map_limits);
            if (!mapped.ok()) {
                return failure(ModuleLoadError::mapping_failed, index,
                               "Host mapping failed with status " +
                                   std::to_string(static_cast<unsigned>(mapped.status.error)) +
                                   " and system error " + std::to_string(mapped.status.system_error) + ".");
            }
            if (!pieces.empty()) {
                // A non-text relocation can still share an RX host page with
                // text. Exclude that page before planning any data writes.
                std::vector<linker::ModuleRange> writable;
                for (const auto& range : relocatable) for (const auto& page : mapped.module->host_regions()) {
                    if (memory::has_permission(page.permissions, memory::Permission::execute)) continue;
                    const auto begin = std::max<std::uint64_t>(range.offset, page.offset);
                    const auto end = std::min<std::uint64_t>(range.offset + range.size, page.offset + page.size);
                    if (begin < end) writable.push_back({begin, end - begin});
                }
                relocatable = std::move(writable);
            }
            if (options.relocation_mode == ModuleRelocationMode::guest_bootstrap) {
                minimum_address = mapped.module->base_address() + mapped.module->mapping_size();
            }
            set.modules_.push_back({std::string(source.name), std::move(*dynamic.module),
                                    std::move(*mapped.module), std::move(relocatable),
                                    original_size, original_mapping_size, extra_size, data_size, false, {}});
            if (!options.preparation.empty() && options.preparation[index].restrict_entry_points) {
                auto& module = set.modules_.back();
                module.restrict_entry_points = true;
                const auto roots = options.preparation[index].admitted_entry_offsets;
                module.admitted_entry_offsets.assign(roots.begin(), roots.end());
            }
        }

        if (options.relocation_mode == ModuleRelocationMode::host_link) {
            std::vector<linker::LinkModule> link_modules;
            std::vector<std::span<std::byte>> destinations;
            link_modules.reserve(set.modules_.size());
            destinations.reserve(set.modules_.size());
            for (auto& module : set.modules_) {
                const auto image = module.mapping.preparation_bytes().first(module.original_image_size);
                link_modules.push_back({module.name, image, &module.metadata,
                    module.mapping.base_address(), module.original_mapping_size,
                    module.relocatable_ranges});
                destinations.push_back(image);
            }
            auto plan = linker::plan_link(link_modules, options.bindings, options.link_limits);
            if (!plan.ok()) {
                const auto index = plan.unresolved.empty()
                    ? std::optional<std::size_t>{}
                    : std::optional<std::size_t>{plan.unresolved.front().module};
                auto result = failure(ModuleLoadError::linking_failed, index, std::move(plan.message));
                result.unresolved_count = plan.unresolved_count;
                result.unresolved = std::move(plan.unresolved);
                return result;
            }
            const auto applied = plan.plan->apply(destinations);
            if (!applied.ok()) {
                return failure(ModuleLoadError::application_failed, applied.module, applied.message);
            }
        }
        std::vector<std::uintptr_t> preparation_bases;
        preparation_bases.reserve(set.modules_.size());
        for (const auto& module : set.modules_) {
            preparation_bases.push_back(module.mapping.base_address());
        }
        const auto prepare_module = [&](std::size_t index) {
            const auto& requested = options.preparation[index];
            if (requested.callback == nullptr) return true;
            auto& module = set.modules_[index];
            const auto bytes = module.mapping.preparation_bytes();
            const auto code = module.appended_code_bytes == 0 ? std::span<std::byte>{} :
                bytes.subspan(module.original_mapping_size + module.appended_data_bytes, module.appended_code_bytes);
            const ModulePreparationView view{
                index, module.name, module.mapping.base_address(),
                bytes.first(module.original_image_size), code,
                module.appended_code_bytes == 0 ? 0 : module.original_mapping_size + module.appended_data_bytes,
                module.metadata, preparation_bases,
                module.appended_data_bytes == 0 ? std::span<std::byte>{} :
                    bytes.subspan(module.original_mapping_size, module.appended_data_bytes),
                module.appended_data_bytes == 0 ? 0 : module.original_mapping_size};
            return requested.callback(view, requested.context);
        };
        std::vector<char> prepared(options.preparation.size(), 1);
        if (options.parallel_preparation && options.preparation.size() > 1) {
            std::vector<std::thread> workers;
            std::size_t started = 0;
            try {
                for (; started + 1 < options.preparation.size(); ++started)
                    workers.emplace_back([&, index = started + 1] { prepared[index] = prepare_module(index); });
            } catch (const std::system_error&) {
                // The modules without a thread run below, on this one.
            }
            prepared[0] = prepare_module(0);
            for (auto index = started + 1; index < options.preparation.size(); ++index)
                prepared[index] = prepare_module(index);
            for (auto& worker : workers) worker.join();
        } else {
            for (std::size_t index = 0; index < options.preparation.size(); ++index)
                if (!(prepared[index] = prepare_module(index))) break;
        }
        for (std::size_t index = 0; index < prepared.size(); ++index) {
            if (!prepared[index]) {
                return failure(ModuleLoadError::preparation_failed, index,
                               "The module preparation callback rejected the load.");
            }
        }
        for (std::size_t index = 0; index < set.modules_.size(); ++index) {
            const auto sealed = set.modules_[index].mapping.seal();
            if (!sealed.ok()) {
                return failure(ModuleLoadError::sealing_failed, index,
                               "Host page sealing failed with system error " +
                                   std::to_string(sealed.system_error) + ".");
            }
        }
        ModuleSetResult result;
        result.modules.emplace(std::move(set));
        return result;
    } catch (const std::bad_alloc&) {
        return failure(ModuleLoadError::resource_limit, std::nullopt, {});
    } catch (const std::length_error&) {
        return failure(ModuleLoadError::resource_limit, std::nullopt, {});
    }
}

std::size_t ModuleSet::relocation_count() const noexcept {
    std::size_t count = 0;
    for (const auto& module : modules_) {
        count += module.metadata.relocations.size() + module.metadata.packed_relative_targets.size();
    }
    return count;
}

std::string_view ModuleSet::module_name(std::size_t index) const {
    return modules_.at(index).name;
}

const platform::HostModule& ModuleSet::mapping(std::size_t index) const {
    return modules_.at(index).mapping;
}

const linker::DynamicModule& ModuleSet::metadata(std::size_t index) const {
    return modules_.at(index).metadata;
}

std::optional<linker::ModuleRange> ModuleSet::appended_code(std::size_t index) const {
    const auto& module = modules_.at(index);
    if (module.appended_code_bytes == 0) return {};
    return linker::ModuleRange{module.original_mapping_size + module.appended_data_bytes, module.appended_code_bytes};
}

bool ModuleSet::permits_guest_entry(std::uint64_t address) const noexcept {
    if (!address || address % 4) return false;
    for (const auto& module : modules_) {
        const auto base = module.mapping.base_address();
        if (address < base) continue;
        const auto offset = address - base;
        if (offset >= module.mapping.image_size() || module.mapping.executable_address(offset) != address) continue;
        return !module.restrict_entry_points ||
            std::binary_search(module.admitted_entry_offsets.begin(), module.admitted_entry_offsets.end(), offset);
    }
    return false;
}

std::optional<std::uintptr_t> ModuleSet::entry_address(
    std::size_t index, std::string_view name) const noexcept {
    if (index >= modules_.size()) return std::nullopt;
    const auto& module = modules_[index];
    for (const auto& symbol : module.metadata.symbols) {
        if (symbol.name == name && symbol.defined() && symbol.section < 0xff00 &&
            (symbol.info & 15u) == 2u &&
            symbol.value < module.original_image_size) {
            const auto address = module.mapping.executable_address(static_cast<std::size_t>(symbol.value));
            return address && permits_guest_entry(*address) ? address : std::nullopt;
        }
    }
    return std::nullopt;
}

} // namespace armigo::runtime
