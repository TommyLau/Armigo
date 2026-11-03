// ============================================================================
// Armigo Audio - Horizon Audio Renderer
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include "renderer/common/audio_renderer_parameter.h"
#include "renderer/common/workbuffer_allocator.h"
#include "renderer/behavior/behavior_info.h"
#include "renderer/splitter/splitter_context.h"
#include "renderer/common/support.h"

namespace armigo::audio::renderer {

SplitterDestinationData* SplitterContext::get_destination_data(const s32 splitter_id,
                                                             const s32 destination_id) {
    // Splitter ids come from guest voices and mixes; an id outside the
    // splitter table has no destinations.
    if (splitter_id < 0 || splitter_id >= info_count) {
        return nullptr;
    }
    return splitter_infos[splitter_id].get_data(destination_id);
}

SplitterInfo& SplitterContext::get_info(const s32 splitter_id) {
    return splitter_infos[splitter_id];
}

u32 SplitterContext::get_data_count() const {
    return destinations_count;
}

u32 SplitterContext::get_info_count() const {
    return info_count;
}

SplitterDestinationData& SplitterContext::get_data(const u32 index) {
    return splitter_destinations[index];
}

void SplitterContext::setup(std::span<SplitterInfo> splitter_infos_, const u32 splitter_info_count_,
                            SplitterDestinationData* splitter_destinations_,
                            const u32 destination_count_, const bool splitter_bug_fixed_,
                            const BehaviorInfo& behavior) {
    splitter_infos = splitter_infos_;
    info_count = splitter_info_count_;
    splitter_destinations = splitter_destinations_;
    destinations_count = destination_count_;
    splitter_bug_fixed = splitter_bug_fixed_;
    splitter_prev_volume_reset_supported = behavior.is_splitter_prev_volume_reset_supported();
    splitter_biquad_param_supported = behavior.is_biquad_filter_parameter_for_splitter_enabled();
    splitter_float_coeff_supported = behavior.is_splitter_destination_v2b_supported();
}

bool SplitterContext::using_splitter() const {
    return splitter_infos.size() > 0 && info_count > 0 && splitter_destinations != nullptr &&
           destinations_count > 0;
}

void SplitterContext::clear_all_new_connection_flag() {
    for (s32 i = 0; i < info_count; i++) {
        splitter_infos[i].set_new_connection_flag();
    }
}

bool SplitterContext::initialize(const BehaviorInfo& behavior,
                                 const AudioRendererParameterInternal& params,
                                 WorkbufferAllocator& allocator) {
    if (behavior.is_splitter_supported() && params.splitter_infos > 0 &&
        params.splitter_destinations > 0) {
        splitter_infos = allocator.allocate<SplitterInfo>(params.splitter_infos, 0x10);

        for (u32 i = 0; i < params.splitter_infos; i++) {
            std::construct_at<SplitterInfo>(&splitter_infos[i], static_cast<s32>(i));
        }

        if (splitter_infos.size() == 0) {
            splitter_infos = {};
            return false;
        }

        splitter_destinations =
            allocator.allocate<SplitterDestinationData>(params.splitter_destinations, 0x10).data();

        for (s32 i = 0; i < params.splitter_destinations; i++) {
            std::construct_at<SplitterDestinationData>(&splitter_destinations[i], i);
        }

        if (params.splitter_destinations <= 0) {
            splitter_infos = {};
            splitter_destinations = nullptr;
            return false;
        }

        setup(splitter_infos, params.splitter_infos, splitter_destinations,
              params.splitter_destinations, behavior.is_splitter_bug_fixed(), behavior);
    }
    return true;
}

bool SplitterContext::update(const u8* input, const u64 size) {
    auto in_params{reinterpret_cast<const InParameterHeader*>(input)};

    if (destinations_count == 0 || info_count == 0) {
        return true;
    }

    // The splitter section is guest data of the header-declared size; every
    // entry below is bounds-checked against it.
    if (size < sizeof(InParameterHeader)) {
        return false;
    }

    if (in_params->magic != get_splitter_in_param_header_magic()) {
        return false;
    }

    for (auto& splitter_info : splitter_infos) {
        splitter_info.clear_new_connection_flag();
    }

    u32 offset{sizeof(InParameterHeader)};
    offset = update_info(input, size, offset, in_params->info_count);
    offset = update_data(input, size, offset, in_params->destination_count);
    return true;
}

u32 SplitterContext::update_info(const u8* input, const u64 size, u32 offset,
                                 const u32 splitter_count) {
    for (u32 i = 0; i < splitter_count; i++) {
        if (offset > size || size - offset < sizeof(SplitterInfo::InParameter)) {
            break;
        }
        auto info_header{reinterpret_cast<const SplitterInfo::InParameter*>(input + offset)};

        if (info_header->magic != get_splitter_info_magic()) {
            continue;
        }

        if (info_header->id < 0 || info_header->id >= info_count) {
            break;
        }

        const u64 destination_ids_size{u64{info_header->destination_count} * sizeof(u32)};
        if (size - offset - sizeof(SplitterInfo::InParameter) < destination_ids_size) {
            break;
        }

        auto& info{splitter_infos[info_header->id]};
        recompose_destination(info, info_header);

        offset += info.update(info_header);
    }

    return offset;
}

u32 SplitterContext::update_data(const u8* input, const u64 size, u32 offset, const u32 count) {
    for (u32 i = 0; i < count; i++) {
        const u64 entry_size{!splitter_biquad_param_supported
                                 ? sizeof(SplitterDestinationData::InParameter)
                             : !splitter_float_coeff_supported
                                 ? sizeof(SplitterDestinationData::InParameterVersion2a)
                                 : sizeof(SplitterDestinationData::InParameterVersion2b)};
        if (offset > size || size - offset < entry_size) {
            break;
        }
        // Version selection based on feature flags:
        // - REV12: integer biquad params (Version2a)
        // - REV15: float coeff/biquad v2b
        // - older: no biquad fields
        if (!splitter_biquad_param_supported) {
            const auto* data_header =
                reinterpret_cast<const SplitterDestinationData::InParameter*>(input + offset);

            if (data_header->magic != get_splitter_send_data_magic()) {
                continue;
            }
            if (data_header->id < 0 || data_header->id >= destinations_count) {
                continue;
            }

            auto modified_params = *data_header;
            if (!splitter_prev_volume_reset_supported) {
                modified_params.reset_prev_volume = false;
            }
            splitter_destinations[data_header->id].update(modified_params);
            offset += sizeof(SplitterDestinationData::InParameter);
        } else if (!splitter_float_coeff_supported) {
            // Version 2a: struct contains legacy fixed-point biquad filter fields (REV12+)
            const auto* data_header_v2a =
                reinterpret_cast<const SplitterDestinationData::InParameterVersion2a*>(input +
                                                                                       offset);

            if (data_header_v2a->magic != get_splitter_send_data_magic()) {
                continue;
            }
            if (data_header_v2a->id < 0 || data_header_v2a->id >= destinations_count) {
                continue;
            }

            // Map common fields to the base format
            SplitterDestinationData::InParameter mapped{};
            mapped.magic = data_header_v2a->magic;
            mapped.id = data_header_v2a->id;
            mapped.mix_volumes = data_header_v2a->mix_volumes;
            mapped.mix_id = data_header_v2a->mix_id;
            mapped.in_use = data_header_v2a->in_use;
            mapped.reset_prev_volume =
                splitter_prev_volume_reset_supported ? data_header_v2a->reset_prev_volume : false;

            auto& destination = splitter_destinations[data_header_v2a->id];
            destination.update(mapped);

            // Convert legacy fixed-point biquad params into float representation
            auto biquad_filters = destination.get_biquad_filters();
            for (size_t filter_idx = 0; filter_idx < MaxBiquadFilters; filter_idx++) {
                const auto& legacy = data_header_v2a->biquad_filters[filter_idx];
                auto& out = biquad_filters[filter_idx];
                out.enabled = legacy.enabled;
                // s16 fixed-point scale: use Q14 like voices (b and a are s16, 1.0 ~= 1<<14)
                constexpr float scale = 1.0f / static_cast<float>(1 << 14);
                out.numerator[0] = static_cast<float>(legacy.b[0]) * scale;
                out.numerator[1] = static_cast<float>(legacy.b[1]) * scale;
                out.numerator[2] = static_cast<float>(legacy.b[2]) * scale;
                out.denominator[0] = static_cast<float>(legacy.a[0]) * scale;
                out.denominator[1] = static_cast<float>(legacy.a[1]) * scale;
            }

            offset += static_cast<u32>(sizeof(SplitterDestinationData::InParameterVersion2a));
        } else {
            // Version 2b: struct contains extra biquad filter fields with float coeffs
            const auto* data_header_v2b =
                reinterpret_cast<const SplitterDestinationData::InParameterVersion2b*>(input +
                                                                                       offset);

            if (data_header_v2b->magic != get_splitter_send_data_magic()) {
                continue;
            }
            if (data_header_v2b->id < 0 || data_header_v2b->id >= destinations_count) {
                continue;
            }

            // Map common fields to the old format
            SplitterDestinationData::InParameter mapped{};
            mapped.magic = data_header_v2b->magic;
            mapped.id = data_header_v2b->id;
            mapped.mix_volumes = data_header_v2b->mix_volumes;
            mapped.mix_id = data_header_v2b->mix_id;
            mapped.in_use = data_header_v2b->in_use;
            mapped.reset_prev_volume =
                splitter_prev_volume_reset_supported ? data_header_v2b->reset_prev_volume : false;

            // Store biquad filters from v2b (REV15+)
            auto& destination = splitter_destinations[data_header_v2b->id];
            destination.update(mapped);

            // Copy biquad filter parameters
            auto biquad_filters = destination.get_biquad_filters();
            for (size_t filter_idx = 0; filter_idx < MaxBiquadFilters; filter_idx++) {
                biquad_filters[filter_idx] = data_header_v2b->biquad_filters[filter_idx];
            }

            offset += static_cast<u32>(sizeof(SplitterDestinationData::InParameterVersion2b));
        }
    }

    return offset;
}

void SplitterContext::update_internal_state() {
    for (s32 i = 0; i < info_count; i++) {
        splitter_infos[i].update_internal_state();
    }
}

void SplitterContext::recompose_destination(SplitterInfo& out_info,
                                           const SplitterInfo::InParameter* info_header) {
    auto destination{out_info.get_data(0)};
    while (destination != nullptr) {
        auto dest{destination->get_next()};
        destination->set_next(nullptr);
        destination = dest;
    }
    out_info.set_destinations(nullptr);

    auto dest_count{info_header->destination_count};
    if (!splitter_bug_fixed) {
        dest_count = (std::min)(dest_count, get_dest_count_per_info_for_compat());
    }

    if (dest_count == 0) {
        return;
    }

    std::span<const u32> destination_ids{reinterpret_cast<const u32*>(&info_header[1]), dest_count};

    for (const auto id : destination_ids) {
        if (id >= static_cast<u32>(destinations_count)) {
            AUDIO_LOG_ERROR("Splitter destination id {} is outside the {} destinations", id,
                            destinations_count);
            return;
        }
    }

    auto head{&splitter_destinations[destination_ids[0]]};
    auto current_destination{head};
    for (u32 i = 1; i < dest_count; i++) {
        auto next_destination{&splitter_destinations[destination_ids[i]]};
        current_destination->set_next(next_destination);
        current_destination = next_destination;
    }

    out_info.set_destinations(head);
    out_info.set_destination_count(dest_count);
}

u32 SplitterContext::get_dest_count_per_info_for_compat() const {
    if (info_count <= 0) {
        return 0;
    }
    return static_cast<u32>(destinations_count / info_count);
}

u64 SplitterContext::calc_work_buffer_size(const BehaviorInfo& behavior,
                                        const AudioRendererParameterInternal& params) {
    u64 size{0};
    if (!behavior.is_splitter_supported()) {
        return size;
    }

    size += params.splitter_destinations * sizeof(SplitterDestinationData) +
            params.splitter_infos * sizeof(SplitterInfo);

    if (behavior.is_splitter_bug_fixed()) {
        size += align_up(params.splitter_destinations * sizeof(u32), 0x10);
    }
    return size;
}

} // namespace armigo::audio::renderer
