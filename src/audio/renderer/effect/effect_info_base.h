// ============================================================================
// Armigo Audio - Horizon Audio Renderer
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#pragma once

#include <array>

#include "renderer/common/common.h"
#include "renderer/behavior/behavior_info.h"
#include "renderer/effect/effect_result_state.h"
#include "renderer/memory/address_info.h"
#include "renderer/memory/pool_mapper.h"
#include "renderer/common/types.h"

namespace armigo::audio::renderer {
// Base of all effects. Holds various data and functions used for all derived effects.
// Should not be used directly.
class EffectInfoBase {
public:
    enum class Type : u8 {
        Invalid,
        Mix,
        Aux,
        Delay,
        Reverb,
        I3dl2Reverb,
        BiquadFilter,
        LightLimiter,
        Capture,
        Compressor,
    };

    enum class UsageState {
        Invalid,
        New,
        Enabled,
        Disabled,
    };

    enum class OutStatus : u8 {
        Invalid,
        New,
        Initialized,
        Used,
        Removed,
    };

    enum class ParameterState : u8 {
        Initialized,
        Updating,
        Updated,
    };

    struct InParameterVersion1 {
        /* 0x00 */ Type type;
        /* 0x01 */ bool is_new;
        /* 0x02 */ bool enabled;
        /* 0x04 */ u32 mix_id;
        /* 0x08 */ CpuAddr workbuffer;
        /* 0x10 */ CpuAddr workbuffer_size;
        /* 0x18 */ u32 process_order;
        /* 0x1C */ char unk1C[0x4];
        /* 0x20 */ std::array<u8, 0xA0> specific;
    };
    static_assert(sizeof(InParameterVersion1) == 0xC0,
                  "EffectInfoBase::InParameterVersion1 has the wrong size!");

    struct InParameterVersion2 {
        /* 0x00 */ Type type;
        /* 0x01 */ bool is_new;
        /* 0x02 */ bool enabled;
        /* 0x04 */ u32 mix_id;
        /* 0x08 */ CpuAddr workbuffer;
        /* 0x10 */ CpuAddr workbuffer_size;
        /* 0x18 */ u32 process_order;
        /* 0x1C */ char unk1C[0x4];
        /* 0x20 */ std::array<u8, 0xA0> specific;
    };
    static_assert(sizeof(InParameterVersion2) == 0xC0,
                  "EffectInfoBase::InParameterVersion2 has the wrong size!");

    struct OutStatusVersion1 {
        /* 0x00 */ OutStatus state;
        /* 0x01 */ char unk01[0xF];
    };
    static_assert(sizeof(OutStatusVersion1) == 0x10,
                  "EffectInfoBase::OutStatusVersion1 has the wrong size!");

    struct OutStatusVersion2 {
        /* 0x00 */ OutStatus state;
        /* 0x01 */ char unk01[0xF];
        /* 0x10 */ EffectResultState result_state;
    };
    static_assert(sizeof(OutStatusVersion2) == 0x90,
                  "EffectInfoBase::OutStatusVersion2 has the wrong size!");

    struct State {
        std::array<u8, 0x500> buffer;
    };
    static_assert(sizeof(State) == 0x500, "EffectInfoBase::State has the wrong size!");

    EffectInfoBase() {
        cleanup();
    }

    virtual ~EffectInfoBase() = default;

    // Cleanup this effect, resetting it to a starting state.
    void cleanup() {
        type = Type::Invalid;
        enabled = false;
        mix_id = UnusedMixId;
        process_order = InvalidProcessOrder;
        buffer_unmapped = false;
        parameter = {};
        for (auto& workbuffer : workbuffers) {
            workbuffer.setup(CpuAddr(0), 0);
        }
    }

    // Forcibly unmap all assigned workbuffers from the AudioRenderer.
    void force_unmap_buffers(const PoolMapper& pool_mapper) {
        for (auto& workbuffer : workbuffers) {
            if (workbuffer.get_reference(false) != 0) {
                pool_mapper.force_unmap_pointer(workbuffer);
            }
        }
    }

    // Check if this effect is enabled.
    bool is_enabled() const {
        return enabled;
    }

    // Check if this effect should not be generated.
    bool should_skip() const {
        return buffer_unmapped;
    }

    // Get the type of this effect.
    Type get_type() const {
        return type;
    }

    // Set the type of this effect.
    void set_type(const Type type_) {
        type = type_;
    }

    // Get the mix id of this effect.
    s32 get_mix_id() const {
        return mix_id;
    }

    // Get the processing order of this effect.
    s32 get_processing_order() const {
        return process_order;
    }

    // Get this effect's parameter data.
    u8* get_parameter() {
        return parameter.data();
    }

    // Get this effect's parameter data.
    u8* get_state_buffer() {
        return state.data();
    }

    // Set this effect's usage state.
    void set_usage(const UsageState usage) {
        usage_state = usage;
    }

    // Check if this effects need to have its workbuffer information updated.
    // Version 1.
    bool should_update_work_buffer_info(const InParameterVersion1& params) const {
        return buffer_unmapped || params.is_new;
    }

    // Check if this effects need to have its workbuffer information updated.
    // Version 2.
    bool should_update_work_buffer_info(const InParameterVersion2& params) const {
        return buffer_unmapped || params.is_new;
    }

    // Get the current usage state of this effect.
    UsageState get_usage() const {
        return usage_state;
    }

    // Write the current state. Version 1.
    void store_status(OutStatusVersion1& out_status, const bool renderer_active) const {
        if (renderer_active) {
            if (usage_state != UsageState::Disabled) {
                out_status.state = OutStatus::Used;
            } else {
                out_status.state = OutStatus::Removed;
            }
        } else if (usage_state == UsageState::New) {
            out_status.state = OutStatus::Used;
        } else {
            out_status.state = OutStatus::Removed;
        }
    }

    // Write the current state. Version 2.
    void store_status(OutStatusVersion2& out_status, const bool renderer_active) const {
        if (renderer_active) {
            if (usage_state != UsageState::Disabled) {
                out_status.state = OutStatus::Used;
            } else {
                out_status.state = OutStatus::Removed;
            }
        } else if (usage_state == UsageState::New) {
            out_status.state = OutStatus::Used;
        } else {
            out_status.state = OutStatus::Removed;
        }
    }

    // Update the info with new parameters, version 1.
    virtual void update(BehaviorInfo::ErrorInfo& error_info,
                        [[maybe_unused]] const InParameterVersion1& params,
                        [[maybe_unused]] const PoolMapper& pool_mapper) {
        error_info.error_code = ResultSuccess;
        error_info.address = CpuAddr(0);
    }

    // Update the info with new parameters, version 2.
    virtual void update(BehaviorInfo::ErrorInfo& error_info,
                        [[maybe_unused]] const InParameterVersion2& params,
                        [[maybe_unused]] const PoolMapper& pool_mapper) {
        error_info.error_code = ResultSuccess;
        error_info.address = CpuAddr(0);
    }

    // Update the info after command generation. Usually only changes its state.
    virtual void update_for_command_generation() {}

    // Initialize a new result state. Version 2 only, unused.
    virtual void initialize_result_state([[maybe_unused]] EffectResultState& result_state) {}

    // Update the host-side state with the ADSP-side state. Version 2 only, unused.
    virtual void update_result_state([[maybe_unused]] EffectResultState& cpu_state,
                                   [[maybe_unused]] EffectResultState& dsp_state) {}

    // Get a workbuffer assigned to this effect with the given index.
    virtual CpuAddr get_workbuffer([[maybe_unused]] s32 index) {
        return 0;
    }

    // Get the first workbuffer assigned to this effect.
    CpuAddr get_single_buffer([[maybe_unused]] const s32 index) {
        if (enabled) {
            return workbuffers[0].get_reference(true);
        }

        if (usage_state != UsageState::Disabled) {
            const auto ref{workbuffers[0].get_reference(false)};
            const auto size{workbuffers[0].get_size()};
            if (ref != 0 && size > 0) {
                // Invalidate DSP cache
            }
        }
        return 0;
    }

    // Get the send buffer info, used by Aux and Capture.
    CpuAddr get_send_buffer_info() const {
        return send_buffer_info;
    }

    // Get the send buffer, used by Aux and Capture.
    CpuAddr get_send_buffer() const {
        return send_buffer;
    }

    // Get the return buffer info, used by Aux and Capture.
    CpuAddr get_return_buffer_info() const {
        return return_buffer_info;
    }

    // Get the return buffer, used by Aux and Capture.
    CpuAddr get_return_buffer() const {
        return return_buffer;
    }

protected:
    // Type of this effect. May be changed
    Type type{Type::Invalid};
    // Is this effect enabled?
    bool enabled{};
    // Are this effect's buffers unmapped?
    bool buffer_unmapped{};
    // Current usage state
    UsageState usage_state{UsageState::Invalid};
    // Mix id of this effect
    s32 mix_id{UnusedMixId};
    // Process order of this effect
    s32 process_order{InvalidProcessOrder};
    // Workbuffers assigned to this effect
    std::array<AddressInfo, 2> workbuffers{AddressInfo(CpuAddr(0), 0), AddressInfo(CpuAddr(0), 0)};
    // Aux/Capture buffer info for reading
    CpuAddr send_buffer_info{};
    // Aux/Capture buffer for reading
    CpuAddr send_buffer{};
    // Aux/Capture buffer info for writing
    CpuAddr return_buffer_info{};
    // Aux/Capture buffer for writing
    CpuAddr return_buffer{};
    // Parameters of this effect
    std::array<u8, sizeof(InParameterVersion2)> parameter{};
    // State of this effect used by the AudioRenderer across calls
    std::array<u8, sizeof(State)> state{};
};

} // namespace armigo::audio::renderer
