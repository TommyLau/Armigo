// ============================================================================
// Armigo Audio - Horizon Audio Renderer
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include "renderer/memory/pool_mapper.h"
#include "renderer/sink/sink_info_base.h"

namespace armigo::audio::renderer {

void SinkInfoBase::clean_up() {
    type = Type::Invalid;
}

void SinkInfoBase::update(BehaviorInfo::ErrorInfo& error_info, OutStatus& out_status,
                          [[maybe_unused]] const InParameter& in_params,
                          [[maybe_unused]] const PoolMapper& pool_mapper) {
    std::memset(&out_status, 0, sizeof(OutStatus));
    error_info.error_code = ResultSuccess;
    error_info.address = CpuAddr(0);
}

void SinkInfoBase::update_for_command_generation() {}

SinkInfoBase::DeviceState* SinkInfoBase::get_device_state() {
    return reinterpret_cast<DeviceState*>(state.data());
}

SinkInfoBase::Type SinkInfoBase::get_type() const {
    return type;
}

bool SinkInfoBase::is_used() const {
    return in_use;
}

bool SinkInfoBase::should_skip() const {
    return buffer_unmapped;
}

u32 SinkInfoBase::get_node_id() const {
    return node_id;
}

u8* SinkInfoBase::get_state() {
    return state.data();
}

u8* SinkInfoBase::get_parameter() {
    return parameter.data();
}

} // namespace armigo::audio::renderer
