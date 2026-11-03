// ============================================================================
// Armigo Audio - Horizon Audio Renderer
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include "renderer/dsp/command_list_processor.h"
#include "renderer/command/command_list_header.h"
#include "renderer/command/commands.h"
#include "renderer/common/guest_memory.h"
#include "renderer/common/support.h"

#include <algorithm>
#include <atomic>

namespace armigo::audio::renderer {

void CommandListProcessor::initialize(GuestMemory& memory_, CpuAddr buffer, u64 size,
                                      PcmOutput* output_) {
    memory = &memory_;
    output = output_;
    header = reinterpret_cast<CommandListHeader*>(buffer);
    commands = reinterpret_cast<u8*>(buffer + sizeof(CommandListHeader));
    commands_buffer_size = size;
    command_count = header->command_count;
    sample_count = header->sample_count;
    target_sample_rate = header->sample_rate;
    mix_buffers = header->samples_buffer;
    buffer_count = header->buffer_count;
    processed_command_count = 0;
    const u64 scratch_size{2 * std::max<u64>(sample_count, TargetSampleCount)};
    if (scratch.size() < scratch_size) {
        scratch.resize(scratch_size);
    }
}

std::span<s32> CommandListProcessor::mix_buffer(const u64 offset, const u64 count) const {
    if (offset <= mix_buffers.size() && count <= mix_buffers.size() - offset) {
        return mix_buffers.subspan(offset, count);
    }
    static std::atomic<u32> reported{0};
    if (reported.fetch_add(1, std::memory_order_relaxed) < 8) {
        AUDIO_LOG_ERROR("Command mix buffer range {:#x}+{:#x} is outside the {:#x} mix buffer samples",
                        offset, count, mix_buffers.size());
    }
    std::fill(scratch.begin(), scratch.end(), 0);
    return {scratch.data(), std::min<u64>(count, scratch.size())};
}

void CommandListProcessor::set_process_time_max(const u64 time) {
    max_process_time = time;
}

u32 CommandListProcessor::get_remaining_command_count() const {
    return command_count - processed_command_count;
}

PcmOutput* CommandListProcessor::get_output() const {
    return output;
}

u64 CommandListProcessor::process([[maybe_unused]] u32 session_id) {
    const auto start_time_{host_time_us()};
    const auto command_base{CpuAddr(commands)};

    if (processed_command_count > 0) {
        current_processing_time += start_time_ - end_time;
    } else {
        start_time = start_time_;
        current_processing_time = 0;
    }

    for (u32 index = 0; index < command_count; index++) {
        auto& command{*reinterpret_cast<ICommand*>(commands)};

        if (command.magic != 0xCAFEBABE) {
            AUDIO_LOG_ERROR("Command has invalid magic! Expected 0xCAFEBABE, got {:08X}",
                            command.magic);
            return host_time_us() - start_time_;
        }

        auto current_offset{CpuAddr(commands) - command_base};

        if (current_offset + command.size > commands_buffer_size) {
            AUDIO_LOG_ERROR("Command exceeded command buffer, buffer size {:08X}, command ends at {:08X}",
                            commands_buffer_size,
                            CpuAddr(commands) + command.size - sizeof(CommandListHeader));
            return host_time_us() - start_time_;
        }

        if (!command.verify(*this)) {
            break;
        }

        if (command.enabled) {
            command.process(*this);
        }

        processed_command_count++;
        commands += command.size;
    }

    end_time = host_time_us();
    return end_time - start_time_;
}

}  // namespace armigo::audio::renderer
