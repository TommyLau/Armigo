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
#include "renderer/command/effect/capture.h"
#include "renderer/effect/aux_.h"
#include "renderer/common/guest_memory.h"

namespace armigo::audio::renderer {
// Reset an AuxBuffer.
static void reset_aux_buffer_dsp(GuestMemory& memory, const CpuAddr aux_info) {
    if (aux_info == 0) {
        AUDIO_LOG_ERROR("Aux info is 0!");
        return;
    }

    memory.write32(CpuAddr(aux_info + offsetof(AuxInfo::AuxInfoDsp, read_offset)), 0);
    memory.write32(CpuAddr(aux_info + offsetof(AuxInfo::AuxInfoDsp, write_offset)), 0);
    memory.write32(CpuAddr(aux_info + offsetof(AuxInfo::AuxInfoDsp, total_sample_count)), 0);
}

// Write the given input mix buffer to the memory at send_buffer, and update send_info_ if
// update_count is set, to notify the game that an update happened.
static u32 write_aux_buffer_dsp(GuestMemory& memory, const CpuAddr send_info_,
                             const CpuAddr send_buffer, u32 count_max, std::span<const s32> input,
                             const u32 write_count_, const u32 write_offset,
                             const u32 update_count) {
    if (write_count_ > count_max) {
        AUDIO_LOG_ERROR("write_count must be smaller than count_max! write_count {}, count_max {}",
                  write_count_, count_max);
        return 0;
    }

    if (send_info_ == 0) {
        AUDIO_LOG_ERROR("send_info is 0!");
        return 0;
    }

    if (input.empty()) {
        AUDIO_LOG_ERROR("input buffer is empty!");
        return 0;
    }

    if (send_buffer == 0) {
        AUDIO_LOG_ERROR("send_buffer is 0!");
        return 0;
    }

    if (count_max == 0) {
        return 0;
    }

    AuxInfo::AuxBufferInfo send_info{};
    memory.read_block(send_info_, &send_info, sizeof(AuxInfo::AuxBufferInfo));

    u32 target_write_offset{send_info.dsp_info.write_offset + write_offset};
    if (target_write_offset > count_max || write_count_ == 0) {
        return 0;
    }

    u32 write_count{write_count_};
    u32 write_pos{0};
    while (write_count > 0) {
        u32 to_write{(std::min)(count_max - target_write_offset, write_count)};

        if (to_write > 0) {
            memory.write_block(send_buffer + target_write_offset * sizeof(s32),
                                    &input[write_pos], to_write * sizeof(s32));
        }

        target_write_offset = (target_write_offset + to_write) % count_max;
        write_count -= to_write;
        write_pos += to_write;
    }

    if (update_count) {
        const auto count_diff{send_info.dsp_info.total_sample_count -
                              send_info.cpu_info.total_sample_count};
        if (count_diff >= count_max) {
            auto dsp_lost_count{send_info.dsp_info.lost_sample_count + update_count};
            if (dsp_lost_count - send_info.cpu_info.lost_sample_count <
                send_info.dsp_info.lost_sample_count - send_info.cpu_info.lost_sample_count) {
                dsp_lost_count = send_info.cpu_info.lost_sample_count - 1;
            }
            send_info.dsp_info.lost_sample_count = dsp_lost_count;
        }

        send_info.dsp_info.write_offset =
            (send_info.dsp_info.write_offset + update_count + count_max) % count_max;

        auto new_sample_count{send_info.dsp_info.total_sample_count + update_count};
        if (new_sample_count - send_info.cpu_info.total_sample_count < count_diff) {
            new_sample_count = send_info.cpu_info.total_sample_count - 1;
        }
        send_info.dsp_info.total_sample_count = new_sample_count;
    }

    memory.write_block(send_info_, &send_info, sizeof(AuxInfo::AuxBufferInfo));

    return write_count_;
}

void CaptureCommand::process(const CommandListProcessor& processor) {
    if (effect_enabled) {
        auto input_buffer{
            processor.mix_buffer(input * processor.sample_count, processor.sample_count)};
        write_aux_buffer_dsp(*processor.memory, send_buffer_info, send_buffer, count_max, input_buffer,
                          processor.sample_count, write_offset, update_count);
    } else {
        reset_aux_buffer_dsp(*processor.memory, send_buffer_info);
    }
}

bool CaptureCommand::verify([[maybe_unused]] const CommandListProcessor& processor) {
    return true;
}

} // namespace armigo::audio::renderer
