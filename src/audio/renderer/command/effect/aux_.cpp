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
#include "renderer/command/effect/aux_.h"
#include "renderer/effect/aux_.h"
#include "renderer/common/guest_memory.h"

namespace armigo::audio::renderer {
// Reset an AuxBuffer.
static void reset_aux_buffer_dsp(GuestMemory& memory, const CpuAddr aux_info) {
    if (aux_info == 0) {
        AUDIO_LOG_ERROR("Aux info is 0!");
        return;
    }

    AuxInfo::AuxInfoDsp info{};
    memory.read_block(aux_info, &info, sizeof(AuxInfo::AuxInfoDsp));

    info.read_offset = 0;
    info.write_offset = 0;
    info.total_sample_count = 0;

    memory.write_block(aux_info, &info, sizeof(AuxInfo::AuxInfoDsp));
}

// Write the given input mix buffer to the memory at send_buffer, and update send_info_ if
// update_count is set, to notify the game that an update happened.
static u32 write_aux_buffer_dsp(GuestMemory& memory, CpuAddr send_info_,
                             [[maybe_unused]] u32 sample_count, CpuAddr send_buffer, u32 count_max,
                             std::span<const s32> input, u32 write_count_, u32 write_offset,
                             u32 update_count) {
    if (write_count_ > count_max) {
        AUDIO_LOG_ERROR("write_count must be smaller than count_max! write_count {}, count_max {}",
                  write_count_, count_max);
        return 0;
    }

    if (send_info_ == 0) {
        AUDIO_LOG_ERROR("send_info_ is 0!");
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

    AuxInfo::AuxInfoDsp send_info{};
    memory.read_block(send_info_, &send_info, sizeof(AuxInfo::AuxInfoDsp));

    u32 target_write_offset{send_info.write_offset + write_offset};
    if (target_write_offset > count_max) {
        return 0;
    }

    u32 write_count{write_count_};
    u32 read_pos{0};
    while (write_count > 0) {
        u32 to_write{(std::min)(count_max - target_write_offset, write_count)};
        if (to_write > 0) {
            const auto write_addr = send_buffer + target_write_offset * sizeof(s32);
            memory.write_block(write_addr, &input[read_pos], to_write * sizeof(s32));
        }
        target_write_offset = (target_write_offset + to_write) % count_max;
        write_count -= to_write;
        read_pos += to_write;
    }

    if (update_count) {
        send_info.write_offset = (send_info.write_offset + update_count) % count_max;
    }

    memory.write_block(send_info_, &send_info, sizeof(AuxInfo::AuxInfoDsp));
    return write_count_;
}

// Read the given memory at return_buffer into the output mix buffer, and update return_info_ if
// update_count is set, to notify the game that an update happened.
static u32 read_aux_buffer_dsp(GuestMemory& memory, CpuAddr return_info_,
                            CpuAddr return_buffer, u32 count_max, std::span<s32> output,
                            u32 read_count_, u32 read_offset, u32 update_count) {
    if (count_max == 0) {
        return 0;
    }

    if (read_count_ > count_max) {
        AUDIO_LOG_ERROR("count must be smaller than count_max! count {}, count_max {}",
                  read_count_, count_max);
        return 0;
    }

    if (return_info_ == 0) {
        AUDIO_LOG_ERROR("return_info_ is 0!");
        return 0;
    }

    if (output.empty()) {
        AUDIO_LOG_ERROR("output buffer is empty!");
        return 0;
    }

    if (return_buffer == 0) {
        AUDIO_LOG_ERROR("return_buffer is 0!");
        return 0;
    }

    AuxInfo::AuxInfoDsp return_info{};
    memory.read_block(return_info_, &return_info, sizeof(AuxInfo::AuxInfoDsp));

    u32 target_read_offset{return_info.read_offset + read_offset};
    if (target_read_offset > count_max) {
        return 0;
    }

    u32 read_count{read_count_};
    u32 write_pos{0};
    while (read_count > 0) {
        u32 to_read{(std::min)(count_max - target_read_offset, read_count)};
        if (to_read > 0) {
            const auto read_addr = return_buffer + target_read_offset * sizeof(s32);
            memory.read_block(read_addr, &output[write_pos], to_read * sizeof(s32));
        }
        target_read_offset = (target_read_offset + to_read) % count_max;
        read_count -= to_read;
        write_pos += to_read;
    }

    if (update_count) {
        return_info.read_offset = (return_info.read_offset + update_count) % count_max;
    }

    memory.write_block(return_info_, &return_info, sizeof(AuxInfo::AuxInfoDsp));
    return read_count_;
}

void AuxCommand::process(const CommandListProcessor& processor) {
    auto input_buffer{
        processor.mix_buffer(input * processor.sample_count, processor.sample_count)};
    auto output_buffer{
        processor.mix_buffer(output * processor.sample_count, processor.sample_count)};

    if (effect_enabled) {
        write_aux_buffer_dsp(*processor.memory, send_buffer_info, processor.sample_count, send_buffer,
                          count_max, input_buffer, processor.sample_count, write_offset,
                          update_count);

        auto read{read_aux_buffer_dsp(*processor.memory, return_buffer_info, return_buffer, count_max,
                                   output_buffer, processor.sample_count, write_offset,
                                   update_count)};

        if (read != processor.sample_count) {
            std::memset(&output_buffer[read], 0, (processor.sample_count - read) * sizeof(s32));
        }
    } else {
        reset_aux_buffer_dsp(*processor.memory, send_buffer_info);
        reset_aux_buffer_dsp(*processor.memory, return_buffer_info);
        if (input != output) {
            std::memcpy(output_buffer.data(), input_buffer.data(), output_buffer.size_bytes());
        }
    }
}

bool AuxCommand::verify([[maybe_unused]] const CommandListProcessor& processor) {
    return true;
}

} // namespace armigo::audio::renderer
