# ============================================================================
# Armigo Build - Native Build and Automation
# Copyright (c) 2026 Tommy Lau.
# SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
#
# This file is licensed under GPL-3.0-only or, by separate written
# agreement, an Armigo proprietary license.
# See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
# ============================================================================

# Portable CPU PCM frame mixer.
#
# The root build integrates this directory with
# add_subdirectory(src/audio). Source and include paths are resolved
# relative to this file so the target works from any including project.

set(ARMIGO_AUDIO_SOURCE_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../..")

add_library(armigo_audio pcm_mixer.cpp)
add_library(armigo::audio ALIAS armigo_audio)
target_include_directories(armigo_audio PUBLIC "${ARMIGO_AUDIO_SOURCE_ROOT}/include")
target_compile_features(armigo_audio PUBLIC cxx_std_20)

# Horizon audio renderer: the audren renderer state, command list generation
# and DSP command processing. Internal headers stay private to the target; the
# service layer uses include/armigo/audio/audio_renderer.h.
add_library(armigo_audio_renderer STATIC
    audio_renderer.cpp
    pcm_output.cpp
    renderer/behavior/behavior_info.cpp
    renderer/behavior/info_updater.cpp
    renderer/command/command_buffer.cpp
    renderer/command/command_generator.cpp
    renderer/command/data_source/adpcm.cpp
    renderer/command/data_source/decode.cpp
    renderer/command/data_source/pcm_float.cpp
    renderer/command/data_source/pcm_int16.cpp
    renderer/command/effect/aux_.cpp
    renderer/command/effect/biquad_filter.cpp
    renderer/command/effect/capture.cpp
    renderer/command/effect/compressor.cpp
    renderer/command/effect/delay.cpp
    renderer/command/effect/i3dl2_reverb.cpp
    renderer/command/effect/light_limiter.cpp
    renderer/command/effect/multi_tap_biquad_filter.cpp
    renderer/command/effect/reverb.cpp
    renderer/command/mix/clear_mix.cpp
    renderer/command/mix/copy_mix.cpp
    renderer/command/mix/depop_for_mix_buffers.cpp
    renderer/command/mix/depop_prepare.cpp
    renderer/command/mix/mix_ramp_grouped.cpp
    renderer/command/mix/mix_ramp.cpp
    renderer/command/mix/mix.cpp
    renderer/command/mix/volume_ramp.cpp
    renderer/command/mix/volume.cpp
    renderer/command/performance/performance.cpp
    renderer/command/resample/downmix_6ch_to_2ch.cpp
    renderer/command/resample/resample.cpp
    renderer/command/resample/upsample.cpp
    renderer/command/sink/circular_buffer.cpp
    renderer/command/sink/device.cpp
    renderer/common/guest_memory.cpp
    renderer/common/support.cpp
    renderer/dsp/command_list_processor.cpp
    renderer/effect/aux_.cpp
    renderer/effect/biquad_filter.cpp
    renderer/effect/buffer_mixer.cpp
    renderer/effect/capture.cpp
    renderer/effect/compressor.cpp
    renderer/effect/delay.cpp
    renderer/effect/effect_context.cpp
    renderer/effect/i3dl2.cpp
    renderer/effect/light_limiter.cpp
    renderer/effect/reverb.cpp
    renderer/memory/memory_pool_info.cpp
    renderer/memory/pool_mapper.cpp
    renderer/mix/mix_context.cpp
    renderer/mix/mix_info.cpp
    renderer/nodes/edge_matrix.cpp
    renderer/nodes/node_states.cpp
    renderer/performance/detail_aspect.cpp
    renderer/performance/entry_aspect.cpp
    renderer/performance/performance_manager.cpp
    renderer/sink/circular_buffer_sink_info.cpp
    renderer/sink/device_sink_info.cpp
    renderer/sink/sink_context.cpp
    renderer/sink/sink_info_base.cpp
    renderer/splitter/splitter_context.cpp
    renderer/splitter/splitter_destinations_data.cpp
    renderer/splitter/splitter_info.cpp
    renderer/system.cpp
    renderer/upsampler/upsampler_info.cpp
    renderer/upsampler/upsampler_manager.cpp
    renderer/voice/voice_context.cpp
    renderer/voice/voice_info.cpp
)
add_library(armigo::audio_renderer ALIAS armigo_audio_renderer)
target_include_directories(armigo_audio_renderer
    PUBLIC "${ARMIGO_AUDIO_SOURCE_ROOT}/include"
    PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
target_compile_features(armigo_audio_renderer PUBLIC cxx_std_20)
find_package(Threads REQUIRED)
target_link_libraries(armigo_audio_renderer PUBLIC Threads::Threads)

if(BUILD_TESTING)
    add_executable(armigo_pcm_mixer_tests
        "${ARMIGO_AUDIO_SOURCE_ROOT}/tests/audio/pcm_mixer_tests.cpp")
    target_link_libraries(armigo_pcm_mixer_tests PRIVATE armigo::audio)
    add_test(NAME audio.pcm_mixer COMMAND armigo_pcm_mixer_tests)
    set_tests_properties(audio.pcm_mixer PROPERTIES LABELS "audio" TIMEOUT 30)

    add_executable(armigo_audio_renderer_render_tests
        "${ARMIGO_AUDIO_SOURCE_ROOT}/tests/audio/audio_renderer_tests.cpp")
    target_include_directories(armigo_audio_renderer_render_tests PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
    target_link_libraries(armigo_audio_renderer_render_tests PRIVATE armigo::audio_renderer)
    add_test(NAME audio.renderer COMMAND armigo_audio_renderer_render_tests)
    set_tests_properties(audio.renderer PROPERTIES LABELS "audio" TIMEOUT 60)
endif()
