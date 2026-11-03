// ============================================================================
// Armigo NVN - Queue Management
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include <armigo/common/diagnostics.h>
#include "backend.h"
#include "entry_table.h"
#include "present_internal.h"

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <exception>
#include <limits>
#include <span>

namespace armigo::nvn {
namespace {

constexpr std::uint32_t queue_tag = object_tag('Q', 'U', 'E', 'U');
constexpr std::uint64_t default_command_memory_size = 65'536;
constexpr std::uint64_t default_compute_memory_size = 262'144;
constexpr std::uint64_t default_control_memory_size = 16'384;
constexpr std::uint64_t default_flush_threshold = 4'096;
constexpr std::uint64_t queue_memory_alignment = 4'096;

struct QueueStorage {
    QueueState state{};
    NVNqueueGetErrorResult error = NVN_QUEUE_GET_ERROR_RESULT_GPU_NO_ERROR;
    std::uint32_t reserved = 0;
    std::uint64_t command_memory_used = 0;
    std::uint64_t control_memory_used = 0;
    std::uint64_t compute_memory_used = 0;
};
static_assert(sizeof(QueueStorage) <= sizeof(NVNqueue));

QueueBuilderState *queue_builder(NVNqueueBuilder *builder) noexcept {
    return reinterpret_cast<QueueBuilderState *>(builder);
}

const QueueBuilderState *queue_builder(const NVNqueueBuilder *builder) noexcept {
    return reinterpret_cast<const QueueBuilderState *>(builder);
}

QueueStorage *queue_storage(NVNqueue *queue) noexcept { return reinterpret_cast<QueueStorage *>(queue); }

const QueueStorage *queue_storage(const NVNqueue *queue) noexcept {
    return reinterpret_cast<const QueueStorage *>(queue);
}

std::size_t required_queue_memory(const QueueBuilderState &builder) noexcept {
    constexpr std::uint64_t maximum = std::numeric_limits<std::size_t>::max();
    const std::uint64_t sizes[] = {
        builder.command_memory_size,
        builder.compute_memory_size,
        builder.control_memory_size,
    };
    std::uint64_t total = 0;
    for (const std::uint64_t size : sizes) {
        if (size > maximum - total) return 0;
        total += size;
    }
    if (total > maximum - (queue_memory_alignment - 1)) return 0;
    return static_cast<std::size_t>((total + queue_memory_alignment - 1) & ~(queue_memory_alignment - 1));
}

void report_queue_failure(NVNqueue *queue, NVNqueueGetErrorResult error, const char *message) noexcept {
    if (queue_valid(queue)) queue_storage(queue)->error = error;
    armigo::diag::write(armigo::diag::Level::error,  "Armigo NVN: queue failure: %s\n", message ? message : "unspecified error");
}

void queue_builder_set_device(NVNqueueBuilder *builder, NVNdevice *device) {
    if (builder) queue_builder(builder)->device = device;
}

void queue_builder_set_defaults(NVNqueueBuilder *builder) {
    if (!builder) return;
    NVNdevice *device = queue_builder(builder)->device;
    *queue_builder(builder) = QueueBuilderState{};
    QueueBuilderState &state = *queue_builder(builder);
    state.device = device;
    state.command_memory_size = default_command_memory_size;
    state.compute_memory_size = default_compute_memory_size;
    state.control_memory_size = default_control_memory_size;
    state.flush_threshold = default_flush_threshold;
}

void queue_builder_set_flags(NVNqueueBuilder *builder, int flags) {
    if (builder) queue_builder(builder)->flags = static_cast<std::uint32_t>(flags);
}

void queue_builder_set_command_memory_size(NVNqueueBuilder *builder, std::size_t size) {
    if (builder) queue_builder(builder)->command_memory_size = size;
}

void queue_builder_set_compute_memory_size(NVNqueueBuilder *builder, std::size_t size) {
    if (builder) queue_builder(builder)->compute_memory_size = size;
}

void queue_builder_set_control_memory_size(NVNqueueBuilder *builder, std::size_t size) {
    if (builder) queue_builder(builder)->control_memory_size = size;
}

std::size_t queue_builder_get_queue_memory_size(const NVNqueueBuilder *builder) {
    return builder ? required_queue_memory(*queue_builder(builder)) : 0;
}

void queue_builder_set_queue_memory(NVNqueueBuilder *builder, void *memory, std::size_t size) {
    if (!builder) return;
    queue_builder(builder)->memory = memory;
    queue_builder(builder)->memory_size = size;
}

void queue_builder_set_command_flush_threshold(NVNqueueBuilder *builder, std::size_t size) {
    if (builder) queue_builder(builder)->flush_threshold = size;
}

NVNboolean queue_initialize(NVNqueue *queue, const NVNqueueBuilder *builder) {
    if (!queue || !builder || !queue_builder(builder)->device || !backends().replay) {
        armigo::diag::write(armigo::diag::Level::error,  "Armigo NVN: queue initialization requires a queue, "
                             "device, builder, and replay backend\n");
        return 0;
    }
    const QueueBuilderState &source = *queue_builder(builder);
    const std::size_t required = required_queue_memory(source);
    if (!required || (source.memory && source.memory_size < required) || (!source.memory && source.memory_size)) {
        armigo::diag::write(armigo::diag::Level::error,  "Armigo NVN: queue initialization rejected invalid queue memory\n");
        return 0;
    }

    QueueStorage initialized{};
    initialized.state.tag = queue_tag;
    initialized.state.device = source.device;
    initialized.state.submits = 0;
    *queue_storage(queue) = initialized;
    return 1;
}

void queue_finalize(NVNqueue *queue) {
    if (queue) *queue_storage(queue) = QueueStorage{};
}

void queue_set_debug_label(NVNqueue *, const char *) {}

void queue_submit_commands(NVNqueue *queue, int count, const NVNcommandHandle *handles) {
    if (!queue_valid(queue)) {
        report_queue_failure(queue, NVN_QUEUE_GET_ERROR_RESULT_GPU_ERROR_UNKNOWN, "submission used an invalid queue");
        return;
    }
    if (count < 0 || (count != 0 && !handles)) {
        report_queue_failure(queue, NVN_QUEUE_GET_ERROR_RESULT_GPU_ERROR_UNKNOWN,
                             "submission has an invalid command handle range");
        return;
    }
    ReplayBackend *replay = backends().replay;
    if (!replay) {
        report_queue_failure(queue, NVN_QUEUE_GET_ERROR_RESULT_GPU_ERROR_UNKNOWN, "submission has no replay backend");
        return;
    }

    try {
        const auto commands = std::span<const NVNcommandHandle>(handles, static_cast<std::size_t>(count));
        // This field is the most recent backend fence, not a submission count.
        queue_storage(queue)->state.submits = replay->submit(commands);
    } catch (const std::exception &exception) {
        report_queue_failure(queue, NVN_QUEUE_GET_ERROR_RESULT_GPU_ERROR_UNKNOWN, exception.what());
    } catch (...) {
        report_queue_failure(queue, NVN_QUEUE_GET_ERROR_RESULT_GPU_ERROR_UNKNOWN,
                             "replay backend threw an unknown exception during submission");
    }
}

void queue_flush(NVNqueue *queue) {
    if (!queue_valid(queue) || !backends().replay) {
        report_queue_failure(queue, NVN_QUEUE_GET_ERROR_RESULT_GPU_ERROR_UNKNOWN,
                             "flush requires a valid queue and replay backend");
        return;
    }
    try {
        backends().replay->flush();
    } catch (const std::exception &exception) {
        report_queue_failure(queue, NVN_QUEUE_GET_ERROR_RESULT_GPU_ERROR_UNKNOWN, exception.what());
    } catch (...) {
        report_queue_failure(queue, NVN_QUEUE_GET_ERROR_RESULT_GPU_ERROR_UNKNOWN,
                             "replay backend threw an unknown exception during flush");
    }
}

void queue_finish(NVNqueue *queue) {
    if (!queue_valid(queue) || !backends().replay) {
        report_queue_failure(queue, NVN_QUEUE_GET_ERROR_RESULT_GPU_ERROR_UNKNOWN,
                             "finish requires a valid queue and replay backend");
        return;
    }
    ReplayBackend *replay = backends().replay;
    const std::uint64_t fence = queue_last_fence(queue);
    try {
        replay->flush();
        if (fence && replay->completed_fence() < fence &&
            !blocking_wait([replay, fence] { return replay->wait_fence(fence, UINT64_MAX); }, "queue_wait")) {
            report_queue_failure(queue, NVN_QUEUE_GET_ERROR_RESULT_GPU_ERROR_TIMEOUT,
                                 "replay backend did not complete queue finish");
        }
    } catch (const std::exception &exception) {
        report_queue_failure(queue, NVN_QUEUE_GET_ERROR_RESULT_GPU_ERROR_UNKNOWN, exception.what());
    } catch (...) {
        report_queue_failure(queue, NVN_QUEUE_GET_ERROR_RESULT_GPU_ERROR_UNKNOWN,
                             "replay backend threw an unknown exception during finish");
    }
}

NVNqueueGetErrorResult queue_get_error(NVNqueue *queue, NVNqueueErrorInfo *info) {
    if (info) std::memset(info, 0, sizeof(*info));
    if (!queue_valid(queue)) return NVN_QUEUE_GET_ERROR_RESULT_GPU_ERROR_UNKNOWN;
    return queue_storage(queue)->error;
}

std::size_t queue_get_total_command_memory_used(NVNqueue *queue) {
    return queue_valid(queue) ? static_cast<std::size_t>(queue_storage(queue)->command_memory_used) : 0;
}

std::size_t queue_get_total_control_memory_used(NVNqueue *queue) {
    return queue_valid(queue) ? static_cast<std::size_t>(queue_storage(queue)->control_memory_used) : 0;
}

std::size_t queue_get_total_compute_memory_used(NVNqueue *queue) {
    return queue_valid(queue) ? static_cast<std::size_t>(queue_storage(queue)->compute_memory_used) : 0;
}

void queue_reset_memory_usage_counts(NVNqueue *queue) {
    if (!queue_valid(queue)) return;
    QueueStorage &state = *queue_storage(queue);
    state.command_memory_used = 0;
    state.control_memory_used = 0;
    state.compute_memory_used = 0;
}

} // namespace

bool queue_valid(const NVNqueue *queue) noexcept { return queue && queue_storage(queue)->state.tag == queue_tag; }

std::uint64_t queue_last_fence(const NVNqueue *queue) noexcept {
    return queue_valid(queue) ? queue_storage(queue)->state.submits : 0;
}

void queue_error(NVNqueue *queue, const char *message) noexcept {
    report_queue_failure(queue, NVN_QUEUE_GET_ERROR_RESULT_GPU_ERROR_UNKNOWN, message);
}

void register_queue_entries() noexcept {
    install<Entry::nvnQueueGetError>(&queue_get_error);
    install<Entry::nvnQueueGetTotalCommandMemoryUsed>(&queue_get_total_command_memory_used);
    install<Entry::nvnQueueGetTotalControlMemoryUsed>(&queue_get_total_control_memory_used);
    install<Entry::nvnQueueGetTotalComputeMemoryUsed>(&queue_get_total_compute_memory_used);
    install<Entry::nvnQueueResetMemoryUsageCounts>(&queue_reset_memory_usage_counts);
    install<Entry::nvnQueueBuilderSetDevice>(&queue_builder_set_device);
    install<Entry::nvnQueueBuilderSetDefaults>(&queue_builder_set_defaults);
    install<Entry::nvnQueueBuilderSetFlags>(&queue_builder_set_flags);
    install<Entry::nvnQueueBuilderSetCommandMemorySize>(&queue_builder_set_command_memory_size);
    install<Entry::nvnQueueBuilderSetComputeMemorySize>(&queue_builder_set_compute_memory_size);
    install<Entry::nvnQueueBuilderSetControlMemorySize>(&queue_builder_set_control_memory_size);
    install<Entry::nvnQueueBuilderGetQueueMemorySize>(&queue_builder_get_queue_memory_size);
    install<Entry::nvnQueueBuilderSetQueueMemory>(&queue_builder_set_queue_memory);
    install<Entry::nvnQueueBuilderSetCommandFlushThreshold>(&queue_builder_set_command_flush_threshold);
    install<Entry::nvnQueueInitialize>(&queue_initialize);
    install<Entry::nvnQueueFinalize>(&queue_finalize);
    install<Entry::nvnQueueSetDebugLabel>(&queue_set_debug_label);
    install<Entry::nvnQueueSubmitCommands>(&queue_submit_commands);
    install<Entry::nvnQueueFlush>(&queue_flush);
    install<Entry::nvnQueueFinish>(&queue_finish);
}

} // namespace armigo::nvn
