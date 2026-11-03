// ============================================================================
// Armigo Platform - Apple Platform Integration
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================
#pragma once

#include <armigo/common/diagnostics.h>
#import <Metal/Metal.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace armigo::platform::metal {

// Diagnostic-only driver storage. Owned by a submission until GPU feedback;
// never shared with guest query heaps or recycled while work is in flight.
class PassCounters {
public:
    static constexpr std::size_t invalid = ~std::size_t{0};
    struct Pass {
        const char* type;
        std::string attachments;
        unsigned long width = 0, height = 0, samples = 0, colors = 0;
        bool depth = false, ended = false;
        std::uint64_t draws = 0, depth_binds = 0, pipeline_binds = 0;
        std::uint64_t blend_changes = 0, blended_draws = 0, depth_draws = 0;
    };
    static bool enabled() {
        static const bool value = [] {
            const char* env = std::getenv("ARMIGO_NVN_PASS_COUNTERS");
            return env && std::strcmp(env, "1") == 0;
        }();
        return value;
    }

    std::size_t begin(id<MTLDevice> device, id<MTL4RenderCommandEncoder> encoder,
                      MTL4RenderPassDescriptor* descriptor, const char* type) {
        if (!enabled()) return invalid;
        if (!heap_ && !failed_) {
            auto* config = [MTL4CounterHeapDescriptor new];
            config.type = MTL4CounterHeapTypeTimestamp;
            config.count = capacity * 2;
            NSError* error = nil;
            heap_ = [device newCounterHeapWithDescriptor:config error:&error];
            if (!heap_) {
                failed_ = true;
                armigo::diag::opt_in( "nvn_pass_error reason=heap_allocation detail=%s\n",
                             error ? error.localizedDescription.UTF8String : "unknown");
            } else {
                [heap_ invalidateCounterRange:NSMakeRange(0, capacity * 2)];

            }
        }
        if (!heap_ || passes.size() == capacity) {
            ++dropped_;
            return invalid;
        }
        Pass pass{type, {}};
        const auto attachment = [&](const char* name, MTLRenderPassAttachmentDescriptor* a) {
            if (!a.texture) return;
            const auto w = std::max<NSUInteger>(1, a.texture.width >> a.level);
            const auto h = std::max<NSUInteger>(1, a.texture.height >> a.level);
            pass.width = pass.width ? std::min(pass.width, w) : w;
            pass.height = pass.height ? std::min(pass.height, h) : h;
            pass.samples = a.texture.sampleCount;
            char text[256];
            std::snprintf(text, sizeof text, "%s:%lux%lu:s%lu:f%lu:l%lu:a%lu:load%lu:store%lu,",
                          name, w, h, a.texture.sampleCount, a.texture.pixelFormat,
                          a.level, a.slice, a.loadAction, a.storeAction);
            pass.attachments += text;
        };
        for (unsigned i = 0; i < 8; ++i) {
            char name[8];
            std::snprintf(name, sizeof name, "c%u", i);
            attachment(name, descriptor.colorAttachments[i]);
            pass.colors += descriptor.colorAttachments[i].texture != nil;
        }
        attachment("d", descriptor.depthAttachment);
        attachment("s", descriptor.stencilAttachment);
        pass.depth = descriptor.depthAttachment.texture != nil;
        const auto index = passes.size();
        passes.push_back(std::move(pass));
        // Metal 4's encoder sampling API targets opaque driver memory. Do not
        // use MTL4CommandBuffer::writeTimestampIntoHeap for pass diagnostics.
        [encoder writeTimestampWithGranularity:MTL4TimestampGranularityRelaxed
                                    afterStage:MTLRenderStageVertex intoHeap:heap_ atIndex:index * 2];
        return index;
    }

    void end(id<MTL4RenderCommandEncoder> encoder, std::size_t index) {
        if (index == invalid) return;
        [encoder writeTimestampWithGranularity:MTL4TimestampGranularityRelaxed
                                    afterStage:MTLRenderStageVertex | MTLRenderStageFragment intoHeap:heap_ atIndex:index * 2 + 1];
        passes[index].ended = true;
    }

    // Called only after final submission feedback, including all earlier
    // synchronized segments. The device supplies GPU tick frequency explicitly.
    void report(id<MTLDevice> device, std::uint64_t fence, bool success, double gpu_ms) {
        if (!enabled()) return;
        NSData* data = heap_ && !passes.empty()
            ? [heap_ resolveCounterRange:NSMakeRange(0, passes.size() * 2)] : nil;
        const auto frequency = [device queryTimestampFrequency];
        const double ns_per_tick = frequency ? 1e9 / double(frequency) : 0;
        double sum = 0;
        std::size_t valid_count = 0;
        for (std::size_t i = 0; i < passes.size(); ++i) {
            const auto& p = passes[i];
            MTL4TimestampHeapEntry entries[2]{};
            const auto offset = i * sizeof entries;
            if (data.length >= offset + sizeof entries)
                std::memcpy(entries, static_cast<const char*>(data.bytes) + offset, sizeof entries);
            const auto start = entries[0].timestamp, end = entries[1].timestamp;
            const bool valid = success && frequency && p.ended && p.draws &&
                end >= start && start != 0 && end != UINT64_MAX;
            const double ms = valid ? double(end - start) * ns_per_tick / 1e6 : -1;
            if (valid) { sum += ms; ++valid_count; }
            armigo::diag::opt_in(
                "nvn_pass frame=%llu pass=%zu type=%s width=%lu height=%lu samples=%lu colors=%lu depth=%u "
                "draws=%llu pipeline_binds=%llu depth_binds=%llu blend_changes=%llu blended_draws=%llu "
                "depth_draws=%llu valid=%u gpu_ms=%.6f start=%llu end=%llu attachments=%s\n",
                (unsigned long long)fence, i, p.type, p.width, p.height, p.samples, p.colors, unsigned(p.depth),
                (unsigned long long)p.draws, (unsigned long long)p.pipeline_binds,
                (unsigned long long)p.depth_binds, (unsigned long long)p.blend_changes,
                (unsigned long long)p.blended_draws, (unsigned long long)p.depth_draws,
                unsigned(valid), ms, (unsigned long long)start, (unsigned long long)end, p.attachments.c_str());
        }
        armigo::diag::opt_in( "nvn_pass_frame frame=%llu passes=%zu valid=%zu dropped=%zu stage_sum_ms=%.6f feedback_ms=%.6f ns_per_tick=%.9f\n",
                     (unsigned long long)fence, passes.size(), valid_count, dropped_, sum, gpu_ms, ns_per_tick);
    }

    std::vector<Pass> passes;
private:
    static constexpr std::size_t capacity = 2048;
    id<MTL4CounterHeap> heap_ = nil;
    bool failed_ = false;
    std::size_t dropped_ = 0;
};

} // namespace armigo::platform::metal
