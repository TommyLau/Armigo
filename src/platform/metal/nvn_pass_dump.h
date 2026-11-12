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

#import <Metal/Metal.h>

#include <cstdint>
#include <string>
#include <vector>
#include <unordered_set>

namespace armigo::platform::metal::nvn_driver {

// Read once, when a replay backend is constructed. Unset variables leave
// `enabled` false: the replay then takes no dump allocations and encodes nothing
// extra. The selected frame is the replay's encoded present count (work before
// the first encoded present is frame 0).
struct PassDumpConfig {
    bool enabled = false;
    std::uint64_t first = 0;
    std::uint64_t last = 0;
    std::uint32_t width = 640;
    std::uint32_t draw_width = 0;
    std::uint32_t draw_height = 0;
    bool inputs = false;
    bool metadata_only = false;
    std::string directory;
};

PassDumpConfig read_pass_dump_config();

// One attachment or storage image. `pixels` is a shared RGBA8 buffer of the
// downscaled image when a PNG will be written; it is nil when the image was
// skipped. `raw` holds one float3 per downscaled pixel for HDR formats.
struct PassDumpImage {
    std::string kind;
    std::uint64_t serial = 0;
    std::string format;
    std::uint64_t native_id = 0;
    std::uint64_t native_root_id = 0;
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    std::uint32_t dst_w = 0;
    std::uint32_t dst_h = 0;
    std::string note;
    bool hdr = false;
    id<MTLBuffer> pixels = nil;
    id<MTLBuffer> raw = nil;
};

// One render pass or compute dispatch, published after its submission retires.
struct PassDumpLine {
    std::string directory;
    std::uint64_t frame = 0;
    std::uint32_t index = 0;
    bool compute = false;
    bool metadata_only = false;
    std::string why;
    std::uint64_t draws = 0;
    std::string vs;
    std::string fs;
    std::string cs;
    std::string attachments;
    // Distinct reflected sampled-texture bindings of the stages used in the
    // pass, including draws that reused their stage's argument table. A
    // bound resource may be unused by the shader's active branch.
    std::vector<std::string> reads;
    std::vector<std::string> draw_metadata;
    std::vector<PassDumpImage> images;
};

// GPU work for one image that `prepare` accepted. The argument table is filled
// on the worker; the encode thread only dispatches it.
struct PassDumpJob {
    PassDumpImage image;
    id<MTLComputePipelineState> pipeline = nil;
    id<MTL4ArgumentTable> table = nil;
    id<MTLTexture> sample = nil;
    id<MTLTexture> blit_from = nil;
    id<MTLBuffer> args = nil;
    id<MTLBuffer> bound_raw = nil;
    bool buffer_copy = false;
};

// Owns the dump kernels, the output directory and the per-run image budget.
// Touched from the replay worker only. PNG encoding is queued separately and
// does not use this object.
class PassDump {
public:
    PassDump(id<MTLDevice> device, PassDumpConfig config);
    ~PassDump();

    bool wants(std::uint64_t frame) const {
        return frame >= config_.first && frame <= config_.last;
    }
    bool stopped() const { return stopped_; }
    bool inputs() const { return config_.inputs && !config_.metadata_only; }
    bool metadata_only() const { return config_.metadata_only; }
    bool first_input(id<MTLTexture> texture) { return inputs_seen_.insert((__bridge const void*)texture).second; }
    bool splits_draws(NSUInteger width, NSUInteger height) const {
        return !config_.metadata_only && config_.draw_width && width == config_.draw_width && height == config_.draw_height;
    }
    const std::string& directory() const { return config_.directory; }

    // Creates the output directory on the first selected pass. A failure stops
    // the dump for the rest of the run.
    bool ensure_directory();
    // Pass index within the current frame, starting at 0. A new frame resets it.
    std::uint32_t next_index(std::uint64_t frame);

    // Describes `texture` and, when it can be sampled and the budget allows,
    // allocates the downsample. `level`, `slice` and `depth_plane` select the
    // stored subresource. A skipped image keeps its description and a note.
    PassDumpJob prepare(id<MTLTexture> texture, std::string kind, std::uint64_t serial,
                        NSUInteger level, NSUInteger slice, NSUInteger depth_plane, bool numeric = false);

    PassDumpJob prepare_buffer(std::string kind, std::uint64_t address, std::uint64_t bytes);

private:
    bool pipelines();
    bool accept(std::uint64_t bytes);
    id<MTLBuffer> scratch();

    id<MTLDevice> device_ = nil;
    PassDumpConfig config_;
    id<MTLComputePipelineState> float_pipeline_ = nil;
    id<MTLComputePipelineState> volume_pipeline_ = nil;
    id<MTLComputePipelineState> copy_pipeline_ = nil;
    std::uint32_t buffer_count_ = 0;
    std::unordered_set<const void*> inputs_seen_;
    id<MTLComputePipelineState> uint_pipeline_ = nil;
    id<MTLComputePipelineState> int_pipeline_ = nil;
    id<MTLComputePipelineState> depth_pipeline_ = nil;
    id<MTLBuffer> scratch_ = nil;
    bool pipelines_ready_ = false;
    bool pipelines_ok_ = false;
    bool directory_ready_ = false;
    bool stopped_ = false;
    bool budget_warned_ = false;
    std::uint32_t images_ = 0;
    std::uint64_t bytes_ = 0;
    bool index_valid_ = false;
    std::uint64_t index_frame_ = 0;
    std::uint32_t index_ = 0;
};

// Encodes every accepted job into `commands`. Failures are ignored: a dump
// must not fail the guest submission.
void encode_pass_dump_jobs(id<MTL4CommandBuffer> commands, id<MTLResidencySet> residency,
                           NSMutableArray<id>* retained, const std::vector<PassDumpJob>& jobs);

// Queues PNG and text output on a serial background queue. The lines retain
// their pixel buffers until that write finishes.
void publish_pass_dumps(std::vector<PassDumpLine> lines);

} // namespace armigo::platform::metal::nvn_driver
