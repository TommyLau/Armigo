// ============================================================================
// Armigo Platform - Apple Platform Integration
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================
#include <armigo/common/diagnostics.h>
// Texture caching owns overlap authority, compatible views, depth and colour
// aliases, private Metal storage, depth/stencil transfers, and GPU block-linear
// staging. Versioning, admission, retention, and transfers remain in Armigo.
// CPU revalidation records a write stamp and 64-bit content hash per layer and
// mip unit. Only written unit ranges are uploaded again, using exact byte spans.

#include "texture_cache.h"
#include "bindings.h"
#include "texture_residency.h"
#include "texture_format.h"
#include "texture_swizzle_gpu.h"
#include "device_context.h"
#include "../../graphics/texture_residency.h"
#include "../../graphics/content_hash.h"

#import <Foundation/Foundation.h>
#include <map>
#include <set>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <limits>
#include <new>

namespace armigo::platform::metal {
namespace {
using namespace graphics;
RendererError failure(const char* detail, CommandStatus status = CommandStatus::Unsupported) {
    return RendererError({status, detail});
}
bool overlaps(BackingRange a, BackingRange b) {
    return a.backing == b.backing && a.offset < b.offset + b.size && b.offset < a.offset + a.size;
}
MTLTextureType native_type(TextureType type) {
    switch (type) {
        case TextureType::Texture1D: return MTLTextureType1D;
        case TextureType::Texture1DArray: return MTLTextureType1DArray;
        case TextureType::Texture3D: return MTLTextureType3D;
        case TextureType::Texture2DArray: return MTLTextureType2DArray;
        case TextureType::Cubemap: return MTLTextureTypeCube;
        case TextureType::CubemapArray: return MTLTextureTypeCubeArray;
        default: return MTLTextureType2D;
    }
}
const char* native_type_name(TextureType type) {
    switch (type) {
        case TextureType::Texture1D: return "1D";
        case TextureType::Texture1DArray: return "1D array";
        case TextureType::Texture3D: return "3D";
        case TextureType::Texture2DArray: return "2D array";
        case TextureType::Cubemap: return "cube";
        case TextureType::CubemapArray: return "cube array";
        default: return "2D";
    }
}
// Metal permits a view to change only the arrayness inside one dimension
// family: 1D<->1D array, 2D<->2D array (rectangle storage is native 2D) and
// cube<->cube array in complete six-face groups, matching the host views
// same-dimension target views require.
bool legal_texture_view(TextureType storage, TextureType view,
                        std::uint32_t first_layer, std::uint32_t layer_count,
                        std::uint32_t storage_layers) {
    if (storage == TextureType::Texture2DRect) storage = TextureType::Texture2D;
    const auto family = [](TextureType type) -> unsigned {
        switch (type) {
        case TextureType::Texture1D: case TextureType::Texture1DArray: return 1;
        case TextureType::Texture2D: case TextureType::Texture2DArray: return 2;
        case TextureType::Cubemap: case TextureType::CubemapArray: return 6;
        default: return 0;
        }
    };
    const unsigned storage_family = family(storage), view_family = family(view);
    if (!storage_family || storage_family != view_family) return false;
    if (!layer_count || first_layer >= storage_layers ||
        layer_count > storage_layers - first_layer)
        return false;
    switch (view) {
        case TextureType::Texture1D:
        case TextureType::Texture2D:
            return layer_count == 1;
        case TextureType::Texture1DArray:
        case TextureType::Texture2DArray:
            return true;
        case TextureType::Cubemap:
            return layer_count == 6 && first_layer % 6 == 0;
        case TextureType::CubemapArray:
            return layer_count % 6 == 0 && first_layer % 6 == 0;
        default: return false;
    }
}
MTLTextureSwizzle native_swizzle(TextureSwizzle swizzle) {
    switch (swizzle) {
        case TextureSwizzle::Zero: return MTLTextureSwizzleZero;
        case TextureSwizzle::Red: return MTLTextureSwizzleRed;
        case TextureSwizzle::Green: return MTLTextureSwizzleGreen;
        case TextureSwizzle::Blue: return MTLTextureSwizzleBlue;
        case TextureSwizzle::Alpha: return MTLTextureSwizzleAlpha;
        default: return MTLTextureSwizzleOne;
    }
}

// Transfers are synchronous coordinator operations on the retained device's
// queue. The buffer is reused only after feedback; timeout drops the pool's
// reference while feedback retains every allocation through GPU quiescence.
struct Transfers {
    std::shared_ptr<DeviceContext> context;
    std::shared_ptr<TextureCacheStatistics> statistics;
    id<MTLBuffer> staging = nil;
    // GPU swizzle pipelines are optional: without them every transfer takes
    // the CPU conversion path explicitly.
    std::shared_ptr<GpuTextureSwizzle> swizzle;

    RendererResult<id<MTLBuffer>> buffer(std::size_t size) {
        if (!size || size > max_texture_layout_size * 2)
            return failure("texture transfer extent exceeds policy limit");
        if (!staging || staging.length < size)
            staging = [context->device() newBufferWithLength:size options:MTLResourceStorageModeShared];
        if (!staging) return failure("texture staging allocation failed");
        return staging;
    }
    RendererStatus execute(id<MTLTexture> texture, id<MTLBuffer> buffer,
                           const std::function<void(id<MTL4ComputeCommandEncoder>)>& encode,
                           std::shared_ptr<const void> owner,
                           NSArray<id<MTLAllocation>>* extra = nil) {
        auto device = context->device();
        id<MTL4CommandAllocator> allocator = [device newCommandAllocator];
        id<MTL4CommandBuffer> commands = [device newCommandBuffer];
        auto* descriptor = [MTLResidencySetDescriptor new];
        descriptor.initialCapacity = 2 + extra.count;
        NSError* error = nil;
        id<MTLResidencySet> residency = [device newResidencySetWithDescriptor:descriptor error:&error];
        if (!allocator || !commands || !residency)
            return failure("texture transfer command allocation failed");
        [residency addAllocation:texture];
        [residency addAllocation:buffer];
        for (id<MTLAllocation> allocation in extra) [residency addAllocation:allocation];
        [residency commit];
        [commands beginCommandBufferWithAllocator:allocator];
        [commands useResidencySet:residency];
        id<MTL4ComputeCommandEncoder> encoder = [commands computeCommandEncoder];
        if (!encoder) {
            [commands endCommandBuffer];
            return failure("texture transfer encoder allocation failed");
        }
        // Swizzle dispatches run on the dispatch stage; the queue-side
        // barriers must order them like the blit copies.
        [encoder barrierAfterQueueStages:MTLStageAll
                            beforeStages:(MTLStageBlit | MTLStageDispatch)
                       visibilityOptions:MTL4VisibilityOptionDevice];
        encode(encoder);
        [encoder barrierAfterStages:(MTLStageBlit | MTLStageDispatch)
                 beforeQueueStages:MTLStageAll
                 visibilityOptions:MTL4VisibilityOptionDevice];
        [encoder endEncoding];
        [commands endCommandBuffer];
        NSArray* retained = extra ? @[device, context->queue(), allocator, commands, residency, texture, buffer, extra]
                                  : @[device, context->queue(), allocator, commands, residency, texture, buffer];
        dispatch_semaphore_t completed = dispatch_semaphore_create(0);
        __block bool failed = false;
        auto* options = [MTL4CommitOptions new];
        [options addFeedbackHandler:^(id<MTL4CommitFeedback> feedback) {
            static_cast<void>(retained.count);
            static_cast<void>(owner);
            failed = feedback.error != nil;
            dispatch_semaphore_signal(completed);
        }];
        id<MTL4CommandBuffer> submitted[] = {commands};
        [context->queue() commit:submitted count:1 options:options];
        if (dispatch_semaphore_wait(completed, dispatch_time(DISPATCH_TIME_NOW, 10 * NSEC_PER_SEC))) {
            staging = nil;
            return failure("texture transfer timed out", CommandStatus::Stopped);
        }
        if (failed) return failure("texture transfer GPU execution failed", CommandStatus::Stopped);
        return renderer_success();
    }
};

struct PlanePlacement {
    std::uint32_t level, layer;
    bool stencil;
    std::size_t offset, row_bytes, image_bytes, payload_row;
};
struct Allocation : std::enable_shared_from_this<Allocation> {
    TextureStorageKey key;
    TextureLayoutDescription layout;
    // Established guest bytes: the readback merge base that keeps retained
    // views byte-exact after their storage is superseded. It is never used
    // for revalidation comparisons; certificates decide those.
    std::vector<std::byte> baseline;
    // Coherence certificates use one registry write stamp per established layer/level
    // unit plus a content hash of its guest bytes. Units are layer/level
    // spans covering the whole allocation, indexed layer * levels + level.
    struct Certificate {
        std::uint64_t stamp = 0, hash = 0;
    };
    std::vector<Certificate> certificates;
    std::uint64_t stamp = 0;
    std::vector<DirtyRange> versions;
    bool has_gpu_snapshot = false;
    std::uint64_t epoch = 0;
    std::uint64_t resident_cost = 0;
    bool invalidated = false;
    // Cache references are residency accounting owners, not consumer pins.
    std::size_t cache_owners = 0;
    std::shared_ptr<Transfers> transfers;
    id<MTLTexture> texture = nil;
    std::vector<PlanePlacement> planes;
    std::size_t transfer_size = 0;
    struct CachedView {
        TextureViewKey key;
        id<MTLTexture> texture;
    };
    std::vector<CachedView> views;
    // Units last brought in from separately rendered producers: the producer
    // identity, the GPU version copied and the shared copy plan.
    struct Composed {
        const void* producer = nullptr;
        ResourceVersion version;
        std::shared_ptr<const SubresourceCopy> plan;
    };
    std::map<std::uint32_t, Composed> composed;

    void plan_planes() {
        const auto info = texture_format_info(key.descriptor.format);
        for (std::uint32_t layer = 0; layer < layout.layers; ++layer) {
            for (std::uint32_t level = 0; level < layout.levels; ++level) {
                const auto& mip = layout.mips[level];
                for (unsigned plane = 0; plane < (info.stencil ? 2u : 1u); ++plane) {
                    const auto unit = plane ? 1u : info.depth ?
                        (key.descriptor.format == TextureFormat::D16Unorm ? 2u : 4u) : info.bytes_per_block;
                    const auto payload_row = mip.block_columns * unit;
                    const auto row = (payload_row + 255u) & ~std::size_t{255};
                    const auto image = row * mip.block_rows;
                    planes.push_back({level, layer, plane != 0, transfer_size, row, image, payload_row});
                    transfer_size += image * mip.depth;
                }
            }
        }
    }
    // Guest byte span of one layer/level unit inside the storage, extended to
    // the next level tile so that the units cover each layer exactly, padding
    // included; the whole-comparison granularity is preserved.
    std::pair<std::size_t, std::size_t> unit_span(std::uint32_t layer, std::uint32_t level) const {
        const auto base = std::size_t{layer} * layout.layer_stride;
        const auto start = base + layout.mips[level].offset;
        const auto end = level + 1 < layout.levels ? base + layout.mips[level + 1].offset
                                                   : base + layout.layer_stride;
        return {std::min(start, std::size_t{key.bytes.size}),
                std::min(std::max(end, start), std::size_t{key.bytes.size})};
    }
    BackingRange unit_bytes(std::uint32_t layer, std::uint32_t level) const {
        const auto [start, end] = unit_span(layer, level);
        return {key.bytes.backing, key.bytes.offset + start, end - start};
    }
    void establish_certificates(std::uint64_t taken_stamp, std::span<const std::byte> bytes) {
        stamp = taken_stamp;
        certificates.assign(std::size_t{layout.layers} * layout.levels, {});
        for (std::uint32_t layer = 0; layer < layout.layers; ++layer)
            for (std::uint32_t level = 0; level < layout.levels; ++level)
                refresh_certificate(layer, level, taken_stamp, bytes);
    }
    void refresh_certificate(std::uint32_t layer, std::uint32_t level,
                             std::uint64_t taken_stamp, std::span<const std::byte> bytes) {
        const auto [start, end] = unit_span(layer, level);
        auto& certificate = certificates[std::size_t{layer} * layout.levels + level];
        certificate.stamp = taken_stamp;
        certificate.hash = content_hash64(bytes.subspan(start, end - start));
    }
    void encode_transfer(id<MTL4ComputeCommandEncoder> encoder, id<MTLBuffer> buffer, bool upload,
                         const std::vector<std::uint32_t>* units = nullptr) const {
        const bool packed = texture.pixelFormat == MTLPixelFormatDepth32Float_Stencil8;
        for (std::size_t index = 0; index < planes.size(); ++index) {
            if (units) {
                const auto& plane = planes[index];
                const auto unit = std::uint32_t{plane.layer * layout.levels + plane.level};
                if (std::find(units->begin(), units->end(), unit) == units->end()) continue;
            }
            const auto& plane = planes[index];
            const auto& mip = layout.mips[plane.level];
            auto option = packed ? (plane.stencil ? MTLBlitOptionStencilFromDepthStencil :
                                         MTLBlitOptionDepthFromDepthStencil) : MTLBlitOptionNone;
            auto size = MTLSizeMake(mip.width, mip.height, mip.depth);
            if (upload) {
                [encoder copyFromBuffer:buffer sourceOffset:plane.offset
                     sourceBytesPerRow:plane.row_bytes sourceBytesPerImage:plane.image_bytes
                            sourceSize:size toTexture:texture destinationSlice:plane.layer
                      destinationLevel:plane.level destinationOrigin:MTLOriginMake(0, 0, 0) options:option];
            } else {
                [encoder copyFromTexture:texture sourceSlice:plane.layer sourceLevel:plane.level
                            sourceOrigin:MTLOriginMake(0, 0, 0) sourceSize:size toBuffer:buffer
                       destinationOffset:plane.offset destinationBytesPerRow:plane.row_bytes
                destinationBytesPerImage:plane.image_bytes options:option];
            }
        }
    }
    // GPU swizzle transfers cover block-linear color formats only: depth and
    // stencil still need the CPU plane packing, and pitch storage has no GOB
    // math for the kernel. The guest extent must fit the transferred image.
    bool swizzle_gpu_eligible(std::span<const std::byte> image) const {
        if (!transfers->swizzle) return false;
        if (layout.storage != TextureStorageLayout::BlockLinear) return false;
        if (layout.total_size > image.size()) return false;
        const auto info = texture_format_info(key.descriptor.format);
        return !info.depth && !info.stencil;
    }
    // Derives one plane's kernel geometry from the calculated layout. gobs_x
    // comes from the level's slice addressing so the sparse-tile column
    // alignment of texture_layout.cpp is honored without exposing it.
    bool swizzle_extent(const PlanePlacement& plane, std::uint64_t guest_base,
                        TextureSwizzleExtent& extent) const {
        const auto& mip = layout.mips[plane.level];
        extent.guest_offset = guest_base +
            std::uint64_t{plane.layer} * layout.layer_stride + mip.offset;
        extent.guest_size = mip.size;
        extent.linear_offset = plane.offset;
        extent.linear_pitch = plane.row_bytes;
        extent.row_bytes = plane.payload_row;
        extent.rows = mip.block_rows;
        extent.slices = mip.depth;
        extent.gob_block_height_log2 = mip.gob_block_height_log2;
        extent.gob_block_depth_log2 = mip.gob_block_depth_log2;
        const std::size_t block_height = std::size_t{1} << mip.gob_block_height_log2;
        const std::size_t block_depth = std::size_t{1} << mip.gob_block_depth_log2;
        const std::size_t robs_y =
            (std::size_t{mip.block_rows} + block_height * texture_gob_height_rows - 1) /
            (block_height * texture_gob_height_rows);
        const std::size_t gob_row_bytes =
            robs_y * texture_gob_size * block_height * block_depth;
        if (!gob_row_bytes || mip.slice_size % gob_row_bytes) return false;
        extent.gobs_x = static_cast<std::uint32_t>(mip.slice_size / gob_row_bytes);
        return extent.gobs_x != 0;
    }
    // Extents for every plane when units is null, otherwise only the listed
    // layer/level units' planes; a plane the kernel cannot address keeps the
    // whole transfer on the CPU path.
    bool swizzle_extents(std::vector<TextureSwizzleExtent>& extents,
                         const std::vector<std::uint32_t>* units = nullptr) const {
        extents.clear();
        extents.reserve(planes.size());
        for (const auto& plane : planes) {
            if (units) {
                const auto unit = std::uint32_t{plane.layer * layout.levels + plane.level};
                if (std::find(units->begin(), units->end(), unit) == units->end()) continue;
            }
            TextureSwizzleExtent extent;
            if (plane.stencil || !swizzle_extent(plane, transfer_size, extent))
                return false;
            extents.push_back(extent);
        }
        return true;
    }
    // Guest byte range of the staging image the extents cover; one memcpy
    // moves it for both full and partial selections.
    std::pair<std::uint64_t, std::uint64_t> staged_guest_range(
        const std::vector<TextureSwizzleExtent>& extents) const {
        auto start = std::numeric_limits<std::uint64_t>::max();
        auto end = std::uint64_t{0};
        for (const auto& extent : extents) {
            start = std::min(start, extent.guest_offset);
            end = std::max(end, extent.guest_offset + extent.guest_size);
        }
        return {start, std::max(end, start)};
    }
    RendererStatus upload_gpu(std::span<const std::byte> source,
                              const std::vector<TextureSwizzleExtent>& extents,
                              const std::vector<std::uint32_t>* units) {
        const std::size_t guest_base = transfer_size;
        auto obtained = transfers->buffer(transfer_size + source.size());
        if (auto* error = std::get_if<RendererError>(&obtained)) return *error;
        id<MTLBuffer> buffer = std::get<id<MTLBuffer>>(obtained);
        // Reserve before the parameters buffer enters the residency set.
        if (auto ready = transfers->swizzle->reserve(extents.size());
            std::holds_alternative<RendererError>(ready))
            return ready;
        // One copy of the needed guest bytes replaces the per-level CPU
        // deswizzle; the compute pass tiles every selected level into its
        // linear plane image on the GPU before the buffer-to-texture copies
        // run. A partial selection stages only the covering range of the
        // selected extents; a full upload copies the whole image at once.
        const auto [staged_start, staged_end] = staged_guest_range(extents);
        if (staged_end > staged_start)
            std::memcpy(static_cast<std::byte*>(buffer.contents) + staged_start,
                        source.data() + (staged_start - guest_base),
                        staged_end - staged_start);
        // Selected planes may skip whole levels, so byte accounting follows
        // the exact unit spans the CPU path reports for the same selection.
        std::uint64_t uploaded_bytes = 0;
        if (units) {
            for (const auto unit : *units) {
                const auto [start, end] =
                    unit_span(unit / layout.levels, unit % layout.levels);
                uploaded_bytes += end - start;
            }
        } else {
            uploaded_bytes = source.size();
        }
        // The result variant defaults to its error alternative; encode
        // failures are stored through the encode callbacks only.
        RendererStatus encode_error = std::monostate{};
        auto status = transfers->execute(texture, buffer, [&](auto encoder) {
            for (std::size_t slot = 0; slot < extents.size(); ++slot) {
                auto encoded = transfers->swizzle->encode_deswizzle(encoder, buffer,
                    buffer, static_cast<std::uint32_t>(slot), extents[slot]);
                if (auto* error = std::get_if<RendererError>(&encoded)) {
                    encode_error = *error;
                    return;
                }
            }
            [encoder barrierAfterEncoderStages:MTLStageDispatch
                           beforeEncoderStages:MTLStageBlit
                             visibilityOptions:MTL4VisibilityOptionDevice];
            encode_transfer(encoder, buffer, true, units);
        }, shared_from_this(), @[transfers->swizzle->parameters_buffer()]);
        if (auto* error = std::get_if<RendererError>(&encode_error)) return *error;
        if (std::holds_alternative<RendererError>(status)) return status;
        ++transfers->statistics->uploads;
        transfers->statistics->upload_bytes += uploaded_bytes;
        return renderer_success();
    }
    // Full upload when units is null; otherwise only the listed layer/level
    // units are converted and encoded, leaving the other units resident.
    RendererStatus upload(std::span<const std::byte> source,
                          const std::vector<std::uint32_t>* units = nullptr) {
        // Unsupported geometry takes the CPU conversion path explicitly;
        // once the GPU pass is committed its failures are reported.
        std::vector<TextureSwizzleExtent> extents;
        if (swizzle_gpu_eligible(source) && swizzle_extents(extents, units))
            return upload_gpu(source, extents, units);
        return upload_cpu(source, units);
    }
    RendererStatus upload_cpu(std::span<const std::byte> source,
                              const std::vector<std::uint32_t>* units) {
        auto obtained = transfers->buffer(transfer_size);
        if (auto* error = std::get_if<RendererError>(&obtained)) return *error;
        id<MTLBuffer> buffer = std::get<id<MTLBuffer>>(obtained);
        std::memset(buffer.contents, 0, transfer_size);
        std::uint64_t uploaded_bytes = 0;
        for (std::uint32_t layer = 0; layer < layout.layers; ++layer) {
            for (std::uint32_t level = 0; level < layout.levels; ++level) {
                const auto unit = std::uint32_t{layer * layout.levels + level};
                if (units && std::find(units->begin(), units->end(), unit) == units->end()) continue;
                const auto [start, end] = unit_span(layer, level);
                uploaded_bytes += end - start;
                auto linear = convert_level_to_linear(layout, source, level, layer);
                auto* data = std::get_if<TextureLevelData>(&linear);
                if (!data) {
                    const auto& error = std::get<TextureLayoutError>(linear);
                    const auto detail = std::string("texture upload layout conversion failed: ") + error.message() +
                        " (value " + std::to_string(error.value) + ", expected " + std::to_string(error.expected) +
                        "; format " + std::to_string(static_cast<unsigned>(key.descriptor.format)) +
                        ", " + std::to_string(key.descriptor.width) + "x" + std::to_string(key.descriptor.height) +
                        "x" + std::to_string(key.descriptor.depth) + ", layers " + std::to_string(layout.layers) +
                        ", levels " + std::to_string(layout.levels) + ", level " + std::to_string(level) +
                        ", layer " + std::to_string(layer) + ", source " + std::to_string(source.size()) +
                        " bytes, layout " + std::to_string(layout.total_size) + " bytes)";
                    return failure(detail.c_str());
                }
                auto converted = metal_texture_planes(key.descriptor.format, data->bytes);
                auto* values = std::get_if<MetalTexturePlanes>(&converted);
                if (!values) return failure("texture upload format conversion failed");
                for (const auto& plane : planes) {
                    if (plane.layer != layer || plane.level != level) continue;
                    const auto& bytes = plane.stencil ? values->stencil : values->color_or_depth;
                    const auto& mip = layout.mips[level];
                    for (std::size_t z = 0; z < mip.depth; ++z)
                        for (std::size_t y = 0; y < mip.block_rows; ++y)
                            std::memcpy(static_cast<std::byte*>(buffer.contents) + plane.offset +
                                z * plane.image_bytes + y * plane.row_bytes,
                                bytes.data() + (z * mip.block_rows + y) * plane.payload_row,
                                plane.payload_row);
                }
            }
        }
        auto status = transfers->execute(texture, buffer,
            [&](auto encoder) { encode_transfer(encoder, buffer, true, units); }, shared_from_this());
        if (std::holds_alternative<RendererError>(status)) return status;
        ++transfers->statistics->uploads;
        transfers->statistics->upload_bytes += uploaded_bytes;
        return renderer_success();
    }
    // The guest bytes replace the former resident CPU baseline as the merge
    // base: read under the coordinator's stopped-guest contract, they equal
    // what the baseline held wherever no newer CPU write exists.
    RendererResult<std::vector<std::byte>> download(std::span<const std::byte> base) const {
        // Same explicit CPU fallback contract as upload().
        std::vector<TextureSwizzleExtent> extents;
        if (swizzle_gpu_eligible(base) && swizzle_extents(extents))
            return download_gpu(extents, base);
        return download_cpu(base);
    }
    RendererResult<std::vector<std::byte>> download_gpu(
        const std::vector<TextureSwizzleExtent>& extents,
        std::span<const std::byte> base) const {
        const std::size_t guest_base = transfer_size;
        auto obtained = transfers->buffer(transfer_size + base.size());
        if (auto* error = std::get_if<RendererError>(&obtained)) return *error;
        id<MTLBuffer> buffer = std::get<id<MTLBuffer>>(obtained);
        // Reserve before the parameters buffer enters the residency set.
        if (auto ready = transfers->swizzle->reserve(extents.size());
            auto* error = std::get_if<RendererError>(&ready))
            return *error;
        // Padding inside GOBs stays untouched by the kernel, so the staged
        // guest image starts from the merge base and the swizzle pass
        // overwrites exactly the covered level bytes.
        std::memcpy(static_cast<std::byte*>(buffer.contents) + guest_base,
                    base.data(), base.size());
        // The result variant defaults to its error alternative; encode
        // failures are stored through the encode callbacks only.
        RendererStatus encode_error = std::monostate{};
        auto status = transfers->execute(texture, buffer, [&](auto encoder) {
            encode_transfer(encoder, buffer, false);
            [encoder barrierAfterEncoderStages:MTLStageBlit
                           beforeEncoderStages:MTLStageDispatch
                             visibilityOptions:MTL4VisibilityOptionDevice];
            for (std::size_t slot = 0; slot < extents.size(); ++slot) {
                auto encoded = transfers->swizzle->encode_swizzle(encoder, buffer,
                    buffer, static_cast<std::uint32_t>(slot), extents[slot]);
                if (auto* error = std::get_if<RendererError>(&encoded)) {
                    encode_error = *error;
                    return;
                }
            }
        }, shared_from_this(), @[transfers->swizzle->parameters_buffer()]);
        if (auto* error = std::get_if<RendererError>(&encode_error)) return *error;
        if (auto* error = std::get_if<RendererError>(&status)) return *error;
        ++transfers->statistics->readbacks;
        std::vector<std::byte> result(base.size());
        std::memcpy(result.data(), static_cast<const std::byte*>(buffer.contents) + guest_base,
                    base.size());
        return result;
    }
    RendererResult<std::vector<std::byte>> download_cpu(std::span<const std::byte> base) const {
        auto obtained = transfers->buffer(transfer_size);
        if (auto* error = std::get_if<RendererError>(&obtained)) return *error;
        id<MTLBuffer> buffer = std::get<id<MTLBuffer>>(obtained);
        auto status = transfers->execute(texture, buffer, [&](auto encoder) { encode_transfer(encoder, buffer, false); }, shared_from_this());
        if (auto* error = std::get_if<RendererError>(&status)) return *error;
        ++transfers->statistics->readbacks;
        auto result = std::vector<std::byte>(base.begin(), base.end());
        for (std::uint32_t layer = 0; layer < layout.layers; ++layer) {
            for (std::uint32_t level = 0; level < layout.levels; ++level) {
                auto original = convert_level_to_linear(layout, base, level, layer);
                auto* data = std::get_if<TextureLevelData>(&original);
                if (!data) return failure("texture readback baseline conversion failed");
                MetalTexturePlanes values;
                for (const auto& plane : planes) {
                    if (plane.layer != layer || plane.level != level) continue;
                    auto& bytes = plane.stencil ? values.stencil : values.color_or_depth;
                    const auto& mip = layout.mips[level];
                    bytes.resize(plane.payload_row * mip.block_rows * mip.depth);
                    for (std::size_t z = 0; z < mip.depth; ++z)
                        for (std::size_t y = 0; y < mip.block_rows; ++y)
                            std::memcpy(bytes.data() + (z * mip.block_rows + y) * plane.payload_row,
                                static_cast<const std::byte*>(buffer.contents) + plane.offset +
                                z * plane.image_bytes + y * plane.row_bytes, plane.payload_row);
                }
                auto merged = metal_merge_texture_planes(key.descriptor.format, values, data->bytes);
                auto* linear = std::get_if<std::vector<std::byte>>(&merged);
                if (!linear) return failure("texture readback format conversion failed");
                auto written = write_level_from_linear(layout, *linear, result, level, layer);
                if (std::holds_alternative<TextureLayoutError>(written))
                    return failure("texture readback tiling conversion failed");
            }
        }
        return result;
    }
};

// A producer whose storage is exactly one layer/level unit of target in the
// same block-linear GOB layout: the unit and the texel extent both cover. A
// texel's bytes depend on the surface width only through its GOB columns, so
// a GOB-aligned producer wider than a small mip level still shares its texels.
struct SubresourceUnit {
    std::uint32_t layer = 0, level = 0, width = 0, height = 0;
};
std::optional<SubresourceUnit> subresource_unit(const TextureStorageKey& target_key,
                                               const TextureLayoutDescription& target_layout,
                                               const Allocation& producer) {
    struct {
        const TextureStorageKey& key;
        const TextureLayoutDescription& layout;
        std::pair<std::size_t, std::size_t> unit_span(std::uint32_t layer, std::uint32_t level) const {
            const auto base = std::size_t{layer} * layout.layer_stride;
            const auto start = base + layout.mips[level].offset;
            const auto end = level + 1 < layout.levels ? base + layout.mips[level + 1].offset
                                                       : base + layout.layer_stride;
            return {std::min(start, std::size_t{key.bytes.size}),
                    std::min(std::max(end, start), std::size_t{key.bytes.size})};
        }
    } target{target_key, target_layout};
    const auto& td = target.key.descriptor;
    const auto& pd = producer.key.descriptor;
    // Equal-size depth and colour encodings of the same texel bits (a shadow
    // cascade rendered as R16Unorm and sampled as a D16 array layer) share
    // their bytes too; their copy is staged through a buffer.
    const auto same_bits = [](TextureFormat a, TextureFormat b) {
        const auto pair = [&](TextureFormat x, TextureFormat y) {
            return (a == x && b == y) || (a == y && b == x);
        };
        return a == b || pair(TextureFormat::D16Unorm, TextureFormat::R16Unorm) ||
               pair(TextureFormat::D32Float, TextureFormat::R32Float);
    };
    if (!(target.key.bytes.backing == producer.key.bytes.backing) ||
        producer.key.bytes.offset < target.key.bytes.offset || !same_bits(td.format, pd.format) ||
        td.layout != TextureLayout::BlockLinear || pd.layout != TextureLayout::BlockLinear ||
        pd.levels != 1 || pd.layers != 1 || pd.depth != 1 || pd.base_level != 0 ||
        producer.layout.mips.size() != 1)
        return std::nullopt;
    const auto info = texture_format_info(td.format);
    if (info.block_width != 1 || info.block_height != 1 || info.stencil)
        return std::nullopt;
    const auto offset = producer.key.bytes.offset - target.key.bytes.offset;
    const auto& produced = producer.layout.mips[0];
    for (std::uint32_t layer = 0; layer < target.layout.layers; ++layer) {
        for (std::uint32_t level = 0; level < target.layout.levels; ++level) {
            const auto [start, end] = target.unit_span(layer, level);
            if (start != offset) continue;
            const auto& mip = target.layout.mips[level];
            const auto columns = [&](const auto& level_layout) {
                return (std::size_t{level_layout.block_columns} * info.bytes_per_block + 63) / 64;
            };
            if (producer.key.bytes.size > end - start || mip.depth != 1 ||
                mip.gob_block_height_log2 != produced.gob_block_height_log2 ||
                mip.gob_block_depth_log2 != produced.gob_block_depth_log2 ||
                columns(mip) != columns(produced))
                return std::nullopt;
            return SubresourceUnit{layer, level, std::min(mip.width, produced.width),
                                   std::min(mip.height, produced.height)};
        }
    }
    return std::nullopt;
}

class View final : public TextureResource {
public:
    std::shared_ptr<Allocation> owner;
    // Pins the admitted address-space mapping while a consumer retains a view.
    RetainedResource lease;
    std::weak_ptr<ResourceMemoryRegistry> memory;
    id<MTLTexture> native = nil;
    BackingRange coverage() const noexcept override { return owner->key.bytes; }
    id<MTLTexture> texture() const noexcept override { return native; }
    id<MTLTexture> allocation() const noexcept override { return owner->texture; }
    RendererStatus read_gpu(BackingRange range, std::span<std::byte> destination) const override {
        const auto bytes = coverage();
        if (range.backing != bytes.backing || range.offset < bytes.offset ||
            range.offset - bytes.offset > bytes.size || !range.size ||
            range.size > bytes.size - (range.offset - bytes.offset) || range.size != destination.size())
            return failure("texture readback range exceeds resident storage");
        try {
            auto result = owner->download(owner->baseline);
            if (auto* error = std::get_if<RendererError>(&result)) return *error;
            const auto& data = std::get<std::vector<std::byte>>(result);
            // Preserve the native allocation's exact depth after packing a CPU
            // copy. Only the registry's current authority may advance these
            // versions; a retained stale view can still be read by a caller.
            if (const auto registry = memory.lock()) {
                auto inspected = registry->inspect_storage(range);
                if (auto* error = std::get_if<RendererError>(&inspected)) return *error;
                const auto& current = std::get<std::vector<ResourceStorageRange>>(inspected);
                const bool authoritative = std::all_of(current.begin(), current.end(),
                    [&](const ResourceStorageRange& interval) {
                        const auto view = std::dynamic_pointer_cast<const View>(interval.storage);
                        return interval.dirty.gpu_dirty && view && view->owner == owner;
                    });
                if (authoritative && !current.empty()) {
                    std::vector<DirtyRange> updated;
                    for (const auto& prior : owner->versions) {
                        if (!overlaps(prior.bytes, range)) {
                            updated.push_back(prior);
                            continue;
                        }
                        if (prior.bytes.offset < range.offset) {
                            auto prefix = prior;
                            prefix.bytes.size = range.offset - prior.bytes.offset;
                            updated.push_back(prefix);
                        }
                        const auto end = range.offset + range.size;
                        const auto prior_end = prior.bytes.offset + prior.bytes.size;
                        if (prior_end > end) {
                            auto suffix = prior;
                            suffix.bytes.offset = end;
                            suffix.bytes.size = prior_end - end;
                            updated.push_back(suffix);
                        }
                    }
                    for (const auto& interval : current)
                        updated.push_back({interval.dirty.bytes, interval.dirty.version, false, false, {}});
                    std::sort(updated.begin(), updated.end(), [](const auto& a, const auto& b) {
                        return a.bytes.offset < b.bytes.offset;
                    });
                    std::size_t count = 0;
                    for (const auto& interval : updated) {
                        if (count && updated[count - 1].version == interval.version &&
                            updated[count - 1].bytes.offset + updated[count - 1].bytes.size ==
                                interval.bytes.offset) {
                            updated[count - 1].bytes.size += interval.bytes.size;
                        } else {
                            updated[count++] = interval;
                        }
                    }
                    updated.resize(count);
                    owner->versions = std::move(updated);
                    std::memcpy(owner->baseline.data() + range.offset - bytes.offset,
                                data.data() + range.offset - bytes.offset, destination.size());
                    // Re-establish only the certificates of units the resolved
                    // range covers, matching the merged baseline; outside units
                    // keep their prior proof.
                    const auto published = memory.lock() ? memory.lock()->cpu_write_stamp() : 0;
                    for (std::uint32_t layer = 0; layer < owner->layout.layers; ++layer)
                        for (std::uint32_t level = 0; level < owner->layout.levels; ++level) {
                            const auto unit = owner->unit_bytes(layer, level);
                            if (overlaps(unit, range))
                                owner->refresh_certificate(layer, level, published, data);
                        }
                    owner->has_gpu_snapshot = true;
                }
            }
            std::memcpy(destination.data(), data.data() + range.offset - bytes.offset, destination.size());
            return renderer_success();
        } catch (const std::bad_alloc&) { return failure("texture readback allocation failed"); }
    }
};

// Calls agree(prior, current) for every overlapping pair. Registry intervals
// of one backing arrive sorted and disjoint, so a merge walk visits each pair
// once; mixed backings or unsorted input fall back to the pairwise scan.
template <class Agree>
bool overlapping_pairs_agree(const std::vector<DirtyRange>& old,
                             const std::vector<ResourceStorageRange>& now, Agree agree) {
    const auto ordered = [](const auto& ranges, auto bytes) {
        for (std::size_t i = 1; i < ranges.size(); ++i) {
            const auto& previous = bytes(ranges[i - 1]);
            const auto& next = bytes(ranges[i]);
            if (!(next.backing == previous.backing) ||
                next.offset < previous.offset + previous.size) return false;
        }
        return true;
    };
    const auto old_bytes = [](const DirtyRange& range) -> const BackingRange& { return range.bytes; };
    const auto now_bytes = [](const ResourceStorageRange& range) -> const BackingRange& {
        return range.dirty.bytes;
    };
    if (old.empty() || now.empty()) return true;
    if (!(old.front().bytes.backing == now.front().dirty.bytes.backing) ||
        !ordered(old, old_bytes) || !ordered(now, now_bytes)) {
        for (const auto& current : now)
            for (const auto& prior : old)
                if (overlaps(current.dirty.bytes, prior.bytes) && !agree(prior, current)) return false;
        return true;
    }
    std::size_t first = 0;
    for (const auto& current : now) {
        const auto& bytes = current.dirty.bytes;
        while (first < old.size() && old[first].bytes.offset + old[first].bytes.size <= bytes.offset)
            ++first;
        for (auto index = first; index < old.size() && old[index].bytes.offset < bytes.offset + bytes.size;
             ++index)
            if (!agree(old[index], current)) return false;
    }
    return true;
}
bool same_authoritative_versions(const std::vector<DirtyRange>& old,
                                 const std::vector<ResourceStorageRange>& now) {
    return overlapping_pairs_agree(old, now, [](const DirtyRange& prior, const ResourceStorageRange& current) {
        if (current.dirty.version.cpu_write != prior.version.cpu_write) return false;
        // Dirty intervals were verified to belong to this allocation. Clean
        // intervals may contain a resolved write from an incompatible alias;
        // those bytes must still match the version originally uploaded.
        return current.dirty.gpu_dirty || current.dirty.version.gpu_write == prior.version.gpu_write;
    });
}
bool same_versions(const std::vector<DirtyRange>& old, const std::vector<ResourceStorageRange>& now) {
    return overlapping_pairs_agree(old, now, [](const DirtyRange& prior, const ResourceStorageRange& current) {
        return current.dirty.version == prior.version;
    });
}
std::vector<DirtyRange> versions(const std::vector<ResourceStorageRange>& ranges) {
    std::vector<DirtyRange> result;
    for (const auto& range : ranges) result.push_back({range.dirty.bytes, range.dirty.version, false, false, {}});
    return result;
}

// Hash the layer/level units whose certificate stamps the tracker still
// suspects; returns the units whose guest bytes no longer match and advances
// every verified unit's stamp to the snapshot stamp (taken before the read).
// A whole-allocation stamp proof skips every unit without a byte access.
std::vector<std::uint32_t> stale_units(Allocation& allocation,
                                       const ResourceMemoryRegistry& memory,
                                       std::span<const std::byte> snapshot,
                                       std::uint64_t snapshot_stamp,
                                       std::span<const BackingRange> gpu_owned = {}) {
    if (!memory.cpu_written_since(allocation.key.bytes, allocation.stamp)) return {};
    std::vector<std::uint32_t> changed;
    const auto levels = allocation.layout.levels;
    for (std::uint32_t layer = 0; layer < allocation.layout.layers; ++layer) {
        for (std::uint32_t level = 0; level < levels; ++level) {
            const auto index = std::size_t{layer} * levels + level;
            // GPU-authoritative units cannot hold a CPU store this check
            // could see; their snapshot bytes were never read.
            const auto unit = allocation.unit_bytes(layer, level);
            if (std::ranges::any_of(gpu_owned, [&](const BackingRange& owned) { return overlaps(owned, unit); }) ||
                !memory.cpu_written_since(unit, allocation.certificates[index].stamp))
                continue;
            const auto [start, end] = allocation.unit_span(layer, level);
            if (content_hash64(snapshot.subspan(start, end - start)) !=
                allocation.certificates[index].hash) {
                changed.push_back(static_cast<std::uint32_t>(index));
            } else {
                allocation.certificates[index].stamp = snapshot_stamp;
            }
        }
    }
    return changed;
}

// ARMIGO_TRACE_TEXTURE_REVALIDATION: a summary line every 500 hashed CPU
// revalidations (bytes snapshotted, units changed, time, largest extent).
void trace_revalidation(std::uint64_t extent, std::size_t changed, std::uint32_t units,
                        std::chrono::steady_clock::duration elapsed) {
    static const bool enabled = std::getenv("ARMIGO_TRACE_TEXTURE_REVALIDATION") != nullptr;
    if (!enabled) return;
    static std::uint64_t count = 0, bytes = 0, changed_units = 0, total_units = 0, largest = 0, ns = 0;
    ++count;
    bytes += extent;
    changed_units += changed;
    total_units += units;
    largest = std::max(largest, extent);
    ns += std::uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(elapsed).count());
    if (count % 500) return;
    armigo::diag::opt_in( "texture_revalidation count=%llu snapshot_mb=%llu changed_units=%llu units=%llu "
        "largest_kb=%llu ms=%llu\n", (unsigned long long)count, (unsigned long long)(bytes >> 20),
        (unsigned long long)changed_units, (unsigned long long)total_units,
        (unsigned long long)(largest >> 10), (unsigned long long)(ns / 1000000));
}

class Cache final : public TextureCache {
public:
    std::shared_ptr<DeviceContext> device;
    std::shared_ptr<ResourceMemoryRegistry> memory;
    std::shared_ptr<TextureCacheStatistics> statistics = std::make_shared<TextureCacheStatistics>();
    std::shared_ptr<Transfers> transfers;
    std::vector<std::shared_ptr<Allocation>> allocations;
    // Cached allocations by guest byte range, in insertion order: compatible
    // storage always names the same bytes, so lookups visit one bucket instead
    // of every allocation. Non-owning; mirrors allocations.
    using BytesKey = std::tuple<std::uint64_t, std::uint64_t, std::uint64_t, std::uint64_t, std::uint64_t>;
    static BytesKey bytes_key(const BackingRange& bytes) {
        return {bytes.backing.owner, bytes.backing.allocation, bytes.backing.generation, bytes.offset, bytes.size};
    }
    std::map<BytesKey, std::vector<Allocation*>> by_bytes;
    void index_add(Allocation* allocation) { by_bytes[bytes_key(allocation->key.bytes)].push_back(allocation); }
    void index_remove(Allocation* allocation) {
        const auto found = by_bytes.find(bytes_key(allocation->key.bytes));
        if (found == by_bytes.end()) return;
        std::erase(found->second, allocation);
        if (found->second.empty()) by_bytes.erase(found);
    }
    std::uint64_t budget;

    struct FastInterval {
        BackingRange bytes;
        ResourceVersion version;
        SubmissionId writer;
        bool cpu_dirty = false, gpu_dirty = false, has_storage = false;
        std::weak_ptr<const ResourceMemoryStorage> storage;
    };
    struct FastEntry {
        bool occupied = false;
        TextureViewKey key;
        MappingId mapping;
        std::uint64_t address = 0, address_generation = 0, allocation_epoch = 0;
        std::uint64_t stamp = 0, generation = 0;
        ResourceAccess access = ResourceAccess::Read;
        TextureAliasScope alias = TextureAliasScope::None;
        ResourceVersion uploaded_version;
        bool invalidated = false;
        std::weak_ptr<Allocation> allocation;
        std::weak_ptr<const ResourceLease> resident_view;
        std::vector<FastInterval> state;
        std::vector<std::weak_ptr<const ResourceLease>> transfers;
    };
    static constexpr std::size_t fast_entry_count = 1024;
    static constexpr std::size_t fast_interval_limit = 16;
    static constexpr std::size_t fast_transfer_limit = 16;
    std::unordered_map<std::uint64_t, FastEntry> fast_entries;
    std::uint64_t fast_generation = 1, fast_hits = 0;

    void invalidate_fast_entries() noexcept {
        if (fast_generation == std::numeric_limits<std::uint64_t>::max()) {
            fast_entries.clear();
            fast_generation = 1;
        } else {
            ++fast_generation;
        }
    }

    // Hash collisions replace the earlier shortcut; full request equality
    // still gates every hit. This table owns no native allocations.
    static std::uint64_t fast_key(const TextureRequest& request) noexcept {
        const auto& bytes = request.key.storage.bytes;
        std::uint64_t hash = 0;
        const auto mix = [&](std::uint64_t value) {
            hash ^= value + 0x9e3779b97f4a7c15ull + (hash << 6) + (hash >> 2);
        };
        mix(bytes.backing.owner); mix(bytes.backing.allocation); mix(bytes.backing.generation);
        mix(bytes.offset); mix(bytes.size); mix(request.key.storage.descriptor.address);
        mix(static_cast<unsigned>(request.key.storage.descriptor.format));
        mix(request.key.first_level); mix(request.key.first_layer);
        mix(request.key.level_count); mix(request.key.layer_count);
        for (auto swizzle : request.key.swizzle) mix(static_cast<unsigned>(swizzle));
        mix(static_cast<unsigned>(request.access)); mix(static_cast<unsigned>(request.alias));
        return hash;
    }

    static bool same_fast_request(const FastEntry& entry,
                                  const TextureRequest& request) noexcept {
        return entry.occupied && entry.key == request.key &&
               entry.key.storage.descriptor == request.key.storage.descriptor &&
               entry.mapping == request.range.mapping &&
               entry.address == request.key.storage.descriptor.address &&
               entry.access == request.access && entry.alias == request.alias;
    }

    static bool same_fast_state(const FastEntry& entry,
                                const std::vector<ResourceStorageRange>& state) {
        if (entry.state.size() != state.size()) return false;
        for (std::size_t index = 0; index < state.size(); ++index) {
            const auto& prior = entry.state[index];
            const auto& current = state[index];
            if (prior.bytes != current.dirty.bytes ||
                prior.version != current.dirty.version ||
                prior.cpu_dirty != current.dirty.cpu_dirty ||
                prior.gpu_dirty != current.dirty.gpu_dirty ||
                prior.has_storage != bool(current.storage) ||
                (prior.gpu_dirty && prior.writer != current.dirty.last_writer.id()))
                return false;
            if (prior.has_storage) {
                const auto storage = prior.storage.lock();
                if (!storage || storage != current.storage) return false;
            }
        }
        return true;
    }

    RendererResult<std::optional<ResidentTexture>> acquire_fast(
        const TextureRequest& request) {
        const auto found = fast_entries.find(fast_key(request));
        if (found == fast_entries.end()) return std::optional<ResidentTexture>{};
        auto& entry = found->second;
        do {
            if (!same_fast_request(entry, request) ||
                entry.generation != fast_generation ||
                entry.address_generation != memory->address_space().generation() ||
                !request.range.lease || request.range.bytes != request.key.storage.bytes ||
                !entry.stamp || memory->cpu_written_since(request.range.bytes, entry.stamp))
                break;
            auto admitted = memory->validate_retained(request.range, request.access);
            if (auto* error = std::get_if<RendererError>(&admitted)) return *error;
            auto inspected = memory->inspect_storage(request.range.bytes);
            if (auto* error = std::get_if<RendererError>(&inspected)) return *error;
            if (!same_fast_state(entry, std::get<std::vector<ResourceStorageRange>>(inspected))) break;
            auto allocation = entry.allocation.lock();
            if (!allocation || allocation->invalidated != entry.invalidated ||
                allocation->epoch != entry.allocation_epoch)
                break;
            std::vector<RetainedResource> copies;
            copies.reserve(entry.transfers.size());
            bool missing_transfer = false;
            for (const auto& cached : entry.transfers) {
                auto transfer = cached.lock();
                if (!transfer) {
                    missing_transfer = true;
                    break;
                }
                const auto copy = std::dynamic_pointer_cast<const SubresourceCopy>(transfer);
                if (!copy || !copy->encoded) copies.push_back(std::move(transfer));
            }
            if (missing_transfer) break;

            auto storage = entry.resident_view.lock();
            if (!storage) {
                id<MTLTexture> native = nil;
                for (const auto& cached : allocation->views) {
                    if (cached.key == request.key) {
                        native = cached.texture;
                        break;
                    }
                }
                if (!native) break;
                auto view = std::make_shared<View>();
                view->owner = std::move(allocation);
                view->lease = request.range.lease;
                view->memory = memory;
                view->native = native;
                storage = std::move(view);
                entry.resident_view = storage;
            }
            ResidentTexture resident{request.key, entry.uploaded_version, std::move(storage), {}, {}};
            resident.transfers = std::move(copies);
            ++statistics->hits;
            ++fast_hits;
            return std::optional<ResidentTexture>(std::move(resident));
        } while (false);
        return std::optional<ResidentTexture>{};
    }

    void remember_fast(const TextureRequest& request,
                       const std::vector<ResourceStorageRange>& state,
                       std::uint64_t stamp,
                       const std::shared_ptr<Allocation>& allocation,
                       const ResidentTexture& resident) {
        const auto forget = [&]() {
            fast_entries.erase(fast_key(request));
        };
        // A same-epoch slow acquisition need not refresh CPU contents. Keep
        // the upload's certificate so a later epoch cannot hide that store.
        if (std::ranges::any_of(state, [](const auto& interval) {
                return !interval.dirty.gpu_dirty;
            })) stamp = std::min(stamp, allocation->stamp);
        if (!stamp || state.empty() || state.size() > fast_interval_limit ||
            !resident.dependencies.empty() ||
            resident.transfers.size() > fast_transfer_limit) {
            forget();
            return;
        }
        for (const auto& interval : state) {
            if (!interval.dirty.gpu_dirty) continue;
            const auto snapshot = interval.dirty.last_writer.snapshot();
            if (snapshot.error || snapshot.phase != SubmissionPhase::Complete) {
                forget();
                return;
            }
        }
        FastEntry entry;
        entry.occupied = true;
        entry.key = request.key;
        entry.mapping = request.range.mapping;
        entry.address = request.key.storage.descriptor.address;
        entry.address_generation = memory->address_space().generation();
        entry.allocation_epoch = allocation->epoch;
        entry.stamp = stamp;
        entry.generation = fast_generation;
        entry.access = request.access;
        entry.alias = request.alias;
        entry.uploaded_version = resident.uploaded_version;
        entry.invalidated = allocation->invalidated;
        entry.allocation = allocation;
        entry.resident_view = resident.storage;
        entry.state.reserve(state.size());
        for (const auto& interval : state) {
            entry.state.push_back({interval.dirty.bytes, interval.dirty.version,
                interval.dirty.gpu_dirty ? interval.dirty.last_writer.id() : SubmissionId{},
                interval.dirty.cpu_dirty, interval.dirty.gpu_dirty, bool(interval.storage),
                interval.storage});
        }
        entry.transfers.reserve(resident.transfers.size());
        for (const auto& transfer : resident.transfers)
            entry.transfers.push_back(transfer);

        if (fast_entries.size() >= fast_entry_count) fast_entries.clear();
        fast_entries.insert_or_assign(fast_key(request), std::move(entry));
    }

    ~Cache() override {
        for (const auto& allocation : allocations) --allocation->cache_owners;
    }

    ResidencyDomain domain() const noexcept override { return {device->identity().device, memory.get()}; }
    RendererResult<ResidentTexture> acquire(const TextureRequest& request) override {
        try {
            auto fast = acquire_fast(request);
            if (auto* error = std::get_if<RendererError>(&fast)) return *error;
            if (auto& resident = std::get<std::optional<ResidentTexture>>(fast))
                return std::move(*resident);
            return acquire_checked(request);
        }
        catch (const std::bad_alloc&) { return failure("texture cache allocation failed"); }
    }
    RendererResult<ResidentTexture> acquire_checked(const TextureRequest& request) {
        auto validated = validate_texture_view(request.key);
        if (auto* error = std::get_if<RendererError>(&validated)) return *error;
        if (!request.range.lease || request.range.bytes != request.key.storage.bytes)
            return failure("texture request does not retain its storage range");
        const auto& d = request.key.storage.descriptor;
        auto retained = memory->retain(d.address, d.backing_extent, request.access);
        if (auto* error = std::get_if<RendererError>(&retained)) return *error;
        const auto& pin = std::get<RetainedRange>(retained);
        if (pin.bytes != request.range.bytes || pin.mapping != request.range.mapping)
            return failure("texture request mapping is stale or belongs to another registry");
        auto admission = memory->validate_retained(request.range, request.access);
        if (auto* error = std::get_if<RendererError>(&admission)) return *error;

        auto mapped = metal_storage_format(d.format);
        if (std::holds_alternative<TextureError>(mapped)) return failure("texture format has no Metal representation");
        if (request.key.view_target &&
            !legal_texture_view(d.type, *request.key.view_target,
                                request.key.first_layer, request.key.layer_count,
                                d.layers))
            return failure((std::string("Metal does not allow a ") +
                            native_type_name(*request.key.view_target) + " view of " +
                            native_type_name(d.type) + " storage with the requested layer range")
                               .c_str());
        const auto info = texture_format_info(d.format);
        if (is_bc_family(d.format) && !device->device().supportsBCTextureCompression)
            return failure("device does not support BC texture compression");
        if ((d.type == TextureType::Texture1D || d.type == TextureType::Texture1DArray) &&
            (d.levels != 1 || info.depth || is_block_compressed(info)))
            return failure("native Metal 1D mipmaps, depth and compressed storage are unsupported");
        if (info.depth && d.type != TextureType::Texture2D && d.type != TextureType::Texture2DArray &&
            d.type != TextureType::Texture2DRect && d.type != TextureType::Cubemap && d.type != TextureType::CubemapArray)
            return failure("depth storage target is unsupported by Metal");
        // Apple Silicon Metal samples BC-compressed 3D textures natively; only
        // writable compressed storage has no Metal equivalent.
        if (is_block_compressed(info) && request.access != ResourceAccess::Read)
            return failure("compressed writable textures are unsupported");
        const std::array<TextureSwizzle, 4> identity{TextureSwizzle::Red, TextureSwizzle::Green,
                                                   TextureSwizzle::Blue, TextureSwizzle::Alpha};
        if (request.access != ResourceAccess::Read && request.key.swizzle != identity)
            return failure("writable texture views require identity component mapping");
        auto inspected = memory->inspect_storage(pin.bytes);
        if (auto* error = std::get_if<RendererError>(&inspected)) return *error;
        const auto fast_stamp = memory->cpu_write_stamp();
        const auto& initial_state =
            std::get<std::vector<ResourceStorageRange>>(inspected);
        const auto fast_state = fast_stamp && initial_state.size() <= fast_interval_limit
            ? initial_state : std::vector<ResourceStorageRange>{};
        // Every GPU-owned range of the request, including intervals composed
        // or superseded below: CPU reads must never touch them.
        std::vector<BackingRange> gpu_owned;
        for (const auto& interval : std::get<std::vector<ResourceStorageRange>>(inspected))
            if (interval.dirty.gpu_dirty) gpu_owned.push_back(interval.dirty.bytes);
        // Units rendered through separate 2D targets at their exact bytes are
        // brought into the selected storage by ordered GPU copies instead of a
        // writeback that would wait for their producers.
        struct ComposedUnit {
            SubresourceUnit unit;
            std::shared_ptr<Allocation> producer;
            ResourceVersion version;
        };
        std::vector<ComposedUnit> composed_units;
        std::vector<CompletionToken> composed_dependencies;
        {
            const auto& initial = std::get<std::vector<ResourceStorageRange>>(inspected);
            const auto& request_layout = std::get<TextureLayoutDescription>(validated);
            std::vector<ResourceStorageRange> remaining;
            for (const auto& interval : initial) {
                std::optional<SubresourceUnit> unit;
                std::shared_ptr<Allocation> producer;
                if (interval.dirty.gpu_dirty) {
                    auto view = std::dynamic_pointer_cast<const View>(interval.storage);
                    if (view && view->owner->transfers->context.get() == device.get() &&
                        !compatible_texture_storage(view->owner->key, request.key.storage, request.alias)) {
                        unit = subresource_unit(request.key.storage, request_layout, *view->owner);
                        producer = view->owner;
                    }
                }
                if (!unit) {
                    remaining.push_back(interval);
                    continue;
                }
                const auto snapshot = interval.dirty.last_writer.snapshot();
                if (snapshot.error) return *snapshot.error;
                if (snapshot.phase == SubmissionPhase::Accepted || snapshot.phase == SubmissionPhase::Submitted)
                    composed_dependencies.push_back(interval.dirty.last_writer);
                if (std::ranges::none_of(composed_units, [&](const auto& composed) {
                        return composed.unit.layer == unit->layer && composed.unit.level == unit->level;
                    }))
                    composed_units.push_back({*unit, producer, interval.dirty.version});
            }
            if (remaining.size() != initial.size()) inspected = std::move(remaining);
        }
        // GPU-newer bytes from an incompatible alias (another layout, format
        // or device) are written back for this texture's range alone once
        // their producers finished, by resolving the overlapping texture
        // before recreating the view. A pending producer still needs the
        // caller to complete prior work and retry.
        // A render target described differently over the same bytes (a
        // downsampling chain reusing one memory range at decreasing sizes) is
        // its own texture, keyed by pool, offset, and description. The attachment
        // supersedes the other description's GPU content instead of reading
        // it back, and is ordered after its producer.
        bool superseded = false, fully_superseded = false;
        {
            const auto& initial = std::get<std::vector<ResourceStorageRange>>(inspected);
            std::shared_ptr<Allocation> owner;
            bool incompatible = false, pending_incompatible = false;
            std::vector<ResourceStorageRange> kept;
            for (const auto& interval : initial) {
                if (!interval.dirty.gpu_dirty) {
                    kept.push_back(interval);
                    continue;
                }
                auto view = std::dynamic_pointer_cast<const View>(interval.storage);
                const bool compatible = view && view->owner->transfers->context.get() == device.get() &&
                    compatible_texture_storage(view->owner->key, request.key.storage, request.alias) &&
                    (!owner || owner == view->owner);
                if (compatible) {
                    owner = view->owner;
                    kept.push_back(interval);
                    continue;
                }
                const auto snapshot = interval.dirty.last_writer.snapshot();
                if (snapshot.error) return *snapshot.error;
                const bool pending = snapshot.phase == SubmissionPhase::Accepted ||
                                     snapshot.phase == SubmissionPhase::Submitted;
                if (request.access != ResourceAccess::Read && view &&
                    view->owner->transfers->context.get() == device.get()) {
                    superseded = true;
                    if (pending) composed_dependencies.push_back(interval.dirty.last_writer);
                    continue;
                }
                // One texture is kept per pool, offset, and description. A sampled
                // description keeps its own contents over bytes another
                // description rendered, with no transfer between them. This
                // differs from Maxwell, where the sample reinterprets the
                // rendered bits; access intents still order the read after
                // the writer.
                if (request.alias == TextureAliasScope::Sampler && view &&
                    view->owner->transfers->context.get() == device.get()) {
                    // Observe this decision before its intervals disappear from
                    // the later incompatible-alias diagnostic. Classify formats,
                    // targets, tiling and writer phase rather than changing
                    // dimensions so transient size changes do not flood logs.
                    static const bool trace = std::getenv("ARMIGO_TRACE_TEXTURE_REVALIDATION") != nullptr;
                    if (trace) {
                        try {
                            const auto& source = view->owner->key;
                            const auto& o = source.descriptor;
                            const auto source_info = texture_format_info(o.format);
                            using Classification = std::array<unsigned, 16>;
                            static thread_local std::map<Classification, std::uint64_t> counts;
                            const Classification classification{
                                unsigned(d.format), unsigned(o.format), unsigned(d.type), unsigned(o.type),
                                unsigned(d.layout), unsigned(o.layout), unsigned(d.block_height_log2),
                                unsigned(o.block_height_log2), unsigned(d.block_depth_log2),
                                unsigned(o.block_depth_log2), unsigned(request.alias), unsigned(request.access),
                                unsigned(snapshot.phase), unsigned(request.key.first_level != 0),
                                unsigned(request.key.first_layer != 0), unsigned(source.bytes == pin.bytes)};
                            auto& count = counts[classification];
                            if (count != std::numeric_limits<std::uint64_t>::max()) ++count;
                            // Each thread logs the first occurrence of every
                            // class, then only powers of two starting at 1024.
                            // Rare block reinterpretations get their own class.
                            if (count == 1 || (count >= 1024 && (count & (count - 1)) == 0)) {
                                const auto overlap_start = std::max(pin.bytes.offset, interval.dirty.bytes.offset);
                                const auto overlap_end = std::min(pin.bytes.offset + pin.bytes.size,
                                    interval.dirty.bytes.offset + interval.dirty.bytes.size);
                                const auto native_level = view->native.parentTexture ? view->native.parentRelativeLevel : 0;
                                const auto native_layer = view->native.parentTexture ? view->native.parentRelativeSlice : 0;
                                armigo::diag::opt_in(
                                    "texture_alias action=keep-description scope=%u access=%u n=%llu pending=%d phase=%u "
                                    "want=[type=%u fmt=%u block=%ux%u/%u %ux%ux%u layers=%u levels=%u bytes=%llu "
                                    "extent=%llu layout=%u tile=%u/%u row=%llu base_level=%u mip=%u+%u layer=%u+%u] "
                                    "have=[type=%u fmt=%u block=%ux%u/%u %ux%ux%u layers=%u levels=%u bytes=%llu "
                                    "extent=%llu layout=%u tile=%u/%u row=%llu base_level=%u native_mip=%llu native_layer=%llu] "
                                    "offset=%lld overlap=[%llu+%llu]\n",
                                    unsigned(request.alias), unsigned(request.access), (unsigned long long)count,
                                    int(pending), unsigned(snapshot.phase),
                                    unsigned(d.type), unsigned(d.format), unsigned(info.block_width),
                                    unsigned(info.block_height), unsigned(info.bytes_per_block), d.width, d.height, d.depth,
                                    d.layers, d.levels, (unsigned long long)pin.bytes.size,
                                    (unsigned long long)d.backing_extent, unsigned(d.layout), unsigned(d.block_height_log2),
                                    unsigned(d.block_depth_log2), (unsigned long long)d.row_bytes, d.base_level,
                                    request.key.first_level, request.key.level_count, request.key.first_layer, request.key.layer_count,
                                    unsigned(o.type), unsigned(o.format), unsigned(source_info.block_width),
                                    unsigned(source_info.block_height), unsigned(source_info.bytes_per_block), o.width, o.height, o.depth,
                                    o.layers, o.levels, (unsigned long long)source.bytes.size,
                                    (unsigned long long)o.backing_extent, unsigned(o.layout), unsigned(o.block_height_log2),
                                    unsigned(o.block_depth_log2), (unsigned long long)o.row_bytes, o.base_level,
                                    (unsigned long long)native_level, (unsigned long long)native_layer,
                                    (long long)source.bytes.offset - (long long)pin.bytes.offset,
                                    (unsigned long long)(overlap_start - pin.bytes.offset),
                                    (unsigned long long)(overlap_end > overlap_start ? overlap_end - overlap_start : 0));
                            }
                        } catch (const std::bad_alloc&) {
                            // Diagnostic bookkeeping must not change admission.
                            static thread_local bool reported = false;
                            if (!reported) {
                                armigo::diag::opt_in( "texture_alias observation allocation failed\n");
                                reported = true;
                            }
                        }
                    }
                    superseded = true;
                    continue;
                }
                kept.push_back(interval);
                incompatible = true;
                if (pending) pending_incompatible = true;
            }
            if (superseded) {
                fully_superseded = kept.empty();
                inspected = std::move(kept);
            }
            if (incompatible && !pending_incompatible) {
                auto resolved = memory->resolve(pin.bytes, ResolveReason::IncompatibleAlias);
                if (auto* error = std::get_if<RendererError>(&resolved)) return *error;
                inspected = memory->inspect_storage(pin.bytes);
                if (auto* error = std::get_if<RendererError>(&inspected)) return *error;
            }
        }
        const auto& state = std::get<std::vector<ResourceStorageRange>>(inspected);
        std::shared_ptr<Allocation> selected;
        std::vector<CompletionToken> dependencies;
        for (const auto& interval : state) {
            if (!interval.dirty.gpu_dirty) continue;
            auto view = std::dynamic_pointer_cast<const View>(interval.storage);
            if (!view || view->owner->transfers->context.get() != device.get() ||
                !compatible_texture_storage(view->owner->key, request.key.storage,
                                            request.alias) ||
                (selected && selected != view->owner)) {
                static const bool trace = std::getenv("ARMIGO_TRACE_TEXTURE_REVALIDATION") != nullptr;
                static std::uint64_t traced = 0;
                if (trace && view && traced++ % 25 == 0) {
                    const auto& o = view->owner->key.descriptor;
                    armigo::diag::opt_in( "texture_alias n=%llu want=[type=%u fmt=%u %ux%u layers=%u levels=%u "
                        "extent=%llu tile=%u] have=[type=%u fmt=%u %ux%u layers=%u levels=%u extent=%llu tile=%u] "
                        "offset=%lld same_device=%d other_owner=%d\n", (unsigned long long)traced,
                        unsigned(d.type), unsigned(d.format), d.width, d.height, d.layers, d.levels,
                        (unsigned long long)d.backing_extent, unsigned(d.block_height_log2),
                        unsigned(o.type), unsigned(o.format), o.width, o.height, o.layers, o.levels,
                        (unsigned long long)o.backing_extent, unsigned(o.block_height_log2),
                        (long long)view->owner->key.bytes.offset - (long long)pin.bytes.offset,
                        int(view->owner->transfers->context.get() == device.get()),
                        int(selected && selected != view->owner));
                }
                return failure("GPU-newer incompatible texture alias requires explicit resolve", CommandStatus::Stopped);
            }
            selected = view->owner;
            const auto snapshot = interval.dirty.last_writer.snapshot();
            if (snapshot.error) return *snapshot.error;
            if (snapshot.phase == SubmissionPhase::Accepted || snapshot.phase == SubmissionPhase::Submitted)
                dependencies.push_back(interval.dirty.last_writer);
        }
        // GPU-dirty intervals stay GPU-authoritative across internal boundaries
        // (deferred writeback); only CPU-backed intervals can be stale by epoch.
        const bool cpu_backed = std::ranges::any_of(state, [](const auto& interval) {
            return !interval.dirty.gpu_dirty;
        });
        if (selected && !same_authoritative_versions(selected->versions, state)) {
            static const bool trace = std::getenv("ARMIGO_TRACE_TEXTURE_REVALIDATION") != nullptr;
            static std::uint64_t traced = 0;
            if (trace && traced++ % 50 == 0) {
                for (const auto& current : state)
                    for (const auto& prior : selected->versions) {
                        if (!overlaps(current.dirty.bytes, prior.bytes)) continue;
                        if (current.dirty.version.cpu_write == prior.version.cpu_write &&
                            (current.dirty.gpu_dirty || current.dirty.version.gpu_write == prior.version.gpu_write))
                            continue;
                        armigo::diag::opt_in( "texture_mixed n=%llu fmt=%u %ux%ux%u levels=%u extent=%llu "
                            "now=[%llu+%llu cpu=%llu gpu=%llu dirty=%d] prior=[%llu+%llu cpu=%llu gpu=%llu] intervals=%zu\n",
                            (unsigned long long)traced, unsigned(d.format), d.width, d.height, d.layers, d.levels,
                            (unsigned long long)d.backing_extent,
                            (unsigned long long)current.dirty.bytes.offset, (unsigned long long)current.dirty.bytes.size,
                            (unsigned long long)current.dirty.version.cpu_write, (unsigned long long)current.dirty.version.gpu_write,
                            int(current.dirty.gpu_dirty), (unsigned long long)prior.bytes.offset,
                            (unsigned long long)prior.bytes.size, (unsigned long long)prior.version.cpu_write,
                            (unsigned long long)prior.version.gpu_write, state.size());
                    }
            }
        }
        // Clean bytes whose versions moved since this allocation recorded them
        // were written through another description (or by this one) and are
        // back in guest memory. Their units re-upload below; a unit that also
        // holds this allocation's pending GPU results cannot be merged.
        std::vector<std::uint32_t> resolved_units;
        bool resolved_alias = false;
        if (selected && !same_authoritative_versions(selected->versions, state)) {
            resolved_alias = true;
            for (const auto& current : state) {
                // Only CPU stores re-upload; bytes another description (or
                // this one) rendered keep this texture's contents, as above.
                const bool moved = std::ranges::any_of(selected->versions, [&](const DirtyRange& prior) {
                    return overlaps(current.dirty.bytes, prior.bytes) &&
                        current.dirty.version.cpu_write != prior.version.cpu_write;
                });
                if (!moved) continue;
                if (current.dirty.gpu_dirty)
                    return failure("mixed texture versions require explicit resolve", CommandStatus::Stopped);
                for (std::uint32_t layer = 0; layer < selected->layout.layers; ++layer)
                    for (std::uint32_t level = 0; level < selected->layout.levels; ++level) {
                        const auto unit = selected->unit_bytes(layer, level);
                        if (!overlaps(unit, current.dirty.bytes)) continue;
                        if (std::ranges::any_of(gpu_owned, [&](const auto& owned) { return overlaps(owned, unit); }))
                            return failure("mixed texture versions require explicit resolve", CommandStatus::Stopped);
                        resolved_units.push_back(layer * selected->layout.levels + level);
                    }
            }
        }
        // One checked CPU snapshot serves every compatible candidate and, on
        // mismatch, the replacement upload. Keep native guest access stopped
        // under the registry's submission-boundary coordinator contract.
        // The stamp is taken before any byte the cache keeps is read.
        const auto stamp = memory->cpu_write_stamp();
        std::vector<std::byte> current;
        const auto snapshot = [&]() {
            if (!current.empty()) return true;
            current.resize(d.backing_extent);
            return memory->address_space().read(d.address, current).ok();
        };
        // Submission-boundary mode cannot observe native CPU stores.
        // Certificate stamps first prove which layer/level units no CPU store
        // reached since their contents were established; the remaining
        // CPU-backed units are hashed from one snapshot and only units whose
        // hash differs re-upload. Storage pinned by a consumer (owners beyond
        // the cache and extra_owners) is invalidated instead: false.
        const auto revalidate = [&](const std::shared_ptr<Allocation>& allocation,
                                    long extra_owners) -> std::variant<bool, RendererError> {
                if (!memory->cpu_written_since(allocation->key.bytes, allocation->stamp)) {
                    ++statistics->revalidations_skipped;
                } else {
                    const auto started = std::chrono::steady_clock::now();
                    // Read only CPU-backed intervals: GPU-dirty bytes may
                    // still be pending and are authoritative on the GPU.
                    if (gpu_owned.empty()) {
                        if (!snapshot())
                            return failure("texture CPU revalidation read failed", CommandStatus::MemoryFault);
                    } else {
                        current.assign(d.backing_extent, std::byte{0});
                        for (const auto& interval : state) {
                            if (interval.dirty.gpu_dirty) continue;
                            const auto offset = interval.dirty.bytes.offset - pin.bytes.offset;
                            if (offset >= current.size()) continue;
                            const auto length = std::min<std::uint64_t>(interval.dirty.bytes.size,
                                                                        current.size() - offset);
                            if (!memory->address_space().read(d.address + offset,
                                    std::span(current).subspan(offset, length)).ok())
                                return failure("texture CPU revalidation read failed", CommandStatus::MemoryFault);
                        }
                    }
                    const auto changed = stale_units(*allocation, *memory, current, stamp, gpu_owned);
                    trace_revalidation(d.backing_extent, changed.size(), allocation->layout.layers *
                        allocation->layout.levels, std::chrono::steady_clock::now() - started);
                    if (!changed.empty() &&
                        allocation.use_count() - extra_owners != static_cast<long>(allocation->cache_owners)) {
                        allocation->invalidated = true;
                        return false;
                    }
                    if (!changed.empty()) {
                        auto status = allocation->upload(current, &changed);
                        if (auto* error = std::get_if<RendererError>(&status)) return *error;
                        ++statistics->reuploads;
                        for (const auto unit : changed) {
                            const auto layer = unit / allocation->layout.levels;
                            const auto level = unit % allocation->layout.levels;
                            const auto [start, end] = allocation->unit_span(layer, level);
                            statistics->reupload_bytes += end - start;
                            // Keep the established bytes and the certificate
                            // describing exactly the uploaded unit.
                            std::memcpy(allocation->baseline.data() + start,
                                        current.data() + start, end - start);
                            allocation->refresh_certificate(layer, level, stamp, current);
                        }
                        allocation->stamp = stamp;
                    } else {
                        // Every certified unit matches the snapshot read
                        // after this stamp; the proof moves forward.
                        allocation->stamp = stamp;
                        ++statistics->revalidations_hashed;
                    }
                }
                allocation->epoch = memory->cpu_cache_epoch();
                return true;
        };
        if (resolved_alias && !resolved_units.empty()) {
            // Hash every moved unit regardless of write stamps; storage still
            // read by queued work stops so the caller completes it first.
            selected->stamp = 0;
            for (const auto unit : resolved_units) selected->certificates[unit].stamp = 0;
            auto revalidated = revalidate(selected, 1);
            if (auto* error = std::get_if<RendererError>(&revalidated)) return *error;
            if (!std::get<bool>(revalidated))
                return failure("mixed texture versions require explicit resolve", CommandStatus::Stopped);
        }
        if (resolved_alias) selected->versions = versions(state);
        // A partly GPU-rendered selection revalidates its CPU-backed units after
        // an epoch advance; changed units in storage still in use stop.
        if (selected && cpu_backed && selected->epoch != memory->cpu_cache_epoch()) {
            auto revalidated = revalidate(selected, 1);
            if (auto* error = std::get_if<RendererError>(&revalidated)) return *error;
            if (!std::get<bool>(revalidated))
                return failure("partly rendered texture changed on the CPU while in use requires explicit resolve",
                               CommandStatus::Stopped);
        }
        if (!selected) {
            std::vector<Allocation*> bucket;
            if (const auto found = by_bytes.find(bytes_key(request.key.storage.bytes)); found != by_bytes.end())
                bucket = found->second;
            for (auto* candidate : bucket) {
                // This reference is the loop's own, beyond the owners below.
                const auto allocation = candidate->shared_from_this();
                if (!compatible_texture_storage(allocation->key, request.key.storage,
                                                request.alias) ||
                    allocation->invalidated || !same_versions(allocation->versions, state)) continue;
                if (allocation->epoch != memory->cpu_cache_epoch()) {
                    auto revalidated = revalidate(allocation, 1);
                    if (auto* error = std::get_if<RendererError>(&revalidated)) return *error;
                    if (!std::get<bool>(revalidated)) continue;
                }
                selected = allocation;
                break;
            }
        }
        if (selected) ++statistics->hits;
        else {
            const auto estimated = std::max<std::uint64_t>(d.backing_extent,
                std::get<TextureLayoutDescription>(validated).total_size);
            if (estimated > budget) return failure("texture exceeds residency budget");
            selected = std::make_shared<Allocation>();
            selected->key = request.key.storage;
            selected->layout = std::get<TextureLayoutDescription>(std::move(validated));
            selected->epoch = memory->cpu_cache_epoch();
            selected->versions = versions(state);
            selected->transfers = transfers;
            selected->plan_planes();
            if (!selected->transfer_size || selected->transfer_size > max_texture_layout_size * 2)
                return failure("texture transfer extent exceeds policy limit");
            // The registry's checked read owns a CPU read transaction for the
            // entire callback; never issue this read while GPU-newer bytes exist.
            if (superseded) {
                // Superseded bytes are not this texture's content; only the
                // remaining CPU-backed intervals are read from guest memory.
                current.assign(d.backing_extent, std::byte{0});
                for (const auto& interval : state) {
                    if (interval.dirty.gpu_dirty) continue;
                    const auto offset = interval.dirty.bytes.offset - pin.bytes.offset;
                    if (offset >= current.size()) continue;
                    const auto length = std::min<std::uint64_t>(interval.dirty.bytes.size, current.size() - offset);
                    if (!memory->address_space().read(d.address + offset,
                            std::span(current).subspan(offset, length)).ok())
                        return failure("texture CPU upload read failed", CommandStatus::MemoryFault);
                }
            } else if (!snapshot())
                return failure("texture CPU upload read failed", CommandStatus::MemoryFault);
            selected->establish_certificates(stamp, current);
            auto* descriptor = [MTLTextureDescriptor new];
            descriptor.textureType = native_type(d.type);
            descriptor.pixelFormat = std::get<MTLPixelFormat>(mapped);
            descriptor.width = d.width;
            descriptor.height = d.height;
            descriptor.depth = d.depth;
            descriptor.mipmapLevelCount = d.levels;
            descriptor.arrayLength = (d.type == TextureType::Cubemap || d.type == TextureType::CubemapArray) ?
                d.layers / 6 : d.layers;
            descriptor.storageMode = MTLStorageModePrivate;
            descriptor.usage = MTLTextureUsageShaderRead | MTLTextureUsagePixelFormatView;
            if (!is_block_compressed(info)) descriptor.usage |= MTLTextureUsageRenderTarget;
            selected->texture = [device->device() newTextureWithDescriptor:descriptor];
            if (!selected->texture) return failure("Metal texture allocation failed");
            // Certificates replace the former resident CPU copy: only the
            // native allocation counts toward the residency budget.
            selected->resident_cost = selected->texture.allocatedSize;
            if (selected->resident_cost > budget) return failure("native texture exceeds residency budget");
            auto trimmed = trim(budget - selected->resident_cost);
            if (auto* error = std::get_if<RendererError>(&trimmed)) return *error;
            // Charge before committing transfer work. A timed-out upload stays
            // pinned by its feedback handler and cannot evade budget accounting.
            selected->invalidated = true;
            allocations.push_back(selected);
            index_add(selected.get());
            ++selected->cache_owners;
            statistics->resident_bytes += selected->resident_cost;
            RendererStatus status;
            // A target whose every byte was superseded has no content to
            // establish and starts undefined.
            try { status = fully_superseded ? renderer_success() : selected->upload(current); }
            catch (const std::bad_alloc&) { status = failure("texture upload allocation failed"); }
            if (auto* error = std::get_if<RendererError>(&status)) {
                // Only the local and cache references remain before submission
                // or after quiescent feedback. Pending feedback owns a third
                // reference, so its allocation stays charged until safe trim.
                if (selected.use_count() == static_cast<long>(selected->cache_owners + 1)) {
                    std::erase(allocations, selected);
                    index_remove(selected.get());
                    --selected->cache_owners;
                    statistics->resident_bytes -= selected->resident_cost;
                }
                return *error;
            }
            selected->baseline = std::move(current);
            selected->invalidated = false;
        }
        if (std::find(allocations.begin(), allocations.end(), selected) == allocations.end()) {
            if (selected->resident_cost > budget) return failure("shared texture exceeds residency budget");
            auto trimmed = trim(budget - selected->resident_cost);
            if (auto* error = std::get_if<RendererError>(&trimmed)) return *error;
            allocations.push_back(selected);
            index_add(selected.get());
            ++selected->cache_owners;
            statistics->resident_bytes += selected->resident_cost;
        }
        std::vector<RetainedResource> copies;
        for (const auto& composed : composed_units) {
            const auto index = composed.unit.layer * selected->layout.levels + composed.unit.level;
            auto& slot = selected->composed[index];
            if (!slot.plan || slot.producer != composed.producer.get() || slot.version != composed.version) {
                auto plan = std::make_shared<SubresourceCopy>();
                plan->source = composed.producer->texture;
                plan->destination = selected->texture;
                plan->layer = composed.unit.layer;
                plan->level = composed.unit.level;
                plan->width = composed.unit.width;
                plan->height = composed.unit.height;
                if (plan->source.pixelFormat != plan->destination.pixelFormat) {
                    const auto texel = texture_format_info(selected->key.descriptor.format).bytes_per_block;
                    plan->bytes_per_row = (std::uint64_t{plan->width} * texel + 255) & ~std::uint64_t{255};
                    plan->staging = [device->device() newBufferWithLength:plan->bytes_per_row * plan->height
                                                                  options:MTLResourceStorageModePrivate];
                    if (!plan->staging) return failure("sub-resource copy staging allocation failed",
                                                       CommandStatus::BudgetExceeded);
                }
                slot = {composed.producer.get(), composed.version, plan};
            }
            if (!slot.plan->encoded) copies.push_back(slot.plan);
        }
        auto pixel = std::get<MTLPixelFormat>(mapped);
        // A depth-aliased selection binds the storage's own depth aspect:
        // The format alias serves the colour-coded lookup from the host
        // depth texture, and Metal has no colour view of a depth texture.
        // sRGB/linear pairs are one storage too, but keep the request's own
        // sRGB view format, so only the depth alias recomputes.
        if (graphics::colour_alias_storage(selected->key.descriptor.format) ==
            std::optional(d.format)) {
            mapped = metal_storage_format(selected->key.descriptor.format);
            if (std::holds_alternative<TextureError>(mapped))
                return failure("aliased texture storage has no Metal representation");
            pixel = std::get<MTLPixelFormat>(mapped);
        }
        if (request.key.aspect == TextureAspect::Stencil) pixel = MTLPixelFormatX32_Stencil8;
        const auto& s = request.key.swizzle;
        auto swizzle = MTLTextureSwizzleChannelsMake(native_swizzle(s[0]), native_swizzle(s[1]),
                                                    native_swizzle(s[2]), native_swizzle(s[3]));
        id<MTLTexture> native = nil;
        for (const auto& cached : selected->views)
            if (cached.key == request.key) { native = cached.texture; break; }
        if (!native) {
            // An admitted same-dimensions view target replaces the storage's
            // own type for the native view; its layer range stays bounded by
            // the admission check above.
            const auto view_type =
                request.key.view_target ? *request.key.view_target : d.type;
            native = [selected->texture newTextureViewWithPixelFormat:pixel
                textureType:native_type(view_type) levels:NSMakeRange(request.key.first_level, request.key.level_count)
                slices:NSMakeRange(request.key.first_layer, request.key.layer_count) swizzle:swizzle];
            if (!native) return failure("Metal texture format or aspect view is unsupported");
            // Bound view metadata independently of storage. Consumer leases
            // retain evicted native views through their actual completion.
            if (selected->views.size() == 64) selected->views.erase(selected->views.begin());
            selected->views.push_back({request.key, native});
        }
        auto view = std::make_shared<View>();
        view->owner = selected;
        view->lease = pin.lease;
        view->memory = memory;
        view->native = native;
        ResourceVersion latest;
        for (const auto& interval : state) {
            latest.cpu_write = std::max(latest.cpu_write, interval.dirty.version.cpu_write);
            latest.gpu_write = std::max(latest.gpu_write, interval.dirty.version.gpu_write);
        }
        dependencies.insert(dependencies.end(), composed_dependencies.begin(), composed_dependencies.end());
        ResidentTexture resident{request.key, latest, view,
                                 std::move(dependencies), {}};
        resident.transfers = std::move(copies);
        remember_fast(request, fast_state, fast_stamp,
                      selected, resident);
        return resident;
    }
    RendererStatus invalidate(BackingRange bytes) override {
        auto inspected = memory->inspect(bytes);
        if (auto* error = std::get_if<RendererError>(&inspected)) return *error;
        // Keep live allocations in accounting even after invalidating lookup.
        // Dirty authority remains reachable through L2; only an unpinned cache
        // entry can be evicted, and every overlapping clean view is invalidated.
        for (auto& allocation : allocations)
            if (overlaps(allocation->key.bytes, bytes)) allocation->invalidated = true;
        invalidate_fast_entries();
        return renderer_success();
    }
    RendererStatus resolve(const TextureViewKey& key, ResolveReason reason) override {
        auto checked = validate_texture_view(key);
        if (auto* error = std::get_if<RendererError>(&checked)) return *error;
        auto status = memory->resolve(key.storage.bytes, reason);
        if (!std::holds_alternative<RendererError>(status)) invalidate_fast_entries();
        return status;
    }
    RendererStatus trim(std::uint64_t target) override {
        for (auto entry = allocations.begin(); entry != allocations.end() && statistics->resident_bytes > target;) {
            // A lease held by a consumer or registry pins its allocation. Never
            // mutate/recycle it or evict the sole authoritative GPU version.
            // Other caches may release their accounting references independently.
            // Views and transfer feedback still pin the allocation in every cache.
            if (entry->use_count() != static_cast<long>((*entry)->cache_owners)) {
                ++entry;
                continue;
            }
            auto& allocation = *entry;
            const auto format = allocation->key.descriptor.format;
            const bool converted_depth = format == TextureFormat::D24UnormS8Uint ||
                format == TextureFormat::S8UintD24Unorm || format == TextureFormat::X8UintD24Unorm;
            if (converted_depth && allocation->has_gpu_snapshot && !allocation->invalidated &&
                memory->has_backing(allocation->key.bytes.backing)) {
                auto inspected = memory->inspect_storage(allocation->key.bytes);
                if (auto* error = std::get_if<RendererError>(&inspected)) return *error;
                bool current = same_versions(allocation->versions,
                    std::get<std::vector<ResourceStorageRange>>(inspected));
                if (current && allocation->epoch != memory->cpu_cache_epoch()) {
                    // A clean cache entry does not pin its original GPU VA.
                    // If it was unmapped or reused, keep precision until an
                    // acquire through a valid alias revalidates the snapshot,
                    // or the backing is retired. Never read a replacement VA.
                    auto retained = memory->retain(allocation->key.descriptor.address,
                                                   allocation->key.bytes.size, ResourceAccess::Read);
                    const auto* pin = std::get_if<RetainedRange>(&retained);
                    if (!pin || pin->bytes != allocation->key.bytes) {
                        ++entry;
                        continue;
                    }
                    try {
                        std::vector<std::byte> bytes(allocation->key.bytes.size);
                        if (!memory->address_space().read(allocation->key.descriptor.address, bytes))
                            return failure("texture eviction revalidation read failed", CommandStatus::MemoryFault);
                        // Certificate hashes keep the D24 rule: the GPU snapshot
                        // stays resident only while every certified unit still
                        // matches the guest bytes exactly.
                        current = stale_units(*allocation, *memory, bytes,
                                              memory->cpu_write_stamp()).empty();
                    } catch (const std::bad_alloc&) {
                        return failure("texture eviction revalidation allocation failed");
                    }
                }
                // Resolving D24 publishes rounded guest bytes but does not make
                // them a lossless replacement for GPU comparison precision.
                // Keep that authority resident until a CPU or alias write
                // supersedes it; budget pressure must stop instead of changing
                // the result of a later identical-depth comparison.
                if (current) {
                    ++entry;
                    continue;
                }
            }
            --(*entry)->cache_owners;
            statistics->resident_bytes -= (*entry)->resident_cost;
            index_remove(entry->get());
            entry = allocations.erase(entry);
            invalidate_fast_entries();
        }
        if (statistics->resident_bytes > target)
            return failure("texture residency budget is pinned by live resources", CommandStatus::Stopped);
        return renderer_success();
    }
};
} // namespace

RendererResult<std::shared_ptr<TextureCache>> create_texture_cache(
    std::shared_ptr<DeviceContext> device, std::shared_ptr<ResourceMemory> memory,
    std::uint64_t budget_bytes) {
    auto registry = std::dynamic_pointer_cast<ResourceMemoryRegistry>(memory);
    if (!device || !device->device() || !device->queue() || !device->identity().device ||
        !registry || !budget_bytes || ![device->device() supportsFamily:MTLGPUFamilyMetal4])
        return failure("texture cache requires a Metal 4 device, queue, L2 registry and budget");
    try {
        auto cache = std::make_shared<Cache>();
        cache->device = std::move(device);
        cache->memory = std::move(registry);
        cache->budget = budget_bytes;
        cache->transfers = std::make_shared<Transfers>();
        cache->transfers->context = cache->device;
        cache->transfers->statistics = cache->statistics;
        // Pipelines are built once per device; without them every transfer
        // takes the CPU conversion path explicitly.
        if (auto created = GpuTextureSwizzle::create(cache->device->device());
            std::holds_alternative<std::shared_ptr<GpuTextureSwizzle>>(created))
            cache->transfers->swizzle =
                std::get<std::shared_ptr<GpuTextureSwizzle>>(std::move(created));
        return std::shared_ptr<TextureCache>(std::move(cache));
    } catch (const std::bad_alloc&) { return failure("texture cache allocation failed"); }
}
ResidentTextureIdentity resident_texture_identity(const ResidentTexture& texture) {
    const auto view = std::dynamic_pointer_cast<const View>(texture.storage);
    const auto& key = texture.key;
    // Exact storage equality deliberately excludes format/depth aliases. A
    // partial view or a composed producer must return through acquire_checked.
    if (!view || view->owner->invalidated || !texture.transfers.empty() ||
        !(view->owner->key == key.storage) || key.first_level || key.first_layer ||
        key.level_count != key.storage.descriptor.levels ||
        key.layer_count != key.storage.descriptor.layers || key.view_target)
        return {};
    return {view->owner, key.aspect};
}

TextureCacheStatistics texture_cache_statistics(const TextureCache& cache) noexcept {
    if (auto* native = dynamic_cast<const Cache*>(&cache)) return *native->statistics;
    return {};
}
std::uint64_t texture_cache_fast_path_hits(const TextureCache& cache) noexcept {
    if (auto* native = dynamic_cast<const Cache*>(&cache)) return native->fast_hits;
    return 0;
}
} // namespace armigo::platform::metal
