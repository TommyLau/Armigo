// ============================================================================
// Armigo Platform - Apple Platform Integration
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================
// One 32-thread threadgroup per GOB: each thread copies one 16-byte GOB
// segment, which is contiguous in block-linear storage, into its place in a
// linear row. The level's strides come from the layout's own addressing, so
// every block height, block depth, array layer, mip and block size lands on
// the bytes the CPU conversion copies.

#include "nvn_deswizzle.h"

#include <cstring>
#include <limits>

namespace armigo::platform::metal::nvn_driver {
namespace {

// Per-dispatch parameters. The field order must match DeswizzleLevel in the
// kernel source below.
struct DeswizzleParameters {
  std::uint64_t source = 0;
  std::uint64_t destination = 0;
  std::uint64_t pitch = 0;
  std::uint64_t row_stride = 0;
  std::uint64_t slice_stride = 0;
  std::uint32_t row_bytes = 0;
  std::uint32_t rows = 0;
  std::uint32_t slices = 0;
  std::uint32_t height_log2 = 0;
  std::uint32_t depth_log2 = 0;
  std::uint32_t gob_order = 0;
};
static_assert(sizeof(DeswizzleParameters) == deswizzle_parameter_bytes);

constexpr std::uint32_t max_block_log2 = 5;

// Thread t starts at byte t * 16 inside a GOB. The selected interleave
// determines which thread bits name the row and 16-byte column.
const char *const deswizzle_source = R"(
#include <metal_stdlib>
using namespace metal;

struct DeswizzleLevel {
  ulong source;
  ulong destination;
  ulong pitch;
  ulong row_stride;
  ulong slice_stride;
  uint row_bytes;
  uint rows;
  uint slices;
  uint height_log2;
  uint depth_log2;
  uint gob_order;
};

kernel void armigo_nvn_deswizzle(device const uchar *source [[buffer(0)]],
                                 device uchar *destination [[buffer(1)]],
                                 constant DeswizzleLevel &p [[buffer(2)]],
                                 uint3 id [[thread_position_in_grid]]) {
  const uint tid = id.x & 31u;
  const uint gob_x = id.x >> 5;
  const uint gob_y = id.y;
  const uint z = id.z;
  const uint segment = ((tid >> 4) << 1) | ((tid >> (p.gob_order ? 2 : 1)) & 1u);
  const uint row_bits = p.gob_order ? (((tid >> 3) & 1u) << 2) | (((tid >> 1) & 1u) << 1)
                                    : ((tid >> 2) & 3u) << 1;
  const uint y = (gob_y << 3) | row_bits | (tid & 1u);
  const uint x = (gob_x << 6) | (segment << 4);
  if (x >= p.row_bytes || y >= p.rows || z >= p.slices) return;
  const uint n = min(16u, p.row_bytes - x);
  const uint height_mask = (1u << p.height_log2) - 1u;
  const uint depth_mask = (1u << p.depth_log2) - 1u;
  const ulong from = p.source + (ulong)(z >> p.depth_log2) * p.slice_stride +
                     (ulong)(gob_y >> p.height_log2) * p.row_stride +
                     ((ulong)gob_x << (9u + p.height_log2 + p.depth_log2)) +
                     (ulong)(gob_y & height_mask) * 512ul +
                     (((ulong)(z & depth_mask) * 512ul) << p.height_log2) +
                     (ulong)tid * 16ul;
  const ulong to = p.destination + ((ulong)z * p.rows + y) * p.pitch + x;
  if (n == 16u && ((from | to) & 15ul) == 0ul)
    *(device uint4 *)(destination + to) = *(device const uint4 *)(source + from);
  else
    for (uint i = 0; i < n; ++i) destination[to + i] = source[from + i];
}
)";

} // namespace

DeswizzleKernel make_deswizzle_kernel(id<MTLDevice> device) {
  DeswizzleKernel kernel;
  if (!device) {
    kernel.error = "no Metal device";
    return kernel;
  }
  NSError *error = nil;
  id<MTLLibrary> library = [device newLibraryWithSource:@(deswizzle_source)
                                                options:nil
                                                  error:&error];
  if (!library) {
    kernel.error = error ? error.localizedDescription.UTF8String
                         : "the kernel library did not compile";
    return kernel;
  }
  id<MTLFunction> function = [library newFunctionWithName:@"armigo_nvn_deswizzle"];
  if (!function) {
    kernel.error = "the kernel library has no deswizzle function";
    return kernel;
  }
  kernel.pipeline = [device newComputePipelineStateWithFunction:function
                                                          error:&error];
  if (!kernel.pipeline)
    kernel.error = error ? error.localizedDescription.UTF8String
                         : "the deswizzle pipeline was not created";
  else if (kernel.pipeline.maxTotalThreadsPerThreadgroup < 32) {
    kernel.pipeline = nil;
    kernel.error = "the deswizzle pipeline cannot run 32-thread threadgroups";
  }
  return kernel;
}

bool deswizzle_supported(const graphics::BlockLinearLevelAddressing &level,
                         std::uint64_t pitch) {
  constexpr std::uint64_t limit = std::numeric_limits<std::uint32_t>::max() - 63;
  return level.row_bytes != 0 && level.rows != 0 && level.slices != 0 &&
         level.row_bytes <= limit && level.rows <= limit &&
         pitch >= level.row_bytes &&
         level.gob_block_height_log2 <= max_block_log2 &&
         level.gob_block_depth_log2 <= max_block_log2;
}

bool encode_deswizzle_metal4(id<MTL4ComputeCommandEncoder> encoder,
                            id<MTLComputePipelineState> pipeline,
                            id<MTL4ArgumentTable> table,
                            id<MTLBuffer> parameter_buffer, std::uint64_t parameter_offset,
                            id<MTLBuffer> source, std::uint64_t source_offset,
                            id<MTLBuffer> destination, std::uint64_t destination_offset,
                            std::uint64_t pitch,
                            const graphics::BlockLinearLevelAddressing& level) {
  if (!encoder || !pipeline || !table || !source || !destination ||
      !parameter_buffer.contents || parameter_offset > parameter_buffer.length ||
      sizeof(DeswizzleParameters) > parameter_buffer.length - parameter_offset ||
      !deswizzle_supported(level, pitch)) return false;
  DeswizzleParameters parameters;
  parameters.source = source_offset + level.guest_base;
  parameters.destination = destination_offset;
  parameters.pitch = pitch;
  parameters.row_stride = level.row_stride;
  parameters.slice_stride = level.slice_stride;
  parameters.row_bytes = static_cast<std::uint32_t>(level.row_bytes);
  parameters.rows = level.rows;
  parameters.slices = level.slices;
  parameters.height_log2 = level.gob_block_height_log2;
  parameters.depth_log2 = level.gob_block_depth_log2;
  parameters.gob_order = static_cast<std::uint32_t>(level.gob_order);
  std::memcpy(static_cast<std::byte*>(parameter_buffer.contents) + parameter_offset,
              &parameters, sizeof parameters);
  [table setAddress:source.gpuAddress atIndex:0];
  [table setAddress:destination.gpuAddress atIndex:1];
  [table setAddress:parameter_buffer.gpuAddress + parameter_offset atIndex:2];
  [encoder setComputePipelineState:pipeline];
  [encoder setArgumentTable:table];
  const NSUInteger gob_columns = (level.row_bytes + 63) / 64;
  const NSUInteger gob_rows = (std::uint64_t{level.rows} + 7) / 8;
  [encoder dispatchThreadgroups:MTLSizeMake(gob_columns, gob_rows, level.slices)
          threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
  return true;
}

} // namespace armigo::platform::metal::nvn_driver
