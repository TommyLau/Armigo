// ============================================================================
// Armigo NVN - Texture Resources
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include <armigo/common/diagnostics.h>
#include "entry_table.h"
#include "resource_layout.h"

#include <armigo/graphics/texflush.h>
#include <armigo/memory/write_tracker.h>
#include "host_stack.h"

#include <algorithm>
#include <unordered_map>
#include <array>
#include <atomic>
#include <cstdio>
#include <cstring>
#include <limits>
#include <new>
#include <memory>
#include <mutex>
#include <vector>
#include <span>
#include <utility>
#include <variant>

namespace armigo::nvn {
namespace {

constexpr std::uint32_t texture_tag = object_tag('T', 'E', 'X', '\0');
constexpr std::uint32_t pool_tag = object_tag('P', 'O', 'O', 'L');
constexpr std::int32_t pitch_flags = NVN_TEXTURE_FLAGS_LINEAR |
                                     NVN_TEXTURE_FLAGS_LINEAR_RENDER_TARGET;

template <typename State, typename Object>
State* state(Object* object) noexcept {
    return reinterpret_cast<State*>(object);
}

template <typename State, typename Object>
const State* state(const Object* object) noexcept {
    return reinterpret_cast<const State*>(object);
}

void diagnose(const char* function, const char* message) noexcept {
    armigo::diag::write(armigo::diag::Level::error,  "Armigo NVN: %s: %s\n", function, message);
}
// A layout refusal names the description it refused.
void diagnose_layout(const char* function, const TextureDescription& description) noexcept {
    armigo::diag::write(armigo::diag::Level::error,
                        "Armigo NVN: %s: texture layout is unsupported or invalid (target %d, format %d, %ux%ux%u, "
                        "levels %u, samples %d, flags 0x%x, stride %u)\n",
                        function, description.target, description.format, description.width, description.height,
                        description.depth, description.levels, description.samples, description.flags,
                        description.stride);
}

// A texture's identity is everything an initialization fixes except its serial
// and snapshot: memory, pool, layout, format and swizzle. Two initializations
// with the same identity are the same texture, so they share one serial, one
// host registration and one immutable snapshot that command records point at.
// The game initializes thousands of short-lived texture objects a second over
// a few thousand identities; a snapshot per initialization grew without bound.
// Records keep raw snapshot pointers, including after the original object is
// copied, finalized or reused, so a snapshot is never freed: an identity whose
// memory pool is released is retired (its snapshot kept, unreachable to new
// initializations), and a later pool at the same address starts new identities.
struct TextureIdentity {
    std::array<std::byte, sizeof(TextureState)> bytes{};
    explicit TextureIdentity(const TextureState& texture) {
        TextureState key = texture;
        key.serial = 0;
        key.snapshot = nullptr;
        std::memcpy(bytes.data(), &key, sizeof key);
    }
    bool operator==(const TextureIdentity&) const = default;
};

struct TextureIdentityHash {
    std::size_t operator()(const TextureIdentity& identity) const noexcept {
        std::uint64_t hash = 0xcbf29ce484222325ull;
        for (std::size_t i = 0; i + 8 <= identity.bytes.size(); i += 8) {
            std::uint64_t word;
            std::memcpy(&word, identity.bytes.data() + i, sizeof word);
            hash = (hash ^ word) * 0x100000001b3ull;
        }
        return static_cast<std::size_t>(hash ^ (hash >> 32));
    }
};

struct TextureIdentities {
    std::mutex mutex;
    std::unordered_map<TextureIdentity, std::unique_ptr<TextureState>, TextureIdentityHash> live;
    std::vector<std::unique_ptr<TextureState>> retired;
};

TextureIdentities& texture_identities() {
    // Never destroyed: the installed driver's replay thread reads snapshots
    // until the process ends, and exit's static destructors must not free
    // them under it.
    static auto& identities = *new TextureIdentities;
    return identities;
}

// The snapshot of `texture`'s identity. The first initialization of an
// identity creates it with a new serial and registers it with the host while
// the identities are locked, so no other initialization can use the serial
// before the host knows it. Null when the allocation fails.
const TextureState* identity_snapshot(const TextureState& texture) {
    auto& identities = texture_identities();
    try {
        TextureIdentity identity(texture);
        std::lock_guard lock(identities.mutex);
        if (const auto found = identities.live.find(identity); found != identities.live.end())
            return found->second.get();
        auto snapshot = std::make_unique<TextureState>(texture);
        snapshot->serial = next_resource_serial();
        snapshot->snapshot = nullptr;
        const auto* result = snapshot.get();
        identities.live.emplace(identity, std::move(snapshot));
        TextureState registered = *result;
        registered.snapshot = result;
        backends().resources->register_texture(registered);
        return result;
    } catch (const std::bad_alloc&) {
        return nullptr;
    }
}

}  // namespace

void retire_pool_textures(const MemoryPoolState* pool) noexcept {
    auto& identities = texture_identities();
    std::vector<const TextureState*> retired;
    {
        std::lock_guard lock(identities.mutex);
        for (auto it = identities.live.begin(); it != identities.live.end();) {
            if (!it->second || it->second->pool != pool) {
                ++it;
                continue;
            }
            // Out of memory keeps the identity live, which is safe: its
            // snapshot stays valid and only the pool's reuse is affected.
            try {
                retired.reserve(retired.size() + 1);
                identities.retired.push_back(std::move(it->second));
            } catch (const std::bad_alloc&) {
                ++it;
                continue;
            }
            retired.push_back(identities.retired.back().get());
            it = identities.live.erase(it);
        }
    }
    if (backends().resources)
        for (const auto* texture : retired) backends().resources->release_texture(*texture);
}

namespace {

struct FormatGeometry {
    std::uint8_t block_width = 1;
    std::uint8_t block_height = 1;
    std::uint8_t bytes_per_block = 0;
};

FormatGeometry format_geometry(std::int32_t format) noexcept {
    static constexpr std::array<std::array<std::uint8_t, 2>, 14> astc{{
        {{4, 4}}, {{5, 4}}, {{5, 5}}, {{6, 5}}, {{6, 6}}, {{8, 5}}, {{8, 6}},
        {{8, 8}}, {{10, 5}}, {{10, 6}}, {{10, 8}}, {{10, 10}}, {{12, 10}}, {{12, 12}},
    }};
    if (format >= NVN_FORMAT_RGBA_ASTC_4x4 && format <= NVN_FORMAT_RGBA_ASTC_12x12_SRGB) {
        const auto index = static_cast<std::size_t>(format - NVN_FORMAT_RGBA_ASTC_4x4) % astc.size();
        return {astc[index][0], astc[index][1], 16};
    }
    if (format >= NVN_FORMAT_RGB_DXT1 && format <= NVN_FORMAT_BPTC_UFLOAT) {
        const bool eight_bytes = format <= NVN_FORMAT_RGBA_DXT1 ||
                                 format == NVN_FORMAT_RGB_DXT1_SRGB ||
                                 format == NVN_FORMAT_RGBA_DXT1_SRGB ||
                                 format == NVN_FORMAT_RGTC1_UNORM ||
                                 format == NVN_FORMAT_RGTC1_SNORM;
        return {4, 4, static_cast<std::uint8_t>(eight_bytes ? 8 : 16)};
    }
    struct Range {
        std::int32_t first, last;
        std::uint8_t bytes;
    };
    static constexpr std::array ranges{
        Range{0x01, 0x04, 1}, Range{0x05, 0x09, 2}, Range{0x0a, 0x0c, 4},
        Range{0x0d, 0x10, 2}, Range{0x11, 0x15, 4}, Range{0x16, 0x18, 8},
        Range{0x19, 0x1c, 3}, Range{0x1d, 0x21, 6}, Range{0x22, 0x24, 12},
        Range{0x25, 0x28, 4}, Range{0x29, 0x2d, 8}, Range{0x2e, 0x30, 16},
        Range{0x31, 0x31, 1}, Range{0x32, 0x32, 2}, Range{0x33, 0x35, 4},
        Range{0x36, 0x36, 8}, Range{0x37, 0x38, 4}, Range{0x39, 0x3c, 2},
        Range{0x3d, 0x40, 4}, Range{0x51, 0x52, 1}, Range{0x53, 0x54, 2},
        Range{0x55, 0x56, 4}, Range{0x57, 0x58, 2}, Range{0x59, 0x5a, 4},
        Range{0x5b, 0x5c, 8}, Range{0x5d, 0x5e, 3}, Range{0x5f, 0x60, 6},
        Range{0x61, 0x62, 12}, Range{0x63, 0x64, 4}, Range{0x65, 0x66, 8},
        Range{0x67, 0x68, 16}, Range{0x69, 0x6c, 4}, Range{0x6d, 0x70, 4},
        Range{0x71, 0x75, 8}, Range{0x76, 0x78, 16}, Range{0x95, 0x98, 2},
        Range{0x99, 0x9c, 4},
    };
    for (const Range& range : ranges) {
        if (format >= range.first && format <= range.last) return {1, 1, range.bytes};
    }
    return {};
}

std::uint32_t pick_gob_block_log2(std::uint32_t gob_count) noexcept {
    if (gob_count >= 16) return 4;
    if (gob_count >= 8) return 3;
    if (gob_count >= 4) return 2;
    if (gob_count >= 2) return 1;
    return 0;
}

bool checked_add(std::size_t left, std::size_t right, std::size_t& result) noexcept {
    if (right > std::numeric_limits<std::size_t>::max() - left) return false;
    result = left + right;
    return true;
}

bool checked_multiply(std::size_t left, std::size_t right, std::size_t& result) noexcept {
    if (left != 0 && right > std::numeric_limits<std::size_t>::max() / left) return false;
    result = left * right;
    return true;
}

bool swizzle_valid(NVNtextureSwizzle value) noexcept {
    return static_cast<std::uint32_t>(value) <= NVN_TEXTURE_SWIZZLE_A;
}

bool pack_checked_swizzle(const char* function, NVNtextureSwizzle r, NVNtextureSwizzle g,
                          NVNtextureSwizzle b, NVNtextureSwizzle a, std::uint16_t& packed) noexcept {
    if (!swizzle_valid(r) || !swizzle_valid(g) || !swizzle_valid(b) || !swizzle_valid(a)) {
        diagnose(function, "invalid texture swizzle");
        return false;
    }
    packed = pack_swizzle(static_cast<unsigned>(r), static_cast<unsigned>(g),
                          static_cast<unsigned>(b), static_cast<unsigned>(a));
    return true;
}

void unpack_swizzle(std::uint16_t packed, NVNtextureSwizzle* r, NVNtextureSwizzle* g,
                    NVNtextureSwizzle* b, NVNtextureSwizzle* a) noexcept {
    if (r) *r = static_cast<NVNtextureSwizzle>(swizzle_channel(packed, 0));
    if (g) *g = static_cast<NVNtextureSwizzle>(swizzle_channel(packed, 1));
    if (b) *b = static_cast<NVNtextureSwizzle>(swizzle_channel(packed, 2));
    if (a) *a = static_cast<NVNtextureSwizzle>(swizzle_channel(packed, 3));
}

bool valid_texture(const TextureState* texture, const char* function) noexcept {
    if (!texture || texture->tag != texture_tag || texture->serial == 0) {
        diagnose(function, "texture is not initialized");
        return false;
    }
    return true;
}

struct TransferGeometry {
    std::uint32_t level = 0;
    std::uint32_t first_layer = 0;
    std::uint32_t image_count = 0;
    std::uint32_t first_slice = 0;
    std::uint32_t x_block = 0;
    std::uint32_t y_block = 0;
    std::uint32_t block_columns = 0;
    std::uint32_t block_rows = 0;
    std::size_t row_bytes = 0;
    std::size_t row_stride = 0;
    std::size_t image_stride = 0;
    bool is_3d = false;
};

bool transfer_geometry(const TextureState& texture,
                       const graphics::TextureLayoutDescription& layout,
                       const TexelTransfer& transfer, TransferGeometry& out) noexcept {
    if (texture.description.samples > 1 ||
        texture.description.target == NVN_TEXTURE_TARGET_2D_MULTISAMPLE ||
        texture.description.target == NVN_TEXTURE_TARGET_2D_MULTISAMPLE_ARRAY)
        return false;
    if (transfer.level >= layout.mips.size()) return false;
    const auto& mip = layout.mips[transfer.level];
    const FormatGeometry format = format_geometry(texture.description.format);
    if (format.bytes_per_block == 0) return false;
    const NVNcopyRegion& region = transfer.region;
    if (region.xoffset < 0 || region.yoffset < 0 || region.zoffset < 0 || region.width < 0 ||
        region.height < 0 || region.depth < 0) {
        return false;
    }
    const auto target = texture.description.target;
    const bool one_d = target == NVN_TEXTURE_TARGET_1D || target == NVN_TEXTURE_TARGET_1D_ARRAY;
    out.level = transfer.level;
    out.is_3d = target == NVN_TEXTURE_TARGET_3D;
    const std::uint32_t x = static_cast<std::uint32_t>(region.xoffset);
    const std::uint32_t y = one_d ? 0u : static_cast<std::uint32_t>(region.yoffset);
    const std::uint32_t width = static_cast<std::uint32_t>(region.width);
    const std::uint32_t height = one_d ? 1u : static_cast<std::uint32_t>(region.height);
    if (width == 0 || (!one_d && height == 0)) {
        out.image_count = 0;
        return true;
    }
    if (target == NVN_TEXTURE_TARGET_1D_ARRAY) {
        out.image_count = static_cast<std::uint32_t>(region.height);
        std::size_t first = 0;
        if (!checked_add(transfer.first_layer, static_cast<std::uint32_t>(region.yoffset), first) ||
            first > std::numeric_limits<std::uint32_t>::max()) return false;
        out.first_layer = static_cast<std::uint32_t>(first);
    } else if (out.is_3d) {
        if (transfer.first_layer != 0) return false;
        out.image_count = static_cast<std::uint32_t>(region.depth);
        out.first_slice = static_cast<std::uint32_t>(region.zoffset);
    } else {
        out.image_count = static_cast<std::uint32_t>(region.depth);
        std::size_t first = 0;
        if (!checked_add(transfer.first_layer, static_cast<std::uint32_t>(region.zoffset), first) ||
            first > std::numeric_limits<std::uint32_t>::max()) return false;
        out.first_layer = static_cast<std::uint32_t>(first);
    }
    if (out.image_count == 0) return true;
    if (std::uint64_t{x} + width > mip.width || std::uint64_t{y} + height > mip.height) return false;
    if (x % format.block_width != 0 || y % format.block_height != 0) return false;
    if (width % format.block_width != 0 && std::uint64_t{x} + width != mip.width) return false;
    if (height % format.block_height != 0 && std::uint64_t{y} + height != mip.height) return false;
    if (out.is_3d) {
        if (std::uint64_t{out.first_slice} + out.image_count > mip.depth) return false;
    } else if (std::uint64_t{out.first_layer} + out.image_count > layout.layers) {
        return false;
    }
    out.x_block = x / format.block_width;
    out.y_block = y / format.block_height;
    out.block_columns = (width + format.block_width - 1) / format.block_width;
    out.block_rows = (height + format.block_height - 1) / format.block_height;
    if (!checked_multiply(out.block_columns, format.bytes_per_block, out.row_bytes)) return false;
    out.row_stride = transfer.row_stride ? transfer.row_stride : out.row_bytes;
    if (out.row_stride < out.row_bytes) return false;
    std::size_t minimum_image = 0;
    if (!checked_multiply(out.row_stride, out.block_rows, minimum_image)) return false;
    out.image_stride = transfer.image_stride ? transfer.image_stride : minimum_image;
    return out.image_stride >= minimum_image;
}

// Address each NVN mip independently: its depth-block reduction and full
// allocation padding differ from the command-stream mip-chain policy.
graphics::TextureLayoutResult<graphics::TextureLayoutDescription> level_storage_layout(
    const graphics::TextureLayoutDescription& layout, std::uint32_t level) {
    const auto& mip = layout.mips[level];
    auto level_description = layout;
    level_description.width = mip.width;
    level_description.height = mip.height;
    level_description.depth = mip.depth;
    level_description.levels = level_description.layers = 1;
    level_description.gob_block_height_log2 = mip.gob_block_height_log2;
    level_description.gob_block_depth_log2 = mip.gob_block_depth_log2;
    return graphics::calculate_texture_size(std::move(level_description));
}

bool transfer_one_level(const graphics::TextureLayoutDescription& layout,
                        const TransferGeometry& geometry, std::span<std::byte> guest,
                        std::byte* bytes, bool write, std::uint32_t layer,
                        std::uint32_t first_image, std::uint32_t image_count) {
    const auto& mip = layout.mips[geometry.level];
    auto calculated = level_storage_layout(layout, geometry.level);
    const auto* level_layout = std::get_if<graphics::TextureLayoutDescription>(&calculated);
    if (!level_layout) return false;
    std::size_t base = 0;
    if (!checked_multiply(layer, layout.layer_stride, base) ||
        !checked_add(base, mip.offset, base) || base > guest.size() ||
        mip.size > guest.size() - base) return false;
    auto level_guest = guest.subspan(base, mip.size);
    auto converted = graphics::convert_level_to_linear(*level_layout,
        std::span<const std::byte>(level_guest), 0, 0);
    auto* linear = std::get_if<graphics::TextureLevelData>(&converted);
    if (!linear) return false;
    for (std::uint32_t image = 0; image < image_count; ++image) {
        const std::uint32_t slice = geometry.is_3d ? geometry.first_slice + image : 0;
        for (std::uint32_t row = 0; row < geometry.block_rows; ++row) {
            std::size_t linear_offset = 0;
            std::size_t plane_row = 0;
            if (!checked_multiply(slice, mip.block_rows, plane_row) ||
                !checked_add(plane_row, geometry.y_block + row, plane_row) ||
                !checked_multiply(plane_row, mip.linear_row_bytes, linear_offset)) return false;
            std::size_t x_offset = 0;
            if (!checked_multiply(geometry.x_block, layout.bytes_per_block, x_offset) ||
                !checked_add(linear_offset, x_offset, linear_offset) ||
                linear_offset > linear->bytes.size() ||
                geometry.row_bytes > linear->bytes.size() - linear_offset) return false;
            std::size_t external_offset = 0;
            if (!checked_multiply(first_image + image, geometry.image_stride, external_offset)) return false;
            std::size_t row_offset = 0;
            if (!checked_multiply(row, geometry.row_stride, row_offset) ||
                !checked_add(external_offset, row_offset, external_offset)) return false;
            if (write) {
                std::memcpy(linear->bytes.data() + linear_offset, bytes + external_offset,
                            geometry.row_bytes);
            } else {
                std::memcpy(bytes + external_offset, linear->bytes.data() + linear_offset,
                            geometry.row_bytes);
            }
        }
    }
    if (!write) return true;
    return std::holds_alternative<std::monostate>(graphics::write_level_from_linear(
        *level_layout, linear->bytes, level_guest, 0, 0));
}

bool view_transfer(const TextureState& texture, const TextureViewState* view,
                   const NVNcopyRegion* region, ptrdiff_t row_stride, ptrdiff_t image_stride,
                   TexelTransfer& transfer, const char* function) noexcept {
    if (!region) {
        diagnose(function, "copy region is null");
        return false;
    }
    if (row_stride < 0 || image_stride < 0) {
        diagnose(function, "negative texel stride is unsupported");
        return false;
    }
    transfer.level = view && view->sets_levels ? view->base_level : 0;
    transfer.first_layer = view && view->sets_layers
                               ? static_cast<std::uint32_t>(static_cast<std::int32_t>(view->min_layer))
                               : 0;
    transfer.region = *region;
    transfer.row_stride = static_cast<std::size_t>(row_stride);
    transfer.image_stride = static_cast<std::size_t>(image_stride);
    if (view && view->sets_layers && view->min_layer < 0) {
        diagnose(function, "negative texture view layer");
        return false;
    }
    if (view && view->sets_format) {
        const FormatGeometry base = format_geometry(texture.description.format);
        const FormatGeometry selected = format_geometry(view->format);
        if (selected.bytes_per_block == 0 || base.block_width != selected.block_width ||
            base.block_height != selected.block_height || base.bytes_per_block != selected.bytes_per_block) {
            diagnose(function, "texture view format changes the storage block geometry");
            return false;
        }
    }
    if (view && view->sets_target &&
        ((view->target == NVN_TEXTURE_TARGET_1D_ARRAY) !=
         (texture.description.target == NVN_TEXTURE_TARGET_1D_ARRAY))) {
        diagnose(function, "texture view target changes one-dimensional array addressing");
        return false;
    }
    return true;
}

std::size_t view_offset(const TextureDescription& description, const TextureViewState* view,
                        bool& valid) noexcept {
    valid = false;
    const auto layout = resource_layout(description);
    if (!layout) return 0;
    const std::uint32_t level = view && view->sets_levels ? view->base_level : 0;
    const std::int32_t layer = view && view->sets_layers ? view->min_layer : 0;
    if (level >= layout->mips.size() || layer < 0 ||
        static_cast<std::uint32_t>(layer) >= layout->layers) return 0;
    std::size_t offset = 0;
    if (!checked_multiply(static_cast<std::size_t>(layer), layout->layer_stride, offset) ||
        !checked_add(offset, layout->mips[level].offset, offset)) return 0;
    valid = true;
    return offset;
}


void texture_builder_set_device(NVNtextureBuilder* builder, NVNdevice* device) {
    state<TextureBuilderState>(builder)->device = device;
}

// Packaged texture data and GL texture names are kept beside the builder: the
// in-place builder has no room for them, and either one makes initialization
// fail as unsupported.
struct BuilderExtras {
    const void* packaged_texture_data = nullptr;
    std::uint32_t gl_texture_name = 0;
};
std::mutex builder_extras_mutex;
std::unordered_map<const NVNtextureBuilder*, BuilderExtras> builder_extras;
BuilderExtras builder_extra(const NVNtextureBuilder* builder) {
    std::lock_guard lock(builder_extras_mutex);
    const auto found = builder_extras.find(builder);
    return found == builder_extras.end() ? BuilderExtras{} : found->second;
}
void set_builder_extra(const NVNtextureBuilder* builder, BuilderExtras extras) {
    std::lock_guard lock(builder_extras_mutex);
    if (!extras.packaged_texture_data && !extras.gl_texture_name) builder_extras.erase(builder);
    else builder_extras[builder] = extras;
}

void texture_builder_set_defaults(NVNtextureBuilder* builder) {
    auto* current = state<TextureBuilderState>(builder);
    NVNdevice* device = current->device;
    std::memset(current, 0, sizeof(*current));
    set_builder_extra(builder, {});
    current->device = device;
    current->description.target = NVN_TEXTURE_TARGET_2D;
    current->description.width = 1;
    current->description.height = 1;
    current->description.depth = 1;
    current->description.levels = 1;
}

void texture_builder_set_flags(NVNtextureBuilder* builder, int flags) {
    state<TextureBuilderState>(builder)->description.flags = flags;
}
void texture_builder_set_target(NVNtextureBuilder* builder, NVNtextureTarget target) {
    state<TextureBuilderState>(builder)->description.target = target;
}
void texture_builder_set_width(NVNtextureBuilder* builder, int width) {
    state<TextureBuilderState>(builder)->description.width = static_cast<std::uint32_t>(width);
}
void texture_builder_set_height(NVNtextureBuilder* builder, int height) {
    state<TextureBuilderState>(builder)->description.height = static_cast<std::uint32_t>(height);
}
void texture_builder_set_depth(NVNtextureBuilder* builder, int depth) {
    state<TextureBuilderState>(builder)->description.depth = static_cast<std::uint32_t>(depth);
}
void texture_builder_set_size_1d(NVNtextureBuilder* builder, int width) {
    auto& description = state<TextureBuilderState>(builder)->description;
    description.width = static_cast<std::uint32_t>(width);
    description.height = description.depth = 1;
}
void texture_builder_set_size_2d(NVNtextureBuilder* builder, int width, int height) {
    auto& description = state<TextureBuilderState>(builder)->description;
    description.width = static_cast<std::uint32_t>(width);
    description.height = static_cast<std::uint32_t>(height);
    description.depth = 1;
}
void texture_builder_set_size_3d(NVNtextureBuilder* builder, int width, int height, int depth) {
    auto& description = state<TextureBuilderState>(builder)->description;
    description.width = static_cast<std::uint32_t>(width);
    description.height = static_cast<std::uint32_t>(height);
    description.depth = static_cast<std::uint32_t>(depth);
}
void texture_builder_set_levels(NVNtextureBuilder* builder, int levels) {
    state<TextureBuilderState>(builder)->description.levels = static_cast<std::uint32_t>(levels);
}
void texture_builder_set_format(NVNtextureBuilder* builder, NVNformat format) {
    state<TextureBuilderState>(builder)->description.format = format;
}
void texture_builder_set_samples(NVNtextureBuilder* builder, int samples) {
    state<TextureBuilderState>(builder)->description.samples = samples;
}
void texture_builder_set_swizzle(NVNtextureBuilder* builder, NVNtextureSwizzle r,
                                 NVNtextureSwizzle g, NVNtextureSwizzle b,
                                 NVNtextureSwizzle a) {
    std::uint16_t packed = 0;
    if (pack_checked_swizzle("nvnTextureBuilderSetSwizzle", r, g, b, a, packed))
        state<TextureBuilderState>(builder)->swizzle = packed;
}
void texture_builder_set_depth_stencil_mode(NVNtextureBuilder* builder,
                                            NVNtextureDepthStencilMode mode) {
    state<TextureBuilderState>(builder)->depth_stencil_mode = static_cast<std::uint32_t>(mode);
}

std::size_t texture_builder_get_storage_size(const NVNtextureBuilder* builder) {
    const auto layout = resource_layout(state<TextureBuilderState>(builder)->description);
    if (!layout) {
        diagnose_layout("nvnTextureBuilderGetStorageSize", state<TextureBuilderState>(builder)->description);
        return 0;
    }
    return layout->total_size;
}

std::size_t texture_builder_get_storage_alignment(const NVNtextureBuilder* builder) {
    const auto& description = state<TextureBuilderState>(builder)->description;
    const std::size_t alignment = resource_alignment(description);
    if (alignment == 0)
        diagnose("nvnTextureBuilderGetStorageAlignment", "texture format or target is unsupported");
    return alignment;
}

void texture_builder_set_storage(NVNtextureBuilder* builder, NVNmemoryPool* pool, ptrdiff_t offset) {
    auto* current = state<TextureBuilderState>(builder);
    current->pool = state<MemoryPoolState>(pool);
    current->offset = offset;
}

void texture_builder_set_packaged_texture_data(NVNtextureBuilder* builder, const void* data) {
    auto extras = builder_extra(builder);
    extras.packaged_texture_data = data;
    set_builder_extra(builder, extras);
}

void texture_builder_set_packaged_texture_layout(NVNtextureBuilder* builder,
                                                 const NVNpackagedTextureLayout* layout) {
    auto* current = state<TextureBuilderState>(builder);
    current->has_packaged_layout = layout != nullptr;
    std::memset(current->packaged_layout, 0, sizeof(current->packaged_layout));
    if (layout) {
        std::memcpy(current->packaged_layout, layout->layout, sizeof(current->packaged_layout));
        current->description.packaged_block_height = static_cast<std::int32_t>(layout->layout[0]) + 1;
    } else {
        current->description.packaged_block_height = 0;
    }
}

void texture_builder_set_stride(NVNtextureBuilder* builder, ptrdiff_t stride) {
    state<TextureBuilderState>(builder)->description.stride = static_cast<std::uint32_t>(stride);
}

void texture_builder_set_gl_texture_name(NVNtextureBuilder* builder, std::uint32_t name) {
    auto extras = builder_extra(builder);
    extras.gl_texture_name = name;
    set_builder_extra(builder, extras);
}

// Every texture lives in ordinary pool memory with one storage class; pools
// accept it for any texture, so allocators that group textures by class keep
// working. Physical and virtual pools are not part of this class.
constexpr NVNstorageClass ordinary_storage_class = 0x100;

NVNstorageClass texture_builder_get_storage_class(const NVNtextureBuilder*) {
    return ordinary_storage_class;
}
NVNtextureFlags texture_builder_get_flags(const NVNtextureBuilder* builder) {
    return static_cast<NVNtextureFlags>(state<TextureBuilderState>(builder)->description.flags);
}
NVNtextureTarget texture_builder_get_target(const NVNtextureBuilder* builder) {
    return static_cast<NVNtextureTarget>(state<TextureBuilderState>(builder)->description.target);
}
int texture_builder_get_width(const NVNtextureBuilder* builder) {
    return static_cast<int>(state<TextureBuilderState>(builder)->description.width);
}
int texture_builder_get_height(const NVNtextureBuilder* builder) {
    return static_cast<int>(state<TextureBuilderState>(builder)->description.height);
}
int texture_builder_get_depth(const NVNtextureBuilder* builder) {
    return static_cast<int>(state<TextureBuilderState>(builder)->description.depth);
}
int texture_builder_get_levels(const NVNtextureBuilder* builder) {
    return static_cast<int>(state<TextureBuilderState>(builder)->description.levels);
}
NVNformat texture_builder_get_format(const NVNtextureBuilder* builder) {
    return static_cast<NVNformat>(state<TextureBuilderState>(builder)->description.format);
}
int texture_builder_get_samples(const NVNtextureBuilder* builder) {
    return state<TextureBuilderState>(builder)->description.samples;
}
void texture_builder_get_swizzle(const NVNtextureBuilder* builder, NVNtextureSwizzle* r,
                                 NVNtextureSwizzle* g, NVNtextureSwizzle* b,
                                 NVNtextureSwizzle* a) {
    unpack_swizzle(state<TextureBuilderState>(builder)->swizzle, r, g, b, a);
}
NVNtextureDepthStencilMode texture_builder_get_depth_stencil_mode(const NVNtextureBuilder* builder) {
    return static_cast<NVNtextureDepthStencilMode>(state<TextureBuilderState>(builder)->depth_stencil_mode);
}
const void* texture_builder_get_packaged_texture_data(const NVNtextureBuilder* builder) {
    return builder_extra(builder).packaged_texture_data;
}
ptrdiff_t texture_builder_get_stride(const NVNtextureBuilder* builder) {
    return state<TextureBuilderState>(builder)->description.stride;
}
void texture_builder_get_sparse_tile_layout(const NVNtextureBuilder*, NVNtextureSparseTileLayout* layout) {
    if (layout) std::memset(layout, 0, sizeof(*layout));
    diagnose("nvnTextureBuilderGetSparseTileLayout", "sparse textures are unsupported");
}
std::uint32_t texture_builder_get_gl_texture_name(const NVNtextureBuilder* builder) {
    return builder_extra(builder).gl_texture_name;
}
std::size_t texture_builder_get_zcull_storage_size(const NVNtextureBuilder*) { return 0; }
NVNmemoryPool* texture_builder_get_memory_pool(const NVNtextureBuilder* builder) {
    return reinterpret_cast<NVNmemoryPool*>(state<TextureBuilderState>(builder)->pool);
}
ptrdiff_t texture_builder_get_memory_offset(const NVNtextureBuilder* builder) {
    return state<TextureBuilderState>(builder)->offset;
}
NVNdevice* texture_builder_get_device(const NVNtextureBuilder* builder) {
    return state<TextureBuilderState>(builder)->device;
}
NVNboolean texture_builder_get_packaged_texture_layout(const NVNtextureBuilder* builder,
                                                       NVNpackagedTextureLayout* layout) {
    const auto* current = state<TextureBuilderState>(builder);
    if (layout) {
        std::memset(layout, 0, sizeof(*layout));
        if (current->has_packaged_layout)
            std::memcpy(layout->layout, current->packaged_layout, sizeof(layout->layout));
    }
    return current->has_packaged_layout;
}
NVNstorageClass texture_builder_get_raw_storage_class(const NVNtextureBuilder*) {
    return ordinary_storage_class;
}
ptrdiff_t texture_builder_get_view_offset(const NVNtextureBuilder* builder,
                                          const NVNtextureView* view) {
    bool valid = false;
    const std::size_t offset = view_offset(state<TextureBuilderState>(builder)->description,
                                           state<TextureViewState>(view), valid);
    if (!valid || offset > static_cast<std::size_t>(std::numeric_limits<ptrdiff_t>::max())) {
        diagnose("nvnTextureBuilderGetViewOffset", "texture view lies outside builder storage");
        return 0;
    }
    return static_cast<ptrdiff_t>(offset);
}

void texture_view_set_defaults(NVNtextureView* view) {
    std::memset(state<TextureViewState>(view), 0, sizeof(TextureViewState));
}
void texture_view_set_levels(NVNtextureView* view, int base, int count) {
    auto* current = state<TextureViewState>(view);
    current->sets_levels = 1;
    current->base_level = static_cast<std::uint8_t>(base);
    current->level_count = static_cast<std::uint8_t>(count);
}
void texture_view_set_layers(NVNtextureView* view, int first, int count) {
    auto* current = state<TextureViewState>(view);
    current->sets_layers = 1;
    current->min_layer = static_cast<std::int16_t>(first);
    current->layer_count = static_cast<std::int16_t>(count);
}
void texture_view_set_format(NVNtextureView* view, NVNformat format) {
    auto* current = state<TextureViewState>(view);
    current->sets_format = 1;
    current->format = format;
}
void texture_view_set_swizzle(NVNtextureView* view, NVNtextureSwizzle r, NVNtextureSwizzle g,
                              NVNtextureSwizzle b, NVNtextureSwizzle a) {
    std::uint16_t ignored = 0;
    if (!pack_checked_swizzle("nvnTextureViewSetSwizzle", r, g, b, a, ignored)) return;
    auto* current = state<TextureViewState>(view);
    current->sets_swizzle = 1;
    current->swizzle[0] = static_cast<std::uint8_t>(r);
    current->swizzle[1] = static_cast<std::uint8_t>(g);
    current->swizzle[2] = static_cast<std::uint8_t>(b);
    current->swizzle[3] = static_cast<std::uint8_t>(a);
}
void texture_view_set_depth_stencil_mode(NVNtextureView* view, NVNtextureDepthStencilMode mode) {
    auto* current = state<TextureViewState>(view);
    current->sets_depth_stencil = 1;
    current->depth_stencil_mode = static_cast<std::uint32_t>(mode) & 3u;
}
void texture_view_set_target(NVNtextureView* view, NVNtextureTarget target) {
    auto* current = state<TextureViewState>(view);
    current->sets_target = 1;
    current->target = static_cast<std::uint8_t>(target);
}
NVNboolean texture_view_get_levels(const NVNtextureView* view, int* base, int* count) {
    const auto* current = state<TextureViewState>(view);
    if (base) *base = current->base_level;
    if (count) *count = current->level_count;
    return current->sets_levels;
}
NVNboolean texture_view_get_layers(const NVNtextureView* view, int* first, int* count) {
    const auto* current = state<TextureViewState>(view);
    if (first) *first = current->min_layer;
    if (count) *count = current->layer_count;
    return current->sets_layers;
}
NVNboolean texture_view_get_format(const NVNtextureView* view, NVNformat* format) {
    const auto* current = state<TextureViewState>(view);
    if (format) *format = static_cast<NVNformat>(current->format);
    return current->sets_format;
}
NVNboolean texture_view_get_swizzle(const NVNtextureView* view, NVNtextureSwizzle* r,
                                    NVNtextureSwizzle* g, NVNtextureSwizzle* b,
                                    NVNtextureSwizzle* a) {
    const auto* current = state<TextureViewState>(view);
    if (r) *r = static_cast<NVNtextureSwizzle>(current->swizzle[0]);
    if (g) *g = static_cast<NVNtextureSwizzle>(current->swizzle[1]);
    if (b) *b = static_cast<NVNtextureSwizzle>(current->swizzle[2]);
    if (a) *a = static_cast<NVNtextureSwizzle>(current->swizzle[3]);
    return current->sets_swizzle;
}
NVNboolean texture_view_get_depth_stencil_mode(const NVNtextureView* view,
                                               NVNtextureDepthStencilMode* mode) {
    const auto* current = state<TextureViewState>(view);
    if (mode) *mode = static_cast<NVNtextureDepthStencilMode>(current->depth_stencil_mode);
    return current->sets_depth_stencil;
}
NVNboolean texture_view_get_target(const NVNtextureView* view, NVNtextureTarget* target) {
    const auto* current = state<TextureViewState>(view);
    if (target) *target = static_cast<NVNtextureTarget>(current->target);
    return current->sets_target;
}
NVNboolean texture_view_compare(const NVNtextureView* left, const NVNtextureView* right) {
    return std::memcmp(state<TextureViewState>(left), state<TextureViewState>(right),
                       sizeof(TextureViewState)) == 0;
}

NVNboolean texture_initialize(NVNtexture* texture, const NVNtextureBuilder* builder) {
    if (!texture || !builder) {
        diagnose("nvnTextureInitialize", "texture or builder is null");
        return 0;
    }
    const auto* source = state<TextureBuilderState>(builder);
    const auto extras = builder_extra(builder);
    if (extras.gl_texture_name || (source->description.flags & NVN_TEXTURE_FLAGS_SPARSE)) {
        diagnose("nvnTextureInitialize", extras.gl_texture_name ? "GL texture interop is unsupported"
                                                                : "sparse texture storage is unsupported");
        return 0;
    }
    const auto layout = resource_layout(source->description);
    if (!layout) {
        diagnose_layout("nvnTextureInitialize", source->description);
        return 0;
    }
    if (!source->pool || source->pool->tag != pool_tag) {
        diagnose("nvnTextureInitialize", "texture has no initialized memory pool");
        return 0;
    }
    if (source->offset < 0 || static_cast<std::uint64_t>(source->offset) > source->pool->size ||
        layout->total_size > source->pool->size - static_cast<std::uint64_t>(source->offset)) {
        diagnose("nvnTextureInitialize", "texture storage lies outside its memory pool");
        return 0;
    }
    if (!backends().resources) {
        diagnose("nvnTextureInitialize", "resource backend is not installed");
        return 0;
    }
    // Packaged texture data is the texture's storage as the texture packager
    // laid it out, in CPU memory: the driver copies it into the storage here,
    // before the texture's first use reads it, and publishes the bytes as a
    // flushed store for textures already made over them.
    if (extras.packaged_texture_data) {
        if (!source->pool->memory) {
            diagnose("nvnTextureInitialize", "packaged texture data needs a pool with CPU storage");
            return 0;
        }
        auto* storage = source->pool->memory + static_cast<std::size_t>(source->offset);
        const std::span bytes(static_cast<const std::byte*>(extras.packaged_texture_data),
                              static_cast<std::size_t>(layout->total_size));
        on_host_stack([&] {
            backends().resources->prepare_pool_write(*source->pool, static_cast<std::uint64_t>(source->offset),
                                                     bytes.size());
        });
        if (!memory::WriteTracker::try_alias_store(storage, bytes)) {
            memory::WriteTracker::prepare_host_write(storage, bytes.size());
            std::memcpy(storage, bytes.data(), bytes.size());
        }
        on_host_stack([&] {
            backends().resources->flush_pool(*source->pool, static_cast<std::uint64_t>(source->offset), bytes.size());
        });
    }
    auto* destination = state<TextureState>(texture);
    TextureState initialized{};
    initialized.tag = texture_tag;
    initialized.description = source->description;
    initialized.size = layout->total_size;
    initialized.layer_size = layout->layer_stride;
    initialized.pool = source->pool;
    initialized.offset = source->offset;
    initialized.memory = source->pool->memory
                             ? source->pool->memory + static_cast<std::size_t>(source->offset)
                             : nullptr;
    initialized.has_packaged_layout = source->has_packaged_layout;
    std::memcpy(initialized.packaged_layout, source->packaged_layout,
                sizeof(initialized.packaged_layout));
    initialized.swizzle = source->swizzle;
    initialized.depth_stencil_mode = source->depth_stencil_mode & 3u;
    const auto* snapshot = identity_snapshot(initialized);
    if (!snapshot) {
        diagnose("nvnTextureInitialize", "texture snapshot allocation failed");
        return 0;
    }
    initialized.serial = snapshot->serial;
    initialized.snapshot = snapshot;
    // The identity stays registered for as long as its pool: other objects of
    // the same identity may still be live, and its serial names the texture.
    *destination = initialized;
    return 1;
}

std::size_t texture_get_zcull_storage_size(const NVNtexture*) { return 0; }
void texture_finalize(NVNtexture* texture) {
    auto* current = state<TextureState>(texture);
    if (!valid_texture(current, "nvnTextureFinalize")) return;
    // The identity's registration outlives this object (see TextureIdentity).
    std::memset(current, 0, sizeof(*current));
}
void texture_set_debug_label(NVNtexture*, const char*) {}
NVNstorageClass texture_get_storage_class(const NVNtexture*) {
    return ordinary_storage_class;
}
ptrdiff_t texture_get_view_offset(const NVNtexture* texture, const NVNtextureView* view) {
    const auto* current = state<TextureState>(texture);
    if (!valid_texture(current, "nvnTextureGetViewOffset")) return 0;
    bool valid = false;
    const std::size_t offset = view_offset(current->description, state<TextureViewState>(view), valid);
    if (!valid || offset > static_cast<std::size_t>(std::numeric_limits<ptrdiff_t>::max())) {
        diagnose("nvnTextureGetViewOffset", "texture view lies outside storage");
        return 0;
    }
    return static_cast<ptrdiff_t>(offset);
}
NVNtextureFlags texture_get_flags(const NVNtexture* texture) {
    return static_cast<NVNtextureFlags>(state<TextureState>(texture)->description.flags);
}
NVNtextureTarget texture_get_target(const NVNtexture* texture) {
    return static_cast<NVNtextureTarget>(state<TextureState>(texture)->description.target);
}
int texture_get_width(const NVNtexture* texture) {
    return static_cast<int>(state<TextureState>(texture)->description.width);
}
int texture_get_height(const NVNtexture* texture) {
    return static_cast<int>(state<TextureState>(texture)->description.height);
}
int texture_get_depth(const NVNtexture* texture) {
    return static_cast<int>(state<TextureState>(texture)->description.depth);
}
int texture_get_levels(const NVNtexture* texture) {
    return static_cast<int>(state<TextureState>(texture)->description.levels);
}
NVNformat texture_get_format(const NVNtexture* texture) {
    return static_cast<NVNformat>(state<TextureState>(texture)->description.format);
}
int texture_get_samples(const NVNtexture* texture) {
    return state<TextureState>(texture)->description.samples;
}
void texture_get_swizzle(const NVNtexture* texture, NVNtextureSwizzle* r, NVNtextureSwizzle* g,
                         NVNtextureSwizzle* b, NVNtextureSwizzle* a) {
    unpack_swizzle(state<TextureState>(texture)->swizzle, r, g, b, a);
}
NVNtextureDepthStencilMode texture_get_depth_stencil_mode(const NVNtexture* texture) {
    return static_cast<NVNtextureDepthStencilMode>(state<TextureState>(texture)->depth_stencil_mode);
}
ptrdiff_t texture_get_stride(const NVNtexture* texture) {
    return state<TextureState>(texture)->description.stride;
}
NVNtextureAddress texture_get_texture_address(const NVNtexture* texture) {
    return reinterpret_cast<NVNtextureAddress>(state<TextureState>(texture)->memory);
}
void texture_get_sparse_tile_layout(const NVNtexture*, NVNtextureSparseTileLayout* layout) {
    if (layout) std::memset(layout, 0, sizeof(*layout));
    diagnose("nvnTextureGetSparseTileLayout", "sparse textures are unsupported");
}

void texture_write_texels_strided(const NVNtexture* texture, const NVNtextureView* view,
                                  const NVNcopyRegion* region, const void* source,
                                  ptrdiff_t row_stride, ptrdiff_t image_stride) {
    const auto* current = state<TextureState>(texture);
    if (!valid_texture(current, "nvnTextureWriteTexelsStrided") || !source) {
        if (!source) diagnose("nvnTextureWriteTexelsStrided", "source pointer is null");
        return;
    }
    TexelTransfer transfer;
    if (!view_transfer(*current, state<TextureViewState>(view), region, row_stride, image_stride,
                       transfer, "nvnTextureWriteTexelsStrided")) return;
    if (!backends().resources) {
        diagnose("nvnTextureWriteTexelsStrided", "resource backend is not installed");
        return;
    }
    if (!backends().resources->write_texels(*current, transfer,
                                            static_cast<const std::byte*>(source)))
        diagnose("nvnTextureWriteTexelsStrided", "resource backend rejected the texel write");
}
void texture_write_texels(const NVNtexture* texture, const NVNtextureView* view,
                          const NVNcopyRegion* region, const void* source) {
    texture_write_texels_strided(texture, view, region, source, 0, 0);
}
void texture_read_texels_strided(const NVNtexture* texture, const NVNtextureView* view,
                                 const NVNcopyRegion* region, void* destination,
                                 ptrdiff_t row_stride, ptrdiff_t image_stride) {
    const auto* current = state<TextureState>(texture);
    if (!valid_texture(current, "nvnTextureReadTexelsStrided") || !destination) {
        if (!destination) diagnose("nvnTextureReadTexelsStrided", "destination pointer is null");
        return;
    }
    TexelTransfer transfer;
    if (!view_transfer(*current, state<TextureViewState>(view), region, row_stride, image_stride,
                       transfer, "nvnTextureReadTexelsStrided")) return;
    if (!backends().resources) {
        diagnose("nvnTextureReadTexelsStrided", "resource backend is not installed");
        return;
    }
    if (!backends().resources->read_texels(*current, transfer, static_cast<std::byte*>(destination)))
        diagnose("nvnTextureReadTexelsStrided", "resource backend rejected the texel read");
}
void texture_read_texels(const NVNtexture* texture, const NVNtextureView* view,
                         const NVNcopyRegion* region, void* destination) {
    texture_read_texels_strided(texture, view, region, destination, 0, 0);
}

void texture_cache_range(const NVNtexture* texture, const NVNtextureView* view,
                         bool invalidate, const char* function) {
    const auto* current = state<TextureState>(texture);
    if (!valid_texture(current, function)) return;
    if (!current->pool || current->pool->tag != pool_tag || !backends().resources) {
        diagnose(function, "texture pool or resource backend is unavailable");
        return;
    }
    std::uint32_t first_layer = 0;
    std::uint32_t layer_count = 1;
    const auto layout = resource_layout(current->description);
    if (!layout) {
        diagnose_layout(function, current->description);
        return;
    }
    if (!current->pool->memory || !pool_cpu_mapped(*current->pool))
        return;
    layer_count = layout->layers;
    // Cache maintenance covers whole layer allocations, ignoring the region
    // and mip selection. Uncached pools and 3D textures also ignore the view.
    const auto* selected = state<TextureViewState>(view);
    if ((current->pool->flags & NVN_MEMORY_POOL_FLAGS_CPU_CACHED) &&
        current->description.target != NVN_TEXTURE_TARGET_3D && selected && selected->sets_layers) {
        if (selected->min_layer < 0 || selected->layer_count < 0 ||
            static_cast<std::uint32_t>(selected->min_layer) > layout->layers) {
            diagnose(function, "texture view layer range is invalid");
            return;
        }
        first_layer = static_cast<std::uint32_t>(selected->min_layer);
        layer_count = selected->layer_count ? static_cast<std::uint32_t>(selected->layer_count)
                                             : layout->layers - first_layer;
        if (layer_count > layout->layers - first_layer) {
            diagnose(function, "texture view layer range is invalid");
            return;
        }
    }
    if (layer_count == 0) return;
    std::size_t begin = 0;
    std::size_t size = 0;
    if (!checked_multiply(first_layer, layout->layer_stride, begin) ||
        !checked_multiply(layer_count, layout->layer_stride, size) ||
        begin > current->size || size > current->size - begin) {
        diagnose(function, "texture cache range exceeds storage");
        return;
    }
    const std::uint64_t pool_offset = static_cast<std::uint64_t>(current->offset) + begin;
    if (invalidate) {
        backends().resources->invalidate_pool(*current->pool, pool_offset, size);
        return;
    }
    // A flush publishes CPU texel stores. The layer certificates say whether a
    // layer changed since it was last published; when none did, the flush is a
    // no-op and costs no upload, GPU wait or readback.
    const auto* storage = reinterpret_cast<const std::byte*>(current->pool->memory) +
        current->offset;
    const auto layer_size = static_cast<std::uint64_t>(layout->layer_stride);
    const auto status = graphics::texflush::verify(current->serial, storage, layer_size,
                                                   layout->layers, first_layer, layer_count);
    if (status == graphics::texflush::Status::clean) return;
    backends().resources->flush_pool(*current->pool, pool_offset, size);
    graphics::texflush::note(current->serial, storage, layer_size, layout->layers,
                             first_layer, layer_count, true);
}
void texture_flush_texels(const NVNtexture* texture, const NVNtextureView* view,
                          const NVNcopyRegion*) {
    texture_cache_range(texture, view, false, "nvnTextureFlushTexels");
}
void texture_invalidate_texels(const NVNtexture* texture, const NVNtextureView* view,
                               const NVNcopyRegion*) {
    texture_cache_range(texture, view, true, "nvnTextureInvalidateTexels");
}
NVNmemoryPool* texture_get_memory_pool(const NVNtexture* texture) {
    return reinterpret_cast<NVNmemoryPool*>(state<TextureState>(texture)->pool);
}
ptrdiff_t texture_get_memory_offset(const NVNtexture* texture) {
    return state<TextureState>(texture)->offset;
}
int texture_get_storage_size(const NVNtexture* texture) {
    const auto size = state<TextureState>(texture)->size;
    if (size > static_cast<std::uint64_t>(std::numeric_limits<int>::max())) {
        diagnose("nvnTextureGetStorageSize", "texture storage size exceeds the ABI result range");
        return std::numeric_limits<int>::max();
    }
    return static_cast<int>(size);
}
NVNboolean texture_compare(const NVNtexture* left, const NVNtexture* right) {
    const auto* a = state<TextureState>(left);
    const auto* b = state<TextureState>(right);
    return std::memcmp(&a->description, &b->description, sizeof(a->description)) == 0 &&
           a->memory == b->memory && a->swizzle == b->swizzle &&
           a->depth_stencil_mode == b->depth_stencil_mode;
}
std::uint64_t texture_get_debug_id(const NVNtexture* texture) {
    return state<TextureState>(texture)->serial;
}
NVNstorageClass texture_get_raw_storage_class(const NVNtexture*) {
    return ordinary_storage_class;
}

template <typename Function>
GenericFunction generic(Function function) noexcept {
    return reinterpret_cast<GenericFunction>(function);
}

} // namespace

std::optional<graphics::TextureFormat> resource_format(std::int32_t format) {
    using graphics::TextureFormat;
    switch (format) {
        case NVN_FORMAT_R8: return TextureFormat::R8Unorm;
        case NVN_FORMAT_R8SN: return TextureFormat::R8Snorm;
        case NVN_FORMAT_R8UI: return TextureFormat::R8Uint;
        case NVN_FORMAT_R8I: return TextureFormat::R8Sint;
        case NVN_FORMAT_R16F: return TextureFormat::R16Float;
        case NVN_FORMAT_R16: return TextureFormat::R16Unorm;
        case NVN_FORMAT_R16SN: return TextureFormat::R16Snorm;
        case NVN_FORMAT_R16UI: return TextureFormat::R16Uint;
        case NVN_FORMAT_R16I: return TextureFormat::R16Sint;
        case NVN_FORMAT_R32F: return TextureFormat::R32Float;
        case NVN_FORMAT_R32UI: return TextureFormat::R32Uint;
        case NVN_FORMAT_R32I: return TextureFormat::R32Sint;
        case NVN_FORMAT_RG8: return TextureFormat::R8G8Unorm;
        case NVN_FORMAT_RG8SN: return TextureFormat::R8G8Snorm;
        case NVN_FORMAT_RG8UI: return TextureFormat::R8G8Uint;
        case NVN_FORMAT_RG8I: return TextureFormat::R8G8Sint;
        case NVN_FORMAT_RG16F: return TextureFormat::R16G16Float;
        case NVN_FORMAT_RG16: return TextureFormat::R16G16Unorm;
        case NVN_FORMAT_RG16SN: return TextureFormat::R16G16Snorm;
        case NVN_FORMAT_RG16UI: return TextureFormat::R16G16Uint;
        case NVN_FORMAT_RG16I: return TextureFormat::R16G16Sint;
        case NVN_FORMAT_RG32F: return TextureFormat::R32G32Float;
        case NVN_FORMAT_RG32UI: return TextureFormat::R32G32Uint;
        case NVN_FORMAT_RG32I: return TextureFormat::R32G32Sint;
        case NVN_FORMAT_RGB32F: return TextureFormat::R32G32B32Float;
        case NVN_FORMAT_RGB32UI: return TextureFormat::R32G32B32Uint;
        case NVN_FORMAT_RGB32I: return TextureFormat::R32G32B32Sint;
        case NVN_FORMAT_RGBA8:
        case NVN_FORMAT_RGBX8: return TextureFormat::R8G8B8A8Unorm;
        case NVN_FORMAT_RGBA8SN:
        case NVN_FORMAT_RGBX8SN: return TextureFormat::R8G8B8A8Snorm;
        case NVN_FORMAT_RGBA8UI:
        case NVN_FORMAT_RGBX8UI: return TextureFormat::R8G8B8A8Uint;
        case NVN_FORMAT_RGBA8I:
        case NVN_FORMAT_RGBX8I: return TextureFormat::R8G8B8A8Sint;
        case NVN_FORMAT_RGBA8_SRGB:
        case NVN_FORMAT_RGBX8_SRGB: return TextureFormat::R8G8B8A8Srgb;
        case NVN_FORMAT_BGRA8:
        case NVN_FORMAT_BGRX8: return TextureFormat::B8G8R8A8Unorm;
        case NVN_FORMAT_BGRA8_SRGB:
        case NVN_FORMAT_BGRX8_SRGB: return TextureFormat::B8G8R8A8Srgb;
        case NVN_FORMAT_RGBA16F:
        case NVN_FORMAT_RGBX16F: return TextureFormat::R16G16B16A16Float;
        case NVN_FORMAT_RGBA16:
        case NVN_FORMAT_RGBX16: return TextureFormat::R16G16B16A16Unorm;
        case NVN_FORMAT_RGBA16SN:
        case NVN_FORMAT_RGBX16SN: return TextureFormat::R16G16B16A16Snorm;
        case NVN_FORMAT_RGBA16UI:
        case NVN_FORMAT_RGBX16UI: return TextureFormat::R16G16B16A16Uint;
        case NVN_FORMAT_RGBA16I:
        case NVN_FORMAT_RGBX16I: return TextureFormat::R16G16B16A16Sint;
        case NVN_FORMAT_RGBA32F:
        case NVN_FORMAT_RGBX32F: return TextureFormat::R32G32B32A32Float;
        case NVN_FORMAT_RGBA32UI:
        case NVN_FORMAT_RGBX32UI: return TextureFormat::R32G32B32A32Uint;
        case NVN_FORMAT_RGBA32I:
        case NVN_FORMAT_RGBX32I: return TextureFormat::R32G32B32A32Sint;
        case NVN_FORMAT_DEPTH16: return TextureFormat::D16Unorm;
        case NVN_FORMAT_DEPTH24: return TextureFormat::X8UintD24Unorm;
        case NVN_FORMAT_DEPTH32F: return TextureFormat::D32Float;
        case NVN_FORMAT_DEPTH24_STENCIL8: return TextureFormat::D24UnormS8Uint;
        case NVN_FORMAT_DEPTH32F_STENCIL8: return TextureFormat::D32FloatS8Uint;
        case NVN_FORMAT_RGBA4: return TextureFormat::R4G4B4A4Unorm;
        case NVN_FORMAT_RGB5:
        case NVN_FORMAT_RGB5A1: return TextureFormat::R5G5B5A1Unorm;
        case NVN_FORMAT_RGB565: return TextureFormat::R5G6B5Unorm;
        case NVN_FORMAT_RGB10A2: return TextureFormat::R10G10B10A2Unorm;
        case NVN_FORMAT_RGB10A2UI: return TextureFormat::R10G10B10A2Uint;
        case NVN_FORMAT_R11G11B10F: return TextureFormat::R11G11B10Float;
        case NVN_FORMAT_RGB9E5F: return TextureFormat::R9G9B9E5Float;
        case NVN_FORMAT_RGB_DXT1:
        case NVN_FORMAT_RGBA_DXT1: return TextureFormat::Bc1RgbaUnorm;
        case NVN_FORMAT_RGBA_DXT3: return TextureFormat::Bc2Unorm;
        case NVN_FORMAT_RGBA_DXT5: return TextureFormat::Bc3Unorm;
        case NVN_FORMAT_RGB_DXT1_SRGB:
        case NVN_FORMAT_RGBA_DXT1_SRGB: return TextureFormat::Bc1RgbaSrgb;
        case NVN_FORMAT_RGBA_DXT3_SRGB: return TextureFormat::Bc2Srgb;
        case NVN_FORMAT_RGBA_DXT5_SRGB: return TextureFormat::Bc3Srgb;
        case NVN_FORMAT_RGTC1_UNORM: return TextureFormat::Bc4Unorm;
        case NVN_FORMAT_RGTC1_SNORM: return TextureFormat::Bc4Snorm;
        case NVN_FORMAT_RGTC2_UNORM: return TextureFormat::Bc5Unorm;
        case NVN_FORMAT_RGTC2_SNORM: return TextureFormat::Bc5Snorm;
        case NVN_FORMAT_BPTC_UNORM: return TextureFormat::Bc7Unorm;
        case NVN_FORMAT_BPTC_UNORM_SRGB: return TextureFormat::Bc7Srgb;
        case NVN_FORMAT_BPTC_SFLOAT: return TextureFormat::Bc6HSfloat;
        case NVN_FORMAT_BPTC_UFLOAT: return TextureFormat::Bc6HUfloat;
        // Scaled integer formats need shader-side numeric conversion and
        // cannot be represented by an integer sampling format alone.
#define ARMIGO_ASTC_CASE(name, value) case NVN_FORMAT_RGBA_ASTC_##name: return TextureFormat::Astc##value##Unorm
#define ARMIGO_ASTC_SRGB_CASE(name, value) case NVN_FORMAT_RGBA_ASTC_##name##_SRGB: return TextureFormat::Astc##value##Srgb
        ARMIGO_ASTC_CASE(4x4, 4x4);
        ARMIGO_ASTC_CASE(5x4, 5x4);
        ARMIGO_ASTC_CASE(5x5, 5x5);
        ARMIGO_ASTC_CASE(6x5, 6x5);
        ARMIGO_ASTC_CASE(6x6, 6x6);
        ARMIGO_ASTC_CASE(8x5, 8x5);
        ARMIGO_ASTC_CASE(8x6, 8x6);
        ARMIGO_ASTC_CASE(8x8, 8x8);
        ARMIGO_ASTC_CASE(10x5, 10x5);
        ARMIGO_ASTC_CASE(10x6, 10x6);
        ARMIGO_ASTC_CASE(10x8, 10x8);
        ARMIGO_ASTC_CASE(10x10, 10x10);
        ARMIGO_ASTC_CASE(12x10, 12x10);
        ARMIGO_ASTC_CASE(12x12, 12x12);
        ARMIGO_ASTC_SRGB_CASE(4x4, 4x4);
        ARMIGO_ASTC_SRGB_CASE(5x4, 5x4);
        ARMIGO_ASTC_SRGB_CASE(5x5, 5x5);
        ARMIGO_ASTC_SRGB_CASE(6x5, 6x5);
        ARMIGO_ASTC_SRGB_CASE(6x6, 6x6);
        ARMIGO_ASTC_SRGB_CASE(8x5, 8x5);
        ARMIGO_ASTC_SRGB_CASE(8x6, 8x6);
        ARMIGO_ASTC_SRGB_CASE(8x8, 8x8);
        ARMIGO_ASTC_SRGB_CASE(10x5, 10x5);
        ARMIGO_ASTC_SRGB_CASE(10x6, 10x6);
        ARMIGO_ASTC_SRGB_CASE(10x8, 10x8);
        ARMIGO_ASTC_SRGB_CASE(10x10, 10x10);
        ARMIGO_ASTC_SRGB_CASE(12x10, 12x10);
        ARMIGO_ASTC_SRGB_CASE(12x12, 12x12);
#undef ARMIGO_ASTC_CASE
#undef ARMIGO_ASTC_SRGB_CASE
        default: return std::nullopt;
    }
}

std::optional<graphics::TextureLayoutDescription> resource_layout(
    const TextureDescription& description) {
    const FormatGeometry format = format_geometry(description.format);
    if (format.bytes_per_block == 0 || description.width == 0 || description.levels == 0 ||
        description.samples < 0) return std::nullopt;
    graphics::TextureLayoutDescription layout{};
    layout.width = description.width;
    layout.height = description.height;
    layout.depth = 1;
    layout.levels = description.levels;
    layout.layers = 1;
    layout.block_width = format.block_width;
    layout.block_height = format.block_height;
    layout.bytes_per_block = format.bytes_per_block;
    switch (description.target) {
        case NVN_TEXTURE_TARGET_1D:
            layout.height = 1;
            break;
        case NVN_TEXTURE_TARGET_1D_ARRAY:
            layout.layers = description.height;
            layout.height = 1;
            break;
        case NVN_TEXTURE_TARGET_3D:
            layout.depth = description.depth ? description.depth : 1;
            break;
        case NVN_TEXTURE_TARGET_2D_ARRAY:
        case NVN_TEXTURE_TARGET_2D_MULTISAMPLE_ARRAY:
            layout.layers = description.depth ? description.depth : 1;
            break;
        case NVN_TEXTURE_TARGET_CUBEMAP:
            layout.layers = 6;
            break;
        case NVN_TEXTURE_TARGET_CUBEMAP_ARRAY:
            if (description.depth > std::numeric_limits<std::uint32_t>::max() / 6) return std::nullopt;
            layout.layers = description.depth * 6;
            break;
        case NVN_TEXTURE_TARGET_2D:
        case NVN_TEXTURE_TARGET_2D_MULTISAMPLE:
        case NVN_TEXTURE_TARGET_RECTANGLE:
            break;
        case NVN_TEXTURE_TARGET_BUFFER: {
            layout.height = 1;
            layout.levels = 1;
            layout.storage = graphics::TextureStorageLayout::Pitch;
            if (!checked_multiply(layout.width, layout.bytes_per_block, layout.pitch_bytes))
                return std::nullopt;
            break;
        }
        default:
            return std::nullopt;
    }
    std::uint32_t sample_x = 1;
    std::uint32_t sample_y = 1;
    if (description.target == NVN_TEXTURE_TARGET_BUFFER) {
        sample_x = sample_y = 1;
    } else if (description.samples == 2) sample_x = 2;
    else if (description.samples == 4) sample_x = sample_y = 2;
    else if (description.samples == 8) {
        sample_x = 4;
        sample_y = 2;
    } else if (description.samples > 1) {
        return std::nullopt;
    }
    if (layout.width > std::numeric_limits<std::uint32_t>::max() / sample_x ||
        layout.height > std::numeric_limits<std::uint32_t>::max() / sample_y) return std::nullopt;
    layout.width *= sample_x;
    layout.height *= sample_y;
    if (description.target != NVN_TEXTURE_TARGET_BUFFER && (description.flags & pitch_flags)) {
        layout.storage = graphics::TextureStorageLayout::Pitch;
        layout.levels = 1;
        const std::size_t columns = (layout.width + layout.block_width - 1) / layout.block_width;
        std::size_t minimum_pitch = 0;
        if (!checked_multiply(columns, layout.bytes_per_block, minimum_pitch)) return std::nullopt;
        const std::size_t alignment = description.flags & NVN_TEXTURE_FLAGS_LINEAR_RENDER_TARGET ? 128 : 32;
        layout.pitch_bytes = description.stride;
        if (layout.pitch_bytes == 0) {
            if (minimum_pitch > std::numeric_limits<std::size_t>::max() - (alignment - 1))
                return std::nullopt;
            layout.pitch_bytes = (minimum_pitch + alignment - 1) & ~(alignment - 1);
        }
    } else if (layout.storage == graphics::TextureStorageLayout::BlockLinear) {
        if (description.packaged_block_height < 0 || description.packaged_block_height > 6)
            return std::nullopt;
        if (description.target == NVN_TEXTURE_TARGET_3D) {
            const std::uint64_t expanded = std::uint64_t{layout.depth} + layout.depth / 2;
            layout.gob_block_depth_log2 = pick_gob_block_log2(static_cast<std::uint32_t>(
                std::min<std::uint64_t>(expanded, std::numeric_limits<std::uint32_t>::max())));
        } else if (description.target != NVN_TEXTURE_TARGET_1D &&
                   description.target != NVN_TEXTURE_TARGET_1D_ARRAY) {
            const std::uint32_t rows = (layout.height + layout.block_height - 1) / layout.block_height;
            const std::uint64_t gobs = (std::uint64_t{rows} + rows / 2 + 7) / 8;
            layout.gob_block_height_log2 = description.packaged_block_height
                                               ? static_cast<std::uint32_t>(description.packaged_block_height - 1)
                                               : pick_gob_block_log2(static_cast<std::uint32_t>(gobs));
        }
    }
    auto calculated = graphics::calculate_texture_size(std::move(layout));
    if (auto* value = std::get_if<graphics::TextureLayoutDescription>(&calculated)) {
        if (value->storage == graphics::TextureStorageLayout::BlockLinear) {
            std::size_t chain_size = 0;
            for (auto& mip : value->mips) {
                auto depth_log2 = value->gob_block_depth_log2;
                while (depth_log2 && mip.depth <= (1u << (depth_log2 - 1))) --depth_log2;
                mip.gob_block_depth_log2 = depth_log2;
                const std::size_t block_height = std::size_t{1} << mip.gob_block_height_log2;
                const std::size_t block_depth = std::size_t{1} << depth_log2;
                std::size_t row_bytes = 0;
                std::size_t blocks = 0;
                if (!checked_multiply(mip.block_columns, value->bytes_per_block, row_bytes) ||
                    !checked_multiply((row_bytes + 63) / 64,
                        (mip.block_rows + 8 * block_height - 1) / (8 * block_height), blocks) ||
                    !checked_multiply(blocks, 512 * block_height * block_depth, mip.slice_size) ||
                    !checked_multiply(mip.slice_size,
                        (mip.depth + block_depth - 1) / block_depth, mip.size)) return std::nullopt;
                mip.offset = chain_size;
                if (!checked_add(chain_size, mip.size, chain_size)) return std::nullopt;
            }
            const bool array = description.target == NVN_TEXTURE_TARGET_1D_ARRAY ||
                description.target == NVN_TEXTURE_TARGET_2D_ARRAY ||
                description.target == NVN_TEXTURE_TARGET_2D_MULTISAMPLE_ARRAY ||
                description.target == NVN_TEXTURE_TARGET_CUBEMAP ||
                description.target == NVN_TEXTURE_TARGET_CUBEMAP_ARRAY;
            if (array) {
                auto height_log2 = value->gob_block_height_log2;
                const auto rows = (std::uint64_t{description.height} + value->block_height - 1) /
                                  value->block_height;
                while (height_log2 && rows <= (8u << (height_log2 - 1))) --height_log2;
                const auto alignment = std::size_t{512} << height_log2;
                if (chain_size > std::numeric_limits<std::size_t>::max() - alignment + 1)
                    return std::nullopt;
                chain_size = (chain_size + alignment - 1) & ~(alignment - 1);
            }
            value->layer_stride = chain_size;
            if (!checked_multiply(chain_size, value->layers, value->total_size) ||
                value->total_size > graphics::max_texture_storage_size) return std::nullopt;
        }
        return std::move(*value);
    }
    return std::nullopt;
}

std::size_t resource_alignment(const TextureDescription& description) {
    const FormatGeometry format = format_geometry(description.format);
    if (format.bytes_per_block == 0) return 0;
    if (description.target == NVN_TEXTURE_TARGET_BUFFER)
        return format.bytes_per_block == 12 ? 4 : format.bytes_per_block;
    if (description.flags & NVN_TEXTURE_FLAGS_LINEAR_RENDER_TARGET) return 128;
    if (description.flags & NVN_TEXTURE_FLAGS_LINEAR) return 32;
    return 512;
}

bool snapshot_texel_input(const TextureState& texture, const TexelTransfer& transfer,
                          const std::byte* input, std::vector<std::byte>& bytes,
                          TexelTransfer& packed) {
    const auto layout = resource_layout(texture.description);
    TransferGeometry geometry;
    if (!input || !layout || !transfer_geometry(texture, *layout, transfer, geometry)) return false;
    packed = transfer;
    if (geometry.image_count == 0) { bytes.clear(); return true; }
    std::size_t image_size = 0, total = 0, external_size = 0, last_row = 0;
    if (!checked_multiply(geometry.row_bytes, geometry.block_rows, image_size) ||
        !checked_multiply(image_size, geometry.image_count, total) ||
        !checked_multiply(geometry.image_count - 1, geometry.image_stride, external_size) ||
        !checked_multiply(geometry.block_rows - 1, geometry.row_stride, last_row) ||
        !checked_add(external_size, last_row, external_size) ||
        !checked_add(external_size, geometry.row_bytes, external_size) ||
        external_size > std::numeric_limits<std::uintptr_t>::max() - reinterpret_cast<std::uintptr_t>(input))
        return false;
    try { bytes.resize(total); } catch (const std::bad_alloc&) { return false; }
    for (std::size_t image = 0; image < geometry.image_count; ++image)
        for (std::size_t row = 0; row < geometry.block_rows; ++row)
            std::memcpy(bytes.data() + image * image_size + row * geometry.row_bytes,
                        input + image * geometry.image_stride + row * geometry.row_stride,
                        geometry.row_bytes);
    packed.row_stride = geometry.row_bytes;
    packed.image_stride = image_size;
    return true;
}

bool transfer_pool_texels(const TextureState& texture, const TexelTransfer& transfer,
                          std::byte* bytes, bool write) {
    if (!bytes || !texture.memory || texture.size == 0) return false;
    const auto layout = resource_layout(texture.description);
    if (!layout || texture.size < layout->total_size) return false;
    TransferGeometry geometry;
    if (!transfer_geometry(texture, *layout, transfer, geometry)) return false;
    if (geometry.image_count == 0) return true;
    std::size_t external_size = 0;
    std::size_t last_row = 0;
    if (!checked_multiply(geometry.image_count - 1, geometry.image_stride, external_size) ||
        !checked_multiply(geometry.block_rows - 1, geometry.row_stride, last_row) ||
        !checked_add(external_size, last_row, external_size) ||
        !checked_add(external_size, geometry.row_bytes, external_size)) return false;
    const auto external_begin = reinterpret_cast<std::uintptr_t>(bytes);
    const auto guest_begin = reinterpret_cast<std::uintptr_t>(texture.memory);
    if (external_size > std::numeric_limits<std::uintptr_t>::max() - external_begin ||
        texture.size > std::numeric_limits<std::uintptr_t>::max() - guest_begin) return false;
    const bool overlap = external_begin < guest_begin + texture.size &&
                         guest_begin < external_begin + external_size;
    std::vector<std::byte> temporary;
    auto* original_bytes = bytes;
    const auto original_row = geometry.row_stride;
    const auto original_image = geometry.image_stride;
    auto copy_rows = [&](std::byte* destination, std::size_t destination_row,
                         std::size_t destination_image, const std::byte* source,
                         std::size_t source_row, std::size_t source_image) {
        for (std::size_t image = 0; image < geometry.image_count; ++image)
            for (std::size_t row = 0; row < geometry.block_rows; ++row)
                std::memcpy(destination + image * destination_image + row * destination_row,
                            source + image * source_image + row * source_row, geometry.row_bytes);
    };
    if (overlap) {
        // Snapshot all source images before any write, including transfers
        // whose linear input or output aliases a later tiled array layer.
        std::size_t image_size = 0;
        std::size_t total = 0;
        if (!checked_multiply(geometry.row_bytes, geometry.block_rows, image_size) ||
            !checked_multiply(image_size, geometry.image_count, total)) return false;
        try { temporary.resize(total); } catch (const std::bad_alloc&) { return false; }
        if (write) copy_rows(temporary.data(), geometry.row_bytes, image_size,
                             bytes, original_row, original_image);
        bytes = temporary.data();
        geometry.row_stride = geometry.row_bytes;
        geometry.image_stride = image_size;
    }
    std::span<std::byte> guest(reinterpret_cast<std::byte*>(texture.memory),
                               static_cast<std::size_t>(texture.size));
    if (geometry.is_3d) {
        if (!transfer_one_level(*layout, geometry, guest, bytes, write, 0, 0,
                                geometry.image_count)) return false;
    } else {
        for (std::uint32_t image = 0; image < geometry.image_count; ++image)
            if (!transfer_one_level(*layout, geometry, guest, bytes, write,
                                    geometry.first_layer + image, image, 1)) return false;
    }
    if (overlap && !write) copy_rows(original_bytes, original_row, original_image,
                                    bytes, geometry.row_stride, geometry.image_stride);
    return true;
}

bool pool_texture_addressing(const TextureState& texture,
                             std::vector<graphics::BlockLinearLevelAddressing>& levels) {
    levels.clear();
    if (!texture.memory || texture.size == 0) return false;
    const auto layout = resource_layout(texture.description);
    if (!layout || layout->storage != graphics::TextureStorageLayout::BlockLinear ||
        texture.size < layout->total_size) return false;
    const bool is_3d = texture.description.target == NVN_TEXTURE_TARGET_3D;
    // A whole-level transfer of each layer, as transfer_pool_texels reads it:
    // the same geometry checks, the same per-level storage layout and the same
    // level placement inside the texture's storage.
    std::vector<graphics::BlockLinearLevelAddressing> per_level;
    try {
        levels.reserve(std::size_t{layout->layers} * layout->mips.size());
        per_level.reserve(layout->mips.size());
    } catch (const std::bad_alloc&) {
        return false;
    }
    for (std::uint32_t level = 0; level < layout->mips.size(); ++level) {
        const auto& mip = layout->mips[level];
        TexelTransfer transfer;
        transfer.level = level;
        transfer.region = {0, 0, 0, static_cast<std::int32_t>(mip.width),
                           static_cast<std::int32_t>(mip.height),
                           static_cast<std::int32_t>(is_3d ? mip.depth : 1)};
        TransferGeometry geometry;
        if (!transfer_geometry(texture, *layout, transfer, geometry) ||
            geometry.image_count != (is_3d ? mip.depth : 1) || geometry.x_block ||
            geometry.y_block || geometry.block_columns != mip.block_columns ||
            geometry.block_rows != mip.block_rows ||
            geometry.row_bytes != mip.linear_row_bytes) return false;
        auto calculated = level_storage_layout(*layout, level);
        const auto* level_layout = std::get_if<graphics::TextureLayoutDescription>(&calculated);
        if (!level_layout) return false;
        auto addressed = graphics::block_linear_level_addressing(*level_layout, 0, 0);
        const auto* addressing = std::get_if<graphics::BlockLinearLevelAddressing>(&addressed);
        if (!addressing || addressing->guest_base != 0 || addressing->guest_size > mip.size ||
            addressing->row_bytes != mip.linear_row_bytes ||
            addressing->rows != mip.block_rows ||
            addressing->slices != (is_3d ? mip.depth : 1)) return false;
        per_level.push_back(*addressing);
    }
    for (std::uint32_t layer = 0; layer < layout->layers; ++layer) {
        for (std::uint32_t level = 0; level < layout->mips.size(); ++level) {
            std::size_t base = 0;
            if (!checked_multiply(layer, layout->layer_stride, base) ||
                !checked_add(base, layout->mips[level].offset, base) || base > texture.size ||
                layout->mips[level].size > texture.size - base) {
                levels.clear();
                return false;
            }
            auto addressing = per_level[level];
            addressing.guest_base = base;
            levels.push_back(addressing);
        }
    }
    return true;
}

void register_texture_entries() {
#define INSTALL(name, function) install<Entry::name>(function)
    INSTALL(nvnTextureBuilderGetDevice, &texture_builder_get_device);
    INSTALL(nvnTextureBuilderSetDevice, &texture_builder_set_device);
    INSTALL(nvnTextureBuilderSetDefaults, &texture_builder_set_defaults);
    INSTALL(nvnTextureBuilderSetFlags, &texture_builder_set_flags);
    INSTALL(nvnTextureBuilderSetTarget, &texture_builder_set_target);
    INSTALL(nvnTextureBuilderSetWidth, &texture_builder_set_width);
    INSTALL(nvnTextureBuilderSetHeight, &texture_builder_set_height);
    INSTALL(nvnTextureBuilderSetDepth, &texture_builder_set_depth);
    INSTALL(nvnTextureBuilderSetSize1D, &texture_builder_set_size_1d);
    INSTALL(nvnTextureBuilderSetSize2D, &texture_builder_set_size_2d);
    INSTALL(nvnTextureBuilderSetSize3D, &texture_builder_set_size_3d);
    INSTALL(nvnTextureBuilderSetLevels, &texture_builder_set_levels);
    INSTALL(nvnTextureBuilderSetFormat, &texture_builder_set_format);
    INSTALL(nvnTextureBuilderSetSamples, &texture_builder_set_samples);
    INSTALL(nvnTextureBuilderSetSwizzle, &texture_builder_set_swizzle);
    INSTALL(nvnTextureBuilderSetDepthStencilMode, &texture_builder_set_depth_stencil_mode);
    INSTALL(nvnTextureBuilderGetStorageSize, &texture_builder_get_storage_size);
    INSTALL(nvnTextureBuilderGetStorageAlignment, &texture_builder_get_storage_alignment);
    INSTALL(nvnTextureBuilderSetStorage, &texture_builder_set_storage);
    INSTALL(nvnTextureBuilderSetPackagedTextureData, &texture_builder_set_packaged_texture_data);
    INSTALL(nvnTextureBuilderSetPackagedTextureLayout, &texture_builder_set_packaged_texture_layout);
    INSTALL(nvnTextureBuilderSetStride, &texture_builder_set_stride);
    INSTALL(nvnTextureBuilderSetGLTextureName, &texture_builder_set_gl_texture_name);
    INSTALL(nvnTextureBuilderGetStorageClass, &texture_builder_get_storage_class);
    INSTALL(nvnTextureBuilderGetFlags, &texture_builder_get_flags);
    INSTALL(nvnTextureBuilderGetTarget, &texture_builder_get_target);
    INSTALL(nvnTextureBuilderGetWidth, &texture_builder_get_width);
    INSTALL(nvnTextureBuilderGetHeight, &texture_builder_get_height);
    INSTALL(nvnTextureBuilderGetDepth, &texture_builder_get_depth);
    INSTALL(nvnTextureBuilderGetLevels, &texture_builder_get_levels);
    INSTALL(nvnTextureBuilderGetFormat, &texture_builder_get_format);
    INSTALL(nvnTextureBuilderGetSamples, &texture_builder_get_samples);
    INSTALL(nvnTextureBuilderGetSwizzle, &texture_builder_get_swizzle);
    INSTALL(nvnTextureBuilderGetDepthStencilMode, &texture_builder_get_depth_stencil_mode);
    INSTALL(nvnTextureBuilderGetPackagedTextureData, &texture_builder_get_packaged_texture_data);
    INSTALL(nvnTextureBuilderGetStride, &texture_builder_get_stride);
    INSTALL(nvnTextureBuilderGetSparseTileLayout, &texture_builder_get_sparse_tile_layout);
    INSTALL(nvnTextureBuilderGetGLTextureName, &texture_builder_get_gl_texture_name);
    INSTALL(nvnTextureBuilderGetZCullStorageSize, &texture_builder_get_zcull_storage_size);
    INSTALL(nvnTextureBuilderGetMemoryPool, &texture_builder_get_memory_pool);
    INSTALL(nvnTextureBuilderGetMemoryOffset, &texture_builder_get_memory_offset);
    install<Entry::nvnTextureBuilderGetPackagedTextureLayout>(
        generic(&texture_builder_get_packaged_texture_layout));
    install<Entry::nvnTextureBuilderGetRawStorageClass>(generic(&texture_builder_get_raw_storage_class));
    install<Entry::nvnTextureBuilderGetViewOffset>(generic(&texture_builder_get_view_offset));
    INSTALL(nvnTextureViewSetDefaults, &texture_view_set_defaults);
    INSTALL(nvnTextureViewSetLevels, &texture_view_set_levels);
    INSTALL(nvnTextureViewSetLayers, &texture_view_set_layers);
    INSTALL(nvnTextureViewSetFormat, &texture_view_set_format);
    INSTALL(nvnTextureViewSetSwizzle, &texture_view_set_swizzle);
    INSTALL(nvnTextureViewSetDepthStencilMode, &texture_view_set_depth_stencil_mode);
    INSTALL(nvnTextureViewSetTarget, &texture_view_set_target);
    INSTALL(nvnTextureViewGetLevels, &texture_view_get_levels);
    INSTALL(nvnTextureViewGetLayers, &texture_view_get_layers);
    INSTALL(nvnTextureViewGetFormat, &texture_view_get_format);
    INSTALL(nvnTextureViewGetSwizzle, &texture_view_get_swizzle);
    INSTALL(nvnTextureViewGetDepthStencilMode, &texture_view_get_depth_stencil_mode);
    INSTALL(nvnTextureViewGetTarget, &texture_view_get_target);
    INSTALL(nvnTextureViewCompare, &texture_view_compare);
    INSTALL(nvnTextureInitialize, &texture_initialize);
    INSTALL(nvnTextureGetZCullStorageSize, &texture_get_zcull_storage_size);
    INSTALL(nvnTextureFinalize, &texture_finalize);
    INSTALL(nvnTextureSetDebugLabel, &texture_set_debug_label);
    INSTALL(nvnTextureGetStorageClass, &texture_get_storage_class);
    INSTALL(nvnTextureGetViewOffset, &texture_get_view_offset);
    INSTALL(nvnTextureGetFlags, &texture_get_flags);
    INSTALL(nvnTextureGetTarget, &texture_get_target);
    INSTALL(nvnTextureGetWidth, &texture_get_width);
    INSTALL(nvnTextureGetHeight, &texture_get_height);
    INSTALL(nvnTextureGetDepth, &texture_get_depth);
    INSTALL(nvnTextureGetLevels, &texture_get_levels);
    INSTALL(nvnTextureGetFormat, &texture_get_format);
    INSTALL(nvnTextureGetSamples, &texture_get_samples);
    INSTALL(nvnTextureGetSwizzle, &texture_get_swizzle);
    INSTALL(nvnTextureGetDepthStencilMode, &texture_get_depth_stencil_mode);
    INSTALL(nvnTextureGetStride, &texture_get_stride);
    INSTALL(nvnTextureGetTextureAddress, &texture_get_texture_address);
    INSTALL(nvnTextureGetSparseTileLayout, &texture_get_sparse_tile_layout);
    INSTALL(nvnTextureWriteTexels, &texture_write_texels);
    INSTALL(nvnTextureWriteTexelsStrided, &texture_write_texels_strided);
    INSTALL(nvnTextureReadTexels, &texture_read_texels);
    INSTALL(nvnTextureReadTexelsStrided, &texture_read_texels_strided);
    INSTALL(nvnTextureFlushTexels, &texture_flush_texels);
    INSTALL(nvnTextureInvalidateTexels, &texture_invalidate_texels);
    INSTALL(nvnTextureGetMemoryPool, &texture_get_memory_pool);
    INSTALL(nvnTextureGetMemoryOffset, &texture_get_memory_offset);
    INSTALL(nvnTextureGetStorageSize, &texture_get_storage_size);
    INSTALL(nvnTextureCompare, &texture_compare);
    INSTALL(nvnTextureGetDebugID, &texture_get_debug_id);
    install<Entry::nvnTextureGetRawStorageClass>(generic(&texture_get_raw_storage_class));
#undef INSTALL
}

} // namespace armigo::nvn
