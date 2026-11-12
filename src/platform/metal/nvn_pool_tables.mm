// ============================================================================
// Armigo Platform - Apple Platform Integration
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include "nvn_pool_tables.h"

#include "nvn_replay.h"

#include <armigo/common/diagnostics.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstring>
#include <mutex>
#include <unordered_map>
#include <utility>

namespace armigo::platform::metal::nvn_driver {
namespace {

constexpr unsigned pool_ids_max = 1024;
constexpr unsigned pool_spares = 16;
constexpr std::uint64_t pool_frame_bytes = 32ull << 20;
constexpr std::uint64_t pool_req_bytes = 128ull << 20;
constexpr double pool_sub_ms = 2.0;
constexpr std::uint32_t pool_reg = 2;
constexpr std::size_t pool_spare_quantum = 64u << 10;

struct IdList {
    std::vector<int> values;
    int count = 0; // -1: the list overflowed and was dropped

    void push(int id) {
        if (count < 0) return;
        if (count == static_cast<int>(pool_ids_max)) {
            values.clear();
            count = -1;
            return;
        }
        values.push_back(id);
        ++count;
    }
};

struct Snap {
    std::shared_ptr<PoolTableVersion> version;
    PoolTableKey key{};
    std::uint64_t reg = 0;
    std::uint64_t ser = 0;
    std::uint64_t frame = 0;
    std::uint64_t sub = 0;
    int left = 0;
    int left_req = 0;
    IdList dead;
    IdList trans;
    // How much of the scan's affected-ID list this table has copied forward.
    // The scan cannot write a published table, so those IDs patch a new version.
    std::uint64_t affected_epoch = 0;
    std::size_t affected_incorporated = 0;
};

struct StampTable {
    std::vector<std::uint32_t> stamp;
    std::uint32_t current = 1;

    void begin() {
        if (!++current) {
            std::fill(stamp.begin(), stamp.end(), 0);
            current = 1;
        }
    }

    bool add(IdList& list, int id) {
        if (list.count < 0 || id < 0) return false;
        if (static_cast<std::size_t>(id) >= stamp.size())
            stamp.resize(static_cast<std::size_t>(id) + 4096, 0);
        if (stamp[static_cast<std::size_t>(id)] == current) return false;
        stamp[static_cast<std::size_t>(id)] = current;
        list.push(id);
        return true;
    }

    void keep(IdList& list, const IdList& old) {
        if (old.count < 0) {
            list.values.clear();
            list.count = -1;
            return;
        }
        for (int index = 0; index < old.count; ++index) add(list, old.values[static_cast<std::size_t>(index)]);
    }
};

MTLPixelFormat component_format(PoolComponent component) {
    switch (component) {
    case PoolComponent::Sint: return MTLPixelFormatR32Sint;
    case PoolComponent::Uint: return MTLPixelFormatR32Uint;
    case PoolComponent::Depth: return MTLPixelFormatDepth32Float;
    case PoolComponent::Float: return MTLPixelFormatR32Float;
    }
    return MTLPixelFormatR32Float;
}

bool component_is_depth(PoolComponent component) { return component == PoolComponent::Depth; }

} // namespace

struct PoolSpareStore {
    std::mutex mutex;
    struct Spare {
        id<MTLBuffer> buffer = nil;
        std::uint64_t fence = 0;
    };
    Spare entries[pool_spares]{};
    int count = 0;
    std::uint64_t made = 0;
    std::uint64_t reused = 0;
};

PoolTableVersion::~PoolTableVersion() {
    if (!buffer || !spares) return;
    // Copy-on-write: a version an encoded submission holds is never written.
    // Its buffer returns here, when the last shared pointer drops after that
    // submission has retired, not when a newer version replaces it.
    std::lock_guard lock(spares->mutex);
    if (spares->count == static_cast<int>(pool_spares)) {
        int oldest = 0;
        for (int index = 1; index < spares->count; ++index)
            if (spares->entries[index].fence < spares->entries[oldest].fence) oldest = index;
        spares->entries[oldest] = spares->entries[--spares->count];
        spares->entries[spares->count].buffer = nil;
    }
    spares->entries[spares->count].buffer = buffer;
    spares->entries[spares->count].fence = retire_fence.load(std::memory_order_acquire);
    ++spares->count;
    buffer = nil;
}

struct PoolTableCache::State {
    id<MTLDevice> device = nil;
    std::shared_ptr<PoolSpareStore> spares = std::make_shared<PoolSpareStore>();
    std::vector<Snap> snaps;
    std::uint64_t generation = 0;
    PoolTableStatistics stats{};
    std::string lookup_failure;
    bool fail_allocation = false;
    std::uint64_t present_count = 0;
    id<MTLBuffer> last_texture_buffer = nil;

    // Versions share a bitmap within an empty epoch. An encoded old version
    // keeps its own bitmap; late reads carry only across unchanged ID epochs.
    std::array<std::uint32_t, pool_seen_words> seen_done{};
    std::array<std::uint32_t, pool_seen_words> seen_todo{};
    id<MTLBuffer> bitmap = nil;
    std::shared_ptr<PoolReadBitmap> current_read_bitmap;
    std::vector<std::shared_ptr<PoolReadBitmap>> retired_read_bitmaps;
    std::uint64_t seen_pool_generation = 0;
    std::uint64_t seen_empty_generation = 0;
    std::uint64_t scanned_submission = ~0ull;
    std::uint64_t cleared_through = 1;
    std::unordered_map<int, std::uint64_t> descriptor_keys;
    std::unordered_map<int, std::uint8_t> element_types;

    std::vector<int> affected;
    std::uint64_t affected_epoch = 0;

    std::uint64_t walk_frame = ~0ull;
    std::uint64_t walk_spent = 0;
    std::uint64_t walk_req = 0;
    std::uint64_t walk_sub = ~0ull;
    double walk_ms = 0;
    bool budget_over = false;

    StampTable dead_stamps;
    StampTable trans_stamps;

    id<MTLTexture> dummies[10][4]{};
    bool dummy_failed[10][4]{};
    id<MTLBuffer> dummy_buffers[10][4]{};
    id<MTLSamplerState> default_samplers[2]{};

    explicit State(id<MTLDevice> metal) : device(metal) {}

    id<MTLBuffer> take(std::size_t need, [[maybe_unused]] std::uint64_t completed) {
        if (fail_allocation || !need) return nil;
        const std::size_t length = (need + pool_spare_quantum - 1) & ~(pool_spare_quantum - 1);
        std::lock_guard lock(spares->mutex);
        for (int index = 0; index < spares->count; ++index) {
            auto& spare = spares->entries[index];
            // A buffer reaches the spares only when its version's last holder
            // let it go, after every submission that bound it retired.
            if (spare.buffer && spare.buffer.length == length) {
                id<MTLBuffer> reused = spare.buffer;
                spare = spares->entries[--spares->count];
                spares->entries[spares->count].buffer = nil;
                ++spares->reused;
                return reused;
            }
        }
        id<MTLBuffer> created = [device newBufferWithLength:length options:MTLResourceStorageModeShared];
        if (created) ++spares->made;
        return created;
    }

    bool commit_residency(PoolTableVersion& version) const {
        auto* descriptor = [MTLResidencySetDescriptor new];
        descriptor.initialCapacity = std::max<NSUInteger>(version.allocations.size(), 1);
        NSError* error = nil;
        version.residency = [device newResidencySetWithDescriptor:descriptor error:&error];
        if (!version.residency) return false;
        for (id<MTLAllocation> allocation : version.allocations)
            if (allocation) [version.residency addAllocation:allocation];
        [version.residency commit];
        return true;
    }

    void remember_texture(PoolTableVersion& version, id<MTLTexture> texture) {
        if (!texture) return;
        const auto add = [&](id object) {
            if (version.allocation_keys.insert((__bridge const void*)object).second)
                version.allocations.push_back((id<MTLAllocation>)object);
        };
        version.textures.push_back(texture);
        add(texture);
        for (id<MTLTexture> parent = texture.parentTexture; parent; parent = parent.parentTexture) {
            version.textures.push_back(parent);
            add(parent);
        }
        if (id<MTLBuffer> backing = texture.buffer) {
            version.extra_buffers.push_back(backing);
            add(backing);
        }
    }

    id<MTLTexture> dummy(PoolTableKey key) {
        if (key.sampler) return nil;
        const int type = static_cast<int>(key.type);
        const int component = static_cast<int>(key.component);
        if (type < 0 || type >= 10 || component < 0 || component >= 4) return nil;
        if (dummies[type][component] || dummy_failed[type][component]) return dummies[type][component];
        const bool depth = component_is_depth(key.component);
        const bool multisample = key.type == MTLTextureType2DMultisample ||
                                 key.type == MTLTextureType2DMultisampleArray;
        const MTLPixelFormat format = component_format(key.component);
        id<MTLTexture> texture = nil;
        if (key.type == MTLTextureTypeTextureBuffer) {
            dummy_buffers[type][component] =
                [device newBufferWithLength:256 options:MTLResourceStorageModeShared];
            if (dummy_buffers[type][component]) {
                auto* descriptor = [MTLTextureDescriptor textureBufferDescriptorWithPixelFormat:format
                                                                                          width:1
                                                                                resourceOptions:MTLResourceStorageModeShared
                                                                                          usage:MTLTextureUsageShaderRead];
                texture = [dummy_buffers[type][component] newTextureWithDescriptor:descriptor
                                                                            offset:0
                                                                       bytesPerRow:256];
            }
        } else {
            auto* descriptor = [MTLTextureDescriptor new];
            descriptor.textureType = key.type;
            descriptor.pixelFormat = format;
            descriptor.width = 1;
            descriptor.height = key.type == MTLTextureType1D || key.type == MTLTextureType1DArray ? 1 : 1;
            descriptor.depth = key.type == MTLTextureType3D ? 1 : 1;
            descriptor.mipmapLevelCount = 1;
            descriptor.sampleCount = multisample ? 2 : 1;
            descriptor.arrayLength = 1;
            if (key.type == MTLTextureTypeCube || key.type == MTLTextureTypeCubeArray) descriptor.arrayLength = 6;
            descriptor.storageMode = MTLStorageModeShared;
            descriptor.usage = (depth || multisample)
                ? (MTLTextureUsageShaderRead | MTLTextureUsageRenderTarget)
                : MTLTextureUsageShaderRead;
            texture = [device newTextureWithDescriptor:descriptor];
            if (texture && !depth && !multisample) {
                const NSUInteger slices = descriptor.arrayLength;
                std::vector<std::uint8_t> zeros(4096, 0);
                for (NSUInteger slice = 0; slice < slices; ++slice) {
                    [texture replaceRegion:MTLRegionMake2D(0, 0, 1, 1)
                               mipmapLevel:0
                                     slice:slice
                                 withBytes:zeros.data()
                               bytesPerRow:256
                             bytesPerImage:256];
                }
            } else if (texture) {
                id<MTLCommandQueue> queue = [device newCommandQueue];
                if (queue) {
                    id<MTLCommandBuffer> commands = [queue commandBuffer];
                    auto* pass = [MTLRenderPassDescriptor renderPassDescriptor];
                    if (depth) {
                        pass.depthAttachment.texture = texture;
                        pass.depthAttachment.loadAction = MTLLoadActionClear;
                        pass.depthAttachment.storeAction = MTLStoreActionStore;
                        pass.depthAttachment.clearDepth = 0;
                    } else {
                        pass.colorAttachments[0].texture = texture;
                        pass.colorAttachments[0].loadAction = MTLLoadActionClear;
                        pass.colorAttachments[0].storeAction = MTLStoreActionStore;
                        pass.colorAttachments[0].clearColor = MTLClearColorMake(0, 0, 0, 0);
                    }
                    id<MTLRenderCommandEncoder> encoder = [commands renderCommandEncoderWithDescriptor:pass];
                    [encoder endEncoding];
                    [commands commit];
                    [commands waitUntilCompleted];
                }
            }
        }
        if (!texture) dummy_failed[type][component] = true;
        dummies[type][component] = texture;
        return texture;
    }

    // One default for every sampler table: nearest, clamp to edge, normalized
    // coordinates, compare never. A registered sampler keeps its own state.
    id<MTLSamplerState> default_sampler(bool) {
        if (default_samplers[1]) return default_samplers[1];
        auto* descriptor = [MTLSamplerDescriptor new];
        descriptor.minFilter = MTLSamplerMinMagFilterNearest;
        descriptor.magFilter = MTLSamplerMinMagFilterNearest;
        descriptor.mipFilter = MTLSamplerMipFilterNotMipmapped;
        descriptor.sAddressMode = MTLSamplerAddressModeClampToEdge;
        descriptor.tAddressMode = MTLSamplerAddressModeClampToEdge;
        descriptor.rAddressMode = MTLSamplerAddressModeClampToEdge;
        descriptor.compareFunction = MTLCompareFunctionNever;
        descriptor.normalizedCoordinates = YES;
        descriptor.supportArgumentBuffers = YES;
        default_samplers[1] = [device newSamplerStateWithDescriptor:descriptor];
        return default_samplers[1];
    }

    std::uint64_t dummy_rid(const PoolTableKey& key) {
        if (key.sampler) {
            id<MTLSamplerState> sampler = default_sampler(key.normalized_coordinates);
            return sampler ? sampler.gpuResourceID._impl : 0;
        }
        id<MTLTexture> texture = dummy(key);
        return texture ? texture.gpuResourceID._impl : 0;
    }

    std::shared_ptr<PoolReadBitmap> make_read_bitmap(std::uint64_t empty_generation) {
        if (fail_allocation) return {};
        auto result = std::make_shared<PoolReadBitmap>();
        result->buffer = [device newBufferWithLength:pool_seen_words * sizeof(std::uint32_t)
                                            options:MTLResourceStorageModeShared];
        if (!result->buffer) return {};
        std::memset(result->buffer.contents, 0, result->buffer.length);
        result->pool_generation = seen_pool_generation;
        result->empty_generation = empty_generation;
        return result;
    }

    void invalidate_texture_tables() {
        std::erase_if(snaps, [](const auto& snap) { return !snap.key.sampler; });
        last_texture_buffer = nil;
        descriptor_keys.clear();
        element_types.clear();
        affected.clear();
        ++affected_epoch;
        ++generation;
        scanned_submission = ~0ull;
    }

    void sync_bitmap_generation(ReplayResources& resources) {
        const std::uint64_t generation = resources.texture_pool_generation();
        if (generation == seen_pool_generation) return;
        seen_pool_generation = generation;
        seen_empty_generation = resources.texture_empty_generation();
        seen_done.fill(0);
        seen_todo.fill(0);
        cleared_through = 1;
        current_read_bitmap.reset();
        bitmap = nil;
        retired_read_bitmaps.clear();
        invalidate_texture_tables();
    }

    bool sync_empty_epochs(ReplayResources& resources) {
        const auto generation = resources.texture_empty_generation();
        if (generation == seen_empty_generation) return true;
        if (!current_read_bitmap) {
            seen_empty_generation = generation;
            return true;
        }
        std::vector<std::uint32_t> clear(pool_seen_words, 0);
        for (unsigned word = 0; word != pool_seen_words; ++word) {
            for (std::uint32_t bits = seen_done[word] | seen_todo[word]; bits; bits &= bits - 1) {
                const int id = static_cast<int>(word) * 32 + __builtin_ctz(bits);
                TextureReadIdentity identity;
                if (!resources.texture_read_identity(id, identity)) {
                    lookup_failure = "texture empty history is unavailable";
                    return false;
                }
                if (identity.empty_epoch > current_read_bitmap->empty_generation)
                    clear[word] |= 1u << (id & 31);
                else if (identity.depth && (seen_done[word] & (1u << (id & 31))))
                    resources.note_pool_depth_read(identity.serial, identity.storage_key);
            }
        }
        auto replacement = make_read_bitmap(generation);
        if (!replacement) {
            lookup_failure = "texture read bitmap allocation failed";
            return false;
        }
        retired_read_bitmaps.push_back(current_read_bitmap);
        current_read_bitmap = std::move(replacement);
        bitmap = current_read_bitmap->buffer;
        for (unsigned word = 0; word != pool_seen_words; ++word) {
            seen_done[word] &= ~clear[word];
            seen_todo[word] &= ~clear[word];
        }
        seen_empty_generation = generation;
        invalidate_texture_tables();
        return true;
    }

    // Read marks belong to descriptor IDs: replacing an occupied ID keeps its
    // admission under the read budget. Only an emptied ID clears its mark.
    // Empty epochs survive overwritten notes and rotate the bitmap before
    // these current registrations patch the table.
    // A missing note advances the cursor: replaying a gap forever would rebuild
    // every table. The rebuild that the same gap forces refreshes the entries.
    bool apply_seen_clears(ReplayResources& resources) {
        const std::uint64_t current = resources.registration_log_generation();
        for (std::uint64_t generation = cleared_through + 1; generation <= current; ++generation) {
            ReplayResources::RegistrationNote note;
            if (!resources.read_registration_note(generation, note)) {
                descriptor_keys.clear();
                element_types.clear();
                for (unsigned word = 0; word != pool_seen_words; ++word) {
                    for (auto bits = seen_done[word]; bits; bits &= bits - 1) {
                        const int id = static_cast<int>(word) * 32 + __builtin_ctz(bits);
                        TextureReadIdentity identity;
                        if (!resources.texture_read_identity(id, identity)) return false;
                        if (identity.depth) resources.note_pool_depth_read(identity.serial, identity.storage_key);
                    }
                }
                cleared_through = current;
                return true;
            }
            if (note.kind != 0) {
                cleared_through = generation;
                continue;
            }
            TextureReadIdentity identity;
            if (!resources.texture_read_identity(note.id, identity)) return false;
            cleared_through = generation;
            const std::uint64_t key = identity.descriptor_key;
            const auto found = descriptor_keys.find(note.id);
            const std::uint64_t previous = found == descriptor_keys.end() ? 0 : found->second;
            if (key == previous) continue;
            descriptor_keys[note.id] = key;
            element_types.erase(note.id);
            if (!key) clear_seen(note.id);
            else if (seen(note.id) && identity.depth)
                resources.note_pool_depth_read(identity.serial, identity.storage_key);
        }
        return true;
    }

    void clear_seen(int id) {
        if (id < 0 || id >= static_cast<int>(pool_seen_words * 32)) return;
        const std::uint32_t bit = 1u << (id & 31);
        seen_done[static_cast<unsigned>(id) >> 5] &= ~bit;
        seen_todo[static_cast<unsigned>(id) >> 5] &= ~bit;
    }

    bool seen(int id) const {
        if (id < 0 || id >= static_cast<int>(pool_seen_words * 32)) return false;
        return (seen_done[static_cast<unsigned>(id) >> 5] >> (id & 31)) & 1u;
    }

    id<MTLBuffer> seen_buffer() {
        if (fail_allocation) return nil;
        if (!bitmap) {
            current_read_bitmap = make_read_bitmap(seen_empty_generation);
            if (!current_read_bitmap) return nil;
            bitmap = current_read_bitmap->buffer;
        }
        return bitmap;
    }

    void note_affected(int id) {
        affected.push_back(id);
        if (affected.size() <= 16384) return;
        affected.clear();
        ++affected_epoch;
    }

    std::uint64_t room(std::uint64_t frame) {
        if (frame != walk_frame) {
            walk_frame = frame;
            walk_spent = 0;
            walk_req = 0;
        }
        const std::uint64_t want = device.recommendedMaxWorkingSetSize / 20;
        const std::uint64_t have = device.currentAllocatedSize;
        budget_over = have >= want;
        std::uint64_t available = have < want ? want - have : 0;
        const std::uint64_t left = walk_spent < pool_frame_bytes ? pool_frame_bytes - walk_spent : 0;
        if (available > left) available = left;
        return available;
    }

    bool unwritten(ReplayResources& resources, int id) const {
        return resources.texture_compressible(id) && !resources.texture_storage_made(id) &&
               !resources.texture_storage_shared(id);
    }

    std::uint64_t walk_limit(ReplayResources& resources, int count, std::uint64_t frame) {
        const std::uint64_t available = room(frame);
        if (!available) return 0;
        std::vector<std::uint64_t> sizes;
        sizes.reserve(static_cast<std::size_t>(std::max(count, 0)));
        std::uint64_t sum = 0;
        for (int id = 1; id < count; ++id) {
            if (!resources.texture_registered(id) || !resources.texture_storage_live(id)) continue;
            if (resources.texture_storage_shared(id)) continue;
            const std::uint64_t size = resources.texture_storage_size(id);
            sizes.push_back(size);
            sum += size;
        }
        if (sum <= available) return UINT64_MAX;
        std::sort(sizes.begin(), sizes.end());
        std::uint64_t used = 0;
        std::uint64_t limit = 0;
        for (const std::uint64_t size : sizes) {
            if (used + size > available) break;
            used += size;
            limit = size;
        }
        return limit;
    }

    struct MadeTexture {
        id<MTLTexture> texture = nil;
        bool created = false;
    };

    MadeTexture make_texture(ReplayResources& resources, int descriptor) {
        const bool before = resources.texture_storage_made(descriptor);
        id<MTLTexture> made = resources.texture_for_id(descriptor);
        const bool after = resources.texture_storage_made(descriptor);
        if (!before && after) ++stats.textures_made;
        return {made, !before && after};
    }

    // One table entry. 0 means the dummy (the caller substitutes it). dead is
    // an ID whose storage does not exist yet. The resources lock is not held
    // across texture_for_id: each query takes it and releases it.
    std::uint64_t entry(ReplayResources& resources, const PoolTableKey& key, int descriptor,
                        std::uint64_t limit, int& left, int& left_req, bool& dead,
                        __strong id<MTLTexture>& texture_out, __strong id<MTLSamplerState>& sampler_out) {
        texture_out = nil;
        sampler_out = nil;
        dead = false;
        if (descriptor <= 0) return 0;
        if (key.sampler) {
            sampler_out = resources.sampler_for_id(descriptor);
            return sampler_out ? sampler_out.gpuResourceID._impl : 0;
        }
        if (!resources.texture_registered(descriptor)) return 0;
        if (!resources.texture_storage_live(descriptor)) {
            dead = true;
            return 0;
        }
        if (unwritten(resources, descriptor)) return 0;
        const bool shared = resources.texture_storage_shared(descriptor);
        const bool made = resources.texture_storage_made(descriptor);
        const std::uint64_t size = resources.texture_storage_size(descriptor);
        if (!shared && size > limit && !made && !(seen(descriptor) && walk_req < pool_req_bytes)) {
            ++left;
            if (seen(descriptor)) ++left_req;
            return 0;
        }
        const auto cached = element_types.find(descriptor);
        if (cached != element_types.end() && cached->second != static_cast<std::uint8_t>(key.type) + 1) return 0;
        if (!shared) {
            walk_spent += size;
            if (seen(descriptor)) walk_req += size;
        }
        const MadeTexture made_texture = make_texture(resources, descriptor);
        texture_out = made_texture.texture;
        if (!texture_out) return 0;
        element_types[descriptor] = static_cast<std::uint8_t>(texture_out.textureType) + 1;
        if (texture_out.textureType != key.type) {
            texture_out = nil;
            return 0;
        }
        return texture_out.gpuResourceID._impl;
    }

    bool fold_read_word(ReplayResources& resources, const std::shared_ptr<PoolReadBitmap>& owner,
                        unsigned word) {
        if (!owner || owner->pool_generation != seen_pool_generation) return true;
        auto* gpu = static_cast<const std::atomic<std::uint32_t>*>(owner->buffer.contents);
        const auto fresh = gpu[word].load(std::memory_order_relaxed) & ~seen_done[word];
        std::uint32_t accepted = 0;
        for (auto bits = fresh; bits; bits &= bits - 1) {
            const int id = static_cast<int>(word) * 32 + __builtin_ctz(bits);
            TextureReadIdentity identity;
            if (!resources.texture_read_identity(id, identity)) {
                lookup_failure = "texture read identity is unavailable";
                return false;
            }
            if (identity.empty_epoch > owner->empty_generation) continue;
            accepted |= 1u << (id & 31);
            if (identity.depth) resources.note_pool_depth_read(identity.serial, identity.storage_key);
        }
        seen_done[word] |= accepted;
        seen_todo[word] |= accepted;
        stats.ids_read += static_cast<std::uint64_t>(__builtin_popcount(accepted));
        return true;
    }

    bool scan(ReplayResources& resources, std::uint64_t submission, [[maybe_unused]] std::uint64_t completed,
              std::uint64_t frame) {
        if (!bitmap) return false;
        // The live bitmap is read without waiting for the submissions that may
        // still write it: bits are only ever set, and one set after this scan
        // is folded by a later one. A held queue keeps the completed fence
        // back, so waiting for it would leave read IDs unmade while it waits.
        if (scanned_submission == submission) return false;
        scanned_submission = submission;
        if (frame != walk_frame) {
            walk_frame = frame;
            walk_spent = 0;
            walk_req = 0;
        }
        const int count = std::max(resources.registered_texture_count(), 0);
        const unsigned words = std::min(pool_seen_words, static_cast<unsigned>(count / 32 + 1));
        // Once no table version owns a retired bitmap, its GPU writers have
        // finished. Fold it one last time before dropping the cache's owner.
        for (std::size_t index = 0; index < retired_read_bitmaps.size();) {
            const auto& owner = retired_read_bitmaps[index];
            if (owner.use_count() != 1) { ++index; continue; }
            for (unsigned word = 0; word != words; ++word)
                if (!fold_read_word(resources, owner, word)) { scanned_submission = ~0ull; return false; }
            retired_read_bitmaps.erase(retired_read_bitmaps.begin() + static_cast<std::ptrdiff_t>(index));
        }
        for (unsigned word = 0; word < words; ++word) {
            if (!fold_read_word(resources, current_read_bitmap, word)) { scanned_submission = ~0ull; return false; }
            for (const auto& owner : retired_read_bitmaps)
                if (!fold_read_word(resources, owner, word)) { scanned_submission = ~0ull; return false; }
            for (std::uint32_t bits = seen_todo[word]; bits; bits &= bits - 1) {
                const int bit = __builtin_ctz(bits);
                const int id = static_cast<int>(word) * 32 + bit;
                if (id <= 0 || !resources.texture_registered(id) || !resources.texture_storage_live(id) ||
                    unwritten(resources, id)) {
                    seen_todo[word] &= ~(1u << bit);
                    continue;
                }
                if (!resources.texture_storage_shared(id) && !resources.texture_storage_made(id)) {
                    if (walk_req >= pool_req_bytes) return true;
                    walk_req += resources.texture_storage_size(id);
                }
                seen_todo[word] &= ~(1u << bit);
                const MadeTexture made_texture = make_texture(resources, id);
                if (!made_texture.texture) continue;
                note_affected(id);
            }
        }
        return true;
    }

    Snap* find(const PoolTableKey& key) {
        for (auto& snap : snaps)
            if (snap.version && snap.key == key) return &snap;
        return nullptr;
    }

    bool table_size(std::uint32_t count, bool sampler, std::size_t& bytes) const {
        const std::uint32_t cap = sampler ? 4096u : (1u << 20);
        if (count < 1 || count > cap) return false;
        const std::uint64_t entries = static_cast<std::uint64_t>(count) + 1;
        if (entries > (UINT64_MAX - 16) / 8) return false;
        bytes = 16 + 8 * static_cast<std::size_t>(entries);
        return true;
    }

    void publish(Snap& snap, std::shared_ptr<PoolTableVersion> version, std::uint64_t reg, std::uint64_t ser,
                 std::uint64_t frame, std::uint64_t submission, int left, int left_req, IdList dead, IdList trans) {
        snap.version = std::move(version);
        snap.reg = reg;
        snap.ser = ser;
        snap.frame = frame;
        snap.sub = trans.count > 0 ? submission : 0;
        snap.left = left;
        snap.left_req = left_req;
        snap.dead = std::move(dead);
        snap.trans = std::move(trans);
        snap.affected_epoch = affected_epoch;
        snap.affected_incorporated = affected.size();
        ++generation;
    }

    void log_walk() const {
        std::uint64_t made = 0, reused = 0;
        {
            std::lock_guard lock(spares->mutex);
            made = spares->made;
            reused = spares->reused;
        }
        armigo::diag::write(armigo::diag::Level::info,
                            "pool walk: tables made %llu, reused %llu; ids read %llu; textures made %llu, left %llu\n",
                            static_cast<unsigned long long>(made), static_cast<unsigned long long>(reused),
                            static_cast<unsigned long long>(stats.ids_read),
                            static_cast<unsigned long long>(stats.textures_made),
                            static_cast<unsigned long long>(stats.textures_left));
    }

    PoolLookup rebuild(ReplayResources& resources, Snap* existing, const PoolTableKey& key, int count,
                       std::uint64_t reg, std::uint64_t ser, std::uint64_t frame, std::uint64_t submission,
                       std::uint64_t completed) {
        PoolLookup failed;
        const auto started = std::chrono::steady_clock::now();
        std::size_t bytes = 0;
        if (!table_size(static_cast<std::uint32_t>(count), key.sampler, bytes)) return failed;
        const std::uint64_t placeholder = dummy_rid(key);
        if (!placeholder && !key.sampler) return failed;
        if (key.sampler && !default_sampler(key.normalized_coordinates)) return failed;
        id<MTLBuffer> buffer = take(bytes, completed);
        if (!buffer) return failed;
        auto version = std::make_shared<PoolTableVersion>();
        version->spares = spares;
        version->buffer = buffer;
        version->capacity = buffer.length;
        version->gpu = buffer.gpuAddress;
        version->n = static_cast<std::uint32_t>(count);
        auto* header = static_cast<std::uint8_t*>(buffer.contents);
        std::memset(header, 0, buffer.length);
        std::memcpy(header, &version->n, sizeof version->n);
        if (!key.sampler) {
            id<MTLBuffer> seen_gpu = seen_buffer();
            if (!seen_gpu) return failed;
            const std::uint64_t address = seen_gpu.gpuAddress;
            version->read_bitmap = current_read_bitmap;
            std::memcpy(header + 8, &address, sizeof address);
        }
        auto* entries = reinterpret_cast<std::uint64_t*>(header + 16);
        const std::uint64_t limit = key.sampler ? UINT64_MAX : walk_limit(resources, count, frame);
        int left = 0, left_req = 0;
        IdList dead, trans;
        dead_stamps.begin();
        trans_stamps.begin();
        if (!key.sampler) {
            id<MTLTexture> dummy_texture = dummy(key);
            if (!dummy_texture) return failed;
            remember_texture(*version, dummy_texture);
        }
        for (int index = 0; index < count; ++index) {
            bool marked_dead = false;
            id<MTLTexture> texture = nil;
            id<MTLSamplerState> sampler = nil;
            const std::uint64_t rid = entry(resources, key, index, limit, left, left_req, marked_dead, texture, sampler);
            if (key.sampler && index > 0) {
                failed.diagnostic = resources.take_sampler_failure();
                if (!failed.diagnostic.empty()) return failed;
            }
            if (marked_dead) dead_stamps.add(dead, index);
            entries[index] = rid ? rid : placeholder;
            if (texture) remember_texture(*version, texture);
            if (sampler) version->samplers.push_back(sampler);
        }
        entries[count] = placeholder;
        if (!commit_residency(*version)) return failed;
        if (!existing) {
            snaps.push_back({});
            existing = &snaps.back();
            existing->key = key;
        }
        if (!key.sampler) stats.textures_left = static_cast<std::uint64_t>(std::max(left, 0));
        publish(*existing, version, reg, ser, frame, submission, left, left_req, std::move(dead), std::move(trans));
        const auto elapsed = std::chrono::steady_clock::now() - started;
        const double milliseconds = std::chrono::duration<double, std::milli>(elapsed).count();
        stats.walk_us += static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count());
        if (!key.sampler) walk_ms += milliseconds;
        if (!key.sampler && stats.builds % 500 == 0) log_walk();
        ++stats.builds;
        if (!key.sampler) last_texture_buffer = version->buffer;
        return {true, version, key.sampler ? nil : bitmap};
    }
};

PoolTableCache::PoolTableCache(id<MTLDevice> device) : state_(std::make_unique<State>(device)) {}

PoolTableCache::~PoolTableCache() = default;

void PoolTableCache::note_present() { ++state_->present_count; }

std::uint64_t PoolTableCache::present_count() const { return state_->present_count; }

std::uint64_t PoolTableCache::generation() const { return state_->generation; }

PoolTableStatistics PoolTableCache::statistics() const { return state_->stats; }

id<MTLBuffer> PoolTableCache::current_texture_buffer() const { return state_->last_texture_buffer; }

void PoolTableCache::set_allocation_failure(bool fail) { state_->fail_allocation = fail; }

PoolLookup PoolTableCache::lookup(const PoolTableKey& requested, ReplayResources& resources,
                                  std::uint64_t submission, std::uint64_t completed, std::uint64_t frame) {
    // One sampler table for every pooled sampling kind. Texture tables do not
    // split on sampler flags either: the entry check is the Metal texture type.
    PoolTableKey key = requested;
    if (key.sampler) {
        key.type = MTLTextureType2D;
        key.component = PoolComponent::Float;
        key.normalized_coordinates = true;
        key.comparison = false;
    } else {
        key.normalized_coordinates = false;
        key.comparison = false;
    }
    PoolLookup failed;
    if (state_->fail_allocation || !state_->device) return failed;
    state_->lookup_failure.clear();
    if (!key.sampler) {
        state_->sync_bitmap_generation(resources);
        if (!state_->sync_empty_epochs(resources)) return {false, {}, nil, state_->lookup_failure};
        if (!state_->apply_seen_clears(resources))
            return {false, {}, nil, "texture read identity is unavailable"};
        // Fold the live OR bits. A GPU write after this scan remains visible
        // to the next lookup without waiting for an unrelated held queue.
        state_->scan(resources, submission, completed, frame);
        if (!state_->lookup_failure.empty()) return {false, {}, nil, state_->lookup_failure};
    }
    const std::uint64_t reg = resources.registration_log_generation();
    const std::uint64_t ser = key.sampler ? 0 : resources.storage_log_generation();
    const int count = std::max(key.sampler ? resources.sampler_count() : resources.registered_texture_count(), 1);
    Snap* snap = state_->find(key);
    const bool stale = snap && snap->sub && snap->sub != submission;
    // Nothing records a transient view, so this list stays empty and stale
    // stays false. The lists are kept so a pool change still replays the IDs
    // whose storage was missing.
    const bool walk_on = snap && snap->left && snap->frame != frame && (snap->left_req || !state_->budget_over);
    const bool affected_pending = snap && !key.sampler &&
        (snap->affected_epoch != state_->affected_epoch || snap->affected_incorporated < state_->affected.size());
    if (snap && snap->version && snap->reg == reg && snap->ser == ser && !stale && !walk_on && !affected_pending) {
        return {true, snap->version, key.sampler ? nil : state_->bitmap};
    }
    if (snap && snap->version && !walk_on) {
        bool rebuild = false;
        if (reg - snap->reg >= 4096 || ser - snap->ser >= 4096) rebuild = true;
        std::uint32_t published = 0;
        if (!rebuild && snap->version->buffer.contents)
            std::memcpy(&published, snap->version->buffer.contents, sizeof published);
        if (!rebuild && published != static_cast<std::uint32_t>(count)) rebuild = true;
        if (!rebuild && !key.sampler && snap->affected_epoch != state_->affected_epoch) rebuild = true;
        std::vector<int> ids;
        bool known = !rebuild;
        bool pools = false;
        if (known) {
            for (std::uint64_t generation = snap->reg + 1; generation <= reg && known; ++generation) {
                ReplayResources::RegistrationNote note;
                known = resources.read_registration_note(generation, note);
                if (!known) break;
                if (note.kind == pool_reg) pools = !key.sampler;
                else if (note.kind == (key.sampler ? 1u : 0u)) ids.push_back(note.id);
            }
        }
        if (known && snap->ser != ser) {
            for (std::uint64_t generation = snap->ser + 1; generation <= ser && known && ids.size() < 8192;
                 ++generation) {
                std::uint64_t serial = 0;
                known = resources.read_storage_note(generation, serial);
                if (!known) break;
                for (const std::int32_t id : resources.descriptor_ids_for_serial(serial)) {
                    if (ids.size() >= 8192) break;
                    ids.push_back(id);
                }
            }
        }
        if (known && pools) {
            if (snap->dead.count < 0) known = false;
            else for (int index = 0; index < snap->dead.count; ++index)
                ids.push_back(snap->dead.values[static_cast<std::size_t>(index)]);
        }
        if (known && stale) {
            if (snap->trans.count < 0) known = false;
            else for (int index = 0; index < snap->trans.count; ++index)
                ids.push_back(snap->trans.values[static_cast<std::size_t>(index)]);
        }
        if (known && !key.sampler && snap->affected_epoch == state_->affected_epoch) {
            for (std::size_t index = snap->affected_incorporated; index < state_->affected.size(); ++index)
                ids.push_back(state_->affected[index]);
        }
        if (!known || ids.size() >= 8192) rebuild = true;
        if (!rebuild) {
            const std::uint64_t available = key.sampler ? UINT64_MAX : state_->room(frame);
            const std::uint64_t spent = state_->walk_spent;
            const std::uint64_t dummy = *reinterpret_cast<std::uint64_t*>(
                static_cast<std::uint8_t*>(snap->version->buffer.contents) + 16 + 8 * static_cast<std::size_t>(count));
            int left = 0, left_req = 0;
            IdList dead, trans;
            state_->dead_stamps.begin();
            state_->trans_stamps.begin();
            if (!pools) state_->dead_stamps.keep(dead, snap->dead);
            if (!stale) state_->trans_stamps.keep(trans, snap->trans);
            std::shared_ptr<PoolTableVersion> patched;
            auto* entries = reinterpret_cast<std::uint64_t*>(
                static_cast<std::uint8_t*>(snap->version->buffer.contents) + 16);
            for (const int descriptor : ids) {
                if (descriptor < 0 || descriptor >= count) continue;
                const std::uint64_t used = state_->walk_spent - spent;
                const std::uint64_t limit = key.sampler ? UINT64_MAX : (used < available ? available - used : 0);
                bool marked_dead = false;
                id<MTLTexture> made_texture = nil;
                id<MTLSamplerState> made_sampler = nil;
                std::uint64_t rid = state_->entry(resources, key, descriptor, limit, left, left_req, marked_dead,
                                                 made_texture, made_sampler);
                if (key.sampler && descriptor > 0) {
                    failed.diagnostic = resources.take_sampler_failure();
                    if (!failed.diagnostic.empty()) return failed;
                }
                if (marked_dead) state_->dead_stamps.add(dead, descriptor);
                if (!rid) rid = dummy;
                std::uint64_t* patched_entries = patched
                    ? reinterpret_cast<std::uint64_t*>(static_cast<std::uint8_t*>(patched->buffer.contents) + 16)
                    : entries;
                if (rid == patched_entries[descriptor]) continue;
                if (!patched) {
                    id<MTLBuffer> buffer = state_->take(snap->version->capacity, completed);
                    if (!buffer) return failed;
                    patched = std::make_shared<PoolTableVersion>();
                    patched->spares = state_->spares;
                    patched->buffer = buffer;
                    patched->capacity = buffer.length;
                    patched->gpu = buffer.gpuAddress;
                    patched->n = snap->version->n;
                    patched->read_bitmap = snap->version->read_bitmap;
                    std::memcpy(buffer.contents, snap->version->buffer.contents, snap->version->capacity);
                    patched->textures = snap->version->textures;
                    patched->samplers = snap->version->samplers;
                    patched->extra_buffers = snap->version->extra_buffers;
                    patched->allocations = snap->version->allocations;
                    patched->allocation_keys = snap->version->allocation_keys;
                    patched_entries = reinterpret_cast<std::uint64_t*>(static_cast<std::uint8_t*>(buffer.contents) + 16);
                }
                patched_entries[descriptor] = rid;
                if (made_texture) state_->remember_texture(*patched, made_texture);
                if (made_sampler) patched->samplers.push_back(made_sampler);
            }
            if (!patched) {
                if (left) {
                    snap->left += left;
                    snap->left_req += left_req;
                    snap->frame = frame;
                }
                snap->reg = reg;
                snap->ser = ser;
                snap->dead = std::move(dead);
                snap->trans = std::move(trans);
                snap->sub = snap->trans.count > 0 ? submission : 0;
                snap->affected_epoch = state_->affected_epoch;
                snap->affected_incorporated = state_->affected.size();
                ++state_->stats.kept;
                if (!key.sampler) {
                    state_->stats.textures_left = static_cast<std::uint64_t>(std::max(snap->left, 0));
                    state_->last_texture_buffer = snap->version->buffer;
                }
                return {true, snap->version, key.sampler ? nil : state_->bitmap};
            }
            if (!state_->commit_residency(*patched)) return failed;
            if (left) {
                snap->left += left;
                snap->left_req += left_req;
                snap->frame = frame;
            }
            snap->dead = std::move(dead);
            snap->trans = std::move(trans);
            state_->publish(*snap, patched, reg, ser, snap->frame, submission, snap->left, snap->left_req,
                            std::move(snap->dead), std::move(snap->trans));
            ++state_->stats.patches;
            if (!key.sampler) {
                state_->last_texture_buffer = patched->buffer;
            }
            return {true, patched, key.sampler ? nil : state_->bitmap};
        }
    }
    if (state_->walk_sub != submission) {
        state_->walk_sub = submission;
        state_->walk_ms = 0;
    }
    if (snap && snap->version && state_->walk_ms >= pool_sub_ms && !stale) {
        return {true, snap->version, key.sampler ? nil : state_->bitmap};
    }
    auto built = state_->rebuild(resources, snap, key, count, reg, ser, frame, submission, completed);
    return built;
}

} // namespace armigo::platform::metal::nvn_driver
