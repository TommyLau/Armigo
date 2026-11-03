// ============================================================================
// Armigo Graphics - GPU Command Processing
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================
// Cached buffers merge overlapping ranges while retaining prior contents and
// provide one coherent view for vertex, index, uniform, and storage uses.
// Guest-to-host synchronization copies only changed intervals, and retained
// slices preserve allocation identity for their consumers. Uniform data is
// snapshotted into immutable upload-ring slices when a draw is recorded.
//
// Fragmented read-only snapshots gather physical components in virtual order.
// Constant-buffer windows may zero-fill missing components. Registry snapshots
// are authoritative; cache metadata records only the versions already copied.
//
// CPU revalidation records a write stamp plus a 64-bit hash per aligned
// 4096-byte page. One snapshot hashes only pages still suspected of change,
// and only pages whose content differs are copied again.

#include "buffer_residency.h"
#include "content_hash.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <iterator>
#include <limits>
#include <mutex>
#include <optional>
#include <string_view>
#include <utility>

namespace armigo::graphics {
namespace {

// Uniform windows of a whole deferred batch share a few arenas.
constexpr std::uint64_t ring_block_bytes = 4ull << 20;
RendererError error(CommandStatus status, std::string_view detail) {
    return RendererError({status, detail});
}
bool ok(const RendererStatus& result) { return std::holds_alternative<std::monostate>(result); }
bool power_of_two(std::uint64_t value) { return value && !(value & (value - 1)); }
bool contains(BackingRange outer, BackingRange inner) {
    return outer.backing == inner.backing && inner.offset >= outer.offset &&
           inner.offset - outer.offset <= outer.size &&
           inner.size <= outer.size - (inner.offset - outer.offset);
}
bool overlaps(BackingRange a, BackingRange b) {
    return a.backing == b.backing && a.offset < b.offset + b.size && b.offset < a.offset + a.size;
}
std::optional<std::uint64_t> align_up(std::uint64_t value, std::uint64_t alignment) {
    if (value > std::numeric_limits<std::uint64_t>::max() - (alignment - 1)) return {};
    return (value + alignment - 1) & ~(alignment - 1);
}
bool pending(const CompletionSnapshot& snapshot) {
    return snapshot.phase == SubmissionPhase::Accepted || snapshot.phase == SubmissionPhase::Submitted;
}
RendererStatus copy_ready(const std::vector<ResourceStorageRange>& state) {
    for (const auto& interval : state) {
        if (!interval.dirty.gpu_dirty) continue;
        const auto snapshot = interval.dirty.last_writer.snapshot();
        if (snapshot.error) return *snapshot.error;
        if (pending(snapshot))
            return error(CommandStatus::Stopped, "buffer copy requires successful GPU feedback; submit and retry");
    }
    return renderer_success();
}

struct Image;
struct ImageDomain {
    std::uint64_t device;
    std::weak_ptr<ResourceMemoryRegistry> memory;
    std::vector<std::weak_ptr<Image>> images;
};

std::shared_ptr<ImageDomain> image_domain(std::uint64_t device,
                                        const std::shared_ptr<ResourceMemoryRegistry>& memory) {
    // Only factory discovery is global. Domain image access follows the
    // registry's serialized coordinator contract, including lease release.
    static std::mutex mutex;
    static std::vector<std::weak_ptr<ImageDomain>> domains;
    const std::lock_guard lock(mutex);
    std::erase_if(domains, [](const auto& domain) { return domain.expired(); });
    for (const auto& weak : domains) {
        auto domain = weak.lock();
        if (domain && domain->device == device && domain->memory.lock() == memory) return domain;
    }
    auto domain = std::make_shared<ImageDomain>();
    domain->device = device;
    domain->memory = memory;
    domains.push_back(domain);
    return domain;
}

struct Image final : BufferStorage {
    std::shared_ptr<BufferAllocation> native;
    std::shared_ptr<BufferAllocator> origin;
    // Retained bindings keep exclusion identity alive across cache recreation.
    std::shared_ptr<ImageDomain> domain;
    BackingRange bytes;
    std::uint64_t base = 0;
    std::shared_ptr<BufferAllocation> allocation() const noexcept override { return native; }
    BackingRange coverage() const noexcept override { return bytes; }
    RendererStatus read_gpu(BackingRange range, std::span<std::byte> destination) const override {
        if (!contains(bytes, range) || destination.size() != range.size)
            return error(CommandStatus::MemoryFault, "storage read leaves the cached image");
        const auto source = native->contents().subspan(base + range.offset - bytes.offset, range.size);
        std::copy(source.begin(), source.end(), destination.begin());
        return renderer_success();
    }
};
struct RingBlock {
    std::shared_ptr<BufferAllocation> native;
    std::uint64_t used = 0;
};
struct Slice final : BufferStorage {
    std::shared_ptr<RingBlock> block;
    std::uint64_t base = 0, size = 0;
    std::shared_ptr<BufferAllocation> allocation() const noexcept override { return block->native; }
    BackingRange coverage() const noexcept override { return {BackingId{}, 0, size}; }
    RendererStatus read_gpu(BackingRange range, std::span<std::byte> destination) const override {
        if (!contains(coverage(), range) || destination.size() != range.size)
            return error(CommandStatus::MemoryFault, "storage read leaves the upload slice");
        const auto source = block->native->contents().subspan(base + range.offset, range.size);
        std::copy(source.begin(), source.end(), destination.begin());
        return renderer_success();
    }
};
struct CopiedRange {
    DirtyRange dirty;
    std::uint64_t epoch = 0;
};
struct Entry {
    std::shared_ptr<Image> image;
    std::uint64_t last_use = 0;
    std::vector<CopiedRange> copied;
    // Certificate established with the last full validation: the registry
    // stamp taken before the guest reads that fed the image, plus one content
    // hash per aligned page of the image bytes.
    std::uint64_t stamp = 0;
    std::vector<std::uint64_t> page_hashes;
};

class Cache final : public BufferCache {
public:
    Cache(std::shared_ptr<BufferAllocator> allocator, std::shared_ptr<ResourceMemoryRegistry> memory,
          std::uint64_t budget)
        : allocator_(std::move(allocator)), memory_(std::move(memory)),
          image_domain_(image_domain(allocator_->device_identity(), memory_)), budget_(budget) {}
    ResidencyDomain domain() const noexcept override { return {allocator_->device_identity(), memory_.get()}; }
    RendererResult<ResidentBuffer> acquire(const BufferRequest& request) override {
        try { return acquire_checked(request); }
        catch (...) { return error(CommandStatus::MemoryFault, "buffer acquire failed"); }
    }
    RendererResult<ResidentBuffer> upload(std::span<const std::byte> bytes, std::uint64_t alignment) override {
        try {
            if (bytes.empty() || !power_of_two(alignment))
                return error(CommandStatus::Unsupported, "upload requires bytes and power-of-two alignment");
            return snapshot(bytes, alignment);
        } catch (...) { return error(CommandStatus::MemoryFault, "buffer upload failed"); }
    }
    RendererResult<ResidentBuffer> upload_filled(std::uint64_t size, std::uint64_t alignment,
                                                UploadFill fill, void* context) override {
        try {
            if (!size || !power_of_two(alignment))
                return error(CommandStatus::Unsupported, "upload requires bytes and power-of-two alignment");
            return snapshot_with(size, alignment, [&](std::span<std::byte> output) { return fill(context, output); });
        } catch (...) { return error(CommandStatus::MemoryFault, "buffer upload failed"); }
    }
    RendererStatus invalidate(BackingRange bytes) override {
        try {
            auto status = memory_->resolve(bytes, ResolveReason::IncompatibleAlias);
            if (!ok(status)) return status;
            std::erase_if(entries_, [&](const auto& entry) { return overlaps(entry->image->bytes, bytes); });
            return renderer_success();
        } catch (...) { return error(CommandStatus::MemoryFault, "buffer invalidation failed"); }
    }
    RendererStatus resolve(BufferKey key, ResolveReason reason) override {
        try {
            // L2 waits before reading current authority and writes only dirty
            // bytes. Never write a cache snapshot over newer guest contents.
            auto status = memory_->resolve(key.bytes, reason);
            if (!ok(status)) return status;
            std::erase_if(entries_, [&](const auto& entry) { return overlaps(entry->image->bytes, key.bytes); });
            return renderer_success();
        } catch (...) { return error(CommandStatus::MemoryFault, "buffer resolve failed"); }
    }
    RendererStatus trim(std::uint64_t budget) override {
        try { trim_to(budget); return renderer_success(); }
        catch (...) { return error(CommandStatus::MemoryFault, "buffer trim failed"); }
    }
    BufferCacheStatistics statistics() const noexcept { return *statistics_; }

private:
    // Weak accounting survives removal from lookup without extending lifetime.
    // Every live allocation, including registry-only or replaced images, counts.
    std::uint64_t resident_bytes() {
        std::erase_if(allocations_, [](const auto& allocation) { return allocation.expired(); });
        std::uint64_t total = 0;
        for (const auto& weak : allocations_) if (auto allocation = weak.lock()) total += allocation->cost();
        return total;
    }
    void trim_to(std::uint64_t budget) {
        while (resident_bytes() > budget) {
            auto victim = entries_.end();
            for (auto it = entries_.begin(); it != entries_.end(); ++it) {
                // Local construction references also pin an entry while trimming.
                if (it->use_count() != 1 || (*it)->image.use_count() != 1) continue;
                if (victim == entries_.end() || (*it)->last_use < (*victim)->last_use) victim = it;
            }
            if (victim == entries_.end()) break;
            entries_.erase(victim);
        }
        for (auto it = ring_.begin(); it != ring_.end() && resident_bytes() > budget;) {
            if (it->use_count() == 1) it = ring_.erase(it);
            else ++it;
        }
    }
    RendererResult<std::shared_ptr<BufferAllocation>> allocate(std::uint64_t size) {
        if (!size || size > budget_)
            return error(CommandStatus::BudgetExceeded, "buffer allocation exceeds residency budget");
        trim_to(budget_ - size);
        if (resident_bytes() > budget_ - size)
            return error(CommandStatus::BudgetExceeded, "retained buffers prevent residency admission");
        auto allocated = allocator_->allocate(size);
        if (auto* failure = std::get_if<RendererError>(&allocated)) return *failure;
        auto native = std::get<std::shared_ptr<BufferAllocation>>(std::move(allocated));
        if (!native || native->contents().size() < size || native->cost() < native->contents().size())
            return error(CommandStatus::MemoryFault, "buffer allocator returned invalid capacity");
        // Native allocation cost may exceed the requested byte length.
        if (native->cost() > budget_)
            return error(CommandStatus::BudgetExceeded, "native buffer allocation exceeds residency budget");
        trim_to(budget_ - native->cost());
        if (resident_bytes() > budget_ - native->cost())
            return error(CommandStatus::BudgetExceeded, "native buffer allocation cannot fit residency budget");
        allocations_.push_back(native);
        return native;
    }

    RendererStatus read_guest(const RetainedRange& retained, BackingRange bytes,
                              std::span<std::byte> output) {
        return memory_->read_retained(retained, bytes, output);
    }

    // Guest bytes of one aligned page of the image, clamped at the image end.
    static std::uint64_t page_bytes(const Image& image, std::uint64_t page) {
        const auto start = page * revalidation_page_bytes;
        return std::min<std::uint64_t>(revalidation_page_bytes, image.bytes.size - start);
    }
    void establish_page_hashes(Entry& entry) const {
        const auto contents = entry.image->native->contents();
        const auto pages = (entry.image->bytes.size + revalidation_page_bytes - 1) /
                           revalidation_page_bytes;
        std::vector<std::uint64_t> hashes(static_cast<std::size_t>(pages));
        for (std::uint64_t page = 0; page < pages; ++page) {
            const auto start = entry.image->base + page * revalidation_page_bytes;
            hashes[static_cast<std::size_t>(page)] =
                content_hash64(contents.subspan(start, page_bytes(*entry.image, page)));
        }
        entry.page_hashes = std::move(hashes);
    }
    // Compare the guest snapshot against the resident image per aligned page.
    // Pages the tracker proves unchanged since the certificate stamp are
    // skipped; whole certified pages compare against the stored hash and edge
    // slices hash both sides exactly, so alignment never hides a difference.
    // Returns the changed page ranges, merged where pages are adjacent.
    RendererResult<std::vector<BackingRange>> suspect_page_deltas(
        const Entry& entry, const std::vector<ResourceStorageRange>& state,
        std::span<const std::byte> current, const BackingRange& request_bytes) const {
        std::vector<BackingRange> changed;
        const auto& image = *entry.image;
        const auto contents = image.native->contents();
        const auto local_end = image.bytes.size;
        for (const auto& interval : state) {
            if (interval.dirty.gpu_dirty) continue;
            const auto bytes = interval.dirty.bytes;
            if (!contains(image.bytes, bytes))
                return error(CommandStatus::MemoryFault, "revalidation interval leaves the cached image");
            const auto local_start = bytes.offset - image.bytes.offset;
            for (auto page = local_start / revalidation_page_bytes;
                 page * revalidation_page_bytes < std::min(local_start + bytes.size, local_end); ++page) {
                const auto page_start = page * revalidation_page_bytes;
                const auto piece_start = std::max(page_start, local_start);
                const auto piece_end = std::min(page_start + revalidation_page_bytes,
                                                local_start + bytes.size);
                const auto size = piece_end - piece_start;
                const auto offset = image.bytes.offset + piece_start;
                if (!memory_->cpu_written_since({bytes.backing, offset, size}, entry.stamp)) continue;
                const bool certified = piece_start == page_start &&
                    size == page_bytes(image, page) && page < entry.page_hashes.size();
                const auto guest = current.subspan(offset - request_bytes.offset, size);
                const auto resident = contents.subspan(image.base + piece_start, size);
                if (content_hash64(guest) ==
                    (certified ? entry.page_hashes[static_cast<std::size_t>(page)]
                               : content_hash64(resident)))
                    continue;
                if (!changed.empty() && changed.back().offset + changed.back().size == offset)
                    changed.back().size += size;
                else
                    changed.push_back({bytes.backing, offset, size});
            }
        }
        return changed;
    }

    static bool copied_covers(const Entry& entry, const DirtyRange& current,
                              std::optional<std::uint64_t> epoch = {}) {
        std::vector<std::uint64_t> points{current.bytes.offset,
                                          current.bytes.offset + current.bytes.size};
        for (const auto& old : entry.copied) {
            if (!overlaps(old.dirty.bytes, current.bytes)) continue;
            points.push_back(std::max(old.dirty.bytes.offset, current.bytes.offset));
            points.push_back(std::min(old.dirty.bytes.offset + old.dirty.bytes.size,
                                      current.bytes.offset + current.bytes.size));
        }
        std::sort(points.begin(), points.end());
        points.erase(std::unique(points.begin(), points.end()), points.end());
        for (std::size_t i = 1; i < points.size(); ++i) {
            const BackingRange piece{current.bytes.backing, points[i - 1], points[i] - points[i - 1]};
            const bool matches = std::any_of(entry.copied.begin(), entry.copied.end(), [&](const auto& old) {
                return contains(old.dirty.bytes, piece) && old.dirty.version == current.version &&
                       (!epoch || old.epoch == *epoch);
            });
            if (!matches) return false;
        }
        return true;
    }

    RendererStatus refresh(Entry& entry, const RetainedRange& retained,
                           const std::vector<ResourceStorageRange>& state,
                           std::span<const std::byte> snapshot = {}, bool revalidated = false,
                           std::span<const BackingRange> changed = {}) {
        const auto bytes = retained.bytes;
        const auto epoch = memory_->cpu_cache_epoch();
        std::vector<CopiedRange> copied;
        // Preserve version records outside this acquire, but never refresh
        // those bytes: an older image can extend beyond the current mappings.
        for (auto old : entry.copied) {
            if (!overlaps(old.dirty.bytes, bytes)) {
                copied.push_back(old);
                continue;
            }
            const auto end = old.dirty.bytes.offset + old.dirty.bytes.size;
            if (old.dirty.bytes.offset < bytes.offset) {
                auto head = old;
                head.dirty.bytes.size = bytes.offset - old.dirty.bytes.offset;
                copied.push_back(head);
            }
            if (end > bytes.offset + bytes.size) {
                old.dirty.bytes.offset = bytes.offset + bytes.size;
                old.dirty.bytes.size = end - old.dirty.bytes.offset;
                copied.push_back(old);
            }
        }
        for (const auto& interval : state) {
            if (!overlaps(interval.dirty.bytes, bytes)) continue;
            auto current = interval.dirty;
            const auto current_end = current.bytes.offset + current.bytes.size;
            const auto request_end = bytes.offset + bytes.size;
            current.bytes.offset = std::max(current.bytes.offset, bytes.offset);
            current.bytes.size = std::min(current_end, request_end) - current.bytes.offset;
            if (current.gpu_dirty) {
                const auto feedback = current.last_writer.snapshot();
                if (feedback.error) return *feedback.error;
                if (interval.storage == entry.image) {
                    copied.push_back({current, epoch});
                    continue;
                }
            }
            // Partition by previous interval boundaries, never by a maximum
            // watermark. Independently stale versions remain independent.
            std::vector<std::uint64_t> points{current.bytes.offset, current.bytes.offset + current.bytes.size};
            for (const auto& old : entry.copied) {
                if (!overlaps(old.dirty.bytes, current.bytes)) continue;
                points.push_back(std::max(old.dirty.bytes.offset, current.bytes.offset));
                points.push_back(std::min(old.dirty.bytes.offset + old.dirty.bytes.size,
                                          current.bytes.offset + current.bytes.size));
            }
            std::sort(points.begin(), points.end());
            points.erase(std::unique(points.begin(), points.end()), points.end());
            for (std::size_t i = 1; i < points.size(); ++i) {
                const BackingRange piece{current.bytes.backing, points[i - 1], points[i] - points[i - 1]};
                // Page certificates prove pieces outside the changed ranges:
                // their bytes equal the guest despite the older epoch record.
                const bool suspect = std::any_of(changed.begin(), changed.end(), [&](const BackingRange& range) {
                    return overlaps(range, piece);
                });
                const bool matches = std::any_of(entry.copied.begin(), entry.copied.end(), [&](const auto& old) {
                    return contains(old.dirty.bytes, piece) && old.dirty.version == current.version &&
                           (revalidated || old.epoch == epoch || (!changed.empty() && !suspect));
                });
                // A replacement must use every captured CPU byte, including
                // pieces previously validated in this epoch. The whole-range
                // comparison may have detected a difference in those pieces.
                if (matches && (snapshot.empty() || current.gpu_dirty)) continue;
                auto output = entry.image->native->contents().subspan(
                    entry.image->base + piece.offset - entry.image->bytes.offset, piece.size);
                RendererStatus status;
                auto local = std::dynamic_pointer_cast<const Image>(interval.storage);
                if (!snapshot.empty() && !current.gpu_dirty) {
                    const auto input = snapshot.subspan(piece.offset - bytes.offset, piece.size);
                    std::copy(input.begin(), input.end(), output.begin());
                    status = renderer_success();
                } else if (current.gpu_dirty && local && local->origin == allocator_) {
                    status = copy_ready({interval});
                    if (ok(status)) status = local->read_gpu(piece, output);
                } else {
                    status = read_guest(retained, piece, output);
                }
                if (!ok(status)) return status;
            }
            copied.push_back({current, epoch});
        }
        entry.copied = std::move(copied);
        return renderer_success();
    }

    RendererResult<std::shared_ptr<Entry>> build(
        const BufferRequest& request, const std::vector<ResourceStorageRange>& request_state,
        std::span<const std::byte> snapshot = {}, const std::shared_ptr<Entry>& replacement = {},
        std::uint64_t stamp = 0) {
        BackingRange bytes = request.range.bytes;
        std::vector<std::shared_ptr<Entry>> overlaps_old;
        for (const auto& entry : entries_) {
            if (!overlaps(entry->image->bytes, request.range.bytes)) continue;
            const auto end = std::max(bytes.offset + bytes.size, entry->image->bytes.offset + entry->image->bytes.size);
            bytes.offset = std::min(bytes.offset, entry->image->bytes.offset);
            bytes.size = end - bytes.offset;
            overlaps_old.push_back(entry);
        }
        // Published bindings contain immutable native offsets and may already
        // be encoded. Replacing their image cannot redirect those bindings.
        // Include retired images so invalidation cannot bypass this exclusion.
        std::erase_if(image_domain_->images, [](const auto& image) { return image.expired(); });
        // Settled accesses still retaining an overlapping image would count
        // as live owners below.
        if (!overlaps_old.empty()) memory_->release_settled_accesses();
        for (const auto& weak : image_domain_->images) {
            auto image = weak.lock();
            if (!image || !overlaps(image->bytes, bytes)) continue;
            const bool cached = std::any_of(entries_.begin(), entries_.end(), [&](const auto& entry) {
                return entry->image == image;
            });
            // The caller's request-state snapshot also retains the image
            // as an authority copy; those references are this call's own.
            const auto snapshot_owners = std::ranges::count_if(request_state, [&](const auto& interval) {
                return interval.storage == image;
            });
            const auto internal_owners = (cached ? 2 : 1) + static_cast<int>(snapshot_owners);
            // An exact same-cache replacement leaves retired bindings on their
            // original image. Growth and cross-cache overlap remain excluded.
            const bool exact_replacement = replacement && image->bytes == bytes &&
                                           image->origin == allocator_;
            // A finished GPU write keeps the image as the registry's dirty
            // authority. Write those bytes back (a coherent, earlier guest
            // update) so the growth copy owns them; a binding that still
            // holds the image, or a pending producer, refuses as before.
            bool authority = false;
            if (!exact_replacement && image.use_count() > internal_owners) {
                {
                    auto state = memory_->inspect_storage(image->bytes);
                    if (auto* failure = std::get_if<RendererError>(&state)) return *failure;
                    authority = std::ranges::any_of(std::get<std::vector<ResourceStorageRange>>(state),
                        [&](const auto& interval) { return interval.dirty.gpu_dirty && interval.storage == image; });
                }
                if (authority) {
                    auto resolved = memory_->resolve(image->bytes, ResolveReason::IncompatibleAlias);
                    if (!ok(resolved)) return std::get<RendererError>(resolved);
                }
            }
            if (!exact_replacement && image.use_count() > internal_owners) {
                char detail[200];
                std::snprintf(detail, sizeof detail,
                    "buffer growth retains overlapping bindings; split or reprepare the submission and retry "
                    "(owners %ld of %d%s, %s image +%#llx/%#llx, union +%#llx/%#llx)",
                    long(image.use_count()), internal_owners, authority ? " after writeback" : "",
                    image->origin == allocator_ ? "cached" : "foreign",
                    (unsigned long long)image->bytes.offset, (unsigned long long)image->bytes.size,
                    (unsigned long long)bytes.offset, (unsigned long long)bytes.size);
                return RendererError({CommandStatus::Stopped, std::string(detail)});
            }
        }
        auto state = memory_->inspect_storage(bytes);
        if (auto* failure = std::get_if<RendererError>(&state)) return *failure;
        auto ready = copy_ready(std::get<std::vector<ResourceStorageRange>>(state));
        if (!ok(ready)) return std::get<RendererError>(ready);
        // Keep the complete union and pad its native origin. No carried head
        // is dropped and every copy is contained in the new allocation.
        const auto relative = request.range.bytes.offset - bytes.offset;
        const auto base = (request.alignment - relative % request.alignment) % request.alignment;
        if (bytes.size > std::numeric_limits<std::uint64_t>::max() - base)
            return error(CommandStatus::BudgetExceeded, "aligned buffer size overflows");
        // Storage loads extract subword elements from uint words. Keep the
        // final native word addressable without retaining or exposing padding
        // as guest bytes; all shader bounds still use the exact logical size.
        auto capacity = align_up(base + bytes.size, 4);
        if (capacity && request.native_extent > request.range.bytes.size)
            capacity = align_up(std::max(*capacity, base + relative + request.native_extent), 4);
        if (!capacity)
            return error(CommandStatus::BudgetExceeded, "buffer word padding overflows");
        auto allocated = allocate(*capacity);
        if (auto* failure = std::get_if<RendererError>(&allocated)) return *failure;
        auto entry = std::make_shared<Entry>();
        entry->image = std::make_shared<Image>();
        entry->image->native = std::get<std::shared_ptr<BufferAllocation>>(std::move(allocated));
        auto padding = entry->image->native->contents().subspan(base + bytes.size);
        std::fill(padding.begin(), padding.end(), std::byte{0});
        entry->image->origin = allocator_;
        entry->image->domain = image_domain_;
        entry->image->bytes = bytes;
        entry->image->base = base;
        for (const auto& old : overlaps_old) {
            auto output = entry->image->native->contents().subspan(base + old->image->bytes.offset - bytes.offset,
                                                                  old->image->bytes.size);
            auto status = old->image->read_gpu(old->image->bytes, output);
            if (!ok(status)) return std::get<RendererError>(status);
            entry->copied.insert(entry->copied.end(), old->copied.begin(), old->copied.end());
        }
        auto status = refresh(*entry, request.range, request_state, snapshot);
        if (!ok(status)) return std::get<RendererError>(status);
        // Every image byte was established by reads taken after this stamp.
        entry->stamp = stamp;
        establish_page_hashes(*entry);
        // Construction pins stable objects; admission may erase vector elements.
        // Replace lookup only after every copy and refresh succeeded.
        std::erase_if(entries_, [&](const auto& old) {
            return std::find(overlaps_old.begin(), overlaps_old.end(), old) != overlaps_old.end();
        });
        image_domain_->images.push_back(entry->image);
        entries_.push_back(entry);
        return entry;
    }

    RendererResult<ResidentBuffer> snapshot(std::span<const std::byte> source, std::uint64_t alignment) {
        return snapshot_with(source.size(), alignment, [&](std::span<std::byte> output) {
            std::copy(source.begin(), source.end(), output.begin());
            return renderer_success();
        });
    }
    template <class Fill>
    RendererResult<ResidentBuffer> snapshot_with(std::uint64_t size, std::uint64_t alignment, Fill&& fill) {
        const auto aligned = align_up(size, alignment);
        if (!aligned) return error(CommandStatus::BudgetExceeded, "upload alignment overflows");
        std::shared_ptr<RingBlock> selected;
        std::uint64_t cursor = 0;
        // Ring discipline: a block's unused tail keeps accepting snapshots
        // while encoded work still reads earlier slices (written once, never
        // rewritten); a full block restarts only once no slice references it.
        // Newest blocks are tried first.
        for (auto it = ring_.rbegin(); it != ring_.rend(); ++it) {
            const auto& block = *it;
            if (block.use_count() == 1) {
                const auto tail = align_up(block->used, alignment);
                if (!tail || *tail > block->native->contents().size() ||
                    *aligned > block->native->contents().size() - *tail) block->used = 0;
            }
            const auto candidate = align_up(block->used, alignment);
            const auto capacity = block->native->contents().size();
            if (!candidate || *candidate > capacity || *aligned > capacity - *candidate) continue;
            selected = block;
            cursor = *candidate;
            break;
        }
        if (!selected) {
            // A block takes at most a quarter of a small residency budget.
            const auto capacity = std::max(*aligned, std::min(ring_block_bytes, budget_ / 4));
            auto allocated = allocate(capacity);
            // The coherent source remains pinned during snapshot admission.
            // A preferred arena must not reject a slice that fits by itself.
            if (const auto* failure = std::get_if<RendererError>(&allocated);
                failure && failure->command().status == CommandStatus::BudgetExceeded && capacity > *aligned)
                allocated = allocate(*aligned);
            if (auto* failure = std::get_if<RendererError>(&allocated)) return *failure;
            selected = std::make_shared<RingBlock>();
            selected->native = std::get<std::shared_ptr<BufferAllocation>>(std::move(allocated));
            ring_.push_back(selected);
        }
        auto output = selected->native->contents().subspan(cursor, size);
        auto filled = fill(output);
        if (auto* failure = std::get_if<RendererError>(&filled)) return *failure;
        selected->used = cursor + *aligned;
        auto slice = std::make_shared<Slice>();
        slice->block = selected;
        slice->base = cursor;
        slice->size = size;
        ResidentBuffer result;
        result.storage = slice;
        result.offset = cursor;
        result.size = size;
        return result;
    }

    RendererResult<ResidentBuffer> acquire_checked(const BufferRequest& request) {
        if (!request.range.lease || !request.range.bytes.size || !power_of_two(request.alignment) ||
            request.range.bytes.offset > std::numeric_limits<std::uint64_t>::max() - request.range.bytes.size)
            return error(CommandStatus::Unsupported, "buffer request has invalid range or alignment");
        if (request.use == BufferUse::Uniform && request.access != ResourceAccess::Read)
            return error(CommandStatus::Unsupported, "immutable uniform slices require read access");
        auto valid = memory_->validate_retained(request.range, request.access);
        if (!ok(valid)) return std::get<RendererError>(valid);
        // Take the stamp before any byte the cache keeps is read.
        const auto stamp = memory_->cpu_write_stamp();
        auto inspected = memory_->inspect_storage(request.range.bytes);
        if (auto* failure = std::get_if<RendererError>(&inspected)) return *failure;
        const auto& initial_state = std::get<std::vector<ResourceStorageRange>>(inspected);
        std::shared_ptr<Entry> entry;
        for (const auto& candidate : entries_) {
            const auto start = candidate->image->base + request.range.bytes.offset - candidate->image->bytes.offset;
            if (contains(candidate->image->bytes, request.range.bytes) && start % request.alignment == 0 &&
                request.native_extent <= candidate->image->native->contents().size() - std::min<std::uint64_t>(
                    start, candidate->image->native->contents().size())) {
                entry = candidate;
                break;
            }
        }
        if (!entry) {
            auto built = build(request, initial_state, {}, {}, stamp);
            if (auto* failure = std::get_if<RendererError>(&built)) return *failure;
            entry = std::get<std::shared_ptr<Entry>>(std::move(built));
        } else {
            const auto epoch = memory_->cpu_cache_epoch();
            bool replacement_required = false;
            bool epoch_revalidation = false;
            for (const auto& interval : initial_state) {
                const bool same_version = copied_covers(*entry, interval.dirty);
                const bool local_authority = interval.dirty.gpu_dirty && interval.storage == entry->image;
                if (!same_version && !local_authority) replacement_required = true;
                if (!interval.dirty.gpu_dirty && same_version &&
                    !copied_covers(*entry, interval.dirty, epoch))
                    epoch_revalidation = true;
            }

            // The certificate stamp proves which images no CPU store reached
            // since their contents were established; those skip every guest
            // access. Otherwise one snapshot feeds page hashes, and only page
            // ranges whose hash differs are re-copied in place while no
            // consumer pins the image; pinned images are replaced wholesale.
            bool certificate_clean = false;
            std::vector<BackingRange> changed_pages;
            std::vector<std::byte> current;
            if (!replacement_required && epoch_revalidation &&
                !memory_->cpu_written_since(entry->image->bytes, entry->stamp)) {
                certificate_clean = true;
                ++statistics_->revalidations_skipped;
            } else if ((replacement_required || epoch_revalidation) &&
                std::any_of(initial_state.begin(), initial_state.end(),
                            [](const auto& interval) { return !interval.dirty.gpu_dirty; })) {
                current.resize(request.range.bytes.size);
                // Snapshot each contiguous CPU-backed span once. GPU-owned
                // pieces stay resident and retain their producer dependency;
                // they must never be read through the CPU backing here.
                for (std::size_t i = 0; i < initial_state.size();) {
                    if (initial_state[i].dirty.gpu_dirty) {
                        ++i;
                        continue;
                    }
                    auto bytes = initial_state[i++].dirty.bytes;
                    while (i < initial_state.size() && !initial_state[i].dirty.gpu_dirty &&
                           bytes.offset + bytes.size == initial_state[i].dirty.bytes.offset)
                        bytes.size += initial_state[i++].dirty.bytes.size;
                    auto output = std::span(current).subspan(
                        bytes.offset - request.range.bytes.offset, bytes.size);
                    auto status = read_guest(request.range, bytes, output);
                    if (!ok(status)) return std::get<RendererError>(status);
                }
            }

            bool identical = false;
            if (certificate_clean) {
                identical = true;
            } else if (epoch_revalidation && !current.empty()) {
                auto deltas = suspect_page_deltas(*entry, initial_state, current, request.range.bytes);
                if (auto* failure = std::get_if<RendererError>(&deltas)) return *failure;
                changed_pages = std::get<std::vector<BackingRange>>(std::move(deltas));
                identical = changed_pages.empty();
            }
            if (replacement_required) {
                auto built = build(request, initial_state, current, entry, stamp);
                if (auto* failure = std::get_if<RendererError>(&built)) return *failure;
                entry = std::get<std::shared_ptr<Entry>>(std::move(built));
            } else if (epoch_revalidation && !identical) {
                if (entry->image.use_count() == 1) {
                    auto status = refresh(*entry, request.range, initial_state, current,
                                          false, changed_pages);
                    if (!ok(status)) return std::get<RendererError>(status);
                    ++statistics_->reuploads;
                    for (const auto& range : changed_pages) statistics_->reupload_bytes += range.size;
                    entry->stamp = stamp;
                    establish_page_hashes(*entry);
                } else {
                    // Consumers keep their original allocation bytes.
                    auto built = build(request, initial_state, current, entry, stamp);
                    if (auto* failure = std::get_if<RendererError>(&built)) return *failure;
                    entry = std::get<std::shared_ptr<Entry>>(std::move(built));
                }
            } else {
                auto status = refresh(*entry, request.range, initial_state, {}, epoch_revalidation);
                if (!ok(status)) return std::get<RendererError>(status);
                if (epoch_revalidation) {
                    if (!certificate_clean) ++statistics_->revalidations_hashed;
                    entry->stamp = stamp;
                }
            }
        }
        entry->last_use = ++clock_;
        inspected = memory_->inspect_storage(request.range.bytes);
        if (auto* failure = std::get_if<RendererError>(&inspected)) return *failure;
        const auto& state = std::get<std::vector<ResourceStorageRange>>(inspected);
        ResidentBuffer result;
        result.offset = entry->image->base + request.range.bytes.offset - entry->image->bytes.offset;
        result.size = request.range.bytes.size;
        result.storage = entry->image;
        if (request.use == BufferUse::Uniform) {
            auto ready = copy_ready(state);
            if (!ok(ready)) return std::get<RendererError>(ready);
            const auto contents = std::span<const std::byte>(entry->image->native->contents()).subspan(result.offset, result.size);
            // A constant window binds its whole extent: the guest's whole
            // words, then zero words past the bound size.
            auto copied = request.native_extent > contents.size()
                ? snapshot_with(request.native_extent, request.alignment, [&](std::span<std::byte> output) {
                      const auto words = contents.size() & ~std::size_t{3};
                      std::copy_n(contents.begin(), words, output.begin());
                      std::fill(output.begin() + std::ptrdiff_t(words), output.end(), std::byte{});
                      return renderer_success();
                  })
                : snapshot(contents, request.alignment);
            if (auto* failure = std::get_if<RendererError>(&copied)) return *failure;
            result = std::get<ResidentBuffer>(std::move(copied));
        }
        result.key = {entry->image->bytes};
        for (const auto& interval : state) {
            result.uploaded_version.cpu_write = std::max(result.uploaded_version.cpu_write, interval.dirty.version.cpu_write);
            result.uploaded_version.gpu_write = std::max(result.uploaded_version.gpu_write, interval.dirty.version.gpu_write);
            if (!interval.dirty.gpu_dirty) continue;
            const auto feedback = interval.dirty.last_writer.snapshot();
            if (feedback.error) return *feedback.error;
            if (pending(feedback) && std::none_of(result.dependencies.begin(), result.dependencies.end(), [&](const auto& token) {
                    return token.snapshot().id == feedback.id;
                })) result.dependencies.push_back(interval.dirty.last_writer);
        }
        return result;
    }

    std::shared_ptr<BufferAllocator> allocator_;
    std::shared_ptr<ResourceMemoryRegistry> memory_;
    std::shared_ptr<ImageDomain> image_domain_;
    std::uint64_t budget_, clock_ = 0;
    std::shared_ptr<BufferCacheStatistics> statistics_ = std::make_shared<BufferCacheStatistics>();
    std::vector<std::shared_ptr<Entry>> entries_;
    std::vector<std::shared_ptr<RingBlock>> ring_;
    std::vector<std::weak_ptr<BufferAllocation>> allocations_;
};

struct PortableAllocation final : BufferAllocation {
    mutable std::vector<std::byte> bytes;
    std::span<std::byte> contents() const noexcept override { return bytes; }
    std::uint64_t cost() const noexcept override { return bytes.size(); }
};
struct PortableAllocator final : BufferAllocator {
    std::uint64_t device_identity() const noexcept override { return 1; }
    RendererResult<std::shared_ptr<BufferAllocation>> allocate(std::uint64_t size) override {
        auto allocation = std::make_shared<PortableAllocation>();
        allocation->bytes.resize(size);
        return std::shared_ptr<BufferAllocation>(allocation);
    }
};
} // namespace

RendererResult<std::vector<std::byte>> snapshot_buffer_range(
    ResourceMemory& memory, std::uint64_t address, std::uint64_t size,
    BufferSnapshotPolicy policy) {
    try {
        if (size > (256u << 20))
            return error(CommandStatus::BudgetExceeded, "buffer snapshot exceeds the staging budget");
        std::vector<std::byte> bytes(static_cast<std::size_t>(size));
        auto status = snapshot_buffer_into(memory, address, bytes, policy);
        if (auto* failure = std::get_if<RendererError>(&status)) return *failure;
        return bytes;
    } catch (...) {
        return error(CommandStatus::MemoryFault, "buffer snapshot allocation or read failed");
    }
}

RendererStatus snapshot_buffer_into(
    ResourceMemory& memory, std::uint64_t address, std::span<std::byte> output,
    BufferSnapshotPolicy policy) {
    const std::uint64_t size = output.size();
    try {
        constexpr std::uint64_t snapshot_limit = 256u << 20;
        if (policy != BufferSnapshotPolicy::RequireMapped &&
            policy != BufferSnapshotPolicy::ZeroMissingConstantBufferBytes)
            return error(CommandStatus::Unsupported, "buffer snapshot policy is invalid");
        const bool zero_missing = policy == BufferSnapshotPolicy::ZeroMissingConstantBufferBytes;
        if (!size || size > std::numeric_limits<std::uint64_t>::max() - address)
            return error(CommandStatus::MemoryFault, "buffer snapshot range is empty or overflows");
        if (size > snapshot_limit)
            return error(CommandStatus::BudgetExceeded, "buffer snapshot exceeds the staging budget");
        struct Fragment {
            std::uint64_t address, offset, size;
            RetainedRange retained;
            std::unique_ptr<CpuReadTransaction> transaction;
        };
        std::vector<Fragment> fragments;
        const auto mappings = memory.address_space().mappings();
        const auto end = address + size;
        auto cursor = address;
        while (cursor != end) {
            const auto after = std::upper_bound(mappings.begin(), mappings.end(), cursor,
                [](std::uint64_t candidate, const Mapping& mapping) {
                    return candidate < mapping.gpu_base;
                });
            auto found = mappings.end();
            if (after != mappings.begin()) {
                const auto candidate = std::prev(after);
                if (cursor - candidate->gpu_base < candidate->mapped_size)
                    found = candidate;
            }
            if (found == mappings.end()) {
                if (!zero_missing)
                    return RendererError({CommandStatus::MemoryFault,
                        "buffer snapshot reaches unmapped GPU memory", cursor});
                const auto next = after == mappings.end() ? end : std::min(end, after->gpu_base);
                std::fill(output.begin() + std::ptrdiff_t(cursor - address),
                          output.begin() + std::ptrdiff_t(next - address), std::byte{});
                cursor = next;
                continue;
            }
            const auto delta = cursor - found->gpu_base;
            const auto backed = found->backing_size - found->buffer_offset;
            if (delta >= backed) {
                if (!zero_missing)
                    return RendererError({CommandStatus::MemoryFault,
                        "buffer snapshot reaches unbacked mapping padding", cursor});
                const auto next = std::min(end, found->gpu_base + found->mapped_size);
                std::fill(output.begin() + std::ptrdiff_t(cursor - address),
                          output.begin() + std::ptrdiff_t(next - address), std::byte{});
                cursor = next;
                continue;
            }
            const auto count = std::min({end - cursor, found->mapped_size - delta, backed - delta});
            auto retained = memory.retain(cursor, count, ResourceAccess::Read);
            if (auto* failure = std::get_if<RendererError>(&retained)) return *failure;
            fragments.push_back({cursor, cursor - address, count,
                std::get<RetainedRange>(std::move(retained)), {}});
            cursor += count;
        }
        // Validate and retain the complete range before reading any fragment.
        // Keep every read transaction alive until the complete copy is owned.
        for (auto& fragment : fragments) {
            auto transaction = memory.prepare_cpu_read(fragment.retained.bytes, ResolveReason::CpuEngineRead);
            if (auto* failure = std::get_if<RendererError>(&transaction)) return *failure;
            fragment.transaction = std::get<std::unique_ptr<CpuReadTransaction>>(std::move(transaction));
        }
        // The held read transactions already resolved GPU authority and armed
        // write tracking; host-addressable backings are copied directly
        // instead of preparing each fragment a second time.
        auto* registry = dynamic_cast<ResourceMemoryRegistry*>(&memory);
        for (const auto& fragment : fragments) {
            const auto& bytes = fragment.retained.bytes;
            const auto host = registry ? registry->backing_host_range(bytes.backing) : std::nullopt;
            if (host && bytes.offset <= host->second && host->second - bytes.offset >= fragment.size) {
                std::memcpy(output.data() + fragment.offset,
                            reinterpret_cast<const void*>(host->first + bytes.offset), fragment.size);
                continue;
            }
            const auto read = memory.address_space().read(fragment.address,
                output.subspan(fragment.offset, fragment.size));
            if (!read) return RendererError({CommandStatus::MemoryFault, read.detail});
        }
        return renderer_success();
    } catch (...) {
        return error(CommandStatus::MemoryFault, "buffer snapshot allocation or read failed");
    }
}

RendererResult<std::shared_ptr<BufferCache>> make_resident_buffer_cache(
    std::shared_ptr<BufferAllocator> allocator, std::shared_ptr<ResourceMemory> memory, std::uint64_t budget) {
    auto registry = std::dynamic_pointer_cast<ResourceMemoryRegistry>(memory);
    if (!allocator || !allocator->device_identity() || !registry || !budget)
        return error(CommandStatus::Unsupported, "buffer cache requires an allocator, registry and nonzero budget");
    return std::shared_ptr<BufferCache>(std::make_shared<Cache>(std::move(allocator), std::move(registry), budget));
}
BufferCacheStatistics buffer_cache_statistics(const BufferCache& cache) noexcept {
    if (auto* native = dynamic_cast<const Cache*>(&cache)) return native->statistics();
    return {};
}
RendererResult<std::shared_ptr<BufferCache>> make_buffer_cache(
    std::shared_ptr<ResourceMemory> memory, std::uint64_t budget) {
    return make_resident_buffer_cache(std::make_shared<PortableAllocator>(), std::move(memory), budget);
}
} // namespace armigo::graphics
