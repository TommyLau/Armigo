// ============================================================================
// Armigo Horizon - Minimal Kernel Compatibility
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#pragma once

#include <armigo/horizon/startup_queries.h>
#include <armigo/horizon/system_resource_pool.h>

namespace armigo::horizon {

enum class IntervalMapError { none, invalid_map, resource_limit, native_failure };

// One authoritative interval tree. Records and intrusive links occupy actual
// 64-byte pool slots. Snapshots and transaction plans are transient host data.
class MemoryIntervalMap final {
public:
    static std::shared_ptr<MemoryIntervalMap> create(std::shared_ptr<SystemResourcePool> pool,
        std::uint64_t end = ProcessRecord::query_end);
    using Update = MemoryMapUpdate;
    ~MemoryIntervalMap();
    // Stage only affected intervals and their immediate merge neighbours. The
    // callback must restore partial native effects on refusal and never reenter.
    IntervalMapError transact(std::span<const Update> updates,
        bool (*apply)(void*) noexcept, void* context, std::size_t node_limit = 1u << 20);
    MemoryInfo query(std::uint64_t address) const;
    bool covered(AddressRange range, std::uint8_t permissions) const;
    bool free(AddressRange range) const;
    std::optional<MemoryBlock> uniform(AddressRange range) const;
    std::vector<MemoryBlock> snapshot(AddressRange range = {0, ProcessRecord::query_end}) const;
    std::size_t node_count() const;
    bool valid_tree() const;
    std::uint64_t assigned_bytes() const noexcept { return pool_->used_bytes(); }
private:
    struct Node {
        MemoryBlock value;
        Node* left = nullptr;
        Node* right = nullptr;
        Node* parent = nullptr;
        bool red = true;
    };
    static_assert(sizeof(Node) == 64);
    MemoryIntervalMap(std::shared_ptr<SystemResourcePool> pool, std::uint64_t end)
        : pool_(std::move(pool)), end_(end) {}
    static bool red(const Node* node) noexcept { return node && node->red; }
    static Node* minimum(Node* node) noexcept;
    static Node* successor(Node* node) noexcept;
    static Node* predecessor(Node* node) noexcept;
    void rotate_left(Node* node) noexcept;
    void rotate_right(Node* node) noexcept;
    void insert(Node* node) noexcept;
    void transplant(Node* from, Node* to) noexcept;
    void detach(Node* node) noexcept;
    void repair_erase(Node* child, Node* parent) noexcept;
    void release(Node* node) noexcept;
    Node* acquire(SystemResourcePool::Allocation&& lease, const MemoryBlock& value) noexcept;
    const Node* lookup(std::uint64_t address) const noexcept;
    std::shared_ptr<SystemResourcePool> pool_;
    std::uint64_t end_;
    Node* root_ = nullptr;
    std::size_t count_ = 0;
    mutable std::mutex mutex_;
};
} // namespace armigo::horizon
