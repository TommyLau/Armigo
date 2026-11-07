// ============================================================================
// Armigo Horizon - Minimal Kernel Compatibility
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include "memory_interval_map.h"
#include "process_memory.h"

#include <algorithm>
#include <limits>
#include <new>
#include <stdexcept>

namespace armigo::horizon {
namespace {
bool same(const MemoryBlock& a, const MemoryBlock& b) noexcept {
    return a.range.address == b.range.address && a.range.size == b.range.size &&
        a.type == b.type && a.attributes == b.attributes && a.permissions == b.permissions &&
        a.ipc_references == b.ipc_references && a.device_references == b.device_references &&
        a.merge_disable == b.merge_disable;
}
}

MemoryIntervalMap::Node* MemoryIntervalMap::minimum(Node* node) noexcept {
    if (node) while (node->left) node = node->left;
    return node;
}
MemoryIntervalMap::Node* MemoryIntervalMap::successor(Node* node) noexcept {
    if (node->right) return minimum(node->right);
    auto* parent = node->parent;
    while (parent && node == parent->right) { node = parent; parent = parent->parent; }
    return parent;
}

std::shared_ptr<MemoryIntervalMap> MemoryIntervalMap::create(
    std::shared_ptr<SystemResourcePool> pool, std::uint64_t end) {
    if (!pool || !end || end > ProcessRecord::query_end || end % guest_page_size) return {};
    try {
        auto map = std::shared_ptr<MemoryIntervalMap>(new MemoryIntervalMap(std::move(pool), end));
        auto lease = map->pool_->allocate(SystemResourcePool::Kind::memory_block);
        if (!lease) return {};
        map->insert(map->acquire(std::move(*lease), {{0, end}, MemoryType::free}));
        return map;
    } catch (const std::bad_alloc&) { return {}; }
}

MemoryIntervalMap::~MemoryIntervalMap() {
    // No observer can hold the map during its destruction. Release children
    // before their parent without allocating a traversal stack.
    auto destroy = [&](auto&& self, Node* node) -> void {
        if (!node) return;
        self(self, node->left); self(self, node->right); release(node);
    };
    destroy(destroy, root_);
}
MemoryIntervalMap::Node* MemoryIntervalMap::predecessor(Node* node) noexcept {
    if (node->left) { node = node->left; while (node->right) node = node->right; return node; }
    auto* parent = node->parent;
    while (parent && node == parent->left) { node = parent; parent = parent->parent; }
    return parent;
}
MemoryIntervalMap::Node* MemoryIntervalMap::acquire(SystemResourcePool::Allocation&& lease,
    const MemoryBlock& value) noexcept {
    auto* node = std::construct_at(reinterpret_cast<Node*>(lease.bytes().data()));
    node->value = value;
    // The tree assumes this slot's ownership; pool_ outlives every tree node.
    // Staged leases retain normal RAII ownership until this publication point.
    lease.data_ = nullptr;
    lease.pool_.reset();
    return node;
}
void MemoryIntervalMap::release(Node* node) noexcept {
    std::destroy_at(node);
    pool_->release(SystemResourcePool::Kind::memory_block, reinterpret_cast<std::byte*>(node));
}
void MemoryIntervalMap::rotate_left(Node* node) noexcept {
    auto* pivot = node->right;
    node->right = pivot->left;
    if (pivot->left) pivot->left->parent = node;
    pivot->parent = node->parent;
    if (!node->parent) root_ = pivot;
    else if (node == node->parent->left) node->parent->left = pivot;
    else node->parent->right = pivot;
    pivot->left = node; node->parent = pivot;
}
void MemoryIntervalMap::rotate_right(Node* node) noexcept {
    auto* pivot = node->left;
    node->left = pivot->right;
    if (pivot->right) pivot->right->parent = node;
    pivot->parent = node->parent;
    if (!node->parent) root_ = pivot;
    else if (node == node->parent->right) node->parent->right = pivot;
    else node->parent->left = pivot;
    pivot->right = node; node->parent = pivot;
}
void MemoryIntervalMap::insert(Node* node) noexcept {
    node->left = node->right = nullptr; node->red = true;
    Node* parent = nullptr; auto* at = root_;
    while (at) { parent = at; at = node->value.range.address < at->value.range.address ? at->left : at->right; }
    node->parent = parent;
    if (!parent) root_ = node;
    else if (node->value.range.address < parent->value.range.address) parent->left = node;
    else parent->right = node;
    while (node->parent && node->parent->red) {
        auto* p = node->parent; auto* g = p->parent;
        if (p == g->left) {
            auto* uncle = g->right;
            if (red(uncle)) { p->red = uncle->red = false; g->red = true; node = g; }
            else {
                if (node == p->right) { node = p; rotate_left(node); p = node->parent; g = p->parent; }
                p->red = false; g->red = true; rotate_right(g);
            }
        } else {
            auto* uncle = g->left;
            if (red(uncle)) { p->red = uncle->red = false; g->red = true; node = g; }
            else {
                if (node == p->left) { node = p; rotate_right(node); p = node->parent; g = p->parent; }
                p->red = false; g->red = true; rotate_left(g);
            }
        }
    }
    root_->red = false; ++count_;
}
void MemoryIntervalMap::transplant(Node* from, Node* to) noexcept {
    if (!from->parent) root_ = to;
    else if (from == from->parent->left) from->parent->left = to;
    else from->parent->right = to;
    if (to) to->parent = from->parent;
}
void MemoryIntervalMap::repair_erase(Node* child, Node* parent) noexcept {
    while (child != root_ && !red(child)) {
        if (child == parent->left) {
            auto* sibling = parent->right;
            if (red(sibling)) { sibling->red = false; parent->red = true; rotate_left(parent); sibling = parent->right; }
            if (!sibling || (!red(sibling->left) && !red(sibling->right))) {
                if (sibling) sibling->red = true;
                child = parent; parent = child->parent;
            } else {
                if (!red(sibling->right)) {
                    if (sibling->left) sibling->left->red = false;
                    sibling->red = true; rotate_right(sibling); sibling = parent->right;
                }
                sibling->red = parent->red; parent->red = false;
                if (sibling->right) sibling->right->red = false;
                rotate_left(parent); child = root_; parent = nullptr;
            }
        } else {
            auto* sibling = parent->left;
            if (red(sibling)) { sibling->red = false; parent->red = true; rotate_right(parent); sibling = parent->left; }
            if (!sibling || (!red(sibling->left) && !red(sibling->right))) {
                if (sibling) sibling->red = true;
                child = parent; parent = child->parent;
            } else {
                if (!red(sibling->left)) {
                    if (sibling->right) sibling->right->red = false;
                    sibling->red = true; rotate_left(sibling); sibling = parent->left;
                }
                sibling->red = parent->red; parent->red = false;
                if (sibling->left) sibling->left->red = false;
                rotate_right(parent); child = root_; parent = nullptr;
            }
        }
    }
    if (child) child->red = false;
}
void MemoryIntervalMap::detach(Node* node) noexcept {
    auto* removed = node; bool was_red = removed->red;
    Node* child = nullptr; Node* parent = nullptr;
    if (!node->left) { child = node->right; parent = node->parent; transplant(node, child); }
    else if (!node->right) { child = node->left; parent = node->parent; transplant(node, child); }
    else {
        removed = minimum(node->right); was_red = removed->red; child = removed->right;
        if (removed->parent == node) { parent = removed; if (child) child->parent = removed; }
        else {
            parent = removed->parent; transplant(removed, child);
            removed->right = node->right; removed->right->parent = removed;
        }
        transplant(node, removed);
        removed->left = node->left; removed->left->parent = removed;
        removed->red = node->red;
    }
    if (!was_red) repair_erase(child, parent);
    node->parent = node->left = node->right = nullptr;
    --count_;
}

IntervalMapError MemoryIntervalMap::transact(std::span<const Update> updates,
    bool (*apply)(void*) noexcept, void* context, std::size_t node_limit) {
    if (!apply) return IntervalMapError::invalid_map;
    for (const auto& update : updates) {
        const auto range = update.range;
        if (!range.size || range.address >= end_ || range.size > end_ - range.address ||
            (range.address | range.size) % guest_page_size ||
            (update.replacement && (update.replacement->permissions & ~7u)))
            return IntervalMapError::invalid_map;
    }
    try {
        std::lock_guard lock(mutex_);
        std::vector<Node*> touched;
        for (const auto& update : updates) {
            auto* node = const_cast<Node*>(lookup(update.range.address));
            if (auto* before = predecessor(node)) touched.push_back(before);
            const auto end = update.range.address + update.range.size;
            do { touched.push_back(node); node = successor(node); }
            while (node && node->value.range.address < end);
            if (node) touched.push_back(node);
        }
        std::sort(touched.begin(), touched.end(), [](auto* a, auto* b) {
            return a->value.range.address < b->value.range.address;
        });
        touched.erase(std::unique(touched.begin(), touched.end()), touched.end());
        std::vector<MemoryBlock> desired;
        desired.reserve(touched.size());
        for (auto* node : touched) desired.push_back(node->value);
        for (const auto& update : updates) {
            std::vector<MemoryBlock> next;
            next.reserve(desired.size() + 2);
            for (const auto& block : desired) {
                if (!process_memory::overlaps(block.range, update.range)) { next.push_back(block); continue; }
                const auto end = block.range.address + block.range.size;
                const auto begin = std::max(block.range.address, update.range.address);
                const auto changed_end = std::min(end, update.range.address + update.range.size);
                if (block.range.address < begin)
                    next.push_back(process_memory::fragment(block, {block.range.address, begin - block.range.address}));
                auto changed = update.replacement.value_or(MemoryBlock{});
                changed.range = {begin, changed_end - begin};
                changed.merge_disable = process_memory::fragment(block, changed.range).merge_disable & ~update.clear_merge;
                if (begin == update.range.address) changed.merge_disable |= update.set_merge;
                next.push_back(changed);
                if (changed_end < end)
                    next.push_back(process_memory::fragment(block, {changed_end, end - changed_end}));
            }
            process_memory::coalesce(next);
            desired.swap(next);
        }
        struct Change { Node* node; MemoryBlock after; };
        std::vector<Change> changes;
        std::vector<Node*> removed;
        std::vector<MemoryBlock> inserted;
        std::vector<SystemResourcePool::Allocation> extra;
        std::size_t old = 0, next = 0;
        while (old < touched.size() || next < desired.size()) {
            auto* node = old < touched.size() ? touched[old] : nullptr;
            if (node && next < desired.size() && node->value.range.address == desired[next].range.address) {
                if (!same(node->value, desired[next])) changes.push_back({node, desired[next]});
                ++old; ++next;
            } else if (node && (next == desired.size() || node->value.range.address < desired[next].range.address)) {
                removed.push_back(node); ++old;
            } else inserted.push_back(desired[next++]);
        }
        if (count_ - removed.size() + inserted.size() > node_limit) return IntervalMapError::resource_limit;
        const auto growth = inserted.size() > removed.size() ? inserted.size() - removed.size() : 0;
        extra.reserve(growth);
        for (std::size_t i = 0; i < growth; ++i) {
            auto allocation = pool_->allocate(SystemResourcePool::Kind::memory_block);
            if (!allocation) return IntervalMapError::resource_limit;
            extra.push_back(std::move(*allocation));
        }
        if (!apply(context)) return IntervalMapError::native_failure;
        // Untouched old nodes provide undo state through the host operation.
        // Publication performs no allocation and cannot report a later failure.
        for (auto* node : removed) detach(node);
        for (const auto& change : changes) change.node->value = change.after;
        const auto recycled = std::min(removed.size(), inserted.size());
        for (std::size_t i = 0; i < recycled; ++i) {
            removed[i]->value = inserted[i]; insert(removed[i]);
        }
        for (std::size_t i = recycled; i < inserted.size(); ++i)
            insert(acquire(std::move(extra[i - recycled]), inserted[i]));
        for (std::size_t i = recycled; i < removed.size(); ++i) release(removed[i]);
        return IntervalMapError::none;
    } catch (const std::bad_alloc&) { return IntervalMapError::resource_limit; }
      catch (const std::length_error&) { return IntervalMapError::resource_limit; }
}

const MemoryIntervalMap::Node* MemoryIntervalMap::lookup(std::uint64_t address) const noexcept {
    auto* node = root_; const Node* candidate = nullptr;
    while (node) {
        if (address < node->value.range.address) node = node->left;
        else { candidate = node; node = node->right; }
    }
    return candidate;
}
MemoryInfo MemoryIntervalMap::query(std::uint64_t address) const {
    std::lock_guard lock(mutex_);
    if (address >= end_) return {{end_, std::uint64_t{0} - end_}, MemoryType::reserved};
    return lookup(address)->value;
}
bool MemoryIntervalMap::covered(AddressRange range, std::uint8_t permissions) const {
    if (!range.size || range.address >= end_ || range.size > end_ - range.address) return false;
    std::lock_guard lock(mutex_);
    auto* node = const_cast<Node*>(lookup(range.address));
    auto cursor = range.address;
    const auto end = range.address + range.size;
    while (node && cursor < end) {
        if (node->value.type == MemoryType::free || (node->value.permissions & permissions) != permissions) return false;
        cursor = node->value.range.address + node->value.range.size;
        node = successor(node);
    }
    return cursor >= end;
}
bool MemoryIntervalMap::free(AddressRange range) const {
    if (!range.size || range.address >= end_ || range.size > end_ - range.address) return false;
    std::lock_guard lock(mutex_);
    for (auto* node = const_cast<Node*>(lookup(range.address));
         node && node->value.range.address < range.address + range.size; node = successor(node))
        if (node->value.type != MemoryType::free) return false;
    return true;
}
std::optional<MemoryBlock> MemoryIntervalMap::uniform(AddressRange range) const {
    if (!range.size || range.address >= end_ || range.size > end_ - range.address) return {};
    std::lock_guard lock(mutex_);
    std::optional<MemoryBlock> result;
    for (auto* node = const_cast<Node*>(lookup(range.address));
         node && node->value.range.address < range.address + range.size; node = successor(node)) {
        const auto& b = node->value;
        if (b.type == MemoryType::free || (result &&
            (result->type != b.type || result->attributes != b.attributes || result->permissions != b.permissions ||
             result->ipc_references != b.ipc_references || result->device_references != b.device_references))) return {};
        result = b;
    }
    result->range = range;
    return result;
}
std::vector<MemoryBlock> MemoryIntervalMap::snapshot(AddressRange range) const {
    std::lock_guard lock(mutex_);
    std::vector<MemoryBlock> result;
    if (!range.size || range.address >= end_) return result;
    const auto end = range.address + std::min(range.size, end_ - range.address);
    for (auto* n = const_cast<Node*>(lookup(range.address)); n && n->value.range.address < end; n = successor(n))
        if (n->value.type != MemoryType::free || n->value.merge_disable) result.push_back(n->value);
    return result;
}
std::size_t MemoryIntervalMap::node_count() const { std::lock_guard lock(mutex_); return count_; }

bool MemoryIntervalMap::valid_tree() const {
    std::lock_guard lock(mutex_);
    if (!root_ || root_->parent || root_->red) return false;
    std::size_t nodes = 0; std::uint64_t end = 0;
    auto walk = [&](auto&& self, Node* node, Node* parent, unsigned depth) -> int {
        if (!node) return 1;
        if (depth > 64 || node->parent != parent ||
            (node->red && (red(node->left) || red(node->right)))) return -1;
        const auto left = self(self, node->left, node, depth + 1);
        if (left < 0 || node->value.range.address != end || !node->value.range.size) return -1;
        end += node->value.range.size; ++nodes;
        const auto right = self(self, node->right, node, depth + 1);
        if (right != left) return -1;
        return left + !node->red;
    };
    return walk(walk, root_, nullptr, 0) > 0 && nodes == count_ && end == end_;
}
} // namespace armigo::horizon
