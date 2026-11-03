// ============================================================================
// Armigo Tests - Native Runtime Verification
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include <armigo/memory/guest_memory_model.h>

#include <cstdio>
#include <cstdlib>
#include <new>

namespace {
bool fail_next_allocation = false;
}

// Fault injection is confined to this test executable.
void* operator new(std::size_t size) {
    if (fail_next_allocation) {
        fail_next_allocation = false;
        throw std::bad_alloc();
    }
    if (void* pointer = std::malloc(size == 0 ? 1 : size)) {
        return pointer;
    }
    throw std::bad_alloc();
}

void operator delete(void* pointer) noexcept {
    std::free(pointer);
}

namespace {
using namespace armigo::memory;

bool require(bool condition, const char* message) {
    if (!condition) {
        std::fprintf(stderr, "%s\n", message);
    }
    return condition;
}

bool permissions_match() {
    return require(!has_permission(Permission::read, Permission::read | Permission::write),
                   "A partial permission mask must not satisfy a composite request.") &&
           require(has_permission(Permission::read | Permission::write,
                                  Permission::read | Permission::write),
                   "A complete permission mask must satisfy a composite request.") &&
           require(has_permission(Permission::none, Permission::none),
                   "An empty request requires no permission bits.");
}

bool protection_merges() {
    for (const std::uint64_t length : {2u, 4u}) {
        auto model = GuestMemoryModel::create(0, 16, 1);
        if (!model || !model->map(0, 4, Permission::read).ok() ||
            !model->map(4, 4, Permission::write).ok() ||
            !model->protect(2, length, Permission::write).ok()) {
            return require(false, "Protection regression setup failed.");
        }
        const auto changed = model->query(2);
        if (!require(model->mapping_count() == 2 && changed &&
                     changed->start == 2 && changed->size == 6,
                     "Protected range must merge with its writable neighbor.")) {
            return false;
        }
    }
    return true;
}

bool allocation_failures_preserve_state() {
    for (int operation = 0; operation < 3; ++operation) {
        auto model = GuestMemoryModel::create(0, 16, 1);
        if (!model || (operation != 0 && !model->map(0, 10, Permission::read).ok())) {
            return require(false, "Allocation-failure regression setup failed.");
        }
        bool allocation_failed = false;
        fail_next_allocation = true;
        try {
            if (operation == 0) {
                static_cast<void>(model->map(0, 10, Permission::read));
            } else if (operation == 1) {
                static_cast<void>(model->unmap(3, 4));
            } else {
                static_cast<void>(model->protect(3, 4, Permission::write));
            }
        } catch (const std::bad_alloc&) {
            allocation_failed = true;
        }
        fail_next_allocation = false;
        if (!require(allocation_failed, "The injected allocation failure was not observed.")) {
            return false;
        }
        if (operation == 0) {
            if (!require(model->mapping_count() == 0, "Failed map changed the model.")) {
                return false;
            }
        } else {
            const auto region = model->query(0);
            if (!require(model->mapping_count() == 1 && region && region->start == 0 &&
                         region->size == 10 && region->permissions == Permission::read,
                         "Failed range edit corrupted the original mapping.")) {
                return false;
            }
        }
    }
    return true;
}
} // namespace

int main() {
    if (!permissions_match() || !protection_merges() || !allocation_failures_preserve_state()) {
        return 1;
    }
    std::puts("Permission, canonicalization, and allocation-failure contracts passed.");
    return 0;
}
