// ============================================================================
// Armigo Audio - Horizon Audio Renderer
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#pragma once

#include <memory>

#include "renderer/common/common.h"
#include "renderer/common/types.h"

namespace armigo::audio::renderer {
// A memory pool: a guest range the renderer may access while it is attached.
class MemoryPoolInfo {
public:
    // The location of this pool.
    // CPU pools are guest memory pinned through the pool mapper.
    // DSP pools are mapped in the current process sysmodule.
    enum class Location {
        CPU = 1,
        DSP = 2,
    };

    // Current state of the pool
    enum class State {
        Invalid,
        Acquired,
        RequestDetach,
        Detached,
        RequestAttach,
        Attached,
        Released,
    };

    // Result code for updating the pool (See InfoUpdater::Update)
    enum class ResultState {
        Success,
        BadParam,
        MapFailed,
        InUse,
    };

    // Input parameters coming from the game which are used to update current pools
    // (See InfoUpdater::Update)
    struct InParameter {
        /* 0x00 */ u64 address;
        /* 0x08 */ u64 size;
        /* 0x10 */ State state;
        /* 0x14 */ bool in_use;
        /* 0x18 */ char unk18[0x8];
    };
    static_assert(sizeof(InParameter) == 0x20, "MemoryPoolInfo::InParameter has the wrong size!");

    // Output status sent back to the game on update (See InfoUpdater::Update)
    struct OutStatus {
        /* 0x00 */ State state;
        /* 0x04 */ char unk04[0xC];
    };
    static_assert(sizeof(OutStatus) == 0x10, "MemoryPoolInfo::OutStatus has the wrong size!");

    MemoryPoolInfo() = default;
    MemoryPoolInfo(Location location_) : location{location_} {}

    // Get the CPU address for this pool.
    CpuAddr get_cpu_address() const;

    // Get the DSP address for this pool.
    CpuAddr get_dsp_address() const;

    // Get the size of this pool.
    u64 get_size() const;

    // Get the location of this pool.
    Location get_location() const;

    // Set the CPU address for this pool.
    void set_cpu_address(CpuAddr address, u64 size);

    // Set the DSP address for this pool.
    void set_dsp_address(CpuAddr address);

    // Check whether the pool contains a given range.
    bool contains(CpuAddr address, u64 size) const;

    // Check whether this pool is mapped, which is when the dsp address is set.
    bool is_mapped() const;

    // Translates a given CPU range into a relative offset for the DSP.
    CpuAddr translate(CpuAddr address, u64 size) const;

    // Set or unset whether this memory pool is in use.
    void set_used(bool used);

    // Get whether this pool is in use.
    bool is_used() const;

private:
    // Base address for the CPU-side memory
    CpuAddr cpu_address{};
    // Base address for the DSP-side memory
    CpuAddr dsp_address{};
    // Size of this pool
    u64 size{};
    // Location of this pool, either CPU or DSP
    Location location{Location::DSP};
    // If this pool is in use
    bool in_use{};
};

} // namespace armigo::audio::renderer
