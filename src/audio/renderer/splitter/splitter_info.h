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

#include "renderer/splitter/splitter_destinations_data.h"
#include "renderer/common/types.h"

namespace armigo::audio::renderer {
// Represents a splitter, wraps multiple output destinations to split an input mix into.
class SplitterInfo {
public:
    struct InParameter {
        /* 0x00 */ u32 magic; // 'SNDI'
        /* 0x04 */ s32 id;
        /* 0x08 */ u32 sample_rate;
        /* 0x0C */ u32 destination_count;
    };
    static_assert(sizeof(InParameter) == 0x10, "SplitterInfo::InParameter has the wrong size!");

    explicit SplitterInfo(s32 id);

    // Initialize the given splitters.
    static void initialize_infos(SplitterInfo* splitters, u32 count);

    // Update this splitter.
    u32 update(const InParameter* params);

    // Get a destination in this splitter.
    SplitterDestinationData* get_data(u32 id);

    // Get the number of destinations in this splitter.
    u32 get_destination_count() const;

    // Set the number of destinations in this splitter.
    void set_destination_count(u32 count);

    // Check if the splitter has a new connection.
    bool has_new_connection() const;

    // Reset the new connection flag.
    void clear_new_connection_flag();

    // Mark as having a new connection.
    void set_new_connection_flag();

    // Update the state of all destinations.
    void update_internal_state();

    // Set this splitter's destinations.
    void set_destinations(SplitterDestinationData* destinations);

private:
    // Id of this splitter
    s32 id;
    // Sample rate of this splitter
    u32 sample_rate{};
    // Number of destinations in this splitter
    u32 destination_count{};
    // Does this splitter have a new connection?
    bool new_connection{true};
    // Pointer to the destinations of this splitter
    SplitterDestinationData* destinations{};
    // Number of channels this splitter manages
    [[maybe_unused]] u32 channel_count{};
};

} // namespace armigo::audio::renderer
