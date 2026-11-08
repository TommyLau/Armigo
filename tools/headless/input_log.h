// ============================================================================
// Armigo Tools - Development and Verification Tools
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#pragma once

#include "input_script.h"
#include <armigo/runtime/guest_clock.h>
#include <armigo/runtime/host_controller.h>
#include <memory>
#include <mutex>
#include <ostream>
#include <vector>

namespace armigo::tools {

// Logs acceptance by the HID host-input setter, not a claim that the guest
// already sampled the shared-memory controller state. Owns no guest memory.
class InputDeliveryLog {
public:
    static constexpr std::size_t maximum_pending = 4096;
    InputDeliveryLog(std::shared_ptr<runtime::GuestClock> clock, runtime::ClockInstant epoch,
                     InputScript::Clock script_clock, std::unique_ptr<std::ostream> output,
                     std::size_t capacity = maximum_pending)
        : clock_(std::move(clock)), epoch_(epoch), script_clock_(script_clock), output_(std::move(output)),
          capacity_(capacity) {
        if (!clock_ || !output_ || !capacity_ || capacity_ > maximum_pending) {
            latch("Invalid scripted input delivery log configuration.");
            return;
        }
        records_.reserve(capacity_);
        *output_ << "elapsed_ns\tinput_ns\tscript_index\tscript_position\tscript_clock"
                    "\tbuttons\tleft_x\tleft_y\tright_x\tright_y\n";
        output_->flush();
        if (!*output_) latch("Could not initialize scripted input delivery log.");
    }
    bool ready() const {
        const std::lock_guard lock(mutex_);
        return !failure_;
    }
    platform::PresenterStatus record(std::size_t index, const ScriptInputState& state) {
        const std::lock_guard lock(mutex_);
        if (failure_) return *failure_;
        if (records_.size() == capacity_)
            return latch("Scripted input delivery log capacity exceeded (" + std::to_string(capacity_) + " pending records).");
        const auto sampled = clock_->sample();
        if (const auto* error = std::get_if<runtime::LaunchFailure>(&sampled)) return latch(error->detail);
        // Capacity is reserved before delivery: no formatting, stream access
        // or vector allocation takes place between due HID setter calls.
        records_.push_back({std::get<runtime::ClockSample>(sampled).instant, index, state.frame,
                            runtime::full_key_input(state.input)});
        return std::monostate{};
    }
    platform::PresenterStatus flush() {
        const std::lock_guard lock(mutex_);
        // Flush the accepted prefix even after overflow, preserving its first
        // failure. The overflowing state was applied but cannot be logged;
        // the presenter reports failure instead of accepting a truncated log.
        for (const auto& record : records_) {
            *output_ << (record.instant - epoch_).count() << '\t' << record.instant.time_since_epoch().count()
                     << '\t' << record.index << '\t' << record.position << '\t'
                     << input_clock_name(script_clock_)
                     << '\t' << record.input.buttons << '\t' << record.input.left_x << '\t' << record.input.left_y
                     << '\t' << record.input.right_x << '\t' << record.input.right_y << '\n';
        }
        if (!records_.empty()) {
            output_->flush();
            if (!*output_) latch("Could not write scripted input delivery log.");
            records_.clear();
        }
        if (failure_) return *failure_;
        return std::monostate{};
    }
private:
    platform::PresentationFailure latch(std::string detail) {
        if (!failure_) failure_ = platform::PresentationFailure{platform::PresentationError::host_failure, std::move(detail)};
        return *failure_;
    }
    struct Record {
        runtime::ClockInstant instant;
        std::size_t index;
        std::uint64_t position;
        services::hid::HostFullKeyInput input;
    };
    std::shared_ptr<runtime::GuestClock> clock_;
    runtime::ClockInstant epoch_;
    InputScript::Clock script_clock_;
    std::unique_ptr<std::ostream> output_;
    std::size_t capacity_;
    std::vector<Record> records_;
    std::optional<platform::PresentationFailure> failure_;
    mutable std::mutex mutex_;
};

} // namespace armigo::tools
