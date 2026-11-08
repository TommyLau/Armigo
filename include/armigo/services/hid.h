// ============================================================================
// Armigo Services - Human Interface Device Service
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#pragma once

#include <armigo/horizon/ipc.h>
#include <armigo/horizon/shared_memory.h>
#include <armigo/horizon/event.h>

#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <vector>

namespace armigo::services::hid {

class AppletResourceSession;

// Host controller input for player one, as the console's HID service would
// sample it: NpadButton bits, and stick positions in [-32767, 32767] with up
// and right positive. Thread-safe; the latest value wins. Once a host has set
// input, the applet resource publishes it every 5 ms while the application's
// FullKey controller is connected. Headless runs that never set input keep the
// neutral controller.
struct HostFullKeyInput {
    std::uint64_t buttons = 0;
    std::int32_t left_x = 0, left_y = 0, right_x = 0, right_y = 0;
    // The controller's motion in the Pro Controller's frame (x right, y
    // forward, z up out of the face): acceleration in G, about (0, 0, -1) at
    // rest face up, and angular velocity in revolutions per second. Without
    // motion the six-axis sensor reports a controller at rest.
    bool has_motion = false;
    bool has_gyroscope = false;  // with motion: whether angular_velocity is measured
    std::array<float, 3> acceleration{0.0f, 0.0f, -1.0f};
    std::array<float, 3> angular_velocity{};
};
void set_host_full_key_input(const HostFullKeyInput& input) noexcept;
// Sets the host input and writes it to every applet resource's shared memory
// now rather than at its sampler's next 5 ms tick, so a test driver that steps
// input with the guest's frames makes each state visible before the guest's
// next read (guest-present input scripts).
void publish_host_full_key_input_now(const HostFullKeyInput& input) noexcept;

// One vibration value as the guest sends it: amplitudes in [0, 1] and
// frequencies in Hz for the low and high bands. Neutral is {0, 160, 0, 320}.
struct VibrationValue {
    float amplitude_low = 0.0f;
    float frequency_low = 160.0f;
    float amplitude_high = 0.0f;
    float frequency_high = 320.0f;
};
// Receives every vibration value the guest sends to player one's FullKey
// controller, after PermitVibration: device 0 is the left motor and 1 the
// right. Called on the HID service's thread; it must not block. Without a
// sink (headless) values are applied to the virtual controller's motors but
// reach no host actuator. Thread-safe.
void set_host_vibration_sink(std::function<void(unsigned device, const VibrationValue& value)> sink) noexcept;

// Logical sampling state for the single registered resource and the manager's
// fallback. This does not own a physical sensor or transfer-memory allocation.
struct SevenSixAxisSamplingState {
    bool resource_running = false;
    bool fallback_running = false;
};

// The `hid` service the application opens during start-up. Command 0 creates the
// applet resource, which is the session the application keeps and later reads input
// state from. The npad configuration commands the game sends afterwards are served
// here. A neutral FullKey controller occupies player one when the application's
// supported styles and IDs permit it. Unknown commands remain explicit errors.
class HidSession final : public horizon::ClientSession {
public:
    // Initial logical flags can be supplied by the host; construction does not
    // start a sensor. Normal launches begin with both flags clear.
    // A Switch 2 session also serves the Switch 2 vibration command (214).
    explicit HidSession(std::uint64_t process_id,
                        SevenSixAxisSamplingState initial_sampling_state = {}, bool switch2 = false);
    horizon::IpcReply request(const horizon::IpcMessage& message,
                              std::uint64_t sender_pid) override;
    horizon::IpcReply request_with_buffers(const horizon::IpcMessage& message,
                                           std::uint64_t sender_pid,
                                           const horizon::GuestBuffers& buffers) override;
    horizon::IpcReply request_with_handles(
        const horizon::IpcMessage& message, std::uint64_t sender_pid,
        const horizon::GuestBuffers& buffers,
        std::span<const std::shared_ptr<horizon::KernelObject>> copied_objects) override;
    bool receives_copied_objects() const noexcept override { return true; }
    std::size_t domain_object_count() const;
    const char* diagnostic() const;
    std::uint32_t supported_styles() const noexcept {
        const std::lock_guard lock(request_mutex_);
        return supported_styles_;
    }
    std::uint32_t revision() const noexcept {
        const std::lock_guard lock(request_mutex_);
        return revision_;
    }
    bool activated() const noexcept {
        const std::lock_guard lock(request_mutex_);
        return activated_;
    }
    bool touch_activated() const noexcept {
        const std::lock_guard lock(request_mutex_);
        return touch_activated_;
    }
    bool mouse_activated() const noexcept {
        const std::lock_guard lock(request_mutex_);
        return mouse_activated_;
    }
    bool keyboard_activated() const noexcept {
        const std::lock_guard lock(request_mutex_);
        return keyboard_activated_;
    }
    std::uint32_t mouse_library_version() const noexcept {
        const std::lock_guard lock(request_mutex_);
        return mouse_library_version_;
    }
    SevenSixAxisSamplingState seven_six_axis_sampling_state() const noexcept {
        const std::lock_guard lock(request_mutex_);
        return seven_six_axis_sampling_state_;
    }
    std::uint32_t npad_joy_hold_type() const noexcept {
        const std::lock_guard lock(request_mutex_);
        return npad_joy_hold_type_;
    }
    std::uint32_t npad_handheld_activation_mode() const noexcept {
        const std::lock_guard lock(request_mutex_);
        return npad_handheld_activation_mode_;
    }

private:
    horizon::IpcReply handle(const horizon::IpcMessage& message, std::uint64_t sender_pid,
                             const horizon::GuestBuffers* buffers,
                             std::span<const std::shared_ptr<horizon::KernelObject>> copied_objects = {});
    horizon::IpcReply unsupported(const char* detail);
    bool accepts_aruid(std::uint64_t aruid) const noexcept;
    void publish_input();

    std::uint64_t process_id_;
    mutable std::mutex request_mutex_;
    std::uint64_t aruid_ = 0;
    bool resource_registered_ = false;
    SevenSixAxisSamplingState seven_six_axis_sampling_state_;
    std::array<std::shared_ptr<horizon::TransferMemoryRecord>, 2> seven_axis_memories_{};
    std::chrono::steady_clock::time_point seven_axis_timestamp_origin_{};
    std::uint64_t seven_axis_active_aruid_ = 0;
    std::uint32_t supported_styles_ = 0;
    std::uint32_t revision_ = 0;
    bool activated_ = false;
    bool touch_activated_ = false;
    bool mouse_activated_ = false;
    bool keyboard_activated_ = false;
    std::uint32_t mouse_library_version_ = 0;
    std::uint32_t npad_joy_hold_type_ = 0;
    std::uint32_t npad_handheld_activation_mode_ = 0;
    std::uint32_t npad_single_device_type_ = 0;
    std::vector<std::uint32_t> supported_ids_;
    std::shared_ptr<AppletResourceSession> resource_;
    std::array<std::shared_ptr<horizon::Event>, 10> style_events_{};
    struct SixAxisState {
        bool running = false;
        std::uint32_t drift_mode = 1;  // Standard until the guest sets one
    };
    // Applies a vibration value to player one's FullKey motor the handle names
    // (neutral while vibration is not permitted) and forwards it to the host.
    void send_vibration(std::uint32_t handle, const VibrationValue& value);
    // The value GetActualVibrationValue reports for the handle.
    VibrationValue actual_vibration(std::uint32_t handle) const;
    std::unordered_map<std::uint32_t, SixAxisState> six_axis_;
    std::unordered_map<std::uint32_t, SixAxisState> fallback_six_axis_;
    bool full_key_connected_ = false;
    bool vibration_permitted_ = true;
    bool switch2_ = false;
    bool switch2_vibration_warned_ = false;
    // The last value sent to each FullKey motor (left, right).
    std::array<VibrationValue, 2> vibration_values_{};
    const char* diagnostic_ = nullptr;
};

// The applet resource the application keeps. It owns the read-write shared
// memory the application maps to read controller state; command 0 hands that
// memory to the caller, and the remaining commands stay explicit unsupported
// operations until the run reaches them.
class AppletResourceSession final : public horizon::ClientSession {
public:
    explicit AppletResourceSession(std::uint64_t process_id);
    ~AppletResourceSession() override;
    horizon::IpcReply request(const horizon::IpcMessage& message,
                              std::uint64_t sender_pid) override;
    const char* diagnostic() const;
    // The shared memory command 0 transfers. The service writes controller state into it and
    // the application's mapping aliases the same bytes.
    const std::shared_ptr<horizon::SharedMemory>& shared_memory() const noexcept {
        return shared_memory_;
    }
    // Serializes every writer of the shared memory's controller state.
    std::mutex& write_mutex() noexcept { return write_mutex_; }
    // Player one's FullKey six-axis sensor: while running, the sampler writes
    // one sample every 5 ms (200 Hz) from the host's motion, or a controller
    // at rest without one. The zero-drift mode sets the gyroscope dead band.
    void set_six_axis(bool running, std::uint32_t zero_drift_mode) noexcept;

private:
    std::uint64_t process_id_;
    mutable std::mutex request_mutex_;
    std::shared_ptr<horizon::SharedMemory> shared_memory_;
    const char* diagnostic_ = nullptr;
    std::mutex write_mutex_;
    std::atomic<bool> six_axis_running_{false};
    std::atomic<std::uint32_t> six_axis_drift_mode_{1};
    // Publishes host input and six-axis samples every 5 ms (see
    // set_host_full_key_input and set_six_axis).
    std::jthread sampler_;
};

}  // namespace armigo::services::hid
