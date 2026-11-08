// ============================================================================
// Armigo Services - Switch 2 Audio Device Manager
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

// The Switch 2 "auddmg:u" service and the virtual output device manager its
// command 0 hands out (nn::audio's virtual output devices), following the
// practical reference. Armigo plays through one stereo 48 kHz output, so there
// is one virtual device, "AudioVirtualOutput0", on the physical output
// "AudioTvOutput". The manager lists it (0), names it as the default (100),
// answers whether a stream format is supported (300), returns its property
// block (401) and an event for its state changes, which never fire (500).
// Attaching and detaching sound-mixer streams (700, 800) need the sound mixer
// service and, like any other request, are unsupported with an explicit
// diagnostic.

#pragma once

#include <armigo/horizon/event.h>
#include <armigo/horizon/ipc.h>

#include <array>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>

namespace armigo::services::auddmg {

// Results of the audio module (153).
constexpr std::uint32_t result(std::uint32_t description) noexcept { return 153u | description << 9; }
constexpr std::uint32_t result_not_supported = result(513);

// Property block of a virtual output device (command 401).
inline constexpr std::size_t device_name_size = 0x100;
inline constexpr std::size_t device_properties_size = 0x228;

class VirtualDeviceManagerSession final : public horizon::ClientSession {
public:
    explicit VirtualDeviceManagerSession(std::uint64_t process_id);
    horizon::IpcReply request(const horizon::IpcMessage&, std::uint64_t sender_pid) override;
    horizon::IpcReply request_with_buffers(const horizon::IpcMessage&, std::uint64_t sender_pid,
                                           const horizon::GuestBuffers& buffers) override;
    const char* diagnostic() const;

private:
    horizon::IpcReply dispatch(const horizon::IpcMessage&, std::uint64_t sender_pid, const horizon::GuestBuffers*);
    horizon::IpcReply unsupported(const char* detail);
    const std::uint64_t process_id_;
    mutable std::mutex mutex_;
    std::shared_ptr<horizon::Event> state_event_;
    const char* diagnostic_ = nullptr;
};

class DeviceManagerSession final : public horizon::ClientSession {
public:
    explicit DeviceManagerSession(std::uint64_t process_id);
    horizon::IpcReply request(const horizon::IpcMessage&, std::uint64_t sender_pid) override;
    const char* diagnostic() const;
    // The 16 bytes of the latest command 0, kept as given (their meaning is
    // not established).
    std::optional<std::array<std::byte, 16>> open_argument() const;

private:
    horizon::IpcReply unsupported(const char* detail);
    const std::uint64_t process_id_;
    mutable std::mutex mutex_;
    std::optional<std::array<std::byte, 16>> argument_;
    const char* diagnostic_ = nullptr;
};

} // namespace armigo::services::auddmg
