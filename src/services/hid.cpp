// ============================================================================
// Armigo Services - Human Interface Device Service
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include <armigo/services/hid.h>

#include "hid_motion.h"
#include "hid_shared_memory.h"

#include <algorithm>
#include <vector>
#include <cstdio>
#include <array>
#include <atomic>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstring>
#include <optional>
#include <limits>

namespace armigo::services::hid {
namespace {
using horizon::IpcDisposition;
using horizon::IpcMessage;
using horizon::IpcReply;

constexpr std::uint32_t sfci = 0x49434653;
constexpr std::uint32_t sfco = 0x4f434653;
// Host pointer-transfer capacity for Npad IDs and batched vibration parameters.
constexpr std::uint16_t session_pointer_capacity = 0x400;

std::uint32_t read(const IpcMessage& message, std::size_t offset, unsigned width = 4) {
    std::uint32_t value = 0;
    for (unsigned index = 0; index < width; ++index)
        value |= std::uint32_t{std::to_integer<unsigned char>(message[offset + index])}
                 << (8 * index);
    return value;
}

std::uint64_t read64(const IpcMessage& message, std::size_t offset) {
    return std::uint64_t{read(message, offset)} | (std::uint64_t{read(message, offset + 4)} << 32);
}

void write(IpcMessage& message, std::size_t offset, std::uint32_t value, unsigned width = 4) {
    for (unsigned index = 0; index < width; ++index)
        message[offset + index] = static_cast<std::byte>((value >> (8 * index)) & 255);
}

IpcReply control_reply(std::uint32_t value, unsigned width) {
    IpcReply reply;
    reply.disposition = IpcDisposition::handled;
    reply.size = 44;
    write(reply.bytes, 4, 9);
    write(reply.bytes, 16, sfco);
    write(reply.bytes, 32, value, width);
    return reply;
}

// A CMIF reply with no output parameters. The HIPC data word count excludes the two
// header words; a zero count makes the client reject the missing output header.
IpcReply success_reply(std::uint32_t result = 0) {
    IpcReply reply;
    reply.disposition = IpcDisposition::handled;
    reply.size = 40;
    write(reply.bytes, 4, (reply.size - 8) / 4);
    write(reply.bytes, 16, sfco);
    write(reply.bytes, 24, result);
    write(reply.bytes, 28, 0x480dd4ea);
    return reply;
}

// A scalar CMIF output followed by the firmware alignment allowance.
IpcReply value_reply(std::uint64_t value, unsigned width = 4) {
    IpcReply reply = success_reply();
    reply.size += (width + 3) & ~3u;
    write(reply.bytes, 4, (reply.size - 8) / 4);
    write(reply.bytes, 32, static_cast<std::uint32_t>(value), std::min(width, 4u));
    if (width == 8) write(reply.bytes, 36, static_cast<std::uint32_t>(value >> 32));
    return reply;
}

IpcReply seven_six_axis_stop_reply(std::uint32_t result) {
    IpcReply reply{IpcDisposition::handled};
    reply.size = 40;
    write(reply.bytes, 4, 8);
    write(reply.bytes, 16, sfco);
    write(reply.bytes, 24, result);
    write(reply.bytes, 28, 0x480dd4ea);
    return reply;
}

struct RequestShape {
    unsigned input_size;
    bool pid;
    unsigned pointers = 0;
    std::optional<unsigned> aruid_offset;
    unsigned copied_handles = 0;
};

// Only ordinary CMIF and counted input pointer descriptors are supported here.
// The raw word count includes the firmware's sixteen-byte alignment allowance.
bool matches_shape(const IpcMessage& message, std::size_t cmif,
                   const horizon::GuestBuffers* buffers, RequestShape shape) {
    const auto type = read(message, 0, 2);
    if (read(message, 0) != (type | (shape.pointers << 16))) return false;
    const auto raw = read(message, 4);
    if ((raw & ~0x3ffu) != (shape.pid ? 0x80000000u : 0)) return false;
    if (shape.pid && read(message, 8) != (1u | (shape.copied_handles << 1))) return false;
    const auto start = 8u + (shape.pid ? 12u : 0u) + 8u * shape.pointers +
                       4u * shape.copied_handles;
    const auto words = raw & 0x3ff;
    if (cmif != ((start + 15u) & ~15u) ||
        words < (shape.input_size + 32u + 3u) / 4u ||
        start + words * 4u > message.size() ||
        cmif + 16u + shape.input_size > start + words * 4u) return false;
    const auto count = buffers ? buffers->size() : 0;
    if (count != shape.pointers) {
        // The dispatcher omits zero-length pointer descriptors from GuestBuffers.
        if (count != 0) return false;
        for (unsigned index = 0; index < shape.pointers; ++index)
            if (read(message, 8u + (shape.pid ? 12u : 0u) + 8u * index) >> 16)
                return false;
    }
    // The transport currently exposes X pointers as both readable and writable.
    // Their input-only wire role was already established by the HIPC header.
    for (unsigned index = 0; index < count; ++index)
        if (!(*buffers)[index].input) return false;
    return true;
}

std::optional<RequestShape> startup_shape(std::uint32_t command) {
    switch (command) {
    case 0: case 11: case 21: case 31: case 101: case 103: case 104: case 121: case 129:
    case 303: case 304: case 307: case 310:
        return RequestShape{8, true, 0, 0};
    case 100: case 107: case 109: case 122: case 124: case 66: case 67: case 79: case 202: case 3150:
        return RequestShape{16, true, 0, 8};
    case 120: case 128: return RequestShape{16, true, 0, 0};
    case 123: case 106: return RequestShape{24, true, 0, 8};
    case 102: return RequestShape{8, true, 1, 0};
    case 200: return RequestShape{4, false, 0, std::nullopt};
    case 201: return RequestShape{32, true, 0, 24};
    case 203: case 205: return RequestShape{0, false, 0, std::nullopt};
    case 204: return RequestShape{1, false, 0, std::nullopt};
    case 206: return RequestShape{8, false, 2, std::nullopt};
    case 211: return RequestShape{16, true, 0, 8};
    case 214: return RequestShape{16, false, 2, std::nullopt};
    case 306: return RequestShape{24, true, 0, 0, 2};
    default: return std::nullopt;
    }
}

bool valid_npad_id(std::uint32_t id) { return id < 8 || id == 16 || id == 32; }

std::size_t npad_index(std::uint32_t id) { return id < 8 ? id : id == 16 ? 9 : 8; }

std::uint32_t vibration_handle_result(std::uint32_t handle) {
    const auto style = handle & 255;
    if (style > 33 || ((std::uint64_t{1} << style) & 0x3000021f8ull) == 0)
        return 0xf4ca;
    if (!valid_npad_id((handle >> 8) & 255)) return 0xf6ca;
    if (((handle >> 16) & 255) > 1) return 0xf8ca;
    return 0;
}

bool initialize_seven_axis_ring(const horizon::GuestBuffers& memory, std::uint64_t address) {
    std::array<std::byte, 0xa70> ring{};
    if (!memory.read_memory(address, ring)) return false;
    std::fill_n(ring.begin(), 0x20, std::byte{});
    ring[8] = std::byte{33};
    for (std::size_t index = 0; index < 33; ++index) {
        const auto entry = ring.begin() + 0x20 + index * 0x50;
        std::fill_n(entry, 8, std::byte{});
        // The firmware preserves each atomic entry's eight-byte padding word.
        std::fill_n(entry + 0x10, 0x40, std::byte{});
    }
    return memory.write_memory(address, ring);
}

bool write_guest_integer(const horizon::GuestBuffers& memory, std::uint64_t address,
                         std::uint64_t value, unsigned width) {
    std::array<std::byte, 8> bytes{};
    for (unsigned index = 0; index < width; ++index)
        bytes[index] = static_cast<std::byte>((value >> (8 * index)) & 255);
    return memory.write_memory(address, std::span{bytes}.first(width));
}

bool initialize_seven_axis_scratch(const horizon::GuestBuffers& memory, std::uint64_t address) {
    return write_guest_integer(memory, address, 0x7efe0, 8) &&
           write_guest_integer(memory, address + 8, 0, 4) &&
           write_guest_integer(memory, address + 0x7eff0, 0, 8) &&
           write_guest_integer(memory, address + 0x7eff8, 2, 4);
}

class ActiveVibrationDeviceList final : public horizon::ClientSession {
public:
    explicit ActiveVibrationDeviceList(std::uint64_t pid) : process_id_(pid) {}
    IpcReply request(const IpcMessage& message, std::uint64_t sender_pid) override {
        const std::lock_guard lock(mutex_);
        if (sender_pid != process_id_) return {};
        const auto type = read(message, 0, 2);
        if (type == 5 || type == 7) {
            if (read(message, 16) == sfci && read(message, 24) == 3)
                return control_reply(session_pointer_capacity, 2);
            return {};
        }
        if (type != 4 && type != 6) return {};
        const auto cmif = horizon::locate_cmif(message);
        if (!cmif || read(message, *cmif + 8) != 0 ||
            !matches_shape(message, *cmif, nullptr, {4, false, 0, std::nullopt})) return {};
        const auto handle = read(message, *cmif + 16);
        auto result = vibration_handle_result(handle);
        if (result == 0 && std::find(handles_.begin(), handles_.end(), handle) == handles_.end()) {
            if (handles_.size() == 256) result = 0xf8ca;
            else handles_.push_back(handle);
        }
        auto reply = success_reply(result);
        write(reply.bytes, 28, 0x679703ea);
        return reply;
    }
private:
    std::uint64_t process_id_;
    std::mutex mutex_;
    // The virtual device has no motors; ownership still lasts for this list's lifetime.
    std::vector<std::uint32_t> handles_;
};
}  // namespace

HidSession::HidSession(std::uint64_t process_id, SevenSixAxisSamplingState initial_sampling_state, bool switch2)
    : process_id_(process_id), seven_six_axis_sampling_state_(initial_sampling_state), switch2_(switch2) {}

IpcReply HidSession::unsupported(const char* detail) {
    diagnostic_ = detail;
    return {};
}

namespace {

// Never destroyed, like the host input below: a service thread may still
// deliver a value during process exit.
std::mutex& vibration_sink_mutex() {
    static auto* mutex = new std::mutex;
    return *mutex;
}
std::function<void(unsigned, const VibrationValue&)>& vibration_sink() {
    static auto* sink = new std::function<void(unsigned, const VibrationValue&)>;
    return *sink;
}

}  // namespace

void set_host_vibration_sink(std::function<void(unsigned device, const VibrationValue& value)> sink) noexcept {
    const std::lock_guard lock(vibration_sink_mutex());
    vibration_sink() = std::move(sink);
}

void HidSession::send_vibration(std::uint32_t handle, const VibrationValue& value) {
    // Player one's FullKey controller (style index 3, npad 0) has two motors:
    // device 0 is the left, device 1 the right.
    const unsigned device = (handle >> 16) & 255;
    if ((handle & 0xffffu) != 3 || device > 1) return;
    const auto finite = [](float v, float neutral) { return std::isfinite(v) ? v : neutral; };
    const auto applied = vibration_permitted_
        ? VibrationValue{finite(value.amplitude_low, 0.0f), finite(value.frequency_low, 160.0f),
                         finite(value.amplitude_high, 0.0f), finite(value.frequency_high, 320.0f)}
        : VibrationValue{};
    vibration_values_[device] = applied;
    std::function<void(unsigned, const VibrationValue&)> sink;
    {
        const std::lock_guard lock(vibration_sink_mutex());
        sink = vibration_sink();
    }
    if (sink) sink(device, applied);
}

VibrationValue HidSession::actual_vibration(std::uint32_t handle) const {
    // The virtual Pro Controller always has both motors mounted, with or
    // without a host actuator behind them.
    const unsigned device = (handle >> 16) & 255;
    if ((handle & 0xffffu) != 3 || device > 1) return {};
    return vibration_values_[device];
}

bool HidSession::accepts_aruid(std::uint64_t aruid) const noexcept {
    // This runtime hosts one application instance; the applet service reports the process
    // identity as its applet resource id. Any other id names an applet resource this service
    // does not own.
    return aruid == 0 || aruid == process_id_;
}

void HidSession::publish_input() {
    if (!resource_) return;
    const bool connected = activated_ && (supported_styles_ & 1) != 0 &&
        std::find(supported_ids_.begin(), supported_ids_.end(), 0) != supported_ids_.end();
    const auto sensor = six_axis_.find(3);
    resource_->set_six_axis(sensor != six_axis_.end() && sensor->second.running,
                            sensor != six_axis_.end() ? sensor->second.drift_mode : 1);
    const std::lock_guard write(resource_->write_mutex());
    resource_->shared_memory()->update([&](std::span<std::byte> bytes) {
        detail::publish_headless_full_key(bytes, connected);
    });
    if (connected != full_key_connected_ && style_events_[0]) style_events_[0]->signal();
    full_key_connected_ = connected;
}

IpcReply HidSession::request(const IpcMessage& message, std::uint64_t sender_pid) {
    return handle(message, sender_pid, nullptr);
}

IpcReply HidSession::request_with_buffers(const IpcMessage& message, std::uint64_t sender_pid,
                                          const horizon::GuestBuffers& buffers) {
    return handle(message, sender_pid, &buffers);
}

IpcReply HidSession::request_with_handles(
    const IpcMessage& message, std::uint64_t sender_pid, const horizon::GuestBuffers& buffers,
    std::span<const std::shared_ptr<horizon::KernelObject>> copied_objects) {
    return handle(message, sender_pid, &buffers, copied_objects);
}

IpcReply HidSession::handle(const IpcMessage& message, std::uint64_t sender_pid,
                            const horizon::GuestBuffers* buffers,
                            std::span<const std::shared_ptr<horizon::KernelObject>> copied_objects) {
    const std::lock_guard lock(request_mutex_);
    diagnostic_ = nullptr;
    if (sender_pid != process_id_) return unsupported("HID session belongs to another process");
    const auto type = read(message, 0, 2);
    if (type == 5 || type == 7) {
        if (read(message, 16) != sfci) return unsupported("HID control magic is unsupported");
        const auto command = read(message, 24);
        if (command == 3) return control_reply(session_pointer_capacity, 2);
        if (command == 0) return control_reply(1, 4);
        return unsupported("HID control command is unsupported");
    }
    if (type != 4 && type != 6) return unsupported("HID request type is unsupported");
    const auto located = horizon::locate_cmif(message);
    if (!located) return unsupported("HID request shape is unsupported");
    const auto cmif = *located;
    const auto command = read(message, cmif + 8);
    if (const auto shape = startup_shape(command)) {
        if (!matches_shape(message, cmif, buffers, *shape))
            return unsupported("HID startup request shape is unsupported");
        if (copied_objects.size() != shape->copied_handles)
            return unsupported("HID request has no validated copied objects");
        if (shape->aruid_offset) {
            const auto aruid = read64(message, cmif + 16 + *shape->aruid_offset);
            if (!accepts_aruid(aruid)) return success_reply(0x60a);
            if (command != 0 && command != 66 && command != 67 && command != 79 &&
                command != 201 && command != 202 &&
                command != 303 && command != 304 && command != 307 && command != 310 &&
                (!resource_registered_ || aruid != aruid_)) {
                if (command == 306)
                    return unsupported("HID seven-axis initialization needs a registered resource");
                return success_reply(0x58cca);
            }
        }
    }
    switch (command) {
    case 305: {
        // This slice supports the ordinary send-PID form with no descriptors or
        // handles. Its raw extent includes the CMIF alignment allowance.
        const auto header = read(message, 0);
        const auto raw_header = read(message, 4);
        const auto words = raw_header & 0x3ff;
        if ((header != 4 && header != 6) || cmif != 32 ||
            (raw_header & ~0x3ffu) != 0x80000000u || read(message, 8) != 1 ||
            words < 10 || words > (message.size() - 20) / 4 ||
            (buffers != nullptr && !buffers->empty())) {
            return unsupported("Seven-six-axis stop request shape is unsupported");
        }
        const auto aruid = read64(message, 48);
        if (aruid != 0 && aruid != sender_pid) return seven_six_axis_stop_reply(0x60a);
        // Zero is passed through to lookup, not replaced with the trusted PID.
        // A missing registration selects the separate manager fallback state.
        auto& running = resource_registered_ && aruid == aruid_
            ? seven_six_axis_sampling_state_.resource_running
            : seven_six_axis_sampling_state_.fallback_running;
        running = false;
        return seven_six_axis_stop_reply(0);
    }
    case 303:
        seven_axis_active_aruid_ = read64(message, cmif + 16);
        return success_reply();
    case 304: {
        const auto aruid = read64(message, cmif + 16);
        auto& running = resource_registered_ && aruid == aruid_
            ? seven_six_axis_sampling_state_.resource_running
            : seven_six_axis_sampling_state_.fallback_running;
        if (!running) seven_axis_timestamp_origin_ = std::chrono::steady_clock::now();
        running = true;
        return success_reply();
    }
    case 306: {
        if (seven_axis_memories_[0] || seven_axis_memories_[1])
            return unsupported("HID seven-axis reinitialization requires finalization");
        constexpr std::array<std::uint64_t, 2> sizes{0x1000, 0x7f000};
        constexpr std::array<std::uint32_t, 2> permissions{1, 0};
        if (read64(message, cmif + 24) != sizes[0] || read64(message, cmif + 32) != sizes[1])
            return unsupported("HID seven-axis transfer-memory sizes are unsupported");
        for (std::size_t index = 0; index < 2; ++index) {
            const auto object = std::dynamic_pointer_cast<horizon::TransferMemoryRecord>(
                copied_objects[index]);
            if (!object || object->permission() != permissions[index] ||
                object->range().size != sizes[index] ||
                object->range().address % 0x1000 != 0 ||
                object->range().address > std::numeric_limits<std::uint64_t>::max() - sizes[index])
                return success_reply(0x878ca);
            const auto address = object->range().address;
            const bool initialized = index == 0
                ? initialize_seven_axis_ring(*buffers, address)
                : initialize_seven_axis_scratch(*buffers, address);
            if (!initialized) return success_reply(0x878ca);
            // Initialization is sequential: failure of the second mapping leaves
            // the first initialized and retained, as on the firmware path.
            seven_axis_memories_[index] = object;
        }
        return success_reply();
    }
    case 307: {
        const auto aruid = read64(message, cmif + 16);
        auto& running = resource_registered_ && aruid == aruid_
            ? seven_six_axis_sampling_state_.resource_running
            : seven_six_axis_sampling_state_.fallback_running;
        running = false;
        if (resource_registered_ && aruid == aruid_) {
            if (seven_axis_memories_[1]) {
                if (!buffers) return unsupported("HID seven-axis finalization needs memory access");
                std::vector<std::byte> cleared(0x7efe0);
                if (!buffers->write_memory(seven_axis_memories_[1]->range().address + 0x10, cleared))
                    return unsupported("HID seven-axis scratch memory is no longer writable");
            }
            seven_axis_memories_[1].reset();
            seven_axis_memories_[0].reset();
        }
        return success_reply();
    }
    case 310:
        seven_axis_timestamp_origin_ = std::chrono::steady_clock::now();
        return success_reply();
    case 0: {
        // CreateAppletResource(aruid): the application keeps the applet resource as a session
        // of its own, so the reply transfers one that answers the resource commands.
        const auto aruid = read64(message, cmif + 16);
        if (resource_registered_ && aruid != aruid_)
            return unsupported("HID supports one registered applet resource");
        aruid_ = aruid;
        if (!resource_) resource_ = std::make_shared<AppletResourceSession>(process_id_);
        auto reply = success_reply();
        reply.transferred[0] = resource_;
        resource_registered_ = true;
        reply.transferred_count = 1;
        return reply;
    }
    case 100: {
        // SetSupportedNpadStyleSet(style, aruid): the application declares the controller
        // styles it wants reported. FullKey is presented only when requested.
        const auto style = read(message, cmif + 16);
        if (!accepts_aruid(read64(message, cmif + 24)))
            return unsupported("HID command names another applet resource");
        supported_styles_ = style;
        publish_input();
        return success_reply();
    }
    case 101:
        return value_reply(supported_styles_);
    case 102: {
        // SetSupportedNpadIdType(aruid, ids[]): the ids the application wants reported.
        if (!accepts_aruid(read64(message, cmif + 16)))
            return unsupported("HID command names another applet resource");
        std::vector<std::uint32_t> ids;
        if (buffers != nullptr && !buffers->empty()) {
            std::array<std::byte, 40> raw{};
            const auto size = (*buffers)[0].size;
            if (size / 4 > 10) return success_reply(0x596ca);
            if (size % 4 != 0) return unsupported("HID supported-id list has a partial element");
            if (!buffers->read(0, std::span<std::byte>(raw.data(), size)))
                return unsupported("HID supported-id list is not readable");
            for (std::uint32_t offset = 0; offset + 4 <= size; offset += 4) {
                std::uint32_t id = 0;
                for (unsigned index = 0; index < 4; ++index)
                    id |= std::uint32_t{std::to_integer<unsigned char>(raw[offset + index])} << (8 * index);
                ids.push_back(id);
            }
        }
        supported_ids_ = std::move(ids);
        publish_input();
        return success_reply();
    }
    case 103:
        revision_ = 0;
        activated_ = true;
        publish_input();
        return success_reply();
    case 104:
        activated_ = false;
        publish_input();
        return success_reply();
    case 106: {
        const auto index = npad_index(read(message, cmif + 16));
        auto& event = style_events_[index];
        if (!event) event = std::make_shared<horizon::Event>();
        // Firmware signals on acquisition, including reacquisition of an existing event.
        event->signal();
        auto reply = success_reply();
        reply.transferred[0] = event;
        reply.transferred_count = 1;
        reply.transferred_as_copy = true;
        return reply;
    }
    case 11: {
        // ActivateTouchScreen(aruid): the host has no touch panel, so the touch
        // section stays without touches; the activation is kept.
        if (!accepts_aruid(read64(message, cmif + 16)))
            return unsupported("HID command names another applet resource");
        touch_activated_ = true;
        return success_reply();
    }
    case 21: {
        // ActivateMouse(aruid): the host reports no mouse, so the mouse section
        // stays without samples; the activation is kept.
        if (!accepts_aruid(read64(message, cmif + 16)))
            return unsupported("HID command names another applet resource");
        mouse_activated_ = true;
        return success_reply();
    }
    case 109: {
        // ActivateNpadWithRevision(revision, aruid): the same activation with the npad
        // revision used by the firmware's supported-style policy.
        const auto revision = read(message, cmif + 16);
        if (!accepts_aruid(read64(message, cmif + 24)))
            return unsupported("HID command names another applet resource");
        revision_ = revision;
        activated_ = true;
        publish_input();
        return success_reply();
    }
    case 31: {
        // ActivateKeyboard(aruid): the keyboard section stays without samples;
        // the activation is kept.
        if (!accepts_aruid(read64(message, cmif + 16)))
            return unsupported("HID command names another applet resource");
        keyboard_activated_ = true;
        return success_reply();
    }
    case 3150: {
        // SetMouseLibraryVersion(version, aruid): the mouse library the application
        // links; the mouse section does not depend on it here.
        const auto version = read(message, cmif + 16);
        if (!accepts_aruid(read64(message, cmif + 24)))
            return unsupported("HID command names another applet resource");
        mouse_library_version_ = version;
        return success_reply();
    }
    case 120: {
        // SetNpadJoyHoldType(aruid, type): orientation applies to detachable Joy-Con.
        if (!accepts_aruid(read64(message, cmif + 16)))
            return unsupported("HID command names another applet resource");
        const auto hold = read64(message, cmif + 24);
        if (hold > 1) return unsupported("HID Joy-Con hold type is unsupported");
        npad_joy_hold_type_ = static_cast<std::uint32_t>(hold);
        return success_reply();
    }
    case 121: {
        // GetNpadJoyHoldType(aruid)
        if (!accepts_aruid(read64(message, cmif + 16)))
            return unsupported("HID command names another applet resource");
        return value_reply(npad_joy_hold_type_, 8);
    }
    case 122:  // SetNpadJoyAssignmentModeSingleByDefault(npad_id, aruid)
    case 124:  // SetNpadJoyAssignmentModeDual(npad_id, aruid)
        // The wire order is the npad id followed by the applet resource user id
        // for these two assignment commands.
        if (!accepts_aruid(read64(message, cmif + 24)))
            return unsupported("HID command names another applet resource");
        return success_reply();
    case 125:  // MergeSingleJoyAsDualJoy
        // No physical Joy-Con is attached; the merge is a no-op.
        return success_reply();
    case 126:  // StartLrAssignmentMode
    case 127:  // StopLrAssignmentMode
    case 130:  // SwapNpadAssignment
        if (!accepts_aruid(read64(message, cmif + 16)))
            return unsupported("HID command names another applet resource");
        return success_reply();
    case 123: {
        // SetNpadJoyAssignmentModeSingle(id, aruid, device_type).
        if (!accepts_aruid(read64(message, cmif + 24)))
            return unsupported("HID command names another applet resource");
        const auto device = read64(message, cmif + 32);
        if (device > 1) return unsupported("HID Joy-Con device type is unsupported");
        npad_single_device_type_ = static_cast<std::uint32_t>(device);
        return success_reply();
    }
    case 128: {
        // SetNpadHandheldActivationMode(aruid, mode)
        if (!accepts_aruid(read64(message, cmif + 16)))
            return unsupported("HID command names another applet resource");
        const auto mode = read64(message, cmif + 24);
        if (mode > 2) return unsupported("HID handheld activation mode is unsupported");
        npad_handheld_activation_mode_ = static_cast<std::uint32_t>(mode);
        return success_reply();
    }
    case 129: {
        // GetNpadHandheldActivationMode(aruid)
        if (!accepts_aruid(read64(message, cmif + 16)))
            return unsupported("HID command names another applet resource");
        return value_reply(npad_handheld_activation_mode_, 8);
    }
    case 66:
    case 67:
    case 79: {
        const auto handle = read(message, cmif + 16);
        // A handle names a style index, a player ID and a device index. Like
        // firmware, a valid handle keeps its sensor state whether or not that
        // controller is connected; only the presented FullKey controller
        // (style index 3, player ID 0) publishes samples. Its firmware resolver
        // ignores the device-index byte (the SDK normally supplies index 2).
        if (!valid_npad_id((handle >> 8) & 255) || ((handle >> 16) & 255) > 2)
            return unsupported("HID six-axis handle names no controller device");
        const auto aruid = read64(message, cmif + 24);
        auto& states = resource_registered_ && aruid == aruid_ ? six_axis_ : fallback_six_axis_;
        auto& sensor = states[handle & 0xffffu];
        if (command == 79) sensor.drift_mode = read(message, cmif + 20);
        else sensor.running = command == 66;
        publish_input();
        return success_reply();
    }
    case 200: {
        const auto handle = read(message, cmif + 16);
        if (const auto error = vibration_handle_result(handle)) return success_reply(error);
        const auto style = handle & 255;
        const std::uint32_t kind = style == 8 ? 2 : style == 13 ? 3 : 1;
        const std::uint32_t position = kind == 1 ? 1 + ((handle >> 16) & 255) : 0;
        return value_reply(std::uint64_t{kind} | (std::uint64_t{position} << 32), 8);
    }
    case 201:
    case 202: {
        const auto handle = read(message, cmif + 16);
        if (const auto error = vibration_handle_result(handle)) return success_reply(error);
        if (command == 201) {
            send_vibration(handle, VibrationValue{
                std::bit_cast<float>(read(message, cmif + 20)), std::bit_cast<float>(read(message, cmif + 24)),
                std::bit_cast<float>(read(message, cmif + 28)), std::bit_cast<float>(read(message, cmif + 32))});
            return success_reply();
        }
        // The actual value is the last one applied to the motor.
        const auto value = actual_vibration(handle);
        auto reply = success_reply();
        reply.size = 56;
        write(reply.bytes, 4, 12);
        write(reply.bytes, 32, std::bit_cast<std::uint32_t>(value.amplitude_low));
        write(reply.bytes, 36, std::bit_cast<std::uint32_t>(value.frequency_low));
        write(reply.bytes, 40, std::bit_cast<std::uint32_t>(value.amplitude_high));
        write(reply.bytes, 44, std::bit_cast<std::uint32_t>(value.frequency_high));
        return reply;
    }
    case 203: {
        auto reply = success_reply();
        reply.transferred[0] = std::make_shared<ActiveVibrationDeviceList>(process_id_);
        reply.transferred_count = 1;
        return reply;
    }
    case 204:
        vibration_permitted_ = (read(message, cmif + 16, 1) & 1) != 0;
        // A refused permission stops both motors.
        if (!vibration_permitted_)
            for (unsigned device = 0; device < 2; ++device) send_vibration(device << 16 | 3, VibrationValue{});
        return success_reply();
    case 205:
        return value_reply(vibration_permitted_, 1);
    case 107: {
        // DisconnectNpad(npad id, aruid; PID). Only player one's FullKey
        // controller (npad 0) is presented, so other controllers are already
        // disconnected. The presented controller has no reconnection here.
        const auto npad = read(message, cmif + 16);
        if (!valid_npad_id(npad)) return unsupported("HID DisconnectNpad names an invalid npad id");
        if (npad == 0) return unsupported("HID DisconnectNpad of the presented controller is not implemented");
        return success_reply();
    }
    case 211: {
        // IsVibrationDeviceMounted(handle, aruid; PID) -> bool: only player
        // one's FullKey controller is presented, with its two motors.
        const auto handle = read(message, cmif + 16);
        if (const auto error = vibration_handle_result(handle)) return success_reply(error);
        const bool mounted = full_key_connected_ && (handle & 0xffffu) == 3 && ((handle >> 16) & 255) <= 1;
        return value_reply(mounted ? 1 : 0, 1);
    }
    case 214: {
        // Switch 2 vibration (u16, aruid; device handles, 16-byte values): the
        // 23.0.1 hid program takes these two input pointer arrays. What the
        // value words mean on Switch 2 is not established; like the practical
        // reference they are sent as vibration values, and the command never
        // fails.
        if (!switch2_) return unsupported("HID command 214 exists only on Switch 2");
        if (!switch2_vibration_warned_) {
            switch2_vibration_warned_ = true;
            std::fprintf(stderr,
                         "Armigo hid: provisional Switch 2 handler vibration 214 sends values as Switch 1 vibration\n");
        }
        if (!buffers || buffers->size() < 2) return success_reply();
        const auto handle_size = (*buffers)[0].size;
        const auto value_size = (*buffers)[1].size;
        if (handle_size % 4 || value_size % 16 || handle_size > session_pointer_capacity ||
            value_size > session_pointer_capacity)
            return unsupported("HID Switch 2 vibration arrays exceed the supported transfer shape");
        std::array<std::byte, session_pointer_capacity> handles{}, values{};
        if (!buffers->read(0, std::span{handles}.first(handle_size)) ||
            !buffers->read(1, std::span{values}.first(value_size)))
            return unsupported("HID Switch 2 vibration arrays are not readable");
        for (std::size_t pair = 0; pair < std::min(handle_size / 4, value_size / 16); ++pair) {
            std::uint32_t handle = 0;
            std::array<std::uint32_t, 4> words{};
            std::memcpy(&handle, handles.data() + pair * 4, 4);
            std::memcpy(words.data(), values.data() + pair * 16, 16);
            if (vibration_handle_result(handle)) continue;
            send_vibration(handle, VibrationValue{std::bit_cast<float>(words[0]), std::bit_cast<float>(words[1]),
                                                  std::bit_cast<float>(words[2]), std::bit_cast<float>(words[3])});
        }
        return success_reply();
    }
    case 206: {
        if (!buffers || buffers->empty()) return success_reply();
        const auto handle_size = (*buffers)[0].size;
        const auto value_size = (*buffers)[1].size;
        if (handle_size / 4 != value_size / 16) return success_reply(0x106ca);
        if (handle_size % 4 || value_size % 16 || value_size > session_pointer_capacity)
            return unsupported("HID vibration arrays exceed the supported transfer shape");
        std::array<std::byte, session_pointer_capacity> handles{}, values{};
        if (!buffers->read(0, std::span{handles}.first(handle_size)) ||
            !buffers->read(1, std::span{values}.first(value_size)))
            return unsupported("HID vibration arrays are not readable");
        for (std::size_t offset = 0; offset < handle_size; offset += 4) {
            std::uint32_t handle = 0;
            for (unsigned index = 0; index < 4; ++index)
                handle |= std::uint32_t{std::to_integer<unsigned char>(handles[offset + index])}
                          << (8 * index);
            if (const auto error = vibration_handle_result(handle)) return success_reply(error);
        }
        for (std::size_t pair = 0; pair < handle_size / 4; ++pair) {
            std::uint32_t handle = 0;
            std::array<std::uint32_t, 4> words{};
            std::memcpy(&handle, handles.data() + pair * 4, 4);
            std::memcpy(words.data(), values.data() + pair * 16, 16);
            send_vibration(handle, VibrationValue{std::bit_cast<float>(words[0]), std::bit_cast<float>(words[1]),
                                                  std::bit_cast<float>(words[2]), std::bit_cast<float>(words[3])});
        }
        return success_reply();
    }
    default:
        break;
    }
    return unsupported("HID command is not implemented");
}

std::size_t HidSession::domain_object_count() const { return 0; }

const char* HidSession::diagnostic() const {
    const std::lock_guard lock(request_mutex_);
    return diagnostic_;
}

namespace {

// Never destroyed: a sampler thread of a runtime retained past a failed teardown can wake
// during process exit, after static destruction began, and must still find a valid lock.
std::mutex& host_input_mutex() {
    static auto* mutex = new std::mutex;
    return *mutex;
}
HostFullKeyInput host_input;
std::atomic<bool> host_input_set{false};

}  // namespace

void set_host_full_key_input(const HostFullKeyInput& input) noexcept {
    const std::lock_guard lock(host_input_mutex());
    host_input = input;
    host_input_set.store(true, std::memory_order_release);
}

namespace {
// Live applet resources, for writing host input to their shared memory at once.
// Never destroyed, like the host input lock.
std::mutex& sessions_mutex() {
    static auto* mutex = new std::mutex;
    return *mutex;
}
std::vector<AppletResourceSession*>& live_sessions() {
    static auto* sessions = new std::vector<AppletResourceSession*>;
    return *sessions;
}
}  // namespace

void publish_host_full_key_input_now(const HostFullKeyInput& input) noexcept {
    set_host_full_key_input(input);
    const std::lock_guard lock(sessions_mutex());
    for (auto* session : live_sessions()) {
        const std::lock_guard write(session->write_mutex());
        session->shared_memory()->update([&](std::span<std::byte> bytes) {
            detail::publish_full_key_input(bytes,
                {input.buttons, input.left_x, input.left_y, input.right_x, input.right_y});
        });
    }
}

AppletResourceSession::AppletResourceSession(std::uint64_t process_id)
    : process_id_(process_id), shared_memory_(std::make_shared<horizon::SharedMemory>(0x40000)) {
    detail::initialize_shared_memory(shared_memory_->bytes());
    {
        const std::lock_guard lock(sessions_mutex());
        live_sessions().push_back(this);
    }
    sampler_ = std::jthread([this](std::stop_token stop) {
        detail::MotionFusion motion;
        auto next = std::chrono::steady_clock::now();
        auto previous = next;
        while (!stop.stop_requested()) {
            next += std::chrono::milliseconds(5);
            std::this_thread::sleep_until(next);
            const bool input_set = host_input_set.load(std::memory_order_acquire);
            const bool six_axis = six_axis_running_.load(std::memory_order_acquire);
            const auto now = std::chrono::steady_clock::now();
            const auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(now - previous).count();
            previous = now;
            if (!input_set && !six_axis) continue;
            // Read the host input under the write lock, so an immediate publication
            // between the read and the write cannot be overwritten by an older state.
            const std::lock_guard write(write_mutex_);
            HostFullKeyInput input;
            if (input_set) {
                const std::lock_guard lock(host_input_mutex());
                input = host_input;
            }
            if (input_set)
                shared_memory_->update([&](std::span<std::byte> bytes) {
                    detail::publish_full_key_input(bytes,
                        {input.buttons, input.left_x, input.left_y, input.right_x, input.right_y});
                });
            if (!six_axis) continue;
            // One sample per tick (200 Hz); a host without motion reports a
            // controller at rest, face up.
            motion.set_zero_drift_mode(six_axis_drift_mode_.load(std::memory_order_relaxed));
            motion.update(input.has_motion ? input.acceleration : HostFullKeyInput{}.acceleration,
                          input.has_motion ? input.angular_velocity : std::array<float, 3>{},
                          static_cast<std::uint64_t>(std::max<std::int64_t>(elapsed, 0)),
                          input.has_motion && input.has_gyroscope);
            detail::SixAxisSample sample;
            sample.delta_time = 5'000'000;
            sample.acceleration = motion.acceleration();
            sample.angular_velocity = motion.angular_velocity();
            sample.angle = motion.angle();
            sample.direction = motion.orientation();
            sample.connected = true;
            shared_memory_->update([&](std::span<std::byte> bytes) {
                detail::publish_full_key_six_axis(bytes, sample);
            });
        }
    });
}

AppletResourceSession::~AppletResourceSession() {
    const std::lock_guard lock(sessions_mutex());
    std::erase(live_sessions(), this);
}

void AppletResourceSession::set_six_axis(bool running, std::uint32_t zero_drift_mode) noexcept {
    six_axis_drift_mode_.store(zero_drift_mode, std::memory_order_relaxed);
    six_axis_running_.store(running, std::memory_order_release);
}

IpcReply AppletResourceSession::request(const IpcMessage& message, std::uint64_t sender_pid) {
    const std::lock_guard lock(request_mutex_);
    diagnostic_ = nullptr;
    if (sender_pid != process_id_) {
        diagnostic_ = "Applet resource belongs to another process";
        return {};
    }
    const auto type = read(message, 0, 2);
    if (type == 5 || type == 7) {
        if (read(message, 16) != sfci) {
            diagnostic_ = "Applet resource control magic is unsupported";
            return {};
        }
        const auto command = read(message, 24);
        if (command == 3) return control_reply(session_pointer_capacity, 2);
        if (command == 0) return control_reply(1, 4);
        diagnostic_ = "Applet resource control command is unsupported";
        return {};
    }
    if (type != 4 && type != 6) {
        diagnostic_ = "Applet resource request shape is unsupported";
        return {};
    }
    const auto located = horizon::locate_cmif(message);
    if (!located) {
        diagnostic_ = "Applet resource request shape is unsupported";
        return {};
    }
    const auto cmif = *located;
    // The firmware creates the applet resource shared memory as a 0x40000-byte object with
    // owner read-write and remote read-only permission, then hands its handle to the caller.
    if (read(message, cmif + 8) == 0) {
        if (!matches_shape(message, cmif, nullptr, {0, false, 0, std::nullopt})) {
            diagnostic_ = "Applet resource handle request shape is unsupported";
            return {};
        }
        IpcReply reply = success_reply();
        write(reply.bytes, 28, 0x1f3e5d57);
        reply.transferred[0] = shared_memory_;
        reply.transferred_count = 1;
        // GetSharedMemoryHandle answers an OutCopyHandle field, so the handle travels in the
        // copy-handle array the client's CMIF proxy reads.
        reply.transferred_as_copy = true;
        return reply;
    }
    diagnostic_ = "Applet resource command is not implemented";
    return {};
}

const char* AppletResourceSession::diagnostic() const {
    const std::lock_guard lock(request_mutex_);
    return diagnostic_;
}

}  // namespace armigo::services::hid
