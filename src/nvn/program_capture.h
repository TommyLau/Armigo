// ============================================================================
// Armigo NVN - Program Diagnostics
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#pragma once

#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace armigo::nvn::program_capture {

struct Limits {
    std::size_t generations = 2;
    std::size_t descriptors = 9;
    std::size_t events = 256;
    std::uint64_t payload_bytes = 72ull << 20;
};

struct Configuration {
    std::filesystem::path output;
    std::vector<std::string> payload_digests;
    std::optional<int> first, count;
    bool stop_after_capture = false;
    Limits limits;
};

// Parsing and filesystem configuration are performed before guest execution.
[[nodiscard]] std::optional<Configuration> read_configuration(const std::filesystem::path &path,
                                                              std::string &error);

struct Descriptor {
    std::uint64_t data_address = 0, control_address = 0, extra_address = 0;
    std::vector<std::byte> control, extra;
    // The owner keeps the resolver's immutable input alive for queued I/O.
    std::shared_ptr<const void> code_owner;
    std::span<const std::byte> code;
    std::string payload_digest, stage_identity;
    std::optional<std::uint32_t> metadata_selector, metadata_route, metadata_auxiliary;
    bool fields_read = false, control_read = false, code_read = false, extra_read = false;
};

struct Inputs {
    std::uint64_t call = 0, result_sequence = 0, program_address = 0;
    std::optional<std::uint64_t> generation_before, generation_after;
    int count = 0;
    unsigned api_major = 0, api_minor = 0;
    bool extended = false, result = false, payload_filter_match = false, tracked_predecessor = false;
    bool generation_before_observed = false, generation_after_observed = false;
    std::vector<Descriptor> descriptors;
    std::vector<std::string> incomplete;
};

struct Cache {
    std::uint64_t generation = 0, capture_call = 0;
    bool failed = false;
    std::string stage_identity, identity_domain;
};

enum class EventKind { bind, mode, direct_draw };
struct Event {
    EventKind kind = EventKind::bind;
    std::uint64_t backend = 0, submission = 0, ordinal = 0;
    std::uint64_t generation = 0, capture_call = 0, api_return = 0;
    std::uint32_t requested = 0, present = 0, failed = 0;
    bool present_observed = false;
    std::optional<std::uint32_t> mesh_route, auxiliary_word;
    int mode_before = 0, mode_after = 0, first = 0, count = 0;
    std::optional<int> requested_mode, error_code;
    bool matches = false, accepted = false, encoded = false;
    std::array<Cache, 10> before, after;
    std::string detail;
};

struct Capture {
    // History starts at the latest matching bind, including its full initial
    // observed cache snapshot. It does not claim earlier bind history.
    Limits limits;
    std::string history_scope = "latest matching bind through matching direct draw";
    std::vector<std::string> payload_filters;
    std::optional<int> first, count;
    std::vector<Inputs> programs;
    std::vector<Event> events;
    std::vector<std::string> incomplete;
    std::uint64_t omitted_events = 0, unselected_windows = 0;
    bool complete = false;
};

struct WriteResult {
    bool success = false;
    std::string error;
};
class Sink {
  public:
    virtual ~Sink() = default;
    virtual WriteResult inputs(const Inputs &) = 0;
    // Success means the manifest and all referenced payloads are durable.
    virtual WriteResult final(const Capture &) = 0;
};
[[nodiscard]] std::shared_ptr<Sink> file_sink(const std::filesystem::path &output);

class Session {
  public:
    explicit Session(Configuration configuration, std::shared_ptr<Sink> sink = {});
    ~Session();
    Session(const Session &) = delete;
    Session &operator=(const Session &) = delete;
    [[nodiscard]] std::uint64_t next_call() noexcept;
    [[nodiscard]] bool matches(const Inputs &) const noexcept;
    // These only enqueue bounded owned data; neither performs filesystem I/O
    // or invokes a completion handler on a caller's thread.
    void submit(Inputs inputs) noexcept;
    void observe(Event event) noexcept;
    void set_completion(std::function<void()> completion);
    void disconnect_completion() noexcept;
    [[nodiscard]] bool finish(std::chrono::milliseconds timeout);
    [[nodiscard]] bool completed() const noexcept;
    [[nodiscard]] bool accepting() const noexcept;
    void record_loss() noexcept;
    [[nodiscard]] const Limits &limits() const noexcept;

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

using ReadMemory = bool (*)(void *, std::uint64_t, std::span<std::byte>);
// Initialization is outside replay/resource/submission locks. Missing env
// configuration leaves every call/event path disabled without input copies.
void configure_from_environment(ReadMemory read, void *context = nullptr) noexcept;
void configure(Configuration configuration, ReadMemory read, void *context = nullptr);
[[nodiscard]] bool enabled() noexcept;

class RunBinding {
  public:
    RunBinding() = default;
    RunBinding(RunBinding &&) noexcept;
    RunBinding &operator=(RunBinding &&) noexcept;
    ~RunBinding();
    // Call after ordinary runtime teardown, on the host control thread.
    void finish(bool runtime_quiescent) noexcept;

  private:
    friend RunBinding bind_run(std::function<void()>);
    std::shared_ptr<Session> session_;
};
[[nodiscard]] RunBinding bind_run(std::function<void()> complete);

class Call {
  public:
    Call(std::uint64_t program, int count, std::uint64_t descriptors, bool extended);
    ~Call();
    Call(const Call &) = delete;
    Call &operator=(const Call &) = delete;
    [[nodiscard]] std::uint64_t id() const noexcept;
    [[nodiscard]] bool matches() const noexcept;
    void finish(bool result, std::optional<std::uint64_t> generation) noexcept;
    static Call *active() noexcept;
    void control(std::size_t index, std::span<const std::byte> bytes, std::uint64_t address = 0) noexcept;
    void metadata(std::size_t index, std::uint32_t selector, std::optional<std::uint32_t> route = {},
                  std::optional<std::uint32_t> auxiliary = {}) noexcept;
    void code(std::size_t index, std::shared_ptr<const void> owner, std::span<const std::byte> bytes,
              std::string_view digest, std::string_view identity, std::uint64_t address = 0) noexcept;

  private:
    struct State;
    std::unique_ptr<State> state_;
    Call *parent_ = nullptr;
    bool borrowed_ = false;
};

void observe(Event event) noexcept;
void record_loss() noexcept;

} // namespace armigo::nvn::program_capture
