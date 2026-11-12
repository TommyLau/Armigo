// ============================================================================
// Armigo NVN - Program Diagnostics
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include "program_capture.h"
#include "objects.h"
#include "entry_table.h"
#include <armigo/common/diagnostics.h>

#include <algorithm>
#include <atomic>
#include <charconv>
#include <cerrno>
#include <climits>
#include <stdexcept>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <fstream>
#include <mutex>
#include <sstream>
#include <thread>
#include <utility>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

namespace armigo::nvn::program_capture {
namespace {
constexpr std::size_t control_bytes = 0xe0, extra_bytes = 24, maximum_code = 4u << 20;
constexpr Limits hard_limits{};
std::string json_string(std::string_view value) {
    std::string out = "\"";
    constexpr char hex[] = "0123456789abcdef";
    for (unsigned char c : value) {
        if (c == '"' || c == '\\') {
            out += '\\';
            out += char(c);
        } else if (c < 0x20) {
            out += "\\u00";
            out += hex[c >> 4];
            out += hex[c & 15];
        } else
            out += char(c);
    }
    return out + '"';
}
std::string address(std::uint64_t value) {
    char text[24];
    std::snprintf(text, sizeof text, "0x%llx", static_cast<unsigned long long>(value));
    return json_string(text);
}
void strings(std::ostream &out, const std::vector<std::string> &values) {
    out << '[';
    for (std::size_t i = 0; i != values.size(); ++i) {
        if (i)
            out << ',';
        out << json_string(values[i]);
    }
    out << ']';
}
void generation(std::ostream &out, std::optional<std::uint64_t> value) {
    if (value)
        out << json_string(std::to_string(*value));
    else
        out << "null";
}
void scalar(std::ostream &out, std::optional<std::uint32_t> value) {
    if (value)
        out << *value;
    else
        out << "null";
}
std::string input_json(const Inputs &input) {
    std::ostringstream out;
    out << "{\"call\":" << json_string(std::to_string(input.call))
        << ",\"result_sequence\":" << json_string(std::to_string(input.result_sequence))
        << ",\"program_address\":" << address(input.program_address) << ",\"api_major\":" << input.api_major
        << ",\"api_minor\":" << input.api_minor << ",\"count\":" << input.count
        << ",\"kind\":" << json_string(input.extended ? "SetShadersExt" : "SetShaders")
        << ",\"result\":" << (input.result ? "true" : "false")
        << ",\"payload_filter_match\":" << (input.payload_filter_match ? "true" : "false")
        << ",\"tracked_predecessor\":" << (input.tracked_predecessor ? "true" : "false")
        << ",\"generation_before\":";
    generation(out, input.generation_before);
    out << ",\"generation_after\":";
    generation(out, input.generation_after);
    out << ",\"generation_before_observed\":" << (input.generation_before_observed ? "true" : "false")
        << ",\"generation_after_observed\":" << (input.generation_after_observed ? "true" : "false");
    out << ",\"control_boundary\":\"native safe-read copy before validation and deduplication\","
           "\"extra_boundary\":\"safe API entry copy before Ext validation\","
           "\"metadata_boundary\":\"program metadata reads after resolver return\",\"incomplete\":";
    strings(out, input.incomplete);
    out << ",\"descriptors\":[";
    for (std::size_t i = 0; i != input.descriptors.size(); ++i) {
        const auto &d = input.descriptors[i];
        if (i)
            out << ',';
        out << "{\"index\":" << i << ",\"data_address\":" << address(d.data_address)
            << ",\"control_address\":" << address(d.control_address)
            << ",\"extra_address\":" << address(d.extra_address)
            << ",\"fields_read\":" << (d.fields_read ? "true" : "false")
            << ",\"control_read\":" << (d.control_read ? "true" : "false")
            << ",\"code_read\":" << (d.code_read ? "true" : "false")
            << ",\"extra_read\":" << (d.extra_read ? "true" : "false")
            << ",\"control_size\":" << d.control.size() << ",\"code_size\":" << d.code.size()
            << ",\"extra_size\":" << d.extra.size() << ",\"payload_digest\":" << json_string(d.payload_digest)
            << ",\"stage_identity\":" << json_string(d.stage_identity) << ",\"control_file\":"
            << (d.control_read ? json_string("call-" + std::to_string(input.call) + "/control-" +
                                             std::to_string(i) + ".bin")
                               : "null")
            << ",\"code_file\":"
            << (d.code_read ? json_string("call-" + std::to_string(input.call) + "/code-" +
                                          std::to_string(i) + ".bin")
                            : "null")
            << ",\"extra_file\":"
            << (d.extra_read ? json_string("call-" + std::to_string(input.call) + "/extra-" +
                                           std::to_string(i) + ".bin")
                             : "null")
            << ",\"metadata_selector\":";
        scalar(out, d.metadata_selector);
        out << ",\"metadata_route\":";
        scalar(out, d.metadata_route);
        out << ",\"metadata_auxiliary\":";
        scalar(out, d.metadata_auxiliary);
        out << '}';
    }
    out << "]}";
    return out.str();
}
void caches(std::ostream &out, const std::array<Cache, 10> &values) {
    constexpr const char *roles[]{
        "vertex",  "fragment", "geometry",      "tessellation-control", "tessellation-evaluation",
        "compute", "task",     "task-fed-mesh", "standalone-mesh",      "auxiliary"};
    out << '[';
    for (std::size_t i = 0; i != values.size(); ++i) {
        if (i)
            out << ',';
        const auto &value = values[i];
        out << "{\"cache\":" << i << ",\"role\":" << json_string(roles[i])
            << ",\"generation\":" << json_string(std::to_string(value.generation))
            << ",\"capture_call\":" << json_string(std::to_string(value.capture_call))
            << ",\"failed\":" << (value.failed ? "true" : "false")
            << ",\"stage_identity\":" << json_string(value.stage_identity)
            << ",\"identity_domain\":" << json_string(value.identity_domain) << '}';
    }
    out << ']';
}
std::string capture_json(const Capture &capture) {
    std::ostringstream out;
    out << "{\"schema\":\"armigo-nvn-program-capture-v1\",\"inputs_complete\":"
        << (capture.complete ? "true" : "false")
        << ",\"writer_completion_required\":true,\"observation\":\"Armigo replay "
           "state\",\"original_driver_execution_observed\":false,"
           "\"history_scope\":"
        << json_string(capture.history_scope)
        << ",\"command_id_domain\":\"submission-fence:flattened-record-ordinal\",\"omitted_events\":"
        << capture.omitted_events << ",\"unselected_windows\":" << capture.unselected_windows
        << ",\"incomplete\":";
    strings(out, capture.incomplete);
    out << ",\"limits\":{\"generations\":" << capture.limits.generations
        << ",\"descriptors\":" << capture.limits.descriptors << ",\"events\":" << capture.limits.events
        << ",\"payload_bytes\":" << capture.limits.payload_bytes << "},\"payload_filters\":";
    strings(out, capture.payload_filters);
    out << ",\"draw_filter\":{\"first\":";
    if (capture.first)
        out << *capture.first;
    else
        out << "null";
    out << ",\"count\":";
    if (capture.count)
        out << *capture.count;
    else
        out << "null";
    out << "},\"programs\":[";
    for (std::size_t i = 0; i != capture.programs.size(); ++i) {
        if (i)
            out << ',';
        out << input_json(capture.programs[i]);
    }
    out << "],\"events\":[";
    for (std::size_t i = 0; i != capture.events.size(); ++i) {
        if (i)
            out << ',';
        const auto &e = capture.events[i];
        const char *kind = e.kind == EventKind::bind   ? "bind"
                           : e.kind == EventKind::mode ? "mode"
                                                       : "direct_draw";
        out << "{\"kind\":" << json_string(kind) << ",\"backend\":" << address(e.backend)
            << ",\"submission_fence\":" << json_string(std::to_string(e.submission))
            << ",\"flattened_record_ordinal\":" << e.ordinal
            << ",\"generation\":" << json_string(std::to_string(e.generation))
            << ",\"capture_call\":" << json_string(std::to_string(e.capture_call))
            << ",\"requested_mask\":" << e.requested << ",\"logical_present_mask\":" << e.present
            << ",\"present_observed\":" << (e.present_observed ? "true" : "false")
            << ",\"failed_mask\":" << e.failed << ",\"mode_before\":" << e.mode_before
            << ",\"mode_after\":" << e.mode_after << ",\"first\":" << e.first << ",\"count\":" << e.count
            << ",\"api_return\":" << address(e.api_return)
            << ",\"api_return_domain\":" << json_string(e.api_return ? "api231-entry-return" : "unavailable")
            << ",\"accepted\":" << (e.accepted ? "true" : "false")
            << ",\"encoded\":" << (e.encoded ? "true" : "false") << ",\"detail\":" << json_string(e.detail)
            << ",\"before\":";
        caches(out, e.before);
        out << ",\"after\":";
        caches(out, e.after);
        out << ",\"metadata_route_byte\":";
        scalar(out, e.mesh_route);
        out << ",\"metadata_auxiliary_word\":";
        scalar(out, e.auxiliary_word);
        out << ",\"requested_mode\":";
        if (e.requested_mode)
            out << *e.requested_mode;
        else
            out << "null";
        out << ",\"error_code\":";
        if (e.error_code)
            out << *e.error_code;
        else
            out << "null";
        out << '}';
    }
    out << "]}\n";
    return out.str();
}
std::span<const std::byte> bytes(std::string_view text) {
    return {reinterpret_cast<const std::byte *>(text.data()), text.size()};
}
bool durable_file(const std::filesystem::path &path, std::span<const std::byte> data, std::string &error) {
    const int fd = ::open(path.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0600);
    if (fd < 0) {
        error = "cannot create capture file";
        return false;
    }
    bool okay = true;
    while (!data.empty()) {
        const auto written = ::write(fd, data.data(), data.size());
        if (written < 0 && errno == EINTR)
            continue;
        if (written <= 0) {
            okay = false;
            break;
        }
        data = data.subspan(static_cast<std::size_t>(written));
    }
    if (okay && ::fsync(fd) != 0)
        okay = false;
    if (::close(fd) != 0)
        okay = false;
    if (!okay)
        error = "capture file did not flush successfully";
    return okay;
}
bool durable_directory(const std::filesystem::path &path) {
    const int fd = ::open(path.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC);
    if (fd < 0)
        return false;
    const bool okay = ::fsync(fd) == 0;
    return ::close(fd) == 0 && okay;
}
class FileSink final : public Sink {
  public:
    explicit FileSink(std::filesystem::path path) : output_(std::move(path)) {}
    WriteResult inputs(const Inputs &input) override {
        if (auto error = open_root(); !error.empty())
            return {false, std::move(error)};
        const auto dir = output_ / ("call-" + std::to_string(input.call));
        std::error_code ec;
        if (!std::filesystem::create_directory(dir, ec))
            return {false, "capture call directory already exists or is unavailable"};
        std::filesystem::permissions(dir, std::filesystem::perms::owner_all, ec);
        if (ec)
            return {false, "cannot restrict capture call directory"};
        std::string error;
        for (std::size_t i = 0; i != input.descriptors.size(); ++i) {
            const auto &d = input.descriptors[i];
            if (d.control_read &&
                !durable_file(dir / ("control-" + std::to_string(i) + ".bin"), d.control, error))
                return {false, error};
            if (d.code_read && !durable_file(dir / ("code-" + std::to_string(i) + ".bin"), d.code, error))
                return {false, error};
            if (d.extra_read && !durable_file(dir / ("extra-" + std::to_string(i) + ".bin"), d.extra, error))
                return {false, error};
        }
        const auto text = input_json(input) + '\n';
        if (!durable_file(dir / "inputs.json", bytes(text), error))
            return {false, error};
        if (!durable_directory(dir) || !durable_directory(output_))
            return {false, "capture input directory did not flush"};
        return {true, {}};
    }
    WriteResult final(const Capture &capture) override {
        if (auto error = open_root(); !error.empty())
            return {false, std::move(error)};
        const auto text = capture_json(capture);
        std::string error;
        if (!durable_file(output_ / "capture.json.tmp", bytes(text), error))
            return {false, error};
        std::error_code ec;
        std::filesystem::rename(output_ / "capture.json.tmp", output_ / "capture.json", ec);
        if (ec || !durable_directory(output_) || !durable_directory(output_.parent_path()))
            return {false, "capture manifest did not publish durably"};
        return {true, {}};
    }

  private:
    std::string open_root() {
        if (opened_)
            return {};
        std::error_code ec;
        if (!std::filesystem::create_directory(output_, ec))
            return "capture output must be a new directory with an existing parent";
        std::filesystem::permissions(output_, std::filesystem::perms::owner_all, ec);
        if (ec)
            return "cannot restrict capture output directory";
        opened_ = true;
        return {};
    }
    std::filesystem::path output_;
    bool opened_ = false;
};
std::string_view trim(std::string_view value) {
    while (!value.empty() && (value.front() == ' ' || value.front() == '\t' || value.front() == '\r'))
        value.remove_prefix(1);
    while (!value.empty() && (value.back() == ' ' || value.back() == '\t' || value.back() == '\r'))
        value.remove_suffix(1);
    return value;
}
bool number(std::string_view value, std::uint64_t &result) {
    auto parsed = std::from_chars(value.data(), value.data() + value.size(), result);
    return parsed.ec == std::errc{} && parsed.ptr == value.data() + value.size();
}
bool valid_digest(std::string_view value) {
    return value.size() == 64 && std::ranges::all_of(value, [](char c) {
               return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f');
           });
}
} // namespace

std::optional<Configuration> read_configuration(const std::filesystem::path &path, std::string &error) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) {
        error = "cannot read program capture configuration";
        return {};
    }
    std::string text;
    char chunk[1024];
    while (stream.read(chunk, sizeof chunk) || stream.gcount()) {
        text.append(chunk, static_cast<std::size_t>(stream.gcount()));
        if (text.size() > 16 * 1024) {
            error = "program capture configuration exceeds 16 KiB";
            return {};
        }
    }
    if (stream.bad()) {
        error = "capture configuration read failed";
        return {};
    }
    Configuration config;
    bool version = false;
    std::istringstream lines(text);
    for (std::string line; std::getline(lines, line);) {
        auto value = trim(line);
        if (value.empty() || value.front() == '#')
            continue;
        const auto split = value.find('=');
        if (split == value.npos) {
            error = "program capture configuration requires key=value lines";
            return {};
        }
        const auto key = trim(value.substr(0, split));
        value = trim(value.substr(split + 1));
        std::uint64_t integer = 0;
        if (key == "version") {
            if (value != "1" || version) {
                error = "unsupported or duplicate capture configuration version";
                return {};
            }
            version = true;
        } else if (key == "output")
            config.output = std::string(value);
        else if (key == "payload_digest") {
            if (!valid_digest(value) || config.payload_digests.size() == hard_limits.descriptors) {
                error = "invalid or excessive capture payload filters";
                return {};
            }
            config.payload_digests.emplace_back(value);
        } else if (key == "stop_after_capture") {
            if (value != "0" && value != "1") {
                error = "capture stop flag must be 0 or 1";
                return {};
            }
            config.stop_after_capture = value == "1";
        } else if (!number(value, integer)) {
            error = "capture numeric field is invalid";
            return {};
        } else if (key == "first" && integer <= INT_MAX)
            config.first = static_cast<int>(integer);
        else if (key == "count" && integer <= INT_MAX)
            config.count = static_cast<int>(integer);
        else if (key == "max_generations" && integer && integer <= hard_limits.generations)
            config.limits.generations = integer;
        else if (key == "max_descriptors" && integer && integer <= hard_limits.descriptors)
            config.limits.descriptors = integer;
        else if (key == "max_events" && integer && integer <= hard_limits.events)
            config.limits.events = integer;
        else if (key == "max_payload_bytes" && integer && integer <= hard_limits.payload_bytes)
            config.limits.payload_bytes = integer;
        else {
            error = "unknown capture option or limit outside the hard cap";
            return {};
        }
    }
    if (!version || !config.output.is_absolute() || config.payload_digests.empty()) {
        error = "capture requires version, absolute output and payload filters";
        return {};
    }
    return config;
}

std::shared_ptr<Sink> file_sink(const std::filesystem::path &output) {
    return std::make_shared<FileSink>(output);
}

struct Session::Impl {
    struct Job {
        std::optional<Inputs> input;
        std::optional<Capture> capture;
    };
    struct History {
        std::uint64_t backend;
        std::deque<Event> events;
        std::uint64_t omitted = 0, unselected = 0, armed = 0;
    };
    Configuration config;
    std::shared_ptr<Sink> sink;
    mutable std::mutex mutex;
    std::condition_variable changed;
    std::deque<Job> jobs;
    std::vector<Inputs> programs;
    std::vector<History> histories;
    std::vector<std::string> issues;
    std::mutex completion_mutex;
    std::function<void()> completion;
    std::thread writer;
    std::atomic_uint64_t next{1}, next_result{1};
    std::atomic_bool completed{false}, accepting{true}, allocation_failed{false}, writer_failed{false};
    std::uint64_t payload_bytes = 0, arm_sequence = 0;
    bool triggered = false, closing = false, exited = false;
    Impl(Configuration c, std::shared_ptr<Sink> s)
        : config(std::move(c)), sink(s ? std::move(s) : file_sink(config.output)) {
        writer = std::thread([this] { write_loop(); });
    }
    void write_loop() {
        std::vector<std::string> write_errors;
        for (;;) {
            Job job;
            {
                std::unique_lock lock(mutex);
                changed.wait(lock, [&] { return closing || !jobs.empty(); });
                if (jobs.empty())
                    break;
                job = std::move(jobs.front());
                jobs.pop_front();
            }
            try {
                WriteResult result;
                if (job.input)
                    result = sink->inputs(*job.input);
                else {
                    auto &capture = *job.capture;
                    capture.incomplete.insert(capture.incomplete.end(), write_errors.begin(),
                                              write_errors.end());
                    if (allocation_failed.load())
                        capture.incomplete.emplace_back("capture observation was lost or truncated");
                    if (writer_failed.load())
                        capture.incomplete.emplace_back("capture writer failed");
                    capture.complete = capture.incomplete.empty();
                    result = sink->final(capture);
                    if (result.success && capture.complete) {
                        completed.store(true, std::memory_order_release);
                        diag::opt_in("nvn_program_capture complete=1 durable=1\n");
                        const std::lock_guard completion_lock(completion_mutex);
                        if (config.stop_after_capture && completion)
                            completion();
                    }
                }
                if (!result.success) {
                    write_errors.emplace_back(result.error.empty() ? "capture write failed" : result.error);
                    diag::write(diag::Level::error, "Armigo NVN program capture: %s\n",
                                write_errors.back().c_str());
                }
            } catch (...) {
                writer_failed.store(true);
                completed.store(false, std::memory_order_release);
                diag::write(diag::Level::error, "Armigo NVN program capture: writer failed\n");
            }
        }
        {
            const std::lock_guard lock(mutex);
            exited = true;
        }
        changed.notify_all();
    }
};

Session::Session(Configuration config, std::shared_ptr<Sink> sink) {
    if (!config.limits.generations || config.limits.generations > hard_limits.generations ||
        !config.limits.descriptors || config.limits.descriptors > hard_limits.descriptors ||
        !config.limits.events || config.limits.events > hard_limits.events || !config.limits.payload_bytes ||
        config.limits.payload_bytes > hard_limits.payload_bytes)
        throw std::invalid_argument("program capture limit exceeds hard cap");
    impl_ = std::make_unique<Impl>(std::move(config), std::move(sink));
}
Session::~Session() {
    if (impl_->writer.joinable()) {
        {
            const std::lock_guard lock(impl_->mutex);
            impl_->closing = true;
        }
        impl_->changed.notify_all();
        impl_->writer.join();
    }
}
std::uint64_t Session::next_call() noexcept { return impl_->next.fetch_add(1, std::memory_order_relaxed); }
const Limits &Session::limits() const noexcept { return impl_->config.limits; }
bool Session::matches(const Inputs &input) const noexcept {
    const bool filtered = std::ranges::all_of(impl_->config.payload_digests, [&](const auto &digest) {
        return std::ranges::any_of(
            input.descriptors, [&](const auto &descriptor) { return descriptor.payload_digest == digest; });
    });
    if (filtered)
        return true;
    try {
        const std::lock_guard lock(impl_->mutex);
        return std::ranges::any_of(impl_->programs, [&](const auto &previous) {
            return previous.program_address == input.program_address && input.generation_before &&
                   previous.generation_after == input.generation_before;
        });
    } catch (...) {
        impl_->allocation_failed.store(true);
        return false;
    }
}
void Session::submit(Inputs input) noexcept {
    try {
        input.result_sequence = impl_->next_result.fetch_add(1, std::memory_order_relaxed);
        const bool filtered = std::ranges::all_of(impl_->config.payload_digests, [&](const auto &digest) {
            return std::ranges::any_of(input.descriptors, [&](const auto &descriptor) {
                return descriptor.payload_digest == digest;
            });
        });
        const std::lock_guard lock(impl_->mutex);
        if (impl_->closing || impl_->triggered)
            return;
        const bool tracked = std::ranges::any_of(impl_->programs, [&](const auto &previous) {
            return previous.program_address == input.program_address && input.generation_before &&
                   previous.generation_after == input.generation_before;
        });
        if (!filtered && !tracked)
            return;
        input.payload_filter_match = filtered;
        input.tracked_predecessor = tracked;
        if (impl_->programs.size() == impl_->config.limits.generations) {
            if (std::ranges::find(impl_->issues, "generation limit exceeded") == impl_->issues.end())
                impl_->issues.emplace_back("generation limit exceeded");
            return;
        }
        if (std::ranges::find(impl_->programs, input.call, &Inputs::call) != impl_->programs.end()) {
            if (std::ranges::find(impl_->issues, "duplicate capture call") == impl_->issues.end())
                impl_->issues.emplace_back("duplicate capture call");
            return;
        }
        if (!input.generation_before_observed || !input.generation_after_observed)
            input.incomplete.emplace_back("program generation was not observed at a call boundary");
        if (input.count < 0 || std::size_t(input.count) != input.descriptors.size())
            input.incomplete.emplace_back("descriptor list is incomplete");
        if (input.descriptors.size() > impl_->config.limits.descriptors) {
            input.descriptors.resize(impl_->config.limits.descriptors);
            input.incomplete.emplace_back("descriptor limit exceeded");
        }
        if (input.incomplete.size() > 32) {
            input.incomplete.resize(32);
            input.incomplete.back() = "input issue list exceeded limit";
        }
        for (auto &reason : input.incomplete)
            if (reason.size() > 256)
                reason.resize(256);
        for (auto &d : input.descriptors) {
            if (d.stage_identity.size() > 128) {
                d.stage_identity.resize(128);
                input.incomplete.emplace_back("stage identity exceeded limit");
            }
            if (!d.fields_read)
                input.incomplete.emplace_back("descriptor fields not completely observed");
            if (d.control.size() > control_bytes) {
                d.control.resize(control_bytes);
                d.control_read = false;
            }
            if (d.extra.size() > extra_bytes) {
                d.extra.resize(extra_bytes);
                d.extra_read = false;
            }
            if (!d.control_read || d.control.size() != control_bytes)
                input.incomplete.emplace_back("control bytes not completely observed");
            std::uint32_t declared_code = 0;
            if (d.control.size() == control_bytes)
                std::memcpy(&declared_code, d.control.data() + 0x44, sizeof declared_code);
            if (!d.code_read || !d.code_owner || !d.code.size() || d.code.size() != declared_code ||
                d.code.size() > maximum_code || !valid_digest(d.payload_digest)) {
                input.incomplete.emplace_back("code bytes not completely observed");
                d.code_read = false;
                d.code = {};
                d.code_owner.reset();
            }
            if (d.extra_address && (!d.extra_read || d.extra.size() != extra_bytes))
                input.incomplete.emplace_back("extended metadata not completely observed");
            if (d.code.size() > impl_->config.limits.payload_bytes - impl_->payload_bytes) {
                input.incomplete.emplace_back("payload byte limit exceeded");
                d.code_read = false;
                d.code = {};
                d.code_owner.reset();
            }
            impl_->payload_bytes += d.code.size();
        }
        impl_->programs.push_back(input);
        impl_->jobs.push_back({std::move(input), {}});
        impl_->changed.notify_all();
    } catch (...) {
        impl_->allocation_failed.store(true);
    }
}
void Session::observe(Event event) noexcept {
    try {
        const std::lock_guard lock(impl_->mutex);
        if (impl_->closing || impl_->triggered)
            return;
        if (event.detail.size() > 4096) {
            event.detail.resize(4096);
            impl_->allocation_failed.store(true);
        }
        for (auto *snapshot : {&event.before, &event.after})
            for (auto &cache : *snapshot)
                if (cache.stage_identity.size() > 128) {
                    cache.stage_identity.resize(128);
                    impl_->allocation_failed.store(true);
                }
        auto history = std::ranges::find(impl_->histories, event.backend, &Impl::History::backend);
        if (event.kind == EventKind::bind && event.matches) {
            if (history == impl_->histories.end()) {
                if (impl_->histories.size() == 16) {
                    if (std::ranges::find(impl_->issues, "backend limit exceeded") == impl_->issues.end())
                        impl_->issues.emplace_back("backend limit exceeded");
                    return;
                }
                impl_->histories.push_back({event.backend, {}, 0, 0});
                history = std::prev(impl_->histories.end());
            }
            if (!history->events.empty())
                ++history->unselected;
            history->events.clear();
            history->omitted = 0;
            history->armed = ++impl_->arm_sequence;
        }
        if (history == impl_->histories.end())
            return;
        if (history->events.size() == impl_->config.limits.events) {
            history->events.pop_front();
            ++history->omitted;
        }
        history->events.push_back(std::move(event));
        const auto &last = history->events.back();
        if (last.kind != EventKind::direct_draw || !last.matches ||
            (impl_->config.first && *impl_->config.first != last.first) ||
            (impl_->config.count && *impl_->config.count != last.count))
            return;
        Capture capture;
        capture.limits = impl_->config.limits;
        capture.payload_filters = impl_->config.payload_digests;
        capture.first = impl_->config.first;
        capture.count = impl_->config.count;
        capture.programs = impl_->programs;
        std::ranges::sort(capture.programs, {}, &Inputs::call);
        capture.events.assign(history->events.begin(), history->events.end());
        capture.incomplete = impl_->issues;
        capture.omitted_events = history->omitted;
        capture.unselected_windows = history->unselected;
        if (history->omitted)
            capture.incomplete.emplace_back("event history limit exceeded");
        const auto found = std::ranges::find(capture.programs, last.capture_call, &Inputs::call);
        if (found == capture.programs.end())
            capture.incomplete.emplace_back("selected program inputs were not retained");
        else if (!found->result || !found->generation_after || *found->generation_after != last.generation)
            capture.incomplete.emplace_back(
                "selected program generation was not published by the captured call");
        for (const auto &program : capture.programs)
            capture.incomplete.insert(capture.incomplete.end(), program.incomplete.begin(),
                                      program.incomplete.end());
        capture.complete = capture.incomplete.empty();
        impl_->jobs.push_back({{}, std::move(capture)});
        impl_->triggered = true;
        impl_->accepting.store(false, std::memory_order_release);
        impl_->changed.notify_all();
    } catch (...) {
        impl_->allocation_failed.store(true);
    }
}
void Session::set_completion(std::function<void()> completion) {
    const std::lock_guard lock(impl_->completion_mutex);
    impl_->completion = std::move(completion);
}
void Session::disconnect_completion() noexcept {
    const std::lock_guard lock(impl_->completion_mutex);
    impl_->completion = {};
}
bool Session::finish(std::chrono::milliseconds timeout) {
    std::unique_lock lock(impl_->mutex);
    if (!impl_->closing && !impl_->triggered) {
        Capture partial;
        partial.limits = impl_->config.limits;
        partial.payload_filters = impl_->config.payload_digests;
        partial.first = impl_->config.first;
        partial.count = impl_->config.count;
        partial.programs = impl_->programs;
        partial.incomplete = impl_->issues;
        std::ranges::sort(partial.programs, {}, &Inputs::call);
        partial.history_scope = "latest armed matching-bind window; no matching direct draw observed";
        if (!impl_->histories.empty()) {
            const auto history = std::ranges::max_element(impl_->histories, {}, &Impl::History::armed);
            partial.events.assign(history->events.begin(), history->events.end());
            partial.omitted_events = history->omitted;
            partial.unselected_windows = history->unselected;
        }
        partial.incomplete.emplace_back("matching direct draw was not observed");
        impl_->jobs.push_back({{}, std::move(partial)});
        impl_->triggered = true;
    }
    impl_->closing = true;
    impl_->accepting.store(false, std::memory_order_release);
    impl_->changed.notify_all();
    if (!impl_->changed.wait_for(lock, timeout, [&] { return impl_->exited; }))
        return false;
    lock.unlock();
    if (impl_->writer.joinable())
        impl_->writer.join();
    return true;
}
bool Session::completed() const noexcept { return impl_->completed.load(std::memory_order_acquire); }
bool Session::accepting() const noexcept { return impl_->accepting.load(std::memory_order_acquire); }
void Session::record_loss() noexcept { impl_->allocation_failed.store(true); }

namespace {
struct Global {
    std::mutex mutex;
    std::optional<Configuration> configuration;
    ReadMemory read = nullptr;
    void *context = nullptr;
    std::shared_ptr<Session> session, retained;
    std::atomic_bool active{false};
    bool from_environment = false;
};
// A failed runtime teardown may leave callbacks alive until process exit.
// The holder must not destroy their session during static destruction.
Global &global = *new Global;
thread_local Call *current_call = nullptr;
std::shared_ptr<Session> active_session() {
    if (!global.active.load(std::memory_order_acquire))
        return {};
    const std::lock_guard lock(global.mutex);
    return global.session;
}
} // namespace

void configure(Configuration configuration, ReadMemory read, void *context) {
    const std::lock_guard lock(global.mutex);
    if (global.session || global.retained)
        throw std::runtime_error("previous program capture session remains owned");
    global.configuration = std::move(configuration);
    global.read = read;
    global.context = context;
    global.from_environment = false;
}
void configure_from_environment(ReadMemory read, void *context) noexcept {
    const char *path = std::getenv("ARMIGO_NVN_PROGRAM_CAPTURE_CONFIG");
    try {
        {
            const std::lock_guard lock(global.mutex);
            if (global.session || global.retained)
                return;
            global.configuration.reset();
            global.read = read;
            global.context = context;
            global.from_environment = true;
        }
        if (!path || !*path)
            return;
        std::string error;
        auto configuration = read_configuration(path, error);
        if (!configuration) {
            diag::write(diag::Level::error, "Armigo NVN program capture: %s\n", error.c_str());
            return;
        }
        const std::lock_guard lock(global.mutex);
        if (!global.session && !global.retained)
            global.configuration = std::move(*configuration);
    } catch (...) {
        diag::write(diag::Level::error, "Armigo NVN program capture: configuration failed\n");
    }
}
bool enabled() noexcept { return global.active.load(std::memory_order_acquire); }
RunBinding bind_run(std::function<void()> completion) {
    RunBinding binding;
    bool refresh;
    ReadMemory read;
    void *context;
    bool configured;
    {
        const std::lock_guard lock(global.mutex);
        refresh = global.from_environment;
        read = global.read;
        context = global.context;
        configured = global.configuration.has_value();
    }
    if (!refresh && !configured) {
        const char *requested = std::getenv("ARMIGO_NVN_PROGRAM_CAPTURE_CONFIG");
        refresh = requested && *requested;
    }
    if (refresh)
        configure_from_environment(read, context);
    const std::lock_guard lock(global.mutex);
    if (!global.configuration)
        return binding;
    if (!global.read) {
        diag::write(diag::Level::error, "Armigo NVN program capture: native input reader is unavailable\n");
        return binding;
    }
    if (global.session || global.retained) {
        diag::write(diag::Level::error, "Armigo NVN program capture: previous session remains owned\n");
        return binding;
    }
    binding.session_ = std::make_shared<Session>(*global.configuration);
    binding.session_->set_completion(std::move(completion));
    global.session = binding.session_;
    global.active.store(true, std::memory_order_release);
    return binding;
}
RunBinding::RunBinding(RunBinding &&other) noexcept : session_(std::move(other.session_)) {}
RunBinding &RunBinding::operator=(RunBinding &&other) noexcept {
    if (this != &other) {
        finish(false);
        session_ = std::move(other.session_);
    }
    return *this;
}
RunBinding::~RunBinding() { finish(false); }
void RunBinding::finish(bool quiescent) noexcept {
    if (!session_)
        return;
    auto session = std::move(session_);
    session->disconnect_completion();
    {
        const std::lock_guard lock(global.mutex);
        global.active.store(false, std::memory_order_release);
        if (global.session == session)
            global.session.reset();
    }
    bool finished = false;
    try {
        if (quiescent)
            finished = session->finish(std::chrono::seconds(5));
    } catch (...) {
    }
    if (!finished) {
        const std::lock_guard lock(global.mutex);
        global.retained = std::move(session);
        diag::write(diag::Level::error,
                    "Armigo NVN program capture: session retained after incomplete teardown\n");
    }
}

struct Call::State {
    std::shared_ptr<Session> session;
    Inputs input;
    bool finished = false, matched = false, observation_failed = false;
};
Call::Call(std::uint64_t program, int count, std::uint64_t descriptors, bool extended) {
    if (!enabled())
        return;
    parent_ = current_call;
    if (parent_) {
        borrowed_ = true;
        return;
    }
    std::shared_ptr<Session> session;
    try {
        session = active_session();
        if (!session || !session->accepting())
            return;
        state_ = std::make_unique<State>();
        state_->session = session;
        auto &input = state_->input;
        input.call = state_->session->next_call();
        input.program_address = program;
        input.count = count;
        input.extended = extended;
        if (const auto api = selected_api_version()) {
            input.api_major = api->major;
            input.api_minor = api->minor;
        }
        ReadMemory read;
        void *context;
        {
            const std::lock_guard lock(global.mutex);
            read = global.read;
            context = global.context;
        }
        if (!read)
            input.incomplete.emplace_back("safe input reader is unavailable");
        ProgramState object{};
        if (!program)
            input.generation_before_observed = true;
        else if (read && read(context, program, {reinterpret_cast<std::byte *>(&object), sizeof object})) {
            input.generation_before_observed = true;
            if (object.tag == object_tag('P', 'R', 'O', 'G'))
                input.generation_before = object.serial;
        } else
            input.incomplete.emplace_back("program generation before call was not observed");
        if (count < 0 || std::uint64_t(count) > state_->session->limits().descriptors)
            input.incomplete.emplace_back("descriptor count outside capture limit");
        const auto stored =
            count < 0 ? 0 : std::min(std::size_t(count), state_->session->limits().descriptors);
        input.descriptors.resize(stored);
        for (std::size_t i = 0; i != stored; ++i) {
            NVNshaderDataExt raw{};
            const auto width = extended ? sizeof raw : sizeof(NVNshaderData);
            const auto offset = i * width;
            if (!read || !descriptors || descriptors > UINT64_MAX - offset ||
                !read(context, descriptors + offset, {reinterpret_cast<std::byte *>(&raw), width})) {
                input.incomplete.emplace_back("descriptor fields were not read");
                continue;
            }
            auto &descriptor = input.descriptors[i];
            descriptor.fields_read = true;
            descriptor.data_address = raw.data;
            descriptor.control_address = reinterpret_cast<std::uint64_t>(raw.control);
            descriptor.extra_address = extended ? reinterpret_cast<std::uint64_t>(raw.extra) : 0;
            if (descriptor.extra_address) {
                descriptor.extra.resize(extra_bytes);
                if (read(context, descriptor.extra_address, descriptor.extra))
                    descriptor.extra_read = true;
                else {
                    descriptor.extra.clear();
                    input.incomplete.emplace_back("extended metadata read failed");
                }
            }
        }
        current_call = this;
    } catch (...) {
        if (session)
            session->record_loss();
        state_.reset();
    }
}
Call::~Call() {
    if (borrowed_ || !state_)
        return;
    if (!state_->finished)
        finish(false, std::nullopt);
    current_call = parent_;
}
std::uint64_t Call::id() const noexcept {
    return borrowed_ ? parent_->id() : state_ ? state_->input.call : 0;
}
bool Call::matches() const noexcept {
    if (borrowed_)
        return parent_->matches();
    return state_ && (state_->finished ? state_->matched : state_->session->matches(state_->input));
}
void Call::finish(bool result, std::optional<std::uint64_t> generation) noexcept {
    if (borrowed_) {
        if (parent_ && parent_->state_ && generation) {
            parent_->state_->input.generation_after = generation;
            parent_->state_->input.generation_after_observed = true;
        }
        return;
    }
    if (!state_ || state_->finished)
        return;
    auto &input = state_->input;
    input.result = result;
    if (generation) {
        input.generation_after = generation;
        input.generation_after_observed = true;
    }
    if (!input.generation_after_observed) {
        try {
            ReadMemory read;
            void *context;
            {
                const std::lock_guard lock(global.mutex);
                read = global.read;
                context = global.context;
            }
            ProgramState object{};
            if (!input.program_address)
                input.generation_after_observed = true;
            else if (read && read(context, input.program_address,
                                  {reinterpret_cast<std::byte *>(&object), sizeof object})) {
                input.generation_after_observed = true;
                if (object.tag == object_tag('P', 'R', 'O', 'G'))
                    input.generation_after = object.serial;
            }
        } catch (...) {
            state_->observation_failed = true;
        }
    }
    try {
        if (state_->observation_failed)
            input.incomplete.emplace_back("input observation allocation failed");
    } catch (...) {
    }
    state_->matched = state_->session->matches(input);
    state_->finished = true;
    state_->session->submit(std::move(input));
}
Call *Call::active() noexcept { return enabled() ? current_call : nullptr; }
void Call::control(std::size_t index, std::span<const std::byte> value, std::uint64_t address) noexcept {
    if (borrowed_)
        return parent_->control(index, value, address);
    if (!state_ || state_->finished)
        return;
    try {
        if (index >= state_->input.descriptors.size()) {
            state_->input.incomplete.emplace_back("control index exceeds captured descriptor list");
            return;
        }
        auto &descriptor = state_->input.descriptors[index];
        if (address && descriptor.control_address != address) {
            state_->input.incomplete.emplace_back("control address changed during the call");
            descriptor.control_address = address;
        }
        descriptor.control.assign(value.begin(), value.begin() + std::min(value.size(), control_bytes));
        descriptor.control_read = value.size() == control_bytes;
    } catch (...) {
        state_->observation_failed = true;
    }
}
void Call::metadata(std::size_t index, std::uint32_t selector, std::optional<std::uint32_t> route,
                    std::optional<std::uint32_t> auxiliary) noexcept {
    if (borrowed_)
        return parent_->metadata(index, selector, route, auxiliary);
    if (!state_ || state_->finished)
        return;
    if (index >= state_->input.descriptors.size()) {
        state_->observation_failed = true;
        return;
    }
    auto &descriptor = state_->input.descriptors[index];
    descriptor.metadata_selector = selector;
    descriptor.metadata_route = route;
    descriptor.metadata_auxiliary = auxiliary;
}
void Call::code(std::size_t index, std::shared_ptr<const void> owner, std::span<const std::byte> value,
                std::string_view digest, std::string_view identity, std::uint64_t address) noexcept {
    if (borrowed_)
        return parent_->code(index, std::move(owner), value, digest, identity, address);
    if (!state_ || state_->finished)
        return;
    try {
        if (index >= state_->input.descriptors.size()) {
            state_->input.incomplete.emplace_back("code index exceeds captured descriptor list");
            return;
        }
        auto &descriptor = state_->input.descriptors[index];
        if (address && descriptor.data_address != address) {
            state_->input.incomplete.emplace_back("code address changed during the call");
            descriptor.data_address = address;
        }
        descriptor.code_owner = std::move(owner);
        descriptor.code = value;
        descriptor.code_read = true;
        descriptor.payload_digest = digest;
        descriptor.stage_identity = identity;
    } catch (...) {
        state_->observation_failed = true;
    }
}
void observe(Event event) noexcept {
    if (!enabled())
        return;
    try {
        if (auto session = active_session()) {
            session->observe(std::move(event));
            if (!session->accepting()) {
                const std::lock_guard lock(global.mutex);
                if (global.session == session)
                    global.active.store(false, std::memory_order_release);
            }
        }
    } catch (...) {
    }
}

void record_loss() noexcept {
    if (!enabled())
        return;
    try {
        if (auto session = active_session())
            session->record_loss();
    } catch (...) {
    }
}

} // namespace armigo::nvn::program_capture
