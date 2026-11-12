// ============================================================================
// Armigo Tests - Native Runtime Verification
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
#include "package_internal.h"
#include <atomic>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <mutex>
#include <thread>
#include <unistd.h>
#include <sys/stat.h>

namespace pc = armigo::nvn::program_capture;
namespace json = armigo::shader::package::detail;
namespace {
unsigned failures = 0;
#define CHECK(x)                                                                                             \
    do {                                                                                                     \
        if (!(x)) {                                                                                          \
            std::fprintf(stderr, "FAIL %d: %s\n", __LINE__, #x);                                             \
            ++failures;                                                                                      \
        }                                                                                                    \
    } while (false)
const std::string digest = "9f64a747e1b97f131fabb6b447296c9b6f0201e79fb3c5356e6c77e89b6a806a";
struct Temporary {
    std::filesystem::path path;
    Temporary() {
        std::string pattern =
            (std::filesystem::temp_directory_path() / "armigo-program-capture-XXXXXX").string();
        std::vector<char> text(pattern.begin(), pattern.end());
        text.push_back(0);
        if (auto *made = mkdtemp(text.data()))
            path = made;
        CHECK(!path.empty());
    }
    ~Temporary() {
        if (!path.empty())
            std::filesystem::remove_all(path);
    }
};
std::string read(const std::filesystem::path &path) {
    std::ifstream stream(path, std::ios::binary);
    return {std::istreambuf_iterator<char>(stream), {}};
}
std::shared_ptr<const std::vector<std::byte>> payload() {
    return std::make_shared<const std::vector<std::byte>>(
        std::initializer_list<std::byte>{std::byte{1}, std::byte{2}, std::byte{3}, std::byte{4}});
}
pc::Inputs inputs(std::uint64_t call = 1, std::uint64_t generation = 42) {
    pc::Inputs out;
    out.call = call;
    out.program_address = 0x1000;
    out.count = 1;
    out.result = true;
    out.generation_before_observed = out.generation_after_observed = true;
    out.api_major = 231;
    out.api_minor = 40;
    out.generation_before = generation - 1;
    out.generation_after = generation;
    pc::Descriptor descriptor;
    descriptor.data_address = 0x2000;
    descriptor.control_address = 0x3000;
    descriptor.fields_read = descriptor.control_read = descriptor.code_read = true;
    descriptor.control.resize(0xe0);
    for (std::size_t i = 0; i != descriptor.control.size(); ++i)
        descriptor.control[i] = std::byte(i);
    const std::uint32_t extent = 4;
    std::memcpy(descriptor.control.data() + 0x44, &extent, 4);
    auto code = payload();
    descriptor.code_owner = code;
    descriptor.code = *code;
    descriptor.payload_digest = digest;
    descriptor.stage_identity = std::string(64, 'b');
    out.descriptors.push_back(std::move(descriptor));
    return out;
}
pc::Event event(pc::EventKind kind, std::uint64_t call = 1, std::uint64_t generation = 42) {
    pc::Event out;
    out.kind = kind;
    out.backend = 7;
    out.submission = 13;
    out.ordinal = 17;
    out.capture_call = call;
    out.generation = generation;
    out.matches = true;
    out.first = 0;
    out.count = 32;
    out.requested = 255;
    out.mode_after = 2;
    out.api_return = 0x1234;
    return out;
}
pc::Configuration config(const std::filesystem::path &output = {}) {
    pc::Configuration out;
    out.output = output;
    out.payload_digests = {digest};
    out.first = 0;
    out.count = 32;
    out.stop_after_capture = true;
    return out;
}
void trigger(pc::Session &session, std::uint64_t call = 1, std::uint64_t generation = 42) {
    session.observe(event(pc::EventKind::bind, call, generation));
    session.observe(event(pc::EventKind::mode, call, generation));
    session.observe(event(pc::EventKind::direct_draw, call, generation));
}
struct RecordingSink : pc::Sink {
    std::vector<pc::Inputs> programs;
    std::optional<pc::Capture> capture;
    bool input_failure = false, final_failure = false;
    pc::WriteResult inputs(const pc::Inputs &input) override {
        programs.push_back(input);
        return {!input_failure, input_failure ? "synthetic input write failure" : ""};
    }
    pc::WriteResult final(const pc::Capture &value) override {
        capture = value;
        return {!final_failure, final_failure ? "synthetic manifest write failure" : ""};
    }
};
void test_durable_files() {
    Temporary temp;
    const auto output = temp.path / "capture";
    pc::Session session(config(output));
    const auto original = inputs();
    std::atomic_uint called{0};
    const auto caller = std::this_thread::get_id();
    session.set_completion([&] {
        CHECK(std::this_thread::get_id() != caller);
        CHECK(read(output / "call-1/code-0.bin") == std::string("\1\2\3\4", 4));
        CHECK(read(output / "call-1/control-0.bin") ==
              std::string(reinterpret_cast<const char *>(original.descriptors[0].control.data()), 0xe0));
        CHECK(std::filesystem::exists(output / "capture.json"));
        ++called;
    });
    session.submit(original);
    trigger(session);
    CHECK(session.finish(std::chrono::seconds(5)));
    CHECK(session.completed() && called.load() == 1);
    const auto text = read(output / "capture.json");
    const auto parsed = json::parse_json({reinterpret_cast<const std::byte *>(text.data()), text.size()});
    CHECK(parsed.object().at("inputs_complete").boolean());
    CHECK(parsed.object().at("writer_completion_required").boolean());
    CHECK(!parsed.object().at("original_driver_execution_observed").boolean());
    CHECK(parsed.object().at("events").array().size() == 3);
    const auto &first = parsed.object().at("programs").array()[0].object();
    CHECK(json::uint_value(first.at("count")) == 1);
    CHECK(first.at("generation_after").string() == "42");
    CHECK(first.at("descriptors").array()[0].object().at("code_file").string() == "call-1/code-0.bin");
    struct stat status{};
    CHECK(::stat((output / "capture.json").c_str(), &status) == 0 && (status.st_mode & 0777) == 0600);
}
void test_incomplete_and_failures() {
    for (unsigned mode = 0; mode != 8; ++mode) {
        auto sink = std::make_shared<RecordingSink>();
        auto settings = config();
        if (mode == 2)
            settings.limits.payload_bytes = 3;
        if (mode == 3)
            settings.limits.generations = 1;
        if (mode == 4)
            settings.limits.events = 2;
        if (mode == 5)
            settings.limits.descriptors = 1;
        sink->input_failure = mode == 0;
        sink->final_failure = mode == 1;
        pc::Session session(settings, sink);
        std::atomic_uint called{0};
        session.set_completion([&] { ++called; });
        auto input = inputs();
        if (mode == 5) {
            input.count = 2;
            input.descriptors.push_back(input.descriptors[0]);
        }
        if (mode == 6) {
            input.extended = true;
            input.descriptors[0].extra_address = 0x4000;
        }
        if (mode == 7)
            input.result = false;
        session.submit(std::move(input));
        if (mode == 3) {
            session.submit(inputs(2, 43));
            trigger(session, 2, 43);
        } else
            trigger(session);
        CHECK(session.finish(std::chrono::seconds(5)));
        CHECK(!session.completed() && called.load() == 0);
        CHECK(sink->capture.has_value());
        if (mode != 1)
            CHECK(!sink->capture->complete);
    }
    Temporary temp;
    const auto existing = temp.path / "existing";
    std::filesystem::create_directory(existing);
    pc::Session session(config(existing));
    std::atomic_uint called{0};
    session.set_completion([&] { ++called; });
    session.submit(inputs());
    trigger(session);
    CHECK(session.finish(std::chrono::seconds(5)));
    CHECK(!session.completed() && called.load() == 0 && !std::filesystem::exists(existing / "capture.json"));
}
void test_latest_bind_and_missing_draw() {
    auto sink = std::make_shared<RecordingSink>();
    pc::Session session(config(), sink);
    session.submit(inputs());
    session.observe(event(pc::EventKind::bind));
    for (unsigned i = 0; i != 300; ++i)
        session.observe(event(pc::EventKind::mode));
    trigger(session);
    CHECK(session.finish(std::chrono::seconds(5)) && session.completed());
    CHECK(sink->capture && sink->capture->events.size() == 3 && sink->capture->omitted_events == 0 &&
          sink->capture->unselected_windows == 1);
    auto missing = std::make_shared<RecordingSink>();
    pc::Session no_draw(config(), missing);
    no_draw.submit(inputs());
    CHECK(no_draw.finish(std::chrono::seconds(5)) && !no_draw.completed());
    CHECK(missing->capture && !missing->capture->complete);
}
void test_queued_owner_and_disconnect() {
    struct BlockingSink final : RecordingSink {
        std::mutex mutex;
        std::condition_variable changed;
        bool entered = false, released = false;
        std::string observed;
        pc::WriteResult inputs(const pc::Inputs &input) override {
            std::unique_lock lock(mutex);
            entered = true;
            changed.notify_all();
            changed.wait(lock, [&] { return released; });
            observed.assign(reinterpret_cast<const char *>(input.descriptors[0].code.data()),
                            input.descriptors[0].code.size());
            return {true, {}};
        }
    };
    auto sink = std::make_shared<BlockingSink>();
    auto session = std::make_unique<pc::Session>(config(), sink);
    auto input = inputs();
    std::weak_ptr<const void> weak = input.descriptors[0].code_owner;
    std::atomic_uint called{0};
    session->set_completion([&] { ++called; });
    session->submit(std::move(input));
    trigger(*session);
    {
        std::unique_lock lock(sink->mutex);
        CHECK(sink->changed.wait_for(lock, std::chrono::seconds(5), [&] { return sink->entered; }));
    }
    CHECK(!weak.expired());
    session->disconnect_completion();
    {
        const std::lock_guard lock(sink->mutex);
        sink->released = true;
    }
    sink->changed.notify_all();
    CHECK(session->finish(std::chrono::seconds(5)) && session->completed());
    CHECK(sink->observed == std::string("\1\2\3\4", 4) && called.load() == 0);
    sink->capture.reset();
    session.reset();
    CHECK(weak.expired());
}
void test_calls_ext_and_deduplication() {
    Temporary temp;
    struct Memory {
        struct Range {
            std::uint64_t address;
            const std::byte *data;
            std::size_t size;
        };
        std::vector<Range> ranges;
        unsigned reads = 0;
        void add(const void *data, std::size_t size) {
            ranges.push_back(
                {reinterpret_cast<std::uint64_t>(data), static_cast<const std::byte *>(data), size});
        }
        static bool read(void *raw, std::uint64_t address, std::span<std::byte> target) {
            auto &memory = *static_cast<Memory *>(raw);
            ++memory.reads;
            for (const auto &range : memory.ranges)
                if (address >= range.address && address - range.address <= range.size &&
                    target.size() <= range.size - (address - range.address)) {
                    std::memcpy(target.data(), range.data + (address - range.address), target.size());
                    return true;
                }
            return false;
        }
    } memory;
    armigo::nvn::select_api_version({231, 40});
    armigo::nvn::ProgramState program{};
    program.tag = armigo::nvn::object_tag('P', 'R', 'O', 'G');
    program.serial = 5;
    auto code = payload();
    auto control = inputs().descriptors[0].control;
    std::array<std::byte, 24> extra{};
    extra.fill(std::byte{0x5a});
    const armigo::nvn::api::NVNshaderDataExt descriptors[]{
        {reinterpret_cast<std::uint64_t>(code->data()), control.data(), extra.data()},
        {reinterpret_cast<std::uint64_t>(code->data()), control.data(), extra.data()}};
    const armigo::nvn::api::NVNshaderData basic[]{{descriptors[0].data, descriptors[0].control},
                                                  {descriptors[1].data, descriptors[1].control}};
    memory.add(&program, sizeof program);
    memory.add(descriptors, sizeof descriptors);
    memory.add(basic, sizeof basic);
    memory.add(extra.data(), extra.size());
    pc::configure(config(temp.path / "capture"), Memory::read, &memory);
    std::atomic_uint stopped{0};
    std::mutex stopped_mutex;
    std::condition_variable stopped_changed;
    auto binding = pc::bind_run([&] {
        ++stopped;
        stopped_changed.notify_all();
    });
    std::uint64_t first = 0, second = 0;
    {
        pc::Call outer(reinterpret_cast<std::uint64_t>(&program), 2,
                       reinterpret_cast<std::uint64_t>(descriptors), true);
        first = outer.id();
        {
            pc::Call inner(reinterpret_cast<std::uint64_t>(&program), 2,
                           reinterpret_cast<std::uint64_t>(basic), false);
            CHECK(inner.id() == first && pc::Call::active() == &outer);
            for (unsigned i = 0; i != 2; ++i) {
                inner.control(i, control);
                inner.code(i, code, *code, digest, std::string(64, 'b'));
            }
            inner.finish(true, 6);
        }
        CHECK(outer.matches());
        outer.finish(true, 6);
    }
    program.serial = 6;
    control[0x30] = std::byte{0};
    {
        pc::Call call(reinterpret_cast<std::uint64_t>(&program), 2, reinterpret_cast<std::uint64_t>(basic),
                      false);
        second = call.id();
        CHECK(second != first);
        for (unsigned i = 0; i != 2; ++i) {
            call.control(i, control);
            call.code(i, code, *code, digest, std::string(64, 'b'));
        }
        call.finish(true, 7);
    }
    pc::observe(event(pc::EventKind::bind, second, 7));
    pc::observe(event(pc::EventKind::direct_draw, second, 7));
    {
        std::unique_lock lock(stopped_mutex);
        CHECK(stopped_changed.wait_for(lock, std::chrono::seconds(5), [&] { return stopped.load() == 1; }));
    }
    binding.finish(true);
    CHECK(stopped.load() == 1 && !pc::enabled() && memory.reads == 8);
    const auto path = temp.path / "capture";
    CHECK(read(path / ("call-" + std::to_string(first)) / "extra-0.bin") == std::string(24, char(0x5a)));
    CHECK(read(path / ("call-" + std::to_string(first)) / "control-0.bin")[0x30] == char(0x30));
    CHECK(read(path / ("call-" + std::to_string(second)) / "control-0.bin")[0x30] == 0);
    const auto text = read(path / "capture.json");
    const auto parsed = json::parse_json({reinterpret_cast<const std::byte *>(text.data()), text.size()});
    const auto &programs = parsed.object().at("programs").array();
    CHECK(programs.size() == 2);
    CHECK(programs[0].object().at("kind").string() == "SetShadersExt");
    CHECK(programs[1].object().at("kind").string() == "SetShaders");
    CHECK(json::uint_value(programs[0].object().at("api_major")) == 231);
    CHECK(programs[0].object().at("descriptors").array().size() == 2);
    unsetenv("ARMIGO_NVN_PROGRAM_CAPTURE_CONFIG");
    pc::configure_from_environment(nullptr);
}

void test_failed_tracked_call() {
    auto sink = std::make_shared<RecordingSink>();
    pc::Session session(config(), sink);
    std::atomic_uint called{0};
    session.set_completion([&] { ++called; });
    session.submit(inputs());
    auto failed = inputs(2, 42);
    failed.generation_before = 42;
    failed.generation_after.reset();
    failed.generation_after_observed = false;
    failed.result = false;
    failed.count = 3;
    failed.descriptors.assign(3, {});
    session.submit(std::move(failed));
    trigger(session);
    CHECK(session.finish(std::chrono::seconds(5)) && !session.completed() && called.load() == 0);
    CHECK(sink->capture && sink->capture->programs.size() == 2);
    const auto &retained = sink->capture->programs[1];
    CHECK(retained.call == 2 && !retained.result && retained.count == 3 && retained.descriptors.size() == 3);
    CHECK(retained.generation_before == 42 && !retained.generation_after && !retained.incomplete.empty());
    CHECK(retained.result_sequence > sink->capture->programs[0].result_sequence);
    CHECK(retained.tracked_predecessor && !retained.payload_filter_match);
}

void test_environment_run_lifetime() {
    Temporary temp;
    const auto file = temp.path / "config.txt";
    {
        std::ofstream out(file);
        out << "version=1\noutput=" << (temp.path / "capture").string() << "\npayload_digest=" << digest
            << "\n";
    }
    setenv("ARMIGO_NVN_PROGRAM_CAPTURE_CONFIG", file.c_str(), 1);
    pc::configure_from_environment([](void *, std::uint64_t, std::span<std::byte>) { return false; });
    auto first = pc::bind_run([] {});
    CHECK(pc::enabled());
    first.finish(true);
    unsetenv("ARMIGO_NVN_PROGRAM_CAPTURE_CONFIG");
    auto disabled = pc::bind_run([] {});
    CHECK(!pc::enabled());
    disabled.finish(true);
    setenv("ARMIGO_NVN_PROGRAM_CAPTURE_CONFIG", (temp.path / "missing").c_str(), 1);
    auto invalid = pc::bind_run([] {});
    CHECK(!pc::enabled());
    invalid.finish(true);
    unsetenv("ARMIGO_NVN_PROGRAM_CAPTURE_CONFIG");
    pc::configure_from_environment(nullptr);
}

void test_config_and_disabled() {
    Temporary temp;
    const auto file = temp.path / "config.txt";
    {
        std::ofstream out(file);
        out << "version=1\noutput=" << (temp.path / "capture").string() << "\npayload_digest=" << digest
            << "\nfirst=0\ncount=32\nstop_after_capture=1\n";
    }
    std::string error;
    auto settings = pc::read_configuration(file, error);
    CHECK(settings && settings->first == 0 && settings->count == 32 && settings->stop_after_capture);
    const auto base_configuration = read(file);
    for (const auto extra : {"max_events=257\n", "max_descriptors=10\n", "unknown=1\n"}) {
        {
            std::ofstream out(file);
            out << base_configuration << extra;
        }
        CHECK(!pc::read_configuration(file, error));
    }
    unsigned reads = 0;
    pc::configure(
        config(temp.path / "disabled"),
        [](void *raw, std::uint64_t, std::span<std::byte>) {
            ++*static_cast<unsigned *>(raw);
            return false;
        },
        &reads);
    {
        pc::Call call(1, 2, 3, true);
        CHECK(call.id() == 0);
        CHECK(pc::Call::active() == nullptr);
    }
    pc::observe(event(pc::EventKind::direct_draw));
    CHECK(reads == 0 && !pc::enabled() && !std::filesystem::exists(temp.path / "disabled"));
    unsetenv("ARMIGO_NVN_PROGRAM_CAPTURE_CONFIG");
    pc::configure_from_environment(nullptr);
}
} // namespace
int main() {
    test_durable_files();
    test_incomplete_and_failures();
    test_latest_bind_and_missing_draw();
    test_queued_owner_and_disconnect();
    test_failed_tracked_call();
    test_config_and_disabled();
    test_calls_ext_and_deduplication();
    test_environment_run_lifetime();
    return failures ? 1 : 0;
}
