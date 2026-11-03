// ============================================================================
// Armigo Tests - Shader Stage Translation
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include "stage_fixture.h"

#include <armigo/shader/source_version.h>
#include <armigo/shader_translation/stage_cache.h>
#include <armigo/shader_translation/stage_translation.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>
#include <pthread.h>
#include <unistd.h>

namespace {
namespace fs = std::filesystem;
namespace tr = armigo::shader::translation;
namespace fixture = armigo::test::stage_fixture;
unsigned failures = 0;
#define CHECK(condition) do { if (!(condition)) { std::fprintf(stderr, "FAIL %d: %s\n", __LINE__, #condition); ++failures; } } while (0)

std::string failure_of(const fixture::Container& stage, const tr::StageOptions& options) {
    try {
        (void)tr::translate_stage(stage.control, stage.code, options);
    } catch (const std::runtime_error& error) {
        return error.what();
    }
    return {};
}

void test_translation() {
    const auto stage = fixture::fragment_stage();
    const auto first = tr::translate_stage(stage.control, stage.code, {});
    CHECK(first.stage == armigo::shader_input::Stage::Fragment);
    // The program is the stage's identity bytes: SPH and instructions, not C1.
    CHECK(std::equal(first.program.begin(), first.program.end(),
                     stage.code.begin() + static_cast<std::ptrdiff_t>(stage.program_begin),
                     stage.code.begin() + static_cast<std::ptrdiff_t>(stage.program_end)));
    CHECK(first.reflection.find("\"stage\":\"fragment\"") != std::string::npos);
    CHECK(first.reflection.find(std::string("\"source_version\":\"") + armigo::shader::source_version + '"') !=
          std::string::npos);
    CHECK(first.source.find("fragment") != std::string::npos);
    // Translation is deterministic.
    const auto second = tr::translate_stage(stage.control, stage.code, {});
    CHECK(second.source == first.source && second.reflection == first.reflection && second.program == first.program);
    // Translations on concurrent threads (as run-time preparation's workers
    // run them, each with a large stack) give the same results.
    {
        struct Worker {
            const fixture::Container* stage;
            const tr::TranslatedStage* expected;
            bool same = true;
            pthread_t thread{};
        };
        std::vector<Worker> workers(8);
        pthread_attr_t attributes;
        pthread_attr_init(&attributes);
        pthread_attr_setstacksize(&attributes, 64u << 20);
        unsigned started = 0;
        for (auto& worker : workers) {
            worker.stage = &stage;
            worker.expected = &first;
            started += pthread_create(&worker.thread, &attributes, [](void* raw) -> void* {
                auto& self = *static_cast<Worker*>(raw);
                for (unsigned round = 0; round != 4; ++round) {
                    try {
                        const auto again = tr::translate_stage(self.stage->control, self.stage->code, {});
                        self.same = self.same && again.source == self.expected->source &&
                                    again.reflection == self.expected->reflection &&
                                    again.program == self.expected->program;
                    } catch (...) {
                        self.same = false;
                    }
                }
                return nullptr;
            }, &worker) == 0;
        }
        pthread_attr_destroy(&attributes);
        CHECK(started == workers.size());
        for (auto& worker : workers) {
            if (worker.thread) pthread_join(worker.thread, nullptr);
            CHECK(worker.same);
        }
    }

    // Failures name their phase, as the offline translator reports them.
    auto broken = stage;
    fixture::put32(broken.control, 0, 0x12345678);
    CHECK(failure_of(broken, {}).rfind("parse", 0) == 0);
    tr::StageOptions vertex_options;
    vertex_options.emission.vertex_depth = armigo::shader::graphics_msl::VertexDepthConvention::ZeroToOne;
    CHECK(failure_of(stage, vertex_options) == "Vertex options are not fragment-stage options");
    tr::StageOptions compute_options;
    compute_options.compute_local_size = std::array<std::uint32_t, 3>{8, 8, 1};
    CHECK(failure_of(stage, compute_options) == "Compute launch options require a compute stage");
}

void test_cache() {
    const auto root = fs::temp_directory_path() / ("armigo-stage-cache-" + std::to_string(getpid()));
    fs::remove_all(root);
    const tr::StageCache cache(root);
    CHECK(cache.directory() == root / armigo::shader::source_version);

    const auto stage = fixture::fragment_stage();
    const auto key = tr::StageCache::key(stage.control, stage.code, "ported stage=1");
    CHECK(key.size() == 64 && key == tr::StageCache::key(stage.control, stage.code, "ported stage=1"));
    // The options, the control and the code each change the key.
    CHECK(key != tr::StageCache::key(stage.control, stage.code, "ported stage=1 fp=1"));
    auto other = stage;
    other.code.back() = std::byte{0x6e};
    CHECK(key != tr::StageCache::key(stage.control, other.code, "ported stage=1"));
    CHECK(key != tr::StageCache::key(other.code, stage.control, "ported stage=1"));

    CHECK(!cache.load(key) && cache.keys().empty());
    const auto translated = tr::translate_stage(stage.control, stage.code, {});
    CHECK(cache.store(key, translated));
    CHECK(fs::is_directory(cache.directory() / key));
    CHECK(cache.keys() == std::vector<std::string>{key});
    // The reflection and program come back; the source stays in its file.
    const auto loaded = cache.load(key);
    CHECK(loaded && loaded->reflection == translated.reflection && loaded->program == translated.program &&
          loaded->stage == armigo::shader_input::Stage::Fragment &&
          loaded->source_file == cache.directory() / key / "source.metal");
    if (loaded) {
        std::ifstream file(loaded->source_file, std::ios::binary);
        const std::string text((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
        CHECK(text == translated.source);
    }
    // A second store keeps the entry; no staging directory is left behind.
    CHECK(cache.store(key, translated));
    std::size_t entries = 0;
    for (const auto& item : fs::directory_iterator(cache.directory())) {
        ++entries;
        CHECK(item.path().filename().string().rfind(".partial-", 0) != 0);
    }
    CHECK(entries == 1);
    // An incomplete entry is not an entry; a malformed key names none.
    fs::remove(cache.directory() / key / "program.bin");
    CHECK(!cache.load(key));
    CHECK(!cache.load("../escape"));
    CHECK(!cache.store("ABC", translated));

    // Pruning removes other translator versions and stale staging directories,
    // and keeps this version's entries and fresh staging directories.
    const auto old_version = root / std::string(64, 'a');
    fs::create_directories(old_version / key);
    const auto stale = cache.directory() / (".partial-" + key + "-1");
    const auto fresh = cache.directory() / (".partial-" + key + "-2");
    fs::create_directories(stale);
    fs::create_directories(fresh);
    fs::last_write_time(stale, fs::file_time_type::clock::now() - std::chrono::hours(1));
    const auto unrelated = root / "notes";
    fs::create_directories(unrelated);
    cache.prune(std::chrono::minutes(10));
    CHECK(!fs::exists(old_version) && !fs::exists(stale) && fs::exists(fresh) && fs::exists(unrelated) &&
          fs::is_directory(cache.directory() / key));
    fs::remove_all(root);
}
} // namespace

int main() {
    test_translation();
    test_cache();
    if (failures == 0) std::puts("Stage translation checks passed");
    return failures == 0 ? 0 : 1;
}
