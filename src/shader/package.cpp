// ============================================================================
// Armigo Shader - Native Shader Translation
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include <armigo/shader/package.h>

#include "package_internal.h"

#include <CommonCrypto/CommonDigest.h>

#include <sys/wait.h>
#include <unistd.h>
#include <uuid/uuid.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <csignal>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <fcntl.h>
#include <fstream>
#include <functional>
#include <future>
#include <mutex>
#include <optional>
#include <set>
#include <spawn.h>
#include <stdexcept>
#include <thread>
#include <tuple>

extern char** environ;

namespace armigo::shader::package {
namespace {

namespace fs = std::filesystem;
using detail::Json;
using detail::JsonNumber;
using Clock = std::chrono::steady_clock;

constexpr auto command_timeout = std::chrono::seconds(60);
const std::array<std::string, 7> compile_flags{
    "-std=metal4.0", "-mmacosx-version-min=26.0", "-fwrapv", "-Werror", "-Wno-unused-variable",
    "-Wno-sometimes-uninitialized", "-fbracket-depth=1024"};

std::string trim(std::string value) {
    const auto first = value.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) return {};
    const auto last = value.find_last_not_of(" \t\r\n");
    return value.substr(first, last - first + 1);
}

std::string first_line(const std::string& value, std::string_view label) {
    const auto end = value.find_first_of("\r\n");
    const std::string line = trim(value.substr(0, end));
    detail::require(!line.empty(), std::string(label) + " returned no version");
    return line;
}

struct ProcessResult {
    int status = -1;
    std::string output;
    std::string error;
    bool timed_out = false;
};

void read_pipe(int descriptor, std::string& output) {
    std::array<char, 16384> buffer{};
    for (;;) {
        const auto count = ::read(descriptor, buffer.data(), buffer.size());
        if (count > 0) output.append(buffer.data(), static_cast<std::size_t>(count));
        else if (count == 0 || (count < 0 && errno != EINTR)) break;
    }
    ::close(descriptor);
}

ProcessResult run_process(const std::vector<std::string>& arguments, bool c_locale = false) {
    detail::require(!arguments.empty(), "Cannot execute an empty shader tool command");
    int output_pipe[2] = {-1, -1}, error_pipe[2] = {-1, -1};
    detail::require(::pipe(output_pipe) == 0 && ::pipe(error_pipe) == 0, "Cannot execute Apple shader tool");
    posix_spawn_file_actions_t actions;
    detail::require(posix_spawn_file_actions_init(&actions) == 0, "Cannot execute Apple shader tool");
    posix_spawn_file_actions_addclose(&actions, output_pipe[0]);
    posix_spawn_file_actions_addclose(&actions, error_pipe[0]);
    posix_spawn_file_actions_adddup2(&actions, output_pipe[1], STDOUT_FILENO);
    posix_spawn_file_actions_adddup2(&actions, error_pipe[1], STDERR_FILENO);
    posix_spawn_file_actions_addclose(&actions, output_pipe[1]);
    posix_spawn_file_actions_addclose(&actions, error_pipe[1]);
    std::vector<char*> argv;
    argv.reserve(arguments.size() + 1);
    for (const std::string& argument : arguments) argv.push_back(const_cast<char*>(argument.c_str()));
    argv.push_back(nullptr);
    std::vector<std::string> environment;
    std::vector<char*> environment_pointer;
    char** child_environment = environ;
    if (c_locale) {
        for (char** item = environ; *item != nullptr; ++item) {
            if (std::string_view(*item).starts_with("LC_ALL=")) continue;
            environment.emplace_back(*item);
        }
        environment.emplace_back("LC_ALL=C");
        for (std::string& item : environment) environment_pointer.push_back(item.data());
        environment_pointer.push_back(nullptr);
        child_environment = environment_pointer.data();
    }
    pid_t pid = -1;
    const int spawned = posix_spawnp(&pid, arguments[0].c_str(), &actions, nullptr, argv.data(), child_environment);
    posix_spawn_file_actions_destroy(&actions);
    ::close(output_pipe[1]);
    ::close(error_pipe[1]);
    if (spawned != 0) {
        ::close(output_pipe[0]);
        ::close(error_pipe[0]);
        throw std::runtime_error("Cannot execute Apple shader tool");
    }
    ProcessResult result;
    std::thread output_reader(read_pipe, output_pipe[0], std::ref(result.output));
    std::thread error_reader(read_pipe, error_pipe[0], std::ref(result.error));
    const auto deadline = Clock::now() + command_timeout;
    int status = 0;
    for (;;) {
        const pid_t waited = ::waitpid(pid, &status, WNOHANG);
        detail::require(waited >= 0, "Cannot wait for Apple shader tool");
        if (waited == pid) break;
        if (Clock::now() >= deadline) {
            result.timed_out = true;
            ::kill(pid, SIGKILL);
            detail::require(::waitpid(pid, &status, 0) == pid, "Cannot wait for Apple shader tool");
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    output_reader.join();
    error_reader.join();
    result.status = WIFEXITED(status) ? WEXITSTATUS(status) : -1;
    return result;
}

std::string query_raw(const std::vector<std::string>& arguments) {
    const ProcessResult result = run_process(arguments);
    if (result.timed_out || result.status != 0) {
        std::string joined;
        for (const std::string& argument : arguments) {
            if (!joined.empty()) joined += ' ';
            joined += argument;
        }
        throw std::runtime_error("Shader tool query failed: " + joined + ": " + trim(result.error).substr(0, 4096));
    }
    return result.output;
}

std::string query(const std::vector<std::string>& arguments) {
    return trim(query_raw(arguments));
}

void run_apple_tool(const std::vector<std::string>& arguments) {
    const ProcessResult result = run_process(arguments, true);
    if (result.timed_out) throw std::runtime_error("Apple shader tool timed out");
    if (result.status != 0) throw std::runtime_error("Apple shader tool failed: " + trim(result.error).substr(0, 4096));
}

std::string random_hex() {
    uuid_t identifier;
    uuid_generate_random(identifier);
    static constexpr char digits[] = "0123456789abcdef";
    std::string result;
    result.reserve(32);
    for (const unsigned char byte : identifier) {
        result += digits[byte >> 4];
        result += digits[byte & 15];
    }
    return result;
}

std::span<const std::byte> as_bytes(std::string_view text) {
    return {reinterpret_cast<const std::byte*>(text.data()), text.size()};
}

class Sha256 {
public:
    Sha256() { detail::require(CC_SHA256_Init(&context_) == 1, "System SHA-256 failed"); }
    void update(std::span<const std::byte> data) {
        detail::require(CC_SHA256_Update(&context_, data.data(), static_cast<CC_LONG>(data.size())) == 1,
                        "System SHA-256 failed");
    }
    void update(std::string_view text) { update(as_bytes(text)); }
    std::array<std::byte, CC_SHA256_DIGEST_LENGTH> finish() {
        std::array<std::byte, CC_SHA256_DIGEST_LENGTH> result{};
        detail::require(CC_SHA256_Final(reinterpret_cast<unsigned char*>(result.data()), &context_) == 1,
                        "System SHA-256 failed");
        return result;
    }
private:
    CC_SHA256_CTX context_{};
};

std::string hex(std::span<const std::byte> bytes) {
    static constexpr char digits[] = "0123456789abcdef";
    std::string result;
    for (const std::byte byte : bytes) {
        const auto value = std::to_integer<unsigned>(byte);
        result += digits[value >> 4];
        result += digits[value & 15];
    }
    return result;
}

std::array<std::byte, 32> compiler_identity() {
    static std::mutex mutex;
    static std::optional<std::array<std::byte, 32>> cached;
    const std::lock_guard lock(mutex);
    if (!cached) {
        Sha256 hasher;
        for (const std::string_view tool : {"metal", "metallib"}) {
            const std::string path = first_line(query({"xcrun", "--sdk", "macosx", "--find", std::string(tool)}),
                                                "xcrun --find " + std::string(tool));
            std::ifstream stream(path, std::ios::binary);
            detail::require(stream.good(), "Cannot hash Apple shader tool");
            std::array<char, 64 * 1024> block{};
            while (stream) {
                stream.read(block.data(), static_cast<std::streamsize>(block.size()));
                const auto count = stream.gcount();
                if (count > 0) hasher.update({reinterpret_cast<const std::byte*>(block.data()), static_cast<std::size_t>(count)});
            }
            detail::require(stream.eof(), "Cannot hash Apple shader tool");
        }
        hasher.update(query_raw({"xcrun", "--sdk", "macosx", "metal", "--version"}));
        cached = hasher.finish();
    }
    return *cached;
}

Json current_profile(const fs::path& translate_tool) {
    const std::string source_version = first_line(query({translate_tool.string(), "--version"}), "translator");
    detail::require(source_version.size() == 64 &&
                        std::all_of(source_version.begin(), source_version.end(), [](char character) {
                            return (character >= '0' && character <= '9') || (character >= 'a' && character <= 'f');
                        }),
                    "Translator source version must be 64 lowercase hexadecimal digits");
    const std::string metal = first_line(query({"xcrun", "--sdk", "macosx", "--find", "metal"}), "xcrun --find metal");
    const std::string metallib = first_line(query({"xcrun", "--sdk", "macosx", "--find", "metallib"}),
                                            "xcrun --find metallib");
    Json::Object profile;
    profile.emplace("compile_options", Json::Array{Json{std::string("-std=metal4.0")},
                                                   Json{std::string("-mmacosx-version-min=26.0")},
                                                   Json{std::string("-fwrapv")},
                                                   Json{std::string("-Werror")}});
    profile.emplace("deployment_target", Json{std::string("26.0")});
    profile.emplace("identity", Json{std::string("span-prefix112-v1")});
    profile.emplace("metal_binary_sha256", Json{detail::sha256_file(metal)});
    profile.emplace("metal_language", Json{std::string("metal4.0")});
    profile.emplace("metal_version", Json{first_line(query({"xcrun", "--sdk", "macosx", "metal", "--version"}), "metal")});
    profile.emplace("metallib_binary_sha256", Json{detail::sha256_file(metallib)});
    profile.emplace("platform", Json{std::string("macos-metal4")});
    profile.emplace("sdk_build", Json{first_line(query({"xcrun", "--sdk", "macosx", "--show-sdk-build-version"}),
                                                    "macOS SDK")});
    profile.emplace("sdk_version", Json{first_line(query({"xcrun", "--sdk", "macosx", "--show-sdk-version"}),
                                                      "macOS SDK")});
    profile.emplace("source_version", Json{source_version});
    profile.emplace("translation_profile", Json{std::string("bounded-float3-clip-depth-positionw-v1")});
    return Json{std::move(profile)};
}

std::optional<fs::path> compile_cache_path(std::span<const std::byte> source,
                                            const std::optional<fs::path>& configured_root) {
    const char* environment_root = std::getenv("ARMIGO_SHADER_COMPILE_CACHE");
    const std::optional<fs::path> root = configured_root
        ? configured_root
        : (environment_root != nullptr && *environment_root != '\0' ? std::optional<fs::path>(environment_root)
                                                                    : std::nullopt);
    if (!root) return std::nullopt;
    Sha256 hasher;
    const auto identity = compiler_identity();
    hasher.update(identity);
    for (std::size_t index = 0; index < compile_flags.size(); ++index) {
        if (index != 0) hasher.update(std::string_view("\0", 1));
        hasher.update(compile_flags[index]);
    }
    hasher.update(std::string_view("\0", 1));
    hasher.update(source);
    return *root / (hex(hasher.finish()) + ".metallib");
}

struct Compiled {
    std::string variant_id;
    fs::path library;
    std::uint64_t size = 0;
    std::string digest;
};

std::uint64_t regular_size(const fs::path& path, std::string_view message) {
    std::error_code error;
    const bool valid = fs::is_regular_file(path, error) && !fs::is_symlink(path, error);
    const auto size = valid ? fs::file_size(path, error) : 0;
    detail::require(valid && !error && size > 0 && size <= detail::maximum_library_size, message);
    return size;
}

// Libraries of earlier packages by the digest of the Metal source they were
// compiled from, with the library digest their manifest records.
struct ReusedLibrary {
    fs::path library;
    std::string digest;
};
using ReusedLibraries = std::map<std::string, ReusedLibrary, std::less<>>;

ReusedLibraries reusable_libraries(const std::vector<fs::path>& packages, const Json& profile) {
    ReusedLibraries libraries;
    Json::Object compiler = profile.object();
    compiler.erase("source_version");
    for (const fs::path& package : packages) {
        const auto manifest = detail::parse_json(
            detail::read_regular(package / "manifest.json", detail::maximum_output_bytes, "reused package manifest"));
        const auto& text = detail::member(manifest, "profile_text").string();
        auto reused = detail::parse_json(as_bytes(text)).object();
        reused.erase("source_version");
        detail::require(reused == compiler, "A reused package was built with another compiler profile");
        for (const Json& entry : detail::member(manifest, "entries").array()) {
            const fs::path directory = package / detail::member(entry, "directory").string();
            // The manifest binds each reflection, which names the digest of the
            // source its library was compiled from.
            const auto reflection_bytes = detail::read_regular(
                directory / "reflection.json", detail::maximum_reflection_size, "reused reflection.json");
            detail::require(detail::sha256(reflection_bytes) == detail::member(entry, "reflection_digest").string(),
                            "A reused reflection does not match its manifest");
            const auto reflection = detail::parse_json(reflection_bytes);
            const auto& source_digest = detail::member(reflection, "source_digest").string();
            libraries.try_emplace(source_digest, ReusedLibrary{directory / "shader.metallib",
                                                               detail::member(entry, "library_digest").string()});
        }
    }
    return libraries;
}

Compiled compile_stage(const detail::Stage& item, const fs::path& work_root,
                       const std::optional<fs::path>& cache_root, const ReusedLibraries& reused,
                       std::atomic<std::size_t>& reused_count) {
    const fs::path work = work_root / item.variant_id;
    fs::create_directory(work);
    const fs::path source = work / "source.metal";
    const fs::path air = work / "shader.air";
    const fs::path library = work / "shader.metallib";
    if (const auto found = reused.find(detail::sha256(item.source)); found != reused.end()) {
        std::error_code error;
        fs::copy_file(found->second.library, library, fs::copy_options::none, error);
        detail::require(!error, "Cannot copy a reused Metal library");
        const auto size = regular_size(library, "A reused Metal library violates the size policy");
        const auto digest = detail::sha256_file(library);
        detail::require(digest == found->second.digest, "A reused Metal library does not match its manifest");
        reused_count.fetch_add(1);
        return Compiled{item.variant_id, library, size, digest};
    }
    detail::write_file(source, item.source);
    const auto cached = compile_cache_path(item.source, cache_root);
    if (cached && fs::is_regular_file(*cached) && !fs::is_symlink(*cached)) {
        std::error_code error;
        const auto size = fs::file_size(*cached, error);
        if (!error && size > 0 && size <= detail::maximum_library_size) {
            fs::copy_file(*cached, library, fs::copy_options::none, error);
            detail::require(!error, "Cannot copy cached Metal library");
            return Compiled{item.variant_id, library, size, detail::sha256_file(library)};
        }
    }
    static const std::string metal = first_line(query({"xcrun", "--sdk", "macosx", "--find", "metal"}),
                                                "xcrun --find metal");
    static const std::string metallib = first_line(query({"xcrun", "--sdk", "macosx", "--find", "metallib"}),
                                                   "xcrun --find metallib");
    std::vector<std::string> compile{metal};
    compile.insert(compile.end(), compile_flags.begin(), compile_flags.end());
    compile.insert(compile.end(), {"-c", source.string(), "-o", air.string()});
    run_apple_tool(compile);
    static_cast<void>(regular_size(air, "Metal AIR output violates the size policy"));
    run_apple_tool({metallib, air.string(), "-o", library.string()});
    const auto size = regular_size(library, "metallib output violates the size policy");
    if (cached) {
        std::error_code error;
        fs::create_directories(cached->parent_path(), error);
        detail::require(!error, "Cannot create shader compile cache");
        const fs::path temporary = cached->parent_path() / (cached->filename().string() + "." + random_hex() + ".tmp");
        fs::copy_file(library, temporary, fs::copy_options::none, error);
        detail::require(!error, "Cannot cache Metal library");
        fs::rename(temporary, *cached, error);
        if (error) {
            fs::remove(temporary);
            throw std::runtime_error("Cannot cache Metal library");
        }
    }
    return Compiled{item.variant_id, library, size, detail::sha256_file(library)};
}

class ThreadPool {
public:
    ThreadPool(unsigned count, const std::optional<fs::path>& cache_root, const ReusedLibraries& reused)
        : cache_root_(cache_root), reused_(reused) {
        detail::require(count >= 1 && count <= 8, "--jobs must be between 1 and 8");
        for (unsigned index = 0; index < count; ++index) workers_.emplace_back([this] { work(); });
    }
    ~ThreadPool() {
        {
            const std::lock_guard lock(mutex_);
            stopping_ = true;
        }
        condition_.notify_all();
        for (std::thread& worker : workers_) worker.join();
    }
    std::future<Compiled> submit(const detail::Stage& stage, const fs::path& work) {
        auto task = std::make_shared<std::packaged_task<Compiled()>>(
            [this, &stage, work] { return compile_stage(stage, work, cache_root_, reused_, reused_count); });
        std::future<Compiled> result = task->get_future();
        {
            const std::lock_guard lock(mutex_);
            tasks_.push_back([task] { (*task)(); });
        }
        condition_.notify_one();
        return result;
    }
private:
    void work() {
        for (;;) {
            std::function<void()> task;
            {
                std::unique_lock lock(mutex_);
                condition_.wait(lock, [this] { return stopping_ || !tasks_.empty(); });
                if (stopping_ && tasks_.empty()) return;
                task = std::move(tasks_.front());
                tasks_.pop_front();
            }
            task();
        }
    }
    std::mutex mutex_;
    std::condition_variable condition_;
    std::deque<std::function<void()>> tasks_;
    std::vector<std::thread> workers_;
    std::optional<fs::path> cache_root_;
    const ReusedLibraries& reused_;
    bool stopping_ = false;
public:
    std::atomic<std::size_t> reused_count{0};
};

bool within(const fs::path& path, const fs::path& root) {
    const auto compared = std::mismatch(root.begin(), root.end(), path.begin(), path.end());
    return compared.first == root.end();
}

fs::path validate_output(const fs::path& supplied, const std::vector<fs::path>& roots) {
    std::error_code error;
    detail::require(!fs::is_symlink(supplied, error), "output cannot be a symbolic link");
    if (fs::exists(supplied, error)) {
        detail::require(fs::is_directory(supplied, error) && fs::directory_iterator(supplied) == fs::directory_iterator{},
                        "output must be new or empty");
    }
    const fs::path resolved = fs::weakly_canonical(supplied, error);
    detail::require(!error, "Cannot resolve package output");
    for (const fs::path& root : roots) detail::require(!within(resolved, root), "output cannot be inside an input directory");
    for (fs::path ancestor = resolved;; ancestor = ancestor.parent_path()) {
        detail::require(!fs::exists(ancestor / ".git", error), "output cannot be inside a Git work tree");
        if (ancestor == ancestor.root_path() || ancestor.empty()) break;
    }
    return resolved;
}

struct RemoveTree {
    fs::path path;
    bool active = true;
    ~RemoveTree() {
        if (active) {
            std::error_code ignored;
            fs::remove_all(path, ignored);
        }
    }
};

std::variant<std::string, BuildReport> build_impl(const BuildOptions& options) {
    detail::require(options.jobs >= 1 && options.jobs <= 8, "--jobs must be between 1 and 8");
    detail::require(options.stages.size() <= detail::maximum_identities, "stage count exceeds the identity policy");
    std::vector<fs::path> roots;
    std::set<std::string> root_names;
    for (const fs::path& path : options.stages) {
        std::error_code error;
        fs::path root = fs::canonical(path, error);
        detail::require(!error && fs::is_directory(root, error), "--stage must name a directory");
        if (root_names.insert(root.string()).second) roots.push_back(std::move(root));
    }
    std::uint64_t declared_bytes = 0;
    for (const fs::path& root : roots) {
        for (const auto& [name, maximum] : std::array<std::pair<std::string_view, std::size_t>, 3>{{
                 {"program.bin", detail::maximum_program_size}, {"source.metal", detail::maximum_source_size},
                 {"reflection.json", detail::maximum_reflection_size}}}) {
            const fs::path path = root / name;
            std::error_code error;
            detail::require(fs::is_regular_file(path, error) && !fs::is_symlink(path, error),
                            std::string(name) + " must be a regular file");
            const auto size = fs::file_size(path, error);
            detail::require(!error && size <= maximum, std::string(name) + " exceeds its size policy");
            declared_bytes += size;
            detail::require(declared_bytes <= detail::maximum_input_bytes,
                            "captured stage inputs exceed the aggregate policy");
        }
    }
    std::vector<detail::Stage> loaded;
    std::uint64_t actual_bytes = 0;
    for (const fs::path& root : roots) {
        detail::Stage stage = detail::load_stage(root);
        actual_bytes += stage.program.size() + stage.source.size() + stage.reflection.size();
        detail::require(actual_bytes <= detail::maximum_input_bytes, "captured stage inputs exceed the aggregate policy");
        loaded.push_back(std::move(stage));
    }
    std::vector<detail::Stage> stages = detail::validate_inventory(std::move(loaded));
    const Json profile = current_profile(options.translate_tool);
    const std::string& source_version = profile.object().at("source_version").string();
    for (const detail::Stage& stage : stages) {
        detail::require(stage.metadata.object().at("source_version").string() == source_version,
                        "stage source version differs from translator");
    }
    const ReusedLibraries reused = reusable_libraries(options.reuse, profile);
    std::size_t reused_libraries = 0;
    const std::string profile_text = detail::canonical_json(profile);
    Sha256 chain_hasher;
    chain_hasher.update("armigo-metal-chain-v1\n");
    chain_hasher.update(profile_text);
    const std::string chain_id = hex(chain_hasher.finish());
    const fs::path output = validate_output(options.output, roots);
    std::error_code error;
    fs::create_directories(output.parent_path(), error);
    detail::require(!error, "Cannot create package output parent");
    const fs::path staging = output.parent_path() / (".armigo-package-" + random_hex());
    fs::create_directory(staging);
    RemoveTree cleanup{staging};
    const fs::path work = staging / ".work";
    fs::create_directory(work);
    std::uint64_t artifact_bytes = 0;
    for (const detail::Stage& stage : stages) artifact_bytes += stage.program.size() + stage.reflection.size();
    detail::require(artifact_bytes <= detail::maximum_output_bytes, "package artifacts exceed the aggregate policy");
    Json::Array entries;
    std::map<std::string, std::pair<fs::path, fs::path>> destinations;
    for (const detail::Stage& stage : stages) {
        const fs::path directory = fs::path("stages") / stage.variant_id;
        const fs::path destination = staging / directory;
        fs::create_directories(destination);
        detail::write_file(destination / "program.bin", stage.program);
        detail::write_file(destination / "reflection.json", stage.reflection);
        destinations.emplace(stage.variant_id, std::pair{directory, destination});
    }
    {
        ThreadPool pool(options.jobs, options.compile_cache, reused);
        std::size_t next = 0;
        std::deque<std::future<Compiled>> pending;
        const auto submit_next = [&] {
            pending.push_back(pool.submit(stages[next], work));
            ++next;
        };
        for (std::size_t index = 0; index < std::min(stages.size(), std::size_t(options.jobs) * 2); ++index) submit_next();
        while (!pending.empty()) {
            std::size_t completed = 0;
            for (auto iterator = pending.begin(); iterator != pending.end();) {
                if (iterator->wait_for(std::chrono::milliseconds(0)) != std::future_status::ready) {
                    ++iterator;
                    continue;
                }
                const Compiled compiled = iterator->get();
                iterator = pending.erase(iterator);
                ++completed;
                detail::require(compiled.size <= detail::maximum_output_bytes - artifact_bytes,
                                "package artifacts exceed the aggregate policy");
                artifact_bytes += compiled.size;
                const auto& [directory, destination] = destinations.at(compiled.variant_id);
                fs::rename(compiled.library, destination / "shader.metallib");
                const detail::Stage& stage = *std::find_if(stages.begin(), stages.end(), [&](const detail::Stage& candidate) {
                    return candidate.variant_id == compiled.variant_id;
                });
                Json::Object entry;
                entry.emplace("directory", Json{directory.generic_string()});
                entry.emplace("library_digest", Json{compiled.digest});
                entry.emplace("library_size", Json{JsonNumber{std::to_string(compiled.size)}});
                entry.emplace("prefix_key", Json{stage.prefix_key});
                entry.emplace("program_key", Json{stage.program_key});
                entry.emplace("program_size", Json{JsonNumber{std::to_string(stage.program.size())}});
                entry.emplace("reflection_digest", Json{detail::sha256(stage.reflection)});
                entry.emplace("stage", Json{stage.stage});
                entry.emplace("variant_id", Json{stage.variant_id});
                entries.emplace_back(std::move(entry));
            }
            for (std::size_t index = 0; index < completed && next < stages.size(); ++index) submit_next();
            if (completed == 0) std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        reused_libraries = pool.reused_count.load();
    }
    std::sort(entries.begin(), entries.end(), [](const Json& left, const Json& right) {
        const auto& a = left.object();
        const auto& b = right.object();
        return std::tie(a.at("stage").string(), a.at("program_key").string(), a.at("variant_id").string()) <
               std::tie(b.at("stage").string(), b.at("program_key").string(), b.at("variant_id").string());
    });
    Json::Object manifest;
    manifest.emplace("schema", Json{std::string("armigo-metal-package-v1")});
    manifest.emplace("source_version", Json{source_version});
    manifest.emplace("chain_id", Json{chain_id});
    manifest.emplace("platform", Json{std::string("macos-metal4")});
    manifest.emplace("profile_text", Json{profile_text});
    manifest.emplace("complete_inventory", Json{false});
    manifest.emplace("runnable", Json{false});
    // Present only when the title opts in. Omitting it keeps a plain manifest
    // byte-identical. The key is outside profile_text, so the chain id is unchanged.
    if (options.hoist_implicit_samples)
        manifest.emplace("sample_hoisting", Json{std::string("implicit")});
    manifest.emplace("entries", std::move(entries));
    std::string manifest_text = detail::canonical_json(Json{std::move(manifest)}) + "\n";
    detail::require(manifest_text.size() <= detail::maximum_output_bytes - artifact_bytes,
                    "package artifacts exceed the aggregate policy");
    detail::write_file(staging / "manifest.json", as_bytes(manifest_text));
    fs::remove_all(work);
    detail::require(validate_output(options.output, roots) == output, "output boundary changed during package build");
    if (fs::exists(output)) fs::remove(output);
    fs::rename(staging, output);
    cleanup.active = false;
    return BuildReport{roots.size(), stages.size(), reused_libraries};
}

} // namespace

std::variant<std::string, BuildReport> build(const BuildOptions& options) {
    try {
        return build_impl(options);
    } catch (const std::bad_alloc&) {
        return std::string("Host memory for shader packaging could not be allocated.");
    } catch (const std::exception& error) {
        return std::string(error.what());
    }
}

} // namespace armigo::shader::package
