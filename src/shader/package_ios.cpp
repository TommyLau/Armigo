// ============================================================================
// Armigo Shader - Native Shader Translation
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include <armigo/shader/ios_package.h>

#include <armigo/shader/graphics_msl.h>
#include <armigo/shader_input/shader_input.h>
#include <armigo/shader_translation/stage_translation.h>
#include "package_internal.h"

#include <CommonCrypto/CommonDigest.h>

#include <sys/wait.h>
#include <unistd.h>
#include <uuid/uuid.h>

#include <pthread.h>

#include <algorithm>
#include <array>
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
#include <map>
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
namespace si = shader_input;
namespace tr = translation;
using detail::Json;
using detail::JsonNumber;
using Clock = std::chrono::steady_clock;

constexpr auto command_timeout = std::chrono::seconds(60);
constexpr std::size_t maximum_manifest_size = 64u << 20;
const std::array<std::string, 7> compile_flags{
    "-std=metal4.0", "-mios-version-min=26.0", "-fwrapv", "-Werror", "-Wno-unused-variable", "-Wno-sometimes-uninitialized",
    "-fbracket-depth=1024"};

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
            const std::string path = first_line(query({"xcrun", "--sdk", "iphoneos", "--find", std::string(tool)}),
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
        hasher.update(query_raw({"xcrun", "--sdk", "iphoneos", "metal", "--version"}));
        cached = hasher.finish();
    }
    return *cached;
}

Json current_profile(std::string_view source_version) {
    const std::string metal = first_line(query({"xcrun", "--sdk", "iphoneos", "--find", "metal"}), "xcrun --find metal");
    const std::string metallib = first_line(query({"xcrun", "--sdk", "iphoneos", "--find", "metallib"}),
                                            "xcrun --find metallib");
    Json::Object profile;
    profile.emplace("compile_options", Json::Array{Json{std::string("-std=metal4.0")},
                                                   Json{std::string("-mios-version-min=26.0")},
                                                   Json{std::string("-fwrapv")},
                                                   Json{std::string("-Werror")}});
    profile.emplace("deployment_target", Json{std::string("26.0")});
    profile.emplace("identity", Json{std::string("span-prefix112-v1")});
    profile.emplace("metal_binary_sha256", Json{detail::sha256_file(metal)});
    profile.emplace("metal_language", Json{std::string("metal4.0")});
    profile.emplace("metal_version", Json{first_line(query({"xcrun", "--sdk", "iphoneos", "metal", "--version"}), "metal")});
    profile.emplace("metallib_binary_sha256", Json{detail::sha256_file(metallib)});
    profile.emplace("platform", Json{std::string("ios-metal4")});
    profile.emplace("sdk_build", Json{first_line(query({"xcrun", "--sdk", "iphoneos", "--show-sdk-build-version"}),
                                                    "iPhoneOS SDK")});
    profile.emplace("sdk_version", Json{first_line(query({"xcrun", "--sdk", "iphoneos", "--show-sdk-version"}),
                                                      "iPhoneOS SDK")});
    profile.emplace("source_version", Json{std::string(source_version)});
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

Compiled compile_stage(const detail::Stage& item, const fs::path& work_root,
                       const std::optional<fs::path>& cache_root) {
    const fs::path work = work_root / item.variant_id;
    const fs::path source = work / "source.metal";
    const fs::path air = work / "shader.air";
    const fs::path library = work / "shader.metallib";
    const auto cached = compile_cache_path(item.source, cache_root);
    if (cached && fs::is_regular_file(*cached) && !fs::is_symlink(*cached)) {
        std::error_code error;
        const auto size = fs::file_size(*cached, error);
        if (!error && size > 0 && size <= detail::maximum_library_size) {
            fs::copy_file(*cached, library, fs::copy_options::overwrite_existing, error);
            detail::require(!error, "Cannot copy cached Metal library");
            return Compiled{item.variant_id, library, size, detail::sha256_file(library)};
        }
    }
    static const std::string metal = first_line(query({"xcrun", "--sdk", "iphoneos", "--find", "metal"}),
                                                "xcrun --find metal");
    static const std::string metallib = first_line(query({"xcrun", "--sdk", "iphoneos", "--find", "metallib"}),
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
    explicit ThreadPool(unsigned count) {
        detail::require(count >= 1 && count <= 8, "--jobs must be between 1 and 8");
        pthread_attr_t attributes;
        detail::require(pthread_attr_init(&attributes) == 0, "A shader worker could not start");
        pthread_attr_setstacksize(&attributes, 64u << 20);
        for (unsigned index = 0; index < count; ++index) {
            pthread_t thread{};
            if (pthread_create(&thread, &attributes, &ThreadPool::entry, this) != 0) {
                pthread_attr_destroy(&attributes);
                stop_and_join();
                throw std::runtime_error("A shader worker could not start");
            }
            workers_.push_back(thread);
        }
        pthread_attr_destroy(&attributes);
    }

    ~ThreadPool() { stop_and_join(); }

    template <typename T, typename Function>
    std::future<T> submit(Function&& function) {
        auto task = std::make_shared<std::packaged_task<T()>>(std::forward<Function>(function));
        std::future<T> result = task->get_future();
        {
            const std::lock_guard lock(mutex_);
            tasks_.push_back([task] { (*task)(); });
        }
        condition_.notify_one();
        return result;
    }

private:
    static void* entry(void* raw) {
        static_cast<ThreadPool*>(raw)->work();
        return nullptr;
    }

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

    void stop_and_join() {
        {
            const std::lock_guard lock(mutex_);
            stopping_ = true;
        }
        condition_.notify_all();
        for (pthread_t thread : workers_) pthread_join(thread, nullptr);
        workers_.clear();
    }

    std::mutex mutex_;
    std::condition_variable condition_;
    std::deque<std::function<void()>> tasks_;
    std::vector<pthread_t> workers_;
    bool stopping_ = false;
};

bool within(const fs::path& path, const fs::path& root) {
    const auto compared = std::mismatch(root.begin(), root.end(), path.begin(), path.end());
    return compared.first == root.end();
}

fs::path validate_output(const fs::path& supplied, const fs::path& input_root) {
    std::error_code error;
    detail::require(!fs::is_symlink(supplied, error), "output cannot be a symbolic link");
    if (fs::exists(supplied, error)) {
        detail::require(fs::is_directory(supplied, error) && fs::directory_iterator(supplied) == fs::directory_iterator{},
                        "output must be new or empty");
    }
    const fs::path resolved = fs::weakly_canonical(supplied, error);
    detail::require(!error, "Cannot resolve package output");
    detail::require(!within(resolved, input_root), "output cannot be inside the input package");
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

const Json& m(const Json& value, std::string_view key) { return detail::member(value, key); }
const std::string& text(const Json& value, std::string_view key) { return m(value, key).string(); }

std::uint64_t number(const Json& value, std::string_view key, std::uint64_t maximum) {
    detail::require(detail::is_uint(m(value, key), maximum), "Package numeric field is invalid");
    return detail::uint_value(m(value, key));
}

template <typename... Keys>
void exact(const Json& value, std::string_view label, Keys... keys) {
    const std::array<std::string_view, sizeof...(Keys)> fields{std::string_view(keys)...};
    detail::require_keys(value, fields, label);
}

std::string digest(const Json& value, std::string_view key) {
    const std::string& result = text(value, key);
    detail::require(result.size() == 64 &&
                        std::all_of(result.begin(), result.end(), [](char character) {
                            return (character >= '0' && character <= '9') || (character >= 'a' && character <= 'f');
                        }),
                    "Package digest is invalid");
    return result;
}

struct InputPackage {
    fs::path root;
    std::string source_version;
    // Copied onto the iOS manifest when the macOS manifest recorded it; the
    // stages are then retranslated with implicit samples hoisted as well.
    std::optional<Json> sample_hoisting;
    std::vector<detail::Stage> stages;
};

InputPackage load_package(const fs::path& supplied) {
    std::error_code error;
    InputPackage result;
    result.root = fs::canonical(supplied, error);
    detail::require(!error && fs::is_directory(result.root, error), "macOS package must name a directory");
    Json manifest = detail::parse_json(
        detail::read_regular(result.root / "manifest.json", maximum_manifest_size, "package manifest"));
    // Optional, and the only extra key. Anything else still fails exact.
    if (manifest.is_object()) {
        auto& fields = manifest.object();
        if (const auto found = fields.find("sample_hoisting"); found != fields.end()) {
            detail::require(found->second.is_string() && found->second.string() == "implicit",
                            "Package sample hoisting is unsupported");
            result.sample_hoisting = found->second;
            fields.erase(found);
        }
    }
    exact(manifest, "package manifest", "schema", "source_version", "chain_id", "platform", "profile_text",
          "complete_inventory", "runnable", "entries");
    detail::require(text(manifest, "schema") == "armigo-metal-package-v1", "Package schema is unsupported");
    detail::require(text(manifest, "platform") == "macos-metal4",
                    "iOS shader conversion requires a macos-metal4 package");
    detail::require(m(manifest, "complete_inventory").is_bool() && !m(manifest, "complete_inventory").boolean() &&
                        m(manifest, "runnable").is_bool() && !m(manifest, "runnable").boolean(),
                    "Current packages must not claim complete or runnable inventory");
    result.source_version = digest(manifest, "source_version");
    const std::string& profile_text = text(manifest, "profile_text");
    const Json profile = detail::parse_json(as_bytes(profile_text));
    detail::require(detail::canonical_json(profile) == profile_text && text(profile, "source_version") == result.source_version &&
                        text(profile, "platform") == "macos-metal4",
                    "Package profile is noncanonical or disagrees with its identity");
    Sha256 chain_hasher;
    chain_hasher.update("armigo-metal-chain-v1\n");
    chain_hasher.update(profile_text);
    detail::require(hex(chain_hasher.finish()) == digest(manifest, "chain_id"), "Package chain ID is invalid");
    detail::require(m(manifest, "entries").is_array() &&
                        m(manifest, "entries").array().size() <= detail::maximum_identities,
                    "Package entries exceed the identity policy");
    std::set<std::string> variants;
    std::uint64_t input_bytes = 0;
    for (const Json& entry : m(manifest, "entries").array()) {
        exact(entry, "package entry", "stage", "program_key", "prefix_key", "program_size", "directory",
              "reflection_digest", "library_digest", "library_size", "variant_id");
        const std::string variant = digest(entry, "variant_id");
        detail::require(variants.insert(variant).second, "Package variant ID is duplicated");
        detail::require(text(entry, "directory") == "stages/" + variant, "Package entry directory is not canonical");
        const fs::path directory = fs::canonical(result.root / text(entry, "directory"), error);
        detail::require(!error && within(directory, result.root) && fs::is_directory(directory, error),
                        "Package stage directory is invalid");
        detail::Stage stage;
        stage.root = directory;
        stage.program = detail::read_regular(directory / "program.bin", detail::maximum_program_size, "program.bin");
        stage.reflection = detail::read_regular(directory / "reflection.json", detail::maximum_reflection_size,
                                                "reflection.json");
        detail::require(stage.program.size() >= detail::prefix_size, "program is shorter than the identity prefix");
        stage.metadata = detail::parse_json(stage.reflection);
        stage.stage = text(stage.metadata, "stage");
        stage.program_key = detail::sha256(stage.program);
        stage.prefix_key = detail::sha256(std::span<const std::byte>(stage.program).first(detail::prefix_size));
        stage.variant_id = detail::sha256(as_bytes(detail::variant_data(stage.metadata)));
        detail::require(stage.variant_id == variant && text(entry, "stage") == stage.stage &&
                            digest(entry, "program_key") == stage.program_key && digest(entry, "prefix_key") == stage.prefix_key &&
                            number(entry, "program_size", detail::maximum_program_size) == stage.program.size() &&
                            digest(entry, "reflection_digest") == detail::sha256(stage.reflection) &&
                            digest(stage.metadata, "source_version") == result.source_version,
                        "Package entry identity does not match its stage");
        const auto library_size = regular_size(directory / "shader.metallib", "Package Metal library is invalid");
        detail::require(number(entry, "library_size", detail::maximum_library_size) == library_size &&
                            digest(entry, "library_digest") == detail::sha256_file(directory / "shader.metallib"),
                        "Package Metal library identity is invalid");
        input_bytes += stage.program.size() + stage.reflection.size() + library_size;
        detail::require(input_bytes <= detail::maximum_input_bytes, "Package inputs exceed the aggregate policy");
        result.stages.push_back(std::move(stage));
    }
    result.stages = detail::validate_inventory(std::move(result.stages));
    return result;
}

void store_u32(std::vector<std::byte>& bytes, std::size_t offset, std::uint32_t value) {
    std::memcpy(bytes.data() + offset, &value, sizeof value);
}

si::Stage input_stage(std::string_view stage) {
    if (stage == "vertex") return si::Stage::Vertex;
    if (stage == "fragment") return si::Stage::Fragment;
    if (stage == "compute") return si::Stage::Compute;
    throw std::runtime_error("Package reflection stage is unsupported");
}

tr::StageOptions stage_options(const detail::Stage& stage, si::Stage kind, bool hoist_implicit_samples) {
    tr::StageOptions options;
    options.texture_handle_cbuf = 2;
    options.texture_descriptors_derived = true;
    options.hoist_implicit_samples = hoist_implicit_samples;
    if (kind == si::Stage::Vertex) {
        const std::string& depth = text(stage.metadata, "vertex_depth");
        detail::require(depth == "minus-one-to-one" || depth == "zero-to-one", "Vertex depth convention is invalid");
        options.emission.vertex_depth = depth == "minus-one-to-one"
            ? graphics_msl::VertexDepthConvention::NegativeOneToOne
            : graphics_msl::VertexDepthConvention::ZeroToOne;
    } else if (kind == si::Stage::Fragment) {
        const std::string& position = text(stage.metadata, "fragment_position");
        detail::require(position == "unspecified" || position == "reciprocal-clip-w",
                        "Fragment position convention is invalid");
        if (position == "reciprocal-clip-w") {
            options.emission.fragment_position = graphics_msl::FragmentPositionConvention::MetalPositionWMatchesGuestPositionW;
        }
    } else {
        const Json& launch = m(stage.metadata, "compute");
        const Json& local = m(launch, "local_size");
        detail::require(local.is_array() && local.array().size() == 3, "Compute launch reflection is invalid");
        std::array<std::uint32_t, 3> dimensions{};
        for (std::size_t index = 0; index < dimensions.size(); ++index) {
            detail::require(detail::is_uint(local.array()[index], 1024), "Compute launch reflection is invalid");
            dimensions[index] = static_cast<std::uint32_t>(detail::uint_value(local.array()[index]));
        }
        options.compute_local_size = dimensions;
        options.compute_shared_memory_bytes = static_cast<std::uint32_t>(number(launch, "shared_memory_bytes", UINT32_MAX));
    }
    return options;
}

detail::Stage retranslate_stage(const detail::Stage& input, const fs::path& work_root, bool hoist_implicit_samples) {
    const si::Stage kind = input_stage(input.stage);
    std::vector<std::byte> control(si::format::control_min_size);
    store_u32(control, 0, si::format::control_magic);
    store_u32(control, 4, si::format::control_tag);
    store_u32(control, si::format::control_version_offset, 14);
    store_u32(control, si::format::control_size_offset, static_cast<std::uint32_t>(control.size()));
    const std::size_t wrapper_size = kind == si::Stage::Compute ? 0 : si::format::graphics_code_offset;
    const auto code_size = static_cast<std::uint32_t>(wrapper_size + input.program.size());
    store_u32(control, si::format::c1_offset_field, code_size);
    store_u32(control, si::format::code_size_field, code_size);
    store_u32(control, si::format::code_offset_field, static_cast<std::uint32_t>(wrapper_size));
    store_u32(control, si::format::stage_field, static_cast<std::uint32_t>(kind));
    std::vector<std::byte> code(code_size);
    if (wrapper_size != 0) store_u32(code, 0, si::format::code_wrapper_magic);
    std::copy(input.program.begin(), input.program.end(), code.begin() + static_cast<std::ptrdiff_t>(wrapper_size));
    // The macOS package records how its stages were translated; translate them the same way.
    const auto options = stage_options(input, kind, hoist_implicit_samples);
    const tr::TranslatedStage translated = tr::translate_stage(control, code, options);
    const auto reflection = as_bytes(translated.reflection);
    detail::require(translated.program == input.program && reflection.size() == input.reflection.size() &&
                        std::equal(reflection.begin(), reflection.end(), input.reflection.begin()),
                    "Packaged stage does not match deterministic native translation: " + input.variant_id);
    const fs::path work = work_root / input.variant_id;
    fs::create_directory(work);
    detail::write_file(work / "source.metal", as_bytes(translated.source));
    detail::write_file(work / "reflection.json", reflection);
    detail::write_file(work / "program.bin", translated.program);
    detail::Stage validated = detail::load_stage(work);
    detail::require(validated.variant_id == input.variant_id, "Translated stage variant identity changed");
    return validated;
}

std::variant<std::string, IosBuildReport> build_impl(const IosBuildOptions& options) {
    detail::require(options.jobs >= 1 && options.jobs <= 8, "--jobs must be between 1 and 8");
    InputPackage input = load_package(options.macos_package);
    const fs::path output = validate_output(options.output, input.root);
    std::error_code error;
    fs::create_directories(output.parent_path(), error);
    detail::require(!error, "Cannot create package output parent");
    const fs::path staging = output.parent_path() / (".armigo-ios-package-" + random_hex());
    fs::create_directory(staging);
    RemoveTree cleanup{staging};
    const fs::path work = staging / ".work";
    fs::create_directory(work);

    std::vector<detail::Stage> translated;
    translated.reserve(input.stages.size());
    {
        ThreadPool pool(options.jobs);
        std::size_t next = 0;
        std::deque<std::future<detail::Stage>> pending;
        const auto submit_next = [&] {
            const detail::Stage& stage = input.stages[next];
            pending.push_back(pool.submit<detail::Stage>([&stage, &work, hoist = input.sample_hoisting.has_value()] {
                return retranslate_stage(stage, work, hoist);
            }));
            ++next;
        };
        for (std::size_t index = 0; index < std::min(input.stages.size(), std::size_t(options.jobs) * 2); ++index) submit_next();
        while (!pending.empty()) {
            std::size_t completed = 0;
            for (auto iterator = pending.begin(); iterator != pending.end();) {
                if (iterator->wait_for(std::chrono::milliseconds(0)) != std::future_status::ready) {
                    ++iterator;
                    continue;
                }
                translated.push_back(iterator->get());
                iterator = pending.erase(iterator);
                ++completed;
            }
            for (std::size_t index = 0; index < completed && next < input.stages.size(); ++index) submit_next();
            if (completed == 0) std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }
    std::vector<detail::Stage> stages = detail::validate_inventory(std::move(translated));
    const Json profile = current_profile(input.source_version);
    const std::string profile_text = detail::canonical_json(profile);
    Sha256 chain_hasher;
    chain_hasher.update("armigo-metal-chain-v1\n");
    chain_hasher.update(profile_text);
    const std::string chain_id = hex(chain_hasher.finish());

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
        ThreadPool pool(options.jobs);
        std::size_t next = 0;
        std::deque<std::future<Compiled>> pending;
        const auto submit_next = [&] {
            const detail::Stage& stage = stages[next];
            pending.push_back(pool.submit<Compiled>([&stage, &work, &options] {
                return compile_stage(stage, work, options.compile_cache);
            }));
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
    }
    std::sort(entries.begin(), entries.end(), [](const Json& left, const Json& right) {
        const auto& a = left.object();
        const auto& b = right.object();
        return std::tie(a.at("stage").string(), a.at("program_key").string(), a.at("variant_id").string()) <
               std::tie(b.at("stage").string(), b.at("program_key").string(), b.at("variant_id").string());
    });
    Json::Object manifest;
    manifest.emplace("schema", Json{std::string("armigo-metal-package-v1")});
    manifest.emplace("source_version", Json{input.source_version});
    manifest.emplace("chain_id", Json{chain_id});
    manifest.emplace("platform", Json{std::string("ios-metal4")});
    manifest.emplace("profile_text", Json{profile_text});
    manifest.emplace("complete_inventory", Json{false});
    manifest.emplace("runnable", Json{false});
    // Unchanged from the macOS manifest. Absent stays absent, so a plain iOS
    // manifest stays byte-identical. The key is outside profile_text.
    if (input.sample_hoisting)
        manifest.emplace("sample_hoisting", *input.sample_hoisting);
    manifest.emplace("entries", std::move(entries));
    const std::string manifest_text = detail::canonical_json(Json{std::move(manifest)}) + "\n";
    detail::require(manifest_text.size() <= detail::maximum_output_bytes - artifact_bytes,
                    "package artifacts exceed the aggregate policy");
    detail::write_file(staging / "manifest.json", as_bytes(manifest_text));
    fs::remove_all(work);
    detail::require(validate_output(options.output, input.root) == output, "output boundary changed during package build");
    if (fs::exists(output)) fs::remove(output);
    fs::rename(staging, output);
    cleanup.active = false;
    return IosBuildReport{stages.size()};
}

} // namespace

std::variant<std::string, IosBuildReport> build_ios_package(const IosBuildOptions& options) {
    try {
        return build_impl(options);
    } catch (const std::bad_alloc&) {
        return std::string("Host memory for iOS shader packaging could not be allocated.");
    } catch (const std::exception& error) {
        return std::string(error.what());
    }
}

} // namespace armigo::shader::package
