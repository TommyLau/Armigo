// ============================================================================
// Armigo Tools - Development and Verification Tools
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include "native_shaders.h"

#include "nvn_native_stages.h"
#include "pso_manifest.h"

#include <armigo/loader/nup.h>
#include <armigo/platform/metal_shader_package.h>
#include <armigo/shader/graphics_msl.h>
#include <armigo/shader/package.h>
#include <armigo/shader_native/package.h>
#include <armigo/shader_input/shader_input.h>
#include <armigo/shader_native/frontend.h>
#include <armigo/shader_native/input.h>

#include <CommonCrypto/CommonDigest.h>

#include <pthread.h>
#include <spawn.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <mutex>
#include <optional>
#include <optional>
#include <set>
#include <span>
#include <stdexcept>
#include <string_view>
#include <thread>
#include <tuple>
#include <unordered_set>
#include <vector>

extern char** environ;

namespace armigo::pack {
namespace {

namespace fs = std::filesystem;
namespace sn = armigo::shader_native;
namespace pl = armigo::shader::pipeline;
namespace gm = armigo::shader::graphics_msl;
namespace ld = armigo::loader;

constexpr auto command_timeout = std::chrono::seconds(60);
// A compile of a large stage (a mesh stage's MSL runs to megabytes) can take
// minutes; the limit only guards against a hung compiler.
constexpr auto compile_timeout = std::chrono::minutes(20);
constexpr std::size_t translation_stack_bytes = 64u << 20;
constexpr std::size_t maximum_library_size = 64u * 1024u * 1024u;
constexpr std::uint64_t maximum_copied_bytes = 8ull << 30;
// Runtime newLibraryWithSource uses Metal 4.0 and does not treat warnings as
// errors. Movie stages compare a value with itself; -Werror turns that into
// a failed library. -fpreserve-invariance matches a vertex position marked
// invariant, and does nothing for a stage that carries no such mark.
// -fbracket-depth raises the front end's nested-bracket limit.
constexpr std::array<std::string_view, 7> compile_flags{
    "-std=metal4.0", "-mmacosx-version-min=26.0", "-fwrapv", "-fpreserve-invariance",
    "-Wno-unused-variable", "-Wno-sometimes-uninitialized", "-fbracket-depth=1024"};

struct Failure : std::runtime_error {
    using std::runtime_error::runtime_error;
};

struct RemoveTree {
    fs::path path;
    bool active = true;
    ~RemoveTree() {
        if (!active) return;
        std::error_code ignored;
        fs::remove_all(path, ignored);
    }
};

std::string trim(std::string value) {
    const auto first = value.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) return {};
    return value.substr(first, value.find_last_not_of(" \t\r\n") - first + 1);
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
        else if (count == 0 || errno != EINTR) break;
    }
    ::close(descriptor);
}

ProcessResult run_process(const std::vector<std::string>& arguments, bool c_locale,
                          std::chrono::steady_clock::duration timeout = command_timeout) {
    if (arguments.empty()) throw Failure("Cannot execute an empty shader tool command");
    int output_pipe[2] = {-1, -1};
    int error_pipe[2] = {-1, -1};
    if (::pipe(output_pipe) != 0 || ::pipe(error_pipe) != 0) throw Failure("Cannot execute Apple shader tool");
    posix_spawn_file_actions_t actions;
    if (posix_spawn_file_actions_init(&actions) != 0) throw Failure("Cannot execute Apple shader tool");
    posix_spawn_file_actions_addclose(&actions, output_pipe[0]);
    posix_spawn_file_actions_addclose(&actions, error_pipe[0]);
    posix_spawn_file_actions_adddup2(&actions, output_pipe[1], STDOUT_FILENO);
    posix_spawn_file_actions_adddup2(&actions, error_pipe[1], STDERR_FILENO);
    posix_spawn_file_actions_addclose(&actions, output_pipe[1]);
    posix_spawn_file_actions_addclose(&actions, error_pipe[1]);
    std::vector<char*> argv;
    for (const std::string& argument : arguments) argv.push_back(const_cast<char*>(argument.c_str()));
    argv.push_back(nullptr);
    std::vector<std::string> environment;
    std::vector<char*> environment_pointer;
    char** child_environment = environ;
    if (c_locale) {
        for (char** item = environ; item != nullptr && *item != nullptr; ++item) {
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
        throw Failure("Cannot execute Apple shader tool");
    }
    ProcessResult result;
    std::thread output_reader(read_pipe, output_pipe[0], std::ref(result.output));
    std::thread error_reader(read_pipe, error_pipe[0], std::ref(result.error));
    const auto deadline = std::chrono::steady_clock::now() + timeout;
    int status = 0;
    for (;;) {
        const pid_t waited = ::waitpid(pid, &status, WNOHANG);
        if (waited < 0) throw Failure("Cannot wait for Apple shader tool");
        if (waited == pid) break;
        if (std::chrono::steady_clock::now() >= deadline) {
            result.timed_out = true;
            ::kill(pid, SIGKILL);
            if (::waitpid(pid, &status, 0) != pid) throw Failure("Cannot wait for Apple shader tool");
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    output_reader.join();
    error_reader.join();
    result.status = WIFEXITED(status) ? WEXITSTATUS(status) : -1;
    return result;
}

std::string query(const std::vector<std::string>& arguments) {
    const ProcessResult result = run_process(arguments, false);
    if (result.timed_out || result.status != 0) throw Failure("Shader tool query failed: " + trim(result.error).substr(0, 4096));
    const auto end = result.output.find_first_of("\r\n");
    const std::string line = trim(result.output.substr(0, end));
    if (line.empty()) throw Failure("Shader tool query returned no version");
    return line;
}

void run_apple_tool(const std::vector<std::string>& arguments) {
    const ProcessResult result = run_process(arguments, true, compile_timeout);
    if (result.timed_out) throw Failure("Apple shader tool timed out");
    if (result.status != 0) throw Failure("Apple shader tool failed: " + trim(result.error).substr(0, 4096));
}

std::string hex_bytes(std::span<const std::byte> bytes) {
    static constexpr char digits[] = "0123456789abcdef";
    std::string text;
    text.reserve(bytes.size() * 2);
    for (const std::byte byte : bytes) {
        const auto value = std::to_integer<unsigned>(byte);
        text += digits[value >> 4];
        text += digits[value & 15];
    }
    return text;
}

std::array<std::byte, CC_SHA256_DIGEST_LENGTH> hash_file(const fs::path& path) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream.good()) throw Failure("Cannot hash Apple shader tool");
    CC_SHA256_CTX context;
    if (CC_SHA256_Init(&context) != 1) throw Failure("System SHA-256 failed");
    std::array<char, 65536> block{};
    while (stream) {
        stream.read(block.data(), static_cast<std::streamsize>(block.size()));
        const auto count = stream.gcount();
        if (count > 0 && CC_SHA256_Update(&context, block.data(), static_cast<CC_LONG>(count)) != 1)
            throw Failure("System SHA-256 failed");
    }
    if (!stream.eof()) throw Failure("Cannot hash Apple shader tool");
    std::array<std::byte, CC_SHA256_DIGEST_LENGTH> digest{};
    if (CC_SHA256_Final(reinterpret_cast<unsigned char*>(digest.data()), &context) != 1)
        throw Failure("System SHA-256 failed");
    return digest;
}

std::array<std::byte, CC_SHA256_DIGEST_LENGTH> compiler_identity() {
    static std::mutex mutex;
    static std::optional<std::array<std::byte, CC_SHA256_DIGEST_LENGTH>> cached;
    const std::lock_guard lock(mutex);
    if (cached) return *cached;
    CC_SHA256_CTX context;
    if (CC_SHA256_Init(&context) != 1) throw Failure("System SHA-256 failed");
    for (const char* tool : {"metal", "metallib"}) {
        const auto digest = hash_file(query({"xcrun", "--sdk", "macosx", "--find", tool}));
        if (CC_SHA256_Update(&context, digest.data(), static_cast<CC_LONG>(digest.size())) != 1)
            throw Failure("System SHA-256 failed");
    }
    const ProcessResult version = run_process({"xcrun", "--sdk", "macosx", "metal", "--version"}, false);
    if (version.timed_out || version.status != 0) throw Failure("Shader tool query failed");
    if (CC_SHA256_Update(&context, version.output.data(), static_cast<CC_LONG>(version.output.size())) != 1)
        throw Failure("System SHA-256 failed");
    std::array<std::byte, CC_SHA256_DIGEST_LENGTH> digest{};
    if (CC_SHA256_Final(reinterpret_cast<unsigned char*>(digest.data()), &context) != 1)
        throw Failure("System SHA-256 failed");
    cached = digest;
    return digest;
}

std::optional<fs::path> compile_cache_file(std::string_view source, const std::optional<fs::path>& root) {
    if (!root) return std::nullopt;
    CC_SHA256_CTX context;
    if (CC_SHA256_Init(&context) != 1) throw Failure("System SHA-256 failed");
    const auto identity = compiler_identity();
    if (CC_SHA256_Update(&context, identity.data(), static_cast<CC_LONG>(identity.size())) != 1)
        throw Failure("System SHA-256 failed");
    for (std::size_t index = 0; index < compile_flags.size(); ++index) {
        if (index != 0 && CC_SHA256_Update(&context, "\0", 1) != 1) throw Failure("System SHA-256 failed");
        if (CC_SHA256_Update(&context, compile_flags[index].data(), static_cast<CC_LONG>(compile_flags[index].size())) != 1)
            throw Failure("System SHA-256 failed");
    }
    if (CC_SHA256_Update(&context, "\0", 1) != 1 ||
        CC_SHA256_Update(&context, source.data(), static_cast<CC_LONG>(source.size())) != 1)
        throw Failure("System SHA-256 failed");
    std::array<std::byte, CC_SHA256_DIGEST_LENGTH> digest{};
    if (CC_SHA256_Final(reinterpret_cast<unsigned char*>(digest.data()), &context) != 1)
        throw Failure("System SHA-256 failed");
    return *root / (hex_bytes(digest) + ".metallib");
}

std::vector<std::byte> read_file(const fs::path& path, std::size_t maximum) {
    std::error_code error;
    if (!fs::is_regular_file(path, error) || fs::is_symlink(path, error)) throw Failure("Metal library is missing");
    const auto size = fs::file_size(path, error);
    if (error || size == 0 || size > maximum) throw Failure("Metal library violates the size policy");
    std::vector<std::byte> bytes(static_cast<std::size_t>(size));
    std::ifstream stream(path, std::ios::binary);
    if (!stream.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(bytes.size())))
        throw Failure("Metal library could not be read");
    return bytes;
}

void write_text(const fs::path& path, std::string_view text) {
    std::ofstream stream(path, std::ios::binary | std::ios::trunc);
    if (!stream.write(text.data(), static_cast<std::streamsize>(text.size()))) throw Failure("Cannot write Metal source");
}

std::string random_name() {
    std::array<unsigned char, 16> bytes{};
    std::ifstream random("/dev/urandom", std::ios::binary);
    if (!random.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(bytes.size())))
        throw Failure("Cannot name a temporary shader file");
    return hex_bytes({reinterpret_cast<const std::byte*>(bytes.data()), bytes.size()});
}

std::vector<std::byte> compile_metal(std::string_view source, const fs::path& work, const std::optional<fs::path>& cache_root) {
    std::error_code error;
    fs::create_directories(work, error);
    if (error) throw Failure("Cannot create a shader work directory");
    const fs::path source_path = work / "source.metal";
    const fs::path air = work / "shader.air";
    const fs::path library = work / "shader.metallib";
    const auto cached = compile_cache_file(source, cache_root);
    if (cached && fs::is_regular_file(*cached, error) && !fs::is_symlink(*cached, error)) {
        const auto size = fs::file_size(*cached, error);
        if (!error && size > 0 && size <= maximum_library_size) {
            fs::copy_file(*cached, library, fs::copy_options::overwrite_existing, error);
            if (!error) return read_file(library, maximum_library_size);
        }
    }
    write_text(source_path, source);
    static const std::string metal = query({"xcrun", "--sdk", "macosx", "--find", "metal"});
    static const std::string metallib = query({"xcrun", "--sdk", "macosx", "--find", "metallib"});
    std::vector<std::string> compile{metal};
    for (const std::string_view flag : compile_flags) compile.emplace_back(flag);
    compile.insert(compile.end(), {"-c", source_path.string(), "-o", air.string()});
    run_apple_tool(compile);
    run_apple_tool({metallib, air.string(), "-o", library.string()});
    auto bytes = read_file(library, maximum_library_size);
    fs::remove(air, error);
    if (cached) {
        fs::create_directories(cached->parent_path(), error);
        if (error) throw Failure("Cannot create shader compile cache");
        const fs::path temporary = cached->parent_path() / (cached->filename().string() + "." + random_name() + ".tmp");
        fs::copy_file(library, temporary, fs::copy_options::none, error);
        if (error) throw Failure("Cannot cache Metal library");
        fs::rename(temporary, *cached, error);
        if (error) {
            fs::remove(temporary, error);
            throw Failure("Cannot cache Metal library");
        }
    }
    return bytes;
}

const char* stage_name(sn::Stage stage) {
    switch (stage) {
        case sn::Stage::vertex: return "vertex";
        case sn::Stage::fragment: return "fragment";
        case sn::Stage::compute: return "compute";
        case sn::Stage::task: return "task";
        case sn::Stage::mesh: return "mesh";
    }
    return "unknown";
}

std::optional<sn::NativeStageKind> packaged_stage_kind(sn::Stage stage) {
    switch (stage) {
        case sn::Stage::vertex: return sn::NativeStageKind::vertex;
        case sn::Stage::fragment: return sn::NativeStageKind::fragment;
        case sn::Stage::compute: return sn::NativeStageKind::compute;
        case sn::Stage::task: return sn::NativeStageKind::task;
        case sn::Stage::mesh: return sn::NativeStageKind::mesh;
    }
    return std::nullopt;
}

const char* diagnostic_name(pl::DiagnosticCode code) {
    switch (code) {
        case pl::DiagnosticCode::malformed_input: return "malformed_input";
        case pl::DiagnosticCode::unsupported_instruction: return "unsupported_instruction";
        case pl::DiagnosticCode::unsupported_modifier: return "unsupported_modifier";
        case pl::DiagnosticCode::missing_fact: return "missing_fact";
        case pl::DiagnosticCode::invalid_fact: return "invalid_fact";
        case pl::DiagnosticCode::conflicting_fact: return "conflicting_fact";
        case pl::DiagnosticCode::invalid_ir: return "invalid_ir";
        case pl::DiagnosticCode::undefined_read: return "undefined_read";
        case pl::DiagnosticCode::unsupported_capability: return "unsupported_capability";
        case pl::DiagnosticCode::unresolved_resource: return "unresolved_resource";
        case pl::DiagnosticCode::limit_exceeded: return "limit_exceeded";
        case pl::DiagnosticCode::duplicate_dispatch: return "duplicate_dispatch";
        case pl::DiagnosticCode::unimplemented: return "unimplemented";
    }
    return "unimplemented";
}

const char* metadata_name(sn::ProgramError code) {
    switch (code) {
        case sn::ProgramError::truncated_control: return "metadata-truncated-control";
        case sn::ProgramError::control_family: return "metadata-control-family";
        case sn::ProgramError::unsupported_stage: return "unsupported-stage";
        case sn::ProgramError::stage_slot: return "metadata-stage-slot";
        case sn::ProgramError::payload_size: return "metadata-payload-size";
        case sn::ProgramError::program_bounds: return "metadata-program-bounds";
        case sn::ProgramError::instruction_alignment: return "metadata-instruction-alignment";
        case sn::ProgramError::nonzero_padding: return "metadata-nonzero-padding";
    }
    return "metadata-rejected";
}

struct CopiedProgram {
    sn::ContainerKind kind = sn::ContainerKind::bnsh;
    std::optional<unsigned> slot;
    std::string member;
    std::vector<std::byte> control;
    std::vector<std::byte> code;
};

struct WorkerStats {
    std::uint64_t translated = 0;
    std::uint64_t compiled = 0;
    std::map<std::string, std::uint64_t> translated_by_stage;
    std::map<std::string, std::uint64_t> compiled_by_stage;
    std::map<std::string, std::uint64_t> refusals;
};

struct BuildShared {
    const std::vector<CopiedProgram>* programs = nullptr;
    sn::NativeSectionWriter* writer = nullptr;
    std::mutex writer_mutex;
    std::optional<fs::path> cache;
    fs::path work_root;
    std::atomic<std::size_t> next{0};
    std::atomic<std::uint64_t> attempts{0};
    std::atomic<bool> failed{false};
    std::mutex error_mutex;
    std::string error;
    std::mutex progress_mutex;
    std::size_t total = 0;
    // Packaged compute identities; sorted before the compute list is written.
    std::vector<std::string> compute_identities;
    // Translate and count diagnostics. Do not compile or store a library.
    bool translate_only = false;
    // Where translated Metal sources are also written, when set.
    std::optional<fs::path> source_directory;
    // The largest Metal source a stage may have (the package's limit for its
    // file policy unless a request overrides it).
    std::size_t maximum_source_bytes = sn::maximum_native_source_bytes;
    // Sources are written next to their libraries (bounded by the file policy).
    bool keep_sources = true;

    void fail(std::string text) {
        const std::lock_guard lock(error_mutex);
        if (error.empty()) error = std::move(text);
        failed.store(true);
    }
};

void note_progress(BuildShared& shared, std::uint64_t attempts) {
    if (attempts == 0 || attempts % 500 != 0) return;
    const std::lock_guard lock(shared.progress_mutex);
    std::cerr << "Native shader package: " << attempts << " translations of " << shared.total << " programs\n";
}

shader_input::Stage input_stage(sn::Stage stage) {
    if (stage == sn::Stage::vertex) return shader_input::Stage::Vertex;
    if (stage == sn::Stage::fragment) return shader_input::Stage::Fragment;
    if (stage == sn::Stage::task) return shader_input::Stage::TessellationControl;
    if (stage == sn::Stage::mesh) return shader_input::Stage::TessellationEvaluation;
    return shader_input::Stage::Compute;
}

pl::Stage pipeline_stage(sn::Stage stage) {
    if (stage == sn::Stage::vertex || stage == sn::Stage::mesh) return pl::Stage::vertex;
    if (stage == sn::Stage::fragment) return pl::Stage::fragment;
    return pl::Stage::compute;
}

const char* depth_label(sn::NativeDepth depth) {
    switch (depth) {
        case sn::NativeDepth::stage: return "stage";
        case sn::NativeDepth::minus_one_to_one: return "minus-one-to-one";
        case sn::NativeDepth::zero_to_one: return "zero-to-one";
    }
    return "stage";
}

// clip_depth: a vertex or mesh stage writes clip position in one depth convention.
bool translate_depth(const sn::Program& program, sn::NativeDepth depth, bool clip_depth, WorkerStats& stats, sn::NativeStageWrite& stage,
                     const fs::path& work, const std::optional<fs::path>& cache, bool compile,
                     const std::optional<fs::path>& source_directory, std::size_t maximum_source_bytes) {
    sn::FrontendOptions options;
    options.pixel_rate = sn::PixelRateState::disabled;
    options.vertex_fetch = false;
    gm::EmitOptions emission;
    emission.vertex_fetch = false;
    if (clip_depth) {
        const bool zero = depth == sn::NativeDepth::zero_to_one;
        options.vertex_depth = zero ? pl::DepthConvention::zero_to_one : pl::DepthConvention::negative_one_to_one;
        emission.vertex_depth = zero ? gm::VertexDepthConvention::ZeroToOne : gm::VertexDepthConvention::NegativeOneToOne;
    } else if (program.stage == sn::Stage::fragment) {
        options.fragment_position = pl::FragmentPositionConvention::metal_position_w_matches_guest_position_w;
        emission.fragment_position = gm::FragmentPositionConvention::MetalPositionWMatchesGuestPositionW;
    }
    auto translated = sn::translate_native(program, options);
    if (const auto* diagnostic = std::get_if<pl::Diagnostic>(&translated)) {
        ++stats.refusals[diagnostic_name(diagnostic->code)];
        if (!compile)
            std::cerr << "Native shader refused " << program.stage_identity << ' ' << stage_name(program.stage) << ' '
                      << depth_label(depth) << ' ' << diagnostic_name(diagnostic->code)
                      << (diagnostic->pc ? " pc " + std::to_string(*diagnostic->pc) : std::string()) << ": "
                      << diagnostic->message << '\n';
        return false;
    }
    const auto& artifact = std::get<pl::MslArtifact>(translated);
    if (source_directory) {
        std::error_code ignored;
        fs::create_directories(*source_directory, ignored);
        std::ofstream(*source_directory / (program.stage_identity + '-' + depth_label(depth) + ".metal"), std::ios::binary)
            << artifact.source;
    }
    // A source the package cannot record is left to run-time translation. A
    // translation census counts it too, so its totals match a package build.
    if (artifact.source.size() > maximum_source_bytes) {
        ++stats.refusals["source-size"];
        std::cerr << "Native shader source-size " << program.stage_identity << ' ' << stage_name(program.stage) << ' '
                  << artifact.source.size() << " bytes\n";
        return false;
    }
    if (!compile) return true;
    const auto launch = armigo::platform::metal::native_mesh_launch(program);
    if ((program.stage == sn::Stage::task || program.stage == sn::Stage::mesh) && !launch) {
        ++stats.refusals["stage-rejected"];
        std::cerr << "Native shader stage-rejected " << program.stage_identity << ' ' << stage_name(program.stage)
                  << " reflection: Native task or mesh stage has no launch facts\n";
        return false;
    }
    std::string reflection;
    try {
        reflection = armigo::platform::metal::native_stage_reflection(artifact, emission, pipeline_stage(program.stage), program.image,
                                                                       launch ? &*launch : nullptr);
    } catch (const std::exception& error) {
        ++stats.refusals["stage-rejected"];
        std::cerr << "Native shader stage-rejected " << program.stage_identity << ' ' << stage_name(program.stage)
                  << " reflection: " << error.what() << '\n';
        return false;
    }
    const std::span<const std::byte> reflection_bytes{reinterpret_cast<const std::byte*>(reflection.data()), reflection.size()};
    auto made = armigo::platform::make_source_stage(input_stage(program.stage), program.image, reflection_bytes, artifact.source);
    if (const auto* rejected = std::get_if<armigo::platform::MetalShaderPackageError>(&made)) {
        ++stats.refusals["stage-rejected"];
        std::cerr << "Native shader stage-rejected " << program.stage_identity << ' ' << stage_name(program.stage)
                  << " stage: " << rejected->detail << '\n';
        return false;
    }
    std::vector<std::byte> library;
    try {
        library = compile_metal(artifact.source, work / (clip_depth ? std::string(depth == sn::NativeDepth::zero_to_one ? "zero" : "minus") : "stage"), cache);
    } catch (const std::exception& error) {
        ++stats.refusals["metal-compile"];
        std::cerr << "Native shader metal-compile " << program.stage_identity << ' ' << error.what() << '\n';
        return false;
    }
    sn::NativeVariantWrite variant;
    variant.depth = depth;
    variant.reflection.assign(reflection_bytes.begin(), reflection_bytes.end());
    variant.source = std::move(artifact.source);
    variant.library = std::move(library);
    stage.variants.push_back(std::move(variant));
    ++stats.compiled;
    ++stats.compiled_by_stage[stage_name(program.stage)];
    return true;
}

void translate_program(BuildShared& shared, std::size_t index, unsigned worker, WorkerStats& stats) {
    const CopiedProgram& owned = (*shared.programs)[index];
    sn::Record record;
    record.kind = owned.kind;
    record.slot = owned.slot;
    record.control = owned.control;
    record.code = owned.code;
    auto parsed = sn::parse_program(record);
    if (const auto* failure = std::get_if<sn::ProgramFailure>(&parsed)) {
        ++stats.refusals[metadata_name(failure->code)];
        return;
    }
    const auto& program = std::get<sn::Program>(parsed);
    const auto packaged = packaged_stage_kind(program.stage);
    if (!packaged) return;
    // A mesh stage is translated once per depth convention, as a vertex stage
    // is. A task stage has one variant. A translation census still asks the
    // frontend, counts the result, and does not compile or store a library.
    const bool clip_depth = program.stage == sn::Stage::vertex || program.stage == sn::Stage::mesh;
    sn::NativeStageWrite stage;
    stage.stage_identity = program.stage_identity;
    stage.stage = *packaged;
    stage.program.assign(program.image.begin(), program.image.end());
    const fs::path work = shared.work_root / std::to_string(worker) / std::to_string(index);
    std::uint64_t attempts = 0;
    bool produced = false;
    const bool compile = !shared.translate_only;
    if (clip_depth) {
        for (const sn::NativeDepth depth : {sn::NativeDepth::minus_one_to_one, sn::NativeDepth::zero_to_one}) {
            produced = translate_depth(program, depth, true, stats, stage, work, shared.cache, compile, shared.source_directory,
                                       shared.maximum_source_bytes) || produced;
            attempts = shared.attempts.fetch_add(1) + 1;
            note_progress(shared, attempts);
        }
    } else {
        produced = translate_depth(program, sn::NativeDepth::stage, false, stats, stage, work, shared.cache, compile,
                                   shared.source_directory, shared.maximum_source_bytes);
        attempts = shared.attempts.fetch_add(1) + 1;
        note_progress(shared, attempts);
    }
    std::error_code ignored;
    fs::remove_all(work, ignored);
    if (!produced) return;
    ++stats.translated;
    ++stats.translated_by_stage[stage_name(program.stage)];
    if (shared.translate_only) return;
    std::lock_guard lock(shared.writer_mutex);
    std::cerr << "Native shader stage " << program.stage_identity << ' ' << stage_name(program.stage) << ' '
              << owned.member << '\n';
    auto added = shared.writer->add(stage);
    if (const auto* failure = std::get_if<std::string>(&added)) {
        shared.fail(*failure);
        return;
    }
    if (stage.stage == sn::NativeStageKind::compute) shared.compute_identities.push_back(stage.stage_identity);
}

std::string write_compute_pipeline_manifest(const fs::path& package_root,
                                            const std::vector<std::string>& identities) {
    const std::string text = armigo::platform::metal::nvn_driver::compute_stage_manifest(identities);
    if (text.empty()) return {};
    const fs::path path = package_root / "native" /
        armigo::platform::metal::nvn_driver::compute_pipeline_manifest_file;
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out) return "Cannot write the compute pipeline list.";
    out.write(text.data(), static_cast<std::streamsize>(text.size()));
    out.close();
    if (!out) return "Cannot write the compute pipeline list.";
    return {};
}

struct WorkerStart {
    BuildShared* shared = nullptr;
    unsigned worker = 0;
    WorkerStats* stats = nullptr;
};

void* worker_main(void* raw) {
    auto* start = static_cast<WorkerStart*>(raw);
    try {
        for (;;) {
            if (start->shared->failed.load()) return nullptr;
            const auto index = start->shared->next.fetch_add(1);
            if (index >= start->shared->programs->size()) return nullptr;
            translate_program(*start->shared, index, start->worker, *start->stats);
        }
    } catch (const std::bad_alloc&) {
        start->shared->fail("Host memory for native shader packaging could not be allocated.");
    } catch (const std::exception& error) {
        start->shared->fail(error.what());
    } catch (...) {
        start->shared->fail("Native shader packaging failed");
    }
    return nullptr;
}

template <class T>
T unwrap(std::variant<ld::NupFailure, T> result) {
    if (const auto* error = std::get_if<ld::NupFailure>(&result))
        throw sn::InputError(error->operation + ": " + error->detail);
    return std::get<T>(std::move(result));
}

struct ArchiveSource {
    std::shared_ptr<ld::NupArchive> archive;
    sn::Buffer read(std::uint32_t index, const sn::Dictionaries& dictionaries, std::size_t limit) const {
        const auto& file = archive->files().at(index);
        if (file.codec || file.stored_size != file.size) throw sn::InputError("Unsupported NUP extent codec");
        if (const auto error = archive->verify(index)) throw sn::InputError(error->operation + ": " + error->detail);
        std::uint64_t position = 0;
        auto reader = [&](std::span<std::byte> buffer) {
            const auto count = unwrap(archive->read(index, position, buffer));
            position += count;
            return count;
        };
        std::array<std::byte, 4> prefix{};
        const auto count = unwrap(archive->read(index, 0, prefix));
        if (count == 4 && sn::little_endian(prefix, 0, 4) == 0xfd2fb528)
            return sn::read_zstandard(reader, dictionaries, limit);
        return sn::read_raw_member(reader, file.size, limit);
    }
};

std::string choose_dictionary(const std::map<std::string, std::uint32_t>& romfs, const std::string& requested) {
    // The title's settings (or the caller) name the dictionary member; none
    // reads the archive without dictionaries.
    if (requested.empty()) return {};
    if (!romfs.contains(requested)) throw sn::InputError("Dictionary archive member was not selected");
    return requested;
}

bool member_selected(const std::string& path, const std::vector<std::string>& members, std::vector<char>& matched) {
    if (members.empty()) return true;
    bool selected = false;
    for (std::size_t index = 0; index < members.size(); ++index) {
        const std::string& member = members[index];
        if (member.empty()) continue;
        const bool exact = path == member;
        const bool suffix = path.size() > member.size() && path.ends_with(member) &&
                            path[path.size() - member.size() - 1] == '/';
        if (!exact && !suffix) continue;
        matched[index] = 1;
        selected = true;
    }
    return selected;
}

bool inside_git(const fs::path& path) {
    std::error_code error;
    const fs::path resolved = fs::weakly_canonical(path, error);
    if (error) return false;
    for (fs::path ancestor = resolved; !ancestor.empty(); ancestor = ancestor.parent_path()) {
        if (fs::exists(ancestor / ".git", error)) return true;
        if (ancestor == ancestor.root_path()) break;
    }
    return false;
}

void require_output(const fs::path& output) {
    std::error_code error;
    if (fs::is_symlink(output, error)) throw Failure("output cannot be a symbolic link");
    if (fs::exists(output, error)) {
        if (!fs::is_directory(output, error) || fs::directory_iterator(output) != fs::directory_iterator{})
            throw Failure("output must be new or empty");
    }
    if (inside_git(output)) throw Failure("output cannot be inside a Git work tree");
}

std::map<std::string, std::uint64_t> stage_counts() { return {{"vertex", 0}, {"fragment", 0}, {"compute", 0}}; }

void add_count(std::map<std::string, std::uint64_t>& counts, const char* key, std::uint64_t value) { counts[key] += value; }

} // namespace

std::variant<std::string, NativeShaderReport> build_native_shader_package(const NativeShaderRequest& request) {
    try {
        if (request.jobs < 1 || request.jobs > 8) return std::string("--jobs must be between 1 and 8");
        if (request.enumerate_only && request.translate_only)
            return std::string("--enumerate-only and --translate-only cannot be combined");
        if (!fs::is_regular_file(request.nup)) return std::string("The .nup archive is not a readable file.");
        if (!fs::is_regular_file(request.translate_tool)) return std::string("The shader translator is missing.");
        require_output(request.output);
        armigo::shader::package::BuildOptions shell;
        shell.translate_tool = request.translate_tool;
        shell.output = request.output;
        shell.jobs = 1;
        auto published = armigo::shader::package::build(shell);
        if (const auto* failure = std::get_if<std::string>(&published)) return *failure;
        RemoveTree cleanup{request.output};
        auto created = sn::NativeSectionWriter::create(request.output, request.keep_sources);
        if (const auto* failure = std::get_if<std::string>(&created)) return *failure;
        auto writer = std::get<sn::NativeSectionWriter>(std::move(created));

        ArchiveSource source{unwrap(ld::NupArchive::open(request.nup))};
        const auto selection = unwrap(source.archive->select());
        const std::string dictionary_member = choose_dictionary(selection.romfs, request.dictionary_member);
        sn::Limits limits;
        sn::Dictionaries dictionaries;
        if (!dictionary_member.empty()) {
            const auto member = selection.romfs.find(dictionary_member);
            auto dictionary_bytes = source.read(member->second, {}, limits.member_bytes);
            dictionaries = sn::read_dictionaries(dictionary_bytes, 16 << 20);
            std::cerr << "Native shader dictionary: " << dictionaries.size() << " entries\n";
        }

        std::vector<CopiedProgram> programs;
        std::set<std::string> identities;
        const std::unordered_set<std::string> selected(request.identities.begin(), request.identities.end());
        std::string current_member;
        NativeShaderReport report;
        report.by_stage = stage_counts();
        std::uint64_t copied = 0;
        sn::Enumerator enumerator([&](const sn::Record& record) {
            ++report.enumerated;
            auto parsed = sn::parse_program(record);
            if (const auto* failure = std::get_if<sn::ProgramFailure>(&parsed)) {
                ++report.refusals[metadata_name(failure->code)];
                return;
            }
            const auto& program = std::get<sn::Program>(parsed);
            if (!selected.empty() && !selected.contains(program.stage_identity)) return;
            if (!identities.insert(program.stage_identity).second) {
                ++report.duplicates;
                return;
            }
            std::cerr << "Native shader admitted " << program.stage_identity << ' ' << stage_name(program.stage) << ' '
                      << current_member << '\n';
            report.admitted.push_back(program.stage_identity + ' ' + stage_name(program.stage) + ' ' + current_member);
            ++report.by_stage[stage_name(program.stage)];
            if (request.enumerate_only) return;
            if (copied > maximum_copied_bytes - record.control.size() - record.code.size())
                throw sn::InputError("Native shader program copies exceed the memory policy");
            copied += record.control.size() + record.code.size();
            CopiedProgram owned;
            owned.kind = record.kind;
            owned.slot = record.slot;
            owned.member = current_member;
            owned.control.assign(record.control.begin(), record.control.end());
            owned.code.assign(record.code.begin(), record.code.end());
            programs.push_back(std::move(owned));
        }, limits, dictionaries);

        using Extent = std::tuple<std::uint64_t, std::uint64_t, std::array<std::byte, 32>>;
        std::map<Extent, std::pair<std::uint32_t, std::string>> extents;
        std::vector<char> matched_members(request.members.size(), 0);
        for (const auto& [name, id] : selection.romfs) {
            if (!member_selected(name, request.members, matched_members)) continue;
            const auto& file = source.archive->files().at(id);
            extents.try_emplace(Extent{file.offset, file.stored_size, file.hash}, id, name);
        }
        for (std::size_t index = 0; index < request.members.size(); ++index) {
            if (!matched_members[index])
                return std::string("Archive member was not selected: ") + request.members[index];
        }
        std::size_t scanned = 0;
        for (const auto& [extent, entry] : extents) {
            static_cast<void>(extent);
            if (entry.second == dictionary_member) continue;
            std::cerr << "Native shader member: " << entry.second << '\n';
            current_member = entry.second;
            try {
                auto bytes = source.read(entry.first, dictionaries, limits.member_bytes);
                const auto before = enumerator.inventory();
                enumerator.scan(bytes, entry.second);
                const auto after = enumerator.inventory();
                std::cerr << "Native shader member inventory records " << (after.records - before.records)
                          << " bnsh " << (after.bnsh - before.bnsh)
                          << " bahs " << (after.bahs - before.bahs)
                          << " nested " << (after.nested_candidates - before.nested_candidates)
                          << " unrecognized " << (after.unrecognized_spans - before.unrecognized_spans)
                          << " bytes " << bytes.size() << '\n';
            } catch (const sn::InputError& error) {
                if (!request.enumerate_only && !request.translate_only) throw;
                std::cerr << "Native shader member skipped " << entry.second << ' ' << error.what() << '\n';
            }
            if (++scanned % 5000 == 0)
                std::cerr << "Scanned " << scanned << " extents; records " << enumerator.inventory().records << '\n';
        }
        dictionaries.clear();
        report.unique = identities.size();
        std::cerr << "Native shader programs: " << report.unique << " unique, " << report.enumerated << " records\n";
        if (!selected.empty()) {
            std::size_t missing = 0;
            for (const auto& identity : selected) {
                if (identities.contains(identity)) continue;
                std::cerr << "Native shader identity was not admitted " << identity << '\n';
                ++missing;
            }
            if (missing) return "Requested stage identities were not admitted: " + std::to_string(missing);
        }
        if (request.enumerate_only) return report;

        const unsigned workers = programs.empty() ? 0 : static_cast<unsigned>(std::min<std::size_t>(request.jobs, programs.size()));
        std::vector<WorkerStats> stats(workers);
        BuildShared shared;
        shared.programs = &programs;
        shared.writer = &writer;
        shared.cache = request.compile_cache.empty()
                           ? (std::getenv("ARMIGO_SHADER_COMPILE_CACHE") && *std::getenv("ARMIGO_SHADER_COMPILE_CACHE")
                                  ? std::optional<fs::path>(std::getenv("ARMIGO_SHADER_COMPILE_CACHE"))
                                  : std::nullopt)
                           : std::optional<fs::path>(request.compile_cache);
        shared.work_root = request.output / "native" / ".work";
        shared.total = programs.size();
        shared.translate_only = request.translate_only;
        shared.source_directory = request.source_directory;
        shared.keep_sources = request.keep_sources;
        shared.maximum_source_bytes = request.maximum_source_bytes.value_or(
            request.keep_sources ? sn::maximum_native_source_bytes : sn::maximum_unstored_source_bytes);
        std::vector<pthread_t> threads(workers);
        std::vector<WorkerStart> starts(workers);
        pthread_attr_t attributes;
        if (workers != 0) {
            if (pthread_attr_init(&attributes) != 0)
                return std::string("The native shader translation thread could not start");
            if (pthread_attr_setstacksize(&attributes, translation_stack_bytes) != 0) {
                pthread_attr_destroy(&attributes);
                return std::string("The native shader translation thread could not start");
            }
        }
        unsigned started = 0;
        for (unsigned worker = 0; worker < workers; ++worker) {
            starts[worker] = WorkerStart{&shared, worker, &stats[worker]};
            if (pthread_create(&threads[worker], &attributes, worker_main, &starts[worker]) != 0) {
                shared.fail("The native shader translation thread could not start");
                break;
            }
            ++started;
        }
        threads.resize(started);
        if (workers != 0) pthread_attr_destroy(&attributes);
        for (unsigned worker = 0; worker < threads.size(); ++worker) pthread_join(threads[worker], nullptr);
        std::error_code ignored;
        fs::remove_all(shared.work_root, ignored);
        if (shared.failed.load()) return shared.error.empty() ? std::string("Native shader packaging failed") : shared.error;

        for (const WorkerStats& worker : stats) {
            report.translated += worker.translated;
            report.compiled += worker.compiled;
            for (const auto& [key, value] : worker.translated_by_stage) add_count(report.translated_by_stage, key.c_str(), value);
            for (const auto& [key, value] : worker.compiled_by_stage) add_count(report.compiled_by_stage, key.c_str(), value);
            for (const auto& [key, value] : worker.refusals) report.refusals[key] += value;
        }
        for (const char* stage : {"vertex", "fragment", "compute"}) {
            report.translated_by_stage.emplace(stage, 0);
            report.compiled_by_stage.emplace(stage, 0);
        }
        if (request.translate_only) return report;
        // Stage counts in the manifest name vertex, fragment, and compute, so
        // a section written before task and mesh kinds still loads. Task and
        // mesh counts stay on the report.
        auto schema_stages = [](std::map<std::string, std::uint64_t> counts) {
            counts.erase("task");
            counts.erase("mesh");
            for (const char* stage : {"vertex", "fragment", "compute"}) counts.emplace(stage, 0);
            return counts;
        };
        sn::NativeCounts counts;
        counts.enumerated = report.enumerated;
        counts.unique = report.unique;
        counts.translated = report.translated;
        counts.compiled = report.compiled;
        counts.duplicates = report.duplicates;
        counts.by_stage = schema_stages(report.by_stage);
        counts.translated_by_stage = schema_stages(report.translated_by_stage);
        counts.compiled_by_stage = schema_stages(report.compiled_by_stage);
        counts.refusals = report.refusals;
        auto finished = writer.finish(std::move(counts), {});
        if (const auto* failure = std::get_if<std::string>(&finished)) return *failure;
        // Workers store stages in completion order; the list is sorted so two
        // builds of one archive write the same list.
        std::sort(shared.compute_identities.begin(), shared.compute_identities.end());
        if (const std::string written = write_compute_pipeline_manifest(request.output, shared.compute_identities);
            !written.empty())
            return written;
        cleanup.active = false;
        std::cerr << "Native shader package: translated " << report.translated << ", compiled " << report.compiled << '\n';
        return report;
    } catch (const sn::InputError& error) {
        return std::string(error.what());
    } catch (const std::bad_alloc&) {
        return std::string("Host memory for native shader packaging could not be allocated.");
    } catch (const std::exception& error) {
        return std::string(error.what());
    }
}

} // namespace armigo::pack
