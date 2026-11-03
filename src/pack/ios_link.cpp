// ============================================================================
// Armigo Tools - Development and Verification Tools
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include <armigo/pack/ios_link.h>
#include <armigo/runtime/embedded_modules.h>

#include <CommonCrypto/CommonDigest.h>
#include <mach-o/loader.h>
#include <mach/machine.h>
#include <mach/vm_prot.h>
#include <spawn.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <cstring>
#include <fstream>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <vector>

extern char** environ;

namespace armigo::pack {
namespace {
namespace fs = std::filesystem;
void require(bool value, const std::string& detail) {
    if (!value) throw std::runtime_error(detail);
}
std::vector<std::byte> read_file(const fs::path& path, std::size_t limit) {
    const auto size = fs::file_size(path);
    require(size <= limit, "Embedded input exceeds its size limit.");
    std::vector<std::byte> bytes(size);
    std::ifstream input(path, std::ios::binary);
    require(bool(input.read(reinterpret_cast<char*>(bytes.data()), bytes.size())), "Cannot read embedded input.");
    return bytes;
}
void write_text(const fs::path& path, const std::string& text) {
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    require(bool(output.write(text.data(), text.size())), "Cannot write generated embedded source.");
}
std::string quote_path(const fs::path& path) {
    std::string result = "\"";
    for (char c : fs::absolute(path).string()) {
        require(c != '\n' && c != '\r' && c != '\0', "Embedded input path contains a control character.");
        if (c == '\\' || c == '"') result += '\\';
        result += c;
    }
    return result + "\"";
}
std::string image_file(const runtime::EmbeddedNativeModule& module, const fs::path& directory) {
    std::array<unsigned char, CC_SHA256_DIGEST_LENGTH> digest{};
    require(module.image.size() <= UINT32_MAX &&
            CC_SHA256(module.image.data(), static_cast<CC_LONG>(module.image.size()), digest.data()),
            "Cannot hash an embedded image.");
    constexpr char hex[] = "0123456789abcdef";
    std::string name;
    for (const auto byte : digest) { name += hex[byte >> 4]; name += hex[byte & 15]; }
    return quote_path(directory / (name + ".armigo-image"));
}
struct ScopedEnvironment {
    std::string name;
    std::optional<std::string> previous;
    ScopedEnvironment(const char* name, const char* value) : name(name) {
        if (const char* existing = std::getenv(name)) previous = existing;
        if (value) require(setenv(name, value, 1) == 0, "Cannot set the iOS embedded sources directory.");
        else unsetenv(name);
    }
    ~ScopedEnvironment() {
        if (previous) setenv(name.c_str(), previous->c_str(), 1);
        else unsetenv(name.c_str());
    }
    ScopedEnvironment(const ScopedEnvironment&) = delete;
    ScopedEnvironment& operator=(const ScopedEnvironment&) = delete;
};
void run(std::vector<std::string> arguments) {
    std::vector<char*> pointers;
    for (auto& argument : arguments) pointers.push_back(argument.data());
    pointers.push_back(nullptr);
    pid_t child = 0;
    const int status = posix_spawnp(&child, pointers[0], nullptr, nullptr, pointers.data(), environ);
    require(status == 0, "Cannot start iOS link tool: " + std::string(std::strerror(status)));
    int result = 0;
    pid_t waited;
    do { waited = waitpid(child, &result, 0); } while (waited < 0 && errno == EINTR);
    require(waited == child && WIFEXITED(result) && WEXITSTATUS(result) == 0,
            "The iOS player configure/build step failed.");
}
void verify_executable(const fs::path& path, std::span<const std::byte> manifest,
                       const runtime::EmbeddedNativeFiles& artifacts) {
    const auto file_size = fs::file_size(path);
    std::ifstream file(path, std::ios::binary);
    mach_header_64 header{};
    require(bool(file.read(reinterpret_cast<char*>(&header), sizeof header)) && header.magic == MH_MAGIC_64 &&
            header.cputype == CPU_TYPE_ARM64 && header.filetype == MH_EXECUTE && header.sizeofcmds <= (16u << 20),
            "The linked iOS player is not a bounded arm64 Mach-O executable.");
    std::vector<std::byte> commands(header.sizeofcmds);
    require(bool(file.read(reinterpret_cast<char*>(commands.data()), commands.size())), "Cannot read linked Mach-O commands.");
    std::vector<std::span<const std::byte>> expected_code;
    std::size_t code_bytes = 0;
    for (const auto& backing : artifacts.backings()) for (const auto& piece : backing.pieces) if (piece.executable) {
        expected_code.push_back(piece.bytes);
        code_bytes += piece.bytes.size();
    }
    require(!manifest.empty() && code_bytes > 0, "Embedded publication requires a manifest and executable bytes.");
    const auto name = [](const char (&text)[16]) { return std::string_view(text, strnlen(text, sizeof text)); };
    const auto within = [](std::uint64_t start, std::uint64_t size, std::uint64_t base, std::uint64_t length) {
        return start >= base && start - base <= length && size <= length - (start - base);
    };
    std::size_t offset = 0;
    unsigned executable_segments = 0, text_segments = 0, code_sections = 0;
    bool ios = false, embedded_manifest = false;
    for (std::uint32_t i = 0; i < header.ncmds; ++i) {
        require(offset <= commands.size() && sizeof(load_command) <= commands.size() - offset,
                "Truncated linked Mach-O command.");
        load_command command{};
        std::memcpy(&command, commands.data() + offset, sizeof command);
        require(command.cmdsize >= sizeof command && command.cmdsize <= commands.size() - offset,
                "Invalid linked Mach-O command size.");
        if (command.cmd == LC_SEGMENT_64) {
            require(command.cmdsize >= sizeof(segment_command_64), "Truncated linked Mach-O segment.");
            segment_command_64 segment{};
            std::memcpy(&segment, commands.data() + offset, sizeof segment);
            const bool text = name(segment.segname) == "__TEXT";
            if (text) ++text_segments;
            const bool immutable_code = segment.initprot == (VM_PROT_READ | VM_PROT_EXECUTE) &&
                                        segment.maxprot == (VM_PROT_READ | VM_PROT_EXECUTE);
            if (segment.initprot & VM_PROT_EXECUTE) {
                ++executable_segments;
                require(text && immutable_code, "The linked player has writable code or an unexpected executable segment.");
            }
            require(within(segment.fileoff, segment.filesize, 0, file_size) &&
                    segment.nsects <= (command.cmdsize - sizeof segment) / sizeof(section_64),
                    "Invalid linked Mach-O segment or section count.");
            for (std::uint32_t j = 0; j < segment.nsects; ++j) {
                section_64 section{};
                std::memcpy(&section, commands.data() + offset + sizeof segment + j * sizeof section, sizeof section);
                const bool code = name(section.sectname) == "__armigocode";
                const bool constants = name(section.sectname) == "__const";
                if (!code && !constants) continue;
                if (!code && !text) continue;
                require(text && immutable_code && name(section.segname) == "__TEXT" &&
                        (section.flags & SECTION_TYPE) == S_REGULAR &&
                        within(section.offset, section.size, segment.fileoff, segment.filesize) &&
                        within(section.addr, section.size, segment.vmaddr, segment.vmsize) &&
                        section.addr - segment.vmaddr == section.offset - segment.fileoff,
                        "Embedded section is outside the immutable executable segment.");
                file.seekg(section.offset);
                if (code) {
                    ++code_sections;
                    require(section.size == code_bytes && section.align >= 14 &&
                            section.offset % 16384 == 0 && section.addr % 16384 == 0,
                            "The linked embedded code has the wrong size or alignment.");
                    std::array<std::byte, 65536> buffer{};
                    for (auto expected : expected_code) while (!expected.empty()) {
                        const auto size = std::min(expected.size(), buffer.size());
                        require(bool(file.read(reinterpret_cast<char*>(buffer.data()), size)) &&
                                std::equal(buffer.begin(), buffer.begin() + size, expected.begin()),
                                "The linked executable bytes differ from the verified artifacts.");
                        expected = expected.subspan(size);
                    }
                } else {
                    require(section.size <= (128u << 20), "Linked constant section exceeds its size limit.");
                    std::vector<std::byte> constants_bytes(section.size);
                    require(bool(file.read(reinterpret_cast<char*>(constants_bytes.data()), constants_bytes.size())),
                            "Cannot read the linked embedded manifest.");
                    embedded_manifest |= std::search(constants_bytes.begin(), constants_bytes.end(),
                        manifest.begin(), manifest.end()) != constants_bytes.end();
                }
            }
        } else if (command.cmd == LC_BUILD_VERSION) {
            require(command.cmdsize >= sizeof(build_version_command), "Truncated Mach-O build identity.");
            build_version_command build{};
            std::memcpy(&build, commands.data() + offset, sizeof build);
            ios = build.platform == PLATFORM_IOS && build.minos == (26u << 16);
        }
        offset += command.cmdsize;
    }
    require(embedded_manifest && code_sections == 1, "The linked player does not contain the requested embedded code and manifest.");
    require(offset == commands.size() && ios && executable_segments == 1 && text_segments == 1,
            "The player must target iOS 26.0 and have exactly one executable segment.");
}
} // namespace

std::variant<std::string, IosLinkReport> generate_ios_embedded_sources(
    const fs::path& embedded_directory, const fs::path& output_directory) {
    try {
        auto opened = runtime::open_embedded_modules(embedded_directory);
        if (const auto* error = std::get_if<runtime::LaunchFailure>(&opened)) return error->detail;
        const auto& files = std::get<runtime::EmbeddedNativeFiles>(opened);
        fs::create_directories(output_directory);
        std::ostringstream assembly, source;
        assembly << "// Generated embedded module backing.\n";
        source << "// Generated embedded module provider.\n#include <armigo/runtime/embedded_modules.h>\n"
                  "using namespace armigo::runtime;\nextern \"C\" {\n"
                  "extern const std::byte armigo_embedded_manifest[];\n";
        const auto manifest = embedded_directory / "modules.armigo-embedded";
        const auto metadata = read_file(manifest, 64u << 20);
        assembly << ".section __TEXT,__const\n.p2align 3\n.globl _armigo_embedded_manifest\n"
                    "_armigo_embedded_manifest:\n.incbin " << quote_path(manifest) << "\n";
        IosLinkReport report;
        report.modules = files.artifacts().modules.size();
        for (std::size_t i = 0; i < report.modules; ++i) {
            const auto input = image_file(files.artifacts().modules[i], embedded_directory);
            const auto& pieces = files.backings()[i].pieces;
            for (std::size_t j = 0; j < pieces.size(); ++j) {
                const auto& piece = pieces[j];
                const auto name = "armigo_embedded_" + std::to_string(i) + "_" + std::to_string(j);
                assembly << (piece.executable ? ".section __TEXT,__armigocode,regular,pure_instructions\n"
                                              : ".section __DATA,__armigodata\n")
                         << ".p2align 14\n.globl _" << name << "\n_" << name << ":\n.incbin " << input
                         << ", " << piece.offset << ", " << piece.bytes.size() << "\n";
                source << "extern const std::byte " << name << "[];\n";
                if (piece.executable) report.executable_bytes += piece.bytes.size();
            }
        }
        source << "}\nnamespace {\n";
        for (std::size_t i = 0; i < report.modules; ++i) {
            source << "const armigo::platform::EmbeddedModulePiece pieces_" << i << "[] = {\n";
            const auto& pieces = files.backings()[i].pieces;
            for (std::size_t j = 0; j < pieces.size(); ++j) {
                const auto& piece = pieces[j];
                source << "{" << piece.offset << ", {armigo_embedded_" << i << '_' << j << ", "
                       << piece.bytes.size() << "}, " << (piece.executable ? "true" : "false") << "},\n";
            }
            source << "};\n";
        }
        source << "const EmbeddedNativeBacking backings[] = {\n";
        for (std::size_t i = 0; i < report.modules; ++i) source << "{pieces_" << i << "},\n";
        source << "};\nconst LinkedEmbeddedBundle bundle{{armigo_embedded_manifest, " << metadata.size()
               << "}, backings};\n}\nextern \"C\" const LinkedEmbeddedBundle* armigo_linked_embedded_bundle() { return &bundle; }\n";
        write_text(output_directory / "embedded.S", assembly.str());
        write_text(output_directory / "embedded.cpp", source.str());
        return report;
    } catch (const std::exception& error) { return std::string(error.what()); }
}

std::variant<std::string, IosLinkReport> link_ios_player(const IosLinkOptions& options) {
    try {
        const auto build = fs::absolute(options.player_build_directory);
        require(fs::is_regular_file(build / "CMakeCache.txt"), "The iOS player build must be configured first.");
        const auto sources = fs::absolute(options.work_directory);
        auto generated = generate_ios_embedded_sources(options.embedded_directory, sources);
        if (std::holds_alternative<std::string>(generated)) return generated;
        // Pass the directory through the environment of this configure and build,
        // then configure again with it unset. The second configure drops those
        // sources from the generated build files on success and on failure.
        std::string build_error;
        try {
            const ScopedEnvironment selection("ARMIGO_IOS_EMBEDDED_SOURCES_DIR", sources.c_str());
            run({"cmake", "-UARMIGO_IOS_EMBEDDED_SOURCES_DIR", build.string()});
            run({"cmake", "--build", build.string(), "--target", "armigo-ios-player", "--parallel", "4"});
        } catch (const std::exception& error) {
            build_error = error.what();
        }
        std::string clear_error;
        try {
            const ScopedEnvironment clear("ARMIGO_IOS_EMBEDDED_SOURCES_DIR", nullptr);
            run({"cmake", "-UARMIGO_IOS_EMBEDDED_SOURCES_DIR", build.string()});
        } catch (const std::exception& error) {
            clear_error = error.what();
        }
        if (!build_error.empty()) {
            if (!clear_error.empty()) return build_error + " " + clear_error;
            return build_error;
        }
        if (!clear_error.empty()) return "The iOS player build kept its generated sources configured: " + clear_error;
        const auto bare_player = build / "armigo-ios-player";
        const auto player = fs::is_regular_file(bare_player) ? bare_player :
            build / "armigo-ios-player.app" / "armigo-ios-player";
        const auto manifest = read_file(options.embedded_directory / "modules.armigo-embedded", 64u << 20);
        auto opened = runtime::open_embedded_modules(options.embedded_directory);
        if (const auto* error = std::get_if<runtime::LaunchFailure>(&opened)) return error->detail;
        verify_executable(player, manifest, std::get<runtime::EmbeddedNativeFiles>(opened));
        const auto output = fs::absolute(options.output_executable);
        fs::create_directories(output.parent_path());
        auto temporary = (output.parent_path() / ".armigo-player-XXXXXX").string();
        const int descriptor = mkstemp(temporary.data());
        require(descriptor >= 0, "Cannot create the player publication file.");
        close(descriptor);
        struct Cleanup { fs::path path; ~Cleanup() { std::error_code error; fs::remove(path, error); } } cleanup{temporary};
        fs::copy_file(player, temporary, fs::copy_options::overwrite_existing);
        fs::permissions(temporary, fs::perms::owner_all | fs::perms::group_read | fs::perms::group_exec |
                                  fs::perms::others_read | fs::perms::others_exec);
        fs::rename(temporary, output);
        return generated;
    } catch (const std::exception& error) { return std::string(error.what()); }
}
} // namespace armigo::pack
