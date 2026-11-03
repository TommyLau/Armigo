// ============================================================================
// Armigo Tests - Native Runtime Verification
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include <armigo/pack/ios_link.h>
#include <armigo/runtime/embedded_modules.h>
#include <armigo/memory/guest_memory_model.h>

#include <mach-o/loader.h>
#include <mach/machine.h>
#include <cstdlib>
#include <unistd.h>
#include <array>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <vector>

namespace {
using namespace armigo;
namespace fs = std::filesystem;
constexpr std::size_t page = 16384;
int failures = 0;
#define CHECK(value) do { if (!(value)) { std::fprintf(stderr, "Failed line %d: %s\n", __LINE__, #value); ++failures; } } while (false)

int generate(const fs::path& output) {
    runtime::EmbeddedNativeArtifacts artifacts;
    artifacts.preparation_identity = runtime::installed_embedded_preparation_identity();
    artifacts.title_id = 1;
    runtime::EmbeddedNativeModule module;
    module.name = "main";
    module.image.resize(page * 4);
    module.original_image_size = page * 2;
    module.key_page_offset = page * 2;
    module.appended_code_offset = page * 3;
    module.text_size = 8;
    module.code_bytes = 8;
    const std::array<std::uint32_t, 2> text{0xd2800540, 0xd65f03c0};
    const std::array<std::uint32_t, 2> appendix{0xd2800560, 0xd65f03c0};
    std::memcpy(module.image.data(), text.data(), sizeof text);
    std::memcpy(module.image.data() + page * 3, appendix.data(), sizeof appendix);
    module.image[page] = std::byte{7};
    artifacts.modules.push_back(std::move(module));
    const auto written = runtime::write_embedded_modules(output / "images", artifacts);
    CHECK(std::holds_alternative<std::monostate>(written));
    const auto emitted = pack::generate_ios_embedded_sources(output / "images", output);
    if (const auto* error = std::get_if<std::string>(&emitted)) std::fprintf(stderr, "%s\n", error->c_str());
    CHECK(std::holds_alternative<pack::IosLinkReport>(emitted));
    if (const auto* report = std::get_if<pack::IosLinkReport>(&emitted)) {
        CHECK(report->modules == 1);
        CHECK(report->executable_bytes == page * 2);
    }
    CHECK(std::holds_alternative<std::string>(pack::generate_ios_embedded_sources(output / "missing", output)));
    return failures ? 1 : 0;
}
#ifdef ARMIGO_TEST_LINKED
std::vector<std::byte> honest_player(const runtime::EmbeddedNativeFiles& files, const std::vector<char>& manifest) {
    std::vector<std::byte> binary(page * 4);
    mach_header_64 header{};
    header.magic = MH_MAGIC_64; header.cputype = CPU_TYPE_ARM64; header.filetype = MH_EXECUTE;
    header.ncmds = 2;
    header.sizeofcmds = sizeof(segment_command_64) + 2 * sizeof(section_64) + sizeof(build_version_command);
    segment_command_64 segment{};
    segment.cmd = LC_SEGMENT_64; segment.cmdsize = sizeof segment + 2 * sizeof(section_64);
    std::strcpy(segment.segname, "__TEXT"); segment.vmaddr = 0x100000000;
    segment.vmsize = binary.size(); segment.filesize = binary.size();
    segment.maxprot = segment.initprot = 5; segment.nsects = 2;
    section_64 code{}, metadata{};
    std::strcpy(code.sectname, "__armigocode"); std::strcpy(code.segname, "__TEXT");
    code.addr = segment.vmaddr + page; code.offset = page; code.size = page * 2; code.align = 14;
    code.flags = S_REGULAR | S_ATTR_PURE_INSTRUCTIONS;
    std::strcpy(metadata.sectname, "__const"); std::strcpy(metadata.segname, "__TEXT");
    metadata.addr = segment.vmaddr + page * 3; metadata.offset = page * 3; metadata.size = manifest.size();
    build_version_command version{};
    version.cmd = LC_BUILD_VERSION; version.cmdsize = sizeof version;
    version.platform = PLATFORM_IOS; version.minos = 26u << 16;
    constexpr auto segment_offset = sizeof header;
    constexpr auto code_offset = segment_offset + sizeof segment;
    constexpr auto metadata_offset = code_offset + sizeof code;
    constexpr auto version_offset = metadata_offset + sizeof metadata;
    const auto store = [&](std::size_t offset, const auto& value) { std::memcpy(binary.data() + offset, &value, sizeof value); };
    store(0, header); store(segment_offset, segment); store(code_offset, code);
    store(metadata_offset, metadata); store(version_offset, version);
    std::size_t cursor = page;
    for (const auto& backing : files.backings()) for (const auto& piece : backing.pieces) if (piece.executable) {
        std::copy(piece.bytes.begin(), piece.bytes.end(), binary.begin() + cursor);
        cursor += piece.bytes.size();
    }
    std::memcpy(binary.data() + metadata.offset, manifest.data(), manifest.size());
    return binary;
}
void publication(const fs::path& images) {
    auto pattern = (fs::temp_directory_path() / "armigo-ios-publication-XXXXXX").string();
    CHECK(mkdtemp(pattern.data()) != nullptr);
    const fs::path root = pattern;
    struct Cleanup { fs::path path; ~Cleanup() { fs::remove_all(path); } } cleanup{root};
    const char* inherited = std::getenv("PATH");
    const std::string previous = inherited ? inherited : "";
    struct RestorePath { std::string value; ~RestorePath() { setenv("PATH", value.c_str(), 1); } } restore{previous};
    fs::create_directory(root / "tools");
    fs::create_directory(root / "build");
    std::ofstream(root / "tools/cmake") << "#!/bin/sh\nexit 0\n";
    fs::permissions(root / "tools/cmake", fs::perms::owner_all);
    std::ofstream(root / "build/CMakeCache.txt") << "CMAKE_SYSTEM_NAME:STRING=iOS\n";
    setenv("PATH", (root / "tools").c_str(), 1);
    const auto loaded = runtime::open_embedded_modules(images);
    CHECK(std::holds_alternative<runtime::EmbeddedNativeFiles>(loaded));
    if (!std::holds_alternative<runtime::EmbeddedNativeFiles>(loaded)) return;
    const auto& files = std::get<runtime::EmbeddedNativeFiles>(loaded);
    std::ifstream input(images / "modules.armigo-embedded", std::ios::binary);
    std::vector<char> manifest((std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());
    const auto honest = honest_player(files, manifest);
    auto binary = honest;
    const auto store = [&](std::size_t offset, const auto& value) { std::memcpy(binary.data() + offset, &value, sizeof value); };
    mach_header_64 header{};
    std::memcpy(&header, binary.data(), sizeof header);
    constexpr auto segment_offset = sizeof header;
    constexpr auto code_offset = segment_offset + sizeof(segment_command_64);
    segment_command_64 segment{};
    std::memcpy(&segment, binary.data() + segment_offset, sizeof segment);
    section_64 code{};
    std::memcpy(&code, binary.data() + code_offset, sizeof code);
    const auto check = [&](bool accepted) {
        std::ofstream output(root / "build/armigo-ios-player", std::ios::binary | std::ios::trunc);
        output.write(reinterpret_cast<const char*>(binary.data()), binary.size()); output.close();
        fs::remove(root / "published");
        const auto linked = pack::link_ios_player({root / "build", images, root / "generated", root / "published"});
        CHECK(std::holds_alternative<pack::IosLinkReport>(linked) == accepted);
        CHECK(fs::exists(root / "published") == accepted);
    };
    check(true);
    binary[page] ^= std::byte{1}; check(false);
    binary = honest;
    auto outside = code; outside.addr = segment.vmaddr + segment.vmsize;
    store(code_offset, outside); check(false);
    binary = honest;
    auto prefix = code; std::strcpy(prefix.sectname, "__armigocodeX");
    store(code_offset, prefix); check(false);
    binary = honest;
    auto writable = segment; writable.maxprot = writable.initprot = 3;
    store(segment_offset, writable);
    auto second = segment; second.cmdsize = sizeof second; second.nsects = 0;
    second.vmaddr += 0x100000000; second.vmsize = page; second.filesize = page;
    std::strcpy(second.segname, "__TEXT_EXEC");
    auto extra = header; extra.ncmds++; extra.sizeofcmds += sizeof second;
    store(0, extra); store(sizeof header + header.sizeofcmds, second); check(false);
    std::strcpy(second.segname, "__TEXT");
    store(sizeof header + header.sizeofcmds, second); check(false);
}

std::string read_text(const fs::path& path) {
    std::ifstream input(path);
    return {std::istreambuf_iterator<char>(input), {}};
}
bool cache_records_embedded_sources(const fs::path& cache) {
    std::ifstream input(cache);
    std::string line;
    constexpr std::string_view key = "ARMIGO_IOS_EMBEDDED_SOURCES_DIR:";
    while (std::getline(input, line)) if (line.starts_with(key)) return true;
    return false;
}
void embedded_sources_cache(const fs::path& images) {
    auto pattern = (fs::temp_directory_path() / "armigo-ios-cache-XXXXXX").string();
    CHECK(mkdtemp(pattern.data()) != nullptr);
    const fs::path root = pattern;
    struct Cleanup { fs::path path; ~Cleanup() { fs::remove_all(path); } } cleanup{root};
    const char* inherited = std::getenv("PATH");
    const std::string previous = inherited ? inherited : "";
    struct RestorePath { std::string value; ~RestorePath() { setenv("PATH", value.c_str(), 1); } } restore{previous};
    fs::create_directories(root / "tools");
    fs::create_directories(root / "build");
    std::ofstream(root / "tools/cmake") << R"script(#!/bin/sh
build=
building=0
unset_var=0
dash_d=
expect_build=0
for arg in "$@"; do
    if [ "$expect_build" -eq 1 ]; then
        build=$arg
        building=1
        expect_build=0
        continue
    fi
    case "$arg" in
        --build) expect_build=1 ;;
        -UARMIGO_IOS_EMBEDDED_SOURCES_DIR) unset_var=1 ;;
        -DARMIGO_IOS_EMBEDDED_SOURCES_DIR=*) dash_d=${arg#-DARMIGO_IOS_EMBEDDED_SOURCES_DIR=} ;;
    esac
done
if [ -z "$build" ]; then
    for arg in "$@"; do
        case "$arg" in
            -*) ;;
            *) build=$arg ;;
        esac
    done
fi
if [ -z "$build" ]; then
    echo "fake cmake: missing build directory" >&2
    exit 2
fi
cache="$build/CMakeCache.txt"
state="$build/selected-sources.txt"
log="$build/cmake-calls.log"
selected=""
if [ -n "${ARMIGO_IOS_EMBEDDED_SOURCES_DIR:-}" ]; then
    selected=$ARMIGO_IOS_EMBEDDED_SOURCES_DIR
elif [ -n "$dash_d" ]; then
    selected=$dash_d
elif [ "$unset_var" -eq 0 ] && [ -f "$cache" ]; then
    selected=$(/usr/bin/sed -n 's/^ARMIGO_IOS_EMBEDDED_SOURCES_DIR:[^=]*=//p' "$cache" | /usr/bin/head -n 1)
fi
if [ "$building" -eq 1 ]; then
    current=$(/bin/cat "$state" 2>/dev/null || true)
    printf 'BUILD %s\n' "$current" >> "$log"
    if [ -f "$build/fail-next-build" ]; then
        /bin/rm -f "$build/fail-next-build"
        exit 1
    fi
    exit 0
fi
if [ -n "$dash_d" ]; then
    if [ -f "$cache" ]; then
        /usr/bin/grep -v '^ARMIGO_IOS_EMBEDDED_SOURCES_DIR:' "$cache" > "$cache.tmp" || true
    else
        : > "$cache.tmp"
    fi
    printf 'ARMIGO_IOS_EMBEDDED_SOURCES_DIR:PATH=%s\n' "$dash_d" >> "$cache.tmp"
    /bin/mv "$cache.tmp" "$cache"
fi
if [ "$unset_var" -eq 1 ] && [ -f "$cache" ]; then
    /usr/bin/grep -v '^ARMIGO_IOS_EMBEDDED_SOURCES_DIR:' "$cache" > "$cache.tmp" || true
    /bin/mv "$cache.tmp" "$cache"
fi
printf '%s' "$selected" > "$state"
printf 'CONFIGURE %s\n' "$selected" >> "$log"
exit 0
)script";
    fs::permissions(root / "tools/cmake", fs::perms::owner_all);
    std::ofstream(root / "build/CMakeCache.txt")
        << "CMAKE_SYSTEM_NAME:STRING=iOS\n"
        << "ARMIGO_IOS_EMBEDDED_SOURCES_DIR:PATH=/previous/pack/sources\n";
    setenv("PATH", (root / "tools").c_str(), 1);
    const auto loaded = runtime::open_embedded_modules(images);
    CHECK(std::holds_alternative<runtime::EmbeddedNativeFiles>(loaded));
    if (!std::holds_alternative<runtime::EmbeddedNativeFiles>(loaded)) return;
    std::ifstream manifest_input(images / "modules.armigo-embedded", std::ios::binary);
    std::vector<char> manifest((std::istreambuf_iterator<char>(manifest_input)), std::istreambuf_iterator<char>());
    const auto player = honest_player(std::get<runtime::EmbeddedNativeFiles>(loaded), manifest);
    {
        std::ofstream output(root / "build/armigo-ios-player", std::ios::binary | std::ios::trunc);
        output.write(reinterpret_cast<const char*>(player.data()), static_cast<std::streamsize>(player.size()));
    }
    const auto link = [&](const char* name, bool succeed) {
        const fs::path work = fs::absolute(root / name);
        const fs::path published = root / (std::string(name) + "-player");
        if (!succeed) std::ofstream(root / "build/fail-next-build") << "fail\n";
        const auto linked = pack::link_ios_player({root / "build", images, work, published});
        if (const auto* error = std::get_if<std::string>(&linked)) {
            if (succeed) std::fprintf(stderr, "%s\n", error->c_str());
        }
        CHECK(std::holds_alternative<pack::IosLinkReport>(linked) == succeed);
        CHECK(fs::exists(published) == succeed);
        CHECK(!cache_records_embedded_sources(root / "build/CMakeCache.txt"));
        CHECK(read_text(root / "build/selected-sources.txt").empty());
        return work.string();
    };
    const auto first = link("work-a", true);
    const auto second = link("work-b", true);
    link("work-c", false);
    std::vector<std::string> builds;
    std::ifstream log(root / "build/cmake-calls.log");
    std::string line;
    while (std::getline(log, line)) if (line.starts_with("BUILD ")) builds.push_back(line.substr(6));
    CHECK(builds.size() == 3);
    if (builds.size() == 3) {
        CHECK(builds[0] == first);
        CHECK(builds[1] == second);
        CHECK(builds[0] != builds[1]);
        CHECK(builds[2] == fs::absolute(root / "work-c").string());
    }
    const auto transcript = read_text(root / "build/cmake-calls.log");
    CHECK(transcript.find("/previous/pack/sources") == std::string::npos);
}

void linked() {
    CHECK(armigo_linked_embedded_bundle != nullptr);
    const auto* bundle = armigo_linked_embedded_bundle();
    CHECK(bundle != nullptr);
    if (!bundle) return;
    const auto opened = runtime::open_linked_embedded_modules(*bundle);
    if (const auto* error = std::get_if<runtime::LaunchFailure>(&opened)) std::fprintf(stderr, "%s\n", error->detail.c_str());
    CHECK(std::holds_alternative<runtime::EmbeddedNativeFiles>(opened));
    if (const auto* files = std::get_if<runtime::EmbeddedNativeFiles>(&opened)) {
        CHECK(files->artifacts().modules.size() == 1);
        const auto& module = files->artifacts().modules.front();
        CHECK(module.image.size() == page * 4);
        using P = memory::Permission;
        const std::array<platform::ModuleRegion, 3> regions{{{0, page, P::read | P::execute},
            {page, page * 2, P::read | P::write}, {page * 3, page, P::read | P::execute}}};
        auto* source_data = const_cast<std::byte*>(bundle->backings[0].pieces[1].bytes.data());
        source_data[0] = std::byte{123};
        source_data[page + 128] = std::byte{0xab};
        auto result = platform::HostModule::create_embedded(module.image.size(), regions, files->backings()[0].pieces);
        source_data[0] = std::byte{7};
        source_data[page + 128] = std::byte{0};
        CHECK(result.status.ok());
        if (result.module) {
            auto& mapping = *result.module;
            CHECK(mapping.preparation_bytes()[page] == std::byte{7});
            CHECK(mapping.preparation_bytes()[page * 2 + 128] == std::byte{0});
            mapping.preparation_bytes()[page] = std::byte{99};
            mapping.preparation_bytes()[page * 2] = std::byte{13};
            CHECK(mapping.seal().ok());
            const auto base = mapping.base_address();
            CHECK(reinterpret_cast<int (*)()>(base)() == 42);
            CHECK(reinterpret_cast<int (*)()>(base + page * 3)() == 43);
            CHECK(module.image[page] == std::byte{7});
            CHECK(module.image[page * 2] == std::byte{0});
            CHECK(files->backings()[0].pieces[1].bytes[0] == std::byte{7});
            CHECK(files->backings()[0].pieces[1].bytes[page] == std::byte{0});
            CHECK(std::memcmp(reinterpret_cast<const void*>(base), module.image.data(), page) == 0);
        }
    }
    std::vector<platform::EmbeddedModulePiece> pieces(bundle->backings[0].pieces.begin(), bundle->backings[0].pieces.end());
    std::array<runtime::EmbeddedNativeBacking, 1> backings{{{pieces}}};
    auto invalid = *bundle;
    invalid.backings = backings;
    pieces[0].executable = false;
    CHECK(std::holds_alternative<runtime::LaunchFailure>(runtime::open_linked_embedded_modules(invalid)));
    pieces[0] = bundle->backings[0].pieces[0];
    pieces[0].offset = page;
    CHECK(std::holds_alternative<runtime::LaunchFailure>(runtime::open_linked_embedded_modules(invalid)));
    pieces[0] = bundle->backings[0].pieces[0];
    pieces[0].bytes = pieces[0].bytes.subspan(1);
    CHECK(std::holds_alternative<runtime::LaunchFailure>(runtime::open_linked_embedded_modules(invalid)));
    pieces[0] = bundle->backings[0].pieces[0];
    alignas(page) static std::array<std::byte, page> unsigned_code{};
    std::copy(pieces[0].bytes.begin(), pieces[0].bytes.end(), unsigned_code.begin());
    pieces[0].bytes = unsigned_code;
    CHECK(std::holds_alternative<runtime::LaunchFailure>(runtime::open_linked_embedded_modules(invalid)));
    pieces[0] = bundle->backings[0].pieces[0];
    std::vector<std::byte> manifest(bundle->manifest.begin(), bundle->manifest.end());
    invalid = *bundle;
    invalid.manifest = manifest;
    CHECK(std::holds_alternative<runtime::LaunchFailure>(runtime::open_linked_embedded_modules(invalid)));
    manifest[0] ^= std::byte{1};
    invalid.manifest = manifest;
    CHECK(std::holds_alternative<runtime::LaunchFailure>(runtime::open_linked_embedded_modules(invalid)));
    invalid = *bundle;
    invalid.backings = {};
    CHECK(std::holds_alternative<runtime::LaunchFailure>(runtime::open_linked_embedded_modules(invalid)));
}
#endif
}
int main(int argc, char** argv) {
    if (argc == 3 && std::string_view(argv[1]) == "--generate") return generate(argv[2]);
    if (argc == 6 && std::string_view(argv[1]) == "--link") {
        const auto result = armigo::pack::link_ios_player({argv[2], argv[3], argv[4], argv[5]});
        if (const auto* error = std::get_if<std::string>(&result)) {
            std::fprintf(stderr, "%s\n", error->c_str());
            return 1;
        }
        return 0;
    }
#ifdef ARMIGO_TEST_LINKED
    linked();
    CHECK(argc == 2);
    if (argc == 2) {
        publication(argv[1]);
        embedded_sources_cache(argv[1]);
    }
    if (!failures) std::puts("Linked embedded code executes; private data and rejection checks passed.");
    return failures ? 1 : 0;
#else
    return 2;
#endif
}
