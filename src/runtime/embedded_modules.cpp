// ============================================================================
// Armigo Runtime - Native Launch Loading
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include <armigo/runtime/embedded_modules.h>
#include <armigo/execution/embedded_patch.h>
#include "sha256.h"
#include <TargetConditionals.h>
#include <mach-o/dyld.h>
#include <mach-o/loader.h>
#include <mach/vm_prot.h>
#include <mach/mach.h>
#include <mach/vm_map.h>

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <set>
#include <stdexcept>
#include <string_view>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <utility>

namespace armigo::runtime {
namespace {
using Digest = std::array<std::byte, 32>;
constexpr std::size_t metadata_limit = 64 * 1024 * 1024;
constexpr std::size_t image_limit = 1ull << 30;
constexpr std::size_t total_image_limit = 2ull << 30;
constexpr std::size_t record_limit = 1'000'000;
constexpr std::size_t page = execution::embedded_key_page_size;
constexpr std::string_view manifest_name = "modules.armigo-embedded";
constexpr std::string_view magic = "ARMGEMB1";

struct FormatError : std::runtime_error { using std::runtime_error::runtime_error; };
void require(bool value, const char* message) { if (!value) throw FormatError(message); }
Digest hash(std::span<const std::byte> bytes) {
    const auto result = loader::detail::sha256(bytes);
    require(result.has_value(), "Embedded artifact hashing failed.");
    return *result;
}
std::string image_name(const Digest& digest) {
    constexpr char hex[] = "0123456789abcdef";
    std::string name;
    for (const auto byte : digest) {
        const auto value = std::to_integer<unsigned>(byte);
        name.push_back(hex[value >> 4]); name.push_back(hex[value & 15]);
    }
    return name + ".armigo-image";
}
struct Bytes {
    std::vector<std::byte> data;
    void raw(std::span<const std::byte> value) {
        require(value.size() <= metadata_limit - data.size(), "Embedded metadata exceeds its byte limit.");
        data.insert(data.end(), value.begin(), value.end());
    }
    void number(std::uint64_t value, unsigned width = 8) {
        std::array<std::byte, 8> bytes{};
        for (unsigned i = 0; i < width; ++i) bytes[i] = std::byte(value >> (i * 8));
        raw(std::span(bytes).first(width));
    }
    void count(std::size_t value, std::size_t limit = record_limit) {
        require(value <= limit, "Embedded metadata count exceeds its limit.");
        number(value, 4);
    }
    void string(std::string_view value, std::size_t limit = 4096) {
        count(value.size(), limit);
        raw(std::as_bytes(std::span(value.data(), value.size())));
    }
    void location(execution::PreparedLocation value) { number(value.module_index, 4); number(value.image_offset); }
};
struct Reader {
    std::span<const std::byte> bytes;
    std::size_t cursor = 0;
    std::span<const std::byte> take(std::size_t count) {
        require(count <= bytes.size() - cursor, "Embedded metadata is truncated.");
        const auto result = bytes.subspan(cursor, count); cursor += count; return result;
    }
    std::uint64_t number(unsigned width = 8) {
        std::uint64_t result = 0;
        const auto value = take(width);
        for (unsigned i = 0; i < width; ++i) result |= std::uint64_t(value[i]) << (i * 8);
        return result;
    }
    std::size_t count(std::size_t minimum, std::size_t limit = record_limit) {
        const auto result = number(4);
        require(result <= limit && result <= (bytes.size() - cursor) / minimum,
                "Embedded metadata count exceeds its bound.");
        return result;
    }
    std::string string(std::size_t limit = 4096) {
        const auto value = take(count(1, limit));
        return {reinterpret_cast<const char*>(value.data()), value.size()};
    }
    Digest digest() { Digest result; const auto value = take(32); std::copy(value.begin(), value.end(), result.begin()); return result; }
    execution::PreparedLocation location() { const auto index = number(4); return {static_cast<std::uint32_t>(index), number()}; }
};

void geometry(const EmbeddedNativeModule& module, std::size_t size) {
    require(!module.name.empty() && module.name.size() <= 256 && module.name.find('\0') == std::string::npos,
            "Embedded module name is invalid.");
    require(size <= image_limit && size % page == 0 && module.original_image_size != 0 &&
        module.original_image_size <= module.key_page_offset && module.key_page_offset < size &&
        module.key_page_offset % page == 0 && module.appended_code_offset == module.key_page_offset + page &&
        module.appended_code_offset < size && module.text_offset % 4 == 0 && module.text_size % 4 == 0 &&
        module.text_offset < module.original_image_size && module.text_size <= module.original_image_size - module.text_offset &&
        module.code_bytes <= size - module.appended_code_offset,
        "Embedded module geometry is invalid.");
}
struct Metadata {
    EmbeddedNativeArtifacts artifacts;
    std::vector<Digest> hashes;
    std::vector<std::size_t> sizes;
};
Bytes encode(const EmbeddedNativeArtifacts& artifacts, std::vector<Digest>& hashes) {
    require(artifacts.version == 1 && artifacts.preparation_identity == installed_embedded_preparation_identity() &&
            !artifacts.modules.empty(), "Unsupported embedded artifact envelope.");
    Bytes bytes;
    bytes.raw(std::as_bytes(std::span(magic.data(), magic.size())));
    bytes.number(artifacts.version, 4); bytes.number(artifacts.title_id);
    bytes.string(artifacts.preparation_identity, 256); bytes.raw(artifacts.code_settings_sha256);
    bytes.count(artifacts.modules.size(), 64);
    std::size_t total = 0;
    std::set<std::string> names;
    for (const auto& module : artifacts.modules) {
        geometry(module, module.image.size());
        require(names.insert(module.name).second && module.image.size() <= total_image_limit - total,
                "Embedded module names or aggregate image size are invalid.");
        total += module.image.size();
        require(std::all_of(module.image.begin() + module.key_page_offset,
                            module.image.begin() + module.appended_code_offset,
                            [](std::byte byte) { return byte == std::byte{0}; }),
                "Embedded image contains runtime key bindings.");
        bytes.string(module.name, 256); bytes.raw(module.source_sha256);
        hashes.push_back(hash(module.image)); bytes.raw(hashes.back()); bytes.number(module.image.size());
        bytes.number(module.original_image_size); bytes.number(module.key_page_offset); bytes.number(module.appended_code_offset);
        bytes.number(module.text_offset); bytes.number(module.text_size); bytes.number(module.host_executes, 1);
        bytes.number(module.code_bytes); bytes.number(module.scalar_sites); bytes.number(module.thread_register_sites);
        bytes.count(module.sites.size());
        for (const auto& site : module.sites) {
            bytes.number(site.return_address_offset); bytes.number(site.instruction, 4);
            bytes.count(site.allowed_indirect_targets.size());
            for (auto location : site.allowed_indirect_targets) bytes.location(location);
        }
        bytes.count(module.entry_offsets.size()); for (auto offset : module.entry_offsets) bytes.number(offset);
        bytes.count(module.foreign_entries.size()); for (auto location : module.foreign_entries) bytes.location(location);
        bytes.count(module.stops.size());
        for (const auto& stop : module.stops) { bytes.number(stop.image_offset); bytes.number(stop.instruction, 4); bytes.string(stop.reason); }
        bytes.count(module.hooks.size(), 64);
        for (const auto& hook : module.hooks) { bytes.number(hook.return_offset); bytes.number(hook.call_return_offset); bytes.location(hook.target); }
    }
    bytes.raw(hash(bytes.data));
    return bytes;
}
Metadata decode(std::span<const std::byte> bytes) {
    require(bytes.size() >= 32 && bytes.size() <= metadata_limit, "Embedded metadata size is invalid.");
    const auto body = bytes.first(bytes.size() - 32);
    const auto digest = hash(body);
    require(std::equal(digest.begin(), digest.end(), bytes.end() - 32), "Embedded metadata checksum mismatch.");
    Reader reader{body};
    const auto signature = reader.take(magic.size());
    require(std::memcmp(signature.data(), magic.data(), magic.size()) == 0, "Unknown embedded metadata format.");
    Metadata result;
    auto& artifacts = result.artifacts;
    artifacts.version = reader.number(4); artifacts.title_id = reader.number();
    artifacts.preparation_identity = reader.string(256); artifacts.code_settings_sha256 = reader.digest();
    require(artifacts.version == 1 && artifacts.preparation_identity == installed_embedded_preparation_identity(),
            "Embedded preparation identity is unsupported.");
    artifacts.modules.resize(reader.count(1, 64));
    require(!artifacts.modules.empty(), "Embedded module roster is empty.");
    std::size_t total = 0;
    std::set<std::string> names;
    for (auto& module : artifacts.modules) {
        module.name = reader.string(256); module.source_sha256 = reader.digest();
        result.hashes.push_back(reader.digest()); const auto size = reader.number();
        module.original_image_size = reader.number(); module.key_page_offset = reader.number();
        module.appended_code_offset = reader.number(); module.text_offset = reader.number(); module.text_size = reader.number();
        const auto executable = reader.number(1);
        require(executable <= 1, "Invalid embedded executable flag."); module.host_executes = executable;
        module.code_bytes = reader.number(); module.scalar_sites = reader.number(); module.thread_register_sites = reader.number();
        geometry(module, size);
        require(names.insert(module.name).second && size <= total_image_limit - total,
                "Embedded module roster or aggregate image size is invalid.");
        total += size; result.sizes.push_back(size);
        module.sites.resize(reader.count(16));
        for (auto& site : module.sites) {
            site.return_address_offset = reader.number(); site.instruction = reader.number(4);
            site.allowed_indirect_targets.resize(reader.count(12));
            for (auto& location : site.allowed_indirect_targets) location = reader.location();
        }
        module.entry_offsets.resize(reader.count(8)); for (auto& offset : module.entry_offsets) offset = reader.number();
        module.foreign_entries.resize(reader.count(12)); for (auto& location : module.foreign_entries) location = reader.location();
        module.stops.resize(reader.count(16));
        for (auto& stop : module.stops) { stop.image_offset = reader.number(); stop.instruction = reader.number(4); stop.reason = reader.string(); }
        module.hooks.resize(reader.count(28, 64));
        for (auto& hook : module.hooks) { hook.return_offset = reader.number(); hook.call_return_offset = reader.number(); hook.target = reader.location(); }
    }
    require(reader.cursor == body.size(), "Embedded metadata has trailing bytes.");
    return result;
}

struct Descriptor {
    int value = -1;
    explicit Descriptor(int descriptor) : value(descriptor) {}
    ~Descriptor() { if (value >= 0) ::close(value); }
};
struct Mapping {
    void* address = MAP_FAILED;
    std::size_t size = 0;
    Mapping() = default;
    Mapping(const Mapping&) = delete;
    Mapping& operator=(const Mapping&) = delete;
    Mapping(Mapping&& other) noexcept : address(std::exchange(other.address, MAP_FAILED)), size(other.size) {}
    ~Mapping() { if (address != MAP_FAILED) ::munmap(address, size); }
};
void atomic_write(const std::filesystem::path& path, std::span<const std::byte> bytes) {
    auto temporary = (path.parent_path() / ".armigo-embedded-XXXXXX").string();
    Descriptor file(::mkstemp(temporary.data()));
    require(file.value >= 0, "Cannot create an embedded artifact temporary file.");
    struct Cleanup { std::string path; ~Cleanup() { ::unlink(path.c_str()); } } cleanup{temporary};
    std::size_t cursor = 0;
    while (cursor < bytes.size()) {
        const auto count = ::write(file.value, bytes.data() + cursor, bytes.size() - cursor);
        if (count < 0 && errno == EINTR) continue;
        require(count > 0, "Cannot write an embedded artifact."); cursor += count;
    }
    require(::rename(temporary.c_str(), path.c_str()) == 0, "Cannot publish an embedded artifact.");
}
LaunchFailure failure(LaunchError code, const char* operation, std::string detail) {
    return {.stage = LaunchStage::preparation, .code = code, .operation = operation, .detail = std::move(detail)};
}
} // namespace

struct EmbeddedNativeFiles::Storage {
    EmbeddedNativeArtifacts artifacts;
    std::vector<Mapping> mappings;
    std::vector<std::vector<platform::EmbeddedModulePiece>> pieces;
    std::vector<EmbeddedNativeBacking> backings;
};
const EmbeddedNativeArtifacts& EmbeddedNativeFiles::artifacts() const noexcept { return storage_->artifacts; }
std::span<const EmbeddedNativeBacking> EmbeddedNativeFiles::backings() const noexcept { return storage_->backings; }

LaunchStatus write_embedded_modules(const std::filesystem::path& directory, const EmbeddedNativeArtifacts& artifacts) {
    try {
        std::vector<Digest> hashes;
        const auto bytes = encode(artifacts, hashes);
        std::filesystem::create_directories(directory);
        for (std::size_t i = 0; i < artifacts.modules.size(); ++i)
            atomic_write(directory / image_name(hashes[i]), artifacts.modules[i].image);
        // Content-addressed images make publishing the manifest last atomic:
        // an older manifest continues to name its complete unchanged images.
        atomic_write(directory / manifest_name, bytes.data);
        return std::monostate{};
    } catch (const std::bad_alloc&) {
        return failure(LaunchError::resource_limit, "write-embedded", "Embedded artifact allocation failed.");
    } catch (const std::exception& error) {
        return failure(LaunchError::host_failure, "write-embedded", error.what());
    }
}

LaunchResult<EmbeddedNativeFiles> open_linked_embedded_modules(const LinkedEmbeddedBundle& bundle) {
    try {
        const auto* header = reinterpret_cast<const mach_header_64*>(_dyld_get_image_header(0));
        require(header && header->magic == MH_MAGIC_64 && header->filetype == MH_EXECUTE,
                "Linked code requires the main Mach-O executable.");
        const auto slide = _dyld_get_image_vmaddr_slide(0);
        std::uintptr_t code_begin = 0, code_end = 0;
        unsigned executable_segments = 0;
        const auto* cursor = reinterpret_cast<const std::byte*>(header + 1);
        const auto* end = cursor + header->sizeofcmds;
        for (std::uint32_t i = 0; i < header->ncmds; ++i) {
            require(static_cast<std::size_t>(end - cursor) >= sizeof(load_command), "Truncated Mach-O command.");
            const auto* command = reinterpret_cast<const load_command*>(cursor);
            require(command->cmdsize >= sizeof(load_command) && command->cmdsize <= end - cursor,
                    "Invalid Mach-O command size.");
            if (command->cmd == LC_SEGMENT_64) {
                require(command->cmdsize >= sizeof(segment_command_64), "Truncated Mach-O segment.");
                const auto* segment = reinterpret_cast<const segment_command_64*>(cursor);
                if (segment->initprot & VM_PROT_EXECUTE) {
                    ++executable_segments;
                    require(!(segment->initprot & VM_PROT_WRITE) && !(segment->maxprot & VM_PROT_WRITE) &&
                            std::string_view(segment->segname, strnlen(segment->segname, sizeof segment->segname)) == "__TEXT",
                            "Linked code is not in an immutable __TEXT segment.");
                    code_begin = segment->vmaddr + slide;
                    require(segment->vmsize <= UINTPTR_MAX - code_begin, "Mach-O code range overflows.");
                    code_end = code_begin + segment->vmsize;
                }
            }
            cursor += command->cmdsize;
        }
        require(cursor == end && executable_segments == 1, "Linked player must have exactly one executable segment.");
        const auto manifest_address = reinterpret_cast<std::uintptr_t>(bundle.manifest.data());
        require(manifest_address >= code_begin && manifest_address < code_end &&
                bundle.manifest.size() <= code_end - manifest_address,
                "Embedded manifest is outside signed Mach-O code.");
        auto decoded = decode(bundle.manifest);
        require(bundle.backings.size() == decoded.artifacts.modules.size(),
                "Linked embedded module count does not match its manifest.");
        auto storage = std::make_shared<EmbeddedNativeFiles::Storage>();
        storage->artifacts = std::move(decoded.artifacts);
        const auto count = bundle.backings.size();
        storage->pieces.resize(count); storage->backings.resize(count);
        for (std::size_t i = 0; i < count; ++i) {
            auto& module = storage->artifacts.modules[i];
            module.image.resize(decoded.sizes[i]);
            const auto text_begin = module.text_offset & ~(page - 1);
            const auto text_end = (module.text_offset + module.text_size + page - 1) & ~(page - 1);
            require(text_end <= module.key_page_offset, "Embedded text overlaps its key page.");
            std::size_t offset = 0;
            for (const auto& piece : bundle.backings[i].pieces) {
                const auto address = reinterpret_cast<std::uintptr_t>(piece.bytes.data());
                const auto size = piece.bytes.size();
                require(piece.offset == offset && size > 0 && size <= module.image.size() - offset &&
                        offset % page == 0 && size % page == 0 && address % page == 0,
                        "Linked embedded pieces are not a complete aligned image.");
                const auto stop = offset + size;
                const bool executable = module.host_executes &&
                    ((offset >= text_begin && stop <= text_end) || offset >= module.appended_code_offset);
                const bool overlaps_code = module.host_executes &&
                    ((offset < text_end && stop > text_begin) || stop > module.appended_code_offset);
                require(piece.executable == executable && (executable || !overlaps_code),
                        "Linked embedded piece crosses a code boundary or has wrong permissions.");
                if (executable) require(address >= code_begin && address < code_end && size <= code_end - address,
                                        "Embedded executable piece is outside signed Mach-O code.");
                auto retained = piece;
                if (!executable) {
                    Mapping snapshot;
                    snapshot.size = size;
                    snapshot.address = ::mmap(nullptr, size, PROT_READ | PROT_WRITE,
                                               MAP_PRIVATE | MAP_ANON, -1, 0);
                    require(snapshot.address != MAP_FAILED, "Cannot snapshot embedded data pages.");
                    std::memcpy(snapshot.address, piece.bytes.data(), size);
                    require(::vm_protect(mach_task_self(),
                                reinterpret_cast<vm_address_t>(snapshot.address), size, true,
                                VM_PROT_READ | VM_PROT_WRITE) == KERN_SUCCESS &&
                            ::mprotect(snapshot.address, size, PROT_READ) == 0,
                            "Cannot protect the embedded data snapshot.");
                    // Validate and later remap this same private snapshot. The
                    // writable linked source may change after this copy.
                    retained.bytes = {static_cast<const std::byte*>(snapshot.address), size};
                    storage->mappings.push_back(std::move(snapshot));
                }
                std::copy(retained.bytes.begin(), retained.bytes.end(), module.image.begin() + offset);
                storage->pieces[i].push_back(retained);
                offset = stop;
            }
            require(offset == module.image.size(), "Linked embedded pieces leave a gap.");
            require(hash(module.image) == decoded.hashes[i], "Linked embedded image checksum mismatch.");
            require(std::all_of(module.image.begin() + module.key_page_offset,
                               module.image.begin() + module.appended_code_offset,
                               [](std::byte value) { return value == std::byte{0}; }),
                    "Linked embedded key page is not empty.");
            storage->backings[i].pieces = storage->pieces[i];
        }
        return EmbeddedNativeFiles(std::move(storage));
    } catch (const std::bad_alloc&) {
        return failure(LaunchError::resource_limit, "open-linked-embedded", "Linked embedded allocation failed.");
    } catch (const std::exception& error) {
        return failure(LaunchError::artifact_mismatch, "open-linked-embedded", error.what());
    }
}

LaunchResult<EmbeddedNativeFiles> open_embedded_modules(const std::filesystem::path& directory) {
#if TARGET_OS_IPHONE
    (void)directory;
    if (!armigo_linked_embedded_bundle)
        return failure(LaunchError::unsupported, "open-embedded", "This iOS player has no linked guest code.");
    const auto* bundle = armigo_linked_embedded_bundle();
    if (!bundle) return failure(LaunchError::artifact_mismatch, "open-embedded", "The linked guest-code provider is empty.");
    return open_linked_embedded_modules(*bundle);
#else
    try {
        Descriptor manifest(::open((directory / manifest_name).c_str(), O_RDONLY | O_CLOEXEC | O_NOFOLLOW));
        struct stat info{};
        require(manifest.value >= 0 && ::fstat(manifest.value, &info) == 0 && S_ISREG(info.st_mode) &&
                info.st_size > 0 && static_cast<std::uint64_t>(info.st_size) <= metadata_limit,
                "Cannot open bounded embedded metadata.");
        std::vector<std::byte> bytes(info.st_size);
        std::size_t cursor = 0;
        while (cursor < bytes.size()) {
            const auto count = ::read(manifest.value, bytes.data() + cursor, bytes.size() - cursor);
            if (count < 0 && errno == EINTR) continue;
            require(count > 0, "Cannot read embedded metadata."); cursor += count;
        }
        auto decoded = decode(bytes);
        auto storage = std::make_shared<EmbeddedNativeFiles::Storage>();
        storage->artifacts = std::move(decoded.artifacts);
        const auto count = storage->artifacts.modules.size();
        storage->pieces.resize(count); storage->backings.resize(count); storage->mappings.reserve(count);
        for (std::size_t i = 0; i < count; ++i) {
            Descriptor file(::open((directory / image_name(decoded.hashes[i])).c_str(), O_RDONLY | O_CLOEXEC | O_NOFOLLOW));
            require(file.value >= 0 && ::fstat(file.value, &info) == 0 && S_ISREG(info.st_mode) && info.st_size > 0 &&
                    static_cast<std::uint64_t>(info.st_size) == decoded.sizes[i], "Embedded image file has the wrong size.");
            Mapping mapping;
            mapping.size = decoded.sizes[i];
            mapping.address = ::mmap(nullptr, mapping.size, PROT_READ | PROT_WRITE, MAP_PRIVATE, file.value, 0);
            require(mapping.address != MAP_FAILED, "Cannot map an embedded image file.");
            const std::span<const std::byte> image{static_cast<const std::byte*>(mapping.address), mapping.size};
            require(hash(image) == decoded.hashes[i], "Embedded image checksum mismatch.");
            auto& module = storage->artifacts.modules[i];
            module.image.assign(image.begin(), image.end());
            require(std::all_of(image.begin() + module.key_page_offset, image.begin() + module.appended_code_offset,
                               [](std::byte byte) { return byte == std::byte{0}; }), "Embedded key page is not empty.");
            const auto text_begin = module.text_offset & ~(page - 1);
            const auto text_end = (module.text_offset + module.text_size + page - 1) & ~(page - 1);
            require(text_end <= module.key_page_offset, "Embedded text overlaps its key page.");
            std::vector<std::size_t> boundaries{0, text_begin, text_end, module.appended_code_offset, image.size()};
            std::sort(boundaries.begin(), boundaries.end());
            boundaries.erase(std::unique(boundaries.begin(), boundaries.end()), boundaries.end());
            for (std::size_t j = 1; j < boundaries.size(); ++j) {
                const auto begin = boundaries[j - 1], end = boundaries[j];
                const bool executable = module.host_executes && ((begin >= text_begin && end <= text_end) || begin >= module.appended_code_offset);
                if (executable) require(::mprotect(static_cast<std::byte*>(mapping.address) + begin, end - begin,
                                                   PROT_READ | PROT_EXEC) == 0, "Cannot map executable embedded file pages.");
                storage->pieces[i].push_back({begin, image.subspan(begin, end - begin), executable});
            }
            storage->backings[i].pieces = storage->pieces[i];
            storage->mappings.push_back(std::move(mapping));
        }
        return EmbeddedNativeFiles(std::move(storage));
    } catch (const std::bad_alloc&) {
        return failure(LaunchError::resource_limit, "open-embedded", "Embedded artifact allocation failed.");
    } catch (const std::exception& error) {
        return failure(LaunchError::artifact_mismatch, "open-embedded", error.what());
    }
#endif
}
} // namespace armigo::runtime
