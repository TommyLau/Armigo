// ============================================================================
// Armigo Platform - Apple Platform Integration
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================
#include <armigo/common/diagnostics.h>
// Matched-code reuse requires exact equality of the relevant guest code bytes.

#include <armigo/platform/metal_shader_package.h>

#include <armigo/graphics/vertex_input.h>
#include <armigo/graphics/render_target.h>
#include <armigo/shader/program_identity.h>
#include <armigo/shader/source_version.h>
#include <armigo/shader/vertex_fetch.h>
#include <armigo/memory/write_tracker.h>
#include "sha256.h"
#include "texture_fact.h"

#import <Foundation/Foundation.h>
#include <TargetConditionals.h>

#include <algorithm>
#include <mach/mach.h>
#include <pthread.h>
#include "../mach_vm.h"
#include <array>
#include <atomic>
#include <charconv>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <map>
#include <mutex>
#include <new>
#include <optional>
#include <set>
#include <span>
#include <exception>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <thread>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

namespace armigo::platform {
namespace {

namespace fs = std::filesystem;
namespace pi = shader::program_identity;
namespace gm = shader::graphics_msl;
using Stage = shader_input::Stage;
using Digest = pi::Digest;

constexpr std::size_t manifest_limit = 64u << 20;
constexpr std::size_t reflection_limit = 4u << 20;
constexpr std::size_t library_limit = 64u << 20;
constexpr std::size_t total_program_limit = 512u << 20;
constexpr std::size_t total_reflection_limit = 512u << 20;
constexpr std::size_t library_cache_limit = 512u << 20;
constexpr std::size_t entry_limit = 1000000;
constexpr std::size_t json_depth_limit = 64;
#if TARGET_OS_IPHONE
constexpr std::string_view platform_name = "ios-metal4";
#else
constexpr std::string_view platform_name = "macos-metal4";
#endif

struct Failure { MetalShaderPackageError error; };
[[noreturn]] void fail(MetalShaderPackageError::Code code, std::string detail) {
    throw Failure{{code, std::move(detail)}};
}

bool lower_hex(std::string_view value) {
    return value.size() == 64 && std::all_of(value.begin(), value.end(), [](char c) {
        return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f');
    });
}

std::string hex(const Digest& digest) {
    constexpr char digits[] = "0123456789abcdef";
    std::string result;
    result.reserve(64);
    for (auto byte : digest) {
        const auto value = std::to_integer<unsigned>(byte);
        result += digits[value >> 4];
        result += digits[value & 15];
    }
    return result;
}

Digest parse_digest(std::string_view value, std::string_view field) {
    if (!lower_hex(value)) fail(MetalShaderPackageError::Code::Manifest,
                                std::string(field) + " must be 64 lowercase hexadecimal characters");
    Digest result{};
    for (std::size_t index = 0; index < result.size(); ++index) {
        unsigned number = 0;
        const auto pair = value.substr(index * 2, 2);
        const auto parsed = std::from_chars(pair.data(), pair.data() + pair.size(), number, 16);
        if (parsed.ec != std::errc{}) fail(MetalShaderPackageError::Code::Manifest,
                                          std::string(field) + " is invalid");
        result[index] = std::byte(number);
    }
    return result;
}

std::optional<Digest> hash_bytes(void*, std::span<const std::byte> bytes) noexcept {
    return armigo::loader::detail::sha256(bytes);
}

Digest digest(std::span<const std::byte> bytes) {
    const auto result = hash_bytes(nullptr, bytes);
    if (!result) fail(MetalShaderPackageError::Code::Integrity, "System SHA-256 failed");
    return *result;
}

std::vector<std::byte> read_file(const fs::path& path, std::size_t limit,
                                 std::string_view label) {
    if (!fs::is_regular_file(path)) fail(MetalShaderPackageError::Code::Input,
                                         std::string(label) + " is not a regular file");
    const auto size = fs::file_size(path);
    if (size > limit) fail(MetalShaderPackageError::Code::Input,
                           std::string(label) + " exceeds its size policy");
    std::vector<std::byte> result(static_cast<std::size_t>(size));
    std::ifstream stream(path, std::ios::binary);
    if (!stream || (!result.empty() &&
        !stream.read(reinterpret_cast<char*>(result.data()), static_cast<std::streamsize>(result.size()))) ||
        stream.peek() != std::char_traits<char>::eof())
        fail(MetalShaderPackageError::Code::Input, "Cannot read stable " + std::string(label));
    return result;
}

NSString* ns(std::string_view value) {
    return [[NSString alloc] initWithBytes:value.data() length:value.size()
                                  encoding:NSUTF8StringEncoding];
}

std::string string(id value, std::string_view field) {
    if (![value isKindOfClass:[NSString class]])
        fail(MetalShaderPackageError::Code::Manifest, std::string(field) + " must be a string");
    NSData* data = [(NSString*)value dataUsingEncoding:NSUTF8StringEncoding];
    if (!data) fail(MetalShaderPackageError::Code::Manifest,
                    std::string(field) + " is not valid UTF-8");
    return {static_cast<const char*>(data.bytes), data.length};
}

bool boolean(id value, std::string_view field) {
    if (![value isKindOfClass:[NSNumber class]] ||
        CFGetTypeID((__bridge CFTypeRef)value) != CFBooleanGetTypeID())
        fail(MetalShaderPackageError::Code::Manifest, std::string(field) + " must be a Boolean");
    return [(NSNumber*)value boolValue];
}

std::uint32_t integer(id value, std::string_view field, std::uint32_t maximum) {
    if (![value isKindOfClass:[NSNumber class]] ||
        CFGetTypeID((__bridge CFTypeRef)value) == CFBooleanGetTypeID() ||
        CFNumberIsFloatType((__bridge CFNumberRef)value))
        fail(MetalShaderPackageError::Code::Manifest, std::string(field) + " must be an integer");
    const long long number = [(NSNumber*)value longLongValue];
    if (number < 0 || static_cast<unsigned long long>(number) > maximum)
        fail(MetalShaderPackageError::Code::Manifest, std::string(field) + " is outside its range");
    return static_cast<std::uint32_t>(number);
}

NSDictionary* object(id value, std::string_view field) {
    if (![value isKindOfClass:[NSDictionary class]])
        fail(MetalShaderPackageError::Code::Manifest, std::string(field) + " must be an object");
    return (NSDictionary*)value;
}

NSArray* array(id value, std::string_view field, std::size_t maximum) {
    if (![value isKindOfClass:[NSArray class]])
        fail(MetalShaderPackageError::Code::Manifest, std::string(field) + " must be an array");
    NSArray* result = (NSArray*)value;
    if (result.count > maximum) fail(MetalShaderPackageError::Code::Manifest,
                                     std::string(field) + " exceeds its count policy");
    return result;
}

void exact(NSDictionary* value, std::initializer_list<const char*> fields,
           std::string_view label) {
    NSMutableSet* expected = [NSMutableSet setWithCapacity:fields.size()];
    for (const char* field : fields) [expected addObject:[NSString stringWithUTF8String:field]];
    NSSet* actual = [NSSet setWithArray:value.allKeys];
    if (![actual isEqualToSet:expected])
        fail(MetalShaderPackageError::Code::Manifest,
             std::string(label) + " has missing or unknown fields");
}

class JsonStructureScanner {
public:
    explicit JsonStructureScanner(std::span<const std::byte> bytes)
        : text_(reinterpret_cast<const char*>(bytes.data()), bytes.size()) {}

    void run(std::string_view label) {
        label_ = label;
        space();
        value(0);
        space();
        if (position_ != text_.size()) malformed();
    }

private:
    [[noreturn]] void malformed() const {
        fail(MetalShaderPackageError::Code::Manifest,
             std::string(label_) + " has malformed or excessively nested JSON structure");
    }
    void space() {
        while (position_ < text_.size() &&
               (text_[position_] == ' ' || text_[position_] == '\t' ||
                text_[position_] == '\r' || text_[position_] == '\n')) ++position_;
    }
    std::string_view string_token() {
        if (position_ >= text_.size() || text_[position_] != '"') malformed();
        const std::size_t begin = position_++;
        while (position_ < text_.size()) {
            const unsigned char c = text_[position_++];
            if (c == '"') return text_.substr(begin, position_ - begin);
            if (c < 0x20) malformed();
            if (c == '\\') {
                if (position_ >= text_.size()) malformed();
                if (text_[position_] == 'u') {
                    ++position_;
                    if (position_ + 4 > text_.size()) malformed();
                    position_ += 4;
                } else {
                    ++position_;
                }
            }
        }
        malformed();
    }
    std::string decoded_key(std::string_view token) {
        // A key of printable ASCII without escapes reads as its own text (the
        // decoded string is the same); any other key is decoded by Foundation.
        const auto inner = token.substr(1, token.size() - 2);
        if (std::all_of(inner.begin(), inner.end(),
                        [](char c) { return c >= 0x20 && c <= 0x7e && c != '\\' && c != '"'; }))
            return std::string(inner);
        std::string wrapper = "[" + std::string(token) + "]";
        NSData* data = [NSData dataWithBytes:wrapper.data() length:wrapper.size()];
        NSError* error = nil;
        id parsed = [NSJSONSerialization JSONObjectWithData:data options:0 error:&error];
        if (error || ![parsed isKindOfClass:[NSArray class]] || [(NSArray*)parsed count] != 1 ||
            ![[(NSArray*)parsed objectAtIndex:0] isKindOfClass:[NSString class]]) malformed();
        return string([(NSArray*)parsed objectAtIndex:0], "JSON object key");
    }
    void object_value(std::size_t depth) {
        ++position_;
        space();
        std::set<std::string> keys;
        if (position_ < text_.size() && text_[position_] == '}') { ++position_; return; }
        for (;;) {
            const auto key = decoded_key(string_token());
            if (!keys.insert(key).second)
                fail(MetalShaderPackageError::Code::Manifest,
                     std::string(label_) + " contains a duplicate JSON object key");
            space();
            if (position_ >= text_.size() || text_[position_++] != ':') malformed();
            space();
            value(depth + 1);
            space();
            if (position_ >= text_.size()) malformed();
            const char delimiter = text_[position_++];
            if (delimiter == '}') return;
            if (delimiter != ',') malformed();
            space();
        }
    }
    void array_value(std::size_t depth) {
        ++position_;
        space();
        if (position_ < text_.size() && text_[position_] == ']') { ++position_; return; }
        for (;;) {
            value(depth + 1);
            space();
            if (position_ >= text_.size()) malformed();
            const char delimiter = text_[position_++];
            if (delimiter == ']') return;
            if (delimiter != ',') malformed();
            space();
        }
    }
    void value(std::size_t depth) {
        if (depth > json_depth_limit || position_ >= text_.size()) malformed();
        if (text_[position_] == '{') return object_value(depth);
        if (text_[position_] == '[') return array_value(depth);
        if (text_[position_] == '"') { static_cast<void>(string_token()); return; }
        const std::size_t begin = position_;
        while (position_ < text_.size() && text_[position_] != ',' &&
               text_[position_] != '}' && text_[position_] != ']' &&
               text_[position_] != ' ' && text_[position_] != '\t' &&
               text_[position_] != '\r' && text_[position_] != '\n') ++position_;
        if (position_ == begin) malformed();
    }

    std::string_view text_;
    std::string_view label_;
    std::size_t position_ = 0;
};

id json(std::span<const std::byte> bytes, std::string_view label) {
    JsonStructureScanner{bytes}.run(label);
    NSData* data = [NSData dataWithBytes:bytes.data() length:bytes.size()];
    NSError* error = nil;
    id result = [NSJSONSerialization JSONObjectWithData:data options:0 error:&error];
    if (!result || error) fail(MetalShaderPackageError::Code::Manifest,
                               std::string(label) + " is not valid JSON");
    return result;
}

bool inside(const fs::path& child, const fs::path& parent) {
    const auto compared = std::mismatch(parent.begin(), parent.end(), child.begin(), child.end());
    return compared.first == parent.end();
}

fs::path package_file(const fs::path& root, const fs::path& stage,
                      std::string_view filename) {
    const auto result = fs::canonical(stage / filename);
    if (!inside(result, root)) fail(MetalShaderPackageError::Code::Input,
                                    "Package file resolves outside its root");
    return result;
}

// The package's identity index: the (stage, length, prefix and full digest)
// of every program, sorted by stage and prefix digest. A lazy package builds
// it from the digests its manifest records, without reading the programs;
// each program is checked against its record when its stage is prepared. The
// lookup is the program-identity lookup over this list (the shader library
// builds its index only from program bytes, and a change to it would change
// the shader source version that packages are bound to).
struct IdentityIndex {
    std::vector<pi::Identity> identities;
};

bool identity_less(const pi::Identity& lhs, const pi::Identity& rhs) noexcept {
    if (lhs.stage != rhs.stage) return lhs.stage < rhs.stage;
    if (lhs.prefix_digest != rhs.prefix_digest) return lhs.prefix_digest < rhs.prefix_digest;
    if (lhs.length != rhs.length) return lhs.length < rhs.length;
    return lhs.full_digest < rhs.full_digest;
}

bool program_header_matches(Stage stage, std::span<const std::byte> bytes) noexcept {
    // Compute programs carry no SPH header and are matched by digest alone.
    if (stage == Stage::Compute) return true;
    std::uint32_t word = 0;
    for (unsigned i = 0; i != 4; ++i) word |= std::uint32_t{std::to_integer<unsigned char>(bytes[i])} << (8 * i);
    return ((word >> shader_input::format::sph_stage_shift) & shader_input::format::sph_stage_mask) ==
           shader_input::sph_stage_type_for(stage);
}

// The size and header checks the identity build applies to every program.
std::optional<pi::Error> check_program(Stage stage, std::span<const std::byte> program) noexcept {
    if (program.size() < pi::prefix_size)
        return pi::Error{pi::Error::Code::ProgramTooShort, 0, program.size()};
    if (program.size() > pi::maximum_program_size)
        return pi::Error{pi::Error::Code::ProgramTooLarge, 0, program.size()};
    if (!program_header_matches(stage, program)) return pi::Error{pi::Error::Code::HeaderStageMismatch};
    return std::nullopt;
}

// Sorts and collapses the records (entries that share a program share an
// identity) and applies the prefix bucket policy.
std::variant<IdentityIndex, pi::Error> make_identity_index(std::vector<pi::Identity> records) {
    if (records.size() > pi::maximum_identities)
        return pi::Error{pi::Error::Code::IdentityLimit, records.size(), pi::maximum_identities};
    for (std::size_t i = 0; i != records.size(); ++i) {
        const auto& record = records[i];
        if (record.length < pi::prefix_size) return pi::Error{pi::Error::Code::ProgramTooShort, i, record.length};
        if (record.length > pi::maximum_program_size)
            return pi::Error{pi::Error::Code::ProgramTooLarge, i, record.length};
    }
    std::sort(records.begin(), records.end(), identity_less);
    records.erase(std::unique(records.begin(), records.end(),
                              [](const pi::Identity& lhs, const pi::Identity& rhs) {
                                  return !identity_less(lhs, rhs) && !identity_less(rhs, lhs);
                              }),
                  records.end());
    for (std::size_t begin = 0; begin < records.size();) {
        std::size_t end = begin + 1;
        while (end < records.size() && records[end].stage == records[begin].stage &&
               records[end].prefix_digest == records[begin].prefix_digest)
            ++end;
        if (end - begin > pi::maximum_bucket_size)
            return pi::Error{pi::Error::Code::BucketLimit, begin, end - begin};
        begin = end;
    }
    return IdentityIndex{std::move(records)};
}

pi::LookupResult lookup_identity(const IdentityIndex& index, Stage stage, std::uint64_t address,
                                 const pi::MemoryReader& reader, const pi::HashProvider& hash) {
    using Code = pi::Error::Code;
    if (stage != Stage::Vertex && stage != Stage::Fragment && stage != Stage::Compute)
        return pi::Error{Code::UnsupportedStage};
    if (reader.function == nullptr) return pi::Error{Code::ReadFailure};
    if (hash.function == nullptr) return pi::Error{Code::HashFailure};
    if (address > std::numeric_limits<std::uint64_t>::max() - pi::prefix_size)
        return pi::Error{Code::AddressOverflow, 0, pi::prefix_size};
    std::array<std::byte, pi::prefix_size> prefix{};
    if (!reader.function(reader.context, address, prefix)) return pi::Error{Code::ReadFailure, 0, pi::prefix_size};
    if (!program_header_matches(stage, prefix)) return pi::Error{Code::HeaderStageMismatch};
    const auto prefix_digest = hash.function(hash.context, prefix);
    if (!prefix_digest) return pi::Error{Code::HashFailure};
    const auto begin = std::lower_bound(index.identities.begin(), index.identities.end(),
        std::pair{stage, *prefix_digest}, [](const pi::Identity& lhs, const std::pair<Stage, Digest>& rhs) {
            if (lhs.stage != rhs.first) return lhs.stage < rhs.first;
            return lhs.prefix_digest < rhs.second;
        });
    auto end = begin;
    while (end != index.identities.end() && end->stage == stage && end->prefix_digest == *prefix_digest) ++end;
    if (begin == end) return pi::Error{Code::Miss};
    if (static_cast<std::size_t>(end - begin) > pi::maximum_bucket_size) return pi::Error{Code::BucketLimit};
    // The candidates' lengths bound the bytes one lookup may read.
    std::size_t work = pi::prefix_size;
    std::size_t previous_length = 0;
    for (auto it = begin; it != end; ++it) {
        if (it != begin && it->length == previous_length) continue;
        if (it->length > pi::maximum_lookup_work_bytes - work) return pi::Error{Code::LookupWorkLimit, 0, work};
        work += it->length;
        previous_length = it->length;
    }
    std::optional<pi::Identity> match;
    for (auto group = begin; group != end;) {
        const std::size_t length = group->length;
        auto group_end = group;
        while (group_end != end && group_end->length == length) ++group_end;
        Digest full{};
        if (length == pi::prefix_size) {
            full = *prefix_digest;
        } else {
            if (address > std::numeric_limits<std::uint64_t>::max() - length)
                return pi::Error{Code::AddressOverflow, 0, length};
            std::vector<std::byte> bytes(length);
            if (!reader.function(reader.context, address, bytes)) return pi::Error{Code::ReadFailure, 0, length};
            const auto hashed = hash.function(hash.context, bytes);
            if (!hashed) return pi::Error{Code::HashFailure, 0, length};
            full = *hashed;
        }
        for (auto candidate = group; candidate != group_end; ++candidate) {
            if (candidate->full_digest != full) continue;
            if (match) return pi::Error{Code::Ambiguous};
            match = *candidate;
        }
        group = group_end;
    }
    if (!match) return pi::Error{Code::Miss};
    return *match;
}

struct Entry {
    Stage stage = Stage::Vertex;
    Digest full{};
    Digest prefix{};
    std::uint32_t program_size = 0;
    std::string variant;
    fs::path stage_directory;
    Digest reflection_digest{};
    fs::path library_path;
    Digest library_digest{};
    std::uint32_t library_size = 0;
    CompiledGraphicsStage metadata;
    std::shared_ptr<const CompiledGraphicsStage> cached;
    std::map<std::uint16_t, std::shared_ptr<const CompiledGraphicsStage>> output_specializations;
    // Set when the reflection is well formed for the package but outside the
    // runtime profile and the caller deferred such entries to lookup time.
    std::string unsupported;
};

struct ReaderContext { const graphics::GpuAddressSpace* memory; };
bool read_guest(void* raw, std::uint64_t address, std::span<std::byte> output) noexcept {
    auto& memory = *static_cast<ReaderContext*>(raw)->memory;
    if (output.size() > std::numeric_limits<std::uint64_t>::max() - address) return false;
    // Address-space mappings are sorted by GPU base and never overlap.
    const auto mappings = memory.mappings();
    while (!output.empty()) {
        std::size_t amount = 0;
        const auto next = std::upper_bound(mappings.begin(), mappings.end(), address,
            [](std::uint64_t value, const auto& mapping) { return value < mapping.gpu_base; });
        if (next != mappings.begin()) {
            const auto& mapping = *std::prev(next);
            if (address - mapping.gpu_base < mapping.mapped_size)
                amount = static_cast<std::size_t>(std::min<std::uint64_t>(
                    output.size(), mapping.mapped_size - (address - mapping.gpu_base)));
        }
        if (amount == 0 || !memory.read(address, output.first(amount))) return false;
        address += amount;
        output = output.subspan(amount);
    }
    return true;
}

Stage parse_stage(std::string_view value) {
    if (value == "vertex") return Stage::Vertex;
    if (value == "fragment") return Stage::Fragment;
    if (value == "compute") return Stage::Compute;
    fail(MetalShaderPackageError::Code::Manifest, "Entry stage is unsupported");
}

const char* stage_name(Stage stage) {
    return stage == Stage::Vertex ? "vertex" : stage == Stage::Compute ? "compute" : "fragment";
}

shader::graphics::Interpolation parse_interpolation(std::string_view value) {
    if (value == "constant") return shader::graphics::Interpolation::Constant;
    if (value == "perspective") return shader::graphics::Interpolation::Perspective;
    if (value == "screen-linear") return shader::graphics::Interpolation::ScreenLinear;
    fail(MetalShaderPackageError::Code::Reflection, "Reflection interpolation is unsupported");
}

[[noreturn]] void reflection_error(std::string detail) {
    fail(MetalShaderPackageError::Code::Reflection, std::move(detail));
}

// The admitted v2 runtime subset: constant-buffer bindings, float32x3 vertex
// streams or class-typed vertex inputs, flat/perspective/linear generic IO on
// both stages, and validated typed colour-output records whose union must
// rebuild the interface summary mask. Which colour-target sets the renderer
// can actually bind is the draw backend's fail-closed admission decision.
// Textures and samplers bind through the runtime's guest descriptor resolution
// on either stage: each pair shares one handle word at a single proven
// constant-buffer slot, and the consumed texture_handle_cbuf fact must still
// hold at draw time.
constexpr std::string_view admitted_v2_profile_identity = "armigo-macos-metal4";
constexpr std::uint32_t admitted_v2_profile_version = 1;
// Profile version 3 adds the armigo_y_negate function constant (index 9) for
// stages that read S2R SR_YDIRECTION, the point-size export of vertex stages
// ("fixed-one-v1" or "program-v1"), and the point-coordinate origin constant
// (index 11) of fragment stages that read the point coordinate.
constexpr std::uint32_t admitted_v3_profile_version = 3;

bool admitted_v2_dependency(std::string_view kind) {
    return kind == "depth_convention" || kind == "vertex_attribute_format" ||
           kind == "input_interpolation" || kind == "fragment_output_type" ||
           kind == "texture_handle_cbuf" || kind == "texture_descriptor" ||
           kind == "local_memory_bytes";
}

shader::graphics::Interpolation parse_interpolation_v2(std::string_view value) {
    if (value == "flat") return shader::graphics::Interpolation::Constant;
    if (value == "perspective") return shader::graphics::Interpolation::Perspective;
    if (value == "linear") return shader::graphics::Interpolation::ScreenLinear;
    fail(MetalShaderPackageError::Code::Reflection,
         "Reflection interpolation is unsupported");
}

TextureFactDimension parse_texture_dimension(std::string_view value) {
    if (value == "d1") return TextureFactDimension::d1;
    if (value == "d2") return TextureFactDimension::d2;
    if (value == "d3") return TextureFactDimension::d3;
    if (value == "cube") return TextureFactDimension::cube;
    reflection_error("Texture dimension is outside the supported profile");
}

TextureSampledScalar parse_texture_scalar(std::string_view value) {
    if (value == "f32") return TextureSampledScalar::f32;
    if (value == "u32") return TextureSampledScalar::u32;
    if (value == "s32") return TextureSampledScalar::s32;
    reflection_error("Texture scalar is outside the supported profile");
}

TextureComponentFormat parse_texture_component_format(std::string_view value) {
    if (value == "floating") return TextureComponentFormat::floating;
    if (value == "unorm") return TextureComponentFormat::unorm;
    if (value == "snorm") return TextureComponentFormat::snorm;
    if (value == "unsigned_integer") return TextureComponentFormat::unsigned_integer;
    if (value == "signed_integer") return TextureComponentFormat::signed_integer;
    if (value == "depth") return TextureComponentFormat::depth;
    reflection_error("Texture component format is outside the supported profile");
}

bool texture_fact_equal(const DerivedTextureFact& left, const DerivedTextureFact& right) {
    return left.cbuf_slot == right.cbuf_slot &&
           left.handle_byte_offset == right.handle_byte_offset &&
           left.normalized_coordinates == right.normalized_coordinates &&
           left.comparison == right.comparison && left.dimension == right.dimension &&
           left.array == right.array && left.multisample == right.multisample &&
           left.scalar == right.scalar && left.component_format == right.component_format;
}

DerivedTextureFact parse_texture_descriptor(NSDictionary* value, std::uint8_t slot,
                                            std::uint32_t offset) {
    DerivedTextureFact fact;
    fact.cbuf_slot = slot;
    fact.handle_byte_offset = offset;
    fact.normalized_coordinates = boolean(value[@"normalized_coordinates"],
                                          "normalized_coordinates");
    fact.comparison = boolean(value[@"comparison"], "comparison");
    fact.dimension = parse_texture_dimension(string(value[@"dimension"], "dimension"));
    fact.array = boolean(value[@"array"], "array");
    fact.multisample = boolean(value[@"multisample"], "multisample");
    fact.scalar = parse_texture_scalar(string(value[@"scalar"], "scalar"));
    fact.component_format = parse_texture_component_format(
        string(value[@"component_format"], "component_format"));

    const bool valid_shape = !fact.multisample &&
        !(fact.array && fact.dimension == TextureFactDimension::d3);
    const bool valid_format =
        (fact.scalar == TextureSampledScalar::f32 &&
         (fact.component_format == TextureComponentFormat::floating ||
          fact.component_format == TextureComponentFormat::unorm ||
          fact.component_format == TextureComponentFormat::snorm ||
          fact.component_format == TextureComponentFormat::depth)) ||
        (fact.scalar == TextureSampledScalar::u32 &&
         fact.component_format == TextureComponentFormat::unsigned_integer) ||
        (fact.scalar == TextureSampledScalar::s32 &&
         fact.component_format == TextureComponentFormat::signed_integer);
    const bool valid_depth = fact.component_format != TextureComponentFormat::depth ||
        fact.dimension == TextureFactDimension::d2 ||
        fact.dimension == TextureFactDimension::cube;
    if (!valid_shape || !valid_format || !valid_depth ||
        (fact.comparison && (fact.scalar != TextureSampledScalar::f32 ||
                             fact.component_format != TextureComponentFormat::depth)))
        reflection_error("Texture descriptor is outside the supported profile");
    return fact;
}

CompiledShaderTextureDescriptor parse_texture_descriptor_v3(NSDictionary* value) {
    CompiledShaderTextureDescriptor result;
    const auto dimension = string(value[@"dimension"], "dimension");
    if (dimension == "d1") result.dimension = CompiledShaderTextureDimension::d1;
    else if (dimension == "d2") result.dimension = CompiledShaderTextureDimension::d2;
    else if (dimension == "d3") result.dimension = CompiledShaderTextureDimension::d3;
    else if (dimension == "cube") result.dimension = CompiledShaderTextureDimension::cube;
    else if (dimension == "buffer") result.dimension = CompiledShaderTextureDimension::buffer;
    else reflection_error("Texture dimension is invalid");
    result.array = boolean(value[@"array"], "array");
    result.multisample = boolean(value[@"multisample"], "multisample");
    const auto scalar = string(value[@"scalar"], "scalar");
    if (scalar == "f16") result.scalar = CompiledShaderTextureScalar::f16;
    else if (scalar == "f32") result.scalar = CompiledShaderTextureScalar::f32;
    else if (scalar == "u32") result.scalar = CompiledShaderTextureScalar::u32;
    else if (scalar == "s32") result.scalar = CompiledShaderTextureScalar::s32;
    else reflection_error("Texture scalar is invalid");
    result.component_format = parse_texture_component_format(
        string(value[@"component_format"], "component_format"));
    result.normalized_coordinates = boolean(value[@"normalized_coordinates"],
                                            "normalized_coordinates");
    result.comparison = boolean(value[@"comparison"], "comparison");
    const bool float_scalar = result.scalar == CompiledShaderTextureScalar::f16 ||
                              result.scalar == CompiledShaderTextureScalar::f32;
    const bool compatible =
        (float_scalar && (result.component_format == TextureComponentFormat::floating ||
                          result.component_format == TextureComponentFormat::unorm ||
                          result.component_format == TextureComponentFormat::snorm ||
                          result.component_format == TextureComponentFormat::depth)) ||
        (result.scalar == CompiledShaderTextureScalar::u32 &&
         result.component_format == TextureComponentFormat::unsigned_integer) ||
        (result.scalar == CompiledShaderTextureScalar::s32 &&
         result.component_format == TextureComponentFormat::signed_integer);
    if (!compatible || (result.comparison &&
        (result.scalar != CompiledShaderTextureScalar::f32 ||
         result.component_format != TextureComponentFormat::depth)))
        reflection_error("Texture scalar and component format are incompatible");
    return result;
}

bool texture_descriptor_equal(const CompiledShaderTextureDescriptor& left,
                              const CompiledShaderTextureDescriptor& right) {
    return left.dimension == right.dimension && left.array == right.array &&
           left.multisample == right.multisample && left.scalar == right.scalar &&
           left.component_format == right.component_format &&
           left.normalized_coordinates == right.normalized_coordinates &&
           left.comparison == right.comparison;
}

std::optional<DerivedTextureFact> legacy_texture_fact(
    const CompiledShaderResource& resource) {
    const auto& descriptor = *resource.texture_descriptor;
    if (descriptor.multisample ||
        (descriptor.array && descriptor.dimension == CompiledShaderTextureDimension::d3) ||
        descriptor.dimension == CompiledShaderTextureDimension::buffer ||
        descriptor.scalar == CompiledShaderTextureScalar::f16 ||
        (descriptor.component_format == TextureComponentFormat::depth &&
         descriptor.dimension != CompiledShaderTextureDimension::d2 &&
         descriptor.dimension != CompiledShaderTextureDimension::cube))
        return std::nullopt;
    DerivedTextureFact fact;
    fact.cbuf_slot = resource.cbuf_slot;
    fact.handle_byte_offset = resource.handle_byte_offset;
    fact.normalized_coordinates = descriptor.normalized_coordinates;
    fact.comparison = descriptor.comparison;
    switch (descriptor.dimension) {
    case CompiledShaderTextureDimension::d1: fact.dimension = TextureFactDimension::d1; break;
    case CompiledShaderTextureDimension::d2: fact.dimension = TextureFactDimension::d2; break;
    case CompiledShaderTextureDimension::d3: fact.dimension = TextureFactDimension::d3; break;
    case CompiledShaderTextureDimension::cube: fact.dimension = TextureFactDimension::cube; break;
    case CompiledShaderTextureDimension::buffer: return std::nullopt;
    }
    fact.array = descriptor.array;
    fact.multisample = descriptor.multisample;
    switch (descriptor.scalar) {
    case CompiledShaderTextureScalar::f32: fact.scalar = TextureSampledScalar::f32; break;
    case CompiledShaderTextureScalar::u32: fact.scalar = TextureSampledScalar::u32; break;
    case CompiledShaderTextureScalar::s32: fact.scalar = TextureSampledScalar::s32; break;
    case CompiledShaderTextureScalar::f16: return std::nullopt;
    }
    fact.component_format = descriptor.component_format;
    return fact;
}

CompiledGraphicsStage parse_reflection_ported(std::span<const std::byte> bytes,
                                              const Entry& entry,
                                              std::string_view source_version,
                                              bool v3) {
    NSDictionary* root = object(json(bytes, "reflection"), "reflection");
    NSMutableDictionary* schema = [root mutableCopy];
    [schema removeObjectForKey:@"fragment_output_mode"];
    [schema removeObjectForKey:@"depth_bias_mode"];
    [schema removeObjectForKey:@"y_direction"];
    [schema removeObjectForKey:@"point_size"];
    [schema removeObjectForKey:@"point_coord"];
    if (v3) [schema removeObjectForKey:@"draw_support"];
    if (v3) [schema removeObjectForKey:@"vertex_fetch"];
    if (v3) [schema removeObjectForKey:@"compute"];
    if (v3) {
        [schema removeObjectForKey:@"storage_buffers"];
        [schema removeObjectForKey:@"custom_border"];
        exact(schema, {"schema","source_version","program_key","program_size","program_file",
                     "source_file","source_digest","stage","entry_point","vertex_depth",
                     "fragment_position","c1_digest","profile","constant_buffers",
                     "support_buffers","textures","vertex_streams","vertex_inputs",
                     "interface","dependencies","argument_buffer"}, "reflection");
    } else {
        exact(schema, {"schema","source_version","program_key","program_size","program_file",
                     "source_file","source_digest","stage","entry_point","vertex_depth",
                     "fragment_position","c1_digest","profile","constant_buffers",
                     "support_buffers","textures","vertex_streams","vertex_inputs",
                     "interface","dependencies"}, "reflection");
    }
    const auto text = [&](const char* key) { return string(root[ns(key)], key); };
    if (text("schema") != (v3 ? "armigo-shader-stage-v3" : "armigo-shader-stage-v2") ||
        text("source_version") != source_version ||
        text("program_key") != hex(entry.full) ||
        integer(root[@"program_size"], "program_size", pi::maximum_program_size) != entry.program_size ||
        text("program_file") != "program.bin" || text("source_file") != "source.metal" ||
        !lower_hex(text("source_digest")) || text("stage") != stage_name(entry.stage) ||
        (entry.stage == Stage::Compute && !v3))
        reflection_error("Reflection identity or fixed fields do not match the manifest");
    CompiledGraphicsStage stage;
    // Compute stages specialize on their launch: the local size and static
    // shared memory the translator was given must equal the queue meta data.
    if ((root[@"compute"] != nil) != (entry.stage == Stage::Compute))
        reflection_error("Compute launch reflection does not match the stage");
    if (entry.stage == Stage::Compute) {
        NSDictionary* launch = object(root[@"compute"], "compute");
        exact(launch, {"local_size", "shared_memory_bytes"}, "compute");
        NSArray* local = array(launch[@"local_size"], "local_size", 3);
        if (local.count != 3) reflection_error("Compute local size must name three dimensions");
        for (unsigned axis = 0; axis < 3; ++axis) {
            const auto extent = integer(local[axis], "local_size", 1024);
            if (!extent) reflection_error("Compute local size dimensions are at least one");
            stage.compute_local_size[axis] = extent;
        }
        stage.compute_shared_memory_bytes =
            integer(launch[@"shared_memory_bytes"], "shared_memory_bytes", 49152);
    }
    if (root[@"fragment_output_mode"] != nil) {
        if (entry.stage != Stage::Fragment ||
            text("fragment_output_mode") != "render-target-v1")
            reflection_error("Fragment output mode is invalid for this stage");
        stage.fragment_output_types_from_render_targets = true;
    }
    if (root[@"depth_bias_mode"] != nil) {
        if (entry.stage != Stage::Fragment || text("depth_bias_mode") != "shader-v1")
            reflection_error("Depth bias mode is invalid for this stage");
        stage.emulates_depth_bias = true;
    }
    if (root[@"y_direction"] != nil) {
        if (text("y_direction") != "function-constant-v1")
            reflection_error("Y-direction mode is invalid for this stage");
        stage.consumes_y_direction = true;
    }
    if (root[@"point_size"] != nil) {
        const auto mode = text("point_size");
        if (entry.stage != Stage::Vertex || (mode != "fixed-one-v1" && mode != "program-v1"))
            reflection_error("Point-size mode is invalid for this stage");
        stage.writes_point_size = true;
        stage.fixed_point_size = mode == "fixed-one-v1";
    }
    if (root[@"point_coord"] != nil) {
        if (entry.stage != Stage::Fragment || text("point_coord") != "attribute-v1")
            reflection_error("Point-coordinate mode is invalid for this stage");
        stage.reads_point_coord = true;
    }
    stage.stage = entry.stage;
    stage.entry_point = text("entry_point");
    if (stage.entry_point != (entry.stage == Stage::Vertex ? "armigo_vertex_main"
                              : entry.stage == Stage::Compute ? "armigo_compute_main"
                                                              : "armigo_fragment_main"))
        reflection_error("Reflection entry point is unsupported");
    const auto depth = text("vertex_depth");
    if (depth == "minus-one-to-one") stage.vertex_depth = gm::VertexDepthConvention::NegativeOneToOne;
    else if (depth == "zero-to-one") stage.vertex_depth = gm::VertexDepthConvention::ZeroToOne;
    else if (depth != "unspecified") reflection_error("Reflection vertex depth is unsupported");
    const auto position = text("fragment_position");
    if (position == "reciprocal-clip-w")
        stage.fragment_position = gm::FragmentPositionConvention::MetalPositionWMatchesGuestPositionW;
    else if (position != "unspecified") reflection_error("Reflection fragment position is unsupported");
    id c1 = root[@"c1_digest"];
    if (c1 != nil && ![c1 isEqual:[NSNull null]] && !lower_hex(string(c1, "c1_digest")))
        reflection_error("Reflection C1 digest is invalid");
    NSDictionary* profile = object(root[@"profile"], "profile");
    exact(profile, {"identity","version"}, "profile");
    if (string(profile[@"identity"], "profile identity") != admitted_v2_profile_identity ||
        integer(profile[@"version"], "profile version", 0xffffffff) !=
            (v3 ? admitted_v3_profile_version : admitted_v2_profile_version))
        reflection_error("Reflection host profile is outside the admitted runtime subset");

    std::set<unsigned> slots, attributes, guests, metals;
    std::map<unsigned, unsigned> constant_buffer_windows;
    for (id raw in array(root[@"constant_buffers"], "constant_buffers", 18)) {
        NSDictionary* item = object(raw, "constant buffer");
        exact(item, {"guest_binding","size_kind","window_bytes","alignment","policy",
                     "metal_buffer_index"}, "constant buffer");
        const auto guest = integer(item[@"guest_binding"], "guest_binding", 17);
        const auto extent = integer(item[@"window_bytes"], "window_bytes", 65536);
        const auto metal = integer(item[@"metal_buffer_index"], "metal_buffer_index", 30);
        if (extent < 4 || (extent & 3) ||
            integer(item[@"alignment"], "alignment", 4) != 4 ||
            string(item[@"size_kind"], "size_kind") != "constant_buffer_window" ||
            string(item[@"policy"], "policy") != "runtime_zero_words_v1" ||
            !guests.insert(guest).second || !metals.insert(metal).second)
            reflection_error("Constant-buffer reflection is invalid");
        stage.constant_buffer_windows.push_back({static_cast<std::uint8_t>(guest), extent,
                                                 static_cast<std::uint8_t>(metal)});
        constant_buffer_windows.emplace(guest, extent);
    }
    NSArray* support = array(root[@"support_buffers"], "support_buffers", 1);
    if (support.count != (stage.constant_buffer_windows.empty() ? 0u : 1u))
        reflection_error("Support buffer must exist exactly when constant-buffer windows exist");
    for (id raw in support) {
        NSDictionary* item = object(raw, "support buffer");
        exact(item, {"schema","version","byte_extent","alignment","metal_buffer_index"},
              "support buffer");
        const auto metal = integer(item[@"metal_buffer_index"], "metal_buffer_index", 30);
        if (string(item[@"schema"], "support schema") != "armigo-cb-sizes" ||
            integer(item[@"version"], "support version", 1) != 1 ||
            integer(item[@"byte_extent"], "support byte_extent", 72) != 72 ||
            integer(item[@"alignment"], "support alignment", 4) != 4 ||
            !metals.insert(metal).second)
            reflection_error("Constant-buffer support reflection is invalid");
        stage.constant_buffer_support_index = static_cast<std::uint8_t>(metal);
    }
    if (v3 && root[@"storage_buffers"] != nil) {
        std::set<std::pair<unsigned, unsigned>> descriptors;
        for (id raw in array(root[@"storage_buffers"], "storage_buffers", 31)) {
            NSDictionary* item = object(raw, "storage buffer");
            exact(item, {"cbuf_slot","descriptor_byte_offset","metal_buffer_index",
                         "access","alignment"}, "storage buffer");
            const auto slot = integer(item[@"cbuf_slot"], "cbuf_slot", 17);
            const auto offset = integer(item[@"descriptor_byte_offset"],
                                        "descriptor_byte_offset", 65524);
            const auto metal = integer(item[@"metal_buffer_index"],
                                       "metal_buffer_index", 30);
            const auto window = constant_buffer_windows.find(slot);
            const auto access_text = string(item[@"access"], "access");
            CompiledShaderResourceAccess access;
            if (access_text == "read") access = CompiledShaderResourceAccess::Read;
            else if (access_text == "write") access = CompiledShaderResourceAccess::Write;
            else if (access_text == "read_write")
                access = CompiledShaderResourceAccess::ReadWrite;
            else reflection_error("Storage-buffer access is unsupported");
            if ((offset & 7u) || window == constant_buffer_windows.end() ||
                offset + 12u > window->second ||
                integer(item[@"alignment"], "alignment", 4) != 4 ||
                !descriptors.emplace(slot, offset).second || !metals.insert(metal).second)
                reflection_error("Storage-buffer reflection is invalid");
            stage.storage_buffers.push_back({static_cast<std::uint8_t>(slot), offset,
                static_cast<std::uint8_t>(metal), access, 4});
        }
    }
    if (stage.emulates_depth_bias && !metals.insert(30).second)
        reflection_error("Depth bias support buffer aliases a reflected buffer");
    if (v3 && root[@"custom_border"] != nil) {
        NSDictionary* border = object(root[@"custom_border"], "custom_border");
        exact(border, {"abi", "buffer_index", "pairs"}, "custom_border");
        const auto index = integer(border[@"buffer_index"], "buffer_index", 30);
        if ((entry.stage != Stage::Fragment && entry.stage != Stage::Vertex) ||
            string(border[@"abi"], "abi") != "coverage-v1" ||
            !metals.insert(index).second)
            reflection_error("Custom border support ABI or buffer index is invalid");
        CompiledCustomBorderSupport support{static_cast<std::uint8_t>(index), {}};
        std::set<std::uint32_t> textures;
        NSArray* pairs = array(border[@"pairs"], "pairs", 4096);
        if (pairs.count == 0) reflection_error("Custom border support has no pairs");
        for (id value in pairs) {
            NSDictionary* pair = object(value, "custom border pair");
            exact(pair, {"texture", "sampler"}, "custom border pair");
            const auto texture = integer(pair[@"texture"], "texture", 0xffffffffu);
            const auto sampler = integer(pair[@"sampler"], "sampler", 0xffffffffu);
            if (!textures.insert(texture).second)
                reflection_error("Custom border texture is repeated");
            support.pairs.push_back({texture, sampler});
        }
        stage.custom_border = std::move(support);
    }
    if (v3 && root[@"draw_support"] != nil) {
        NSDictionary* support = object(root[@"draw_support"], "draw_support");
        exact(support, {"schema","version","byte_extent","alignment","metal_buffer_index"},
              "draw_support");
        const auto index = integer(support[@"metal_buffer_index"], "draw support index", 30);
        if (entry.stage != Stage::Vertex ||
            string(support[@"schema"], "draw support schema") != "armigo-draw-parameters" ||
            integer(support[@"version"], "draw support version", 1) != 1 ||
            integer(support[@"byte_extent"], "draw support extent", 4) != 4 ||
            integer(support[@"alignment"], "draw support alignment", 4) != 4 ||
            !metals.insert(index).second)
            reflection_error("Draw-parameter support reflection is invalid");
        stage.draw_support_buffer_index = static_cast<std::uint8_t>(index);
    }
    if (v3 && root[@"vertex_fetch"] != nil) {
        NSDictionary* fetch = object(root[@"vertex_fetch"], "vertex_fetch");
        exact(fetch, {"schema","version","byte_extent","alignment","metal_buffer_index"}, "vertex_fetch");
        const auto index = integer(fetch[@"metal_buffer_index"], "vertex fetch index", 30);
        if (entry.stage != Stage::Vertex ||
            string(fetch[@"schema"], "vertex fetch schema") != shader::vertex_fetch_schema ||
            integer(fetch[@"version"], "vertex fetch version", shader::vertex_fetch_version) !=
                shader::vertex_fetch_version ||
            integer(fetch[@"byte_extent"], "vertex fetch extent", shader::vertex_fetch_bytes) !=
                shader::vertex_fetch_bytes ||
            integer(fetch[@"alignment"], "vertex fetch alignment", 8) != 8 ||
            !metals.insert(index).second)
            reflection_error("Vertex fetch reflection is invalid");
        stage.vertex_fetch_buffer_index = static_cast<std::uint8_t>(index);
    }
    if (v3 && root[@"argument_buffer"] != nil &&
        ![root[@"argument_buffer"] isEqual:[NSNull null]]) {
        NSDictionary* layout = object(root[@"argument_buffer"], "argument_buffer");
        exact(layout, {"abi_version","buffer_index","byte_extent","alignment"},
              "argument_buffer");
        const auto buffer = integer(layout[@"buffer_index"], "buffer_index", 30);
        const auto extent = integer(layout[@"byte_extent"], "byte_extent", 0xffffffffu);
        if (integer(layout[@"abi_version"], "abi_version", 1) != 1 ||
            integer(layout[@"alignment"], "alignment", 8) != 8 || !extent || (extent & 7u) ||
            !metals.insert(buffer).second)
            reflection_error("Argument-buffer layout is invalid");
        stage.argument_buffer = CompiledArgumentBufferLayout{
            1, static_cast<std::uint8_t>(buffer), extent, 8};
    } else if (v3 && root[@"argument_buffer"] == nil) {
        reflection_error("Argument-buffer layout is required");
    }
    // Bound resources name the handle words they were proven from. V3 retains
    // each texture and sampler independently; older schemas require paired
    // provenance. Fetch- and query-only resources need no sampler declaration.
    std::map<unsigned, unsigned> texture_indices, sampler_indices;
    std::map<unsigned, unsigned> handle_slots;
    std::map<unsigned, DerivedTextureFact> texture_entry_facts;
    std::map<unsigned, std::pair<bool, bool>> sampler_states;
    std::set<unsigned> texture_metal, sampler_metal;
    std::optional<unsigned> proven_slot;
    if (v3) {
        std::array<std::vector<std::pair<std::uint32_t, std::uint32_t>>, 2> direct_ranges;
        std::map<std::pair<unsigned, unsigned>, std::size_t> direct_textures;
        std::map<std::pair<unsigned, unsigned>, std::size_t> direct_samplers;
        std::set<std::uint32_t> resource_ids;
        std::vector<std::pair<std::uint32_t, std::uint32_t>> member_ranges;
        NSArray* resources = array(root[@"textures"], "textures", 4096);
        for (std::size_t ordinal = 0; ordinal < resources.count; ++ordinal) {
            NSDictionary* item = object(resources[ordinal], "texture");
            const auto kind_text = string(item[@"kind"], "texture kind");
            const bool is_texture = kind_text == "texture";
            if (!is_texture && kind_text != "sampler")
                reflection_error("Texture kind is unsupported");
            exact(item, is_texture
                ? std::initializer_list<const char*>{"cbuf_slot","handle_byte_offset","kind",
                      "dimension","array","multisample","scalar","component_format",
                      "normalized_coordinates","comparison","resource_id","array_length",
                      "location","resource_kind","access"}
                : std::initializer_list<const char*>{"cbuf_slot","handle_byte_offset","kind",
                      "comparison","normalized_coordinates","resource_id","array_length",
                      "location"}, "texture entry");
            const auto resource_id = integer(item[@"resource_id"], "resource_id", 0xffffffffu);
            const auto length = integer(item[@"array_length"], "array_length", 0xffffffffu);
            const auto slot = integer(item[@"cbuf_slot"], "cbuf_slot", 17);
            const auto offset = integer(item[@"handle_byte_offset"], "handle_byte_offset", 65532);
            const auto provenance_end = static_cast<std::uint64_t>(offset) +
                                        static_cast<std::uint64_t>(length) * 4;
            if (!resource_ids.insert(resource_id).second || !length ||
                (offset & 3u) || provenance_end > 65536)
                reflection_error("Texture provenance reflection is invalid");

            CompiledShaderResource resource;
            resource.resource_id = resource_id;
            resource.array_length = length;
            resource.cbuf_slot = static_cast<std::uint8_t>(slot);
            resource.handle_byte_offset = offset;
            resource.comparison = boolean(item[@"comparison"], "comparison");
            resource.normalized_coordinates = boolean(
                item[@"normalized_coordinates"], "normalized_coordinates");
            if (is_texture) {
                const auto resource_kind = string(item[@"resource_kind"], "resource_kind");
                if (resource_kind == "sampled")
                    resource.kind = CompiledShaderResourceKind::SampledTexture;
                else if (resource_kind == "storage_image")
                    resource.kind = CompiledShaderResourceKind::StorageImage;
                else reflection_error("Texture resource kind is unsupported");
                const auto access = string(item[@"access"], "access");
                if (access == "read") resource.access = CompiledShaderResourceAccess::Read;
                else if (access == "write") resource.access = CompiledShaderResourceAccess::Write;
                else if (access == "read_write")
                    resource.access = CompiledShaderResourceAccess::ReadWrite;
                else reflection_error("Texture access is unsupported");
                resource.texture_descriptor = parse_texture_descriptor_v3(item);
                if (resource.kind == CompiledShaderResourceKind::SampledTexture &&
                    resource.access != CompiledShaderResourceAccess::Read)
                    reflection_error("Sampled texture resource must be read-only");
            } else {
                resource.kind = CompiledShaderResourceKind::Sampler;
            }

            NSDictionary* location = object(item[@"location"], "resource location");
            if (stage.argument_buffer) {
                exact(location, {"kind","byte_offset","count","stride"}, "resource location");
                const auto byte_offset = integer(location[@"byte_offset"], "byte_offset", 0xffffffffu);
                const auto count = integer(location[@"count"], "count", 0xffffffffu);
                if (string(location[@"kind"], "location kind") != "argument_buffer_member" ||
                    (byte_offset & 7u) || count != length ||
                    integer(location[@"stride"], "stride", 0xffffffffu) != 8)
                    reflection_error("Argument-buffer member location is invalid");
                const auto member_end = static_cast<std::uint64_t>(byte_offset) +
                                        static_cast<std::uint64_t>(length) * 8;
                if (member_end > 0xffffffffu || member_end > stage.argument_buffer->byte_extent)
                    reflection_error("Argument-buffer member exceeds its layout");
                if (std::any_of(member_ranges.begin(), member_ranges.end(), [&](const auto& range) {
                        return member_end > range.first && byte_offset < range.second;
                    }))
                    reflection_error("Argument-buffer members overlap");
                member_ranges.emplace_back(byte_offset, static_cast<std::uint32_t>(member_end));
                resource.location.kind = CompiledShaderResourceLocationKind::ArgumentBufferMember;
                resource.location.byte_offset = byte_offset;
                resource.location.count = count;
                resource.location.stride = 8;
            } else {
                exact(location, {"kind","namespace","first","count"}, "resource location");
                const auto namespace_text = string(location[@"namespace"], "namespace");
                const auto expected_namespace = is_texture ? "texture" : "sampler";
                const auto first = integer(location[@"first"], "first", 0xffffffffu);
                const auto count = integer(location[@"count"], "count", 0xffffffffu);
                const auto capacity = is_texture ? 128u : 16u;
                const auto end = static_cast<std::uint64_t>(first) + count;
                if (string(location[@"kind"], "location kind") != "direct" ||
                    namespace_text != expected_namespace || count != length || end > capacity)
                    reflection_error("Direct resource location is invalid");
                auto& occupied = direct_ranges[is_texture ? 0 : 1];
                if (std::any_of(occupied.begin(), occupied.end(), [&](const auto& range) {
                        return end > range.first && first < range.second;
                    }))
                    reflection_error("Direct resource locations overlap");
                occupied.emplace_back(first, static_cast<std::uint32_t>(end));
                resource.location.resource_namespace = is_texture
                    ? CompiledShaderResourceNamespace::Texture
                    : CompiledShaderResourceNamespace::Sampler;
                resource.location.first = first;
                resource.location.count = count;
                if (!(is_texture ? direct_textures : direct_samplers)
                        .emplace(std::make_pair(slot, offset), stage.resources.size()).second)
                    stage.legacy_texture_bindings_compatible = false;
            }
            if (!proven_slot) proven_slot = slot;
            else if (*proven_slot != slot) stage.legacy_texture_bindings_compatible = false;
            stage.resources.push_back(std::move(resource));
        }
        if (stage.argument_buffer) {
            std::sort(member_ranges.begin(), member_ranges.end());
            std::uint32_t cursor = 0;
            for (const auto& [start, end] : member_ranges) {
                if (start != cursor) reflection_error("Argument-buffer members must be dense");
                cursor = end;
            }
            if (cursor != stage.argument_buffer->byte_extent)
                reflection_error("Argument-buffer extent does not match its dense members");
            stage.legacy_texture_bindings_compatible = false;
        } else {
            for (const auto& [key, index] : direct_textures) {
                const auto& texture = stage.resources[index];
                const auto sampler = direct_samplers.find(key);
                const auto fact = legacy_texture_fact(texture);
                if (texture.array_length != 1 ||
                    texture.kind != CompiledShaderResourceKind::SampledTexture ||
                    texture.access != CompiledShaderResourceAccess::Read || !fact) {
                    stage.legacy_texture_bindings_compatible = false;
                    continue;
                }
                std::uint8_t sampler_index = gm::TextureBinding::no_sampler;
                if (sampler != direct_samplers.end()) {
                    const auto& state = stage.resources[sampler->second];
                    if (state.array_length != 1 || state.comparison != texture.comparison ||
                        state.normalized_coordinates != texture.normalized_coordinates ||
                        !fact->normalized_coordinates ||
                        fact->scalar != TextureSampledScalar::f32) {
                        stage.legacy_texture_bindings_compatible = false;
                        continue;
                    }
                    sampler_index = static_cast<std::uint8_t>(state.location.first);
                } else if (!fact->normalized_coordinates &&
                           fact->dimension != TextureFactDimension::d2) {
                    stage.legacy_texture_bindings_compatible = false;
                    continue;
                }
                stage.textures.push_back({texture.handle_byte_offset,
                    static_cast<std::uint8_t>(texture.location.first), sampler_index});
                stage.texture_facts.push_back(*fact);
            }
            if (std::any_of(direct_samplers.begin(), direct_samplers.end(),
                    [&](const auto& item) { return !direct_textures.contains(item.first); }))
                stage.legacy_texture_bindings_compatible = false;
        }
    } else for (id raw in array(root[@"textures"], "textures", 32)) {
        NSDictionary* item = object(raw, "texture");
        const auto kind = string(item[@"kind"], "texture kind");
        if (kind != "texture" && kind != "sampler")
            reflection_error("Texture kind is unsupported");
        const bool is_texture = kind == "texture";
        exact(item, is_texture
            ? std::initializer_list<const char*>{"cbuf_slot", "handle_byte_offset", "kind",
                  "dimension", "array", "multisample", "scalar", "component_format",
                  "normalized_coordinates", "comparison", "texture_index"}
            : std::initializer_list<const char*>{"cbuf_slot", "handle_byte_offset", "kind",
                  "comparison", "normalized_coordinates", "sampler_index"},
            "texture entry");
        const auto slot = integer(item[@"cbuf_slot"], "cbuf_slot", 17);
        const auto offset = integer(item[@"handle_byte_offset"], "handle_byte_offset", 32764);
        auto& metal_indices = is_texture ? texture_indices : sampler_indices;
        if (offset & 3u || !metal_indices.emplace(offset, 0).second)
            reflection_error("Texture reflection is invalid");
        // A texture and its sampler share one handle word and must agree on
        // the constant buffer that word was proven from.
        const auto provenance = handle_slots.emplace(offset, slot);
        if (!provenance.second && provenance.first->second != slot)
            reflection_error("Texture provenance reflection is invalid");
        if (proven_slot && *proven_slot != slot)
            reflection_error("Texture provenance reflection is invalid");
        proven_slot = slot;
        if (is_texture) {
            texture_entry_facts.emplace(offset, parse_texture_descriptor(
                item, static_cast<std::uint8_t>(slot), offset));
            const auto texture = integer(item[@"texture_index"], "texture_index", 15);
            if (!texture_metal.insert(texture).second)
                reflection_error("Texture reflection is invalid");
            texture_indices[offset] = texture;
        } else {
            sampler_states.emplace(offset, std::make_pair(
                boolean(item[@"comparison"], "comparison"),
                boolean(item[@"normalized_coordinates"], "normalized_coordinates")));
            const auto sampler = integer(item[@"sampler_index"], "sampler_index", 15);
            if (!sampler_metal.insert(sampler).second)
                reflection_error("Sampler reflection is invalid");
            sampler_indices[offset] = sampler;
        }
    }
    for (const auto& [offset, texture] : texture_indices) {
        const auto sampler = sampler_indices.find(offset);
        const auto& fact = texture_entry_facts.at(offset);
        if (sampler != sampler_indices.end()) {
            const auto state = sampler_states.at(offset);
            if (state.first != fact.comparison || state.second != fact.normalized_coordinates)
                reflection_error("Reflected texture and sampler states disagree");
            if (!fact.normalized_coordinates || fact.scalar != TextureSampledScalar::f32)
                reflection_error("Sampled texture reflection is outside the supported profile");
        } else if (!fact.normalized_coordinates && fact.dimension != TextureFactDimension::d2) {
            // The emitter supports unnormalized rectangle descriptors only on
            // sampler-less 2D fetch/query paths.
            reflection_error("Unnormalized texture reflection is outside the supported profile");
        }
        stage.textures.push_back({static_cast<std::uint32_t>(offset),
                                  static_cast<std::uint8_t>(texture),
                                  sampler == sampler_indices.end()
                                      ? gm::TextureBinding::no_sampler
                                      : static_cast<std::uint8_t>(sampler->second)});
        stage.texture_facts.push_back(fact);
    }
    if (std::any_of(sampler_indices.begin(), sampler_indices.end(),
                    [&](const auto& item) { return !texture_indices.contains(item.first); }))
        reflection_error("Reflected sampler has no texture at its handle offset");
    // Vertex streams bind at 18 + reflected ordinal; only the float32x3 fetch
    // the draw backend implements is admitted.
    for (id raw in array(root[@"vertex_streams"], "vertex_streams", 32)) {
        NSDictionary* item = object(raw, "vertex stream");
        exact(item, {"slot","attribute_index","format","components","byte_stride"},
              "vertex stream");
        const auto slot = integer(item[@"slot"], "slot", 31);
        const auto attribute = integer(item[@"attribute_index"], "attribute_index", 30);
        const auto components = integer(item[@"components"], "components", 4);
        const auto byte_stride = integer(item[@"byte_stride"], "byte_stride", 64);
        if (string(item[@"format"], "format") != "f32" || components != 3 ||
            byte_stride != 12 || !slots.insert(slot).second || !attributes.insert(attribute).second)
            reflection_error("Vertex stream reflection is outside the supported profile");
        stage.vertex_attributes.push_back({static_cast<std::uint8_t>(slot),
            static_cast<std::uint8_t>(attribute), gm::VertexFormat::Float32x3});
    }
    // Legacy stream entries bind at 18 + ordinal; those Metal indices must
    // stay inside the namespace and disjoint from the CB windows and support
    // buffer recorded above.
    for (std::size_t ordinal = 0; ordinal < stage.vertex_attributes.size(); ++ordinal) {
        const auto metal = 18 + ordinal;
        if (metal >= 31 || !metals.insert(metal).second)
            reflection_error("Constant/support buffer plan aliases or exceeds vertex-stream indices");
    }
    // Class-typed vertex inputs: the translated declaration's storage class
    // was derived from the code, so the runtime must feed the location from a
    // guest attribute of the same class. Streams and classes never mix; the
    // stream each class input reads is resolved from guest state at draw time.
    std::map<unsigned, graphics::VertexFormatClass> class_slots;
    for (id raw in array(root[@"vertex_inputs"], "vertex_inputs", 32)) {
        NSDictionary* item = object(raw, "vertex input");
        exact(item, {"location","class"}, "vertex input");
        const auto location = integer(item[@"location"], "location", 31);
        const auto class_text = string(item[@"class"], "class");
        graphics::VertexFormatClass input_class;
        if (class_text == "float") input_class = graphics::VertexFormatClass::float_class;
        else if (class_text == "uint") input_class = graphics::VertexFormatClass::uint_class;
        else if (class_text == "sint") input_class = graphics::VertexFormatClass::sint_class;
        else reflection_error("Vertex input class is unsupported");
        if (!class_slots.emplace(location, input_class).second)
            reflection_error("Vertex input reflection is invalid");
        stage.vertex_class_inputs.push_back({static_cast<std::uint8_t>(location), input_class});
    }
    NSDictionary* interface = object(root[@"interface"], "interface");
    exact(interface, {"position_input_mask","position_output_mask","color_output_mask",
                      "color_outputs","writes_fragment_depth","writes_sample_mask",
                      "generic_inputs","generic_outputs"}, "interface");
    stage.interface.position_input_mask = static_cast<std::uint8_t>(
        integer(interface[@"position_input_mask"], "position_input_mask", 15));
    stage.interface.position_output_mask = static_cast<std::uint8_t>(
        integer(interface[@"position_output_mask"], "position_output_mask", 15));
    stage.interface.color_output_mask = integer(interface[@"color_output_mask"],
                                                "color_output_mask", 0xffffffffu);
    // Typed colour exports: one validated record per written render target.
    // The records must rebuild the summary mask exactly, at four bits per
    // target, and every record restates the type its fragment_output_type
    // dependency proved (checked after the dependency list is parsed).
    std::map<unsigned, CompiledColorOutput> color_records;
    std::uint32_t colour_summary = 0;
    for (id raw in array(interface[@"color_outputs"], "color_outputs", 8)) {
        NSDictionary* item = object(raw, "colour output");
        exact(item, {"location","component_mask","scalar","components"}, "colour output");
        const auto location = integer(item[@"location"], "location", 7);
        const auto mask = integer(item[@"component_mask"], "component_mask", 15);
        const auto components = integer(item[@"components"], "components", 4);
        const auto scalar = string(item[@"scalar"], "scalar");
        CompiledColorOutput record;
        record.location = static_cast<std::uint8_t>(location);
        record.component_mask = static_cast<std::uint8_t>(mask);
        record.components = static_cast<std::uint8_t>(components);
        if (scalar == "f32") record.scalar = ColorOutputScalar::f32;
        else if (scalar == "u32") record.scalar = ColorOutputScalar::u32;
        else if (scalar == "s32") record.scalar = ColorOutputScalar::s32;
        else record.scalar = static_cast<ColorOutputScalar>(0xff);
        if (!mask || !components || (mask >> components) ||
            record.scalar == static_cast<ColorOutputScalar>(0xff) ||
            !color_records.emplace(location, record).second)
            reflection_error("Colour output reflection is invalid");
        colour_summary |= mask << (4 * location);
    }
    for (const auto& [location, record] : color_records) {
        if (stage.fragment_output_types_from_render_targets &&
            (record.scalar != ColorOutputScalar::u32 || record.components != 4))
            reflection_error("Adaptive fragment outputs require raw uint32x4 storage");
        stage.color_outputs.push_back(record);
    }
    stage.writes_fragment_depth = boolean(interface[@"writes_fragment_depth"],
                                          "writes_fragment_depth");
    stage.writes_sample_mask = boolean(interface[@"writes_sample_mask"],
                                       "writes_sample_mask");
    if (stage.emulates_depth_bias && stage.writes_fragment_depth)
        reflection_error("Shader depth bias cannot replace a guest fragment-depth export");
    if (colour_summary != stage.interface.color_output_mask)
        reflection_error("Colour output records disagree with the summary mask");
    const auto generic = [&](NSString* key, std::vector<gm::GenericInterface>& output) {
        std::set<unsigned> seen;
        const bool inputs = [key isEqualToString:@"generic_inputs"];
        for (id raw in array(interface[key], "generic interface", 32)) {
            NSDictionary* item = object(raw, "generic interface");
            exact(item, {"slot","component_mask","scalar","components","interpolation"},
                  "generic interface");
            const auto slot = integer(item[@"slot"], "slot", 31);
            const auto mask = integer(item[@"component_mask"], "component_mask", 15);
            const auto components = integer(item[@"components"], "components", 4);
            const auto scalar = string(item[@"scalar"], "scalar");
            // Class-typed vertex inputs declare uint4/int4; everything else
            // stays float32. Fragment inputs are always float.
            const bool integer_scalar = scalar == "u32" || scalar == "s32";
            if (!mask || !components || components > 4 ||
                (scalar != "f32" && !(integer_scalar && entry.stage == Stage::Vertex && inputs)) ||
                (integer_scalar && components != 4) || !seen.insert(slot).second)
                reflection_error("Generic interface is invalid");
            output.push_back({static_cast<std::uint8_t>(slot), static_cast<std::uint8_t>(mask),
                              parse_interpolation_v2(string(item[@"interpolation"], "interpolation"))});
        }
    };
    generic(@"generic_inputs", stage.interface.generic_inputs);
    generic(@"generic_outputs", stage.interface.generic_outputs);
    unsigned dependencies = 0;
    std::optional<unsigned> recorded_handle_slot;
    std::set<std::pair<unsigned, unsigned>> recorded_fact_offsets, recorded_array_lengths;
    std::map<unsigned, std::pair<std::string, unsigned>> output_type_facts;
    for (id raw in array(root[@"dependencies"], "dependencies", 1024)) {
        NSDictionary* item = object(raw, "dependency");
        exact(item, {"kind","index","byte_offset","component","origin","value"}, "dependency");
        const auto kind = string(item[@"kind"], "dependency kind");
        if (string(item[@"origin"], "dependency origin") != "validated_header" &&
            string(item[@"origin"], "dependency origin") != "captured_state" &&
            string(item[@"origin"], "dependency origin") != "explicit_option" &&
            string(item[@"origin"], "dependency origin") != "derived")
            reflection_error("Dependency origin is unsupported");
        if (!admitted_v2_dependency(kind) && !(v3 && kind == "texture_array_length") &&
            !(entry.stage == Stage::Compute &&
              (kind == "compute_local_size" || kind == "compute_shared_memory_bytes")))
            reflection_error("Consumed fact " + kind + " is outside the admitted runtime subset");
        if (kind == "compute_local_size") {
            NSArray* value = array(item[@"value"], "compute_local_size value", 3);
            if (value.count != 3) reflection_error("Compute local size dependency is invalid");
            for (unsigned axis = 0; axis < 3; ++axis)
                if (integer(value[axis], "compute_local_size value", 1024) != stage.compute_local_size[axis])
                    reflection_error("Compute local size dependency disagrees with the launch reflection");
        }
        if (kind == "compute_shared_memory_bytes" &&
            integer(item[@"value"], "compute_shared_memory_bytes value", 49152) !=
                stage.compute_shared_memory_bytes)
            reflection_error("Compute shared memory dependency disagrees with the launch reflection");
        if (kind == "depth_convention") {
            if (entry.stage != Stage::Vertex || stage.vertex_depth == gm::VertexDepthConvention::Unspecified)
                reflection_error("Depth dependency requires a vertex stage with an explicit convention");
        }
        if (kind == "vertex_attribute_format" &&
            (entry.stage != Stage::Vertex || stage.vertex_attributes.empty()))
            reflection_error("Attribute-format dependency requires declared vertex streams");
        if (kind == "fragment_output_type") {
            // The colour output type fact: consumed once per written target.
            // Its value must restate exactly what the interface record claims.
            NSDictionary* value = object(item[@"value"], "dependency value");
            exact(value, {"scalar","components"}, "fragment_output_type value");
            const auto target = integer(item[@"index"], "dependency index", 7);
            if (integer(item[@"byte_offset"], "dependency byte_offset", 0xffffffffu) != 0 ||
                integer(item[@"component"], "dependency component", 3) != 0)
                reflection_error("Fragment output type dependency key is invalid");
            const auto components = integer(value[@"components"], "components", 4);
            const auto scalar = string(value[@"scalar"], "scalar");
            if (!components || components > 4 ||
                (scalar != "f32" && scalar != "u32" && scalar != "s32") ||
                !output_type_facts.emplace(target, std::make_pair(scalar, components)).second)
                reflection_error("Fragment output type dependency is invalid");
        }
        if (kind == "input_interpolation" &&
            stage.interface.generic_inputs.empty())
            reflection_error("Interpolation dependency requires reflected generic inputs");
        // The local memory size comes from the program's own header and sizes
        // a per-thread array inside the compiled stage; nothing to bind.
        if (kind == "local_memory_bytes" &&
            string(item[@"origin"], "dependency origin") != "validated_header")
            reflection_error("Local memory dependency must come from the validated header");
        if (kind == "texture_handle_cbuf") {
            if (recorded_handle_slot)
                reflection_error("Dependency reflection is invalid");
            recorded_handle_slot = integer(item[@"value"], "dependency value", 17);
        }
        if (v3 && kind == "texture_array_length") {
            const auto slot = integer(item[@"index"], "dependency index", 17);
            const auto offset = integer(item[@"byte_offset"], "dependency byte_offset", v3 ? 65532 : 32764);
            const auto length = integer(item[@"value"], "array length", 0xffffffffu);
            if ((offset & 3u) || !length ||
                integer(item[@"component"], "dependency component", 3) != 0 ||
                !recorded_array_lengths.emplace(slot, offset).second)
                reflection_error("Resource array-length dependency is invalid");
            bool matched = false;
            for (const auto& resource : stage.resources) {
                if (resource.cbuf_slot != slot || resource.handle_byte_offset != offset) continue;
                matched |= resource.array_length == length;
                if (resource.array_length != 1 && resource.array_length != length)
                    reflection_error("Resource array-length dependency disagrees with its provenance");
            }
            if (!matched) reflection_error("Resource array-length dependency names no bound array");
        }
        if (kind == "texture_descriptor") {
            NSDictionary* value = object(item[@"value"], "dependency value");
            exact(value, {"dimension","array","multisample","scalar","component_format",
                          "normalized_coordinates","comparison"},
                  "texture_descriptor value");
            const auto slot = integer(item[@"index"], "dependency index", 17);
            const auto offset = integer(item[@"byte_offset"], "dependency byte_offset", v3 ? 65532 : 32764);
            if (offset & 3u || !recorded_fact_offsets.emplace(slot, offset).second)
                reflection_error("Texture descriptor dependency is invalid");
            // The fact must name a reflected bound handle at its recorded slot
            // and agree with it; draw time re-checks both against the live TIC.
            bool agrees = false;
            if (v3) {
                const auto descriptor = parse_texture_descriptor_v3(value);
                if (integer(item[@"component"], "dependency component", 3) != 0)
                    reflection_error("Resource descriptor dependency component is invalid");
                for (const auto& resource : stage.resources) {
                    if (resource.cbuf_slot != slot || resource.handle_byte_offset != offset) continue;
                    const bool match = resource.texture_descriptor
                        ? texture_descriptor_equal(*resource.texture_descriptor, descriptor)
                        : resource.comparison == descriptor.comparison &&
                          resource.normalized_coordinates == descriptor.normalized_coordinates;
                    if (!match) reflection_error("Resource descriptor dependency disagrees with its provenance");
                    agrees = true;
                }
            } else {
                const auto fact = parse_texture_descriptor(
                    value, static_cast<std::uint8_t>(slot), offset);
                const auto bound = texture_entry_facts.find(offset);
                agrees = bound != texture_entry_facts.end() &&
                          handle_slots.find(offset) != handle_slots.end() &&
                          handle_slots.find(offset)->second == slot &&
                          texture_fact_equal(bound->second, fact);
            }
            if (!agrees)
                reflection_error("Texture descriptor fact names no agreeing bound handle");
        }
        ++dependencies;
    }
    // Every written colour target consumed its type fact exactly once, and the
    // record must restate the fact's type. Vertex stages never consume it.
    if (entry.stage == Stage::Fragment) {
        if (output_type_facts.size() != stage.color_outputs.size())
            reflection_error("Written colour targets and fragment_output_type facts disagree");
        for (const auto& output : stage.color_outputs) {
            const auto fact = output_type_facts.find(output.location);
            const std::string scalar = output.scalar == ColorOutputScalar::f32 ? "f32" :
                output.scalar == ColorOutputScalar::u32 ? "u32" : "s32";
            if (fact == output_type_facts.end() ||
                fact->second.first != scalar || fact->second.second != output.components)
                reflection_error("Colour output disagrees with its fragment_output_type fact");
        }
    } else if (!output_type_facts.empty()) {
        reflection_error("A vertex stage cannot consume fragment output type facts");
    }
    // The recorded slot is the draw-time contract for register 0x982: bound
    // handles resolve through it, so it must match the reflected provenance.
    if (!v3 && proven_slot && recorded_handle_slot && *recorded_handle_slot != *proven_slot)
        reflection_error("Texture provenance differs from the consumed texture_handle_cbuf fact");
    if (!v3 && (recorded_handle_slot || proven_slot))
        stage.texture_handle_cbuf =
            static_cast<std::uint8_t>(recorded_handle_slot ? *recorded_handle_slot : *proven_slot);
    if (v3) {
        // Independent resource provenance does not erase a consumed selector fact.
        if (recorded_handle_slot)
            stage.texture_handle_cbuf = static_cast<std::uint8_t>(*recorded_handle_slot);
        if (recorded_handle_slot && proven_slot && *recorded_handle_slot != *proven_slot)
            stage.legacy_texture_bindings_compatible = false;
    }
    if (entry.stage == Stage::Compute) {
        // A compute stage has no graphics interface; its resources bind
        // through the same planner as draw stages, from queue-meta-data
        // constant buffers.
        if (stage.vertex_depth != gm::VertexDepthConvention::Unspecified ||
            stage.fragment_position != gm::FragmentPositionConvention::Unspecified ||
            !stage.vertex_attributes.empty() || !stage.vertex_class_inputs.empty() ||
            stage.interface.position_input_mask || stage.interface.position_output_mask ||
            stage.interface.color_output_mask || !stage.color_outputs.empty() ||
            !stage.interface.generic_inputs.empty() || !stage.interface.generic_outputs.empty() ||
            stage.writes_fragment_depth || stage.writes_sample_mask ||
            stage.custom_border || stage.draw_support_buffer_index || stage.vertex_fetch_buffer_index)
            reflection_error("Compute reflection is outside the supported profile");
    } else if (entry.stage == Stage::Vertex) {
        // Vertex stages admit the same texture provenance facts as fragment
        // stages: the draw backend resolves and verifies both stages through
        // one binding path. Generic input and output interpolation is drawn
        // from the translation and matched at link time, so every parsed
        // interpolation is admissible here.
        if (stage.vertex_depth == gm::VertexDepthConvention::Unspecified ||
            stage.fragment_position != gm::FragmentPositionConvention::Unspecified ||
            stage.interface.position_input_mask || stage.interface.position_output_mask != 15 ||
            stage.interface.color_output_mask || !stage.color_outputs.empty() ||
            stage.writes_fragment_depth || stage.writes_sample_mask ||
            (!stage.vertex_class_inputs.empty() && !stage.vertex_attributes.empty()) ||
            (stage.vertex_class_inputs.empty() &&
             stage.interface.generic_inputs.size() != stage.vertex_attributes.size()) ||
            dependencies == 0)
            reflection_error("Vertex reflection is outside the supported profile");
        if (!stage.vertex_class_inputs.empty()) {
            if (stage.interface.generic_inputs.size() != stage.vertex_class_inputs.size())
                reflection_error("Vertex input options do not cover reflected inputs");
            for (const auto& input : stage.interface.generic_inputs) {
                const auto bound = class_slots.find(input.slot);
                if (bound == class_slots.end())
                    reflection_error("Vertex input options do not cover reflected inputs");
            }
        }
        for (const auto& option : stage.vertex_attributes) {
            const auto match = std::find_if(stage.interface.generic_inputs.begin(),
                stage.interface.generic_inputs.end(), [&](const auto& item) {
                    return item.slot == option.generic_slot && item.component_mask &&
                           !(item.component_mask & ~7u);
                });
            if (match == stage.interface.generic_inputs.end())
                reflection_error("Vertex input options do not cover reflected inputs");
        }
    } else {
        const bool perspective = std::any_of(stage.interface.generic_inputs.begin(),
            stage.interface.generic_inputs.end(), [](const auto& item) {
                return item.interpolation == shader::graphics::Interpolation::Perspective;
            });
        // Perspective inputs do not require a declared position input: the
        // translator corrects through the compiled convention below, which is
        // the same requirement the draw backend enforces per input.
        // Colour output admission is the draw backend's decision: the loader
        // accepts any validated record set whose union matches the summary
        // mask and whose types their facts proved.
        if (stage.vertex_depth != gm::VertexDepthConvention::Unspecified ||
            !stage.vertex_attributes.empty() || !stage.vertex_class_inputs.empty() ||
            stage.interface.position_output_mask ||
            !stage.interface.generic_outputs.empty() ||
            (stage.interface.position_input_mask != 0 && stage.interface.position_input_mask != 8) ||
            ((stage.interface.position_input_mask == 8 || perspective) &&
             stage.fragment_position != gm::FragmentPositionConvention::MetalPositionWMatchesGuestPositionW) ||
            (stage.interface.position_input_mask == 0 && !perspective &&
             stage.fragment_position != gm::FragmentPositionConvention::Unspecified))
            reflection_error("Fragment reflection is outside the supported profile");
    }
    if (!metal::valid_custom_border_pairs(stage))
        reflection_error("Custom border pair is outside the admitted sampling profile");
    return stage;
}

CompiledGraphicsStage parse_reflection(std::span<const std::byte> bytes, const Entry& entry,
                                       std::string_view source_version) {
    NSDictionary* root = object(json(bytes, "reflection"), "reflection");
    exact(root, {"schema","source_version","program_key","program_size","program_file",
                 "source_file","source_digest","stage","entry_point","vertex_depth",
                 "fragment_position","vertex_attributes","constant_buffers","textures","interface"},
          "reflection");
    const auto text = [&](const char* key) { return string(root[ns(key)], key); };
    if (text("schema") != "armigo-shader-stage-v1" || text("source_version") != source_version ||
        text("program_key") != hex(entry.full) ||
        integer(root[@"program_size"], "program_size", pi::maximum_program_size) != entry.program_size ||
        text("program_file") != "program.bin" || text("source_file") != "source.metal" ||
        !lower_hex(text("source_digest")) || text("stage") != (entry.stage == Stage::Vertex ? "vertex" : "fragment"))
        reflection_error("Reflection identity or fixed fields do not match the manifest");
    CompiledGraphicsStage stage;
    stage.stage = entry.stage;
    stage.entry_point = text("entry_point");
    if (stage.entry_point != (entry.stage == Stage::Vertex ? "armigo_vertex_main" : "armigo_fragment_main"))
        reflection_error("Reflection entry point is unsupported");
    const auto depth = text("vertex_depth");
    if (depth == "minus-one-to-one") stage.vertex_depth = gm::VertexDepthConvention::NegativeOneToOne;
    else if (depth == "zero-to-one") stage.vertex_depth = gm::VertexDepthConvention::ZeroToOne;
    else if (depth != "unspecified") reflection_error("Reflection vertex depth is unsupported");
    const auto position = text("fragment_position");
    if (position == "reciprocal-clip-w")
        stage.fragment_position = gm::FragmentPositionConvention::MetalPositionWMatchesGuestPositionW;
    else if (position != "unspecified") reflection_error("Reflection fragment position is unsupported");

    std::set<unsigned> slots, attributes, guests, metals, handles, textures, samplers;
    for (id raw in array(root[@"vertex_attributes"], "vertex_attributes", 32)) {
        NSDictionary* item = object(raw, "vertex attribute");
        exact(item, {"generic_slot","attribute_index","format"}, "vertex attribute");
        const auto slot = integer(item[@"generic_slot"], "generic_slot", 31);
        const auto attribute = integer(item[@"attribute_index"], "attribute_index", 30);
        if (string(item[@"format"], "format") != "float32x3" || !slots.insert(slot).second ||
            !attributes.insert(attribute).second) reflection_error("Vertex attribute reflection is invalid");
        stage.vertex_attributes.push_back({static_cast<std::uint8_t>(slot),
            static_cast<std::uint8_t>(attribute), gm::VertexFormat::Float32x3});
    }
    for (id raw in array(root[@"constant_buffers"], "constant_buffers", 18)) {
        NSDictionary* item = object(raw, "constant buffer");
        exact(item, {"guest_binding","minimum_size","metal_buffer_index"}, "constant buffer");
        const auto guest = integer(item[@"guest_binding"], "guest_binding", 17);
        const auto minimum = integer(item[@"minimum_size"], "minimum_size", 65536);
        const auto metal = integer(item[@"metal_buffer_index"], "metal_buffer_index", 17);
        if (!minimum || (minimum & 3) || !guests.insert(guest).second || !metals.insert(metal).second)
            reflection_error("Constant-buffer reflection is invalid");
        stage.constant_buffers.push_back({static_cast<std::uint8_t>(guest), minimum,
                                          static_cast<std::uint8_t>(metal)});
    }
    for (id raw in array(root[@"textures"], "textures", 16)) {
        NSDictionary* item = object(raw, "texture");
        exact(item, {"handle_byte_offset","texture_index","sampler_index"}, "texture");
        const auto offset = integer(item[@"handle_byte_offset"], "handle_byte_offset", 32764);
        const auto texture = integer(item[@"texture_index"], "texture_index", 15);
        const auto sampler = integer(item[@"sampler_index"], "sampler_index", 15);
        if ((offset & 3) || !handles.insert(offset).second ||
            !textures.insert(texture).second || !samplers.insert(sampler).second)
            reflection_error("Texture reflection is invalid");
        stage.textures.push_back({offset, static_cast<std::uint8_t>(texture),
                                  static_cast<std::uint8_t>(sampler)});
    }
    NSDictionary* interface = object(root[@"interface"], "interface");
    exact(interface, {"position_input_mask","position_output_mask","color_output_mask",
                      "generic_inputs","generic_outputs"}, "interface");
    stage.interface.position_input_mask = static_cast<std::uint8_t>(
        integer(interface[@"position_input_mask"], "position_input_mask", 15));
    stage.interface.position_output_mask = static_cast<std::uint8_t>(
        integer(interface[@"position_output_mask"], "position_output_mask", 15));
    stage.interface.color_output_mask = integer(interface[@"color_output_mask"],
                                                "color_output_mask", 0xffffffffu);
    // The v1 schema predates typed colour records; its only admissible
    // fragment export is the known target-0 float32x4 record, synthesized here
    // so both schemas share the compiled-stage representation. Depth and
    // sample-mask writes have no v1 form at all.
    if (entry.stage == Stage::Fragment)
        stage.color_outputs.push_back({0, 15, ColorOutputScalar::f32, 4});
    const auto generic = [&](NSString* key, std::vector<gm::GenericInterface>& output) {
        std::set<unsigned> seen;
        for (id raw in array(interface[key], "generic interface", 32)) {
            NSDictionary* item = object(raw, "generic interface");
            exact(item, {"slot","component_mask","interpolation"}, "generic interface");
            const auto slot = integer(item[@"slot"], "slot", 31);
            const auto mask = integer(item[@"component_mask"], "component_mask", 15);
            if (!mask || !seen.insert(slot).second) reflection_error("Generic interface is invalid");
            output.push_back({static_cast<std::uint8_t>(slot), static_cast<std::uint8_t>(mask),
                              parse_interpolation(string(item[@"interpolation"], "interpolation"))});
        }
    };
    generic(@"generic_inputs", stage.interface.generic_inputs);
    generic(@"generic_outputs", stage.interface.generic_outputs);

    if (entry.stage == Stage::Vertex) {
        if (stage.vertex_depth == gm::VertexDepthConvention::Unspecified ||
            stage.fragment_position != gm::FragmentPositionConvention::Unspecified ||
            stage.interface.position_input_mask || stage.interface.position_output_mask != 15 ||
            stage.interface.color_output_mask || !stage.textures.empty() ||
            stage.interface.generic_inputs.size() != stage.vertex_attributes.size())
            reflection_error("Vertex reflection is outside the supported profile");
        const auto screen_linear = [](const auto& item) {
            return item.interpolation == shader::graphics::Interpolation::ScreenLinear;
        };
        if (!std::all_of(stage.interface.generic_inputs.begin(),
                         stage.interface.generic_inputs.end(), screen_linear) ||
            !std::all_of(stage.interface.generic_outputs.begin(),
                         stage.interface.generic_outputs.end(), screen_linear))
            reflection_error("Vertex generic interfaces require screen-linear metadata");
        for (const auto& option : stage.vertex_attributes) {
            const auto match = std::find_if(stage.interface.generic_inputs.begin(),
                stage.interface.generic_inputs.end(), [&](const auto& item) {
                    return item.slot == option.generic_slot && item.component_mask &&
                           !(item.component_mask & ~7u);
                });
            if (match == stage.interface.generic_inputs.end())
                reflection_error("Vertex input options do not cover reflected inputs");
        }
    } else {
        const bool perspective = std::any_of(stage.interface.generic_inputs.begin(),
            stage.interface.generic_inputs.end(), [](const auto& item) {
                return item.interpolation == shader::graphics::Interpolation::Perspective;
            });
        if (stage.vertex_depth != gm::VertexDepthConvention::Unspecified ||
            !stage.vertex_attributes.empty() || stage.interface.position_output_mask ||
            stage.interface.color_output_mask != 15 || !stage.interface.generic_outputs.empty() ||
            (stage.interface.position_input_mask != 0 && stage.interface.position_input_mask != 8) ||
            (perspective && stage.interface.position_input_mask != 8) ||
            ((stage.interface.position_input_mask == 8 || perspective) &&
             stage.fragment_position != gm::FragmentPositionConvention::MetalPositionWMatchesGuestPositionW) ||
            (stage.interface.position_input_mask == 0 && !perspective &&
             stage.fragment_position != gm::FragmentPositionConvention::Unspecified))
            reflection_error("Fragment reflection is outside the supported profile");
    }
    return stage;
}

std::string identity_key(Stage stage, const Digest& digest_value, std::size_t size) {
    const char* prefix = stage == Stage::Vertex ? "v:" : stage == Stage::Compute ? "c:" : "f:";
    return std::string(prefix) + hex(digest_value) + ':' + std::to_string(size);
}

// The stage's identity across package rebuilds: the program's complete digest
// plus what separates its variants. Manifests name stages by it, so they stay
// valid when the shader source version changes the package variant IDs.
std::string stage_identity_text(const Entry& entry) {
    std::string text = std::to_string(static_cast<unsigned>(entry.stage)) + '.' + hex(entry.full) + '.';
    if (entry.stage == Stage::Vertex)
        text += std::to_string(static_cast<unsigned>(entry.metadata.vertex_depth));
    else if (entry.stage == Stage::Compute)
        text += std::to_string(entry.metadata.compute_local_size[0]) + 'x' +
                std::to_string(entry.metadata.compute_local_size[1]) + 'x' +
                std::to_string(entry.metadata.compute_local_size[2]) + 's' +
                std::to_string(entry.metadata.compute_shared_memory_bytes);
    else text += '0';
    return text;
}

// The compute contract mirrors the offline validator in build_package.py:
// constant buffers are not reflected (queue meta data slot i binds Metal
// buffer i), storage descriptors live in slots 0..7 at runtime-checked
// offsets, and launch specialization (local size, shared memory) is part of
// the artifact identity the manifest variant digest covers.
CompiledGraphicsStage parse_reflection_compute(std::span<const std::byte> bytes,
                                               const Entry& entry,
                                               std::string_view source_version) {
    NSDictionary* root = object(json(bytes, "reflection"), "reflection");
    exact(root, {"schema","source_version","program_key","program_size","program_file",
                 "source_file","source_digest","stage","entry_point","local_size",
                 "shared_memory_bytes","profile","storage_buffers","textures"},
          "reflection");
    const auto text = [&](const char* key) { return string(root[ns(key)], key); };
    if (text("schema") != "armigo-shader-stage-compute-v1" ||
        text("source_version") != source_version ||
        text("program_key") != hex(entry.full) ||
        integer(root[@"program_size"], "program_size", pi::maximum_program_size) != entry.program_size ||
        text("program_file") != "program.bin" || text("source_file") != "source.metal" ||
        text("stage") != "compute" || text("entry_point") != "armigo_compute_main")
        reflection_error("Compute reflection identity or fixed fields do not match the manifest");
    NSDictionary* profile = object(root[@"profile"], "profile");
    exact(profile, {"identity","version"}, "profile");
    if (string(profile[@"identity"], "profile identity") != admitted_v2_profile_identity ||
        integer(profile[@"version"], "profile version", 0xffffffff) !=
            admitted_v3_profile_version)
        reflection_error("Compute reflection host profile is outside the admitted runtime subset");
    NSArray* local = array(root[@"local_size"], "local_size", 3);
    if (local.count != 3) reflection_error("Compute local size must name three dimensions");
    CompiledGraphicsStage stage;
    stage.stage = Stage::Compute;
    stage.entry_point = "armigo_compute_main";
    for (unsigned axis = 0; axis < 3; ++axis) {
        const auto extent = integer(local[axis], "local_size", 65535);
        if (!extent) reflection_error("Compute local size dimensions are at least one");
        stage.compute_local_size[axis] = extent;
    }
    stage.compute_shared_memory_bytes =
        integer(root[@"shared_memory_bytes"], "shared_memory_bytes", 32768);
    std::set<unsigned> metals;
    std::set<std::pair<unsigned, unsigned>> descriptors;
    for (id raw in array(root[@"storage_buffers"], "storage_buffers", 31)) {
        NSDictionary* item = object(raw, "storage buffer");
        exact(item, {"cbuf_slot","descriptor_byte_offset","metal_buffer_index",
                     "access","alignment"}, "storage buffer");
        const auto slot = integer(item[@"cbuf_slot"], "cbuf_slot", 7);
        const auto offset = integer(item[@"descriptor_byte_offset"],
                                    "descriptor_byte_offset", 65524);
        const auto metal = integer(item[@"metal_buffer_index"],
                                   "metal_buffer_index", 30);
        const auto access_text = string(item[@"access"], "access");
        CompiledShaderResourceAccess access;
        if (access_text == "read") access = CompiledShaderResourceAccess::Read;
        else if (access_text == "write") access = CompiledShaderResourceAccess::Write;
        else if (access_text == "read_write")
            access = CompiledShaderResourceAccess::ReadWrite;
        else reflection_error("Storage-buffer access is unsupported");
        // Descriptor fit against the live queue-meta-data size is a dispatch
        // duty; offline identity reserves Metal buffer indices 8..30 (slot i
        // of the meta data owns Metal buffer i).
        if ((offset & 7u) || metal < 8 || integer(item[@"alignment"], "alignment", 4) != 4 ||
            !descriptors.emplace(slot, offset).second || !metals.insert(metal).second)
            reflection_error("Compute storage-buffer reflection is invalid");
        stage.storage_buffers.push_back({static_cast<std::uint8_t>(slot), offset,
            static_cast<std::uint8_t>(metal), access, 4});
    }
    std::set<unsigned> texture_handles, sampler_handles;
    for (id raw in array(root[@"textures"], "textures", 32)) {
        NSDictionary* item = object(raw, "texture entry");
        const bool is_texture = string(item[@"kind"], "texture kind") == "texture";
        const auto slot = integer(item[@"cbuf_slot"], "cbuf_slot", 7);
        const auto offset = integer(item[@"handle_byte_offset"], "handle_byte_offset", 65532);
        if ((offset & 3u)) reflection_error("Compute texture handle offset is unaligned");
        if (is_texture) {
            exact(item, {"cbuf_slot","handle_byte_offset","kind","dimension","array",
                         "multisample","scalar","component_format","normalized_coordinates",
                         "comparison","texture_index"}, "texture");
            const auto index = integer(item[@"texture_index"], "texture_index", 15);
            // The declared shape joins the stage's consumed facts so dispatch
            // re-checks the live TIC exactly like the draw path.
            stage.texture_facts.push_back(parse_texture_descriptor(item,
                static_cast<std::uint8_t>(slot), offset));
            if (!texture_handles.insert(offset).second)
                reflection_error("Compute texture handle is repeated");
            stage.textures.push_back({offset, static_cast<std::uint8_t>(index),
                gm::TextureBinding::no_sampler});
        } else {
            exact(item, {"cbuf_slot","handle_byte_offset","kind","comparison",
                         "normalized_coordinates","sampler_index"}, "sampler");
            const auto index = integer(item[@"sampler_index"], "sampler_index", 15);
            if (!sampler_handles.insert(offset).second)
                reflection_error("Compute sampler handle is repeated");
            for (auto& binding : stage.textures) {
                if (binding.handle_byte_offset == offset)
                    binding.sampler_index = static_cast<std::uint8_t>(index);
            }
        }
    }
    stage.constant_buffers_reflected = false;
    return stage;
}

// Reflection files the package builder writes start with their schema, which
// the chosen parser checks again against the document it parses. Reading it
// from the text spares a second full parse of every stage; any other file
// takes the full parse below.
std::string_view leading_schema(std::span<const std::byte> bytes) {
    constexpr std::string_view opening = R"({"schema":")";
    const std::string_view text(reinterpret_cast<const char*>(bytes.data()), bytes.size());
    if (!text.starts_with(opening)) return {};
    const auto end = text.find_first_of("\"\\", opening.size());
    if (end == std::string_view::npos || text[end] != '"') return {};
    const auto schema = text.substr(opening.size(), end - opening.size());
    constexpr std::array known{std::string_view{"armigo-shader-stage-v1"},
                               std::string_view{"armigo-shader-stage-v2"},
                               std::string_view{"armigo-shader-stage-v3"},
                               std::string_view{"armigo-shader-stage-compute-v1"}};
    return std::find(known.begin(), known.end(), schema) == known.end() ? std::string_view{} : schema;
}

// Dispatches on the reflection schema: v1 keeps its exact contract, while v2
// and v3 load the pipeline plan through their versioned host profiles.
CompiledGraphicsStage parse_reflection_dispatch(std::span<const std::byte> bytes,
                                                const Entry& entry,
                                                std::string_view source_version) {
    std::string schema(leading_schema(bytes));
    if (schema.empty()) {
        NSDictionary* root = object(json(bytes, "reflection"), "reflection");
        if (![root[@"schema"] isKindOfClass:[NSString class]])
            reflection_error("Reflection schema must be a string");
        schema = string(root[@"schema"], "schema");
    }
    if (schema == "armigo-shader-stage-compute-v1")
        return parse_reflection_compute(bytes, entry, source_version);
    if (schema == "armigo-shader-stage-v2")
        return parse_reflection_ported(bytes, entry, source_version, false);
    if (schema == "armigo-shader-stage-v3")
        return parse_reflection_ported(bytes, entry, source_version, true);
    return parse_reflection(bytes, entry, source_version);
}

}  // namespace

struct MetalShaderPackage::Impl {
    fs::path root;
    std::vector<Entry> entries;
    IdentityIndex index;
    std::map<std::string, std::vector<std::size_t>> variants;
    // variants by identity fields, so lookups skip formatting the text key.
    // Filled by lookups on any thread (the command-stream renderer's lookups
    // are not serialized with the NVN driver's), so it has its own lock.
    std::mutex variant_index_mutex;
    std::map<std::tuple<Stage, std::size_t, Digest>, const std::vector<std::size_t>*> variant_index;
    const std::vector<std::size_t>* find_variants(const pi::Identity& identity) {
        const auto key = std::tuple{identity.stage, identity.length, identity.full_digest};
        const std::lock_guard lock(variant_index_mutex);
        if (const auto found = variant_index.find(key); found != variant_index.end()) return found->second;
        const auto found = variants.find(identity_key(identity.stage, identity.full_digest, identity.length));
        const std::vector<std::size_t>* list = found == variants.end() ? nullptr : &found->second;
        variant_index.emplace(key, list);
        return list;
    }
    std::array<std::array<char, 513>, 64> diagnostics{};
    std::size_t diagnostic_count = 0;
    std::size_t cached_library_bytes = 0;
    // Guards each entry's cached stage and the byte count above: warm-up
    // threads load stages by identity while the draw thread loads by lookup.
    std::mutex cache_mutex;
    bool by_digest_built = false;
    std::string chain;
    std::string content;
    // Lazy packages parse an entry's reflection on first use, so opening does
    // not touch the stage directories of stages the title never draws with.
    // prepared[i] is published with release once entry i has its metadata,
    // unsupported text and library path; readers acquire it through prepare().
    std::string source;
    bool defer_unsupported = false;
    std::unique_ptr<std::atomic<bool>[]> prepared;
    // Entries prepare concurrently; one stripe serializes the threads that
    // want the same entry.
    std::array<std::mutex, 64> prepare_stripes;
    std::atomic<std::size_t> reflection_total{0};
    // Opening starts a few background threads that prepare every entry, so
    // the parsing overlaps the guest's startup instead of landing on the
    // first draws that select an entry.
    std::atomic<std::size_t> prefetch_next{0};
    std::atomic<bool> prefetch_stop{false};
    // Entries the title is known to select first, set once by prefetch_first
    // and read by the workers after `priority_ready`; the workers take them
    // before the rest. `prefetch_claimed[i]` is set by the worker that takes
    // entry i, so an entry in both lists is prepared once.
    std::vector<std::size_t> priority;
    std::atomic<bool> priority_ready{false};
    std::atomic<bool> priority_offered{false};
    std::atomic<std::size_t> priority_next{0};
    std::unique_ptr<std::atomic<bool>[]> prefetch_claimed;
    std::vector<std::thread> prefetchers;
    // Prefetch outcome: entries prepared and failed by the background threads,
    // the first failure's text, and the workers still running (the last one
    // prints the summary line).
    std::atomic<std::size_t> prefetch_prepared{0};
    std::atomic<std::size_t> prefetch_failed{0};
    std::atomic<unsigned> prefetch_active{0};
    std::chrono::steady_clock::time_point prefetch_started;
    std::mutex prefetch_join_mutex;
    std::mutex prefetch_failure_mutex;
    std::string prefetch_first_failure;
    // A stage that failed to prepare keeps its error, so later selections of
    // it fail at once instead of reading its files again. Guarded by the
    // entry's stripe.
    std::vector<std::optional<MetalShaderPackageError>> prepare_failures;
    // Stage and complete digest to the entries that share them, built on the
    // first lookup by identity text (the variant suffix needs reflection).
    std::unordered_map<std::string, std::vector<std::size_t>> by_digest;
    // Guest code last matched at each stage address. Comparing cached program
    // code with memory lets unchanged code keep its identity
    // without rehashing; any difference repeats the full lookup.
    // With write tracking, host bytes armed before the copy and untouched
    // since (stamp) still equal the copy, so the comparison is skipped.
    struct MatchedCode {
        pi::Identity identity;
        std::vector<std::byte> code;
        const void* host = nullptr;
        std::uint64_t stamp = 0;
    };
    std::map<std::pair<Stage, std::uint64_t>, MatchedCode> matched;
    std::vector<std::byte> compared;

    Impl(fs::path path, std::vector<Entry> loaded, IdentityIndex built,
         std::map<std::string, std::vector<std::size_t>> mapped)
        : root(std::move(path)), entries(std::move(loaded)), index(std::move(built)),
          variants(std::move(mapped)) {}

    graphics::CommandResult error(std::string_view text,
                                  graphics::CommandStatus status = graphics::CommandStatus::Unsupported) noexcept {
        static constexpr std::string_view exhausted =
            "Metal shader package diagnostic retention limit reached";
        // A defect that repeats on every draw keeps its one slot.
        for (std::size_t i = 0; i != diagnostic_count; ++i)
            if (text.size() < diagnostics[i].size() && text == std::string_view(diagnostics[i].data()))
                return {status, std::string_view(diagnostics[i].data(), text.size())};
        if (diagnostic_count == diagnostics.size()) return {status, exhausted};
        auto& slot = diagnostics[diagnostic_count++];
        const auto length = std::min<std::size_t>(text.size(), slot.size() - 1);
        if (length) std::memcpy(slot.data(), text.data(), length);
        slot[length] = '\0';
        return {status, std::string_view(slot.data(), length)};
    }

    ~Impl() {
        prefetch_stop.store(true, std::memory_order_relaxed);
        join_prefetch();
    }

    void join_prefetch() {
        const std::lock_guard lock(prefetch_join_mutex);
        for (auto& worker : prefetchers)
            if (worker.joinable()) worker.join();
    }

    // Prefetch is optional: a thread that cannot be created leaves its share
    // of the entries to the lookups that select them.
    // The first lookup by identity text builds the digest index; the entries'
    // stage and digest come from the manifest, so no stage file is read.
    void build_digest_index() {
        const std::lock_guard lock(cache_mutex);
        if (by_digest_built) return;
        for (std::size_t index = 0; index != entries.size(); ++index) {
            const auto& entry = entries[index];
            by_digest[std::to_string(static_cast<unsigned>(entry.stage)) + '.' + hex(entry.full) + '.']
                .push_back(index);
        }
        by_digest_built = true;
    }

    // The next entry for a prefetch worker: the title's first-selected entries
    // once they are known, then the rest in package order.
    std::optional<std::size_t> next_prefetch() {
        if (priority_ready.load(std::memory_order_acquire)) {
            for (;;) {
                const auto position = priority_next.fetch_add(1, std::memory_order_relaxed);
                if (position >= priority.size()) break;
                if (!prefetch_claimed[priority[position]].exchange(true, std::memory_order_relaxed))
                    return priority[position];
            }
        }
        for (;;) {
            const auto index = prefetch_next.fetch_add(1, std::memory_order_relaxed);
            if (index >= entries.size()) return std::nullopt;
            if (!prefetch_claimed[index].exchange(true, std::memory_order_relaxed)) return index;
        }
    }

    void start_prefetch(unsigned workers) {
        prefetch_claimed = std::make_unique<std::atomic<bool>[]>(entries.size());
        prefetch_started = std::chrono::steady_clock::now();
        prefetch_active.store(workers, std::memory_order_relaxed);
        for (unsigned i = 0; i != workers; ++i) {
            try {
                prefetchers.emplace_back([this] { prefetch_worker(); });
            } catch (const std::system_error&) {
                if (prefetch_active.fetch_sub(1, std::memory_order_acq_rel) == 1) report_prefetch();
            }
        }
    }

    void prefetch_worker() {
        pthread_setname_np("armigo.shader.prefetch");
        pthread_set_qos_class_self_np(QOS_CLASS_UTILITY, 0);
        for (;;) {
            if (prefetch_stop.load(std::memory_order_relaxed)) break;
            const auto next = next_prefetch();
            if (!next) break;
            const auto index = *next;
            // A defect is reported again by the lookup that selects the entry.
            try {
                prepare(index);
                prefetch_prepared.fetch_add(1, std::memory_order_relaxed);
            } catch (...) {
                std::string text = "unexpected failure";
                try { throw; }
                catch (const Failure& failure) { text = failure.error.detail; }
                catch (const std::exception& error) { text = error.what(); }
                catch (...) {}
                if (prefetch_failed.fetch_add(1, std::memory_order_relaxed) == 0) {
                    const std::lock_guard lock(prefetch_failure_mutex);
                    prefetch_first_failure = std::move(text);
                }
            }
        }
        if (prefetch_active.fetch_sub(1, std::memory_order_acq_rel) == 1) report_prefetch();
    }

    // One summary line per package, like the pipeline warm-up's.
    void report_prefetch() {
        if (prefetch_stop.load(std::memory_order_relaxed)) return;
        const auto seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - prefetch_started).count();
        const auto failed = prefetch_failed.load(std::memory_order_relaxed);
        std::string first;
        if (failed) {
            const std::lock_guard lock(prefetch_failure_mutex);
            first = prefetch_first_failure;
        }
        armigo::diag::write(armigo::diag::Level::info,  "Armigo shader package: prepared %zu of %zu stages in %.1f s%s%s\n",
                     prefetch_prepared.load(std::memory_order_relaxed), entries.size(), seconds,
                     failed ? "; first failure: " : "", first.c_str());
    }

    // Preparation runs on prefetch threads and on the guest threads that
    // select a stage first; neither has an autorelease pool of its own, so
    // each preparation drains the Foundation objects it creates.
    void prepare(std::size_t index) {
        if (prepared[index].load(std::memory_order_acquire)) return;
        std::exception_ptr error;
        @autoreleasepool {
            try { prepare_entry(index); } catch (...) { error = std::current_exception(); }
        }
        if (error) std::rethrow_exception(error);
    }

    // A failure carries the stage that raised it, since it surfaces from a
    // background thread or a draw that does not know which entry it selected.
    MetalShaderPackageError named_failure(std::size_t index, MetalShaderPackageError::Code code,
                                          const std::string& detail) const {
        const auto& entry = entries[index];
        return {code, "stage " + entry.variant + " (" + std::string(stage_name(entry.stage)) + "): " + detail};
    }

    void prepare_entry(std::size_t index) {
        const std::lock_guard lock(prepare_stripes[index % prepare_stripes.size()]);
        if (prepared[index].load(std::memory_order_relaxed)) return;
        if (const auto& failed = prepare_failures[index]) throw Failure{*failed};
        try {
            prepare_files(index);
        } catch (const Failure& failure) {
            prepare_failures[index] = named_failure(index, failure.error.code, failure.error.detail);
            throw Failure{*prepare_failures[index]};
        } catch (const std::exception& error) {
            prepare_failures[index] = named_failure(index, MetalShaderPackageError::Code::Input, error.what());
            throw Failure{*prepare_failures[index]};
        }
    }

    // The program artifact is read when its stage is prepared (a lazy package
    // indexes the manifest's identities without reading it) and must be the
    // program the manifest records.
    void check_program_artifact(const Entry& entry) const {
        const auto program_path = package_file(root, entry.stage_directory, "program.bin");
        if (fs::file_size(program_path) != entry.program_size)
            fail(MetalShaderPackageError::Code::Integrity,
                 "Program file size differs from the manifest");
        const auto program = read_file(program_path, pi::maximum_program_size, "program");
        if (program.size() != entry.program_size || program.size() < pi::prefix_size ||
            digest(program) != entry.full ||
            digest(std::span(program).first(pi::prefix_size)) != entry.prefix)
            fail(MetalShaderPackageError::Code::Integrity,
                 "Program size, full digest or prefix digest differs from the manifest");
        if (const auto error = check_program(entry.stage, program))
            fail(MetalShaderPackageError::Code::Identity,
                 std::string("Program identity rejected the program: ") + error->message());
    }

    void prepare_files(std::size_t index) {
        auto& entry = entries[index];
        entry.stage_directory = fs::canonical(entry.stage_directory);
        if (!inside(entry.stage_directory, root))
            fail(MetalShaderPackageError::Code::Input, "Stage directory escapes the package root");
        std::set<std::string> files;
        for (const auto& child : fs::directory_iterator(entry.stage_directory))
            files.insert(child.path().filename().string());
        if (files != std::set<std::string>{"program.bin","reflection.json","shader.metallib"})
            fail(MetalShaderPackageError::Code::Input, "Stage directory has missing or extra files");
        const auto reflection_path = package_file(root, entry.stage_directory, "reflection.json");
        const auto reflection_size = fs::file_size(reflection_path);
        if (reflection_size > reflection_limit ||
            reflection_total.fetch_add(static_cast<std::size_t>(reflection_size)) + reflection_size >
                total_reflection_limit) {
            if (reflection_size <= reflection_limit)
                reflection_total.fetch_sub(static_cast<std::size_t>(reflection_size));
            fail(MetalShaderPackageError::Code::Input,
                 "Reflection metadata exceeds its aggregate policy");
        }
        try {
            check_program_artifact(entry);
            auto library_path = package_file(root, entry.stage_directory, "shader.metallib");
            auto reflection_bytes = read_file(reflection_path, reflection_limit, "reflection");
            if (digest(reflection_bytes) != entry.reflection_digest)
                fail(MetalShaderPackageError::Code::Integrity,
                     "Reflection digest differs from the manifest");
            try {
                entry.metadata = parse_reflection_dispatch(reflection_bytes, entry, source);
            } catch (const Failure& failure) {
                if (!defer_unsupported || failure.error.code != MetalShaderPackageError::Code::Reflection)
                    throw;
                entry.unsupported = failure.error.detail;
            }
            entry.library_path = std::move(library_path);
        } catch (...) {
            reflection_total.fetch_sub(static_cast<std::size_t>(reflection_size));
            throw;
        }
        prepared[index].store(true, std::memory_order_release);
    }

    std::shared_ptr<const CompiledGraphicsStage> load(
        Entry& entry, const graphics::RenderTargetPlan* targets = nullptr, bool unspecialized = false) {
        prepare(static_cast<std::size_t>(&entry - entries.data()));
        std::uint16_t output_key = 0;
        auto loaded = entry.metadata;
        loaded.guest_code_bytes = std::max<std::uint64_t>(entry.program_size, pi::prefix_size);
        if (loaded.fragment_output_types_from_render_targets && !unspecialized) {
            if (!targets) fail(MetalShaderPackageError::Code::Compatibility,
                               "Adaptive fragment outputs require render-target state");
            for (auto& output : loaded.color_outputs) {
                const auto& attachment = targets->colors[output.location];
                // Maxwell discards an export to an unbound logical target.
                // Keep its raw uint32 class so Metal can specialize the
                // declaration while the pipeline applies a zero write mask.
                if (attachment) {
                    switch (graphics::texture_numeric_class(attachment->texture.format)) {
                    case graphics::TextureNumericClass::Float: output.scalar = ColorOutputScalar::f32; break;
                    case graphics::TextureNumericClass::Uint: output.scalar = ColorOutputScalar::u32; break;
                    case graphics::TextureNumericClass::Sint: output.scalar = ColorOutputScalar::s32; break;
                    }
                }
                output_key |= static_cast<std::uint16_t>(output.scalar) << (2 * output.location);
            }
            const auto found = entry.output_specializations.find(output_key);
            if (found != entry.output_specializations.end()) return found->second;
        } else {
            const std::lock_guard lock(cache_mutex);
            if (entry.cached) return entry.cached;
        }
        const auto current_path = fs::canonical(entry.library_path);
        if (current_path != entry.library_path || !inside(current_path, root))
            fail(MetalShaderPackageError::Code::Input,
                 "Metal library path changed after package validation");
        const auto actual_size = fs::file_size(current_path);
        std::size_t cached_bytes = 0;
        {
            const std::lock_guard lock(cache_mutex);
            cached_bytes = cached_library_bytes;
        }
        if (actual_size != entry.library_size || actual_size > library_limit ||
            actual_size > library_cache_limit - cached_bytes)
            fail(MetalShaderPackageError::Code::Input,
                 "Metal library exceeds its declared size or package cache budget");
        auto bytes = read_file(entry.library_path, library_limit, "Metal library");
        if (bytes.size() != entry.library_size || digest(bytes) != entry.library_digest)
            fail(MetalShaderPackageError::Code::Integrity,
                 "Metal library size or digest differs from the manifest");
        loaded.metallib = std::move(bytes);
        loaded.identity = stage_identity_text(entry);
        auto result = std::make_shared<const CompiledGraphicsStage>(std::move(loaded));
        const std::lock_guard lock(cache_mutex);
        if (entry.metadata.fragment_output_types_from_render_targets && !unspecialized)
            entry.output_specializations.emplace(output_key, result);
        else if (entry.cached) return entry.cached;   // another thread loaded it meanwhile
        else entry.cached = result;
        cached_library_bytes += entry.library_size;
        return result;
    }

    pi::LookupResult identify(Stage stage, std::uint64_t address, const pi::MemoryReader& reads,
                              const pi::HashProvider& hashes,
                              const graphics::GpuAddressSpace* memory = nullptr) {
        const auto key = std::pair{stage, address};
        // Stamp and arm the guest bytes before they are copied.
        const auto arm = [&](std::size_t size, MatchedCode& code) {
            code.host = nullptr;
            code.stamp = 0;
            if (!memory) return;
            const auto mappings = memory->mappings();
            const auto after = std::upper_bound(mappings.begin(), mappings.end(), address,
                [](std::uint64_t value, const graphics::Mapping& mapping) { return value < mapping.gpu_base; });
            if (after == mappings.begin()) return;
            const auto& mapping = *std::prev(after);
            const auto offset = address - mapping.gpu_base;
            if (offset >= mapping.mapped_size || size > mapping.mapped_size - offset ||
                offset > mapping.backing_size - mapping.buffer_offset ||
                size > mapping.backing_size - mapping.buffer_offset - offset)
                return;
            const auto* host = reinterpret_cast<const void*>(mapping.guest_address + mapping.buffer_offset + offset);
            const auto stamp = armigo::memory::WriteTracker::now();
            if (!armigo::memory::WriteTracker::protect_checked(host, size)) return;
            code.host = host;
            code.stamp = stamp;
        };
        if (const auto found = matched.find(key); found != matched.end()) {
            auto& entry = found->second;
            if (entry.host && !armigo::memory::WriteTracker::written_since(entry.host, entry.code.size(), entry.stamp))
                return entry.identity;
            arm(entry.code.size(), entry);
            compared.resize(entry.code.size());
            if (reads.function(reads.context, address, compared) && compared == entry.code)
                return entry.identity;
            entry.host = nullptr;
        }
        auto result = lookup_identity(index, stage, address, reads, hashes);
        if (const auto* identity = std::get_if<pi::Identity>(&result)) {
            MatchedCode entry{*identity, std::vector<std::byte>(std::max(identity->length, pi::prefix_size))};
            arm(entry.code.size(), entry);
            if (reads.function(reads.context, address, entry.code))
                matched.insert_or_assign(key, std::move(entry));
            else matched.erase(key);
        }
        return result;
    }

    Entry& select(const pi::Identity& identity, const graphics::RenderStop& stop) {
        const auto* found = find_variants(identity);
        if (!found) fail(MetalShaderPackageError::Code::Identity, "Matched identity has no package variant");
        std::vector<std::size_t> matches;
        const std::string* deferred = nullptr;
        // The first mismatch of each candidate, so a miss names what differs.
        std::string reasons;
        const auto note = [&](std::size_t candidate, const std::string& reason) {
            if (reasons.size() < 600)
                reasons += (reasons.empty() ? "" : "; ") + std::string("variant ") +
                           std::to_string(candidate) + ": " + reason;
        };
        for (auto index_value : *found) {
            prepare(index_value);
            auto& entry = entries[index_value];
            if (!entry.unsupported.empty()) {
                deferred = &entry.unsupported;
                continue;
            }
            bool match = true;
            if (entry.stage == Stage::Vertex) {
                gm::VertexDepthConvention depth = gm::VertexDepthConvention::Unspecified;
                if (stop.registers[0x35f] == 0) depth = gm::VertexDepthConvention::NegativeOneToOne;
                else if (stop.registers[0x35f] == 1) depth = gm::VertexDepthConvention::ZeroToOne;
                else fail(MetalShaderPackageError::Code::Compatibility,
                          "Guest clip-depth convention is unsupported");
                if (entry.metadata.vertex_depth != depth) {
                    match = false;
                    note(index_value, "clip-depth convention " +
                         std::to_string(static_cast<unsigned>(entry.metadata.vertex_depth)) + " vs guest " +
                         std::to_string(static_cast<unsigned>(depth)));
                }
                for (const auto& attribute : entry.metadata.vertex_attributes) {
                    if (!match) break;
                    const auto planned = graphics::plan_vertex_float3(stop, attribute.generic_slot);
                    if (!std::holds_alternative<graphics::VertexFloat3Plan>(planned)) {
                        match = false;
                        note(index_value, "float3 attribute at generic slot " +
                             std::to_string(attribute.generic_slot) + " does not plan as float3");
                    }
                }
                for (const auto& input : entry.metadata.vertex_class_inputs) {
                    if (!match) break;
                    const auto planned = graphics::plan_vertex_attribute(stop, input.location);
                    if (!std::holds_alternative<graphics::VertexAttributePlan>(planned)) {
                        match = false;
                        note(index_value, "attribute " + std::to_string(input.location) +
                             " has no runtime plan");
                    } else if (const auto& actual = std::get<graphics::VertexAttributePlan>(planned).format;
                               !graphics::vertex_format_as_class(actual, input.input_class)) {
                        match = false;
                        note(index_value, "attribute " + std::to_string(input.location) + " class " +
                             std::to_string(static_cast<unsigned>(input.input_class)) + " vs guest " +
                             std::to_string(static_cast<unsigned>(actual.format_class)) + " format " +
                             std::to_string(static_cast<unsigned>(actual.format)));
                    }
                }
            }
            if (match) matches.push_back(index_value);
        }
        if (matches.empty() && deferred)
            fail(MetalShaderPackageError::Code::Reflection,
                 "Shader package entry is outside the runtime profile: " + *deferred);
        if (matches.empty()) fail(MetalShaderPackageError::Code::Compatibility,
                                  "No compiled variant matches current graphics state (" + reasons + ")");
        if (matches.size() != 1) fail(MetalShaderPackageError::Code::Compatibility,
                                      "Multiple compiled variants match current graphics state");
        return entries[matches.front()];
    }

    // Selects the one compute variant whose launch specialization matches the
    // decoded queue meta data exactly. A miss names the program and both
    // sides' sizes: never silently skipped.
    Entry& select_compute(const pi::Identity& identity, const graphics::ComputeLaunch& launch) {
        const auto* found = find_variants(identity);
        if (!found) fail(MetalShaderPackageError::Code::Identity, "Matched identity has no package variant");
        const auto request = launch.block;
        const auto shared = launch.shared_memory_size;
        std::vector<std::size_t> matches;
        std::string available;
        for (auto index_value : *found) {
            prepare(index_value);
            auto& entry = entries[index_value];
            if (!entry.unsupported.empty()) {
                available += (available.empty() ? "" : ", ") +
                    std::string("deferred: ") + entry.unsupported;
                continue;
            }
            // The launch's shared allocation comes in hardware granules and may
            // exceed what the program declares; any variant compiled for at most
            // the allocated size is valid, and the largest such one is used.
            if (entry.metadata.compute_local_size == request &&
                entry.metadata.compute_shared_memory_bytes <= shared) {
                matches.push_back(index_value);
            }
            available += (available.empty() ? "" : ", ") +
                std::to_string(entry.metadata.compute_local_size[0]) + "x" +
                std::to_string(entry.metadata.compute_local_size[1]) + "x" +
                std::to_string(entry.metadata.compute_local_size[2]) + " shared " +
                std::to_string(entry.metadata.compute_shared_memory_bytes);
        }
        if (matches.empty()) {
            fail(MetalShaderPackageError::Code::Compatibility,
                 "No compiled compute variant matches program offset " +
                 std::to_string(launch.program_offset) + " with local size " +
                 std::to_string(request[0]) + "x" + std::to_string(request[1]) + "x" +
                 std::to_string(request[2]) + " and shared memory " +
                 std::to_string(shared) + " bytes (package offers: " + available + ")");
        }
        const auto best = std::max_element(matches.begin(), matches.end(), [&](std::size_t a, std::size_t b) {
            return entries[a].metadata.compute_shared_memory_bytes < entries[b].metadata.compute_shared_memory_bytes;
        });
        return entries[*best];
    }

    CompiledComputeStageResult lookup_compute(const graphics::RenderStop& snapshot,
                                              const graphics::GpuAddressSpace& memory) {
        if (snapshot.kind != graphics::RenderKind::Compute ||
            !snapshot.has_compute_descriptor || !snapshot.compute ||
            !snapshot.programs[0].enabled)
            return error("Compute lookup requires a compute stop with a queue meta data descriptor");
        ReaderContext reader{&memory};
        const pi::MemoryReader reads{read_guest, &reader};
        const pi::HashProvider hashes{hash_bytes, nullptr};
        const auto find = [&](std::uint64_t address) -> pi::Identity {
            auto result = identify(Stage::Compute, address, reads, hashes);
            if (const auto* failure = std::get_if<pi::Error>(&result)) {
                std::string detail = std::string("Shader package compute lookup failed: ") +
                    failure->message();
                if (failure->code == pi::Error::Code::Miss) {
                    std::array<std::byte, pi::prefix_size> prefix{};
                    if (read_guest(&reader, address, prefix)) {
                        const auto prefix_hash = hash_bytes(nullptr, prefix);
                        if (prefix_hash) detail += " (prefix-sha256 " + hex(*prefix_hash) + ")";
                    }
                }
                throw error(std::move(detail),
                    failure->code == pi::Error::Code::ReadFailure ||
                        failure->code == pi::Error::Code::AddressOverflow
                        ? graphics::CommandStatus::MemoryFault
                        : graphics::CommandStatus::Unsupported);
            }
            return std::get<pi::Identity>(result);
        };
        Entry& entry = select_compute(find(snapshot.programs[0].address), *snapshot.compute);
        return load(entry);
    }
};

MetalShaderPackage::MetalShaderPackage(std::unique_ptr<Impl> impl) noexcept
    : impl_(std::move(impl)) {}
MetalShaderPackage::~MetalShaderPackage() = default;

void MetalShaderPackage::prefetch_first(std::span<const std::string> identities) noexcept {
    try {
        if (impl_->prefetchers.empty() || impl_->priority_offered.exchange(true)) return;
        impl_->build_digest_index();
        std::vector<char> listed(impl_->entries.size(), 0);
        for (const auto& identity : identities) {
            const auto split = identity.rfind('.');
            if (split == std::string::npos) continue;
            const auto found = impl_->by_digest.find(identity.substr(0, split + 1));
            if (found == impl_->by_digest.end()) continue;
            for (const auto index : found->second) {
                if (listed[index]) continue;
                listed[index] = 1;
                impl_->priority.push_back(index);
            }
        }
        impl_->priority_ready.store(true, std::memory_order_release);
    } catch (...) {
        // The workers keep package order.
    }
}

MetalShaderPackagePrefetch MetalShaderPackage::finish_prefetch() noexcept {
    impl_->join_prefetch();
    MetalShaderPackagePrefetch result;
    result.prepared = impl_->prefetch_prepared.load(std::memory_order_relaxed);
    result.failed = impl_->prefetch_failed.load(std::memory_order_relaxed);
    if (result.failed) {
        const std::lock_guard lock(impl_->prefetch_failure_mutex);
        result.first_failure = impl_->prefetch_first_failure;
    }
    return result;
}

SourceStageResult make_source_stage(shader_input::Stage stage_kind, std::span<const std::byte> program,
                                   std::span<const std::byte> reflection, std::string source,
                                   std::string source_file) {
    try {
        @autoreleasepool {
            if (reflection.size() > reflection_limit)
                fail(MetalShaderPackageError::Code::Input, "Reflection metadata exceeds its size policy");
            if (program.size() < pi::prefix_size || program.size() > pi::maximum_program_size)
                fail(MetalShaderPackageError::Code::Integrity, "Program size is outside the identity bounds");
            if (const auto error = check_program(stage_kind, program))
                fail(MetalShaderPackageError::Code::Identity,
                     std::string("Program identity rejected the program: ") + error->message());
            Entry entry;
            entry.stage = stage_kind;
            entry.full = digest(program);
            entry.prefix = digest(program.first(pi::prefix_size));
            entry.program_size = static_cast<std::uint32_t>(program.size());
            entry.variant = "runtime";
            NSDictionary* root = object(json(reflection, "reflection"), "reflection");
            const auto source_digest = string(root[@"source_digest"], "source_digest");
            if (!lower_hex(source_digest))
                fail(MetalShaderPackageError::Code::Reflection, "Reflection source digest is not a SHA-256");
            if (source.empty() == source_file.empty())
                fail(MetalShaderPackageError::Code::Input, "A source stage needs its source or its source file");
            const std::span<const std::byte> source_bytes{reinterpret_cast<const std::byte*>(source.data()),
                                                          source.size()};
            if (!source.empty() && source_digest != hex(digest(source_bytes)))
                fail(MetalShaderPackageError::Code::Integrity, "Metal source digest differs from the reflection");
            entry.metadata = parse_reflection_dispatch(reflection, entry, armigo::shader::source_version);
            auto loaded = entry.metadata;
            loaded.guest_code_bytes = std::max<std::uint64_t>(entry.program_size, pi::prefix_size);
            loaded.identity = stage_identity_text(entry);
            loaded.msl_source = std::move(source);
            loaded.msl_source_file = std::move(source_file);
            loaded.msl_source_digest = source_digest;
            return std::make_shared<const CompiledGraphicsStage>(std::move(loaded));
        }
    } catch (const Failure& failure) {
        return failure.error;
    } catch (const std::bad_alloc&) {
        return MetalShaderPackageError{MetalShaderPackageError::Code::Allocation, "Source stage allocation failed"};
    } catch (const std::exception& error) {
        return MetalShaderPackageError{MetalShaderPackageError::Code::Input, error.what()};
    }
}

std::variant<std::string, MetalShaderPackageError> stage_source(const CompiledGraphicsStage& stage) {
    if (!stage.msl_source.empty()) return stage.msl_source;
    try {
        if (stage.msl_source_file.empty())
            fail(MetalShaderPackageError::Code::Input, "The stage has no Metal source");
        const auto bytes = read_file(stage.msl_source_file, library_limit, "Metal source");
        if (hex(digest(bytes)) != stage.msl_source_digest) {
            // A damaged run-time cache entry (<key>/source.metal) is removed, so
            // the next run translates the stage again instead of failing on it.
            const fs::path file(stage.msl_source_file);
            if (file.filename() == "source.metal" && file.parent_path().filename().string().size() == 64) {
                std::error_code ignored;
                fs::remove_all(file.parent_path(), ignored);
            }
            fail(MetalShaderPackageError::Code::Integrity, "Metal source file differs from its reflection digest");
        }
        return std::string(reinterpret_cast<const char*>(bytes.data()), bytes.size());
    } catch (const Failure& failure) {
        return failure.error;
    } catch (const std::exception& error) {
        return MetalShaderPackageError{MetalShaderPackageError::Code::Input, error.what()};
    }
}

MetalShaderPackage::OpenResult MetalShaderPackage::open(const fs::path& directory,
                                                        Options options) {
    @autoreleasepool {
    try {
        if (options.defer_unsupported_entries && !options.allow_development_subset)
            fail(MetalShaderPackageError::Code::InvalidOptions,
                 "Deferring unsupported entries requires a development-subset package");
        if (!lower_hex(options.expected_chain_id))
            fail(MetalShaderPackageError::Code::InvalidOptions,
                 "Trusted expected_chain_id is required and must be lowercase SHA-256");
        const auto root = fs::canonical(directory);
        if (!fs::is_directory(root)) fail(MetalShaderPackageError::Code::Input,
                                          "Shader package root is not a directory");
        auto manifest_bytes = read_file(package_file(root, root, "manifest.json"),
                                        manifest_limit, "package manifest");
        const auto manifest_digest = hex(digest(manifest_bytes));
        NSDictionary* manifest = object(json(manifest_bytes, "package manifest"), "package manifest");
        exact(manifest, {"schema","source_version","chain_id","platform","profile_text",
                         "complete_inventory","runnable","entries"}, "package manifest");
        const auto source = string(manifest[@"source_version"], "source_version");
        const auto chain = string(manifest[@"chain_id"], "chain_id");
        const auto platform = string(manifest[@"platform"], "platform");
        const auto profile_text = string(manifest[@"profile_text"], "profile_text");
        if (string(manifest[@"schema"], "schema") != "armigo-metal-package-v1" ||
            !lower_hex(source) || source != shader::source_version ||
            !lower_hex(chain) || chain != options.expected_chain_id || platform != platform_name)
            fail(MetalShaderPackageError::Code::Compatibility,
                 "Package schema, source version, chain or platform is incompatible");
        if (boolean(manifest[@"complete_inventory"], "complete_inventory") ||
            boolean(manifest[@"runnable"], "runnable"))
            fail(MetalShaderPackageError::Code::Manifest,
                 "Current packages must not claim complete or runnable inventory");
        if (!options.allow_development_subset)
            fail(MetalShaderPackageError::Code::Compatibility,
                 "Incomplete shader package requires explicit development-subset mode");

        if (!std::all_of(profile_text.begin(), profile_text.end(),
                         [](unsigned char value) { return value < 0x80; }))
            fail(MetalShaderPackageError::Code::Compatibility,
                 "Canonical package profile is restricted to ASCII JSON text");
        auto profile_bytes = std::as_bytes(std::span(profile_text.data(), profile_text.size()));
        id profile_json = json(profile_bytes, "profile_text");
        NSDictionary* profile = object(profile_json, "profile_text");
        NSError* canonical_error = nil;
        NSData* canonical = [NSJSONSerialization dataWithJSONObject:profile
            options:(NSJSONWritingSortedKeys | NSJSONWritingWithoutEscapingSlashes)
            error:&canonical_error];
        if (!canonical || canonical_error)
            fail(MetalShaderPackageError::Code::Compatibility,
                 "Profile text cannot be canonicalized");
        std::string canonical_text(static_cast<const char*>(canonical.bytes), canonical.length);
        if (canonical_text != profile_text ||
            string(profile[@"source_version"], "profile source_version") != source ||
            string(profile[@"platform"], "profile platform") != platform)
            fail(MetalShaderPackageError::Code::Compatibility,
                 "Profile text is noncanonical or disagrees with package compatibility fields");
        std::string chain_input = "armigo-metal-chain-v1\n" + profile_text;
        if (hex(digest(std::as_bytes(std::span(chain_input.data(), chain_input.size())))) != chain)
            fail(MetalShaderPackageError::Code::Integrity,
                 "Package chain ID does not match canonical profile text");

        NSArray* raw_entries = array(manifest[@"entries"], "entries", entry_limit);
        std::vector<Entry> entries;
        entries.reserve(raw_entries.count);
        std::set<std::string> variants_seen;
        std::size_t declared_program_total = 0;
        for (id raw in raw_entries) {
            NSDictionary* item = object(raw, "entry");
            exact(item, {"stage","program_key","prefix_key","program_size","directory",
                         "reflection_digest","library_digest","library_size","variant_id"}, "entry");
            Entry entry;
            entry.stage = parse_stage(string(item[@"stage"], "stage"));
            entry.full = parse_digest(string(item[@"program_key"], "program_key"), "program_key");
            entry.prefix = parse_digest(string(item[@"prefix_key"], "prefix_key"), "prefix_key");
            entry.program_size = integer(item[@"program_size"], "program_size", pi::maximum_program_size);
            if (entry.program_size < pi::prefix_size ||
                entry.program_size > total_program_limit - declared_program_total)
                fail(MetalShaderPackageError::Code::Manifest,
                     "Declared program bytes exceed the aggregate policy");
            declared_program_total += entry.program_size;
            entry.variant = string(item[@"variant_id"], "variant_id");
            if (!lower_hex(entry.variant) || !variants_seen.insert(entry.variant).second)
                fail(MetalShaderPackageError::Code::Manifest, "Variant ID is invalid or duplicated");
            const auto relative = string(item[@"directory"], "directory");
            if (relative != "stages/" + entry.variant)
                fail(MetalShaderPackageError::Code::Manifest, "Entry directory is not canonical");
            entry.library_digest = parse_digest(string(item[@"library_digest"], "library_digest"),
                                                "library_digest");
            entry.library_size = integer(item[@"library_size"], "library_size", library_limit);
            if (!entry.library_size) fail(MetalShaderPackageError::Code::Manifest,
                                          "Metal library size must be nonzero");
            entry.reflection_digest = parse_digest(
                string(item[@"reflection_digest"], "reflection_digest"), "reflection_digest");
            if (options.lazy_entries) {
                // The directory is resolved and checked when the stage is prepared.
                entry.stage_directory = root / relative;
            } else {
                entry.stage_directory = fs::canonical(root / relative);
                if (!inside(entry.stage_directory, root) || !fs::is_directory(entry.stage_directory))
                    fail(MetalShaderPackageError::Code::Input, "Stage directory escapes the package root");
            }
            entries.push_back(std::move(entry));
        }

        if (!options.lazy_entries) {
            // Preflight every metadata file before allocating or reading any of them.
            std::size_t reflection_total = 0;
            for (auto& entry : entries) {
                std::set<std::string> files;
                for (const auto& child : fs::directory_iterator(entry.stage_directory))
                    files.insert(child.path().filename().string());
                if (files != std::set<std::string>{"program.bin","reflection.json","shader.metallib"})
                    fail(MetalShaderPackageError::Code::Input, "Stage directory has missing or extra files");
                const auto reflection_path = package_file(root, entry.stage_directory, "reflection.json");
                const auto reflection_size = fs::file_size(reflection_path);
                if (reflection_size > reflection_limit ||
                    reflection_size > total_reflection_limit - reflection_total)
                    fail(MetalShaderPackageError::Code::Input,
                         "Reflection metadata exceeds its aggregate policy");
                reflection_total += static_cast<std::size_t>(reflection_size);
                entry.library_path = package_file(root, entry.stage_directory, "shader.metallib");
            }
            // Accept every reflection field before reading any program artifact.
            for (auto& entry : entries) {
                auto reflection_bytes = read_file(package_file(root, entry.stage_directory, "reflection.json"),
                                                  reflection_limit, "reflection");
                if (digest(reflection_bytes) != entry.reflection_digest)
                    fail(MetalShaderPackageError::Code::Integrity,
                         "Reflection digest differs from the manifest");
                try {
                    entry.metadata = parse_reflection_dispatch(reflection_bytes, entry, source);
                } catch (const Failure& failure) {
                    if (!options.defer_unsupported_entries ||
                        failure.error.code != MetalShaderPackageError::Code::Reflection)
                        throw;
                    entry.unsupported = failure.error.detail;
                }
            }
        }
        struct Program { Stage stage; std::vector<std::byte> bytes; };
        std::vector<Program> programs;
        programs.reserve(entries.size());
        // A lazy package indexes the identities its manifest records; each
        // program artifact is read and checked against them when its stage is
        // prepared, so opening does not read the programs.
        auto built = [&]() -> std::variant<IdentityIndex, pi::Error> {
            if (options.lazy_entries) {
                std::vector<pi::Identity> identities;
                identities.reserve(entries.size());
                for (const auto& entry : entries)
                    identities.push_back({entry.stage, entry.program_size, entry.prefix, entry.full});
                return make_identity_index(std::move(identities));
            }
            std::size_t program_total = 0;
            for (const auto& entry : entries) {
                const auto program_path = package_file(root, entry.stage_directory, "program.bin");
                if (fs::file_size(program_path) != entry.program_size)
                    fail(MetalShaderPackageError::Code::Integrity,
                         "Program file size differs from the manifest");
                auto program = read_file(program_path,
                                         pi::maximum_program_size, "program");
                if (program.size() != entry.program_size || digest(program) != entry.full ||
                    program.size() < pi::prefix_size ||
                    digest(std::span(program).first(pi::prefix_size)) != entry.prefix)
                    fail(MetalShaderPackageError::Code::Integrity,
                         "Program size, full digest or prefix digest differs from the manifest");
                if (program.size() > total_program_limit - program_total)
                    fail(MetalShaderPackageError::Code::Input,
                         "Owned program bytes exceed the aggregate policy");
                program_total += program.size();
                programs.push_back({entry.stage, std::move(program)});
            }
            std::vector<pi::ProgramInput> inputs;
            inputs.reserve(programs.size());
            for (const auto& program : programs) inputs.push_back({program.stage, program.bytes});
            auto from_programs = pi::build(inputs, {hash_bytes, nullptr});
            if (const auto* error = std::get_if<pi::Error>(&from_programs)) return *error;
            const auto records = std::get<pi::Index>(from_programs).identities();
            return IdentityIndex{std::vector<pi::Identity>(records.begin(), records.end())};
        }();
        if (const auto* error = std::get_if<pi::Error>(&built))
            fail(MetalShaderPackageError::Code::Identity,
                 std::string("Program identity index rejected package: ") + error->message());
        auto index = std::get<IdentityIndex>(std::move(built));
        std::map<std::string, std::vector<std::size_t>> mapped;
        for (std::size_t i = 0; i < entries.size(); ++i)
            mapped[identity_key(entries[i].stage, entries[i].full, entries[i].program_size)].push_back(i);
        programs.clear();
        programs.shrink_to_fit();
        auto impl = std::make_unique<Impl>(root, std::move(entries), std::move(index), std::move(mapped));
        impl->chain = chain;
        impl->content = manifest_digest;
        impl->source = source;
        impl->defer_unsupported = options.defer_unsupported_entries;
        impl->prepared = std::make_unique<std::atomic<bool>[]>(impl->entries.size());
        impl->prepare_failures.resize(impl->entries.size());
        if (!options.lazy_entries)
            for (std::size_t i = 0; i != impl->entries.size(); ++i)
                impl->prepared[i].store(true, std::memory_order_relaxed);
        else if (options.prefetch_threads)
            impl->start_prefetch(options.prefetch_threads);
        return std::shared_ptr<MetalShaderPackage>(new MetalShaderPackage(std::move(impl)));
    } catch (const Failure& failure) {
        return failure.error;
    } catch (const fs::filesystem_error& error) {
        return MetalShaderPackageError{MetalShaderPackageError::Code::Input, error.what()};
    } catch (const std::bad_alloc&) {
        return MetalShaderPackageError{MetalShaderPackageError::Code::Allocation,
                                       "Shader package allocation failed"};
    } catch (const std::length_error&) {
        return MetalShaderPackageError{MetalShaderPackageError::Code::Allocation,
                                       "Shader package allocation failed"};
    } catch (const std::exception& error) {
        return MetalShaderPackageError{MetalShaderPackageError::Code::Input, error.what()};
    }
    }
}

CompiledComputeStageResult MetalShaderPackage::lookup_stage(
    Stage stage, std::span<const std::byte> program,
    gm::VertexDepthConvention depth) noexcept {
    try {
        const pi::MemoryReader reader{
            [](void* context, std::uint64_t address, std::span<std::byte> destination) noexcept {
                const auto bytes = *static_cast<const std::span<const std::byte>*>(context);
                if (address > bytes.size() || destination.size() > bytes.size() - address) return false;
                std::memcpy(destination.data(), bytes.data() + address, destination.size());
                return true;
            }, &program};
        const auto identified = lookup_identity(impl_->index, stage, 0, reader, {hash_bytes, nullptr});
        if (const auto* error = std::get_if<pi::Error>(&identified))
            return impl_->error(error->message());
        const auto* variants = impl_->find_variants(std::get<pi::Identity>(identified));
        Entry* selected = nullptr;
        if (variants) for (const auto index : *variants) {
            impl_->prepare(index);
            auto& candidate = impl_->entries[index];
            if (!candidate.unsupported.empty()) continue;
            if (stage == Stage::Vertex && candidate.metadata.vertex_depth != depth) continue;
            if (selected) return impl_->error("NVN program has ambiguous compiled stage variants");
            selected = &candidate;
        }
        if (!selected) return impl_->error("NVN program has no compatible compiled stage variant");
        return impl_->load(*selected, nullptr, true);
    } catch (const Failure& failure) {
        return impl_->error(failure.error.detail);
    } catch (const std::exception& error) {
        return impl_->error(error.what());
    } catch (...) {
        return impl_->error("Unexpected NVN stage lookup failure");
    }
}

std::shared_ptr<const CompiledGraphicsStage> MetalShaderPackage::stage_by_identity(
    std::string_view identity) noexcept {
    try {
        Entry* selected = nullptr;
        impl_->build_digest_index();
        // Identity text is "<stage>.<digest>.<variant suffix>"; only the
        // entries that share the first two fields need their reflection.
        const auto split = identity.rfind('.');
        if (split == std::string_view::npos) return nullptr;
        const auto candidates = impl_->by_digest.find(std::string(identity.substr(0, split + 1)));
        if (candidates == impl_->by_digest.end()) return nullptr;
        for (const auto index : candidates->second) {
            impl_->prepare(index);
            const auto& entry = impl_->entries[index];
            if (entry.unsupported.empty() && stage_identity_text(entry) == identity) {
                selected = &impl_->entries[index];
                break;
            }
        }
        if (!selected) return nullptr;
        return impl_->load(*selected, nullptr, true);
    } catch (...) {
        return nullptr;
    }
}

const std::string& MetalShaderPackage::chain_id() const noexcept { return impl_->chain; }
const std::string& MetalShaderPackage::content_id() const noexcept { return impl_->content; }

CompiledComputeStageResult MetalShaderPackage::lookup_stage_at(
    Stage stage, std::uint64_t program,
    gm::VertexDepthConvention depth) noexcept {
    try {
        const pi::MemoryReader reader{
            [](void* context, std::uint64_t offset, std::span<std::byte> destination) noexcept {
                const auto base = *static_cast<const std::uint64_t*>(context);
                if (offset > std::numeric_limits<std::uint64_t>::max() - base) return false;
                mach_vm_size_t copied = 0;
                return mach_vm_read_overwrite(mach_task_self(), base + offset, destination.size(),
                           reinterpret_cast<mach_vm_address_t>(destination.data()), &copied) == KERN_SUCCESS &&
                       copied == destination.size();
            }, &program};
        const auto identified = lookup_identity(impl_->index, stage, 0, reader, {hash_bytes, nullptr});
        if (const auto* error = std::get_if<pi::Error>(&identified))
            return impl_->error(error->message());
        const auto* variants = impl_->find_variants(std::get<pi::Identity>(identified));
        Entry* selected = nullptr;
        if (variants) for (const auto index : *variants) {
            impl_->prepare(index);
            auto& candidate = impl_->entries[index];
            if (!candidate.unsupported.empty()) continue;
            if (stage == Stage::Vertex && candidate.metadata.vertex_depth != depth) continue;
            if (selected) return impl_->error("NVN program has ambiguous compiled stage variants");
            selected = &candidate;
        }
        if (!selected) return impl_->error("NVN program has no compatible compiled stage variant");
        return impl_->load(*selected, nullptr, true);
    } catch (const Failure& failure) {
        return impl_->error(failure.error.detail);
    } catch (const std::exception& error) {
        return impl_->error(error.what());
    } catch (...) {
        return impl_->error("Unexpected NVN stage lookup failure");
    }
}

CompiledGraphicsLookupResult MetalShaderPackage::lookup(
    const graphics::RenderStop& snapshot, const graphics::GpuAddressSpace& memory) noexcept {
    try {
        if (snapshot.kind != graphics::RenderKind::Draw ||
            !snapshot.programs[1].enabled || !snapshot.programs[5].enabled ||
            snapshot.programs[0].enabled || snapshot.programs[2].enabled ||
            snapshot.programs[3].enabled || snapshot.programs[4].enabled)
            return impl_->error("Shader package requires exactly vertex and fragment stages");
        ReaderContext reader{&memory};
        const pi::MemoryReader reads{read_guest, &reader};
        const pi::HashProvider hashes{hash_bytes, nullptr};
        const auto find = [&](Stage stage, std::uint64_t address) -> pi::Identity {
            auto result = impl_->identify(stage, address, reads, hashes, &memory);
            if (const auto* error = std::get_if<pi::Error>(&result)) {
                std::string detail = std::string("Shader package ") +
                    (stage == Stage::Vertex ? "vertex" : "fragment") + " lookup failed: " + error->message();
                if (error->code == pi::Error::Code::Miss) {
                    std::array<std::byte, pi::prefix_size> prefix{};
                    if (read_guest(&reader, address, prefix)) {
                        const auto prefix_hash = hash_bytes(nullptr, prefix);
                        if (prefix_hash) detail += " (prefix-sha256 " + hex(*prefix_hash) + ")";
                    }
                }
                throw impl_->error(std::move(detail),
                    error->code == pi::Error::Code::ReadFailure ||
                    error->code == pi::Error::Code::AddressOverflow
                        ? graphics::CommandStatus::MemoryFault
                        : graphics::CommandStatus::Unsupported);
            }
            return std::get<pi::Identity>(result);
        };
        const auto vertex_identity = find(Stage::Vertex, snapshot.programs[1].address);
        const auto fragment_identity = find(Stage::Fragment, snapshot.programs[5].address);
        Entry& vertex_entry = impl_->select(vertex_identity, snapshot);
        Entry& fragment_entry = impl_->select(fragment_identity, snapshot);
        const auto& vertex = vertex_entry.metadata;
        const auto& fragment = fragment_entry.metadata;
        for (const auto& input : fragment.interface.generic_inputs) {
            const auto output = std::find_if(vertex.interface.generic_outputs.begin(),
                vertex.interface.generic_outputs.end(), [&](const auto& item) {
                    return item.slot == input.slot;
                });
            if (output == vertex.interface.generic_outputs.end() ||
                (output->component_mask & input.component_mask) != input.component_mask)
                return impl_->error("Compiled vertex and fragment interfaces do not link");
        }
        std::optional<graphics::RenderTargetPlan> targets;
        if (fragment.fragment_output_types_from_render_targets) {
            auto planned = graphics::plan_render_targets(snapshot);
            if (const auto* error = std::get_if<graphics::RendererError>(&planned))
                return impl_->error(error->command().detail, error->command().status);
            targets = std::get<graphics::RenderTargetPlan>(std::move(planned));
        }
        return CompiledGraphicsPair{impl_->load(vertex_entry),
            impl_->load(fragment_entry, targets ? &*targets : nullptr)};
    } catch (const graphics::CommandResult& error) {
        return error;
    } catch (const Failure& failure) {
        return impl_->error(failure.error.detail);
    } catch (const std::bad_alloc&) {
        return graphics::CommandResult{graphics::CommandStatus::BudgetExceeded,
                                       "Shader package lookup allocation failed"};
    } catch (const std::exception& error) {
        return impl_->error(error.what());
    } catch (...) {
        return impl_->error("Unexpected shader package lookup failure");
    }
}

CompiledGraphicsLookupResult MetalShaderPackage::callback(
    void* context, const graphics::RenderStop& snapshot,
    const graphics::GpuAddressSpace& memory) noexcept {
    if (!context) return graphics::CommandResult{
        graphics::CommandStatus::Unsupported, "Metal shader package context is null"};
    return static_cast<MetalShaderPackage*>(context)->lookup(snapshot, memory);
}

CompiledComputeStageResult MetalShaderPackage::lookup_compute(
    const graphics::RenderStop& snapshot, const graphics::GpuAddressSpace& memory) noexcept {
    try {
        return impl_->lookup_compute(snapshot, memory);
    } catch (const graphics::CommandResult& error) {
        return error;
    } catch (const Failure& failure) {
        return impl_->error(failure.error.detail);
    } catch (const std::bad_alloc&) {
        return graphics::CommandResult{graphics::CommandStatus::BudgetExceeded,
                                       "Shader package compute lookup allocation failed"};
    } catch (const std::exception& error) {
        return impl_->error(error.what());
    } catch (...) {
        return impl_->error("Unexpected shader package compute lookup failure");
    }
}

CompiledComputeStageResult MetalShaderPackage::compute_callback(
    void* context, const graphics::RenderStop& snapshot,
    const graphics::GpuAddressSpace& memory) noexcept {
    if (!context) return graphics::CommandResult{
        graphics::CommandStatus::Unsupported, "Metal shader package context is null"};
    return static_cast<MetalShaderPackage*>(context)->lookup_compute(snapshot, memory);
}

}  // namespace armigo::platform
