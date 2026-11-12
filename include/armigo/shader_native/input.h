// ============================================================================
// Armigo Shader Input - Shader Input Processing
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <map>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace armigo::shader_native {
using Bytes = std::span<const std::byte>;
using Buffer = std::vector<std::byte>;

struct InputError : std::runtime_error {
    std::uint64_t offset;
    InputError(std::string message, std::uint64_t at = 0)
        : std::runtime_error(std::move(message)), offset(at) {}
};
struct Limits {
    std::size_t member_bytes = 512ull << 20;
    std::uint64_t traversal_bytes = 64ull << 30;
    std::size_t spans = 1'000'000;
    std::size_t records = 1'000'000;
    std::size_t variations = 65'536;
    std::size_t table_bytes = 64ull << 20;
    unsigned depth = 8;
};
using Reader = std::function<std::size_t(std::span<std::byte>)>;
using Dictionaries = std::map<unsigned, Buffer>;
// The reader returns zero only at EOF. Exactly one Zstandard frame is admitted.
Buffer read_zstandard(Reader reader, const Dictionaries& dictionaries, std::size_t limit);
Dictionaries read_dictionaries(Bytes sarc, std::size_t limit);
// Oversized raw archives are refused before a signature-only media scan can
// mistake compressed members for an absence of shader containers.
Buffer read_raw_member(Reader reader, std::uint64_t size, std::size_t limit);
std::string digest(Bytes bytes);
std::uint64_t little_endian(Bytes bytes, std::size_t offset, unsigned width);

struct Location {
    std::string member;
    std::uint64_t offset = 0;
};
// A runtime record is the control and code an NVN driver receives from the
// title (the 0xe0 control bytes its version-20 reader consumes, and the code
// at the program's data address); it carries no container extents.
enum class ContainerKind { bnsh, bahs, runtime };
struct Record {
    ContainerKind kind{};
    std::vector<Location> ancestry;
    std::uint64_t container_offset{}, descriptor_offset{}, control_offset{}, code_offset{};
    std::optional<std::uint32_t> variation;
    std::optional<unsigned> slot;
    Bytes control, code;
};
struct Inventory {
    std::uint64_t visited_bytes{}, spans{}, bnsh{}, bahs{}, records{}, duplicate_candidates{};
    std::uint64_t nested_candidates{}, rejected_markers{}, unrecognized_spans{};
};
// Record views live only during the callback. Membership is never deduplicated.
class Enumerator {
public:
    using Visitor = std::function<void(const Record&)>;
    explicit Enumerator(Visitor visitor, Limits limits = {}, Dictionaries dictionaries = {});
    void scan(Bytes bytes, std::string source);
    const Inventory& inventory() const { return inventory_; }
private:
    void scan_inner(Bytes bytes, std::vector<Location>& ancestry, unsigned depth);
    void bnsh(Bytes bytes, const std::vector<Location>& ancestry, std::size_t offset);
    void bahs(Bytes bytes, const std::vector<Location>& ancestry, std::size_t offset);
    // True when the span is an NDVE container whose pack or network tables hold shaders.
    bool ndve(Bytes bytes, std::vector<Location>& ancestry, std::size_t offset);
    // True when the span is an NDSR pack and at least one shader record was emitted.
    bool ndsr(Bytes bytes, const std::vector<Location>& ancestry, std::size_t offset);
    void emit(Record record);
    Visitor visitor_;
    Limits limits_;
    Inventory inventory_;
    Dictionaries dictionaries_;
    std::size_t resident_bytes_{};
};
} // namespace armigo::shader_native
