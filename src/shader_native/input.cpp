// ============================================================================
// Armigo Shader Input - Shader Input Processing
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include <armigo/shader_native/input.h>
#include <armigo/shader_native/program.h>
#include <armigo/shader_input/archive.h>
#include <CommonCrypto/CommonDigest.h>
#include <zstd.h>
#include <algorithm>
#include <array>
#include <cstring>
#include <memory>
#include <set>

namespace armigo::shader_native {
namespace {
bool covers(Bytes bytes, std::uint64_t at, std::uint64_t size) {
    return at <= bytes.size() && size <= bytes.size() - at;
}
Bytes slice(Bytes bytes, std::uint64_t at, std::uint64_t size) {
    if (!covers(bytes, at, size)) throw InputError("Native container range is outside its parent", at);
    return bytes.subspan(at, size);
}
bool magic(Bytes bytes, std::size_t at, std::string_view text) {
    return covers(bytes, at, text.size()) && std::memcmp(bytes.data() + at, text.data(), text.size()) == 0;
}
void zcheck(std::size_t status) {
    if (ZSTD_isError(status)) throw InputError(std::string("Zstandard: ") + ZSTD_getErrorName(status));
}
void control_prefix(Bytes control) {
    const auto version = control.size() >= 12 ? little_endian(control, 8, 4) : 0;
    // Version 17 keeps the fields read below at the version-20 offsets.
    if (control.size() < 0xd8 || little_endian(control, 0, 4) != 0x98761234 ||
        little_endian(control, 4, 4) != 0xc9 || (version != 20 && version != 17))
        throw InputError("Unsupported native shader control family or version");
}
// Bytes that belong to the control section itself. An NVN2 section's key is
// its first 0x48 bytes. A Maxwell section uses the length at +0x20 when that
// length is a short section, and 0x880 otherwise. The length an entry claims
// is not this span.
std::uint64_t control_own_length(Bytes control) {
    if (control.size() >= 0x48 && little_endian(control, 0, 4) == 0x98761234 &&
        little_endian(control, 4, 4) == 0xc9)
        return 0x48;
    const auto declared = control.size() >= 0x24 ? little_endian(control, 0x20, 4) : 0;
    return declared >= 0x760 && declared < 0x880 ? declared : 0x880;
}
}
std::uint64_t little_endian(Bytes bytes, std::size_t at, unsigned width) {
    if (width == 0 || width > 8 || !covers(bytes, at, width))
        throw InputError("Truncated little-endian field", at);
    std::uint64_t value = 0;
    for (unsigned i = 0; i < width; ++i)
        value |= std::uint64_t(std::to_integer<unsigned char>(bytes[at + i])) << (8 * i);
    return value;
}
std::string digest(Bytes bytes) {
    CC_SHA256_CTX context;
    CC_SHA256_Init(&context);
    while (!bytes.empty()) {
        const auto chunk = bytes.first(std::min<std::size_t>(bytes.size(), 1 << 20));
        CC_SHA256_Update(&context, chunk.data(), static_cast<CC_LONG>(chunk.size()));
        bytes = bytes.subspan(chunk.size());
    }
    std::array<unsigned char, CC_SHA256_DIGEST_LENGTH> hash{};
    CC_SHA256_Final(hash.data(), &context);
    constexpr char digits[] = "0123456789abcdef";
    std::string result;
    for (auto byte : hash) { result += digits[byte >> 4]; result += digits[byte & 15]; }
    return result;
}
Buffer read_zstandard(Reader reader, const Dictionaries& dictionaries, std::size_t limit) {
    std::unique_ptr<ZSTD_DCtx, decltype(&ZSTD_freeDCtx)> context(ZSTD_createDCtx(), ZSTD_freeDCtx);
    if (!context) throw InputError("Cannot allocate Zstandard decoder");
    // Bound the decoder's internal window independently of the advertised output size.
    zcheck(ZSTD_DCtx_setParameter(context.get(), ZSTD_d_windowLogMax, 27));
    std::array<std::byte, 131072> input{}, output{};
    std::size_t filled = 0;
    while (filled < 18) {
        auto n = reader(std::span(input).subspan(filled, 18 - filled));
        if (n > 18 - filled) throw InputError("Reader exceeded requested extent");
        if (!n) break;
        filled += n;
    }
    if (filled < 4 || little_endian(Bytes(input).first(filled), 0, 4) != 0xfd2fb528)
        throw InputError("Input is not an ordinary Zstandard frame");
    const auto id = ZSTD_getDictID_fromFrame(input.data(), filled);
    if (id) {
        auto dictionary = dictionaries.find(id);
        if (dictionary == dictionaries.end()) throw InputError("Missing Zstandard dictionary " + std::to_string(id));
        zcheck(ZSTD_DCtx_loadDictionary(context.get(), dictionary->second.data(), dictionary->second.size()));
    }
    Buffer result;
    ZSTD_inBuffer in{input.data(), filled, 0};
    for (;;) {
        ZSTD_outBuffer out{output.data(), output.size(), 0};
        const auto before = in.pos;
        const auto remaining = ZSTD_decompressStream(context.get(), &out, &in);
        zcheck(remaining);
        if (out.pos > limit - result.size()) throw InputError("Zstandard output exceeds member limit");
        result.insert(result.end(), output.begin(), output.begin() + out.pos);
        if (!remaining) {
            if (in.pos != in.size || reader(std::span(input).first(1)))
                throw InputError("Trailing data or concatenated Zstandard frames are unsupported");
            return result;
        }
        if (in.pos == in.size) {
            auto n = reader(input);
            if (n > input.size()) throw InputError("Reader exceeded requested extent");
            if (!n && out.pos != output.size()) throw InputError("Truncated Zstandard frame");
            in = {input.data(), n, 0};
        } else if (before == in.pos && !out.pos) throw InputError("Zstandard decoder made no progress");
    }
}
Buffer read_raw_member(Reader reader, std::uint64_t size, std::size_t limit) {
    if (size <= limit) {
        Buffer bytes(static_cast<std::size_t>(size));
        std::size_t position = 0;
        while (position < bytes.size()) {
            const auto n = reader(std::span(bytes).subspan(position));
            if (!n || n > bytes.size() - position) throw InputError("Truncated or invalid raw member read", position);
            position += n;
        }
        return bytes;
    }
    std::array<std::byte, (1 << 20) + 7> window{};
    std::size_t carry = 0;
    std::uint64_t position = 0;
    while (position < size) {
        const auto request = static_cast<std::size_t>(std::min<std::uint64_t>(size - position, 1 << 20));
        const auto n = reader(std::span(window).subspan(carry, request));
        if (!n || n > request) throw InputError("Truncated or invalid oversized raw member read", position);
        const auto bytes = Bytes(window).first(carry + n);
        // Keep the first bytes until a complete root signature can be checked,
        // including readers that return fewer than four bytes per call.
        if (position - carry == 0 && magic(bytes, 0, "SARC"))
            throw InputError("Oversized SARC requires bounded archive traversal; member limit exceeded");
        for (std::size_t at = 0; at + 4 <= bytes.size(); ++at) {
            if (magic(bytes, at, "BAHS") || magic(bytes, at, std::string_view("BNSH\0\0\0\0", 8)))
                throw InputError("Oversized raw member contains a shader container marker", position - carry + at);
        }
        carry = std::min<std::size_t>(7, bytes.size());
        std::memmove(window.data(), bytes.data() + bytes.size() - carry, carry);
        position += n;
    }
    return {};
}
Dictionaries read_dictionaries(Bytes bytes, std::size_t limit) {
    auto parsed = shader_input::parse_sarc(bytes);
    if (auto error = std::get_if<shader_input::ArchiveError>(&parsed)) throw InputError(error->message(), error->offset);
    Dictionaries result;
    std::size_t total = 0;
    for (auto& member : std::get<shader_input::SarcArchive>(parsed).members) {
        if (member.data.size() > limit - total) throw InputError("Dictionary storage exceeds limit");
        total += member.data.size();
        auto id = ZSTD_getDictID_fromDict(member.data.data(), member.data.size());
        if (!id) throw InputError("SARC member is not an identified Zstandard dictionary");
        if (!result.emplace(id, Buffer(member.data.begin(), member.data.end())).second)
            throw InputError("Duplicate Zstandard dictionary identifier");
    }
    if (result.empty()) throw InputError("Dictionary archive is empty");
    return result;
}
Enumerator::Enumerator(Visitor visitor, Limits limits, Dictionaries dictionaries)
    : visitor_(std::move(visitor)), limits_(limits), dictionaries_(std::move(dictionaries)) {}
void Enumerator::scan(Bytes bytes, std::string source) {
    resident_bytes_ = bytes.size();
    std::vector<Location> ancestry{{std::move(source), 0}};
    scan_inner(bytes, ancestry, 0);
}
void Enumerator::emit(Record record) {
    if (inventory_.records == limits_.records) throw InputError("Native shader record limit exceeded");
    control_prefix(record.control);
    const auto declared = little_endian(record.control, 0x30, 4);
    if ((record.kind == ContainerKind::bnsh ? declared != record.control.size() : declared && declared != record.control.size()) ||
        little_endian(record.control, 0x44, 4) != record.code.size())
        throw InputError("Native shader descriptor sizes disagree with control", record.descriptor_offset);
    visitor_(record);
    ++inventory_.records;
}
void Enumerator::bnsh(Bytes b, const std::vector<Location>& ancestry, std::size_t offset) {
    if (!magic(b, 0x60, "grsc")) throw InputError("BNSH resource block is missing", offset);
    const auto count = little_endian(b, 0x7c, 4), table = little_endian(b, 0x80, 8);
    if (count > limits_.variations) throw InputError("BNSH variation limit exceeded");
    slice(b, table, count * 0x40);
    ++inventory_.bnsh;
    for (std::uint32_t i = 0; i < count; ++i) {
        const auto program = little_endian(b, table + i * 0x40 + 0x10, 8);
        if (!program) continue;
        slice(b, program, 0x38);
        for (unsigned slot = 0; slot < 6; ++slot) {
            auto descriptor = little_endian(b, program + 8 + slot * 8, 8);
            if (!descriptor) continue;
            slice(b, descriptor, 0x20);
            auto control = little_endian(b, descriptor + 8, 8), code = little_endian(b, descriptor + 16, 8);
            emit({ContainerKind::bnsh, ancestry, offset, descriptor, control, code, i, slot,
                  slice(b, control, little_endian(b, descriptor + 28, 4)),
                  slice(b, code, little_endian(b, descriptor + 24, 4))});
        }
        // NVN2's task and mesh stages follow at +0x40 (slot 7) and +0x48
        // (slot 8). An older program block ends before them, so a pointer
        // there counts only when it leads, inside the container, to a
        // control section of that stage (selector 6 or 7).
        for (unsigned slot = 7; slot < 9; ++slot) {
            if (!covers(b, program + 8 + slot * 8, 8)) continue;
            const auto descriptor = little_endian(b, program + 8 + slot * 8, 8);
            if (!descriptor || !covers(b, descriptor, 0x20)) continue;
            const auto control = little_endian(b, descriptor + 8, 8), code = little_endian(b, descriptor + 16, 8);
            const auto control_size = little_endian(b, descriptor + 28, 4), code_size = little_endian(b, descriptor + 24, 4);
            if (control_size <= 0x56 || !covers(b, control, control_size) || !covers(b, code, code_size) ||
                little_endian(b, control, 4) != 0x98761234 || little_endian(b, control + 0x56, 1) != slot - 1)
                continue;
            emit({ContainerKind::bnsh, ancestry, offset, descriptor, control, code, i, slot,
                  slice(b, control, control_size), slice(b, code, code_size)});
        }
    }
}
void Enumerator::bahs(Bytes b, const std::vector<Location>& ancestry, std::size_t offset) {
    const auto start = little_endian(b, 0x1c, 4), size = little_endian(b, 0x18, 4);
    if (start < 32 || start > limits_.table_bytes || !covers(b, start, size))
        throw InputError("BAHS code region or table limit is invalid", offset);
    ++inventory_.bahs;
    struct Candidate {
        std::size_t at = 0;
        std::uint64_t control = 0;
        std::uint64_t code = 0;
        Bytes control_bytes;
        Bytes code_bytes;
    };
    std::vector<Candidate> candidates;
    for (std::size_t at = 0; at + 16 <= start; at += 4) {
        const auto code = little_endian(b, at, 4), extent = little_endian(b, at + 8, 4);
        const auto displacement = static_cast<std::int64_t>(little_endian(b, at + 12, 4));
        const auto signed_displacement = displacement & 0x80000000 ? displacement - 0x100000000ll : displacement;
        const auto control = static_cast<std::int64_t>(at) + signed_displacement;
        if (little_endian(b, at + 4, 4) || code % 256 || extent < 0xd8 || extent > (1 << 20) ||
            control < 0 || std::uint64_t(control) > start || extent > start - control) continue;
        auto c = slice(b, control, extent);
        if (little_endian(c, 0, 4) != 0x98761234 || little_endian(c, 0xd4, 4) != extent) continue;
        const auto code_size = little_endian(c, 0x44, 4);
        if (code > size || code_size > size - code) throw InputError("BAHS recognized candidate code is out of range", at);
        candidates.push_back({at, std::uint64_t(control), code, c, slice(b, start + code, code_size)});
    }
    // A table position inside a candidate's own control span is that section's
    // words read as an entry, not another program. Duplicate recognition follows
    // the drop, so a later entry that names the same control and code is kept.
    std::vector<std::pair<std::uint64_t, std::uint64_t>> spans;
    spans.reserve(candidates.size());
    for (const auto& candidate : candidates) {
        const auto length = control_own_length(candidate.control_bytes);
        spans.emplace_back(candidate.control, candidate.control + length);
    }
    std::set<std::pair<std::uint64_t, std::uint64_t>> seen;
    for (const auto& candidate : candidates) {
        const bool nested = std::any_of(spans.begin(), spans.end(), [&](const auto& span) {
            return span.first <= candidate.at && candidate.at < span.second;
        });
        if (nested) { ++inventory_.nested_candidates; continue; }
        if (!seen.emplace(candidate.control, candidate.code).second) {
            ++inventory_.duplicate_candidates;
            continue;
        }
        emit({ContainerKind::bahs, ancestry, offset, candidate.at, candidate.control, start + candidate.code, {}, {},
              candidate.control_bytes, candidate.code_bytes});
    }
}
bool Enumerator::ndve(Bytes bytes, std::vector<Location>& ancestry, std::size_t offset) {
    if (!magic(bytes, 0, "NDVE") || bytes.size() < 0x18) return false;
    // The NDSR pack follows the 8-byte model records counted at +0x14; the
    // networks follow the pack, whose length is at +0x0c.
    const auto models = little_endian(bytes, 0x14, 4);
    if (models > (bytes.size() - 0x18) / 8) return false;
    const std::uint64_t base = 0x18 + 8 * models;
    const auto first = little_endian(bytes, 0x0c, 4);
    const auto second = little_endian(bytes, 0x10, 4);
    if (first > bytes.size() - base || second != bytes.size() - base - first) return false;
    const auto section = slice(bytes, base + first, second);
    auto emit_program = [&](std::uint64_t control_at, std::uint64_t control_size, std::uint64_t code_size) {
        if (control_size < 0xe0 || code_size == 0 || control_at > section.size() ||
            control_size > section.size() - control_at || code_size > section.size() - control_at - control_size)
            throw InputError("NDVE shader extent is outside its group", offset + control_at);
        const auto control = slice(section, control_at, control_size);
        if (little_endian(control, 0, 4) != 0x98761234 || little_endian(control, 0x30, 4) != control_size ||
            little_endian(control, 0xd4, 4) != control_size || little_endian(control, 0x44, 4) != code_size)
            throw InputError("NDVE shader control does not match its group entry", offset + control_at);
        const auto code_at = control_at + control_size;
        emit({ContainerKind::bahs, ancestry, offset, control_at, offset + base + first + control_at,
              offset + base + first + code_at, {}, {}, control, slice(section, code_at, code_size)});
    };
    std::size_t position = 0;
    bool grouped = false;
    bool packed = false;
    if (first >= 4) packed = ndsr(slice(bytes, base, first), ancestry, offset + base);
    while (position + 0x18 <= section.size()) {
        const auto count = little_endian(section, position, 4);
        const auto flags = little_endian(section, position + 4, 4);
        const auto header = little_endian(section, position + 8, 4);
        if (flags == 1 && count >= 1 && count <= 64 && header == 0x10 + count * 16 &&
            position + header <= section.size()) {
            std::uint64_t end = 0;
            for (std::uint64_t index = 0; index < count; ++index) {
                const auto entry = position + 0x10 + index * 16;
                const auto relative = little_endian(section, entry, 4);
                const auto control_size = little_endian(section, entry + 4, 4);
                const auto code_size = little_endian(section, entry + 8, 4);
                if (little_endian(section, entry + 12, 4) != 0)
                    throw InputError("NDVE shader group entry is not terminated", offset + entry);
                emit_program(position + relative, control_size, code_size);
                end = position + relative + control_size + code_size;
            }
            if (end < position + header) throw InputError("NDVE shader group is unordered", offset + position);
            position = static_cast<std::size_t>(end);
            grouped = true;
            continue;
        }
        if (count == 1 && flags == 0 && header == 0x18 && position + 0x18 <= section.size() &&
            little_endian(section, position + 0x14, 4) == 0) {
            const auto control_size = little_endian(section, position + 0x0c, 4);
            const auto code_size = little_endian(section, position + 0x10, 4);
            emit_program(position + 0x18, control_size, code_size);
            position = static_cast<std::size_t>(position + 0x18 + control_size + code_size);
            grouped = true;
            continue;
        }
        break;
    }
    return grouped || packed;
}

bool Enumerator::ndsr(Bytes bytes, const std::vector<Location>& ancestry, std::size_t offset) {
    if (!magic(bytes, 0, "NDSR") || bytes.size() < 0x20) return false;
    // Records are 16 bytes. A short header puts the first one at +0x10; one
    // extra header word puts it at +0x14. The start is the one whose first
    // records point at control sections.
    const auto points = [&](std::size_t start) {
        unsigned hits = 0;
        const auto count = std::min<std::size_t>(16, (bytes.size() - start) / 16);
        for (std::size_t index = 0; index < count; ++index) {
            const auto control_at = little_endian(bytes, start + index * 16, 4);
            if (control_at + 0x48 <= bytes.size() && little_endian(bytes, control_at, 4) == 0x98761234 &&
                little_endian(bytes, control_at + 4, 4) == 0xc9)
                ++hits;
        }
        return hits;
    };
    const auto from_10 = points(0x10);
    const auto from_14 = bytes.size() >= 0x24 ? points(0x14) : 0u;
    // A zero score still walks: a data entry of 8 bytes shifts the next shader
    // off the 16-byte grid the score uses, and that shader is still a record.
    std::size_t cursor = from_14 > from_10 ? 0x14 : 0x10;
    bool emitted = false;
    while (cursor + 16 <= bytes.size()) {
        const auto control_at = little_endian(bytes, cursor, 4);
        const auto control_word = little_endian(bytes, cursor + 4, 4);
        const auto code_at = little_endian(bytes, cursor + 8, 4);
        const auto code_word = little_endian(bytes, cursor + 12, 4);
        if ((control_at | control_word | code_at | code_word) == 0) break;
        const bool control = control_at + 0x48 <= bytes.size() &&
                             little_endian(bytes, control_at, 4) == 0x98761234 &&
                             little_endian(bytes, control_at + 4, 4) == 0xc9;
        // A data entry is 8 bytes, so the next shader can sit 8 bytes on.
        if (!control) {
            cursor += 8;
            continue;
        }
        cursor += 16;
        const auto declared = little_endian(bytes, control_at + 0x30, 4);
        const auto record_length = control_word >> 4;
        const auto control_size = declared ? declared : std::max<std::uint64_t>(record_length, 0xe0);
        const auto code_size = little_endian(bytes, control_at + 0x44, 4);
        if (control_size < 0xd8 || control_at > bytes.size() || control_size > bytes.size() - control_at ||
            code_at > bytes.size() || code_size > bytes.size() - code_at || code_size == 0)
            continue;
        const auto version = little_endian(bytes, control_at + 8, 4);
        if (version != 17 && version != 20) continue;
        // A pack control carries no container extent: +0x30 and +0xd4 are
        // clear and the section is the reader span. A BAHS record stores its
        // extent at +0xd4, so this one is a runtime control.
        const auto file_extent = control_size > 0xd4 ? little_endian(bytes, control_at + 0xd4, 4) : 0;
        const auto kind = !declared && file_extent == 0 && control_size == runtime_control_bytes
                              ? ContainerKind::runtime
                              : ContainerKind::bahs;
        emit({kind, ancestry, offset, cursor - 16, offset + control_at, offset + code_at, {}, {},
              slice(bytes, control_at, control_size), slice(bytes, code_at, code_size)});
        emitted = true;
    }
    return emitted;
}

void Enumerator::scan_inner(Bytes b, std::vector<Location>& ancestry, unsigned depth) {
    if (depth > limits_.depth || inventory_.spans == limits_.spans || b.size() > limits_.member_bytes ||
        b.size() > limits_.traversal_bytes - inventory_.visited_bytes)
        throw InputError("Native container traversal limit exceeded");
    ++inventory_.spans;
    inventory_.visited_bytes += b.size();
    if (b.size() >= 4 && little_endian(b, 0, 4) == 0xfd2fb528) {
        std::size_t position = 0;
        auto expanded = read_zstandard([&](std::span<std::byte> output) {
            auto n = std::min(output.size(), b.size() - position);
            std::copy_n(b.begin() + position, n, output.begin()); position += n; return n;
        }, dictionaries_, limits_.member_bytes - resident_bytes_);
        resident_bytes_ += expanded.size();
        ancestry.push_back({"zstandard:" + digest(expanded), 0});
        scan_inner(expanded, ancestry, depth + 1);
        resident_bytes_ -= expanded.size();
        ancestry.pop_back();
        return;
    }
    if (ndve(b, ancestry, ancestry.empty() ? 0 : ancestry.back().offset)) return;
    if (ndsr(b, ancestry, ancestry.empty() ? 0 : ancestry.back().offset)) return;
    if (magic(b, 0, "SARC")) {
        auto result = shader_input::parse_sarc(b);
        if (auto error = std::get_if<shader_input::ArchiveError>(&result)) throw InputError(error->message(), error->offset);
        std::size_t ordinal = 0;
        for (auto& member : std::get<shader_input::SarcArchive>(result).members) {
            ancestry.push_back({member.has_name ? std::string(member.name) : "#" + std::to_string(ordinal),
                                std::uint64_t(member.data.data() - b.data())});
            scan_inner(member.data, ancestry, depth + 1);
            ancestry.pop_back();
            ++ordinal;
        }
        return;
    }
    bool recognized = false;
    for (std::size_t at = 0; at + 4 <= b.size(); ++at) {
        if (magic(b, at, std::string_view("BNSH\0\0\0\0", 8))) {
            const auto size = little_endian(b, at + 0x1c, 4);
            if (size < 0x88) throw InputError("BNSH declared size is too small", at);
            bnsh(slice(b, at, size), ancestry, at);
            at += size - 1;
            recognized = true;
        } else if (magic(b, at, "BAHS")) {
            if (!covers(b, at, 32) || little_endian(b, at + 4, 4) != 11) {
                if (!at) throw InputError("Unsupported or truncated BAHS header");
                ++inventory_.rejected_markers; continue;
            }
            const auto size = little_endian(b, at + 8, 4), start = little_endian(b, at + 28, 4);
            const auto codes = little_endian(b, at + 24, 4);
            if (size < 32 || !covers(b, at, size) || start < 32 || start > size || codes != size - start) {
                if (!at) throw InputError("Malformed BAHS header");
                ++inventory_.rejected_markers; continue;
            }
            bahs(slice(b, at, size), ancestry, at);
            at += size - 1;
            recognized = true;
        }
    }
    if (!recognized) ++inventory_.unrecognized_spans;
}
} // namespace armigo::shader_native
