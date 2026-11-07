// ============================================================================
// Armigo Runtime - Native Module Runtime
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

// NPDM admission and launch profile derivation.
//
// Admission follows Horizon 20.5.0. It validates META/ACID/ACI0 and the
// program-id interval before checking requested capabilities against the ACID
// catalogue. Capability descriptors identify core/priority, syscall, range,
// I/O-page, region and interrupt permissions. Duplicate singleton kinds are
// rejected; program type, kernel version, handles and debug flags are checked.
//
// Armigo supports a subset of valid process configurations. Physical mappings,
// interrupts, other address spaces and pools, authentication, and owner lists
// remain explicit unsupported results; they are not firmware-invalid inputs.
// No authentication is performed; see NpdmTrustPolicy.

#include <armigo/runtime/launch_profile.h>

#include "npdm.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace armigo::runtime {
namespace {

using armigo::loader::NpdmSection;

// Armigo currently represents four guest cores. Firmware capability IDs are
// virtual cores 0..63 (kernel 0xC82C), with a separate physical-core mapping.
constexpr std::uint32_t kGuestCoreCount = 4;

// Service names are at most eight bytes in the service manager namespace.
constexpr std::size_t kMaxServiceNameBytes = 8;

// FS 0x76F04/0x76F0C: ACI0 has a 0x1C-byte header; ACID has 0x2C bytes.
constexpr std::size_t kFsAccessBlockSize = 0x1C;
constexpr std::size_t kFsRestrictionHeaderSize = 0x2C;

constexpr std::uint32_t kAcidFlagProduction = 1u << 0;
constexpr std::uint32_t kAcidFlagPoolPartitionShift = 2;
constexpr std::uint32_t kAcidFlagPoolPartitionMask = 0xFu << kAcidFlagPoolPartitionShift;
constexpr std::uint32_t kAcidPoolPartitionApplication = 0;

constexpr std::uint8_t kMetaFlagIs64Bit = 1u << 0;
constexpr std::uint8_t kMetaAddressSpaceShift = 1;
constexpr std::uint8_t kMetaAddressSpaceMask = 7u << kMetaAddressSpaceShift;
constexpr std::uint8_t kMetaAddressSpace64Bit = 3u << kMetaAddressSpaceShift;
// Bits 4 and 5 (optimized memory allocation, device address space merge
// disable) only steer kernel placement policies the fixed 39-bit profile does
// not expose; they are accepted as no-ops. Bit 6 enables the alias region
// extra size (Loader 0x57D0..0x57E4). Bit 7 is rejected by the structural
// loader check at 0x3E9C, before this stage.
constexpr std::uint8_t kMetaFlagAliasRegionExtraSize = 1u << 6;

LaunchFailure failure(LaunchError code, std::string operation, std::string detail) {
    LaunchFailure result;
    result.stage = LaunchStage::profile;
    result.code = code;
    result.operation = std::move(operation);
    result.detail = std::move(detail);
    return result;
}

// A bounds-checked little-endian reader over one byte range.
class RangeReader {
public:
    explicit RangeReader(std::span<const std::byte> bytes) : bytes_(bytes) {}

    [[nodiscard]] bool read_u32(std::size_t offset, std::uint32_t& value) const {
        if (offset > bytes_.size() || bytes_.size() - offset < sizeof(std::uint32_t)) {
            return false;
        }
        value = 0;
        for (std::size_t index = 0; index < sizeof(std::uint32_t); ++index) {
            value |= static_cast<std::uint32_t>(bytes_[offset + index]) << (8 * index);
        }
        return true;
    }

    [[nodiscard]] bool read_u64(std::size_t offset, std::uint64_t& value) const {
        if (offset > bytes_.size() || bytes_.size() - offset < sizeof(std::uint64_t)) {
            return false;
        }
        value = 0;
        for (std::size_t index = 0; index < sizeof(std::uint64_t); ++index) {
            value |= static_cast<std::uint64_t>(bytes_[offset + index]) << (8 * index);
        }
        return true;
    }

private:
    std::span<const std::byte> bytes_;
};

[[nodiscard]] std::span<const std::byte> section_bytes(
    std::span<const std::byte> image, const NpdmSection& section) {
    if (std::uint64_t{section.offset} + section.size > image.size()) {
        return {};
    }
    return image.subspan(section.offset, section.size);
}

// Capability descriptor identities. A descriptor word carries its identity
// as a low bit pattern of id set bits, so the id is the position of the
// lowest clear bit: id = countr_zero(~word). The all-zero word names no
// capability, and the all-ones word is padding. Ids outside this list are
// rejected, never skipped.
enum class CapabilityId : std::uint32_t {
    invalid = 0,
    core_priority = 3,
    syscall_mask = 4,
    map_range = 6,
    map_io_page = 7,
    map_region = 10,
    interrupt_pair = 11,
    program_type = 13,
    kernel_version = 14,
    handle_table = 15,
    debug_flags = 16,
    padding = 32,
};

[[nodiscard]] CapabilityId capability_id(std::uint32_t word) {
    return static_cast<CapabilityId>(std::countr_zero(~word));
}

// The decoded capability content of one kernel access control list.
struct KernelCapabilitySet {
    bool has_core_priority = false;
    std::uint32_t min_thread_priority = 0; // numerically smallest allowed (best)
    std::uint32_t max_thread_priority = 0; // numerically largest allowed (worst)
    std::uint32_t min_core = 0;
    std::uint32_t max_core = 0;

    std::array<std::uint32_t, 8> syscall_slot_mask{};
    std::array<bool, 8> syscall_slot_present{};

    bool has_kernel_version = false;
    std::uint32_t kernel_version = 0; // packed descriptor word as written

    bool has_handle_table = false;
    std::uint32_t handle_table_size = 0;

    bool has_program_type = false;
    std::uint32_t program_type = 0;

    bool has_debug_flags = false;
    std::uint32_t debug_flags = 0;

    bool requests_memory_mapping = false;
    bool requests_interrupt_access = false;
};

// Kernel validation applies to the ACI0 request, after loader restrictions.
// ACID is a catalogue, not another process capability list (Loader 0x35A0).
[[nodiscard]] std::optional<LaunchFailure> decode_kernel_capabilities(
    std::span<const std::byte> section, KernelCapabilitySet& out) {
    // Loader 0x4F40/0x4F4C passes floor(size / 4) descriptors to the kernel.
    std::uint32_t once_flags = 0;
    std::uint32_t present_slots = 0;
    const std::size_t descriptor_count = section.size() / sizeof(std::uint32_t);
    RangeReader reader(section);

    for (std::size_t index = 0; index < descriptor_count; ++index) {
        std::uint32_t word = 0;
        static_cast<void>(reader.read_u32(index * sizeof(std::uint32_t), word));
        const CapabilityId id = capability_id(word);

        if (id == CapabilityId::padding) {
            continue;
        }

        // Initialize-once kinds reject a second descriptor outright.
        std::uint32_t once_bit = 0;
        switch (id) {
            case CapabilityId::core_priority: once_bit = 1u << 0; break;
            case CapabilityId::program_type: once_bit = 1u << 1; break;
            case CapabilityId::kernel_version: once_bit = 1u << 2; break;
            case CapabilityId::handle_table: once_bit = 1u << 3; break;
            case CapabilityId::debug_flags: once_bit = 1u << 4; break;
            default: break;
        }
        if (once_bit != 0) {
            if ((once_flags & once_bit) != 0) {
                return failure(LaunchError::invalid_input, "npdm.kernel_access_control",
                               "an initialize-once capability appears twice");
            }
            once_flags |= once_bit;
        }

        switch (id) {
            case CapabilityId::core_priority: {
                const std::uint32_t max_thread_priority = (word >> 4) & 0x3F;
                const std::uint32_t min_thread_priority = (word >> 10) & 0x3F;
                const std::uint32_t min_core = (word >> 16) & 0xFF;
                const std::uint32_t max_core = (word >> 24) & 0xFF;
                if (min_core > max_core || min_thread_priority > max_thread_priority) {
                    return failure(LaunchError::invalid_input, "npdm.kernel_access_control",
                                   "the thread info capability has an inverted range");
                }
                if (max_core >= 64) {
                    return failure(LaunchError::invalid_input, "npdm.kernel_access_control",
                                   "the thread info capability names core " +
                                       std::to_string(max_core) + " outside the 64 virtual-core IDs");
                }
                // Kernel 0xC8E4 rejects masks containing priorities 0..3.
                if (min_thread_priority < 4) {
                    return failure(LaunchError::invalid_input, "npdm.kernel_access_control",
                                   "the thread info capability reaches kernel-only priorities");
                }
                out.has_core_priority = true;
                out.min_thread_priority = min_thread_priority;
                out.max_thread_priority = max_thread_priority;
                out.min_core = min_core;
                out.max_core = max_core;
                break;
            }
            case CapabilityId::syscall_mask: {
                const std::uint32_t slot = (word >> 29) & 7;
                if ((present_slots & (1u << slot)) != 0) {
                    return failure(LaunchError::invalid_input, "npdm.kernel_access_control",
                                   "syscall mask slot " + std::to_string(slot) +
                                       " is described twice");
                }
                present_slots |= 1u << slot;
                out.syscall_slot_present[slot] = true;
                out.syscall_slot_mask[slot] = (word >> 5) & 0xFFFFFF;
                break;
            }
            case CapabilityId::map_range: {
                // A map range consumes the next descriptor as its size word.
                if (index + 1 >= descriptor_count) {
                    return failure(LaunchError::invalid_input, "npdm.kernel_access_control",
                                   "a map range descriptor has no size descriptor");
                }
                std::uint32_t size_word = 0;
                static_cast<void>(reader.read_u32((index + 1) * sizeof(std::uint32_t), size_word));
                if (capability_id(size_word) != CapabilityId::map_range) {
                    return failure(LaunchError::invalid_input, "npdm.kernel_access_control",
                                   "a map range size descriptor has a different kind");
                }
                if (((size_word >> 27) & 0xF) != 0) {
                    return failure(LaunchError::invalid_input, "npdm.kernel_access_control",
                                   "a map range size descriptor uses reserved bits");
                }
                if (((size_word >> 7) & 0xFFFFF) == 0) {
                    return failure(LaunchError::invalid_input, "npdm.kernel_access_control",
                                   "a map range descriptor maps zero pages");
                }
                const std::uint64_t address = std::uint64_t{(word >> 7) & 0xFFFFFF} << 12;
                const std::uint64_t size = std::uint64_t{(size_word >> 7) & 0xFFFFF} << 12;
                if (address + size > (std::uint64_t{1} << 36)) {
                    return failure(LaunchError::invalid_input, "npdm.kernel_access_control",
                                   "a map range exceeds the 36-bit physical address space");
                }
                ++index;
                out.requests_memory_mapping = true;
                break;
            }
            case CapabilityId::map_io_page: {
                out.requests_memory_mapping = true;
                break;
            }
            case CapabilityId::map_region: {
                const std::uint32_t region0 = (word >> 11) & 0x3F;
                const std::uint32_t region1 = (word >> 18) & 0x3F;
                const std::uint32_t region2 = (word >> 25) & 0x3F;
                if (region0 > 3 || region1 > 3 || region2 > 3) {
                    return failure(LaunchError::invalid_input, "npdm.kernel_access_control",
                                   "a map region descriptor names an unknown memory region");
                }
                if (region0 != 0 || region1 != 0 || region2 != 0) {
                    out.requests_memory_mapping = true;
                }
                break;
            }
            case CapabilityId::interrupt_pair: {
                const std::uint32_t first = (word >> 12) & 0x3FF;
                const std::uint32_t second = (word >> 22) & 0x3FF;
                // Kernel 0xC918..0xC92C also checks the hardware GIC limit.
                // Its maximum is 1020; actual interrupt grants remain unsupported.
                if ((first >= 1020 && first != 0x3FF) ||
                    (second >= 1020 && second != 0x3FF)) {
                    return failure(LaunchError::invalid_input, "npdm.kernel_access_control",
                                   "an interrupt ID exceeds the architectural interrupt limit");
                }
                // 0x3FF is the descriptor's padding id and requests nothing.
                if (first != 0x3FF || second != 0x3FF) {
                    out.requests_interrupt_access = true;
                }
                break;
            }
            case CapabilityId::program_type: {
                if ((word >> 17) != 0) {
                    return failure(LaunchError::invalid_input, "npdm.kernel_access_control",
                                   "a program type descriptor uses reserved bits");
                }
                out.has_program_type = true;
                out.program_type = (word >> 14) & 7;
                break;
            }
            case CapabilityId::kernel_version: {
                // The intended kernel version is a packed word; a zero major
                // version names no kernel at all.
                if ((word >> 19) == 0) {
                    return failure(LaunchError::invalid_input, "npdm.kernel_access_control",
                                   "a kernel version descriptor has no major version");
                }
                out.has_kernel_version = true;
                out.kernel_version = word;
                break;
            }
            case CapabilityId::handle_table: {
                if ((word >> 26) != 0) {
                    return failure(LaunchError::invalid_input, "npdm.kernel_access_control",
                                   "a handle table descriptor uses reserved bits");
                }
                out.has_handle_table = true;
                out.handle_table_size = (word >> 16) & 0x3FF;
                break;
            }
            case CapabilityId::debug_flags: {
                if ((word >> 20) != 0) {
                    return failure(LaunchError::invalid_input, "npdm.kernel_access_control",
                                   "a debug flags descriptor uses reserved bits");
                }
                const std::uint32_t flags = (word >> 17) & 7;
                // Kernel 0xCA44..0xCA58 admits at most one debug flag.
                if ((flags & (flags - 1)) != 0) {
                    return failure(LaunchError::invalid_input, "npdm.kernel_access_control",
                                   "a debug flags descriptor sets more than one flag");
                }
                out.has_debug_flags = true;
                out.debug_flags = flags;
                break;
            }
            default:
                return failure(LaunchError::invalid_input, "npdm.kernel_access_control",
                               "the capability list contains an unknown descriptor kind");
        }
    }
    return std::nullopt;
}

// Loader 0x35A0 scans the raw ACID catalogue for each ACI0 descriptor.
// Singleton searches use the first matching restriction; syscall and mapping
// searches may find any covering entry. ACID duplicates and broad debug grants
// are therefore legal even though the kernel rejects them in an ACI0 request.
[[nodiscard]] std::optional<LaunchFailure> check_within_acid_restrictions(
    std::span<const std::byte> request_bytes, std::span<const std::byte> restriction_bytes) {
    const auto words = [](std::span<const std::byte> bytes) {
        std::vector<std::uint32_t> result(bytes.size() / 4);
        RangeReader reader(bytes);
        for (std::size_t i = 0; i < result.size(); ++i) {
            static_cast<void>(reader.read_u32(i * 4, result[i]));
        }
        return result;
    };
    const auto requests = words(request_bytes);
    const auto restrictions = words(restriction_bytes);
    const auto first = [&restrictions](CapabilityId id) {
        const auto found = std::find_if(restrictions.begin(), restrictions.end(),
            [id](std::uint32_t word) { return capability_id(word) == id; });
        return found == restrictions.end() ? 0u : *found;
    };
    const auto any = [&restrictions](CapabilityId id, const auto& covers) {
        return std::any_of(restrictions.begin(), restrictions.end(),
            [&](std::uint32_t word) { return capability_id(word) == id && covers(word); });
    };
    const auto field = [](std::uint32_t word, unsigned shift, std::uint32_t mask) {
        return (word >> shift) & mask;
    };
    for (std::size_t index = 0; index < requests.size(); ++index) {
        const auto word = requests[index];
        const auto id = capability_id(word);
        const auto bound = first(id);
        bool covered = false;
        switch (id) {
            case CapabilityId::padding: covered = true; break;
            case CapabilityId::core_priority:
                covered = bound != 0 && field(word, 4, 63) <= field(bound, 4, 63) &&
                    field(word, 10, 63) >= field(bound, 10, 63) &&
                    field(word, 10, 63) <= field(word, 4, 63) &&
                    field(word, 16, 255) >= field(bound, 16, 255) &&
                    field(word, 24, 255) <= field(bound, 24, 255) &&
                    field(word, 16, 255) <= field(word, 24, 255);
                break;
            case CapabilityId::syscall_mask:
                covered = any(id, [&](auto limit) {
                    return word >> 29 == limit >> 29 &&
                        (field(word, 5, 0xFFFFFF) & ~field(limit, 5, 0xFFFFFF)) == 0;
                });
                break;
            case CapabilityId::map_range: {
                if (++index >= requests.size() || capability_id(requests[index]) != id) {
                    break;
                }
                const auto size_word = requests[index];
                if ((size_word & 0x78000000u) != 0) { break; }
                for (std::size_t i = 0; i < restrictions.size(); ++i) {
                    const auto address_limit = restrictions[i];
                    if (capability_id(address_limit) != id) { continue; }
                    if (++i >= restrictions.size() || capability_id(restrictions[i]) != id) {
                        break;
                    }
                    const auto size_limit = restrictions[i];
                    const auto start = field(word, 7, 0xFFFFFF);
                    const auto lower = field(address_limit, 7, 0xFFFFFF);
                    if ((size_limit & 0x78000000u) == 0 &&
                        (((word ^ address_limit) | (size_word ^ size_limit)) & 0x80000000u) == 0 &&
                        start >= lower && start + field(size_word, 7, 0xFFFFFF) <=
                            lower + field(size_limit, 7, 0xFFFFFF)) {
                        covered = true;
                        break;
                    }
                }
                break;
            }
            case CapabilityId::map_io_page:
                covered = any(id, [word](auto limit) { return word == limit; });
                break;
            case CapabilityId::map_region:
                covered = true;
                for (const unsigned shift : {11u, 18u, 25u}) {
                    const auto region = field(word, shift, 0x7F);
                    if ((region & 63) == 0) { continue; }
                    covered = covered && any(id, [&](auto limit) {
                        for (const unsigned position : {11u, 18u, 25u}) {
                            const auto grant = field(limit, position, 0x7F);
                            if ((region & 63) == (grant & 63) &&
                                ((region & 64) != 0 || (grant & 64) == 0)) { return true; }
                        }
                        return false;
                    });
                }
                break;
            case CapabilityId::interrupt_pair:
                covered = true;
                for (const unsigned shift : {12u, 22u}) {
                    const auto interrupt = field(word, shift, 0x3FF);
                    if (interrupt == 0x3FF) { continue; }
                    covered = covered && any(id, [&](auto limit) {
                        const auto a = field(limit, 12, 0x3FF);
                        const auto b = field(limit, 22, 0x3FF);
                        return (a == 0x3FF && b == 0x3FF) || a == interrupt || b == interrupt;
                    });
                }
                break;
            case CapabilityId::program_type:
            case CapabilityId::kernel_version:
                covered = word == bound;
                break;
            case CapabilityId::handle_table:
                covered = bound != 0 && field(word, 16, 0x3FF) <= field(bound, 16, 0x3FF);
                break;
            case CapabilityId::debug_flags:
                covered = (word & ~bound) == 0;
                break;
            default:
                return failure(LaunchError::invalid_input, "npdm.capability_restriction",
                               "the capability list contains an unknown descriptor kind");
        }
        if (!covered) {
            return failure(LaunchError::permission_denied, "npdm.capability_restriction",
                           "the ACID restrictions do not cover capability kind " +
                               std::to_string(static_cast<std::uint32_t>(id)));
        }
    }
    return std::nullopt;
}

struct ServiceAccessEntry {
    std::array<char, kMaxServiceNameBytes> name{};
    std::size_t length = 0;
    bool host = false;
};

[[nodiscard]] std::optional<LaunchFailure> parse_service_access_control(
    std::span<const std::byte> section, std::vector<ServiceAccessEntry>& entries) {
    std::size_t position = 0;
    while (position < section.size()) {
        const std::uint8_t control = static_cast<std::uint8_t>(section[position]);
        const std::size_t length = (control & 7) + 1;
        // SM 0x1930 and 0x1B20 stop at an incomplete trailing entry.
        if (section.size() - position < length + 1) {
            break;
        }
        ServiceAccessEntry entry;
        entry.length = length;
        entry.host = (control & 0x80) != 0;
        for (std::size_t index = 0; index < length; ++index) {
            const char character = static_cast<char>(section[position + 1 + index]);
            entry.name[index] = character;
        }
        entries.push_back(std::move(entry));
        position += length + 1;
    }
    return std::nullopt;
}

[[nodiscard]] bool names_equal(const ServiceAccessEntry& left, const ServiceAccessEntry& right) {
    // SM compares zero-padded 64-bit names, including embedded zero bytes.
    return left.name == right.name;
}

[[nodiscard]] bool is_wildcard(const ServiceAccessEntry& entry) {
    return entry.length != 0 && entry.name[entry.length - 1] == '*';
}

// SM 20.5.0 0x1880/0x1AC0: a request entry is covered when a restriction entry
// with the same host flag matches it exactly, or fuzzy-matches it with a
// wildcard whose prefix agrees (the '*' byte itself excluded).
[[nodiscard]] bool service_allowed_by(const std::vector<ServiceAccessEntry>& restrictions,
                                      const ServiceAccessEntry& entry) {
    const bool entry_wildcard = is_wildcard(entry);
    for (const ServiceAccessEntry& restriction : restrictions) {
        if (restriction.host != entry.host) {
            continue;
        }
        const bool restriction_wildcard = is_wildcard(restriction);
        if (restriction_wildcard == entry_wildcard) {
            if (names_equal(restriction, entry)) {
                return true;
            }
        } else if (restriction_wildcard) {
            const std::size_t prefix = restriction.length - 1;
            if (std::equal(restriction.name.begin(),
                           restriction.name.begin() + static_cast<std::ptrdiff_t>(prefix),
                           entry.name.begin())) {
                return true;
            }
        }
    }
    return false;
}

struct FsAccessBlock {
    std::uint8_t version = 0;
    std::uint64_t permissions = 0;
};

// FS 20.5.0 0x76EC0 consumes both records. Version is one byte; unequal or
// zero versions produce empty access (0x76F14..0x76F28). Permissions intersect
// at 0x76F70. Armigo selects the non-debug access configuration, masking bits
// 62 and 63 (0x76F34..0x76F78, 0xA43D0). The external configuration value and
// privileged internal registration are outside this profile's verified scope.
// ACI0 owner payload pairs are offset/size at +0xC/+0x10 and +0x14/+0x18.
// ACID counts at +1/+2 describe lists after +0x2C; its four qwords at +0xC
// are owner ID range restrictions. Unrequested owner grants need no profile
// state. Requested owner payloads require a future owner-aware FS contract.
[[nodiscard]] std::optional<LaunchFailure> derive_fs_access(
    std::span<const std::byte> request, std::span<const std::byte> restriction,
    FsAccessBlock& out) {
    if (request.size() < kFsAccessBlockSize || restriction.size() < kFsRestrictionHeaderSize) {
        return failure(LaunchError::invalid_input, "npdm.file_system_access",
                       "a file-system access record is shorter than its fixed header");
    }
    out.version = static_cast<std::uint8_t>(request[0]);
    if (out.version == 0 || out.version != static_cast<std::uint8_t>(restriction[0])) {
        return std::nullopt;
    }
    RangeReader reader(request);
    const std::size_t owner_count = static_cast<std::uint8_t>(restriction[1]) +
                                    static_cast<std::uint8_t>(restriction[2]);
    if (owner_count > (restriction.size() - kFsRestrictionHeaderSize) / 8) {
        return failure(LaunchError::invalid_input, "npdm.file_system_access",
                       "the ACID owner lists extend past their access record");
    }
    for (const std::size_t field : {0xCu, 0x14u}) {
        std::uint32_t offset = 0;
        std::uint32_t size = 0;
        static_cast<void>(reader.read_u32(field, offset));
        static_cast<void>(reader.read_u32(field + 4, size));
        if (offset > request.size() || size > request.size() - offset) {
            return failure(LaunchError::invalid_input, "npdm.file_system_access",
                           "an ACI0 owner payload extends past its access record");
        }
        if (size != 0) {
            return failure(LaunchError::unsupported, "npdm.file_system_access",
                           "owner payloads require an owner-aware file-system profile");
        }
    }
    std::uint64_t requested = 0;
    std::uint64_t granted = 0;
    static_cast<void>(reader.read_u64(4, requested));
    static_cast<void>(RangeReader(restriction).read_u64(4, granted));
    out.permissions = requested & granted & 0x3FFFFFFFFFFFFFFFULL;
    return std::nullopt;
}

} // namespace

LaunchResult<LaunchProfile> derive_launch_profile(std::span<const std::byte> npdm,
                                                  NpdmTrustPolicy trust) {
    // No signature verifier exists in this build. Authentication is never
    // silently downgraded to structural validation.
    if (trust == NpdmTrustPolicy::require_authenticated) {
        return failure(LaunchError::unsupported, "npdm.trust_policy",
                       "no NPDM signature verifier exists, so an authenticated profile "
                       "cannot be derived");
    }

    const auto parsed = armigo::loader::parse_npdm(npdm);
    if (std::holds_alternative<armigo::loader::NpdmParseFailure>(parsed)) {
        const auto& parse_failure = std::get<armigo::loader::NpdmParseFailure>(parsed);
        return failure(LaunchError::invalid_input, parse_failure.operation,
                       parse_failure.detail);
    }
    const auto& meta = std::get<armigo::loader::ParsedNpdm>(parsed);

    // This profile models a production application. Loader 0x3F3C conditionally
    // requires the production bit on retail systems. Bit 1 only relaxes code
    // signature admission (0x5174), outside the explicit extracted-input trust
    // policy. Other flag bits are not rejected by the loader.
    if ((meta.acid_flags & kAcidFlagProduction) == 0) {
        return failure(LaunchError::unsupported, "npdm.acid_flags",
                       "the profile requires production metadata");
    }
    const auto pool = (meta.acid_flags & kAcidFlagPoolPartitionMask) >>
                      kAcidFlagPoolPartitionShift;
    // Loader 0x5480 admits pool values 0..3.
    if (pool > 3) {
        return failure(LaunchError::invalid_input, "npdm.acid_flags",
                       "the ACID names an invalid memory pool partition");
    }
    if (pool != kAcidPoolPartitionApplication) {
        return failure(LaunchError::unsupported, "npdm.acid_flags",
                       "the ACID names a non-application memory pool partition");
    }

    // The ACI0 program id must sit inside the signed ACID range.
    if (meta.aci0_program_id < meta.acid_program_id_min ||
        meta.aci0_program_id > meta.acid_program_id_max) {
        return failure(LaunchError::permission_denied, "npdm.program_id_restriction",
                       "the ACI0 program id lies outside the ACID title id range");
    }

    // META flags: only the 64-bit 39-bit-address-space profile exists.
    if ((meta.meta_flags & kMetaAddressSpaceMask) != kMetaAddressSpace64Bit) {
        return failure(LaunchError::unsupported, "npdm.address_space",
                       "the metadata requests an address space other than the supported "
                       "64-bit 39-bit profile");
    }
    if ((meta.meta_flags & kMetaFlagIs64Bit) == 0) {
        return failure(LaunchError::invalid_input, "npdm.meta_flags",
                       "the metadata marks a 32-bit process with a 64-bit address space");
    }
    // Loader 0x5868..0x58EC requires 2 MiB granularity and at most 0x1FE00000.
    if (meta.system_resource_size % 0x200000 != 0 ||
        meta.system_resource_size > 0x1FE00000) {
        return failure(LaunchError::invalid_input, "npdm.system_resource",
                       "the system resource size is not 2 MiB aligned or exceeds the loader limit");
    }
    // Armigo's initial-stack contract requires explicit page-aligned storage.
    // The kernel StartProcess allocation/rounding path is not verified here.
    if (meta.main_thread_stack_size == 0 ||
        meta.main_thread_stack_size % horizon::guest_page_size != 0) {
        return failure(LaunchError::invalid_input, "npdm.stack_size",
                       "the main thread stack size is zero or not guest-page aligned");
    }

    // Loader restrictions precede kernel interpretation; ACID is not parsed
    // as a process request. This preserves first-match and catalogue semantics.
    const auto request_caps = section_bytes(npdm, meta.aci0_kernel_access_control);
    if (auto check = check_within_acid_restrictions(
            request_caps, section_bytes(npdm, meta.acid_kernel_access_control))) {
        return *check;
    }
    KernelCapabilitySet aci0_capabilities;
    if (auto check = decode_kernel_capabilities(request_caps, aci0_capabilities)) {
        return *check;
    }
    const auto supported_cores = meta.platform == GuestPlatform::switch2 ? 6u : kGuestCoreCount;
    if (aci0_capabilities.max_core >= supported_cores) {
        return failure(LaunchError::unsupported, "npdm.capability_support",
                       "the requested virtual cores exceed the selected platform profile");
    }
    if (aci0_capabilities.requests_memory_mapping) {
        return failure(LaunchError::unsupported, "npdm.capability_support",
                       "the capability list requests physical memory mappings, which no "
                       "launch profile can represent");
    }
    if (aci0_capabilities.requests_interrupt_access) {
        return failure(LaunchError::unsupported, "npdm.capability_support",
                       "the capability list requests hardware interrupt access, which no "
                       "launch profile can represent");
    }
    // Armigo requires explicit quantities for this profile; firmware defaults
    // for omitted capabilities are not a supported alternative profile.
    if (!aci0_capabilities.has_core_priority || !aci0_capabilities.has_kernel_version ||
        !aci0_capabilities.has_handle_table) {
        return failure(LaunchError::invalid_input, "npdm.absent_metadata",
                       "the capability list is missing the required thread info, kernel "
                       "version, or handle table capability");
    }
    if (!aci0_capabilities.has_program_type || aci0_capabilities.program_type != 1) {
        // Loader 0x3BC0 defaults an absent program type to zero.
        return failure(LaunchError::unsupported, "npdm.capability_support",
                       "the capability list does not select the supported application profile");
    }

    // FS and SM enforce their own restrictions; Loader 0x2F8C..0x30C0
    // transports their ACID and ACI0 records without comparing the payloads.
    FsAccessBlock aci0_fs;
    if (auto check = derive_fs_access(section_bytes(npdm, meta.aci0_fs_access_header),
                                     section_bytes(npdm, meta.acid_fs_access_control), aci0_fs)) {
        return *check;
    }
    const std::uint64_t effective_fs_permissions = aci0_fs.permissions;

    // SM 0x18A0 limits the stored request SAC to 0x200 bytes.
    if (meta.aci0_service_access_control.size > 0x200) {
        return failure(LaunchError::invalid_input, "npdm.service_access_control",
                       "the service access request exceeds the service manager limit");
    }
    // Service access: every complete ACI0 entry must be covered by an ACID entry.
    std::vector<ServiceAccessEntry> acid_services;
    if (auto check = parse_service_access_control(
            section_bytes(npdm, meta.acid_service_access_control), acid_services)) {
        return *check;
    }
    std::vector<ServiceAccessEntry> aci0_services;
    if (auto check = parse_service_access_control(
            section_bytes(npdm, meta.aci0_service_access_control), aci0_services)) {
        return *check;
    }
    for (const ServiceAccessEntry& entry : aci0_services) {
        if (!service_allowed_by(acid_services, entry)) {
            return failure(LaunchError::permission_denied, "npdm.service_restriction",
                           "the ACI0 requests a service the ACID does not grant");
        }
    }

    // The main thread must sit inside the admitted ranges.
    if (meta.main_thread_priority < aci0_capabilities.min_thread_priority ||
        meta.main_thread_priority > aci0_capabilities.max_thread_priority) {
        return failure(LaunchError::invalid_input, "npdm.main_thread",
                       "the main thread priority lies outside the admitted priority range");
    }
    if (meta.main_thread_ideal_core < aci0_capabilities.min_core ||
        meta.main_thread_ideal_core > aci0_capabilities.max_core) {
        return failure(LaunchError::invalid_input, "npdm.main_thread",
                       "the main thread ideal core lies outside the admitted core range");
    }

    // Assemble the profile. Placement, entropy and accounting remain runtime
    // decisions; only the NPDM-admitted quantities are written here.
    LaunchProfile profile;
    profile.origin = ProfileOrigin::npdm;
    profile.title_id = meta.aci0_program_id;
    profile.is_64_bit = true;
    profile.process.platform = meta.platform;
    if (meta.platform == GuestPlatform::switch2) profile.process.baseline = horizon::Baseline::horizon_23_0_1;
    profile.process.address_bits = 39;
    profile.process.alias_extra_enabled =
        (meta.meta_flags & kMetaFlagAliasRegionExtraSize) != 0;
    profile.process.private_system_resource_bytes = meta.system_resource_size;
    std::uint64_t core_mask = 0;
    for (std::uint32_t core = aci0_capabilities.min_core;
         core <= aci0_capabilities.max_core; ++core) {
        core_mask |= std::uint64_t{1} << core;
    }
    profile.process.allowed_core_mask = core_mask;

    for (std::size_t slot = 0; slot < aci0_capabilities.syscall_slot_present.size(); ++slot) {
        if (!aci0_capabilities.syscall_slot_present[slot]) {
            continue;
        }
        for (std::uint32_t bit = 0; bit < 24; ++bit) {
            if ((aci0_capabilities.syscall_slot_mask[slot] & (1u << bit)) != 0) {
                profile.permissions.supervisor_calls.push_back(
                    static_cast<std::uint16_t>(24 * slot + bit));
            }
        }
    }

    const auto aci0_service_bytes = section_bytes(npdm, meta.aci0_service_access_control);
    profile.permissions.service_access_control.assign(aci0_service_bytes.begin(),
                                                      aci0_service_bytes.end());

    // Effective file-system access in the same 0x1C-byte header shape: the
    // ACI0 version with the permission mask intersected with the ACID grant.
    std::array<std::byte, kFsAccessBlockSize> fs_block{};
    const auto write_u32 = [&fs_block](std::size_t offset, std::uint32_t value) {
        for (std::size_t index = 0; index < sizeof(std::uint32_t); ++index) {
            fs_block[offset + index] =
                static_cast<std::byte>((value >> (8 * index)) & 0xFF);
        }
    };
    const auto write_u64 = [&fs_block](std::size_t offset, std::uint64_t value) {
        for (std::size_t index = 0; index < sizeof(std::uint64_t); ++index) {
            fs_block[offset + index] =
                static_cast<std::byte>((value >> (8 * index)) & 0xFF);
        }
    };
    write_u32(0x00, aci0_fs.version);
    write_u64(0x04, effective_fs_permissions);
    write_u32(0x0C, static_cast<std::uint32_t>(kFsAccessBlockSize));
    write_u32(0x10, 0);
    write_u32(0x14, static_cast<std::uint32_t>(kFsAccessBlockSize));
    profile.permissions.file_system_access_control.assign(fs_block.begin(), fs_block.end());

    profile.permissions.handle_table_capacity = aci0_capabilities.handle_table_size;
    profile.permissions.intended_kernel_version = aci0_capabilities.kernel_version;
    profile.permissions.program_type = aci0_capabilities.program_type;
    profile.permissions.debug_flags = aci0_capabilities.debug_flags;
    profile.permissions.highest_priority = aci0_capabilities.min_thread_priority;
    profile.permissions.lowest_priority = aci0_capabilities.max_thread_priority;
    profile.main_thread.priority = meta.main_thread_priority;
    profile.main_thread.ideal_core = static_cast<std::int32_t>(meta.main_thread_ideal_core);
    profile.main_thread.stack_size = meta.main_thread_stack_size;

    return profile;
}

} // namespace armigo::runtime
