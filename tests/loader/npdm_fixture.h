// ============================================================================
// Armigo Tests - Native Runtime Verification
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

// Independently authored synthetic NPDM byte arrays for the loader and
// runtime profile tests. Every image is assembled from scratch here; no
// bytes derive from any real title. The builder keeps the raw byte vector
// reachable so tests can surgically corrupt fields, and mirrors the fixed
// field offsets the parser reads.

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <string>
#include <utility>
#include <vector>

namespace npdm_fixture {

using Bytes = std::vector<std::byte>;

// Fixed META header offsets, exposed for surgical corruption tests.
inline constexpr std::size_t kMetaFlags = 0x0C;
inline constexpr std::size_t kMetaMainThreadPriority = 0x0E;
inline constexpr std::size_t kMetaMainThreadIdealCore = 0x0F;
inline constexpr std::size_t kMetaSystemResource = 0x14;
inline constexpr std::size_t kMetaVersion = 0x18;
inline constexpr std::size_t kMetaStackSize = 0x1C;
inline constexpr std::size_t kMetaAci0Offset = 0x70;
inline constexpr std::size_t kMetaAci0Size = 0x74;
inline constexpr std::size_t kMetaAcidOffset = 0x78;
inline constexpr std::size_t kMetaAcidSize = 0x7C;

inline void put_u8(Bytes& bytes, std::size_t offset, std::uint8_t value) {
    bytes[offset] = static_cast<std::byte>(value);
}

inline void put_u32(Bytes& bytes, std::size_t offset, std::uint32_t value) {
    for (std::size_t index = 0; index < 4; ++index) {
        bytes[offset + index] = static_cast<std::byte>((value >> (8 * index)) & 0xFF);
    }
}

inline void put_u64(Bytes& bytes, std::size_t offset, std::uint64_t value) {
    for (std::size_t index = 0; index < 8; ++index) {
        bytes[offset + index] = static_cast<std::byte>((value >> (8 * index)) & 0xFF);
    }
}

inline Bytes bytes_of(std::initializer_list<int> values) {
    Bytes bytes;
    bytes.reserve(values.size());
    for (int value : values) {
        bytes.push_back(static_cast<std::byte>(value & 0xFF));
    }
    return bytes;
}

inline Bytes raw_bytes(std::size_t count, std::uint8_t fill = 0) {
    return Bytes(count, static_cast<std::byte>(fill));
}

// Capability descriptor builders, one field position per firmware encoding.
// Each descriptor carries its identity as id low set bits, i.e. the
// pattern (1 << id) - 1, followed by the payload fields.
inline constexpr std::uint32_t cap_padding() {
    return 0xFFFFFFFFu;
}

inline constexpr std::uint32_t cap_core_priority(std::uint32_t worst_priority,
                                                 std::uint32_t best_priority,
                                                 std::uint32_t min_core,
                                                 std::uint32_t max_core) {
    return ((1u << 3) - 1) | (worst_priority << 4) | (best_priority << 10) | (min_core << 16) |
           (max_core << 24);
}

inline constexpr std::uint32_t cap_syscall_mask(std::uint32_t slot, std::uint32_t mask) {
    return ((1u << 4) - 1) | ((mask & 0xFFFFFF) << 5) | ((slot & 7) << 29);
}

inline constexpr std::uint32_t cap_map_range_address(std::uint32_t address_pages,
                                                     bool read_only) {
    return ((1u << 6) - 1) | ((address_pages & 0xFFFFFF) << 7) | (read_only ? 1u << 31 : 0);
}

inline constexpr std::uint32_t cap_map_range_size(std::uint32_t pages, bool normal) {
    return ((1u << 6) - 1) | ((pages & 0xFFFFF) << 7) | (normal ? 1u << 31 : 0);
}

inline constexpr std::uint32_t cap_map_io_page(std::uint32_t address_pages) {
    return ((1u << 7) - 1) | ((address_pages & 0xFFFFFF) << 8);
}

inline constexpr std::uint32_t cap_map_region(std::uint32_t region0, std::uint32_t region1,
                                              std::uint32_t region2) {
    return ((1u << 10) - 1) | ((region0 & 0x3F) << 11) | ((region1 & 0x3F) << 18) |
           ((region2 & 0x3F) << 25);
}

inline constexpr std::uint32_t cap_interrupt_pair(std::uint32_t first, std::uint32_t second) {
    return ((1u << 11) - 1) | ((first & 0x3FF) << 12) | ((second & 0x3FF) << 22);
}

inline constexpr std::uint32_t cap_program_type(std::uint32_t type) {
    return ((1u << 13) - 1) | ((type & 7) << 14);
}

inline constexpr std::uint32_t cap_kernel_version(std::uint32_t major, std::uint32_t minor) {
    return ((1u << 14) - 1) | ((minor & 0xF) << 15) | ((major & 0x1FFF) << 19);
}

inline constexpr std::uint32_t cap_handle_table(std::uint32_t size) {
    return ((1u << 15) - 1) | ((size & 0x3FF) << 16);
}

inline constexpr std::uint32_t cap_debug_flags(bool allow_debug, bool force_debug_prod,
                                               bool force_debug) {
    return ((1u << 16) - 1) | (allow_debug ? 1u << 17 : 0) | (force_debug_prod ? 1u << 18 : 0) |
           (force_debug ? 1u << 19 : 0);
}

inline Bytes capability_words(std::initializer_list<std::uint32_t> words) {
    Bytes bytes;
    for (std::uint32_t word : words) {
        for (std::size_t index = 0; index < 4; ++index) {
            bytes.push_back(static_cast<std::byte>((word >> (8 * index)) & 0xFF));
        }
    }
    return bytes;
}

// One service access control entry: name plus host (register) flag.
using ServiceEntry = std::pair<const char*, bool>;

inline Bytes service_access_control(std::initializer_list<ServiceEntry> entries) {
    Bytes bytes;
    for (const ServiceEntry& entry : entries) {
        const std::string name(entry.first);
        bytes.push_back(static_cast<std::byte>((name.size() - 1) | (entry.second ? 0x80 : 0)));
        for (char character : name) {
            bytes.push_back(static_cast<std::byte>(character));
        }
    }
    return bytes;
}

// The fixed 0x1C-byte file-system access header used by the ACI0: version,
// permission mask, and three size words. `extra_tail` appends bytes past the
// standard size.
inline Bytes fs_access_block(std::uint32_t version, std::uint64_t permissions,
                             std::size_t extra_tail = 0) {
    Bytes bytes = raw_bytes(0x1C + extra_tail);
    put_u32(bytes, 0x00, version);
    put_u64(bytes, 0x04, permissions);
    put_u32(bytes, 0x0C, 0x1C);
    put_u32(bytes, 0x10, 0);
    put_u32(bytes, 0x14, 0x1C);
    return bytes;
}

// The fixed 0x2C-byte ACID file-system access header: a version byte and
// owner-count fields, permission mask, and owner ranges. This synthetic
// fixture leaves every owner count and range zero (the FS module).
inline Bytes acid_fs_access_block(std::uint32_t version, std::uint64_t permissions) {
    Bytes bytes = raw_bytes(0x2C);
    put_u8(bytes, 0x00, static_cast<std::uint8_t>(version));
    put_u64(bytes, 0x04, permissions);
    return bytes;
}

// Synthetic program id range; no real title uses this block.
inline constexpr std::uint64_t kProgramIdMin = 0x0200000000000000;
inline constexpr std::uint64_t kProgramIdMax = 0x0200000000000FFF;
inline constexpr std::uint64_t kProgramId = 0x0200000000000A42;

struct Parts {
    bool switch2 = false;
    std::uint8_t meta_flags = 0x07; // 64-bit process, address space type 3 (39-bit)
    std::uint8_t main_priority = 44;
    std::uint8_t main_core = 1;
    std::uint32_t system_resource = 0x200000;
    std::uint32_t stack_size = 0x40000;
    std::uint32_t meta_version = 0;
    std::uint64_t program_id = kProgramId;
    std::uint64_t program_id_min = kProgramIdMin;
    std::uint64_t program_id_max = kProgramIdMax;
    std::uint32_t acid_flags = 0x01; // production, application pool partition
    std::uint32_t fs_version = 1;
    std::uint64_t aci0_fs_permissions = 1;
    std::uint64_t acid_fs_permissions = 3;
    // Requested FS owners after the ACI0 header, and the ACID's owner lists
    // and its content/save-data owner id ranges (min, max, min, max).
    std::vector<std::uint64_t> aci0_content_owners;
    std::vector<std::pair<std::uint64_t, std::uint8_t>> aci0_save_owners;
    std::vector<std::uint64_t> acid_content_owners;
    std::vector<std::uint64_t> acid_save_owners;
    std::array<std::uint64_t, 4> acid_owner_ranges{};
    std::vector<ServiceEntry> aci0_services{{"fsp-srv", false}, {"acc:u0", false}};
    std::vector<ServiceEntry> acid_services{{"fsp-srv", false}, {"acc:u0", false}};
    std::vector<std::uint32_t> aci0_capabilities{
        cap_core_priority(59, 28, 0, 2),
        cap_syscall_mask(0, 0x31),
        cap_syscall_mask(1, 0x2),
        cap_handle_table(512),
        cap_kernel_version(20, 5),
        cap_padding(),
        cap_program_type(1),
    };
    std::vector<std::uint32_t> acid_capabilities{
        cap_core_priority(59, 28, 0, 3),
        cap_syscall_mask(0, 0x31),
        cap_syscall_mask(1, 0x2),
        cap_handle_table(1000),
        cap_kernel_version(20, 5),
        cap_program_type(1),
    };
};

struct NpdmImage {
    Bytes bytes;

    // A bare META header; sections can be appended or replaced afterwards.
    NpdmImage() : bytes(raw_bytes(0x80)) {
        put_u32(bytes, 0x00, 0x4154454Du); // "META"
    }
};

inline void put_u32_at(NpdmImage& image, std::size_t offset, std::uint32_t value) {
    put_u32(image.bytes, offset, value);
}

// Assembles a complete NPDM from the given parts: fixed META header, the
// ACI0 section (header plus its three subsections) at 0x80, and the ACID
// section afterwards.
inline NpdmImage build_npdm(const Parts& parts) {
    NpdmImage image;
    put_u8(image.bytes, kMetaFlags, parts.meta_flags);
    put_u8(image.bytes, kMetaMainThreadPriority, parts.main_priority);
    put_u8(image.bytes, kMetaMainThreadIdealCore, parts.main_core);
    put_u32(image.bytes, kMetaSystemResource, parts.system_resource);
    put_u32(image.bytes, kMetaVersion, parts.meta_version);
    put_u32(image.bytes, kMetaStackSize, parts.stack_size);

    Bytes aci0 = raw_bytes(0x40);
    put_u32(aci0, 0x00, 0x30494341u); // "ACI0"
    put_u64(aci0, 0x10, parts.program_id);
    Bytes aci0_fs = fs_access_block(parts.fs_version, parts.aci0_fs_permissions);
    if (!parts.aci0_content_owners.empty()) {
        put_u32(aci0_fs, 0x0C, static_cast<std::uint32_t>(aci0_fs.size()));
        put_u32(aci0_fs, 0x10, static_cast<std::uint32_t>(4 + 8 * parts.aci0_content_owners.size()));
        Bytes payload = raw_bytes(4 + 8 * parts.aci0_content_owners.size());
        put_u32(payload, 0, static_cast<std::uint32_t>(parts.aci0_content_owners.size()));
        for (std::size_t i = 0; i < parts.aci0_content_owners.size(); ++i)
            put_u64(payload, 4 + 8 * i, parts.aci0_content_owners[i]);
        aci0_fs.insert(aci0_fs.end(), payload.begin(), payload.end());
    }
    if (!parts.aci0_save_owners.empty()) {
        const std::size_t count = parts.aci0_save_owners.size();
        const std::size_t ids = (4 + count + 3) & ~std::size_t{3};
        put_u32(aci0_fs, 0x14, static_cast<std::uint32_t>(aci0_fs.size()));
        put_u32(aci0_fs, 0x18, static_cast<std::uint32_t>(ids + 8 * count));
        Bytes payload = raw_bytes(ids + 8 * count);
        put_u32(payload, 0, static_cast<std::uint32_t>(count));
        for (std::size_t i = 0; i < count; ++i) {
            put_u8(payload, 4 + i, parts.aci0_save_owners[i].second);
            put_u64(payload, ids + 8 * i, parts.aci0_save_owners[i].first);
        }
        aci0_fs.insert(aci0_fs.end(), payload.begin(), payload.end());
    }
    Bytes aci0_sac = service_access_control({});
    for (const ServiceEntry& entry : parts.aci0_services) {
        Bytes entry_bytes = service_access_control({entry});
        aci0_sac.insert(aci0_sac.end(), entry_bytes.begin(), entry_bytes.end());
    }
    Bytes aci0_kac = capability_words({});
    for (std::uint32_t word : parts.aci0_capabilities) {
        Bytes word_bytes = capability_words({word});
        aci0_kac.insert(aci0_kac.end(), word_bytes.begin(), word_bytes.end());
    }
    const std::uint32_t aci0_fs_offset = 0x40;
    const std::uint32_t aci0_sac_offset = aci0_fs_offset + static_cast<std::uint32_t>(aci0_fs.size());
    const std::uint32_t aci0_kac_offset =
        (aci0_sac_offset + static_cast<std::uint32_t>(aci0_sac.size()) + 3) & ~3u;
    put_u32(aci0, 0x20, aci0_fs_offset);
    put_u32(aci0, 0x24, static_cast<std::uint32_t>(aci0_fs.size()));
    put_u32(aci0, 0x28, aci0_sac_offset);
    put_u32(aci0, 0x2C, static_cast<std::uint32_t>(aci0_sac.size()));
    put_u32(aci0, 0x30, aci0_kac_offset);
    put_u32(aci0, 0x34, static_cast<std::uint32_t>(aci0_kac.size()));
    aci0.insert(aci0.end(), aci0_fs.begin(), aci0_fs.end());
    aci0.insert(aci0.end(), aci0_sac.begin(), aci0_sac.end());
    aci0.resize(aci0_kac_offset);
    aci0.insert(aci0.end(), aci0_kac.begin(), aci0_kac.end());

    const std::uint32_t body = parts.switch2 ? 0x300 : 0x200;
    Bytes acid = raw_bytes(body + 0x40);
    put_u32(acid, body + 0x00, 0x44494341u); // "ACID"
    put_u32(acid, body + 0x0C, parts.acid_flags);
    put_u64(acid, body + 0x10, parts.program_id_min);
    put_u64(acid, body + 0x18, parts.program_id_max);
    Bytes acid_fs = acid_fs_access_block(parts.fs_version, parts.acid_fs_permissions);
    put_u8(acid_fs, 0x01, static_cast<std::uint8_t>(parts.acid_content_owners.size()));
    put_u8(acid_fs, 0x02, static_cast<std::uint8_t>(parts.acid_save_owners.size()));
    for (std::size_t i = 0; i < parts.acid_owner_ranges.size(); ++i)
        put_u64(acid_fs, 0x0C + 8 * i, parts.acid_owner_ranges[i]);
    for (const auto* list : {&parts.acid_content_owners, &parts.acid_save_owners}) {
        for (const std::uint64_t id : *list) {
            Bytes entry = raw_bytes(8);
            put_u64(entry, 0, id);
            acid_fs.insert(acid_fs.end(), entry.begin(), entry.end());
        }
    }
    Bytes acid_sac;
    for (const ServiceEntry& entry : parts.acid_services) {
        Bytes entry_bytes = service_access_control({entry});
        acid_sac.insert(acid_sac.end(), entry_bytes.begin(), entry_bytes.end());
    }
    Bytes acid_kac;
    for (std::uint32_t word : parts.acid_capabilities) {
        Bytes word_bytes = capability_words({word});
        acid_kac.insert(acid_kac.end(), word_bytes.begin(), word_bytes.end());
    }
    const std::uint32_t acid_fs_offset = body + 0x40;
    const std::uint32_t acid_sac_offset = acid_fs_offset + static_cast<std::uint32_t>(acid_fs.size());
    const std::uint32_t acid_kac_offset =
        (acid_sac_offset + static_cast<std::uint32_t>(acid_sac.size()) + 3) & ~3u;
    put_u32(acid, body + 0x20, acid_fs_offset);
    put_u32(acid, body + 0x24, static_cast<std::uint32_t>(acid_fs.size()));
    put_u32(acid, body + 0x28, acid_sac_offset);
    put_u32(acid, body + 0x2C, static_cast<std::uint32_t>(acid_sac.size()));
    put_u32(acid, body + 0x30, acid_kac_offset);
    put_u32(acid, body + 0x34, static_cast<std::uint32_t>(acid_kac.size()));
    acid.insert(acid.end(), acid_fs.begin(), acid_fs.end());
    acid.insert(acid.end(), acid_sac.begin(), acid_sac.end());
    acid.resize(acid_kac_offset);
    acid.insert(acid.end(), acid_kac.begin(), acid_kac.end());

    put_u32(acid, body + 0x04, static_cast<std::uint32_t>(acid.size()) - (parts.switch2 ? 0x180 : 0x100));

    const std::uint32_t aci0_offset = 0x80;
    const std::uint32_t acid_offset =
        (aci0_offset + static_cast<std::uint32_t>(aci0.size()) + 7) & ~7u;
    put_u32(image.bytes, kMetaAci0Offset, aci0_offset);
    put_u32(image.bytes, kMetaAci0Size, static_cast<std::uint32_t>(aci0.size()));
    put_u32(image.bytes, kMetaAcidOffset, acid_offset);
    put_u32(image.bytes, kMetaAcidSize, static_cast<std::uint32_t>(acid.size()));
    image.bytes.insert(image.bytes.end(), aci0.begin(), aci0.end());
    image.bytes.resize(acid_offset);
    image.bytes.insert(image.bytes.end(), acid.begin(), acid.end());
    return image;
}

// The two acceptance profiles: same shape, different admitted permissions,
// cores, priorities, stack and resource settings.
inline NpdmImage profile_image_a() {
    return build_npdm(Parts{});
}

inline NpdmImage profile_image_b() {
    Parts parts;
    parts.meta_flags = 0x07 | (1u << 6); // alias region extra size enabled
    parts.main_priority = 20;
    parts.main_core = 3;
    parts.system_resource = 0x600000;
    parts.stack_size = 0x80000;
    parts.aci0_fs_permissions = 1 | (1u << 3);
    parts.acid_fs_permissions = ~std::uint64_t{0};
    parts.aci0_services = {{"vi:m", false}, {"nvdrv", false}, {"lm", false}};
    parts.acid_services = {{"vi:m", false}, {"nvdrv", false}, {"lm", false}};
    parts.aci0_capabilities = {
        cap_core_priority(47, 16, 1, 3),
        cap_syscall_mask(2, 0x40),
        cap_handle_table(1000),
        cap_kernel_version(18, 1),
        cap_program_type(1),
    };
    parts.acid_capabilities = {
        cap_core_priority(47, 16, 1, 3),
        cap_syscall_mask(2, 0x40),
        cap_handle_table(1023),
        cap_kernel_version(18, 1),
        cap_program_type(1),
    };
    return build_npdm(parts);
}

} // namespace npdm_fixture
