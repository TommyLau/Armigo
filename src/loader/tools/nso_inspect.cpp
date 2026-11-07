// ============================================================================
// Armigo Loader - Guest Module Loading
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include <armigo/common/diagnostics.h>
// Development tool: reads NSO0 files with the Armigo module reader and prints
// a structural summary. It never executes, maps, or writes module content.
//
// Usage: armigo-nso-inspect [--verify-all] [--image-digest] FILE...

#include <armigo/loader/nso_module.h>

#include "../sha256.h"

#include <cstdio>
#include <cstring>
#include <fstream>
#include <vector>

namespace {

using namespace armigo::loader;

const char* permission_text(SegmentPermission permission) {
    switch (static_cast<unsigned>(permission)) {
    case 1:
        return "r--";
    case 3:
        return "rw-";
    case 5:
        return "r-x";
    case 4:
        return "--x";
    default:
        return "???";
    }
}

bool read_file(const char* path, std::uint64_t limit, std::vector<std::byte>& out) {
    std::ifstream stream(path, std::ios::binary | std::ios::ate);
    if (!stream) {
        armigo::diag::write(armigo::diag::Level::error,  "%s: cannot open\n", path);
        return false;
    }
    const std::streamoff size = stream.tellg();
    if (size < 0 || static_cast<std::uint64_t>(size) > limit) {
        armigo::diag::write(armigo::diag::Level::error,  "%s: size exceeds the input limit\n", path);
        return false;
    }
    out.resize(static_cast<std::size_t>(size));
    stream.seekg(0);
    if (!stream.read(reinterpret_cast<char*>(out.data()), size)) {
        armigo::diag::write(armigo::diag::Level::error,  "%s: read failed\n", path);
        return false;
    }
    return true;
}

int inspect(const char* path, const NsoReadOptions& options, bool image_digest) {
    std::vector<std::byte> file;
    if (!read_file(path, options.limits.max_input_bytes, file)) {
        return 1;
    }
    const NsoReadResult result = read_nso_module(file, options);
    if (!result.ok()) {
        const NsoReadError& error = result.error;
        std::printf("%s: error: %s", path, to_string(error.code));
        if (error.segment) {
            std::printf(" [segment %s]", to_string(*error.segment));
        }
        if (error.lz4 != Lz4Error::success) {
            std::printf(" [%s]", to_string(error.lz4));
        }
        if (!error.decoder_detail.empty()) std::printf(" [%s]", error.decoder_detail.c_str());
        std::printf(" [detail 0x%llx]\n", static_cast<unsigned long long>(error.detail));
        return 1;
    }
    const NsoModuleImage& image = *result.image;
    std::printf("%s: ok, input 0x%zx bytes, image 0x%zx bytes, bss 0x%llx+0x%llx\n", path,
                file.size(), image.bytes.size(),
                static_cast<unsigned long long>(image.bss_offset),
                static_cast<unsigned long long>(image.bss_size));
    for (const NsoSegment& segment : image.segments) {
        std::printf("  %-6s offset 0x%08llx content 0x%08llx mapped 0x%08llx %s %s file 0x%llx "
                    "hash %s\n",
                    to_string(segment.kind), static_cast<unsigned long long>(segment.image_offset),
                    static_cast<unsigned long long>(segment.content_size),
                    static_cast<unsigned long long>(segment.mapped_size),
                    permission_text(segment.permission),
                    segment.compressed ? (image.zbic_compression ? "zbic" : "lz4") : "raw",
                    static_cast<unsigned long long>(segment.file_size),
                    segment.hash == HashCheck::verified ? "verified" : "not-checked");
    }
    if (image.module_header) {
        const ModuleHeaderInfo& info = *image.module_header;
        std::printf("  MOD0 at 0x%llx: dynamic 0x%llx bss 0x%llx-0x%llx eh_frame_hdr "
                    "0x%llx-0x%llx module 0x%llx\n",
                    static_cast<unsigned long long>(info.header_offset),
                    static_cast<unsigned long long>(info.dynamic_offset),
                    static_cast<unsigned long long>(info.bss_start_offset),
                    static_cast<unsigned long long>(info.bss_end_offset),
                    static_cast<unsigned long long>(info.eh_frame_hdr_start_offset),
                    static_cast<unsigned long long>(info.eh_frame_hdr_end_offset),
                    static_cast<unsigned long long>(info.module_object_offset));
    } else {
        std::printf("  MOD0 absent\n");
    }
    if (image_digest) {
        const auto digest = armigo::loader::detail::sha256(image.bytes);
        if (!digest) {
            armigo::diag::write(armigo::diag::Level::error,  "%s: digest failed\n", path);
            return 1;
        }
        std::printf("  image-sha256 ");
        for (std::byte value : *digest) {
            std::printf("%02x", static_cast<unsigned>(value));
        }
        std::printf("\n");
    }
    return 0;
}

} // namespace

int main(int argc, char** argv) {
    NsoReadOptions options;
    bool image_digest = false;
    int first = 1;
    for (; first < argc && std::strncmp(argv[first], "--", 2) == 0; ++first) {
        if (std::strcmp(argv[first], "--verify-all") == 0) {
            options.hash_policy = NsoHashPolicy::verify_all;
        } else if (std::strcmp(argv[first], "--image-digest") == 0) {
            image_digest = true;
        } else if (std::strcmp(argv[first], "--help") == 0) {
            std::printf("Usage: armigo-nso-inspect [--verify-all] [--image-digest] FILE...\n");
            return 0;
        } else {
            armigo::diag::write(armigo::diag::Level::error,  "unknown option %s\n", argv[first]);
            return 2;
        }
    }
    if (first == argc) {
        armigo::diag::write(armigo::diag::Level::error,  "Usage: armigo-nso-inspect [--verify-all] [--image-digest] FILE...\n");
        return 2;
    }
    int status = 0;
    for (int i = first; i < argc; ++i) {
        status |= inspect(argv[i], options, image_digest);
    }
    return status;
}
