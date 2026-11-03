// ============================================================================
// Armigo Platform - Apple Platform Integration
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#pragma once

#import <Metal/Metal.h>

#include "pso_manifest.h"

#include <CommonCrypto/CommonDigest.h>
#include <dispatch/dispatch.h>

#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <string>
#include <vector>

namespace armigo::platform::metal::nvn_driver {

// Metal's archive loader trusts the file: a damaged one (a bad header, a
// truncation) crashes the process instead of failing. So an archive file is
// named for the SHA-256 of its own bytes, "<tag>-<first 32 hex digits>.marchive",
// and is only handed to Metal when the bytes still hash to that name. Files are
// written under another name and renamed once complete.
inline std::string file_digest_prefix(const std::filesystem::path& path) {
    std::FILE* file = std::fopen(path.c_str(), "rb");
    if (!file) return {};
    CC_SHA256_CTX context;
    CC_SHA256_Init(&context);
    std::vector<unsigned char> chunk(1 << 20);
    for (std::size_t got; (got = std::fread(chunk.data(), 1, chunk.size(), file)) != 0;)
        CC_SHA256_Update(&context, chunk.data(), static_cast<CC_LONG>(got));
    const bool failed = std::ferror(file) != 0;
    std::fclose(file);
    if (failed) return {};
    unsigned char digest[CC_SHA256_DIGEST_LENGTH];
    CC_SHA256_Final(digest, &context);
    return pso_hex_encode(digest, 16);
}

// Renames a completely written file to its digest name in the same directory;
// the new path, or empty when the file could not be read or renamed.
inline std::filesystem::path seal_archive(const std::filesystem::path& written, const std::string& tag) {
    const auto digest = file_digest_prefix(written);
    if (digest.empty()) return {};
    auto sealed = written.parent_path() / (tag + "-" + digest + ".marchive");
    std::error_code error;
    std::filesystem::rename(written, sealed, error);
    return error ? std::filesystem::path() : sealed;
}

// Whether a file's name carries the digest of its bytes.
inline bool archive_intact(const std::filesystem::path& path) {
    const auto stem = path.stem().string();
    const auto dash = stem.rfind('-');
    return dash != std::string::npos && stem.size() - dash - 1 == 32 &&
           stem.compare(dash + 1, std::string::npos, file_digest_prefix(path)) == 0;
}

// The Metal 4 binary archives in a directory, for the compiler to look
// pipelines up in, in the order they were written (their names start with the
// time). A file that fails its digest, or does not open as an archive (written
// by another Metal build), is left out and listed, never an error: its
// pipelines compile as usual. A large archive is several files, whose digests
// are checked in parallel.
struct ArchiveSet {
    NSMutableArray* archives = [NSMutableArray array];
    std::vector<std::filesystem::path> loaded, rejected;
};

inline ArchiveSet open_archives(id<MTLDevice> device, const std::filesystem::path& directory) {
    ArchiveSet set;
    std::error_code error;
    std::vector<std::filesystem::path> paths;
    for (const auto& entry : std::filesystem::directory_iterator(directory, error))
        if (entry.path().extension() == ".marchive") paths.push_back(entry.path());
    std::sort(paths.begin(), paths.end());
    std::vector<char> intact(paths.size());
    const auto* listed = paths.data();
    auto* checked = intact.data();
    dispatch_apply(paths.size(), dispatch_get_global_queue(QOS_CLASS_DEFAULT, 0), ^(std::size_t index) {
        checked[index] = archive_intact(listed[index]);
    });
    for (std::size_t index = 0; index != paths.size(); ++index) {
        id<MTL4Archive> archive = nil;
        if (intact[index]) {
            NSError* open_error = nil;
            archive = [device newArchiveWithURL:[NSURL fileURLWithPath:@(paths[index].c_str())]
                                          error:&open_error];
        }
        if (archive) {
            [set.archives addObject:archive];
            set.loaded.push_back(paths[index]);
        } else {
            set.rejected.push_back(paths[index]);
        }
    }
    return set;
}

} // namespace armigo::platform::metal::nvn_driver
