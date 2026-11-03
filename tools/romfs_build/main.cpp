// ============================================================================
// Armigo Tools - RomFS image builder
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

// Command line front end: builds one RomFS image from a host directory.

#include <armigo/romfs/romfs_image.h>

#include <cstdio>
#include <exception>
#include <fstream>

int main(int argc, char** argv) try {
    if (argc != 3) {
        std::fprintf(stderr, "usage: %s <input-directory> <output-image>\n", argv[0]);
        return 2;
    }
    armigo::romfs::ImageLayout layout;
    std::string error;
    const auto image = armigo::romfs::build(argv[1], layout, error);
    if (!image) {
        std::fprintf(stderr, "%s\n", error.c_str());
        return 1;
    }
    std::ofstream out(argv[2], std::ios::binary | std::ios::trunc);
    if (!out) {
        std::fprintf(stderr, "cannot create %s\n", argv[2]);
        return 1;
    }
    out.write(reinterpret_cast<const char*>(image->data()),
              static_cast<std::streamsize>(image->size()));
    if (!out) {
        std::fprintf(stderr, "cannot write %s\n", argv[2]);
        return 1;
    }
    std::printf("romfs_image bytes=%zu directories=%zu files=%zu data_offset=%u\n", image->size(),
                layout.directories.size(), layout.files.size(), layout.file_data_offset);
    return 0;
} catch (const std::exception& error) {
    std::fprintf(stderr, "romfs build failed: %s\n", error.what());
    return 1;
}
