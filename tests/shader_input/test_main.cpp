// ============================================================================
// Armigo Tests - Native Runtime Verification
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

// Test runner for the shader input parser suite: runs all registered tests,
// or only the groups named on the command line.
#include "test_support.h"

#include <cstring>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    using namespace armigo::shader_input::test;

    std::vector<std::string> selected_groups;
    for (int i = 1; i < argc; ++i) {
        selected_groups.emplace_back(argv[i]);
    }

    unsigned executed = 0;
    for (const TestCase& entry : registry()) {
        if (!selected_groups.empty()) {
            bool selected = false;
            for (const std::string& group : selected_groups) {
                if (group == entry.group) {
                    selected = true;
                    break;
                }
            }
            if (!selected) {
                continue;
            }
        }
        ++executed;
        std::printf("[%s] %s\n", entry.group, entry.name);
        entry.function();
    }

    if (selected_groups.size() == executed && executed == 0) {
        std::printf("no tests matched the requested groups\n");
        return 2;
    }
    std::printf("%u tests, %u failure(s)\n", executed, failure_count());
    return failure_count() == 0 ? 0 : 1;
}
