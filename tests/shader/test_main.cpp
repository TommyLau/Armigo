// ============================================================================
// Armigo Tests - Native Runtime Verification
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include "fixtures.h"
#include "test_support.h"

#include <armigo/shader/msl_emitter.h>

#include <cstdio>
#include <fstream>
#include <iostream>
#include <set>
#include <string>
#include <vector>

namespace armigo::shader::test {
namespace {

int g_failures_in_current_test = 0;

int emit_fixtures(const std::string& directory) {
    int failures = 0;
    for (const NamedProgram& fixture : msl_fixtures()) {
        const MslEmitResult result = emit_msl(fixture.program);
        if (!result.ok()) {
            std::cerr << "fixture " << fixture.name << " failed to emit:\n"
                      << describe_diagnostics(result.diagnostics);
            ++failures;
            continue;
        }
        const std::string path = directory + "/" + fixture.name + ".metal";
        std::ofstream file(path, std::ios::binary);
        file << result.kernel.source;
        if (!file) {
            std::cerr << "cannot write " << path << "\n";
            ++failures;
        }
    }
    return failures == 0 ? 0 : 1;
}

}  // namespace

std::vector<TestCase>& registry() {
    static std::vector<TestCase> cases;
    return cases;
}

void record_failure(const char* file, int line, const std::string& message) {
    ++g_failures_in_current_test;
    std::cerr << file << ":" << line << ": check failed: " << message << "\n";
}

std::string describe_diagnostics(const std::vector<Diagnostic>& diagnostics) {
    std::string text;
    for (const Diagnostic& diagnostic : diagnostics) {
        text += "  " + format(diagnostic) + "\n";
    }
    return text;
}

}  // namespace armigo::shader::test

// Usage:
//   armigo_shader_tests [group...]       run all tests or the named groups
//   armigo_shader_tests --list           list registered tests
//   armigo_shader_tests --emit-fixtures DIR
//                                        write MSL fixtures for compilation
int main(int argc, char** argv) {
    using namespace armigo::shader::test;

    std::vector<std::string> arguments(argv + 1, argv + argc);
    if (arguments.size() == 2 && arguments[0] == "--emit-fixtures") {
        return emit_fixtures(arguments[1]);
    }
    if (arguments.size() == 1 && arguments[0] == "--list") {
        for (const TestCase& test : registry()) {
            std::cout << test.group << "." << test.name << "\n";
        }
        return 0;
    }

    const std::set<std::string> groups(arguments.begin(), arguments.end());
    std::set<std::string> known_groups;
    for (const TestCase& test : registry()) {
        known_groups.insert(test.group);
    }
    for (const std::string& group : groups) {
        if (known_groups.count(group) == 0) {
            std::cerr << "unknown test group: " << group << "\n";
            return 2;
        }
    }

    int run = 0;
    int failed = 0;
    for (const TestCase& test : registry()) {
        if (!groups.empty() && groups.count(test.group) == 0) {
            continue;
        }
        g_failures_in_current_test = 0;
        test.function();
        ++run;
        if (g_failures_in_current_test != 0) {
            ++failed;
            std::cerr << "FAILED " << test.group << "." << test.name << "\n";
        }
    }
    std::cout << run - failed << " of " << run << " tests passed\n";
    return failed == 0 && run > 0 ? 0 : 1;
}
