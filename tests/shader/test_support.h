// ============================================================================
// Armigo Tests - Native Runtime Verification
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

// Minimal self-contained test harness for the shader foundation tests.
#pragma once

#include <armigo/shader/diagnostics.h>

#include <cstdint>
#include <sstream>
#include <string>
#include <vector>

namespace armigo::shader::test {

using TestFunction = void (*)();

struct TestCase {
    const char* group;
    const char* name;
    TestFunction function;
};

std::vector<TestCase>& registry();

struct Registrar {
    Registrar(const char* group, const char* name, TestFunction function) {
        registry().push_back({group, name, function});
    }
};

void record_failure(const char* file, int line, const std::string& message);

template <typename T>
std::string describe(const T& value) {
    std::ostringstream stream;
    stream << value;
    return stream.str();
}

inline std::string describe(DiagnosticCode code) { return to_string(code); }

// Renders diagnostics as one code per line for failure messages.
std::string describe_diagnostics(const std::vector<Diagnostic>& diagnostics);

}  // namespace armigo::shader::test

#define ARMIGO_TEST(group, name)                                      \
    static void group##_##name();                                     \
    static const ::armigo::shader::test::Registrar                    \
        group##_##name##_registrar(#group, #name, &group##_##name);   \
    static void group##_##name()

#define ARMIGO_CHECK(condition)                                         \
    do {                                                                \
        if (!(condition)) {                                             \
            ::armigo::shader::test::record_failure(__FILE__, __LINE__,  \
                                                   #condition);         \
        }                                                               \
    } while (false)

#define ARMIGO_CHECK_EQ(actual, expected)                                   \
    do {                                                                    \
        const auto& armigo_actual = (actual);                               \
        const auto& armigo_expected = (expected);                           \
        if (!(armigo_actual == armigo_expected)) {                          \
            ::armigo::shader::test::record_failure(                         \
                __FILE__, __LINE__,                                         \
                std::string(#actual " == " #expected "\n  actual:   ") +    \
                    ::armigo::shader::test::describe(armigo_actual) +       \
                    "\n  expected: " +                                      \
                    ::armigo::shader::test::describe(armigo_expected));     \
        }                                                                   \
    } while (false)
