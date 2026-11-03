// ============================================================================
// Armigo Tests - Native Runtime Verification
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

// Minimal self-contained test harness for the shader input parser tests.
#pragma once

#include <armigo/shader_input/shader_input.h>

#include <cstdint>
#include <cstdio>
#include <sstream>
#include <string>
#include <vector>

namespace armigo::shader_input::test {

struct TestCase {
    const char* group;
    const char* name;
    void (*function)();
};

inline std::vector<TestCase>& registry() {
    static std::vector<TestCase> cases;
    return cases;
}

inline unsigned& failure_count() {
    static unsigned count = 0;
    return count;
}

struct Registrar {
    Registrar(const char* group, const char* name, void (*function)()) {
        registry().push_back({group, name, function});
    }
};

inline void record_failure(const char* file, int line,
                           const std::string& message) {
    ++failure_count();
    std::printf("  FAIL %s:%d: %s\n", file, line, message.c_str());
}

template <typename T>
std::string describe(const T& value) {
    std::ostringstream stream;
    stream << value;
    return stream.str();
}

inline std::string describe(const ParseError& error) {
    std::ostringstream stream;
    stream << error.message() << " (code " << static_cast<unsigned>(error.code)
           << ", value " << error.value << ", bound " << error.bound << ")";
    return stream.str();
}

inline std::string describe(ParseError::Code code) {
    return describe(ParseError{code, 0, 0});
}

// Parses `input`, expecting success, and returns the parsed structure by
// value; records a failure and returns a zeroed instance when the parse is
// rejected. The result borrows the caller's buffers (see the lifetime
// contract), so callers must keep their fixtures alive.
inline ParsedShader must_parse(const ShaderInput& input, const char* file,
                               int line) {
    const ParseResult result = parse_shader(input);
    if (const ParsedShader* parsed = std::get_if<ParsedShader>(&result)) {
        return *parsed;
    }
    record_failure(file, line,
                   "parse_shader unexpectedly failed: " +
                       describe(std::get<ParseError>(result)));
    return ParsedShader{};
}

// Parses `input`, expecting the given rejection code; records a failure when
// the parse succeeds or is rejected with a different code.
inline void must_reject(const ShaderInput& input, ParseError::Code expected,
                        const char* file, int line) {
    const ParseResult result = parse_shader(input);
    if (const ParseError* error = std::get_if<ParseError>(&result)) {
        if (error->code != expected) {
            record_failure(file, line,
                           "expected rejection " + describe(expected) +
                               " but got " + describe(*error));
        }
        return;
    }
    record_failure(file, line,
                   "parse_shader unexpectedly succeeded; expected rejection " +
                       describe(expected));
}

}  // namespace armigo::shader_input::test

#define ARMIGO_SI_TEST(group, name)                                          \
    static void group##_##name();                                            \
    static const ::armigo::shader_input::test::Registrar                     \
        group##_##name##_registrar(#group, #name, &group##_##name);          \
    static void group##_##name()

#define ARMIGO_SI_CHECK(condition)                                         \
    do {                                                                   \
        if (!(condition)) {                                                \
            ::armigo::shader_input::test::record_failure(                  \
                __FILE__, __LINE__, #condition);                           \
        }                                                                  \
    } while (false)

#define ARMIGO_SI_CHECK_EQ(actual, expected)                               \
    do {                                                                   \
        const auto& actual_value = (actual);                               \
        const auto& expected_value = (expected);                           \
        if (!(actual_value == expected_value)) {                           \
            ::armigo::shader_input::test::record_failure(                  \
                __FILE__, __LINE__,                                        \
                std::string(#actual " == ") +                              \
                    ::armigo::shader_input::test::describe(expected_value) + \
                    " (actual " +                                          \
                    ::armigo::shader_input::test::describe(actual_value) + \
                    ")");                                                  \
        }                                                                  \
    } while (false)
