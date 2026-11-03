// ============================================================================
// Armigo Tools - Development and Verification Tools
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include "armigo/platform/headless_probe.h"

#include "report_json.h"
#include "runner_shell.h"

#import <Foundation/Foundation.h>

#include <cstdio>
#include <cstdlib>
#include <string>
#include <string_view>
#include <vector>

namespace {

int run_headless(int argc, char* argv[]) {
    const std::vector<char*> arguments(argv + 2, argv + argc);
    const auto parsed = armigo::tools::parse_runner_arguments(
        armigo::platform::PresentationMode::headless, arguments);
    if (parsed.help) {
        std::printf("Usage: armigo-headless run --exefs DIR --romfs DIR [--shaders DIR] [options]\n"
                    "%s\n",
                    armigo::tools::runner_run_usage().c_str());
        return 0;
    }
    if (!parsed.ok) {
        std::fprintf(stderr, "armigo-headless run: %s\n", parsed.error.c_str());
        return 2;
    }
    @autoreleasepool {
        const auto environment = armigo::tools::default_shell_environment();
        const auto report =
            armigo::tools::run_headless_shell(parsed.arguments, environment);
        const auto fields = armigo::tools::report_fields(report);
        if (!armigo::tools::emit_shell_report(@"run-headless", fields,
                                              parsed.arguments.report_file)) {
            armigo::tools::exit_process_if_runtime_retained(1);
            return 1;
        }
        const int status = static_cast<int>(report.exit_code);
        armigo::tools::exit_process_if_runtime_retained(status);
        return status;
    }
}

} // namespace

int main(int argc, char* argv[]) {
    armigo::tools::raise_open_file_limit();
    if (argc == 2 && std::string_view(argv[1]) == "--help") {
        std::puts("Usage: armigo-headless --probe | --self-test | --help\n"
                  "Usage: armigo-headless run --exefs DIR --romfs DIR [--shaders DIR] [options]\n"
                  "Usage: armigo-headless pso-archive --shaders DIR --manifest FILE... [--cache DIR] [--threads N] "
                  "[--warm-only]\n"
                  "A headless Metal 4 platform check and runner shell. "
                  "Guest execution is not implemented.");
        return 0;
    }
    if (argc >= 2 && std::string_view(argv[1]) == "run") {
        return run_headless(argc, argv);
    }
    if (argc >= 2 && std::string_view(argv[1]) == "pso-archive") {
        return armigo::tools::run_pso_archive_command(argc, argv);
    }
    if (argc != 2 || (std::string_view(argv[1]) != "--probe" &&
                      std::string_view(argv[1]) != "--self-test")) {
        std::fputs("Expected --probe, --self-test, --help, run, or pso-archive.\n", stderr);
        return 2;
    }
    @autoreleasepool {
        const bool self_test = std::string_view(argv[1]) == "--self-test";
        const auto result = armigo::platform::probe_metal(self_test);
        NSDictionary* report = @{
            @"schema_version": @1,
            @"mode": self_test ? @"self-test" : @"probe",
            @"device": [NSString stringWithUTF8String:result.device_name.c_str()],
            @"metal4_supported": @(result.supported),
            @"self_test_ran": @(result.self_test_ran),
            @"self_test_passed": @(result.self_test_passed),
            @"verified_buffer_bytes": @(result.verified_buffer_bytes),
            @"verified_pixels": @(result.verified_pixels),
            @"guest_execution_implemented": @NO,
            @"error": [NSString stringWithUTF8String:result.error.c_str()]
        };
        NSError* serialization_error = nil;
        NSData* json = [NSJSONSerialization dataWithJSONObject:report
                                                      options:NSJSONWritingSortedKeys
                                                        error:&serialization_error];
        if (!json) {
            std::fputs("Could not serialize the platform check result.\n", stderr);
            return 1;
        }
        std::fwrite(json.bytes, 1, json.length, stdout);
        std::fputc('\n', stdout);
        return result.supported && (!self_test || result.self_test_passed) ? 0 : 1;
    }
}
