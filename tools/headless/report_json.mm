// ============================================================================
// Armigo Tools - Development and Verification Tools
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include "report_json.h"

#include <cstdio>
#include <string>

namespace armigo::tools {
namespace {

NSString* stage_name(runtime::LaunchStage stage) {
    switch (stage) {
        case runtime::LaunchStage::inputs: return @"inputs";
        case runtime::LaunchStage::profile: return @"profile";
        case runtime::LaunchStage::preparation: return @"preparation";
        case runtime::LaunchStage::modules: return @"modules";
        case runtime::LaunchStage::memory: return @"memory";
        case runtime::LaunchStage::threads: return @"threads";
        case runtime::LaunchStage::services: return @"services";
        case runtime::LaunchStage::graphics: return @"graphics";
        case runtime::LaunchStage::presentation: return @"presentation";
        case runtime::LaunchStage::shutdown: return @"shutdown";
    }
    return @"unknown";
}

NSString* error_name(runtime::LaunchError code) {
    switch (code) {
        case runtime::LaunchError::invalid_input: return @"invalid_input";
        case runtime::LaunchError::permission_denied: return @"permission_denied";
        case runtime::LaunchError::unsupported: return @"unsupported";
        case runtime::LaunchError::artifact_mismatch: return @"artifact_mismatch";
        case runtime::LaunchError::resource_limit: return @"resource_limit";
        case runtime::LaunchError::host_failure: return @"host_failure";
        case runtime::LaunchError::guest_fault: return @"guest_fault";
        case runtime::LaunchError::cancelled: return @"cancelled";
        case runtime::LaunchError::deadline_expired: return @"deadline_expired";
    }
    return @"unknown";
}

NSDictionary* failure_dictionary(const runtime::LaunchFailure& failure) {
    NSMutableDictionary* dictionary = [NSMutableDictionary dictionary];
    dictionary[@"stage"] = stage_name(failure.stage);
    dictionary[@"code"] = error_name(failure.code);
    dictionary[@"operation"] = [NSString stringWithUTF8String:failure.operation.c_str()];
    dictionary[@"detail"] = [NSString stringWithUTF8String:failure.detail.c_str()];
    if (failure.thread_id) {
        dictionary[@"thread_id"] = @(*failure.thread_id);
    }
    if (failure.guest_pc) {
        dictionary[@"guest_pc"] = @(*failure.guest_pc);
    }
    if (failure.command) {
        dictionary[@"command"] = @(*failure.command);
    }
    if (failure.guest_result) {
        dictionary[@"guest_result"] = @(*failure.guest_result);
    }
    return dictionary;
}

} // namespace

NSDictionary* shell_report_dictionary(NSString* mode, const ShellReportFields& fields) {
    NSMutableDictionary* report = [NSMutableDictionary dictionary];
    report[@"schema_version"] = @1;
    report[@"mode"] = mode;
    report[@"outcome"] = [NSString stringWithUTF8String:fields.outcome.c_str()];
    report[@"exit_code"] = @(fields.exit_code);
    report[@"quiescent"] = @(fields.quiescent);
    report[@"runtime_retained"] = @(fields.runtime_retained);
    // Container launches report verified only when every container
    // signature verified during extraction; extracted input is validated,
    // not authenticated.
    report[@"metadata_signature"] =
        fields.metadata_signature_verified ? @"verified" : @"not_verified";
    report[@"statistics"] = @{
        @"queued_frames": @(fields.statistics.queued_frames),
        @"completed_frames": @(fields.statistics.completed_frames),
        @"completed_draws": @(fields.statistics.completed_draws),
        @"elapsed_ns": @(static_cast<long long>(fields.statistics.elapsed.count())),
    };
    if (fields.frame_target) {
        report[@"frame_target"] = @(*fields.frame_target);
    }
    if (fields.guest_exit_status) {
        report[@"guest_exit_status"] = @(*fields.guest_exit_status);
    }
    if (fields.timeout_limit_ns) {
        report[@"timeout_limit_ns"] = @(*fields.timeout_limit_ns);
    }
    if (fields.cancel_reason) {
        report[@"cancel_reason"] = [NSString stringWithUTF8String:fields.cancel_reason->c_str()];
    }
    if (fields.failure) {
        report[@"failure"] = failure_dictionary(*fields.failure);
    }
    if (fields.teardown_failure) {
        report[@"teardown_failure"] = failure_dictionary(*fields.teardown_failure);
    }
    return [report copy];
}

bool emit_shell_report(NSString* mode, const ShellReportFields& fields,
                       const std::optional<std::filesystem::path>& report_file) {
    NSDictionary* report = shell_report_dictionary(mode, fields);
    NSError* serialization_error = nil;
    NSData* json = [NSJSONSerialization dataWithJSONObject:report
                                                   options:NSJSONWritingSortedKeys
                                                     error:&serialization_error];
    if (!json) {
        std::fputs("Could not serialize the run report.\n", stderr);
        return false;
    }
    std::fwrite(json.bytes, 1, json.length, stdout);
    std::fputc('\n', stdout);
    if (report_file) {
        NSString* path = [NSString stringWithUTF8String:report_file->string().c_str()];
        if (![json writeToFile:path options:NSDataWritingAtomic error:&serialization_error]) {
            std::fprintf(stderr, "Could not write the run report to %s: %s\n",
                         report_file->string().c_str(),
                         serialization_error.localizedDescription.UTF8String);
            return false;
        }
    }
    return true;
}

} // namespace armigo::tools
