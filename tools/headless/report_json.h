// ============================================================================
// Armigo Tools - Development and Verification Tools
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#pragma once

#include "runner_shell.h"

#import <Foundation/Foundation.h>

#include <optional>
#include <filesystem>

namespace armigo::tools {

// Builds the structured run-report dictionary shared by the tool frontends.
NSDictionary* shell_report_dictionary(NSString* mode, const ShellReportFields& fields);
// Prints the JSON encoding to stdout and, when report_file is set, writes the
// same bytes there. Returns false and reports on stderr when either fails.
bool emit_shell_report(NSString* mode, const ShellReportFields& fields,
                       const std::optional<std::filesystem::path>& report_file);

} // namespace armigo::tools
