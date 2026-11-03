// ============================================================================
// Armigo Tools - Development and Verification Tools
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include <armigo/common/diagnostics.h>
#include "inputs.h"

#include <cstdio>
#include <filesystem>
#include <string>
#include <vector>

namespace {

bool same_location(const armigo::pack::ContentLocation& a,
                   const armigo::pack::ContentLocation& b) {
    return a == b;
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        armigo::diag::write(armigo::diag::Level::error,  "usage: %s KEYS_FILE INPUT...\n", argv[0]);
        return 2;
    }
    std::vector<std::filesystem::path> paths;
    for (int i = 2; i < argc; ++i) {
        paths.emplace_back(argv[i]);
    }
    armigo::pack::InputsOptions options;
    options.keys_file = argv[1];
    auto classified = armigo::pack::classify_inputs(paths, options);
    if (const auto* failure = std::get_if<armigo::pack::InputsFailure>(&classified)) {
        armigo::diag::write(armigo::diag::Level::error,  "error: %s\n", failure->detail.c_str());
        return 1;
    }
    const armigo::pack::Inputs& inputs = std::get<armigo::pack::Inputs>(classified);
    for (const armigo::pack::ContentSet& set : inputs.content_sets) {
        std::string decision;
        if (static_cast<std::uint8_t>(set.type) < 0x80) {
            decision = "ignored: system content";
        } else if (same_location(set.location, inputs.application.location)) {
            decision = "application (selected)";
        } else if (inputs.patch && same_location(set.location, inputs.patch->location)) {
            decision = "patch (selected)";
        } else {
            for (const armigo::pack::SupersededUpdate& update : inputs.superseded_updates) {
                if (same_location(set.location, update.update.location)) {
                    decision = "patch (superseded)";
                    break;
                }
            }
            if (decision.empty()) {
                for (const armigo::pack::SupersededUpdate& update : inputs.duplicate_updates) {
                    if (same_location(set.location, update.update.location)) {
                        decision = "patch (duplicate)";
                        break;
                    }
                }
            }
            if (decision.empty()) {
                for (const armigo::pack::ContentSet& dlc : inputs.add_on_content) {
                    if (same_location(set.location, dlc.location)) {
                        decision = "dlc (accepted)";
                        break;
                    }
                }
            }
            if (decision.empty()) {
                for (const armigo::pack::SupersededUpdate& dlc : inputs.superseded_add_on_content) {
                    if (same_location(set.location, dlc.update.location)) {
                        decision = "dlc (superseded)";
                        break;
                    }
                }
            }
            if (decision.empty()) {
                for (const armigo::pack::RefusedInput& refusal : inputs.refused) {
                    if (refusal.content_set_index &&
                        refusal.file == set.location.file &&
                        *refusal.content_set_index == set.location.content_set_index) {
                        decision = "refused: " + refusal.reason;
                        break;
                    }
                }
            }
        }
        std::printf("%s | set %zu | %s | %s | v%u | %s\n",
                    set.location.file.filename().c_str(), set.location.content_set_index,
                    armigo::pack::format_title_id(set.title_id).c_str(),
                    armigo::pack::to_string(set.type), set.version, decision.c_str());
    }
    for (const armigo::pack::RefusedInput& refusal : inputs.refused) {
        bool represented = false;
        if (refusal.content_set_index) {
            for (const armigo::pack::ContentSet& set : inputs.content_sets) {
                if (set.location.file == refusal.file &&
                    set.location.content_set_index == *refusal.content_set_index) {
                    represented = true;
                    break;
                }
            }
        }
        if (!represented) {
            if (refusal.content_set_index) {
                std::printf("%s | set %zu | refused: %s\n", refusal.file.filename().c_str(),
                            *refusal.content_set_index, refusal.reason.c_str());
            } else {
                std::printf("%s | refused: %s\n", refusal.file.filename().c_str(),
                            refusal.reason.c_str());
            }
        }
    }
    std::printf("control | %s set %zu | nacp %zu bytes | %zu icons\n",
                inputs.control.source.file.filename().c_str(),
                inputs.control.source.content_set_index, inputs.control.nacp.size(),
                inputs.control.icons.size());
    return 0;
}
