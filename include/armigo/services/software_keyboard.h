// ============================================================================
// Armigo Services - Software Keyboard Library Applet
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <span>
#include <string>
#include <vector>

namespace armigo::services::am {

// The software keyboard applet (17) in full-screen mode: the application pushes
// its common arguments, the keyboard configuration and a work buffer, starts the
// applet and reads one output storage after it exits. Only the configuration
// fields the applet presents or obeys are kept here.
struct KeyboardRequest {
    // Library version from the common arguments; it fixes the configuration size.
    std::uint32_t library_version = 0;
    std::uint32_t type = 0;
    std::u16string header;
    std::u16string sub;
    std::u16string guide;
    // Read from the work buffer at the configured offset, always UTF-16.
    std::u16string initial;
    std::uint32_t min_length = 0;
    // Zero in the configuration means no limit of its own; the applet then caps
    // input at 500 UTF-16 units.
    std::uint32_t max_length = 0;
    bool password = false;
    // The output text is UTF-8 instead of UTF-16.
    bool utf8 = false;
    // The application checks the text through interactive storages before the
    // applet exits.
    bool text_check = false;
    // Where the initial text sits in the work buffer, in bytes and UTF-16 units.
    std::uint32_t initial_offset = 0;
    std::uint32_t initial_length = 0;
};

// What the player decided: OK with the text, or cancel.
struct KeyboardResponse {
    bool accepted = false;
    std::u16string text;
};

// Supplies the player's input for one keyboard request. The headless default
// accepts the initial text; a windowed host can install one that asks the player.
using KeyboardInput = std::function<KeyboardResponse(const KeyboardRequest&)>;

// Common-argument and configuration sizes of the supported library version.
inline constexpr std::size_t keyboard_common_arguments_size = 0x20;
inline constexpr std::size_t keyboard_config_common_size = 0x3d4;
// Output storage: u32 close result (0 OK, 1 cancel) then the text area.
inline constexpr std::size_t keyboard_string_size = 0x7d4;
inline constexpr std::size_t keyboard_output_size = 4 + keyboard_string_size;

// Parses the common arguments and the configuration storage. Returns a
// diagnostic when either is malformed or names an unsupported layout.
const char* parse_keyboard_request(std::span<const std::byte> common_arguments,
                                   std::span<const std::byte> config, KeyboardRequest& request);

// Headless input: OK with the initial text, or with "Player" when there is none.
// The text is then fitted to the configured length limits.
KeyboardResponse headless_keyboard_input(const KeyboardRequest& request);

// Fits text to the configured limits as the applet would accept it: text
// shorter than the minimum is doubled (space-separated) until it is long
// enough, then anything past the maximum is cut.
std::u16string fit_keyboard_text(const KeyboardRequest& request, std::u16string text);

// The output storage for a response: the close result and the NUL-terminated
// text, UTF-8 or UTF-16 as configured, cut to the text area.
std::vector<std::byte> keyboard_output(const KeyboardRequest& request,
                                       const KeyboardResponse& response);

} // namespace armigo::services::am
