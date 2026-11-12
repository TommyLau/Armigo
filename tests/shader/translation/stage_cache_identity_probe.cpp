// ============================================================================
// Armigo Shader - Stage Translation
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include <armigo/shader_translation/stage_cache.h>
#include <array>
#include <fstream>
#include <iostream>
#include <iterator>

namespace tr = armigo::shader::translation;
int main(int argc, char **argv) {
  if (argc != 3)
    return 2;
  const std::array control{std::byte{1}, std::byte{2}},
      code{std::byte{3}, std::byte{4}};
  const auto key =
      tr::StageCache::key(control, code, "synthetic identical options");
  tr::StageCache cache(argv[1]);
  const auto loaded = cache.load(key);
  const bool wanted = std::string_view(argv[2]) == "hit";
  if (bool(loaded) != wanted) {
    std::cerr << "same-input stage cache expected " << (wanted ? "hit" : "miss")
              << " in " << cache.directory() << "\n";
    return 1;
  }
  if (loaded) {
    std::ifstream input(loaded->source_file);
    std::string source{std::istreambuf_iterator<char>(input), {}};
    if (loaded->reflection != "{\"stage\":\"compute\"}" ||
        loaded->program != std::vector{std::byte{3}, std::byte{4}} ||
        source != "kernel void synthetic_stage() {}\n")
      return 1;
  } else {
    tr::TranslatedStage stage;
    stage.stage = armigo::shader_input::Stage::Compute;
    stage.source = "kernel void synthetic_stage() {}\n";
    stage.reflection = "{\"stage\":\"compute\"}";
    stage.program = {code.begin(), code.end()};
    if (!cache.store(key, stage))
      return 1;
  }
  std::cout << cache.directory().filename().string() << " " << key << "\n";
}
