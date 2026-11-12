// ============================================================================
// Armigo NVN - Entry Point Table
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include <armigo/compat/patches.h>
#include <cstdio>
#include <fstream>
#include <unistd.h>

namespace {
unsigned failures = 0;
#define CHECK(value) do { if (!(value)) { std::fprintf(stderr, "line %d: %s\n", __LINE__, #value); ++failures; } } while (false)
armigo::compat::PatchReadResult parse(const std::string& flag, const std::string& extra = "") {
    // Independently authored identity used only for parser tests.
    const auto contents = std::string(R"({"version":1,"patches":[{"title_id":"0x0000000000000001","main_build_id":")") +
        std::string(64, '0') + R"(","nvn_driver":)" + flag + extra + "}]}";
    auto pattern = (std::filesystem::temp_directory_path() / "armigo-nvn-compat-XXXXXX").string();
    const int fd = mkstemp(pattern.data());
    if (fd < 0) return std::string("Cannot create synthetic compatibility file.");
    close(fd);
    { std::ofstream out(pattern); out << contents; }
    auto result = armigo::compat::read_patch_descriptions(pattern);
    unlink(pattern.c_str());
    return result;
}
}
int main() {
    using namespace armigo::compat;
    for (const auto flag : {"true", "false"}) {
        const auto result = parse(flag);
        CHECK(std::holds_alternative<PatchDescriptions>(result));
        if (const auto* descriptions = std::get_if<PatchDescriptions>(&result)) {
            CHECK(descriptions->size() == 1);
            CHECK(descriptions->front().nvn_driver == (std::string_view(flag) == "true"));
            CHECK(descriptions->front().module.empty());
            CHECK(descriptions->front().calls.empty());
            CHECK(validate_patch(descriptions->front(), {}).empty());
        }
    }
    for (const auto flag : {"1", "0", "null", "[]", "{}", "\"true\""})
        CHECK(std::holds_alternative<std::string>(parse(flag)));
    {
        // The title's pipeline list is an optional file name in the profile's
        // folder; the description holds its path there.
        const auto folder = std::filesystem::temp_directory_path();
        const auto list = folder / ("armigo-nvn-compat-list-" + std::to_string(getpid()) + ".tsv");
        { std::ofstream out(list); out << "S\t0.ab.0\n"; }
        const auto result = parse("true", ",\"pipelines\":\"" + list.filename().string() + "\"");
        const auto* descriptions = std::get_if<PatchDescriptions>(&result);
        CHECK(descriptions && descriptions->front().pso_manifest == list.string());
        std::filesystem::remove(list);
        const auto absent = parse("true");
        const auto* plain = std::get_if<PatchDescriptions>(&absent);
        CHECK(plain && plain->front().pso_manifest.empty());
        CHECK(std::holds_alternative<std::string>(parse("true", ",\"pipelines\":\"\"")));
        CHECK(std::holds_alternative<std::string>(parse("true", ",\"pipelines\":7")));
        CHECK(std::holds_alternative<std::string>(parse("true", ",\"pipelines\":\"missing.tsv\"")));
        CHECK(std::holds_alternative<std::string>(parse("true", ",\"pipelines\":\"../x.tsv\"")));
        CHECK(std::holds_alternative<std::string>(parse("true", ",\"pso_manifest\":\"/a.tsv\"")));
    }
    {
        // The title's NVN policy: an object of known members (booleans and
        // named modes); absent members keep the driver's default.
        const auto set = parse("true", ",\"nvn_policy\":{\"cpu_cached_texture_writes\":true}");
        const auto* descriptions = std::get_if<PatchDescriptions>(&set);
        CHECK(descriptions && descriptions->front().nvn_policy.cpu_cached_texture_writes);
        const auto empty = parse("true", ",\"nvn_policy\":{}");
        const auto* defaults = std::get_if<PatchDescriptions>(&empty);
        CHECK(defaults && !defaults->front().nvn_policy.cpu_cached_texture_writes);
        CHECK(std::holds_alternative<std::string>(parse("true", ",\"nvn_policy\":{\"cpu_cached_texture_writes\":1}")));
        CHECK(std::holds_alternative<std::string>(parse("true", ",\"nvn_policy\":{\"unknown\":true}")));
        const auto segment = parse("true", ",\"nvn_policy\":{\"timestamps\":\"segment\"}");
        const auto* segmented = std::get_if<PatchDescriptions>(&segment);
        CHECK(segmented && segmented->front().nvn_policy.segment_timestamps &&
              !segmented->front().nvn_policy.cpu_cached_texture_writes);
        const auto report = parse("true", ",\"nvn_policy\":{\"timestamps\":\"report\"}");
        const auto* reported = std::get_if<PatchDescriptions>(&report);
        CHECK(reported && !reported->front().nvn_policy.segment_timestamps);
        CHECK(defaults && !defaults->front().nvn_policy.segment_timestamps);
        CHECK(std::holds_alternative<std::string>(parse("true", ",\"nvn_policy\":{\"timestamps\":\"frame\"}")));
        CHECK(std::holds_alternative<std::string>(parse("true", ",\"nvn_policy\":{\"timestamps\":true}")));
        const auto game = parse("true", ",\"nvn_policy\":{\"barriers\":\"game\"}");
        const auto* gamed = std::get_if<PatchDescriptions>(&game);
        CHECK(gamed && gamed->front().nvn_policy.game_barriers && !gamed->front().nvn_policy.segment_timestamps);
        const auto full = parse("true", ",\"nvn_policy\":{\"barriers\":\"full\"}");
        const auto* fulled = std::get_if<PatchDescriptions>(&full);
        CHECK(fulled && !fulled->front().nvn_policy.game_barriers);
        CHECK(defaults && !defaults->front().nvn_policy.game_barriers);
        CHECK(std::holds_alternative<std::string>(parse("true", ",\"nvn_policy\":{\"barriers\":\"none\"}")));
        const auto hoist = parse("true", ",\"nvn_policy\":{\"hoist_implicit_samples\":true}");
        const auto* hoisted = std::get_if<PatchDescriptions>(&hoist);
        CHECK(hoisted && hoisted->front().nvn_policy.hoist_implicit_samples &&
              !hoisted->front().nvn_policy.game_barriers);
        CHECK(defaults && !defaults->front().nvn_policy.hoist_implicit_samples);
        CHECK(std::holds_alternative<std::string>(
            parse("true", ",\"nvn_policy\":{\"hoist_implicit_samples\":\"yes\"}")));
        const auto replay = parse("true", ",\"nvn_policy\":{\"conditional_render\":\"replay\"}");
        const auto* replayed = std::get_if<PatchDescriptions>(&replay);
        CHECK(replayed && replayed->front().nvn_policy.replay_conditional_render &&
              !replayed->front().nvn_policy.hoist_implicit_samples);
        const auto exact = parse("true", ",\"nvn_policy\":{\"conditional_render\":\"exact\"}");
        const auto* exacted = std::get_if<PatchDescriptions>(&exact);
        CHECK(exacted && !exacted->front().nvn_policy.replay_conditional_render);
        CHECK(defaults && !defaults->front().nvn_policy.replay_conditional_render);
        CHECK(std::holds_alternative<std::string>(
            parse("true", ",\"nvn_policy\":{\"conditional_render\":\"foo\"}")));
        CHECK(std::holds_alternative<std::string>(
            parse("true", ",\"nvn_policy\":{\"conditional_render\":true}")));
        const auto current = parse("true", ",\"nvn_policy\":{\"handle_reads\":\"current\"}");
        const auto* currented = std::get_if<PatchDescriptions>(&current);
        CHECK(currented && currented->front().nvn_policy.current_handle_reads &&
              !currented->front().nvn_policy.replay_conditional_render);
        const auto ordered = parse("true", ",\"nvn_policy\":{\"handle_reads\":\"ordered\"}");
        const auto* ordereded = std::get_if<PatchDescriptions>(&ordered);
        CHECK(ordereded && !ordereded->front().nvn_policy.current_handle_reads);
        CHECK(defaults && !defaults->front().nvn_policy.current_handle_reads);
        CHECK(std::holds_alternative<std::string>(
            parse("true", ",\"nvn_policy\":{\"handle_reads\":\"later\"}")));
        const auto retired = parse("true", ",\"nvn_policy\":{\"guest_store_textures\":\"retired\"}");
        const auto* retireded = std::get_if<PatchDescriptions>(&retired);
        CHECK(retireded && retireded->front().nvn_policy.retired_guest_store_textures &&
              !retireded->front().nvn_policy.current_handle_reads);
        const auto store_ordered = parse("true", ",\"nvn_policy\":{\"guest_store_textures\":\"ordered\"}");
        const auto* store_ordereded = std::get_if<PatchDescriptions>(&store_ordered);
        CHECK(store_ordereded && !store_ordereded->front().nvn_policy.retired_guest_store_textures);
        CHECK(defaults && !defaults->front().nvn_policy.retired_guest_store_textures);
        CHECK(std::holds_alternative<std::string>(
            parse("true", ",\"nvn_policy\":{\"guest_store_textures\":\"later\"}")));
        const auto core = parse("true", ",\"nvn_policy\":{\"replay\":\"core\"}");
        const auto* cored = std::get_if<PatchDescriptions>(&core);
        CHECK(cored && cored->front().nvn_policy.core_replay);
        const auto standard = parse("true", ",\"nvn_policy\":{\"replay\":\"standard\"}");
        const auto* standarded = std::get_if<PatchDescriptions>(&standard);
        CHECK(standarded && !standarded->front().nvn_policy.core_replay);
        CHECK(defaults && !defaults->front().nvn_policy.core_replay);
        CHECK(std::holds_alternative<std::string>(
            parse("true", ",\"nvn_policy\":{\"replay\":\"other\"}")));
        const auto slots = parse("true", ",\"nvn_policy\":{\"submission_slots\":12}");
        const auto* slotted = std::get_if<PatchDescriptions>(&slots);
        CHECK(slotted && slotted->front().nvn_policy.submission_slots == 12);
        CHECK(defaults && defaults->front().nvn_policy.submission_slots == 0);
        CHECK(std::holds_alternative<std::string>(
            parse("true", ",\"nvn_policy\":{\"submission_slots\":0}")));
        CHECK(std::holds_alternative<std::string>(
            parse("true", ",\"nvn_policy\":{\"submission_slots\":17}")));
        CHECK(std::holds_alternative<std::string>(parse("true", ",\"nvn_policy\":true")));
    }
    CHECK(std::holds_alternative<std::string>(parse("true", ",\"unknown\":0")));
    CHECK(std::holds_alternative<std::string>(parse("true", ",\"calls\":[]")));
    std::printf("NVN compatibility failures: %u\n", failures);
    return failures ? 1 : 0;
}
