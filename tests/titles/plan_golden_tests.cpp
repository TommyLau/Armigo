// ============================================================================
// Armigo Tests - Title Patch Plan Characterization
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

// Pins what every title folder selects, so a refactor of the patch planning
// cannot change a single word: for each description and each requested mode,
// with and without the NVN driver, the ordered selected words, the selected
// calls and the hook bytes generated for them, and for resolution sites the
// planned words on a synthetic module. The expected file keeps one digest per
// case and never the words themselves (those live only in titles/<TitleID>/).
// Usage: plan_golden_tests <titles> <expected> [--write]
// ARMIGO_GOLDEN_DUMP=<file> also writes the canonical text the digests hash,
// for diffing two trees.

#include <armigo/compat/title_settings.h>
#include <armigo/titles/resolution.h>
#include <armigo/loader/nso_module.h>

#include <algorithm>
#include <bit>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

namespace {

using namespace armigo;
namespace title = titles::resolution;

std::string hex(std::uint64_t value) {
    char text[24];
    std::snprintf(text, sizeof text, "%" PRIx64, value);
    return text;
}

std::uint64_t digest(const std::string& text) {
    std::uint64_t hash = 0xcbf29ce484222325ull;
    for (const unsigned char c : text) hash = (hash ^ c) * 0x100000001b3ull;
    return hash;
}

class Recorder {
public:
    void add(const std::string& key, const std::string& canonical) {
        char text[24];
        std::snprintf(text, sizeof text, "%016" PRIx64, digest(canonical));
        lines_[key] = text;
        dump_ << "== " << key << "\n" << canonical;
    }
    const std::map<std::string, std::string>& lines() const { return lines_; }
    std::string dump() const { return dump_.str(); }
private:
    std::map<std::string, std::string> lines_;
    std::ostringstream dump_;
};

void describe(std::ostringstream& out, const compat::PatchDescription& patch) {
    out << "module=" << patch.module << " callee=" << hex(patch.original_callee)
        << " nvn=" << patch.nvn_driver << " startup_user=" << patch.startup_user
        << " present_link=" << patch.present_link
        << " pipelines=" << std::filesystem::path(patch.pso_manifest).filename().string() << "\n";
    const auto& policy = patch.nvn_policy;
    out << "policy " << policy.cpu_cached_texture_writes << policy.segment_timestamps
        << policy.game_barriers << policy.hoist_implicit_samples << "\n";
    const auto& layout = patch.layout;
    out << "layout " << hex(layout.framework_pointer) << ' ' << hex(layout.framework_period) << ' '
        << hex(layout.override_flag) << ' ' << hex(layout.ratio) << ' ' << hex(layout.interval) << ' '
        << hex(layout.base_interval) << ' ' << hex(layout.frame_step) << ' ' << int(layout.variant) << ' '
        << layout.object_register << ' ' << hex(layout.state) << ' ' << layout.active_state << ' '
        << hex(layout.counter) << ' ' << hex(layout.repeat_flag) << ' ' << layout.repeat_frames << "\n";
    if (patch.frame_rate) {
        out << "rates default=" << patch.frame_rate->default_rate
            << " display=" << patch.frame_rate->display_refresh_rate << " supported=";
        for (auto rate : patch.frame_rate->supported_rates) out << rate << ',';
        out << " experimental=";
        for (auto rate : patch.frame_rate->experimental_rates) out << rate << ',';
        out << "\n";
    }
    const auto& inputs = patch.shader_inputs;
    out << "shaders containers=" << int(inputs.containers) << " depth=" << int(inputs.vertex_depth) << " caches=";
    for (const auto& cache : inputs.caches) out << cache << ',';
    for (const auto& fallback : inputs.fallbacks) {
        out << " fallback=" << fallback.cache << ':';
        for (const auto& alternative : fallback.alternatives) out << alternative << ',';
    }
    out << " ignored=";
    for (auto offset : inputs.ignored_needle_offsets) out << hex(offset) << ',';
    // Native selections are written only when set, so other descriptions keep their text.
    if (!inputs.dictionary_member.empty()) out << " dictionary=" << inputs.dictionary_member;
    if (!inputs.members.empty()) {
        out << " members=";
        for (const auto& member : inputs.members) out << member << ',';
    }
    if (!inputs.identities.empty()) {
        out << " identities=";
        for (const auto& identity : inputs.identities) out << identity << ',';
    }
    out << "\nplatforms=";
    for (const auto& platform : patch.platforms) out << platform << ',';
    out << "\nwords=" << patch.words.size() << " calls=" << patch.calls.size() << "\n";
}

void word_line(std::ostringstream& out, const compat::WordPatch& word) {
    out << "word " << hex(word.offset) << ' ' << hex(word.expected_instruction) << ' '
        << hex(word.replacement_instruction) << " nvn=" << word.nvn_driver_only << " rate=" << word.frame_rate
        << " note=" << word.note << "\n";
}

// Hooks are emitted for a module at zero, stubs after its text and the gate
// beyond them, in the order the launchers install them.
constexpr std::uint64_t stub_base = 0x6000000, gate_base = 0x7000000;

void call_line(std::ostringstream& out, const compat::PatchDescription& patch, const compat::CallPatch& call,
               std::uint64_t stub) {
    out << "call " << hex(call.offset) << ' ' << hex(call.expected_instruction) << " rate=" << call.frame_rate
        << " nvn=" << call.nvn_driver_only << " callee=" << hex(compat::call_callee(patch, call));
    if (call.phases)
        out << " phases=" << hex(call.phases->original_callee) << ':' << hex(call.phases->counter) << ':'
            << call.phases->phase_count;
    if (call.gpu_budget) out << " gpu_budget=" << call.gpu_budget->target_rate;
    if (call.frame_ratio) out << " frame_ratio=" << call.frame_ratio->divisor_register;
    out << " note=" << call.note << "\n";
    const auto hook = compat::make_patch_hook(call, call.expected_instruction, call.offset, stub,
                                              {gate_base, 0x1000});
    if (!hook.ok()) {
        out << "  hook error=" << int(hook.error) << "\n";
        return;
    }
    out << "  hook " << hex(hook.patch->replacement_instruction);
    for (auto word : hook.patch->trampoline) out << ' ' << hex(word);
    out << " site=" << hex(hook.patch->site.return_address) << ':' << hex(hook.patch->site.call_return_address)
        << ':' << hex(hook.patch->site.target_address) << "\n";
}

void set_environment(const char* name, const char* value) {
    if (value) setenv(name, value, 1);
    else unsetenv(name);
}

void characterize_modes(Recorder& recorder, const std::string& prefix, const compat::PatchDescription& patch) {
    static const char* const modes[] = {"", "0", "30", "45", "60", "120", "variable"};
    for (const char* mode : modes) {
        for (const bool nvn : {true, false}) {
            std::ostringstream out;
            std::uint32_t admitted = 0;
            const auto words = compat::select_patch_words(patch, mode, nvn, &admitted);
            if (const auto* error = std::get_if<std::string>(&words)) out << "words error=" << *error << "\n";
            else {
                out << "admitted=" << admitted << "\n";
                for (const auto& word : std::get<std::vector<compat::WordPatch>>(words)) word_line(out, word);
            }
            const auto calls = compat::select_patch_calls(patch, mode, nvn);
            if (const auto* error = std::get_if<std::string>(&calls)) out << "calls error=" << *error << "\n";
            else {
                std::uint64_t stub = stub_base;
                for (const auto& call : std::get<std::vector<compat::CallPatch>>(calls)) {
                    call_line(out, patch, call, stub);
                    stub += compat::call_stub_bytes(call);
                }
            }
            recorder.add(prefix + " mode=" + (*mode ? mode : "default") + " nvn=" + (nvn ? "1" : "0"), out.str());
        }
    }
}

std::uint32_t move(bool wide, bool keep, unsigned shift, std::uint16_t immediate, unsigned rd) {
    return (std::uint32_t(wide) << 31) | ((keep ? 3u : 2u) << 29) | (0b100101u << 23) |
        ((shift / 16) << 21) | (std::uint32_t(immediate) << 5) | rd;
}

// A synthetic module holding a console output of 1280x720 at every resolution
// site: render-base 1024, a 0.75 floor, a load scale of 30 and small heaps.
loader::NsoModuleImage resolution_module(const compat::ResolutionSites& sites) {
    std::uint64_t end = 0;
    for (const auto& pair : sites.tv_size) end = std::max({end, pair[0], pair[1]});
    for (auto offset : {sites.screen_widths[0], sites.screen_widths[1], sites.display_size[0], sites.display_size[1],
                        sites.window_crop[0], sites.window_crop[1], sites.floor[0], sites.floor[1],
                        sites.render_target_pool, sites.graphics_heap, sites.process_heap, sites.root_heap,
                        sites.load_scale})
        end = std::max(end, offset);
    loader::NsoModuleImage image;
    image.bytes.resize(end + 0x1000);
    image.segments[0].image_offset = 0;
    image.segments[0].content_size = end + 0x100;
    const auto put = [&](std::uint64_t offset, std::uint32_t word) {
        std::memcpy(image.bytes.data() + offset, &word, 4);
    };
    const auto floor = std::bit_cast<std::uint32_t>(0.75f), scale = std::bit_cast<std::uint32_t>(30.0f);
    for (const auto& pair : sites.tv_size) {
        put(pair[0], move(true, false, 0, 1280, 9));
        put(pair[1], move(true, true, 32, 720, 9));
    }
    put(sites.screen_widths[0], move(true, false, 16, std::bit_cast<std::uint32_t>(1280.0f) >> 16, 8));
    put(sites.screen_widths[1], move(true, true, 48, std::bit_cast<std::uint32_t>(1024.0f) >> 16, 8));
    put(sites.display_size[0], move(true, false, 0, 1280, 8));
    put(sites.display_size[1], move(true, true, 32, 720, 8));
    put(sites.window_crop[0], move(false, false, 0, 1280, 3));
    put(sites.window_crop[1], move(false, false, 0, 720, 4));
    put(sites.floor[0], move(false, false, 0, std::uint16_t(floor), 8));
    put(sites.floor[1], move(false, true, 16, std::uint16_t(floor >> 16), 8));
    put(sites.render_target_pool, move(false, false, 16, 0x100, 10));
    put(sites.graphics_heap, move(false, false, 16, 0x200, 0));
    put(sites.process_heap, move(false, false, 16, 0x8000, 0));
    put(sites.root_heap, move(false, false, 16, 0x7f00, 8));
    put(sites.load_scale, move(false, false, 16, std::uint16_t(scale >> 16), 8));
    return image;
}

void characterize_resolution(Recorder& recorder, const std::string& prefix, const compat::PatchDescription& patch) {
    const auto image = resolution_module(*patch.resolution);
    static const char* const widths[] = {nullptr, "1280", "1600", "1920", "2560", "3840", "640"};
    struct Rates { const char* frame_rate; const char* resolution_rate; const char* gpu_busy; };
    static const Rates rates[] = {{nullptr, nullptr, nullptr}, {"60", "60", nullptr}, {"120", "120", nullptr},
                                  {"120", "120", "0"}, {"60", "60", "1"}};
    for (const char* width : widths) {
        for (const auto& rate : rates) {
            set_environment("ARMIGO_COMPAT_RESOLUTION_WIDTH", width);
            set_environment("ARMIGO_COMPAT_FRAME_RATE", rate.frame_rate);
            set_environment("ARMIGO_COMPAT_RESOLUTION_RATE", rate.resolution_rate);
            set_environment("ARMIGO_COMPAT_RESOLUTION_GPU_BUSY", rate.gpu_busy);
            std::ostringstream out;
            const auto settings = title::resolution_settings(*patch.resolution);
            if (const auto* error = std::get_if<std::string>(&settings)) out << "settings error=" << *error << "\n";
            else {
                const auto& chosen = std::get<title::ResolutionSettings>(settings);
                out << "settings width=" << chosen.maximum_width << " rate=" << chosen.target_rate
                    << " gpu_budget=" << chosen.gpu_budget << "\n";
                const auto planned = title::resolution_words(*patch.resolution, chosen, image);
                if (const auto* error = std::get_if<std::string>(&planned)) out << "plan error=" << *error << "\n";
                else {
                    const auto& plan = std::get<title::ResolutionPlan>(planned);
                    out << "plan " << plan.width << 'x' << plan.height << ' ' << hex(plan.render_target_pool) << ' '
                        << hex(plan.graphics_heap) << ' ' << hex(plan.process_heap) << ' ' << hex(plan.root_heap)
                        << ' ' << hex(std::bit_cast<std::uint32_t>(plan.floor)) << ' '
                        << hex(std::bit_cast<std::uint32_t>(plan.load_scale)) << "\n";
                    for (const auto& word : plan.words) word_line(out, word);
                    if (plan.gpu_budget_hook) call_line(out, patch, *plan.gpu_budget_hook, stub_base);
                }
            }
            std::string key = prefix + " resolution width=" + (width ? width : "default") +
                " rate=" + (rate.frame_rate ? rate.frame_rate : "default") + " gpu_busy=" +
                (rate.gpu_busy ? rate.gpu_busy : "default");
            recorder.add(key, out.str());
        }
    }
    for (const char* name : {"ARMIGO_COMPAT_RESOLUTION_WIDTH", "ARMIGO_COMPAT_FRAME_RATE",
                             "ARMIGO_COMPAT_RESOLUTION_RATE", "ARMIGO_COMPAT_RESOLUTION_GPU_BUSY"})
        unsetenv(name);
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: %s <titles> <expected> [--write]\n", argv[0]);
        return 2;
    }
    const std::filesystem::path titles = argv[1], expected = argv[2];
    const bool write = argc > 3 && std::string_view(argv[3]) == "--write";
    for (const char* name : {"ARMIGO_COMPAT_RESOLUTION_WIDTH", "ARMIGO_COMPAT_FRAME_RATE",
                             "ARMIGO_COMPAT_RESOLUTION_RATE", "ARMIGO_COMPAT_RESOLUTION_GPU_BUSY"})
        unsetenv(name);

    std::vector<std::filesystem::path> folders;
    for (const auto& entry : std::filesystem::directory_iterator(titles))
        if (entry.is_directory()) folders.push_back(entry.path());
    std::sort(folders.begin(), folders.end());
    Recorder recorder;
    for (const auto& folder : folders) {
        const auto name = folder.filename().string();
        const auto id = compat::parse_title_id_text(name);
        if (!id) continue;
        const auto read = compat::read_title_settings(folder, *id);
        if (const auto* error = std::get_if<std::string>(&read)) {
            std::fprintf(stderr, "%s: %s\n", name.c_str(), error->c_str());
            return 1;
        }
        const auto& settings = std::get<compat::TitleSettings>(read);
        for (std::size_t index = 0; index < settings.patches.size(); ++index) {
            const auto& patch = settings.patches[index];
            const auto prefix = name + " d" + std::to_string(index);
            std::ostringstream out;
            describe(out, patch);
            recorder.add(prefix + " description", out.str());
            characterize_modes(recorder, prefix, patch);
            if (patch.resolution) characterize_resolution(recorder, prefix, patch);
        }
    }

    if (const char* dump = std::getenv("ARMIGO_GOLDEN_DUMP"); dump && *dump) {
        std::ofstream out(dump);
        out << recorder.dump();
    }
    if (write) {
        std::ofstream out(expected);
        for (const auto& [key, value] : recorder.lines()) out << key << " = " << value << "\n";
        std::printf("wrote %zu cases\n", recorder.lines().size());
        return 0;
    }
    std::map<std::string, std::string> wanted;
    std::ifstream in(expected);
    for (std::string line; std::getline(in, line);) {
        const auto separator = line.rfind(" = ");
        if (separator != std::string::npos) wanted[line.substr(0, separator)] = line.substr(separator + 3);
    }
    unsigned failures = 0;
    for (const auto& [key, value] : recorder.lines()) {
        const auto found = wanted.find(key);
        if (found == wanted.end()) { std::fprintf(stderr, "new case: %s\n", key.c_str()); ++failures; }
        else if (found->second != value) { std::fprintf(stderr, "changed: %s\n", key.c_str()); ++failures; }
    }
    for (const auto& [key, value] : wanted)
        if (!recorder.lines().count(key)) { std::fprintf(stderr, "missing case: %s\n", key.c_str()); ++failures; }
    if (failures) {
        std::fprintf(stderr, "%u plan cases differ from %s (ARMIGO_GOLDEN_DUMP=<file> prints them)\n",
                     failures, expected.string().c_str());
        return 1;
    }
    std::printf("%zu plan cases match\n", recorder.lines().size());
    return 0;
}
