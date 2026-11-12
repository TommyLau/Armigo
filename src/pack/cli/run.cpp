// ============================================================================
// Armigo Tools - Development and Verification Tools
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include "run.h"

#include "ios.h"
#include "package_internal.h"

#include <spawn.h>
#include <sys/clonefile.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <optional>
#include <span>
#include <string_view>
#include <system_error>
#include <utility>

extern char** environ;

namespace armigo::pack {
namespace {

namespace fs = std::filesystem;
namespace detail = armigo::shader::package::detail;

void set_env(RunPlan& plan, std::string key, std::string value) {
    for (auto& entry : plan.environment) {
        if (entry.first == key) {
            entry.second = std::move(value);
            return;
        }
    }
    plan.environment.emplace_back(std::move(key), std::move(value));
}

void add_option(RunPlan& plan, std::string_view flag, std::string value) {
    plan.arguments.emplace_back(flag);
    plan.arguments.push_back(std::move(value));
}

void add_option(RunPlan& plan, std::string_view flag, const fs::path& value) {
    add_option(plan, flag, value.string());
}

std::string lower_extension(const fs::path& path) {
    std::string extension = path.extension().string();
    for (char& character : extension) {
        if (character >= 'A' && character <= 'Z') character = static_cast<char>(character - 'A' + 'a');
    }
    return extension;
}

bool has_flag(const RunPlan& plan, std::string_view flag) {
    return std::find(plan.arguments.begin(), plan.arguments.end(), std::string(flag)) != plan.arguments.end();
}

std::string list_addons(const fs::path& root, RunPlan& plan) {
    std::error_code error;
    if (!fs::exists(root, error)) {
        if (error) return "Cannot list add-on content: " + root.string() + ".";
        return {};
    }
    if (!fs::is_directory(root, error)) return "Add-on content is not a directory: " + root.string() + ".";
    std::vector<fs::path> addons;
    error.clear();
    for (fs::directory_iterator cursor(root, error), end; !error && cursor != end; cursor.increment(error)) {
        std::error_code entry_error;
        if (cursor->is_directory(entry_error)) addons.push_back(cursor->path());
        else if (entry_error) return "Cannot list add-on content: " + root.string() + ".";
    }
    if (error) return "Cannot list add-on content: " + root.string() + ".";
    std::sort(addons.begin(), addons.end());
    for (const fs::path& addon : addons) add_option(plan, "--addon", addon);
    return {};
}

std::string append_app(const Command& command, const fs::path& game, const fs::path& scratch, RunPlan& plan) {
    std::error_code error;
    if (!fs::is_directory(game, error)) return "The packed app is not a directory: " + game.string() + ".";
    auto identity = read_packed_identity(game);
    if (const auto* failure = std::get_if<std::string>(&identity)) return *failure;
    const PackedIdentity& packed = std::get<PackedIdentity>(identity);
    const fs::path resources = game / "Contents" / "Resources" / "Armigo";
    const fs::path exefs = resources / "game" / "exefs";
    const fs::path romfs = resources / "game" / "romfs";
    // A packed .nup title keeps its archive whole as game/title.nup.
    const fs::path archive = resources / "game" / "title.nup";
    if (!fs::is_directory(romfs, error) && fs::is_regular_file(archive, error)) {
        add_option(plan, "--nup", archive);
    } else if (!fs::is_directory(exefs, error) || !fs::is_directory(romfs, error)) {
        return "The packed app has no game code and data: " + game.string() + ".";
    } else {
        add_option(plan, "--exefs", exefs);
        add_option(plan, "--romfs", romfs);
    }
    if (!command.run_options.shaders.empty()) add_option(plan, "--shaders", command.run_options.shaders);
    else if (fs::is_directory(resources / "shaders", error)) add_option(plan, "--shaders", resources / "shaders");
    if (const std::string failure = list_addons(resources / "game" / "addons", plan); !failure.empty()) return failure;

    set_env(plan, "ARMIGO_TITLES", (scratch / "titles").string());
    plan.directories.push_back(scratch / "titles");
    error.clear();
    const fs::path title = fs::absolute(resources / "title", error);
    if (error) return "Cannot resolve the packed title settings: " + (resources / "title").string() + ".";
    plan.links.emplace_back(title, scratch / "titles" / packed.title_id);

    if (command.run_options.fonts.empty() && fs::is_directory(resources / "system", error)) {
        set_env(plan, "ARMIGO_SHARED_FONTS", (resources / "system").string());
    }
    const fs::path cache = scratch / "pso";
    set_env(plan, "ARMIGO_PSO_CACHE", cache.string());
    // The launch reads a private copy so the packed archive stays unchanged.
    if (fs::is_directory(resources / "pipelines", error)) plan.clones.emplace_back(resources / "pipelines", cache);
    else plan.directories.push_back(cache);
    return {};
}

std::string append_nup(const fs::path& game, RunPlan& plan) {
    std::error_code error;
    if (!fs::is_regular_file(game, error)) return "The .nup package is not a file: " + game.string() + ".";
    add_option(plan, "--nup", game);
    return {};
}

std::string append_container(const Command& command, const fs::path& game, const fs::path& scratch, RunPlan& plan) {
    const RunOptions& options = command.run_options;
    if (options.keys.empty()) return "An XCI or NSP requires --keys.";
    std::error_code error;
    if (!fs::is_regular_file(game, error)) return "The container is not a file: " + game.string() + ".";
    if (!fs::is_regular_file(options.keys, error)) return "The keys file is not a file: " + options.keys.string() + ".";
    if (!options.title_keys.empty() && !fs::is_regular_file(options.title_keys, error)) {
        return "The title keys file is not a file: " + options.title_keys.string() + ".";
    }
    if (!options.update.empty() && !fs::is_regular_file(options.update, error)) {
        return "The update is not a file: " + options.update.string() + ".";
    }
    add_option(plan, "--container", game);
    add_option(plan, "--keys", options.keys);
    if (!options.title_keys.empty()) add_option(plan, "--title-keys", options.title_keys);
    if (!options.update.empty()) add_option(plan, "--update", options.update);
    const fs::path cache = scratch / "extract";
    add_option(plan, "--cache", cache);
    plan.directories.push_back(cache);
    return {};
}

std::string append_input(const Command& command, const fs::path& scratch, RunPlan& plan) {
    const fs::path game = command.game.lexically_normal();
    const std::string extension = lower_extension(game);
    std::error_code error;
    if (extension == ".app") return append_app(command, game, scratch, plan);
    if (extension == ".nup") return append_nup(game, plan);
    if (extension == ".xci" || extension == ".nsp") return append_container(command, game, scratch, plan);
    if (fs::is_directory(game, error) && fs::is_directory(game / "exefs", error) &&
        fs::is_directory(game / "romfs", error)) {
        add_option(plan, "--exefs", game / "exefs");
        add_option(plan, "--romfs", game / "romfs");
        return {};
    }
    return "The game is not a packed app, a .nup package, an XCI or NSP, or an extracted folder: " + game.string() +
           ".";
}

void append_common(const Command& command, const fs::path& scratch, RunPlan& plan) {
    const RunOptions& options = command.run_options;
    const fs::path saves = options.saves.empty() ? scratch / "saves" : options.saves;
    const fs::path report = options.report.empty() ? scratch / "report.json" : options.report;
    if (!options.shaders.empty() && !has_flag(plan, "--shaders")) add_option(plan, "--shaders", options.shaders);
    plan.directories.push_back(saves);
    if (!report.parent_path().empty()) plan.directories.push_back(report.parent_path());
    add_option(plan, "--saves", saves);
    add_option(plan, "--frames", std::to_string(options.frames));
    add_option(plan, "--time-limit", options.time_limit);
    add_option(plan, "--report", report);
    if (options.width != 0 && options.height != 0) {
        add_option(plan, "--width", std::to_string(options.width));
        add_option(plan, "--height", std::to_string(options.height));
    }
}

void append_environment(const Command& command, RunPlan& plan) {
    const RunOptions& options = command.run_options;
    // Headless stays muted when ARMIGO_AUDIO is unset, so --audio has to set a
    // value that plays and replaces an inherited mute.
    set_env(plan, "ARMIGO_AUDIO", options.audio ? "1" : "0");
    if (!options.rate.empty()) set_env(plan, "ARMIGO_COMPAT_FRAME_RATE", options.rate);
    if (!options.fonts.empty()) set_env(plan, "ARMIGO_SHARED_FONTS", options.fonts.string());
}

std::string one_decimal(double value) {
    if (value < 0) value = 0;
    const auto scaled = static_cast<unsigned long long>(std::llround(value * 10.0));
    return std::to_string(scaled / 10) + "." + std::to_string(scaled % 10);
}

std::optional<std::uint64_t> whole_number(const detail::Json& value) {
    if (!value.is_number()) return std::nullopt;
    const std::string& text = std::get<detail::JsonNumber>(value.value).text;
    if (text.empty() || text.front() == '-') return std::nullopt;
    std::size_t used = 0;
    try {
        const auto parsed = std::stoull(text, &used);
        if (used != text.size()) return std::nullopt;
        return parsed;
    } catch (const std::exception&) {
        return std::nullopt;
    }
}

void print_command_line(const std::vector<std::string>& arguments) {
    for (std::size_t index = 0; index < arguments.size(); ++index) {
        if (index != 0) std::fputc(' ', stdout);
        const std::string& argument = arguments[index];
        const bool quote = argument.empty() || argument.find_first_of(" \t\n\"\\") != std::string::npos;
        if (!quote) {
            std::fputs(argument.c_str(), stdout);
            continue;
        }
        std::fputc('"', stdout);
        for (const char character : argument) {
            if (character == '"' || character == '\\') std::fputc('\\', stdout);
            std::fputc(character, stdout);
        }
        std::fputc('"', stdout);
    }
    std::fputc('\n', stdout);
    std::fflush(stdout);
}

void print_error(const std::string& message) {
    std::fprintf(stderr, "%s\n", message.c_str());
}

std::string ignored_note(const Command& command) {
    std::vector<std::string> names;
    if (!command.run_options.keys.empty()) names.emplace_back("--keys");
    if (!command.run_options.title_keys.empty()) names.emplace_back("--title-keys");
    if (!command.run_options.update.empty()) names.emplace_back("--update");
    if (names.empty()) return {};
    std::string text = "Ignoring ";
    for (std::size_t index = 0; index < names.size(); ++index) {
        if (index != 0) text += index + 1 == names.size() ? " and " : ", ";
        text += names[index];
    }
    text += " for this input.";
    return text;
}

// Deletes a tree without following a directory symlink, so a title-settings
// link cannot remove the directory it points at.
void remove_tree(const fs::path& path) {
    try {
        std::error_code error;
        const fs::file_status status = fs::symlink_status(path, error);
        if (error || status.type() == fs::file_type::not_found || status.type() == fs::file_type::none) return;
        if (status.type() == fs::file_type::directory) {
            for (fs::directory_iterator cursor(path, error), end; !error && cursor != end; cursor.increment(error)) {
                remove_tree(cursor->path());
            }
        }
        error.clear();
        fs::remove(path, error);
    } catch (const std::exception&) {
    }
}

// True when `candidate` is `directory` or a path inside it. An inconclusive
// check reports true so a user directory is kept.
bool path_contains(const fs::path& directory, const fs::path& candidate) {
    std::error_code error;
    const fs::path root = fs::absolute(directory, error).lexically_normal();
    if (error) return true;
    error.clear();
    const fs::path child = fs::absolute(candidate, error).lexically_normal();
    if (error) return true;
    const fs::path relative = child.lexically_relative(root);
    if (relative.empty()) return true;
    for (const fs::path& part : relative) {
        if (part == "..") return false;
    }
    return true;
}

struct TemporaryDirectory {
    fs::path path;
    fs::path user_saves;

    ~TemporaryDirectory() {
        if (path.empty()) return;
        if (!user_saves.empty() && path_contains(path, user_saves)) return;
        remove_tree(path);
    }
};

std::string clone_or_copy(const fs::path& source, const fs::path& destination) {
    if (::clonefile(source.c_str(), destination.c_str(), 0) == 0) return {};
    remove_tree(destination);
    std::error_code error;
    fs::copy(source, destination, fs::copy_options::recursive | fs::copy_options::copy_symlinks, error);
    if (error) return "Cannot copy the pipeline archive to " + destination.string() + ".";
    return {};
}

std::string prepare_plan(const RunPlan& plan) {
    std::error_code error;
    for (const fs::path& directory : plan.directories) {
        if (directory.empty()) continue;
        error.clear();
        fs::create_directories(directory, error);
        if (error) return "Cannot create " + directory.string() + ".";
    }
    for (const auto& [target, link] : plan.links) {
        error.clear();
        fs::create_directory_symlink(target, link, error);
        if (error) return "Cannot link title settings to " + link.string() + ".";
    }
    for (const auto& [source, destination] : plan.clones) {
        if (const std::string failure = clone_or_copy(source, destination); !failure.empty()) return failure;
    }
    return {};
}

fs::path report_path(const RunPlan& plan) {
    for (std::size_t index = 0; index + 1 < plan.arguments.size(); ++index) {
        if (plan.arguments[index] == "--report") return plan.arguments[index + 1];
    }
    return {};
}

std::string read_report(const fs::path& path) {
    std::error_code error;
    if (!fs::is_regular_file(path, error)) return {};
    const auto size = fs::file_size(path, error);
    constexpr std::uintmax_t maximum = 64u * 1024u * 1024u;
    if (error || size > maximum) return {};
    std::ifstream input(path, std::ios::binary);
    if (!input) return {};
    std::string text(static_cast<std::size_t>(size), '\0');
    input.read(text.data(), static_cast<std::streamsize>(size));
    if (!input) return {};
    return text;
}

std::string environment_name(std::string_view entry) {
    const auto equals = entry.find('=');
    return std::string(equals == std::string_view::npos ? entry : entry.substr(0, equals));
}

bool environment_overridden(const RunPlan& plan, std::string_view name) {
    for (const auto& entry : plan.environment) {
        if (entry.first == name) return true;
    }
    return false;
}

std::optional<int> launch(const RunPlan& plan) {
    std::vector<std::string> environment;
    if (environ != nullptr) {
        for (char** entry = environ; *entry != nullptr; ++entry) {
            if (!environment_overridden(plan, environment_name(*entry))) environment.emplace_back(*entry);
        }
    }
    for (const auto& [key, value] : plan.environment) environment.push_back(key + "=" + value);
    std::vector<char*> envp;
    envp.reserve(environment.size() + 1);
    for (std::string& entry : environment) envp.push_back(entry.data());
    envp.push_back(nullptr);

    std::vector<std::string> arguments = plan.arguments;
    std::vector<char*> argv;
    argv.reserve(arguments.size() + 1);
    for (std::string& argument : arguments) argv.push_back(argument.data());
    argv.push_back(nullptr);

    pid_t pid = 0;
    const int spawned = ::posix_spawn(&pid, argv[0], nullptr, nullptr, argv.data(), envp.data());
    if (spawned != 0) {
        print_error("Cannot start " + arguments.front() + ": " + std::strerror(spawned) + ".");
        return std::nullopt;
    }
    int status = 0;
    pid_t waited = 0;
    do {
        waited = ::waitpid(pid, &status, 0);
    } while (waited < 0 && errno == EINTR);
    if (waited < 0) {
        print_error("Cannot wait for the headless runner.");
        return std::nullopt;
    }
    return status;
}

} // namespace

std::variant<std::string, RunPlan> plan_run(const Command& command, const fs::path& runner, const fs::path& scratch) {
    std::error_code error;
    if (!command.run_options.shaders.empty() && !fs::is_directory(command.run_options.shaders, error)) {
        return "--shaders is not a directory: " + command.run_options.shaders.string() + ".";
    }
    if (!command.run_options.fonts.empty() && !fs::is_directory(command.run_options.fonts, error)) {
        return "--fonts is not a directory: " + command.run_options.fonts.string() + ".";
    }
    RunPlan plan;
    plan.arguments.push_back(runner.string());
    plan.arguments.emplace_back("run");
    if (const std::string failure = append_input(command, scratch, plan); !failure.empty()) return failure;
    append_common(command, scratch, plan);
    append_environment(command, plan);
    return plan;
}

std::string summarize_report(std::string_view report_json) {
    try {
        const auto parsed = detail::parse_json(std::as_bytes(std::span(report_json.data(), report_json.size())));
        const detail::Json* outcome = detail::find_member(parsed, "outcome");
        const detail::Json* statistics = detail::find_member(parsed, "statistics");
        if (!outcome || !outcome->is_string() || !statistics) return {};
        const detail::Json* frames = detail::find_member(*statistics, "completed_frames");
        const detail::Json* elapsed = detail::find_member(*statistics, "elapsed_ns");
        if (!frames || !elapsed) return {};
        const auto frame_count = whole_number(*frames);
        const auto elapsed_ns = whole_number(*elapsed);
        if (!frame_count || !elapsed_ns) return {};
        const double seconds = static_cast<double>(*elapsed_ns) / 1000000000.0;
        const double fps = seconds > 0.0 ? static_cast<double>(*frame_count) / seconds : 0.0;
        if (!std::isfinite(seconds) || !std::isfinite(fps)) return {};
        std::string summary = "Run: " + outcome->string() + ", " + std::to_string(*frame_count) + " frames in " +
            one_decimal(seconds) + " s, " + one_decimal(fps) + " fps";
        // A frame-target run keeps its outcome when teardown fails afterwards;
        // the runner's exit code says so.
        if (const detail::Json* teardown = detail::find_member(parsed, "teardown_failure");
            teardown && !teardown->is_null())
            summary += "; teardown failed";
        return summary;
    } catch (const std::exception&) {
        return {};
    }
}

int run_game(const Command& command, const fs::path& executable) {
    try {
        if (executable.empty()) {
            print_error("Cannot locate the packer executable.");
            return 1;
        }
        const fs::path runner = executable.parent_path() / "armigo-headless";
        std::error_code error;
        if (!fs::is_regular_file(runner, error)) {
            print_error("The headless runner is missing: " + runner.string() + ".");
            return 1;
        }
        std::string scratch_name = (fs::temp_directory_path() / "armigo-run-XXXXXX").string();
        if (::mkdtemp(scratch_name.data()) == nullptr) {
            print_error(std::string("Cannot create a temporary directory: ") + std::strerror(errno) + ".");
            return 1;
        }
        TemporaryDirectory scratch;
        scratch.path = scratch_name;
        scratch.user_saves = command.run_options.saves;

        const auto planned = plan_run(command, runner, scratch.path);
        if (const auto* failure = std::get_if<std::string>(&planned)) {
            print_error(*failure);
            return 1;
        }
        const RunPlan& plan = std::get<RunPlan>(planned);
        if (!has_flag(plan, "--container")) {
            if (const std::string note = ignored_note(command); !note.empty()) {
                std::fprintf(stderr, "%s\n", note.c_str());
            }
        }
        if (const std::string failure = prepare_plan(plan); !failure.empty()) {
            print_error(failure);
            return 1;
        }
        // A report left by an earlier run must not stand in for this one.
        if (const fs::path report = report_path(plan); !report.empty()) {
            std::error_code removed;
            fs::remove(report, removed);
        }
        print_command_line(plan.arguments);
        const std::optional<int> status = launch(plan);
        if (!status) return 1;
        const int exit_code = WIFEXITED(*status) ? WEXITSTATUS(*status) : 1;
        const std::string summary = summarize_report(read_report(report_path(plan)));
        if (summary.empty()) std::printf("Run: no report (runner exit %d)\n", exit_code);
        else std::printf("%s\n", summary.c_str());
        std::fflush(stdout);
        return exit_code;
    } catch (const std::exception& failure) {
        print_error(failure.what());
        return 1;
    }
}

} // namespace armigo::pack
