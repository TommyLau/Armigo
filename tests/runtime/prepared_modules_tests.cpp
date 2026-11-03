// ============================================================================
// Armigo Tests - Native Runtime Verification
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include <armigo/runtime/preparation_format.h>
#include <armigo/runtime/prepared_modules.h>

#include "preparation_fixture.h"
#include "../../src/runtime/prepared_artifact.h"

#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <ranges>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

#include <fcntl.h>
#include <sys/wait.h>
#include <unistd.h>

namespace {
using namespace armigo;
using namespace armigo::runtime;

unsigned failures = 0;
#define CHECK(condition) do { if (!(condition)) { \
    std::fprintf(stderr, "Check failed at line %u: %s\n", __LINE__, #condition); \
    ++failures; \
} } while (false)

template<class T>
const T* value(const LaunchResult<T>& result) {
    return std::get_if<T>(&result);
}

template<class T>
bool failed(const LaunchResult<T>& result, LaunchError code) {
    const auto* failure = std::get_if<LaunchFailure>(&result);
    return failure != nullptr && failure->stage == LaunchStage::preparation &&
           failure->code == code && !failure->operation.empty() && !failure->detail.empty();
}

bool failed(const LaunchStatus& result) {
    return std::holds_alternative<LaunchFailure>(result);
}

struct TemporaryDirectory {
    std::filesystem::path path;

    TemporaryDirectory() {
        auto pattern = (std::filesystem::temp_directory_path() /
                        "armigo-prepared-modules-XXXXXX").string();
        std::vector<char> writable(pattern.begin(), pattern.end());
        writable.push_back('\0');
        if (char* created = ::mkdtemp(writable.data())) path = created;
    }
    ~TemporaryDirectory() {
        std::error_code ignored;
        std::filesystem::remove_all(path, ignored);
    }
};

using FileTree = std::vector<std::pair<std::filesystem::path, std::vector<std::byte>>>;

FileTree read_tree(const std::filesystem::path& root) {
    FileTree tree;
    for (const auto& entry : std::filesystem::recursive_directory_iterator(root)) {
        if (!entry.is_regular_file()) continue;
        std::ifstream stream(entry.path(), std::ios::binary);
        std::vector<char> chars((std::istreambuf_iterator<char>(stream)),
                                std::istreambuf_iterator<char>());
        std::vector<std::byte> bytes(chars.size());
        std::transform(chars.begin(), chars.end(), bytes.begin(),
                       [](char byte) { return std::byte(static_cast<unsigned char>(byte)); });
        tree.emplace_back(std::filesystem::relative(entry.path(), root), std::move(bytes));
    }
    std::ranges::sort(tree, {}, &FileTree::value_type::first);
    return tree;
}

bool equal_artifacts(const PreparedModuleArtifacts& left,
                     const PreparedModuleArtifacts& right) {
    if (left.format_version != right.format_version ||
        left.preparation_identity != right.preparation_identity ||
        left.modules.size() != right.modules.size()) return false;
    for (std::size_t index = 0; index < left.modules.size(); ++index) {
        const auto& a = left.modules[index];
        const auto& b = right.modules[index];
        if (a.module_name != b.module_name || a.source_sha256 != b.source_sha256 ||
            a.payload != b.payload) return false;
    }
    return true;
}

bool write_file(const std::filesystem::path& path, std::span<const std::byte> bytes) {
    std::ofstream stream(path, std::ios::binary);
    stream.write(reinterpret_cast<const char*>(bytes.data()),
                 static_cast<std::streamsize>(bytes.size()));
    return stream.good();
}

int run_tool(const std::filesystem::path& executable,
             std::vector<std::string> arguments) {
    const auto process = ::fork();
    if (process < 0) return -1;
    if (process == 0) {
        const int null_output = ::open("/dev/null", O_WRONLY);
        if (null_output >= 0) {
            ::dup2(null_output, STDOUT_FILENO);
            ::dup2(null_output, STDERR_FILENO);
            ::close(null_output);
        }
        std::vector<char*> raw;
        raw.reserve(arguments.size() + 2);
        const auto executable_string = executable.string();
        raw.push_back(const_cast<char*>(executable_string.c_str()));
        for (auto& argument : arguments) raw.push_back(argument.data());
        raw.push_back(nullptr);
        ::execv(executable_string.c_str(), raw.data());
        ::_exit(127);
    }
    int status = 0;
    while (::waitpid(process, &status, 0) < 0) {
        if (errno != EINTR) return -1;
    }
    return WIFEXITED(status) ? WEXITSTATUS(status) : -1;
}

PreparationOptions options(PreparationMode mode = PreparationMode::synthetic_test) {
    return {
        .required_preparation_identity = installed_preparation_identity(),
        .max_artifact_bytes = default_prepared_artifact_limit,
        .mode = mode,
    };
}

void deterministic_round_trip() {
    auto input = test::two_module_input();
    const auto first = prepare_modules(input, options());
    const auto second = prepare_modules(input, options());
    CHECK(value(first) != nullptr);
    CHECK(value(second) != nullptr);
    if (!value(first) || !value(second)) return;
    CHECK(value(first)->format_version == prepared_modules_format_version);
    CHECK(value(first)->preparation_identity == installed_preparation_identity());
    CHECK(value(first)->modules.size() == input.modules.size());
    CHECK(equal_artifacts(*value(first), *value(second)));

    TemporaryDirectory temporary;
    CHECK(!temporary.path.empty());
    if (temporary.path.empty()) return;
    const auto first_path = temporary.path / "first";
    const auto second_path = temporary.path / "second";
    CHECK(std::holds_alternative<std::monostate>(
        write_prepared_modules(first_path, *value(first))));
    CHECK(std::holds_alternative<std::monostate>(
        write_prepared_modules(second_path, *value(second))));
    CHECK(read_tree(first_path) == read_tree(second_path));

    const auto read = read_prepared_modules(first_path, input, options());
    CHECK(value(read) != nullptr && equal_artifacts(*value(first), *value(read)));
}

void rootless_companion_round_trip() {
    const auto input = test::rootless_companion_input();
    const auto prepared = prepare_modules(input, options());
    CHECK(value(prepared) != nullptr);
    if (!value(prepared)) return;

    TemporaryDirectory temporary;
    CHECK(std::holds_alternative<std::monostate>(
        write_prepared_modules(temporary.path, *value(prepared))));
    const auto read = read_prepared_modules(temporary.path, input, options());
    CHECK(value(read) != nullptr && equal_artifacts(*value(prepared), *value(read)));
}

void seeds_original_process_entry_without_a_symbol() {
    const auto with_rtld = test::original_process_input();
    const auto prepared_rtld = prepare_modules(with_rtld, options(PreparationMode::automatic));
    CHECK(value(prepared_rtld) != nullptr);
    if (value(prepared_rtld)) {
        for (std::size_t i = 0; i < with_rtld.modules.size(); ++i) {
            const auto payload = preparation::decode_payload(
                value(prepared_rtld)->modules[i].payload);
            CHECK(value(payload) != nullptr);
            if (!value(payload)) continue;
            const bool rooted = std::binary_search(
                value(payload)->roots.begin(), value(payload)->roots.end(), 0);
            CHECK(rooted == (with_rtld.modules[i].name == "rtld"));
        }
    }

    const auto without_rtld = test::original_process_input(false);
    const auto prepared_main = prepare_modules(without_rtld, options(PreparationMode::automatic));
    CHECK(value(prepared_main) != nullptr);
    if (value(prepared_main)) {
        const auto payload = preparation::decode_payload(
            value(prepared_main)->modules.front().payload);
        CHECK(value(payload) != nullptr);
        if (value(payload)) CHECK(std::binary_search(
            value(payload)->roots.begin(), value(payload)->roots.end(), 0));
    }
}

void automatic_read_preserves_explicit_synthetic_orphans() {
    auto input = test::original_process_input();
    input.process_metadata.reset();
    test::ModuleSpec orphan;
    orphan.symbols = {{"orphan_body", 0x100, 4, true}};
    test::put_word(orphan, 0x100, aarch64::encode_ret().word);
    input.modules.push_back({"subsdk1", test::build_module(std::move(orphan))});
    const auto prepared = prepare_modules(input, options(PreparationMode::synthetic_test));
    CHECK(value(prepared) != nullptr);
    if (!value(prepared)) return;
    TemporaryDirectory temporary;
    CHECK(std::holds_alternative<std::monostate>(
        write_prepared_modules(temporary.path, *value(prepared))));
    const auto read = read_prepared_modules(temporary.path, input, options(PreparationMode::automatic));
    CHECK(value(read) != nullptr);
    if (value(read)) {
        CHECK(equal_artifacts(*value(prepared), *value(read)));
        CHECK(value(read)->ignored_modules.empty());
    }
    CHECK(failed(read_prepared_modules(temporary.path, input,
        options(PreparationMode::original_process)), LaunchError::artifact_mismatch));
}

void generated_fixup_limit() {
    static_assert(prepared_fixup_limit % 2 == 0);

    const auto boundary = prepare_modules(
        test::fixup_count_input(prepared_fixup_limit / 2), options());
    CHECK(value(boundary) != nullptr);

    if (value(boundary)) {
        TemporaryDirectory temporary;
        CHECK(std::holds_alternative<std::monostate>(
            write_prepared_modules(temporary.path, *value(boundary))));
        const auto read = read_prepared_modules(temporary.path,
            test::fixup_count_input(prepared_fixup_limit / 2), options());
        CHECK(value(read) != nullptr && equal_artifacts(*value(boundary), *value(read)));
    }

    const auto over_limit = prepare_modules(
        test::fixup_count_input(prepared_fixup_limit / 2 + 1), options());
    CHECK(failed(over_limit, LaunchError::resource_limit));
}

void identity_schema_and_input_mismatch() {
    auto input = test::two_module_input();
    const auto prepared = prepare_modules(input, options());
    CHECK(value(prepared) != nullptr);
    if (!value(prepared)) return;
    TemporaryDirectory temporary;
    const auto valid_path = temporary.path / "valid";
    CHECK(std::holds_alternative<std::monostate>(
        write_prepared_modules(valid_path, *value(prepared))));

    auto wrong_identity = options();
    wrong_identity.required_preparation_identity += "-different";
    CHECK(failed(prepare_modules(input, wrong_identity), LaunchError::artifact_mismatch));
    CHECK(failed(read_prepared_modules(valid_path, input, wrong_identity),
                 LaunchError::artifact_mismatch));

    auto changed_input = input;
    changed_input.modules[0].bytes.back() ^= std::byte{1};
    CHECK(failed(read_prepared_modules(valid_path, changed_input, options()),
                 LaunchError::artifact_mismatch));

    auto wrong_schema = *value(prepared);
    ++wrong_schema.format_version;
    const auto schema_path = temporary.path / "schema";
    const auto schema_write = write_prepared_modules(schema_path, wrong_schema);
    bool schema_rejected = failed(schema_write);
    if (!schema_rejected) {
        schema_rejected = failed(read_prepared_modules(schema_path, input, options()),
                                 LaunchError::artifact_mismatch);
    }
    CHECK(schema_rejected);

    auto self_declared_identity = *value(prepared);
    self_declared_identity.preparation_identity += "-stale";
    const auto identity_path = temporary.path / "identity";
    const auto identity_write = write_prepared_modules(identity_path, self_declared_identity);
    bool identity_rejected = failed(identity_write);
    if (!identity_rejected) {
        identity_rejected = failed(read_prepared_modules(identity_path, input, options()),
                                   LaunchError::artifact_mismatch);
    }
    CHECK(identity_rejected);
}

void corruption_duplicates_and_limits() {
    const auto input = test::two_module_input();
    const auto prepared = prepare_modules(input, options());
    CHECK(value(prepared) != nullptr);
    if (!value(prepared)) return;
    TemporaryDirectory temporary;
    const auto path = temporary.path / "corrupt";
    CHECK(std::holds_alternative<std::monostate>(write_prepared_modules(path, *value(prepared))));
    const auto payload = path / "modules.armigo-prepared";
    CHECK(std::filesystem::is_regular_file(payload));
    if (std::filesystem::is_regular_file(payload)) {
        std::fstream stream(payload, std::ios::in | std::ios::out | std::ios::binary);
        stream.seekg(-1, std::ios::end);
        char byte = 0;
        stream.read(&byte, 1);
        byte ^= 0x5a;
        stream.seekp(-1, std::ios::end);
        stream.write(&byte, 1);
        CHECK(stream.good());
        stream.close();
        CHECK(failed(read_prepared_modules(path, input, options()),
                     LaunchError::artifact_mismatch));
    }

    auto duplicate = input;
    duplicate.modules.push_back(duplicate.modules.front());
    CHECK(failed(prepare_modules(duplicate, options()), LaunchError::invalid_input));

    auto limited = options();
    limited.max_artifact_bytes = 1;
    CHECK(failed(prepare_modules(input, limited), LaunchError::resource_limit));
    const auto limited_read = read_prepared_modules(path, input, limited);
    CHECK(failed(limited_read, LaunchError::resource_limit));
}

void cli_round_trip(const std::filesystem::path& executable, bool rootless = false) {
    CHECK(!executable.empty());
    TemporaryDirectory temporary;
    const auto exefs = temporary.path / "exefs";
    const auto first = temporary.path / "first";
    const auto second = temporary.path / "second";
    std::error_code error;
    std::filesystem::create_directory(exefs, error);
    CHECK(!error);
    auto input = rootless ? test::rootless_companion_input() : test::two_module_input();
    for (const auto& module : input.modules) {
        CHECK(write_file(exefs / module.name, module.bytes));
    }
    const auto prepare_arguments = [&](const std::filesystem::path& output) {
        return std::vector<std::string>{"--synthetic-test", "--exefs", exefs.string(),
                                        "--output", output.string()};
    };
    CHECK(run_tool(executable, prepare_arguments(first)) == 0);
    CHECK(run_tool(executable, prepare_arguments(second)) == 0);
    CHECK(read_tree(first) == read_tree(second));
    CHECK(run_tool(executable, {"--verify", "--synthetic-test", "--exefs", exefs.string(),
                                "--output", first.string()}) == 0);

    auto& main = input.modules.front().bytes;
    main.back() ^= std::byte{1};
    CHECK(write_file(exefs / input.modules.front().name, main));
    CHECK(run_tool(executable, {"--verify", "--synthetic-test", "--exefs", exefs.string(),
                                "--output", first.string()}) == 1);
}

void cli_original_metadata_and_orphan_round_trip(const std::filesystem::path& executable) {
    TemporaryDirectory temporary;
    const auto exefs = temporary.path / "exefs";
    const auto output = temporary.path / "prepared";
    std::filesystem::create_directory(exefs);
    auto input = test::original_process_input();
    for (const auto& module : input.modules)
        CHECK(write_file(exefs / module.name, module.bytes));
    CHECK(write_file(exefs / "main.npdm", *input.process_metadata));
    CHECK(write_file(exefs / "subsdk1", std::vector<std::byte>{std::byte{0xff}}));
    CHECK(run_tool(executable, {"--exefs", exefs.string(), "--output", output.string()}) == 0);
    auto read = read_prepared_modules(output, input, options(PreparationMode::original_process));
    CHECK(value(read) != nullptr);
    if (!value(read)) return;
    const auto payload = preparation::decode_payload(value(read)->modules[1].payload);
    CHECK(value(payload) != nullptr);
    if (value(payload)) CHECK(value(payload)->original_process && preparation::covers(value(payload)->image, 0));
    // The same CLI output also admits later detached metadata: original
    // entry preparation depends on the executable roster, not NPDM presence.
    std::filesystem::remove(exefs / "main.npdm");
    const auto detached_output = temporary.path / "detached-prepared";
    CHECK(run_tool(executable, {"--exefs", exefs.string(), "--output", detached_output.string()}) == 0);
    const auto detached = read_prepared_modules(detached_output, input, options(PreparationMode::original_process));
    CHECK(value(detached) != nullptr);
    if (value(detached)) CHECK(equal_artifacts(*value(read), *value(detached)));
}

void refuses_register_addressed_code_reads() {
    for (const bool page_relative : {false, true}) {
        test::ModuleSpec module;
        module.symbols = {{"_start", 0x100, 16, true}, {"helper", 0x180, 8, true}};
        if (page_relative) {
            test::put_word(module, 0x100, aarch64::encode_adrp(0, 0x100, 0).word);
            test::put_word(module, 0x104, 0x91060000u); // ADD X0, X0, #0x180.
            test::put_word(module, 0x108, 0xb9400000u); // LDR W0, [X0].
            test::put_word(module, 0x10c, aarch64::encode_ret().word);
        } else {
            test::put_word(module, 0x100, aarch64::encode_adr(0, 0x100, 0x180).word);
            test::put_word(module, 0x104, 0xb9400000u); // LDR W0, [X0].
            test::put_word(module, 0x108, aarch64::encode_ret().word);
        }
        test::put_word(module, 0x180, 0x91001652u); // ADD X18, X18, #5.
        test::put_word(module, 0x184, aarch64::encode_ret().word);
        loader::ExeFsDirectoryInput input;
        input.modules.push_back({"main", test::build_module(std::move(module))});
        const auto result = prepare_modules(input, options());
        CHECK(failed(result, LaunchError::unsupported));
        const auto* failure = std::get_if<LaunchFailure>(&result);
        CHECK(failure && failure->detail.find("data reference") != std::string::npos);
    }
}

void refuses_unrelocated_and_cross_module_address_producers() {
    // These authored inputs deliberately collide with the offline address
    // namespace. Numeric coincidence must not become relocation provenance.
    constexpr std::uint64_t first_base = 0x100000000ull;
    constexpr std::uint64_t second_base = 0x200000000ull;
    test::ModuleSpec first;
    first.symbols = {{"_start", 0x100, 16, true}, {"target", 0x180, 4, true}};
    test::put_word(first, 0x100, aarch64::encode_adr(18, 0x100, test::pointer_slot_offset).word);
    test::put_word(first, 0x104, test::load64(18, 18));
    test::put_word(first, 0x108, aarch64::encode_blr(18).word);
    test::put_word(first, 0x10c, aarch64::encode_ret().word);
    test::put_word(first, 0x180, aarch64::encode_ret().word);
    test::put_wide(first, test::pointer_slot_offset, first_base + 0x180);
    loader::ExeFsDirectoryInput raw;
    raw.modules.push_back({"main", test::build_module(first)});
    CHECK(failed(prepare_modules(raw, options()), LaunchError::unsupported));

    test::ModuleSpec second;
    second.symbols = {{"other", 0x100, 4, true}};
    test::put_word(second, 0x100, aarch64::encode_ret().word);
    first.relocations = {{test::pointer_slot_offset, 0, 1027,
        static_cast<std::int64_t>(second_base - first_base + 0x100)}};
    test::put_wide(first, test::pointer_slot_offset, 0);
    loader::ExeFsDirectoryInput crossing;
    crossing.modules = {{"main", test::build_module(first)}, {"sdk", test::build_module(second)}};
    CHECK(failed(prepare_modules(crossing, options()), LaunchError::unsupported));

    // ADRP can span these synthetic namespaces, but independently mapped
    // modules cannot retain that displacement at runtime.
    first.symbols = {{"_start", 0x100, 4, true}};
    test::put_word(first, 0x100, aarch64::encode_ret().word);
    first.relocations.front().addend = 0x100;
    second.symbols.front().size = 16;
    const auto cross_page = aarch64::encode_adrp(9, second_base + 0x100,
                                                first_base + test::pointer_slot_offset);
    CHECK(cross_page.ok());
    test::put_word(second, 0x100, cross_page.word);
    test::put_word(second, 0x104, test::load64(10, 9));
    test::put_word(second, 0x108, aarch64::encode_blr(10).word);
    test::put_word(second, 0x10c, aarch64::encode_ret().word);
    crossing.modules = {{"main", test::build_module(first)}, {"sdk", test::build_module(second)}};
    CHECK(failed(prepare_modules(crossing, options()), LaunchError::unsupported));
}

} // namespace

int main(int argc, char** argv) {
    deterministic_round_trip();
    rootless_companion_round_trip();
    seeds_original_process_entry_without_a_symbol();
    generated_fixup_limit();
    automatic_read_preserves_explicit_synthetic_orphans();
    identity_schema_and_input_mismatch();
    corruption_duplicates_and_limits();
    refuses_register_addressed_code_reads();
    refuses_unrelocated_and_cross_module_address_producers();
    CHECK(argc == 1 || argc == 2);
    if (argc == 2) {
        cli_round_trip(argv[1]);
        cli_round_trip(argv[1], true);
        cli_original_metadata_and_orphan_round_trip(argv[1]);
    }
    if (failures != 0) return 1;
    std::puts("Prepared artifacts are deterministic, bounded, and reject stale or corrupt input.");
    return 0;
}
