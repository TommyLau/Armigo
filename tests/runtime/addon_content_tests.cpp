// ============================================================================
// Armigo Tests - Native Runtime Verification
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include <armigo/runtime/addon_content.h>

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <unistd.h>
#include <variant>
#include <vector>

namespace {

namespace fs = std::filesystem;
unsigned failures = 0;
#define CHECK(c) do { if (!(c)) { std::fprintf(stderr, "Failed line %d: %s\n", __LINE__, #c); ++failures; } } while (false)

using armigo::runtime::LaunchFailure;
using armigo::runtime::open_addon_content;
using Catalog = std::shared_ptr<const armigo::services::AddOnContentCatalog>;

constexpr std::uint64_t application = 0x0100000000010000ull;  // synthetic

fs::path make_item(const fs::path& root, const std::string& name) {
    const auto directory = root / name;
    fs::create_directories(directory / "Data");
    std::ofstream(directory / "Data" / "file.bin") << "add-on";
    return directory;
}

bool failed_with(const auto& result, const char* text) {
    const auto* failure = std::get_if<LaunchFailure>(&result);
    return failure && failure->operation == "open-addon-content" &&
           failure->detail.find(text) != std::string::npos;
}

void test_catalog_from_directories(const fs::path& root) {
    // No directories: the base id alone.
    const std::vector<fs::path> none;
    auto empty = open_addon_content(application, none);
    CHECK(std::holds_alternative<Catalog>(empty));
    if (const auto* catalog = std::get_if<Catalog>(&empty)) {
        CHECK((*catalog)->base_id == application + 0x1000);
        CHECK((*catalog)->items.empty());
    }
    // Items sorted by data id, names in either case.
    const std::vector<fs::path> items{make_item(root, "010000000001100C"), make_item(root, "010000000001100a"),
                                      make_item(root, "0100000000011001")};
    auto opened = open_addon_content(application, items);
    CHECK(std::holds_alternative<Catalog>(opened));
    if (const auto* catalog = std::get_if<Catalog>(&opened)) {
        CHECK((*catalog)->items.size() == 3);
        if ((*catalog)->items.size() == 3) {
            CHECK((*catalog)->items[0].data_id == application + 0x1001);
            CHECK((*catalog)->items[1].data_id == application + 0x100a);
            CHECK((*catalog)->items[2].data_id == application + 0x100c);
            CHECK((*catalog)->items[0].storage && (*catalog)->items[0].storage->size() > 0);
        }
        CHECK((*catalog)->find(application + 0x100c) != nullptr);
        CHECK((*catalog)->find(application + 0x1002) == nullptr);
    }
    // A trailing separator names the same directory.
    const std::vector<fs::path> trailing{fs::path(items[2].string() + "/")};
    CHECK(std::holds_alternative<Catalog>(open_addon_content(application, trailing)));
    const auto bundled = make_item(root / "0100000000011002", "romfs");
    const std::vector<fs::path> bundle_items{bundled};
    const auto bundle = open_addon_content(application, bundle_items);
    CHECK(std::holds_alternative<Catalog>(bundle));
    if (const auto* catalog = std::get_if<Catalog>(&bundle))
        CHECK((*catalog)->find(application + 0x1002) && (*catalog)->items[0].storage->size() > 0);
    const std::vector<fs::path> bundle_trailing{fs::path(bundled.string() + "/")};
    CHECK(std::holds_alternative<Catalog>(open_addon_content(application, bundle_trailing)));
    // Rejected: a name that is not a data id, another application's add-on, the
    // base id itself, an id given twice, and any id for an application whose base id would wrap.
    const std::vector<fs::path> misnamed{make_item(root, "dlc-one")};
    CHECK(failed_with(open_addon_content(application, misnamed), "16 hexadecimal digits"));
    const std::vector<fs::path> foreign{make_item(root, "0100000000022001")};
    CHECK(failed_with(open_addon_content(application, foreign), "does not belong"));
    const std::vector<fs::path> base{make_item(root, "0100000000011000")};
    CHECK(failed_with(open_addon_content(application, base), "does not belong"));
    const std::vector<fs::path> twice{items[0], items[0]};
    CHECK(failed_with(open_addon_content(application, twice), "more than once"));
    const std::vector<fs::path> wrapped{make_item(root, "0000000000000FF1")};
    CHECK(failed_with(open_addon_content(~std::uint64_t{0} - 0xf, wrapped), "does not belong"));
}

}  // namespace

int main() {
    const auto root = fs::temp_directory_path() / ("armigo-addon-tests-" + std::to_string(::getpid()));
    fs::remove_all(root);
    test_catalog_from_directories(root);
    fs::remove_all(root);
    if (failures) return 1;
    std::puts("add-on content input checks passed");
    return 0;
}
