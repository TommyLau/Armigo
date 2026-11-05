// ============================================================================
// Armigo Tests - Native Runtime Verification
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

// Container reader tests: keys parsing, AES primitives, signature
// verification and end-to-end synthetic NSP/XCI extraction. All fixtures
// are runtime-generated synthetic bytes; see container_fixture.h.

#include "container_fixture.h"

#include "container/crypto.h"

#include <algorithm>
#include <filesystem>
#include <optional>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <string_view>

namespace {

using namespace container_test;
namespace loader = armigo::loader;

int failures = 0;

#define CHECK(condition)                                                         \
    do {                                                                         \
        if (!(condition)) {                                                      \
            std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #condition);     \
            ++failures;                                                          \
        }                                                                        \
    } while (false)

[[nodiscard]] bool read_file_bytes(const std::filesystem::path& path,
                                   std::vector<std::byte>& out) {
    FILE* file = std::fopen(path.c_str(), "rb");
    if (!file) {
        return false;
    }
    std::byte buffer[4096];
    std::size_t read = 0;
    while ((read = std::fread(buffer, 1, sizeof(buffer), file)) > 0) {
        out.insert(out.end(), buffer, buffer + read);
    }
    const bool ok = std::ferror(file) == 0;
    std::fclose(file);
    return ok;
}

[[nodiscard]] std::filesystem::path make_temp_directory(const std::string& name) {
    std::filesystem::path path = std::filesystem::temp_directory_path() / name;
    std::filesystem::remove_all(path);
    std::filesystem::create_directories(path);
    return path;
}

void test_keys_parsing() {
    const std::string text =
        "# comment\n"
        "; another comment\n"
        "\n"
        "alpha = 0011ff\n"
        "beta =  AABBCCDD \n"
        "header_key = " + std::string(64, 'a') + "\n";
    const std::string_view view(text);
    const auto parsed = loader::parse_container_keys(std::span(
        reinterpret_cast<const std::byte*>(view.data()), view.size()));
    const auto* keys = std::get_if<loader::ContainerKeys>(&parsed);
    CHECK(keys != nullptr);
    if (!keys) {
        return;
    }
    const auto* alpha = keys->find("alpha");
    CHECK(alpha != nullptr && alpha->size() == 3);
    CHECK(alpha && (*alpha)[0] == std::byte{0x00} && (*alpha)[2] == std::byte{0xff});
    const auto* beta = keys->find("beta");
    CHECK(beta != nullptr && beta->size() == 4);
    CHECK(keys->find("missing") == nullptr);
    CHECK(keys->find("header_key") != nullptr && keys->find("header_key")->size() == 32);

    const std::string odd = "k = 0\n";
    auto odd_parsed = loader::parse_container_keys(std::span(
        reinterpret_cast<const std::byte*>(odd.data()), odd.size()));
    CHECK(std::get_if<loader::ContainerKeysFailure>(&odd_parsed) != nullptr);

    const std::string duplicate = "k = 00\nk = 11\n";
    auto duplicate_parsed = loader::parse_container_keys(std::span(
        reinterpret_cast<const std::byte*>(duplicate.data()), duplicate.size()));
    CHECK(std::get_if<loader::ContainerKeysFailure>(&duplicate_parsed) != nullptr);

    const std::string no_equals = "justaname\n";
    auto failed_parsed = loader::parse_container_keys(std::span(
        reinterpret_cast<const std::byte*>(no_equals.data()), no_equals.size()));
    CHECK(std::get_if<loader::ContainerKeysFailure>(&failed_parsed) != nullptr);
}

void test_aes_primitives() {
    // A library XTS decryption must invert the fixture's independent XTS
    // encryption, and vice versa, across several sectors.
    std::array<std::byte, 32> key{};
    for (std::size_t i = 0; i < key.size(); ++i) {
        key[i] = std::byte(static_cast<std::uint8_t>(i * 13 + 7));
    }
    std::vector<std::byte> plain(0x200 * 5); // five whole sectors
    for (std::size_t i = 0; i < plain.size(); ++i) {
        plain[i] = std::byte(static_cast<std::uint8_t>(i * 31 + 3));
    }
    std::vector<std::byte> cipher = plain;
    CHECK(fixture_xts_encrypt(key.data(), cipher, 4, 0x200));
    std::vector<std::byte> decrypted = cipher;
    CHECK(loader::container_detail::aes_xts_decrypt(key, decrypted, 4, 0x200));
    CHECK(decrypted == plain);
    // Known answer from an independent IEEE 1619 XTS-AES-128 implementation
    // (OpenSSL): key bytes 0..31, data unit 0, plaintext bytes 0..63. Data
    // unit 0 has the same tweak in the NCA and IEEE layouts, so this pins the
    // GF(2^128) tweak multiplication independently of the fixture, which
    // shares the library's conventions.
    {
        std::array<std::byte, 32> known_key{};
        for (std::size_t i = 0; i < known_key.size(); ++i) {
            known_key[i] = std::byte(static_cast<std::uint8_t>(i));
        }
        const std::array<std::uint8_t, 64> vector = {
            0x74, 0xa1, 0x09, 0xaa, 0xbf, 0x19, 0x37, 0xc0, 0x22, 0xd1, 0x9d, 0xa4, 0xb9, 0x6c, 0xbc, 0x40,
            0xb8, 0xdd, 0xc9, 0xc0, 0x65, 0x3a, 0x7f, 0xb0, 0xdc, 0x84, 0x25, 0xc7, 0xef, 0x27, 0x6d, 0xea,
            0xc4, 0x5a, 0x55, 0xf7, 0xe6, 0xf1, 0x0d, 0x80, 0x4a, 0x52, 0x94, 0xef, 0xb2, 0x82, 0x63, 0xbe,
            0xfa, 0x7d, 0x53, 0x15, 0x0e, 0xe1, 0x0e, 0x59, 0x3a, 0xa1, 0xb0, 0x80, 0xbe, 0xa9, 0xf6, 0xef,
        };
        std::vector<std::byte> known(vector.size());
        std::memcpy(known.data(), vector.data(), vector.size());
        CHECK(loader::container_detail::aes_xts_decrypt(known_key, known, 0, known.size()));
        bool matches = true;
        for (std::size_t i = 0; i < known.size(); ++i) {
            matches = matches && known[i] == std::byte(static_cast<std::uint8_t>(i));
        }
        CHECK(matches);
    }
    // XTS rejects a trailing partial sector instead of corrupting it.
    std::vector<std::byte> ragged = cipher;
    ragged.push_back(std::byte{0xAA});
    CHECK(!loader::container_detail::aes_xts_decrypt(key, ragged, 4, 0x200));
    std::array<std::byte, 16> aes_key{};
    for (std::size_t i = 0; i < aes_key.size(); ++i) {
        aes_key[i] = std::byte(static_cast<std::uint8_t>(i * 13 + 7));
    }

    // CTR uses the NCA counter convention: an 8-byte section nonce in the
    // high half and the 16-byte block number of the absolute offset in the
    // low half. Encrypt block-by-block with the fixture, then decrypt with
    // the library, including through an unaligned mid-stream window that
    // exercises the reader's aligned-window slicing.
    std::vector<std::byte> stream(0x1234);
    for (std::size_t i = 0; i < stream.size(); ++i) {
        stream[i] = std::byte(static_cast<std::uint8_t>(i * 5 + 11));
    }
    const auto nca_counter = [](std::array<std::uint8_t, 8> nonce, std::uint64_t block_number) {
        std::array<std::uint8_t, 16> value = {};
        for (std::size_t i = 0; i < 8; ++i) {
            value[i] = nonce[i];
        }
        for (std::size_t i = 0; i < 8; ++i) {
            value[15 - i] = static_cast<std::uint8_t>(block_number >> (8 * i));
        }
        return value;
    };
    const std::array<std::uint8_t, 8> nonce = {1, 2, 3, 4, 5, 6, 7, 8};
    std::vector<std::byte> encrypted = stream;
    for (std::size_t block = 0; block * 16 < encrypted.size(); ++block) {
        const std::size_t begin = block * 16;
        const std::size_t end = std::min(begin + 16, encrypted.size());
        std::vector<std::byte> chunk(encrypted.begin() + static_cast<std::ptrdiff_t>(begin),
                                     encrypted.begin() + static_cast<std::ptrdiff_t>(end));
        CHECK(fixture_aes_ctr_apply(aes_key.data(), nca_counter(nonce, block).data(), chunk));
        std::memcpy(encrypted.data() + begin, chunk.data(), chunk.size());
    }

    std::vector<std::byte> whole = encrypted;
    std::array<std::uint8_t, 16> base = nca_counter(nonce, 0);
    CHECK(loader::container_detail::aes_ctr_crypt(aes_key, base, whole));
    CHECK(whole == stream);

    // A window starting mid-block: decrypt from the window's block start
    // with the matching counter, then slice the same bytes.
    constexpr std::size_t kWindowStart = 0x801;
    constexpr std::size_t kWindowEnd = 0x9FF;
    std::array<std::uint8_t, 16> block_counter =
        nca_counter(nonce, kWindowStart / 16);
    const std::size_t aligned_start = kWindowStart & ~std::size_t{0xF};
    const std::size_t aligned_length = ((kWindowEnd - aligned_start) + 0xF) & ~std::size_t{0xF};
    std::vector<std::byte> window(encrypted.begin() + static_cast<std::ptrdiff_t>(aligned_start),
                                  encrypted.begin() +
                                      static_cast<std::ptrdiff_t>(aligned_start + aligned_length));
    CHECK(loader::container_detail::aes_ctr_crypt(aes_key, block_counter, window));
    CHECK(std::equal(window.begin() + (kWindowStart - aligned_start),
                     window.begin() + (kWindowStart - aligned_start) + (kWindowEnd - kWindowStart),
                     stream.begin() + kWindowStart));
}

void test_rsa_pss() {
    const auto key = TestRsaKey::generate();
    CHECK(key.has_value());
    if (!key) {
        return;
    }
    std::vector<std::byte> message(0x200);
    for (std::size_t i = 0; i < message.size(); ++i) {
        message[i] = std::byte(static_cast<std::uint8_t>(i * 17 + 1));
    }
    std::array<std::uint8_t, 256> signature{};
    CHECK(key->sign(message, signature));
    CHECK(loader::container_detail::rsa2048_pss_sha256_verify(key->modulus(), message,
                                                              signature));
    // Any change to the message, the signature or the anchor must fail.
    auto tampered_message = message;
    tampered_message[3] = std::byte{tampered_message[3]} ^ std::byte{0xFF};
    CHECK(!loader::container_detail::rsa2048_pss_sha256_verify(key->modulus(), tampered_message,
                                                               signature));
    auto tampered_signature = signature;
    tampered_signature[10] = static_cast<std::uint8_t>(tampered_signature[10] ^ 0x01);
    CHECK(!loader::container_detail::rsa2048_pss_sha256_verify(key->modulus(), message,
                                                               tampered_signature));
    auto other_key = TestRsaKey::generate();
    CHECK(other_key.has_value());
    if (other_key) {
        CHECK(!loader::container_detail::rsa2048_pss_sha256_verify(other_key->modulus(), message,
                                                                   signature));
    }
}

// Builds a complete verified container and asserts the extraction.
void check_container_extraction(const std::filesystem::path& container_file,
                                const std::filesystem::path& keys_file,
                                const std::filesystem::path& output,
                                const TestRsaKey& sign_key) {
    const TestKeys keys = TestKeys::generate();
    CHECK(write_file(keys_file, std::span(
        reinterpret_cast<const std::byte*>(keys.keys_text.data()), keys.keys_text.size())));
    loader::ContainerExtractionOptions options;
    options.output_directory = output;
    options.anchors = test_anchors(sign_key);
    auto parsed_keys = loader::read_container_keys_file(keys_file);
    const auto* parsed = std::get_if<loader::ContainerKeys>(&parsed_keys);
    CHECK(parsed != nullptr);
    if (!parsed) {
        return;
    }
    options.keys = *parsed;
    const auto result = loader::extract_program_container(container_file, options);
    const auto* extraction = std::get_if<loader::ContainerExtraction>(&result);
    CHECK(extraction != nullptr);
    if (!extraction) {
        const auto* failure = std::get_if<loader::ContainerFailure>(&result);
        if (failure) {
            std::printf("  extraction failed: %s: %s\n", loader::to_string(failure->code),
                        failure->detail.c_str());
        }
        return;
    }
    CHECK(extraction->exefs_directory == output / "exefs");
    CHECK(extraction->romfs_directory == output / "romfs");

    std::vector<std::byte> main_module;
    CHECK(read_file_bytes(extraction->exefs_directory / "main", main_module));
    CHECK(main_module.size() == 0x300);
    CHECK(main_module[5] == std::byte{5 * 7 + 1});
    std::vector<std::byte> npdm;
    CHECK(read_file_bytes(extraction->exefs_directory / "main.npdm", npdm));
    CHECK(!npdm.empty());

    std::vector<std::byte> hello;
    CHECK(read_file_bytes(extraction->romfs_directory / "hello.txt", hello));
    std::vector<std::byte> nested;
    CHECK(read_file_bytes(extraction->romfs_directory / "data" / "nested.bin", nested));
    CHECK(hello.size() == 0x123 && nested.size() == 0x47);
    CHECK(hello[0x122] == std::byte((0x122 * 3 + 5) & 0xFF));
    CHECK(nested[0x10] == std::byte((0x10 * 11 + 9) & 0xFF));
}

void test_nsp_extraction() {
    const auto sign_key = TestRsaKey::generate();
    CHECK(sign_key.has_value());
    if (!sign_key) {
        return;
    }
    const TestKeys keys = TestKeys::generate();
    const auto npdm = build_npdm(*sign_key);
    const BuiltNca nca = build_program_nca(keys, *sign_key, npdm);

    const auto directory = make_temp_directory("armigo-container-nsp-test");
    // A decoy entry exercises the candidate skip path.
    const auto container = build_nsp({{"decoy.bin", {std::byte{1}, std::byte{2}, std::byte{3}}},
                                      {"0100nca.nca", nca.image}});
    const auto container_file = directory / "synthetic.nsp";
    const auto keys_file = directory / "prod.keys";
    CHECK(write_file(container_file, container));
    check_container_extraction(container_file, keys_file, directory / "out", *sign_key);
    std::filesystem::remove_all(directory);
}

void test_xci_extraction() {
    const auto sign_key = TestRsaKey::generate();
    CHECK(sign_key.has_value());
    if (!sign_key) {
        return;
    }
    const TestKeys keys = TestKeys::generate();
    const auto npdm = build_npdm(*sign_key);
    const BuiltNca nca = build_program_nca(keys, *sign_key, npdm);

    const auto directory = make_temp_directory("armigo-container-xci-test");
    const auto container = build_xci("program0.nca", nca.image);
    const auto container_file = directory / "synthetic.xci";
    const auto keys_file = directory / "prod.keys";
    CHECK(write_file(container_file, container));
    check_container_extraction(container_file, keys_file, directory / "out", *sign_key);
    std::filesystem::remove_all(directory);
}

// Extraction must cover every entry the RomFS tables link, whatever the
// tree's shape. The walk's pending-sibling stack grows with fanout across
// levels, so a wide tree drives it far past any depth the directory names
// express; a shape like this silently lost whole subtrees when the walk
// bounded that stack.
void test_romfs_wide_and_deep_tree() {
    const auto sign_key = TestRsaKey::generate();
    CHECK(sign_key.has_value());
    if (!sign_key) {
        return;
    }
    const TestKeys keys = TestKeys::generate();
    const auto npdm = build_npdm(*sign_key);

    RomFsDirSpec root;
    // One hundred sibling directories: while the walk registers them, more
    // than sixty-four stack entries are pending at once.
    RomFsDirSpec wide;
    wide.name = "wide";
    for (int i = 0; i < 100; ++i) {
        char index[8];
        std::snprintf(index, sizeof(index), "s%02d", i);
        RomFsDirSpec child;
        child.name = index;
        child.files.push_back({"a", 16, 1});
        child.files.push_back({"b", 24, 2});
        wide.dirs.push_back(std::move(child));
    }
    root.dirs.push_back(std::move(wide));
    // A chain of seventy nested directories with the files at the bottom.
    RomFsDirSpec levels[70];
    for (int i = 69; i >= 0; --i) {
        char index[8];
        std::snprintf(index, sizeof(index), "n%02d", i);
        levels[i].name = index;
        if (i == 69) {
            levels[i].files.push_back({"leaf.bin", 40, 3});
        } else {
            levels[i].dirs.push_back(std::move(levels[i + 1]));
        }
    }
    RomFsDirSpec deep;
    deep.name = "deep";
    deep.dirs.push_back(std::move(levels[0]));
    root.dirs.push_back(std::move(deep));
    // One directory holding three hundred files: a long sibling chain.
    RomFsDirSpec many;
    many.name = "many";
    for (int i = 0; i < 300; ++i) {
        char index[8];
        std::snprintf(index, sizeof(index), "f%03d", i);
        many.files.push_back({index, 32, static_cast<std::uint8_t>(i)});
    }
    root.dirs.push_back(std::move(many));

    const BuiltRomFsTree tree = build_romfs_tree(root);
    const BuiltNca nca = build_program_nca(keys, *sign_key, npdm, true, false, 0, tree.image);

    const auto directory = make_temp_directory("armigo-container-romfs-tree-test");
    const auto container_file = directory / "synthetic.nsp";
    const auto keys_file = directory / "prod.keys";
    CHECK(write_file(container_file, build_nsp({{"0100nca.nca", nca.image}})));
    CHECK(write_file(keys_file, std::span(
        reinterpret_cast<const std::byte*>(keys.keys_text.data()), keys.keys_text.size())));
    loader::ContainerExtractionOptions options;
    options.output_directory = directory / "out";
    options.anchors = test_anchors(*sign_key);
    auto parsed_keys = loader::read_container_keys_file(keys_file);
    const auto* parsed = std::get_if<loader::ContainerKeys>(&parsed_keys);
    CHECK(parsed != nullptr);
    if (!parsed) {
        return;
    }
    options.keys = *parsed;
    const auto result = loader::extract_program_container(container_file, options);
    const auto* extraction = std::get_if<loader::ContainerExtraction>(&result);
    CHECK(extraction != nullptr);
    if (!extraction) {
        const auto* failure = std::get_if<loader::ContainerFailure>(&result);
        if (failure) {
            std::printf("  romfs tree extraction failed: %s: %s\n",
                        loader::to_string(failure->code), failure->detail.c_str());
        }
        std::filesystem::remove_all(directory);
        return;
    }

    // The extracted path set must equal the spec's, and every file must be
    // byte-identical to its pattern.
    std::vector<std::string> expected;
    for (const auto& [path, spec] : tree.files) {
        expected.push_back(path);
    }
    std::vector<std::string> found;
    for (std::filesystem::recursive_directory_iterator it(extraction->romfs_directory),
         end; it != end; ++it) {
        if (it->is_regular_file()) {
            found.push_back(
                std::filesystem::relative(it->path(), extraction->romfs_directory)
                    .generic_string());
        }
    }
    std::sort(expected.begin(), expected.end());
    std::sort(found.begin(), found.end());
    if (found != expected) {
        for (const std::string& path : expected) {
            if (std::find(found.begin(), found.end(), path) == found.end()) {
                std::printf("  missing from extraction: %s\n", path.c_str());
            }
        }
        for (const std::string& path : found) {
            if (std::find(expected.begin(), expected.end(), path) == expected.end()) {
                std::printf("  unexpected in extraction: %s\n", path.c_str());
            }
        }
    }
    CHECK(found.size() == expected.size());
    CHECK(found == expected);
    for (const auto& [path, spec] : tree.files) {
        std::vector<std::byte> bytes;
        CHECK(read_file_bytes(extraction->romfs_directory / path, bytes));
        CHECK(bytes.size() == spec.size);
        bool matches = bytes.size() == spec.size;
        for (std::uint64_t i = 0; matches && i < spec.size; ++i) {
            matches = bytes[static_cast<std::size_t>(i)] ==
                      std::byte(static_cast<std::uint8_t>(i * 3 + spec.fill));
        }
        CHECK(matches);
    }
    std::filesystem::remove_all(directory);
}

void test_plaintext_header_container() {
    const auto sign_key = TestRsaKey::generate();
    CHECK(sign_key.has_value());
    if (!sign_key) {
        return;
    }
    const TestKeys keys = TestKeys::generate();
    const auto npdm = build_npdm(*sign_key);
    const BuiltNca nca = build_program_nca(keys, *sign_key, npdm, false);

    const auto directory = make_temp_directory("armigo-container-plain-test");
    const auto container_file = directory / "plain.nsp";
    const auto keys_file = directory / "prod.keys";
    CHECK(write_file(container_file, build_nsp({{"a.nca", nca.image}})));
    // Keys without header_key still admit a plaintext-header container.
    const std::string area_only =
        "key_area_key_application_00 = " + to_hex(keys.area_key) + "\n";
    CHECK(write_file(keys_file, std::span(
        reinterpret_cast<const std::byte*>(area_only.data()), area_only.size())));
    loader::ContainerExtractionOptions options;
    options.output_directory = directory / "out";
    options.anchors = test_anchors(*sign_key);
    auto parsed_keys = loader::read_container_keys_file(keys_file);
    const auto* parsed = std::get_if<loader::ContainerKeys>(&parsed_keys);
    CHECK(parsed != nullptr);
    if (parsed) {
        options.keys = *parsed;
        const auto result = loader::extract_program_container(container_file, options);
        if (auto* failure = std::get_if<loader::ContainerFailure>(&result)) {
            std::printf("  plain extraction failed: %s: %s\n",
                        loader::to_string(failure->code), failure->detail.c_str());
        }
        CHECK(std::holds_alternative<loader::ContainerExtraction>(result));
    }
    std::filesystem::remove_all(directory);
}

void test_signature_failures() {
    const auto sign_key = TestRsaKey::generate();
    CHECK(sign_key.has_value());
    if (!sign_key) {
        return;
    }
    const TestKeys keys = TestKeys::generate();
    const auto npdm = build_npdm(*sign_key);
    const auto directory = make_temp_directory("armigo-container-sig-test");
    const auto keys_file = directory / "prod.keys";
    CHECK(write_file(keys_file, std::span(
        reinterpret_cast<const std::byte*>(keys.keys_text.data()), keys.keys_text.size())));

    // Production anchors reject the test-signed container.
    {
        const BuiltNca nca = build_program_nca(keys, *sign_key, npdm);
        const auto container_file = directory / "signed-by-test.nsp";
        CHECK(write_file(container_file, build_nsp({{"a.nca", nca.image}})));
        loader::ContainerExtractionOptions options;
        options.output_directory = directory / "out-a";
        options.keys = std::get<loader::ContainerKeys>(
            loader::read_container_keys_file(keys_file));
        const auto result = loader::extract_program_container(container_file, options);
        const auto* failure = std::get_if<loader::ContainerFailure>(&result);
        CHECK(failure != nullptr && failure->code == loader::ContainerError::signature_invalid);
    }

    // A corrupted signature byte fails admission against the test anchor.
    {
        BuiltNca nca = build_program_nca(keys, *sign_key, npdm);
        nca.image[5] = nca.image[5] ^ std::byte{0xFF}; // signature 1 corruption
        const auto container_file = directory / "corrupt.nsp";
        CHECK(write_file(container_file, build_nsp({{"a.nca", nca.image}})));
        loader::ContainerExtractionOptions options;
        options.output_directory = directory / "out-b";
        options.keys = std::get<loader::ContainerKeys>(
            loader::read_container_keys_file(keys_file));
        options.anchors = test_anchors(*sign_key);
        const auto result = loader::extract_program_container(container_file, options);
        const auto* failure = std::get_if<loader::ContainerFailure>(&result);
        CHECK(failure != nullptr && failure->code == loader::ContainerError::signature_invalid);
    }

    // A container without any program NCA reports the missing program.
    {
        const auto container_file = directory / "empty.nsp";
        CHECK(write_file(container_file,
                         build_nsp({{"only.bin", {std::byte{9}, std::byte{9}}}})));
        loader::ContainerExtractionOptions options;
        options.output_directory = directory / "out-c";
        options.keys = std::get<loader::ContainerKeys>(
            loader::read_container_keys_file(keys_file));
        options.anchors = test_anchors(*sign_key);
        const auto result = loader::extract_program_container(container_file, options);
        const auto* failure = std::get_if<loader::ContainerFailure>(&result);
        CHECK(failure != nullptr &&
              failure->code == loader::ContainerError::program_nca_missing);
    }
    std::filesystem::remove_all(directory);
}

void test_missing_keys() {
    const auto sign_key = TestRsaKey::generate();
    CHECK(sign_key.has_value());
    if (!sign_key) {
        return;
    }
    const TestKeys keys = TestKeys::generate();
    const auto npdm = build_npdm(*sign_key);
    const BuiltNca nca = build_program_nca(keys, *sign_key, npdm);
    const auto directory = make_temp_directory("armigo-container-keys-test");
    const auto container_file = directory / "a.nsp";
    CHECK(write_file(container_file, build_nsp({{"a.nca", nca.image}})));

    // Without the key-area key the failure names the key, never a value.
    {
        const auto keys_file = directory / "no-area.keys";
        const std::string text = "header_key = " + to_hex(keys.header_key) + "\n";
        CHECK(write_file(keys_file, std::span(
            reinterpret_cast<const std::byte*>(text.data()), text.size())));
        loader::ContainerExtractionOptions options;
        options.output_directory = directory / "out-a";
        options.keys = std::get<loader::ContainerKeys>(
            loader::read_container_keys_file(keys_file));
        options.anchors = test_anchors(*sign_key);
        const auto result = loader::extract_program_container(container_file, options);
        const auto* failure = std::get_if<loader::ContainerFailure>(&result);
        CHECK(failure != nullptr && failure->code == loader::ContainerError::key_missing);
        CHECK(failure != nullptr &&
              failure->detail.find("key_area_key_application_00") != std::string::npos);
        CHECK(failure != nullptr && failure->detail.find(to_hex(keys.area_key)) ==
                                          std::string::npos);
    }

    // A rights-id NCA without a ticket or title key is a missing key naming
    // the rights id, after signature admission.
    {
        const BuiltNca titled = build_program_nca(keys, *sign_key, npdm, true, true);
        const auto keys_file = directory / "full.keys";
        CHECK(write_file(keys_file, std::span(
            reinterpret_cast<const std::byte*>(keys.keys_text.data()),
            keys.keys_text.size())));
        const auto titled_file = directory / "titled.nsp";
        CHECK(write_file(titled_file, build_nsp({{"a.nca", titled.image}})));
        loader::ContainerExtractionOptions options;
        options.output_directory = directory / "out-b";
        options.keys = std::get<loader::ContainerKeys>(
            loader::read_container_keys_file(keys_file));
        options.anchors = test_anchors(*sign_key);
        const auto result = loader::extract_program_container(titled_file, options);
        const auto* failure = std::get_if<loader::ContainerFailure>(&result);
        CHECK(failure != nullptr && failure->code == loader::ContainerError::key_missing);
        CHECK(failure != nullptr &&
              failure->detail.find("0102030405060708090a0b0c0d0e0f10") != std::string::npos);
    }

    // An unknown container format is rejected before any key is consulted.
    {
        const auto junk_file = directory / "junk.bin";
        const std::vector<std::byte> junk = {std::byte{0}, std::byte{1}, std::byte{2}};
        CHECK(write_file(junk_file, junk));
        loader::ContainerExtractionOptions options;
        options.output_directory = directory / "out-c";
        const auto result = loader::extract_program_container(junk_file, options);
        const auto* failure = std::get_if<loader::ContainerFailure>(&result);
        CHECK(failure != nullptr &&
              failure->code == loader::ContainerError::unknown_container_format);
    }
    std::filesystem::remove_all(directory);
}

// Single-byte corruption at every level of the signed header chain must
// produce a structured content-hash failure naming the section and level,
// and never a partial extraction.
void test_hash_chain_corruption() {
    const auto sign_key = TestRsaKey::generate();
    CHECK(sign_key.has_value());
    if (!sign_key) {
        return;
    }
    const TestKeys keys = TestKeys::generate();
    const auto npdm = build_npdm(*sign_key);
    const BuiltNca nca = build_program_nca(keys, *sign_key, npdm);
    const auto directory = make_temp_directory("armigo-container-hash-test");
    const auto keys_file = directory / "prod.keys";
    CHECK(write_file(keys_file, std::span(
        reinterpret_cast<const std::byte*>(keys.keys_text.data()), keys.keys_text.size())));

    struct CorruptionCase {
        const char* name;
        std::uint64_t offset;      // absolute container offset to flip
        const char* expected_name; // substring the failure detail must contain
    };
    const CorruptionCase cases[] = {
        // FS headers live in the encrypted header; a flipped cipher byte
        // scrambles the decrypted header its hash table entry covers.
        {"ExeFS FS header", 0x402, "FS header 0"},
        {"RomFS FS header", 0x602, "FS header 1"},
        {"PFS0 hash table", nca.exefs_table_offset + 3, "master hash"},
        {"PFS0 data", nca.exefs_data_offset + 0x30, "entry"},
        {"IVFC level 1", nca.ivfc_level_offsets[0] + 2, "level 1"},
        {"IVFC level 2", nca.ivfc_level_offsets[1] + 2, "level 2"},
        {"IVFC level 3", nca.ivfc_level_offsets[2] + 2, "level 3"},
        {"IVFC level 4", nca.ivfc_level_offsets[3] + 2, "level 4"},
        {"IVFC level 5", nca.ivfc_level_offsets[4] + 2, "level 5"},
        {"RomFS data", nca.ivfc_level_offsets[5] + 0x10, "level 6"},
    };
    for (const CorruptionCase& item : cases) {
        std::vector<std::byte> corrupted = nca.image;
        corrupted[static_cast<std::size_t>(item.offset)] =
            corrupted[static_cast<std::size_t>(item.offset)] ^ std::byte{0xFF};
        const auto container_file = directory / (std::string(item.name) + ".nsp");
        CHECK(write_file(container_file, build_nsp({{"a.nca", corrupted}})));
        loader::ContainerExtractionOptions options;
        options.output_directory = directory / "out";
        auto parsed_keys = loader::read_container_keys_file(keys_file);
        const auto* parsed = std::get_if<loader::ContainerKeys>(&parsed_keys);
        CHECK(parsed != nullptr);
        if (!parsed) {
            continue;
        }
        options.keys = *parsed;
        options.anchors = test_anchors(*sign_key);
        const auto result = loader::extract_program_container(container_file, options);
        const auto* failure = std::get_if<loader::ContainerFailure>(&result);
        CHECK(failure != nullptr);
        if (!failure) {
            std::printf("  %s: extraction unexpectedly succeeded\n", item.name);
            continue;
        }
        if (failure->code != loader::ContainerError::content_hash_invalid) {
            std::printf("  %s: wrong failure %s: %s\n", item.name,
                        loader::to_string(failure->code), failure->detail.c_str());
        }
        CHECK(failure->code == loader::ContainerError::content_hash_invalid);
        if (failure->detail.find(item.expected_name) == std::string::npos) {
            std::printf("  %s: detail names the wrong level: %s\n", item.name,
                        failure->detail.c_str());
        }
        CHECK(failure->detail.find(item.expected_name) != std::string::npos);
        CHECK(!std::filesystem::exists(options.output_directory));
    }
    std::filesystem::remove_all(directory);
}

// Base + update extraction: titlekey decryption through a common ticket or
// the title-keys table, BKTR patch layering over the base RomFS, and the
// structured failure paths. All fixtures are synthetic; see
// build_update_program_nca in container_fixture.h.
void test_update_extraction() {
    const auto sign_key = TestRsaKey::generate();
    CHECK(sign_key.has_value());
    if (!sign_key) {
        return;
    }
    const TestKeys keys = TestKeys::generate();
    const auto npdm = build_npdm(*sign_key);
    const std::uint64_t title_id = 0x01000000000a2c4eULL; // synthetic
    const BuiltNca base = build_program_nca(keys, *sign_key, npdm, true, false, title_id);
    const BuiltUpdateNca update = build_update_program_nca(keys, *sign_key, npdm, title_id);
    CHECK(!update.image.empty());
    if (update.image.empty()) {
        return;
    }

    const auto directory = make_temp_directory("armigo-container-update-test");
    const auto keys_file = directory / "prod.keys";
    const std::string keys_text =
        keys.keys_text + "titlekek_00 = " + to_hex(keys.titlekek) + "\n";
    CHECK(write_file(keys_file, std::span(
        reinterpret_cast<const std::byte*>(keys_text.data()), keys_text.size())));

    const auto base_file = directory / "base.nsp";
    CHECK(write_file(base_file, build_nsp({{"base.nca", base.image}})));

    // The expected patched RomFS bytes for assertions below.
    const std::vector<std::byte> hello_patched = [&] {
        std::vector<std::byte> value(0x123);
        for (std::size_t i = 0; i < value.size(); ++i) {
            value[i] = std::byte(static_cast<std::uint8_t>(i * 5 + 0x71));
        }
        return value;
    }();
    const std::vector<std::byte> hello_base = [&] {
        std::vector<std::byte> value(0x123);
        for (std::size_t i = 0; i < value.size(); ++i) {
            value[i] = std::byte(static_cast<std::uint8_t>(i * 3 + 5));
        }
        return value;
    }();
    const std::vector<std::byte> module_update = [&] {
        std::vector<std::byte> value(0x300);
        for (std::size_t i = 0; i < value.size(); ++i) {
            value[i] = std::byte(static_cast<std::uint8_t>(i * 7 + 2));
        }
        return value;
    }();

    auto run = [&](const std::filesystem::path& update_path,
                   const loader::ContainerKeys* title_keys,
                   const std::filesystem::path& output)
        -> std::optional<loader::ContainerExtraction> {
        loader::ContainerExtractionOptions options;
        options.output_directory = output;
        options.anchors = test_anchors(*sign_key);
        options.keys = std::get<loader::ContainerKeys>(
            loader::read_container_keys_file(keys_file));
        if (title_keys != nullptr) {
            options.title_keys = *title_keys;
        }
        options.update_container_file = update_path;
        const auto result = loader::extract_program_container(base_file, options);
        auto* extraction = std::get_if<loader::ContainerExtraction>(&result);
        if (!extraction) {
            const auto* failure = std::get_if<loader::ContainerFailure>(&result);
            if (failure) {
                std::printf("  update extraction failed: %s: %s\n",
                            loader::to_string(failure->code), failure->detail.c_str());
            }
        }
        if (extraction) {
            CHECK(extraction->update_applied);
            CHECK(extraction->title_id == title_id);
        }
        if (extraction) {
            return std::optional<loader::ContainerExtraction>(*extraction);
        }
        return std::nullopt;
    };

    // 1. Ticket path: the update NSP carries a matching .tik; the patched
    //    hello.txt replaces the base file and the ExeFS comes from the
    //    update module.
    {
        const auto update_file = directory / "update.nsp";
        CHECK(write_file(update_file, build_nsp(
            {{"update.nca", update.image}, {"update_01000000000a2c4e.tik", update.ticket}})));
        const auto extraction = run(update_file, nullptr, directory / "out-ticket");
        CHECK(extraction.has_value());
        if (extraction) {
            std::vector<std::byte> hello;
            std::vector<std::byte> nested;
            std::vector<std::byte> main_module;
            CHECK(read_file_bytes(extraction->romfs_directory / "hello.txt", hello));
            CHECK(read_file_bytes(extraction->romfs_directory / "data" / "nested.bin",
                                  nested));
            CHECK(read_file_bytes(extraction->exefs_directory / "main", main_module));
            CHECK(hello == hello_patched);
            CHECK(hello != hello_base);
            CHECK(nested.size() == 0x47);
            CHECK(main_module == module_update);
        }
    }

    // A retail subsection tree may include the relocation tree in its total.
    // Both representations extract the same patched RomFS, while every other
    // total remains malformed.
    {
        constexpr std::uint64_t kRelocationTreeSize = 0x8000;
        const BuiltUpdateNca alternate = build_update_program_nca(
            keys, *sign_key, npdm, title_id, kRelocationTreeSize);
        CHECK(!alternate.image.empty());
        if (!alternate.image.empty()) {
            const auto update_file = directory / "update-subsection-tree-total.nsp";
            CHECK(write_file(update_file, build_nsp(
                {{"update.nca", alternate.image}, {"update.tik", alternate.ticket}})));
            const auto extraction = run(update_file, nullptr,
                                        directory / "out-subsection-tree-total");
            CHECK(extraction.has_value());
            if (extraction) {
                std::vector<std::byte> hello;
                CHECK(read_file_bytes(extraction->romfs_directory / "hello.txt", hello));
                CHECK(hello == hello_patched);
            }
        }

        const BuiltUpdateNca invalid = build_update_program_nca(
            keys, *sign_key, npdm, title_id, kRelocationTreeSize + 1);
        CHECK(!invalid.image.empty());
        if (!invalid.image.empty()) {
            const auto update_file = directory / "update-invalid-subsection-total.nsp";
            CHECK(write_file(update_file, build_nsp(
                {{"update.nca", invalid.image}, {"update.tik", invalid.ticket}})));
            const auto extraction = run(update_file, nullptr,
                                        directory / "out-invalid-subsection-total");
            CHECK(!extraction.has_value());
        }
    }

    // 2. Title-keys table path: no ticket in the update NSP; the rights-id
    //    key comes from the title-keys file.
    {
        const auto update_file = directory / "update-no-ticket.nsp";
        CHECK(write_file(update_file, build_nsp({{"update.nca", update.image}})));
        const std::string title_keys_text =
            update.rights_id_hex + " = " + to_hex(update.title_key) + "\n";
        const auto title_keys_file = directory / "title.keys";
        CHECK(write_file(title_keys_file, std::span(
            reinterpret_cast<const std::byte*>(title_keys_text.data()),
            title_keys_text.size())));
        const auto parsed = loader::read_container_keys_file(title_keys_file);
        const auto* title_keys = std::get_if<loader::ContainerKeys>(&parsed);
        CHECK(title_keys != nullptr);
        if (title_keys) {
            const auto extraction = run(update_file, title_keys, directory / "out-table");
            CHECK(extraction.has_value());
            if (extraction) {
                std::vector<std::byte> hello;
                CHECK(read_file_bytes(extraction->romfs_directory / "hello.txt", hello));
                CHECK(hello == hello_patched);
            }
        }
    }

    // 3. A wrong title key decrypts to noise and fails the content hash
    //    chain; no partial extraction is written.
    {
        const auto update_file = directory / "update-wrong.nsp";
        CHECK(write_file(update_file, build_nsp({{"update.nca", update.image}})));
        std::array<std::byte, 16> wrong = update.title_key;
        wrong[0] = wrong[0] ^ std::byte{0xFF};
        const std::string title_keys_text =
            update.rights_id_hex + " = " + to_hex(wrong) + "\n";
        const auto title_keys_file = directory / "wrong.keys";
        CHECK(write_file(title_keys_file, std::span(
            reinterpret_cast<const std::byte*>(title_keys_text.data()),
            title_keys_text.size())));
        const auto parsed = loader::read_container_keys_file(title_keys_file);
        const auto* title_keys = std::get_if<loader::ContainerKeys>(&parsed);
        CHECK(title_keys != nullptr);
        if (title_keys) {
            loader::ContainerExtractionOptions options;
            options.output_directory = directory / "out-wrong";
            options.anchors = test_anchors(*sign_key);
            options.keys = std::get<loader::ContainerKeys>(
                loader::read_container_keys_file(keys_file));
            options.title_keys = *title_keys;
            options.update_container_file = update_file;
            const auto result = loader::extract_program_container(base_file, options);
            const auto* failure = std::get_if<loader::ContainerFailure>(&result);
            CHECK(failure != nullptr &&
                  failure->code == loader::ContainerError::content_hash_invalid);
            CHECK(!std::filesystem::exists(options.output_directory));
        }
    }

    // 4. A corrupted patch byte fails the update section's IVFC chain over
    //    the patched image, for both the patched tables and the patched
    //    image data.
    for (const std::uint64_t probe : {update.patch_table_offset, update.patch_data_offset}) {
        std::vector<std::byte> corrupted = update.image;
        corrupted[static_cast<std::size_t>(probe)] ^= std::byte{0xFF};
        const auto update_file = directory / "update-corrupt.nsp";
        CHECK(write_file(update_file, build_nsp(
            {{"update.nca", corrupted}, {"update.tik", update.ticket}})));
        loader::ContainerExtractionOptions options;
        options.output_directory = directory / "out-corrupt";
        options.anchors = test_anchors(*sign_key);
        options.keys = std::get<loader::ContainerKeys>(
            loader::read_container_keys_file(keys_file));
        options.update_container_file = update_file;
        const auto result = loader::extract_program_container(base_file, options);
        const auto* failure = std::get_if<loader::ContainerFailure>(&result);
        CHECK(failure != nullptr);
        if (failure && failure->code != loader::ContainerError::content_hash_invalid) {
            std::printf("  patch corruption: wrong failure %s: %s\n",
                        loader::to_string(failure->code), failure->detail.c_str());
        }
        CHECK(failure != nullptr &&
              failure->code == loader::ContainerError::content_hash_invalid);
    }

    // 5. Without the titlekek the ticket cannot be decrypted: the failure
    //    names the key, never a value.
    {
        const auto update_file = directory / "update.nsp";
        CHECK(write_file(update_file, build_nsp(
            {{"update.nca", update.image}, {"update.tik", update.ticket}})));
        loader::ContainerExtractionOptions options;
        options.output_directory = directory / "out-kek";
        options.anchors = test_anchors(*sign_key);
        options.keys = std::get<loader::ContainerKeys>(
            loader::read_container_keys_file(keys_file));
        // Overwrite the parsed keys with the text lacking the titlekek.
        options.keys = std::get<loader::ContainerKeys>(
            loader::parse_container_keys(std::span(
                reinterpret_cast<const std::byte*>(keys.keys_text.data()),
                keys.keys_text.size())));
        options.update_container_file = update_file;
        const auto result = loader::extract_program_container(base_file, options);
        const auto* failure = std::get_if<loader::ContainerFailure>(&result);
        CHECK(failure != nullptr && failure->code == loader::ContainerError::key_missing);
        CHECK(failure != nullptr &&
              failure->detail.find("titlekek_00") != std::string::npos);
        CHECK(failure != nullptr &&
              failure->detail.find(to_hex(keys.titlekek)) == std::string::npos);
    }

    // 6. An update program NCA with a different title id never matches.
    {
        const BuiltUpdateNca other = build_update_program_nca(keys, *sign_key, npdm,
                                                              title_id + 1);
        CHECK(!other.image.empty());
        if (!other.image.empty()) {
            const auto update_file = directory / "update-other.nsp";
            CHECK(write_file(update_file, build_nsp(
                {{"update.nca", other.image}, {"update.tik", other.ticket}})));
            loader::ContainerExtractionOptions options;
            options.output_directory = directory / "out-other";
            options.anchors = test_anchors(*sign_key);
            options.keys = std::get<loader::ContainerKeys>(
                loader::read_container_keys_file(keys_file));
            options.update_container_file = update_file;
            const auto result = loader::extract_program_container(base_file, options);
            const auto* failure = std::get_if<loader::ContainerFailure>(&result);
            CHECK(failure != nullptr &&
                  failure->code == loader::ContainerError::program_nca_missing);
        }
    }

    // A BKTR RomFS data offset addresses the patched image, so it may lie
    // past the stored section. The relocation tree still bounds it.
    {
        const BuiltUpdateNca beyond = build_update_program_nca(
            keys, *sign_key, npdm, title_id, 0, 0x100000000ULL);
        CHECK(!beyond.image.empty());
        if (!beyond.image.empty()) {
            const auto update_file = directory / "update-virtual-offset.nsp";
            CHECK(write_file(update_file, build_nsp(
                {{"update.nca", beyond.image}, {"update.tik", beyond.ticket}})));
            loader::ContainerExtractionOptions options;
            options.output_directory = directory / "out-virtual-offset";
            options.anchors = test_anchors(*sign_key);
            options.keys = std::get<loader::ContainerKeys>(
                loader::read_container_keys_file(keys_file));
            options.update_container_file = update_file;
            const auto result = loader::extract_program_container(base_file, options);
            const auto* failure = std::get_if<loader::ContainerFailure>(&result);
            CHECK(failure != nullptr);
            CHECK(failure != nullptr &&
                  failure->detail.find("RomFS data offset leaves its section") ==
                      std::string::npos);
            CHECK(failure != nullptr &&
                  failure->detail.find("leaves the patched image") != std::string::npos);
        }
    }

    // The same patched bytes extract when every IVFC offset sits past the
    // stored section and inside the relocation tree's virtual image.
    {
        constexpr std::uint64_t kVirtualBase = 0x100000;
        const BuiltUpdateNca shifted = build_update_program_nca(
            keys, *sign_key, npdm, title_id, 0, 0, kVirtualBase);
        CHECK(!shifted.image.empty());
        CHECK(shifted.stored_section_size > 0);
        CHECK(shifted.stored_section_size <= shifted.romfs_offset);
        CHECK(shifted.romfs_offset < shifted.patch_virtual_size);
        if (!shifted.image.empty() &&
            shifted.stored_section_size <= shifted.romfs_offset &&
            shifted.romfs_offset < shifted.patch_virtual_size) {
            const auto update_file = directory / "update-shifted-virtual.nsp";
            CHECK(write_file(update_file, build_nsp(
                {{"update.nca", shifted.image}, {"update.tik", shifted.ticket}})));
            const auto extraction = run(update_file, nullptr,
                                        directory / "out-shifted-virtual");
            CHECK(extraction.has_value());
            if (extraction) {
                std::vector<std::byte> hello;
                std::vector<std::byte> nested;
                CHECK(read_file_bytes(extraction->romfs_directory / "hello.txt", hello));
                CHECK(read_file_bytes(extraction->romfs_directory / "data" / "nested.bin",
                                      nested));
                CHECK(hello == hello_patched);
                CHECK(nested.size() == 0x47);
            }
        }
    }

    // Declared hash counts that wrap, a parent table shorter than those
    // digests, and a relocation whose physical range wraps. Each image is
    // rejected before any patched file is produced.
    auto expect_rejection = [&](const char* name, container_test::UpdateFixtureDefect defect,
                                std::string_view expected) {
        const BuiltUpdateNca image = build_update_program_nca(
            keys, *sign_key, npdm, title_id, 0, 0, 0, defect);
        CHECK(!image.image.empty());
        if (image.image.empty()) {
            std::printf("  %s fixture was not built\n", name);
            return;
        }
        const auto update_file = directory / (std::string(name) + ".nsp");
        CHECK(write_file(update_file, build_nsp(
            {{"update.nca", image.image}, {"update.tik", image.ticket}})));
        loader::ContainerExtractionOptions options;
        options.output_directory = directory / (std::string(name) + "-out");
        options.anchors = test_anchors(*sign_key);
        options.keys = std::get<loader::ContainerKeys>(
            loader::read_container_keys_file(keys_file));
        options.update_container_file = update_file;
        const auto result = loader::extract_program_container(base_file, options);
        const auto* failure = std::get_if<loader::ContainerFailure>(&result);
        CHECK(failure != nullptr);
        if (failure == nullptr) {
            std::printf("  %s extracted\n", name);
            return;
        }
        CHECK(failure->detail.find(expected) != std::string::npos);
        if (failure->detail.find(expected) == std::string::npos) {
            std::printf("  %s detail: %s\n", name, failure->detail.c_str());
        }
    };
    expect_rejection("digest-overflow", container_test::UpdateFixtureDefect::digest_overflow,
                     "The section's hash table or hashed data leaves its section.");
    expect_rejection("short-parent", container_test::UpdateFixtureDefect::short_parent,
                     "The section's hash table or hashed data leaves its section.");
    expect_rejection("relocation-wrap", container_test::UpdateFixtureDefect::relocation_wrap,
                     "Reading a hashed data block failed.");
    std::filesystem::remove_all(directory);
}

} // namespace

int main() {
    test_keys_parsing();
    test_aes_primitives();
    test_rsa_pss();
    test_nsp_extraction();
    test_xci_extraction();
    test_romfs_wide_and_deep_tree();
    test_plaintext_header_container();
    test_signature_failures();
    test_missing_keys();
    test_hash_chain_corruption();
    test_update_extraction();
    if (failures == 0) {
        std::printf("container tests passed\n");
        return 0;
    }
    std::printf("%d container test check(s) failed\n", failures);
    return 1;
}
