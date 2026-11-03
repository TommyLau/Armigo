#!/usr/bin/env python3
# ============================================================================
# Armigo Tests - Native Runtime Verification
# Copyright (c) 2026 Tommy Lau.
# SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
#
# This file is licensed under GPL-3.0-only or, by separate written
# agreement, an Armigo proprietary license.
# See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
# ============================================================================

"""Verify private-path detection and rejection of signing material."""

from pathlib import Path, PurePosixPath
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from contextlib import contextmanager

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
from check_repository import (
    PERSONAL_PATH,
    cmake_game_name_failures,
    game_name_token,
    signing_content_failures,
)

_REPOSITORY = Path(__file__).resolve().parents[2]
_CHECKER = _REPOSITORY / "tools" / "check_repository.py"
_SIGNING_FAILURE = re.compile(
    r":\d+: (?:signing-extension|signing-filename|embedded-certificate"
    r"|embedded-private-key|signing-setting-team|signing-setting-identity"
    r"|signing-setting-profile|signing-setting-style|codesign-identity"
    r"|key-file-assignment|key-file-master-key|key-file-titlekek)$"
)


def _git_env() -> dict[str, str]:
    env = os.environ.copy()
    for name in ("GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE"):
        env.pop(name, None)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return env


def _hex_digits(count: int) -> str:
    return ("01234567" * ((count // 8) + 1))[:count]


def _certificate_mark() -> str:
    return "-----" + "BEGIN CERTIFICATE" + "-----"


def _private_key_mark(label: str = "") -> str:
    return "-----" + "BEGIN " + label + "PRIVATE KEY" + "-----"


def _setting(*parts: str) -> str:
    return "".join(parts)


def _key_line(name: str, value: str, prefix: str = "", suffix: str = "") -> str:
    return f"{prefix}{name} = {value}{suffix}"


def _codesign(*arguments: str) -> str:
    return " ".join(("codesign", *arguments))


def _python_source() -> str:
    lines = Path(__file__).read_text(encoding="utf-8").splitlines()
    closings = [
        index for index, line in enumerate(lines)
        if line.startswith("# " + ("=" * 76))
    ]
    end = closings[1]
    return "\n".join(lines[:end + 1] + ["", "print('ok')", ""])


@contextmanager
def _indexed_repository(files: dict[str, str]):
    root = Path(tempfile.mkdtemp(prefix="armigo-repo-guard-"))
    env = _git_env()
    try:
        init = subprocess.run(
            ["git", "init", "-q", "-b", "main"],
            cwd=root, env=env, check=False, capture_output=True, text=True,
        )
        if init.returncode != 0:
            raise AssertionError(init.stderr)
        for name, content in files.items():
            relative = PurePosixPath(name)
            if relative.is_absolute() or ".." in relative.parts:
                raise AssertionError(name)
            path = root.joinpath(*relative.parts)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8", newline="\n")
        added = subprocess.run(
            ["git", "add", "-f", "--", *files],
            cwd=root, env=env, check=False, capture_output=True, text=True,
        )
        if added.returncode != 0:
            raise AssertionError(added.stderr)
        yield root
    finally:
        shutil.rmtree(root)


def _run_check(root: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(_CHECKER)],
        cwd=root, env=_git_env(), check=False, capture_output=True, text=True,
    )


class PrivacyPolicyTests(unittest.TestCase):
    def test_personal_machine_locations(self):
        candidates = [
            "/" + "Users" + "/sample/private-input",
            "/" + "home" + "/sample/project",
            "/" + "Volumes" + "/sample/project",
            "/" + "private" + "/" + "tmp" + "/sample",
            "/" + "var" + "/" + "tmp" + "/sample",
            "/" + "tmp" + "/sample",
            "C:" + "\\Users\\" + "sample\\project",
            "~" + "/private-input",
        ]
        for candidate in candidates:
            with self.subTest(candidate=candidate):
                self.assertIsNotNone(PERSONAL_PATH.search(candidate))

    def test_local_uri_and_neighbor_assumptions(self):
        self.assertIsNotNone(PERSONAL_PATH.search("file:" + "//" + "private-input"))
        self.assertIsNotNone(PERSONAL_PATH.search(".." + "/" + "armigo-private"))
        self.assertIsNotNone(PERSONAL_PATH.search(".." + "/" + "armigo-reference/header.h"))

    def test_portable_text(self):
        for text in (
            "src/platform/host_module.cpp",
            "../include/armigo/runtime.h",
            "cmake --preset macos-debug",
            "https://example.invalid/home/guide",
            "https://github.com/TommyLau/Armigo",
            "The caller supplies an external input.",
            "~/Library/Caches/Armigo",
            "~/Library/Caches/Armigo/pso",
            "~/Library/Application Support/Armigo",
            "~/Library/Application Support/Armigo/title",
            "~/Documents/Armigo",
            "~/Documents/Armigo/0100",
        ):
            with self.subTest(text=text):
                self.assertIsNone(PERSONAL_PATH.search(text))

    def test_other_tilde_paths_stay_personal(self):
        candidates = [
            "~" + "/Library/Caches/Other",
            "~" + "/Library/Caches/ArmigoExtra",
            "~" + "/Library/Application Support/Other",
            "~" + "/Documents/Other",
            "~" + "/Documents/ArmigoExtra",
            "~" + "/" + "tmp" + "/note",
        ]
        for candidate in candidates:
            with self.subTest(candidate=candidate):
                self.assertIsNotNone(PERSONAL_PATH.search(candidate))


class SigningMaterialTests(unittest.TestCase):
    def assert_rejected(self, files, lines, secrets=()):
        with _indexed_repository(files) as root:
            result = _run_check(root)
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertEqual(result.stdout, "")
        self.assertEqual(result.stderr.splitlines(), lines)
        combined = result.stdout + result.stderr
        for secret in secrets:
            self.assertNotIn(secret, combined)
            self.assertGreaterEqual(len(secret), 2)

    def test_signing_extensions(self):
        suffixes = (
            ".p12", ".pfx", ".cer", ".crt", ".der", ".pem", ".key", ".keys",
            ".mobileprovision", ".provisionprofile", ".keychain",
            ".keychain-db", ".xcconfig", ".XCConfig",
        )
        for suffix in suffixes:
            name = "material" + suffix
            with self.subTest(name=name):
                self.assert_rejected(
                    {name: "placeholder\n"},
                    [f"{name}:0: signing-extension"],
                )

    def test_signing_filenames(self):
        for name in ("prod.keys", "title.keys", "dev.keys", "Nested/PROD.KEYS"):
            with self.subTest(name=name):
                self.assert_rejected(
                    {name: "placeholder\n"},
                    [f"{name}:0: signing-filename"],
                )

    def test_certificate_armor(self):
        mark = _certificate_mark()
        body = "MII" + "FAKE" + "BODY"
        self.assert_rejected(
            {"notes.txt": "preface\n" + mark + "\n" + body + "\n"},
            ["notes.txt:2: embedded-certificate"],
            (mark, body),
        )

    def test_private_key_armor(self):
        body = "MII" + "FAKE" + "KEY"
        for label in ("", "RSA ", "EC ", "ENCRYPTED "):
            mark = _private_key_mark(label)
            with self.subTest(label=label or "plain"):
                self.assert_rejected(
                    {"notes.txt": mark + "\n" + body + "\n"},
                    ["notes.txt:1: embedded-private-key"],
                    (mark, body),
                )

    def test_signing_settings(self):
        samples = (
            ("signing-setting-team", ("DEVELOPMENT", "_TEAM")),
            ("signing-setting-identity", ("CODE", "_SIGN_", "IDENTITY")),
            ("signing-setting-profile", ("PROVISIONING", "_PROFILE")),
            ("signing-setting-style", ("CODE", "_SIGN_", "STYLE")),
        )
        for rule, parts in samples:
            token = _setting(*parts)
            with self.subTest(rule=rule):
                self.assert_rejected(
                    {"settings.txt": token + " = EXAMPLE\n"},
                    [f"settings.txt:1: {rule}"],
                    (token, "EXAMPLE"),
                )

    def test_codesign_identity(self):
        identity = "Example" + "Developer"
        quoted = "Example" + " Developer"
        samples = (
            _codesign("--sign", identity),
            _codesign("-s", identity),
            _codesign("--force", "--sign", identity),
            _codesign("--sign=" + identity),
            'COMMAND ' + _codesign("--force", "--sign", identity),
            _codesign("--sign", '"' + quoted + '"'),
            _codesign("--sign", "-", "--sign", identity),
        )
        for sample in samples:
            with self.subTest(sample=sample.split()[0:3]):
                self.assert_rejected(
                    {"build.txt": "echo start\n" + sample + "\n"},
                    ["build.txt:2: codesign-identity"],
                    (identity, quoted),
                )

    def test_key_file_lines(self):
        secret = _hex_digits(32)
        longer = _hex_digits(64)
        upper = "AB" * 16
        samples = (
            _key_line("header" + "_key", secret),
            _key_line("header" + "_key", longer),
            _key_line("header" + "_key_1a", secret),
            _key_line("header" + "_key", upper),
            "\t" + _key_line("header" + "_key", secret).replace(" = ", "="),
            _key_line("key_area" + "_key_application_00", secret),
            _key_line("master" + "_key_0" + "0", secret),
            _key_line("titlekek" + "_0" + "0", longer),
            _key_line("alpha", longer),
            _key_line("header" + "_key_1", secret),
        )
        for line in samples:
            with self.subTest(line=line[:24]):
                secrets = tuple(
                    value for value in (secret, longer, upper)
                    if value and value in line
                )
                self.assert_rejected(
                    {"keys.txt": "\n" + line + "\n"},
                    ["keys.txt:2: key-file-assignment"],
                    secrets,
                )

    def test_ordinary_code_is_not_a_key_file(self):
        digest = _hex_digits(64)
        samples = (
            'const char text[] = "' + digest + '";',
            "constexpr auto word = 0x" + _hex_digits(32) + ";",
            '{"sha256":"' + digest + '"}',
            digest,
            'SHA256("") = ' + digest,
            _key_line("header" + "_key", digest, suffix=" # note"),
            _key_line("header" + "_key", digest, prefix="# "),
            _key_line("header" + "_key", _hex_digits(31)),
        )
        for sample in samples:
            with self.subTest(sample=sample[:32]):
                self.assertEqual(signing_content_failures("sample.cpp", sample), [])

    def test_several_findings_hide_material(self):
        secret = _hex_digits(32)
        mark = _certificate_mark()
        token = _setting("DEVELOPMENT", "_TEAM")
        self.assert_rejected(
            {
                "a.txt": "\n".join((
                    _key_line("header" + "_key", secret),
                    mark,
                    token,
                )) + "\n",
                "b.pem": "placeholder\n",
            },
            [
                "a.txt:1: key-file-assignment",
                "a.txt:2: embedded-certificate",
                "a.txt:3: signing-setting-team",
                "b.pem:0: signing-extension",
            ],
            (secret, mark, token),
        )

    def test_clean_tree_passes(self):
        notes = "\n".join((
            "portable notes",
            "codesign --force --sign -",
            "codesign -s -",
            "codesign --sign=-",
            'codesign --sign "-"',
            "codesign --verify --verbose=4",
            "codesign -d --entitlements -",
            'COMMAND codesign --force --sign - "$<TARGET_BUNDLE_DIR:armigo>"',
            "notcodesign --sign OTHER",
            "-----BEGIN PUBLIC KEY-----",
            "-----BEGIN CERTIFICATE REQUEST-----",
            "development team",
            "CODE_SIGN",
            "a certificate is not armor",
            "private key material stays outside",
            _key_line("header" + "_key", _hex_digits(31)),
            _key_line("master" + "_key_0", "abcd"),
            _key_line("master" + "_key_" + "00", ""),
            _key_line("master" + "_key_" + "gg", "ab"),
            _key_line("titlekek" + "_0" + "0", "zz"),
            _key_line("header" + "_key", _hex_digits(64), suffix=" # note"),
            _key_line("header" + "_key", _hex_digits(64), prefix="# "),
            'const char text[] = "' + _hex_digits(64) + '";',
            "constexpr auto word = 0x" + _hex_digits(32) + ";",
            '{"sha256":"' + _hex_digits(64) + '"}',
            _hex_digits(64),
            'SHA256("") = ' + _hex_digits(64),
            "~/Library/Caches/Armigo/pso",
            "~/Library/Application Support/Armigo",
            "~/Documents/Armigo/title",
            "prod.keys.txt is a text note",
        )) + "\n"
        files = {
            "notes.txt": notes,
            "prod.keys.txt": "not a key file\n",
            "ok.py": _python_source(),
        }
        with _indexed_repository(files) as root:
            result = _run_check(root)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stderr, "")
        self.assertIn("Repository content checks passed for 3 indexed files.", result.stdout)

    def test_existing_rules_still_fail(self):
        samples = {
            "game.nsp": (
                "placeholder\n",
                ["game.nsp: prohibited content or managed dependency extension"],
            ),
            "build/out.txt": (
                "notes\n",
                ["build/out.txt: generated or private directory is not allowed"],
            ),
            "plain.py": (
                "print('ok')\n",
                ["plain.py: missing or invalid source copyright banner"],
            ),
            "where.txt": (
                "see " + "/" + "Users" + "/sample/data\n",
                ["where.txt: personal path, local file URL, or neighboring checkout reference"],
            ),
            "text.txt": (
                "label " + chr(0x4E00) + "\n",
                ["text.txt: project content must be English"],
            ),
        }
        for name, (content, lines) in samples.items():
            with self.subTest(name=name):
                self.assert_rejected({name: content}, lines)

    def test_indexed_tree_has_no_signing_material(self):
        result = _run_check(_REPOSITORY)
        signing = [
            line for line in result.stderr.splitlines() if _SIGNING_FAILURE.search(line)
        ]
        self.assertEqual(signing, [])


def _banner(prefix: str, title: str) -> str:
    tail = [
        "Copyright (c) 2026 Tommy Lau.",
        "SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary",
        "",
        "This file is licensed under GPL-3.0-only or, by separate written",
        "agreement, an Armigo proprietary license.",
        "See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.",
        "=" * 76,
    ]
    lines = [prefix + " " + ("=" * 76), prefix + " " + title]
    for line in tail:
        lines.append(prefix + (" " + line if line else ""))
    return "\n".join(lines) + "\n"


class GameNameTests(unittest.TestCase):
    def test_tokens_match_titles_and_skip_ordinary_words(self):
        self.assertEqual(game_name_token("botw_frame"), "botw")
        self.assertEqual(game_name_token("BOTW"), "botw")
        self.assertEqual(game_name_token("breath-of-the-wild"), "breath-of-the-wild")
        self.assertEqual(game_name_token("tears_of_the_kingdom"), "tears-of-the-kingdom")
        self.assertEqual(game_name_token("persona"), "persona")
        self.assertEqual(game_name_token("p5r"), "p5r")
        self.assertEqual(game_name_token("totk"), "totk")
        self.assertEqual(game_name_token("zelda"), "zelda")
        self.assertEqual(game_name_token("unicorn"), "unicorn")
        self.assertEqual(game_name_token("overlord"), "overlord")
        self.assertIsNone(game_name_token("personal"))
        self.assertIsNone(game_name_token("Personality"))
        self.assertIsNone(game_name_token("title.01007EF00011E000.frame_rate"))
        self.assertIsNone(game_name_token("src/titles/01007EF00011E000/frame_rate.cpp"))

    def test_cmake_names_and_foreach_items(self):
        clean = "\n".join((
            "add_library(armigo_compat STATIC frame_rate.cpp)",
            "add_executable(armigo_01007EF00011E000_frame_rate_tests frame_rate_tests.cpp)",
            "add_test(NAME title.01007EF00011E000.frame_rate COMMAND armigo_01007EF00011E000_frame_rate_tests)",
            "foreach(test_name IN ITEMS abi counting)",
            "    add_executable(armigo_nvn_${test_name}_tests ${test_name}.cpp)",
            "    add_test(NAME nvn.${test_name} COMMAND armigo_nvn_${test_name}_tests)",
            "endforeach()",
            "# add_executable(armigo_botw_tests botw.cpp)",
        ))
        self.assertEqual(cmake_game_name_failures("CMakeLists.txt", clean), [])
        named = "add_executable(armigo_p5r_tests p5r.cpp)\n"
        self.assertEqual(
            cmake_game_name_failures("CMakeLists.txt", named),
            ["CMakeLists.txt:1: game-name"],
        )
        listed = "\n".join((
            "foreach(test_name IN ITEMS abi botw)",
            "    add_test(NAME nvn.${test_name} COMMAND driver)",
            "endforeach()",
        ))
        self.assertEqual(
            cmake_game_name_failures("CMakeLists.txt", listed),
            ["CMakeLists.txt:1: game-name"],
        )

    def test_indexed_names_reject_game_tokens_and_allow_prose(self):
        source = _banner("//", "Armigo Tests - Synthetic") + "int main() { return 0; }\n"
        cmake = _banner("#", "Armigo Build - Targets") + (
            "add_executable(armigo_zelda_tests zelda.cpp)\n"
        )
        command = cmake.splitlines()[-1]
        files = {
            "src/compat/botw_frame.cpp": source,
            "notes-personal.txt": "personal notes\n",
            "notes/story.md": "title 01007EF00011E000 (Breath of the Wild)\n",
            "CMakeLists.txt": cmake,
            "src/titles/01007EF00011E000/frame_rate.cpp": source,
        }
        with _indexed_repository(files) as root:
            result = _run_check(root)
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertEqual(result.stdout, "")
        self.assertEqual(sorted(result.stderr.splitlines()), sorted([
            "src/compat/botw_frame.cpp:0: game-name",
            f"CMakeLists.txt:{cmake.splitlines().index(command) + 1}: game-name",
        ]))

    def test_indexed_tree_has_no_game_names(self):
        result = _run_check(_REPOSITORY)
        names = [line for line in result.stderr.splitlines() if line.endswith(": game-name")]
        self.assertEqual(names, [], result.stderr)


if __name__ == "__main__":
    unittest.main()
