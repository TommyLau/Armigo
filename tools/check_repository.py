#!/usr/bin/env python3
# ============================================================================
# Armigo Tools - Development and Verification Tools
# Copyright (c) 2026 Tommy Lau.
# SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
#
# This file is licensed under GPL-3.0-only or, by separate written
# agreement, an Armigo proprietary license.
# See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
# ============================================================================

"""Check indexed repository content without inspecting private runtime data.

Signing, certificate and key-file findings name the indexed path, the line
and the rule. They never include the matched text. Path rules use line 0.
"""

from pathlib import PurePosixPath
import re
import subprocess
import sys


PROHIBITED_SUFFIXES = {
    ".nsp", ".nsz", ".xci", ".xcz", ".nca", ".ncz", ".nso", ".nro",
    ".rom", ".bin", ".keys", ".key", ".pem", ".p12", ".mobileprovision",
    ".idb", ".i64", ".gpr", ".metallib", ".air", ".log", ".trace", ".dmp",
    ".dll", ".exe", ".pdb", ".cs", ".csproj", ".sln", ".fsproj", ".vbproj", ".hpp",
}
PROHIBITED_DIRECTORIES = {
    "private", "runtime-data", "captures", "saves", "reports", "build",
    "out", ".cache", ".local", ".worktrees",
}
ARTWORK_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".psd"}
CJK_TEXT = re.compile("[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\U00020000-\U0003134f]")
LFS_POINTER = re.compile(
    rb"version https://git-lfs.github.com/spec/v1\n"
    rb"oid sha256:[0-9a-f]{64}\nsize [0-9]+\n"
)
SOURCE_SUFFIXES = {
    ".h", ".c", ".cc", ".cpp", ".m", ".mm", ".s", ".metal", ".swift",
    ".py", ".sh", ".cmake", ".yaml", ".yml",
}
COPYRIGHT_TAIL = [
    "Copyright (c) 2026 Tommy Lau.",
    "SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary",
    "",
    "This file is licensed under GPL-3.0-only or, by separate written",
    "agreement, an Armigo proprietary license.",
    "See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.",
    "=" * 76,
]
# These three prefixes, and paths under them, are Armigo's own per-user
# locations. The spelling is exact; every other tilde path stays personal.
_ARMIGO_HOME = (
    r"(?-i:(?:Library[\\/]Caches[\\/]Armigo"
    r"|Library[\\/]Application Support[\\/]Armigo"
    r"|Documents[\\/]Armigo)(?=[\\/]|[\s\"']|$))"
)
PERSONAL_PATH = re.compile(
    r"(?<![A-Za-z0-9:/])(?:(?:/Users|/home|/Volumes|/tmp|/var/tmp)/[^\s/\"']+"
    r"|/private/(?:tmp|var)/[^\s/\"']+"
    r"|[A-Za-z]:[\\/]Users[\\/][^\s\\/\"']+"
    r"|~[\\/](?!" + _ARMIGO_HOME + r")[^\s\"']+)"
    r"|file:[/]{2}"
    r"|[.][.][/\\](?:armigo-private|armigo-reference)(?=[/\\\s\"']|$)",
    re.IGNORECASE,
)
# Certificate, key and signing-setting paths. An xcconfig may carry signing
# settings, so the extension itself is rejected.
SIGNING_EXTENSIONS = {
    ".p12", ".pfx", ".cer", ".crt", ".der", ".pem", ".key", ".keys",
    ".mobileprovision", ".provisionprofile", ".keychain", ".keychain-db",
    ".xcconfig",
}
SIGNING_FILENAMES = {"prod.keys", "title.keys", "dev.keys"}
_CERTIFICATE_MARK = "-----" + "BEGIN CERTIFICATE" + "-----"
_PRIVATE_KEY_MARK = re.compile(
    "-----" + "BEGIN " + "(?:RSA |EC |ENCRYPTED )?" + "PRIVATE KEY" + "-----"
)
_SIGNING_SETTINGS = (
    ("signing-setting-team", "DEVELOPMENT" + "_TEAM"),
    ("signing-setting-identity", "CODE" + "_SIGN_" + "IDENTITY"),
    ("signing-setting-profile", "PROVISIONING" + "_PROFILE"),
    ("signing-setting-style", "CODE" + "_SIGN_" + "STYLE"),
)
# A key-file line is only an identifier, "=", and 32 or more hex digits.
_KEY_ASSIGNMENT_LINE = re.compile(
    r"[A-Za-z_][A-Za-z0-9_]*[ \t]*=[ \t]*[0-9A-Fa-f]{32,}[ \t]*"
)
_CODESIGN_CALL = re.compile(r"(?:^|[\s/;&|`(])codesign(?=\s)")
# Whole-word game titles and abbreviations. Separators (_, space) fold to '-',
# so a file or target such as botw_frame or breath-of-the-wild matches, while
# an ordinary word such as "personal" does not match "persona".
GAME_NAME_TOKENS = (
    "botw",
    "p5r",
    "totk",
    "persona",
    "zelda",
    "unicorn",
    "overlord",
    "breath-of-the-wild",
    "tears-of-the-kingdom",
)
_GAME_NAME = re.compile(
    r"(?<![a-z0-9])(?:"
    + "|".join(re.escape(token) for token in sorted(GAME_NAME_TOKENS, key=len, reverse=True))
    + r")(?![a-z0-9])"
)
# Indentation is spaces or tabs only. A newline-matching \s would make the
# match start on an earlier blank line and report the wrong line number.
_CMAKE_TARGET = re.compile(
    r"^[ \t]*add_(?:library|executable|custom_target)[ \t]*\([ \t]*([^\s)#]+)",
    re.MULTILINE,
)
_CMAKE_TEST = re.compile(
    r"^[ \t]*add_test[ \t]*\([ \t]*(?:NAME[ \t]+)?([^\s)#]+)",
    re.MULTILINE,
)
# Names built as armigo_${suite}_tests pick up the words in the matching
# foreach(... IN ITEMS ...) list. Those words are part of the target name.
_CMAKE_FOREACH_ITEMS = re.compile(
    r"^[ \t]*foreach[ \t]*\([ \t]*([A-Za-z_][A-Za-z0-9_]*)[ \t]+IN[ \t]+ITEMS[ \t]+([^)]*)\)",
    re.MULTILINE,
)


def valid_source_banner(path: PurePosixPath, text: str) -> bool:
    if path.suffix.lower() not in SOURCE_SUFFIXES and path.name != "CMakeLists.txt":
        return True
    prefix = "#" if path.suffix.lower() in {".py", ".sh", ".cmake", ".yaml", ".yml"} \
        or path.name == "CMakeLists.txt" else "//"
    lines = text.splitlines()
    if lines and lines[0].startswith("#!"):
        lines = lines[1:]
    tail = [prefix + (" " + line if line else "") for line in COPYRIGHT_TAIL]
    end = 2 + len(tail)
    return len(lines) >= end and lines[0] == prefix + " " + "=" * 76 \
        and lines[1].startswith(prefix + " Armigo ") and " - " in lines[1] \
        and lines[2:end] == tail


def game_name_token(text: str) -> str | None:
    folded = re.sub(r"[_\s]+", "-", text).casefold()
    match = _GAME_NAME.search(folded)
    return match.group(0) if match else None


def cmake_game_name_failures(name: str, text: str) -> list[str]:
    code = "\n".join(line.split("#", 1)[0] for line in text.splitlines())
    failures = []
    declared = []
    for pattern in (_CMAKE_TARGET, _CMAKE_TEST):
        declared.extend(pattern.finditer(code))
    for match in declared:
        if game_name_token(match.group(1)):
            failures.append(_violation(name, code[:match.start()].count("\n") + 1, "game-name"))
    referenced = {
        variable
        for match in declared
        for variable in re.findall(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}", match.group(1))
    }
    if referenced:
        for items in _CMAKE_FOREACH_ITEMS.finditer(code):
            if items.group(1) not in referenced:
                continue
            if any(game_name_token(word) for word in items.group(2).split()):
                failures.append(_violation(name, code[:items.start()].count("\n") + 1, "game-name"))
    return failures


def signing_path_rule(path: PurePosixPath) -> str | None:
    if path.name.casefold() in SIGNING_FILENAMES:
        return "signing-filename"
    if path.suffix.casefold() in SIGNING_EXTENSIONS:
        return "signing-extension"
    return None


def _violation(name: str, line: int, rule: str) -> str:
    return f"{name}:{line}: {rule}"


def _command_tokens(text: str) -> list[str]:
    tokens = []
    index = 0
    length = len(text)
    while index < length:
        character = text[index]
        if character.isspace():
            index += 1
            continue
        if character in "&|;":
            break
        if character in "\"'":
            quote = character
            index += 1
            start = index
            while index < length and text[index] != quote:
                index += 1
            tokens.append(text[start:index])
            if index < length:
                index += 1
            continue
        start = index
        while index < length and not text[index].isspace() and text[index] not in "&|;\"'":
            index += 1
        tokens.append(text[start:index])
    return tokens


def _sign_identities(tokens: list[str]) -> list[str]:
    identities = []
    index = 0
    count = len(tokens)
    while index < count:
        token = tokens[index]
        if token in {"--sign", "-s"} and index + 1 < count:
            identities.append(tokens[index + 1])
            index += 2
            continue
        if token.startswith("--sign="):
            identities.append(token[len("--sign="):])
        index += 1
    return identities


def _forbidden_codesign(line: str) -> bool:
    # Ad-hoc signing passes the identity "-". Any other identity is rejected.
    for match in _CODESIGN_CALL.finditer(line):
        identities = _sign_identities(_command_tokens(line[match.end():]))
        if any(identity != "-" for identity in identities):
            return True
    return False


def _key_file_rule(line: str) -> str | None:
    if _KEY_ASSIGNMENT_LINE.fullmatch(line.strip()):
        return "key-file-assignment"
    return None


def signing_content_failures(name: str, text: str) -> list[str]:
    failures = []
    for number, line in enumerate(text.splitlines(), start=1):
        rules = []
        if _CERTIFICATE_MARK in line:
            rules.append("embedded-certificate")
        if _PRIVATE_KEY_MARK.search(line):
            rules.append("embedded-private-key")
        for rule, token in _SIGNING_SETTINGS:
            if token in line:
                rules.append(rule)
        if _forbidden_codesign(line):
            rules.append("codesign-identity")
        key_rule = _key_file_rule(line)
        if key_rule is not None:
            rules.append(key_rule)
        failures.extend(_violation(name, number, rule) for rule in rules)
    return failures


def git(*arguments: str) -> bytes:
    return subprocess.check_output(["git", *arguments])


def main() -> int:
    failures = []
    entries = git("ls-files", "--stage", "-z").split(b"\0")
    checked = 0
    for entry in entries:
        if not entry:
            continue
        metadata, encoded_path = entry.split(b"\t", 1)
        mode, object_id, stage = metadata.split()
        name = encoded_path.decode("utf-8")
        path = PurePosixPath(name)
        checked += 1
        if stage != b"0" or mode not in {b"100644", b"100755"}:
            failures.append(f"{name}: unresolved entry, symlink, or submodule is not allowed")
            continue
        signing_rule = signing_path_rule(path)
        if signing_rule is not None:
            failures.append(_violation(name, 0, signing_rule))
            continue
        if path.suffix.lower() in PROHIBITED_SUFFIXES:
            failures.append(f"{name}: prohibited content or managed dependency extension")
            continue
        if any(part.lower() in PROHIBITED_DIRECTORIES for part in path.parts[:-1]):
            failures.append(f"{name}: generated or private directory is not allowed")
            continue
        if game_name_token(name):
            failures.append(_violation(name, 0, "game-name"))
        content = git("cat-file", "blob", object_id.decode("ascii"))
        if path.parts[:2] == ("assets", "original") and path.suffix.lower() in ARTWORK_SUFFIXES:
            if not LFS_POINTER.fullmatch(content):
                failures.append(f"{name}: original artwork must be an LFS pointer")
            continue
        try:
            text = content.decode("utf-8")
        except UnicodeDecodeError:
            failures.append(f"{name}: unexpected binary content")
            continue
        if "\0" in text:
            failures.append(f"{name}: unexpected binary content")
        if CJK_TEXT.search(text):
            failures.append(f"{name}: project content must be English")
        if PERSONAL_PATH.search(text):
            failures.append(f"{name}: personal path, local file URL, or neighboring checkout reference")
        if not valid_source_banner(path, text):
            failures.append(f"{name}: missing or invalid source copyright banner")
        if path.name == "CMakeLists.txt" or path.suffix == ".cmake":
            failures.extend(cmake_game_name_failures(name, text))
        failures.extend(signing_content_failures(name, text))

    for failure in failures:
        print(failure, file=sys.stderr)
    if failures:
        return 1
    print(f"Repository content checks passed for {checked} indexed files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
