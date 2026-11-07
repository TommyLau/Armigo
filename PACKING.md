<!--
  Armigo - Packing your own app
  Copyright (c) 2026 Tommy Lau.
  SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary

  This file is licensed under GPL-3.0-only or, by separate written
  agreement, an Armigo proprietary license.
  See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
-->

# Packing your own app

`armigo` turns one game you own into one player app: a macOS `.app`, or an iOS
`.ipa`. Build Armigo first (see [README.md](README.md)). Packing
supports the titles listed in the README's status table.

```text
armigo <game.xci|nsp or folder> [--update FILE]... [--dlc FILE]...
       --keys FILE --firmware PATH [--title-keys FILE] [--output DIR]
       [--platform macos|ios] [--reuse APP] [--ios-build DIR] [--keep-app] [--no-ipa]
```

## Inputs

All inputs are your own dumps.

| Input | What it is |
| --- | --- |
| game | The base game: an `.xci` or `.nsp` file, or a folder of `.xci`, `.nsp` or `.nca` files |
| `--update FILE` | An update `.nsp`; repeat for several, the newest applies |
| `--dlc FILE` | An add-on content `.nsp`; repeat for each |
| `--keys FILE` | Your console's `prod.keys`; it must contain every key your game's content uses (the packer names any missing key) |
| `--firmware PATH` | Your console's firmware: a folder of its `.nca` files, or a `.zip` of them. Armigo's baseline is 20.5.0; other versions pack with a warning |
| `--title-keys FILE` | Optional `title.keys`. Without it, title keys come from the tickets inside the NSPs |
| `--output DIR` | Where the app is written (default: the current directory); never inside a git checkout |

Keys are used in memory only. They are never written into the app or anywhere
else. The packer checks that the game's build is one the title's settings in
`titles/<TitleID>/profile.json` support, and stops with the reason if not.

## What the app contains

The app carries everything it needs, so it starts without options:

- the game's ExeFS and RomFS with the newest update applied, and each add-on;
- the game's translated shaders and, on macOS, its prepared pipelines;
- the title's settings from `titles/<TitleID>/`;
- the system data the player uses from your firmware (the shared fonts);
- the title's name in each of its languages, and its icon.

The app's bundle identifier is `com.tommylau.armigo.<TitleID>`. Its name is
the game's English title, then its Simplified Chinese title, then the first one
available; the Home Screen, Dock and Finder show the title in your language
when the game has it.

Saves live in `~/Documents/Armigo/<TitleID>/` and caches in
`~/Library/Caches/Armigo/<TitleID>/` (on iOS, inside the app's container).
Packing a newer version of the same game keeps your saves.

## macOS

```sh
build/macos-release/armigo "My Game.xci" --keys prod.keys \
  --firmware Firmware.20.5.0.zip --output "$HOME/Games"
```

The app keeps only the ad-hoc signature its player gets when it is built, and
Armigo signs nothing else. It opens on the Mac that packed it; on another Mac,
macOS may ask you to allow it in System Settings › Privacy & Security the first
time, or you can sign it with your own identity.

## iOS

```sh
build/macos-release/armigo "My Game.xci" --keys prod.keys \
  --firmware Firmware.20.5.0.zip --platform ios --output "$HOME/Games"
```

This writes one unsigned `<Name>.ipa` that contains the game. It needs iOS 26 or
later and runs without JIT. The first iOS pack configures and
builds the iOS player itself (the `ios-release` preset); `--ios-build DIR`
uses another build. If a macOS app of the same game is already in the output
folder, the packer reuses its game files and shaders (`--reuse APP` names one
explicitly).

`--keep-app` also keeps `<Name>.app` and the `entitlements.plist` the app
needs; `--no-ipa` keeps only those. Armigo never signs the app: sign it with
your own Apple developer identity and a provisioning profile whose App ID
grants the entitlements in `entitlements.plist`, then install it with Xcode,
Apple Configurator or your usual tool.

## Problems

The packer stops with a message that names the missing or invalid input. A
packed app writes its warnings and errors to `last-run.log` in its cache
folder; attach it when you report a problem, together with the title ID and
version, but never your keys, firmware or game files.
