# Armigo

Armigo turns your own Nintendo Switch game files into a standalone player app
for macOS and iOS. It is a native compatibility
runtime: a thin Horizon (firmware 23.0.1) layer that runs the game's own code
and SDK unchanged, with graphics on Metal 4.

You give Armigo a game you own, its updates and add-on content, your console's
keys and firmware. It writes one app for that game: double-click it on a Mac,
or sign and install the `.ipa` on iOS. Armigo ships no games, no
keys and no firmware, and it signs nothing.

## Status

Armigo supports only the titles with a folder in [`titles/`](titles), and only
the game builds listed in that folder's `profile.json`:

| Title ID | Title | State |
| --- | --- | --- |
| `01007EF00011E000` | The Legend of Zelda: Breath of the Wild | Playable on macOS and iOS: 60 fps, with an experimental 120 fps mode on 120 Hz displays |
| `01004B10157F2000` | Persona 5 Royal | Playable on macOS: 30 fps by default, with 60 and 120 fps modes (120 on 120 Hz displays); packs for iOS, iOS performance not yet measured |

Other titles are not supported. Tears of the Kingdom and Unicorn Overlord are
planned, not supported.

## Requirements

- An Apple Silicon Mac with macOS 26 or later.
- Xcode 26 or later with its Metal toolchain installed, and the iOS SDK if
  you pack for iOS.
- CMake 3.25 or later and Ninja (`brew install cmake ninja`). Homebrew puts
  them in `/opt/homebrew/bin`; that directory must be on your `PATH` in the
  shell you build and pack from.
- Free disk space for the app plus the packer's work files: about twice the
  finished app. Breath of the Wild with its update and both add-on packs makes
  a 15.9 GiB app, so start with at least 35 GB free.
- For iOS: a device with iOS 26 or later, and your own way to sign apps.

## Build

```sh
cmake --preset macos-release
cmake --build --preset macos-release
```

This builds the packer `build/macos-release/armigo` and the player it puts in
every app.

## What you need from your console

All of these come from a console and games you own; Armigo provides none of them.

- **The game**: an `.xci` or `.nsp` of the base game, plus `.nsp` files of its
  update and add-on content. Only the titles and game versions listed in
  `titles/<TitleID>/profile.json` are supported.
- **Keys**: your console's `prod.keys`. They must be at least as new as the
  firmware you give Armigo. If packing stops with "The keys do not provide
  'key_area_key_application_NN'", your keys are older than the firmware: dump
  them again from a console running that firmware version or newer.
- **Firmware**: your console's 23.0.1 firmware, as a `.zip` or a folder of its
  `.nca` files.

## Pack your own game

macOS:

```sh
build/macos-release/armigo "My Game.xci" --update "My Game v1.2.nsp" \
  --dlc "My Game DLC.nsp" --keys prod.keys --firmware Firmware.23.0.1.zip \
  --output "$HOME/Games"
```

This writes `<Name>.app` into that folder. For iOS and iPadOS, add
`--platform ios` to write `<Name>.ipa`.

Packing translates and compiles every shader and pipeline the game uses, so it
takes several minutes and uses every CPU core. The app is self-contained: the
game, its prepared code, shaders, pipelines and settings are inside it, and
your original files are not needed to play.

## Signing

Armigo signs nothing. A macOS app runs as written, on the linker's ad-hoc
signature. To install the `.ipa` on an iPhone or iPad, sign it with your own
Apple developer identity and a provisioning profile for your device.

[PACKING.md](PACKING.md) explains every input and option, what goes into the
app, and how to sign the iOS app yourself.

## Playing

A packed app opens straight into the game. Saves are kept in
`~/Documents/Armigo/<TitleID>/`.

**Settings** (macOS: Settings in the app menu, ⌘,; iOS: the app's page in
the Settings app):

| Setting | Choices |
| --- | --- |
| Console Mode | Docked (default), Handheld |
| Resolution | Auto (default: native on macOS, 1080p on iOS), 720p, 900p, 1080p, 1440p, 4K (macOS), Native |
| Frame Rate | Auto (default: the title's own rate, 60 for Breath of the Wild and 30 for Persona 5 Royal), 30, 60, 120 (on 120 Hz displays; experimental for Breath of the Wild) |
| V-Sync | On (default), Off (macOS) |
| Performance Overlay | Off (default), On |

The game's language follows the language you choose for the app in System
Settings (or the device language), and the region follows that language.

**Controls.** A connected game controller works on both platforms. On iOS an
on-screen controller appears when no controller is connected. On macOS the
keyboard and mouse also work:

| Key | Switch |
| --- | --- |
| W A S D | Left stick |
| I J K L | Right stick |
| Arrow keys | D-pad |
| E / Shift / Space / U | A / B / X / Y |
| Q / O | ZL / ZR |
| F / R | L / R |
| C / V | Left / right stick click |
| Return / Tab | + / − |
| Click the window, then move the mouse | Right stick (camera); left and right buttons are Y and ZR; Esc releases the mouse |
| F1 | Show or hide this key guide |

## Your files, your responsibility

You must lawfully own the games you pack and the console you take keys and
firmware from. Armigo provides no games, keys, firmware or ways to get them,
and it opposes piracy. A packed app contains your own game and system files and
is for your own devices; do not share it.

## License

Armigo is available under the GNU General Public License version 3 only, or
under a separate written Armigo proprietary license. See [LICENSE](LICENSE),
[COMMERCIAL-LICENSE.md](COMMERCIAL-LICENSE.md),
[DISCLAIMER.md](DISCLAIMER.md) and [PRIVACY.md](PRIVACY.md).

Armigo is an independent project. It is not affiliated with or endorsed by
Nintendo or any game publisher. Nintendo Switch, Breath of the Wild and other
names are trademarks of their owners. Armigo is provided without warranty,
subject to the license and mandatory law.
