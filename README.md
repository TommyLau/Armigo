# Armigo

Armigo turns your own Nintendo Switch game files into a standalone player app
for Apple Silicon Macs and for iPhone and iPad. It is a native compatibility
runtime: a thin Horizon (firmware 20.5.0) layer that runs the game's own code
and SDK unchanged, with graphics on Metal 4.

You give Armigo a game you own, its updates and add-on content, your console's
keys and firmware. It writes one app for that game: double-click it on a Mac,
or sign and install the `.ipa` on an iPhone or iPad. Armigo ships no games, no
keys and no firmware, and it signs nothing.

## Status

Armigo supports only the titles with a folder in [`titles/`](titles), and only
the game builds listed in that folder's `profile.json`:

| Title ID | Title | State |
| --- | --- | --- |
| `01007EF00011E000` | The Legend of Zelda: Breath of the Wild | Playable on macOS and iPadOS |
| `01004B10157F2000` | Persona 5 Royal | In progress; packing is not supported yet |

Other titles are not supported. Tears of the Kingdom and Unicorn Overlord are
planned, not supported.

## Requirements

- An Apple Silicon Mac with macOS 26 or later.
- Xcode 26 or later with its Metal toolchain installed, and the iOS SDK if
  you pack for iOS.
- CMake 3.25 or later and Ninja (`brew install cmake ninja`).
- For iOS: an iPhone or iPad with iOS/iPadOS 26 or later, and your own way to
  sign apps.

## Build

```sh
cmake --preset macos-release
cmake --build --preset macos-release
```

This builds the packer `build/macos-release/armigo` and the player it puts in
every app. To run the tests:

```sh
ctest --preset macos-release
```

## Pack your game

```sh
build/macos-release/armigo "My Game.nsp" --update "My Game v1.2.nsp" \
  --dlc "My Game DLC.nsp" --keys prod.keys --firmware firmware/ --output "$HOME/Games"
```

This writes `<Name>.app` into that folder. Add `--platform ios` to write `<Name>.ipa`
instead. [PACKING.md](PACKING.md) explains every input and option, what goes
into the app, and how to sign the iOS app yourself.

## Playing

A packed app opens straight into the game. Saves are kept in
`~/Documents/Armigo/<TitleID>/`.

**Settings** (macOS: Settings in the app menu, ⌘,; iOS: the app's page in
the Settings app):

| Setting | Choices |
| --- | --- |
| Console Mode | Docked (default), Handheld |
| Resolution | Auto (default: native on macOS, 1080p on iOS), 720p, 900p, 1080p, 1440p, 4K (macOS), Native |
| Frame Rate | Auto, 30, 60 (default) |
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
