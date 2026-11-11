// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <string_view>

namespace Common {

/// nup: the Skitchii release this build is (window title and the updater). Releases are the
/// GitHub releases of skolvolt/Lichii tagged skitchii-v<version>, each with the package
/// Skitchii-v<version>-Windows-x64.zip (one top folder, Skitchii-v<version>).
inline constexpr std::string_view SkitchiiVersion = "1.4";

} // namespace Common
