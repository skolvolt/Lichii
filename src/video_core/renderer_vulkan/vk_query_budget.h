// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <cstddef>

namespace Vulkan::OcclusionQueryBudget {

// See SamplesStreamer::CloseCounter. Eight banks retain the >1,024-slot gameplay
// windows observed in OoT Water Temple A3 while keeping unreported menu queries bounded.
inline constexpr std::size_t SlotsPerBank = 256;
inline constexpr std::size_t MaxRunningSlots = 8 * SlotsPerBank;

[[nodiscard]] constexpr bool MustDrop(std::size_t running_slots) noexcept {
    return running_slots > MaxRunningSlots;
}

} // namespace Vulkan::OcclusionQueryBudget
