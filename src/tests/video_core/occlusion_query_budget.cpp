// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <catch2/catch_test_macros.hpp>

#include "video_core/renderer_vulkan/vk_query_budget.h"

TEST_CASE("Occlusion query windows retain the Water Temple counts within a bounded bank budget",
          "[video_core][query_cache]") {
    using namespace Vulkan::OcclusionQueryBudget;

    STATIC_REQUIRE(MaxRunningSlots == 2048);
    REQUIRE_FALSE(MustDrop(0));
    REQUIRE_FALSE(MustDrop(346));   // earlier Hyrule/Kakariko measurements
    REQUIRE_FALSE(MustDrop(1024));  // previous inclusive boundary
    REQUIRE_FALSE(MustDrop(1025));  // Water Temple A3
    REQUIRE_FALSE(MustDrop(1027));
    REQUIRE_FALSE(MustDrop(1035));
    REQUIRE_FALSE(MustDrop(MaxRunningSlots));
    REQUIRE(MustDrop(MaxRunningSlots + 1));
    REQUIRE(MustDrop(16384));       // unreported menu counter still gets capped
}
