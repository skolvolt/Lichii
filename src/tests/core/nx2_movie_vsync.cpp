// SPDX-License-Identifier: GPL-3.0-or-later

#include <catch2/catch_test_macros.hpp>

#include "core/hle/service/nx2_movie_vsync.h"

namespace {
constexpr s64 Frame = 16667;
constexpr s64 Start = 5'000'000;
constexpr u32 Limit = 170;
} // namespace

TEST_CASE("NX2 movie vsync keeps the frame rate limit without a movie", "[nx2_movie_vsync]") {
    const Service::Nx2MovieVsync vsync;
    REQUIRE_FALSE(vsync.Playing(0));
    REQUIRE_FALSE(vsync.Playing(Start));
    REQUIRE(vsync.RateHz(Start, Limit) == Limit);
}

TEST_CASE("NX2 movie vsync runs at the display rate while a movie plays", "[nx2_movie_vsync]") {
    Service::Nx2MovieVsync vsync;
    for (s64 frame = 0; frame < 120; ++frame) {
        const s64 now = Start + frame * Frame;
        vsync.FrameWritten(now, 1080);
        REQUIRE(vsync.RateHz(now + Frame - 1, Limit) == Service::Nx2MovieVsync::DisplayRateHz);
    }
    const s64 last = Start + 119 * Frame;
    REQUIRE(vsync.Playing(last + Service::Nx2MovieVsync::HoldUs - 1));
    // Stopped or paused: back to the limit.
    REQUIRE(vsync.RateHz(last + Service::Nx2MovieVsync::HoldUs, Limit) == Limit);
}

TEST_CASE("NX2 movie vsync ignores videos smaller than 720p", "[nx2_movie_vsync]") {
    Service::Nx2MovieVsync vsync;
    vsync.FrameWritten(Start, 480);
    REQUIRE_FALSE(vsync.Playing(Start));
    vsync.FrameWritten(Start, 720);
    REQUIRE(vsync.Playing(Start));
}
