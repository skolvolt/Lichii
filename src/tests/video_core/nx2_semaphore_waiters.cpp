// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <catch2/catch_test_macros.hpp>

#include "video_core/nx2.h"

namespace {
using Waiters = Tegra::Nx2SemaphoreWaiters;
constexpr u64 address = 0xd6d2044;
} // Anonymous namespace

TEST_CASE("A delayed release meets acquires before its write", "[video_core][nx2]") {
    auto& waiters = Waiters::Get();
    const int before = 0;
    const int after = 0;

    waiters.Add(&before, address, 5, Waiters::AcquireGequal);
    waiters.OnDelayedRelease(address, 5);
    CHECK(waiters.Latched(&before));

    waiters.Add(&after, address, 5, Waiters::AcquireEqual);
    CHECK(waiters.Latched(&after));

    waiters.OnDelayedReleaseDone(address, 5);
    waiters.Add(&after, address, 5, Waiters::AcquireEqual);
    CHECK_FALSE(waiters.Latched(&after));

    waiters.Remove(&before);
    waiters.Remove(&after);
}

TEST_CASE("A finished release leaves a newer delayed release in place", "[video_core][nx2]") {
    auto& waiters = Waiters::Get();
    const int owner = 0;

    waiters.OnDelayedRelease(address, 6);
    waiters.OnDelayedRelease(address, 7);
    waiters.OnDelayedReleaseDone(address, 6);
    waiters.Add(&owner, address, 7, Waiters::AcquireEqual);
    CHECK(waiters.Latched(&owner));

    waiters.OnDelayedReleaseDone(address, 7);
    waiters.Remove(&owner);
}
