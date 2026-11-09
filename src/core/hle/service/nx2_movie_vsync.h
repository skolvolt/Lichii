// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <atomic>
#include <limits>

#include "common/common_types.h"

namespace Service {

// The nx2:vi2 vsync event rate while a full-screen movie plays: the 60 Hz the display reports
// (QueryDisplayRefreshRate), not the frame rate limit. Games time what they draw over a movie in
// vsyncs. At a 170 Hz limit Ocarina of Time ran its new-game intro's subtitles and narration
// ~2.4x fast, so the lines overlapped, and ended the intro's overlay ~60 s into the 154 s movie,
// showing black until the movie ended.
class Nx2MovieVsync {
public:
    static constexpr u32 DisplayRateHz = 60;
    // Smaller videos are drawn into a scene (or a menu) and don't pace it.
    static constexpr u32 MinHeight = 720;
    // Frames come every 16.7 ms while a movie plays; a movie paused or stopped this long is over.
    static constexpr s64 HoldUs = 250'000;

    // A decoder wrote a frame of the given height at now_us (CoreTiming global time).
    void FrameWritten(s64 now_us, u32 height) {
        if (height >= MinHeight) {
            last_frame_us.store(now_us, std::memory_order_relaxed);
        }
    }

    bool Playing(s64 now_us) const {
        const s64 last = last_frame_us.load(std::memory_order_relaxed);
        return last != Never && now_us - last < HoldUs;
    }

    u32 RateHz(s64 now_us, u32 configured_hz) const {
        return Playing(now_us) ? DisplayRateHz : configured_hz;
    }

private:
    static constexpr s64 Never = std::numeric_limits<s64>::min();
    std::atomic<s64> last_frame_us{Never};
};

} // namespace Service
