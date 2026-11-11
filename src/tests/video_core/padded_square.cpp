// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <catch2/catch_test_macros.hpp>

#include "video_core/texture_cache/image_info.h"
#include "video_core/texture_cache/util.h"

namespace {

using VideoCommon::ImageInfo;
using VideoCommon::ImageType;
using VideoCore::Surface::PixelFormat;

ImageInfo Depth(PixelFormat format, u32 width, u32 height) {
    ImageInfo info;
    info.format = format;
    info.type = ImageType::e2D;
    info.size = {width, height, 1};
    info.block = {0, 4, 0};
    return info;
}

} // namespace

TEST_CASE("A depth target padded to whole GOBs is a padded square", "[video_core][texture_cache]") {
    // Ocarina of Time's local shadow maps: 4 bytes a texel, 16 texels a GOB.
    REQUIRE(VideoCommon::IsPaddedSquare(Depth(PixelFormat::D32_FLOAT, 800, 793)));
    REQUIRE(VideoCommon::IsPaddedSquare(Depth(PixelFormat::D32_FLOAT, 1024, 1017)));
    REQUIRE(VideoCommon::IsPaddedSquare(Depth(PixelFormat::D32_FLOAT, 144, 129)));
    // 2 bytes a texel: 32 texels a GOB
    REQUIRE(VideoCommon::IsPaddedSquare(Depth(PixelFormat::D16_UNORM, 800, 793)));
}

TEST_CASE("Other depth targets are not padded squares", "[video_core][texture_cache]") {
    REQUIRE_FALSE(VideoCommon::IsPaddedSquare(Depth(PixelFormat::D32_FLOAT, 793, 793)));
    REQUIRE_FALSE(VideoCommon::IsPaddedSquare(Depth(PixelFormat::D32_FLOAT, 793, 800)));
    // One more texel needs another GOB column.
    REQUIRE_FALSE(VideoCommon::IsPaddedSquare(Depth(PixelFormat::D32_FLOAT, 801, 793)));
    REQUIRE_FALSE(VideoCommon::IsPaddedSquare(Depth(PixelFormat::D32_FLOAT, 816, 793)));
    REQUIRE_FALSE(VideoCommon::IsPaddedSquare(Depth(PixelFormat::D32_FLOAT, 1920, 1080)));
    // 2 bytes a texel: 816 texels need a 26th GOB column.
    REQUIRE_FALSE(VideoCommon::IsPaddedSquare(Depth(PixelFormat::D16_UNORM, 816, 793)));

    ImageInfo multisampled = Depth(PixelFormat::D32_FLOAT, 800, 793);
    multisampled.num_samples = 4;
    REQUIRE_FALSE(VideoCommon::IsPaddedSquare(multisampled));

    ImageInfo volume = Depth(PixelFormat::D32_FLOAT, 800, 793);
    volume.type = ImageType::e3D;
    REQUIRE_FALSE(VideoCommon::IsPaddedSquare(volume));
}
