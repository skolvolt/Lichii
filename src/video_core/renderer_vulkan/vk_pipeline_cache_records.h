// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <filesystem>
#include <span>
#include <unordered_set>
#include <utility>
#include <vector>

#include "common/common_types.h"
#include "video_core/shader_environment.h"

namespace Vulkan {

/// A compute translation read from the disk cache, and where it is in the file
struct CachedComputeTranslation {
    VideoCommon::FileEnvironment env;
    VideoCommon::PipelineRecord record;
};

/// Adds the translations of a key that no load chooses on a host with these `read_properties`, once
/// `chosen` is the one that translated (the count when none did): those tried before it that lack a
/// recorded value (see FileEnvironment::LacksRecordedValue), and copies of an earlier record that
/// stays, which come after it in the order and fail wherever it fails. The first in the file stays
/// unless the next one kept binds the same pointers, as the order of the others is relative to it.
/// Others stay, those binding fewer pointers than the chosen one too: a device that can't build
/// that one (it binds more storage buffers than the device does) takes the next that builds.
void AddUnchosenComputeTranslations(std::span<const CachedComputeTranslation> candidates,
                                    size_t chosen, u32 read_properties,
                                    std::vector<VideoCommon::PipelineRecord>& incomplete,
                                    std::vector<VideoCommon::PipelineRecord>& copies);

/// A graphics record after one with the same key hash and stages, and where that one's key is
struct DuplicateGraphicsRecord {
    VideoCommon::PipelineRecord record;
    u64 first_key;
};

/// Adds the duplicates whose key equals the first one's: the load builds only the first record of a
/// key. This reads the file without the cache's lock, which is fine: RemovePipelines removes a
/// duplicate only while it is still where the load read it, and as records are only ever appended,
/// cut off or removed, the first one before it is then unchanged as well.
void AddEqualGraphicsKeys(const std::filesystem::path& filename,
                          std::span<const DuplicateGraphicsRecord> duplicates,
                          std::vector<VideoCommon::PipelineRecord>& records);

/// Adds the graphics records lacking a value the translator reads (with their key hashes) to
/// `removed`. Such a record never builds, and while its key is in `disk_pipelines` the pipeline
/// built when first needed is not recorded. Without a later record of its key, the key leaves
/// `disk_pipelines` so that the pipeline is recorded; a later record is no longer a duplicate, as
/// it takes the record's place once that is removed, so the key is recorded once however often
/// removing it fails.
void AddIncompleteGraphics(std::span<const std::pair<u64, VideoCommon::PipelineRecord>> incomplete,
                           std::vector<DuplicateGraphicsRecord>& duplicates,
                           std::unordered_set<u64>& disk_pipelines,
                           std::vector<VideoCommon::PipelineRecord>& removed);

} // namespace Vulkan
