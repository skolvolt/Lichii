// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <fstream>
#include <numeric>

#include "video_core/renderer_vulkan/vk_graphics_pipeline.h"
#include "video_core/renderer_vulkan/vk_pipeline_cache_records.h"

namespace Vulkan {

void AddUnchosenComputeTranslations(std::span<const CachedComputeTranslation> candidates,
                                    size_t chosen, u32 read_properties,
                                    std::vector<VideoCommon::PipelineRecord>& incomplete,
                                    std::vector<VideoCommon::PipelineRecord>& copies) {
    if (candidates.empty()) {
        return;
    }
    std::vector<bool> removed(candidates.size());
    for (size_t index = 0; index < chosen && index < candidates.size(); ++index) {
        removed[index] = candidates[index].env.LacksRecordedValue(read_properties);
    }
    std::vector<size_t> file_order(candidates.size());
    std::iota(file_order.begin(), file_order.end(), size_t{0});
    std::ranges::sort(file_order, {}, [&](size_t index) { return candidates[index].record.begin; });
    const size_t first{file_order.front()};
    const auto next{
        std::ranges::find(file_order, false, [&](size_t index) -> bool { return removed[index]; })};
    if (removed[first] && next != file_order.end() &&
        candidates[*next].env.MappedStoragePointerKeys() !=
            candidates[first].env.MappedStoragePointerKeys()) {
        removed[first] = false;
    }
    for (size_t index = 0; index < candidates.size(); ++index) {
        if (removed[index]) {
            incomplete.push_back(candidates[index].record);
        }
    }
    // A copy has the same bytes as an earlier record kept, so they bind the same pointers and the
    // order tries the earlier one first. RemovePipelines checks the bytes again.
    for (auto later = file_order.begin(); later != file_order.end(); ++later) {
        const VideoCommon::PipelineRecord& record{candidates[*later].record};
        if (removed[*later] || *later == chosen) {
            continue;
        }
        const auto original{std::find_if(file_order.begin(), later, [&](size_t index) {
            const VideoCommon::PipelineRecord& kept{candidates[index].record};
            return !removed[index] && kept.hash == record.hash &&
                   kept.end - kept.begin == record.end - record.begin;
        })};
        if (original != later) {
            removed[*later] = true;
            VideoCommon::PipelineRecord removed_copy{record};
            removed_copy.copy_of = candidates[*original].record.begin;
            copies.push_back(removed_copy);
        }
    }
}

void AddEqualGraphicsKeys(const std::filesystem::path& filename,
                          std::span<const DuplicateGraphicsRecord> duplicates,
                          std::vector<VideoCommon::PipelineRecord>& records) {
    std::ifstream file{filename, std::ios::binary};
    const auto read_key{[&file](u64 offset) {
        GraphicsPipelineCacheKey key{};
        file.seekg(static_cast<std::streamoff>(offset));
        file.read(reinterpret_cast<char*>(&key), sizeof(key));
        return key;
    }};
    for (const auto& [record, first_key] : duplicates) {
        const GraphicsPipelineCacheKey key{read_key(record.end - sizeof(GraphicsPipelineCacheKey))};
        if (key == read_key(first_key) && file) {
            records.push_back(record);
        }
    }
}

void AddIncompleteGraphics(std::span<const std::pair<u64, VideoCommon::PipelineRecord>> incomplete,
                           std::vector<DuplicateGraphicsRecord>& duplicates,
                           std::unordered_set<u64>& disk_pipelines,
                           std::vector<VideoCommon::PipelineRecord>& removed) {
    for (const auto& [hash, record] : incomplete) {
        const size_t later_records{
            std::erase_if(duplicates, [&](const DuplicateGraphicsRecord& duplicate) {
                return duplicate.first_key == record.end - sizeof(GraphicsPipelineCacheKey);
            })};
        if (later_records == 0) {
            disk_pipelines.erase(hash);
        }
        removed.push_back(record);
    }
}

} // namespace Vulkan
