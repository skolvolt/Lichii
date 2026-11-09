// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <thread>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "shader_recompiler/info_serialize.h"
#include "video_core/nx2_recipe.h"
#include "video_core/renderer_vulkan/vk_pipeline_cache.h"
#include "video_core/shader_environment.h"
#include "video_core/shader_warm/prepared_store.h"
#include "video_core/shader_warm/warm_records.h"

namespace {

using VideoCommon::ShaderWarm::PreparedStore;

constexpr std::string_view Identity = "game=1 translator=2 device=3 settings=4";

struct Scratch {
    std::filesystem::path path =
        std::filesystem::temp_directory_path() /
        ("lichii-preparation-test-" +
         std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    ~Scratch() {
        std::error_code error;
        std::filesystem::remove(path, error);
    }
};

std::vector<u8> Bytes(std::string_view text) {
    return {text.begin(), text.end()};
}

std::vector<u32> Ids(const PreparedStore& store) {
    std::vector<u32> ids;
    for (const auto& record : store.Records()) {
        ids.push_back(record.id);
    }
    std::ranges::sort(ids);
    return ids;
}

/// A prepared record as the pipeline cache writes one: blobs first, then the record naming them.
void Prepare(PreparedStore& store, u32 id, std::string_view info, std::string_view spirv) {
    const auto info_at = store.AddBlob(Bytes(info));
    const auto code_at = store.AddBlob(Bytes(spirv));
    REQUIRE(info_at);
    REQUIRE(code_at);
    std::vector<u8> payload(sizeof(*info_at) * 2);
    std::memcpy(payload.data(), &*info_at, sizeof(*info_at));
    std::memcpy(payload.data() + sizeof(*info_at), &*code_at, sizeof(*code_at));
    REQUIRE(store.AddRecord(id, payload));
}

void Corrupt(const std::filesystem::path& path, u64 offset) {
    std::fstream file(path, std::ios::binary | std::ios::in | std::ios::out);
    file.seekg(static_cast<std::streamoff>(offset));
    const char byte = static_cast<char>(file.get());
    file.seekp(static_cast<std::streamoff>(offset));
    file.put(static_cast<char>(byte ^ 0x5a));
}

} // Anonymous namespace

TEST_CASE("Native preparation resumes from its checkpoints and drops a torn tail",
          "[video_core][preparation]") {
    Scratch scratch;
    {
        PreparedStore store;
        REQUIRE(store.Open(scratch.path, std::string{Identity}));
        Prepare(store, 7, "info seven", "spirv seven");
        Prepare(store, 8, "info eight", "spirv eight");
    }
    const auto complete_size = std::filesystem::file_size(scratch.path);
    {
        // A crash in the middle of an entry: its header and part of its frame were written.
        std::ofstream file(scratch.path, std::ios::binary | std::ios::app);
        const u32 header[3]{1, 9, 4096};
        file.write(reinterpret_cast<const char*>(header), sizeof(header));
        file << "partial";
    }
    PreparedStore resumed;
    REQUIRE(resumed.Open(scratch.path, std::string{Identity}));
    REQUIRE(Ids(resumed) == std::vector<u32>{7, 8});
    REQUIRE(std::filesystem::file_size(scratch.path) == complete_size);

    // Blobs are shared by content, across sessions too.
    const auto before = std::filesystem::file_size(scratch.path);
    Prepare(resumed, 9, "info seven", "spirv eight");
    const auto record_only = std::filesystem::file_size(scratch.path) - before;
    REQUIRE(record_only < 64);
    REQUIRE(Ids(resumed) == std::vector<u32>{7, 8, 9});
}

TEST_CASE("Native preparation starts over for another game, translator, device or settings",
          "[video_core][preparation]") {
    Scratch scratch;
    {
        PreparedStore store;
        REQUIRE(store.Open(scratch.path, std::string{Identity}));
        Prepare(store, 1, "info", "spirv");
    }
    for (const char* identity :
         {"game=1 translator=DIRTY device=3 settings=4", "game=1 translator=2 device=3 settings=5",
          "game=UPDATED translator=2 device=3 settings=4"}) {
        PreparedStore other;
        REQUIRE(other.Open(scratch.path, identity));
        REQUIRE(other.Records().empty());
        Prepare(other, 1, "info", "spirv");
    }
    PreparedStore original;
    REQUIRE(original.Open(scratch.path, std::string{Identity}));
    REQUIRE(original.Records().empty());
    REQUIRE_FALSE(original.Open(scratch.path, ""));
}

TEST_CASE("Damaged native preparation entries are dropped and prepared again",
          "[video_core][preparation]") {
    Scratch scratch;
    PreparedStore::Location blob{};
    u64 first_record{};
    {
        PreparedStore store;
        REQUIRE(store.Open(scratch.path, std::string{Identity}));
        blob = *store.AddBlob(Bytes("spirv words"));
        REQUIRE(store.AddRecord(1, Bytes("record one")));
        first_record = store.Records()[0].at.offset;
        REQUIRE(store.AddRecord(2, Bytes("record two")));
        REQUIRE(store.AddRecord(3, Bytes("record three")));
    }

    SECTION("a damaged record in the middle loses only that record") {
        Corrupt(scratch.path, first_record + 9);
        PreparedStore store;
        REQUIRE(store.Open(scratch.path, std::string{Identity}));
        REQUIRE(Ids(store) == std::vector<u32>{2, 3});
        REQUIRE(store.AddRecord(1, Bytes("record one again")));
        REQUIRE(Ids(store) == std::vector<u32>{1, 2, 3});
    }
    SECTION("a damaged last entry is cut off") {
        Corrupt(scratch.path, std::filesystem::file_size(scratch.path) - 1);
        PreparedStore store;
        REQUIRE(store.Open(scratch.path, std::string{Identity}));
        REQUIRE(Ids(store) == std::vector<u32>{1, 2});
        REQUIRE(store.AddRecord(3, Bytes("record three")));
        PreparedStore reopened;
        REQUIRE(reopened.Open(scratch.path, std::string{Identity}));
        REQUIRE(Ids(reopened) == std::vector<u32>{1, 2, 3});
    }
    SECTION("a damaged blob is refused when read, so the draw translates instead") {
        std::ifstream file(scratch.path, std::ios::binary);
        REQUIRE(PreparedStore::ReadBlob(file, blob) == Bytes("spirv words"));
        file.close();
        Corrupt(scratch.path, blob.offset + blob.size - 1);
        std::ifstream damaged(scratch.path, std::ios::binary);
        REQUIRE(PreparedStore::ReadBlob(damaged, blob).empty());
        REQUIRE(PreparedStore::ReadBlob(damaged, {blob.offset, 0}).empty());
        REQUIRE(PreparedStore::ReadBlob(damaged, {~0ULL - 8, blob.size}).empty());
    }
    SECTION("a file of another kind is replaced") {
        std::ofstream(scratch.path, std::ios::binary | std::ios::trunc) << "NX2INFO3 garbage";
        PreparedStore store;
        REQUIRE(store.Open(scratch.path, std::string{Identity}));
        REQUIRE(store.Records().empty());
    }
}

TEST_CASE("Native preparation checkpoints from parallel workers survive a stop",
          "[video_core][preparation]") {
    Scratch scratch;
    constexpr u32 Workers = 8;
    constexpr u32 PerWorker = 64;
    std::atomic<bool> stop{};
    std::atomic<u32> completed{};
    {
        PreparedStore store;
        REQUIRE(store.Open(scratch.path, std::string{Identity}));
        std::vector<std::jthread> threads;
        for (u32 worker = 0; worker < Workers; ++worker) {
            threads.emplace_back([&, worker] {
                for (u32 i = 0; i < PerWorker && !stop; ++i) {
                    const u32 id = worker * PerWorker + i;
                    // Half the stages are shared between workers, as programs are.
                    const auto info = store.AddBlob(Bytes(std::to_string(id % 16)));
                    const auto code = store.AddBlob(Bytes("code " + std::to_string(id)));
                    if (info && code && store.AddRecord(id, Bytes("record"))) {
                        if (completed.fetch_add(1) + 1 == Workers * PerWorker / 2) {
                            stop = true; // the loading screen's Skip
                        }
                    }
                }
            });
        }
    }
    PreparedStore resumed;
    REQUIRE(resumed.Open(scratch.path, std::string{Identity}));
    REQUIRE(resumed.Records().size() == completed.load());
    REQUIRE(completed.load() < Workers * PerWorker);
}

TEST_CASE("Prepared shader infos roundtrip and refuse truncation and invalid values",
          "[video_core][preparation]") {
    Shader::Info info{};
    info.is_mesh = true;
    info.frag_color_components[2] = 11;
    info.storage_buffers_descriptors.emplace_back();
    info.storage_buffers_descriptors.back().cbuf_index = 3;
    info.storage_buffers_descriptors.back().write_extent.bounded = true;
    info.handle_programs = {{1, 2, 3}, {4, 5}};
    std::vector<u8> bytes;
    Shader::SerializeInfo(info, bytes);
    Shader::Info restored{};
    std::span<const u8> input(bytes);
    REQUIRE(Shader::DeserializeInfo(input, restored));
    REQUIRE(input.empty());
    std::vector<u8> roundtrip;
    Shader::SerializeInfo(restored, roundtrip);
    REQUIRE(bytes == roundtrip);
    for (size_t size : {size_t{0}, size_t{1}, bytes.size() / 2, bytes.size() - 1}) {
        input = std::span<const u8>(bytes).first(size);
        REQUIRE_FALSE(Shader::DeserializeInfo(input, restored));
    }
    bytes[0] = 2; // a bool that is neither 0 nor 1
    input = bytes;
    REQUIRE_FALSE(Shader::DeserializeInfo(input, restored));
}

TEST_CASE("Prepared translations are used only when the runtime state answers the same",
          "[video_core][preparation]") {
    namespace ShaderWarm = VideoCommon::ShaderWarm;
    Scratch scratch;
    const auto make = [&](u32 local_memory) {
        ShaderWarm::LoweredCode lowered;
        lowered.code.assign(16 * sizeof(u64), 0x11);
        const ShaderWarm::ComputeControl control{
            .workgroup = {64, 1, 1}, .shared_memory = 256, .local_memory = local_memory};
        ShaderWarm::ComputeKey key{};
        const auto entry = ShaderWarm::ComputeEntry(lowered, control, key);
        std::ofstream(scratch.path, std::ios::binary | std::ios::trunc)
            .write(reinterpret_cast<const char*>(entry.data()),
                   static_cast<std::streamsize>(entry.size()));
        std::ifstream in(scratch.path, std::ios::binary);
        in.exceptions(std::ifstream::failbit);
        u32 count{};
        in.read(reinterpret_cast<char*>(&count), sizeof(count));
        VideoCommon::FileEnvironment env;
        env.Deserialize(in, static_cast<std::streamoff>(entry.size()));
        return env;
    };
    auto recorded = make(1024);
    auto same = make(1024);
    auto other = make(2048);
    const char* why = nullptr;
    REQUIRE(recorded.AnswersHoldFor(same, &why));
    REQUIRE_FALSE(recorded.AnswersHoldFor(other, &why));
    REQUIRE(why != nullptr);
}

TEST_CASE("Recipes from elsewhere or damaged ones are refused", "[video_core][preparation]") {
    namespace Recipe = VideoCommon::Nx2Recipe;
    REQUIRE_FALSE(Recipe::ReadHeader({}));
    const auto foreign = Bytes("FRITCHII pipeline cache, not a Lichii recipe");
    REQUIRE_FALSE(Recipe::ReadHeader(foreign));
    // A recipe header whose body is not a zstd frame of the size it names.
    std::vector<u8> damaged = Bytes("NX2RCP01");
    damaged.resize(32, 0xff);
    REQUIRE_FALSE(Recipe::ReadHeader(damaged));
    Recipe::Report report;
    REQUIRE_THROWS(
        Recipe::Assemble(damaged, {}, VideoCommon::ShaderWarm::WarmCacheVersion, report, 1, {}));
}
