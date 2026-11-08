// SPDX-License-Identifier: GPL-3.0-or-later

#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <unordered_map>

#include <catch2/catch_test_macros.hpp>

#include "video_core/pipeline_cache_files.h"
#include "video_core/renderer_vulkan/vk_pipeline_cache.h"
#include "video_core/shader_environment.h"

namespace {

constexpr std::array<char, 8> Magic{'y', 'u', 'z', 'u', 'c', 'a', 'c', 'h'};
constexpr std::array<char, 8> OtherMagic{'y', 'u', 'z', 'u', 'v', 'k', 'c', 'h'};

struct TempDir {
    std::filesystem::path path;
    TempDir() {
        const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
        path = std::filesystem::temp_directory_path() /
               ("eden-pipeline-cache-files-" + std::to_string(stamp));
        std::filesystem::create_directories(path);
    }
    ~TempDir() {
        std::error_code error;
        std::filesystem::remove_all(path, error);
    }
};

void WriteCache(const std::filesystem::path& file, const std::array<char, 8>& magic, u32 version,
                std::string_view body) {
    std::ofstream stream(file, std::ios::binary);
    stream.write(magic.data(), magic.size());
    stream.write(reinterpret_cast<const char*>(&version), sizeof(version));
    stream.write(body.data(), static_cast<std::streamsize>(body.size()));
}

std::string ReadAll(const std::filesystem::path& file) {
    std::ifstream stream(file, std::ios::binary);
    return {std::istreambuf_iterator<char>(stream), std::istreambuf_iterator<char>()};
}

class SerializedEnvironment final : public VideoCommon::GenericEnvironment {
public:
    explicit SerializedEnvironment(Shader::Stage shader_stage) {
        stage = shader_stage;
        code = {0x123456789abcdef0, 0xfedcba9876543210};
        cached_lowest = 0;
        cached_highest = sizeof(u64);
        mesh_registers = {0x12345678, 0x87654321, 0xabcdef01};
        std::memset(&mesh_primitive_header, 0x5a, sizeof(mesh_primitive_header));
        texture_bound = 2;
        cbuf_values.emplace((u64{1} << 32) | 16, 0x1357);
        texture_types.emplace(3, Shader::TextureType::Color2D);
    }

    u32 ReadCbufValue(u32, u32) override {
        return 0;
    }
    Shader::TextureType ReadTextureType(u32) override {
        return Shader::TextureType::Color2D;
    }
    Shader::TexturePixelFormat ReadTexturePixelFormat(u32) override {
        return {};
    }
    bool IsTexturePixelFormatInteger(u32) override {
        return false;
    }
    u32 ReadViewportTransformState() override {
        return 1;
    }
    std::optional<Shader::ReplaceConstant> GetReplaceConstBuffer(u32, u32) override {
        return std::nullopt;
    }
};

} // Anonymous namespace

TEST_CASE("Task and mesh disk environments roundtrip all translation state and record boundaries",
          "[pipeline_cache_files][mesh_serialization]") {
    TempDir temp;
    const auto path = temp.path / "mesh.bin";
    constexpr u64 key = 0xabcdef;
    for (const auto stage : {Shader::Stage::Task, Shader::Stage::Mesh, Shader::Stage::Fragment}) {
        SerializedEnvironment source{stage};
        const std::array<const VideoCommon::GenericEnvironment*, 1> envs{&source};
        VideoCommon::SerializePipeline(key, envs, path, VideoCommon::VulkanPipelineCacheVersion);
    }
    size_t read = 0;
    VideoCommon::LoadPipelines(
        {}, path, VideoCommon::VulkanPipelineCacheVersion,
        [](std::ifstream&, VideoCommon::FileEnvironment) { FAIL("Unexpected compute record"); },
        [&](std::ifstream& file, std::vector<VideoCommon::FileEnvironment> envs) {
            const std::array stages{Shader::Stage::Task, Shader::Stage::Mesh,
                                    Shader::Stage::Fragment};
            REQUIRE(read < stages.size());
            REQUIRE(envs.size() == 1);
            auto& restored = envs.front();
            SerializedEnvironment source{stages[read++]};
            REQUIRE(restored.ShaderStage() == source.ShaderStage());
            REQUIRE(restored.ReadInstruction(0) == 0x123456789abcdef0);
            REQUIRE(restored.ReadInstruction(8) == 0xfedcba9876543210);
            REQUIRE(restored.ReadCbufValue(1, 16) == 0x1357);
            REQUIRE(restored.ReadTextureType(3) == Shader::TextureType::Color2D);
            REQUIRE(restored.TextureBoundBuffer() == 2);
            REQUIRE(restored.MemorySize() > sizeof(restored) + 2 * sizeof(u64));
            if (source.ShaderStage() != Shader::Stage::Fragment) {
                REQUIRE(restored.MeshRegisters() == source.MeshRegisters());
                REQUIRE(std::memcmp(&restored.MeshPrimitiveHeader(), &source.MeshPrimitiveHeader(),
                                    sizeof(Shader::ProgramHeader)) == 0);
            }
            u64 restored_key{};
            file.read(reinterpret_cast<char*>(&restored_key), sizeof(restored_key));
            REQUIRE(restored_key == key);
        });
    REQUIRE(read == 3);
}

TEST_CASE("Compatible v52 pipelines migrate without changing or discarding the old warm cache",
          "[pipeline_cache_files][mesh_serialization]") {
    TempDir temp;
    const auto old = VideoCommon::VersionedPipelineCachePath(temp.path, "vulkan", 52);
    const auto current = VideoCommon::VersionedPipelineCachePath(
        temp.path, "vulkan", VideoCommon::VulkanPipelineCacheVersion);
    SerializedEnvironment source{Shader::Stage::Fragment};
    const std::array<const VideoCommon::GenericEnvironment*, 1> envs{&source};
    constexpr u64 key = 42;
    VideoCommon::SerializePipeline(key, envs, old, 52);
    const auto before = ReadAll(old);
    REQUIRE(VideoCommon::CopyCompatiblePipelineCache(old, current, Magic, 52,
                                                     VideoCommon::VulkanPipelineCacheVersion));
    REQUIRE(ReadAll(old) == before);
    REQUIRE(ReadAll(current).substr(12) == before.substr(12));
    REQUIRE_FALSE(VideoCommon::CopyCompatiblePipelineCache(
        old, current, Magic, 52, VideoCommon::VulkanPipelineCacheVersion));
    bool loaded = false;
    VideoCommon::LoadPipelines(
        {}, current, VideoCommon::VulkanPipelineCacheVersion,
        [](std::ifstream&, VideoCommon::FileEnvironment) { FAIL("Unexpected compute record"); },
        [&](std::ifstream& file, std::vector<VideoCommon::FileEnvironment> restored) {
            REQUIRE(restored.size() == 1);
            REQUIRE(restored.front().ShaderStage() == Shader::Stage::Fragment);
            u64 restored_key{};
            file.read(reinterpret_cast<char*>(&restored_key), sizeof(restored_key));
            REQUIRE(restored_key == key);
            loaded = true;
        });
    REQUIRE(loaded);
    const auto other = temp.path / "wrong.bin";
    REQUIRE_FALSE(VideoCommon::CopyCompatiblePipelineCache(old, other, Magic, 51, 53));
    REQUIRE_FALSE(std::filesystem::exists(other));
}

TEST_CASE("Compatible pipeline cache migration retries after an interrupted temporary copy",
          "[pipeline_cache_files][mesh_serialization][migration_retry]") {
    TempDir temp;
    const auto old = VideoCommon::VersionedPipelineCachePath(temp.path, "vulkan", 52);
    const auto current = VideoCommon::VersionedPipelineCachePath(
        temp.path, "vulkan", VideoCommon::VulkanPipelineCacheVersion);
    auto temporary = current;
    temporary += ".migrating";
    WriteCache(old, Magic, 52, "complete pipeline records");
    const auto before = ReadAll(old);
    {
        std::ofstream stream(temporary, std::ios::binary);
        stream.write(before.data(), 4);
    }
    REQUIRE(ReadAll(temporary) == before.substr(0, 4));
    REQUIRE(VideoCommon::CopyCompatiblePipelineCache(old, current, Magic, 52,
                                                     VideoCommon::VulkanPipelineCacheVersion));
    REQUIRE(ReadAll(old) == before);
    REQUIRE(VideoCommon::PipelineCacheHeaderMatches(
        current, Magic, VideoCommon::VulkanPipelineCacheVersion));
    REQUIRE(ReadAll(current).substr(12) == before.substr(12));
    REQUIRE_FALSE(std::filesystem::exists(temporary));
    const auto migrated = ReadAll(current);
    REQUIRE_FALSE(VideoCommon::CopyCompatiblePipelineCache(
        old, current, Magic, 52, VideoCommon::VulkanPipelineCacheVersion));
    REQUIRE(ReadAll(current) == migrated);
    REQUIRE(ReadAll(old) == before);
}

TEST_CASE("A new cache version starts from the newest compatible cache on disk",
          "[pipeline_cache_files][carry_over]") {
    TempDir temp;
    const auto v52 = VideoCommon::VersionedPipelineCachePath(temp.path, "vulkan", 52);
    const auto v53 = VideoCommon::VersionedPipelineCachePath(temp.path, "vulkan", 53);
    const auto v54 = VideoCommon::VersionedPipelineCachePath(temp.path, "vulkan", 54);
    WriteCache(v52, Magic, 52, "older pipelines");
    WriteCache(v53, Magic, 53, "newer pipelines");
    const auto before_v52 = ReadAll(v52);
    const auto before_v53 = ReadAll(v53);
    constexpr std::array<u32, 2> versions{53, 52};
    const auto carried =
        VideoCommon::CarryOverPipelineCache(temp.path, "vulkan", Magic, versions, 54);
    REQUIRE(carried.version == 53u);
    REQUIRE_FALSE(carried.failed);
    REQUIRE(VideoCommon::PipelineCacheHeaderMatches(v54, Magic, 54));
    REQUIRE(ReadAll(v54).substr(12) == "newer pipelines");
    REQUIRE(ReadAll(v52) == before_v52);
    REQUIRE(ReadAll(v53) == before_v53);
    // A cache of its own version is never replaced, even when an older one changes later.
    WriteCache(v53, Magic, 53, "written by the older build since");
    const auto again =
        VideoCommon::CarryOverPipelineCache(temp.path, "vulkan", Magic, versions, 54);
    REQUIRE_FALSE(again.version);
    REQUIRE_FALSE(again.failed);
    REQUIRE(ReadAll(v54).substr(12) == "newer pipelines");
}

TEST_CASE("A new cache version skips an older file that isn't its cache",
          "[pipeline_cache_files][carry_over]") {
    TempDir temp;
    constexpr std::array<u32, 2> versions{53, 52};
    WriteCache(VideoCommon::VersionedPipelineCachePath(temp.path, "vulkan", 53), OtherMagic, 53,
               "another cache");
    WriteCache(VideoCommon::VersionedPipelineCachePath(temp.path, "vulkan", 52), Magic, 52,
               "older pipelines");
    REQUIRE(VideoCommon::CarryOverPipelineCache(temp.path, "vulkan", Magic, versions, 54).version ==
            52u);
    REQUIRE(ReadAll(VideoCommon::VersionedPipelineCachePath(temp.path, "vulkan", 54)).substr(12) ==
            "older pipelines");
}

TEST_CASE("A newer cache that can't be copied doesn't make way for an older one",
          "[pipeline_cache_files][carry_over]") {
    TempDir temp;
    constexpr std::array<u32, 2> versions{53, 52};
    const auto v54 = VideoCommon::VersionedPipelineCachePath(temp.path, "vulkan", 54);
    WriteCache(VideoCommon::VersionedPipelineCachePath(temp.path, "vulkan", 53), Magic, 53,
               "newer pipelines");
    WriteCache(VideoCommon::VersionedPipelineCachePath(temp.path, "vulkan", 52), Magic, 52,
               "older pipelines");
    // A directory where the copy goes makes it fail.
    auto temporary = v54;
    temporary += ".migrating";
    std::filesystem::create_directory(temporary);
    const auto carried =
        VideoCommon::CarryOverPipelineCache(temp.path, "vulkan", Magic, versions, 54);
    REQUIRE(carried.failed);
    REQUIRE_FALSE(carried.version);
    REQUIRE_FALSE(std::filesystem::exists(v54));
}

TEST_CASE("A new cache version without a compatible cache starts empty",
          "[pipeline_cache_files][carry_over]") {
    TempDir temp;
    constexpr std::array<u32, 2> versions{53, 52};
    WriteCache(VideoCommon::VersionedPipelineCachePath(temp.path, "vulkan", 51), Magic, 51,
               "another record layout");
    const auto carried =
        VideoCommon::CarryOverPipelineCache(temp.path, "vulkan", Magic, versions, 54);
    REQUIRE_FALSE(carried.version);
    REQUIRE_FALSE(carried.failed);
    REQUIRE_FALSE(
        std::filesystem::exists(VideoCommon::VersionedPipelineCachePath(temp.path, "vulkan", 54)));
}

TEST_CASE("The current Vulkan cache starts from the newest version it lists",
          "[pipeline_cache_files][carry_over]") {
    const std::span<const u32> versions{VideoCommon::CompatibleVulkanPipelineCacheVersions};
    if (versions.empty()) {
        return;
    }
    TempDir temp;
    WriteCache(VideoCommon::VersionedPipelineCachePath(temp.path, "vulkan", versions.front()),
               Magic, versions.front(), "pipelines");
    const auto carried =
        VideoCommon::CarryOverPipelineCache(temp.path, "vulkan", VideoCommon::PipelineCacheMagic,
                                            versions, VideoCommon::VulkanPipelineCacheVersion);
    REQUIRE(carried.version == versions.front());
    REQUIRE(VideoCommon::PipelineCacheHeaderMatches(
        VideoCommon::VersionedPipelineCachePath(temp.path, "vulkan",
                                                VideoCommon::VulkanPipelineCacheVersion),
        VideoCommon::PipelineCacheMagic, VideoCommon::VulkanPipelineCacheVersion));
}

TEST_CASE("A cleared cache stays empty while an older cache could replace it",
          "[pipeline_cache_files][carry_over]") {
    TempDir temp;
    constexpr std::array<u32, 2> versions{53, 52};
    const auto v53 = VideoCommon::VersionedPipelineCachePath(temp.path, "vulkan", 53);
    const auto v54 = VideoCommon::VersionedPipelineCachePath(temp.path, "vulkan", 54);
    WriteCache(v53, Magic, 53, "pipelines");
    const auto before_v53 = ReadAll(v53);
    REQUIRE(VideoCommon::CarryOverPipelineCache(temp.path, "vulkan", Magic, versions, 54).version);
    REQUIRE(VideoCommon::ClearPipelineCache(temp.path, "vulkan", Magic, versions, 54) ==
            VideoCommon::PipelineCacheClear::Cleared);
    REQUIRE(ReadAll(v54).size() == 12);
    REQUIRE(VideoCommon::PipelineCacheHeaderMatches(v54, Magic, 54));
    REQUIRE(ReadAll(v53) == before_v53);
    const auto carried =
        VideoCommon::CarryOverPipelineCache(temp.path, "vulkan", Magic, versions, 54);
    REQUIRE_FALSE(carried.version);
    REQUIRE_FALSE(carried.failed);
    REQUIRE(ReadAll(v54).size() == 12);
}

TEST_CASE("A cleared cache without an older one is removed", "[pipeline_cache_files][carry_over]") {
    TempDir temp;
    constexpr std::array<u32, 2> versions{53, 52};
    const auto v54 = VideoCommon::VersionedPipelineCachePath(temp.path, "vulkan", 54);
    REQUIRE(VideoCommon::ClearPipelineCache(temp.path, "vulkan", Magic, versions, 54) ==
            VideoCommon::PipelineCacheClear::Missing);
    WriteCache(v54, Magic, 54, "pipelines");
    REQUIRE(VideoCommon::ClearPipelineCache(temp.path, "vulkan", Magic, versions, 54) ==
            VideoCommon::PipelineCacheClear::Cleared);
    REQUIRE_FALSE(std::filesystem::exists(v54));
    REQUIRE(VideoCommon::ClearPipelineCache(temp.path, "vulkan", Magic, versions, 54) ==
            VideoCommon::PipelineCacheClear::Missing);
}

TEST_CASE("Truncated mesh state is rejected before a pipeline can be queued",
          "[pipeline_cache_files][mesh_serialization]") {
    TempDir temp;
    const auto path = temp.path / "truncated-mesh.bin";
    SerializedEnvironment source{Shader::Stage::Mesh};
    const std::array<const VideoCommon::GenericEnvironment*, 1> envs{&source};
    constexpr u64 key = 42;
    VideoCommon::SerializePipeline(key, envs, path, VideoCommon::VulkanPipelineCacheVersion);
    std::filesystem::resize_file(path, std::filesystem::file_size(path) - sizeof(key) - 1);
    VideoCommon::LoadPipelines(
        {}, path, VideoCommon::VulkanPipelineCacheVersion,
        [](std::ifstream&, VideoCommon::FileEnvironment) { FAIL("Unexpected compute record"); },
        [](std::ifstream&, std::vector<VideoCommon::FileEnvironment>) {
            FAIL("Incomplete mesh state reached pipeline creation");
        });
    REQUIRE_FALSE(std::filesystem::exists(path));
}

TEST_CASE("Cancelled mesh loading preserves its disk cache and queues no work",
          "[pipeline_cache_files][mesh_serialization]") {
    TempDir temp;
    const auto path = temp.path / "cancelled-mesh.bin";
    SerializedEnvironment source{Shader::Stage::Mesh};
    const std::array<const VideoCommon::GenericEnvironment*, 1> envs{&source};
    constexpr u64 key = 42;
    VideoCommon::SerializePipeline(key, envs, path, VideoCommon::VulkanPipelineCacheVersion);
    const auto before = ReadAll(path);
    std::stop_source stop;
    stop.request_stop();
    VideoCommon::LoadPipelines(
        stop.get_token(), path, VideoCommon::VulkanPipelineCacheVersion,
        [](std::ifstream&, VideoCommon::FileEnvironment) { FAIL("Unexpected compute record"); },
        [](std::ifstream&, std::vector<VideoCommon::FileEnvironment>) {
            FAIL("Cancelled load reached pipeline creation");
        });
    REQUIRE(ReadAll(path) == before);
}

TEST_CASE("Pipeline cache names are version-suffixed per build", "[pipeline_cache_files]") {
    const std::filesystem::path dir{"shader"};
    CHECK(VideoCommon::VersionedPipelineCachePath(dir, "vulkan", 39).filename() ==
          "vulkan.fork-v39.bin");
    CHECK(VideoCommon::VersionedPipelineCachePath(dir, "vulkan_pipelines", 39).filename() ==
          "vulkan_pipelines.fork-v39.bin");
    CHECK(VideoCommon::VersionedPipelineCachePath(dir, "vulkan", 18) !=
          VideoCommon::VersionedPipelineCachePath(dir, "vulkan", 39));
}

TEST_CASE("A legacy cache of this version is adopted with its contents",
          "[pipeline_cache_files]") {
    TempDir temp;
    const auto legacy = temp.path / "vulkan.bin";
    const auto versioned = VideoCommon::VersionedPipelineCachePath(temp.path, "vulkan", 39);
    WriteCache(legacy, Magic, 39, "pipelines");
    const std::string before = ReadAll(legacy);
    REQUIRE(VideoCommon::AdoptLegacyPipelineCache(legacy, versioned, Magic, 39));
    CHECK_FALSE(std::filesystem::exists(legacy));
    CHECK(ReadAll(versioned) == before);
    // Idempotent: nothing left to adopt.
    CHECK_FALSE(VideoCommon::AdoptLegacyPipelineCache(legacy, versioned, Magic, 39));
}

TEST_CASE("Another build's legacy cache is never touched", "[pipeline_cache_files]") {
    TempDir temp;
    const auto legacy = temp.path / "vulkan.bin";
    const auto versioned = VideoCommon::VersionedPipelineCachePath(temp.path, "vulkan", 39);
    WriteCache(legacy, Magic, 18, "official");
    const std::string before = ReadAll(legacy);
    CHECK_FALSE(VideoCommon::AdoptLegacyPipelineCache(legacy, versioned, Magic, 39));
    CHECK(ReadAll(legacy) == before);
    CHECK_FALSE(std::filesystem::exists(versioned));

    WriteCache(legacy, OtherMagic, 39, "wrong magic");
    CHECK_FALSE(VideoCommon::AdoptLegacyPipelineCache(legacy, versioned, Magic, 39));
    CHECK(std::filesystem::exists(legacy));

    // Truncated header.
    {
        std::ofstream stream(legacy, std::ios::binary | std::ios::trunc);
        stream.write(Magic.data(), 4);
    }
    CHECK_FALSE(VideoCommon::AdoptLegacyPipelineCache(legacy, versioned, Magic, 39));
    CHECK(std::filesystem::exists(legacy));
}

TEST_CASE("An existing versioned cache is never overwritten by a legacy one",
          "[pipeline_cache_files]") {
    TempDir temp;
    const auto legacy = temp.path / "vulkan.bin";
    const auto versioned = VideoCommon::VersionedPipelineCachePath(temp.path, "vulkan", 39);
    WriteCache(legacy, Magic, 39, "older");
    WriteCache(versioned, Magic, 39, "current");
    const std::string current = ReadAll(versioned);
    CHECK_FALSE(VideoCommon::AdoptLegacyPipelineCache(legacy, versioned, Magic, 39));
    CHECK(ReadAll(versioned) == current);
    CHECK(std::filesystem::exists(legacy));
    CHECK_FALSE(VideoCommon::AdoptLegacyPipelineCache(temp.path / "missing.bin", versioned,
                                                      Magic, 39));
}

TEST_CASE("Cached compute failures retain their cause and the current dispatch address",
          "[pipeline_cache][diagnostics]") {
    const Vulkan::ComputePipelineCacheKey key{
        .unique_hash = 0x1234,
        .shared_memory_size = 1024,
        .workgroup_size = {32, 1, 1},
    };
    std::unordered_map<Vulkan::ComputePipelineCacheKey, Vulkan::ComputePipelineCacheEntry> cache;
    const std::string cause =
        "SPIR-V emission failed at IR opcode UndefU8: SPIR-V Instruction is not implemented";
    cache[key].failure = cause;
    for (const GPUVAddr address : {0x211ddd9000ULL, 0x21ebdfa000ULL}) {
        const auto [entry, inserted] = cache.try_emplace(key);
        REQUIRE_FALSE(inserted);
        CHECK(entry->second.Get(false, address, key) == nullptr);
        try {
            static_cast<void>(entry->second.Get(true, address, key));
            FAIL("Native compute failure was not propagated");
        } catch (const Shader::RuntimeError& exception) {
            const std::string message{exception.what()};
            CHECK(message.find(cause) != std::string::npos);
            CHECK(message.find(fmt::format("address={:#x}", address)) != std::string::npos);
            CHECK(message.find(fmt::format("shader={:#016x}", key.unique_hash)) != std::string::npos);
            CHECK(message.find(fmt::format("pipeline={:#016x}", key.Hash())) != std::string::npos);
        }
        CHECK(entry->second.failure == cause);
        CHECK_FALSE(entry->second.pipeline);
    }
    auto other = key;
    other.workgroup_size[0] = 64;
    const auto [entry, inserted] = cache.try_emplace(other);
    REQUIRE(inserted);
    CHECK(entry->second.failure.empty());
    CHECK(entry->second.Get(false, 0x2120633c00, other) == nullptr);
    REQUIRE_THROWS_AS(entry->second.Get(true, 0x2120633c00, other), Shader::RuntimeError);
}
