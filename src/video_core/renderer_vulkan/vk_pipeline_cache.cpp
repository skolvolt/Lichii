// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2019 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <atomic>
#include <bit>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <fstream>
#include <functional>
#include <iostream>
#include <memory>
#include <mutex>
#include <numeric>
#include <optional>
#include <span>
#include <thread>
#include <unordered_set>
#include <vector>
#include <zstd.h>
#include "common/cityhash.h"
#include "common/fs/fs.h"
#include "common/fs/path_util.h"
#include "common/scm_rev.h"
#include "common/scope_exit.h"
#include "common/thread.h"
#include "common/thread_worker.h"
#include "common/zstd_compression.h"
#include "core/core.h"
#include "native_preparation_revision.h"
#include "shader_recompiler/backend/spirv/emit_spirv.h"
#include "shader_recompiler/environment.h"
#include "shader_recompiler/exception.h"
#include "shader_recompiler/frontend/maxwell/control_flow.h"
#include "shader_recompiler/frontend/maxwell/translate_program.h"
#include "shader_recompiler/info_serialize.h"
#include "shader_recompiler/ir_opt/passes.h"
#include "shader_recompiler/program_header.h"
#include "video_core/dirty_flags.h"
#include "video_core/engines/kepler_compute.h"
#include "video_core/engines/maxwell_3d.h"
#include "video_core/gpu_logging/gpu_logging.h"
#include "video_core/memory_manager.h"
#include "video_core/nx2_bindmemo_stats.h"
#include "video_core/nx2_perf.h"
#include "video_core/pipeline_cache_files.h"
#include "video_core/renderer_vulkan/async_draw_policy.h"
#include "video_core/renderer_vulkan/fixed_pipeline_state.h"
#include "video_core/renderer_vulkan/maxwell_to_vk.h"
#include "video_core/renderer_vulkan/mesh_fallback_limits.h"
#include "video_core/renderer_vulkan/pipeline_helper.h"
#include "video_core/renderer_vulkan/pipeline_shader_emission.h"
#include "video_core/renderer_vulkan/pipeline_statistics.h"
#include "video_core/renderer_vulkan/vertex_repack.h"
#include "video_core/renderer_vulkan/vk_compute_pipeline.h"
#include "video_core/renderer_vulkan/vk_cooperative_hmma_test.h"
#include "video_core/renderer_vulkan/vk_descriptor_pool.h"
#include "video_core/renderer_vulkan/vk_pipeline_cache.h"
#include "video_core/renderer_vulkan/vk_pipeline_cache_records.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/renderer_vulkan/vk_shader_util.h"
#include "video_core/renderer_vulkan/vk_update_descriptor.h"
#include "video_core/shader_cache.h"
#include "video_core/shader_environment.h"
#include "video_core/shader_notify.h"
#include "video_core/shader_warm/prepared_store.h"
#include "video_core/shader_warm/shader_warm_builder.h"
#include "video_core/surface.h"
#include "video_core/vulkan_common/vulkan_device.h"
#include "video_core/vulkan_common/vulkan_wrapper.h"

#ifdef __ANDROID__
#include "../../android/app/src/main/jni/android_settings.h"
#endif

namespace Vulkan {

namespace {
using Shader::Backend::SPIRV::EmitSPIRV;
using Shader::Maxwell::ConvertLegacyToGeneric;
using Shader::Maxwell::GenerateGeometryPassthrough;
using Shader::Maxwell::MergeDualVertexPrograms;
using Shader::Maxwell::TranslateProgram;
using VideoCommon::ComputeEnvironment;
using VideoCommon::FileEnvironment;
using VideoCommon::GenericEnvironment;
using VideoCommon::GuessMissing;
using VideoCommon::GraphicsEnvironment;

constexpr u32 CACHE_VERSION = VideoCommon::VulkanPipelineCacheVersion;
constexpr size_t VULKAN_CACHE_FLUSH_PIPELINES = 128;
constexpr size_t VULKAN_CACHE_FLUSH_MIN_SECONDS = 30;
constexpr auto& VULKAN_CACHE_MAGIC_NUMBER = VideoCommon::VulkanDriverPipelineCacheMagic;

template <typename Container>
auto MakeSpan(Container& container) {
    return std::span(container.data(), container.size());
}

Shader::OutputTopology MaxwellToOutputTopology(Maxwell::PrimitiveTopology topology) {
    switch (topology) {
    case Maxwell::PrimitiveTopology::Points:
        return Shader::OutputTopology::PointList;
    case Maxwell::PrimitiveTopology::LineStrip:
        return Shader::OutputTopology::LineStrip;
    default:
        return Shader::OutputTopology::TriangleStrip;
    }
}

Shader::CompareFunction MaxwellToCompareFunction(Maxwell::ComparisonOp comparison) {
    switch (comparison) {
    case Maxwell::ComparisonOp::Never_D3D:
    case Maxwell::ComparisonOp::Never_GL:
        return Shader::CompareFunction::Never;
    case Maxwell::ComparisonOp::Less_D3D:
    case Maxwell::ComparisonOp::Less_GL:
        return Shader::CompareFunction::Less;
    case Maxwell::ComparisonOp::Equal_D3D:
    case Maxwell::ComparisonOp::Equal_GL:
        return Shader::CompareFunction::Equal;
    case Maxwell::ComparisonOp::LessEqual_D3D:
    case Maxwell::ComparisonOp::LessEqual_GL:
        return Shader::CompareFunction::LessThanEqual;
    case Maxwell::ComparisonOp::Greater_D3D:
    case Maxwell::ComparisonOp::Greater_GL:
        return Shader::CompareFunction::Greater;
    case Maxwell::ComparisonOp::NotEqual_D3D:
    case Maxwell::ComparisonOp::NotEqual_GL:
        return Shader::CompareFunction::NotEqual;
    case Maxwell::ComparisonOp::GreaterEqual_D3D:
    case Maxwell::ComparisonOp::GreaterEqual_GL:
        return Shader::CompareFunction::GreaterThanEqual;
    case Maxwell::ComparisonOp::Always_D3D:
    case Maxwell::ComparisonOp::Always_GL:
        return Shader::CompareFunction::Always;
    }
    UNIMPLEMENTED_MSG("Unimplemented comparison op={}", comparison);
    return {};
}

Shader::AttributeType CastAttributeType(const FixedPipelineState::VertexAttribute& attr) {
    if (attr.enabled == 0) {
        return Shader::AttributeType::Disabled;
    }
    switch (attr.Type()) {
    case Maxwell::VertexAttribute::Type::UnusedEnumDoNotUseBecauseItWillGoAway:
        ASSERT_MSG(false, "Invalid vertex attribute type!");
        return Shader::AttributeType::Disabled;
    case Maxwell::VertexAttribute::Type::SNorm:
    case Maxwell::VertexAttribute::Type::UNorm:
    case Maxwell::VertexAttribute::Type::Float:
        return Shader::AttributeType::Float;
    case Maxwell::VertexAttribute::Type::SInt:
        return Shader::AttributeType::SignedInt;
    case Maxwell::VertexAttribute::Type::UInt:
        return Shader::AttributeType::UnsignedInt;
    case Maxwell::VertexAttribute::Type::UScaled:
        return Shader::AttributeType::UnsignedScaled;
    case Maxwell::VertexAttribute::Type::SScaled:
        return Shader::AttributeType::SignedScaled;
    }
    return Shader::AttributeType::Float;
}

Shader::AttributeType AttributeType(const FixedPipelineState& state, size_t index) {
    switch (state.DynamicAttributeType(index)) {
    case 0:
        return Shader::AttributeType::Disabled;
    case 1:
        return Shader::AttributeType::Float;
    case 2:
        return Shader::AttributeType::SignedInt;
    case 3:
        return Shader::AttributeType::UnsignedInt;
    }
    return Shader::AttributeType::Disabled;
}

Shader::RuntimeInfo MakeRuntimeInfo(std::span<const Shader::IR::Program> programs,
                                    const GraphicsPipelineCacheKey& key,
                                    const Shader::IR::Program& program,
                                    const Shader::IR::Program* previous_program,
                                    const Vulkan::Device& device) {
    Shader::RuntimeInfo info;
    if (previous_program) {
        info.previous_stage_stores = previous_program->info.stores;
        info.previous_stage_legacy_stores_mapping = previous_program->info.legacy_stores_mapping;
        if (previous_program->is_geometry_passthrough) {
            info.previous_stage_stores.mask |= previous_program->info.passthrough.mask;
        }
    } else {
        info.previous_stage_stores.mask.set();
    }
    if (previous_program && previous_program->stage == Shader::Stage::Mesh) {
        for (const u32 line : previous_program->mesh.primitive_lines) {
            const auto attribute{static_cast<Shader::IR::Attribute>(line)};
            if (Shader::IR::IsGeneric(attribute)) {
                info.per_primitive_generics.set(Shader::IR::GenericAttributeIndex(attribute));
            }
        }
    }
    const Shader::Stage stage{program.stage};
    const bool has_geometry{key.unique_hashes[4] != 0 && !programs[4].is_geometry_passthrough};
    const bool gl_ndc{key.state.ndc_minus_one_to_one != 0};
    const float point_size{std::bit_cast<float>(key.state.point_size)};
    switch (stage) {
    case Shader::Stage::VertexB:
        if (!has_geometry) {
            if (key.state.topology == Maxwell::PrimitiveTopology::Points) {
                info.fixed_state_point_size = point_size;
            }
            if (key.state.xfb_enabled) {
                if (device.IsExtTransformFeedbackSupported()) {
                    auto [varyings, count] =
                        VideoCommon::MakeTransformFeedbackVaryings(key.state.xfb_state);
                    info.xfb_varyings = varyings;
                    info.xfb_count = count;
                } else {
                    LOG_WARNING(Render_Vulkan, "XFB requested in pipeline key but device lacks VK_EXT_transform_feedback; ignoring XFB decorations");
                }
            }
            info.convert_depth_mode = gl_ndc;
        }
        if (key.state.dynamic_vertex_input) {
            for (size_t index = 0; index < Maxwell::NumVertexAttributes; ++index) {
                info.generic_input_types[index] = AttributeType(key.state, index);
            }
        } else {
            std::ranges::transform(key.state.attributes, info.generic_input_types.begin(),
                                   &CastAttributeType);
        }
        break;
    case Shader::Stage::TessellationEval:
        info.tess_clockwise = key.state.tessellation_clockwise != 0;
        info.tess_primitive = [&key] {
            const u32 raw{key.state.tessellation_primitive.Value()};
            switch (static_cast<Maxwell::Tessellation::DomainType>(raw)) {
            case Maxwell::Tessellation::DomainType::Isolines:
                return Shader::TessPrimitive::Isolines;
            case Maxwell::Tessellation::DomainType::Triangles:
                return Shader::TessPrimitive::Triangles;
            case Maxwell::Tessellation::DomainType::Quads:
                return Shader::TessPrimitive::Quads;
            }
            ASSERT(false);
            return Shader::TessPrimitive::Triangles;
        }();
        info.tess_spacing = [&] {
            const u32 raw{key.state.tessellation_spacing};
            switch (static_cast<Maxwell::Tessellation::Spacing>(raw)) {
            case Maxwell::Tessellation::Spacing::Integer:
                return Shader::TessSpacing::Equal;
            case Maxwell::Tessellation::Spacing::FractionalOdd:
                return Shader::TessSpacing::FractionalOdd;
            case Maxwell::Tessellation::Spacing::FractionalEven:
                return Shader::TessSpacing::FractionalEven;
            }
            ASSERT(false);
            return Shader::TessSpacing::Equal;
        }();
        break;
    case Shader::Stage::Mesh:
        info.convert_depth_mode = gl_ndc;
        if (program.mesh.topology == 0) {
            info.fixed_state_point_size = point_size;
        }
        break;
    case Shader::Stage::Geometry:
        if (program.output_topology == Shader::OutputTopology::PointList) {
            info.fixed_state_point_size = point_size;
        }
        if (key.state.xfb_enabled != 0) {
            if (device.IsExtTransformFeedbackSupported()) {
                auto [varyings, count] =
                    VideoCommon::MakeTransformFeedbackVaryings(key.state.xfb_state);
                info.xfb_varyings = varyings;
                info.xfb_count = count;
            } else {
                LOG_WARNING(Render_Vulkan, "XFB requested in pipeline key but device lacks VK_EXT_transform_feedback; ignoring XFB decorations");
            }
        }
        info.convert_depth_mode = gl_ndc;
        break;
    case Shader::Stage::Fragment: {
        info.alpha_test_func = MaxwellToCompareFunction(
            key.state.UnpackComparisonOp(key.state.alpha_test_func.Value()));
        info.alpha_test_reference = std::bit_cast<float>(key.state.alpha_test_ref);

        info.dual_source_blend = key.state.attachment0_dual_source_blend != 0;

        if (device.IsMoltenVK()) {
            for (size_t i = 0; i < 8; ++i) {
                const auto format = static_cast<Tegra::RenderTargetFormat>(key.state.color_formats[i]);
                if (format == Tegra::RenderTargetFormat::NONE) {
                    info.color_output_types[i] = Shader::AttributeType::Float;
                    continue;
                }
                const auto pixel_format = VideoCore::Surface::PixelFormatFromRenderTargetFormat(format);
                if (VideoCore::Surface::IsPixelFormatInteger(pixel_format)) {
                    if (VideoCore::Surface::IsPixelFormatSignedInteger(pixel_format)) {
                        info.color_output_types[i] = Shader::AttributeType::SignedInt;
                    } else {
                        info.color_output_types[i] = Shader::AttributeType::UnsignedInt;
                    }
                } else {
                    info.color_output_types[i] = Shader::AttributeType::Float;
                }
            }
        }
        break;
    }
    default:
        break;
    }
    switch (key.state.topology) {
    case Maxwell::PrimitiveTopology::Points:
        info.input_topology = Shader::InputTopology::Points;
        break;
    case Maxwell::PrimitiveTopology::Lines:
    case Maxwell::PrimitiveTopology::LineLoop:
    case Maxwell::PrimitiveTopology::LineStrip:
        info.input_topology = Shader::InputTopology::Lines;
        break;
    case Maxwell::PrimitiveTopology::Triangles:
    case Maxwell::PrimitiveTopology::TriangleStrip:
    case Maxwell::PrimitiveTopology::TriangleFan:
    case Maxwell::PrimitiveTopology::Quads:
    case Maxwell::PrimitiveTopology::QuadStrip:
    case Maxwell::PrimitiveTopology::Polygon:
    case Maxwell::PrimitiveTopology::Patches:
        info.input_topology = Shader::InputTopology::Triangles;
        break;
    case Maxwell::PrimitiveTopology::LinesAdjacency:
    case Maxwell::PrimitiveTopology::LineStripAdjacency:
        info.input_topology = Shader::InputTopology::LinesAdjacency;
        break;
    case Maxwell::PrimitiveTopology::TrianglesAdjacency:
    case Maxwell::PrimitiveTopology::TriangleStripAdjacency:
        info.input_topology = Shader::InputTopology::TrianglesAdjacency;
        break;
    }
    info.force_early_z = key.state.early_z != 0;
    info.y_negate = key.state.y_negate != 0;
    // Transform feedback would capture the scaled values, a passthrough geometry shader doesn't
    // scale the outputs it writes itself, and indexed attribute reads in the fragment stage don't
    // undo the scaling, so all three keep native noperspective interpolation.
    const bool geometry_passthrough{key.unique_hashes[4] != 0 &&
                                    programs[4].is_geometry_passthrough};
    if (key.unique_hashes[5] != 0 && !geometry_passthrough && key.state.xfb_enabled == 0 &&
        !programs[5].info.loads_indexed_attributes) {
        const bool has_tess{key.unique_hashes[3] != 0};
        const bool last_pre_raster{
            (stage == Shader::Stage::VertexB && !has_tess && !has_geometry) ||
            (stage == Shader::Stage::TessellationEval && !has_geometry) ||
            (stage == Shader::Stage::Geometry && has_geometry)};
        if (stage == Shader::Stage::Fragment) {
            info.scaled_noperspective_inputs =
                previous_program && (previous_program->stage == Shader::Stage::VertexB ||
                                     previous_program->stage == Shader::Stage::TessellationEval ||
                                     previous_program->stage == Shader::Stage::Geometry);
        }
        if (last_pre_raster) {
            const auto& fragment_info{programs[5].info};
            for (size_t index = 0; index < 32; ++index) {
                if (fragment_info.interpolation[index] == Shader::Interpolation::NoPerspective) {
                    info.noperspective_outputs.set(index);
                }
            }
        }
    }
    return info;
}

std::array<std::atomic<std::chrono::nanoseconds::rep>, 2> runtime_spirv_ns{};

/// Runtime pipelines built from precompiled infos, and those whose recorded answers differed
/// (by the first answer that did), since the last 10 s report.
std::atomic<u32> precompiled_drawn{};
std::atomic<u32> precompiled_failed{};
std::atomic<u32> precompiled_created{};
std::atomic<u32> precompiled_from_spirv{};
std::atomic<std::chrono::nanoseconds::rep> precompiled_gpu_ns{};
std::mutex precompiled_reasons_mutex;
std::map<std::string, u32> precompiled_reasons;

void CountPrecompiledReason(const char* reason) {
    std::scoped_lock lock{precompiled_reasons_mutex};
    ++precompiled_reasons[reason];
}

/// EDEN_NX2_PRECOMPILED_FAST=0: precompiled pipelines are translated on the GPU thread like the
/// others (their kept infos and SPIR-V unused), to compare.
bool PrecompiledFastPath() {
    static const bool enabled = [] {
        const char* value = std::getenv("EDEN_NX2_PRECOMPILED_FAST");
        const bool on = !(value && value[0] == '0');
        if (!on) {
            LOG_INFO(Render_Vulkan, "EDEN_NX2_PRECOMPILED_FAST=0: precompiled pipelines are "
                                    "translated at runtime");
        }
        return on;
    }();
    return enabled;
}

bool PrecompiledAudit() {
    static const bool enabled = [] {
        const char* value = std::getenv("EDEN_NX2_PRECOMPILED_AUDIT");
        return value != nullptr && value[0] == '1';
    }();
    return enabled;
}
std::atomic<u32> audit_same{};
std::atomic<u32> audit_differ{};

/// The words of a kept SPIR-V blob, on the GPU thread.
std::vector<u32> ReadKeptSpirv(std::ifstream& file, std::pair<u64, u32> blob) {
    const auto compressed =
        VideoCommon::ShaderWarm::PreparedStore::ReadBlob(file, {blob.first, blob.second});
    const auto size = ZSTD_getFrameContentSize(compressed.data(), compressed.size());
    if (size < 20 || size > VideoCommon::ShaderWarm::PreparedStore::MaxPayloadBytes ||
        size % 4 != 0) {
        throw Shader::LogicError("Invalid prepared SPIR-V size");
    }
    std::vector<u32> words(size / sizeof(u32));
    if (ZSTD_decompress(words.data(), size, compressed.data(), compressed.size()) != size ||
        words[0] != 0x07230203 || words[4] != 0) {
        throw Shader::LogicError("Invalid prepared SPIR-V");
    }
    return words;
}

vk::ShaderModule BuildKeptSpirv(const Device& device, const std::filesystem::path& path,
                                std::pair<u64, u32> blob, u64 pipeline_hash) {
    std::ifstream file(path, std::ios::binary);
    return BuildShader(device, ReadKeptSpirv(file, blob));
}
constexpr u32 NoInfo = ~0U;

/// nup: reads `files` (directories: every file in them) start to end at low priority. A prepared
/// pipeline reads a few small pieces of the ~2.5 GB of prepared files (and the driver reads its
/// own cache entry); on a hard disk each such read cost ~10-20 ms (first visits: 24 ms a pipeline,
/// 3 s for one area's 122) unless the file is already in the OS file cache. One sequential pass
/// puts it there. ponytail: relies on the OS keeping it cached; a RAM copy if that proves too weak.
void PrereadFiles(std::stop_token stop, std::vector<std::filesystem::path> paths) {
    Common::SetCurrentThreadName("ShaderPreread");
    Common::SetCurrentThreadPriority(Common::ThreadPriority::Low);
    std::vector<std::filesystem::path> files;
    for (const auto& path : paths) {
        std::error_code error;
        if (std::filesystem::is_directory(path, error)) {
            for (const auto& entry : std::filesystem::recursive_directory_iterator(
                     path, std::filesystem::directory_options::skip_permission_denied, error)) {
                if (entry.is_regular_file(error)) {
                    files.push_back(entry.path());
                }
            }
        } else if (std::filesystem::is_regular_file(path, error)) {
            files.push_back(path);
        }
    }
    const auto begin = std::chrono::steady_clock::now();
    std::vector<char> buffer(size_t{4} << 20);
    u64 total = 0;
    for (const auto& file_path : files) {
        std::ifstream file(file_path, std::ios::binary);
        while (file && !stop.stop_requested()) {
            file.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
            total += static_cast<u64>(file.gcount());
        }
        if (stop.stop_requested()) {
            return;
        }
    }
    LOG_INFO(Render_Vulkan, "Read ahead {} MiB of prepared shader files ({} files) in {:.1f} s",
             total >> 20, files.size(),
             std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count());
}

void AddRuntimeSpirvTime(size_t thread, std::chrono::steady_clock::time_point begin) {
    const auto elapsed = std::chrono::steady_clock::now() - begin;
    runtime_spirv_ns[thread].fetch_add(
        std::chrono::duration_cast<std::chrono::nanoseconds>(elapsed).count(),
        std::memory_order::relaxed);
}

vk::ShaderModule EmitShaderModule(const Device& device, const Shader::Profile& profile,
                                  const Shader::RuntimeInfo& runtime_info,
                                  Shader::IR::Program& program, Shader::Backend::Bindings& binding,
                                  u64 shader_hash, size_t stage_index,
                                  std::span<const u32> reference = {},
                                  std::vector<u32>* code_out = nullptr) {
    const auto code =
        EmitPipelineShader(profile, runtime_info, program, binding, shader_hash, reference);
    if (code_out) {
        *code_out = code;
    }
    device.SaveShader(code);
    vk::ShaderModule module{BuildShader(device, code)};

    // Text log + .spv dump. Text log is gated by gpu_log_level != Off; .spv dump
    // is independent and gated only by gpu_log_shader_dumps.
    const bool should_log = GPU::Logging::IsActive();
    const bool should_dump = Settings::values.gpu_log_shader_dumps.GetValue();
    if (should_log || should_dump) {
        static constexpr std::array stage_names{"vertex", "tess_control", "tess_eval", "geometry",
                                               "fragment"};
        const std::string shader_name =
            fmt::format("shader_{:016x}_{}", shader_hash, stage_names[stage_index]);
        if (should_log) {
            const std::string shader_info = fmt::format(
                "SPIR-V size: {} bytes, hash: {:016x}", code.size() * sizeof(u32), shader_hash);
            GPU::Logging::GPULogger::GetInstance().LogShaderCompilation(shader_name, shader_info);
        }
        if (should_dump) {
            GPU::Logging::DumpSpirvShader(shader_hash,
                                         std::span<const u32>(code.data(), code.size()));
        }
    }

    if (device.HasDebuggingToolAttached()) {
        const std::string name{fmt::format("Shader {:016x}", shader_hash)};
        module.SetObjectNameEXT(name.c_str());
    }
    return module;
}

struct DeferredStages {
    std::unique_ptr<EmissionBudget::Reservation> reservation;
    std::shared_ptr<ShaderPools> pools;
    std::array<Shader::IR::Program, Maxwell::MaxShaderProgram> programs;
    std::array<std::optional<Shader::RuntimeInfo>, Maxwell::MaxShaderStage> runtime_infos;
    std::array<std::vector<u32>, Maxwell::MaxShaderStage> reference_spirv;
};

size_t GetTotalPipelineWorkers() {
    const size_t max_core_threads =
        std::max<size_t>(static_cast<size_t>(std::thread::hardware_concurrency()), 2ULL) - 1ULL;
#ifdef __ANDROID__
    const int configured = AndroidSettings::values.pipeline_worker_count.GetValue();
    const int clamped = std::clamp(configured, 2, 8);
    const size_t desired = static_cast<size_t>(clamped);
    if (desired == 0) {
        return 1ULL;
    }
    return std::min(max_core_threads, desired);
#else
    return max_core_threads;
#endif
}

} // Anonymous namespace

struct PipelineCache::GraphicsStages {
    std::array<Shader::IR::Program, Maxwell::MaxShaderProgram> programs;
    /// Per pipeline stage (vertex B to fragment), for the stages the pipeline has.
    std::array<std::optional<Shader::RuntimeInfo>, Maxwell::MaxShaderStage> runtime_infos;
    std::array<std::vector<u32>, Maxwell::MaxShaderStage> reference_spirv;
    std::optional<MeshEmulationSetup> mesh_emulation;
};

size_t ComputePipelineCacheKey::Hash() const noexcept {
    const u64 hash = Common::CityHash64(reinterpret_cast<const char*>(this), sizeof *this);
    return static_cast<size_t>(hash);
}

bool ComputePipelineCacheKey::operator==(const ComputePipelineCacheKey& rhs) const noexcept {
    return std::memcmp(&rhs, this, sizeof *this) == 0;
}

size_t GraphicsPipelineCacheKey::Hash() const noexcept {
    const u64 hash = Common::CityHash64(reinterpret_cast<const char*>(this), Size());
    return static_cast<size_t>(hash);
}

bool GraphicsPipelineCacheKey::operator==(const GraphicsPipelineCacheKey& rhs) const noexcept {
    if (unique_hashes != rhs.unique_hashes) {
        return false;
    }
    return std::memcmp(reinterpret_cast<const char*>(this) + sizeof(unique_hashes),
                       reinterpret_cast<const char*>(&rhs) + sizeof(unique_hashes),
                       state.Size()) == 0;
}

namespace {
// NVIDIA's and AMD's drivers keep a disk cache of built pipelines of their own. Eden's copy on top
// only kept every pipeline the title ever built in host memory (title 1: a 245 MB file, ~0.35 GB
// of RAM for the session) and copied it whole at each save, which stalled frames; pipelines built
// no slower without it.
bool DriverCachesPipelines(const Device& device) {
    switch (device.GetDriverID()) {
    case VK_DRIVER_ID_NVIDIA_PROPRIETARY:
    case VK_DRIVER_ID_AMD_PROPRIETARY:
        return true;
    default:
        return false;
    }
}
} // Anonymous namespace

PipelineCache::PipelineCache(Tegra::MaxwellDeviceMemoryManager& device_memory_,
                             const Device& device_, Scheduler& scheduler_,
                             DescriptorPool& descriptor_pool_,
                             GuestDescriptorQueue& guest_descriptor_queue_,
                             DescriptorBufferRing& descriptor_buffer_ring_,
                             RenderPassCache& render_pass_cache_, BufferCache& buffer_cache_,
                             TextureCache& texture_cache_, VideoCore::ShaderNotify& shader_notify_)
    : VideoCommon::ShaderCache{device_memory_}, device{device_}, scheduler{scheduler_},
      descriptor_pool{descriptor_pool_}, guest_descriptor_queue{guest_descriptor_queue_},
      descriptor_buffer_ring{descriptor_buffer_ring_}, render_pass_cache{render_pass_cache_},
      buffer_cache{buffer_cache_}, texture_cache{texture_cache_}, shader_notify{shader_notify_},
      use_asynchronous_shaders{Settings::values.use_asynchronous_shaders.GetValue()},
      use_vulkan_pipeline_cache{Settings::values.use_vulkan_driver_pipeline_cache.GetValue() &&
                                !DriverCachesPipelines(device_)},
      num_pipeline_workers{device.HasBrokenParallelShaderCompiling() ? 1ULL
                                                                     : GetTotalPipelineWorkers()},
      emission_budget{num_pipeline_workers},
      workers(num_pipeline_workers, "VkPipelineBuilder", {}, Common::ThreadPlacement::Background),
      serialization_thread(1, "VkPipelineSerialization", {}, Common::ThreadPlacement::Background),
      background_pipelines{64, 16 * 1024 * 1024, device.HasBrokenParallelShaderCompiling()} {
    const auto& float_control{device.FloatControlProperties()};
    const VkDriverId driver_id{device.GetDriverID()};
    const VkShaderStageFlags subgroup_stages{device.GetSubgroupSupportedStages()};
    const auto subgroup_stage_bit{[subgroup_stages](VkShaderStageFlags flag, Shader::Stage stage) {
        return (subgroup_stages & flag) != 0 ? (1u << static_cast<u32>(stage)) : 0u;
    }};
    u32 supported_subgroup_stages{
        subgroup_stage_bit(VK_SHADER_STAGE_VERTEX_BIT, Shader::Stage::VertexA) |
        subgroup_stage_bit(VK_SHADER_STAGE_VERTEX_BIT, Shader::Stage::VertexB) |
        subgroup_stage_bit(VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT,
                           Shader::Stage::TessellationControl) |
        subgroup_stage_bit(VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT,
                           Shader::Stage::TessellationEval) |
        subgroup_stage_bit(VK_SHADER_STAGE_GEOMETRY_BIT, Shader::Stage::Geometry) |
        subgroup_stage_bit(VK_SHADER_STAGE_FRAGMENT_BIT, Shader::Stage::Fragment) |
        subgroup_stage_bit(VK_SHADER_STAGE_COMPUTE_BIT, Shader::Stage::Compute) |
        subgroup_stage_bit(VK_SHADER_STAGE_TASK_BIT_EXT, Shader::Stage::Task) |
        subgroup_stage_bit(VK_SHADER_STAGE_MESH_BIT_EXT, Shader::Stage::Mesh)};
    if (driver_id == VK_DRIVER_ID_MESA_TURNIP) {
        supported_subgroup_stages &= ~(1u << static_cast<u32>(Shader::Stage::Geometry));
    }
    profile = Shader::Profile{
        .supported_spirv = device.SupportedSpirvVersion(),
        .unified_descriptor_binding = true,
        .support_descriptor_aliasing = device.IsDescriptorAliasingSupported(),
        .support_int8 = device.IsInt8Supported(),
        .support_uniform_and_storage_buffer_8bit =
            device.IsUniformAndStorageBuffer8BitAccessSupported(),
        .support_storage_buffer_8bit = device.IsStorageBuffer8BitAccessSupported(),
        .support_int16 = device.IsShaderInt16Supported(),
        .support_uniform_and_storage_buffer_16bit =
            device.IsUniformAndStorageBuffer16BitAccessSupported(),
        .support_storage_buffer_16bit = device.IsStorageBuffer16BitAccessSupported(),
        .support_int64 = device.IsShaderInt64Supported(),
        .support_vertex_instance_id = false,
        .support_float_controls = device.IsKhrShaderFloatControlsSupported(),
        .support_separate_denorm_behavior =
            float_control.denormBehaviorIndependence == VK_SHADER_FLOAT_CONTROLS_INDEPENDENCE_ALL,
        .support_separate_rounding_mode =
            float_control.roundingModeIndependence == VK_SHADER_FLOAT_CONTROLS_INDEPENDENCE_ALL,
        .support_fp16_denorm_preserve = float_control.shaderDenormPreserveFloat16 != VK_FALSE,
        .support_fp32_denorm_preserve = float_control.shaderDenormPreserveFloat32 != VK_FALSE,
        .support_fp16_denorm_flush = float_control.shaderDenormFlushToZeroFloat16 != VK_FALSE,
        .support_fp32_denorm_flush = float_control.shaderDenormFlushToZeroFloat32 != VK_FALSE,
        .support_fp16_signed_zero_nan_preserve =
            float_control.shaderSignedZeroInfNanPreserveFloat16 != VK_FALSE,
        .support_fp32_signed_zero_nan_preserve =
            float_control.shaderSignedZeroInfNanPreserveFloat32 != VK_FALSE,
        .support_fp64_signed_zero_nan_preserve =
            float_control.shaderSignedZeroInfNanPreserveFloat64 != VK_FALSE,
        .support_explicit_workgroup_layout = device.IsKhrWorkgroupMemoryExplicitLayoutSupported(),
        .support_workgroup_layout_8bit_access =
            device.IsWorkgroupMemoryExplicitLayout8BitAccessSupported(),
        .support_workgroup_layout_16bit_access =
            device.IsWorkgroupMemoryExplicitLayout16BitAccessSupported(),
        .support_shader_quad_control = device.IsKhrShaderQuadControlSupported(),
        .support_quad_shuffles = device.IsSubgroupFeatureSupported(VK_SUBGROUP_FEATURE_QUAD_BIT),
        .support_vote = device.IsSubgroupFeatureSupported(VK_SUBGROUP_FEATURE_VOTE_BIT),
        .support_shuffle_relative =
            device.IsSubgroupFeatureSupported(VK_SUBGROUP_FEATURE_SHUFFLE_RELATIVE_BIT),
        .supported_subgroup_stages = supported_subgroup_stages,
        .support_viewport_index_layer_non_geometry =
            device.IsExtShaderViewportIndexLayerSupported(),
        .support_viewport_mask = device.IsNvViewportArray2Supported(),
        .support_typeless_image_loads = device.IsFormatlessImageLoadSupported(),
        .support_demote_to_helper_invocation =
            device.IsExtShaderDemoteToHelperInvocationSupported(),
        .support_int64_atomics = device.IsExtShaderAtomicInt64Supported(),
        .support_shared_int64_atomics = device.IsSharedInt64AtomicsSupported(),
        .support_derivative_control = true,
        .support_geometry_shader_passthrough = device.IsNvGeometryShaderPassthroughSupported(),
        .support_native_ndc = device.IsExtDepthClipControlSupported(),
        .support_scaled_attributes = !device.MustEmulateScaledFormats(),
        .support_multi_viewport = device.SupportsMultiViewport(),
        .support_geometry_streams = device.AreTransformFeedbackGeometryStreamsSupported(),
        .support_sampled_image_array_nonuniform_indexing =
            device.IsSampledImageArrayNonUniformIndexingSupported(),
        .support_storage_image_array_nonuniform_indexing =
            device.IsStorageImageArrayNonUniformIndexingSupported(),
        .support_uniform_texel_buffer_array_nonuniform_indexing =
            device.IsUniformTexelBufferArrayNonUniformIndexingSupported(),
        .support_storage_texel_buffer_array_nonuniform_indexing =
            device.IsStorageTexelBufferArrayNonUniformIndexingSupported(),

        .warp_size_potentially_larger_than_guest = device.IsWarpSizePotentiallyBiggerThanGuest(),

        .lower_left_origin_mode = false,
        .need_declared_frag_colors = false,
        .need_gather_subpixel_offset = driver_id == VK_DRIVER_ID_AMD_PROPRIETARY ||
                                       driver_id == VK_DRIVER_ID_AMD_OPEN_SOURCE ||
                                       driver_id == VK_DRIVER_ID_MESA_RADV ||
                                       driver_id == VK_DRIVER_ID_INTEL_PROPRIETARY_WINDOWS ||
                                       driver_id == VK_DRIVER_ID_INTEL_OPEN_SOURCE_MESA ||
                                       driver_id == VK_DRIVER_ID_QUALCOMM_PROPRIETARY ||
                                       driver_id == VK_DRIVER_ID_MESA_TURNIP,

        .has_broken_spirv_clamp = driver_id == VK_DRIVER_ID_INTEL_PROPRIETARY_WINDOWS,
        .has_broken_spirv_position_input = driver_id == VK_DRIVER_ID_QUALCOMM_PROPRIETARY,
        .has_broken_unsigned_image_offsets = false,
        .has_broken_signed_operations = false,
        .has_broken_fp16_float_controls = driver_id == VK_DRIVER_ID_NVIDIA_PROPRIETARY,
        .max_compute_shared_memory_size = device.GetMaxComputeSharedMemorySize(),
        .has_broken_fp32_denorm_flush = driver_id == VK_DRIVER_ID_QUALCOMM_PROPRIETARY,
        .ignore_nan_fp_comparisons = false,
        .has_broken_spirv_subgroup_mask_vector_extract_dynamic =
            driver_id == VK_DRIVER_ID_QUALCOMM_PROPRIETARY &&
            device.GetDriverVersion() < VK_MAKE_VERSION(512, 672, 0),
        .has_broken_spirv_subgroup_shuffle = driver_id == VK_DRIVER_ID_QUALCOMM_PROPRIETARY,
        .has_broken_spirv_fragment_shuffle = driver_id == VK_DRIVER_ID_AMD_PROPRIETARY,
        .max_subgroup_size = device.GetMaxSubgroupSize(),
        .has_broken_robust =
            device.IsNvidia() && device.GetNvidiaArch() <= NvidiaArchitecture::Arch_Pascal,
        .min_ssbo_alignment = device.GetStorageBufferAlignment(),
        .max_user_clip_distances = device.GetMaxUserClipDistances()
    };
    profile.support_cooperative_hmma =
        device.IsCooperativeHmmaSupported() && TestCooperativeHmma(device);
    profile.support_cooperative_hmma_16x16x16 =
        !profile.support_cooperative_hmma && device.IsCooperativeHmma16x16x16Supported();
    if (profile.support_cooperative_hmma_16x16x16) {
        LOG_INFO(Render_Vulkan, "Guest tensor operations run on 16x16x16 cooperative matrices");
    }

    host_info = Shader::HostTranslateInfo{
        .min_ssbo_alignment = device.GetStorageBufferAlignment(),
        .max_per_stage_descriptor_sampled_images = device.GetMaxPerStageDescriptorSampledImages(),
        .max_per_stage_resources = device.GetMaxPerStageResources(),
        .max_per_stage_descriptor_storage_buffers = device.GetMaxPerStageDescriptorStorageBuffers(),
        .max_descriptor_set_samplers = device.GetMaxDescriptorSetSamplers(),
        .max_descriptor_set_uniform_buffers = device.GetMaxDescriptorSetUniformBuffers(),
        .max_descriptor_set_uniform_buffers_dynamic = device.GetMaxDescriptorSetUniformBuffersDynamic(),
        .max_descriptor_set_storage_buffers = device.GetMaxDescriptorSetStorageBuffers(),
        .max_descriptor_set_storage_buffers_dynamic = device.GetMaxDescriptorSetStorageBuffersDynamic(),
        .max_descriptor_set_sampled_images = device.GetMaxDescriptorSetSampledImages(),
        .max_descriptor_set_storage_images = device.GetMaxDescriptorSetStorageImages(),
        .max_descriptor_set_input_attachements = device.GetMaxDescriptorSetInputAttachments(),
        .support_float64 = device.IsFloat64Supported(),
        .support_float16 = device.IsFloat16Supported(),
        .support_int64 = device.IsShaderInt64Supported(),
        .needs_demote_reorder = driver_id == VK_DRIVER_ID_AMD_PROPRIETARY ||
                                driver_id == VK_DRIVER_ID_AMD_OPEN_SOURCE ||
                                driver_id == VK_DRIVER_ID_SAMSUNG_PROPRIETARY,
        .support_snorm_render_buffer = true,
        .support_viewport_index_layer = device.IsExtShaderViewportIndexLayerSupported(),
        .support_viewport_mask = device.IsNvViewportArray2Supported(),
        .support_geometry_shader_passthrough = device.IsNvGeometryShaderPassthroughSupported(),
        .support_conditional_barrier = device.SupportsConditionalBarriers(),
        .single_lane_geometry_subgroups = !profile.SupportsSubgroupStage(Shader::Stage::Geometry),
        .support_byte_addressed_r16 = true,
    };
    host_info.ApplyDescriptorLimitPolicy();

    if (device.GetMaxVertexInputAttributes() < Maxwell::NumVertexAttributes) {
        LOG_WARNING(Render_Vulkan, "maxVertexInputAttributes is too low: {} < {}",
                    device.GetMaxVertexInputAttributes(), Maxwell::NumVertexAttributes);
    }
    if (device.GetMaxVertexInputBindings() < Maxwell::NumVertexArrays) {
        LOG_WARNING(Render_Vulkan, "maxVertexInputBindings is too low: {} < {}",
                    device.GetMaxVertexInputBindings(), Maxwell::NumVertexArrays);
    }

    LOG_INFO(Render_Vulkan, "DynamicState setting value: {}", u32(Settings::values.dyna_state.GetValue()));

    dynamic_features = {};
    dynamic_features.driver_id = device.GetDriverID();
    dynamic_features.driver_version = device.GetDriverVersion();

    // User granularity enforced in vulkan_device.cpp switch statement:
    //   Level 0: Core Dynamic States only
    //   Level 1: Core + EDS1
    //   Level 2: Core + EDS1 + EDS2 (accumulative)
    //   Level 3: Core + EDS1 + EDS2 + EDS3 (accumulative)
    // Here we only verify if extensions were successfully loaded by the device

    dynamic_features.has_extended_dynamic_state =
        device.IsExtExtendedDynamicStateSupported();

    dynamic_features.has_extended_dynamic_state_2 =
        device.IsExtExtendedDynamicState2Supported();
    dynamic_features.has_extended_dynamic_state_2_logic_op =
        device.IsExtExtendedDynamicState2ExtrasSupported();
    dynamic_features.has_extended_dynamic_state_2_patch_control_points = false;

    dynamic_features.has_extended_dynamic_state_3_blend =
        device.IsExtExtendedDynamicState3BlendingSupported();
    dynamic_features.has_extended_dynamic_state_3_enables =
        device.IsExtExtendedDynamicState3EnablesSupported();
    dynamic_features.has_color_write_enable =
        device.IsExtColorWriteEnableSupported();
    dynamic_features.has_dynamic_state3_depth_clamp_enable =
        dynamic_features.has_extended_dynamic_state_3_enables &&
        device.SupportsDynamicState3DepthClampEnable();
    dynamic_features.has_dynamic_state3_logic_op_enable =
        dynamic_features.has_extended_dynamic_state_3_enables &&
        device.SupportsDynamicState3LogicOpEnable();
    dynamic_features.has_dynamic_state3_line_stipple_enable =
        dynamic_features.has_extended_dynamic_state_3_enables &&
        device.SupportsDynamicState3LineStippleEnable();

    // VIDS: Independent toggle (not affected by dyna_state levels)
    dynamic_features.has_dynamic_vertex_input =
        device.IsExtVertexInputDynamicStateSupported() &&
        Settings::values.vertex_input_dynamic_state.GetValue();
    dynamic_features.has_static_vertex_repack =
        StaticVertexRepackEnabled(device.GetMinVertexInputBindingStrideAlignment());

    dynamic_features.has_provoking_vertex = device.IsExtProvokingVertexSupported();
    dynamic_features.has_provoking_vertex_first_mode =
        device.SupportsProvokingVertexFirstMode();
    dynamic_features.has_provoking_vertex_last_mode =
        device.SupportsProvokingVertexLastMode();
    dynamic_features.has_provoking_vertex_tf_preserve =
        device.SupportsTransformFeedbackProvokingVertexPreservation();
}

namespace {
// <title>/vulkan_used.bin: the graphics pipelines recent sessions used, as key hash + sessions
// since the last use.
constexpr u32 UsedPipelinesMagic = 0x31505545; // "EUP1"
// Pipelines the last two sessions used are built at boot. Each holds ~0.45 MB of driver memory:
// title 1 built 4,800 from eight sessions, 1.5 GB more than building them on first use (but that
// stalled frames 4x as often), while one session of its title screen uses ~1,500.
// MoltenVK keeps the previous eight sessions: a pipeline built while playing goes through
// SPIR-V -> MSL -> Metal, far more than the 2-3 ms the shorter window assumes, so each one
// built at boot is a stutter avoided.
// Ocarina still built pipelines for 1-2 s of every 10 s early in play with two sessions kept:
// keep sixteen (#161).
constexpr u8 UsedPipelinesMaxAge = 16;
constexpr u8 UsedPipelinesMaxAgeMoltenVK = 8;

u8 MaxUsedPipelineAge(const Device& device) {
    if (device.IsMoltenVK()) {
        return UsedPipelinesMaxAgeMoltenVK;
    }
    return UsedPipelinesMaxAge;
}

std::unordered_map<u64, u8> LoadUsedPipelines(const std::filesystem::path& path, u8 max_age) {
    std::unordered_map<u64, u8> used;
    std::ifstream file{path, std::ios::binary};
    u32 magic{}, count{};
    if (!file.read(reinterpret_cast<char*>(&magic), sizeof(magic)) || magic != UsedPipelinesMagic ||
        !file.read(reinterpret_cast<char*>(&count), sizeof(count))) {
        return used;
    }
    used.reserve(count);
    for (u32 i = 0; i < count; ++i) {
        u64 hash{};
        u8 age{};
        if (!file.read(reinterpret_cast<char*>(&hash), sizeof(hash)) ||
            !file.read(reinterpret_cast<char*>(&age), sizeof(age))) {
            break;
        }
        if (age < max_age) { // lists saved with a longer window
            used.emplace(hash, age);
        }
    }
    return used;
}

// Without a list of the pipelines recent sessions used (the first boot with this cache, or a session
// that ended before saving it), this many cached graphics pipelines at most are built at boot and the
// rest when first needed. A cache that grew over many builds held 48k pipelines, ~21 GB of driver
// memory built before the game started; the list is only saved at exit or after two minutes, so a
// boot that ran out of memory never saved one and the next boot did the same.
constexpr size_t MaxPipelinesWithoutUsedList = 3000;

// Captured environments the shader precache holds for skipped graphics pipelines; the rest wait
// for a later precache.
constexpr size_t PrecacheCaptureLimit = size_t{1} << 30;

// EDEN_NX2_PRELOAD_ALL_PIPELINES=1 builds every cached pipeline at boot, as before.
bool PreloadAllPipelines() {
    static const bool all = [] {
        const char* value = std::getenv("EDEN_NX2_PRELOAD_ALL_PIPELINES");
        return value != nullptr && value[0] == '1';
    }();
    return all;
}
} // Anonymous namespace

PipelineCache::~PipelineCache() {
    serialization_stop.request_stop();
    background_pipelines.Stop();
    workers.WaitForRequests();
    serialization_thread.WaitForRequests();
    SaveUsedPipelines();
    if (use_vulkan_pipeline_cache && !vulkan_pipeline_cache_filename.empty()) {
        SerializeVulkanPipelineCache(vulkan_pipeline_cache_filename, vulkan_pipeline_cache,
                                     CACHE_VERSION);
    }
    graphics_cache.clear();
    evicted_pipelines.clear();
}

void PipelineCache::SaveUsedPipelines() {
    used_pipelines_saved = std::chrono::steady_clock::now();
    // A session too short to say what the game uses (a boot to the menu) keeps the old list.
    constexpr size_t MinimumUsed = 64;
    if (used_pipelines_filename.empty() ||
        used_this_session.size() + used_and_evicted.size() < MinimumUsed) {
        return;
    }
    std::unordered_map<u64, u8> next;
    next.reserve(used_pipelines.size() + used_this_session.size());
    for (const auto& [key, pipeline] : graphics_cache) {
        if (used_this_session.contains(pipeline.get())) {
            next.emplace(key.Hash(), u8{0});
        }
    }
    for (const u64 hash : used_and_evicted) {
        next.emplace(hash, u8{0});
    }
    for (const auto& [hash, age] : used_pipelines) {
        if (age + 1 < MaxUsedPipelineAge(device)) {
            next.try_emplace(hash, static_cast<u8>(age + 1));
        }
    }
    std::ofstream file{used_pipelines_filename, std::ios::binary | std::ios::trunc};
    const u32 count = static_cast<u32>(next.size());
    file.write(reinterpret_cast<const char*>(&UsedPipelinesMagic), sizeof(UsedPipelinesMagic));
    file.write(reinterpret_cast<const char*>(&count), sizeof(count));
    for (const auto& [hash, age] : next) {
        file.write(reinterpret_cast<const char*>(&hash), sizeof(hash));
        file.write(reinterpret_cast<const char*>(&age), sizeof(age));
    }
    // This session's first uses, then the earlier order for the pipelines it didn't reach.
    std::ofstream order{order_filename, std::ios::binary | std::ios::trunc};
    const std::unordered_set<u64> seen(first_use_order.begin(), first_use_order.end());
    for (const u64 hash : first_use_order) {
        order.write(reinterpret_cast<const char*>(&hash), sizeof(hash));
    }
    for (const u64 hash : last_order) {
        if (!seen.contains(hash)) {
            order.write(reinterpret_cast<const char*>(&hash), sizeof(hash));
        }
    }
    LOG_INFO(Render_Vulkan, "Follow-on prebuild: {} queued, {} taken prebuilt, {} built on demand",
             follow_on_queued, follow_on_taken, built_on_demand);
}

void PipelineCache::FollowOn(u64 hash) {
    first_use_order.push_back(hash);
    // EDEN_NX2_FOLLOW_ON=N: how far ahead of a first use to prebuild (0 = off).
    static const size_t ahead = [] {
        const char* value = std::getenv("EDEN_NX2_FOLLOW_ON");
        return value ? static_cast<size_t>(std::max(std::atoi(value), 0)) : size_t{32};
    }();
    const auto it = last_order_index.find(hash);
    if (ahead == 0 || it == last_order_index.end()) {
        return;
    }
    const size_t end = std::min(last_order.size(), it->second + 1 + ahead);
    for (size_t i = it->second + 1; i < end; ++i) {
        auto node = skipped_records.extract(last_order[i]);
        if (node.empty()) {
            continue;
        }
        const auto [key, begin] = node.mapped();
        if (background_pipelines.Add(
                key, 0,
                [this, key, begin, path = pipeline_cache_filename]() -> std::unique_ptr<GraphicsPipeline> {
                    std::ifstream file{path, std::ios::binary};
                    file.exceptions(std::ifstream::failbit | std::ifstream::badbit);
                    file.seekg(0, std::ios::end);
                    const std::streampos file_end = file.tellg();
                    file.seekg(static_cast<std::streamoff>(begin));
                    u32 num_envs{};
                    file.read(reinterpret_cast<char*>(&num_envs), sizeof(num_envs));
                    if (num_envs == 0 || num_envs > Maxwell::MaxShaderProgram) {
                        return {};
                    }
                    std::vector<FileEnvironment> envs(num_envs);
                    boost::container::static_vector<Shader::Environment*, Maxwell::MaxShaderProgram>
                        env_ptrs;
                    for (auto& env : envs) {
                        env.Deserialize(file, file_end);
                        env_ptrs.push_back(&env);
                    }
                    // A rewrite after the load moves records: build only the record still there.
                    GraphicsPipelineCacheKey stored;
                    file.read(reinterpret_cast<char*>(&stored), sizeof(stored));
                    if (stored.unique_hashes != key.unique_hashes) {
                        return {};
                    }
                    ShaderPools pools;
                    auto pipeline = CreateGraphicsPipeline(pools, key, MakeSpan(env_ptrs), nullptr, false);
                    if (pipeline && pipeline->Failed()) {
                        return {};
                    }
                    return pipeline;
                },
                false)) {
            ++follow_on_queued;
        }
    }
}

GraphicsPipeline* PipelineCache::CurrentGraphicsPipeline() {
    namespace Memo = VideoCommon::Nx2BindMemo;
    using Stat = Memo::Stat;
    Memo::Note(Stat::PipeCalls);
    bool shaders_dirty = maxwell3d->dirty.flags[VideoCommon::Dirty::Shaders];
    // EDEN_NX2_PIPE_SHADER_SAME: the shader registers were written; the stages may hash the same.
    const bool compare_shaders =
        shaders_dirty && key_pipeline != nullptr && Memo::PipeShaderSameEnabled();
    decltype(graphics_key.unique_hashes) previous_hashes{};
    if (compare_shaders) {
        previous_hashes = graphics_key.unique_hashes;
    }
    if (!RefreshStages(graphics_key.unique_hashes)) {
        current_pipeline = nullptr;
        key_pipeline = nullptr;
        return nullptr;
    }
    // nup: #717's lookup skip runs on the raw state; on native (Switch 2) pipelines #1021 keys the
    // pipeline by a canonicalized copy of it (CanonicalizeNative), so precompiled keys match.
    const bool native = maxwell3d->GetNativeMacroEngine().native_turing;
    FixedPipelineState& raw_state = native ? native_state : graphics_key.state;
    const bool shaders_same = compare_shaders && previous_hashes == graphics_key.unique_hashes;
    if (shaders_same) {
        shaders_dirty = false;
    }
    // EDEN_NX2_PIPE_SKIP: when nothing Refresh reads changed, the state part of the key stands;
    // with the same shaders too, so does the pipeline the last lookup found for it.
    const bool skip = PipelineSkipEnabled();
    bool state_same = false;
    // EDEN_NX2_PIPE_PATCH: blending or attribute types alone moved: redo only those parts.
    PipelineStateWatch::Patch patch{};
    if (skip) {
        if (raw_state.xfb_enabled != 0) {
            Memo::Note(Stat::PipeMissXfb);
        } else {
            state_same = state_watch.StateUnchanged(*maxwell3d, raw_state,
                                                    Memo::PipePatchEnabled() ? &patch : nullptr);
        }
    }
    if (state_same && patch.Any()) {
        // The key's state changed in those parts only; Next() finds the pipeline for it.
        PipelineStateWatch::ApplyPatch(*maxwell3d, raw_state, patch);
    } else if (state_same && !shaders_dirty) {
        if (key_pipeline != nullptr && key_pipeline == current_pipeline) {
            Memo::Note(Stat::PipeSkip);
            if (shaders_same) {
                Memo::Note(Stat::PipeSkipShadersSame);
            }
            current_pipeline->MarkUsed(frame_count);
            return BuiltPipeline(current_pipeline);
        }
        Memo::Note(Stat::PipeMissKeyStale);
    } else if (state_same) {
        Memo::Note(Stat::PipeStateSameShadersChanged);
    }
    if (!state_same) {
        raw_state.Refresh(*maxwell3d, dynamic_features);
        if (skip) {
            state_watch.Store(*maxwell3d, dynamic_features);
        }
    }
    if (native) {
        graphics_key.state = native_state;
        const VideoCommon::ShaderInfo* const vertex{shader_infos[1]};
        u32 mask = ~0U;
        if (maxwell3d->regs.IsMeshEnabled()) {
            mask = 0;
        } else if (vertex && !IsUnmaskedVertexProgram(vertex->unique_hash)) {
            mask = vertex->generic_read_mask;
        }
        graphics_key.state.CanonicalizeNative(mask);
    }
    key_pipeline = nullptr;

    if (current_pipeline) {
        GraphicsPipeline* const next{current_pipeline->Next(graphics_key)};
        if (next) {
            Memo::Note(next == current_pipeline ? Stat::PipeNextSame : Stat::PipeNextTransition);
            current_pipeline = next;
            if (current_pipeline->IsPrecompiled() && current_pipeline->Failed()) {
                return ReplaceFailedPrecompiled();
            }
            current_pipeline->MarkUsed(frame_count);
            key_pipeline = current_pipeline;
            return BuiltPipeline(current_pipeline);
        }
    }
    Memo::Note(Stat::PipeSlowPath);
    return CurrentGraphicsPipelineSlowPath();
}

ComputePipeline* ComputePipelineCacheEntry::Get(bool native, GPUVAddr address,
                                                const ComputePipelineCacheKey& key) const {
    if (!pipeline && native) {
        throw Shader::RuntimeError(
            "Native compute pipeline unavailable: address={:#x} shader={:#016x} pipeline={:#016x}: {}",
            address, key.unique_hash, key.Hash(),
            failure.empty() ? "No compiler diagnostic was recorded" : failure);
    }
    return pipeline.get();
}

constexpr u32 MaxComputeRebuilds = 4;

namespace {
/// Which pointer slots a native program's arguments fill depends on the area (Ocarina of Time's
/// foliage culling fills different list slots in Hyrule Field and Castle Town), so a translation
/// rebuilt in one area may lack pointers the first cached one binds. Try first the translations
/// that bind every pointer the first one binds, those binding the most first, then the others in
/// cache order.
void OrderComputeTranslations(std::vector<CachedComputeTranslation>& candidates) {
    if (candidates.size() < 2) {
        return;
    }
    std::vector<std::vector<u64>> keys;
    keys.reserve(candidates.size());
    for (const CachedComputeTranslation& candidate : candidates) {
        keys.push_back(candidate.env.MappedStoragePointerKeys());
    }
    std::vector<size_t> order(candidates.size());
    std::iota(order.begin(), order.end(), size_t{0});
    const auto keeps_first = [&](size_t index) {
        return std::ranges::includes(keys[index], keys.front());
    };
    std::ranges::stable_sort(order, [&](size_t lhs, size_t rhs) {
        const bool lhs_keeps{keeps_first(lhs)};
        if (lhs_keeps != keeps_first(rhs)) {
            return lhs_keeps;
        }
        return lhs_keeps && keys[lhs].size() > keys[rhs].size();
    });
    std::vector<CachedComputeTranslation> ordered;
    ordered.reserve(candidates.size());
    for (const size_t index : order) {
        ordered.push_back(std::move(candidates[index]));
    }
    candidates = std::move(ordered);
}
} // Anonymous namespace

bool PipelineCache::StaleNullPointerSlot(const ComputePipelineCacheEntry& entry) const {
    const auto& qmd{kepler_compute->launch_description};
    for (const auto& [index, offset] : entry.null_pointer_slots) {
        if (index >= qmd.const_buffer_config.size() ||
            ((qmd.const_buffer_enable_mask >> index) & 1) == 0) {
            continue;
        }
        const auto& cbuf{qmd.const_buffer_config[index]};
        if (offset + sizeof(u64) > cbuf.size) {
            continue; // The translator reads past the bound size as zero, too
        }
        const GPUVAddr slot{cbuf.Address() + offset};
        if (!gpu_memory->IsFullyMappedRange(slot, sizeof(u64))) {
            continue;
        }
        const GPUVAddr address{gpu_memory->Read<u64>(slot)};
        if (address != 0 && gpu_memory->IsWithinGPUAddressRange(address) &&
            gpu_memory->IsFullyMappedRange(address, 1)) {
            LOG_INFO(Render_Vulkan,
                     "Rebuilding compute program {:#x}: c{}[{:#x}] was null at translation and "
                     "now points to {:#x}",
                     kepler_compute->regs.code_loc.Address() + qmd.program_start, index, offset, address);
            return true;
        }
    }
    return false;
}

ComputePipeline* PipelineCache::CurrentComputePipeline(u64& native_key) {

    const ShaderInfo* const shader{ComputeShader()};
    if (!shader) {
        return nullptr;
    }
    native_key = shader->lowered_native ? shader->lowered_native->native_program_key : 0;
    const auto& qmd{kepler_compute->launch_description};
    const ComputePipelineCacheKey key{
        .unique_hash = shader->unique_hash,
        .shared_memory_size = qmd.shared_alloc,
        .workgroup_size{qmd.block_dim_x, qmd.block_dim_y, qmd.block_dim_z},
    };
    const auto [pair, is_new]{compute_cache.try_emplace(key)};
    auto& entry{pair->second};
    if (is_new) {
        entry = CreateComputePipeline(key, shader);
    } else if (entry.pending) {
        // A rebuild (below) compiles on a worker; switch once it is built (zortos293/Lichii#969:
        // the next dispatch used to wait for it, up to 2.8 s in Ocarina of Time).
        if (entry.pending->pipeline->IsBuilt()) {
            auto rebuilt{std::move(entry.pending)};
            if (!rebuilt->pipeline->BuildFailed()) {
                retired_compute_pipelines.emplace_back(scheduler.CurrentTick(),
                                                       std::move(entry.pipeline));
                entry = std::move(*rebuilt);
            } else {
                LOG_WARNING(Render_Vulkan, "Keeping compute shader {:016x}: rebuild failed",
                            key.unique_hash);
                retired_compute_pipelines.emplace_back(scheduler.CurrentTick(),
                                                       std::move(rebuilt->pipeline));
                entry.rebuilds = MaxComputeRebuilds;
            }
        }
    } else if (entry.pipeline && entry.pipeline->IsPrecompiled() && entry.pipeline->BuildFailed()) {
        precompiled_rejected.insert(key.Hash());
        precompiled_failed.fetch_add(1, std::memory_order_relaxed);
        retired_compute_pipelines.emplace_back(scheduler.CurrentTick(), std::move(entry.pipeline));
        entry = CreateComputePipeline(key, shader);
    } else if (!entry.null_pointer_slots.empty() && entry.rebuilds < MaxComputeRebuilds &&
               StaleNullPointerSlot(entry)) {
        // The translation dropped accesses through a pointer that was null back then. Which slots
        // the game fills depends on the area, so the new one keeps binding every pointer this one
        // binds, null now or not: a null pointer binds nothing.
        auto rebuilt{CreateComputePipeline(key, shader, entry.mapped_pointers)};
        if (rebuilt.pipeline) {
            rebuilt.rebuilds = entry.rebuilds + 1;
            if (entry.pipeline && !entry.pipeline->BuildFailed()) {
                // The current translation keeps dispatching meanwhile: for a few frames it misses
                // the accesses through the newly filled pointer, instead of a freeze.
                entry.pending = std::make_unique<ComputePipelineCacheEntry>(std::move(rebuilt));
            } else {
                if (entry.pipeline) {
                    retired_compute_pipelines.emplace_back(scheduler.CurrentTick(),
                                                           std::move(entry.pipeline));
                }
                entry = std::move(rebuilt);
            }
        } else {
            // Keep the working translation when the new one cannot be built (for example a
            // narrow access that now stays global across two buffers), and stop retrying.
            LOG_WARNING(Render_Vulkan, "Keeping compute shader {:016x}: rebuild failed: {}",
                        key.unique_hash, rebuilt.failure);
            entry.rebuilds = MaxComputeRebuilds;
        }
    }
    return entry.Get(kepler_compute->IsNativeMode(),
                     kepler_compute->regs.code_loc.Address() + qmd.program_start, key);
}

void PipelineCache::LoadDiskResources(u64 title_id, std::stop_token stop_loading,
                                      const VideoCore::DiskResourceLoadCallback& callback) {
    if (title_id == 0) {
        return;
    }
    if (!pipeline_cache_filename.empty()) {
        background_pipelines.Stop();
        workers.WaitForRequests();
        serialization_thread.WaitForRequests();
        if (use_vulkan_pipeline_cache && !vulkan_pipeline_cache_filename.empty()) {
            SerializeVulkanPipelineCache(vulkan_pipeline_cache_filename, vulkan_pipeline_cache,
                                         CACHE_VERSION);
        }
    }

    const auto shader_dir{Common::FS::GetEdenPath(Common::FS::EdenPath::ShaderDir)};
    const auto base_dir{shader_dir / fmt::format("{:016x}", title_id)};
    if (!Common::FS::CreateDir(shader_dir) || !Common::FS::CreateDir(base_dir)) {
        LOG_ERROR(Common_Filesystem, "Failed to create pipeline cache directories");
        return;
    }
    pipeline_cache_filename =
        VideoCommon::VersionedPipelineCachePath(base_dir, "vulkan", CACHE_VERSION);
    used_pipelines_filename =
        VideoCommon::VersionedPipelineCachePath(base_dir, "vulkan_used", CACHE_VERSION);
    const auto carried = VideoCommon::CarryOverPipelineCache(
        base_dir, "vulkan", VideoCommon::PipelineCacheMagic,
        VideoCommon::CompatibleVulkanPipelineCacheVersions, CACHE_VERSION);
    if (carried.version) {
        // The used list goes with the pipelines it was saved beside, never one this version kept
        // after its own cache was removed.
        const auto carried_used =
            VideoCommon::VersionedPipelineCachePath(base_dir, "vulkan_used", *carried.version);
        std::error_code migration_error;
        if (std::filesystem::exists(carried_used, migration_error)) {
            std::filesystem::copy_file(carried_used, used_pipelines_filename,
                                       std::filesystem::copy_options::overwrite_existing,
                                       migration_error);
        } else {
            std::filesystem::remove(used_pipelines_filename, migration_error);
        }
        LOG_INFO(Render_Vulkan, "Pipeline cache started from the version {} cache",
                 *carried.version);
    } else if (carried.failed) {
        LOG_WARNING(Render_Vulkan, "Couldn't copy the older pipeline cache; this one starts empty");
    }
    used_pipelines = LoadUsedPipelines(used_pipelines_filename, MaxUsedPipelineAge(device));
    used_pipelines_saved = std::chrono::steady_clock::now();
    order_filename = VideoCommon::VersionedPipelineCachePath(base_dir, "vulkan_order", CACHE_VERSION);
    last_order.clear();
    last_order_index.clear();
    first_use_order.clear();
    skipped_records.clear();
    {
        std::ifstream order{order_filename, std::ios::binary};
        for (u64 hash; order.read(reinterpret_cast<char*>(&hash), sizeof(hash));) {
            if (last_order_index.try_emplace(hash, static_cast<u32>(last_order.size())).second) {
                last_order.push_back(hash);
            }
        }
    }
    // nup: EDEN_NX2_BOOT_AGE=N builds at boot only the pipelines used in the last N sessions
    // (default all 16 kept); follow-on prebuild covers the rest as the game gets near them.
    static const u8 boot_age = [] {
        const char* value = std::getenv("EDEN_NX2_BOOT_AGE");
        return value ? static_cast<u8>(std::clamp(std::atoi(value), 0, 255)) : u8{255};
    }();
    const auto used_at_boot = [this](u64 hash) {
        const auto it = used_pipelines.find(hash);
        return it != used_pipelines.end() && it->second < boot_age;
    };
    size_t skipped_pipelines = 0;
    size_t deferred_pipelines = 0;
    size_t queued_graphics = 0;

    // With the shader precache this boot, the cached graphics pipelines skipped below are built
    // on the loading screen too, unless an earlier precache built them for this driver.
    namespace ShaderWarm = VideoCommon::ShaderWarm;
    const bool precache = ShaderWarm::PrecacheBeforeStart(
        Settings::values.precache_shaders.GetValue(), ShaderWarm::TakeWarmBeforeStart());
    std::unordered_set<u64> precached_pipelines;
    if (precache) {
        precached_pipelines = ShaderWarm::ReadPrecachedPipelines(
            ShaderWarm::PrecachedPipelinesPath(title_id),
            {device.GetDriverVersion(), device.GetDeviceID(), CACHE_VERSION});
    }
    std::vector<PrecacheJob> precache_graphics;
    size_t precache_capture_bytes = 0;

    if (use_vulkan_pipeline_cache) {
        vulkan_pipeline_cache_filename =
            VideoCommon::VersionedPipelineCachePath(base_dir, "vulkan_pipelines", CACHE_VERSION);
        if (VideoCommon::CarryOverPipelineCache(
                base_dir, "vulkan_pipelines", VULKAN_CACHE_MAGIC_NUMBER,
                VideoCommon::CompatibleVulkanPipelineCacheVersions, CACHE_VERSION)
                .failed) {
            LOG_WARNING(Render_Vulkan,
                        "Couldn't copy the older driver pipeline cache; this one starts empty");
        }
        vulkan_pipeline_cache =
            LoadVulkanPipelineCache(vulkan_pipeline_cache_filename, CACHE_VERSION);
    }

    struct {
        std::mutex mutex;
        size_t total{};
        size_t built{};
        bool has_loaded{};
        std::unique_ptr<PipelineStatistics> statistics;
        /// Records the load never chooses: those lacking a value the translator reads (graphics
        /// ones with their key hash), and copies of a compute translation that stays
        std::vector<VideoCommon::PipelineRecord> incomplete;
        std::vector<std::pair<u64, VideoCommon::PipelineRecord>> incomplete_graphics;
        std::vector<VideoCommon::PipelineRecord> copies;
    } state;
    // A rebuilt translation is cached beside the stale one. Choose between them before building, so
    // loading only ever adds entries (see OrderComputeTranslations), and take the next when a
    // translation fails.
    ::Common::unordered_map<ComputePipelineCacheKey, std::vector<CachedComputeTranslation>>
        compute_envs;
    // Where each record read begins and ends, to remove those the load never chooses.
    u64 record_begin{VideoCommon::PipelineCacheMagic.size() + sizeof(CACHE_VERSION)};
    const auto read_record{[&](std::ifstream& file) {
        const auto record{VideoCommon::ReadPipelineRecord(file, record_begin)};
        record_begin = record.end;
        return record;
    }};
    /// For each graphics key hash read, where the first record's key is and its stages
    std::unordered_map<u64, std::pair<u64, u64>> first_graphics;
    std::vector<DuplicateGraphicsRecord> graphics_duplicates;

    if (device.IsKhrPipelineExecutablePropertiesEnabled()) {
        state.statistics = std::make_unique<PipelineStatistics>(device);
    }
    const auto load_compute{[&](std::ifstream& file, FileEnvironment env) {
        ComputePipelineCacheKey key;
        file.read(reinterpret_cast<char*>(&key), sizeof(key));
        VideoCommon::CheckComputeRecord(env, key.shared_memory_size, key.workgroup_size);
        compute_envs[key].push_back({std::move(env), read_record(file)});
    }};
    const auto queue_compute{[&](const ComputePipelineCacheKey& key,
                                 std::vector<CachedComputeTranslation> candidates) {
        workers.QueueWork([this, key, candidates_ = std::move(candidates), &state,
                           &callback]() mutable {
            ShaderPools pools;
            ComputePipelineCacheEntry entry;
            size_t chosen{};
            for (; chosen < candidates_.size(); ++chosen) {
                pools.ReleaseContents();
                entry = CreateComputePipeline(pools, key, candidates_[chosen].env,
                                              state.statistics.get(), false);
                if (entry.pipeline) {
                    break;
                }
            }
            std::scoped_lock lock{state.mutex};
            if (entry.pipeline) {
                compute_cache.emplace(key, std::move(entry));
            }
            AddUnchosenComputeTranslations(candidates_, chosen,
                                           host_info.EnvironmentReadProperties(), state.incomplete,
                                           state.copies);
            ++state.built;
            if (state.has_loaded) {
                callback(VideoCore::LoadCallbackStage::Build, state.built, state.total);
            }
        });
        ++state.total;
    }};
    const auto load_graphics{[&](std::ifstream& file, std::vector<FileEnvironment> envs) {
        GraphicsPipelineCacheKey key;
        file.read(reinterpret_cast<char*>(&key), sizeof(key));
        VideoCommon::CheckGraphicsRecord(key.unique_hashes, envs.size());
        const auto record{read_record(file)};
        u64 stages{};
        for (const FileEnvironment& env : envs) {
            stages = (stages << 8) | static_cast<u32>(env.ShaderStage());
        }
        // The checks below and disk_pipelines take the first record of a key: a later one with the
        // same key and stages is never built, whatever the device.
        if (const auto [first, inserted] =
                first_graphics.try_emplace(key.Hash(), record.end - sizeof(key), stages);
            !inserted && first->second.second == stages) {
            graphics_duplicates.push_back({record, first->second.first});
        }

        if ((key.state.extended_dynamic_state != 0) !=
                dynamic_features.has_extended_dynamic_state ||
            (key.state.extended_dynamic_state_2 != 0) !=
                dynamic_features.has_extended_dynamic_state_2 ||
            (key.state.extended_dynamic_state_2_logic_op != 0) !=
                dynamic_features.has_extended_dynamic_state_2_logic_op ||
            (key.state.extended_dynamic_state_3_blend != 0) !=
                dynamic_features.has_extended_dynamic_state_3_blend ||
            (key.state.extended_dynamic_state_3_enables != 0) !=
                dynamic_features.has_extended_dynamic_state_3_enables ||
            (key.state.color_write_enable_dynamic != 0) !=
                dynamic_features.has_color_write_enable ||
            (key.state.dynamic_vertex_input != 0) !=
                dynamic_features.has_dynamic_vertex_input) {
            return;
        }

        const bool key_requests_provoking_last = key.state.provoking_vertex_last != 0;
        if (key_requests_provoking_last && !dynamic_features.has_provoking_vertex_last_mode) {
            return;
        }

        const bool key_uses_transform_feedback = key.state.xfb_enabled != 0;
        if (key_uses_transform_feedback && key_requests_provoking_last &&
            !dynamic_features.has_provoking_vertex_tf_preserve) {
            return;
        }

        // Records written before keys were canonical are found under their old hash in the
        // used list; the pipeline itself is keyed the way draws now look it up.
        const u64 stored_hash = key.Hash();
        {
            boost::container::static_vector<const Shader::Environment*, Maxwell::MaxShaderProgram>
                env_ptrs;
            for (const auto& env : envs) {
                env_ptrs.push_back(&env);
            }
            CanonicalizeNativeKey(key, env_ptrs);
        }
        const u64 hash = key.Hash();
        if (!disk_pipelines.insert(hash).second) {
            return;
        }
        if (!PreloadAllPipelines() &&
            (used_pipelines.empty()
                 ? queued_graphics >= MaxPipelinesWithoutUsedList
                 : !used_at_boot(hash) && !used_at_boot(stored_hash))) {
            ++skipped_pipelines;
            skipped_records.try_emplace(hash, key, record.begin);
            size_t capture_bytes = (envs.capacity() - envs.size()) * sizeof(FileEnvironment);
            for (const auto& env : envs) {
                capture_bytes += env.MemorySize();
            }
            if (precache && !precached_pipelines.contains(hash) &&
                capture_bytes <= PrecacheCaptureLimit - precache_capture_bytes) {
                precache_capture_bytes += capture_bytes;
                precache_graphics.push_back({hash, [this, key, envs = std::move(envs)]() mutable {
                    ShaderPools pools;
                    boost::container::static_vector<Shader::Environment*,
                                                    Maxwell::MaxShaderProgram>
                        env_ptrs;
                    for (auto& env : envs) {
                        env_ptrs.push_back(&env);
                    }
                    const auto pipeline =
                        CreateGraphicsPipeline(pools, key, MakeSpan(env_ptrs), nullptr, false);
                    return pipeline != nullptr && !pipeline->Failed();
                }});
                return;
            }
            if (background_pipelines.Add(
                    key, capture_bytes, [this, key, envs = std::move(envs)]() mutable {
                        ShaderPools pools;
                        boost::container::static_vector<Shader::Environment*,
                                                        Maxwell::MaxShaderProgram>
                            env_ptrs;
                        for (auto& env : envs) {
                            env_ptrs.push_back(&env);
                        }
                        auto pipeline =
                            CreateGraphicsPipeline(pools, key, MakeSpan(env_ptrs), nullptr, false);
                        if (pipeline && pipeline->Failed()) {
                            return std::unique_ptr<GraphicsPipeline>{};
                        }
                        return pipeline;
                    })) {
                ++deferred_pipelines;
            }
            return;
        }
        ++queued_graphics;

        workers.QueueWork([this, key, envs_ = std::move(envs), record, &state,
                           &callback]() mutable {
            ShaderPools pools;
            boost::container::static_vector<Shader::Environment*, Maxwell::MaxShaderProgram>
                env_ptrs;
            for (auto& env : envs_) {
                env_ptrs.push_back(&env);
            }
            auto pipeline{CreateGraphicsPipeline(pools, key, MakeSpan(env_ptrs),
                                                 state.statistics.get(), false)};
            const bool incomplete{!pipeline && std::ranges::any_of(envs_, [this](const auto& env) {
                return env.LacksRecordedValue(host_info.EnvironmentReadProperties());
            })};

            std::scoped_lock lock{state.mutex};
            if (pipeline) {
                graphics_cache.emplace(key, std::move(pipeline));
            }
            if (incomplete) {
                state.incomplete_graphics.emplace_back(key.Hash(), record);
            }
            ++state.built;
            if (state.has_loaded) {
                callback(VideoCore::LoadCallbackStage::Build, state.built, state.total);
            }
        });
        ++state.total;
    }};
    VideoCommon::LoadPipelines(stop_loading, pipeline_cache_filename, CACHE_VERSION, load_compute,
                               load_graphics);
    for (auto& [key, candidates] : compute_envs) {
        OrderComputeTranslations(candidates);
        queue_compute(key, std::move(candidates));
    }
    compute_envs.clear();

    LOG_INFO(Render_Vulkan, "Total Pipeline Count: {}", state.total);
    if (skipped_pipelines != 0) {
        LOG_INFO(
            Render_Vulkan,
            "{} cached graphics pipelines skipped at boot, {} admitted for background prebuild",
            skipped_pipelines, deferred_pipelines);
    }

    std::unique_lock lock{state.mutex};
    callback(VideoCore::LoadCallbackStage::Build, 0, state.total);
    state.has_loaded = true;
    lock.unlock();

    workers.WaitForRequests(stop_loading);

    AddIncompleteGraphics(state.incomplete_graphics, graphics_duplicates, disk_pipelines,
                          state.incomplete);
    // Records the load never chooses are read again at every boot. The serialization thread
    // rewrites the file without them, before it appends the pipelines built from now on: at once
    // for incomplete ones, else once the superseded ones take up an eighth of the file.
    u64 superseded_bytes{};
    for (const auto& record : state.copies) {
        superseded_bytes += record.end - record.begin;
    }
    for (const auto& duplicate : graphics_duplicates) {
        superseded_bytes += duplicate.record.end - duplicate.record.begin;
    }
    if (!stop_loading.stop_requested() &&
        (!state.incomplete.empty() ||
         (superseded_bytes != 0 && superseded_bytes >= record_begin / 8))) {
        auto removed{std::move(state.incomplete)};
        removed.insert(removed.end(), state.copies.begin(), state.copies.end());
        serialization_thread.QueueWork([filename = pipeline_cache_filename,
                                        loaded_size = record_begin, records = std::move(removed),
                                        duplicates = std::move(graphics_duplicates), stop_loading,
                                        destroyed = serialization_stop.get_token()]() mutable {
            // Stopped by a cancelled load, or by the cache's destruction.
            std::stop_source stop;
            const std::stop_callback on_cancel{stop_loading, [&stop] { stop.request_stop(); }};
            const std::stop_callback on_destroy{destroyed, [&stop] { stop.request_stop(); }};
            if (stop.stop_requested()) {
                return;
            }
            AddEqualGraphicsKeys(filename, duplicates, records);
            VideoCommon::RemovePipelines(stop.get_token(), filename, CACHE_VERSION, loaded_size,
                                         std::move(records));
        });
    }

    PreparePrecompiled(base_dir, precache, stop_loading, callback);

    if (use_vulkan_pipeline_cache) {
        SerializeVulkanPipelineCache(vulkan_pipeline_cache_filename, vulkan_pipeline_cache,
                                     CACHE_VERSION);
        size_t size = 0;
        vulkan_pipeline_cache.Read(&size, nullptr);
        last_cache_size.store(size, std::memory_order_relaxed);
        last_flush = std::chrono::steady_clock::now();
    }

    if (state.statistics) {
        state.statistics->Report();
    }
    if (precache && !stop_loading.stop_requested()) {
        PrecacheShaders(title_id, std::move(precache_graphics), std::move(precached_pipelines),
                        stop_loading, callback);
    }
    if (stop_loading.stop_requested()) {
        background_pipelines.Stop();
    } else {
        background_pipelines.Start(std::clamp<size_t>(num_pipeline_workers / 4, 1, 2));
    }
}

void PipelineCache::PrecacheShaders(u64 title_id, std::vector<PrecacheJob> jobs,
                                    std::unordered_set<u64> precached, std::stop_token stop_loading,
                                    const VideoCore::DiskResourceLoadCallback& callback) {
    namespace ShaderWarm = VideoCommon::ShaderWarm;
    const size_t graphics_jobs = jobs.size();
    const std::filesystem::path warm_file = ShaderWarm::ComputeWarmPath(title_id);
    const std::string path = Common::FS::PathToUTF8String(warm_file);
    const auto stamp = [&](u64 entries) {
        return ShaderWarm::CurrentWarmStamp(warm_file, entries, device.GetDriverVersion(),
                                            device.GetDeviceID(), CACHE_VERSION);
    };
    // Compute programs of the precache file, unless a matching marker says they were built.
    size_t file_entries = 0;
    bool compute_pending = false;
    std::error_code ec;
    const auto current = stamp(0);
    const auto marker = ShaderWarm::ReadWarmMarker(warm_file);
    if (!std::filesystem::exists(warm_file, ec)) {
        LOG_INFO(Render_Vulkan, "Shader precache: no precache file at {}", path);
    } else if (current && marker && marker->Matches(*current)) {
        LOG_INFO(Render_Vulkan, "Shader precache: compute programs already done for this driver");
    } else {
        // Read the whole file here (its own reader: the disk cache loader deletes files it rejects).
        try {
            std::ifstream file(warm_file, std::ios::binary | std::ios::ate);
            file.exceptions(std::ifstream::failbit);
            const auto end{file.tellg()};
            file.seekg(0, std::ios::beg);
            std::array<char, 8> magic;
            u32 version{};
            file.read(magic.data(), magic.size())
                .read(reinterpret_cast<char*>(&version), sizeof(version));
            constexpr std::array<char, 8> expected_magic{'y', 'u', 'z', 'u', 'c', 'a', 'c', 'h'};
            if (magic != expected_magic || version != CACHE_VERSION) {
                LOG_WARNING(Render_Vulkan, "Shader precache: {} is not a version {} file", path,
                            CACHE_VERSION);
                throw std::runtime_error("wrong version");
            }
            while (file.tellg() != end) {
                u32 num_envs{};
                file.read(reinterpret_cast<char*>(&num_envs), sizeof(num_envs));
                FileEnvironment env;
                env.Deserialize(file, end);
                if (num_envs != 1 || env.ShaderStage() != Shader::Stage::Compute) {
                    LOG_WARNING(Render_Vulkan, "Shader precache: unexpected entry in {}", path);
                    throw std::runtime_error("unexpected entry");
                }
                ComputePipelineCacheKey key;
                file.read(reinterpret_cast<char*>(&key), sizeof(key));
                ++file_entries;
                // Programs already in the disk cache were built above; leave them out.
                if (compute_cache.contains(key)) {
                    continue;
                }
                jobs.push_back({0, [this, key, env = std::move(env)]() mutable {
                                    static const GuessMissing guess{};
                                    env.SetGuessMissing(&guess);
                                    ShaderPools pools;
                                    return CreateComputePipeline(pools, key, env, nullptr, false)
                                               .pipeline != nullptr;
                                }});
            }
            compute_pending = true;
        } catch (const std::exception& e) {
            LOG_WARNING(Render_Vulkan, "Shader precache: reading {} failed: {}", path, e.what());
            jobs.resize(graphics_jobs);
            file_entries = 0;
        }
    }
    if (jobs.empty() && !compute_pending) {
        LOG_INFO(Render_Vulkan, "Shader precache: nothing to precache");
        return;
    }

    const auto begin = std::chrono::steady_clock::now();
    const size_t num_threads = std::clamp<size_t>(std::thread::hardware_concurrency(), 1, 64);
    std::vector<u8> built(jobs.size());
    std::atomic<size_t> next{};
    std::atomic<size_t> done{};
    std::atomic<bool> cancel{false};
    bool skipped = false;
    LOG_INFO(Render_Vulkan,
             "Shader precache: {} cached graphics pipelines and {} of {} compute programs to "
             "build on {} threads",
             graphics_jobs, jobs.size() - graphics_jobs, file_entries, num_threads);
    if (!jobs.empty()) {
        std::vector<std::jthread> threads;
        for (size_t thread = 0; thread < num_threads; ++thread) {
            threads.emplace_back([&jobs, &built, &next, &done, &cancel] {
                Common::SetCurrentThreadName("VkShaderPrecache");
                while (!cancel.load(std::memory_order_relaxed)) {
                    const size_t index = next.fetch_add(1);
                    if (index >= jobs.size()) {
                        break;
                    }
                    try {
                        // Built and destroyed here: only the driver's cache keeps it.
                        built[index] = jobs[index].build();
                    } catch (const std::exception& e) {
                        LOG_DEBUG(Render_Vulkan, "Shader precache: {}", e.what());
                    }
                    jobs[index].build = {};
                    done.fetch_add(1);
                }
            });
        }
        while (done < jobs.size()) {
            if (stop_loading.stop_requested() || ShaderWarm::TakeSkipWarm()) {
                // Every thread stops after its current build; the scope end joins them.
                cancel = true;
                skipped = true;
                break;
            }
            if (callback) {
                callback(VideoCore::LoadCallbackStage::Warm, done, jobs.size());
            }
            std::this_thread::sleep_for(std::chrono::milliseconds{100});
        }
    }
    const size_t total_built = static_cast<size_t>(std::count(built.begin(), built.end(), u8{1}));
    if (compute_pending && !skipped) {
        if (const auto stamp_done = stamp(file_entries)) {
            ShaderWarm::WriteWarmMarker(warm_file, *stamp_done);
        }
    }
    const size_t precached_before = precached.size();
    for (size_t index = 0; index < graphics_jobs; ++index) {
        if (built[index]) {
            precached.insert(jobs[index].graphics_hash);
        }
    }
    if (precached.size() != precached_before) {
        ShaderWarm::WritePrecachedPipelines(
            ShaderWarm::PrecachedPipelinesPath(title_id),
            {device.GetDriverVersion(), device.GetDeviceID(), CACHE_VERSION}, precached);
    }
    if (!stop_loading.stop_requested() && total_built != 0) {
        // Without this, the builds reach the file only at the next boot, game change or clean
        // close, and are lost if the app ends otherwise.
        QueueVulkanPipelineCacheSave();
    }
    LOG_INFO(Render_Vulkan, "Shader precache: {}, {} built, {} failed, {} not reached, {:.1f} s",
             stop_loading.stop_requested() ? "stopped" : (skipped ? "skipped" : "finished"),
             total_built, done - total_built, jobs.size() - std::min(jobs.size(), done.load()),
             std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count());
}

void PipelineCache::CanonicalizeNativeKey(
    GraphicsPipelineCacheKey& key,
    std::span<const Shader::Environment* const> envs) const noexcept {
    if (envs.empty() || !std::ranges::all_of(envs, [](const Shader::Environment* env) {
            return env->IsNativeProgram();
        })) {
        return;
    }
    u32 mask = ~0U;
    for (const Shader::Environment* const env : envs) {
        if (env->ShaderStage() == Shader::Stage::Task ||
            env->ShaderStage() == Shader::Stage::Mesh) {
            mask = 0;
            break;
        }
        if (env->ShaderStage() == Shader::Stage::VertexB &&
            !IsUnmaskedVertexProgram(key.unique_hashes[1])) {
            mask = VideoCommon::GenericReadMask(env->SPH());
        }
    }
    key.state.CanonicalizeNative(mask);
}

std::string PipelineCache::NativePreparationStamp(const std::filesystem::path& path) const {
    // The file is assembled locally from the player's game files; foreign caches are not read.
    const auto content = VideoCommon::ShaderWarm::ContentHash(path);
    if (content.empty()) {
        return {};
    }
    const auto properties = device.GetPhysical().GetProperties();
    std::string stamp = fmt::format(
        "lichii-2 {} {} {} {} {} {} {}", content, LICHII_PREPARATION_REVISION, CACHE_VERSION,
        properties.vendorID, properties.deviceID, properties.driverVersion, properties.apiVersion);
    for (const auto byte : properties.pipelineCacheUUID) {
        stamp += fmt::format("{:02x}", byte);
    }
    const auto setting = [&](auto value) { stamp += fmt::format(" {}", value); };
    setting(profile.supported_spirv);
    setting(profile.unified_descriptor_binding);
    setting(profile.support_descriptor_aliasing);
    setting(profile.support_int8);
    setting(profile.support_uniform_and_storage_buffer_8bit);
    setting(profile.support_storage_buffer_8bit);
    setting(profile.support_int16);
    setting(profile.support_uniform_and_storage_buffer_16bit);
    setting(profile.support_storage_buffer_16bit);
    setting(profile.support_int64);
    setting(profile.support_vertex_instance_id);
    setting(profile.support_float_controls);
    setting(profile.support_separate_denorm_behavior);
    setting(profile.support_separate_rounding_mode);
    setting(profile.support_fp16_denorm_preserve);
    setting(profile.support_fp32_denorm_preserve);
    setting(profile.support_fp16_denorm_flush);
    setting(profile.support_fp32_denorm_flush);
    setting(profile.support_fp16_signed_zero_nan_preserve);
    setting(profile.support_fp32_signed_zero_nan_preserve);
    setting(profile.support_fp64_signed_zero_nan_preserve);
    setting(profile.support_explicit_workgroup_layout);
    setting(profile.support_workgroup_layout_8bit_access);
    setting(profile.support_workgroup_layout_16bit_access);
    setting(profile.support_shader_quad_control);
    setting(profile.support_quad_shuffles);
    setting(profile.support_vote);
    setting(profile.support_shuffle_relative);
    setting(profile.supported_subgroup_stages);
    setting(profile.support_viewport_index_layer_non_geometry);
    setting(profile.support_viewport_mask);
    setting(profile.support_typeless_image_loads);
    setting(profile.support_demote_to_helper_invocation);
    setting(profile.support_int64_atomics);
    setting(profile.support_shared_int64_atomics);
    setting(profile.support_derivative_control);
    setting(profile.support_geometry_shader_passthrough);
    setting(profile.support_native_ndc);
    setting(profile.support_gl_nv_gpu_shader_5);
    setting(profile.support_gl_amd_gpu_shader_half_float);
    setting(profile.support_gl_texture_shadow_lod);
    setting(profile.support_gl_warp_intrinsics);
    setting(profile.support_gl_variable_aoffi);
    setting(profile.support_gl_sparse_textures);
    setting(profile.support_gl_derivative_control);
    setting(profile.support_scaled_attributes);
    setting(profile.support_multi_viewport);
    setting(profile.support_geometry_streams);
    setting(profile.support_sampled_image_array_nonuniform_indexing);
    setting(profile.support_storage_image_array_nonuniform_indexing);
    setting(profile.support_uniform_texel_buffer_array_nonuniform_indexing);
    setting(profile.support_storage_texel_buffer_array_nonuniform_indexing);
    setting(profile.warp_size_potentially_larger_than_guest);
    setting(profile.lower_left_origin_mode);
    setting(profile.need_declared_frag_colors);
    setting(profile.need_fastmath_off);
    setting(profile.need_gather_subpixel_offset);
    setting(profile.has_broken_spirv_clamp);
    setting(profile.has_broken_spirv_position_input);
    setting(profile.has_broken_unsigned_image_offsets);
    setting(profile.has_broken_signed_operations);
    setting(profile.has_broken_fp16_float_controls);
    setting(profile.max_compute_shared_memory_size);
    setting(profile.has_broken_fp32_denorm_flush);
    setting(profile.has_gl_component_indexing_bug);
    setting(profile.has_gl_precise_bug);
    setting(profile.has_gl_cbuf_ftou_bug);
    setting(profile.has_gl_bool_ref_bug);
    setting(profile.ignore_nan_fp_comparisons);
    setting(profile.has_broken_spirv_subgroup_mask_vector_extract_dynamic);
    setting(profile.has_broken_spirv_subgroup_shuffle);
    setting(profile.has_broken_spirv_fragment_shuffle);
    setting(profile.support_cooperative_hmma);
    setting(profile.support_cooperative_hmma_16x16x16);
    setting(profile.max_subgroup_size);
    setting(profile.gl_max_compute_smem_size);
    setting(profile.has_broken_robust);
    setting(profile.min_ssbo_alignment);
    setting(profile.max_user_clip_distances);
    setting(host_info.min_ssbo_alignment);
    setting(host_info.max_per_stage_descriptor_sampled_images);
    setting(host_info.max_per_stage_resources);
    setting(host_info.max_per_stage_descriptor_storage_buffers);
    setting(host_info.max_descriptor_set_samplers);
    setting(host_info.max_descriptor_set_uniform_buffers);
    setting(host_info.max_descriptor_set_uniform_buffers_dynamic);
    setting(host_info.max_descriptor_set_storage_buffers);
    setting(host_info.max_descriptor_set_storage_buffers_dynamic);
    setting(host_info.max_descriptor_set_sampled_images);
    setting(host_info.max_descriptor_set_storage_images);
    setting(host_info.max_descriptor_set_input_attachements);
    setting(host_info.support_float64);
    setting(host_info.support_float16);
    setting(host_info.support_int64);
    setting(host_info.needs_demote_reorder);
    setting(host_info.support_snorm_render_buffer);
    setting(host_info.support_viewport_index_layer);
    setting(host_info.support_viewport_mask);
    setting(host_info.support_geometry_shader_passthrough);
    setting(host_info.support_conditional_barrier);
    setting(host_info.single_lane_geometry_subgroups);
    setting(host_info.support_byte_addressed_r16);
    setting(Settings::values.resolution_info.active);
    setting(Settings::values.resolution_info.up_scale);
    setting(Settings::values.resolution_info.down_shift);
    setting(Settings::values.resolution_info.up_factor);
    setting(Settings::values.resolution_info.down_factor);
    setting(Settings::values.disable_shader_loop_safety_checks.GetValue());
    setting(Settings::values.fix_bloom_effects.GetValue());
    setting(Settings::values.renderer_debug.GetValue());
    for (const char* name : {"EDEN_NX2_UNDEF_ZERO", "EDEN_NX2_NO_TENSOR"}) {
        const char* value = std::getenv(name);
        stamp += fmt::format(" {}={}", name, value ? value : "<unset>");
    }
    return stamp;
}

void PipelineCache::PreparePrecompiled(const std::filesystem::path& base_dir, bool prepare,
                                       std::stop_token stop_loading,
                                       const VideoCore::DiskResourceLoadCallback& callback) {
    namespace Warm = VideoCommon::ShaderWarm;
    const auto path = base_dir / "native-preparation.bin";
    std::error_code error;
    if (!std::filesystem::exists(path, error)) {
        return;
    }
    const char* enabled = std::getenv("EDEN_NX2_PRECOMPILED");
    if (enabled && enabled[0] == '0') {
        return;
    }
    const std::string stamp = NativePreparationStamp(path);
    const auto infos_path = base_dir / "native-prepared.journal";
    Warm::PreparedStore store;
    if (stamp.empty() || !store.Open(infos_path, stamp)) {
        LOG_WARNING(Render_Vulkan, "Cannot open the native shader preparation journal");
        return;
    }
    // Without a driver cache of its own the prepared pipelines would only warm this session.
    if (prepare && !DriverCachesPipelines(device) && !(enabled && enabled[0] == '1')) {
        LOG_INFO(Render_Vulkan, "Native shader preparation skipped: the driver keeps no pipeline "
                                "cache of its own (EDEN_NX2_PRECOMPILED=1 prepares anyway)");
        prepare = false;
    }
    if (prepare) {
        RunNativePreparation(path, store, stop_loading, callback);
    }
    LoadPrecompiledInfos(path, infos_path, store);
}

void PipelineCache::RunNativePreparation(const std::filesystem::path& path,
                                         VideoCommon::ShaderWarm::PreparedStore& store,
                                         std::stop_token stop_loading,
                                         const VideoCore::DiskResourceLoadCallback& callback) {
    namespace Warm = VideoCommon::ShaderWarm;
    std::unordered_set<u32> done_before;
    for (const auto& record : store.Records()) {
        done_before.insert(record.id);
    }
    std::ifstream file(path, std::ios::binary);
    file.exceptions(std::ifstream::failbit | std::ifstream::badbit);
    const auto get = [&file](auto& value) {
        file.read(reinterpret_cast<char*>(&value), sizeof(value));
    };
    struct {
        std::mutex mutex;
        std::condition_variable_any cv;
        size_t in_flight{};
        size_t done{};
        size_t built{};
        size_t failed{};
        size_t incompatible{};
    } state;
    std::atomic<bool> cancel{};
    // Each worker holds one pipeline at a time; it is dropped once its translation is kept.
    const size_t max_in_flight = std::max<size_t>(num_pipeline_workers * 2, 4);
    const auto begin = std::chrono::steady_clock::now();
    u32 count{};
    try {
        const u64 file_size = std::filesystem::file_size(path);
        std::array<char, 8> magic{};
        u32 version{}, env_count{}, reserved{};
        get(magic);
        get(version);
        get(env_count);
        get(count);
        get(reserved);
        // Layout (tools/nx2-precompile/README.md): "NX2PREP1", u32 cache version, u32 environment
        // count, u32 record count, u32 reserved; per environment u64 offset and u64 size; per
        // record u8 kind (0 graphics, 1 compute), u8 environment count, u16 reserved, u32
        // environment indices in stage order, then the key.
        if (std::string_view(magic.data(), magic.size()) != "NX2PREP1" ||
            version != CACHE_VERSION || reserved != 0 ||
            env_count > Warm::PreparedStore::MaxEntries ||
            count > Warm::PreparedStore::MaxEntries || env_count > (file_size - 24) / 16) {
            throw std::runtime_error("invalid native preparation header");
        }
        auto env_index = std::make_shared<std::vector<std::pair<u64, u64>>>(env_count);
        for (auto& [offset, size] : *env_index) {
            get(offset);
            get(size);
            if (offset > file_size || size > file_size - offset ||
                size > Warm::PreparedStore::MaxPayloadBytes) {
                throw std::runtime_error("invalid native preparation environment");
            }
        }
        LOG_INFO(Render_Vulkan, "Preparing native pipelines: {} records, {} already prepared",
                 count, done_before.size());
        callback(VideoCore::LoadCallbackStage::Warm, 0, count);
        const auto report = [&] {
            size_t done;
            {
                std::scoped_lock lock{state.mutex};
                done = state.done;
            }
            callback(VideoCore::LoadCallbackStage::Warm, done, count);
        };
        for (u32 index = 0; index < count; ++index) {
            if (stop_loading.stop_requested() || Warm::SkipWarmRequested()) {
                cancel = true;
                break;
            }
            u8 kind{}, env_total{};
            u16 pad{};
            get(kind);
            get(env_total);
            get(pad);
            if (kind > 1 || env_total == 0 || env_total > 5 || (kind == 1 && env_total != 1) ||
                pad != 0) {
                throw std::runtime_error("invalid native preparation record");
            }
            auto record = std::make_shared<PrecompiledRecord>();
            record->env_count = env_total;
            record->envs.fill(NoInfo);
            for (u8 i = 0; i < env_total; ++i) {
                get(record->envs[i]);
                if (record->envs[i] >= env_count) {
                    throw std::runtime_error("invalid native preparation environment index");
                }
            }
            GraphicsPipelineCacheKey graphics{};
            ComputePipelineCacheKey compute{};
            if (kind == 1) {
                get(compute);
            } else {
                get(graphics);
            }
            bool skip = done_before.contains(index);
            if (!skip && kind == 0) {
                // Records carry the recording machine's driver; draws key them with this one.
                auto& key_state = graphics.state;
                key_state.driver_id = static_cast<u32>(device.GetDriverID());
                key_state.driver_version = device.GetDriverVersion();
                const bool other_device =
                    (key_state.extended_dynamic_state != 0) !=
                        dynamic_features.has_extended_dynamic_state ||
                    (key_state.extended_dynamic_state_2 != 0) !=
                        dynamic_features.has_extended_dynamic_state_2 ||
                    (key_state.extended_dynamic_state_2_logic_op != 0) !=
                        dynamic_features.has_extended_dynamic_state_2_logic_op ||
                    (key_state.extended_dynamic_state_3_blend != 0) !=
                        dynamic_features.has_extended_dynamic_state_3_blend ||
                    (key_state.extended_dynamic_state_3_enables != 0) !=
                        dynamic_features.has_extended_dynamic_state_3_enables ||
                    (key_state.color_write_enable_dynamic != 0) !=
                        dynamic_features.has_color_write_enable ||
                    (key_state.dynamic_vertex_input != 0) !=
                        dynamic_features.has_dynamic_vertex_input ||
                    (key_state.provoking_vertex_last != 0 &&
                     !dynamic_features.has_provoking_vertex_last_mode);
                if (other_device) {
                    std::scoped_lock lock{state.mutex};
                    ++state.incompatible;
                    skip = true;
                }
            }
            if (skip) {
                std::scoped_lock lock{state.mutex};
                ++state.done;
                continue;
            }
            {
                std::unique_lock lock{state.mutex};
                if (!state.cv.wait(lock, stop_loading,
                                   [&] { return state.in_flight < max_in_flight; })) {
                    cancel = true;
                    break;
                }
                ++state.in_flight;
            }
            workers.QueueWork([this, &path, &store, &state, &cancel, &stop_loading, env_index,
                               record, index, kind, graphics, compute] {
                bool ok = false;
                if (!cancel && !stop_loading.stop_requested()) {
                    try {
                        PrepareNativeRecord(path, *env_index, store, index, kind == 1, *record,
                                            graphics, compute);
                        ok = true;
                    } catch (const std::exception& e) {
                        LOG_DEBUG(Render_Vulkan, "Native preparation record {}: {}", index,
                                  e.what());
                    }
                }
                std::scoped_lock lock{state.mutex};
                --state.in_flight;
                ++state.done;
                ++(ok ? state.built : state.failed);
                state.cv.notify_all();
            });
            if (index % 64 == 0) {
                report();
            }
        }
        // The queued jobs reference this frame: wait for all of them, stopped or not.
        workers.WaitForRequests();
        report();
    } catch (const std::exception& e) {
        cancel = true;
        workers.WaitForRequests();
        LOG_WARNING(Render_Vulkan, "Native shader preparation stopped: {}", e.what());
    }
    const double seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count();
    LOG_INFO(Render_Vulkan,
             "Native shader preparation{}: {} of {} records done ({} built now, {} prepared "
             "before, {} failed, {} for another device) in {:.1f} s; completed entries are kept",
             cancel ? " stopped" : "", state.done, count, state.built, done_before.size(),
             state.failed, state.incompatible, seconds);
}

void PipelineCache::PrepareNativeRecord(const std::filesystem::path& path,
                                        std::span<const std::pair<u64, u64>> env_index,
                                        VideoCommon::ShaderWarm::PreparedStore& store, u32 index,
                                        bool compute, const PrecompiledRecord& record,
                                        const GraphicsPipelineCacheKey& graphics,
                                        const ComputePipelineCacheKey& compute_key) {
    namespace Warm = VideoCommon::ShaderWarm;
    std::ifstream file(path, std::ios::binary);
    file.exceptions(std::ifstream::failbit | std::ifstream::badbit);
    std::vector<FileEnvironment> environments(record.env_count);
    boost::container::static_vector<Shader::Environment*, Maxwell::MaxShaderProgram> pointers;
    for (size_t i = 0; i < record.env_count; ++i) {
        const auto [offset, size] = env_index[record.envs[i]];
        file.seekg(static_cast<std::streamoff>(offset));
        environments[i].Deserialize(file, static_cast<std::streamoff>(offset + size));
        pointers.push_back(&environments[i]);
    }
    ShaderPools pools;
    std::array<std::vector<u32>, Maxwell::MaxShaderStage> spirv;
    std::array<std::vector<u8>, 5> infos;
    if (compute) {
        const auto entry =
            CreateComputePipeline(pools, compute_key, environments[0], nullptr, false, &spirv[0]);
        if (!entry.pipeline || entry.pipeline->BuildFailed()) {
            throw std::runtime_error("compute pipeline build failed");
        }
        Shader::SerializeInfo(entry.pipeline->StageInfo(), infos[0]);
    } else {
        const auto pipeline = CreateGraphicsPipeline(pools, graphics, MakeSpan(pointers), nullptr,
                                                     false, nullptr, nullptr, &spirv);
        if (!pipeline || pipeline->Failed()) {
            throw std::runtime_error("graphics pipeline build failed");
        }
        // Emulated mesh pipelines are warmed in the driver but always translated at runtime.
        if (pipeline->IsEmulatedMesh()) {
            return;
        }
        for (size_t stage = 0; stage < infos.size(); ++stage) {
            if (const auto* const info = pipeline->StageInfo(stage)) {
                Shader::SerializeInfo(*info, infos[stage]);
            }
        }
    }
    // A translation that asked what the recipe did not record is not one runtime can reproduce.
    for (const auto& env : environments) {
        if (env.LacksRecordedValue(host_info.EnvironmentReadProperties())) {
            throw std::runtime_error("recipe lacks translation inputs");
        }
    }
    std::vector<u8> payload;
    const auto put = [&payload](const auto& value) {
        const auto* const bytes = reinterpret_cast<const u8*>(&value);
        payload.insert(payload.end(), bytes, bytes + sizeof(value));
    };
    put(static_cast<u8>(compute ? 1 : 0));
    put(record.env_count);
    put(u16{0});
    put(record.envs);
    if (compute) {
        put(compute_key);
    } else {
        put(graphics);
    }
    for (size_t stage = 0; stage < infos.size(); ++stage) {
        if (infos[stage].empty() != spirv[stage].empty()) {
            throw std::runtime_error("incomplete prepared shader stage");
        }
        Warm::PreparedStore::Location info{}, code{};
        if (!infos[stage].empty()) {
            const auto compressed = Common::Compression::CompressDataZSTD(
                reinterpret_cast<const u8*>(spirv[stage].data()), spirv[stage].size() * sizeof(u32),
                3);
            const auto info_at = store.AddBlob(infos[stage]);
            const auto code_at = store.AddBlob(compressed);
            if (!info_at || !code_at) {
                throw std::runtime_error("cannot keep prepared shader");
            }
            info = *info_at;
            code = *code_at;
        }
        put(info.offset);
        put(info.size);
        put(code.offset);
        put(code.size);
    }
    if (!store.AddRecord(index, payload)) {
        throw std::runtime_error("cannot checkpoint prepared shader");
    }
}

bool PipelineCache::LoadPrecompiledInfos(const std::filesystem::path& precompiled,
                                         const std::filesystem::path& infos,
                                         const VideoCommon::ShaderWarm::PreparedStore& store) {
    namespace Warm = VideoCommon::ShaderWarm;
    precompiled_records.clear();
    precompiled_compute_records.clear();
    precompiled_info_blobs.clear();
    precompiled_file.close();
    precompiled_infos_file.close();
    u32 record_count{};
    try {
        std::ifstream prep(precompiled, std::ios::binary);
        prep.exceptions(std::ifstream::failbit | std::ifstream::badbit);
        std::array<char, 8> magic{};
        std::array<u32, 4> header{};
        prep.read(magic.data(), magic.size());
        prep.read(reinterpret_cast<char*>(header.data()), sizeof(header));
        const u64 prep_size = std::filesystem::file_size(precompiled);
        if (std::string_view(magic.data(), magic.size()) != "NX2PREP1" ||
            header[0] != CACHE_VERSION || header[1] > Warm::PreparedStore::MaxEntries ||
            header[1] > (prep_size - 24) / 16) {
            return false;
        }
        record_count = header[2];
        precompiled_env_index.resize(header[1]);
        for (auto& [offset, size] : precompiled_env_index) {
            prep.read(reinterpret_cast<char*>(&offset), sizeof(offset));
            prep.read(reinterpret_cast<char*>(&size), sizeof(size));
            if (offset > prep_size || size > prep_size - offset ||
                size > Warm::PreparedStore::MaxPayloadBytes) {
                precompiled_env_index.clear();
                return false;
            }
        }
    } catch (const std::exception&) {
        precompiled_env_index.clear();
        return false;
    }
    std::ifstream file(infos, std::ios::binary);
    std::unordered_map<u64, u32> blob_ids;
    const auto blob_id = [&](const Warm::PreparedStore::Location& at) {
        const auto [it, added] =
            blob_ids.try_emplace(at.offset, static_cast<u32>(precompiled_info_blobs.size()));
        if (added) {
            precompiled_info_blobs.emplace_back(at.offset, at.size);
        }
        return it->second;
    };
    size_t rejected = 0;
    for (const auto& entry : store.Records()) {
        // A damaged record only loses that pipeline: its draws are translated as usual.
        const auto bytes = Warm::PreparedStore::ReadFrame(file, entry.at);
        size_t at = 0;
        const auto get = [&](auto& value) {
            if (sizeof(value) > bytes.size() - at) {
                return false;
            }
            std::memcpy(&value, bytes.data() + at, sizeof(value));
            at += sizeof(value);
            return true;
        };
        PrecompiledRecord record;
        u8 kind{}, count{};
        u16 pad{};
        bool valid = get(kind) && get(count) && get(pad) && get(record.envs) && kind <= 1 &&
                     count > 0 && count <= 5 && (kind == 0 || count == 1) && pad == 0 &&
                     entry.id < record_count;
        for (u8 i = 0; valid && i < count; ++i) {
            valid = record.envs[i] < precompiled_env_index.size();
        }
        u64 hash{};
        if (valid && kind == 1) {
            ComputePipelineCacheKey key;
            valid = get(key);
            hash = key.Hash();
            record.key.assign(reinterpret_cast<const u8*>(&key),
                              reinterpret_cast<const u8*>(&key) + sizeof(key));
        } else if (valid) {
            GraphicsPipelineCacheKey key;
            valid = get(key);
            hash = key.Hash();
            record.key.assign(reinterpret_cast<const u8*>(&key),
                              reinterpret_cast<const u8*>(&key) + sizeof(key));
        }
        record.env_count = count;
        record.infos.fill(NoInfo);
        record.spirv.fill(NoInfo);
        size_t stages = 0;
        for (size_t stage = 0; valid && stage < 5; ++stage) {
            Warm::PreparedStore::Location info{}, code{};
            valid = get(info.offset) && get(info.size) && get(code.offset) && get(code.size) &&
                    (info.size == 0) == (code.size == 0) &&
                    (info.size == 0 || (store.IsBlob(info) && store.IsBlob(code)));
            if (valid && info.size != 0) {
                record.infos[stage] = blob_id(info);
                record.spirv[stage] = blob_id(code);
                ++stages;
            }
        }
        if (!valid || stages != count || at != bytes.size()) {
            ++rejected;
            continue;
        }
        if (kind == 1) {
            precompiled_compute_records.emplace(hash, std::move(record));
        } else {
            precompiled_records.emplace(hash, std::move(record));
        }
    }
    precompiled_path = precompiled;
    precompiled_infos_path = infos;
    LOG_INFO(Render_Vulkan,
             "Prepared native shaders: {} graphics and {} compute pipelines, {} damaged records "
             "ignored",
             precompiled_records.size(), precompiled_compute_records.size(), rejected);
    // nup EDEN_NX2_PRECOMPILED_PREREAD (default on, 0 = off): see PrereadFiles.
    static const bool preread = [] {
        const char* const value = std::getenv("EDEN_NX2_PRECOMPILED_PREREAD");
        return value == nullptr || value[0] != '0';
    }();
    if (preread && !precompiled_records.empty() && !prepared_preread.joinable()) {
        std::vector<std::filesystem::path> paths{precompiled, infos};
        if (const char* const driver_cache = std::getenv("__GL_SHADER_DISK_CACHE_PATH")) {
            paths.emplace_back(driver_cache);
        }
        prepared_preread = std::jthread(PrereadFiles, std::move(paths));
    }
    return true;
}

std::unique_ptr<GraphicsPipeline> PipelineCache::CreatePrecompiledGraphicsPipeline(
    GraphicsEnvironments& environments) {
    if (precompiled_records.empty() || !PrecompiledFastPath()) {
        return nullptr;
    }
    const u64 hash = graphics_key.Hash();
    const auto it = precompiled_records.find(hash);
    if (it == precompiled_records.end() || precompiled_rejected.contains(hash)) {
        return nullptr;
    }
    const PrecompiledRecord& record = it->second;
    if (record.key.size() != sizeof(graphics_key) ||
        std::memcmp(record.key.data(), &graphics_key, sizeof(graphics_key)) != 0) {
        return nullptr;
    }
    const auto live = environments.Span();
    // A vertex A program is merged into vertex B: not one of these.
    if (graphics_key.unique_hashes[0] != 0 || live.size() != record.env_count) {
        return nullptr;
    }
    auto recorded = std::make_shared<std::vector<FileEnvironment>>(record.env_count);
    std::array<Shader::Info, 5> infos{};
    std::array<const Shader::Info*, 5> info_ptrs{};
    try {
        if (!precompiled_file.is_open()) {
            precompiled_file.open(precompiled_path, std::ios::binary);
            precompiled_file.exceptions(std::ifstream::failbit | std::ifstream::badbit);
        }
        for (size_t i = 0; i < record.env_count; ++i) {
            precompiled_file.clear();
            const auto [offset, size] = precompiled_env_index.at(record.envs[i]);
            precompiled_file.seekg(static_cast<std::streamoff>(offset));
            (*recorded)[i].Deserialize(precompiled_file,
                                       static_cast<std::streamoff>(offset + size));
        }
        if (!precompiled_infos_file.is_open()) {
            precompiled_infos_file.open(precompiled_infos_path, std::ios::binary);
        }
        std::vector<u8> bytes;
        for (size_t stage = 0; stage < infos.size(); ++stage) {
            if (record.infos[stage] == NoInfo) {
                continue;
            }
            const auto [offset, size] = precompiled_info_blobs.at(record.infos[stage]);
            bytes = VideoCommon::ShaderWarm::PreparedStore::ReadBlob(precompiled_infos_file,
                                                                     {offset, size});
            std::span<const u8> in{bytes};
            if (!Shader::DeserializeInfo(in, infos[stage]) || !in.empty() ||
                infos[stage].requires_layer_emulation) {
                CountPrecompiledReason("infos unreadable or with layer emulation");
                return nullptr;
            }
            info_ptrs[stage] = &infos[stage];
        }
    } catch (const std::exception&) {
        CountPrecompiledReason("files unreadable");
        precompiled_file.close();
        precompiled_infos_file.close();
        return nullptr;
    }
    // The pixel format traits the translation used: integer formats always (sampled textures),
    // the others only with descriptors that use them.
    u32 format_traits = FileEnvironment::FormatTraitInteger;
    for (const auto* const info : info_ptrs) {
        if (!info) {
            continue;
        }
        if (std::ranges::any_of(info->image_descriptors,
                                [](const auto& desc) { return desc.is_byte_addressed; })) {
            format_traits |= FileEnvironment::FormatTraitByteAddressed;
        }
        if (!info->texture_buffer_descriptors.empty()) {
            format_traits |= FileEnvironment::FormatTraitSnorm;
        }
    }
    for (size_t i = 0; i < live.size(); ++i) {
        const char* why = "";
        if (!(*recorded)[i].AnswersHoldFor(*live[i], &why, format_traits)) {
            CountPrecompiledReason(why);
            return nullptr;
        }
    }
    std::array<bool, 5> present{};
    for (size_t stage = 0; stage < present.size(); ++stage) {
        present[stage] = info_ptrs[stage] != nullptr;
    }
    // The worker builds the SPIR-V the preparation step kept: the translation of the recorded
    // environments by this build, which the check above found to give the game state's.
    // EDEN_NX2_PRECOMPILED_VERIFY=1 translates them again instead and compares the infos.
    static const bool verify = [] {
        const char* value = std::getenv("EDEN_NX2_PRECOMPILED_VERIFY");
        return value != nullptr && value[0] == '1';
    }();
    bool kept_spirv = !verify;
    std::array<std::pair<u64, u32>, 5> spirv_blobs{};
    for (size_t stage = 0; stage < present.size() && kept_spirv; ++stage) {
        if (!present[stage]) {
            continue;
        }
        if (record.spirv[stage] == NoInfo || record.spirv[stage] >= precompiled_info_blobs.size()) {
            kept_spirv = false;
            break;
        }
        spirv_blobs[stage] = precompiled_info_blobs[record.spirv[stage]];
    }
    if (kept_spirv) {
        auto build_kept = [this, key = graphics_key, path = precompiled_infos_path, spirv_blobs,
                           present] {
            std::array<vk::ShaderModule, Maxwell::MaxShaderStage> modules;
            for (size_t stage = 0; stage < modules.size(); ++stage) {
                if (present[stage]) {
                    modules[stage] = BuildKeptSpirv(device, path, spirv_blobs[stage], key.Hash());
                }
            }
            precompiled_from_spirv.fetch_add(1, std::memory_order_relaxed);
            return modules;
        };
        auto pipeline = std::make_unique<GraphicsPipeline>(
            scheduler, buffer_cache, texture_cache, vulkan_pipeline_cache, &shader_notify, device,
            descriptor_pool, guest_descriptor_queue, descriptor_buffer_ring, &workers, nullptr,
            render_pass_cache, graphics_key,
            std::array<vk::ShaderModule, Maxwell::MaxShaderStage>{}, info_ptrs,
            std::move(build_kept), std::nullopt);
        pipeline->MarkPrecompiled();
        return pipeline;
    }
    // The worker translates the recorded environments, which give the same programs as the
    // game's state does; the infos drawn with must be the ones that translation gives.
    auto emit_stages = [this, key = graphics_key, recorded, infos, present] {
        ShaderPools pools;
        boost::container::static_vector<Shader::Environment*, Maxwell::MaxShaderProgram> env_ptrs;
        for (auto& env : *recorded) {
            env_ptrs.push_back(&env);
        }
        GraphicsStages stages;
        if (!TranslateGraphicsStages(pools, key, MakeSpan(env_ptrs), stages, false) ||
            stages.mesh_emulation) {
            throw Shader::LogicError("Precompiled pipeline {:016x} is not drawn natively",
                                     key.Hash());
        }
        std::array<vk::ShaderModule, Maxwell::MaxShaderStage> modules;
        Shader::Backend::Bindings binding;
        for (size_t stage = 0; stage < modules.size(); ++stage) {
            if (stages.runtime_infos[stage].has_value() != present[stage]) {
                throw Shader::LogicError("Precompiled pipeline {:016x}: other stages", key.Hash());
            }
            if (!present[stage]) {
                continue;
            }
            modules[stage] = EmitShaderModule(device, profile, *stages.runtime_infos[stage],
                                              stages.programs[stage + 1], binding,
                                              key.unique_hashes[stage + 1], stage);
            std::vector<u8> translated, stored;
            Shader::SerializeInfo(stages.programs[stage + 1].info, translated);
            Shader::SerializeInfo(infos[stage], stored);
            if (translated != stored) {
                throw Shader::LogicError(
                    "Precompiled pipeline {:016x}: stage {} infos differ from its translation",
                    key.Hash(), stage);
            }
        }
        precompiled_drawn.fetch_add(1, std::memory_order_relaxed);
        return modules;
    };
    auto pipeline = std::make_unique<GraphicsPipeline>(
        scheduler, buffer_cache, texture_cache, vulkan_pipeline_cache, &shader_notify, device,
        descriptor_pool, guest_descriptor_queue, descriptor_buffer_ring, &workers, nullptr,
        render_pass_cache, graphics_key, std::array<vk::ShaderModule, Maxwell::MaxShaderStage>{},
        info_ptrs, std::move(emit_stages), std::nullopt);
    pipeline->MarkPrecompiled();
    return pipeline;
}

ComputePipelineCacheEntry PipelineCache::CreatePrecompiledComputePipeline(
    const ComputePipelineCacheKey& key, ComputeEnvironment& env,
    const PrecompiledRecord** matched) {
    if (precompiled_compute_records.empty() || !PrecompiledFastPath()) {
        return {};
    }
    const u64 hash = key.Hash();
    if (precompiled_rejected.contains(hash)) {
        return {};
    }
    // One record for each storage pointer state the program was prepared in.
    const auto [first, last] = precompiled_compute_records.equal_range(hash);
    const char* first_reason = nullptr;
    for (auto it = first; it != last; ++it) {
        const PrecompiledRecord& record = it->second;
        if (record.key.size() != sizeof(key) ||
            std::memcmp(record.key.data(), &key, sizeof(key)) != 0) {
            continue;
        }
        if (record.env_count != 1 || record.infos[0] == NoInfo || record.spirv[0] == NoInfo ||
            record.spirv[0] >= precompiled_info_blobs.size()) {
            continue;
        }
        FileEnvironment recorded;
        Shader::Info info{};
        try {
            if (!precompiled_file.is_open()) {
                precompiled_file.open(precompiled_path, std::ios::binary);
                precompiled_file.exceptions(std::ifstream::failbit | std::ifstream::badbit);
            }
            precompiled_file.clear();
            const auto [env_offset, env_size] = precompiled_env_index.at(record.envs[0]);
            precompiled_file.seekg(static_cast<std::streamoff>(env_offset));
            recorded.Deserialize(precompiled_file,
                                 static_cast<std::streamoff>(env_offset + env_size));
            if (!precompiled_infos_file.is_open()) {
                precompiled_infos_file.open(precompiled_infos_path, std::ios::binary);
            }
            const auto [offset, size] = precompiled_info_blobs.at(record.infos[0]);
            auto bytes = VideoCommon::ShaderWarm::PreparedStore::ReadBlob(precompiled_infos_file,
                                                                          {offset, size});
            std::span<const u8> in{bytes};
            if (!Shader::DeserializeInfo(in, info) || !in.empty()) {
                if (!first_reason) {
                    first_reason = "infos unreadable or with layer emulation";
                }
                continue;
            }
        } catch (const std::exception&) {
            CountPrecompiledReason("files unreadable");
            precompiled_file.close();
            precompiled_infos_file.close();
            return {};
        }
        u32 format_traits = FileEnvironment::FormatTraitInteger;
        if (std::ranges::any_of(info.image_descriptors,
                                [](const auto& desc) { return desc.is_byte_addressed; })) {
            format_traits |= FileEnvironment::FormatTraitByteAddressed;
        }
        if (!info.texture_buffer_descriptors.empty()) {
            format_traits |= FileEnvironment::FormatTraitSnorm;
        }
        const char* why = "";
        if (!recorded.AnswersHoldFor(env, &why, format_traits)) {
            if (!first_reason) {
                first_reason = why;
            }
            continue;
        }
        auto emit_module = [this, path = precompiled_infos_path,
                            blob = precompiled_info_blobs[record.spirv[0]], hash] {
            auto module = BuildKeptSpirv(device, path, blob, hash);
            precompiled_from_spirv.fetch_add(1, std::memory_order_relaxed);
            return module;
        };
        ComputePipelineCacheEntry entry{
            .pipeline = std::make_unique<ComputePipeline>(
                device, scheduler, vulkan_pipeline_cache, descriptor_pool, guest_descriptor_queue,
                descriptor_buffer_ring, &workers, nullptr, &shader_notify, info, vk::ShaderModule{},
                key.unique_hash, std::move(emit_module)),
            // The live environment answered the same questions, as the translation would have.
            .null_pointer_slots = env.IsNativeProgram() ? env.NullStoragePointerSlots()
                                                        : std::vector<std::pair<u32, u32>>{},
            .mapped_pointers =
                env.IsNativeProgram() ? env.MappedStoragePointerKeys() : std::vector<u64>{}};
        entry.pipeline->MarkPrecompiled();
        if (matched) {
            *matched = &record;
        }
        return entry;
    }
    if (first_reason) {
        CountPrecompiledReason(first_reason);
    }
    return {};
}

void PipelineCache::AuditPrecompiled(std::span<Shader::Environment* const> envs, bool compute,
                                     u64 key_hash, const PrecompiledRecord* given) {
    const auto it = precompiled_records.find(key_hash);
    if (!given && (compute || it == precompiled_records.end())) {
        return;
    }
    const PrecompiledRecord& record = given ? *given : it->second;
    const auto differ = [&](size_t stage, const char* what) {
        if (audit_differ.fetch_add(1, std::memory_order_relaxed) < 32) {
            LOG_ERROR(Render_Vulkan, "Precompiled audit: {} pipeline {:016x} stage {}: {} differ",
                      compute ? "compute" : "graphics", key_hash, stage, what);
        }
    };
    try {
        std::array<std::vector<u32>, Maxwell::MaxShaderStage> live_code;
        std::array<std::vector<u8>, Maxwell::MaxShaderStage> live_info;
        std::array<bool, Maxwell::MaxShaderStage> present{};
        main_pools.ReleaseContents();
        if (compute) {
            Shader::Maxwell::Flow::CFG cfg{*envs[0], main_pools.flow_block,
                                           envs[0]->StartAddress()};
            auto program{
                TranslateProgram(main_pools.inst, main_pools.block, *envs[0], cfg, host_info)};
            live_code[0] = EmitSPIRV(profile, program);
            Shader::SerializeInfo(program.info, live_info[0]);
            present[0] = true;
        } else {
            GraphicsStages stages;
            if (!TranslateGraphicsStages(main_pools, graphics_key, envs, stages, false)) {
                differ(0, "the pipeline (skipped by its translation)");
                return;
            }
            Shader::Backend::Bindings binding;
            for (size_t stage = 0; stage < live_code.size(); ++stage) {
                if (!stages.runtime_infos[stage]) {
                    continue;
                }
                live_code[stage] = EmitPipelineShader(profile, *stages.runtime_infos[stage],
                                                      stages.programs[stage + 1], binding,
                                                      graphics_key.unique_hashes[stage + 1], {});
                Shader::SerializeInfo(stages.programs[stage + 1].info, live_info[stage]);
                present[stage] = true;
            }
        }
        bool same = true;
        for (size_t stage = 0; stage < present.size(); ++stage) {
            if (present[stage] != (record.infos[stage] != NoInfo)) {
                differ(stage, "stages");
                same = false;
                continue;
            }
            if (!present[stage]) {
                continue;
            }
            const auto [offset, size] = precompiled_info_blobs.at(record.infos[stage]);
            const auto kept_info = VideoCommon::ShaderWarm::PreparedStore::ReadBlob(
                precompiled_infos_file, {offset, size});
            if (kept_info != live_info[stage]) {
                differ(stage, "infos");
                same = false;
            }
            if (record.spirv[stage] != NoInfo &&
                ReadKeptSpirv(precompiled_infos_file,
                              precompiled_info_blobs.at(record.spirv[stage])) != live_code[stage]) {
                differ(stage, "SPIR-V words");
                same = false;
            }
        }
        if (same) {
            audit_same.fetch_add(1, std::memory_order_relaxed);
        }
    } catch (const std::exception& exception) {
        differ(0, exception.what());
    }
}

GraphicsPipeline* PipelineCache::ReplaceFailedPrecompiled() {
    precompiled_rejected.insert(graphics_key.Hash());
    precompiled_failed.fetch_add(1, std::memory_order_relaxed);
    if (const auto it = graphics_cache.find(graphics_key); it != graphics_cache.end()) {
        used_this_session.erase(it->second.get());
        evicted_pipelines.emplace_back(scheduler.CurrentTick(), std::move(it->second));
        graphics_cache.erase(it);
    }
    for (auto& [key, pipeline] : graphics_cache) {
        if (pipeline) {
            pipeline->ClearTransitions();
        }
    }
    current_pipeline = nullptr;
    return CurrentGraphicsPipelineSlowPath();
}

void PipelineCache::QueueVulkanPipelineCacheFlush() {
    if (!use_vulkan_pipeline_cache || vulkan_pipeline_cache_filename.empty()) {
        return;
    }
    if (++pipelines_since_flush < VULKAN_CACHE_FLUSH_PIPELINES) {
        return;
    }
    const auto now = std::chrono::steady_clock::now();
    const auto megabytes = last_cache_size.load(std::memory_order_relaxed) / (1024 * 1024);
    const std::chrono::seconds interval{
        std::max<size_t>(VULKAN_CACHE_FLUSH_MIN_SECONDS, megabytes)};
    if (last_flush.time_since_epoch().count() != 0 && now - last_flush < interval) {
        return;
    }
    if (flush_in_flight.exchange(true, std::memory_order_acq_rel)) {
        return;
    }
    pipelines_since_flush = 0;
    last_flush = now;
    serialization_thread.QueueWork([this] {
        SerializeVulkanPipelineCache(vulkan_pipeline_cache_filename, vulkan_pipeline_cache,
                                     CACHE_VERSION);
        size_t size = 0;
        vulkan_pipeline_cache.Read(&size, nullptr);
        last_cache_size.store(size, std::memory_order_relaxed);
        flush_in_flight.store(false, std::memory_order_release);
    });
}

void PipelineCache::QueueVulkanPipelineCacheSave() {
    if (!use_vulkan_pipeline_cache || vulkan_pipeline_cache_filename.empty()) {
        return;
    }
    // Not skipped when a timed flush is in flight: that one may have read the cache before the
    // stage's last builds. The serialization thread runs the two one after the other.
    serialization_thread.QueueWork([this] {
        SerializeVulkanPipelineCache(vulkan_pipeline_cache_filename, vulkan_pipeline_cache,
                                     CACHE_VERSION);
        size_t size = 0;
        vulkan_pipeline_cache.Read(&size, nullptr);
        last_cache_size.store(size, std::memory_order_relaxed);
        LOG_INFO(Render_Vulkan, "Driver pipeline cache saved after the shader precache, {} bytes",
                 size);
    });
}

namespace {
// EDEN_NX2_PIPELINE_DIFF=1: for a new graphics pipeline whose shaders an earlier one uses too,
// log which parts of the fixed state differ from that one (finds state that splits the same
// draws into many pipelines, each translated on the GPU thread).
void ReportNewPipelineKey(const GraphicsPipelineCacheKey& key) {
    static const bool enabled = std::getenv("EDEN_NX2_PIPELINE_DIFF") != nullptr;
    if (!enabled) {
        return;
    }
    static std::unordered_map<u64, GraphicsPipelineCacheKey> last_by_shaders;
    static u32 logged{};
    const u64 shaders = Common::CityHash64(reinterpret_cast<const char*>(key.unique_hashes.data()),
                                           sizeof(key.unique_hashes));
    const auto [it, first] = last_by_shaders.try_emplace(shaders, key);
    if (first) {
        return;
    }
    const FixedPipelineState& now = key.state;
    const FixedPipelineState& before = it->second.state;
    std::string diffs;
    if (now.raw1 != before.raw1) diffs += fmt::format(" raw1^{:#x}", now.raw1 ^ before.raw1);
    if (now.raw2 != before.raw2) diffs += fmt::format(" raw2^{:#x}", now.raw2 ^ before.raw2);
    if (now.color_formats != before.color_formats) {
        diffs += fmt::format(" color_formats {:02x} -> {:02x}", fmt::join(before.color_formats, ""),
                             fmt::join(now.color_formats, ""));
    }
    if (now.alpha_test_ref != before.alpha_test_ref) {
        diffs += fmt::format(" alpha_test_ref {:#x} -> {:#x} (func {})", before.alpha_test_ref,
                             now.alpha_test_ref, now.alpha_test_func.Value());
    }
    if (now.point_size != before.point_size) {
        diffs += fmt::format(" point_size {:#x} -> {:#x}", before.point_size, now.point_size);
    }
    if (now.viewport_swizzles != before.viewport_swizzles) diffs += " viewport_swizzles";
    if (now.attribute_types != before.attribute_types) {
        diffs += fmt::format(" attribute_types^{:#x}", now.attribute_types ^ before.attribute_types);
    }
    if (std::memcmp(&now.dynamic_state, &before.dynamic_state, sizeof(now.dynamic_state)) != 0) {
        diffs += " dynamic_state";
    }
    if (now.driver_id != before.driver_id || now.driver_version != before.driver_version) {
        diffs += " driver";
    }
    const size_t size = now.Size();
    if (diffs.empty() && size > offsetof(FixedPipelineState, attachments)) {
        diffs = fmt::format(" past attachments (key size {})", size);
    }
    it->second = key;
    if (logged++ < 400) {
        LOG_INFO(Render_Vulkan, "New pipeline for known shaders {:016x} differs in:{}", shaders,
                 diffs.empty() ? std::string{" nothing in the key prefix"} : diffs);
    }
}

// Without VK_EXT_mesh_shader: logs, once per shader set, what a skipped task/mesh pipeline would
// need from an emulation (task stage, per-primitive outputs, topology, output sizes).
template <typename Programs>
void ReportUnsupportedMeshPipeline(const GraphicsPipelineCacheKey& key, const Programs& programs) {
    static std::mutex mutex;
    static std::unordered_set<u64> seen;
    static u32 with_task{}, per_primitive{}, non_triangle{}, total{};
    u64 shaders{};
    for (const u64 hash : key.unique_hashes) {
        shaders = shaders * 31 + hash;
    }
    std::scoped_lock lock{mutex};
    if (!seen.insert(shaders).second) {
        return;
    }
    const Shader::IR::Program* task{};
    const Shader::IR::Program* mesh{};
    bool has_pixel{};
    for (size_t index = 0; index < programs.size(); ++index) {
        if (key.unique_hashes[index] == 0) {
            continue;
        }
        const auto& program{programs[index]};
        if (program.stage == Shader::Stage::Task) {
            task = &program;
        } else if (program.stage == Shader::Stage::Mesh) {
            mesh = &program;
        } else if (program.stage == Shader::Stage::Fragment) {
            has_pixel = true;
        }
    }
    ++total;
    if (mesh == nullptr) {
        LOG_WARNING(Render_Vulkan, "Mesh emulation survey: pipeline {:016x} without a mesh stage",
                    shaders);
        return;
    }
    const auto& info{mesh->mesh};
    u32 position{}, generic{}, other{};
    std::string others;
    for (const u32 line : info.vertex_lines) {
        const auto attribute{static_cast<Shader::IR::Attribute>(line)};
        if (attribute >= Shader::IR::Attribute::PositionX &&
            attribute <= Shader::IR::Attribute::PositionW) {
            ++position;
        } else if (Shader::IR::IsGeneric(attribute)) {
            ++generic;
        } else {
            ++other;
            others += fmt::format(" {}", line);
        }
    }
    with_task += task != nullptr;
    per_primitive += !info.primitive_lines.empty();
    non_triangle += info.topology != 4;
    LOG_WARNING(Render_Vulkan,
                "Mesh emulation survey #{}: task={} (payload {} words, wg {}x{}x{}) mesh wg "
                "{}x{}x{} max_vertices={} max_primitives={} topology={} vertex_lines={} "
                "(position {} generic {} other {}:{}) primitive_lines={} pixel={} | totals: "
                "task {} per-primitive {} non-triangle {}",
                total, task != nullptr, info.payload_words, task ? task->workgroup_size[0] : 0,
                task ? task->workgroup_size[1] : 0, task ? task->workgroup_size[2] : 0,
                mesh->workgroup_size[0], mesh->workgroup_size[1], mesh->workgroup_size[2],
                info.max_vertices, info.max_primitives, info.topology, info.vertex_lines.size(),
                position, generic, other, others, info.primitive_lines.size(), has_pixel,
                with_task, per_primitive, non_triangle);
}
} // Anonymous namespace

GraphicsPipeline* PipelineCache::CurrentGraphicsPipelineSlowPath() {
    const auto [pair, is_new]{graphics_cache.try_emplace(graphics_key)};
    auto& pipeline{pair->second};
    if (is_new) {
        ReportNewPipelineKey(graphics_key);
        pipeline = background_pipelines.Take(graphics_key);
        if (pipeline) {
            ++follow_on_taken;
        } else {
            ++built_on_demand;
            pipeline = CreateGraphicsPipeline();
        }
    }
    if (!pipeline) {
        return nullptr;
    }
    constexpr auto save_interval = std::chrono::seconds{15};
    if (used_this_session.insert(pipeline.get()).second) {
        FollowOn(graphics_key.Hash());
        if (std::chrono::steady_clock::now() - used_pipelines_saved > save_interval) {
            SaveUsedPipelines();
        }
    }
    if (current_pipeline) {
        current_pipeline->AddTransition(pipeline.get());
    }
    current_pipeline = pipeline.get();
    if (current_pipeline->IsPrecompiled() && current_pipeline->Failed()) {
        return ReplaceFailedPrecompiled();
    }
    current_pipeline->MarkUsed(frame_count);
    key_pipeline = current_pipeline;
    return BuiltPipeline(current_pipeline);
}

void PipelineCache::TickFrame() {
    ++frame_count;
    native_wait_budget_us =
        static_cast<s64>(Settings::values.native_pipeline_wait_ms.GetValue()) * 1000;
    VideoCommon::Nx2BindMemo::Report();
    while (!evicted_pipelines.empty() && scheduler.IsFree(evicted_pipelines.front().first)) {
        evicted_pipelines.pop_front();
    }
    while (!retired_compute_pipelines.empty() &&
           scheduler.IsFree(retired_compute_pipelines.front().first) &&
           retired_compute_pipelines.front().second->IsBuildTaskDone()) {
        retired_compute_pipelines.pop_front();
    }
    constexpr u64 EvictCheckFrames = 600;
    if (frame_count % EvictCheckFrames == 0) {
        EvictUnusedPipelines();
    }
}

void PipelineCache::EvictUnusedPipelines() {
    // Building a pipeline again on MoltenVK means a Metal compile mid-game, far dearer than the
    // 2-3 ms below, so none are evicted there.
    if (device.IsMoltenVK()) {
        return;
    }
    // EDEN_NX2_PRELOAD_ALL_PIPELINES asked for every cached pipeline built: evicting the ones not
    // used yet threw the preload away after 9000 frames (Ocarina: 22,510 of 24,285, then rebuilt
    // one by one in new places).
    if (PreloadAllPipelines()) {
        return;
    }
    // Each built pipeline holds ~0.45 MB of driver memory and none was ever destroyed, so it grew
    // with every place a session went (title 1: 1,500 on its title screen alone). One unused for
    // this long is destroyed; it takes 2-3 ms to build again from the driver's pipeline cache.
    // EDEN_NX2_PIPELINE_EVICT_FRAMES=<frames> overrides the age; 0 never evicts. Ocarina keeps
    // ~1,700 pipelines (~0.8 GB): evicting 961 at once stalled the GPU thread ~300 ms, and the
    // evicted ones are rebuilt when the player returns. A desktop can keep them all.
    static const u64 EvictAfterFrames = [] {
        const char* const value = std::getenv("EDEN_NX2_PIPELINE_EVICT_FRAMES");
        return value != nullptr && *value != '\0' ? std::strtoull(value, nullptr, 10) : u64{9000};
    }();
    if (EvictAfterFrames == 0 || frame_count <= EvictAfterFrames) {
        return;
    }
    const u64 cutoff = frame_count - EvictAfterFrames;
    const u64 tick = scheduler.CurrentTick();
    // ~0.9 GB of driver memory at ~0.45 MB each.
    constexpr size_t MaxUnusedPrebuiltPipelines = 2000;
    size_t kept_prebuilt = 0;
    size_t evicted = 0;
    for (auto it = graphics_cache.begin(); it != graphics_cache.end();) {
        GraphicsPipeline* const pipeline = it->second.get();
        if (pipeline == nullptr || pipeline == current_pipeline || !pipeline->IsBuilt() ||
            pipeline->LastUse() >= cutoff) {
            ++it;
            continue;
        }
        // A pipeline built from the disk cache at boot that no draw has used yet stays, up to a
        // budget: evicting it throws the prebuild away, and the game translates it again on the
        // GPU thread when it reaches that content (OoT: 1174 of 1346 evicted on the title screen,
        // 684 rebuilt in 10 s on entering the field). Past the budget they go like the rest, so a
        // large cache can't hold their driver memory for the whole session.
        if (!pipeline->WasUsed() && kept_prebuilt < MaxUnusedPrebuiltPipelines) {
            ++kept_prebuilt;
            ++it;
            continue;
        }
        if (used_this_session.erase(pipeline) != 0) {
            used_and_evicted.insert(it->first.Hash());
        }
        evicted_pipelines.emplace_back(tick, std::move(it->second));
        it = graphics_cache.erase(it);
        ++evicted;
    }
    if (evicted == 0) {
        return;
    }
    for (auto& [key, pipeline] : graphics_cache) {
        if (pipeline) {
            pipeline->ClearTransitions();
        }
    }
    LOG_INFO(Render_Vulkan, "Evicted {} graphics pipelines unused for {} frames, {} left", evicted,
             EvictAfterFrames, graphics_cache.size());
}

/// Returns `pipeline` when it is built or the current draw must wait for it, and nullptr when the
/// draw may be skipped while it builds (asynchronous shaders; native draws only with
/// EDEN_NX2_ASYNC_DISCARD=1). Each nullptr returned for an unbuilt pipeline is counted for the log.
GraphicsPipeline* PipelineCache::BuiltPipeline(GraphicsPipeline* pipeline) noexcept {
    if (pipeline->IsBuilt()) {
        return pipeline;
    }
    if (!use_asynchronous_shaders) {
        return pipeline;
    }
    // EDEN_NX2_ASYNC_DISCARD=1: skip native draws whose pipeline is still building instead of
    // waiting for it (experimental; the draw may be missing for a frame or more).
    /// Reads EDEN_NX2_ASYNC_DISCARD once and logs when it is on.
    static const bool discard_native = [] {
        const char* value = std::getenv("EDEN_NX2_ASYNC_DISCARD");
        const bool enabled = value != nullptr && value[0] == '1';
        if (enabled) {
            LOG_INFO(Render_Vulkan, "EDEN_NX2_ASYNC_DISCARD: native draws skip unbuilt pipelines");
        }
        return enabled;
    }();
    const auto& draw_state = maxwell3d->draw_manager.draw_state;
    const bool native = maxwell3d->GetNativeMacroEngine().native_turing;
    // The environment variable wins over the setting; the setting is read per call.
    NativePendingMode mode = NativePendingMode::Wait;
    if (native) {
        switch (Settings::values.native_async_draws.GetValue()) {
        case Settings::NativeAsyncDraws::Bounded:
            mode = NativePendingMode::Bounded;
            break;
        case Settings::NativeAsyncDraws::Skip:
            mode = NativePendingMode::Skip;
            break;
        default:
            break;
        }
        if (discard_native) {
            mode = NativePendingMode::Skip;
        }
    }
    // Native draws are judged by the heuristic only when a mode other than Wait is chosen.
    const bool can_discard = CanDiscardPendingDraw(
        native, maxwell3d->regs.zeta_enable != 0, draw_state.draw_element_count,
        draw_state.draw_element_count, native && mode != NativePendingMode::Wait);
    bool discard = can_discard;
    if (native) {
        switch (DecideNativePending(mode, can_discard, native_wait_budget_us)) {
        case NativeDrawAction::Wait:
            discard = false;
            break;
        case NativeDrawAction::Skip:
            discard = true;
            break;
        case NativeDrawAction::WaitTimed: {
            const auto begin = std::chrono::steady_clock::now();
            const bool built =
                pipeline->WaitBuilt(std::chrono::microseconds{native_wait_budget_us});
            const auto spent = std::chrono::duration_cast<std::chrono::microseconds>(
                                   std::chrono::steady_clock::now() - begin)
                                   .count();
            native_wait_budget_us -= spent;
            native_wait_used_us += spent;
            static auto window = std::chrono::steady_clock::now();
            if (std::chrono::steady_clock::now() - window > std::chrono::seconds{10}) {
                LOG_INFO(Render_Vulkan,
                         "Bounded pipeline wait: {:.1f} ms spent in the last 10 s (budget {} ms "
                         "per frame)",
                         static_cast<double>(native_wait_used_us) / 1000.0,
                         Settings::values.native_pipeline_wait_ms.GetValue());
                native_wait_used_us = 0;
                window = std::chrono::steady_clock::now();
            }
            if (built) {
                return pipeline;
            }
            discard = true;
            break;
        }
        }
    }
    if (discard) {
        static std::atomic<u64> skipped{};
        if ((skipped.fetch_add(1, std::memory_order_relaxed) & 255) == 0) {
            // Counts lookups, not draws: a draw state can look its pipeline up more than once.
            LOG_INFO(Render_Vulkan,
                     "Async shaders: {} pipeline lookups skipped for unbuilt pipelines",
                     skipped.load(std::memory_order_relaxed));
        }
    }
    return discard ? nullptr : pipeline;
}

bool PipelineCache::TranslateGraphicsStages(ShaderPools& pools, const GraphicsPipelineCacheKey& key,
                                            std::span<Shader::Environment* const> envs,
                                            GraphicsStages& out, bool reference_spirv) {
    auto hash = key.Hash();
    size_t env_index{0};
    auto& programs{out.programs};
    const bool uses_vertex_a{key.unique_hashes[0] != 0};
    const bool uses_vertex_b{key.unique_hashes[1] != 0};

    // Layer passthrough generation for devices without VK_EXT_shader_viewport_index_layer
    Shader::IR::Program* layer_source_program{};
    bool mesh_unsupported{};

    for (size_t index = 0; index < Maxwell::MaxShaderProgram; ++index) {
        const bool is_emulated_stage = layer_source_program != nullptr &&
                                       index == static_cast<u32>(Maxwell::ShaderType::Geometry);
        if (key.unique_hashes[index] == 0 && is_emulated_stage) {
            auto topology = MaxwellToOutputTopology(key.state.topology);
            programs[index] = GenerateGeometryPassthrough(pools.inst, pools.block, host_info,
                                                          *layer_source_program, topology);
            continue;
        }
        if (key.unique_hashes[index] == 0) {
            continue;
        }
        Shader::Environment& env{*envs[env_index]};
        ++env_index;
        if (index == static_cast<size_t>(Maxwell::ShaderType::Geometry) &&
            !device.IsGeometryShaderSupported() && env.IsNativeProgram()) {
            // A native (Switch 2) geometry program on a device without geometry shaders: the
            // pipeline cannot run it, so its draws are skipped.
            static std::atomic<bool> logged{};
            if (!logged.exchange(true)) {
                LOG_WARNING(Render_Vulkan, "Skipping native geometry shader pipelines: the device has no geometry shaders");
            }
            return false;
        }
        // Without VK_EXT_mesh_shader the driver cannot build a task/mesh pipeline; the draw is
        // skipped (see the pipeline check in the rasterizer). The stages are still translated
        // so the skipped pipeline's needs can be reported below.
        if ((env.ShaderStage() == Shader::Stage::Task ||
             env.ShaderStage() == Shader::Stage::Mesh) &&
            !device.IsExtMeshShaderSupported()) {
            static std::atomic<bool> logged{};
            if (!logged.exchange(true, std::memory_order_relaxed)) {
                LOG_WARNING(Render_Vulkan,
                            "{} has no VK_EXT_mesh_shader: pipelines with task/mesh shaders are "
                            "skipped, so the geometry they draw (such as foliage and props) will "
                            "be missing",
                            device.GetModelName());
            }
            mesh_unsupported = true;
        }

        const u32 cfg_offset{static_cast<u32>(env.StartAddress() + sizeof(Shader::ProgramHeader))};
        Shader::Maxwell::Flow::CFG cfg(env, pools.flow_block, cfg_offset, index == 0);
        if (!uses_vertex_a || index != 1) {
            // Normal path
            programs[index] = TranslateProgram(pools.inst, pools.block, env, cfg, host_info);
        } else {
            // VertexB path when VertexA is present.
            auto& program_va{programs[0]};
            auto program_vb{TranslateProgram(pools.inst, pools.block, env, cfg, host_info)};
            programs[index] = MergeDualVertexPrograms(program_va, program_vb, env, host_info);
        }

        // Canonical native keys keep the attribute types the header's input map lists. A program
        // that loads another one would read it as disabled: this pipeline is dropped and the
        // program's keys keep every attribute type from now on.
        if (index == 1 && key.state.dynamic_vertex_input && env.IsNativeProgram() &&
            env.ShaderStage() == Shader::Stage::VertexB) {
            u32 loads = 0;
            for (u32 attribute = 0; attribute < 32; ++attribute) {
                if (programs[index].info.loads.Generic(attribute)) {
                    loads |= 1U << attribute;
                }
            }
            const u32 listed = VideoCommon::GenericReadMask(env.SPH());
            if ((loads & ~listed) != 0 && !IsUnmaskedVertexProgram(key.unique_hashes[1])) {
                {
                    std::scoped_lock lock{unmasked_mutex};
                    unmasked_vertex_programs.insert(key.unique_hashes[1]);
                    any_unmasked_vertex_program.store(true, std::memory_order_release);
                }
                LOG_ERROR(Render_Vulkan,
                          "Vertex program {:016x} loads attributes {:#010x} its header does not "
                          "list ({:#010x}); its pipelines now keep every attribute type",
                          key.unique_hashes[1], loads, listed);
                return false;
            }
        }

        if (device.GetDriverID() != VK_DRIVER_ID_NVIDIA_PROPRIETARY && Shader::Optimization::HasBrokenPattern(programs[index])) {
            LOG_WARNING(Render_Vulkan, "0x{:016x}: Skipping shader with known issues", key.unique_hashes[index]);
            return false;
        }

        if (Settings::values.dump_guest_shaders) {
            env.Dump(hash, key.unique_hashes[index]);
        }

        if (programs[index].info.requires_layer_emulation) {
            layer_source_program = &programs[index];
        }
    }
    // Mesh emulation without VK_EXT_mesh_shader (prototype): a mesh program, after an optional
    // task program, drawing triangles without per-primitive outputs, before a pixel program only.
    auto& mesh_emulation{out.mesh_emulation};
    size_t mesh_index{};
    if (mesh_unsupported) {
        ReportUnsupportedMeshPipeline(key, programs);
        const bool has_task{key.unique_hashes[1] != 0 &&
                            programs[1].stage == Shader::Stage::Task};
        for (size_t index = 1; index < Maxwell::MaxShaderProgram; ++index) {
            if (key.unique_hashes[index] != 0 && programs[index].stage == Shader::Stage::Mesh) {
                mesh_index = index;
                break;
            }
        }
        bool eligible{mesh_index != 0 && (mesh_index == 1 || has_task) &&
                      layer_source_program == nullptr};
        const Shader::IR::Program& mesh{programs[mesh_index]};
        eligible = eligible && mesh.mesh.topology == 4 && mesh.mesh.primitive_lines.empty() &&
                   !mesh.mesh.vertex_lines.empty() && mesh.mesh.max_vertices != 0 &&
                   mesh.mesh.max_primitives != 0;
        for (size_t index = 1; index < Maxwell::MaxShaderProgram && eligible; ++index) {
            const bool emulated{index == mesh_index || (has_task && index == 1)};
            eligible = emulated || key.unique_hashes[index] == 0 ||
                       programs[index].stage == Shader::Stage::Fragment;
        }
        if (!eligible) {
            return false;
        }
        // The output buffer's binding follows every program's (unified bindings, one per
        // descriptor element, as the SPIR-V backend numbers them). The task and mesh programs
        // share the compute stage, so their descriptors and the emulation's own buffers count
        // against one set of per-stage limits.
        u32 output_binding{};
        u32 compute_storage_buffers{has_task ? 4U : 1U};
        u32 compute_sampled_images{};
        u32 compute_resources{compute_storage_buffers};
        for (size_t index = 1; index < Maxwell::MaxShaderProgram; ++index) {
            if (key.unique_hashes[index] == 0) {
                continue;
            }
            const Shader::Info& info{programs[index].info};
            const bool emulated{index == mesh_index || (has_task && index == 1)};
            const auto add = [&](const auto& descriptors, u32* category) {
                const u32 count{Shader::NumDescriptors(descriptors)};
                output_binding += count;
                if (emulated) {
                    compute_resources += count;
                    if (category) {
                        *category += count;
                    }
                }
            };
            add(info.constant_buffer_descriptors, nullptr);
            add(info.storage_buffers_descriptors, &compute_storage_buffers);
            add(info.texture_buffer_descriptors, &compute_sampled_images);
            add(info.image_buffer_descriptors, nullptr);
            add(info.texture_descriptors, &compute_sampled_images);
            add(info.image_descriptors, nullptr);
        }
        if (compute_storage_buffers > device.GetMaxPerStageDescriptorStorageBuffers() ||
            compute_sampled_images > device.GetMaxPerStageDescriptorSampledImages() ||
            compute_resources > device.GetMaxPerStageResources()) {
            LOG_WARNING(Render_Vulkan,
                        "Mesh emulation skipped: {} storage buffers, {} sampled images and {} "
                        "resources exceed the compute stage limits",
                        compute_storage_buffers, compute_sampled_images, compute_resources);
            return false;
        }
        mesh_emulation.emplace();
        mesh_emulation->output_binding = output_binding;
        if (has_task) {
            // After the output: the task payload, offset and argument buffers.
            mesh_emulation->has_task = true;
            mesh_emulation->mesh_stage = static_cast<u32>(mesh_index - 1);
            mesh_emulation->payload_stride = (std::max)(programs[1].mesh.payload_words, 2U);
            mesh_emulation->payload_binding = output_binding + 1;
            mesh_emulation->offsets_binding = output_binding + 2;
            mesh_emulation->args_binding = output_binding + 3;
            mesh_emulation->distribution_spirv =
                Shader::Backend::SPIRV::EmitMeshEmulationDistribution(profile);
        }
        mesh_emulation->block_words = Shader::Backend::SPIRV::MeshEmulationBlockWords(mesh.mesh);
        mesh_emulation->max_primitives = mesh.mesh.max_primitives;
        mesh_emulation->max_vertices = mesh.mesh.max_vertices;
        mesh_emulation->vertex_lines = static_cast<u32>(mesh.mesh.vertex_lines.size());
        for (u32 line = 0; line < mesh.mesh.vertex_lines.size(); ++line) {
            const u32 attribute{mesh.mesh.vertex_lines[line]};
            const u32 position_x{static_cast<u32>(Shader::IR::Attribute::PositionX)};
            if (attribute >= position_x && attribute < position_x + 4) {
                mesh_emulation->position_lines[attribute - position_x] = line;
            }
        }
        static std::atomic<u32> emulated{};
        if (const u32 count{++emulated}; count <= 64) {
            // The input region addresses the mesh program reads (0 is its workgroup index).
            std::string reads;
            for (const Shader::IR::Block* block : mesh.blocks) {
                for (const Shader::IR::Inst& inst : block->Instructions()) {
                    if (inst.GetOpcode() != Shader::IR::Opcode::IsbeRead) {
                        continue;
                    }
                    const Shader::IR::Value address{inst.Arg(0)};
                    reads += address.IsImmediate() ? fmt::format(" {:#x}", address.U32())
                                                   : std::string{" dynamic"};
                }
            }
            LOG_INFO(Render_Vulkan,
                     "Mesh emulation #{}: pipeline {:016x}, {} vertices x {} lines, {} triangles "
                     "per workgroup, output binding {}, task {} (payload {} words), input "
                     "reads:{}",
                     count, hash, mesh.mesh.max_vertices, mesh.mesh.vertex_lines.size(),
                     mesh.mesh.max_primitives, output_binding, has_task,
                     mesh_emulation->payload_stride, reads.empty() ? " none" : reads);
        }
    }
    // A tessellation pipeline the driver would refuse is skipped like a missing mesh pipeline:
    // the draw is dropped (see the pipeline check in the rasterizer) instead of failing the build.
    {
        const Shader::IR::Program* control{};
        const Shader::IR::Program* evaluation{};
        for (size_t index = 0; index < Maxwell::MaxShaderProgram; ++index) {
            if (key.unique_hashes[index] == 0) {
                continue;
            }
            if (programs[index].stage == Shader::Stage::TessellationControl) {
                control = &programs[index];
            } else if (programs[index].stage == Shader::Stage::TessellationEval) {
                evaluation = &programs[index];
            }
        }
        if (control != nullptr || evaluation != nullptr) {
            const u32 max_patch_size = device.GetMaxTessellationPatchSize();
            const char* reason = nullptr;
            if (!device.IsTessellationSupported()) {
                reason = "the device has no tessellation shaders";
            } else if (control == nullptr || evaluation == nullptr) {
                reason = "it has only one of the control and evaluation stages";
            } else if (control->invocations == 0 || control->invocations > max_patch_size) {
                reason = "the control shader's output vertex count is out of range";
            }
            if (reason != nullptr) {
                LOG_WARNING(Render_Vulkan,
                            "Skipping tessellation pipeline: {} (tcs={} tes={}, tcs output "
                            "vertices {}, patch control points {}, device max patch size {})",
                            reason, control != nullptr, evaluation != nullptr,
                            control != nullptr ? control->invocations : 0,
                            key.state.patch_control_points_minus_one.Value() + 1, max_patch_size);
                return false;
            }
        }
    }
    // Each stage stays within the limit on its own; together they share one descriptor set.
    if (const u32 set_limit = host_info.max_descriptor_set_storage_buffers; set_limit != 0) {
        size_t storage_buffers{};
        for (size_t index = uses_vertex_a && uses_vertex_b ? 1 : 0;
             index < Maxwell::MaxShaderProgram; ++index) {
            if (key.unique_hashes[index] != 0) {
                storage_buffers +=
                    Shader::NumDescriptors(programs[index].info.storage_buffers_descriptors);
            }
        }
        // The fallback binds its output buffer (and, for task draws, payload,
        // offsets and indirect arguments) into the same descriptor set.
        const size_t auxiliary_buffers{
            mesh_emulation ? (mesh_emulation->has_task ? 4U : 1U) : 0U};
        if (!FitsStorageBufferSetLimit(storage_buffers, auxiliary_buffers, set_limit)) {
            throw Shader::NotImplementedException(
                "Binding {} storage buffers ({} for mesh fallback) to a descriptor set limited to {}",
                storage_buffers + auxiliary_buffers, auxiliary_buffers, set_limit);
        }
    }
    const Shader::IR::Program* previous_stage{};
    Shader::Backend::Bindings binding;
    for (size_t index = uses_vertex_a && uses_vertex_b ? 1 : 0; index < Maxwell::MaxShaderProgram;
         ++index) {
        const bool is_emulated_stage = layer_source_program != nullptr &&
                                       index == static_cast<u32>(Maxwell::ShaderType::Geometry);
        if (key.unique_hashes[index] == 0 && !is_emulated_stage) {
            continue;
        }
        UNIMPLEMENTED_IF(index == 0);

        Shader::IR::Program& program{programs[index]};
        const size_t stage_index{index - 1};

        auto runtime_info{MakeRuntimeInfo(programs, key, program, previous_stage, device)};
        if (mesh_emulation && (program.stage == Shader::Stage::Mesh ||
                               program.stage == Shader::Stage::Task)) {
            runtime_info.mesh_as_compute = true;
            runtime_info.mesh_output_binding = mesh_emulation->output_binding;
            if (mesh_emulation->has_task) {
                runtime_info.mesh_task_payload_binding = mesh_emulation->payload_binding;
                runtime_info.mesh_task_offsets_binding = mesh_emulation->offsets_binding;
                runtime_info.mesh_task_args_binding = mesh_emulation->args_binding;
                runtime_info.mesh_task_payload_stride = mesh_emulation->payload_stride;
            }
            if (program.stage == Shader::Stage::Mesh) {
                mesh_emulation->vertex_spirv = Shader::Backend::SPIRV::EmitMeshEmulationVertex(
                    profile, program.mesh, mesh_emulation->output_binding,
                    mesh_emulation->has_task ? mesh_emulation->args_binding : ~0U,
                    runtime_info.convert_depth_mode && !profile.support_native_ndc);
            }
        }
        ConvertLegacyToGeneric(program, runtime_info);
        // EDEN_NX2_MESH_EMULATION_DUMP=<dir>: write the emulation's vertex shader for spirv-val.
        // (Emitting the mesh program a second time here to dump its compute shader crashed the
        // build that followed, so only the vertex shader is written.)
        if (mesh_emulation && program.stage == Shader::Stage::Mesh) {
            if (const char* const dump{std::getenv("EDEN_NX2_MESH_EMULATION_DUMP")}) {
                const auto& code{mesh_emulation->vertex_spirv};
                std::ofstream file{fmt::format("{}/{:016x}.vert.spv", dump, hash),
                                   std::ios::binary};
                file.write(reinterpret_cast<const char*>(code.data()),
                           static_cast<std::streamsize>(code.size() * sizeof(u32)));
            }
        }
        out.runtime_infos[stage_index] = runtime_info;
        if (reference_spirv && VideoCommon::Nx2Perf::Verify()) {
            out.reference_spirv[stage_index] = EmitSPIRV(profile, runtime_info, program, binding);
        }
        previous_stage = &program;
    }
    return true;
}

std::unique_ptr<GraphicsPipeline> PipelineCache::CreateGraphicsPipeline(
    ShaderPools& pools, const GraphicsPipelineCacheKey& key,
    std::span<Shader::Environment* const> envs, PipelineStatistics* statistics,
    bool build_in_parallel, std::shared_ptr<ShaderPools> emit_pools,
    std::unique_ptr<EmissionBudget::Reservation> reservation,
    std::array<std::vector<u32>, Maxwell::MaxShaderStage>* spirv_out) try {
    LOG_DEBUG(Render_Vulkan, "{:#016x}", key.Hash());
    GraphicsStages stages;
    if (!TranslateGraphicsStages(pools, key, envs, stages, emit_pools != nullptr)) {
        return nullptr;
    }
    std::array<const Shader::Info*, Maxwell::MaxShaderStage> infos{};
    for (size_t stage = 0; stage < infos.size(); ++stage) {
        if (stages.runtime_infos[stage]) {
            infos[stage] = &stages.programs[stage + 1].info;
        }
    }
    std::array<vk::ShaderModule, Maxwell::MaxShaderStage> modules;
    std::function<std::array<vk::ShaderModule, Maxwell::MaxShaderStage>()> emit_stages;
    if (emit_pools) {
        auto job = std::make_shared<DeferredStages>();
        job->reservation = std::move(reservation);
        job->pools = emit_pools;
        job->programs = std::move(stages.programs);
        job->runtime_infos = std::move(stages.runtime_infos);
        job->reference_spirv = std::move(stages.reference_spirv);
        for (size_t stage = 0; stage < infos.size(); ++stage) {
            if (infos[stage]) {
                infos[stage] = &job->programs[stage + 1].info;
            }
        }
        emit_stages = [this, job = std::move(job), hashes = key.unique_hashes]() mutable {
            const auto begin = std::chrono::steady_clock::now();
            std::array<vk::ShaderModule, Maxwell::MaxShaderStage> stage_modules;
            Shader::Backend::Bindings stage_binding;
            for (size_t stage = 0; stage < stage_modules.size(); ++stage) {
                if (job->runtime_infos[stage]) {
                    stage_modules[stage] =
                        EmitShaderModule(device, profile, *job->runtime_infos[stage],
                                         job->programs[stage + 1], stage_binding,
                                         hashes[stage + 1], stage, job->reference_spirv[stage]);
                }
            }
            auto completed_pools = std::move(job->pools);
            job.reset();
            completed_pools->ReleaseContents();
            {
                std::scoped_lock lock{spare_pools_mutex};
                if (spare_pools.size() < 4) {
                    spare_pools.push_back(std::move(completed_pools));
                }
            }
            AddRuntimeSpirvTime(1, begin);
            return stage_modules;
        };
    } else {
        Shader::Backend::Bindings binding;
        for (size_t stage = 0; stage < modules.size(); ++stage) {
            if (!stages.runtime_infos[stage]) {
                continue;
            }
            const auto begin = std::chrono::steady_clock::now();
            modules[stage] =
                EmitShaderModule(device, profile, *stages.runtime_infos[stage],
                                 stages.programs[stage + 1], binding, key.unique_hashes[stage + 1],
                                 stage, {}, spirv_out ? &(*spirv_out)[stage] : nullptr);
            if (build_in_parallel) {
                AddRuntimeSpirvTime(0, begin);
            }
        }
    }
    Common::ThreadWorker* const thread_worker{build_in_parallel ? &workers : nullptr};
    return std::make_unique<GraphicsPipeline>(
        scheduler, buffer_cache, texture_cache, vulkan_pipeline_cache, &shader_notify, device,
        descriptor_pool, guest_descriptor_queue, descriptor_buffer_ring, thread_worker, statistics,
        render_pass_cache, key, std::move(modules), infos, std::move(emit_stages),
        std::move(stages.mesh_emulation));

} catch (const Shader::Exception& exception) {
    auto hash = key.Hash();
    size_t env_index{0};
    for (size_t index = 0; index < Maxwell::MaxShaderProgram; ++index) {
        if (key.unique_hashes[index] == 0) {
            continue;
        }
        Shader::Environment& env{*envs[env_index]};
        ++env_index;

        const u32 cfg_offset{static_cast<u32>(env.StartAddress() + sizeof(Shader::ProgramHeader))};
        try {
            // Walks the program for the dump; the one that failed may fail here again, and an
            // exception leaving this handler would end the process.
            Shader::Maxwell::Flow::CFG cfg(env, pools.flow_block, cfg_offset, index == 0);
        } catch (const Shader::Exception&) {
        }
        env.Dump(hash, key.unique_hashes[index]);
    }
    LOG_ERROR(Render_Vulkan, "{}", exception.what());
    return nullptr;
}

namespace {
// Pipelines built on the GPU thread at run time (shader translation, and the SPIR-V unless a worker
// emits it; the driver compile is on workers): the first 400 over 2 ms and every one over 100 ms
// are logged, and the totals every 10 s, to tell how much of the stutter on new content is
// translation.
void ReportRuntimePipeline(const char* kind, std::chrono::steady_clock::time_point begin,
                           const std::string& shaders) {
    static std::chrono::steady_clock::duration total{};
    static u32 count{}, logged{};
    static auto window = std::chrono::steady_clock::now();
    const auto now = std::chrono::steady_clock::now();
    const auto elapsed = now - begin;
    total += elapsed;
    ++count;
    const auto ms = [](auto d) { return std::chrono::duration<double, std::milli>(d).count(); };
    // Slow builds are always logged with their shaders, to find the programs that are slow to
    // translate. EDEN_NX2_PIPELINE_LOG_LIMIT=<n> changes how many of the others are.
    static const u32 limit = [] {
        const char* value = std::getenv("EDEN_NX2_PIPELINE_LOG_LIMIT");
        return value ? static_cast<u32>(std::strtoul(value, nullptr, 10)) : 400U;
    }();
    if (elapsed > std::chrono::milliseconds{100} ||
        (elapsed > std::chrono::milliseconds{2} && logged < limit)) {
        ++logged;
        LOG_INFO(Render_Vulkan, "{} pipeline built on the GPU thread in {:.1f} ms: {}", kind,
                 ms(elapsed), shaders);
    }
    if (now - window > std::chrono::seconds{10}) {
        const auto take = [](auto& counter) {
            return std::chrono::nanoseconds{counter.exchange(0, std::memory_order::relaxed)};
        };
        LOG_INFO(Render_Vulkan,
                 "Runtime pipelines in the last 10 s: {} built, {:.1f} ms on the GPU thread "
                 "({:.1f} ms of it graphics SPIR-V), {:.1f} ms graphics SPIR-V on workers, "
                 "{:.1f} ms of draws held for unbuilt pipelines",
                 count, ms(total), ms(take(runtime_spirv_ns[0])), ms(take(runtime_spirv_ns[1])),
                 ms(GraphicsPipeline::TakeHoldTime()));
        std::string reasons;
        {
            std::scoped_lock lock{precompiled_reasons_mutex};
            for (const auto& [reason, n] : precompiled_reasons) {
                reasons += fmt::format(", {} {}", n, reason);
            }
            precompiled_reasons.clear();
        }
        LOG_INFO(Render_Vulkan,
                 "Precompiled pipelines in the last 10 s: {} created from kept infos ({:.1f} ms on "
                 "the GPU thread), {} built from kept SPIR-V, {} translated on workers from their "
                 "recorded environments, {} failed and replaced{}",
                 precompiled_created.exchange(0, std::memory_order::relaxed),
                 ms(std::chrono::nanoseconds{
                     precompiled_gpu_ns.exchange(0, std::memory_order::relaxed)}),
                 precompiled_from_spirv.exchange(0, std::memory_order::relaxed),
                 precompiled_drawn.exchange(0, std::memory_order::relaxed),
                 precompiled_failed.exchange(0, std::memory_order::relaxed), reasons);
        if (PrecompiledAudit()) {
            LOG_INFO(Render_Vulkan,
                     "Precompiled audit in the last 10 s: {} pipelines translated from the game's "
                     "state gave the kept SPIR-V and infos, {} differed",
                     audit_same.exchange(0, std::memory_order::relaxed),
                     audit_differ.exchange(0, std::memory_order::relaxed));
        }
        total = {};
        count = 0;
        window = now;
    }
}
} // Anonymous namespace

std::unique_ptr<GraphicsPipeline> PipelineCache::CreateGraphicsPipeline() {
    const auto begin = std::chrono::steady_clock::now();
    SCOPE_EXIT {
        std::string shaders = fmt::format("pipeline={:016x}", graphics_key.Hash());
        for (size_t stage = 0; stage < graphics_key.unique_hashes.size(); ++stage) {
            if (graphics_key.unique_hashes[stage] != 0) {
                shaders += fmt::format(" s{}:{:016x}", stage, graphics_key.unique_hashes[stage]);
            }
        }
        ReportRuntimePipeline("Graphics", begin, shaders);
    };
    GraphicsEnvironments environments;
    GetGraphicsEnvironments(environments, graphics_key.unique_hashes);

    const auto precompiled_begin = std::chrono::steady_clock::now();
    auto pipeline{CreatePrecompiledGraphicsPipeline(environments)};
    if (pipeline) {
        precompiled_created.fetch_add(1, std::memory_order_relaxed);
        precompiled_gpu_ns.fetch_add(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                         std::chrono::steady_clock::now() - precompiled_begin)
                                         .count(),
                                     std::memory_order_relaxed);
        if (PrecompiledAudit()) {
            AuditPrecompiled(environments.Span(), false, graphics_key.Hash());
        }
    } else {
        std::shared_ptr<ShaderPools> emit_pools;
        auto reservation =
            emission_budget.TryAcquire(maxwell3d->GetNativeMacroEngine().native_turing);
        if (reservation) {
            {
                std::scoped_lock lock{spare_pools_mutex};
                if (!spare_pools.empty()) {
                    emit_pools = std::move(spare_pools.back());
                    spare_pools.pop_back();
                }
            }
            if (!emit_pools) {
                emit_pools = std::make_shared<ShaderPools>();
            }
        } else {
            main_pools.ReleaseContents();
        }
        ShaderPools& pools = emit_pools ? *emit_pools : main_pools;
        pipeline = CreateGraphicsPipeline(pools, graphics_key, environments.Span(), nullptr, true,
                                          std::move(emit_pools), std::move(reservation));
    }
    if (!pipeline || pipeline_cache_filename.empty()) {
        return pipeline;
    }
    if (disk_pipelines.contains(graphics_key.Hash())) {
        // In the disk cache already, only not built at boot.
        QueueVulkanPipelineCacheFlush();
        return pipeline;
    }
    disk_pipelines.insert(graphics_key.Hash());
    serialization_thread.QueueWork([this, key = graphics_key, envs = std::move(environments.envs)] {
        boost::container::static_vector<const GenericEnvironment*, Maxwell::MaxShaderProgram>
            env_ptrs;
        for (size_t index = 0; index < Maxwell::MaxShaderProgram; ++index) {
            if (key.unique_hashes[index] != 0) {
                env_ptrs.push_back(&envs[index]);
            }
        }
        SerializePipeline(key, env_ptrs, pipeline_cache_filename, CACHE_VERSION,
                          host_info.EnvironmentReadProperties());
    });
    QueueVulkanPipelineCacheFlush();
    return pipeline;
}

ComputePipelineCacheEntry PipelineCache::CreateComputePipeline(const ComputePipelineCacheKey& key,
                                                               const ShaderInfo* shader,
                                                               std::span<const u64> keep_pointers) {
    const auto begin = std::chrono::steady_clock::now();
    SCOPE_EXIT {
        ReportRuntimePipeline("Compute", begin,
                              fmt::format("pipeline={:016x} s6:{:016x} address={:#x}", key.Hash(),
                                          key.unique_hash,
                                          kepler_compute->regs.code_loc.Address() +
                                              kepler_compute->launch_description.program_start));
    };
    const GPUVAddr program_base{kepler_compute->regs.code_loc.Address()};
    const auto& qmd{kepler_compute->launch_description};
    ComputeEnvironment env{*kepler_compute, *gpu_memory, program_base, qmd.program_start,
                           shader->lowered_native};
    env.SetCachedSize(shader->size_bytes);
    env.KeepStoragePointers(keep_pointers);

    // A rebuild too (keep_pointers): its environment answers the kept pointers as pointers before
    // anything is asked, as its translation would see them.
    ComputePipelineCacheEntry entry;
    {
        const auto precompiled_begin = std::chrono::steady_clock::now();
        const PrecompiledRecord* matched = nullptr;
        entry = CreatePrecompiledComputePipeline(key, env, &matched);
        if (entry.pipeline && PrecompiledAudit()) {
            Shader::Environment* const env_ptr = &env;
            AuditPrecompiled(std::span(&env_ptr, 1), true, key.Hash(), matched);
        }
        if (entry.pipeline) {
            precompiled_created.fetch_add(1, std::memory_order_relaxed);
            precompiled_gpu_ns.fetch_add(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                             std::chrono::steady_clock::now() - precompiled_begin)
                                             .count(),
                                         std::memory_order_relaxed);
        }
    }
    if (!entry.pipeline) {
        main_pools.ReleaseContents();
        entry = CreateComputePipeline(main_pools, key, env, nullptr, true);
    }
    if (const char* directory = std::getenv("EDEN_NX2_SHADER_DUMP");
        directory && !entry.pipeline && env.CanBeSerialized()) {
        const auto path = std::filesystem::path{directory} /
                          fmt::format("failed-compute-{:016x}-v{}.env", key.unique_hash, CACHE_VERSION);
        std::ofstream file{path, std::ios::binary};
        if (file) {
            env.Serialize(file, host_info.EnvironmentReadProperties());
            LOG_INFO(Render_Vulkan, "Captured failed compute environment: {} written={}",
                     path.string(), file.good());
        }
    }
    if (!entry.pipeline || pipeline_cache_filename.empty()) {
        return entry;
    }
    serialization_thread.QueueWork([this, key, env_ = std::move(env)] {
        SerializePipeline(key, std::array<const GenericEnvironment*, 1>{&env_},
                          pipeline_cache_filename, CACHE_VERSION,
                          host_info.EnvironmentReadProperties());
    });
    QueueVulkanPipelineCacheFlush();
    return entry;
}

ComputePipelineCacheEntry PipelineCache::CreateComputePipeline(
    ShaderPools& pools, const ComputePipelineCacheKey& key, Shader::Environment& env,
    PipelineStatistics* statistics, bool build_in_parallel, std::vector<u32>* spirv_out) try {
    auto hash = key.Hash();
    if (device.HasBrokenCompute()) {
        LOG_ERROR(Render_Vulkan, "Skipping {:#016x}", hash);
        return {.failure = "Device reports broken compute support"};
    }

    LOG_DEBUG(Render_Vulkan, "{:#016x}", hash);

    Shader::Maxwell::Flow::CFG cfg{env, pools.flow_block, env.StartAddress()};

    // Dump it before error.
    if (Settings::values.dump_guest_shaders) {
        env.Dump(hash, key.unique_hash);
    }

    auto program{TranslateProgram(pools.inst, pools.block, env, cfg, host_info)};
    const VkDriverIdKHR driver_id = device.GetDriverID();
    const bool needs_shared_mem_clamp =
        driver_id == VK_DRIVER_ID_QUALCOMM_PROPRIETARY ||
        driver_id == VK_DRIVER_ID_ARM_PROPRIETARY || driver_id == VK_DRIVER_ID_MOLTENVK;
    const u32 max_shared_memory = device.GetMaxComputeSharedMemorySize();
    if (needs_shared_mem_clamp && program.shared_memory_size > max_shared_memory) {
        LOG_WARNING(Render_Vulkan,
                    "Compute shader {:#016x} requests {}KB shared memory but device max is {}KB - clamping",
                    key.unique_hash,
                    program.shared_memory_size / 1024,
                    max_shared_memory / 1024);
        program.shared_memory_size = max_shared_memory;
    }
    std::vector<u32> code{EmitSPIRV(profile, program)};
    if (spirv_out) {
        *spirv_out = code;
    }
    device.SaveShader(code);
    vk::ShaderModule spv_module{BuildShader(device, code)};

    // Text log + .spv dump. Same split as the graphics path.
    const bool should_log = GPU::Logging::IsActive();
    const bool should_dump = Settings::values.gpu_log_shader_dumps.GetValue();
    if (should_log || should_dump) {
        const std::string shader_name = fmt::format("shader_{:016x}_compute", key.unique_hash);
        if (should_log) {
            const std::string shader_info = fmt::format("SPIR-V size: {} bytes, hash: {:016x}",
                code.size() * sizeof(u32), key.unique_hash);
            GPU::Logging::GPULogger::GetInstance().LogShaderCompilation(shader_name, shader_info);
        }
        if (should_dump) {
            GPU::Logging::DumpSpirvShader(key.unique_hash,
                                          std::span<const u32>(code.data(), code.size()));
        }
    }

    if (device.HasDebuggingToolAttached()) {
        const auto name{fmt::format("Shader {:016x}", key.unique_hash)};
        spv_module.SetObjectNameEXT(name.c_str());
    }
    Common::ThreadWorker* const thread_worker{build_in_parallel ? &workers : nullptr};
    return {.pipeline = std::make_unique<ComputePipeline>(
                device, scheduler, vulkan_pipeline_cache, descriptor_pool, guest_descriptor_queue,
                descriptor_buffer_ring, thread_worker, statistics, &shader_notify, program.info,
                std::move(spv_module), key.unique_hash),
            .null_pointer_slots = env.IsNativeProgram() ? env.NullStoragePointerSlots()
                                                        : std::vector<std::pair<u32, u32>>{},
            .mapped_pointers =
                env.IsNativeProgram() ? env.MappedStoragePointerKeys() : std::vector<u64>{}};

} catch (const Shader::Exception& exception) {
    LOG_ERROR(Render_Vulkan, "Compute pipeline compilation failed shader={:#016x} pipeline={:#016x}: {}",
              key.unique_hash, key.Hash(), exception.what());
    return {.failure = exception.what()};
}

void PipelineCache::SerializeVulkanPipelineCache(const std::filesystem::path& filename,
                                                 const vk::PipelineCache& pipeline_cache,
                                                 u32 cache_version) {
    // Written beside the cache and renamed over it: the emulator stopped mid-write left the cache
    // empty, and every later boot failed to read it.
    auto temporary = filename;
    temporary += ".tmp";
    std::ofstream file(temporary, std::ios::binary);
    if (!file.is_open()) {
        LOG_ERROR(Common_Filesystem, "Failed to open Vulkan driver pipeline cache file {}",
                  Common::FS::PathToUTF8String(temporary));
        return;
    }
    try {
        file.exceptions(std::ifstream::failbit);
        file.write(VULKAN_CACHE_MAGIC_NUMBER.data(), VULKAN_CACHE_MAGIC_NUMBER.size())
            .write(reinterpret_cast<const char*>(&cache_version), sizeof(cache_version));

        size_t cache_size = 0;
        std::vector<char> cache_data;
        if (pipeline_cache) {
            pipeline_cache.Read(&cache_size, nullptr);
            cache_data.resize(cache_size);
            pipeline_cache.Read(&cache_size, cache_data.data());
        }
        file.write(cache_data.data(), cache_size);
        file.close();
    } catch (const std::ios_base::failure& e) {
        // The cache stays as it was: only the file being written goes.
        LOG_ERROR(Common_Filesystem, "{}", e.what());
        file.exceptions(std::ios::goodbit);
        file.close();
        std::error_code remove_error;
        std::filesystem::remove(temporary, remove_error);
        return;
    }
    std::error_code error;
    std::filesystem::rename(temporary, filename, error);
    if (error) {
        LOG_ERROR(Common_Filesystem, "Failed to replace Vulkan driver pipeline cache file {}: {}",
                  Common::FS::PathToUTF8String(filename), error.message());
        std::filesystem::remove(temporary, error);
        return;
    }

    LOG_INFO(Render_Vulkan, "Vulkan driver pipelines cached at: {}",
             Common::FS::PathToUTF8String(filename));
}

vk::PipelineCache PipelineCache::LoadVulkanPipelineCache(const std::filesystem::path& filename,
                                                         u32 expected_cache_version) {
    const auto create_pipeline_cache = [this](size_t data_size, const void* data) {
        VkPipelineCacheCreateInfo pipeline_cache_ci = {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .initialDataSize = data_size,
            .pInitialData = data};
        return device.GetLogical().CreatePipelineCache(pipeline_cache_ci);
    };
    try {
        std::ifstream file(filename, std::ios::binary | std::ios::ate);
        if (!file.is_open()) {
            return create_pipeline_cache(0, nullptr);
        }
        file.exceptions(std::ifstream::failbit);
        const auto end{file.tellg()};
        file.seekg(0, std::ios::beg);

        std::array<char, 8> magic_number;
        u32 cache_version;
        // A file cut short (the emulator stopped while writing it) would throw from the reads
        // below, and macOS's libc++ throws an ios_base::failure the handler doesn't match.
        if (static_cast<size_t>(end) < magic_number.size() + sizeof(cache_version)) {
            file.close();
            LOG_ERROR(Common_Filesystem, "Truncated Vulkan driver pipeline cache file, deleting it");
            Common::FS::RemoveFile(filename);
            return create_pipeline_cache(0, nullptr);
        }
        file.read(magic_number.data(), magic_number.size())
            .read(reinterpret_cast<char*>(&cache_version), sizeof(cache_version));
        if (magic_number != VULKAN_CACHE_MAGIC_NUMBER || cache_version != expected_cache_version) {
            file.close();
            if (Common::FS::RemoveFile(filename)) {
                if (magic_number != VULKAN_CACHE_MAGIC_NUMBER) {
                    LOG_ERROR(Common_Filesystem, "Invalid Vulkan driver pipeline cache file");
                }
                if (cache_version != expected_cache_version) {
                    LOG_INFO(Common_Filesystem, "Deleting old Vulkan driver pipeline cache");
                }
            } else {
                LOG_ERROR(Common_Filesystem,
                          "Invalid Vulkan pipeline cache file and failed to delete it in \"{}\"",
                          Common::FS::PathToUTF8String(filename));
            }
            return create_pipeline_cache(0, nullptr);
        }

        static constexpr size_t header_size = magic_number.size() + sizeof(cache_version);
        const size_t cache_size = static_cast<size_t>(end) - header_size;
        std::vector<char> cache_data(cache_size);
        file.read(cache_data.data(), cache_size);

        LOG_INFO(Render_Vulkan,
                 "Loaded Vulkan driver pipeline cache: {}", Common::FS::PathToUTF8String(filename));

        try {
            return create_pipeline_cache(cache_size, cache_data.data());
        } catch (const vk::Exception& e) {
            // The blob came from another driver or driver version (e.g. switching between
            // MoltenVK and KosmicKrisp on macOS): the driver rejects it. Start empty instead of
            // aborting, and drop the stale file so the next run doesn't retry it.
            LOG_ERROR(Render_Vulkan,
                      "Driver rejected the Vulkan pipeline cache file ({}), starting with an empty "
                      "cache and deleting it",
                      e.what());
            Common::FS::RemoveFile(filename);
            return create_pipeline_cache(0, nullptr);
        }

    } catch (const std::ios_base::failure& e) {
        LOG_ERROR(Common_Filesystem, "{}", e.what());
        if (!Common::FS::RemoveFile(filename)) {
            LOG_ERROR(Common_Filesystem, "Failed to delete Vulkan driver pipeline cache file {}",
                      Common::FS::PathToUTF8String(filename));
        }

        return create_pipeline_cache(0, nullptr);
    }
}

} // namespace Vulkan
