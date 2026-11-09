// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <cstddef>
#include <filesystem>
#include <functional>
#include <optional>
#include <stop_token>
#include <string>
#include <unordered_set>

#include "common/common_types.h"
#include "common/settings_enums.h"

namespace Core {
class System;
}

namespace VideoCommon::ShaderWarm {

/// Where the shader precache file of a title lives: next to its pipeline cache
/// (<shader dir>/<title id>/warm-compute.bin). It holds the compute programs found in the game
/// files, in pipeline cache format with no recorded reads.
std::filesystem::path ComputeWarmPath(u64 title_id);

/// The title's precompile recipe (tools/nx2-precompile), kept once a build used it, and the
/// pipeline file assembled from it with the player's game files, which the loading screen prepares.
std::filesystem::path NativeRecipePath(u64 title_id);
std::filesystem::path NativePreparationPath(u64 title_id);

/// The recipe for this title: the kept one, else the first <exe dir>/recipes/*.nx2recipe whose
/// header names this title and this build's cache version (as Fritchii ships them). Empty: none.
std::filesystem::path FindNativeRecipe(u64 title_id);

/// "Already precached" marker of a precache file (same folder, extension .done), written by the
/// emulator after the precache ran to the end. A matching marker skips the precache at boot.
std::filesystem::path WarmMarkerPath(const std::filesystem::path& warm_file);

/// Entry count of a precache file, written by the builder next to it (extension .count): the
/// entry count and the file's size. The file format has no count, so the loading screen shows
/// exact progress only with this.
std::filesystem::path WarmCountPath(const std::filesystem::path& warm_file);

/// The entry count recorded for `warm_file`, or 0 when there is none or it was written for a
/// file of another size.
u64 ReadWarmCount(const std::filesystem::path& warm_file);

/// What a marker records: the precache file's size and last-write time, its entry count, the GPU
/// driver version and device ID and the pipeline cache version.
struct WarmStamp {
    u64 size{};
    s64 write_time{};
    u64 entries{};
    u32 driver{};
    u32 device_id{};
    u32 cache_version{};

    /// True when `other` was written for the same file, driver, device and cache version (the
    /// entry count is informative).
    [[nodiscard]] bool Matches(const WarmStamp& other) const noexcept {
        return size == other.size && write_time == other.write_time && driver == other.driver &&
               device_id == other.device_id && cache_version == other.cache_version;
    }
};

/// The stamp of `warm_file` as it is now, empty when its size or time cannot be read.
std::optional<WarmStamp> CurrentWarmStamp(const std::filesystem::path& warm_file, u64 entries,
                                          u32 driver, u32 device_id, u32 cache_version);

/// The marker of `warm_file`, empty when there is none or it cannot be read.
std::optional<WarmStamp> ReadWarmMarker(const std::filesystem::path& warm_file);

/// Writes the marker of `warm_file`. Returns false when it cannot be written.
bool WriteWarmMarker(const std::filesystem::path& warm_file, const WarmStamp& stamp);

/// True when the title's precache has never finished: no precache file yet, or no marker next
/// to it, or a recipe (FindNativeRecipe) not assembled yet. A marker that no longer matches (new
/// driver) is not seen here: the emulator checks that at boot.
bool WarmPending(u64 title_id);

/// Whether this boot precaches before the game starts: Always, or Ask when the frontend asked and
/// the user agreed (`asked`). Off never does.
[[nodiscard]] constexpr bool PrecacheBeforeStart(Settings::PrecacheShaders mode, bool asked) {
    return mode == Settings::PrecacheShaders::Always ||
           (mode == Settings::PrecacheShaders::Ask && asked);
}

/// The launch's answer to "precache shaders before the game starts" (precache_shaders = Ask):
/// set by the frontend before booting, taken (and cleared) by the renderer when it loads. Setting
/// it also clears an earlier skip request.
void RequestWarmBeforeStart(bool before_start);
bool TakeWarmBeforeStart();

/// Skip button of the loading screen: the running precache stops after its current builds and the
/// game starts. Taken (and cleared) by the renderer.
void RequestSkipWarm();
bool TakeSkipWarm();
/// Whether Skip was pressed, without clearing it (the native preparation runs before the
/// precache, and one Skip stops both).
bool SkipWarmRequested();

/// The GPU driver, device and pipeline cache version pipelines were precached for.
struct DriverStamp {
    u32 driver{};
    u32 device_id{};
    u32 cache_version{};

    bool operator==(const DriverStamp&) const = default;
};

/// The title's list of cached graphics pipelines an earlier precache built
/// (<shader dir>/<title id>/precache-graphics.bin).
std::filesystem::path PrecachedPipelinesPath(u64 title_id);

/// The pipeline key hashes in `file`, empty when there is none or it was written for another
/// driver, device or cache version.
std::unordered_set<u64> ReadPrecachedPipelines(const std::filesystem::path& file,
                                               const DriverStamp& stamp);

/// Writes `hashes` for `stamp` to `file`. Returns false when it cannot be written.
bool WritePrecachedPipelines(const std::filesystem::path& file, const DriverStamp& stamp,
                             const std::unordered_set<u64>& hashes);

/// Removes the title's precache file, its marker and count, the list of precached graphics
/// pipelines and a leftover temporary file (rebuild or delete from the UI). Returns false when a file exists but cannot be removed, with
/// the reason in `error` when given.
bool DeleteWarmFiles(u64 title_id, std::string* error = nullptr);

struct Progress {
    std::string stage;   ///< "Reading game files", "Extracting shaders", "Converting", "Writing"
    size_t done{};       ///< Items done in this stage
    size_t total{};      ///< Items in this stage, 0 when unknown
};

struct Result {
    bool ok{};
    std::string error;           ///< Empty when ok
    size_t archives{};           ///< Shader archives found in the game files
    size_t native_pipelines{};   ///< Pipelines assembled from the recipe
    size_t native_dropped{};     ///< Recipe pipelines whose programs this copy lacks
    size_t native_programs{};    ///< Native programs found in the game files
    size_t native_combinations{}; ///< Program combinations (variations) found
    size_t compute_programs{};   ///< Entries written to the precache file
    size_t skipped{};            ///< Programs that failed to extract or convert
};

/// Builds the precache file for the game at `game_path` (any format the loader opens, Switch 2
/// .nup first), from the user's own copy and keys, with no external tools. Runs on the calling
/// thread; `progress` is called from it; stops early when `stop` is requested.
Result BuildWarmFiles(Core::System& system, const std::filesystem::path& game_path, u64 title_id,
                      const std::function<void(const Progress&)>& progress,
                      std::stop_token stop);

/// Overrides for BuildWarmFilesWith (tests and tools).
struct BuildOptions {
    /// Folder for warm-compute.bin; empty: <shader dir>/<title id>.
    std::filesystem::path output_dir;
    /// Recipe to assemble; empty: the one kept from an earlier build, if any.
    std::filesystem::path recipe_path;
    /// Conversion threads, 0: a quarter of the CPU threads, 2 to 6.
    size_t threads{};
};

/// BuildWarmFiles with an explicit output folder.
Result BuildWarmFilesWith(Core::System& system, const std::filesystem::path& game_path,
                          u64 title_id, const BuildOptions& options,
                          const std::function<void(const Progress&)>& progress,
                          std::stop_token stop);

} // namespace VideoCommon::ShaderWarm
