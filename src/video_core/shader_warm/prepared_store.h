// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <array>
#include <filesystem>
#include <fstream>
#include <istream>
#include <mutex>
#include <optional>
#include <span>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include "common/common_types.h"

namespace VideoCommon::ShaderWarm {

/// Append-only journal of prepared shaders: content-deduplicated blobs (shader infos, SPIR-V) and
/// the records that point at them. An entry is reused only once its checksummed frame is flushed;
/// opening drops a torn tail, and another identity (game, translator, device, settings) starts a
/// fresh generation. Blob payloads are checked when read, records when the journal opens.
class PreparedStore {
public:
    static constexpr u32 MaxPayloadBytes = 64 * 1024 * 1024;
    static constexpr u32 MaxEntries = 4'000'000;

    /// Where an entry's frame (u64 checksum, payload) lies in the file.
    struct Location {
        u64 offset{};
        u32 size{};
        bool operator==(const Location&) const = default;
    };
    struct Record {
        u32 id{};
        Location at;
    };

    PreparedStore() = default;
    PreparedStore(const PreparedStore&) = delete;
    PreparedStore& operator=(const PreparedStore&) = delete;

    bool Open(const std::filesystem::path& path, const std::string& identity);
    void Close();

    /// Thread safe. Appends the blob unless the same content is stored already.
    std::optional<Location> AddBlob(std::span<const u8> bytes);
    /// Thread safe. A later record with the same id replaces an earlier one.
    bool AddRecord(u32 id, std::span<const u8> payload);

    /// Records whose checksum held when the journal was opened or that were added since.
    std::vector<Record> Records() const;
    bool IsBlob(const Location& location) const;

    /// A frame's payload, empty when it is unreadable or its checksum fails.
    static std::vector<u8> ReadFrame(std::istream& file, const Location& location);
    /// A blob's bytes (its payload without the content key), empty when damaged.
    static std::vector<u8> ReadBlob(std::istream& file, const Location& location);

private:
    using Key = std::array<u64, 2>;
    struct KeyHash {
        size_t operator()(const Key& key) const {
            return static_cast<size_t>(key[0] ^ (key[1] * 0x9E3779B97F4A7C15ULL));
        }
    };
    bool Append(u32 tag, u32 id, std::span<const u8> payload, Location& location);

    mutable std::mutex mutex;
    std::fstream output;
    std::unordered_map<Key, Location, KeyHash> blobs;
    std::unordered_set<u64> blob_offsets;
    std::unordered_map<u32, Location> records;
};

/// SHA-256 of a file's contents as hex, empty when it can't be read.
std::string ContentHash(const std::filesystem::path& file);
bool AtomicWrite(const std::filesystem::path& path, std::span<const u8> bytes);

} // namespace VideoCommon::ShaderWarm
