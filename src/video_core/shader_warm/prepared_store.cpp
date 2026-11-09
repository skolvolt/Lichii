// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include "video_core/shader_warm/prepared_store.h"

#include <cstring>
#include <memory>
#include <system_error>
#include <openssl/evp.h>
#include "common/cityhash.h"
#ifdef _WIN32
#include <windows.h>
#endif

namespace VideoCommon::ShaderWarm {
namespace {

// Header: magic, u32 identity size, identity. Entry: u32 tag, u32 id, u32 frame size, frame
// (u64 CityHash64 of the payload, payload). A blob's payload starts with its 16-byte content key.
constexpr std::string_view Magic = "LCHPREP3";
constexpr u32 TagBlob = 0;
constexpr u32 TagRecord = 1;
constexpr u32 EntryHeader = 12;
constexpr u32 KeySize = 16;

u64 Checksum(std::span<const u8> bytes) {
    return Common::CityHash64(reinterpret_cast<const char*>(bytes.data()), bytes.size());
}

template <typename T>
bool Get(std::istream& file, T& value) {
    return static_cast<bool>(file.read(reinterpret_cast<char*>(&value), sizeof(value)));
}

template <typename T>
void Put(std::ostream& file, const T& value) {
    file.write(reinterpret_cast<const char*>(&value), sizeof(value));
}

} // Anonymous namespace

std::vector<u8> PreparedStore::ReadFrame(std::istream& file, const Location& location) {
    if (location.size <= sizeof(u64) || location.size > MaxPayloadBytes + sizeof(u64)) {
        return {};
    }
    file.clear();
    file.seekg(static_cast<std::streamoff>(location.offset));
    u64 checksum{};
    if (!Get(file, checksum)) {
        return {};
    }
    std::vector<u8> bytes(location.size - sizeof(checksum));
    if (!file.read(reinterpret_cast<char*>(bytes.data()),
                   static_cast<std::streamsize>(bytes.size())) ||
        Checksum(bytes) != checksum) {
        return {};
    }
    return bytes;
}

std::vector<u8> PreparedStore::ReadBlob(std::istream& file, const Location& location) {
    auto bytes = ReadFrame(file, location);
    if (bytes.size() <= KeySize) {
        return {};
    }
    bytes.erase(bytes.begin(), bytes.begin() + KeySize);
    return bytes;
}

bool PreparedStore::Open(const std::filesystem::path& path, const std::string& identity) {
    std::scoped_lock lock{mutex};
    output.close();
    blobs.clear();
    blob_offsets.clear();
    records.clear();
    if (identity.empty() || identity.size() > 64 * 1024) {
        return false;
    }
    std::error_code error;
    const u64 file_size = std::filesystem::file_size(path, error);
    std::ifstream input(path, std::ios::binary);
    std::array<char, 8> magic{};
    u32 length{};
    bool matching = !error && input.read(magic.data(), magic.size()) && Get(input, length) &&
                    std::string_view(magic.data(), magic.size()) == Magic &&
                    length == identity.size();
    if (matching) {
        std::string stored(length, '\0');
        matching = input.read(stored.data(), length) && stored == identity;
    }
    u64 end = Magic.size() + sizeof(u32) + identity.size();
    if (matching) {
        // Blobs are checked when read: scanning only reads their keys. The last entry is checked
        // whole, since a crash can leave its frame partly written.
        for (u32 n = 0; n < MaxEntries; ++n) {
            u32 tag{}, id{}, size{};
            input.clear();
            input.seekg(static_cast<std::streamoff>(end));
            if (!Get(input, tag) || !Get(input, id) || !Get(input, size) || tag > TagRecord ||
                size <= sizeof(u64) + (tag == TagBlob ? KeySize : 0) ||
                size > MaxPayloadBytes + sizeof(u64) || size > file_size - end - EntryHeader) {
                break;
            }
            const Location location{end + EntryHeader, size};
            const bool last = location.offset + size == file_size;
            if (tag == TagRecord || last) {
                const auto payload = ReadFrame(input, location);
                if (payload.empty()) {
                    if (last) {
                        break;
                    }
                    end = location.offset + size;
                    continue;
                }
                if (tag == TagRecord) {
                    records.insert_or_assign(id, location);
                } else {
                    Key key;
                    std::memcpy(key.data(), payload.data(), KeySize);
                    blobs.try_emplace(key, location);
                    blob_offsets.insert(location.offset);
                }
            } else {
                Key key;
                input.clear();
                input.seekg(static_cast<std::streamoff>(location.offset + sizeof(u64)));
                if (!input.read(reinterpret_cast<char*>(key.data()), KeySize)) {
                    break;
                }
                blobs.try_emplace(key, location);
                blob_offsets.insert(location.offset);
            }
            end = location.offset + size;
        }
    }
    input.close();
    if (matching) {
        if (end != file_size) {
            std::filesystem::resize_file(path, end, error);
            if (error) {
                return false;
            }
        }
    } else {
        std::ofstream fresh(path, std::ios::binary | std::ios::trunc);
        fresh.write(Magic.data(), Magic.size());
        Put(fresh, static_cast<u32>(identity.size()));
        fresh.write(identity.data(), static_cast<std::streamsize>(identity.size()));
        fresh.close();
        if (!fresh) {
            return false;
        }
    }
    output.open(path, std::ios::binary | std::ios::in | std::ios::out);
    output.seekp(0, std::ios::end);
    return static_cast<bool>(output);
}

void PreparedStore::Close() {
    std::scoped_lock lock{mutex};
    output.close();
}

bool PreparedStore::Append(u32 tag, u32 id, std::span<const u8> payload, Location& location) {
    if (!output.is_open() || !output || payload.empty() || payload.size() > MaxPayloadBytes ||
        blobs.size() + records.size() >= MaxEntries) {
        return false;
    }
    const u32 size = static_cast<u32>(sizeof(u64) + payload.size());
    output.seekp(0, std::ios::end);
    const u64 start = static_cast<u64>(output.tellp());
    Put(output, tag);
    Put(output, id);
    Put(output, size);
    Put(output, Checksum(payload));
    output.write(reinterpret_cast<const char*>(payload.data()),
                 static_cast<std::streamsize>(payload.size()));
    output.flush();
    if (!output) {
        // Leave nothing half written behind for this session's later entries.
        output.clear();
        output.close();
        return false;
    }
    location = {start + EntryHeader, size};
    return true;
}

std::optional<PreparedStore::Location> PreparedStore::AddBlob(std::span<const u8> bytes) {
    if (bytes.empty() || bytes.size() > MaxPayloadBytes - KeySize) {
        return std::nullopt;
    }
    const Key key = Common::CityHash128(reinterpret_cast<const char*>(bytes.data()), bytes.size());
    std::vector<u8> payload(KeySize + bytes.size());
    std::memcpy(payload.data(), key.data(), KeySize);
    std::memcpy(payload.data() + KeySize, bytes.data(), bytes.size());
    std::scoped_lock lock{mutex};
    if (const auto it = blobs.find(key); it != blobs.end()) {
        return it->second;
    }
    Location location;
    if (!Append(TagBlob, 0, payload, location)) {
        return std::nullopt;
    }
    blobs.emplace(key, location);
    blob_offsets.insert(location.offset);
    return location;
}

bool PreparedStore::AddRecord(u32 id, std::span<const u8> payload) {
    std::scoped_lock lock{mutex};
    Location location;
    if (!Append(TagRecord, id, payload, location)) {
        return false;
    }
    records.insert_or_assign(id, location);
    return true;
}

std::vector<PreparedStore::Record> PreparedStore::Records() const {
    std::scoped_lock lock{mutex};
    std::vector<Record> result;
    result.reserve(records.size());
    for (const auto& [id, location] : records) {
        result.push_back({id, location});
    }
    return result;
}

bool PreparedStore::IsBlob(const Location& location) const {
    std::scoped_lock lock{mutex};
    return blob_offsets.contains(location.offset);
}

std::string ContentHash(const std::filesystem::path& path) {
    std::ifstream file(path, std::ios::binary);
    std::unique_ptr<EVP_MD_CTX, decltype(&EVP_MD_CTX_free)> ctx(EVP_MD_CTX_new(), EVP_MD_CTX_free);
    if (!file || !ctx || EVP_DigestInit_ex(ctx.get(), EVP_sha256(), nullptr) != 1) {
        return {};
    }
    std::vector<char> buffer(1 << 20);
    do {
        file.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
        if (EVP_DigestUpdate(ctx.get(), buffer.data(), static_cast<size_t>(file.gcount())) != 1) {
            return {};
        }
    } while (file);
    if (!file.eof()) {
        return {};
    }
    std::array<unsigned char, 32> digest{};
    unsigned length{};
    if (EVP_DigestFinal_ex(ctx.get(), digest.data(), &length) != 1 || length != digest.size()) {
        return {};
    }
    constexpr char Hex[] = "0123456789abcdef";
    std::string result;
    for (const auto byte : digest) {
        result += Hex[byte >> 4];
        result += Hex[byte & 15];
    }
    return result;
}

bool AtomicWrite(const std::filesystem::path& path, std::span<const u8> bytes) {
    auto temporary = path;
    temporary += ".tmp";
    std::ofstream file(temporary, std::ios::binary | std::ios::trunc);
    file.write(reinterpret_cast<const char*>(bytes.data()),
               static_cast<std::streamsize>(bytes.size()));
    file.close();
    if (!file) {
        return false;
    }
#ifdef _WIN32
    return MoveFileExW(temporary.c_str(), path.c_str(),
                       MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH) != 0;
#else
    std::error_code error;
    std::filesystem::rename(temporary, path, error);
    return !error;
#endif
}

} // namespace VideoCommon::ShaderWarm
