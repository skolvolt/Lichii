// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <cstring>

#include "common/cityhash.h"
#include "shader_recompiler/frontend/ampere/instruction.h"
#include "shader_recompiler/frontend/ampere/lower_to_maxwell.h"
#include "shader_recompiler/frontend/ampere/program_capture.h"
#include "video_core/shader_warm/warm_records.h"

namespace VideoCommon::ShaderWarm {
namespace {

constexpr u64 AddressMapMagic = 0x50414d4e4544454EULL;
constexpr u64 HashXor = 0x4F554E4345000002ULL;
constexpr size_t EnvHeaderSize = 6 * 8 + 7 * 4;
constexpr u32 ComputeStage = 5;

template <typename T>
void Put(std::vector<u8>& out, T value) {
    const size_t at = out.size();
    out.resize(at + sizeof(T));
    std::memcpy(out.data() + at, &value, sizeof(T));
}

void PutBytes(std::vector<u8>& out, std::span<const u8> bytes) {
    out.insert(out.end(), bytes.begin(), bytes.end());
}

template <typename T>
T Get(std::span<const u8> data, size_t offset) {
    T value;
    std::memcpy(&value, data.data() + offset, sizeof(T));
    return value;
}

/// The GenericEnvironment::Serialize header: six u64 counts and seven u32 fields.
void PutEnvHeader(std::vector<u8>& out, u64 code_size, u64 cbuf_count, u32 local_memory,
                  u32 texture_bound, u32 start_address, u32 stage) {
    Put<u64>(out, code_size);
    Put<u64>(out, 0); // texture types
    Put<u64>(out, 0); // texture pixel formats
    Put<u64>(out, cbuf_count);
    Put<u64>(out, 0); // storage pointer types
    Put<u64>(out, 0); // cbuf replacements
    Put<u32>(out, local_memory);
    Put<u32>(out, texture_bound);
    Put<u32>(out, start_address);
    Put<u32>(out, 0);                                  // read lowest
    Put<u32>(out, static_cast<u32>(code_size - 8));    // read highest
    Put<u32>(out, 1);                                  // viewport transform state
    Put<u32>(out, stage);
}

} // Anonymous namespace

std::optional<std::string> LowerProgram(std::span<const u8> program, LoweredCode& out) {
    using Shader::Ampere::Instruction;
    if (program.empty() || program.size() % sizeof(Instruction) ||
        program.size() / sizeof(Instruction) > 65536) {
        return "Native input must contain 1..65536 complete 16-byte instructions";
    }
    std::vector<Instruction> native(program.size() / sizeof(Instruction));
    std::memcpy(native.data(), program.data(), native.size() * sizeof(Instruction));
    Shader::Ampere::ProgramCaptureState capture;
    bool complete = false;
    for (u32 index = 0; index < native.size(); ++index) {
        const auto result = capture.Observe(native[index], index);
        if (result == Shader::Ampere::ProgramCaptureState::Result::InvalidTarget) {
            return "Native allocation has an out-of-range branch target";
        }
        if (result == Shader::Ampere::ProgramCaptureState::Result::Complete) {
            native.resize(index + 1);
            complete = true;
            break;
        }
    }
    if (!complete) {
        return "Native allocation lacks a complete runtime capture";
    }
    for (const auto& instruction : native) {
        if (instruction.Bits(0, 12) == 0x95c) {
            return "Trap guards require captured constant buffers";
        }
    }
    const auto lowered = Shader::Ampere::LowerToMaxwell(native, false, true, {},
                                                        Shader::Ampere::TessellationStage::None);
    if (!lowered) {
        return std::string(lowered.error);
    }
    const auto& offsets = lowered.source_instruction_offsets;
    if (lowered.source_size != native.size() * sizeof(Instruction) ||
        offsets.size() != native.size()) {
        return "Cannot encode the lowering as an override record";
    }
    std::vector<u64> words(10, 0);
    words.reserve(10 + lowered.instructions.size() * 5 / 4 + 4 + offsets.size());
    for (const u64 instruction : lowered.instructions) {
        if (words.size() % 4 == 0) {
            words.push_back(0); // Maxwell scheduling word
        }
        words.push_back(instruction);
    }
    words.push_back(0);
    if (!offsets.empty()) {
        // AppendNativeAddressMap: lowered index -> byte address inside the code.
        for (const u32 i : offsets) {
            words.push_back(0x50 + u64{i} * 8 + ((u64{i} + 1) / 3) * 8);
        }
        words.push_back(offsets.size());
        words.push_back(AddressMapMagic);
    }
    out.code.resize(words.size() * sizeof(u64));
    std::memcpy(out.code.data(), words.data(), out.code.size());
    out.texture_buffer = lowered.required_texture_buffer;
    return std::nullopt;
}

u64 StageHash(std::span<const u8> code) {
    return Common::CityHash64(reinterpret_cast<const char*>(code.data()), code.size()) ^ HashXor;
}

std::optional<ComputeControl> ReadComputeControl(std::span<const u8> control) {
    if (control.size() < 108) {
        return std::nullopt;
    }
    ComputeControl result;
    for (size_t i = 0; i < 3; ++i) {
        result.workgroup[i] = Get<u32>(control, 88 + i * 4);
    }
    result.shared_memory =
        static_cast<u32>((u64{Get<u32>(control, 100)} + 255) / 256 * 256);
    result.local_memory = Get<u32>(control, 104);
    return result;
}

std::vector<u8> ComputeEntry(const LoweredCode& lowered, const ComputeControl& control,
                             ComputeKey& key) {
    std::vector<u8> out;
    out.reserve(4 + EnvHeaderSize + lowered.code.size() + 16 + key.size());
    Put<u32>(out, 1);
    // Like InitializeNativeCode: the program's own bank when it names one.
    const u32 texture_bound = lowered.texture_buffer != ~0U ? lowered.texture_buffer : 0;
    PutEnvHeader(out, lowered.code.size(), 0, control.local_memory, texture_bound, 80,
                 ComputeStage);
    PutBytes(out, lowered.code);
    for (const u32 size : control.workgroup) {
        Put<u32>(out, size);
    }
    Put<u32>(out, control.shared_memory);
    std::vector<u8> key_bytes;
    Put<u64>(key_bytes, StageHash(lowered.code));
    Put<u32>(key_bytes, control.shared_memory);
    for (const u32 size : control.workgroup) {
        Put<u32>(key_bytes, size);
    }
    std::memcpy(key.data(), key_bytes.data(), key.size());
    PutBytes(out, key_bytes);
    return out;
}

} // namespace VideoCommon::ShaderWarm
