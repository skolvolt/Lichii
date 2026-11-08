// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <array>
#include <optional>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include "shader_recompiler/environment.h"
#include "shader_recompiler/frontend/ir/ir_emitter.h"
#include "shader_recompiler/host_translate_info.h"
#include "shader_recompiler/ir_opt/passes.h"
#include "shader_recompiler/storage_write_extent.h"

namespace {
namespace IR = Shader::IR;
using Shader::StorageWriteExtent;

class TestEnvironment final : public Shader::Environment {
public:
    bool native{true};
    TestEnvironment() { stage = Shader::Stage::Compute; }
    bool IsNativeProgram() const override { return native; }
    bool IsCbufStorageBufferPointer(u32 index, u32) override { return index == 0; }
    u64 ReadInstruction(u32) override { return 0; }
    u32 ReadCbufValue(u32, u32) override { return 0; }
    Shader::TextureType ReadTextureType(u32) override { return Shader::TextureType::Color2D; }
    Shader::TexturePixelFormat ReadTexturePixelFormat(u32) override {
        return Shader::TexturePixelFormat::A8B8G8R8_UNORM;
    }
    bool IsTexturePixelFormatInteger(u32) override { return false; }
    u32 ReadViewportTransformState() override { return 1; }
    u32 TextureBoundBuffer() const override { return 0; }
    u32 LocalMemorySize() const override { return 0; }
    u32 SharedMemorySize() const override { return 0; }
    std::array<u32, 3> WorkgroupSize() const override { return {64, 1, 1}; }
    bool HasHLEMacroState() const override { return false; }
    std::optional<Shader::ReplaceConstant> GetReplaceConstBuffer(u32, u32) override {
        return std::nullopt;
    }
    void Dump(u64, u64) override {}
};

struct KernelProgram {
    Shader::ObjectPool<IR::Inst> pool{256};
    IR::Block entry{pool};
    IR::Block loop{pool};
    IR::Program program;
    TestEnvironment env;
    bool ran{};

    KernelProgram() {
        program.stage = Shader::Stage::Compute;
        program.workgroup_size = {64, 1, 1};
        program.blocks = {&entry};
        program.post_order_blocks = {&entry};
    }

    /// Runs the pass once and returns the descriptor of the buffer whose pointer is at c0[offset]
    const Shader::StorageBufferDescriptor& Run(u32 offset) {
        if (!ran) {
            Shader::HostTranslateInfo host{};
            host.min_ssbo_alignment = 16;
            Shader::Optimization::GlobalMemoryToStorageBufferPass(env, program, host);
            ran = true;
        }
        for (const auto& descriptor : program.info.storage_buffers_descriptors) {
            if (descriptor.cbuf_index == 0 && descriptor.cbuf_offset == offset) {
                return descriptor;
            }
        }
        FAIL("No descriptor for the pointer");
        return program.info.storage_buffers_descriptors.front();
    }
};

IR::U32 Word(IR::IREmitter& ir, u32 offset) {
    return ir.GetCbuf(ir.Imm32(0), ir.Imm32(offset));
}

/// The 64-bit pointer at c0[offset] plus a 32-bit byte offset
IR::U64 Address(IR::IREmitter& ir, u32 offset, const IR::Value& bytes) {
    return ir.PackUint2x32(ir.CompositeConstruct(
        IR::U32{ir.IAdd(Word(ir, offset), IR::U32{bytes})}, Word(ir, offset + 4)));
}

std::optional<u64> At(const StorageWriteExtent& extent, std::array<u32, 3> block,
                      std::optional<std::array<u32, 3>> grid) {
    return Shader::ResolveStorageWriteExtent(extent, block, grid);
}
} // Anonymous namespace

TEST_CASE("Stores at constant offsets end at the last byte written", "[shader][storage_extent]") {
    KernelProgram fixture;
    fixture.env.native = GENERATE(false, true);
    IR::IREmitter ir{fixture.entry};
    ir.WriteGlobal32(Address(ir, 0x170, ir.Imm32(0x40)), ir.Imm32(1));
    ir.WriteGlobal128(Address(ir, 0x170, ir.Imm32(0x10)), ir.CompositeConstruct(
        ir.Imm32(0), ir.Imm32(0), ir.Imm32(0), ir.Imm32(0)));
    const auto& descriptor{fixture.Run(0x170)};
    REQUIRE(descriptor.is_written);
    const StorageWriteExtent& extent{descriptor.write_extent};
    CHECK(extent.bounded);
    CHECK_FALSE(extent.guarded);
    CHECK(extent.const_end == 0x44);
    CHECK(extent.local_stride == std::array<u32, 3>{});
    CHECK(extent.group_stride == std::array<u32, 3>{});
    CHECK(At(extent, {64, 1, 1}, std::array<u32, 3>{100, 1, 1}) == 0x44);
    CHECK(At(extent, {64, 1, 1}, std::nullopt) == 0x44);
}

TEST_CASE("Stores indexed by the global invocation id scale with the dispatch",
          "[shader][storage_extent]") {
    // ((ctaid.x << 6) + tid.x) << 8 + 236: one 256-byte record per invocation, as Ocarina of
    // Time's effect and skinning programs write them
    KernelProgram fixture;
    IR::IREmitter ir{fixture.entry};
    const IR::U32 global{ir.IAdd(ir.ShiftLeftLogical(ir.WorkgroupIdX(), ir.Imm32(6)),
                                 ir.LocalInvocationIdX())};
    const IR::U32 record{ir.ShiftLeftLogical(global, ir.Imm32(8))};
    ir.WriteGlobal64(Address(ir, 0x150, ir.IAdd(record, ir.Imm32(236))),
                     ir.CompositeConstruct(ir.Imm32(0), ir.Imm32(0)));
    const StorageWriteExtent& extent{fixture.Run(0x150).write_extent};
    REQUIRE(extent.bounded);
    CHECK(extent.const_end == 244);
    CHECK(extent.local_stride == std::array<u32, 3>{256, 0, 0});
    CHECK(extent.group_stride == std::array<u32, 3>{16384, 0, 0});
    // 64 invocations: one group
    CHECK(At(extent, {64, 1, 1}, std::array<u32, 3>{1, 1, 1}) == 244 + 256 * 63);
    CHECK(At(extent, {64, 1, 1}, std::array<u32, 3>{3, 1, 1}) == 244 + 256 * 63 + 16384 * 2);
    // An indirect dispatch's group count is unknown
    CHECK_FALSE(At(extent, {64, 1, 1}, std::nullopt).has_value());
}

TEST_CASE("Constant multiplications and masks keep the bound", "[shader][storage_extent]") {
    KernelProgram fixture;
    IR::IREmitter ir{fixture.entry};
    const IR::U32 lane{ir.BitwiseAnd(ir.LocalInvocationIdX(), ir.Imm32(31))};
    const IR::U32 row{ir.IMul(ir.WorkgroupIdY(), ir.Imm32(48))};
    ir.WriteGlobal32(Address(ir, 0x160, ir.IAdd(ir.ShiftLeftLogical(lane, ir.Imm32(2)), row)),
                     ir.Imm32(0));
    const StorageWriteExtent& extent{fixture.Run(0x160).write_extent};
    REQUIRE(extent.bounded);
    CHECK(extent.const_end == 31 * 4 + 4);
    CHECK(extent.local_stride == std::array<u32, 3>{});
    CHECK(extent.group_stride == std::array<u32, 3>{0, 48, 0});
}

TEST_CASE("A multiplication split in 16-bit halves is bounded while its operand fits",
          "[shader][storage_extent]") {
    // Native IMAD lowered through XMAD: lo(id) * 48 + ((hi(id) * 48) << 16)
    KernelProgram fixture;
    IR::IREmitter ir{fixture.entry};
    const IR::U32 id{ir.IAdd(ir.ShiftLeftLogical(ir.WorkgroupIdX(), ir.Imm32(6)),
                             ir.LocalInvocationIdX())};
    const IR::U32 low_half{ir.BitFieldExtract(id, ir.Imm32(0), ir.Imm32(16), false)};
    const IR::U32 high_half{ir.BitFieldExtract(id, ir.Imm32(16), ir.Imm32(16), false)};
    const IR::U32 product{ir.IAdd(ir.IMul(low_half, ir.Imm32(48)),
                                  ir.ShiftLeftLogical(ir.IMul(high_half, ir.Imm32(48)),
                                                      ir.Imm32(16)))};
    ir.WriteGlobal32(Address(ir, 0x170, product), ir.Imm32(0));
    const StorageWriteExtent& extent{fixture.Run(0x170).write_extent};
    REQUIRE(extent.bounded);
    CHECK(extent.guarded);
    CHECK(extent.local_stride == std::array<u32, 3>{48, 0, 0});
    CHECK(extent.group_stride == std::array<u32, 3>{48 * 64, 0, 0});
    // 1024 groups of 64: the largest id is 0xffff, its high half still zero
    CHECK(At(extent, {64, 1, 1}, std::array<u32, 3>{1024, 1, 1}) == 4 + 48 * (1024 * 64 - 1));
    // 1025 groups: the id reaches 2^16 and its high half is no longer zero
    CHECK_FALSE(At(extent, {64, 1, 1}, std::array<u32, 3>{1025, 1, 1}).has_value());
}

TEST_CASE("Offsets read from memory or other parameters are unbounded",
          "[shader][storage_extent]") {
    KernelProgram fixture;
    IR::IREmitter ir{fixture.entry};
    const bool loaded = GENERATE(false, true);
    // An index loaded from another buffer, or a count or offset parameter from the constant
    // buffer
    const IR::U32 index{
        loaded ? ir.LoadGlobal32(Address(
                     ir, 0x180, IR::U32{ir.ShiftLeftLogical(ir.LocalInvocationIdX(), ir.Imm32(2))}))
               : IR::U32{ir.IAdd(ir.LocalInvocationIdX(), Word(ir, 0x1a0))}};
    ir.WriteGlobal32(Address(ir, 0x170, ir.ShiftLeftLogical(index, ir.Imm32(2))), ir.Imm32(0));
    const auto& descriptor{fixture.Run(0x170)};
    CHECK(descriptor.is_written);
    CHECK_FALSE(descriptor.write_extent.bounded);
    CHECK_FALSE(At(descriptor.write_extent, {64, 1, 1}, std::array<u32, 3>{1, 1, 1}));
    if (loaded) {
        // The buffer only read keeps no extent and is not written
        const auto& source{fixture.Run(0x180)};
        CHECK_FALSE(source.is_written);
        CHECK_FALSE(source.write_extent.bounded);
    }
}

TEST_CASE("Loop-carried store offsets are unbounded", "[shader][storage_extent]") {
    KernelProgram fixture;
    fixture.entry.AddBranch(&fixture.loop);
    fixture.loop.AddBranch(&fixture.loop);
    fixture.program.blocks = {&fixture.entry, &fixture.loop};
    fixture.program.post_order_blocks = {&fixture.loop, &fixture.entry};
    IR::IREmitter entry{fixture.entry};
    const IR::U32 first{entry.ShiftLeftLogical(entry.LocalInvocationIdX(), entry.Imm32(2))};
    IR::IREmitter body{fixture.loop};
    fixture.loop.AppendNewInst(IR::Opcode::Phi, {});
    IR::Inst* const cursor{&fixture.loop.back()};
    cursor->SetFlags(IR::Type::U32);
    body.WriteGlobal32(Address(body, 0x170, IR::U32{IR::Value{cursor}}), body.Imm32(0));
    const IR::U32 next{body.IAdd(IR::U32{IR::Value{cursor}}, body.Imm32(256))};
    cursor->AddPhiOperand(&fixture.entry, first);
    cursor->AddPhiOperand(&fixture.loop, next);
    CHECK_FALSE(fixture.Run(0x170).write_extent.bounded);
}

TEST_CASE("Selected offsets take the larger alternative", "[shader][storage_extent]") {
    KernelProgram fixture;
    IR::IREmitter ir{fixture.entry};
    const IR::U1 condition{ir.IEqual(ir.LocalInvocationIdX(), ir.Imm32(0))};
    const IR::U32 offset{ir.Select(condition, ir.Imm32(0x20),
                                   ir.ShiftLeftLogical(ir.LocalInvocationIdX(), ir.Imm32(4)))};
    ir.WriteGlobal32(Address(ir, 0x170, offset), ir.Imm32(0));
    const StorageWriteExtent& extent{fixture.Run(0x170).write_extent};
    REQUIRE(extent.bounded);
    CHECK(At(extent, {64, 1, 1}, std::array<u32, 3>{1, 1, 1}) == 0x24 + 16 * 63);
}

TEST_CASE("A global store left untracked makes every written buffer unbounded",
          "[shader][storage_extent]") {
    KernelProgram fixture;
    IR::IREmitter ir{fixture.entry};
    ir.WriteGlobal32(Address(ir, 0x170, ir.Imm32(0)), ir.Imm32(0));
    // An address with no constant-buffer pointer at all stays a global store
    ir.WriteGlobal32(ir.PackUint2x32(ir.CompositeConstruct(ir.LocalInvocationIdX(), ir.Imm32(0))),
                     ir.Imm32(0));
    CHECK_FALSE(fixture.Run(0x170).write_extent.bounded);
}

TEST_CASE("Store extents merge to cover both", "[shader][storage_extent]") {
    StorageWriteExtent lhs{.bounded = true, .const_end = 16, .local_stride = {4, 0, 0}};
    const StorageWriteExtent rhs{.bounded = true, .const_end = 8, .group_stride = {0, 32, 0}};
    Shader::MergeStorageWriteExtent(lhs, rhs);
    CHECK(lhs.bounded);
    CHECK(lhs.const_end == 16);
    CHECK(lhs.local_stride == std::array<u32, 3>{4, 0, 0});
    CHECK(lhs.group_stride == std::array<u32, 3>{0, 32, 0});
    Shader::MergeStorageWriteExtent(lhs, StorageWriteExtent{});
    CHECK_FALSE(lhs.bounded);
}

TEST_CASE("An offset that can wrap below zero does not shrink the extent",
          "[shader][storage_extent]") {
    // Invocation 0 computes 0 - 8192 = 0xffffe000 (or 0 - 255), whose low 16 bits times 4 put the
    // store at byte 0x38000 (or 0x3fc04). Neither a select nor a mask may hide that.
    KernelProgram fixture;
    IR::IREmitter ir{fixture.entry};
    const IR::U32 tid{ir.LocalInvocationIdX()};
    const bool masked = GENERATE(false, true);
    IR::U32 value;
    u32 reach;
    if (masked) {
        // tid & 0xff is at most 255, but 255 is no lower bound: (tid & 0xff) - 255 wraps
        value = IR::U32{ir.ISub(ir.BitwiseAnd(tid, ir.Imm32(0xff)), ir.Imm32(255))};
        reach = 0xff01 * 4 + 4;
    } else {
        const IR::U1 first{ir.IEqual(tid, ir.Imm32(0))};
        value = IR::U32{ir.Select(first, ir.ISub(tid, ir.Imm32(8192)), ir.Imm32(0))};
        reach = 0xe000 * 4 + 4;
    }
    const IR::U32 low_half{ir.BitFieldExtract(value, ir.Imm32(0), ir.Imm32(16), false)};
    ir.WriteGlobal32(Address(ir, 0x170, ir.IMul(low_half, ir.Imm32(4))), ir.Imm32(0));
    const StorageWriteExtent& extent{fixture.Run(0x170).write_extent};
    const std::optional<u64> marked{At(extent, {64, 1, 1}, std::array<u32, 3>{1, 1, 1})};
    INFO("marked " << (marked ? *marked : 0) << " of " << reach);
    CHECK((!marked || *marked >= reach));
}

TEST_CASE("A value that is only an upper bound is not subtracted or used as a multiplier",
          "[shader][storage_extent]") {
    // 0x10000 - (tid & 0xff), also as 0xffffffff * (tid & 0xff) + 0x10000: 255 bounds tid & 0xff
    // from above only, so invocation 0 stores at 0x10000, past 0x10000 - 255
    KernelProgram fixture;
    IR::IREmitter ir{fixture.entry};
    const IR::U32 lane{ir.BitwiseAnd(ir.LocalInvocationIdX(), ir.Imm32(0xff))};
    const bool multiplied = GENERATE(false, true);
    const IR::U32 offset{multiplied
                             ? IR::U32{ir.IAdd(ir.IMul(ir.Imm32(0xffffffff), lane),
                                               ir.Imm32(0x10000))}
                             : IR::U32{ir.ISub(ir.Imm32(0x10000), lane)}};
    ir.WriteGlobal32(Address(ir, 0x170, offset), ir.Imm32(0));
    const StorageWriteExtent& extent{fixture.Run(0x170).write_extent};
    const std::optional<u64> marked{At(extent, {64, 1, 1}, std::array<u32, 3>{1, 1, 1})};
    INFO("marked " << (marked ? *marked : 0));
    CHECK((!marked || *marked >= 0x10000 + 4));
}

TEST_CASE("Scaling a large constant gives up instead of overflowing", "[shader][storage_extent]") {
    // tid + 4 * 0x80000000 is tid (and tid + 5 * 0x7fffffff is tid + 0x7ffffffb) in 32 bits, but
    // the analysis sees the constant as -2^33 (or about 2^33.3), and << 31 overflows it in 64 bits
    KernelProgram fixture;
    IR::IREmitter ir{fixture.entry};
    const bool negative = GENERATE(true, false);
    IR::U32 value{ir.LocalInvocationIdX()};
    for (int i = 0; i < (negative ? 4 : 5); ++i) {
        value = IR::U32{ir.IAdd(value, ir.Imm32(negative ? 0x80000000U : 0x7fffffffU))};
    }
    ir.WriteGlobal32(Address(ir, 0x170, ir.ShiftLeftLogical(value, ir.Imm32(31))), ir.Imm32(0));
    const StorageWriteExtent& extent{fixture.Run(0x170).write_extent};
    // Every invocation stores at 0 or 0x80000000
    const std::optional<u64> marked{At(extent, {64, 1, 1}, std::array<u32, 3>{1, 1, 1})};
    CHECK((!marked || *marked >= 0x80000004ULL));
}

TEST_CASE("A store left global between pointers makes every written buffer unbounded",
          "[shader][storage_extent]") {
    // The store picks c0[0x150] or c0[0x160] at run time and stays global. The backend routes it to
    // the descriptor whose pointer is the closest one at or below the address, which can be
    // c0[0x170]'s, so c0[0x170]'s own stores no longer bound what lands in its binding.
    KernelProgram fixture;
    IR::IREmitter ir{fixture.entry};
    const IR::U32 tid{ir.LocalInvocationIdX()};
    const IR::U32 bytes{ir.ShiftLeftLogical(tid, ir.Imm32(2))};
    ir.WriteGlobal32(Address(ir, 0x170, bytes), ir.Imm32(0));
    const IR::U1 first{ir.IEqual(tid, ir.Imm32(0))};
    const IR::U32 low{ir.IAdd(IR::U32{ir.Select(first, Word(ir, 0x150), Word(ir, 0x160))},
                              ir.ShiftLeftLogical(ir.WorkgroupIdX(), ir.Imm32(20)))};
    const IR::U32 high{ir.Select(first, Word(ir, 0x154), Word(ir, 0x164))};
    ir.WriteGlobal32(ir.PackUint2x32(ir.CompositeConstruct(low, high)), ir.Imm32(0));
    REQUIRE(fixture.Run(0x150).is_written);
    CHECK_FALSE(fixture.Run(0x150).write_extent.bounded);
    CHECK_FALSE(fixture.Run(0x160).write_extent.bounded);
    REQUIRE(fixture.program.info.native_global_bounds);
    CHECK_FALSE(fixture.Run(0x170).write_extent.bounded);
}

TEST_CASE("Offsets whose constant wraps below zero are unbounded", "[shader][storage_extent]") {
    // ((tid.x - 0x40000000) << 2) + 0x2000 is 0x2000 + 4 * tid.x modulo 2^32
    KernelProgram fixture;
    IR::IREmitter ir{fixture.entry};
    const IR::U32 shifted{ir.ShiftLeftLogical(
        ir.ISub(ir.LocalInvocationIdX(), ir.Imm32(0x40000000)), ir.Imm32(2))};
    ir.WriteGlobal32(Address(ir, 0x170, ir.IAdd(shifted, ir.Imm32(0x2000))), ir.Imm32(0));
    CHECK_FALSE(fixture.Run(0x170).write_extent.bounded);
}
