# Native shader preparation

Prepares a Switch 2 game's native graphics and compute pipelines before gameplay. The Vulkan
driver compiles them on the loading screen, and Lichii keeps its translated SPIR-V and shader
metadata, so later draws neither translate nor compile them again. Targeted at The Legend of
Zelda: Ocarina of Time (`0400486024DEC000`) on Windows/Vulkan; nothing in it is specific to that
game.

## Sources

This ports and hardens work that already existed; it does not reinvent it.

| Source | Ref | What is used |
|---|---|---|
| zortos293/Lichii#895 (`precompile`, open) | `6eea979a02f9c55a5b7011a3f9c2c00003e3e526` | Native program extraction (`core/file_sys/nx2_shader_extract`), recipes (`video_core/nx2_recipe`), canonical native keys, `tools/nx2-precompile` |
| zortos293/Lichii#896 (`engine-async-translation`, open) | `ca8567f0530e7613c775ce52ecaa5badf54d9710` | `Shader::Info` serialization, kept SPIR-V, runtime check that recorded answers hold (`FileEnvironment::AnswersHoldFor`), precompiled draw path |
| Frizti/Fritchii `fritchii-v5` | `62194f32e9b9af25504528a5c1037ffc1e97bebb` | Code reference only. Fritchii uses the #895 recipes. It gets render state from those recipes, or, for a title with no cache, from `warm_defaults.cpp` (the most common state in one title's recorded caches). Its caches, keys and file formats are not read or converted. |

The upstream survey (`Djihads80/Eglantine` PRs, issues and `upstream/nx2`) found no
preparation or precompile work. Merging `upstream/nx2` gives conflicts in the renderer, the shader
compiler, services and CI; that is left out of this branch.

## Where render state comes from

Game files contain native programs and the program combinations (variations) the game draws, but
not the fixed-function state (blend, depth, vertex input, render target formats) or the
constant-buffer and texture facts that translation reads. Like Fritchii, Lichii does not invent
them. They come from a **recipe** that `tools/nx2-precompile` learns from recorded runtime
pipeline caches (see its README). For every pipeline, a recipe holds its render state, a
fingerprint for each program, and the facts its translation read. It contains no program code
(`audit` checks this), so a recipe can be shared but extracted code cannot.

So graphics preparation needs a recipe for the game version and cache version. Compute programs
are still precached from the game files alone, as before.

## Flow

1. **Extract and assemble** (game list → *Build Shader Precache from Recipe...*; or *Build/Rebuild
   Shader Precache*, and the launch-time precache, with a kept recipe or one in
   `<exe dir>/recipes/*.nx2recipe` whose header names the title and this cache version, as
   Fritchii ships them). The existing compute precache is built first. Then:
   - The recipe is checked: magic, title id, cache version and bounded sizes. Other emulators'
     caches are refused.
   - The game's effective RomFS is opened (base, update and LayeredFS mods, as the game sees it).
     A SHA-256 over every file's path, size and contents identifies it; timestamps are not used.
   - The native programs and combinations are extracted, and the recipe is assembled with them
     into `<shader dir>/<title id>/native-preparation.bin`. The recipe is copied to
     `native.nx2recipe`, and `native-source.txt` records the identity. A rebuild with the same
     identity and an intact output skips extraction and assembly.
2. **Prepare** (loading screen, with *Precache shaders before game start* on). The stage is
   reported as *Precaching shaders n / total*, and Skip stops it.
   `PipelineCache::PreparePrecompiled` builds every record not prepared before on the pipeline
   workers, at most 2 × workers in flight. Each pipeline is dropped once built; the driver's own
   cache keeps the compiled code. Its `Shader::Info` and zstd SPIR-V go to
   `native-prepared.journal`.
3. **Use** (gameplay). A draw whose canonical key matches a prepared record (same hash, then
   same key bytes) is created from the kept infos, and a worker builds it from the kept SPIR-V,
   but only when the game's state answers the recorded environments' questions as they were
   answered (`AnswersHoldFor`). Anything else is translated as before.

## Journal (`native-prepared.journal`)

`"LCHPREP3"`, the identity, then append-only entries: `u32 tag` (0 blob, 1 record), `u32 id`,
`u32 frame size`, frame = `u64 CityHash64(payload)` + payload. A blob payload is a 16-byte
CityHash128 content key + bytes; infos and SPIR-V are stored once however many pipelines share
them. A record payload is the kind, environment ids, the full pipeline key, and for each stage the
location of its info and SPIR-V blobs. A record is appended only after its blobs are flushed.

- **Durable progress and resumption.** Each record is checkpointed as soon as it is prepared. A
  stopped, skipped or crashed preparation resumes with the records not yet journaled, and a fully
  prepared title only reads the record headers.
- **Corruption.** On open, records are checksummed. A damaged record is dropped and prepared again,
  and a torn or damaged last entry is truncated. Blobs are checksummed when they are read. A damaged
  blob, info that won't deserialize, or invalid SPIR-V (size, magic, zstd size) falls back to
  runtime translation.
- **Invalidation.** The identity changes, and the journal starts over, when any of these change:
  the SHA-256 of `native-preparation.bin`; `LICHII_PREPARATION_REVISION`, a hash of every working-tree
  source under `src/shader_recompiler` and `src/video_core` plus compiler id, version and flags
  (`CMakeModules/NativePreparationRevision.cmake`), so dirty builds don't share translations;
  the pipeline cache version; the GPU vendor, device, driver version, API version and pipeline
  cache UUID; every host profile and `HostTranslateInfo` field; resolution scaling,
  `disable_shader_loop_safety_checks`, `fix_bloom_effects`, `renderer_debug`; and
  `EDEN_NX2_UNDEF_ZERO` and `EDEN_NX2_NO_TENSOR`.
- **Device compatibility.** Records whose dynamic-state features differ from this device are
  counted and skipped. Driver fields in keys are rewritten to this driver's, as draws key them.

After a game update, the old `native-preparation.bin` still names the old programs. Their shader
hashes are in the keys, so updated programs don't match and are translated at runtime. *Rebuild
Shader Precache* assembles the file again. *Delete Shader Precache* removes the recipe, the
assembled file, the journal and the identity with the other precache files.

## Bounds

- Vulkan pipelines: only those in flight (≤ 2 × pipeline workers) exist at once.
- Journal index in memory: one location per distinct blob and per record.
- Recipe ≤ 64 MiB; journal payload ≤ 64 MiB; ≤ 4,000,000 entries; counts in the preparation file
  and recipe are bounded by the remaining bytes before anything is allocated.
- **Not bounded yet:** extraction keeps the extracted programs in memory, and `Nx2Recipe::Assemble`
  returns the whole preparation file as one buffer (the README cites 1.17 GB for 133,016 Ocarina
  pipelines). Peak memory during *Build* is therefore about the file size plus the programs.
- Every boot with a preparation file hashes it (SHA-256) for the journal identity, which reads the
  whole file once.
- Entries are flushed to the OS as they are written but not fsynced. A process crash keeps every
  complete entry; after a power loss the damaged tail is cut off as above.
- A blob that turns out damaged after the journal opened stays referenced for that generation, so
  its pipelines keep falling back to runtime translation until the identity changes or the journal
  is deleted.
- Preparation runs only where the driver keeps its own pipeline cache (NVIDIA, AMD proprietary),
  or with `EDEN_NX2_PRECOMPILED=1`. `EDEN_NX2_PRECOMPILED=0` disables loading and preparing.
  `EDEN_NX2_PRECOMPILED_FAST=0` and `EDEN_NX2_PRECOMPILED_AUDIT=1` (from #896) still apply.
- On NVIDIA, set Shader Cache Size to Unlimited, as the #895 README explains. Otherwise the driver
  silently evicts most prepared pipelines.

## Build

Windows, VS 2026 Build Tools environment, clang-cl, Ninja, Qt 6.8.3 (`C:/Qt/6.8.3/msvc2022_64`):

```bat
"C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat"
cmake -S C:/dev/Lichii -B C:/dev/Lichii/build-native -G Ninja ^
  -DCMAKE_C_COMPILER="C:/Program Files/LLVM/bin/clang-cl.exe" ^
  -DCMAKE_CXX_COMPILER="C:/Program Files/LLVM/bin/clang-cl.exe" ^
  -DCMAKE_BUILD_TYPE=Release -DENABLE_QT=ON -DYUZU_USE_BUNDLED_QT=OFF ^
  -DCMAKE_PREFIX_PATH=C:/Qt/6.8.3/msvc2022_64 -DENABLE_TESTS=ON -DENABLE_DEBUG_TOOLS=ON ^
  -DENABLE_WEB_SERVICE=OFF -DENABLE_UPDATE_CHECKER=OFF -DUSE_DISCORD_PRESENCE=OFF ^
  -DENABLE_LSFG=OFF -DENABLE_WERROR=OFF ^
  "-DCMAKE_CXX_FLAGS=/DWIN32 /D_WINDOWS /EHsc /D_USE_STD_VECTOR_ALGORITHMS=0"
ninja -C C:/dev/Lichii/build-native -j12 tests yuzu nx2-precompile
C:\dev\Lichii\build-native\bin\tests.exe "[preparation]"
```

`/D_USE_STD_VECTOR_ALGORITHMS=0` works around MSVC STL 14.51 with clang 23. Without it,
`std::find` on a 16-byte struct fails a `static_assert` in `xutility` while compiling the
unchanged `global_memory_to_storage_buffer_pass.cpp`. The owner's PC build (`C:/lx`, clang-cl
LLVM 22, VS 2022 SDK) may not need it.

## Running it (isolated)

Use a separate user folder so normal caches are untouched: copy `lichii.exe` into a fresh folder
next to an empty `user` directory (portable mode), add keys and firmware, then:

1. Make or get a recipe for the exact game version and cache version (`nx2-precompile
   cache-version`; tools/nx2-precompile/README.md).
2. Right-click the game → *Build Shader Precache from Recipe...* → pick the `.nx2recipe`.
3. Configure → Graphics → Advanced → *Precache shaders before game start*: On/Ask. Start the game.
   The loading screen shows the preparation, Skip stops it, and the next launch resumes it.
4. The log reports `Native shader preparation: n of m records done (...)`, then
   `Prepared native shaders: ... ignored`. During play, the 10 s report shows precompiled pipelines
   created, drawn and failed, and the reasons the precompiled path was refused.

## Measured (Ocarina of Time v131072, RTX 4090, Ryzen 9 5900X)

Fritchii v5.3's shipped recipe (`OoT v131072 (cache v54).nx2recipe`) with the player's `.nup`:

- `nx2-precompile assemble`: 41,216 of 41,216 programs found, 0 not lowered, 120,461 pipelines,
  0 dropped, 22 s; `check` replayed 62,740 environments through Lichii's translator with 22
  failures (uncached storage pointers) and 0 mismatched keys.
- In Lichii: the builder found 56,382 programs and 110,779 combinations; the loading screen built
  120,436 records (25 failed) in 855.6 s, and 75,161 graphics and 4,085 compute pipelines were
  loaded for draws (records sharing a key after the driver fields are rewritten are one).
- During play, every 10 s report showed prepared pipelines created from kept infos and built
  from kept SPIR-V with 0 failed and replaced (for example 683 in one report, 114 ms of GPU
  thread time), and the player reported no shader stutter.

## Verification status

Measured: the unit tests (`tests.exe "[preparation]"` and the whole suite) and the run above.
Not measured: peak RAM during *Build*, boot time with a complete journal, driver-cache retention
across reboots, AMD and Intel drivers. The 1.17 GB / 133,016 pipeline figures and the cache-size
behaviour come from the #895 README (RTX 3080).
