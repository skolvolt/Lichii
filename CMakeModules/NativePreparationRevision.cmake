# Include working-tree contents, not just HEAD: two dirty builds must never share translations.
file(GLOB_RECURSE inputs LIST_DIRECTORIES false
    "${SOURCE}/src/shader_recompiler/*.cpp" "${SOURCE}/src/shader_recompiler/*.h"
    "${SOURCE}/src/shader_recompiler/*.inc"
    "${SOURCE}/src/video_core/*.cpp" "${SOURCE}/src/video_core/*.h")
list(SORT inputs)
set(material "${TOOLCHAIN}")
foreach(input IN LISTS inputs)
    file(SHA256 "${input}" digest)
    string(APPEND material "${input}:${digest}\n")
endforeach()
string(SHA256 revision "${material}")
set(content "#pragma once\n#define LICHII_PREPARATION_REVISION \"${revision}\"\n")
if(EXISTS "${OUTPUT}")
    file(READ "${OUTPUT}" previous)
endif()
if(NOT content STREQUAL previous)
    file(WRITE "${OUTPUT}" "${content}")
endif()
