// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <cstddef>

namespace Vulkan {

// The mesh fallback adds storage-buffer bindings for its output (mesh-only), or
// output, task payload, offsets and indirect arguments (task + mesh). Check
// against the final descriptor set limit without overflowing guest + auxiliary.
[[nodiscard]] constexpr bool FitsStorageBufferSetLimit(std::size_t guest_buffers,
                                                       std::size_t auxiliary_buffers,
                                                       std::size_t set_limit) noexcept {
    return set_limit == 0 ||
           (guest_buffers <= set_limit && auxiliary_buffers <= set_limit - guest_buffers);
}

} // namespace Vulkan
