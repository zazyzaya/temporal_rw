#pragma once

#include "extensions.h"

namespace temporal_rw {
TEMPORAL_RW_API int64_t cuda_version() noexcept;

namespace detail {
TEMPORAL_RW_INLINE_VARIABLE int64_t _cuda_version = cuda_version();
} // namespace detail
} // namespace temporal_rw

TEMPORAL_RW_API std::tuple<torch::Tensor, torch::Tensor>
continuous_trw(torch::Tensor rowptr, torch::Tensor col, torch::Tensor ts, torch::Tensor start,
    int64_t walk_length, torch::Tensor t_start, torch::Tensor t_end, bool reverse);