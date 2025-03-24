#pragma once

#include "../extensions.h"

std::tuple<torch::Tensor, torch::Tensor>
continuous_trw_cuda(torch::Tensor rowptr, torch::Tensor col, torch::Tensor ts, torch::Tensor start,
                 int64_t walk_length, torch::Tensor t_start, torch::Tensor t_end, bool reverse);
