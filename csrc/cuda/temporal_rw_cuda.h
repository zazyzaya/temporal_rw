#pragma once

#include "../extensions.h"

std::tuple<torch::Tensor, torch::Tensor>
temporal_random_walk_cuda(torch::Tensor rowptr, torch::Tensor col, torch::Tensor ts, torch::Tensor start,
                 int64_t walk_length, int64_t t_start, int64_t t_end, bool reverse);
