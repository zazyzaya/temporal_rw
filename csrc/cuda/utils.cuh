#pragma once

#include "../extensions.h"

#define CHECK_CUDA(x)                                                          \
  AT_ASSERTM(x.device().is_cuda(), #x " must be CUDA tensor")
#define CHECK_INPUT(x) AT_ASSERTM(x, "Input mismatch")
#define CHECK_CONTIGUOUS(x)                                                    \
  AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")

#ifndef UTILS_CUH
#define UTILS_CUH

extern __device__ int64_t binary_search_min_cuda(const int64_t tgt,
    const int64_t st_idx,
    const int64_t en_idx,
    const int64_t *ts);

extern __device__ int64_t binary_search_max_cuda(const int64_t tgt,
  const int64_t st_idx,
  const int64_t en_idx,
  const int64_t *ts);

#endif // UTILS_CUH