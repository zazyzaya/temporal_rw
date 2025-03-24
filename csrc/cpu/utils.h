#pragma once

#include "../extensions.h"

#define CHECK_CPU(x) AT_ASSERTM(x.device().is_cpu(), #x " must be CPU tensor")
#define CHECK_INPUT(x) AT_ASSERTM(x, "Input mismatch")
#define CHECK_CONTIGUOUS(x)                                                    \
  AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")

int64_t binary_search_min_cpu(const int64_t tgt, const int64_t st_idx,
  const int64_t en_idx, const int64_t *ts);
int64_t binary_search_max_cpu(const int64_t tgt, const int64_t st_idx,
  const int64_t en_idx, const int64_t *ts);
