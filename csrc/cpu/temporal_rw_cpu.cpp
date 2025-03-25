#include "temporal_rw_cpu.h"

#include <ATen/Parallel.h>

#include "utils.h"


int64_t binary_search_min_cpu(const int64_t tgt, const int64_t st_idx, const int64_t en_idx, const int64_t *ts) {
  /*
    Return lowest idx of ts that is >= tgt
    Assumes ts is sorted between ts[start] and ts[end]
  */
  int st = st_idx;
  int en = en_idx;

  while (en-st > 1) {
    auto len = en-st;
    auto half = len >> 1;
    auto val_at_half = ts[st+half];

    if (val_at_half >= tgt) {
        en = en-half;
    } else {
        st = st+half;
    }
  }

  if (ts[st] >= tgt) {
    return st;
  }
  return en;
}

int64_t binary_search_max_cpu(const int64_t tgt, const int64_t st_idx, const int64_t en_idx, const int64_t *ts) {
  /*
    Return highest idx of ts that is <= tgt
    Assumes ts is sorted between ts[start] and ts[end]
  */
  int st = st_idx;
  int en = en_idx;

  while (en-st > 1) {
    auto len = en-st;
    auto half = len >> 1;
    auto val_at_half = ts[st+half];

    if (val_at_half <= tgt) {
        st = st+half;
    } else {
        en = en-half;
    }
  }

  if (ts[en] <= tgt) {
    return en;
  }
  return st;
}

void uniform_sampling(const int64_t *rowptr, const int64_t *col, const int64_t *ts,
                      const int64_t *start, int64_t *n_out, int64_t *e_out,
                      const int64_t numel, const int64_t walk_length,
                      const int64_t t_start, const int64_t t_end, const bool reverse) {

  auto rand = torch::rand({numel, walk_length});
  auto rand_data = rand.data_ptr<float>();

  int64_t grain_size = at::internal::GRAIN_SIZE / walk_length;
  at::parallel_for(0, numel, grain_size, [&](int64_t begin, int64_t end) {
    for (auto n = begin; n < end; n++) {
      int64_t n_cur = start[n], e_cur = 0, row_start, row_end, idx;

      n_out[n * (walk_length + 1)] = n_cur;

      int64_t t = (reverse) ? t_end : t_start;
      for (auto l = 0; l < walk_length; l++) {
        row_start = rowptr[n_cur], row_end = rowptr[n_cur + 1];

        // Skip search if we already know we're at a terminal node/time
        if (e_cur == -1) {
          row_start = row_end;
        } else {
          // When traversing backward keep row_start the same, and decrease row_end
          if (reverse) {
            row_start = (t_start) ? binary_search_min_cpu(t_start, row_start, row_end, ts) : row_start;
            row_end = binary_search_max_cpu(t, row_start, row_end, ts);
          // Else, keep row_end the same and increase row_start
          } else {
            row_start = binary_search_min_cpu(t, row_start, row_end, ts);
            row_end = (t_end) ? binary_search_max_cpu(t_end, row_start, row_end, ts) : row_end;
          }
        }

        if (row_end - row_start == 0) {
          e_cur = -1;
        } else {
          idx = int64_t(rand_data[n * walk_length + l] * (row_end - row_start));
          e_cur = row_start + idx;
          n_cur = col[e_cur];
          t = ts[e_cur];
        }
        n_out[n * (walk_length + 1) + (l + 1)] = n_cur;
        e_out[n * walk_length + l] = e_cur;
      }
    }
  });
}


std::tuple<torch::Tensor, torch::Tensor>
temporal_random_walk_cpu(torch::Tensor rowptr, torch::Tensor col, torch::Tensor ts, torch::Tensor start,
                int64_t walk_length, int64_t t_start, int64_t t_end, bool reverse) {
  CHECK_CPU(rowptr);
  CHECK_CPU(col);
  CHECK_CPU(start);
  CHECK_CPU(ts);

  CHECK_INPUT(rowptr.dim() == 1);
  CHECK_INPUT(col.dim() == 1);
  CHECK_INPUT(start.dim() == 1);
  CHECK_INPUT(ts.dim() == 1);

  auto n_out = torch::empty({start.size(0), walk_length + 1}, start.options());
  auto e_out = torch::empty({start.size(0), walk_length}, start.options());

  auto rowptr_data = rowptr.data_ptr<int64_t>();
  auto col_data = col.data_ptr<int64_t>();
  auto ts_data = ts.data_ptr<int64_t>();
  auto start_data = start.data_ptr<int64_t>();
  auto n_out_data = n_out.data_ptr<int64_t>();
  auto e_out_data = e_out.data_ptr<int64_t>();

  uniform_sampling(rowptr_data, col_data, ts_data, start_data, n_out_data, e_out_data,
                     start.numel(), walk_length, t_start, t_end, reverse);

  return std::make_tuple(n_out, e_out);
}
