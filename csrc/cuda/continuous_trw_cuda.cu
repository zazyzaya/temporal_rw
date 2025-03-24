#include "continuous_trw_cuda.h"

#include <ATen/cuda/CUDAContext.h>
#include <curand.h>
#include <curand_kernel.h>

#include "utils.cuh"

#define THREADS 1024
#define BLOCKS(N) (N + THREADS - 1) / THREADS


__global__ void continuous_uniform_sampling_kernel(const int64_t *rowptr,
                                        const int64_t *col,
                                        const int64_t *ts,
                                        const int64_t *start, const float *rand,
                                        int64_t *n_out, int64_t *e_out,
                                        const int64_t walk_length,
                                        int64_t *t_start,
                                        int64_t *t_end,
                                        const bool reverse,
                                        const int64_t numel) {

  const int64_t thread_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (thread_idx < numel) {
    int64_t n_cur = start[thread_idx], e_cur = 0, row_start, row_end, rnd;

    n_out[thread_idx] = n_cur;

    int64_t t = (reverse) ? t_end[thread_idx] : t_start[thread_idx];

    for (int64_t l = 0; l < walk_length; l++) {
      row_start = rowptr[n_cur], row_end = rowptr[n_cur + 1];

      // Skip search if we already know we're at a terminal node/time
      if (e_cur == -1) {
        row_start = row_end;
      } else {
        // When traversing backward keep row_start the same, and decrease row_end
        if (reverse) {
          row_start = (t_start[thread_idx]) ? binary_search_min_cuda(t_start[thread_idx], row_start, row_end, ts) : row_start;
          row_end = binary_search_max_cuda(t, row_start, row_end, ts);
        // Else, keep row_end the same and increase row_start
        } else {
          row_start = binary_search_min_cuda(t, row_start, row_end, ts);
          row_end = (t_end[thread_idx]) ? binary_search_max_cuda(t_end[thread_idx], row_start, row_end, ts) : row_end;
        }
      }

      if (row_end - row_start == 0) {
        e_cur = -1;
      } else {
        rnd = int64_t(rand[l * numel + thread_idx] * (row_end - row_start));
        e_cur = row_start + rnd;
        n_cur = col[e_cur];
        t = ts[e_cur];
      }
      n_out[(l + 1) * numel + thread_idx] = n_cur;
      e_out[l * numel + thread_idx] = e_cur;
    }
  }
}


std::tuple<torch::Tensor, torch::Tensor>
continuous_trw_cuda(torch::Tensor rowptr, torch::Tensor col, torch::Tensor ts, torch::Tensor start,
                 int64_t walk_length, torch::Tensor t_start, torch::Tensor t_end, bool reverse) {
  CHECK_CUDA(rowptr);
  CHECK_CUDA(col);
  CHECK_CUDA(start);
  CHECK_CUDA(ts);
  CHECK_CUDA(t_start);
  CHECK_CUDA(t_end);
  c10::cuda::MaybeSetDevice(rowptr.get_device());

  CHECK_INPUT(rowptr.dim() == 1);
  CHECK_INPUT(col.dim() == 1);
  CHECK_INPUT(start.dim() == 1);
  CHECK_INPUT(ts.dim() == 1);
  CHECK_INPUT(t_start.dim() == 1);
  CHECK_INPUT(t_end.dim() == 1);

  auto n_out = torch::empty({walk_length + 1, start.size(0)}, start.options());
  auto e_out = torch::empty({walk_length, start.size(0)}, start.options());

  auto stream = at::cuda::getCurrentCUDAStream();

  auto rand = torch::rand({start.size(0), walk_length},
                            start.options().dtype(torch::kFloat));

  continuous_uniform_sampling_kernel<<<BLOCKS(start.numel()), THREADS, 0, stream>>>(
      rowptr.data_ptr<int64_t>(), col.data_ptr<int64_t>(), ts.data_ptr<int64_t>(),
      start.data_ptr<int64_t>(), rand.data_ptr<float>(),
      n_out.data_ptr<int64_t>(), e_out.data_ptr<int64_t>(),
      walk_length, t_start.data_ptr<int64_t>(), t_end.data_ptr<int64_t>(),
      reverse, start.numel());

  return std::make_tuple(n_out.t().contiguous(), e_out.t().contiguous());
}
