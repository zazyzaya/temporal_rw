#include "temporal_rw_cuda.h"

#include <ATen/cuda/CUDAContext.h>
#include <curand.h>
#include <curand_kernel.h>

#include "utils.cuh"

#define THREADS 1024
#define BLOCKS(N) (N + THREADS - 1) / THREADS

__device__ int64_t binary_search_cuda(const int64_t tgt,
                                 const int64_t st_idx,
                                 const int64_t en_idx,
                                 const int64_t *ts) {
  /*
    Return lowest idx of ts that is >= tgt
    Assumes ts is sorted between ts[start] and ts[end]

    Rewritten as a loop instead of recursion for performance improvement
  */
  // Convert from const
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

__global__ void uniform_sampling_kernel(const int64_t *rowptr,
                                        const int64_t *col,
                                        const int64_t *ts,
                                        const int64_t *start, const float *rand,
                                        int64_t *n_out, int64_t *e_out,
                                        const int64_t walk_length,
                                        const int64_t numel) {

  const int64_t thread_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (thread_idx < numel) {
    int64_t n_cur = start[thread_idx], e_cur = 0, row_start, row_end, rnd;

    n_out[thread_idx] = n_cur;

    int64_t t = 0; // Assume t0 = 0

    for (int64_t l = 0; l < walk_length; l++) {
      row_start = rowptr[n_cur], row_end = rowptr[n_cur + 1];

      if (e_cur == -1) {
        row_start = row_end;
      } else {
        row_start = binary_search_cuda(t, row_start, row_end, ts);
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
temporal_random_walk_cuda(torch::Tensor rowptr, torch::Tensor col, torch::Tensor ts, torch::Tensor start,
                 int64_t walk_length, double p, double q) {
  CHECK_CUDA(rowptr);
  CHECK_CUDA(col);
  CHECK_CUDA(start);
  CHECK_CUDA(ts);
  c10::cuda::MaybeSetDevice(rowptr.get_device());

  CHECK_INPUT(rowptr.dim() == 1);
  CHECK_INPUT(col.dim() == 1);
  CHECK_INPUT(start.dim() == 1);
  CHECK_INPUT(ts.dim() == 1);

  auto n_out = torch::empty({walk_length + 1, start.size(0)}, start.options());
  auto e_out = torch::empty({walk_length, start.size(0)}, start.options());

  auto stream = at::cuda::getCurrentCUDAStream();

  auto rand = torch::rand({start.size(0), walk_length},
                            start.options().dtype(torch::kFloat));

  // p and q not implemented
  uniform_sampling_kernel<<<BLOCKS(start.numel()), THREADS, 0, stream>>>(
      rowptr.data_ptr<int64_t>(), col.data_ptr<int64_t>(), ts.data_ptr<int64_t>(),
      start.data_ptr<int64_t>(), rand.data_ptr<float>(),
      n_out.data_ptr<int64_t>(), e_out.data_ptr<int64_t>(), walk_length,
      start.numel());

  return std::make_tuple(n_out.t().contiguous(), e_out.t().contiguous());
}
