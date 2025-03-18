#ifdef WITH_PYTHON
#include <Python.h>
#endif
#include <torch/script.h>

#include "cpu/temporal_rw_cpu.h"

#ifdef WITH_CUDA
#include "cuda/temporal_rw_cuda.h"
#endif

#ifdef _WIN32
#ifdef WITH_PYTHON
#ifdef WITH_CUDA
PyMODINIT_FUNC PyInit__temporal_rw_cuda(void) { return NULL; }
#else
PyMODINIT_FUNC PyInit__temporal_rw_cpu(void) { return NULL; }
#endif
#endif
#endif

TEMPORAL_RW_API std::tuple<torch::Tensor, torch::Tensor>
temporal_random_walk(torch::Tensor rowptr, torch::Tensor col, torch::Tensor ts, torch::Tensor start,
            int64_t walk_length) {
  if (rowptr.device().is_cuda()) {
#ifdef WITH_CUDA
    return temporal_random_walk_cuda(rowptr, col, ts, start, walk_length);
#else
    AT_ERROR("Not compiled with CUDA support");
#endif
  } else {
    return temporal_random_walk_cpu(rowptr, col, ts, start, walk_length);
  }
}

static auto registry =
    torch::RegisterOperators().op("temporal_rw::temporal_random_walk", &temporal_random_walk);
