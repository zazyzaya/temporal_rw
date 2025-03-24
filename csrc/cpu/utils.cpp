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
