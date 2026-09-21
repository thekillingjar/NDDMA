#ifndef HW_GE_ATT_NDDMA_TILING_H
#define HW_GE_ATT_NDDMA_TILING_H

#include <cstdint>

struct NddmaTilingData {
  uint32_t dtype_id;
  uint32_t dim;
  uint32_t repeat;
  uint32_t enable_store;
  uint32_t block_dim;
  uint32_t total_elems;
  uint32_t logical_total_elems;
  uint32_t ub_buffer_bytes;
  uint32_t src_offset_elem;
  uint32_t dst_offset_elem;
  uint32_t src_core_stride_elems;
  uint32_t dst_core_stride_elems;
  int64_t output_dims[5];
  int64_t output_stride[5];
  int64_t input_stride[5];
};

enum NddmaDtypeId : uint32_t {
  NDDMA_DTYPE_INT64 = 0,
  NDDMA_DTYPE_INT32 = 1,
  NDDMA_DTYPE_INT16 = 2,
  NDDMA_DTYPE_INT8 = 3,
};

#endif
