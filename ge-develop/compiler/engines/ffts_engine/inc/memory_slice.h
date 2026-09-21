/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FFTS_ENGINE_TASK_BUILDER_MODE_MEMORY_SLICE_H_
#define FFTS_ENGINE_TASK_BUILDER_MODE_MEMORY_SLICE_H_
#include "common/sgt_slice_type.h"
#include "inc/ffts_error_codes.h"
#include "inc/ffts_log.h"
#include "graph/compute_graph.h"
#include "common/string_utils.h"

namespace ffts {
struct FftsPlusContextPath {
  uint32_t ctx_id;
  uint32_t pre_cnt;
  uint16_t policy_pri;
  int32_t max_pre_index;
  vector<uint32_t> cmo_list; // label context_id
  vector<uint32_t> label_list; // label context_id
  vector<uint32_t> pre_list; // pre list context_id
  vector<uint32_t> succ_list; // succ lsit context_id
};

#define RT_CTX_TYPE_NOOP_LABEL (RT_CTX_TYPE_LABEL + 1)

struct DataContextParam {
  int64_t num_inner;
  int64_t len_inner;
  int64_t stride_inner;
  int64_t num_outter;
  int64_t stride_outter;
  int64_t base_addr_offset;
  uint32_t index;
};
const int32_t kBlockDim = 3;
struct Block {
  int64_t dim[kBlockDim];
  int64_t dim_stride[kBlockDim];
  int64_t offset;
  int64_t count;
  int64_t stride;
};

class MemorySlice {
 public:
  MemorySlice();
  ~MemorySlice();

  static Status GenerateManualDataCtxParam(const ge::NodePtr &node, int32_t index, bool is_input, int64_t burst_len,
                                           std::vector<DataContextParam> &data_ctx);

  static Status GenerateAutoDataCtxParam(const ge::NodePtr &node, int32_t index, bool is_input, int64_t burst_len,
                                         std::vector<DataContextParam> &param_nontail_tail);

  static Status GenerateDataCtxParam(const std::vector<int64_t> &shape, const std::vector<DimRange> &slice,
                                     ge::DataType dtype, int64_t burst_len, std::vector<DataContextParam> &data_ctx);

 private:
  MemorySlice(const MemorySlice &builder) = delete;
  MemorySlice &operator=(const MemorySlice &builder) = delete;
};
}  // namespace ffts
#endif // FFTS_ENGINE_TASK_BUILDER_MODE_MEMORY_SLICE_H_
