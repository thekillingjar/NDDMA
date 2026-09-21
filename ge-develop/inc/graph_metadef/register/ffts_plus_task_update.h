/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef INC_REGISTER_FFTS_PLUS_TASK_UPDATE_H_
#define INC_REGISTER_FFTS_PLUS_TASK_UPDATE_H_

#include <vector>

#include "graph/node.h"
#include "register/op_tiling_registry.h"
#include "runtime/rt_ffts_plus.h"
#include "external/ge_common/ge_api_error_codes.h"

namespace ge {
struct AutoThreadSubTaskFlush {
  int32_t device_id{0};
  void *args_base{nullptr};
  std::vector<optiling::utils::OpRunInfo> op_run_info;

  uintptr_t aic_non_tail_task_start_pc{0U};
  uintptr_t aic_tail_task_start_pc{0U};
  uint32_t aic_icache_prefetch_cnt{0U};

  uintptr_t aiv_non_tail_task_start_pc{0U};
  uintptr_t aiv_tail_task_start_pc{0U};
  uint32_t aiv_icache_prefetch_cnt{0U};

  // Task I/O Addrs.
  std::vector<uintptr_t> input_addr_base;
  std::vector<uintptr_t> output_addr_base;
};

struct AutoThreadParam {
  uint16_t thread_dim{0U};  // thread dim after Pre-Thread
  uint32_t input_output_num{0U};  // input + output
  std::vector<uint64_t> task_addr_offset; // input + output + workspace

  // Task Thread Dims.
  std::vector<std::vector<std::vector<int64_t>>> *task_input_shape{nullptr}; // thread<input>
  std::vector<std::vector<std::vector<int64_t>>> *task_output_shape{nullptr}; // thread<output>
};

class FFTSPlusTaskUpdate {
 public:
  FFTSPlusTaskUpdate() = default;
  virtual ~FFTSPlusTaskUpdate() = default;

  virtual Status GetAutoThreadParam(const NodePtr &node, const std::vector<optiling::utils::OpRunInfo> &op_run_info,
                                    AutoThreadParam &auto_thread_param) {
    (void)node;
    (void)op_run_info;
    (void)auto_thread_param;
    return SUCCESS;
  }

  virtual Status UpdateSubTaskAndCache(const NodePtr &node, const AutoThreadSubTaskFlush &sub_task_flush,
                                       rtFftsPlusTaskInfo_t &ffts_plus_task_info) {
    (void)node;
    (void)sub_task_flush;
    (void)ffts_plus_task_info;
    return SUCCESS;
  }

  virtual Status UpdateCommonCtx(const ComputeGraphPtr &sgt_graph, rtFftsPlusTaskInfo_t &task_info) {
    (void)sgt_graph;
    (void)task_info;
    return SUCCESS;
  }

  virtual Status UpdateStaticDataCtx(size_t ctx_num, std::vector<uint64_t> &io_addrs, size_t align_offset,
                                     size_t host_io_base, std::map<size_t, std::vector<uint32_t>> &ctx_ids_map) {
    (void)ctx_num;
    (void)io_addrs;
    (void)align_offset;
    (void)host_io_base;
    (void)ctx_ids_map;
    return SUCCESS;
  }
};
} // namespace ge
#endif // INC_REGISTER_FFTS_PLUS_TASK_UPDATE_H_
