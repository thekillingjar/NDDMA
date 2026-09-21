/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_GRAPH_BUILD_RUN_CONTEXT_H_
#define GE_GRAPH_BUILD_RUN_CONTEXT_H_

#include <vector>
#include "framework/common/ge_inner_error_codes.h"
#include "common/opskernel/ops_kernel_info_types.h"
#include "framework/common/types.h"
#include "graph/compute_graph.h"
#include "graph/model.h"
#include "runtime/rt.h"

namespace ge {
/*lint -e148*/
class RunContextUtil {
 public:
  RunContextUtil() = default;

  virtual ~RunContextUtil() = default;

  // Init mem info.
  ge::Status InitMemInfo(uint8_t *data_mem_base, uint64_t data_mem_size,
                         std::map<int64_t, uint8_t *> mem_type_to_data_mem_base,
                         std::map<int64_t, uint64_t> mem_type_to_data_mem_size,
                         uint8_t *weight_mem_base, uint64_t weight_mem_size);

  ge::Status CreateRunContext(Model &model, const ComputeGraphPtr &graph, Buffer &buffer,
                              const uint64_t session_id);

  RunContext &GetRunContext();

  void PrintMemInfo() const;

  RunContext run_context_;

 private:
  // Mem info
  uint8_t *data_mem_base_ = nullptr;
  uint64_t data_mem_size_ = 0;
  uint8_t *weight_mem_base_ = nullptr;
  uint64_t weight_mem_size_ = 0;
  std::map<int64_t, uint8_t *> mem_type_to_data_mem_base_;
  std::map<int64_t, uint64_t> mem_type_to_data_mem_size_;
};
}  // namespace ge
#endif  // GE_GRAPH_BUILD_RUN_CONTEXT_H_
