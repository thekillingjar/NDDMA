/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FUSION_ENGINE_OPTIMIZER_GRAPH_OPTIMIZER_OP_JUDGE_OP_JUDGE_BASE_H_
#define FUSION_ENGINE_OPTIMIZER_GRAPH_OPTIMIZER_OP_JUDGE_OP_JUDGE_BASE_H_

#include "common/fe_inner_error_codes.h"
#include "common/fe_log.h"
#include "common/fe_utils.h"
#include "common/fe_op_info_common.h"
#include "common/util/op_info_util.h"
#include "graph/compute_graph.h"

namespace fe {
class OpJudgeBase {
 public:
  explicit OpJudgeBase(const std::string& engine_name);
  virtual ~OpJudgeBase();

  /**
   * judge the set the highest prior imply type of op
   * @param [in|out] graph compute graph
   * @return SUCCESS or FAILED
   */
  virtual Status Judge(ge::ComputeGraph &graph) = 0;
  virtual Status SetFormat(ge::ComputeGraph &graph) = 0;

 protected:
  std::string engine_name_;
};
}  // namespace fe

#endif  // FUSION_ENGINE_OPTIMIZER_GRAPH_OPTIMIZER_OP_JUDGE_OP_JUDGE_BASE_H_
