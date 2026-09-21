/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef CANN_GRAPH_ENGINE_FUSION_PASS_EXECUTOR_H
#define CANN_GRAPH_ENGINE_FUSION_PASS_EXECUTOR_H
#include "graph/compute_graph.h"
#include "register/register_custom_pass.h"
#include "ge/fusion/pass/fusion_pass_reg.h"
namespace ge {
namespace fusion {
class FusionPassExecutor {
 public:
  /**
   * 对compute graph执行通过REG_FUSION_PASS、REGISTER_CUSTOM_PASS注册到ge的fusion pass
   * @param compute_graph
   * @param stage
   * @return
   */
  Status RunPassesWithLegacyCustom(const ComputeGraphPtr &compute_graph, CustomPassStage stage);

  /**
   * 对compute graph执行通过REG_FUSION_PASS注册到ge的fusion pass
   * 该函数不会对子图再次执行pass，需要pass自行处理
   * @param compute_graph
   * @param stage
   * @return
   */
  Status RunPasses(const ComputeGraphPtr &compute_graph, CustomPassStage stage);

  ~FusionPassExecutor();
private:
 Status InitPassesIfNeed(CustomPassStage stage);
 std::vector<std::pair<std::string, FusionBasePass *>> names_to_fusion_passes_;
 std::map<std::string, bool> pass_name_to_switches_;
};
} // namespace fusion
}  // namespace ge

#endif  // CANN_GRAPH_ENGINE_FUSION_PASS_EXECUTOR_H
