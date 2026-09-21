/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_RUNTIME_V2_LOWERING_PASS_PRUNER_H_
#define AIR_CXX_RUNTIME_V2_LOWERING_PASS_PRUNER_H_
#include "graph/node.h"
#include "graph/fast_graph/fast_node.h"
namespace gert {
namespace bg {
class Pruner {
 public:
  /**
   * 起始节点是否必须被删除，默认为起始节点不满足剪枝条件时，就不会被删除。
   * 如果调用了本接口，那么认为起始节点必须被删除，此时起始节点不满足剪枝条件时，剪枝失败
   * @return this
   */
  Pruner &StartNodesMustBeDeleted();
  ge::graphStatus PruneFromNodes(const std::vector<ge::FastNode *> &start_nodes, bool &changed);

 private:
  bool start_nodes_must_be_deleted_ = false;
};
}  // namespace bg
}  // namespace gert

#endif  // AIR_CXX_RUNTIME_V2_LOWERING_PASS_PRUNER_H_
