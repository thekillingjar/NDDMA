/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_COMPILER_GRAPH_OPTIMIZE_AUTOFUSE_CAN_FUSE_STRATEGY_CUBE_FUSION_STRATEGY_H_
#define AIR_CXX_COMPILER_GRAPH_OPTIMIZE_AUTOFUSE_CAN_FUSE_STRATEGY_CUBE_FUSION_STRATEGY_H_
#include "ge_common/ge_api_types.h"
#include "can_fuse/strategy/fusion_strategy.h"

namespace ge {
class CubeFusionStrategy : public FusionStrategy {
 public:
  CubeFusionStrategy() = default;

  ~CubeFusionStrategy() override = default;

  CubeFusionStrategy(const CubeFusionStrategy &) = delete;
  CubeFusionStrategy &operator=(const CubeFusionStrategy &) = delete;

  // 检查两个节点是否可以融合
  bool CanFuse(const NodePtr &node1, const NodePtr &node2) override;

  // 获取融合对的优先级
  FusionPriority GetFusionPairPriority(const NodePtr &node1, const NodePtr &node2) override;

  uint64_t GetMaxFusionNodesSize(const NodePtr &node1, const NodePtr &node2) override;
};
}  // namespace ge

#endif  // AIR_CXX_COMPILER_GRAPH_OPTIMIZE_AUTOFUSE_CAN_FUSE_STRATEGY_CUBE_FUSION_STRATEGY_H_
