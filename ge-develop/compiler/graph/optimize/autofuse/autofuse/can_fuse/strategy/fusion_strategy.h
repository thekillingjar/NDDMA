/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AUTOFUSE_CAN_FUSE_FUSION_STRATEGY_H_
#define AUTOFUSE_CAN_FUSE_FUSION_STRATEGY_H_
#include "ge_common/ge_api_types.h"
#include "graph/node.h"
#include "utils/autofuse_attrs.h"
#include "can_fuse/backend/asc_graph_axis_mapping.h"

namespace ge {
class FusionStrategy {
 public:
  FusionStrategy() = default;

  virtual ~FusionStrategy() = default;

  FusionStrategy(const FusionStrategy &) = delete;
  FusionStrategy &operator=(const FusionStrategy &) = delete;

  // 检查两个节点是否可以融合
  virtual bool CanFuse(const NodePtr &node1, const NodePtr &node2);

  // 检查两个节点是否可以循环合并
  virtual bool CanMergeLoop(const NodePtr &node1, const NodePtr &node2);

  // 获取融合对的优先级
  virtual FusionPriority GetFusionPairPriority(const NodePtr &node1, const NodePtr &node2);

  // 获取融合节点里原始节点最大个数限制
  virtual uint64_t GetMaxFusionNodesSize(const NodePtr &node1, const NodePtr &node2);

  // 获取融合节点输入最大个数限制
  virtual uint32_t GetMaxFusionNodeInputSize(const NodePtr &node1, const NodePtr &node2);

  // 获取node1和node2是否同时有水平融合以及垂直融合
  virtual bool OnlyVerticalMapping(const NodePtr &node1, const NodePtr &node2, const NodeFuseInfo &fuse_info);

  // 检查node1和node2的循环轴是否相同
  virtual bool CheckSameSchedAxis(const NodePtr &node1, const NodePtr &node2, const AxisPairSet &node1_map,
                                  const AxisPairSet &node2_map, const NodeFuseInfo &node_fuse_info);
};
}  // namespace ge

#endif  // AUTOFUSE_CAN_FUSE_FUSION_STRATEGY_H_
