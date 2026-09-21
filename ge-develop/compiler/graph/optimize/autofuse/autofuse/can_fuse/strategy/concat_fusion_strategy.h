/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AUTOFUSE_CAN_FUSE_STRATEGY_CONCAT_FUSION_STRATEGY_H_
#define AUTOFUSE_CAN_FUSE_STRATEGY_CONCAT_FUSION_STRATEGY_H_
#include "ge_common/ge_api_types.h"
#include "can_fuse/strategy/fusion_strategy.h"

namespace ge {
class ConcatFusionStrategy : public FusionStrategy {
 public:
  ConcatFusionStrategy() = default;

  ~ConcatFusionStrategy() override = default;

  ConcatFusionStrategy(const ConcatFusionStrategy &) = delete;
  ConcatFusionStrategy &operator=(const ConcatFusionStrategy &) = delete;

  // 检查两个节点是否可以融合
  bool CanFuse(const NodePtr &node1, const NodePtr &node2) override;

  // 检查两个节点是否可以循环合并
  bool CanMergeLoop(const NodePtr &node1, const NodePtr &node2) override;

  // 获取融合对的优先级
  FusionPriority GetFusionPairPriority(const NodePtr &node1, const NodePtr &node2) override;

  // 获取融合节点里原始节点最大个数限制
  uint64_t GetMaxFusionNodesSize(const NodePtr &node1, const NodePtr &node2) override;

  // 获取融合节点输入最大个数限制
  uint32_t GetMaxFusionNodeInputSize(const NodePtr &node1, const NodePtr &node2) override;

  // 检查node1和node2的循环轴是否相同，如果相同则可以融合，反之不能融合
  bool CheckSameSchedAxis(const NodePtr &node1, const NodePtr &node2, const AxisPairSet &node1_map,
                          const AxisPairSet &node2_map, const NodeFuseInfo &node_fuse_info) override;

 private:
  static Status CanFuseBackward(const NodePtr &node1, const NodePtr &node2, const AutoFuseAttrs *attr2);
  static bool IsFirstDimSplit(const NodePtr &node);
  static bool IsFirstDimConcat(const NodePtr &node);
  static Status CanFuseSplit(const NodePtr &node1, const NodePtr &node2);
  static AscNodePtr FindConcatNode(const NodePtr &backend_node);
  static bool IsConcatOrSplitFirstDim(const AscNodePtr &concat_node);
  static Status IsSplitLinkToBackwardFusionNode(const NodePtr &split_node, const NodePtr &concat_node, bool &can_fuse);
  static Status CollectBackwardFusionRelatedInputs(const NodePtr &fused_node, std::set<int32_t> &indices);
};
}  // namespace ge

#endif  // AUTOFUSE_CAN_FUSE_STRATEGY_CONCAT_FUSION_STRATEGY_H_
