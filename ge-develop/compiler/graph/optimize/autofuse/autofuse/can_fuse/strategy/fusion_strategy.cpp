/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "fusion_strategy.h"
#include "can_fuse/backend/backend_utils.h"
#include "utils/autofuse_attrs.h"
#include "utils/auto_fuse_config.h"

namespace ge {
bool FusionStrategy::CanFuse(const NodePtr &node1, const NodePtr &node2) {
  GELOGI("node1 %s(*) and node2 %s(*) can fuse.", node1->GetNamePtr(), node2->GetNamePtr());
  return true;
}

bool FusionStrategy::CanMergeLoop(const NodePtr &node1, const NodePtr &node2) {
  GELOGI("node1 %s(*) and node2 %s(*) can merge loop.", node1->GetNamePtr(), node2->GetNamePtr());
  return true;
}

FusionPriority FusionStrategy::GetFusionPairPriority(const NodePtr &node1, const NodePtr &node2) {
  GELOGI("node1 %s(*) and node2 %s(*) priority:%u.", node1->GetNamePtr(), node2->GetNamePtr(), FusionPriority::DEFAULT);
  return FusionPriority::DEFAULT;
}

uint64_t FusionStrategy::GetMaxFusionNodesSize(const NodePtr &node1, const NodePtr &node2) {
  const auto &config = autofuse::AutoFuseConfig::Config().GetFusionStrategySolver();
  GELOGI("node1 %s(*) and node2 %s(*) max_fusion_nodes_size:%lu.", node1->GetNamePtr(), node2->GetNamePtr(),
         config.max_fusion_size);
  return config.max_fusion_size;
}

uint32_t FusionStrategy::GetMaxFusionNodeInputSize(const NodePtr &node1, const NodePtr &node2) {
  const auto &config = autofuse::AutoFuseConfig::Config().GetFusionStrategySolver();
  GELOGI("node1 %s(*) and node2 %s(*) max_fusion_node_input_size:%u.", node1->GetNamePtr(), node2->GetNamePtr(),
         config.max_input_nums_after_fuse);
  return config.max_input_nums_after_fuse;
}

bool FusionStrategy::OnlyVerticalMapping(const NodePtr &node1, const NodePtr &node2,
                                         const NodeFuseInfo &fuse_info) {
  bool is_both_horizontal_and_vertical =
      !fuse_info.GetNode1ToNode2LinkMap().empty() && !fuse_info.GetSameInputMap().empty();
  std::string is_both_horizontal_and_vertical_str = is_both_horizontal_and_vertical ? "true" : "false";
  GELOGD("node1 %s(%s) and node2 %s(%s) is_both_horizontal_and_vertical: %s.", node1->GetNamePtr(),
         node1->GetType().c_str(), node2->GetNamePtr(), node2->GetType().c_str(),
         is_both_horizontal_and_vertical_str.c_str());
  return is_both_horizontal_and_vertical;
}

bool FusionStrategy::CheckSameSchedAxis(const NodePtr &node1, const NodePtr &node2,
                                        const AxisPairSet &node1_map, const AxisPairSet &node2_map,
                                        const NodeFuseInfo &node_fuse_info) {
  (void)node1_map;
  (void)node2_map;
  (void)node_fuse_info;
  GELOGI("node1 %s(%s) and node2 %s(%s) can fuse by same sched axis.", node1->GetNamePtr(), node1->GetType().c_str(),
         node2->GetNamePtr(), node2->GetType().c_str());
  return true;
}
}  // namespace ge
