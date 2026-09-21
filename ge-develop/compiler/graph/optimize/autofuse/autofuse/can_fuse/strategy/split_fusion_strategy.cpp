/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "utils/auto_fuse_config.h"
#include "can_fuse/backend/backend_utils.h"
#include "utils/autofuse_attrs.h"
#include "can_fuse/strategy/fusion_strategy_registry.h"
#include "split_fusion_strategy.h"
#include "utils/not_fuse_reason_code.h"

namespace ge {
bool SplitFusionStrategy::CanFuse(const NodePtr &node1, const NodePtr &node2) {
  const auto attr1 = BackendUtils::GetNodeAutoFuseAttr(node1);
  GE_ASSERT_NOTNULL(attr1);
  const auto attr2 = BackendUtils::GetNodeAutoFuseAttr(node2);
  GE_ASSERT_NOTNULL(attr2);

  if (attr1->HasFuseType(loop::FuseType::kSplit) && attr2->HasFuseType(loop::FuseType::kSplit)) {
    GELOGI("node1 %s(%s) has split global id %d, node2 %s(%s) has split global id %d.",
           node1->GetName().c_str(), node1->GetType().c_str(), attr1->GetSplitGlobalId(),
           node2->GetName().c_str(), node2->GetType().c_str(), attr2->GetSplitGlobalId());
    // 不同的Ascend IR split在Canfuse阶段不相融
    if (attr1->GetSplitGlobalId() != attr2->GetSplitGlobalId()) {
      GELOGI("node1 %s(%s) and node2 %s(%s) can not fuse, the reason is [%s][node1 and node2 are horizontal fuse, "
             "node1 is split, node2 is split, and they are different split nodes before lowering.]",
             node1->GetName().c_str(), node1->GetType().c_str(), node2->GetName().c_str(), node2->GetType().c_str(),
             ge::NotFuseReasonCode(ge::NotFuseReason::kSplitCanNotFuseSplitHorizontal));
      return false;
    }
    // 前端必须保证同一个Ascend IR split lowering成的N个split AscBackend在canfuse阶段融合到同一个AscBackend内
    return true;
  }

  // canfuse框架已经预测过split融合比例过低，复用判断结果提前返回false
  if (attr1->GetSplitFusionRatioRequirementState() == SplitFusionRatioRequirementState::NOT_SATISFIED) {
    // Split类型不支持低融合比例, 当前只支持split后向融合
    GELOGI("node1 %s(%s) and node2 %s(%s) can not fuse, the reason is [%s][node1 or node2 has low fuse ratio]",
           node1->GetName().c_str(), node1->GetType().c_str(), node2->GetName().c_str(), node2->GetType().c_str(),
           ge::NotFuseReasonCode(ge::NotFuseReason::kSplitLowFuseRatio));
    return false;
  }
  // split不与reduction融合
  if ((attr1->HasFuseType(loop::FuseType::kSplit) && attr2->HasFuseType(loop::FuseType::kReduction)) ||
      (attr1->HasFuseType(loop::FuseType::kReduction) && attr2->HasFuseType(loop::FuseType::kSplit))) {
    GELOGI("node1 %s(%s) and node2 %s(%s) can not fuse, the reason is [%s][split cannot fuse reduction]",
           node1->GetName().c_str(), node1->GetType().c_str(), node2->GetName().c_str(), node2->GetType().c_str(),
           ge::NotFuseReasonCode(ge::NotFuseReason::kSplitCanNotFuseReduction));
    return false;
  }
  if (BackendUtils::IsVertical(node1, node2) && attr2->HasFuseType(loop::FuseType::kSplit)) {
    // node2 为 Split 类型时，不支持向前融合
    GELOGI("node1 %s(%s) and node2 %s(%s) can not fuse, the reason is [%s][node1 and node2 are vertical fuse, "
           "and node2 is split, split can not fuse forward]", node1->GetName().c_str(), node1->GetType().c_str(),
           node2->GetName().c_str(), node2->GetType().c_str(),
           ge::NotFuseReasonCode(ge::NotFuseReason::kSplitCanNotFuseForward));
    return false;
  }
  // split不做水平融合
  if (!BackendUtils::IsVertical(node1, node2)) {
    GELOGI("node1 %s(%s) and node2 %s(%s) can not fuse, the reason is [%s][split cannot fuse other node horizontal.]",
           node1->GetName().c_str(), node1->GetType().c_str(), node2->GetName().c_str(), node2->GetType().c_str(),
           ge::NotFuseReasonCode(ge::NotFuseReason::kSplitCanNotFuseHorizontal));
    return false;
  }
  return true;
}

uint64_t SplitFusionStrategy::GetMaxFusionNodesSize(const NodePtr &node1, const NodePtr &node2) {
  const auto &config = autofuse::AutoFuseConfig::Config().GetFusionStrategySolver();
  uint64_t max_fusion_size = config.max_fusion_size;
  // split和输出融合时不限制个数
  if (BackendUtils::IsVertical(node1, node2)) {
    const auto attr = BackendUtils::GetNodeAutoFuseAttr(node1);
    GE_ASSERT_NOTNULL(attr);
    if (attr->HasFuseType(loop::FuseType::kSplit)) {
      max_fusion_size = std::numeric_limits<uint64_t>::max();
    }
  } else if (BackendUtils::IsHorizontal(node1, node2)) {
    const auto attr1 = BackendUtils::GetNodeAutoFuseAttr(node1);
    GE_ASSERT_NOTNULL(attr1);
    const auto attr2 = BackendUtils::GetNodeAutoFuseAttr(node2);
    GE_ASSERT_NOTNULL(attr2);
    if (attr1->HasFuseType(loop::FuseType::kSplit) && attr2->HasFuseType(loop::FuseType::kSplit)) {
      max_fusion_size = std::numeric_limits<uint64_t>::max();
    }
  }
  GELOGI("node1 %s(*) and node2 %s(*) max_fusion_nodes_size: %lu.", node1->GetName().c_str(), node2->GetName().c_str(),
         max_fusion_size);
  return max_fusion_size;
}

FusionPriority SplitFusionStrategy::GetFusionPairPriority(const NodePtr &node1, const NodePtr &node2) {
  const auto attr1 = BackendUtils::GetNodeAutoFuseAttr(node1);
  GE_ASSERT_NOTNULL(attr1);
  const auto attr2 = BackendUtils::GetNodeAutoFuseAttr(node2);
  GE_ASSERT_NOTNULL(attr2);
  auto fusion_priority = FusionPriority::HIGHER;
  if ((attr1->GetFuseType() == loop::FuseType::kSplit) && (attr2->GetFuseType() == loop::FuseType::kSplit)) {
    fusion_priority = FusionPriority::HIGHEST;
    GELOGI("node1 %s(%s) and node2 %s(%s) priority:%u.", node1->GetName().c_str(), node1->GetType().c_str(),
      node2->GetName().c_str(), node2->GetType().c_str(), fusion_priority);
  }
  return fusion_priority;
}

REGISTER_FUSION_STRATEGY(SplitFusionStrategy, loop::FuseType::kSplit);
} // namespace ge