/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "cube_fusion_strategy.h"
#include "backend/backend_spec.h"
#include "can_fuse/backend/backend_utils.h"
#include "post_process/scheduler_adapter/adaption_complete_node_attrs.h"
#include "utils/autofuse_attrs.h"
#include "can_fuse/strategy/fusion_strategy_registry.h"
#include "utils/auto_fuse_config.h"
#include "utils/autofuse_utils.h"
#include "utils/not_fuse_reason_code.h"

namespace ge {
using namespace autofuse;
bool HasBatchBroadcast(int32_t axis_size, std::vector<ViewOpAttrInfo> &attr_infos, const NodePtr &node2) {
  auto batch_axis_size = axis_size - 2U;
  for (const auto &attr_info : attr_infos) {
    GELOGD("try to find batch axis(axis size %d) broadcast with broadcast info(%s) in node2 %s(%s)", axis_size,
           AutofuseUtils::VectorToStr(attr_info.broadcast_info).c_str(), node2->GetNamePtr(), node2->GetType().c_str());
    for (auto axis_id : attr_info.broadcast_info) {
      // 非m、n轴broadcast则是batch轴broadcast，n、m轴总是总是在内轴
      if (axis_id < batch_axis_size) {
        return true;
      }
    }
  }
  return false;
}

bool IsIndirectLinksWithoutBatchBroadcast(const NodePtr &node1, const NodePtr &node2,
                                          const std::vector<std::pair<int32_t, int32_t>> &direct_links) {
  std::unordered_set<int32_t> direct_links_node2_indices;
  for (const auto &link : direct_links) {
    direct_links_node2_indices.insert(link.second);
  }

  TensorAttrInfo temp_graph_attr;
  auto asc_graph = BackendUtils::GetNodeFusedAscGraph(node1);
  GE_ASSERT_NOTNULL(asc_graph);
  GE_ASSERT_SUCCESS(BackendUtils::GetGraphAttrInfo(*asc_graph, temp_graph_attr));
  GE_ASSERT_TRUE(temp_graph_attr.axis.size() < static_cast<uint32_t>(std::numeric_limits<int32_t>::max()));
  auto axis_size = static_cast<int32_t>(temp_graph_attr.axis.size());
  ComputeGraphPtr graph;
  GE_ASSERT_SUCCESS(BackendUtils::GetNodeFusedGraph(node2, graph));
  const auto subgraph_input_nodes = graph->GetInputNodes();
  for (size_t i = 0U; i < subgraph_input_nodes.size(); ++i) {
    int32_t index = static_cast<int32_t>(i);
    GELOGD("try to find batch axis broadcast in node1 %s(%s) input(%d)", node1->GetNamePtr(), node1->GetType().c_str(),
           index);
    if (direct_links_node2_indices.find(index) == direct_links_node2_indices.end()) {
      std::vector<ViewOpAttrInfo> attr_infos;
      if (!BackendUtils::CurNodeInputIsSimplestLoad(node2, index, attr_infos, false) &&
          HasBatchBroadcast(axis_size, attr_infos, node2)) {
        GELOGI(
            "node1 %s(%s) and node2 %s(%s) can not fuse, the reason is[%s] [node1 is Cube, node2 is Pointwise but "
            "node2 has batch axis broadcast].",
            node1->GetNamePtr(), node1->GetType().c_str(), node2->GetNamePtr(), node2->GetType().c_str(),
            ge::NotFuseReasonCode(ge::NotFuseReason::kCubeCanNotFuseWithViewElementwise));
        return false;
      }
    }
  }
  return true;
}

bool CanFuseWithElementwise(const NodePtr &node1, const NodePtr &node2) {
  const auto attr1 = BackendUtils::GetNodeAutoFuseAttr(node1);
  GE_ASSERT_NOTNULL(attr1);
  const auto attr2 = BackendUtils::GetNodeAutoFuseAttr(node2);
  GE_ASSERT_NOTNULL(attr2);
  // cube不支持水平融合
  if (!BackendUtils::IsVertical(node1, node2)) {
    GELOGI(
        "node1 %s(%s) and node2 %s(%s) can not fuse, the reason is[%s] [node1 is Cube, node2 is Pointwise but "
        "node1 and node2 is not vertical relation].",
        node1->GetNamePtr(), node1->GetType().c_str(), node2->GetNamePtr(), node2->GetType().c_str(),
        ge::NotFuseReasonCode(ge::NotFuseReason::kCubeCanNotFuseHoriZontal));
    return false;
  }
  // node1与node2（elewisement有view只能是broadcast）直连路径中不能包含broadcast，否则后端没法算vector内存占用
  NodeFuseInfo node_fuse_info;
  GE_ASSERT_SUCCESS(node_fuse_info.UpdateNodeFuseInfo(node1, node2));
  for (const auto &subgraph_link : node_fuse_info.GetNode1ToNode2LinkMap()) {
    std::vector<ViewOpAttrInfo> attr_infos;
    if (!BackendUtils::CurNodeInputIsSimplestLoad(node2, subgraph_link.second, attr_infos)) {
      GELOGI(
          "node1 %s(%s) and node2 %s(%s) can not fuse, the reason is[%s] [node1 is Cube, node2 is Pointwise but "
          "node2 input, which link to node1 has view op].",
          node1->GetNamePtr(), node1->GetType().c_str(), node2->GetNamePtr(), node2->GetType().c_str(),
          ge::NotFuseReasonCode(ge::NotFuseReason::kCubeCanNotFuseWithViewElementwise));
      return false;
    }
  }
  // node1和node2非直连路径中包含batch轴和broadcast，后端暂不支持融合
  if (!IsIndirectLinksWithoutBatchBroadcast(node1, node2, node_fuse_info.GetNode1ToNode2LinkMap())) {
    return false;
  }
  std::vector<std::string> target_types1 = {}; // 如果不支持有某个node类型，可以加在这里例如 kScalarType
  if (BackendUtils::HasTypesInAscgraph(node2, target_types1)) {
    GELOGI(
        "node1 %s(%s) and node2 %s(%s) can not fuse, the reason is[%s] [node1 is Cube, node2 is Pointwise but "
        "node2 has type(scalar or cast), which is unsupported].",
        node1->GetNamePtr(), node1->GetType().c_str(), node2->GetNamePtr(), node2->GetType().c_str(),
        ge::NotFuseReasonCode(ge::NotFuseReason::kCubeCanNotFuseWithUnsupportedType));
    return false;
  }
  // node2暂不支持仅有"ExpandDims", "Reshape", "Squeeze", "Unsqueeze"的elementwise,会改轴且走不到ub全载模板
  std::vector<std::string> target_types2 = {kExpandDimsType, kReshapeType, kSqueezeType, kUnsqueezeType};
  if (BackendUtils::OnlyHasTypesInAscgraph(node2, target_types2)) {
    GELOGI(
        "node1 %s(%s) and node2 %s(%s) can not fuse, the reason is[%s] [node1 is Cube, node2 is Pointwise but "
        "node2 has only ExpandDims or Reshape or Squeeze or Unsqueeze].",
        node1->GetNamePtr(), node1->GetType().c_str(), node2->GetNamePtr(), node2->GetType().c_str(),
        ge::NotFuseReasonCode(ge::NotFuseReason::kCubeCanNotFuseWithNotComputeNode));
    return false;
  }
  return true;
}

bool CubeFusionStrategy::CanFuse(const NodePtr &node1, const NodePtr &node2) {
  // 1、cube在部分芯片不支持融合
  const auto backend_spec = optimize::BackendSpec::GetInstance();
  GE_ASSERT_NOTNULL(backend_spec);
  if (!backend_spec->enable_matmul_lowering_to_matmul) {
    GELOGI("node1 %s(%s) and node2 %s(%s) can not fuse, the reason is[%s] [Cube can not fuse in this chip type].",
           node1->GetNamePtr(), node1->GetType().c_str(), node2->GetNamePtr(), node2->GetType().c_str(),
           ge::NotFuseReasonCode(ge::NotFuseReason::kCubeCanNotFuseInThisChip));
    return false;
  }
  const auto attr1 = BackendUtils::GetNodeAutoFuseAttr(node1);
  GE_ASSERT_NOTNULL(attr1);
  const auto attr2 = BackendUtils::GetNodeAutoFuseAttr(node2);
  GE_ASSERT_NOTNULL(attr2);

  // 2.cube不能前融合
  if (attr2->HasFuseType(loop::FuseType::kCube)) {
    GELOGI("node1 %s(%s) and node2 %s(%s) can not fuse, the reason is[%s] [Cube can not fuse forward].",
           node1->GetNamePtr(), node1->GetType().c_str(), node2->GetNamePtr(), node2->GetType().c_str(),
           ge::NotFuseReasonCode(ge::NotFuseReason::kCubeCanNotFuseForward));
    return false;
  }

  // 3.cube不能和非elementwise融合
  uint64_t supported_type = (1UL << static_cast<uint64_t>(loop::FuseType::kPointwise));
  if (!(attr1->HasFuseType(loop::FuseType::kCube) && (attr2->GetAllFuseType() == supported_type))) {
    GELOGI("node1 %s(%s) and node2 %s(%s) can not fuse, the reason is[%s] [node1 is Cube, node2 is not Pointwise].",
           node1->GetNamePtr(), node1->GetType().c_str(), node2->GetNamePtr(), node2->GetType().c_str(),
           ge::NotFuseReasonCode(ge::NotFuseReason::kCubeCanNotFuseWithNotElementwise));
    return false;
  }

  // 4. cube只能和elementwise垂直融合
  if (!CanFuseWithElementwise(node1, node2)) {
    return false;
  }

  return true;
}

// cube 向下垂直融合，设置优先级
FusionPriority CubeFusionStrategy::GetFusionPairPriority(const NodePtr &node1, const NodePtr &node2) {
  auto attr = BackendUtils::GetNodeAutoFuseAttr(node1);
  GE_ASSERT_NOTNULL(attr);
  FusionPriority fusion_priority = FusionPriority::DEFAULT;
  // 首轮融合才要处理，只有AscBackend场景
  if ((attr->GetFuseType() == loop::FuseType::kCube) && BackendUtils::IsVertical(node1, node2)) {
    fusion_priority = FusionPriority::LOW; // 防止影响当前网络已融合结构，在已融合结构基础上再做cube融合
    GELOGI("node1 %s(Cube) --> node2 %s(*) priority:%u.", node1->GetNamePtr(), node2->GetNamePtr(),
           fusion_priority);
  }
  return fusion_priority;
}

uint64_t CubeFusionStrategy::GetMaxFusionNodesSize(const NodePtr &node1, const NodePtr &node2) {
  const auto &config = AutoFuseConfig::Config().GetFusionStrategySolver();
  // cube支持和64个vector融合，1+64
  uint64_t max_fusion_size = config.max_fusion_size + 1U;
  GELOGI("node1 %s(*) and node2 %s(*) max_fusion_nodes_size:%lu.", node1->GetNamePtr(), node2->GetNamePtr(),
         max_fusion_size);
  return max_fusion_size;
}

REGISTER_FUSION_STRATEGY(CubeFusionStrategy, loop::FuseType::kCube);
}  // namespace ge