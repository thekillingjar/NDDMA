/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph/passes/shape_optimize/reshape_remove_pass.h"

#include  <map>
#include  <string>

#include "framework/common/util.h"
#include "framework/common/types.h"
#include "graph/passes/pass_utils.h"
#include "graph/utils/node_utils.h"
#include "graph/debug/ge_attr_define.h"

namespace ge {
namespace {
const int32_t kReshapeDataIndex = 0;
const int32_t kReshapeShapeIndex = 1;
enum OpHashValue {
  kReshapeType = 0,
  kReformatType = 1,
  kOpNoDelete = -1
};

std::map<std::string, OpHashValue> kToBeDeleteOp = {{RESHAPE, kReshapeType}, {REFORMAT, kReformatType}};
// todo 临时方案，不应该判断节点类型，应该找到这类节点的共同点，或者最终把reshape全部删除
const std::set<std::string> kInputShapeContinue = {GATHERSHAPES, GATHERND};

bool EnablePass(const ge::NodePtr &node) {
  // todo 临时方案，编译时不应该感知单算子
  // 单算子图中（ATTR_SINGLE_OP_SCENE）,如果图上有动态算子，跳过该pass
  const auto compute_graph = node->GetOwnerComputeGraph();
  bool is_single_op = false;
  (void)ge::AttrUtils::GetBool(compute_graph, ge::ATTR_SINGLE_OP_SCENE, is_single_op);
  if (is_single_op) {
    for (const auto &n : compute_graph->GetDirectNode()) {
      bool is_dynamic = false;
      // todo 这个属性是FE打的，单算子场景下ge没有打动态属性，可能略不靠谱，后面要整改
      (void)ge::AttrUtils::GetBool(n->GetOpDesc(), "_unknown_shape", is_dynamic);
      if (is_dynamic) {
        GELOGD("Node %s owner graph %s is dynamic. Skip pass.", node->GetName().c_str(),
               compute_graph->GetName().c_str());
        return false;
      }
    }
  }
  if (compute_graph->GetGraphUnknownFlag()) {
    GELOGD("Node %s owner graph %s is dynamic. Skip pass.", node->GetName().c_str(), compute_graph->GetName().c_str());
    return false;
  }
  return true;
}

bool IsOutDataNodeRequireInputShapeContinuous(const ge::NodePtr &node) {
  for (const auto &out_data_node : node->GetOutDataNodes()) {
    if (kInputShapeContinue.count(out_data_node->GetType()) != 0U) {
      GELOGD("Node: %s, out data node: %s, type: %s, require input shape to be continuous.", node->GetName().c_str(),
             out_data_node->GetName().c_str(), out_data_node->GetType().c_str());
      return true;
    }
  }
  return false;
}

// if reshape is the output of static subgraph, it may be the real input of nodes in dynamic subgraph, which require
// input shape to be continuous when doing infershape.
bool IsOutputOfSubGraph(const ge::NodePtr &node) {
  for (const auto &out_data_node : node->GetOutDataNodes()) {
    if (out_data_node->GetType() != NETOUTPUT) {
      continue;
    }
    const auto &parent_node = node->GetOwnerComputeGraph()->GetParentNode();
    // reshape can not exist in ffts+ graph
    if ((parent_node != nullptr) && (parent_node->GetOpDesc() != nullptr) &&
        (!parent_node->GetOpDesc()->HasAttr(ATTR_NAME_FFTS_PLUS_SUB_GRAPH)) &&
        (!parent_node->GetOpDesc()->HasAttr(ATTR_NAME_FFTS_SUB_GRAPH))) {
      GELOGD("Reshape node: %s is the output of subgraph: %s, should be retained.", node->GetName().c_str(),
             node->GetOwnerComputeGraph()->GetName().c_str());
      return true;
    }
  }
  return false;
}
}

Status ReshapeRemovePass::Run(NodePtr &node) {
  GE_CHECK_NOTNULL(node);
  GE_CHECK_NOTNULL(node->GetOpDesc());
  const auto it = kToBeDeleteOp.find(node->GetType());
  int32_t key = (it == kToBeDeleteOp.cend()) ? kOpNoDelete : it->second;
  switch (key) {
    case kReshapeType: {
      if (!EnablePass(node)) {
        return SUCCESS;
      }
      bool is_shape_unknown = false;
      if (NodeUtils::GetNodeUnknownShapeStatus(*node, is_shape_unknown) == GRAPH_SUCCESS) {
        if (is_shape_unknown) {
          GELOGI("op:%s is unknown shape, can not be deleted.",
                 node->GetName().c_str());
          return SUCCESS;
        }
      }
      if (IsOutputOfSubGraph(node) || IsOutDataNodeRequireInputShapeContinuous(node)) {
        return SUCCESS;
      }
      break;
    }
    case kReformatType:
      break;
    default:
      return SUCCESS;
  }

  GELOGI("Remove %s node %s.", node->GetType().c_str(), node->GetName().c_str());
  auto shape_const = NodeUtils::GetInDataNodeByIndex(*node, kReshapeShapeIndex);
  auto ret = IsolateAndDeleteNode(node, {kReshapeDataIndex});
  GE_CHK_STATUS(ret, "Failed to isolate and delete node %s.", node->GetName().c_str());
  // try delete shape const which only has control out
  if (shape_const == nullptr) {
    return SUCCESS;
  }
  if ((shape_const->GetType() != CONSTANT) && (shape_const->GetType() != CONSTANTOP)) {
    return SUCCESS;
  }
  if ((shape_const->GetOutDataNodesSize() == 0U) && (shape_const->GetInControlNodes().size() == 0U)) {
    return IsolateAndDeleteNode(shape_const, {});
  }
  return SUCCESS;
}

REG_PASS_OPTION("ReshapeRemovePass").LEVELS(OoLevel::kO1);
}  // namespace ge
