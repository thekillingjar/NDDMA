/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph/passes/feature/data_flow_prepare_pass.h"
#include <atomic>
#include "graph/utils/node_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/type_utils.h"
#include "graph/utils/op_type_utils.h"
#include "common/checker.h"

namespace ge {
namespace {
const char_t *const kMaxSize = "max_size";
const std::unordered_set<std::string> kDataFlowSources = {STACK};
const std::unordered_set<std::string> kDataFlowOperations = {STACKPUSH, STACKPOP, STACKCLOSE};
inline bool IsDataFlowSource(const std::string &op_type) {
  return kDataFlowSources.count(op_type) !=  0UL;
}
inline bool IsDataFlowOperations(const std::string &op_type) {
  return kDataFlowOperations.count(op_type) != 0UL;
}
inline bool IsDataFlowOps(const std::string &op_type) {
  return (IsDataFlowSource(op_type)) || (IsDataFlowOperations(op_type));
}
}  // namespace
Status DataFlowPreparePass::Run(ge::ComputeGraphPtr graph) {
  GE_CHECK_NOTNULL(graph);
  if (graph->GetParentGraph() != nullptr) {
    GELOGD("Subgraph %s is not need to process", graph->GetName().c_str());
    return SUCCESS;
  }
  std::map<std::string, std::unordered_set<NodePtr>> data_flow_ops_groups;
  for (const NodePtr &node : graph->GetAllNodes()) {
    if (!IsDataFlowOps(node->GetType())) {
      continue;
    }
    if (IsDataFlowSource(node->GetType())) {
      SetMaxSize(node);
      (void)data_flow_ops_groups[node->GetName()].insert(node);
    } else {
      NodePtr data_flow_src;
      const auto ret = GetResourceInputNode(node, 0, data_flow_src);
      if (ret != SUCCESS) {
        GELOGE(INTERNAL_ERROR, "[Get][Resource]node:%s, type:%s.",
               node->GetName().c_str(), node->GetType().c_str());
        return INTERNAL_ERROR;
      }
      GE_ASSERT_NOTNULL(data_flow_src);
      (void)data_flow_ops_groups[data_flow_src->GetName()].insert(node);
    }
  }
  const Status ret = SetHandle(data_flow_ops_groups);
  if (ret != SUCCESS) {
    GELOGE(INTERNAL_ERROR, "[Set][Handle]graph:%s.", graph->GetName().c_str());
    return INTERNAL_ERROR;
  }
  return SUCCESS;
}

void DataFlowPreparePass::SetMaxSize(const NodePtr &node) {
  const auto size_idx = static_cast<uint32_t>(node->GetOpDesc()->GetInputIndexByName(kMaxSize));
  const GeTensor *weight = OpDescUtils::GetInputConstData(OpDescUtils::CreateOperatorFromNode(node), size_idx);
  if (weight != nullptr) {
    const auto &tensor_desc = weight->GetTensorDesc();
    if (tensor_desc.GetDataType() == DT_INT32) {
      const int32_t *weight_data = reinterpret_cast<const int32_t *>(weight->GetData().data());
      if (weight_data != nullptr) {
        const int64_t max_size = static_cast<int64_t>(*weight_data);
        (void)AttrUtils::SetInt(node->GetOpDesc(), ATTR_NAME_DATA_FLOW_MAX_SIZE, max_size);
        GELOGD("Data flow node:%s, type:%s, get max size %ld.",
               node->GetName().c_str(), node->GetType().c_str(), max_size);
      } else {
        GELOGW("Data flow max_size weight data is nullptr, node:%s, type:%s",
               node->GetName().c_str(), node->GetType().c_str());
      }
    } else {
      GELOGW("Data flow max_size type is not int32, node:%s, type:%s, data_type:%s",
             node->GetName().c_str(), node->GetType().c_str(),
             TypeUtils::DataTypeToSerialString(tensor_desc.GetDataType()).c_str());
    }
  } else {
    GELOGI("Data flow max_size is not const, node:%s, type:%s", node->GetName().c_str(), node->GetType().c_str());
  }
}

std::pair<NodePtr, int32_t> DataFlowPreparePass::GetPeerOutNodeAndIndexByInputIndex(const NodePtr &node,
                                                                                    const int32_t input_index) {
  if (node == nullptr) {
    return {nullptr, -1};
  }
  if (node->GetInDataAnchor(input_index) == nullptr) {
    return {nullptr, -1};
  }
  auto peer_out_anchor = node->GetInDataAnchor(input_index)->GetPeerOutAnchor();
  if (peer_out_anchor == nullptr) {
    return {nullptr, -1};
  }
  NodePtr input_node = peer_out_anchor->GetOwnerNode();
  while (input_node != nullptr) {
    if (!OpTypeUtils::IsDataNode(input_node->GetType())) {
      return {input_node, peer_out_anchor->GetIdx()};
    }
    const auto parent_node_idx = NodeUtils::GetParentInputAndAnchor(input_node);
    if (parent_node_idx.first == nullptr || parent_node_idx.second == nullptr) {
      return {input_node, peer_out_anchor->GetIdx()};
    }
    input_node = parent_node_idx.first;
    peer_out_anchor = parent_node_idx.second;
  }
  return {input_node, peer_out_anchor->GetIdx()};
}

Status DataFlowPreparePass::GetResourceInputNode(const NodePtr &node, const int32_t in_idx, NodePtr &src_node) {
  // Resource handles are either consumed or passed.
  // Before the framework supported resource classes, here is a list of known transients to make the scene concrete.
  // A map of <type <out_index, in_index>>, indicate relationship of resource pass through in and out
  static const std::unordered_map<std::string, std::unordered_map<int32_t, int32_t>> kPassThroughResourceIoMap = {
      {REFIDENTITY, {{0, 0}}},
      {ENTER, {{0, 0}}},
      {SWITCH, {{0, 0}, {1, 0}}}
  };

  NodePtr in_node = node;
  int32_t input_index = in_idx;
  bool is_get_resource_input_node = false;
  do {
    const auto in_node_idx = GetPeerOutNodeAndIndexByInputIndex(in_node, input_index);
    in_node = in_node_idx.first;
    const int32_t peer_out_index = in_node_idx.second;
    if (in_node == nullptr || peer_out_index == -1) {
      GELOGE(INTERNAL_ERROR, "[Get][ResourceInput]Invalid input, node:%s, input index:%d",
             node->GetType().c_str(), input_index);
      return INTERNAL_ERROR;
    }
    const auto iter = kPassThroughResourceIoMap.find(in_node->GetType());
    if (iter != kPassThroughResourceIoMap.end()) {
      const auto idx_iter = iter->second.find(peer_out_index);
      if (idx_iter != iter->second.end()) {
        input_index = idx_iter->second;
      } else {
        GELOGE(INTERNAL_ERROR, "[Get][ResourceInput]type:%s, output index:%d",
               in_node->GetType().c_str(), peer_out_index);
        return INTERNAL_ERROR;
      }
    } else {
      src_node = in_node;
      is_get_resource_input_node = true;
    }
  } while (!is_get_resource_input_node);
  return SUCCESS;
}

Status DataFlowPreparePass::SetHandle(const std::map<std::string, std::unordered_set<NodePtr>> &data_flow_ops_groups) {
  static std::atomic<int64_t> atomic_handle(0);
  for (const auto &item : data_flow_ops_groups) {
    const int64_t handle = atomic_handle++;
    for (const auto &data_flow_op : item.second) {
      GE_CHECK_NOTNULL(data_flow_op);
      (void)AttrUtils::SetInt(data_flow_op->GetOpDesc(), ATTR_NAME_DATA_FLOW_HANDLE, handle);
      GELOGD("Data flow op handle %ld, node:%s, type:%s", handle,
             data_flow_op->GetName().c_str(), data_flow_op->GetType().c_str());
    }
  }
  return SUCCESS;
}

REG_PASS_OPTION("DataFlowPreparePass").LEVELS(OoLevel::kO0);
}  // namespace ge
