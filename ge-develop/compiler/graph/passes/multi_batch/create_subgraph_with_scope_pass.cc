/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph/passes/multi_batch/create_subgraph_with_scope_pass.h"

#include "mmpa/mmpa_api.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/tensor_utils.h"
#include "graph/ge_context.h"
#include "graph/preprocess/multi_batch_options.h"
#include "graph/utils/op_type_utils.h"
#include "exec_runtime/execution_runtime_utils.h"
#include "common/plugin/ge_make_unique_util.h"
#include "common/math/math_util.h"
#include "common/context/local_context.h"
#include "common/checker.h"
#include "register/op_registry.h"
#include "api/aclgrph/option_utils.h"
#include "formats/utils/formats_trans_utils.h"

namespace ge {
namespace {
constexpr const char *kOriginType = "original_type";
const char *const kSubstrOfGetNextNosinkName = "IteratorGetNext";
constexpr int32_t kDecimal = 10;

/// @brief Check each dimension is the same
/// @param [in] param
/// @return Status
Status CheckSameDims(const vector<vector<int64_t>> &param) {
  if (param.size() == 0) {
    GELOGE(FAILED, "2D array is empty");
    return FAILED;
  }
  const size_t first_dim = param[0].size();
  for (const auto &item : param) {
    if (item.size() != first_dim) {
      GELOGE(FAILED, "Check 2D array dimensions failed");
      return FAILED;
    }
  }
  return SUCCESS;
}

Status UpdateTensorShape(GeTensorDescPtr &ge_tensor, const std::vector<int64_t> &dims) {
  GE_CHECK_NOTNULL(ge_tensor);
  const auto &shape = ge_tensor->GetShape();
  GELOGD("Update tensor shape is:[%s]", shape.ToString().c_str());
  if (shape.IsUnknownShape()) {
    return SUCCESS;
  }
  size_t origin_dim = shape.GetDimNum();
  if ((!shape.ToString().empty()) && (origin_dim != dims.size())) {
    GELOGE(FAILED, "Original shape dim[%zu] and updated dim[%zu] not match", origin_dim, dims.size());
    return FAILED;
  }
  ge_tensor->SetShape(GeShape(dims));
  ge_tensor->SetOriginShape(GeShape(dims));
  return SUCCESS;
}

Status SetTensorShapeRange(const NodePtr &node) {
  const auto &op_desc = node->GetOpDesc();
  GE_CHECK_NOTNULL(op_desc);
  std::string shape_str;
  if (!AttrUtils::GetStr(op_desc, ATTR_NAME_OP_MAX_SHAPE, shape_str)) {
    return SUCCESS;
  }
  GELOGD("Node [%s] ATTR_NAME_OP_MAX_SHAPE is [%s]", node->GetName().c_str(), shape_str.c_str());
  std::vector<std::string> shape_vec = StringUtils::Split(shape_str, ';');
  for (size_t i = 0U; i < shape_vec.size(); i++) {
    std::vector<std::string> tmp_str_vec = StringUtils::Split(shape_vec[i], ',');
    std::vector<std::pair<int64_t, int64_t>> shape_range;
    for (size_t j = 0U; j < tmp_str_vec.size(); ++j) {
      int64_t dim = std::strtol(tmp_str_vec[j].c_str(), nullptr, kDecimal);
      shape_range.push_back(std::make_pair(dim, dim));
    }
    GeTensorDesc output_tensor = op_desc->GetOutputDesc(i);
    output_tensor.SetShapeRange(shape_range);
    output_tensor.SetOriginShapeRange(shape_range);
    op_desc->UpdateOutputDesc(i, output_tensor);
  }
  for (size_t i = 0UL; i < op_desc->GetAllOutputsDescSize(); i++) {
    auto output_desc = op_desc->MutableOutputDesc(i);
    GE_CHECK_NOTNULL(output_desc);
    ge::TensorUtils::SetSize(*output_desc, 0);
    GELOGI("Node[%s] output[%zu] tensor setsize 0", node->GetName().c_str(), i);
  }
  return SUCCESS;
}

Status SetOpMaxShape(OpDescPtr &op_desc, size_t id, std::vector<std::vector<int64_t>> &dims_range) {
  if (dims_range.empty()) {
    GELOGE(FAILED, "Set op max shape failed, dims_range is empty");
    return FAILED;
  }
  GE_CHK_STATUS_RET(CheckSameDims(dims_range), "Check 2D array dims_range failed");
  std::vector<int64_t> max_dim_range;
  int64_t max_mem_size = 0;
  for (const auto &dim_range : dims_range) {
    int64_t mem_size = 1;
    for (const int64_t dim : dim_range) {
      GE_CHK_BOOL_RET_STATUS(dim > 0, FAILED, "dim [%ld] info error.", dim);
      GE_CHK_STATUS_RET(CheckIntMulOverflow(mem_size, dim), "Multiply result is out of range, dim is [%ld].", dim);
      mem_size = mem_size * dim;
    }
    if (mem_size > max_mem_size) {
      max_mem_size = mem_size;
      max_dim_range = dim_range;
    }
  }

  std::string max_shape_str;
  for (size_t i = 0U; i < max_dim_range.size(); ++i) {
    max_shape_str.append(std::to_string(max_dim_range[i]));
    if (i != max_dim_range.size() - 1U) {
      max_shape_str.append(",");
    }
  }
  GELOGI("Op max shape is [%s].", max_shape_str.c_str());
  std::string op_max_shape;
  std::vector<std::string> max_shape;
  if (AttrUtils::GetStr(op_desc, ATTR_NAME_OP_MAX_SHAPE, op_max_shape)) {
    max_shape = StringUtils::Split(op_max_shape, ';');
    GE_RETURN_WITH_LOG_IF_TRUE(id >= max_shape.size(), "Op name[%s] id [%zu] value is greater than max_shape[%s] size",
                               op_desc->GetName().c_str(), id, op_max_shape.c_str());
    max_shape[id] = max_shape_str;
  } else {
    for (size_t i = 0U; i < op_desc->GetOutputsSize(); ++i) {
      if (i == id) {
        max_shape.push_back(max_shape_str);
      } else {
        GE_ASSERT_NOTNULL(op_desc->GetOutputDescPtr(i));
        max_shape.push_back(formats::ShapeToString(op_desc->GetOutputDescPtr(i)->GetShape()));
      }
    }
  }
  std::string shape_str;
  for (size_t i = 0U; i < max_shape.size(); ++i) {
    shape_str.append(max_shape[i]).append(";");
  }
  shape_str = shape_str.substr(0, shape_str.size() - 1);
  AttrUtils::SetStr(op_desc, ATTR_NAME_OP_MAX_SHAPE, shape_str);
  return SUCCESS;
}
}  // namespace

Status CreateSubGraphWithScopePass::Run(ComputeGraphPtr graph) {
  GELOGD("CreateSubGraphWithScopePass start.");

  GE_CHECK_NOTNULL(graph);
  if (graph->GetParentGraph() != nullptr) {
    return SUCCESS;
  }

  GE_CHK_STATUS_RET(ProcessHeterogeneousMultiBatch(graph), "Create graph scope failed.");
  // add nodes to scope list by index
  GE_CHK_STATUS_RET(CollectScopeNodesByIndex(graph), "Collect scope nodes failed.");

  for (const auto &it : scopes_) {
    const int32_t scope_index = it.first;
    const std::vector<NodePtr> &scope_nodes = it.second;
    ComputeGraphPtr subgraph;

    GE_CHK_STATUS_RET(ParseMultiDimsAttr(scope_nodes),
                      "Parse dynamic dims attribute failed for scope[%d].", scope_index);
    GE_CHK_STATUS_RET(GetSubgraphFromScope(graph, it, subgraph),
                      "Get subgraph From scope[%d] failed.", scope_index);
    GE_CHK_STATUS_RET(CollectIoNodes(subgraph), "Collect Io nodes from subgraph[%s] failed.",
                      subgraph->GetName().c_str());
    GE_CHK_STATUS_RET(CreateNewPartitionedCall(graph, subgraph, scope_nodes),
                      "Create new partitioned call failed.");
    GE_CHK_STATUS_RET(MergeNodesToSubgraph(scope_nodes, subgraph), "Merge scope nodes to subgraph[%s] failed.",
                      subgraph->GetName().c_str());
    GE_CHK_STATUS_RET(SetParentIndexToData(subgraph), "Set parent index to data nodes for subgraph[%s] failed.",
                      subgraph->GetName().c_str());
    GE_CHK_STATUS_RET(SetParentIndexToNetOutput(subgraph), "Set parent index to net_output for subgraph[%s] failed.",
                      subgraph->GetName().c_str());
    GE_CHK_STATUS_RET(AddSubgraphToNewPartitionedCall(graph, subgraph),
                      "Add subgraph to new partitioned_call failed.");
    GE_CHK_STATUS_RET(UpdateTensorMaxShape(graph, subgraph), "Update tensor max shape failed");
    (void)AttrUtils::SetBool(subgraph, ATTR_NAME_SUBGRAPH_IS_MULTI_DIMS, true);
    (void)AttrUtils::SetBool(subgraph, ATTR_NAME_NO_NEED_DYNAMIC_SHAPE_PARTITION, true);
    GELOGI("Create multi dims original subgraph[%s] with scope[%d] success.",
           subgraph->GetName().c_str(), scope_index);
  }
  GE_CHK_STATUS_RET(ParseMultiBatchParams(), "Parse multi-batch parameters failed.");
  GELOGD("CreateSubGraphWithScopePass end.");
  return SUCCESS;
}

bool CreateSubGraphWithScopePass::IsGraphMultiBatchCondition(const ComputeGraphPtr &graph) {
  std::string input_shape_optimize;
  (void)GetContext().GetOption(INPUT_SHAPE, input_shape_optimize);
  std::string dynamic_dims_optimize;
  (void)GetContext().GetOption(kDynamicDims, dynamic_dims_optimize);
  std::string dynamic_node_type;
  (void)GetContext().GetOption(DYNAMIC_NODE_TYPE, dynamic_node_type);
  is_set_dynamic_config_ =
      (!input_shape_optimize.empty()) && (!dynamic_dims_optimize.empty()) && (!dynamic_node_type.empty());
  const bool is_heterogeneous = ExecutionRuntimeUtils::IsHeterogeneous();
  GELOGI(
      "Graph[%s] input_shape_optimize[%s], dynamic_dims_optimize [%s], dynamic_node_type[%s], "
      "is_set_dynamic_config[%d], is_heterogeneous[%d]",
      graph->GetName().c_str(), input_shape_optimize.c_str(), dynamic_dims_optimize.c_str(), dynamic_node_type.c_str(),
      is_set_dynamic_config_, static_cast<int32_t>(is_heterogeneous));
  if ((!is_set_dynamic_config_) || (!is_heterogeneous)) {
    GELOGI("No need to process graph multi batch.");
    return false;
  }

  constexpr size_t kTotleNodes = 2U;
  if (graph->GetDirectNode().size() == kTotleNodes) {
    for (const auto &node : graph->GetDirectNode()) {
      if (node->GetType() == NETOUTPUT) {
        GELOGD("There are 2 nodes in graph [%s], one of the node[%s] is output", graph->GetName().c_str(),
               node->GetName().c_str());
        return false;
      }
    }
  }

  for (const auto &node : graph->GetDirectNode()) {
    const auto &op_desc = node->GetOpDesc();
    int32_t node_index = -1;
    if (AttrUtils::GetInt(op_desc, ATTR_NAME_SUBGRAPH_MULTI_DIMS_INDEX, node_index) && node_index >= 0) {
      GELOGW("The multi-batch configuration takes effect and subgraph multi-batch is invalid.");
      break;
    }
  }

  return true;
}

Status CreateSubGraphWithScopePass::ProcessHeterogeneousMultiBatch(const ComputeGraphPtr &graph) {
  if (!IsGraphMultiBatchCondition(graph)) {
    return SUCCESS;
  }
  std::vector<NodePtr> dynamic_shape_nodes;
  GE_CHK_STATUS_RET(CollectDynamicNodes(graph, dynamic_shape_nodes), "Collect dynamic nodes failed.");
  GE_CHK_STATUS_RET(UpdateDynamicConfigAttrs(dynamic_shape_nodes), "Update dynamic cConfig attrs failed.");
  GE_CHK_STATUS_RET(CreateMultiBatchScope(graph), "Create multi batch scope failed.");
  return SUCCESS;
}

Status CreateSubGraphWithScopePass::CollectDynamicNodes(const ComputeGraphPtr &graph,
                                                        std::vector<NodePtr> &dynamic_shape_nodes) {
  std::string dynamic_node_type;
  (void)GetContext().GetOption(DYNAMIC_NODE_TYPE, dynamic_node_type);
  GE_CHK_BOOL_RET_STATUS((dynamic_node_type == "0" || dynamic_node_type == "1"), FAILED,
                         "Node type value should be 0 or 1.");
  std::function<bool(const NodePtr &)> is_get_next_type = [](const NodePtr &node) {
    const auto &op_desc = node->GetOpDesc();
    std::string original_type;
    (void)AttrUtils::GetStr(op_desc, kOriginType, original_type);
    return (ITERATORV2 == original_type);
  };
  for (const auto &node : graph->GetDirectNode()) {
    GE_CHECK_NOTNULL(node);
    if (dynamic_node_type == "0" && is_get_next_type(node)) {
      dynamic_shape_nodes.emplace_back(node);
      GELOGI("push in dynamic shape nodes, node: %s, type: IteratorV2.", node->GetName().c_str());
    }
    const auto &op_desc = node->GetOpDesc();
    GE_CHECK_NOTNULL(op_desc);
    if (OpTypeUtils::IsDataNode(op_desc->GetType())) {
      if (op_desc->GetName().find(kSubstrOfGetNextNosinkName) != std::string::npos) {
        if (dynamic_node_type == "0") {
          dynamic_shape_nodes.emplace_back(node);
          GELOGI("push in dynamic shape nodes, node: %s, type: %s.", node->GetName().c_str(), node->GetType().c_str());
        }
      } else {
        if (dynamic_node_type == "1") {
          dynamic_shape_nodes.emplace_back(node);
          GELOGI("push in dynamic shape nodes, node: %s, type: %s.", node->GetName().c_str(), node->GetType().c_str());
        }
      }
    }
  }

  (void)std::sort(dynamic_shape_nodes.begin(), dynamic_shape_nodes.end(),
                  [](const NodePtr &node1, const NodePtr &node2) -> bool {
                    return (node1->GetName() < node2->GetName());
                  });
  return SUCCESS;
}

Status CreateSubGraphWithScopePass::UpdateDynamicConfigAttrs(const std::vector<NodePtr> &dynamic_shape_nodes) {
  if (dynamic_shape_nodes.empty()) {
    GELOGD("Dynamic_shape_nodes is empty, skip parse dynamic config.");
    return SUCCESS;
  }
  std::vector<std::pair<std::string, std::vector<int64_t>>>
      user_shape_map;  // e.g. "data0:-1,3;data1:-1,4"-> [{"data0", [-1,3]}, {"data1", [-1,4]}]
  std::vector<std::pair<std::string, std::vector<int64_t>>>
      max_shape_map;  // e.g. "3,3;4,4" -> [["3","3"],["4","4"]] , digitvec [[3,3], [4,4]]
  std::string input_shape_optimize;
  (void)GetContext().GetOption(INPUT_SHAPE, input_shape_optimize);
  std::string dynamic_dims_optimize;
  (void)GetContext().GetOption(kDynamicDims, dynamic_dims_optimize);
  std::vector<std::vector<std::string>> dynamic_dims_vec;
  GE_CHK_STATUS_RET(multibatch::ParseDynamicShapesAndDims(input_shape_optimize, dynamic_dims_optimize, user_shape_map,
                                                          dynamic_dims_vec, max_shape_map));
  size_t input_count = 0UL;
  for (const auto &node : dynamic_shape_nodes) {
    const OpDescPtr node_desc = node->GetOpDesc();
    GE_CHECK_NOTNULL(node_desc);
    input_count += static_cast<size_t>(node_desc->GetOutputsSize());
  }
  std::string dynamic_node_type;
  (void)GetContext().GetOption(DYNAMIC_NODE_TYPE, dynamic_node_type);
  const std::string node_type_str = (dynamic_node_type == "0" ? "dataset" : "placeholder");
  GE_CHK_BOOL_RET_STATUS(user_shape_map.size() == input_count, FAILED,
                         "user_shape_map size %zu and input count %zu not match, node type is %s.",
                         user_shape_map.size(), input_count, node_type_str.c_str());

  std::vector<std::string> subgraph_multi_dims_input_shape;
  std::vector<std::string> subgraph_multi_dims_input_dims;
  GE_RETURN_IF_ERROR(multibatch::BuildSubgraphMuliDimsInput(
      user_shape_map, dynamic_dims_vec, subgraph_multi_dims_input_shape, subgraph_multi_dims_input_dims));
  size_t input_index_begin = 0UL;
  for (size_t i = 0UL; i < dynamic_shape_nodes.size(); ++i) {
    const NodePtr src_node = dynamic_shape_nodes[i];
    for (const auto &dst_node_and_in_anchor : src_node->GetOutDataNodesAndAnchors()) {
      const InDataAnchorPtr dst_in_anchor = dst_node_and_in_anchor.second;
      GE_CHECK_NOTNULL(dst_in_anchor);
      int idx = dst_in_anchor->GetIdx();
      const OutDataAnchorPtr peer_out_anchor = dst_in_anchor->GetPeerOutAnchor();
      GE_CHECK_NOTNULL(peer_out_anchor);
      const size_t input_index = input_index_begin + static_cast<size_t>(peer_out_anchor->GetIdx());

      NodePtr dst_node = dst_node_and_in_anchor.first;
      OpDescPtr dst_node_desc = dst_node->GetOpDesc();
      GE_CHECK_NOTNULL(dst_node_desc);
      std::string pre_subgraph_input_shape;
      std::string pre_subgraph_input_dims;
      (void)AttrUtils::GetStr(dst_node_desc, ATTR_NAME_SUBGRAPH_MULTI_DIMS_INPUT_SHAPE, pre_subgraph_input_shape);
      (void)AttrUtils::GetStr(dst_node_desc, ATTR_NAME_SUBGRAPH_MULTI_DIMS_INPUT_DIMS, pre_subgraph_input_dims);
      GE_CHK_BOOL_RET_STATUS((pre_subgraph_input_shape.empty()) == (pre_subgraph_input_dims.empty()), FAILED,
                             "pre_subgraph_input_shape[%s] and pre_subgraph_input_dims[%s] not match",
                             pre_subgraph_input_shape.c_str(), pre_subgraph_input_dims.c_str());
      GE_CHK_BOOL_RET_STATUS((input_index < subgraph_multi_dims_input_shape.size()) &&
                                 (input_index < subgraph_multi_dims_input_dims.size()),
                             FAILED, "input_index[%zu] is greater than input_shape[%zu] or input_dims[%zu]",
                             input_index, subgraph_multi_dims_input_shape.size(),
                             subgraph_multi_dims_input_dims.size());
      if ((subgraph_multi_dims_input_shape[input_index].empty()) ||
          (subgraph_multi_dims_input_dims[input_index].empty())) {
        continue;
      }
      std::string subgraph_input_shape = std::to_string(idx) + ":" + subgraph_multi_dims_input_shape[input_index];
      if (pre_subgraph_input_shape.empty() && pre_subgraph_input_dims.empty()) {
        (void)AttrUtils::SetStr(dst_node_desc, ATTR_NAME_SUBGRAPH_MULTI_DIMS_INPUT_SHAPE, subgraph_input_shape);
        (void)AttrUtils::SetStr(dst_node_desc, ATTR_NAME_SUBGRAPH_MULTI_DIMS_INPUT_DIMS,
                                subgraph_multi_dims_input_dims[input_index]);
      } else {
        GE_RETURN_IF_ERROR(UpdateSubgraphMultiDimsAttr(dst_node, pre_subgraph_input_shape, pre_subgraph_input_dims,
                                                       subgraph_input_shape,
                                                       subgraph_multi_dims_input_dims[input_index]));
      }
    }
    const OpDescPtr src_node_desc = src_node->GetOpDesc();
    GE_CHECK_NOTNULL(src_node_desc);
    input_index_begin += src_node_desc->GetOutputsSize();
  }
  return SUCCESS;
}

Status CreateSubGraphWithScopePass::UpdateSubgraphMultiDimsAttr(NodePtr node, const std::string &pre_input_shape,
                                                                const std::string &pre_input_dims,
                                                                const std::string &new_input_shape,
                                                                const std::string &new_input_dims) const {
  std::vector<std::string> pre_input_dims_vec = ge::StringUtils::Split(pre_input_dims, ';');
  std::vector<std::string> new_input_dims_vec = ge::StringUtils::Split(new_input_dims, ';');
  GE_CHK_BOOL_RET_STATUS(pre_input_dims_vec.size() == new_input_dims_vec.size(), FAILED,
                         "pre_input_dims size[%zu] and new_input_dims size[%zu] not match", pre_input_dims_vec.size(),
                         new_input_dims_vec.size());
  std::string update_input_dims;
  for (size_t i = 0U; i < new_input_dims_vec.size(); ++i) {
    update_input_dims.append(pre_input_dims_vec[i]).append(",").append(new_input_dims_vec[i]).append(";");
  }
  update_input_dims = update_input_dims.substr(0, update_input_dims.size() - 1);
  std::string update_input_shape = pre_input_shape + ";" + new_input_shape;
  GE_CHECK_NOTNULL(node);
  OpDescPtr node_desc = node->GetOpDesc();
  (void)AttrUtils::SetStr(node_desc, ATTR_NAME_SUBGRAPH_MULTI_DIMS_INPUT_SHAPE, update_input_shape);
  (void)AttrUtils::SetStr(node_desc, ATTR_NAME_SUBGRAPH_MULTI_DIMS_INPUT_DIMS, update_input_dims);
  GELOGI("update node %s ATTR_NAME_SUBGRAPH_MULTI_DIMS_INPUT_SHAPE [%s]", node->GetName().c_str(),
         update_input_shape.c_str());
  GELOGI("update node %s ATTR_NAME_SUBGRAPH_MULTI_DIMS_INPUT_DIMS [%s]", node->GetName().c_str(),
         update_input_dims.c_str());
  return SUCCESS;
}

Status CreateSubGraphWithScopePass::CreateMultiBatchScope(const ComputeGraphPtr &graph) {
  std::vector<NodePtr> scope_node;
  for (const auto &node : graph->GetDirectNode()) {
    const auto &op_desc = node->GetOpDesc();
    GE_CHECK_NOTNULL(op_desc);
    std::string origin_type;
    (void)AttrUtils::GetStr(op_desc, kOriginType, origin_type);
    GELOGD("Node type [%s], node name [%s], origin_type [%s]", op_desc->GetType().c_str(), op_desc->GetName().c_str(),
           origin_type.c_str());
    if (OpTypeUtils::IsDataNode(node->GetType()) || (node->GetType() == NETOUTPUT || (origin_type == ITERATORV2))) {
      continue;
    }
    scope_node.emplace_back(node);
  }

  size_t num_of_node_with_dim_attr = 0U;
  for (const auto &node : scope_node) {
    const auto &op_desc = node->GetOpDesc();
    GE_CHECK_NOTNULL(op_desc);
    std::string input_shape;
    std::string multi_dims;
    // example: input shape: "0:-1,224; 1:-1,-1"  multi dims: "112,1,1; 224,2,2; 448,4,4"
    if (!AttrUtils::GetStr(op_desc, ATTR_NAME_SUBGRAPH_MULTI_DIMS_INPUT_SHAPE, input_shape) ||
        !AttrUtils::GetStr(op_desc, ATTR_NAME_SUBGRAPH_MULTI_DIMS_INPUT_DIMS, multi_dims)) {
      continue;
    }
    num_of_node_with_dim_attr++;
  }
  if (num_of_node_with_dim_attr == 0U) {
    GELOGW("Nodes in scope do not have specified attr.");
    return SUCCESS;
  }
  for (const auto &node : scope_node) {
    const auto &op_desc = node->GetOpDesc();
    GE_CHECK_NOTNULL(op_desc);
    (void)AttrUtils::SetInt(op_desc, ATTR_NAME_SUBGRAPH_MULTI_DIMS_INDEX, 0);
    GELOGD("Op[%s][%s] set subgraph multi dims index 0", op_desc->GetName().c_str(), op_desc->GetType().c_str());
  }
  return SUCCESS;
}

Status CreateSubGraphWithScopePass::CollectScopeNodesByIndex(const ComputeGraphPtr &graph) {
  for (const auto &node : graph->GetDirectNode()) {
    const auto &op_desc = node->GetOpDesc();
    GE_CHECK_NOTNULL(op_desc);
    int32_t node_index = -1;
    if (AttrUtils::GetInt(op_desc, ATTR_NAME_SUBGRAPH_MULTI_DIMS_INDEX, node_index) && node_index >= 0) {
      if (OpTypeUtils::IsDataNode(node->GetType()) || (node->GetType() == NETOUTPUT)) {
        REPORT_INNER_ERR_MSG("E19999", "Not support DATA/NETOUTPUT node[%s] in scope.", node->GetName().c_str());
        GELOGE(PARAM_INVALID, "Not support DATA/NETOUTPUT node[%s] in scope.", node->GetName().c_str());
        return PARAM_INVALID;
      }
      scopes_[node_index].push_back(node);
      GELOGD("Get node[%s] scope index[%d].", node->GetName().c_str(), node_index);
    }
  }
  return SUCCESS;
}

Status CreateSubGraphWithScopePass::ParseMultiDimsAttr(const std::vector<NodePtr> &scope_nodes) {
  for (const auto &node : scope_nodes) {
    const auto &op_desc = node->GetOpDesc();
    GE_CHECK_NOTNULL(op_desc);
    std::string input_shape;
    std::string multi_dims;
    // example: input shape: "0:1,1,-1,224; 1:-1,-1,224,224"
    //          multi dims: "112,1,1; 224,2,2; 448,4,4"
    if (!AttrUtils::GetStr(op_desc, ATTR_NAME_SUBGRAPH_MULTI_DIMS_INPUT_SHAPE, input_shape) ||
        !AttrUtils::GetStr(op_desc, ATTR_NAME_SUBGRAPH_MULTI_DIMS_INPUT_DIMS, multi_dims)) {
      continue;
    }
    GELOGI("Input node[%s] has dynamic dims attr, input shape[%s], input multi dims[%s].", node->GetName().c_str(),
           input_shape.c_str(), multi_dims.c_str());
    GE_CHK_STATUS_RET(ParseSubGraphMultiAttrs(node, input_shape, multi_dims), "Parse subgraph mulit attrs failed");
    GE_CHK_STATUS_RET(RefreshTensorShape(node), "Refresh tensor shape failed");
  }
  return SUCCESS;
}
Status CreateSubGraphWithScopePass::ParseSubGraphMultiAttrs(const NodePtr &node, const std::string &input_shape,
                                                            const std::string &multi_dims) {
  const auto &op_desc = node->GetOpDesc();
  GE_CHECK_NOTNULL(op_desc);
  std::map<int64_t, std::vector<int64_t>> shape_vec;
  for (const auto &it : ge::StringUtils::Split(input_shape, ';')) {
    std::vector<std::string> tensor_shape = ge::StringUtils::Split(it, ':');
    const int32_t kDefaultShapePairSize = 2;
    GE_RT_PARAM_INVALID_WITH_LOG_IF_TRUE(tensor_shape.size() != kDefaultShapePairSize,
                                         "parse tensor_shape failed, shape is [%s]", it.c_str());
    int64_t index = std::strtol(tensor_shape[0U].c_str(), nullptr, kDecimal);  // decimal
    GE_RT_PARAM_INVALID_WITH_LOG_IF_TRUE(index >= static_cast<int64_t>(op_desc->GetAllInputsSize()),
                                         "Invalid input shape for node[%s], input size[%zu], index[%ld]",
                                         op_desc->GetName().c_str(), op_desc->GetAllInputsSize(), index);
    std::vector<int64_t> shape;
    for (const auto &dim : ge::StringUtils::Split(tensor_shape[1], ',')) {
      shape.push_back(std::strtol(dim.c_str(), nullptr, kDecimal));
    }
    shape_vec[index] = std::move(shape);
  }
  node_to_input_shape_[node] = shape_vec;  // e.g. {{0:3,-1}, {1:2,-1}}
  std::vector<std::vector<int64_t>> parsed_dims;
  for (const auto &it : ge::StringUtils::Split(multi_dims, ';')) {
    std::vector<int64_t> dims;
    for (const auto &grade : ge::StringUtils::Split(it, ',')) {
      dims.push_back(std::strtol(grade.c_str(), nullptr, kDecimal));
    }
    parsed_dims.push_back(dims);  // e.g. [[3,3],[4,4],[5,5,]]
  }
  GE_CHK_STATUS_RET(CheckSameDims(parsed_dims), "Check 2D array parsed_dims failed");
  std::map<int64_t, std::vector<std::vector<int64_t>>> dims_map;
  for (const auto &parsed_dim : parsed_dims) {
    size_t negative = 0U;
    for (const auto &shape : shape_vec) {
      std::vector<int64_t> dims;
      for (const auto &dim : shape.second) {
        if (dim == ge::UNKNOWN_DIM) {
          dims.push_back(parsed_dim[negative]);
          negative++;
        } else {
          dims.push_back(dim);
        }
      }
      if (dims_map.find(shape.first) == dims_map.end()) {
        dims_map[shape.first] = {dims};
      } else {
        dims_map.at(shape.first).push_back(dims);
      }
    }
  }
  // e.g. node_to_multi_dims_[node0] = {0:[[3,3],[4,4],[5,5]], 1:[[2,3],[2,4],[2,5]]}
  node_to_multi_dims_[node] = std::move(dims_map);
  return SUCCESS;
}

Status CreateSubGraphWithScopePass::RefreshTensorShape(const NodePtr &node) {
  if (!is_set_dynamic_config_) {
    return SUCCESS;
  }
  const auto &op_desc = node->GetOpDesc();
  GE_CHECK_NOTNULL(op_desc);
  std::map<int64_t, std::vector<int64_t>> &shape_vec = node_to_input_shape_[node];
  std::map<int64_t, std::vector<std::vector<int64_t>>> &dims_map = node_to_multi_dims_[node];
  const std::function<Status(const NodePtr &, const int64_t, OutDataAnchorPtr &, NodePtr &)> get_peer_node =
      [](const NodePtr &n, const int64_t index, OutDataAnchorPtr &peer_anchor, NodePtr &peer_node) -> Status {
    auto input_anchor = n->GetInDataAnchor(index);
    GE_CHECK_NOTNULL(input_anchor);
    peer_anchor = input_anchor->GetPeerOutAnchor();
    GE_CHECK_NOTNULL(peer_anchor);
    peer_node = peer_anchor->GetOwnerNode();
    GE_CHECK_NOTNULL(peer_node);
    return SUCCESS;
  };
  for (auto it = shape_vec.cbegin(); it != shape_vec.cend(); ++it) {
    int64_t index = it->first;
    // update input tensor shape
    auto input_tensor = op_desc->MutableInputDesc(index);
    GE_CHECK_NOTNULL(input_tensor);
    GE_CHK_STATUS_RET_NOLOG(UpdateTensorShape(input_tensor, shape_vec[index]));
    OutDataAnchorPtr peer_anchor;
    NodePtr peer_node;
    const Status ret = get_peer_node(node, index, peer_anchor, peer_node);
    if (ret != SUCCESS) {
      continue;
    }

    const auto &peer_desc = peer_node->GetOpDesc();
    GE_CHECK_NOTNULL(peer_desc);
    auto peer_tensor = peer_desc->MutableOutputDesc(peer_anchor->GetIdx());
    GE_CHECK_NOTNULL(peer_tensor);
    GELOGD("Node[%s] input[%ld], node[%s] output[%d]", op_desc->GetName().c_str(), index, peer_desc->GetName().c_str(),
           peer_anchor->GetIdx());
    GE_CHK_STATUS_RET_NOLOG(UpdateTensorShape(peer_tensor, shape_vec[index]));
    if (OpTypeUtils::IsDataNode(peer_desc->GetType()) || (peer_desc->GetType() == ITERATORV2) ||
        (peer_desc->GetType() == "QueueData")) {
      auto peer_input_tensor = peer_desc->MutableInputDesc(0);
      if (peer_input_tensor != nullptr) {
        GE_CHK_STATUS_RET_NOLOG(UpdateTensorShape(peer_input_tensor, shape_vec[index]));
      }
    }
  }
  for (auto it = dims_map.begin(); it != dims_map.end(); ++it) {
    int64_t index = it->first;
    std::vector<std::vector<int64_t>> &dims_range = it->second;
    OutDataAnchorPtr peer_anchor;
    NodePtr peer_node;
    const Status ret = get_peer_node(node, index, peer_anchor, peer_node);
    if (ret != SUCCESS) {
      continue;
    }
    int32_t out_id = peer_anchor->GetIdx();

    auto peer_desc = peer_node->GetOpDesc();
    GE_CHECK_NOTNULL(peer_desc);
    std::string origin_type;
    (void)AttrUtils::GetStr(peer_desc, kOriginType, origin_type);
    if (OpTypeUtils::IsDataNode(peer_desc->GetType()) ||
        ((peer_desc->GetType() == FRAMEWORKOP) && (origin_type == ITERATORV2))) {
      GE_CHK_STATUS_RET_NOLOG(SetOpMaxShape(peer_desc, static_cast<size_t>(out_id), dims_range));
      GELOGD("Node[%s] type[%s] update tensor max shape success", peer_node->GetName().c_str(),
             peer_desc->GetType().c_str());
    }
  }
  return SUCCESS;
}

Status CreateSubGraphWithScopePass::AddSubgraphToNewPartitionedCall(const ComputeGraphPtr &root_graph,
                                                                    const ComputeGraphPtr &subgraph) {
  const auto &parent_node = subgraph->GetParentNode();
  GE_CHECK_NOTNULL(parent_node);
  const auto &parent_desc = parent_node->GetOpDesc();
  GE_CHECK_NOTNULL(parent_desc);
  subgraph->SetParentNode(new_partitioned_call_);
  const auto &op_desc = new_partitioned_call_->GetOpDesc();
  GE_CHECK_NOTNULL(op_desc);
  op_desc->CopyAttrsFrom(*parent_desc);

  GE_CHK_STATUS_RET(root_graph->RemoveNode(parent_node), "Remove node[%s] failed.", parent_node->GetName().c_str());
  GE_CHK_STATUS_RET(op_desc->AddSubgraphName(subgraph->GetName()), "Add subgraph[%s] to partitioned_call[%s] failed.",
                    subgraph->GetName().c_str(), op_desc->GetName().c_str());
  GE_CHK_STATUS_RET(op_desc->SetSubgraphInstanceName(0U, subgraph->GetName()),
                    "Add subgraph[%s] to partitioned_call[%s] failed.",
                    subgraph->GetName().c_str(), op_desc->GetName().c_str());
  return SUCCESS;
}

Status CreateSubGraphWithScopePass::CollectIoNodes(const ComputeGraphPtr &subgraph) {
  const auto &parent_node = subgraph->GetParentNode();
  GE_CHECK_NOTNULL(parent_node);

  anchor_to_data_nodes_.clear();
  for (const auto &node : subgraph->GetDirectNode()) {
    if (OpTypeUtils::IsDataNode(node->GetType())) {
      int32_t parent_index = -1;
      if (!AttrUtils::GetInt(node->GetOpDesc(), ATTR_NAME_PARENT_NODE_INDEX, parent_index) || parent_index < 0) {
        REPORT_INNER_ERR_MSG("E19999", "Data node[%s] has no parent index.", node->GetName().c_str());
        GELOGE(PARAM_INVALID, "Data node[%s] has no parent index.", node->GetName().c_str());
        return PARAM_INVALID;
      }

      const auto &anchor = parent_node->GetInDataAnchor(parent_index);
      GE_CHECK_NOTNULL(anchor);
      anchor_to_data_nodes_[anchor] = node;
      GELOGD("Add anchor[%d] and node[%s] to anchor_to_data_nodes_", parent_index, node->GetName().c_str());
    }
  }
  anchor_to_output_.clear();
  const auto &output = subgraph->FindFirstNodeMatchType(NETOUTPUT);
  if (output == nullptr) {
    return SUCCESS;
  }
  const auto &op_desc = output->GetOpDesc();
  GE_CHECK_NOTNULL(op_desc);
  for (const auto &input_anchor : output->GetAllInDataAnchors()) {
    int32_t parent_index = -1;
    const auto &tensor = op_desc->GetInputDescPtr(input_anchor->GetIdx());
    if (!AttrUtils::GetInt(tensor, ATTR_NAME_PARENT_NODE_INDEX, parent_index) || parent_index < 0) {
      REPORT_INNER_ERR_MSG("E19999", "NetOutput[%s] tensor[%d] has no parent index.",
                         output->GetName().c_str(), input_anchor->GetIdx());
      GELOGE(PARAM_INVALID, "NetOutput[%s] tensor[%d] has no parent index.",
             output->GetName().c_str(), input_anchor->GetIdx());
      return PARAM_INVALID;
    }

    const auto &anchor = parent_node->GetOutDataAnchor(parent_index);
    GE_CHECK_NOTNULL(anchor);
    anchor_to_output_[anchor] = input_anchor;
    GELOGD("Add anchor[%d] and netoutput anchor[%d] to anchor_to_output_", parent_index, input_anchor->GetIdx());
  }

  return SUCCESS;
}

Status CreateSubGraphWithScopePass::GetSubgraphFromScope(const ComputeGraphPtr &root_graph,
                                                         const std::pair<const int32_t, std::vector<NodePtr>> &scope,
                                                         ComputeGraphPtr &subgraph) const {
  const int32_t scope_index = scope.first;
  for (const auto &node : scope.second) {
    if (node->GetType() == PARTITIONEDCALL) {
      const auto &op_desc = node->GetOpDesc();
      GE_CHECK_NOTNULL(op_desc);
      subgraph = root_graph->GetSubgraph(op_desc->GetSubgraphInstanceName(0U));
      GE_CHECK_NOTNULL(subgraph);
      GELOGD("Get subgraph:%s from scope[%d].", subgraph->GetName().c_str(), scope_index);
      return SUCCESS;
    }
  }

  // no partitioned_call in scope, create new subgraph
  subgraph = MakeShared<ComputeGraph>(root_graph->GetName() + "_scope_subgraph_" + std::to_string(scope_index));
  GE_CHECK_NOTNULL(subgraph);
  const auto op_desc = MakeShared<OpDesc>("scope_partitioned_call_" + std::to_string(scope_index), PARTITIONEDCALL);
  GE_CHECK_NOTNULL(op_desc);
  const auto partition_op_vec = root_graph->FuseNodeKeepTopo(scope.second, {op_desc});
  GE_ASSERT_TRUE(!partition_op_vec.empty());
  const NodePtr partition_op = partition_op_vec.front();
  (void)op_desc->AddSubgraphName(subgraph->GetName());
  (void)op_desc->SetSubgraphInstanceName(0U, subgraph->GetName());
  subgraph->SetParentNode(partition_op);
  subgraph->SetParentGraph(root_graph);
  (void)root_graph->AddSubGraph(subgraph);
  std::string session_graph_id;
  (void)AttrUtils::GetStr(root_graph, ATTR_NAME_SESSION_GRAPH_ID, session_graph_id);
  (void)AttrUtils::SetStr(subgraph, ATTR_NAME_SESSION_GRAPH_ID, session_graph_id);
  GELOGD("Scope[%d] has no subgraph, create new[%s].", scope_index, subgraph->GetName().c_str());

  return SUCCESS;
}

Status CreateSubGraphWithScopePass::MergeNodesToSubgraph(const std::vector<NodePtr> &scope_nodes,
                                                         const ComputeGraphPtr &subgraph) {
  const auto &partitioned_call = subgraph->GetParentNode();
  GE_CHECK_NOTNULL(partitioned_call);
  const auto &op_desc = partitioned_call->GetOpDesc();
  GE_CHECK_NOTNULL(op_desc);

  if ((scope_nodes.size() == 1U) && (scope_nodes[0U] == partitioned_call)) {
    GELOGI("Scope has only one node[%s] type[%s], copy attrs to data nodes.",
           partitioned_call->GetName().c_str(), partitioned_call->GetType().c_str());
    return CopyPartitionedCallAttrToData(subgraph);
  }

  for (const auto &node : scope_nodes) {
    // skip partitioned_call it self
    if (node == partitioned_call) {
      continue;
    }
    // if node has control anchor outside scope, return failed
    GE_CHK_STATUS_RET(CheckCtrlAnchorInvalid(node, scope_nodes),
                      "[CHECK]Not support control edge cross scope, error node[%s].", node->GetName().c_str());
    const auto &parent_graph = node->GetOwnerComputeGraph();
    GE_CHECK_NOTNULL(parent_graph);
    if (GraphUtils::RemoveJustNode(parent_graph, node) != GRAPH_SUCCESS) {
      REPORT_INNER_ERR_MSG("E19999", "Remove node[%s] from root graph[%s] failed.",
                         node->GetName().c_str(), parent_graph->GetName().c_str());
      GELOGE(FAILED, "Remove node[%s] from root graph[%s] failed.",
             node->GetName().c_str(), parent_graph->GetName().c_str());
      return FAILED;
    }
    (void)subgraph->AddNode(node);
    if (node->SetOwnerComputeGraph(subgraph) != GRAPH_SUCCESS) {
      REPORT_INNER_ERR_MSG("E19999", "Set owner graph[%s] to node[%s] failed.",
                         subgraph->GetName().c_str(), node->GetName().c_str());
      GELOGE(FAILED, "Set owner graph[%s] to node[%s] failed.", subgraph->GetName().c_str(), node->GetName().c_str());
      return FAILED;
    }
    GE_CHK_STATUS_RET(MergeInputAnchors(subgraph, node, scope_nodes),
                      "Merge node[%s]'s input anchors to subgraph[%s] failed.",
                      node->GetName().c_str(), subgraph->GetName().c_str());
    GE_CHK_STATUS_RET(MergeOutputAnchors(subgraph, node, scope_nodes),
                      "Merge node[%s]'s output anchors to subgraph[%s] failed.",
                      node->GetName().c_str(), subgraph->GetName().c_str());
    GELOGI("Node[%s] has been moved to subgraph[%s].", node->GetName().c_str(), subgraph->GetName().c_str());
  }
  return SUCCESS;
}

Status CreateSubGraphWithScopePass::CopyPartitionedCallAttrToData(const ComputeGraphPtr &subgraph) {
  const auto &partitioned_call = subgraph->GetParentNode();
  GE_CHECK_NOTNULL(partitioned_call);

  const std::map<NodePtr, std::map<int64_t, std::vector<int64_t>>>::const_iterator &input_shape =
    node_to_input_shape_.find(partitioned_call);
  if (input_shape == node_to_input_shape_.cend()) {
    REPORT_INNER_ERR_MSG("E19999", "No input shape for the only node[%s] in scope.",
                       partitioned_call->GetName().c_str());
    GELOGE(PARAM_INVALID, "No input shape for the only node[%s] in scope.", partitioned_call->GetName().c_str());
    return PARAM_INVALID;
  }

  const std::map<NodePtr, std::map<int64_t, std::vector<std::vector<int64_t>>>>::const_iterator &dyn_dims =
    node_to_multi_dims_.find(partitioned_call);
  if (dyn_dims == node_to_multi_dims_.cend()) {
    REPORT_INNER_ERR_MSG("E19999", "No dynamic dims for the only node[%s] in scope.",
                       partitioned_call->GetName().c_str());
    GELOGE(PARAM_INVALID, "No dynamic dims for the only node[%s] in scope.", partitioned_call->GetName().c_str());
    return PARAM_INVALID;
  }

  for (const auto &node : subgraph->GetDirectNode()) {
    if (node->GetType() != DATA) {
      GELOGI("Node type is [%s].", node->GetType().c_str());
      continue;
    }
    const auto &op_desc = node->GetOpDesc();
    GE_CHECK_NOTNULL(op_desc);
    int32_t parent_index = -1;
    if (!AttrUtils::GetInt(op_desc, ATTR_NAME_PARENT_NODE_INDEX, parent_index) || parent_index < 0) {
      REPORT_INNER_ERR_MSG("E19999", "Invalid parent index for data node[%s].", node->GetName().c_str());
      GELOGE(PARAM_INVALID, "Invalid parent index for data node[%s].", node->GetName().c_str());
      return PARAM_INVALID;
    }
    const std::map<int64_t, std::vector<int64_t>>::const_iterator &shape = input_shape->second.find(parent_index);
    if (shape != input_shape->second.cend()) {
      auto tensor = op_desc->MutableInputDesc(0U);
      GE_CHECK_NOTNULL(tensor);
      tensor->SetShape(GeShape(shape->second));
      tensor->SetOriginShape(GeShape(shape->second));
    }
    const auto &dims = dyn_dims->second.find(parent_index);
    if (dims != dyn_dims->second.end()) {
      std::vector<int64_t> merged_dims;
      for (const auto &it : dims->second) {
        for (const auto &dim : it) {
          merged_dims.push_back(dim);
        }
      }
      (void)AttrUtils::SetListInt(op_desc, ATTR_NAME_OP_MULTI_DIMS_INPUT_DIMS, merged_dims);
    }
  }
  return SUCCESS;
}

Status CreateSubGraphWithScopePass::CheckCtrlAnchorInvalid(const NodePtr &node,
                                                           const std::vector<NodePtr> &scope_nodes) const {
  GE_CHECK_NOTNULL(node);
  const auto &in_ctrl_anchor = node->GetInControlAnchor();
  if (in_ctrl_anchor != nullptr) {
    for (const auto &peer_anchor : in_ctrl_anchor->GetPeerOutControlAnchors()) {
      if (peer_anchor == nullptr) {
        continue;
      }
      const auto &peer_node = peer_anchor->GetOwnerNode();
      if (peer_node == nullptr) {
        continue;
      }
      const auto &it = std::find(scope_nodes.begin(), scope_nodes.end(), peer_node);
      if (it == scope_nodes.end()) {
        REPORT_INNER_ERR_MSG("E19999", "Exit control edge bwtween [%s] and [%s].",
                           peer_node->GetName().c_str(), node->GetName().c_str());
        GELOGE(PARAM_INVALID, "Exit control edge bwtween [%s] and [%s].",
               peer_node->GetName().c_str(), node->GetName().c_str());
        return FAILED;
      }
    }
  }
  const auto &out_ctrl_anchor = node->GetOutControlAnchor();
  if (out_ctrl_anchor != nullptr) {
    for (const auto &peer_anchor : out_ctrl_anchor->GetPeerInControlAnchors()) {
      if (peer_anchor == nullptr) {
        continue;
      }
      const auto &peer_node = peer_anchor->GetOwnerNode();
      if (peer_node == nullptr) {
        continue;
      }
      const auto &it = std::find(scope_nodes.begin(), scope_nodes.end(), peer_node);
      if (it == scope_nodes.end() && peer_node->GetType() == NETOUTPUT && is_set_dynamic_config_) {
        GELOGI("Exist control edge between [%s] and net output [%s].", node->GetName().c_str(),
               peer_node->GetName().c_str());
        GE_CHK_STATUS_RET(GraphUtils::RemoveEdge(out_ctrl_anchor, peer_anchor),
                          "Remove ctrl edge failed between [%s] and net output [%s].", node->GetName().c_str(),
                          peer_node->GetName().c_str());
        continue;
      }

      if (it == scope_nodes.end()) {
        REPORT_INNER_ERR_MSG("E19999", "Exist control edge bwtween [%s] and [%s].",
                           node->GetName().c_str(), peer_node->GetName().c_str());
        GELOGE(PARAM_INVALID, "Exist control edge bwtween [%s] and [%s].",
               node->GetName().c_str(), peer_node->GetName().c_str());
        return FAILED;
      }
    }
  }
  return SUCCESS;
}

Status CreateSubGraphWithScopePass::MergeInputAnchors(const ComputeGraphPtr &subgraph, const NodePtr &node,
                                                      const std::vector<NodePtr> &scope_nodes) {
  for (const auto &input_anchor : node->GetAllInDataAnchors()) {
    if (input_anchor == nullptr) {
      continue;
    }
    const auto peer_anchor = input_anchor->GetPeerOutAnchor();
    if (peer_anchor == nullptr) {
      continue;
    }
    const auto &peer_node = peer_anchor->GetOwnerNode();
    GE_CHECK_NOTNULL(peer_node);

    if (peer_node == subgraph->GetParentNode()) {
      // proc node linked with partitioned call
      return ProcPartitionOutputAnchor(peer_anchor, input_anchor);
    }

    const auto &it = std::find(scope_nodes.begin(), scope_nodes.end(), peer_node);
    if (it != scope_nodes.end()) {
      GELOGD("Peer node[%s] is also in scope, no need merge anchors.", peer_node->GetName().c_str());
      continue;
    }

    const size_t parent_index = new_partitioned_call_->GetAllInDataAnchorsSize();
    const auto &peer_desc = peer_node->GetOpDesc();
    GE_CHECK_NOTNULL(peer_desc);
    NodePtr data_node = nullptr;
    OpDescPtr data_desc = nullptr;
    bool scope_data_node_exist = false;
    const auto iter = data_node_to_scope_data_node_.find(peer_node);
    if (iter != data_node_to_scope_data_node_.end()) {
      data_node = iter->second;
      scope_data_node_exist = true;
      data_desc = data_node->GetOpDesc();
      GE_CHECK_NOTNULL(data_desc);
      GELOGI("Scope data node [%s] exist.", data_node->GetName().c_str());
    } else {
      data_desc = MakeShared<OpDesc>(
          subgraph->GetName() + peer_node->GetName() + "_data_" + std::to_string(parent_index), DATA);
      GE_CHECK_NOTNULL(data_desc);
      GE_CHK_STATUS_RET(data_desc->AddInputDesc(peer_desc->GetOutputDesc(peer_anchor->GetIdx())),
                        "Add input desc fail");
      GE_CHK_STATUS_RET(data_desc->AddOutputDesc(peer_desc->GetOutputDesc(peer_anchor->GetIdx())),
                        "Add output desc fail");
      data_node = subgraph->AddNode(data_desc);
      data_node_to_scope_data_node_[peer_node] = data_node;
    }

    GE_CHECK_NOTNULL(data_node);
    GE_CHK_STATUS_RET(SetDataDynDimsInfo(node, data_node, input_anchor->GetIdx()),
                      "Set dyn dims info from node[%s] input[%d] to data[%s] failed.",
                      node->GetName().c_str(), input_anchor->GetIdx(), data_node->GetName().c_str());
    std::string max_op_shape;
    if (AttrUtils::GetStr(peer_desc, ATTR_NAME_OP_MAX_SHAPE, max_op_shape)) {
      std::vector<std::string> max_shape = StringUtils::Split(max_op_shape, ';');
      const int32_t peer_anchor_idx = peer_anchor->GetIdx();
      GE_RETURN_WITH_LOG_IF_TRUE(peer_anchor_idx < 0, "Value of peer_anchor_idx[%d] is less than 0.", peer_anchor_idx);
      GE_RETURN_WITH_LOG_IF_TRUE(max_shape.size() <= static_cast<size_t>(peer_anchor_idx),
                                 "Value of max_op_shape[%s] invalid, peer ancher index[%d]", max_op_shape.c_str(),
                                 peer_anchor_idx);
      (void)AttrUtils::SetStr(data_desc, ATTR_NAME_OP_MAX_SHAPE, max_shape[peer_anchor_idx]);
      GELOGI("Node[%s] max_op_shape:[%s], max_shape[%d]:[%s]", node->GetName().c_str(), max_op_shape.c_str(),
             peer_anchor_idx, max_shape[peer_anchor_idx].c_str());
    }
    GELOGI("Peer node[%s] is out of scope, add new data node[%s] into subgraph[%s].",
           peer_node->GetName().c_str(), data_node->GetName().c_str(), subgraph->GetName().c_str());

    if (GraphUtils::RemoveEdge(peer_anchor, input_anchor) != SUCCESS) {
      REPORT_INNER_ERR_MSG("E19999", "Remove edge from node[%s] output[%d] to node[%s] input[%d] failed.",
                         peer_node->GetName().c_str(), peer_anchor->GetIdx(),
                         node->GetName().c_str(), input_anchor->GetIdx());
      GELOGE(FAILED, "Remove edge from node[%s] output [%d] to node[%s] input[%d] failed.",
             peer_node->GetName().c_str(), peer_anchor->GetIdx(), node->GetName().c_str(), input_anchor->GetIdx());
      return FAILED;
    }
    if (GraphUtils::AddEdge(data_node->GetOutDataAnchor(0), input_anchor) != SUCCESS) {
      REPORT_INNER_ERR_MSG("E19999", "Add edge from node[%s] output[0] to node[%s] input[%d] failed.",
                         data_node->GetName().c_str(), node->GetName().c_str(), input_anchor->GetIdx());
      GELOGE(FAILED, "Add edge from node[%s] output[0] to node[%s] input[%d] failed.", data_node->GetName().c_str(),
             node->GetName().c_str(), input_anchor->GetIdx());
      return FAILED;
    }
    if (scope_data_node_exist) {
      continue;
    }
    const auto &op_desc = new_partitioned_call_->GetOpDesc();
    GE_CHECK_NOTNULL(op_desc);
    GE_CHK_STATUS_RET(op_desc->AddInputDesc(peer_desc->GetOutputDesc(peer_anchor->GetIdx())),
                      "Add input desc for [%s] failed.", op_desc->GetName().c_str());
    (void)NodeUtils::AppendInputAnchor(new_partitioned_call_, parent_index + 1);
    auto new_anchor = new_partitioned_call_->GetInDataAnchor(parent_index);
    GE_CHECK_NOTNULL(new_anchor);
    if (GraphUtils::AddEdge(peer_anchor, new_anchor) != SUCCESS) {
      REPORT_INNER_ERR_MSG("E19999", "Add edge from node[%s] output[%d] to node[%s] input[%d] failed.",
                         peer_node->GetName().c_str(), peer_anchor->GetIdx(),
                         new_partitioned_call_->GetName().c_str(), new_anchor->GetIdx());
      GELOGE(FAILED, "Add edge from node[%s] output [%d] to node[%s] input[%d] failed.",
             peer_node->GetName().c_str(), peer_anchor->GetIdx(),
             new_partitioned_call_->GetName().c_str(), new_anchor->GetIdx());
      return FAILED;
    }

    anchor_to_data_nodes_[new_anchor] = data_node;
  }
  return SUCCESS;
}

Status CreateSubGraphWithScopePass::SetDataDynDimsInfo(const NodePtr &node, const NodePtr &data_node,
                                                       const int32_t input_index) {
  const std::map<NodePtr, std::map<int64_t, std::vector<int64_t>>>::const_iterator &input_shape =
    node_to_input_shape_.find(node);
  if (input_shape == node_to_input_shape_.cend()) {
    GELOGI("No node exist in node_to_input_shape_");
    return SUCCESS;
  }
  const std::map<int64_t, std::vector<int64_t>>::const_iterator &shape = input_shape->second.find(input_index);
  if (shape == input_shape->second.cend()) {
    return SUCCESS;
  }
  const auto &data_desc = data_node->GetOpDesc();
  GE_CHECK_NOTNULL(data_desc);
  const auto &tensor_desc = data_desc->MutableInputDesc(0U);
  GE_CHECK_NOTNULL(tensor_desc);
  tensor_desc->SetShape(GeShape(shape->second));
  tensor_desc->SetOriginShape(GeShape(shape->second));

  const std::map<NodePtr, std::map<int64_t, std::vector<std::vector<int64_t>>>>::const_iterator &multi_dims =
    node_to_multi_dims_.find(node);
  if (multi_dims == node_to_multi_dims_.end()) {
    REPORT_INNER_ERR_MSG("E19999", "Input node[%s] has input_shape but no multi_dims.", node->GetName().c_str());
    GELOGE(FAILED, "Input node[%s] has input_shape but no multi_dims.", node->GetName().c_str());
    return FAILED;
  }
  const auto &dims = multi_dims->second.find(input_index);
  if (dims == multi_dims->second.end()) {
    REPORT_INNER_ERR_MSG("E19999", "Node[%s] input[%d] has input_shape but no multi_dims.",
                       node->GetName().c_str(), input_index);
    GELOGE(FAILED, "Node[%s] input[%d] has input_shape but no multi_dims.",
           node->GetName().c_str(), input_index);
    return FAILED;
  }
  std::vector<int64_t> merged_dims;
  for (const auto &it : dims->second) {
    for (const auto &dim : it) {
      merged_dims.push_back(dim);
    }
  }
  (void)AttrUtils::SetListInt(data_desc, ATTR_NAME_OP_MULTI_DIMS_INPUT_DIMS, merged_dims);
  GELOGI("Set multi dims to data node[%s] from [%s].", node->GetName().c_str(), data_node->GetName().c_str());
  return SUCCESS;
}

Status CreateSubGraphWithScopePass::MergeOutputAnchors(const ComputeGraphPtr &subgraph, const NodePtr &node,
                                                       const std::vector<NodePtr> &scope_nodes) {
  auto node_output = subgraph->FindFirstNodeMatchType(NETOUTPUT);
  if (node_output == nullptr) {
    const OpDescPtr new_desc = MakeShared<OpDesc>(subgraph->GetName() + "_node_output", NETOUTPUT);
    GE_CHECK_NOTNULL(new_desc);
    node_output = subgraph->AddNode(new_desc);
    GE_CHECK_NOTNULL(node_output);
  }
  const auto &node_output_desc = node_output->GetOpDesc();
  GE_CHECK_NOTNULL(node_output_desc);
  const auto &partition_desc = new_partitioned_call_->GetOpDesc();
  GE_CHECK_NOTNULL(partition_desc);
  const auto &node_desc = node->GetOpDesc();
  GE_CHECK_NOTNULL(node_desc);

  for (auto output_anchor : node->GetAllOutDataAnchors()) {
    for (auto peer_anchor : output_anchor->GetPeerInDataAnchors()) {
      if (peer_anchor == nullptr) {
        continue;
      }
      const auto &peer_node = peer_anchor->GetOwnerNode();
      GE_CHECK_NOTNULL(peer_node);

      if (peer_node == subgraph->GetParentNode()) {
      // proc node linked with partitioned call
        return ProcPartitionInputAnchor(subgraph, output_anchor, peer_anchor);
      }

      const auto &it = std::find(scope_nodes.begin(), scope_nodes.end(), peer_node);
      if (it != scope_nodes.end()) {
        GELOGD("Peer node[%s] is also in scope, no need merge.", peer_node->GetName().c_str());
        continue;
      }

      GELOGI("Peer node[%s] is out of scope, add output edge into subgraph[%s].",
             peer_node->GetName().c_str(), subgraph->GetName().c_str());
      if (GraphUtils::RemoveEdge(output_anchor, peer_anchor) != SUCCESS) {
        REPORT_INNER_ERR_MSG("E19999", "Remove edge from node[%s] output[%d] to node[%s] input[%d] failed.",
                           node->GetName().c_str(), output_anchor->GetIdx(),
                           peer_node->GetName().c_str(), peer_anchor->GetIdx());
        GELOGE(FAILED, "Remove edge from node[%s] output[%d] to node[%s] input[%d] failed.",
               node->GetName().c_str(), output_anchor->GetIdx(), peer_node->GetName().c_str(), peer_anchor->GetIdx());
        return FAILED;
      }

      const size_t parent_index = new_partitioned_call_->GetAllOutDataAnchorsSize();
      const auto &peer_desc = peer_node->GetOpDesc();
      GE_CHECK_NOTNULL(peer_desc);
      GeTensorDesc tensor_desc(peer_desc->GetInputDesc(peer_anchor->GetIdx()));
      (void)AttrUtils::SetInt(tensor_desc, ATTR_NAME_PARENT_NODE_INDEX, parent_index);
      GE_CHK_STATUS_RET(node_output_desc->AddInputDesc(tensor_desc), "Add input desc fail");
      const auto input_size = node_output->GetAllInDataAnchorsSize();
      (void)NodeUtils::AppendInputAnchor(node_output, input_size + 1);
      if (GraphUtils::AddEdge(output_anchor, node_output->GetInDataAnchor(input_size)) != SUCCESS) {
        REPORT_INNER_ERR_MSG("E19999", "Add edge from node[%s] output[%d] to node[%s] input[%u] failed.",
                           node->GetName().c_str(), output_anchor->GetIdx(),
                           node_output->GetName().c_str(), input_size);
        GELOGE(FAILED, "Add edge from node[%s] output[%d] to node[%s] input[%u] failed.",
               node->GetName().c_str(), output_anchor->GetIdx(), node_output->GetName().c_str(), input_size);
        return FAILED;
      }

      (void)partition_desc->AddOutputDesc(node_desc->GetOutputDesc(output_anchor->GetIdx()));
      (void)NodeUtils::AppendOutputAnchor(new_partitioned_call_, parent_index + 1);
      auto new_anchor = new_partitioned_call_->GetOutDataAnchor(parent_index);
      GE_CHECK_NOTNULL(new_anchor);

      if (GraphUtils::AddEdge(new_anchor, peer_anchor) != SUCCESS) {
        REPORT_INNER_ERR_MSG("E19999", "Add edge from node[%s] output[%zu] to node[%s] input[%d] failed.",
                           new_partitioned_call_->GetName().c_str(), parent_index,
                           peer_node->GetName().c_str(), peer_anchor->GetIdx());
        GELOGE(FAILED, "Add edge from node[%s] output[%zu] to node[%s] input[%d] failed.",
               new_partitioned_call_->GetName().c_str(), parent_index,
               peer_node->GetName().c_str(), peer_anchor->GetIdx());
        return FAILED;
      }
      anchor_to_output_[new_anchor] = node_output->GetInDataAnchor(input_size);
    }
  }
  return SUCCESS;
}

Status CreateSubGraphWithScopePass::ProcPartitionInputAnchor(const ComputeGraphPtr &subgraph,
                                                             const OutDataAnchorPtr &node_anchor,
                                                             const InDataAnchorPtr &partition_anchor) {
  const std::map<InDataAnchorPtr, NodePtr>::const_iterator &it = anchor_to_data_nodes_.find(partition_anchor);
  if (it == anchor_to_data_nodes_.cend()) {
    REPORT_INNER_ERR_MSG("E19999", "Can't find data node by input anchor[%d].", partition_anchor->GetIdx());
    GELOGE(PARAM_INVALID, "Can't find data node by input anchor[%d].", partition_anchor->GetIdx());
    return PARAM_INVALID;
  }
  const auto &data_node = it->second;
  GE_CHECK_NOTNULL(data_node);
  const auto &output_anchor = data_node->GetOutDataAnchor(0);
  if (output_anchor != nullptr) {
    for (auto peer_anchor : output_anchor->GetPeerInDataAnchors()) {
      if (peer_anchor == nullptr) {
        continue;
      }
      const auto &peer_node = peer_anchor->GetOwnerNode();
      GE_CHECK_NOTNULL(peer_node);
      if (GraphUtils::RemoveEdge(output_anchor, peer_anchor) != SUCCESS) {
        REPORT_INNER_ERR_MSG("E19999", "Remove edge from node[%s] output[%d] to node[%s] input[%d] failed.",
                           data_node->GetName().c_str(), output_anchor->GetIdx(),
                           peer_node->GetName().c_str(), peer_anchor->GetIdx());
        GELOGE(FAILED, "Remove edge from node[%s] output[%d] to node[%s] input[%d] failed.",
               data_node->GetName().c_str(), output_anchor->GetIdx(),
               peer_node->GetName().c_str(), peer_anchor->GetIdx());
        return FAILED;
      }
      if (GraphUtils::AddEdge(node_anchor, peer_anchor) != SUCCESS) {
        REPORT_INNER_ERR_MSG("E19999", "Add edge from node[%s] output[%d] to partitioned_call input[%d] failed.",
                           node_anchor->GetOwnerNode()->GetName().c_str(), node_anchor->GetIdx(),
                           partition_anchor->GetIdx());
        GELOGE(FAILED, "Add edge from node[%s] output[%d] to partitioned_call input[%d] failed.",
               node_anchor->GetOwnerNode()->GetName().c_str(), node_anchor->GetIdx(), partition_anchor->GetIdx());
        return FAILED;
      }
    }
  }
  if (GraphUtils::RemoveEdge(node_anchor, partition_anchor) != SUCCESS) {
    REPORT_INNER_ERR_MSG("E19999", "Remove edge from node[%s] output[%d] to partitioned_call input[%d].",
                       node_anchor->GetOwnerNode()->GetName().c_str(), node_anchor->GetIdx(),
                       partition_anchor->GetIdx());
    GELOGE(FAILED, "Remove edge from node[%s] output[%d] to partitioned_call input[%d].",
           node_anchor->GetOwnerNode()->GetName().c_str(), node_anchor->GetIdx(), partition_anchor->GetIdx());
    return FAILED;
  }
  GE_CHK_STATUS_RET(subgraph->RemoveNode(data_node), "Remove node[%s] from subgraph[%s] failed.",
                    data_node->GetName().c_str(), subgraph->GetName().c_str());
  anchor_to_data_nodes_.erase(it);
  return SUCCESS;
}

Status CreateSubGraphWithScopePass::ProcPartitionOutputAnchor(const OutDataAnchorPtr &partition_anchor,
                                                              const InDataAnchorPtr &node_anchor) {
  const std::map<OutDataAnchorPtr, InDataAnchorPtr>::const_iterator &it = anchor_to_output_.find(partition_anchor);
  if (it == anchor_to_output_.cend()) {
    REPORT_INNER_ERR_MSG("E19999", "Can't find subgraph anchor by netoutput anchor[%d].", partition_anchor->GetIdx());
    GELOGE(PARAM_INVALID, "Can't find subgraph anchor by netoutput anchor[%d].", partition_anchor->GetIdx());
    return PARAM_INVALID;
  }
  const auto &output_anchor = it->second;
  GE_CHECK_NOTNULL(output_anchor);
  const auto &peer_anchor = output_anchor->GetPeerOutAnchor();
  GE_CHECK_NOTNULL(peer_anchor);
  const auto &peer_node = peer_anchor->GetOwnerNode();
  GE_CHECK_NOTNULL(peer_node);

  // unlink partitioned_call and current node
  if (GraphUtils::RemoveEdge(partition_anchor, node_anchor) != SUCCESS) {
    REPORT_INNER_ERR_MSG("E19999", "Remove edge from partitioned_call output[%d] to node[%s] input[%d] failed.",
                       partition_anchor->GetIdx(),
                       node_anchor->GetOwnerNode()->GetName().c_str(), node_anchor->GetIdx());
    GELOGE(FAILED, "Remove edge from partitioned_call output[%d] to node[%s] input[%d] failed.",
           partition_anchor->GetIdx(), node_anchor->GetOwnerNode()->GetName().c_str(), node_anchor->GetIdx());
    return FAILED;
  }
  if (GraphUtils::AddEdge(peer_anchor, node_anchor) != SUCCESS) {
    REPORT_INNER_ERR_MSG("E19999", "Add edge from node[%s] output[%d] to node[%s] input[%d] failed.",
                       peer_node->GetName().c_str(), peer_anchor->GetIdx(),
                       node_anchor->GetOwnerNode()->GetName().c_str(), node_anchor->GetIdx());
    GELOGE(FAILED, "Add edge from node[%s] output[%d] to node[%s] input[%d] failed.",
           peer_node->GetName().c_str(), peer_anchor->GetIdx(),
           node_anchor->GetOwnerNode()->GetName().c_str(), node_anchor->GetIdx());
    return FAILED;
  }
  if (partition_anchor->GetPeerAnchorsSize() == 0U) {
    // unlink net_output and it's input node
    if (GraphUtils::RemoveEdge(peer_anchor, output_anchor) != SUCCESS) {
      REPORT_INNER_ERR_MSG("E19999", "Remove edge from node[%s] output[%d] to node[%s] input[%d] failed.",
                         peer_node->GetName().c_str(), peer_anchor->GetIdx(),
                         output_anchor->GetOwnerNode()->GetName().c_str(), output_anchor->GetIdx());
      GELOGE(FAILED, "Remove edge from node[%s] output[%d] to node[%s] input[%d] failed.",
             peer_node->GetName().c_str(), peer_anchor->GetIdx(),
             output_anchor->GetOwnerNode()->GetName().c_str(), output_anchor->GetIdx());
      return FAILED;
    }
    anchor_to_output_.erase(it);
  }
  return SUCCESS;
}

Status CreateSubGraphWithScopePass::CreateNewPartitionedCall(const ComputeGraphPtr &root_graph,
                                                             const ComputeGraphPtr &subgraph,
                                                             const std::vector<NodePtr> &scope_nodes) {
  const auto &partitioned_call = subgraph->GetParentNode();
  GE_CHECK_NOTNULL(partitioned_call);

  const OpDescPtr op_desc = MakeShared<OpDesc>(partitioned_call->GetName() + "_copy", PARTITIONEDCALL);
  GE_CHECK_NOTNULL(op_desc);
  GELOGD("Copy new partitioned call from [%s].", partitioned_call->GetName().c_str());
  new_partitioned_call_ = root_graph->InsertNode(partitioned_call, op_desc);
  GE_CHECK_NOTNULL(new_partitioned_call_);

  GE_CHK_STATUS_RET(CopyPartitionedCallStaticInput(partitioned_call, scope_nodes), "Copy static input failed.");
  GE_CHK_STATUS_RET(CopyPartitionedCallStaticOutput(partitioned_call, scope_nodes), "Copy static output failed.");

  return SUCCESS;
}

Status CreateSubGraphWithScopePass::CopyPartitionedCallStaticInput(const NodePtr &src_node,
                                                                   const std::vector<NodePtr> &scope_nodes) {
  const auto &src_desc = src_node->GetOpDesc();
  GE_CHECK_NOTNULL(src_desc);
  const auto &new_desc = new_partitioned_call_->GetOpDesc();
  GE_CHECK_NOTNULL(new_desc);
  size_t copy_input_cnt = 0U;
  for (const auto &input_anchor : src_node->GetAllInDataAnchors()) {
    if (input_anchor == nullptr) {
      continue;
    }
    const auto &peer_anchor = input_anchor->GetPeerOutAnchor();
    if (peer_anchor == nullptr) {
      continue;
    }
    const auto &peer_node = peer_anchor->GetOwnerNode();
    if (peer_node == nullptr) {
      continue;
    }
    const auto &it = std::find(scope_nodes.begin(), scope_nodes.end(), peer_node);
    if (it != scope_nodes.end()) {
      GELOGD("Peer node[%s] is in scope, no need copy anchor.", peer_node->GetName().c_str());
      continue;
    }
    const auto &input_tensor = src_desc->GetInputDesc(input_anchor->GetIdx());
    GE_CHK_STATUS_RET(new_desc->AddInputDesc(input_tensor),
                      "Add input desc for [%s] fail", new_desc->GetName().c_str());
    if (GraphUtils::RemoveEdge(peer_anchor, input_anchor) != GRAPH_SUCCESS) {
      REPORT_INNER_ERR_MSG("E19999", "Remove edge from node[%s] output[%d] to node[%s] input[%d] failed",
                        peer_node->GetName().c_str(), peer_anchor->GetIdx(),
                        src_node->GetName().c_str(), input_anchor->GetIdx());
      GELOGE(FAILED, "Remove edge from node[%s] output[%d] to node[%s] input[%d] failed",
             peer_node->GetName().c_str(), peer_anchor->GetIdx(),
             src_node->GetName().c_str(), input_anchor->GetIdx());
      return FAILED;
    }
    (void)NodeUtils::AppendInputAnchor(new_partitioned_call_, copy_input_cnt + 1);
    const auto &new_anchor = new_partitioned_call_->GetInDataAnchor(copy_input_cnt);
    if (GraphUtils::AddEdge(peer_anchor, new_anchor) != GRAPH_SUCCESS) {
      REPORT_INNER_ERR_MSG("E19999", "Add edge from node[%s] output[%d] to node[%s] input[%zu] failed",
                        peer_node->GetName().c_str(), peer_anchor->GetIdx(),
                        new_partitioned_call_->GetName().c_str(), copy_input_cnt);
      GELOGE(FAILED, "Add edge from node[%s] output[%d] to node[%s] input[%zu] failed",
             peer_node->GetName().c_str(), peer_anchor->GetIdx(),
             new_partitioned_call_->GetName().c_str(), copy_input_cnt);
      return FAILED;
    }
    copy_input_cnt++;
    const auto &iter = anchor_to_data_nodes_.find(input_anchor);
    if (iter != anchor_to_data_nodes_.end()) {
      anchor_to_data_nodes_[new_anchor] = iter->second;
      anchor_to_data_nodes_.erase(iter);
    }
  }
  return SUCCESS;
}

Status CreateSubGraphWithScopePass::CopyPartitionedCallStaticOutput(const NodePtr &src_node,
                                                                    const std::vector<NodePtr> &scope_nodes) {
  const auto &src_desc = src_node->GetOpDesc();
  GE_CHECK_NOTNULL(src_desc);
  const auto &new_desc = new_partitioned_call_->GetOpDesc();
  GE_CHECK_NOTNULL(new_desc);

  size_t copy_output_cnt = 0U;
  for (const auto &output_anchor : src_node->GetAllOutDataAnchors()) {
    std::vector<InDataAnchorPtr> in_scope;
    std::vector<InDataAnchorPtr> out_scope;
    GE_CHECK_NOTNULL(output_anchor);
    for (const auto &peer_anchor : output_anchor->GetPeerInDataAnchors()) {
      if (peer_anchor == nullptr) {
        continue;
      }
      const auto &peer_node = peer_anchor->GetOwnerNode();
      if (peer_node == nullptr) {
        continue;
      }
      const auto &it = std::find(scope_nodes.begin(), scope_nodes.end(), peer_node);
      if (it != scope_nodes.end()) {
        in_scope.push_back(peer_anchor);
        GELOGD("Peer node[%s] is in scope, no need copy anchor.", peer_node->GetName().c_str());
      } else {
        out_scope.push_back(peer_anchor);
      }
    }
    if (!out_scope.empty()) {
      const auto &output_tensor = src_desc->GetOutputDesc(output_anchor->GetIdx());
      GE_CHK_STATUS_RET(new_desc->AddOutputDesc(output_tensor), "Add output desc for [%s] fail",
                        new_desc->GetName().c_str());
      (void)NodeUtils::AppendOutputAnchor(new_partitioned_call_, copy_output_cnt + 1);
      const auto &new_anchor = new_partitioned_call_->GetOutDataAnchor(copy_output_cnt);
      for (auto it : out_scope) {
        if (GraphUtils::RemoveEdge(output_anchor, it) != GRAPH_SUCCESS) {
          REPORT_INNER_ERR_MSG("E19999", "Remove edge from node[%s] output[%d] to node[%s] input[%d] failed",
                            src_node->GetName().c_str(), output_anchor->GetIdx(),
                            it->GetOwnerNode()->GetName().c_str(), it->GetIdx());
          GELOGE(FAILED, "Remove edge from node[%s] output[%d] to node[%s] input[%d] failed",
                 src_node->GetName().c_str(), output_anchor->GetIdx(),
                 it->GetOwnerNode()->GetName().c_str(), it->GetIdx());
          return FAILED;
        }
        if (GraphUtils::AddEdge(new_anchor, it) != GRAPH_SUCCESS) {
          REPORT_INNER_ERR_MSG("E19999", "Add edge from node[%s] output[%zu] to node[%s] input[%d] failed",
                            new_partitioned_call_->GetName().c_str(), copy_output_cnt,
                            it->GetOwnerNode()->GetName().c_str(), it->GetIdx());
          GELOGE(FAILED, "Add edge from node[%s] output[%zu] to node[%s] input[%d] failed",
                 new_partitioned_call_->GetName().c_str(), copy_output_cnt,
                 it->GetOwnerNode()->GetName().c_str(), it->GetIdx());
          return FAILED;
        }
      }
      copy_output_cnt++;
      if (in_scope.empty()) {
        const auto &iter = anchor_to_output_.find(output_anchor);
        if (iter != anchor_to_output_.end()) {
          anchor_to_output_[new_anchor] = iter->second;
          anchor_to_output_.erase(iter);
        }
      }
    }
  }
  return SUCCESS;
}

Status CreateSubGraphWithScopePass::SetParentIndexToData(const ComputeGraphPtr &subgraph) {
  for (const auto &it : anchor_to_data_nodes_) {
    if ((it.first != nullptr) && (it.first->GetPeerOutAnchor() != nullptr)) {
      const int32_t parent_index = it.first->GetIdx();
      const auto &data_desc = it.second->GetOpDesc();
      GE_CHECK_NOTNULL(data_desc);
      (void)AttrUtils::SetInt(data_desc, ATTR_NAME_PARENT_NODE_INDEX, parent_index);
      GELOGD("Set data node[%s] parent index[%d].", data_desc->GetName().c_str(), parent_index);
    } else {
      subgraph->RemoveNode(it.second);
    }
  }
  return SUCCESS;
}

void CreateSubGraphWithScopePass::SortAnchorByIndex(std::set<std::pair<int32_t, InDataAnchorPtr>> &index_anchor) const {
  for (const auto &it : anchor_to_output_) {
    if (it.first == nullptr || it.first->GetPeerAnchorsSize() == 0U ||
        it.second == nullptr || it.second->GetPeerAnchorsSize() == 0U) {
      continue;
    }
    index_anchor.emplace(std::make_pair(it.first->GetIdx(), it.second));
  }
}

Status CreateSubGraphWithScopePass::SetParentIndexToNetOutput(const ComputeGraphPtr &subgraph) const {
  const auto &net_output = subgraph->FindFirstNodeMatchType(NETOUTPUT);
  GE_CHECK_NOTNULL(net_output);
  const auto &old_desc = net_output->GetOpDesc();
  GE_CHECK_NOTNULL(old_desc);

  const OpDescPtr op_desc = MakeShared<OpDesc>(net_output->GetName() + "_copy", NETOUTPUT);
  GE_CHECK_NOTNULL(op_desc);
  GELOGD("Copy new node_output from [%s].", net_output->GetName().c_str());
  const auto new_net_output = subgraph->AddNode(op_desc);
  GE_CHECK_NOTNULL(new_net_output);

  // we should add input in order of parent index
  std::set<std::pair<int32_t, InDataAnchorPtr>> index_anchor;
  SortAnchorByIndex(index_anchor);

  size_t all_input_size = 0U;
  for (const auto &it : index_anchor) {
    GE_CHK_STATUS_RET(op_desc->AddInputDesc(old_desc->GetInputDesc(it.second->GetIdx())),
                      "Add input desc for [%s] failed.", op_desc->GetName().c_str());
    (void)NodeUtils::AppendInputAnchor(new_net_output, all_input_size + 1);
    const auto &new_anchor = new_net_output->GetInDataAnchor(all_input_size);
    GE_CHECK_NOTNULL(new_anchor);

    const auto &peer_anchor = it.second->GetPeerOutAnchor();
    GE_CHECK_NOTNULL(peer_anchor);
    if (GraphUtils::RemoveEdge(peer_anchor, it.second) != SUCCESS) {
      REPORT_INNER_ERR_MSG("E19999", "Remove edge from node[%s] output[%d] to node[%s] input[%d] failed.",
                         peer_anchor->GetOwnerNode()->GetName().c_str(), peer_anchor->GetIdx(),
                         net_output->GetName().c_str(), it.second->GetIdx());
      GELOGE(FAILED, "Remove edge from node[%s] output[%d] to node[%s] input[%d] failed.",
             peer_anchor->GetOwnerNode()->GetName().c_str(), peer_anchor->GetIdx(),
             net_output->GetName().c_str(), it.second->GetIdx());
      return FAILED;
    }
    if (GraphUtils::AddEdge(peer_anchor, new_anchor) != SUCCESS) {
      REPORT_INNER_ERR_MSG("E19999", "Add edge from node[%s] output[%d] to node[%s] input[%d] failed.",
                         peer_anchor->GetOwnerNode()->GetName().c_str(), peer_anchor->GetIdx(),
                         new_net_output->GetName().c_str(), new_anchor->GetIdx());
      GELOGE(FAILED, "Add edge from node[%s] output [%d] to node[%s] input[%d] failed.",
             peer_anchor->GetOwnerNode()->GetName().c_str(), peer_anchor->GetIdx(),
             new_net_output->GetName().c_str(), new_anchor->GetIdx());
      return FAILED;
    }

    const auto &new_tensor = op_desc->MutableInputDesc(all_input_size);
    const int32_t parent_index = it.first;
    (void)AttrUtils::SetInt(new_tensor, ATTR_NAME_PARENT_NODE_INDEX, parent_index);
    GELOGD("Set parent index[%d] to new net_output[%s] input[%zu].",
           parent_index, new_net_output->GetName().c_str(), all_input_size);
    all_input_size++;
  }
  (void)subgraph->RemoveNode(net_output);
  return SUCCESS;
}
Status CreateSubGraphWithScopePass::UpdateTensorMaxShape(const ComputeGraphPtr &root_graph,
                                                         const ComputeGraphPtr &subgraph) const {
  if (!is_set_dynamic_config_) {
    return SUCCESS;
  }
  for (const auto &node : root_graph->GetDirectNode()) {
    GE_CHK_STATUS_RET(SetTensorShapeRange(node), "SetTensorShapeRange node[%s] failed",
                      node->GetName().c_str());
  }
  for (const auto &node : subgraph->GetDirectNode()) {
    GE_CHK_STATUS_RET(SetTensorShapeRange(node), "SetTensorShapeRange node[%s] failed",
                      node->GetName().c_str());
  }
  return SUCCESS;
}

Status CreateSubGraphWithScopePass::ParseMultiBatchParams() const {
  if ((!is_set_dynamic_config_) || (scopes_.empty())) {
    return SUCCESS;
  }
  GetLocalOmgContext().is_subgraph_multi_batch = true;
  return SUCCESS;
}

REG_PASS_OPTION("CreateSubGraphWithScopePass").LEVELS(OoLevel::kO1);
}  // namespace ge
