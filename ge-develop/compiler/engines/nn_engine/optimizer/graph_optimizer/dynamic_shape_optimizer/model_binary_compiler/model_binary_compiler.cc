/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph_optimizer/dynamic_shape_optimizer/model_binary_compiler/model_binary_compiler.h"

#include <nlohmann/json.hpp>
#include <type_traits>

#include "common/aicore_util_constants.h"
#include "common/fe_inner_attr_define.h"
#include "common/fe_utils.h"
#include "common/fe_op_info_common.h"
#include "lowering/bg_ir_attrs.h"
#include "ge/ge_api_types.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/utils/op_desc_utils.h"
#include "register/graph_optimizer/fusion_common/unknown_shape_utils.h"
namespace fe {
namespace {
std::string GetVectorStr(std::vector<uint32_t> v, std::string name) {
  nlohmann::json vector_str;
  vector_str[name] = v;
  return vector_str.dump(4);
}

std::string GetThreeVectorStr(std::vector<std::vector<std::vector<int64_t>>> vvv, std::string name) {
  nlohmann::json vector_str;
  vector_str[name] = vvv;
  return vector_str.dump(4);
}

std::string GetIndexStr(const std::set<uint32_t> &index) {
  std::string index_str = "{";
  for (auto i = index.begin(); i != index.end(); ++i) {
    index_str = index_str + std::to_string(*i) + ",";
  }
  index_str = index_str + "}";
  return index_str;
}

std::string GetVectorDimRangeStr(std::vector<std::vector<std::vector<ffts::DimRange>>> tensor_slice, std::string name) {
  std::vector<std::vector<std::vector<std::vector<int64_t>>>> vvvv;
  for (size_t i = 0; i < tensor_slice.size(); ++i) {
    std::vector<std::vector<std::vector<int64_t>>> vvv;
    for (size_t j = 0; j < tensor_slice[i].size(); ++j) {
      std::vector<std::vector<int64_t>> vv;
      for (size_t k = 0; k < tensor_slice[i][j].size(); ++k) {
        std::vector<int64_t> v;
        v.emplace_back(tensor_slice[i][j][k].higher);
        v.emplace_back(tensor_slice[i][j][k].lower);
        vv.emplace_back(v);
      }
      vvv.emplace_back(vv);
    }
    vvvv.emplace_back(vvv);
  }
  nlohmann::json vector_str;
  vector_str[name] = vvvv;
  return vector_str.dump(4);
}

void PrintTensorSliceInfo(const ffts::ThreadSliceMapPtr &slice_info_ptr) {
  FE_LOGD("************************Tensor slice info begin*****************************************");
  FE_LOGD("%s.", GetThreeVectorStr(slice_info_ptr->ori_input_tensor_shape, "ori_input_tensor_shape").c_str());
  FE_LOGD("%s.", GetThreeVectorStr(slice_info_ptr->ori_output_tensor_shape, "ori_output_tensor_shape").c_str());
  FE_LOGD("%s.", GetVectorStr(slice_info_ptr->input_axis, "input_axis").c_str());
  FE_LOGD("%s.", GetVectorStr(slice_info_ptr->output_axis, "output_axis").c_str());
  FE_LOGD("%s.", GetVectorStr(slice_info_ptr->input_tensor_indexes, "input_tensor_indexes").c_str());
  FE_LOGD("%s.", GetVectorStr(slice_info_ptr->output_tensor_indexes, "output_tensor_indexes").c_str());
  FE_LOGD("%s.", GetVectorDimRangeStr(slice_info_ptr->input_tensor_slice, "input_tensor_slice").c_str());
  FE_LOGD("%s.", GetVectorDimRangeStr(slice_info_ptr->output_tensor_slice, "output_tensor_slice").c_str());
  FE_LOGD("%s.", GetVectorDimRangeStr(slice_info_ptr->ori_input_tensor_slice, "ori_input_tensor_slice").c_str());
  FE_LOGD("%s.", GetVectorDimRangeStr(slice_info_ptr->ori_output_tensor_slice, "ori_output_tensor_slice").c_str());
  FE_LOGD("************************Tensor slice info end*******************************************");
}

void CopyTensorSliceBasicsInfo(const ffts::ThreadSliceMapPtr &slice_info_ptr,
    const ffts::ThreadSliceMapPtr &new_slice_info_ptr) {
  new_slice_info_ptr->thread_scope_id = slice_info_ptr->thread_scope_id;
  new_slice_info_ptr->is_first_node_in_topo_order = slice_info_ptr->is_first_node_in_topo_order;
  new_slice_info_ptr->thread_mode = slice_info_ptr->thread_mode;
  new_slice_info_ptr->node_num_in_thread_scope = slice_info_ptr->node_num_in_thread_scope;
  new_slice_info_ptr->is_input_node_of_thread_scope = slice_info_ptr->is_input_node_of_thread_scope;
  new_slice_info_ptr->is_output_node_of_thread_scope = slice_info_ptr->is_output_node_of_thread_scope;
  new_slice_info_ptr->original_node = slice_info_ptr->original_node;
  new_slice_info_ptr->slice_instance_num = slice_info_ptr->slice_instance_num;
  new_slice_info_ptr->parallel_window_size = slice_info_ptr->parallel_window_size;
  new_slice_info_ptr->thread_id = slice_info_ptr->thread_id;
  new_slice_info_ptr->core_num = slice_info_ptr->core_num;
  new_slice_info_ptr->cut_type = slice_info_ptr->cut_type;
  new_slice_info_ptr->dependencies = slice_info_ptr->dependencies;
}
} // namespace

ModelBinaryCompiler::ModelBinaryCompiler() {}

ModelBinaryCompiler::~ModelBinaryCompiler() {}

void ModelBinaryCompiler::UpdateVariableAttrFromOriNodeToCurNode(const ge::NodePtr &ori_node,
                                                                 const ge::NodePtr &cur_node) const {
  const auto ori_op_desc = ori_node->GetOpDesc();
  const auto cur_op_desc = cur_node->GetOpDesc();

  std::vector<std::string> variable_attr;
  if (ge::AttrUtils::GetListStr(cur_op_desc, "variable_attr", variable_attr)) {
    for (const auto &variable_attr_name : variable_attr) {
      ge::GeAttrValue variable_attr_value;
      if (ori_op_desc->GetAttr(variable_attr_name, variable_attr_value) != ge::GRAPH_SUCCESS) {
        FE_LOGW("Get attr[%s] from node[%s] failed.", variable_attr_name.c_str(), ori_node->GetName().c_str());
        continue;
      }

      if (cur_op_desc->SetAttr(variable_attr_name, variable_attr_value) != ge::GRAPH_SUCCESS) {
        FE_LOGW("Setting attr[%s] to node[%s] failed.", variable_attr_name.c_str(), cur_node->GetName().c_str());
        continue;
      }
    }
  } else {
    FE_LOGW("Get attr[variable_attr] from node [%s] failed.", cur_node->GetName().c_str());
  }
}

Status ModelBinaryCompiler::UpdateFuncSubGraphNetOutputInfo(const ge::NodePtr &ori_node,
                                                            const ge::NodePtr &net_output) const {
  const auto ori_op_desc = ori_node->GetOpDesc();
  const auto net_output_op_desc = net_output->GetOpDesc();
  const size_t output_size = ori_op_desc->GetOutputsSize();
  const size_t input_size = net_output_op_desc->GetAllInputsSize();
  if (output_size != input_size) {
    REPORT_FE_ERROR("[SubGraphOpt][ModelBinary] output_size[%zu] of node[%s] is not equal to "
                    "input_size[%zu] of net_output[%s].",
                    output_size, ori_node->GetName().c_str(), input_size, net_output->GetName().c_str());
    return FAILED;
  }

  for (size_t i = 0; i < input_size; ++i) {
    const auto out_tensor_desc = ori_op_desc->MutableOutputDesc(static_cast<uint32_t>(i));
    const auto in_tensor_desc = net_output_op_desc->MutableInputDesc(static_cast<uint32_t>(i));
    if (out_tensor_desc == nullptr || in_tensor_desc == nullptr) {
      continue;
    }

    in_tensor_desc->SetFormat(out_tensor_desc->GetFormat());
    in_tensor_desc->SetOriginFormat(out_tensor_desc->GetOriginFormat());
  }

  return SUCCESS;
}

Status ModelBinaryCompiler::GetSubGraphsByCurNode(const ge::NodePtr &node,
                                                  std::vector<ge::ComputeGraphPtr> &sub_graphs) const {
  const auto sub_graph_names = node->GetOpDesc()->GetSubgraphInstanceNames();
  if (sub_graph_names.empty()) {
    return SUCCESS;
  }

  auto owner_graph = node->GetOwnerComputeGraph();
  if (owner_graph == nullptr) {
    REPORT_FE_ERROR("[SubGraphOpt][ModelBinary] The owner_graph of node [%s] is null.", node->GetName().c_str());
    return FAILED;
  }

  for (const auto &name : sub_graph_names) {
    if (name.empty()) {
      FE_LOGW("[SubGraphOpt][ModelBinary][GetSubGraph] Node [%s] contains an empty subgraph instance name.",
              node->GetName().c_str());
      continue;
    }

    auto sub_graph = owner_graph->GetSubgraph(name);
    if (sub_graph == nullptr) {
      FE_LOGW("[SubGraphOpt][ModelBinary][GetSubGraph] The subgraph [%s] for node [%s] is null.",
              name.c_str(), node->GetName().c_str());
      continue;
    }
    sub_graphs.emplace_back(sub_graph);
  }
  return SUCCESS;
}

Status ModelBinaryCompiler::UpdateConstValueByAttrBuffer(const ge::NodePtr &ori_node, const ge::NodePtr &node) const {
  bool binary_attr_required = false;
  (void)ge::AttrUtils::GetBool(node->GetOpDesc(), "_binary_attr_required", binary_attr_required);
  if (!binary_attr_required) {
    FE_LOGD("The _binary_attr_required for node [%s] is false.", node->GetName().c_str());
    return SUCCESS;
  }

  const std::vector<ge::GeTensorPtr> weights = ge::OpDescUtils::MutableWeights(node);
  if (weights.empty()) {
    REPORT_FE_ERROR("[SubGraphOpt][ModelBinary][UpdateConstValue][Op %s,type %s]: Const node does not have weight.",
                    node->GetName().c_str(), node->GetType().c_str());
    return FAILED;
  }

  if (weights[0] == nullptr) {
    REPORT_FE_ERROR("[SubGraphOpt][ModelBinary][UpdateConstValue][Op %s, type %s]: Const tensor pointer is null.",
                    node->GetName().c_str(), node->GetType().c_str());
    return FAILED;
  }

  size_t attr_size = 0;
  const auto attr_buf = gert::bg::CreateAttrBuffer(ori_node, attr_size);
  FE_CHECK_NOTNULL(attr_buf);
  weights[0]->SetData(reinterpret_cast<const uint8_t *>(attr_buf.get()), attr_size * sizeof(uint8_t));
  weights[0]->MutableTensorDesc().SetDataType(ge::DT_UINT8);
  FE_LOGD("[Op %s, type %s]: Successfully set data for const tensor.", node->GetName().c_str(), node->GetType().c_str());

  return SUCCESS;
}

bool ModelBinaryCompiler::NeedCopyTensorSliceInfo(const ge::NodePtr &ori_node,
    ffts::ThreadSliceMapPtr &slice_info_ptr) const {
  if (UnknownShapeUtils::IsUnknownShapeOp(*ori_node->GetOpDesc())) {
    FE_LOGD("Original node [%s] is an unknown shape op.", ori_node->GetName().c_str());
    return false;
  }

  const uint32_t manual_mode = 0;
  slice_info_ptr = ori_node->GetOpDesc()->TryGetExtAttr(ffts::kAttrSgtStructInfo, slice_info_ptr);
  if (slice_info_ptr != nullptr && slice_info_ptr->thread_mode == manual_mode) {
    FE_LOGD("slice_info_ptr != nullptr && slice_info_ptr->thread_mode == manual_mode node[%s]",
            ori_node->GetName().c_str());
    return true;
  }

  return false;
}

Status ModelBinaryCompiler::GetOutputTensorSliceNodesInfo(const ge::NodePtr &output_node,
    std::unordered_map<ge::NodePtr, std::set<uint32_t>> &last_nodes_info) const {
  std::set<uint32_t> output_index;
  std::vector<uint32_t> parent_index;
  const auto net_output_op_desc = output_node->GetOpDesc();
  const size_t input_size = net_output_op_desc->GetAllInputsSize();
  for (size_t i = 0; i < input_size; ++i) {
    const auto in_tensor_desc = net_output_op_desc->MutableInputDesc(static_cast<uint32_t>(i));
    if (in_tensor_desc == nullptr) {
      FE_LOGD("Node[%s] input[%zu] in_tensor_desc is null.", output_node->GetName().c_str(), i);
      continue;
    }

    int32_t parent_node_index = -1;
    if (!ge::AttrUtils::GetInt(in_tensor_desc, ge::ATTR_NAME_PARENT_NODE_INDEX, parent_node_index)) {
      REPORT_FE_ERROR("[SubGraphOpt][ModelBinary][LastNodesInfo] attribute [%s] is missing for node [%s], input [%zu].",
                      ge::ATTR_NAME_PARENT_NODE_INDEX.c_str(), output_node->GetName().c_str(), i);
      return FAILED;
    }
    if (parent_node_index < 0) {
      REPORT_FE_ERROR("[SubGraphOpt][ModelBinary][LastNodesInfo] The parent_node_index of node [%s] input [%zu] is invalid.",
                      output_node->GetName().c_str(), i);
      return FAILED;
    }
    FE_LOGD("Node[%s] input[%zu] parent_node_index is %d.", output_node->GetName().c_str(), i, parent_node_index);
    parent_index.emplace_back(parent_node_index);
  }

  if (parent_index.size() != output_node->GetAllInDataAnchors().size()) {
    REPORT_FE_ERROR("[SubGraphOpt][ModelBinary][LastNodesInfo] parent_index size[%zu] does not match anchor size[%zu].",
                    parent_index.size(), output_node->GetAllInDataAnchors().size());
    return FAILED;
  }

  for (size_t i = 0; i < output_node->GetAllInDataAnchors().size(); ++i) {
    const auto in_anchor = output_node->GetInDataAnchor(i);
    if (in_anchor == nullptr) {
      FE_LOGD("Node[%s] input[%zu] in anchor is null.", output_node->GetName().c_str(), i);
      continue;
    }
    const auto out_anchor = in_anchor->GetPeerOutAnchor();
    if (out_anchor == nullptr || out_anchor->GetOwnerNode() == nullptr) {
      FE_LOGD("Node[%s] input[%zu] in anchor PeerOutAnchor or PeerOutNode is null.", output_node->GetName().c_str(), i);
      continue;
    }
    const auto last_node = out_anchor->GetOwnerNode();
    const auto iter = last_nodes_info.find(last_node);
    if (iter == last_nodes_info.end()) {
      last_nodes_info[last_node] = {parent_index[i]};
    } else {
      iter->second.insert(parent_index[i]);
    }
  }

  return SUCCESS;
}

Status ModelBinaryCompiler::GetSubGraphTensorSliceNodesInfo(const ge::ComputeGraph &om_sub_graph,
    std::unordered_map<ge::NodePtr, std::set<uint32_t>> &first_nodes_info,
    std::unordered_map<ge::NodePtr, std::set<uint32_t>> &last_nodes_info) const {
  for (const auto &node : om_sub_graph.GetDirectNode()) {
    if (node->GetType() == NETOUTPUT) {
      if (GetOutputTensorSliceNodesInfo(node, last_nodes_info) != SUCCESS) {
        REPORT_FE_ERROR("[SubGraphOpt][ModelBinary][NodesInfo] Failed to get output tensor slice node info for node [%s].",
                        node->GetName().c_str());
        return FAILED;
      }
      continue;
    }

    std::set<uint32_t> input_index;
    for (const auto &in_node : node->GetInDataNodes()) {
      // if in_node has no input nodes, means it is input node and node is first node of sub graph
      if (!in_node->GetInAllNodes().empty() || in_node->GetType() != DATA) {
        FE_LOGD("The in_node [%s] of node [%s] is not an input node.", in_node->GetName().c_str(), node->GetName().c_str());
        continue;
      }
      const auto data_op_desc = in_node->GetOpDesc();
      FE_CHECK_NOTNULL(data_op_desc);
      int32_t parent_node_index = -1;
      if (!ge::AttrUtils::GetInt(data_op_desc, ge::ATTR_NAME_PARENT_NODE_INDEX, parent_node_index)) {
        REPORT_FE_ERROR("[SubGraphOpt][ModelBinary][NodesInfo] Attribute [%s] is missing for node [%s].",
                        ge::ATTR_NAME_PARENT_NODE_INDEX.c_str(), data_op_desc->GetName().c_str());
        return FAILED;
      }
      if (parent_node_index < 0) {
        REPORT_FE_ERROR("[SubGraphOpt][ModelBinary][NodesInfo] The parent_node_index of node [%s] is invalid.",
                        data_op_desc->GetName().c_str());
        return FAILED;
      }
      FE_LOGD("Node[%s] parent_node_index is %d.", data_op_desc->GetName().c_str(), parent_node_index);
      input_index.insert(parent_node_index);
    }

    if (!input_index.empty()) {
      first_nodes_info[node] = input_index;
    }
  }

  return SUCCESS;
}

template<typename T>
Status ModelBinaryCompiler::UpdateSingleTensorSliceInfo(const std::set<uint32_t> &index,
    std::vector<std::vector<T>> &slice_infos, std::vector<std::vector<T>> &new_slice_infos) const {
  typename std::vector<std::vector<T>>::iterator slice_info;
  for (slice_info = slice_infos.begin(); slice_info != slice_infos.end(); ++slice_info) {
    std::vector<T> new_slice_info;
    for (std::set<uint32_t>::iterator index_id = index.begin(); index_id != index.end(); ++index_id) {
      if (static_cast<uint32_t>((*slice_info).size()) < *index_id) {
        REPORT_FE_ERROR("[SubGraphOpt][ModelBinary][UpdateSliceInfo] index id[%u] is bigger than slice_info size[%zu].",
                        *index_id, (*slice_info).size());
        return FAILED;
      }
      new_slice_info.emplace_back((*slice_info)[*index_id]);
    }
    new_slice_infos.emplace_back(new_slice_info);
  }

  return SUCCESS;
}

Status ModelBinaryCompiler::UpdateAxisAndTensorIndex(const std::set<uint32_t> &index,
    const std::vector<uint32_t> &tensor_indexes, const std::vector<uint32_t> &axis,
    std::vector<uint32_t> &new_tensor_indexes, std::vector<uint32_t> &new_axis) const {
  if (tensor_indexes.size() != axis.size()) {
    REPORT_FE_ERROR("[SubGraphOpt][ModelBinary][UpdateSliceInfo] tensor_index size[%zu] not equal axis size[%zu].",
                    tensor_indexes.size(), axis.size());
    return FAILED;
  }

  for (size_t i = 0; i < tensor_indexes.size(); ++i) {
    if (index.find(tensor_indexes[i]) != index.end()) {
      new_tensor_indexes.emplace_back(tensor_indexes[i]);
      new_axis.emplace_back(axis[i]);
    }
  }

  return SUCCESS;
}

Status ModelBinaryCompiler::SetTensorSliceInfo(const ge::NodePtr &node, const std::set<uint32_t> &index,
    const bool &is_input, const ffts::ThreadSliceMapPtr &slice_info_ptr) const {
  ffts::ThreadSliceMapPtr new_slice_info_ptr = nullptr;
  FE_MAKE_SHARED(new_slice_info_ptr = std::make_shared<ffts::ThreadSliceMap>(), return FAILED);
  FE_LOGD("index is %s.", GetIndexStr(index).c_str());
  FE_LOGD("********************************original tensor slice info*********************************************");
  PrintTensorSliceInfo(slice_info_ptr);
  CopyTensorSliceBasicsInfo(slice_info_ptr, new_slice_info_ptr);
  Status status = SUCCESS;
  if (is_input) {
    status += UpdateSingleTensorSliceInfo(index, slice_info_ptr->ori_input_tensor_shape,
                                          new_slice_info_ptr->ori_input_tensor_shape);
    status += UpdateSingleTensorSliceInfo(index, slice_info_ptr->input_tensor_slice,
                                          new_slice_info_ptr->input_tensor_slice);
    status += UpdateSingleTensorSliceInfo(index, slice_info_ptr->ori_input_tensor_slice,
                                          new_slice_info_ptr->ori_input_tensor_slice);
    status += UpdateAxisAndTensorIndex(index, slice_info_ptr->input_tensor_indexes, slice_info_ptr->input_axis,
                                       new_slice_info_ptr->input_tensor_indexes, new_slice_info_ptr->input_axis);
    if (status != SUCCESS) {
      return FAILED;
    }
  } else {
    status += UpdateSingleTensorSliceInfo(index, slice_info_ptr->ori_output_tensor_shape,
                                          new_slice_info_ptr->ori_output_tensor_shape);
    status += UpdateSingleTensorSliceInfo(index, slice_info_ptr->output_tensor_slice,
                                          new_slice_info_ptr->output_tensor_slice);
    status += UpdateSingleTensorSliceInfo(index, slice_info_ptr->ori_output_tensor_slice,
                                          new_slice_info_ptr->ori_output_tensor_slice);
    status += UpdateAxisAndTensorIndex(index, slice_info_ptr->output_tensor_indexes, slice_info_ptr->output_axis,
                                       new_slice_info_ptr->output_tensor_indexes, new_slice_info_ptr->output_axis);
    if (status != SUCCESS) {
      return FAILED;
    }
  }
  FE_LOGD("********************************new tensor slice info*********************************************");
  PrintTensorSliceInfo(new_slice_info_ptr);
  if (!node->GetOpDesc()->SetExtAttr(ffts::kAttrSgtStructInfo, new_slice_info_ptr)) {
    REPORT_FE_ERROR("[SubGraphOpt][ModelBinary][SetSliceInfo][Name:%s, Type:%s] Failed to set slice info.",
                    node->GetName().c_str(), node->GetType().c_str());
    return FAILED;
  }

  return SUCCESS;
}

Status ModelBinaryCompiler::UpdateTensorSliceInfoToNode(bool is_input, const ffts::ThreadSliceMapPtr &slice_info_ptr,
    const std::unordered_map<ge::NodePtr, std::set<uint32_t>> &nodes_info) const {
  FE_LOGD("Update %s tensor slice info.", is_input ? "is_input" : "is_output");
  for (const auto &node_info : nodes_info) {
    if (SetTensorSliceInfo(node_info.first, node_info.second, is_input, slice_info_ptr) != SUCCESS) {
      return FAILED;
    }
  }
  return SUCCESS;
}

Status ModelBinaryCompiler::UpdateTensorSliceInfo(const ge::NodePtr &ori_node,
    const ge::ComputeGraph &om_sub_graph) const {
  ffts::ThreadSliceMapPtr slice_info_ptr = nullptr;
  if (!NeedCopyTensorSliceInfo(ori_node, slice_info_ptr)) {
    FE_LOGD("Node [%s] does not have tensor slice info.", ori_node->GetName().c_str());
    return SUCCESS;
  }

  std::unordered_map<ge::NodePtr, std::set<uint32_t>> first_nodes_info;
  std::unordered_map<ge::NodePtr, std::set<uint32_t>> last_nodes_info;
  if (GetSubGraphTensorSliceNodesInfo(om_sub_graph, first_nodes_info, last_nodes_info) != SUCCESS) {
    REPORT_FE_ERROR("[SubGraphOpt][ModelBinary][UpdateSliceInfo]Get slice nodes info in sub graph[%s] node[%s] failed.",
                    om_sub_graph.GetName().c_str(), ori_node->GetName().c_str());
    return FAILED;
  }

  FE_LOGD("Node[%s] first nodes size %zu, last nodes size %zu.",
          ori_node->GetName().c_str(), first_nodes_info.size(), last_nodes_info.size());
  if (UpdateTensorSliceInfoToNode(true, slice_info_ptr, first_nodes_info) != SUCCESS) {
    REPORT_FE_ERROR("[SubGraphOpt][ModelBinary][UpdateSliceInfo] Failed to update first nodes for sub graph [%s] node [%s].",
                    om_sub_graph.GetName().c_str(), ori_node->GetName().c_str());
    return FAILED;
  }

  if (UpdateTensorSliceInfoToNode(false, slice_info_ptr, last_nodes_info) != SUCCESS) {
    REPORT_FE_ERROR("[SubGraphOpt][ModelBinary][UpdateSliceInfo] Failed to update last nodes for sub graph [%s] node [%s].",
                    om_sub_graph.GetName().c_str(), ori_node->GetName().c_str());
    return FAILED;
  }

  for (const auto &node : om_sub_graph.GetDirectNode()) {
    std::vector<ge::ComputeGraphPtr> sub_graphs;
    if (GetSubGraphsByCurNode(node, sub_graphs) != SUCCESS) {
      REPORT_FE_ERROR("[SubGraphOpt][ModelBinary][UpdateSliceInfo] Failed to get sub graphs of node [%s].",
                      node->GetName().c_str());
      return FAILED;
    }

    for (const auto &sub_graph : sub_graphs) {
      if (UpdateTensorSliceInfo(node, *(sub_graph.get())) != SUCCESS) {
        REPORT_FE_ERROR("[SubGraphOpt][ModelBinary][UpdateSliceInfo] Failed to update tensor slice info by node [%s].",
                        node->GetName().c_str());
        return FAILED;
      }
    }
  }
  return SUCCESS;
}

Status ModelBinaryCompiler::RecoverAttrsWithKernelPrefix(const ge::OpDescPtr &op_desc, const std::string &kernel_prefix,
    const KernelLookup &lookup) const {
  std::string kernel_name;
  if (kernel_prefix.empty()) {
    (void)ge::AttrUtils::GetStr(op_desc, kKernelName, kernel_name);
  } else {
    (void)ge::AttrUtils::GetStr(op_desc, kernel_prefix + ge::ATTR_NAME_TBE_KERNEL_NAME, kernel_name);
  }
  if (kernel_name.empty()) {
    REPORT_FE_ERROR("[SubGraphOpt][ModelBinary][RecoverAttr] Node[%s] kernel name is empty.",
                    op_desc->GetName().c_str());
    return FAILED;
  }

  FE_LOGD("node[%s] kernel prefix is %s, kernel name is %s.",
          op_desc->GetName().c_str(), kernel_prefix.c_str(), kernel_name.c_str());
  const ge::OpKernelBinPtr tbe_kernel_ptr = lookup(kernel_name);
  if (tbe_kernel_ptr == nullptr) {
    REPORT_FE_ERROR("[SubGraphOpt][ModelBinary][RecoverAttr] tbeKernelPtr of kernelName [%s] for node [%s] is nullptr.",
                    kernel_name.c_str(), op_desc->GetName().c_str());
    return FAILED;
  }

  op_desc->SetExtAttr(kernel_prefix + ge::OP_EXTATTR_NAME_TBE_KERNEL, tbe_kernel_ptr);
  (void)ge::AttrUtils::SetStr(op_desc, kernel_prefix + ge::ATTR_NAME_TBE_KERNEL_NAME, tbe_kernel_ptr->GetName());
  FE_LOGD("node[%s]'s tbe kernel name is %s.", op_desc->GetName().c_str(), tbe_kernel_ptr->GetName().c_str());
  ge::Buffer tbe_kernel_buffer(tbe_kernel_ptr->GetBinDataSize());
  tbe_kernel_buffer = ge::Buffer::CopyFrom(tbe_kernel_ptr->GetBinData(), tbe_kernel_ptr->GetBinDataSize());
  (void)ge::AttrUtils::SetBytes(op_desc, kernel_prefix + ge::ATTR_NAME_TBE_KERNEL_BUFFER, tbe_kernel_buffer);
  size_t tbe_kernel_size = tbe_kernel_ptr->GetBinDataSize();
  (void)ge::AttrUtils::SetInt(op_desc, kernel_prefix + ATTR_NAME_TBE_KERNEL_SIZE, tbe_kernel_size);
  FE_LOGD("node[%s]'s tbe kernel buffer size is %lu.", op_desc->GetName().c_str(), tbe_kernel_buffer.GetSize());

  return SUCCESS;
}

Status ModelBinaryCompiler::RecoverAttrsForSubGraphNode(const KernelLookup &lookup, const ge::NodePtr &node) const {
  const auto op_desc = node->GetOpDesc();
  if (!op_desc->HasAttr(ge::TVM_ATTR_NAME_MAGIC)) {
    FE_LOGD("node[%s] has no attr %s, is not aicore node.",
            op_desc->GetName().c_str(), ge::TVM_ATTR_NAME_MAGIC.c_str());
    return SUCCESS;
  }

  std::vector<std::string> kernel_prefix_list;
  (void)ge::AttrUtils::GetListStr(op_desc, kKernelNamesPrefix, kernel_prefix_list);

  FE_LOGD("node[%s] kernel_prefix_list size is %zu.", op_desc->GetName().c_str(), kernel_prefix_list.size());
  if (kernel_prefix_list.empty()) {
    const std::string null_prefix;
    if (RecoverAttrsWithKernelPrefix(op_desc, null_prefix, lookup) != SUCCESS) {
      return FAILED;
    }
  } else {
    for (const auto &kernel_prefix : kernel_prefix_list) {
      if (RecoverAttrsWithKernelPrefix(op_desc, kernel_prefix, lookup) != SUCCESS) {
        return FAILED;
      }
    }
  }

  return SUCCESS;
}

Status ModelBinaryCompiler::UpdateSubGraphNodeInfoByOriginNode(const ge::NodePtr &ori_node,
    const KernelLookup &lookup, const ge::NodePtr &node) const {
  if (node->GetType() == CONSTANT) {
    return UpdateConstValueByAttrBuffer(ori_node, node);
  }

  /* format of netoutput of om_sub_graph is ND if it is not inner format in om build,
    * but in cur_graph, output format of ori_node will be chenged to inner format,
    * so need to refresh format of netoutput of om_sub_graph by ori_node's output format
    */
  if (node->GetType() == NETOUTPUT) {
    return UpdateFuncSubGraphNetOutputInfo(ori_node, node);
  }

  if (node->GetOpDesc()->HasAttr("variable_attr")) {
    UpdateVariableAttrFromOriNodeToCurNode(ori_node, node);
  }

  if (RecoverAttrsForSubGraphNode(lookup, node) != SUCCESS) {
    REPORT_FE_ERROR("[SubGraphOpt][ModelBinary][RecoverAttr] Recover attrs from node[%s] to node[%s] failed.",
                    ori_node->GetName().c_str(), node->GetName().c_str());
    return FAILED;
  }
  return SUCCESS;
}

Status ModelBinaryCompiler::UpdateNodeInfoInOmSubGraph(ge::ComputeGraph &om_sub_graph,
    const KernelLookup &lookup) const {
  const ge::NodePtr parent_node = om_sub_graph.GetParentNode();
  if (parent_node == nullptr) {
    REPORT_FE_ERROR("[SubGraphOpt][ModelBinary] The parent node of sub graph [%s] is null.",
                    om_sub_graph.GetName().c_str());
    return FAILED;
  }

  for (const auto &node : om_sub_graph.GetAllNodes()) {
    if (UpdateSubGraphNodeInfoByOriginNode(parent_node, lookup, node) != SUCCESS) {
      REPORT_FE_ERROR("[SubGraphOpt][ModelBinary] Failed to update sub-graph node [%s] info by original node [%s].",
                      node->GetName().c_str(), parent_node->GetName().c_str());
      return FAILED;
    }
  }

  if (UpdateTensorSliceInfo(parent_node, om_sub_graph) != SUCCESS) {
    REPORT_FE_ERROR("[SubGraphOpt][ModelBinary] Failed to update om_sub_graph[%s] tensor slice info by ori node[%s].",
                    om_sub_graph.GetName().c_str(), parent_node->GetName().c_str());
    return FAILED;
  }

  return SUCCESS;
}
}  // namespace fe
