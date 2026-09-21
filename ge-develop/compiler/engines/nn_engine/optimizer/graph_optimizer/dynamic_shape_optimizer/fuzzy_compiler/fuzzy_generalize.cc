/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph_optimizer/dynamic_shape_optimizer/fuzzy_compiler/fuzzy_generalize.h"
#include "common/graph/fe_graph_utils.h"
#include "common/configuration.h"
#include "adapter/common/op_store_adapter_manager.h"
#include "register/graph_optimizer/fusion_common/unknown_shape_utils.h"

namespace fe {
using namespace nlohmann;
namespace {
const int64_t kExpandNum = 2;
}

FuzzyGeneralize::FuzzyGeneralize(ge::OptimizeUtility *optimize_utility,
                                 const FEOpsKernelInfoStorePtr &ops_kernel_info_store_ptr,
                                 const FusionPriorityMgrPtr &fusion_priority_mgr_ptr)
    : optimize_utility_(optimize_utility),
      ops_kernel_info_store_ptr_(ops_kernel_info_store_ptr),
      fusion_priority_mgr_ptr_(fusion_priority_mgr_ptr) {}

FuzzyGeneralize::~FuzzyGeneralize() {}

bool FuzzyGeneralize::CheckIsFirstNode(const ge::NodePtr &node) const {
  for (const auto &in_node : node->GetInDataNodes()) {
    if (CheckIsExternalNode(in_node)) {
      return true;
    }
  }

  return false;
}

Status FuzzyGeneralize::CheckAndUpdateLimitedNodes(const OpStoreAdapterPtr &op_store_adapter,
                                                   const std::vector<ge::NodePtr> &limited_nodes,
                                                   bool &generalize_flag) {
  Status ret;
  bool is_supported = true;
  NodeGeneralInfoPtr node_info_ptr;
  FE_LOGD("[GraphOpt][Prepare][CheckAndUpdateLimitedNodes] Begin to check and update limited nodes; the number of nodes is %zu.",
          limited_nodes.size());

  for (const auto &limited_node : limited_nodes) {
    std::vector<size_t> upper_limited_input_indexs;
    std::vector<size_t> lower_limited_input_indexs;
    FE_LOGD("[GraphOpt][Prepare][CheckAndUpdateLimitedNodes] Current node: [%s, %s].",
            limited_node->GetName().c_str(), limited_node->GetType().c_str());
    Status result = LimitNode(limited_node);
    if (result == FAILED) {
      continue;
    }

    auto iter = node_info_map_.find(limited_node);
    if (iter == node_info_map_.end()) {
      FE_LOGW("[GraphOpt][Prepare][CheckAndUpdateLimitedNodes] Unable to locate the current node [%s].",
              limited_node->GetName().c_str());
      return FAILED;
    }
    node_info_ptr = iter->second;
    FE_CHECK_NOTNULL_WARNLOG(node_info_ptr);

    if (op_store_adapter->LimitedNodesCheck(is_supported, *(node_info_ptr->op_info.get()), upper_limited_input_indexs,
                                            lower_limited_input_indexs) != SUCCESS) {
      FE_LOGW("[GraphOpt][Prepare][CheckAndUpdateLimitedNodes] Node[%s] LimitedNodesCheck failed.",
              limited_node->GetName().c_str());
      return FAILED;
    }

    FE_LOGD("[GraphOpt][Prepare][Generalize] Node[%s] LimitedNodesCheck successfully.",
            limited_node->GetName().c_str());
    if (!is_supported) {
      bool is_range_out = false;
      ret = InputNodeDowngrades(limited_node, upper_limited_input_indexs, lower_limited_input_indexs, is_range_out);
      if (ret != SUCCESS) {
        FE_LOGW("[GraphOpt][Prepare][CheckAndUpdateLimitedNodes] Input node [%s] downgrade failed.",
                limited_node->GetName().c_str());
        return FAILED;
      }
      FE_LOGD("[GraphOpt][Prepare][Generalize] Node[%s], downgrades successfully.",
              limited_node->GetName().c_str());
      if (!is_range_out) {
        return SUCCESS;
      }
    }
  }

  generalize_flag = false;
  return SUCCESS;
}

Status FuzzyGeneralize::LimitNode(const ge::NodePtr &limited_node) const {
  if (!UnknownShapeUtils::IsUnknownShapeOp(*(limited_node->GetOpDesc()))) {
    FE_LOGD("[GraphOpt][Prepare][CheckAndUpdateLimitedNodes] Current node [%s, %s] is not an unknownShape op.",
            limited_node->GetName().c_str(), limited_node->GetType().c_str());
    return FAILED;
  }

  if (CheckIsFirstNode(limited_node)) {
    FE_LOGD("[GraphOpt][Prepare][CheckAndUpdateLimitedNodes] First node [%s, %s] does not require range checking.",
            limited_node->GetName().c_str(), limited_node->GetType().c_str());
    return FAILED;
  }
  return SUCCESS;
}

Status FuzzyGeneralize::SingleOpDowngrades(const ge::NodePtr &external_node, const bool &is_upper_limited,
                                           bool &is_range_out) {
  if (is_upper_limited) {
    std::string op_name = external_node->GetName();
    auto iter_decent = decent_times_count_.find(op_name);
    if (iter_decent == decent_times_count_.end()) {
      FE_LOGW("[GraphOpt][Prepare][SingleOpDowngrades] Unable to find expected info for node[%s].",
              op_name.c_str());
      return FAILED;
    }
    if (iter_decent->second >= MAX_DECENT_TIMES) {
      is_range_out = true;
    } else {
      if (RangeDecent(external_node, iter_decent->second) != SUCCESS) {
        FE_LOGW(
            "[GraphOpt][Prepare][SingleOpDowngrades] Node[%s]: fail to make range-decent.", op_name.c_str());
        return FAILED;
      }
    }
  } else {
    is_range_out = true;
  }

  return SUCCESS;
}

Status FuzzyGeneralize::Downgrades(const ge::NodePtr &cur_node, const bool &is_upper_limited,
                                   const std::vector<size_t> &limited_input_indexs, bool &is_range_out) {
  std::map<ge::NodePtr, NodeGeneralInfoPtr>::const_iterator iter_node = node_info_map_.find(cur_node);
  if (iter_node == node_info_map_.end()) {
    FE_LOGW("[GraphOpt][Prepare][Downgrades] Node[%s], could not find cur_node in node_info_map, generalization failed.",
            cur_node->GetName().c_str());
    return FAILED;
  }
  bool all_input_nodes_static = true;
  auto node_info = iter_node->second;
  for (auto &idx : limited_input_indexs) {
    auto input_desc = cur_node->GetOpDesc()->MutableInputDesc(idx);
    auto iter = node_info->inputs_root_map.find(input_desc);
    if (iter == node_info->inputs_root_map.end()) {
      FE_LOGW("[GraphOpt][Prepare][Downgrades] Cannot find the root set by index[%zu], node[%s], generalize failed.",
              idx, cur_node->GetName().c_str());
      return FAILED;
    }
    for (auto &external_node : iter->second) {
      if (!UnknownShapeUtils::IsUnknownShapeOp(*(external_node->GetOpDesc()))) {
        continue;
      }

      all_input_nodes_static = false;
      if (SingleOpDowngrades(external_node, is_upper_limited, is_range_out) != SUCCESS) {
        FE_LOGW("[GraphOpt][Prepare][Downgrades] Node [%s] downgrade failed.", external_node->GetName().c_str());
        return FAILED;
      }
    }
  }

  if (all_input_nodes_static) {
    FE_LOGW(
        "[GraphOpt][Prepare][Downgrades] Node[%s]: all input nodes have static shape, but it is still not supported.",
        cur_node->GetName().c_str());
    return FAILED;
  }
  FE_LOGD("[GraphOpt][Prepare][Downgrades] Node [%s, %s], downgraded successfully by lower_limited_indexs.",
          cur_node->GetName().c_str(), cur_node->GetType().c_str());
  return SUCCESS;
}

Status FuzzyGeneralize::InputNodeDowngrades(const ge::NodePtr &cur_node,
                                            const std::vector<size_t> &upper_limited_input_indexs,
                                            const std::vector<size_t> &lower_limited_input_indexs,
                                            bool &is_range_out) {
  bool is_upper_limited = true;
  if (!upper_limited_input_indexs.empty()) {
    if (Downgrades(cur_node, is_upper_limited, upper_limited_input_indexs, is_range_out) != SUCCESS) {
      return FAILED;
    }
  }

  if (!lower_limited_input_indexs.empty()) {
    is_upper_limited = false;
    if (Downgrades(cur_node, is_upper_limited, lower_limited_input_indexs, is_range_out) != SUCCESS) {
      return FAILED;
    }
  }

  FE_LOGD("[GraphOpt][Prepare][InputNodeDowngrades] Node[%s], run func[InputNodeDowngrades] successfully.",
          cur_node->GetName().c_str());
  return SUCCESS;
}

void FuzzyGeneralize::UpdateOpAttrs(const ge::GeTensorDescPtr &cur_tensor_desc,
    const ge::GeTensorDescPtr &ori_tensor_desc, const ge::OpDescPtr &op_desc_ptr, bool is_input) const {
  // delete const value
  if (is_input) {
    if (ge::AttrUtils::HasAttr(ori_tensor_desc, ge::ATTR_NAME_VALUE) &&
        !ge::AttrUtils::HasAttr(cur_tensor_desc, ge::ATTR_NAME_VALUE)) {
      ori_tensor_desc->DelAttr(ge::ATTR_NAME_VALUE);
      FE_LOGD("[GraphOpt][Prepare][Generalize] Peer node[%s] input tensor desc has ATTR_NAME_VALUE.",
              op_desc_ptr->GetName().c_str());
    }
  } else {
    if (ge::AttrUtils::HasAttr(ori_tensor_desc, ge::ATTR_NAME_VALUE)) {
      ori_tensor_desc->DelAttr(ge::ATTR_NAME_VALUE);
      FE_LOGD("[GraphOpt][Prepare][Generalize] Peer node [%s] output tensor description contains ATTR_NAME_VALUE.",
              op_desc_ptr->GetName().c_str());
    }
  }

  // set op generalized flag
  if (cur_tensor_desc->GetOriginShape().IsUnknownShape() &&
      !ori_tensor_desc->GetOriginShape().IsUnknownShape()) {
    FE_LOGD("[GraphOpt][Prepare][Generalize] node[%s] has generalized.", op_desc_ptr->GetName().c_str());
    (void)ge::AttrUtils::SetBool(op_desc_ptr, "_is_op_generalized", true);
  }
}

bool FuzzyGeneralize::IsOpDescExtInputOutputTensor(const ge::OpDescPtr &opdesc, bool input) const {
  if (input) {
    return !(opdesc->GetInputsSize() == 0);
  } else {
    return !(opdesc->GetOutputsSize() == 0);
  }
}

Status FuzzyGeneralize::UpdateDynamicShapeToNewInputNode(const std::unordered_set<ge::NodePtr> &external_input_nodes,
                                                         const std::map<std::string, ge::NodePtr> &new_input_nodes)
                                                         const {
  for (const ge::NodePtr &input_node : external_input_nodes) {
    auto iter = new_input_nodes.find(input_node->GetName());
    if (iter == new_input_nodes.end()) {
      FE_LOGW("[GraphOpt][Prepare][Generalize] Node [%s], unable to locate this node on ori_graph; generalization failed.",
              input_node->GetName().c_str());
      return FAILED;
    }

    bool is_value_depend = false;
    (void)ge::AttrUtils::GetBool(input_node->GetOpDesc(), ge::ATTR_NAME_VALUE_DEPEND, is_value_depend);
    if (is_value_depend) {
      if (!ge::AttrUtils::SetBool(iter->second->GetOpDesc(), ge::ATTR_NAME_VALUE_DEPEND, is_value_depend)) {
        FE_LOGW("[GraphOpt][Prepare][Generalize] Node[%s] set is_value_depend failed.", input_node->GetName().c_str());
      }
    }

    bool exist_input_desc = IsOpDescExtInputOutputTensor(input_node->GetOpDesc(), true) &&
        IsOpDescExtInputOutputTensor(iter->second->GetOpDesc(), true);
    bool exist_output_desc = IsOpDescExtInputOutputTensor(input_node->GetOpDesc(), false) &&
        IsOpDescExtInputOutputTensor(iter->second->GetOpDesc(), false);
    if (exist_input_desc) {
      ge::GeTensorDescPtr cur_input_tensor_desc = input_node->GetOpDesc()->MutableInputDesc(0);
      ge::GeTensorDescPtr ori_input_tensor_desc = iter->second->GetOpDesc()->MutableInputDesc(0);
      FE_CHECK_NOTNULL_WARNLOG(cur_input_tensor_desc);
      FE_CHECK_NOTNULL_WARNLOG(ori_input_tensor_desc);
      FE_LOGD("[GraphOpt][Prepare][Generalize] Update dynamic shape to origin graph for node [%s, %s] input.",
              input_node->GetName().c_str(), input_node->GetType().c_str());
      UpdateOpAttrs(cur_input_tensor_desc, ori_input_tensor_desc, iter->second->GetOpDesc(), true);
      UpdateTensorDesc(cur_input_tensor_desc, ori_input_tensor_desc);
    }
    if (exist_output_desc) {
      ge::GeTensorDescPtr cur_output_tensor_desc = input_node->GetOpDesc()->MutableOutputDesc(0);
      ge::GeTensorDescPtr ori_output_tensor_desc = iter->second->GetOpDesc()->MutableOutputDesc(0);
      FE_CHECK_NOTNULL_WARNLOG(cur_output_tensor_desc);
      FE_CHECK_NOTNULL_WARNLOG(ori_output_tensor_desc);
      FE_LOGD("[GraphOpt][Prepare][Generalize] Update dynamic shape to origin graph for node [%s, %s] output.",
              input_node->GetName().c_str(), input_node->GetType().c_str());
      UpdateOpAttrs(cur_output_tensor_desc, ori_output_tensor_desc, iter->second->GetOpDesc(), false);
      UpdateTensorDesc(cur_output_tensor_desc, ori_output_tensor_desc);
    }
  }

  return SUCCESS;
}

Status FuzzyGeneralize::UpdateDynamicShapeToNewBakGraph(const ge::ComputeGraph &graph) const {
  std::map<std::string, ge::NodePtr> new_inputs_node;
  for (const ge::NodePtr &cur_node : graph.GetDirectNode()) {
    if (CheckIsExternalNode(cur_node)) {
      new_inputs_node.emplace(cur_node->GetName(), cur_node);
    }
  }

  Status ret = UpdateDynamicShapeToNewInputNode(external_input_nodes_, new_inputs_node);
  if (ret != SUCCESS) {
    FE_LOGW("[GraphOpt][Prepare][Generalize] Failed to UpdateDynamicShapeToNewInputNode for Graph[%s] on new backup graph.",
            graph.GetName().c_str());
    return FAILED;
  }

  return SUCCESS;
}

Status FuzzyGeneralize::GetReshapeTypeByOpStore(const ge::NodePtr &node, const std::string &input_name,
                                                std::string &reshape_type) const {
  auto iter_node_info = node_info_map_.find(node);
  if (iter_node_info == node_info_map_.end()) {
    FE_LOGW("[GraphOpt][Prepare][Generalize] Unable to locate node [%s, %s] within node_info_map._.", node->GetName().c_str(),
            node->GetType().c_str());
    return FAILED;
  }

  auto node_info = iter_node_info->second;
  FE_CHECK_NOTNULL_WARNLOG(node_info);

  auto op_kernel = node_info->op_kernel;
  if (op_kernel == nullptr) {
    FE_LOGD("[GraphOpt][Prepare][Generalize] The op kernel for node [%s] is null.", node->GetName().c_str());
    return SUCCESS;
  }

  InputOrOutputInfoPtr input_info;
  (void)op_kernel->GetInputInfoByName(input_name, input_info);
  if (input_info == nullptr) {
    FE_LOGD("[GraphOpt][Prepare][Generalize] The input_info for the input name [%s] of node [%s] is null.", input_name.c_str(),
            node->GetName().c_str());
    return FAILED;
  }

  reshape_type = input_info->GetReshapeType();

  return SUCCESS;
}

Status FuzzyGeneralize::GetReshapeType(const ge::Format &origin_format, ge::GeShape &ori_shape,
                                       const ge::NodePtr &first_node, const std::string &input_name,
                                       std::string &reshape_type) const {
  std::string op_name = first_node->GetName();
  reshape_type = ge::TypeUtils::FormatToSerialString(origin_format);
  if (origin_format == ge::FORMAT_ND) {
    Status ret = GetReshapeTypeByOpStore(first_node, input_name, reshape_type);
    if (ret != SUCCESS) {
      FE_LOGW("[GraphOpt][Prepare][Generalize] Failed to obtain the reshape type for input op [%s] of node [%s].", input_name.c_str(),
              op_name.c_str());
      return FAILED;
    }
  } else {
    if (ori_shape.GetDims().size() < DIM_DEFAULT_SIZE) {
      std::string reshape_type_temp;
      if (GetReshapeTypeByOpStore(first_node, input_name, reshape_type_temp) != SUCCESS) {
        FE_LOGW("[GraphOpt][Prepare][Generalize] Failed to get reshape type for input[%s] of node[%s].", input_name.c_str(),
                op_name.c_str());
        return FAILED;
      }
      FE_LOGD("[GraphOpt][Prepare][Generalize] Node[%s] reshape_type from opStore is: %s.", op_name.c_str(),
              reshape_type_temp.c_str());

      if (reshape_type_temp.empty() &&
          GetDefaultReshapeType(origin_format, ori_shape.GetDims().size(), reshape_type_temp) != SUCCESS) {
        FE_LOGW("[GraphOpt][Prepare][Generalize] node[%s] get default reshape type failed.", op_name.c_str());
        return FAILED;
      }
      reshape_type = reshape_type_temp;
    }
  }

  FE_LOGD("[GraphOpt][Prepare][Generalize] Node[%s] origin_format[%s], shape size is %zu, reshape_type[%s].",
          op_name.c_str(), ge::TypeUtils::FormatToSerialString(origin_format).c_str(), ori_shape.GetDims().size(),
          reshape_type.c_str());
  return SUCCESS;
}

Status FuzzyGeneralize::CorrectCAxisByOriginalFormat(const ge::Format &origin_format, const ge::NodePtr &input_node,
                                                     const ge::NodePtr &first_node, const std::string &input_name)
                                                     const {
  auto iter_input = original_input_nodes_.find(input_node->GetName());
  if (iter_input == original_input_nodes_.end()) {
    FE_LOGW("[GraphOpt][Prepare][Generalize] can not find node[%s] in original_input_nodes.", input_node->GetNamePtr());
    return FAILED;
  }
  ge::NodePtr ori_input_node = iter_input->second;
  FE_CHECK_NOTNULL_WARNLOG(ori_input_node);
  ge::GeTensorDescPtr ori_tensor_desc = ori_input_node->GetOpDesc()->MutableInputDesc(0);
  FE_CHECK_NOTNULL_WARNLOG(ori_tensor_desc);
  ge::GeShape ori_shape = ori_tensor_desc->GetOriginShape();

  std::string reshape_type;
  if (GetReshapeType(origin_format, ori_shape, first_node, input_name, reshape_type) != SUCCESS) {
    return FAILED;
  }

  size_t c_idx = reshape_type.find('C');
  if (c_idx == std::string::npos) {
    FE_LOGD("[GraphOpt][Prepare][Generalize] reshape_type does not contain char C, input name [%s], node [%s].",
            input_name.c_str(), input_node->GetNamePtr());
    return SUCCESS;
  }

  if (reshape_type.size() <= DIM_DEFAULT_SIZE) {
    ge::GeTensorDescPtr tensor_desc = input_node->GetOpDesc()->MutableInputDesc(0);
    FE_CHECK_NOTNULL_WARNLOG(tensor_desc);
    std::vector<std::pair<int64_t, int64_t>> range;
    (void)tensor_desc->GetOriginShapeRange(range);
    ge::GeShape shape = tensor_desc->GetOriginShape();
    FE_LOGD("[GraphOpt][Prepare][Generalize] Before correcting C axis, c_idx[%zu], ori_shape[%s], shape[%s], range[%s].",
            c_idx, ShapeToString(ori_shape.GetDims()).c_str(), ShapeToString(shape.GetDims()).c_str(),
            RangeToString(range).c_str());
    if (c_idx >= range.size()) {
      FE_LOGW("[GraphOpt][Prepare][Generalize] Param invalid, c idx is %zu, but range size is %zu.", c_idx,
              range.size());
      return FAILED;
    }

    auto dim_val = ori_shape.GetDim(c_idx);
    (void)shape.SetDim(c_idx, dim_val);
    range[c_idx].first = dim_val;
    range[c_idx].second = dim_val;
    tensor_desc->SetOriginShape(shape);
    (void)tensor_desc->SetOriginShapeRange(range);
    ge::GeTensorDescPtr out_tensor_desc = input_node->GetOpDesc()->MutableOutputDesc(0);
    UpdateTensorDesc(tensor_desc, out_tensor_desc);
    FE_LOGD("[GraphOpt][Prepare][Generalize] After correct C axis, shape is %s, range is %s.",
            ShapeToString(shape.GetDims()).c_str(), RangeToString(range).c_str());
  } else {
    FE_LOGW("[GraphOpt][Prepare][Generalize] The formatted size of node [%s] exceeds 4.", input_node->GetNamePtr());
    return FAILED;
  }

  return SUCCESS;
}

Status FuzzyGeneralize::CorrectInputNodeCAxisByFirstNode(const ge::NodePtr &input_node) const {
  Status ret;
  for (auto &output_anchor : input_node->GetAllOutDataAnchors()) {
    auto peer_in_anchors = output_anchor->GetPeerInDataAnchors();
    for (size_t i = 0; i < peer_in_anchors.size(); ++i) {
      ge::NodePtr next_node = peer_in_anchors.at(i)->GetOwnerNode();
      FE_CHECK_NOTNULL_WARNLOG(next_node);

      ge::OpDescPtr next_node_desc = next_node->GetOpDesc();
      FE_CHECK_NOTNULL_WARNLOG(next_node_desc);

      uint32_t in_data_anchor_index = peer_in_anchors.at(i)->GetIdx();
      ge::GeTensorDescPtr tensor_desc = next_node_desc->MutableInputDesc(in_data_anchor_index);
      FE_CHECK_NOTNULL_WARNLOG(tensor_desc);

      if (tensor_desc->GetOriginShape().IsUnknownDimNum()) {
        FE_LOGD("Shape contains -2, no correction needed for c-axis.");
        return SUCCESS;
      }

      ret = CorrectCAxisByOriginalFormat(tensor_desc->GetOriginFormat(), input_node, next_node,
                                         next_node_desc->GetInputNameByIndex(in_data_anchor_index));
      if (ret != SUCCESS) {
        FE_LOGW("[GraphOpt][Prepare][Generalize] Node[%s] Failed to Correct CAxis By Original Format.",
                input_node->GetName().c_str());
        return FAILED;
      }
    }
  }

  return SUCCESS;
}

Status FuzzyGeneralize::CAxisCorrection() const {
  Status ret;
  for (const auto &input_node : external_input_nodes_) {
    if (!UnknownShapeUtils::IsUnknownShapeOp(*(input_node->GetOpDesc()))) {
      FE_LOGD("[GraphOpt][Prepare][Generalize] Current node [%s] is not an unknownShape op.",
              input_node->GetName().c_str());
      continue;
    }

    ge::GeTensorDescPtr tensor_desc = input_node->GetOpDesc()->MutableInputDesc(0);
    if (tensor_desc == nullptr) {
      FE_LOGD("[GraphOpt][Prepare][Generalize] The first input tensor of the current node [%s] is null.",
              input_node->GetName().c_str());
      continue;
    }
    int64_t format = ge::FORMAT_RESERVED;
    (void)ge::AttrUtils::GetInt(*tensor_desc, ge::ATTR_NAME_STORAGE_FORMAT, format);
    ge::Format storage_format = static_cast<ge::Format>(format);
    if (storage_format != ge::FORMAT_NC1HWC0) {
      FE_LOGD("[GraphOpt][Prepare][Generalize] The storage_format of node [%s] is not NC1HWC0.",
              input_node->GetName().c_str());
      continue;
    }

    ret = CorrectInputNodeCAxisByFirstNode(input_node);
    if (ret != SUCCESS) {
      FE_LOGW(
          "[GraphOpt][Prepare][Generalize] Node [%s] CorrectInputNodeCAxisByFirstNode unsuccessful, generalization failed.",
          input_node->GetName().c_str());
      return FAILED;
    }
  }

  return SUCCESS;
}

Status FuzzyGeneralize::UpdateDynamicShapeToFirstNode(const ge::NodePtr &ori_input_node) const {
  for (auto &output_anchor : ori_input_node->GetAllOutDataAnchors()) {
    auto peer_in_anchors = output_anchor->GetPeerInDataAnchors();
    auto output_desc_ptr = ori_input_node->GetOpDesc()->MutableInputDesc(output_anchor->GetIdx());
    for (size_t i = 0; i < peer_in_anchors.size(); ++i) {
      ge::NodePtr next_node = peer_in_anchors.at(i)->GetOwnerNode();
      FE_CHECK_NOTNULL_WARNLOG(next_node);

      ge::OpDescPtr next_node_desc = next_node->GetOpDesc();
      FE_CHECK_NOTNULL_WARNLOG(next_node_desc);

      uint32_t in_data_anchor_index = peer_in_anchors.at(i)->GetIdx();
      ge::GeTensorDescPtr tensor_desc = next_node_desc->MutableInputDesc(in_data_anchor_index);
      FE_CHECK_NOTNULL_WARNLOG(tensor_desc);

      if (ge::AttrUtils::HasAttr(tensor_desc, ge::ATTR_NAME_VALUE) &&
          !ge::AttrUtils::HasAttr(output_desc_ptr, ge::ATTR_NAME_VALUE)) {
        tensor_desc->DelAttr(ge::ATTR_NAME_VALUE);
      }

      UpdateTensorDesc(output_desc_ptr, tensor_desc);
    }
  }

  return SUCCESS;
}

bool FuzzyGeneralize::IsAllUnknownDimNum(const ge::GeShape &shape) {
  const auto &shape_dims = shape.GetDims();
  return std::all_of(shape_dims.begin(), shape_dims.end(),
      [](const int64_t &shape_dim_num) {
          return (shape_dim_num == -1);
      });
}

bool FuzzyGeneralize::IsAllUnknownDim(const std::vector<std::pair<int64_t, int64_t>> &range) {
  const std::pair<int64_t, int64_t> std_unknown_dim(1, -1);
  return std::all_of(range.begin(), range.end(),
      [&std_unknown_dim](const std::pair<int64_t, int64_t> &range_dim) {
          return (range_dim == std_unknown_dim);
      });
}

bool FuzzyGeneralize::IsAllUnKnownShapeAndRange(const ge::GeTensorDescPtr &output_tensor_desc) {
  const ge::GeShape &shape = output_tensor_desc->MutableShape();
  const ge::GeShape &ori_shape = output_tensor_desc->GetOriginShape();
  if (!IsAllUnknownDimNum(shape) || !IsAllUnknownDimNum(ori_shape)) {
    return false;
  }
  std::vector<std::pair<int64_t, int64_t>> range;
  std::vector<std::pair<int64_t, int64_t>> ori_range;
  output_tensor_desc->GetShapeRange(range);
  output_tensor_desc->GetShapeRange(ori_range);
  return (IsAllUnknownDim(range) && IsAllUnknownDim(ori_range));
}

void FuzzyGeneralize::FurtherGeneralizeShapeAndRange(const ge::GeTensorDescPtr &output_tensor_desc) {
  ge::GeShape unknown_dim({-2});
  (void)output_tensor_desc->SetUnknownDimNumShape();
  (void)output_tensor_desc->SetOriginShape(unknown_dim);
  std::vector<std::pair<int64_t, int64_t>> null_range;
  (void)output_tensor_desc->SetShapeRange(null_range);
  (void)output_tensor_desc->SetOriginShapeRange(null_range);
}

void FuzzyGeneralize::GeneralizeNodeToUnknownDim(const ge::NodePtr &single_node) {
  auto inputs_desc = single_node->GetOpDesc()->GetAllInputsDescPtr();
  auto outputs_desc = single_node->GetOpDesc()->GetAllOutputsDescPtr();

  for (auto &input_desc : inputs_desc) {
    FurtherGeneralizeShapeAndRange(input_desc);
  }

  for (auto &output_desc : outputs_desc) {
    FurtherGeneralizeShapeAndRange(output_desc);
  }
}

Status FuzzyGeneralize::DoFurtherGeneralize(const std::string &graph_name) const {
  FE_LOGD("[GraphOpt][Prepare][DoFurtherGeneralize] Start to do further generalize for graph %s.", graph_name.c_str());
  if (!is_single_op_graph_) {
    FE_LOGD("[GraphOpt][Prepare][DoFurtherGeneralize] Graph[%s]: no need to perform further generalization for the single op scenario.",
            graph_name.c_str());
    return SUCCESS;
  }

  ge::NodePtr single_node;
  for (auto &node_info : node_info_map_) {
    single_node = node_info.first;
    FE_CHECK_NOTNULL_WARNLOG(single_node);
    if (!CheckIsExternalNode(single_node)) {
      break;
    }
  }

  BinaryKernelInfo &bin_kernel_info = BinaryKernelInfo::Instance();
  FE_CHECK_NOTNULL(single_node);
  if (!bin_kernel_info.IsBinSupportDynamicRank(single_node->GetType())) {
    FE_LOGD("[GraphOpt][Prepare][DoFurtherGeneralize] Current op [%s] does not support dynamic rank in the binary information storage.",
            single_node->GetName().c_str());
    return SUCCESS;
  }

  for (auto &input_desc : single_node->GetOpDesc()->GetAllInputsDescPtr()) {
    if (!IsAllUnKnownShapeAndRange(input_desc)) {
      FE_LOGD("[GraphOpt][Prepare][DoFurtherGeneralize] Input shape or range for single node [%s] is not entirely unknown dimensions.",
              single_node->GetName().c_str());
      return SUCCESS;
    }
  }
  FE_LOGD("[GraphOpt][Prepare][DoFurtherGeneralize] Node[%s]: beginning further generalization.",
          single_node->GetName().c_str());
  GeneralizeNodeToUnknownDim(single_node);

  for (auto &input_node : external_input_nodes_) {
    GeneralizeNodeToUnknownDim(input_node);
  }
  FE_LOGD("[GraphOpt][Prepare][DoFurtherGeneralize] Node[%s]: finished doing further generalization.",
          single_node->GetName().c_str());
  return SUCCESS;
}

Status FuzzyGeneralize::UpdateDynamicShapeToOriginalGraph(const ge::ComputeGraph &graph) const {
  std::string graph_name = graph.GetName();
  Status ret = CAxisCorrection();
  if (ret != SUCCESS) {
    FE_LOGW("[GraphOpt][Prepare][Generalize] Graph[%s] CAxisCorrection failed, generalization unsuccessful.",
            graph_name.c_str());
    return FAILED;
  }

  if (DoFurtherGeneralize(graph.GetName()) != SUCCESS) {
    FE_LOGW("[GraphOpt][Prepare][Generalize] Graph[%s] DoFurtherGeneralize unsuccessful, generalization failed.",
            graph_name.c_str());
    return FAILED;
  }

  ret = UpdateDynamicShapeToNewInputNode(external_input_nodes_, original_input_nodes_);
  if (ret != SUCCESS) {
    FE_LOGW("[GraphOpt][Prepare][Generalize] Graph[%s] failed to update DynamicShape to original graph, generalization not successfully.",
            graph.GetName().c_str());
    return FAILED;
  }

  for (auto &original_input_node : original_input_nodes_) {
    ret = UpdateDynamicShapeToFirstNode(original_input_node.second);
    if (ret != SUCCESS) {
      FE_LOGW("[GraphOpt][Prepare][Generalize] Failed to update dynamic shape for the first node in Graph [%s].",
              graph.GetName().c_str());
      return FAILED;
    }
  }

  return SUCCESS;
}

Status FuzzyGeneralize::GraphDynamicShapeInfer(const OpStoreAdapterPtr &op_store_adapter,
                                               ge::ComputeGraphPtr &graph_bak, const ge::ComputeGraphPtr &ori_graph) {
  Status ret;
  bool generalize_flag = true;
  bool is_first_bak_graph = true;
  uint32_t graph_bak_index = 0;
  std::string graph_name = graph_bak->GetName();
  FE_LOGD("[GraphOpt][Prepare][GraphDynamicShapeInfer] Current graph name: %s.", graph_name.c_str());
  FE_TIMECOST_START(GraphDynamicShapeInfer);

  /* 1.generalize_flag = true means need to continue to generalize, if passed the limited node check,
   *   it will be changed to false
   * 2.graph_bak_index means downgrades total count, current threshold is 30, prevent unlimited downgrades
   **/
  while (generalize_flag && graph_bak_index < kDowngradesTimeMax) {
    // not first time to infer shape dynamic shape graph, need to copy new graph from original graph
    if (!is_first_bak_graph) {
      graph_name = ori_graph->GetName() + "_bak_" + std::to_string(graph_bak_index);
      FE_LOGD("[GraphOpt][Prepare][GraphDynamicShapeInfer] graph_bak name: %s.", graph_name.c_str());
      ge::ComputeGraphPtr new_graph_bak = nullptr;
      FE_MAKE_SHARED(new_graph_bak = std::make_shared<ge::ComputeGraph>(graph_name), return FAILED);

      if (ge::GraphUtils::CopyComputeGraph(ori_graph, new_graph_bak) != ge::GRAPH_SUCCESS) {
        FE_LOGW("[GraphOpt][Prepare][GraphDynamicShapeInfer] Failed to copy graph [%s] by GE.", graph_name.c_str());
        return FAILED;
      }

      if (UpdateDynamicShapeToNewBakGraph(*new_graph_bak) != SUCCESS) {
        FE_LOGW("[GraphOpt][Prepare][GraphDynamicShapeInfer] Failed to update dynamic shape to original for graph [%s].",
                graph_name.c_str());
        return FAILED;
      }

      graph_bak = new_graph_bak;
      graph_bak_index++;
    } else {
      is_first_bak_graph = false;
    }
    ret = optimize_utility_->InferShape(graph_bak);
    if (ret != SUCCESS) {
      FE_LOGW("[GraphOpt][Prepare][GraphDynamicShapeInfer] Dynamic shape inference for graph [%s] failed by GE.",
              graph_name.c_str());
      return FAILED;
    }

    FeGraphUtils::DumpGraphAndOnnx(*graph_bak, "OptimizeGraph_GeneralizeDowngradesAfter");
    FeGraphUtils::DumpSubGraphAndOnnx(*graph_bak, "OptimizeGraph_GeneralizeDowngradesAfter_Subgraph");

    // traversing graph, limit type notarize, get input_nodes/limited_nodes/input_nodes... of every node
    ret = GraphPreprocessing(*graph_bak, op_store_adapter);
    if (ret != SUCCESS) {
      FE_LOGW("[GraphOpt][Prepare][GraphDynamicShapeInfer] graph[%s] preprocessing failed.", graph_name.c_str());
      return FAILED;
    }

    // fater infer shape, call tbe generalize func to check the validity of shape and range for limited nodes
    ret = CheckAndUpdateLimitedNodes(op_store_adapter, limited_range_nodes_, generalize_flag);
    if (ret != SUCCESS) {
      FE_LOGW("[GraphOpt][Prepare][GraphDynamicShapeInfer] Graph [%s] limited nodes check failed.", graph_name.c_str());
      return FAILED;
    }
  }

  // if dynamic shape graph generalize success, update shape and range of input_nodes from bak graph to original graph
  FE_LOGD("[GraphOpt][Prepare][GraphDynamicShapeInfer] generalize_flag: %d.", generalize_flag);
  if (!generalize_flag) {
    if (UpdateDynamicShapeToOriginalGraph(*ori_graph) != SUCCESS) {
      FE_LOGW(
          "[GraphOpt][Prepare][UpdateDynamicShapeToOriginalGraph] Graph[%s]: unable to update dynamic shape to original.",
          graph_name.c_str());
      return FAILED;
    }
  }

  FE_TIMECOST_END(GraphDynamicShapeInfer, "FEGraphOptimizer::GraphDynamicShapeInfer");
  return SUCCESS;
}

Status FuzzyGeneralize::InitOriginalGraphInfos(const ge::ComputeGraph &graph) {
  if (external_input_nodes_.empty()) {
    FE_LOGW("[GraphOpt][Prepare][InitOriginalGraphInfos] external_input_nodes_ is null, generalization for graph [%s] failed.",
            graph.GetName().c_str());
    return FAILED;
  }

  std::vector<double> non_vec;
  for (auto &node : external_input_nodes_) {
    decent_times_count_.insert(std::make_pair(node->GetName(), 1));
    decent_steps_.insert(std::make_pair(node->GetName(), non_vec));
  }

  for (const auto &cur_node : graph.GetDirectNode()) {
    FE_LOGD("[GraphOpt][Prepare][InitOriginalGraphInfos] Graph[%s] Initializing original graph info, node[%s, %s].",
            graph.GetName().c_str(), cur_node->GetName().c_str(), cur_node->GetType().c_str());
    bool node_can_be_generalized = (CheckIsExternalNode(cur_node) &&
                                    !CheckOpConstOrVariableInOriGraph(cur_node->GetOpDesc()));
    if (node_can_be_generalized) {
      original_input_nodes_.emplace(std::make_pair(cur_node->GetName(), cur_node));
    }
  }

  return SUCCESS;
}

Status FuzzyGeneralize::GeneralizeGraph(ge::ComputeGraph &graph) {
  const std::string graph_name = graph.GetName();
  const std::string graph_bak_name = graph_name + "_bak";

  ge::ComputeGraphPtr tmp_graph_ptr = nullptr;
  FE_MAKE_SHARED(tmp_graph_ptr = std::make_shared<ge::ComputeGraph>(graph), return FAILED);
  ge::ComputeGraphPtr new_graph_bak = nullptr;
  FE_MAKE_SHARED(new_graph_bak = std::make_shared<ge::ComputeGraph>(graph_bak_name), return FAILED);

  (void)ge::AttrUtils::GetBool(tmp_graph_ptr, ge::ATTR_SINGLE_OP_SCENE, is_single_op_graph_);
  FE_LOGD("[GraphOpt][Prepare][GeneralizeGraph] Cur graph[%s] is single_op_scene_flag[%d].",
          graph_name.c_str(), is_single_op_graph_);

  if (ge::GraphUtils::CopyComputeGraph(tmp_graph_ptr, new_graph_bak) != ge::GRAPH_SUCCESS) {
    FE_LOGW("[GraphOpt][Generalize][GeneralizeGraph] Failed to copy graph by ge, generalizing graph [%s] failed.",
            graph_name.c_str());
    return FAILED;
  }

  // static shape integral graph infer shape for first node generalize
  if (optimize_utility_ == nullptr) {
    FE_LOGW("[GraphOpt][Generalize][GeneralizeGraph] optimize_utility_ is null; graph [%s] generalization failed.",
            graph_name.c_str());
    return FAILED;
  }
  Status ret = optimize_utility_->InferShape(new_graph_bak);
  if (ret != SUCCESS) {
    FE_LOGW(
        "[GraphOpt][Generalize][GeneralizeGraph] Failed to infer static shape for Graph[%s] by GE.",
        graph_name.c_str());
    return FAILED;
  }
  const std::string &engine_name = ops_kernel_info_store_ptr_->GetFEOpsKernelInfoStoreName();
  OpStoreAdapterPtr op_store_adapter = nullptr;
  if (OpStoreAdapterManager::Instance(engine_name).GetOpStoreAdapter(EN_IMPL_HW_TBE, op_store_adapter) != SUCCESS) {
    FE_LOGW("[GraphOpt][Prepare][GetOpStoreAdapter] Failed to get op_store_adapter, causing graph [%s] generalization failure.",
            graph_name.c_str());
    return FAILED;
  }
  FE_CHECK_NOTNULL(op_store_adapter);

  // traversing graph, limit type notarize, get input_nodes/limited_nodes/input_nodes of every node
  ret = GraphPreprocessing(*new_graph_bak, op_store_adapter);
  if (ret != SUCCESS) {
    FE_LOGW("[GraphOpt][Prepare][GraphPreprocessing] Preprocessing failed: graph [%s] generalization failed.",
            graph_name.c_str());
    return FAILED;
  }
  FE_LOGD("[GraphOpt][Prepare][GeneralizeGraph] Preprocessing graph[%s] successfully.", graph_bak_name.c_str());

  if (!is_need_generalize_graph_) {
    FE_LOGD("[GraphOpt][Prepare][GeneralizeGraph] Graph[%s] no need to generalize.", graph_bak_name.c_str());
    return SUCCESS;
  }

  // initialize range_decent infos for input nodes
  ret = InitOriginalGraphInfos(graph);
  if (ret != SUCCESS) {
    FE_LOGW(
        "[GraphOpt][Prepare][InitOriginalGraphInfos] Graph[%s]: fail to initialize range_decent infos.",
        graph_name.c_str());
    return FAILED;
  }
  FE_LOGD("[GraphOpt][Prepare][GeneralizeGraph] Init range_decent infos with graph[%s] successfully.",
          graph_bak_name.c_str());

  FusionAttrManagerPtr fusion_attr_mgr = nullptr;
  FE_MAKE_SHARED(fusion_attr_mgr = std::make_shared<FusionAttrManager>(fusion_priority_mgr_ptr_),
                 return GRAPH_OPTIMIZER_MAKE_SHARED_FAILED);
  GraphType graph_type = {is_range_limited_graph_, is_single_op_graph_};
  // all input_nodes generalize
  InputNodeGeneralize input_node_generalize(external_input_nodes_, graph_type, node_info_map_,
                                            op_store_adapter, fusion_attr_mgr);
  ret = input_node_generalize.GeneralizeAllInputNodesInGraph();
  if (ret != SUCCESS) {
    FE_LOGW("[GraphOpt][Prepare][GeneralizeGraph] Failed to generalize input nodes, graph [%s] generalization failed.",
            graph_name.c_str());
    return FAILED;
  }
  FE_LOGD("[GraphOpt][Prepare][GeneralizeGraph] Successfully generalized all input nodes in graph [%s].",
          graph_bak_name.c_str());

  if (!is_range_limited_graph_) {
    FE_LOGD("[GraphOpt][Prepare][GeneralizeGraph] Graph[%s]: all nodes are range_unlimited.", graph_bak_name.c_str());
    if (UpdateDynamicShapeToOriginalGraph(graph) != SUCCESS) {
      FE_LOGW(
          "[GraphOpt][Prepare][GraphDynamicShapeInfer] Failed to update dynamic shape to original shape for Graph[%s].",
          graph_name.c_str());
      return FAILED;
    }
  } else {
    // dynamic shape limited graph infer shape, include limited_nodes check
    ret = GraphDynamicShapeInfer(op_store_adapter, new_graph_bak, tmp_graph_ptr);
    if (ret != SUCCESS) {
      FE_LOGW(
          "[GraphOpt][Prepare][GraphDynamicShapeInfer] Failed to perform dynamic shape inference, graph [%s] generalization failed.",
          graph_name.c_str());
      return FAILED;
    }
    FE_LOGD("[GraphOpt][Prepare][GeneralizeGraph] graph[%s] dynamic shape infer successfully.",
            graph_bak_name.c_str());
  }
  FE_LOGD("[GraphOpt][Prepare][GeneralizeGraph] graph[%s] generalize successfully.",
          graph_name.c_str());
  return SUCCESS;
}

Status FuzzyGeneralize::CalDecentSteps(const ge::NodePtr &external_node, const ge::OpDescPtr &opdesc) {
  auto output_desc = external_node->GetOpDesc()->MutableOutputDesc(0);
  const std::string op_name = external_node->GetName();
  FE_LOGD("[GraphOpt][Prepare][CalDecentSteps] FeedDecentSteps: %s.", op_name.c_str());

  std::vector<std::pair<int64_t, int64_t>> shape_range;
  FE_CHECK_NOTNULL(output_desc);
  (void)output_desc->GetShapeRange(shape_range);

  double step_temp;
  for (size_t idx = 0; idx < shape_range.size(); ++idx) {
    ge::GeTensorDescPtr ori_output_desc = opdesc->MutableOutputDesc(0);
    FE_CHECK_NOTNULL_WARNLOG(ori_output_desc);
    int64_t shape_dim = ori_output_desc->GetShape().GetDim(idx);
    FE_LOGD("[GraphOpt][Prepare][CalDecentSteps] Tensor_idx: %lu, shape_dim: %ld, ori_shape_dim: %ld.", idx, shape_dim,
            output_desc->GetShape().GetDim(0));
    if (shape_range[idx].second == MAX_RANGE_UPPER) {
      step_temp = static_cast<double>(shape_dim) / static_cast<double>(TOTAL_DECENT_TIMES - 1);
    } else {
      step_temp = static_cast<double>(shape_range[idx].second - shape_dim) / static_cast<double>(TOTAL_DECENT_TIMES);
    }
    FE_LOGD("[GraphOpt][Prepare][CalDecentSteps] index: %zu, step_temp: %lf.", idx, step_temp);
    decent_steps_[op_name].emplace_back(step_temp);
  }

  return SUCCESS;
}

Status FuzzyGeneralize::RangeDecent(const ge::NodePtr &external_node, uint32_t &decent_times) {
  auto op_desc = external_node->GetOpDesc();
  const std::string op_name = external_node->GetName();
  FE_LOGD("[GraphOpt][Prepare][RangeDecent] RangeDecent %s.", op_name.c_str());
  std::vector<std::pair<int64_t, int64_t>> range;

  std::map<std::string, ge::NodePtr>::const_iterator iter_input = original_input_nodes_.find(op_name);
  if (iter_input == original_input_nodes_.end()) {
    FE_LOGW("Cannot find the node [%s] in original_input_nodes_; generalization failed.", op_name.c_str());
    return FAILED;
  }
  const ge::OpDescPtr ori_op_desc = iter_input->second->GetOpDesc();

  if (decent_times == 1) {  // The first time to decend shape-range, we need calculate decent-steps.
    if (CalDecentSteps(external_node, ori_op_desc) != SUCCESS) {
      FE_LOGW("[GraphOpt][Prepare][RangeDecent] Node [%s] failed in CalDecentSteps.", op_name.c_str());
      return FAILED;
    }
  }

  ge::GeTensorDescPtr input_tensor_desc = op_desc->MutableInputDesc(0);
  ge::GeTensorDescPtr output_tensor_desc = op_desc->MutableOutputDesc(0);
  FE_CHECK_NOTNULL(input_tensor_desc);
  input_tensor_desc->GetOriginShapeRange(range);
  FE_LOGD("[GraphOpt][Prepare][RangeDecent] Original range is %s.", RangeToString(range).c_str());
  for (size_t idx = 0; idx < range.size(); ++idx) {
    int64_t range_high = range[idx].second;
    FE_CHECK_NOTNULL(ori_op_desc->MutableInputDesc(0));
    int64_t dim_num = ori_op_desc->MutableInputDesc(0)->GetShape().GetDim(idx);
    if (range_high == MAX_RANGE_UPPER) {
      range[idx].second = dim_num * kExpandNum;
      continue;
    }
    double step = decent_steps_[op_name][idx];
    range_high = static_cast<int64_t>(ceil(range[idx].second - step));
    FE_LOGW("[GraphOpt][Prepare][RangeDecent] Get index %zu, step: %lf, %ld.", idx, step, range_high);
    if (range_high <= dim_num) {
      range_high = dim_num + 1;
    }
    range[idx].second = range_high;
  }
  input_tensor_desc->SetShapeRange(range);
  input_tensor_desc->SetOriginShapeRange(range);
  FE_CHECK_NOTNULL(output_tensor_desc);
  output_tensor_desc->SetShapeRange(range);
  output_tensor_desc->SetOriginShapeRange(range);
  FE_LOGD("[GraphOpt][Prepare][RangeDecent] decent range is %s.", RangeToString(range).c_str());

  ++decent_times;
  return SUCCESS;
}

void FuzzyGeneralize::CheckIsSubGraphNode(const ge::NodePtr &node_ptr, const NodeGeneralInfoPtr &node_info_ptr) const {
  if (node_ptr->GetOwnerComputeGraph()->GetParentGraph() != nullptr) {
    node_info_ptr->is_sub_graph_node = true;
  }
}

bool FuzzyGeneralize::CheckIsExternalNode(const ge::NodePtr &node) const {
  if (node->GetInAllNodes().empty() && (node->GetOwnerComputeGraph()->GetParentGraph() == nullptr)) {
    return true;
  }
  FE_LOGD("[GraphOpt][Prepare][CheckIsExternalNode] Current node[%s] is not external-input node.",
          node->GetName().c_str());
  return false;
}

Status FuzzyGeneralize::FeedInputsRootSet(const ge::NodePtr &node_ptr, const NodeGeneralInfoPtr &node_info_ptr) const {
  for (const ge::InDataAnchorPtr &input_data_anchor : node_ptr->GetAllInDataAnchors()) {
    if (input_data_anchor == nullptr) {
      FE_LOGD("[GraphOpt][Prepare][FeedInputsRootSet] The input_data_anchor of node[%s] is nullptr.",
              node_ptr->GetName().c_str());
      continue;
    }
    ge::OutDataAnchorPtr peer_node_out_anchor = input_data_anchor->GetPeerOutAnchor();
    if (peer_node_out_anchor == nullptr) {
      FE_LOGD("[GraphOpt][Prepare][FeedInputsRootSet] The peer_node_out_anchor of node [%s] is nullptr.",
              node_ptr->GetName().c_str());
      continue;
    }

    ge::NodePtr peer_node = peer_node_out_anchor->GetOwnerNode();
    if (peer_node == nullptr) {
      FE_LOGD("[GraphOpt][Prepare][FeedInputsRootSet] Peer output node of node[%s] is nullptr.",
              node_ptr->GetName().c_str());
      continue;
    }
    auto iter = node_info_map_.find(peer_node);
    if (iter == node_info_map_.end()) {
      FE_LOGD("[GraphOpt][Prepare][FeedInputsRootSet] Could not find node [%s] in node_info_map.",
              peer_node->GetName().c_str());
      continue;
    }
    NodeGeneralInfoPtr peer_node_info = iter->second;
    if (peer_node_info == nullptr) {
      FE_LOGW("[GraphOpt][Prepare][FeedInputsRootSet] The GeneralInfo of node[%s] is a nullptr.",
              peer_node->GetName().c_str());
      return FAILED;
    }

    for (auto &i : peer_node_info->disjoint_root_set) {
      node_info_ptr->disjoint_root_set.emplace(i);
    }

    if (node_ptr->GetOpDesc() == nullptr) {
      FE_LOGD("[GraphOpt][Prepare][FeedInputsRootSet] The opdesc of node [%s] is a nullptr.", peer_node->GetName().c_str());
      continue;
    }

    ge::GeTensorDescPtr cur_input = node_ptr->GetOpDesc()->MutableInputDesc(input_data_anchor->GetIdx());
    if (cur_input == nullptr) {
      FE_LOGD("[GraphOpt][Prepare][FeedInputsRootSet] Null pointer encountered! Node: [%s].", peer_node->GetName().c_str());
      continue;
    }
    node_info_ptr->inputs_root_map.insert(std::make_pair(cur_input, peer_node_info->disjoint_root_set));
  }
  return SUCCESS;
}

Status FuzzyGeneralize::GetCurNodeInfo(const ge::NodePtr &node,
                                       const OpStoreAdapterPtr &op_store_adapter,
                                       NodeGeneralInfoPtr &node_info_ptr) {
  FE_LOGD("[GraphOpt][Prepare][GetCurNodeInfo] Begin to feed node_info for current node[%s].", node->GetName().c_str());
  CheckIsSubGraphNode(node, node_info_ptr);
  Status ret = op_store_adapter->FeedNodeGeneralInfoFromOpStore(node, node_info_ptr);
  if (ret != SUCCESS) {
    FE_LOGD("[GraphOpt][Prepare][GetCurNodeInfo] Node[%s]: failed to feed general info by opstore.",
            node->GetName().c_str());
    return FAILED;
  }
  FE_LOGD("[GraphOpt][Prepare][GetCurNodeInfo] Node[%s]: feed general info by opstore successfully.",
          node->GetName().c_str());

  if (FeedInputsRootSet(node, node_info_ptr) != SUCCESS) {
    FE_LOGW("[GraphOpt][Prepare][GetCurNodeInfo] Feed input root set for node[%s] failed.", node->GetName().c_str());
    return FAILED;
  }
  node_info_map_.insert(std::make_pair(node, node_info_ptr));
  return SUCCESS;
}

Status FuzzyGeneralize::GetRangeLimitValue(const OpStoreAdapterPtr &op_store_adapter,
                                           const NodeGeneralInfoPtr &node_info_ptr,
                                           const ge::NodePtr &node) {
  Status ret = op_store_adapter->GetRangeLimit(node_info_ptr, node);
  if (ret == SUCCESS && node_info_ptr->is_limited_range) {
    limited_range_nodes_.emplace_back(node);
    FE_LOGD("[GraphOpt][Prepare][GetRangeLimitValue] Current node[%s] is range_limited.", node->GetName().c_str());
  }
  return SUCCESS;
}

Status FuzzyGeneralize::GraphPreprocessing(const ge::ComputeGraph &graph,
                                           const OpStoreAdapterPtr &op_store_adapter) {
  node_info_map_.clear();
  external_input_nodes_.clear();
  limited_range_nodes_.clear();
  FE_LOGD("[GraphOpt][Prepare][GraphPreprocessing] Begin pre-processing graph [%s].", graph.GetName().c_str());
  for (auto &node_ptr : graph.GetAllNodes()) {
    NodeGeneralInfo node_info;
    NodeGeneralInfoPtr node_info_ptr = nullptr;
    node_ptr->GetOpDesc()->DelAttr(ATTR_NAME_UNKNOWN_SHAPE);
    FE_MAKE_SHARED(node_info_ptr = std::make_shared<NodeGeneralInfo>(node_info), return FAILED);
    FE_LOGD("[GraphOpt][Prepare][GraphPreprocessing] Ready to process current node[%s].",
            node_ptr->GetName().c_str());

    if (CheckIsExternalNode(node_ptr)) {
      node_info_ptr->disjoint_root_set.emplace(node_ptr);
      if (!CheckOpConstOrVariableInOriGraph(node_ptr->GetOpDesc())) {
        external_input_nodes_.emplace(node_ptr);
      }
      node_info_map_.insert(std::make_pair(node_ptr, node_info_ptr));
      FE_LOGD("[GraphOpt][Prepare][GraphPreprocessing] Current node[%s] is external input node.",
              node_ptr->GetName().c_str());
      continue;
    }

    if (GetCurNodeInfo(node_ptr, op_store_adapter, node_info_ptr) != SUCCESS) {
      FE_LOGW(
          "[GraphOpt][Prepare][GraphPreprocessing] Failed to get current node [%s] info for graph [%s].",
          node_ptr->GetName().c_str(), graph.GetName().c_str());
      return FAILED;
    }

    if (node_info_ptr->is_found_in_opstore && !node_info_ptr->is_support_dynamic_shape) {
      is_need_generalize_graph_ = false;
      FE_LOGD("[GraphOpt][Prepare][GraphPreprocessing] Graph[%s] does not need to be generalized.", graph.GetName().c_str());
      return SUCCESS;
    }

    FE_LOGD("[GraphOpt][Prepare][GraphPreprocessing] Get current node[%s] info successfully.",
            node_ptr->GetName().c_str());
    if (GetRangeLimitValue(op_store_adapter, node_info_ptr, node_ptr) != SUCCESS) {
      FE_LOGW("[GraphOpt][Prepare][GraphPreprocessing] Failed to get range_limit value, node[%s].",
              node_ptr->GetName().c_str());
      return FAILED;
    }

    FE_LOGD("[GraphOpt][Prepare][GraphPreprocessing] Get range limit value successfully, node[%s].",
            node_ptr->GetName().c_str());
    if (!is_range_limited_graph_ && node_info_ptr->is_limited_range) {
      is_range_limited_graph_ = true;
      FE_LOGD("[GraphOpt][Prepare][GraphPreprocessing] Graph [%s] is a range-limited graph.",
              graph.GetName().c_str());
    }
  }
  return SUCCESS;
}
}  // namespace fe
