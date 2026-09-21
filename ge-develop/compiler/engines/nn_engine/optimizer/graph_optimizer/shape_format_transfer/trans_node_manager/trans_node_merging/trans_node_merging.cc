/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph_optimizer/shape_format_transfer/trans_node_manager/trans_node_merging/trans_node_merging.h"
#include <memory>
#include <stack>
#include <string>
#include <set>
#include "common/fe_utils.h"
#include "common/fe_type_utils.h"
#include "common/platform_utils.h"
#include "graph_optimizer/shape_format_transfer/trans_node_manager/trans_node_manager.h"
#include "register/graph_optimizer/fusion_common/unknown_shape_utils.h"

namespace fe {
namespace {
const std::vector<int64_t> kUnknownDimNum = {-2};
const std::string kStageMrgOneNd = "[GraphOptJdgInst][ShapeTrans][MrgOneNd]";
const std::set<std::string> kTransOpTypeSet = {TRANSDATA, CAST, TRANSPOSE, TRANSPOSED, RESHAPE, REFORMAT};
const std::set<std::string> kTransposeTypeSet = {TRANSPOSE, TRANSPOSED};

bool IsTransOpType(const string &op_type) {
  return kTransOpTypeSet.count(op_type) != 0;
}

bool IsTransposeType(const string &op_type) {
  return kTransposeTypeSet.count(op_type) != 0;
}

bool CheckAxisC(ge::DataType original_data_type, int64_t multiply_result1, int64_t multiply_result2, int64_t dims1,
                int64_t dims2) {
  int64_t c0 = GetC0ValByDataType(original_data_type);

  /* If size_of_dims1 == axis_c_unsigned, this branch will not be reached.
   * This means if thr origianl_format is not valid, GetAxisIndexByFormat
   * will return -1, and we just check the product of all elements. */
  if (multiply_result1 != multiply_result2) {
    return false;
  } else {
    if (dims1 != dims2 && ((dims1 % c0) || (dims2 % c0))) {
      /* If axis C is not equal, if any of the dims cannot be divided by
       * c0, return false; */
      return false;
    }
  }
  return true;
}

/* Equivalent means:
 * 1. Two dims are completely the same.
 * 2. Two dims are same in memory. One may be the reshape version
 * of the other. For example:
 * dims1  8, 9, 28, 28, 32
 * dims2  72, 1, 28, 28, 32
 * Same in memory means:
 * 1. Dims size must be the same.
 * 2. Product of two dims are the same.
 * 3. Each dimension of dims and dims can be divided by each other, either
 * dims1[i] mod dims2[i] is equal to 0 or
 * dims2[i] mod dims1[i] is equal to 0.
 *
 *
 * For NCHW, NHWC and HWCN need to ADDTIONAL check the product of all axis at
 * the left of C and C itself is the same and the axis c can be divided by
 * C0. If the axis C is the same for both dims, we will not check this.
 * For example:
 * For NCHW, we need to check whether the product the dim N and dim C is the
 * same.
 * For NHWC, we just check all.
 * For HWCN, we need to addtionally check the product of H * W * C. */
bool CheckTwoDimsEquivalent(const std::vector<int64_t> &dims1, const std::vector<int64_t> &dims2,
                            ge::Format original_format, ge::DataType original_data_type, string op_type) {
  if (dims1 == kUnknownDimNum || dims2 == kUnknownDimNum) {
    return false;
  }
  if (dims1 == dims2) {
    return true;
  } else if (op_type == TRANSDATA) {
    FE_LOGD("Check two Transdata equivalents. Dims1 = %s, dims2 = %s.", StringUtils::IntegerVecToString(dims1).c_str(),
            StringUtils::IntegerVecToString(dims2).c_str());
    FE_LOGD("Original format is %u and original dtype is %u.", original_format, original_data_type);
    size_t size_of_dims1 = dims1.size();
    size_t size_of_dims2 = dims2.size();
    size_t axis_c_unsigned;
    int64_t axis_c = GetAxisIndexByFormat(original_format, C_AXIS_NAME);
    if (axis_c >= static_cast<int64_t>(size_of_dims1) || axis_c < 0) {
      axis_c_unsigned = size_of_dims1;
    } else {
      axis_c_unsigned = static_cast<size_t>(axis_c);
    }
    if (size_of_dims1 != size_of_dims2) {
      return false;
    }
    int64_t multiply_result1 = 1;
    int64_t multiply_result2 = 1;
    for (size_t i = 0; i < size_of_dims1; i++) {
      if (dims1[i] == 0 || dims2[i] == 0) {
        /* if one tensor contains a dim 0, and two tensor's shape are not the
         * same, for simplicity, we just consider they are not the same tensor.
         * But they might be the same, for example:
         * {2,0,3,4,6} and {0,7,8,9,5} is the same. */
        return false;
      }
      if (ge::MulOverflow(multiply_result1, dims1[i], multiply_result1)) {
        FE_LOGW("Int64 %ld and %ld multiplication overflow!", multiply_result1, dims1[i]);
        return false;
      }
      if (ge::MulOverflow(multiply_result2, dims2[i], multiply_result2)) {
        FE_LOGW("Int64 %ld and %ld multiplication overflow!", multiply_result2, dims2[i]);
        return false;
      }
      if (multiply_result1 % multiply_result2 != 0 && multiply_result2 % multiply_result1 != 0) {
        return false;
      }
      if ((i == axis_c_unsigned) &&
          !CheckAxisC(original_data_type, multiply_result1, multiply_result2, dims1[i], dims2[i])) {
        return false;
      }
    }
    FE_LOGD("%ld %ld", multiply_result1, multiply_result2);
    return multiply_result1 == multiply_result2;
  }
  return false;
}

bool GetPeerConstNodeData(const ge::NodePtr &node, std::vector<int64_t>& perm_vec) {
  if (node->GetType() == TRANSPOSED) {
    ge::OpDescPtr trans_op_desc = node->GetOpDesc();
    FE_CHECK_NOTNULL(trans_op_desc);
    (void)ge::AttrUtils::GetListInt(trans_op_desc, PERM, perm_vec);
    return true;
  }
  ge::InDataAnchorPtr perm_in_anchor = node->GetInDataAnchor(kTransposeInputPerm);
  bool perm_node_nullflag = (perm_in_anchor == nullptr || perm_in_anchor->GetPeerOutAnchor() == nullptr ||
                             perm_in_anchor->GetPeerOutAnchor()->GetOwnerNode() == nullptr);
  if (perm_node_nullflag) {
    FE_LOGW("InAnchor or its predecessor of Node (name [%s]) is null!", node->GetNamePtr());
    return false;
  }

  auto const_node = perm_in_anchor->GetPeerOutAnchor()->GetOwnerNode();
  auto weight_list = ge::OpDescUtils::GetWeights(const_node);
  if (weight_list.empty() || weight_list[0] == nullptr) {
    FE_LOGW("Get weight of Node (name [%s]) is empty!", node->GetNamePtr());
    return false;
  }

  ge::ConstGeTensorPtr perm_tensor = weight_list[0];
  const auto &tensor_desc = perm_tensor->GetTensorDesc();
  std::size_t data_size = perm_tensor->GetData().size();
  if (tensor_desc.GetDataType() == ge::DT_INT32) {
    const auto *perm_data = reinterpret_cast<const int32_t *>(perm_tensor->GetData().data());
    for (std::size_t i = 0; i * sizeof(int32_t) < data_size; ++i) {
      perm_vec.emplace_back(static_cast<int64_t>(perm_data[i]));
    }
  } else if (tensor_desc.GetDataType() == ge::DT_INT64) {
    const auto *perm_data = reinterpret_cast<const int64_t *>(perm_tensor->GetData().data());
    for (std::size_t i = 0; i * sizeof(int64_t) < data_size; ++i) {
      perm_vec.emplace_back(perm_data[i]);
    }
  }
  return true;
}

bool CheckTwoTransposePermAttrValid(const ge::NodePtr &src_node, const ge::NodePtr &dst_node,
                                    const ge::OpDescPtr &src_op_desc, const ge::OpDescPtr &dst_op_desc) {
  std::vector<int64_t> src_perm_vec;
  std::vector<int64_t> dst_perm_vec;
  if (!GetPeerConstNodeData(src_node, src_perm_vec) || !GetPeerConstNodeData(dst_node, dst_perm_vec)) {
    return false;
  }

  if (src_perm_vec.size() != dst_perm_vec.size()) {
    FE_LOGD("Perm attr size of transpose op[%s] and [%s] are not same.",
            src_op_desc->GetNamePtr(), dst_op_desc->GetNamePtr());
    return false;
  }
  std::vector<int32_t> merge_perm_vec;
  for (size_t i = 0; i < dst_perm_vec.size(); i++) {
    if (dst_perm_vec[i] < 0 || static_cast<size_t>(dst_perm_vec[i]) >= src_perm_vec.size()) {
      FE_LOGD("Perm value [%d] is invalid because it exceeds the perm vector size [%zu], or it is a negative integer.",
              dst_perm_vec[i], src_perm_vec.size());
      continue;
    }
    merge_perm_vec.push_back(src_perm_vec[static_cast<size_t>(dst_perm_vec[i])]);
  }
  if (merge_perm_vec.size() != src_perm_vec.size()) {
    FE_LOGD("Merged perm vec size [%zu] does not match the current perm vec size [%zu].",
            merge_perm_vec.size(), src_perm_vec.size());
    return false;
  }
  for (size_t i = 0; i < merge_perm_vec.size(); i++) {
    if (merge_perm_vec[i] < 0) {
      FE_LOGD("Merged perm vec is invalid: perm value [%d] is negative.", merge_perm_vec[i]);
      return false;
    }
    if (static_cast<size_t>(merge_perm_vec[i]) != i) {
      FE_LOGD("Merged perm vec is invalid for perm value[%d] is not same with its index[%zu].",
              merge_perm_vec[i], i);
      return false;
    }
  }
  return true;
}

void UpdateAttrNames(ge::OpDescPtr dst_op_desc, ge::OpDescPtr old_src_op_desc, ge::OpDescPtr new_src_op_desc) {
  auto src_name_of_dst_node = dst_op_desc->GetSrcName();
  auto input_name_of_dst_node = dst_op_desc->GetInputName();

  vector<string> new_src_name;
  vector<string> new_input_name;

  if(old_src_op_desc == nullptr || new_src_op_desc == nullptr) {
    FE_LOGD("old_src_op_desc or input_name_of_dst_node is nullptr, no attr update.");
    return;
  }
  auto old_src_name = old_src_op_desc->GetName();
  auto new_name = new_src_op_desc->GetName();
  if (src_name_of_dst_node.size() == 0) {
    new_src_name.push_back(new_name);
  } else {
    for (auto iter = src_name_of_dst_node.begin(); iter != src_name_of_dst_node.end(); iter++) {
      if (*iter != old_src_name) {
        new_src_name.push_back(*iter);
      } else {
        new_src_name.push_back(new_name);
      }
    }
  }

  if (input_name_of_dst_node.size() == 0) {
    new_input_name.push_back(new_name);
  } else {
    for (auto iter = input_name_of_dst_node.begin(); iter != input_name_of_dst_node.end(); iter++) {
      if (*iter != old_src_name) {
        new_input_name.push_back(*iter);
      } else {
        new_input_name.push_back(new_name);
      }
    }
  }

  dst_op_desc->SetSrcName(new_src_name);
  dst_op_desc->SetInputName(new_input_name);
}

Status RemoveTransposePerm(const BasicInfoForRemovingNode &info, ge::ComputeGraph &fused_graph) {
  if (info.node->GetType() == TRANSPOSE) {
    ge::InDataAnchorPtr perm_in_anchor = info.node->GetInDataAnchor(kTransposeInputPerm);
    if(!(perm_in_anchor == nullptr || perm_in_anchor->GetPeerOutAnchor() == nullptr ||
                                perm_in_anchor->GetPeerOutAnchor()->GetOwnerNode() == nullptr)) {
      auto const_node = perm_in_anchor->GetPeerOutAnchor()->GetOwnerNode();
      ge::OutDataAnchorPtr out_anchor_of_node = const_node->GetOutDataAnchor(0);
      FE_CHECK_NOTNULL(out_anchor_of_node);

      if (out_anchor_of_node->GetPeerInDataAnchors().size() == kTransposeOutputSize) {
        if (ge::GraphUtils::RemoveNodeWithoutRelink(fused_graph.shared_from_this(), const_node) != ge::GRAPH_SUCCESS) {
          REPORT_FE_ERROR("[GraphOptJdgInst][ShapeTrans][RmNdEg] Graph[%s]: failed to remove node[%s, %s].",
                          fused_graph.GetName().c_str(), const_node->GetNamePtr(), const_node->GetTypePtr());
          return FAILED;
        }
      } else if (ge::GraphUtils::RemoveEdge(out_anchor_of_node, perm_in_anchor) != ge::GRAPH_SUCCESS) {
        REPORT_FE_ERROR("[GraphOptJdgInst][ShapeTrans][RmNdEg] [1]:Failed to remove edge from [%s] to [%s].",
                        const_node->GetNamePtr(), info.node->GetNamePtr());
        return FAILED;
      }
    }
  }
  return SUCCESS;
}

Status RemoveNodeAndEdges(const BasicInfoForRemovingNode &info, bool father_node_nullflag,
                          std::stack<uint32_t> &in_anchor_index_stack, ge::ComputeGraph &fused_graph) {
  (void)father_node_nullflag;
  /* Third: Remove node itself from graph. */
  /* If we encounter multiple peer in anchors, we cannot remove the node. So
   * we remove the anchor relation between node and it's father, and add a
   * edge between node and it's father's father.
   */
  if (info.dst_in_anchor_size_of_node == 1) {
    if (RemoveTransposePerm(info, fused_graph) != SUCCESS) {
      REPORT_FE_ERROR("[GraphOptJdgInst][ShapeTrans][RmNdEg] Failed to remove perm of transpose op[%s]",
                      info.node->GetNamePtr());
      return FAILED;
    }
    if (ge::GraphUtils::IsolateNode(info.node, {0}) != ge::GRAPH_SUCCESS) {
      FE_CHECK_NOTNULL(info.src_op_desc);
      REPORT_FE_ERROR("[GraphOptJdgInst][ShapeTrans][RmNdEg][3]: Failed to remove edge from [%s] to [%s].",
                      info.src_op_desc->GetNamePtr(), info.node->GetNamePtr());
      return FAILED;
    }
    FE_LOGD("[Trans][Merge] Isolate Node %s.", info.node->GetNamePtr());
    if (ge::GraphUtils::RemoveNodeWithoutRelink(fused_graph.shared_from_this(), info.node) != ge::GRAPH_SUCCESS) {
      REPORT_FE_ERROR("[GraphOptJdgInst][ShapeTrans][RmNdEg] Graph[%s]: failed to remove node[%s, %s].",
                      fused_graph.GetName().c_str(), info.node->GetNamePtr(), info.node->GetTypePtr());
      return FAILED;
    }
    uint32_t new_in_anchor_index = info.src_out_anchor->GetPeerInDataAnchors().size() - 1;
    in_anchor_index_stack.push(new_in_anchor_index);
    FE_LOGD("[Trans][Merge] Remove node (name [%s] type [%s])!", info.node->GetNamePtr(), info.node->GetTypePtr());
  } else {
    // relink data edge
    if (ge::GraphUtils::RemoveEdge(info.out_anchor_of_node, info.dst_in_anchor) != ge::GRAPH_SUCCESS) {
      REPORT_FE_ERROR("[GraphOptJdgInst][ShapeTrans][RmNdEg] [1]:Failed to remove edge from [%s] to [%s].",
                      info.node->GetNamePtr(), info.dst_node->GetNamePtr());
      return FAILED;
    }
    FE_LOGD("[Trans][Merge] Remove edge from [%s] to [%s].", info.node->GetNamePtr(), info.dst_node->GetNamePtr());

    bool result = info.src_out_anchor != nullptr &&
            ge::GraphUtils::AddEdge(info.src_out_anchor, info.dst_in_anchor) != ge::GRAPH_SUCCESS;
    if (result) {
      REPORT_FE_ERROR("[GraphOptJdgInst][ShapeTrans][RmNdEg] [2]:Failed to add edge from [%s] to [%s].",
                      info.src_op_desc->GetNamePtr(), info.dst_node->GetNamePtr());
      return FAILED;
    }

    // relink control edges
    if (info.node->GetInControlAnchor() != nullptr && !info.node->GetInControlAnchor()->IsPeerOutAnchorsEmpty()) {
      for (const ge::OutControlAnchorPtr &out_ctrl_anchor :
           info.node->GetInControlAnchor()->GetPeerOutControlAnchors()) {
        (void)ge::GraphUtils::AddEdge(out_ctrl_anchor, info.dst_node->GetInControlAnchor());
      }
    }
    if (info.node->GetOutControlAnchor() != nullptr &&
        !info.node->GetOutControlAnchor()->GetPeerInControlAnchors().empty()) {
      for (const ge::InControlAnchorPtr &in_ctrl_anchor : info.node->GetOutControlAnchor()->GetPeerInControlAnchors()) {
        (void)ge::GraphUtils::AddEdge(info.src_node->GetOutControlAnchor(), in_ctrl_anchor);
      }
    }

    if (info.src_out_anchor->GetPeerInDataAnchors().empty()) {
      REPORT_FE_ERROR("[GraphOptJdgInst][ShapeTrans][RmNdEg] Anchor %d's peer in data anchor is empty.",
                      info.src_out_anchor->GetIdx());
      return FAILED;
    }
    FE_CHECK_NOTNULL(info.src_op_desc);
    FE_LOGD("[Trans][Merge] Add edge from [%s] to [%s].",
            info.src_op_desc->GetNamePtr(), info.dst_node->GetNamePtr());

    FE_LOGI("We keep node (name [%s] type [%s]) in graph because its output has [%zu] peer input anchors.",
            info.node->GetNamePtr(), info.node->GetTypePtr(), info.dst_in_anchor_size_of_node);
    uint32_t new_in_anchor_index = info.src_out_anchor->GetPeerInDataAnchors().size() - 1;
    in_anchor_index_stack.push(new_in_anchor_index);
  }
  UpdateAttrNames(info.dst_op_desc, info.node->GetOpDesc(), info.src_op_desc);
  return SUCCESS;
}
} // namespace
TransNodeMerging::TransNodeMerging() {}

TransNodeMerging::~TransNodeMerging() {}

bool TransNodeMerging::CheckTwoTransOpDescMergable(const ge::NodePtr &src_node,
                                                   const ge::NodePtr &dst_node,
                                                   const std::vector<ge::NodePtr> &origin_cast_list,
                                                   const bool check_list_flag) const {
  // dst_op_desc -> src_node
  FE_CHECK(src_node == nullptr || dst_node == nullptr, FE_LOGD("Src node or dst node is null."), return false);

  ge::OpDescPtr src_op_desc = src_node->GetOpDesc();
  ge::OpDescPtr dst_op_desc = dst_node->GetOpDesc();
  FE_CHECK(src_op_desc == nullptr, FE_LOGD("Source op desc is null."), return false);
  bool unsupport_merge = (src_op_desc->GetType() == RESHAPE && dst_op_desc->GetType() == RESHAPE) &&
     (UnknownShapeUtils::IsUnknownShapeOp(*src_op_desc) || UnknownShapeUtils::IsUnknownShapeOp(*dst_op_desc));
  if (unsupport_merge) {
    return false;
  }

  string src_op_type = src_op_desc->GetType();
  string dst_op_type = dst_op_desc->GetType();
  // transposed and transpose are processed with the same logic
  bool flag = (!(src_op_type == dst_op_type || (IsTransposeType(src_op_type) && IsTransposeType(dst_op_type))) ||
              !IsTransOpType(src_op_type) ||
              !IsTransOpType(dst_op_type));
  if (flag) {
    return false;
  }

  bool has_dumpable_op = IsDumpableOp(src_op_desc) || IsDumpableOp(dst_op_desc);
  if (has_dumpable_op) {
    return false;
  }

  ge::ConstGeTensorDescPtr src_out_tensor_desc_ptr = src_op_desc->GetOutputDescPtr(0);
  ge::ConstGeTensorDescPtr src_in_tensor_desc_ptr = src_op_desc->GetInputDescPtr(0);
  ge::ConstGeTensorDescPtr dst_in_tensor_desc_ptr = dst_op_desc->GetInputDescPtr(0);
  ge::ConstGeTensorDescPtr dst_out_tensor_desc_ptr = dst_op_desc->GetOutputDescPtr(0);
  FE_CHECK(src_out_tensor_desc_ptr == nullptr || src_in_tensor_desc_ptr == nullptr ||
               dst_in_tensor_desc_ptr == nullptr || dst_out_tensor_desc_ptr == nullptr,
           FE_LOGD("Tensor_desc_ptr is null."), return false);

  if (dst_op_type == CAST) { // can't optimize if not casting between float types and inserted by user
    if (!(IsFloatingCast(src_in_tensor_desc_ptr->GetDataType(), src_out_tensor_desc_ptr->GetDataType()) ||
        IsFloatingCast(dst_in_tensor_desc_ptr->GetDataType(), dst_out_tensor_desc_ptr->GetDataType())) &&
        (!check_list_flag || !IsOptimizableCast(src_node, origin_cast_list) ||
         !IsOptimizableCast(dst_node, origin_cast_list))) {
      FE_LOGD("Keeping nodes [%s] and [%s] in the graph.", src_node->GetName().c_str(), dst_node->GetName().c_str());
      return false;
    }
  }
  
  if (IsTransposeType(dst_op_type) && !CheckTwoTransposePermAttrValid(src_node, dst_node, src_op_desc, dst_op_desc)) {
    return false;
  }

  const ge::GeShape &src_out_shape = src_out_tensor_desc_ptr->GetShape();
  const ge::GeShape &src_in_shape = src_in_tensor_desc_ptr->GetShape();

  const ge::GeShape &dst_in_shape = dst_in_tensor_desc_ptr->GetShape();
  const ge::GeShape &dst_out_shape = dst_out_tensor_desc_ptr->GetShape();

  auto src_in_format = src_in_tensor_desc_ptr->GetFormat();
  auto src_out_format = src_out_tensor_desc_ptr->GetFormat();
  auto dst_in_format = dst_in_tensor_desc_ptr->GetFormat();
  auto dst_out_format = dst_out_tensor_desc_ptr->GetFormat();
  ge::DataType src_in_datatype = src_in_tensor_desc_ptr->GetDataType();
  int64_t src_in_c0 = ge::GetC0Value(src_in_format);
  int64_t src_out_c0 = ge::GetC0Value(src_out_format);
  int64_t dst_in_c0 = ge::GetC0Value(dst_in_format);
  int64_t dst_out_c0 = ge::GetC0Value(dst_out_format);
  bool fp32_has_c08 = src_in_datatype == ge::DT_FLOAT &&
                      (src_in_c0 == SHAPE_NUMBER_8 || src_out_c0 == SHAPE_NUMBER_8 ||
                       dst_in_c0 == SHAPE_NUMBER_8 || dst_out_c0 == SHAPE_NUMBER_8);
  if (fp32_has_c08 && (src_in_c0 != dst_out_c0 || src_out_c0 != dst_in_c0)) {
    return false;
  }
  auto src_in_primary_format = static_cast<ge::Format>(ge::GetPrimaryFormat(src_in_format));
  auto src_out_primary_format = static_cast<ge::Format>(ge::GetPrimaryFormat(src_out_format));
  auto dst_in_primary_format = static_cast<ge::Format>(ge::GetPrimaryFormat(dst_in_format));
  auto dst_out_primary_format = static_cast<ge::Format>(ge::GetPrimaryFormat(dst_out_format));
  if (src_in_primary_format == dst_out_primary_format &&
      src_in_tensor_desc_ptr->GetDataType() == dst_out_tensor_desc_ptr->GetDataType() &&
      CheckTwoDimsEquivalent(src_in_shape.GetDims(), dst_out_shape.GetDims(), dst_out_primary_format,
                             src_in_tensor_desc_ptr->GetDataType(), src_op_type) &&
      src_out_primary_format == dst_in_primary_format &&
      src_out_tensor_desc_ptr->GetDataType() == dst_in_tensor_desc_ptr->GetDataType() &&
      CheckTwoDimsEquivalent(src_out_shape.GetDims(), dst_in_shape.GetDims(), dst_in_primary_format,
                             src_out_tensor_desc_ptr->GetDataType(), src_op_type)) {
    return true;
  } else {
    return false;
  }
}

Status TransNodeMerging::MergeAllTransOps(ge::ComputeGraph &fused_graph,
                                          const std::vector<ge::NodePtr> &origin_cast_list,
                                          const bool check_list_flag) {
  for (ge::NodePtr &node : fused_graph.GetDirectNode()) {
    if (node == nullptr || node->GetOpDesc() == nullptr) {
      continue;
    }

    if (!IsTransOpType(node->GetOpDesc()->GetType())) {
      FE_LOGD("Merge trans op from normal op[%s, %s] backwards.", node->GetNamePtr(), node->GetTypePtr());
      for (auto &in_anchor : node->GetAllInDataAnchors()) {
        if (in_anchor == nullptr || in_anchor->GetPeerOutAnchor() == nullptr ||
            in_anchor->GetPeerOutAnchor()->GetOwnerNode() == nullptr) {
          continue;
        }
        ge::NodePtr src_node = in_anchor->GetPeerOutAnchor()->GetOwnerNode();

        /* Every Time we merge all trans nodes between two non-trans nodes. */
        Status ret = MergeTransOpBetweenTwoNormalOp(fused_graph, src_node, in_anchor, origin_cast_list,
                                                    check_list_flag);
        if (ret == FAILED) {
          FE_LOGD("Merge not successfully. After merging, the graph [%s] is as follows:", fused_graph.GetName().c_str());
          if (IsDebugLogLevel) {
            for (auto &node_tmp : fused_graph.GetDirectNode()) {
              FE_LOGD("Node named [%s]", node_tmp->GetName().c_str());
            }
          }
          return FAILED;
        }
      }
    }
  }
  FE_LOGD("Successfully merged transaction op for graph [%s].", fused_graph.GetName().c_str());
  return SUCCESS;
}

Status TransNodeMerging::MergeOneNode(ge::ComputeGraph &fused_graph, ge::NodePtr node,
                                      const uint32_t &current_in_anchor_index,
                                      std::stack<uint32_t> &in_anchor_index_stack) {
  /* First: Remove edge from source to node, source is predecessor of node.
   * Here TransOp(TransData, Cast, Transpose) will only have one input, so we
   * get in data anchor 0. */
  FE_LOGD("Try to remove node[%s].", node->GetNamePtr());
  ge::InDataAnchorPtr in_anchor_of_node = node->GetInDataAnchor(0);
  ge::OutDataAnchorPtr src_out_anchor = nullptr;
  ge::NodePtr src_node = nullptr;
  ge::OpDescPtr src_op_desc = nullptr;

  bool father_node_nullflag = (in_anchor_of_node == nullptr || in_anchor_of_node->GetPeerOutAnchor() == nullptr ||
                               in_anchor_of_node->GetPeerOutAnchor()->GetOwnerNode() == nullptr ||
                               in_anchor_of_node->GetPeerOutAnchor()->GetOwnerNode()->GetOpDesc() == nullptr);
  if (father_node_nullflag) {
    FE_LOGW("InAnchor or its predecessor of Node (name [%s]) in subGraph (name [%s]) is null!",
            node->GetNamePtr(), fused_graph.GetName().c_str());
  } else {
    src_out_anchor = in_anchor_of_node->GetPeerOutAnchor();
    src_node = src_out_anchor->GetOwnerNode();
    src_op_desc = src_node->GetOpDesc();
  }
  /* Second: Remove edges from node to all its successors. */
  ge::OutDataAnchorPtr out_anchor_of_node = node->GetOutDataAnchor(0);
  bool out_anchor_null_flag = (out_anchor_of_node == nullptr || out_anchor_of_node->GetPeerInDataAnchors().empty());
  if (out_anchor_null_flag) {
    FE_LOGW("outAnchorOfNode or its successor node is null! Failed to remove node [%s] from graph %s!",
            node->GetNamePtr(), fused_graph.GetName().c_str());
  } else {
    auto dst_in_anchors = out_anchor_of_node->GetPeerInDataAnchors();
    /* Default dst_anchor is anchor 0, we use it to get the op type */
    if (current_in_anchor_index > dst_in_anchors.size()) {
      REPORT_FE_ERROR(
          "[GraphOptJdgInst][ShapeTrans][MrgOneNd] Anchor index %u exceeds the size of destination anchors [%zu].",
          current_in_anchor_index, dst_in_anchors.size());
      return FAILED;
    }

    ge::InDataAnchorPtr dst_in_anchor = dst_in_anchors.at(current_in_anchor_index);
    bool dst_anchor_null_flag = (dst_in_anchor == nullptr || dst_in_anchor->GetOwnerNode() == nullptr);
    if (dst_anchor_null_flag) {
      REPORT_FE_ERROR("[GraphOptJdgInst][ShapeTrans][MrgOneNd] The dstAnchor or its successor node is null!");
      REPORT_FE_ERROR("[GraphOptJdgInst][ShapeTrans][MrgOneNd] Failed to remove node [%s, %s] from graph [%s]!",
                      fused_graph.GetName().c_str(), node->GetNamePtr(), node->GetTypePtr());
      return FAILED;
    } else {
      auto dst_node = dst_in_anchor->GetOwnerNode();
      auto dst_op_desc = dst_node->GetOpDesc();
      auto dst_in_anchor_size_of_node = dst_in_anchors.size();
      if (current_in_anchor_index >= dst_in_anchor_size_of_node) {
        REPORT_FE_ERROR("%s CurNode[%s], DstNode[%s]: index %u exceeds the size of dst_in_anchors [%zu].",
                        kStageMrgOneNd.c_str(), node->GetNamePtr(), dst_node->GetNamePtr(),
                        current_in_anchor_index, dst_in_anchor_size_of_node);
        return FAILED;
      }
      BasicInfoForRemovingNode info = {
          src_node,   src_op_desc, dst_node,          dst_op_desc,        node,
          dst_in_anchor, src_out_anchor, in_anchor_of_node, out_anchor_of_node, dst_in_anchor_size_of_node};
      if (RemoveNodeAndEdges(info, father_node_nullflag, in_anchor_index_stack, fused_graph) != SUCCESS) {
        return FAILED;
      }
    }
  }
  return SUCCESS;
}

uint32_t TransNodeMerging::FindIndexOfCurrentNode(const Vistor<ge::InDataAnchorPtr> &in_data_anchor_ptr_vec,
                                                  const ge::InDataAnchorPtr &in_anchor) const {
  for (uint32_t i = 0; i < in_data_anchor_ptr_vec.size(); i++) {
    if (in_data_anchor_ptr_vec.at(i) == in_anchor) {
      return i;
    }
  }
  return 0xffffffff;
}

Status TransNodeMerging::MergeTransOpBetweenTwoNormalOp(ge::ComputeGraph &fused_graph, ge::NodePtr src_node,
                                                        const ge::InDataAnchorPtr &in_anchor,
                                                        const std::vector<ge::NodePtr> &origin_cast_list,
                                                        const bool check_list_flag) {
  /* We use stack to store all TransNodes between two normal Nodes. */
  std::stack<ge::NodePtr> trans_node_stack;
  std::stack<uint32_t> in_anchor_index_stack;
  FE_CHECK_NOTNULL(src_node->GetOutDataAnchor(0));
  in_anchor_index_stack.push(FindIndexOfCurrentNode(src_node->GetOutDataAnchor(0)->GetPeerInDataAnchors(), in_anchor));
  uint32_t loop_count = 0;
  while (IsTransOpType(src_node->GetType()) && loop_count < 100) {
    loop_count++; /* Use loop count to prevent infinite loops */
                  /* Check stack empty first and then use its top() function.
                   * Otherwise, the program will get a seg-fault. */
    if (!trans_node_stack.empty() &&
        CheckTwoTransOpDescMergable(trans_node_stack.top(), src_node, origin_cast_list, check_list_flag)) {
      ge::NodePtr node_after_src_node = trans_node_stack.top();
      ge::InDataAnchorPtr in_anchor_of_src_node = src_node->GetInDataAnchor(0);
      ge::NodePtr node_before_src_node = nullptr;
      if (in_anchor_of_src_node == nullptr || in_anchor_of_src_node->GetPeerOutAnchor() == nullptr) {
        node_before_src_node = nullptr;
      } else {
        node_before_src_node = in_anchor_of_src_node->GetPeerOutAnchor()->GetOwnerNode();
      }
      /* Remove two nodes and correct relationships of edges. */
      in_anchor_index_stack.pop();

      uint32_t in_anchor_index_of_node_after_src_node = in_anchor_index_stack.top();

      in_anchor_index_stack.pop();

      if (MergeOneNode(fused_graph, node_after_src_node, in_anchor_index_of_node_after_src_node,
                       in_anchor_index_stack) != SUCCESS) {
        return FAILED;
      }
      in_anchor_index_of_node_after_src_node = in_anchor_index_stack.top();

      in_anchor_index_stack.pop();

      if (MergeOneNode(fused_graph, src_node, in_anchor_index_of_node_after_src_node, in_anchor_index_stack) !=
          SUCCESS) {
        return FAILED;
      }
      trans_node_stack.pop();

      if (node_before_src_node == nullptr) {
        FE_LOGD("We merged two nodes [%s, %s] between the source [nullptr] and the destination node.",
                src_node->GetNamePtr(), node_after_src_node->GetNamePtr());
        return MERGE_TRANS_OP_NO_MORE_PREDECESSOR;
      } else {
        FE_LOGD("We merged two nodes [%s, %s] between the source [%s] and the destination node.",
                src_node->GetNamePtr(), node_after_src_node->GetNamePtr(), node_before_src_node->GetNamePtr());
        src_node = node_before_src_node;
      }
    } else {
      trans_node_stack.push(src_node);
      /* TransOp only have one input. */
      ge::InDataAnchorPtr in_anchor_of_src_node = src_node->GetInDataAnchor(0);
      bool flag = (in_anchor_of_src_node == nullptr || in_anchor_of_src_node->GetPeerOutAnchor() == nullptr ||
                  in_anchor_of_src_node->GetPeerOutAnchor()->GetOwnerNode() == nullptr);
      if (flag) {
        return MERGE_TRANS_OP_NO_MORE_PREDECESSOR;
      }

      src_node = in_anchor_of_src_node->GetPeerOutAnchor()->GetOwnerNode();
      uint32_t in_anchor_index =
          FindIndexOfCurrentNode(src_node->GetOutDataAnchor(0)->GetPeerInDataAnchors(), in_anchor_of_src_node);
      in_anchor_index_stack.push(in_anchor_index);
    }
  }
  return SUCCESS;
}

bool TransNodeMerging::IsOptimizableCast(const ge::NodePtr &node,
                                         const std::vector<ge::NodePtr> &origin_cast_list) const {
  return (find(origin_cast_list.begin(), origin_cast_list.end(), node) == origin_cast_list.end());
}

bool TransNodeMerging::IsFloatingCast(const ge::DataType &in_data_type,
                                      const ge::DataType &out_data_type) const {
  return IsFloatingType(in_data_type) && IsFloatingType(out_data_type);
}

bool TransNodeMerging::IsFloatingType(const ge::DataType &data_type) const {
  return data_type == ge::DT_FLOAT || data_type == ge::DT_FLOAT16 || data_type == ge::DT_DOUBLE ||
         data_type == ge::DT_BF16;
}
}  // namespace fe
