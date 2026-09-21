/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "node_checker_utils.h"
#include "ge_common/ge_api_error_codes.h"
#include "framework/common/debug/ge_log.h"
#include "graph/utils/op_type_utils.h"
#include "graph/utils/node_utils.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/build/memory/block_mem_assigner.h"
#include "common/checker.h"
#include "graph/utils/tensor_utils.h"
#include "graph/utils/type_utils.h"
#include "graph/optimize/mem_layout_conflict_optimize/mem_layout_conflict_util.h"

namespace ge {
namespace {
constexpr size_t KNameMax = 200UL;
const std::string kNoPaddingContinuousInputAttrErrorLog =
    "attr _output_offset_list_for_continuous and _output_offset_for_buffer_fusion can only have one. Moreover, "
    "all input nodes either have or do not have the attr _output_offset_list_for_continuous at the same time";

const std::string kNoPaddingContinuousOutputAttrErrorLog = "all output nodes either have or do not have the attr"
    " _input_offset_list_for_continuous at the same time";

void GetDimIndexAttr(const Node *const node, std::stringstream &ss) {
  int64_t attr_dim_index;
  const bool get_attr_dim_flag =
      ge::AttrUtils::GetInt(node->GetOpDescBarePtr(), ATTR_NAME_REUSE_INPUT_ON_DIM_INDEX, attr_dim_index);
  if (get_attr_dim_flag) {
    ss << ", _reuse_input_on_dim_index: " << attr_dim_index;
  }
}

void GetFirstOutputShape(const Node *const node, std::stringstream &ss) {
  if (node->GetOpDescBarePtr()->GetOutputsSize() > 0U) {
    const auto out_desc = node->GetOpDescBarePtr()->GetOutputDesc(0);
    int64_t size = -1;
    (void)TensorUtils::GetSize(out_desc, size);
    ss << ", output0 shape: [" << out_desc.GetShape().ToString() << "], "
       << TypeUtils::DataTypeToSerialString(out_desc.GetDataType()) << ", size: " << size;
  }
}

void GetFirstInputShape(const Node *const node, std::stringstream &ss) {
  if (node->GetOpDescBarePtr()->GetInputsSize() > 0U) {
    const auto input_desc = node->GetOpDescBarePtr()->GetInputDesc(0);
    int64_t size = -1;
    (void)TensorUtils::GetSize(input_desc, size);
    ss << ", input0 shape: [" << input_desc.GetShape().ToString() << "], "
       << TypeUtils::DataTypeToSerialString(input_desc.GetDataType()) << ", size: " << size;
  }
}

std::string GetAttrsForNoPaddingContinuousInput(const Node *const peer_node, const int32_t out_index) {
  std::stringstream ss;
  std::vector<int64_t> offsets_for_fusion = {};
  const bool has_lx_fusion_attr = AttrUtils::GetListInt(peer_node->GetOpDescBarePtr(),
                                                        ATTR_NAME_OUTPUT_OFFSET_FOR_BUFFER_FUSION, offsets_for_fusion);
  if (has_lx_fusion_attr) {
    GE_ASSERT_TRUE(static_cast<size_t>(out_index) < offsets_for_fusion.size(),
                   "out_index: %d, size: %zu, peer_node name: %s", out_index, offsets_for_fusion.size(),
                   peer_node->GetNamePtr());
    ss << ", _output_offset_for_buffer_fusion: " << offsets_for_fusion.at(out_index);
  }
  std::vector<int64_t> offset_list = {};
  const bool has_offset_list =
      AttrUtils::GetListInt(peer_node->GetOpDescBarePtr(), ATTR_NAME_OUTPUT_OFFSET_LIST_FOR_CONTINUOUS, offset_list);
  if (has_offset_list && (!offset_list.empty())) {
    GE_ASSERT_TRUE(static_cast<size_t>(out_index) < offset_list.size(), "out_index: %d, size: %zu, peer_node name: %s",
                   out_index, offset_list.size(), peer_node->GetNamePtr());
    ss << ", _output_offset_list_for_continuous: " << offset_list.at(out_index);
  }
  if (ss.str().empty()) {
    ss << ", no attrs";
  }
  return ss.str();
}

std::string GetAttrsForNoPaddingContinuousOutput(const OutDataAnchor *const out_data_anchor) {
  std::stringstream ss;
  const auto node = out_data_anchor->GetOwnerNodeBarePtr();
  for (const auto &in_data_anchor : out_data_anchor->GetPeerInDataAnchorsPtr()) {
    const auto peer_node = in_data_anchor->GetOwnerNodeBarePtr();
    GE_ASSERT_NOTNULL(peer_node);
    std::vector<int64_t> offset_list = {};
    const bool has_offset_list =
        AttrUtils::GetListInt(peer_node->GetOpDescBarePtr(), ATTR_NAME_INPUT_OFFSET_LIST_FOR_CONTINUOUS, offset_list);
    if (has_offset_list && (!offset_list.empty())) {
      GE_ASSERT_TRUE(static_cast<size_t>(in_data_anchor->GetIdx()) < offset_list.size(),
                     "out_index: %d, size: %zu, peer_node name: %s", in_data_anchor->GetIdx(), offset_list.size(),
                     peer_node->GetNamePtr());
      ss << ", _input_offset_list_for_continuous: " << offset_list.at(in_data_anchor->GetIdx());
      break;
    }
  }
  if (out_data_anchor->GetPeerInDataAnchorsPtr().empty()) {
    ss << ", suspend output";
  }
  GetDimIndexAttr(node, ss);

  if (ss.str().empty()) {
    ss << ", no attrs";
  }
  return ss.str();
}

Status CalcSizeWithDimIndex(const OpDesc *op_desc, const GeTensorDesc &output_desc, int64_t &output_mem_size) {
  const GeShape &output_shape = output_desc.GetShape();
  std::vector<int64_t> output_dims = output_shape.GetDims();
  int64_t dim_index = 0;
  (void)ge::AttrUtils::GetInt(op_desc, ATTR_NAME_REUSE_INPUT_ON_DIM_INDEX, dim_index);
  GE_ASSERT_TRUE((dim_index >= 0) && (dim_index < static_cast<int64_t>(output_dims.size())),
                 "Inner param dim_index value:%" PRId64 " invalid, bigger than dim size:%zu, shape:[%s], node: %s",
                 dim_index, output_dims.size(), output_shape.ToString().c_str(), op_desc->GetNamePtr());
  for (int64_t index = 0; index < dim_index; index++) {
    output_dims[index] = 1;
  }

  const Format out_format = output_desc.GetFormat();
  const DataType data_type = output_desc.GetDataType();
  const auto new_shape = GeShape(output_dims);
  GE_ASSERT_SUCCESS(ge::TensorUtils::CalcTensorMemSize(new_shape, out_format, data_type, output_mem_size));
  GE_ASSERT_TRUE(output_mem_size >= 0,
                 "After calculating, tensor memory size:%" PRId64 " invalid, less than 0. "
                 "new shape:%s, format:%s, dtype:%s, maybe has dynamic shape",
                 output_mem_size, new_shape.ToString().c_str(), TypeUtils::FormatToSerialString(out_format).c_str(),
                 TypeUtils::DataTypeToSerialString(data_type).c_str());
  return SUCCESS;
}

bool NeedSkip(const Node *const node) {
  return node->GetOpDescBarePtr()->HasAttr(ge::ATTR_NAME_BUFFER_POOL_ID) &&
         node->GetOpDescBarePtr()->HasAttr(ge::ATTR_NAME_BUFFER_POOL_SIZE);
}

Status GetSize(const Node *const node, const int32_t out_index, int64_t &size, int64_t &aligned_size,
               size_t &no_aligned_size) {
  const auto out_tensor = node->GetOpDescBarePtr()->GetOutputDescPtr(out_index);
  GE_ASSERT_NOTNULL(out_tensor);

  GE_ASSERT_SUCCESS(TensorUtils::GetSize(*out_tensor, size), "node: %s, out_index: %d", node->GetNamePtr(), out_index);

  aligned_size = size;
  GE_ASSERT_SUCCESS(NodeCheckerUtils::AlignMemSize(aligned_size), "size: %" PRId64 " < 0, node: %s, out_index: %d",
                    size, node->GetNamePtr(), out_index);
  GE_ASSERT_SUCCESS(MemReuseUtils::GetOutputNoAlignSize(*node->GetOpDescBarePtr(), out_index, no_aligned_size),
                    "node: %s, out_index: %d", NodeCheckerUtils::NodeName(node).c_str(), out_index);
  return SUCCESS;
}

Status GetOutputOffsetForBufferFusion(const Node *const peer_node, const int32_t out_index, int64_t &stride,
                                      bool &has_lx_fusion_attr) {
  std::vector<int64_t> offsets_for_fusion = {};
  has_lx_fusion_attr = AttrUtils::GetListInt(peer_node->GetOpDescBarePtr(), ATTR_NAME_OUTPUT_OFFSET_FOR_BUFFER_FUSION,
                                             offsets_for_fusion);
  if (has_lx_fusion_attr) {
    GE_ASSERT_TRUE(static_cast<size_t>(out_index) < offsets_for_fusion.size(),
                   "out_index: %d, size: %zu, peer_node name: %s", out_index, offsets_for_fusion.size(),
                   peer_node->GetNamePtr());
    stride = offsets_for_fusion[out_index];
  }
  return SUCCESS;
}

Status ChangeToNoAlignSizeIfNeed(const OutDataAnchor *const out_data_anchor, int64_t &mem_size) {
  // NoPaddingContinuousInput node input, mem_size need use no-aligned size
  InDataAnchor *continous_node_in_anchor = nullptr;
  for (const auto peer_in : out_data_anchor->GetPeerInDataAnchorsPtr()) {
    if (!MemLayoutConflictUtil::IsContinuousInputThroughRefNode(peer_in, true,
                                                                continous_node_in_anchor)) {
      continue;
    }
    GE_ASSERT_NOTNULL(continous_node_in_anchor);
    const auto op_desc = out_data_anchor->GetOwnerNodeBarePtr()->GetOpDescBarePtr();
    GE_ASSERT_NOTNULL(op_desc);
    size_t no_aligned_size = 0U;
    GE_CHK_STATUS_RET(MemReuseUtils::GetOutputNoAlignSize(*op_desc, static_cast<uint32_t>(out_data_anchor->GetIdx()),
        no_aligned_size), "Get node_name:%s, output_index:%d no align size failed", op_desc->GetNamePtr(),
        out_data_anchor->GetIdx());
    GE_ASSERT_TRUE(no_aligned_size < static_cast<size_t>(std::numeric_limits<int64_t>::max()));
    mem_size = static_cast<int64_t>(no_aligned_size);
  }

  return SUCCESS;
}
}  // namespace
Status NodeCheckerUtils::AlignMemSize(int64_t &size) {
  GE_ASSERT_TRUE(size > 0, "size: %" PRId64 "", size);
  GE_ASSERT_TRUE(size < std::numeric_limits<int64_t>::max() - MEM_ALIGN_SIZE);
  size = (size + MEM_ALIGN_SIZE - 1) / MEM_ALIGN_SIZE * MEM_ALIGN_SIZE;
  return SUCCESS;
}

std::string NodeCheckerUtils::NodeName(const Node *const node) {
  if ((node == nullptr) || (node->GetOpDesc() == nullptr)) {
    return "node is null";
  }
  auto name = node->GetName().substr(0, KNameMax);
  if (name.size() == KNameMax) {
    name += "...topo_id_" + std::to_string(node->GetOpDesc()->GetId());
  } else {
    name += ", topo_id: " + std::to_string(node->GetOpDesc()->GetId());
  }
  return name;
}

SpecailNodeTypes NodeCheckerUtils::GetSpecialNodeTypes(const Node *const node) {
  SpecailNodeTypes types;
  if (NeedSkip(node)) {
    return types;
  }
  if (MemLayoutConflictUtil::IsContinuousInput(node)) {
    types.emplace_back(kSpecialNodeTypeContinuousInput);
  }
  if (MemLayoutConflictUtil::IsContinuousOutput(node)) {
    types.emplace_back(kSpecialNodeTypeContinuousOutput);
  }
  if (MemLayoutConflictUtil::IsNoPaddingContinuousInput(node)) {
    types.emplace_back(kSpecialNodeTypeNoPaddingContinuousInput);
  }
  if (MemLayoutConflictUtil::IsNoPaddingContinuousOutput(node)) {
    types.emplace_back(kSpecialNodeTypeNoPaddingContinuousOutput);
  }
  return types;
}

Status NodeCheckerUtils::GetOutputOffset(const OpDesc *const op_desc, const int32_t out_index, int64_t &offset) {
  GE_ASSERT_NOTNULL(op_desc);
  const auto output_offsets = op_desc->GetOutputOffset();
  GE_ASSERT_TRUE(static_cast<size_t>(out_index) < output_offsets.size(),
                 "index exceed size, out_index: %d, size: %zu, node: %s", out_index, output_offsets.size(),
                 op_desc->GetNamePtr());

  offset = output_offsets.at(out_index);
  return SUCCESS;
}

Status NodeCheckerUtils::GetStrideForContinuousInput(const Node *const peer_node, const int32_t out_index,
                                                     int64_t &stride) {
  GE_ASSERT_NOTNULL(peer_node);
  GE_ASSERT_NOTNULL(peer_node->GetOpDescBarePtr());
  bool has_lx_fusion_attr;
  GE_ASSERT_SUCCESS(GetOutputOffsetForBufferFusion(peer_node, out_index, stride, has_lx_fusion_attr));
  if (has_lx_fusion_attr) {
    return SUCCESS;
  }

  const auto out_tensor = peer_node->GetOpDescBarePtr()->GetOutputDescPtr(out_index);
  GE_ASSERT_NOTNULL(out_tensor);
  int64_t size = -1;
  GE_ASSERT_SUCCESS(TensorUtils::GetSize(*out_tensor, size), "node: %s, out_index: %d", peer_node->GetNamePtr(),
                    out_index);
  GE_ASSERT_SUCCESS(NodeCheckerUtils::AlignMemSize(size), "size: %" PRId64 " < 0, node: %s, out_index: %d", size,
                    peer_node->GetNamePtr(), out_index);
  stride = size;
  return SUCCESS;
}

Status NodeCheckerUtils::GetStrideForNoPaddingContinuousInput(const Node *const node, const Node *const peer_node,
                                                              const int32_t out_index, int64_t &stride) {
  GE_ASSERT_NOTNULL(peer_node);
  bool has_lx_fusion_attr;
  GE_ASSERT_SUCCESS(GetOutputOffsetForBufferFusion(peer_node, out_index, stride, has_lx_fusion_attr));
  if (has_lx_fusion_attr) {
    return SUCCESS;
  }
  GE_ASSERT_NOTNULL(peer_node->GetOpDescBarePtr());
  const auto &output_desc = peer_node->GetOpDescBarePtr()->GetOutputDesc(out_index);
  GE_ASSERT_SUCCESS(CalcSizeWithDimIndex(node->GetOpDescBarePtr(), output_desc, stride),
                    "node: %s, peer_node: %s, out_index: %d", NodeCheckerUtils::NodeName(node).c_str(),
                    NodeCheckerUtils::NodeName(peer_node).c_str(), out_index);
  return SUCCESS;
}

Status NodeCheckerUtils::GetStrideForNoPaddingContinuousOutput(const Node *const node, const int32_t out_index,
                                                               int64_t &stride) {
  GE_ASSERT_NOTNULL(node);
  GE_ASSERT_NOTNULL(node->GetOpDescBarePtr());
  const auto &output_desc = node->GetOpDescBarePtr()->GetOutputDesc(out_index);
  GE_ASSERT_SUCCESS(CalcSizeWithDimIndex(node->GetOpDescBarePtr(), output_desc, stride), "node: %s, out_index: %d",
                    NodeCheckerUtils::NodeName(node).c_str(), out_index);
  return SUCCESS;
}

Status NodeCheckerUtils::ErrorLogAllInputs(const NodeCheckerParam &param) {
  GE_ASSERT_NOTNULL(param.node);
  std::stringstream ss;
  GetFirstOutputShape(param.node, ss);
  GetDimIndexAttr(param.node, ss);
  REPORT_INNER_ERR_MSG("E19999", "node: %s(%s), %s, input nodes size: %zu%s",
                     NodeCheckerUtils::NodeName(param.node).c_str(), param.node->GetTypePtr(),
                     SpecialNodeTypeStr(param.type), param.node->GetOpDescBarePtr()->GetInputsSize(), ss.str().c_str());
  GELOGE(FAILED, "node: %s(%s), %s, input nodes size: %zu%s", NodeCheckerUtils::NodeName(param.node).c_str(),
         param.node->GetTypePtr(), SpecialNodeTypeStr(param.type), param.node->GetOpDescBarePtr()->GetInputsSize(),
         ss.str().c_str());

  int64_t size;
  int64_t aligned_size;
  size_t no_aligned_size;
  int64_t offset;
  for (const auto &in_data_anchor : param.node->GetAllInDataAnchorsPtr()) {
    const auto &peer_out_data_anchor = in_data_anchor->GetPeerOutAnchor();
    GE_ASSERT_NOTNULL(peer_out_data_anchor);
    const auto peer_node = peer_out_data_anchor->GetOwnerNodeBarePtr();
    GE_ASSERT_NOTNULL(peer_node);
    GE_ASSERT_NOTNULL(peer_node->GetOpDescBarePtr());
    const auto out_index = peer_out_data_anchor->GetIdx();
    const auto out_tensor = peer_node->GetOpDescBarePtr()->GetOutputDescPtr(out_index);
    GE_ASSERT_NOTNULL(out_tensor);

    GE_ASSERT_SUCCESS(GetSize(peer_node, out_index, size, aligned_size, no_aligned_size),
                      "peer_node: %s, out_index: %d", peer_node->GetNamePtr(), out_index);
    GE_ASSERT_SUCCESS(NodeCheckerUtils::GetOutputOffset(peer_node->GetOpDescBarePtr(), out_index, offset),
                      "peer_node: %s, out_index: %d", peer_node->GetNamePtr(), out_index);

    REPORT_INNER_ERR_MSG("E19999", "input %d: [%s out %d, offset: %" PRId64 ", get size: %" PRId64 ", 512_aligned_size: "
        "%" PRId64 ", no_aligned_size: %zu, shape: [%s] %s%s]", in_data_anchor->GetIdx(), NodeName(peer_node).c_str(),
        out_index, offset, size, aligned_size, no_aligned_size, out_tensor->GetShape().ToString().c_str(),
        TypeUtils::DataTypeToSerialString(out_tensor->GetDataType()).c_str(),
        GetAttrsForNoPaddingContinuousInput(peer_node, out_index).c_str());
    GELOGE(FAILED, "input %d: [%s out %d, offset: %" PRId64 ", get size: %" PRId64 ", 512_aligned_size: %" PRId64 ","
        " no_aligned_size: %zu, shape: [%s] %s%s]", in_data_anchor->GetIdx(), NodeName(peer_node).c_str(), out_index,
        offset, size, aligned_size, no_aligned_size, out_tensor->GetShape().ToString().c_str(),
        TypeUtils::DataTypeToSerialString(out_tensor->GetDataType()).c_str(),
        GetAttrsForNoPaddingContinuousInput(peer_node, out_index).c_str());
  }
  return SUCCESS;
}

Status NodeCheckerUtils::ErrorLogAllOutputs(const NodeCheckerParam &param) {
  const Node *const node = param.node;
  GE_ASSERT_NOTNULL(node);
  std::stringstream ss;
  GetFirstInputShape(node, ss);
  REPORT_INNER_ERR_MSG("E19999", "node: %s(%s), %s, output nodes size: %zu%s", NodeCheckerUtils::NodeName(node).c_str(),
                     node->GetTypePtr(), SpecialNodeTypeStr(param.type), node->GetOpDescBarePtr()->GetOutputsSize(),
                     ss.str().c_str());
  GELOGE(FAILED, "node: %s(%s), %s, output nodes size: %zu%s", NodeCheckerUtils::NodeName(node).c_str(),
         node->GetTypePtr(), SpecialNodeTypeStr(param.type), node->GetOpDescBarePtr()->GetOutputsSize(),
         ss.str().c_str());
  int64_t size;
  int64_t aligned_size;
  size_t no_aligned_size;
  int64_t offset = 0;
  for (const auto &out_data_anchor : node->GetAllOutDataAnchorsPtr()) {
    const auto out_index = out_data_anchor->GetIdx();
    const auto out_tensor = node->GetOpDescBarePtr()->GetOutputDescPtr(out_index);
    GE_ASSERT_NOTNULL(out_tensor);

    GE_ASSERT_SUCCESS(GetSize(node, out_index, size, aligned_size, no_aligned_size), "node: %s, out_index: %d",
                      node->GetNamePtr(), out_index);
    (void)NodeCheckerUtils::GetOutputOffset(node->GetOpDescBarePtr(), out_index, offset);

    REPORT_INNER_ERR_MSG("E19999", "output %d: [offset: %" PRId64 ", get size: %" PRId64 ", 512_aligned_size: %" PRId64 ","
        " no_aligned_size: %zu, shape: [%s] %s%s]", out_index, offset, size, aligned_size, no_aligned_size,
        out_tensor->GetShape().ToString().c_str(), TypeUtils::DataTypeToSerialString(out_tensor->GetDataType()).c_str(),
        GetAttrsForNoPaddingContinuousOutput(out_data_anchor).c_str());
    GELOGE(FAILED, "output %d: [offset: %" PRId64 ", get size: %" PRId64 ", 512_aligned_size: %" PRId64 ","
        " no_aligned_size: %zu, shape: [%s] %s%s]", out_index, offset, size, aligned_size, no_aligned_size,
        out_tensor->GetShape().ToString().c_str(), TypeUtils::DataTypeToSerialString(out_tensor->GetDataType()).c_str(),
        GetAttrsForNoPaddingContinuousOutput(out_data_anchor).c_str());
  }
  return SUCCESS;
}

Status NodeCheckerUtils::CheckNoPaddingContinuousInputNodeAttrs(const Node *const node) {
  GE_ASSERT_NOTNULL(node);
  if (!MemLayoutConflictUtil::IsNoPaddingContinuousInput(node)) {
    return SUCCESS;
  }
  bool is_first_valid_node = true;
  bool first_in_node_has_attr = false;
  for (const auto &in_data_anchor : node->GetAllInDataAnchorsPtr()) {
    const auto &peer_out_anchor = in_data_anchor->GetPeerOutAnchor().get();
    GE_ASSERT_NOTNULL(peer_out_anchor);
    const auto in_node = peer_out_anchor->GetOwnerNodeBarePtr();
    GE_ASSERT_NOTNULL(in_node);
    if (MemLayoutConflictUtil::IsNoPaddingContinuousInput(in_node)) {
      continue;
    }
    std::vector<int64_t> offset_list = {};
    const bool has_offset_list =
        AttrUtils::GetListInt(in_node->GetOpDescBarePtr(), ATTR_NAME_OUTPUT_OFFSET_LIST_FOR_CONTINUOUS, offset_list);
    if (is_first_valid_node) {
      is_first_valid_node = false;
      first_in_node_has_attr = has_offset_list;
    }

    std::vector<int64_t> offsets_for_fusion = {};
    const bool has_lx_fusion_attr = AttrUtils::GetListInt(
        in_node->GetOpDescBarePtr(), ATTR_NAME_OUTPUT_OFFSET_FOR_BUFFER_FUSION, offsets_for_fusion);
    if ((has_offset_list && has_lx_fusion_attr) || (first_in_node_has_attr != has_offset_list)) {
      REPORT_INNER_ERR_MSG("E19999", "as input nodes for %s(%s), %s", NodeCheckerUtils::NodeName(node).c_str(),
                         node->GetTypePtr(), kNoPaddingContinuousInputAttrErrorLog.c_str());
      GELOGE(FAILED, "as input nodes for %s(%s), %s", NodeCheckerUtils::NodeName(node).c_str(), node->GetTypePtr(),
             kNoPaddingContinuousInputAttrErrorLog.c_str());
      NodeCheckerParam param{node->GetOwnerComputeGraph(), node, kSpecialNodeTypeNoPaddingContinuousInput};
      NodeCheckerUtils::ErrorLogAllInputs(param);
      return FAILED;
    }
  }
  return SUCCESS;
}

Status NodeCheckerUtils::CheckNoPaddingContinuousOutputNodeAttrs(const Node *const node) {
  GE_ASSERT_NOTNULL(node);
  if (!MemLayoutConflictUtil::IsNoPaddingContinuousOutput(node)) {
    return SUCCESS;
  }
  bool is_first_valid_node = true;
  bool first_out_node_has_attr = false;
  for (const auto &out_data_anchor : node->GetAllOutDataAnchorsPtr()) {
    bool has_offset_list = false;
    bool skip_node = out_data_anchor->GetPeerInDataAnchorsPtr().empty();
    for (const auto &peer_in_anchor : out_data_anchor->GetPeerInDataAnchorsPtr()) {
      const auto out_node = peer_in_anchor->GetOwnerNodeBarePtr();
      GE_ASSERT_NOTNULL(out_node);
      if (MemLayoutConflictUtil::IsNoPaddingContinuousOutput(out_node)) {
        skip_node = true;
        break;
      }
      std::vector<int64_t> offset_list = {};
      has_offset_list =
          AttrUtils::GetListInt(out_node->GetOpDescBarePtr(), ATTR_NAME_INPUT_OFFSET_LIST_FOR_CONTINUOUS, offset_list);
      (void)offset_list;
      if (has_offset_list) {
        break;
      }
    }
    if (skip_node) {
      continue;
    }
    if (is_first_valid_node) {
      is_first_valid_node = false;
      first_out_node_has_attr = has_offset_list;
    }

    if (has_offset_list != first_out_node_has_attr) {
      REPORT_INNER_ERR_MSG("E19999", "as output nodes for %s(%s), %s", NodeCheckerUtils::NodeName(node).c_str(),
                         node->GetTypePtr(), kNoPaddingContinuousOutputAttrErrorLog.c_str());
      GELOGE(FAILED, "as output nodes for %s(%s), %s", NodeCheckerUtils::NodeName(node).c_str(), node->GetTypePtr(),
             kNoPaddingContinuousOutputAttrErrorLog.c_str());
      NodeCheckerParam param{node->GetOwnerComputeGraph(), node, kSpecialNodeTypeNoPaddingContinuousOutput};
      NodeCheckerUtils::ErrorLogAllOutputs(param);
      return FAILED;
    }
  }
  return SUCCESS;
}

// PhonyConcat means NoPaddingContinuousInputNode
Status NodeCheckerUtils::CheckPhonyConcatOutputSize(const Node *const node) {
  GE_ASSERT_NOTNULL(node);
  if (!MemLayoutConflictUtil::IsNoPaddingContinuousInput(node)) {
    return SUCCESS;
  }
  GE_ASSERT_TRUE(node->GetOpDescBarePtr()->GetOutputsSize() == 1U,
                 "node %s need nopadding continuous input, should only has one output, but actual output size: %zu",
                 NodeCheckerUtils::NodeName(node).c_str(), node->GetOpDescBarePtr()->GetOutputsSize());
  return SUCCESS;
}

// PhonySplit means NoPaddingContinuousOutputNode
Status NodeCheckerUtils::CheckPhonySplitInputSize(const Node *const node) {
  GE_ASSERT_NOTNULL(node);
  if (!MemLayoutConflictUtil::IsNoPaddingContinuousOutput(node)) {
    return SUCCESS;
  }
  GE_ASSERT_TRUE(node->GetOpDescBarePtr()->GetInputsSize() == 1U,
                 "node %s need nopadding continuous output, should only has one input, but actual input size: %zu",
                 NodeCheckerUtils::NodeName(node).c_str(), node->GetOpDescBarePtr()->GetInputsSize());
  return SUCCESS;
}

// 如果node的index输出直连或经RefNode间接连接到了NOPADDING连续输入节点，那么mem_size使用非对齐大小
Status NodeCheckerUtils::GetOutputSize(const Node *const node, const int32_t index, int64_t &mem_size) {
  GE_ASSERT_NOTNULL(node);
  const auto op_desc = node->GetOpDescBarePtr();
  GE_ASSERT_NOTNULL(op_desc);
  const auto output_tensor_desc = op_desc->MutableOutputDesc(static_cast<uint32_t>(index));
  GE_ASSERT_NOTNULL(output_tensor_desc, "node %s output[%d] tensor desc is null", node->GetNamePtr(), index);
  const auto out_anchor_ptr = node->GetOutDataAnchor(index).get();
  GE_ASSERT_NOTNULL(out_anchor_ptr);

  GE_ASSERT_SUCCESS(MemReuseUtils::GetTensorSize(*output_tensor_desc, mem_size),
                    "get node %s output[%d] tensor size failed", node->GetNamePtr(), index);
  GE_ASSERT_SUCCESS(ChangeToNoAlignSizeIfNeed(out_anchor_ptr, mem_size));
  return SUCCESS;
}

// 如果node的index输出直连或经RefNode间接连接到了NOPADDING连续输入节点，那么mem_size使用非对齐大小
Status NodeCheckerUtils::GetInputSize(const Node *const node, const int32_t index, int64_t &mem_size) {
  GE_ASSERT_NOTNULL(node);
  const auto op_desc = node->GetOpDescBarePtr();
  GE_ASSERT_NOTNULL(op_desc);
  const auto in_tensor_desc = op_desc->MutableInputDesc(static_cast<uint32_t>(index));
  GE_ASSERT_NOTNULL(in_tensor_desc, "node %s input[%d] tensor desc is null", node->GetNamePtr(), index);
  const auto in_anchor_ptr = node->GetInDataAnchor(index).get();
  GE_ASSERT_NOTNULL(in_anchor_ptr);

  const auto peer_out_anchor = in_anchor_ptr->GetPeerOutAnchor();
  GE_ASSERT_NOTNULL(peer_out_anchor);
  const auto in_node = peer_out_anchor->GetOwnerNode();
  GE_ASSERT_NOTNULL(in_node);
  GE_ASSERT_SUCCESS(MemReuseUtils::GetTensorSize(*in_tensor_desc, mem_size),
                    "node %s get %d input tensor size failed", node->GetNamePtr(), index);

  InDataAnchor *continous_node_in_anchor = nullptr;
  if (MemLayoutConflictUtil::IsContinuousInputThroughRefNode(in_anchor_ptr, true, continous_node_in_anchor)) {
    (void)continous_node_in_anchor;
    size_t no_aligned_size = 0U;
    GE_CHK_STATUS_RET(MemReuseUtils::GetNoAlignSize(*in_tensor_desc, no_aligned_size),
                      "Get node_name:%s, input_index:%d no align size failed", op_desc->GetNamePtr(), index);
    GE_ASSERT_TRUE(no_aligned_size < static_cast<size_t>(std::numeric_limits<int64_t>::max()));
    mem_size = static_cast<int64_t>(no_aligned_size);
  }
  return SUCCESS;
}
}  // namespace ge