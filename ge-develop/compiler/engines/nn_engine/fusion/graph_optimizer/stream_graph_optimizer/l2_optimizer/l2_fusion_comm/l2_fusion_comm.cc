/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph_optimizer/stream_graph_optimizer/l2_optimizer/l2_fusion_comm/l2_fusion_comm.h"
#include <set>
#include "common/math_util.h"
#include "common/fe_inner_attr_define.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/tensor_utils.h"
#include "param_calculate/tensor_compute_util.h"
#include "common/fe_utils.h"

namespace fe {
const uint64_t ONE_KILO_BYTE = 1024;  // 1KB
const uint64_t MINI_L2_SIZE_MB = 8;   // 8MB
const uint64_t PAGE_NUM = 64;
const uint64_t L2_SIZE = MINI_L2_SIZE_MB * ONE_KILO_BYTE * ONE_KILO_BYTE;

void L2FusionComm::GetL2HardwareSet(L2BufferInfo &l2) {
  l2.l2_buffer_size = L2_SIZE;
  l2.page_num = PAGE_NUM;
  const uint64_t max_data_num = 8;
  l2.max_data_num = max_data_num;
}

Status L2FusionComm::CalcTensorSize(const ge::GeTensorDesc &tensor_desc, int64_t &tensor_size) {
  // verify the tensor
  FE_CHECK(TensorComputeUtil::VerifyTensor(tensor_desc) != SUCCESS,
           REPORT_FE_ERROR("[StreamOpt][L2Opt][CaclTensorSize] Failed to verify this tensor."),
           return FAILED);

  int64_t element_cnt;
  FE_CHECK(TensorComputeUtil::GetElementCountByMultiply(tensor_desc, element_cnt) != SUCCESS,
           REPORT_FE_ERROR("[StreamOpt][L2Opt][CalcTensorSize] Failed to obtain element count."), return FAILED);
  ge::DataType data_type = tensor_desc.GetDataType();
  int32_t output_real_calc_flag = 0;
  FE_CHECK(
      TensorComputeUtil::GetTensorSizeByDataType(element_cnt, data_type, tensor_size, output_real_calc_flag) != SUCCESS,
      REPORT_FE_ERROR("[StreamOpt][L2Opt][CalcTensorSize] Failed to calculate tensor size using element count and data type."),
      return FAILED);
  return SUCCESS;
}

// output param: data_size
Status L2FusionComm::GetGraphDataSize(const ge::OpDescPtr &op_desc, const ge::GeTensorDesc &tensor_desc,
                                      const uint32_t &data_type, int64_t &data_size) {
  data_size = 0;
  // AIPP situation, the input size, we should use the size of fmk
  if (data_type == INPUT_DATA) {
    bool support_aipp = false;
    if (ge::AttrUtils::GetBool(op_desc, AIPP_CONV_FLAG, support_aipp) && support_aipp) {
      int64_t size = 0;
      FE_CHECK(ge::TensorUtils::GetSize(tensor_desc, size) != ge::GRAPH_SUCCESS,
               REPORT_FE_ERROR("[StreamOpt][L2Opt][GetGphDataSize] Get size failed!"),
               return FAILED);
      data_size = size;
      return SUCCESS;
    }
  }

  FE_CHECK(CalcTensorSize(tensor_desc, data_size) != SUCCESS,
           REPORT_FE_ERROR("[StreamOpt][L2Opt][GetGphDataSize] Calc tensor size failed!"), return FAILED);
  return SUCCESS;
}

// data_type: 0-input, 1-output
Status L2FusionComm::GetGraphDataSize(const ge::NodePtr &node, const size_t &tensor_index, const uint32_t &data_type,
                                      int64_t &data_size) {
  if (data_type == INPUT_DATA) {
    // dual type, we need get the related size
    for (size_t i = 0; i < node->GetAllInDataAnchors().size(); ++i) {
      auto in_data_anchor = node->GetInDataAnchor(i);
      FE_CHECK_NOTNULL(in_data_anchor);
      auto peer_out_data_anchor = in_data_anchor->GetPeerOutAnchor();
      if (peer_out_data_anchor == nullptr) {
        continue;
      }
      if (IsNonValidOp(peer_out_data_anchor->GetOwnerNode())) {
        continue;
      }

      auto peer_out_op_desc = peer_out_data_anchor->GetOwnerNode()->GetOpDesc();
      ge::GeTensorDesc output_tensor = peer_out_op_desc->GetInputDesc(peer_out_data_anchor->GetIdx());
      bool is_dual_output = (in_data_anchor->GetIdx() == static_cast<int32_t>(tensor_index)) &&
                            (output_tensor.GetDataType() == ge::DT_DUAL);

      if (is_dual_output) {
        FE_CHECK(GetGraphDataSize(peer_out_op_desc, output_tensor, OUTPUT_DATA, data_size) != SUCCESS,
                 REPORT_FE_ERROR("[StreamOpt][L2Opt][GetGphDataSize] GetGraphDataSize operation failed."), return FAILED);
        return SUCCESS;
      }
    }

    ge::GeTensorDesc tensor = node->GetOpDesc()->GetInputDesc(tensor_index);
    FE_CHECK(GetGraphDataSize(node->GetOpDesc(), tensor, INPUT_DATA, data_size) != SUCCESS,
             REPORT_FE_ERROR("[StreamOpt][L2Opt][GetGphDataSize] GetGraphDataSize operation failed."), return FAILED);
  } else {
    ge::GeTensorDesc tensor = node->GetOpDesc()->GetOutputDesc(tensor_index);
    FE_CHECK(GetGraphDataSize(node->GetOpDesc(), tensor, OUTPUT_DATA, data_size) != SUCCESS,
             REPORT_FE_ERROR("[StreamOpt][L2Opt][GetGphDataSize] GetGraphDataSize operation failed."), return FAILED);
  }

  return SUCCESS;
}

void L2FusionComm::DisplayOpL2DataInfo(const std::vector<OpL2DataInfo> &l2_data_vec) {
  FE_LOGD("Size of L2 data is %zu.", l2_data_vec.size());
  for (const OpL2DataInfo &l2_data_info : l2_data_vec) {
    FE_LOGD("Node id[%u], node name[%s], node task size[%u], input size[%zu], output size[%zu].", l2_data_info.node_id,
            l2_data_info.node_name.c_str(), l2_data_info.node_task_num, l2_data_info.input.size(),
            l2_data_info.output.size());
    DisplayTensorL2DataInfo(l2_data_info.input);
    DisplayTensorL2DataInfo(l2_data_info.output);
  }
}

void L2FusionComm::DisplayTensorL2DataInfo(const TensorL2DataMap &tensor_l2_map) {
  for (TensorL2DataMap::const_iterator iter = tensor_l2_map.begin(); iter != tensor_l2_map.end(); ++iter) {
    FE_LOGD("L2 tensor data: Data id[%lu], ddr addr[%lu], data size[%ld].",
            iter->second.id, iter->second.ddr_addr, iter->second.data_size);
  }
}

void L2FusionComm::DisplayTensorL2AllocInfo(const TensorL2AllocMap &input, rtL2Ctrl_t l2ctrl,
                                            const TensorL2AllocMap &tensor_alloc_map) {
  int64_t page_size = L2_SIZE / PAGE_NUM;
  for (auto it = tensor_alloc_map.cbegin(); it != tensor_alloc_map.cend(); ++it) {
    FE_LOGD("Data_id = %lu, data_in_l2_id = %d, if_input = %u, data_size = %u.",
            it->first, it->second.data_in_l2_id, (uint32_t)(input.count(it->first)),
            l2ctrl.data[it->second.data_in_l2_id].L2_data_section_size);
    FE_LOGD("l2PageN = %lu, l2_addr0 = %lu, l2Addr=%lu, ddr_addr_key = %lu, ddr_addr = %lu.", it->second.l2PageNum,
            it->second.data_in_l2_addr - (l2ctrl.data[it->second.data_in_l2_id].L2_page_offset_base) * page_size,
            it->second.data_in_l2_addr, it->first, l2ctrl.data[it->second.data_in_l2_id].L2_mirror_addr);
    FE_LOGD("offset = %2u, previous_offset = %2d", static_cast<uint32_t>(l2ctrl.data[it->second.data_in_l2_id].L2_page_offset_base),
            static_cast<int32_t>(l2ctrl.data[it->second.data_in_l2_id].prev_L2_page_offset_base));
  }
}

void L2FusionComm::DisplayOpL2AllocInfo(const OpL2AllocMap &l2_alloc_map) {
  for (auto it = l2_alloc_map.cbegin(); it != l2_alloc_map.cend(); ++it) {
    const rtL2Ctrl_t &l2ctrl = it->second.l2ctrl;
    const OpL2AllocInfo &l2_info = it->second;

    FE_LOGD("Node[%u, %s] l2ctrl.size = %lu.", l2_info.node_id, l2_info.node_name.c_str(), l2ctrl.size);
    FE_LOGD("Input size[%zu], output size[%zu], converge size[%zu], standing size[%zu].",
            l2_info.input.size(), l2_info.output.size(), l2_info.converge.size(), l2_info.standing_data.size());

    FE_LOGD("The following is the standing allocation information.");
    DisplayTensorL2AllocInfo(l2_info.input, l2ctrl, l2_info.standing_data);
    FE_LOGD("Following is input alloc info.");
    DisplayTensorL2AllocInfo(l2_info.input, l2ctrl, l2_info.output);
    FE_LOGD("The following is the converge allocation information.");
    DisplayTensorL2AllocInfo(l2_info.input, l2ctrl, l2_info.converge);
  }
}

bool L2FusionComm::CheckData(const ge::OpDescPtr &op_desc) {
  if (op_desc->GetType() == OP_TYPE_PLACE_HOLDER) {
    string parent_op_type;
    if (!ge::AttrUtils::GetStr(op_desc, PARENT_OP_TYPE, parent_op_type)) {
      return false;
    }

    if (IsRootGraphData(parent_op_type)) {
      return true;
    }
  }
  return false;
}

// nonvalid op include Const,Constant,End
// PlaceHolder except that the parent_op_type is data.
bool L2FusionComm::IsNonValidOp(const ge::NodePtr &node) {
  std::set<std::string> set;
  // add const op
  set.insert(CONSTANT);
  set.insert(CONSTANTOP);
  set.insert(OP_TYPE_END);
  std::set<std::string>::const_iterator iter = set.find(node->GetType());
  bool is_non_valid = iter != set.end();

  if (node->GetType() == OP_TYPE_PLACE_HOLDER) {
    if (!CheckData(node->GetOpDesc())) {
      is_non_valid = true;
    }
  }
  FE_LOGD("Node named %s, op type is %s, is_non_valid is %d.", node->GetName().c_str(), node->GetType().c_str(),
          is_non_valid);

  return is_non_valid;
}

bool L2FusionComm::IsNonValidOpOrData(const ge::NodePtr &node) {
  bool is_data = IsRootGraphData(node->GetType()) || CheckData(node->GetOpDesc());
  FE_LOGD("Node named %s, op type is %s, is_data value is %d.", node->GetName().c_str(), node->GetType().c_str(),
          is_data);

  return (is_data || IsNonValidOp(node));
}

bool L2FusionComm::IsConstInput(const ge::ConstNodePtr &node, const size_t &index) {
  FE_CHECK(node == nullptr, FE_LOGW("node is null!"), return false);
  return IsConstInput(*node, index);
}

// const input include PlaceHolder nddes except that the parent_op_type is data.
bool L2FusionComm::IsConstInput(const ge::Node &node, const size_t &index) {
  bool ret = true;
  if (index < node.GetAllInDataAnchors().size()) {
    for (const auto &anchor : node.GetAllInDataAnchors()) {
      if (anchor->GetIdx() != static_cast<int32_t>(index)) {
        continue;
      }
      auto peer_anchor = anchor->GetPeerOutAnchor();
      if (!peer_anchor) {
        break;
      }
      auto owner_node = peer_anchor->GetOwnerNode();
      if (!owner_node) {
        break;
      }
      if (owner_node->GetOpDesc()->GetType() != OP_TYPE_PLACE_HOLDER) {
        return false;
      }
      ret = (!CheckData(owner_node->GetOpDesc()));
    }
  }
  return ret;
}
}  // namespace fe
