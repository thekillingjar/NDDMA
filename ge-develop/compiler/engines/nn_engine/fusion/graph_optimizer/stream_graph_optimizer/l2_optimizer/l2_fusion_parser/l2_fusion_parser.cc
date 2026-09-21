/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph_optimizer/stream_graph_optimizer/l2_optimizer/l2_fusion_parser/l2_fusion_parser.h"
#include <string>
#include <vector>
#include "common/fe_type_utils.h"
#include "common/fe_utils.h"
#include "common/op_info_common.h"
#include "common/fe_inner_attr_define.h"
#include "graph/utils/tensor_utils.h"
#include "ops_store/ops_kernel_manager.h"

namespace fe {
Status L2FusionParser::GetDataDependentCountMap(const ge::ComputeGraph &graph,
                                                std::map<uint64_t, size_t> &data_dependent_count_map) {
  for (const ge::NodePtr &node : graph.GetDirectNode()) {
    FE_LOGD("Node [%s, %s]", node->GetName().c_str(), node->GetType().c_str());
    // jump over CONSTANT and other non-valid node
    if (L2FusionComm::IsNonValidOp(node)) {
      continue;
    }
    int64_t node_id = node->GetOpDesc()->GetId();
    for (auto out_anchor : node->GetAllOutDataAnchors()) {
      FE_CHECK(out_anchor == nullptr,
               REPORT_FE_ERROR("[StreamOpt][L2Opt][GetDataDepdCountMap] Node %s out anchor is null.",
               node->GetName().c_str()), return FAILED);
      size_t peer_anchor_num = out_anchor->GetPeerAnchors().size();
      if (peer_anchor_num == 0) {
        continue;
      }
      if (peer_anchor_num > 1) {
        set<ge::NodePtr> peer_anchor_nodes;
        for (auto peer_anchor : out_anchor->GetPeerAnchors()) {
          peer_anchor_nodes.insert(peer_anchor->GetOwnerNode());
        }
        peer_anchor_num = peer_anchor_nodes.size();
      }
      uint64_t tmp_id = DATA_OVERALL_ID(static_cast<uint64_t>(node_id), OUTPUT_DATA,
                                        static_cast<uint32_t>(out_anchor->GetIdx()));
      if (data_dependent_count_map.find(tmp_id) != data_dependent_count_map.end()) {
        FE_LOGD("There are already output tensors for node %s, index %d.", node->GetName().c_str(), out_anchor->GetIdx());
      }
      data_dependent_count_map[tmp_id] = peer_anchor_num;
      FE_LOGD("node %s: node_id %ld, out_anchor_index %d; id is %lu, peer_anchor_num is %zu.",
              node->GetName().c_str(), node_id, out_anchor->GetIdx(), tmp_id, peer_anchor_num);
    }
  }

  return SUCCESS;
}

bool L2FusionParser::HasAtomicNode(const ge::NodePtr &nodePtr) {
  for (auto &input_node : nodePtr->GetInControlNodes()) {
    if (input_node->GetType() == ATOMIC_CLEAN_OP_TYPE) {
      return true;
    }
  }
  return false;
}

Status L2FusionParser::GetDataFromGraph(const ge::ComputeGraph &graph,
                                        const OpReferTensorIndexMap &refer_tensor_index_map,
                                        std::vector<OpL2DataInfo> &op_l2_data_vec) {
  std::vector<ge::NodePtr> end_node_vec;
  for (const ge::NodePtr &node : graph.GetDirectNode()) {
    FE_CHECK(node == nullptr, REPORT_FE_ERROR("[StreamOpt][L2Opt][GetDataFromGph] node is nullptr."),
             return FAILED);
    FE_CHECK(node->GetOpDesc() == nullptr, REPORT_FE_ERROR("[StreamOpt][L2Opt][GetDataFromGph] op desc is nullptr."),
             return FAILED);
    if (NoNeedAllocOpL2Info(node)) {
      continue;
    }

    OpL2DataInfo op_l2_data;
    op_l2_data.node_id = node->GetOpDesc()->GetId();
    op_l2_data.node_name = node->GetName();
    FE_LOGD("Begin generating L2 data info for node [%s, %s].", node->GetName().c_str(), node->GetType().c_str());
    FE_CHECK(GetDataFromNode(node, refer_tensor_index_map, op_l2_data, end_node_vec) != SUCCESS,
             REPORT_FE_ERROR("[StreamOpt][L2Opt][GetDataFromGph] GetDataFromNode failed!"), return FAILED);
    op_l2_data_vec.push_back(op_l2_data);
  }
  for (const ge::NodePtr &node : end_node_vec) {
    ClearReferDataInfo(node, refer_tensor_index_map, op_l2_data_vec);
  }
  return SUCCESS;
}

Status L2FusionParser::GetDataFromNode(const ge::NodePtr &node, const OpReferTensorIndexMap &refer_tensor_index_map,
                                       OpL2DataInfo &op_l2_data, std::vector<ge::NodePtr> &end_node_vec) {
  op_l2_data.node_task_num = 1;
  // the node is not in L2 white listTensorL2DataInfo
  FE_CHECK(GetInputDataFromNode(node, (op_l2_data.input)) != SUCCESS,
           REPORT_FE_ERROR("[StreamOpt][L2Opt][GetDataFromGph] GetInputDataFromNode failed!"),
           return FAILED);
  if (NoNeedAllocOutputs(node)) {
    // if this node is refer node, clear the refer tensor info
    end_node_vec.push_back(node);
    return SUCCESS;
  }
  FE_CHECK(GetOutputDataFromNode(node, refer_tensor_index_map, (op_l2_data.output)) != SUCCESS,
           REPORT_FE_ERROR("[StreamOpt][L2Opt][GetDataFromGph] Failed to get output data from node!"),
           return FAILED);
  return SUCCESS;
}

Status L2FusionParser::GetInputDataFromNode(const ge::NodePtr &node, TensorL2DataMap &input_tensor_l2_map) {
  ge::OpDescPtr opdef = node->GetOpDesc();
  FE_CHECK(opdef == nullptr,
           REPORT_FE_ERROR("[StreamOpt][L2Opt][GetInDataFromInNd] opdef is nullptr."),
           return FAILED);

  // get input
  vector<int64_t> op_input_offset = opdef->GetInputOffset();
  size_t input_size = opdef->GetAllInputsSize();
  for (size_t i = 0; i < input_size; ++i) {
    if (L2FusionComm::IsConstInput(node, i)) {
      FE_LOGD("Peer output node of node[%s]'s input[%zu] is a constant node, thus not processed.", node->GetName().c_str(), i);
      continue;
    }
    ge::InDataAnchorPtr in_anchor = node->GetInDataAnchor(i);
    FE_CHECK(in_anchor == nullptr,
             REPORT_FE_ERROR("[StreamOpt][L2Opt][GetInDataFromInNd] node %s in anchor is null",
             node->GetName().c_str()), return FAILED);
    // no out anchor, do continue
    if (in_anchor->GetPeerOutAnchor() == nullptr) {
      continue;
    }

    const ge::GeTensorDesc tensor_desc = opdef->GetInputDesc(i);
    TensorL2DataInfo l2_data;
    l2_data.id = GetDataUnitDataId(node, i, INPUT_DATA, tensor_desc);
    FE_LOGD("Data ID of node [%s, %s]'s input [%zu] is [%lu].", node->GetName().c_str(), node->GetType().c_str(),
            i, l2_data.id);
    l2_data.ddr_addr = (op_input_offset.size() > i) ? op_input_offset[i] : 0;
    FE_CHECK(L2FusionComm::GetGraphDataSize(node, i, INPUT_DATA, l2_data.data_size) != SUCCESS,
             REPORT_FE_ERROR("[StreamOpt][L2Opt][GetInDataFromInNd] Failed to get graph data size!"), return FAILED);
    l2_data.occupy_data_ids.clear();
    input_tensor_l2_map.insert(TensorL2DataPair(l2_data.id, l2_data));
  }
  return SUCCESS;
}

Status L2FusionParser::GetOutputDataFromNode(const ge::NodePtr &node,
                                             const OpReferTensorIndexMap &refer_tensor_index_map,
                                             TensorL2DataMap &output_tensor_l2_map) {
  ge::OpDescPtr op_desc_ptr = node->GetOpDesc();
  FE_CHECK(op_desc_ptr == nullptr,
           REPORT_FE_ERROR("[StreamOpt][L2Opt][GetOutDataFromInNd] op_desc_ptr is nullptr."), return FAILED);

  vector<int64_t> output_offset = op_desc_ptr->GetOutputOffset();
  size_t output_size = op_desc_ptr->GetOutputsSize();
  for (size_t i = 0; i < output_size; ++i) {
    auto output_desc = op_desc_ptr->GetOutputDesc(i);
    if (NoNeedAllocOutput(output_desc)) {
      FE_LOGI("Node[type=%s, name=%s]: does not allocate L2 for the output %s.", op_desc_ptr->GetType().c_str(),
              op_desc_ptr->GetName().c_str(), op_desc_ptr->GetOutputNameByIndex(i).c_str());
      continue;
    }

    TensorL2DataInfo l2_data;
    l2_data.id = GetDataUnitDataId(node, i, OUTPUT_DATA, output_desc);
    FE_LOGD("Data ID of node [%s, %s]'s output [%zu] is [%lu].", node->GetName().c_str(), node->GetType().c_str(),
            i, l2_data.id);
    l2_data.ddr_addr = (output_offset.size() > i) ? output_offset[i] : 0;  // hyz,output
    FE_CHECK(L2FusionComm::GetGraphDataSize(node, i, OUTPUT_DATA, l2_data.data_size) != SUCCESS,
             REPORT_FE_ERROR("[StreamOpt][L2Opt][GetOutDataFromInNd] GetGraphDataSize operation failed!"), return FAILED);
    l2_data.occupy_data_ids.clear();
    l2_data.refer_data_id = 0;
    OpReferTensorIndexMap::const_iterator iter = refer_tensor_index_map.find(l2_data.id);
    if (iter != refer_tensor_index_map.cend()) {
      l2_data.refer_data_id = GetDataUnitDataId(node, static_cast<size_t>(iter->second), INPUT_DATA, output_desc);
    }
    output_tensor_l2_map.insert(TensorL2DataPair(l2_data.id, l2_data));
  }
  return SUCCESS;
}

void L2FusionParser::ClearReferTensorData(const ge::InDataAnchorPtr &in_data_anchor,
                                          const OpReferTensorIndexMap &refer_tensor_index_map,
                                          std::vector<OpL2DataInfo> &op_l2_data_vec) {
  if (in_data_anchor == nullptr) {
    return;
  }
  const ge::OutDataAnchorPtr peer_out_anchor = in_data_anchor->GetPeerOutAnchor();
  if (peer_out_anchor == nullptr) {
    return;
  }
  ge::NodePtr peer_in_node = peer_out_anchor->GetOwnerNode();
  if (peer_in_node == nullptr) {
    return;
  }
  int32_t output_index = peer_out_anchor->GetIdx();
  uint64_t data_id = DATA_OVERALL_ID(static_cast<uint64_t>(peer_in_node->GetOpDesc()->GetId()), OUTPUT_DATA,
                                     static_cast<uint64_t>(output_index));
  for (size_t i = 0; i < op_l2_data_vec.size(); ++i) {
    if (op_l2_data_vec[i].node_id == peer_in_node->GetOpDesc()->GetId()) {
      TensorL2DataMap::const_iterator out_iter = op_l2_data_vec[i].output.find(data_id);
      if (out_iter != op_l2_data_vec[i].output.end()) {
        op_l2_data_vec[i].output.erase(out_iter);
      }
      break;
    }
  }

  OpReferTensorIndexMap::const_iterator iter = refer_tensor_index_map.find(data_id);
  if (iter != refer_tensor_index_map.end()) {
    FE_LOGD("The reference of output[%d] of node[%s, %s] is input[%u].",
            output_index, peer_in_node->GetName().c_str(), peer_in_node->GetType().c_str(), iter->second);
    ge::InDataAnchorPtr peer_in_data_anchor = peer_in_node->GetInDataAnchor(static_cast<int32_t>(iter->second));
    ClearReferTensorData(peer_in_data_anchor, refer_tensor_index_map, op_l2_data_vec);
  }
}

void L2FusionParser::ClearReferDataInfo(const ge::NodePtr &node, const OpReferTensorIndexMap &refer_tensor_index_map,
                                        std::vector<OpL2DataInfo> &op_l2_data_vec) {
  FE_LOGD("Begin cleaning reference data for node [%s, %s].", node->GetName().c_str(), node->GetType().c_str());
  size_t output_size = node->GetOpDesc()->GetOutputsSize();
  for (size_t i = 0; i < output_size; ++i) {
    uint64_t data_id = DATA_OVERALL_ID(static_cast<uint64_t>(node->GetOpDesc()->GetId()), OUTPUT_DATA,
                                       static_cast<uint64_t>(i));
    OpReferTensorIndexMap::const_iterator iter = refer_tensor_index_map.find(data_id);
    if (iter != refer_tensor_index_map.end()) {
      FE_LOGD("The reference of output[%zu] from node[%s, %s] is input[%u].",
              i, node->GetName().c_str(), node->GetType().c_str(), iter->second);
      ge::InDataAnchorPtr in_data_anchor = node->GetInDataAnchor(static_cast<int32_t>(iter->second));
      ClearReferTensorData(in_data_anchor, refer_tensor_index_map, op_l2_data_vec);
    }
  }
}

bool L2FusionParser::IsNotSupportOp(const ge::NodePtr &node) {
  ge::OpDescPtr op_desc = node->GetOpDesc();
  if (CheckVirtualOp(op_desc)) {
    FE_LOGD("Node %s is a virtual node; do not allocate l2 buffer.", node->GetName().c_str());
    return true;
  }

  if (HasAtomicNode(node)) {
    FE_LOGD("Node %s has an atomic input node; do not allocate L2 buffer.", node->GetName().c_str());
    return true;
  }
  return false;
}

bool L2FusionParser::NoNeedAllocOpL2Info(const ge::NodePtr &node) {
  // jump over CONSTANT and other non-valid node
  if (L2FusionComm::IsNonValidOpOrData(node)) {
    return true;
  }
  return IsNotSupportOp(node);
}

bool L2FusionParser::NoNeedAllocOutputs(const ge::NodePtr &node) {
  if (node->GetOutDataNodes().size() == 0) {
    FE_LOGD("Node [%s] has no output data nodes; skipping L2 buffer allocation for its outputs.", node->GetName().c_str());
    return true;
  }
  for (const auto &anchor : node->GetAllOutDataAnchors()) {
    if (anchor != nullptr) {
      for (const auto &dst_anchor : anchor->GetPeerInDataAnchors()) {
        FE_CHECK(dst_anchor == nullptr,
                 REPORT_FE_ERROR("[StreamOpt][L2Opt][NoNeedAllocOut] Node %s peer in anchor is null.",
                 node->GetName().c_str()),
                 return FAILED);
        auto peer_node = dst_anchor->GetOwnerNode();
        if (peer_node->GetType() == OP_TYPE_END) {
          FE_LOGD("Node %s is the final node; do not allocate L2 buffer for its outputs.", node->GetName().c_str());
          return true;
        }
      }
    }
  }

  for (const auto &next_node : node->GetOutNodes()) {
    if (IsNotSupportOp(next_node)) {
      FE_LOGD("Next node %s don't alloc l2 buffer, so don't alloc l2 buffer for node:%s outputs.",
              next_node->GetName().c_str(), node->GetName().c_str());
      return true;
    }
  }
  return false;
}

bool L2FusionParser::NoNeedAllocOutput(const ge::GeTensorDesc &tensor_desc) {
  return IsMemoryEmpty(tensor_desc);
}

uint64_t L2FusionParser::GetDataUnitDataId(const ge::NodePtr &node, const size_t &tensor_index,
                                           const uint32_t &data_type, const ge::GeTensorDesc &tensor) {
  int64_t node_id = node->GetOpDesc()->GetId();
  // output data
  if (data_type == OUTPUT_DATA) {
    bool is_output_tensor = false;
    FE_CHECK(ge::TensorUtils::GetOutputTensor(tensor, is_output_tensor) != ge::GRAPH_SUCCESS,
             REPORT_FE_ERROR("[StreamOpt][L2Opt][NoNeedAllocOut] GetOutputTensor failed!"), return FAILED);
    if (!is_output_tensor) {  // the data is not graph's output
      return DATA_OVERALL_ID(static_cast<uint64_t>(node_id), OUTPUT_DATA, static_cast<uint64_t>(tensor_index));
    }
  }

  // input data, only src in graph can be allocted
  if (data_type == INPUT_DATA) {
    ge::InDataAnchorPtr in_anchor = node->GetInDataAnchor(static_cast<int32_t>(tensor_index));
    FE_CHECK(in_anchor == nullptr,
             REPORT_FE_ERROR("[StreamOpt][L2Opt][NoNeedAllocOut] Node %s in anchor is null.", node->GetName().c_str()),
             return FAILED);
    ge::OutDataAnchorPtr peer_out_anchor = in_anchor->GetPeerOutAnchor();
    if (peer_out_anchor != nullptr) {
      ge::OpDescPtr peer_out_op = peer_out_anchor->GetOwnerNode()->GetOpDesc();
      int64_t peer_out_node_id = peer_out_op->GetId();
      return DATA_OVERALL_ID(static_cast<uint64_t>(peer_out_node_id), OUTPUT_DATA,
                             static_cast<uint64_t>(peer_out_anchor->GetIdx()));  // find the src data id
    }
  }

  // left data without src node
  return -1;
}

void L2FusionParser::GetOpReferTensorIndexMap(const ge::ComputeGraph &graph,
                                              OpReferTensorIndexMap &refer_tensor_index_map) {
  for (const ge::NodePtr &node : graph.GetDirectNode()) {
    // jump over CONSTANT and other non-valid node
    if (L2FusionComm::IsNonValidOp(node)) {
      continue;
    }
    ge::OpDescPtr op_desc = node->GetOpDesc();
    OpKernelInfoPtr op_kernel_info_ptr = OpsKernelManager::Instance(AI_CORE_NAME).GetOpKernelInfoByOpDesc(op_desc);
    if (op_kernel_info_ptr == nullptr) {
      continue;
    }

    if (op_kernel_info_ptr->GetReferTensorNameVec().empty()) {
      continue;
    }
    for (const std::string &tensor_name : op_kernel_info_ptr->GetReferTensorNameVec()) {
      int32_t output_index = op_desc->GetOutputIndexByName(tensor_name);
      int32_t input_index = op_desc->GetInputIndexByName(tensor_name);
      if (output_index >= 0 && input_index >= 0) {
        uint64_t data_id = DATA_OVERALL_ID(static_cast<uint64_t>(op_desc->GetId()), OUTPUT_DATA,
                                           static_cast<uint32_t>(output_index));
        refer_tensor_index_map.emplace(data_id, static_cast<uint32_t>(input_index));
      }
    }
  }
}
}  // namespace fe
