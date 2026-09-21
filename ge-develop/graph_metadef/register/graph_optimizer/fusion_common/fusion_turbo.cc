/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "register/graph_optimizer/fusion_common/fusion_turbo.h"
#include "graph/operator_factory.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/debug/ge_attr_define.h"

#include <array>

namespace fe {
const std::string kNetOutput = "NetOutput";
WeightInfo::WeightInfo(const ge::GeTensorDesc &tensor_desc, void *data_p)
    : data(reinterpret_cast<uint8_t*>(data_p)) {
  shape = tensor_desc.GetShape();
  ori_shape = tensor_desc.GetOriginShape();
  datatype = tensor_desc.GetDataType();
  ori_datatype = tensor_desc.GetOriginDataType();
  format = tensor_desc.GetFormat();
  ori_format = tensor_desc.GetOriginFormat();
  CalcTotalDataSize();
}

WeightInfo::WeightInfo(const ge::NodePtr &node, const int32_t &index, void *data_p)
    : data(reinterpret_cast<uint8_t*>(data_p)) {
  if (node == nullptr) {
    return;
  }
  const auto tensor = node->GetOpDesc()->MutableInputDesc(static_cast<uint32_t>(index));
  if (tensor == nullptr) {
    return;
  }
  shape = tensor->GetShape();
  ori_shape = tensor->GetOriginShape();
  datatype = tensor->GetDataType();
  ori_datatype = tensor->GetOriginDataType();
  format = tensor->GetFormat();
  ori_format = tensor->GetOriginFormat();
  CalcTotalDataSize();
}

WeightInfo::WeightInfo(const ge::GeShape &shape_p, const ge::GeShape &ori_shape_p,
                       const ge::DataType &datatype_p, const ge::DataType &ori_datatype_p,
                       const ge::Format &format_p, const ge::Format &ori_format_p, void *data_p)
    : shape(shape_p),
      ori_shape(ori_shape_p),
      datatype(datatype_p),
      ori_datatype(ori_datatype_p),
      format(format_p),
      ori_format(ori_format_p),
      data(reinterpret_cast<uint8_t*>(data_p)) {
  CalcTotalDataSize();
}

WeightInfo::WeightInfo(ge::GeShape &&shape_p, ge::GeShape &&ori_shape_p,
                       const ge::DataType &datatype_p, const ge::DataType &ori_datatype_p,
                       const ge::Format &format_p, const ge::Format &ori_format_p, void *data_p)
    : shape(std::move(shape_p)),
      ori_shape(std::move(ori_shape_p)),
      datatype(datatype_p),
      ori_datatype(ori_datatype_p),
      format(format_p),
      ori_format(ori_format_p),
      data(reinterpret_cast<uint8_t*>(data_p)) {
  CalcTotalDataSize();
}

WeightInfo::WeightInfo(const ge::GeShape &shape_p, const ge::DataType &datatype_p,
                       const ge::Format &format_p, void *data_p)
    : shape(shape_p),
      ori_shape(shape_p),
      datatype(datatype_p),
      ori_datatype(datatype_p),
      format(format_p),
      ori_format(format_p),
      data(reinterpret_cast<uint8_t*>(data_p)) {
  CalcTotalDataSize();
}

WeightInfo::WeightInfo(ge::GeShape &&shape_p, const ge::DataType &datatype_p,
                       const ge::Format &format_p, void *data_p)
    :shape(std::move(shape_p)),
     ori_shape(shape_p),
     datatype(datatype_p),
     ori_datatype(datatype_p),
     format(format_p),
     ori_format(format_p),
     data(reinterpret_cast<uint8_t*>(data_p)) {
  CalcTotalDataSize();
}

FusionTurbo::FusionTurbo(const ge::ComputeGraphPtr &graph) : graph_(graph) {}

FusionTurbo::FusionTurbo(ge::ComputeGraph &graph) : graph_(graph.shared_from_this()) {}

FusionTurbo::~FusionTurbo() {}

Status FusionTurbo::BreakInput(const ge::NodePtr &node,
                               const vector<int32_t> &input_index) {
  for (const auto &index : input_index) {
    const auto in_anchor = node->GetInDataAnchor(index);
    if (in_anchor == nullptr) {
      continue;
    }

    in_anchor->UnlinkAll();
  }
  return SUCCESS;
}

Status FusionTurbo::BreakOutput(const ge::NodePtr &node,
                                const vector<int32_t> &output_index) {
  for (const auto &index : output_index) {
    const auto out_anchor = node->GetOutDataAnchor(index);
    if (out_anchor == nullptr) {
      continue;
    }

    out_anchor->UnlinkAll();
  }
  return SUCCESS;
}

Status FusionTurbo::BreakAllInput(const ge::NodePtr &node) {
  const auto input_anchors = node->GetAllInDataAnchors();
  for (const auto &in_anchor : input_anchors) {
    if (in_anchor == nullptr) {
      continue;
    }

    in_anchor->UnlinkAll();
  }
  return SUCCESS;
}

Status FusionTurbo::BreakAllOutput(const ge::NodePtr &node) {
  const auto output_anchors = node->GetAllOutDataAnchors();
  for (const auto &out_anchor : output_anchors) {
    if (out_anchor == nullptr) {
      continue;
    }

    out_anchor->UnlinkAll();
  }
  return SUCCESS;
}

Status FusionTurbo::RemoveNodeWithRelink(const ge::NodePtr &node, const std::initializer_list<int32_t> &io_map) {
  return RemoveNodeWithRelink(node, std::vector<int32_t>(io_map));
}

Status FusionTurbo::RemoveNodeWithRelink(const ge::NodePtr &node, const std::vector<int32_t> &io_map) {
  FUSION_TURBO_NOTNULL(node, PARAM_INVALID);
  if (ge::GraphUtils::IsolateNode(node, io_map) != ge::GRAPH_SUCCESS) {
    return FAILED;
  }

  if (ge::GraphUtils::RemoveNodeWithoutRelink(graph_, node) != ge::GRAPH_SUCCESS) {
    return FAILED;
  }

  return SUCCESS;
}

/* Just remove the node and all its relative data and control anchors. */
Status FusionTurbo::RemoveNodeOnly(const ge::NodePtr &node) {
  FUSION_TURBO_NOTNULL(node, PARAM_INVALID);
  ge::NodeUtils::UnlinkAll(*node);

  if (ge::GraphUtils::RemoveNodeWithoutRelink(graph_, node) != ge::GRAPH_SUCCESS) {
    return FAILED;
  }
  return SUCCESS;
}

Status FusionTurbo::RemoveDanglingNode(const ge::NodePtr &node, const bool &only_care_data_nodes) {
  FUSION_TURBO_NOTNULL(node, PARAM_INVALID);
  bool able_to_remove = false;
  if (only_care_data_nodes) {
    if (!HasOutData(node)) {
      able_to_remove = true;
    }
  } else {
    if (!HasOutData(node) && !HasOutControl(node)) {
      able_to_remove = true;
    }
  }
  if (able_to_remove) {
    return RemoveNodeOnly(node);
  }
  return FAILED;
}

Status FusionTurbo::RemoveMultiNodesOnly(const std::vector<ge::NodePtr> &nodes) {
  for (const auto &ele : nodes) {
    if (RemoveNodeOnly(ele) != SUCCESS) {
      return FAILED;
    }
  }
  return SUCCESS;
}

static ge::GeTensorPtr GenerateWeightTensor(const WeightInfo &w_info) {
  ge::GeTensorDesc new_weight_tensor;
  new_weight_tensor.SetShape(w_info.shape);
  new_weight_tensor.SetDataType(w_info.datatype);
  new_weight_tensor.SetFormat(w_info.format);
  new_weight_tensor.SetOriginShape(w_info.ori_shape);
  new_weight_tensor.SetOriginDataType(w_info.ori_datatype);
  new_weight_tensor.SetOriginFormat(w_info.ori_format);
  if (w_info.total_data_size == 0) {
    return nullptr;
  }
  ge::GeTensorPtr w = nullptr;
  GE_MAKE_SHARED(w = std::make_shared<ge::GeTensor>(
      new_weight_tensor,
      reinterpret_cast<uint8_t *>(w_info.data), w_info.total_data_size),
                 return nullptr);
  return w;
}

static void UpdateTensor(const ge::GeTensorDescPtr &tensor, const WeightInfo &w_info) {
  if (tensor == nullptr) {
    return;
  }
  tensor->SetDataType(w_info.datatype);
  tensor->SetOriginDataType(w_info.ori_datatype);
  tensor->SetFormat(w_info.format);
  tensor->SetOriginFormat(w_info.ori_format);
  tensor->SetShape(w_info.shape);
  tensor->SetOriginShape(w_info.ori_shape);
}

ge::NodePtr FusionTurbo::AddConstNode(const ge::NodePtr &node, const WeightInfo &w_info,
                                      const int32_t index) const {
  const auto node_in_tensor = std::const_pointer_cast<ge::GeTensorDesc>(
                              node->GetOpDesc()->GetInputDescPtrDfault(static_cast<uint32_t>(index)));
  FUSION_TURBO_NOTNULL(node_in_tensor, nullptr);
  UpdateTensor(node_in_tensor, w_info);

  ge::GeTensorPtr const_out_tenosr = nullptr;
  GE_MAKE_SHARED(const_out_tenosr = std::make_shared<ge::GeTensor>(*node_in_tensor), return nullptr);

  const Status ret = const_out_tenosr->SetData(reinterpret_cast<uint8_t *>(w_info.data),
                                               w_info.total_data_size);
  if (ret != SUCCESS) {
    GELOGE(FAILED, "[FusionTurbo][AddWeight][AddConstNode] Failed to set data.");
    return nullptr;
  }
  ge::OpDescPtr const_op_desc = ge::OpDescUtils::CreateConstOp(const_out_tenosr);

  auto const_node = graph_->AddNode(const_op_desc);
  if (const_node == nullptr) {
    GELOGE(FAILED, "[FusionTurbo][AddWeight][AddConstNode] Failed to add const node.");
    return nullptr;
  }

  GELOGD("Successfully created const input [%s] for node [%s].", const_op_desc->GetName().c_str(),
         node->GetName().c_str());
  if (ge::GraphUtils::AddEdge(const_node->GetOutDataAnchor(0), node->GetInDataAnchor(index)) != SUCCESS) {
    GELOGE(FAILED, "[FusionTurbo][AddWeight][AddConstNode] Failed to add edge between const %s and index %d of %s.",
           const_node->GetName().c_str(), index, node->GetName().c_str());
  }

  return const_node;
}

ge::NodePtr FusionTurbo::UpdateConst(const ge::NodePtr &node, const int32_t &index,
                                     const WeightInfo &w_info) const {
  auto const_node = FusionTurboUtils::GetConstInput(node, index);
  if (const_node == nullptr) {
    return nullptr;
  }
  const auto const_op = const_node->GetOpDesc();
  const auto const_out_tensor_desc = const_op->MutableOutputDesc(0);
  UpdateTensor(const_out_tensor_desc, w_info);

  const auto node_in_tensor = node->GetOpDesc()->MutableInputDesc(static_cast<uint32_t>(index));
  FUSION_TURBO_NOTNULL(node_in_tensor, nullptr);
  UpdateTensor(node_in_tensor, w_info);

  std::vector<ge::GeTensorPtr> weights = ge::OpDescUtils::MutableWeights(const_node);
  /* Substitute the const value with a new one. */
  Status ret;
  if (weights.empty()) {
    GELOGD("The weight for %s is missing; creating a new one.", const_node->GetName().c_str());
    ge::GeTensorPtr const_out = nullptr;
    FUSION_TURBO_NOTNULL(const_out_tensor_desc, nullptr);
    GE_MAKE_SHARED(const_out = std::make_shared<ge::GeTensor>(*const_out_tensor_desc), return nullptr);
    if (w_info.data == nullptr) {
      GELOGE(FAILED, "[FusionTurbo][AddWeight][UpdateConst] Data is null.");
      return nullptr;
    }
    ret = const_out->SetData(reinterpret_cast<uint8_t *>(w_info.data), w_info.total_data_size);
  } else {
    GELOGD("The weight for %s is not null, updating data.", const_node->GetName().c_str());
    ge::GeTensorPtr &const_out = weights.at(0);
    FUSION_TURBO_NOTNULL(const_out_tensor_desc, nullptr);
    const_out->SetTensorDesc(*const_out_tensor_desc);
    ret = const_out->SetData(reinterpret_cast<uint8_t *>(w_info.data), w_info.total_data_size);
  }
  if (ret != SUCCESS) {
    GELOGE(FAILED, "[FusionTurbo][AddWeight][UpdateConst] Failed to set data.");
    return nullptr;
  }

  return const_node;
}

ge::NodePtr FusionTurbo::AddWeightAfter(const ge::NodePtr &node, const int32_t &index,
                                        const WeightInfo &w_info) const {
  FUSION_TURBO_NOTNULL(node, nullptr);
  const auto output_anchor = node->GetOutDataAnchor(index);
  FUSION_TURBO_NOTNULL(output_anchor, nullptr);
  const auto peer_in_anchors = output_anchor->GetPeerInDataAnchors();
  if (peer_in_anchors.empty()) {
    GELOGD("Node %s does not have peer in anchors.", node->GetName().c_str());
    return nullptr;
  }

  const auto& first_peer_in_anchor = peer_in_anchors.at(0);
  const auto first_peer_in_node = first_peer_in_anchor->GetOwnerNode();
  FUSION_TURBO_NOTNULL(first_peer_in_node, nullptr);
  Relations output_relation(0, {node, index, PEER});

  output_anchor->UnlinkAll();
  /* Add weight in front of first peer input of node. */
  auto const_node = AddWeight(first_peer_in_node, first_peer_in_anchor->GetIdx(), w_info);
  FUSION_TURBO_NOTNULL(const_node, nullptr);

  if (LinkOutput(output_relation, const_node) != SUCCESS) {
    return nullptr;
  }
  return const_node;
}

ge::NodePtr FusionTurbo::AddWeight(const ge::NodePtr &node, const int32_t &index, const WeightInfo &w_info) const {
  FUSION_TURBO_NOTNULL(node, nullptr);
  const size_t input_size = node->GetAllInDataAnchorsSize();
  if (static_cast<size_t>(index) >= input_size) {
    GELOGD("Index %d is larger than input size %zu of %s.", index, input_size, node->GetName().c_str());
    return AddWeight(node, w_info);
  } else {
    const auto in_anchor = node->GetInDataAnchor(index);
    /* 1. If the peer node of this input index is nullptr, we add a const node
     *    as input and update tensor desc. */
    FUSION_TURBO_NOTNULL(in_anchor, nullptr);
    if (in_anchor->GetPeerOutAnchor() == nullptr) {
      auto const_node = AddConstNode(node, w_info, index);
      return const_node;
    }

    /* 2. If the peer node of this input index is Const, we substitute the data
     *    of current Const and update tensor desc. */
    return UpdateConst(node, index, w_info);
  }
}

ge::NodePtr FusionTurbo::AddWeight(const ge::NodePtr &node, const string& tensor_name, const WeightInfo &w_info) const {
  FUSION_TURBO_NOTNULL(node, nullptr);
  const auto index = node->GetOpDesc()->GetInputIndexByName(tensor_name);
  if (index == -1) {
    return nullptr;
  }
  return AddWeight(node, index, w_info);
}

ge::NodePtr FusionTurbo::AddWeight(const ge::NodePtr &node,
                                   const WeightInfo &w_info) const {
  FUSION_TURBO_NOTNULL(node, nullptr);
  /* 1. Collect all existing weights. */
  vector<ge::GeTensorPtr> weights = ge::OpDescUtils::MutableWeights(node);

  /* 2. Create new weight and link edges. */
  ge::GeTensorPtr w = GenerateWeightTensor(w_info);
  if (w == nullptr) {
    GELOGE(FAILED, "[FusionTurbo][AddWeight]Failed to generate weight for node %s.", node->GetName().c_str());
    return nullptr;
  }
  (void)weights.emplace_back(w);
  if (ge::OpDescUtils::SetWeights(node, weights) != ge::GRAPH_SUCCESS) {
    return nullptr;
  }

  /* 3. Return new weight node. */
  const auto in_size = static_cast<int32_t>(node->GetAllInDataAnchorsSize());
  const auto i = in_size - 1;
  return GetPeerOutNode(node, i);
}

std::vector<ge::NodePtr> FusionTurbo::AddWeights(const ge::NodePtr &node,
                                                 const vector<WeightInfo> &w_infos) const {
  std::vector<ge::NodePtr> ret;
  FUSION_TURBO_NOTNULL(node, ret);
  /* 1. Colloect all existing weights. */
  vector<ge::GeTensorPtr> weights = ge::OpDescUtils::MutableWeights(node);

  /* 2. Create new weights and link edges. */
  for (auto &w_info : w_infos) {
    ge::GeTensorPtr w = GenerateWeightTensor(w_info);
    if (w == nullptr) {
      GELOGE(FAILED, "[FusionTurbo][AddWeights]Failed to generate weight for node %s.", node->GetName().c_str());
      return ret;
    }
    (void)weights.emplace_back(w);
  }
  if (ge::OpDescUtils::SetWeights(node, weights) != ge::GRAPH_SUCCESS) {
    return ret;
  }

  /* 3. Return new weight nodes. */
  const auto in_size = static_cast<size_t>(node->GetAllInDataAnchorsSize());
  GELOGD("in_size %zu, w_info size %zu", in_size, w_infos.size());
  for (size_t i = in_size - w_infos.size(); i < in_size; i++) {
    auto peer_node = GetPeerOutNode(node, static_cast<int32_t>(i));
    (void)ret.emplace_back(peer_node);
  }
  return ret;
}

ge::GeTensorPtr FusionTurbo::MutableWeight(const ge::NodePtr &node, int32_t index) {
  FUSION_TURBO_NOTNULL(node, nullptr);
  const auto const_node = FusionTurboUtils::GetConstInput(node, index);
  if (const_node == nullptr) {
    return nullptr;
  }

  std::vector<ge::GeTensorPtr> weights = ge::OpDescUtils::MutableWeights(const_node);
  if (weights.empty()) {
    return nullptr;
  }

  return weights.at(0);
}

ge::NodePtr FusionTurbo::AddNodeOnly(const string &op_name, const string &op_type) const {
  return AddNodeOnly(*graph_, op_name, op_type, 0);
}

ge::NodePtr FusionTurbo::AddNodeOnly(ge::ComputeGraph &graph, const string &op_name, const string &op_type) {
  return AddNodeOnly(graph, op_name, op_type, 0);
}

ge::NodePtr FusionTurbo::AddNodeOnly(const string &op_name, const string &op_type, size_t dynamic_num) const {
  return AddNodeOnly(*graph_, op_name, op_type, dynamic_num);
}

ge::OpDescPtr FusionTurbo::CreateOpDesc(const string &op_name,
                                        const string &op_type, const size_t dynamic_num) {
  const auto op = ge::OperatorFactory::CreateOperator(op_name.c_str(), op_type.c_str());
  if (op.IsEmpty()) {
    GELOGW("Failed to create operator %s %s.", op_name.c_str(), op_type.c_str());
    return nullptr;
  }

  auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
  if (dynamic_num != 0) {
    size_t index = 0;
    const auto &ir_inputs = op_desc->GetIrInputs();
    for (auto &ir_input : ir_inputs) {
      if (ir_input.second == ge::kIrInputDynamic) {
        (void)op_desc->AddInputDescMiddle(ir_input.first, static_cast<uint32_t>(dynamic_num), index);
        index += dynamic_num;
      } else {
        ++index;
      }
    }

    index = 0U;
    const auto &ir_outputs = op_desc->GetIrOutputs();
    for (auto &ir_output : ir_outputs) {
      if (ir_output.second == ge::kIrOutputDynamic) {
        (void)op_desc->AddOutputDescMiddle(ir_output.first, static_cast<uint32_t>(dynamic_num), index);
        index += dynamic_num;
      } else {
        ++index;
      }
    }
  }
  return op_desc;
}

ge::NodePtr FusionTurbo::AddNodeOnly(ge::ComputeGraph &graph, const string &op_name,
                                     const string &op_type, size_t dynamic_num) {
  auto op_desc = CreateOpDesc(op_name, op_type, dynamic_num);
  auto ret_node = graph.AddNode(op_desc);
  return ret_node;
}

ge::NodePtr FusionTurbo::InsertNodeOnly(const string &op_name, const string &op_type,
                                        const ge::NodePtr &origin_node,
                                        const size_t dynamic_num) const {
  return InsertNodeOnly(*graph_, op_name, op_type, origin_node, dynamic_num);
}

ge::NodePtr FusionTurbo::InsertNodeOnly(ge::ComputeGraph &graph, const string &op_name, const string &op_type,
                                        const ge::NodePtr &origin_node,
                                        const size_t dynamic_num) {
  auto op_desc = CreateOpDesc(op_name, op_type, dynamic_num);
  auto ret_node = graph.InsertNode(origin_node, op_desc);
  return ret_node;
}

ge::NodePtr FusionTurbo::InsertNodeBefore(const string &op_name, const string &op_type,
                                          const ge::NodePtr &base_node, const int32_t &base_input_index,
                                          const int32_t &input_index, const int32_t &output_index) const {
  FUSION_TURBO_NOTNULL(base_node, nullptr);
  const auto base_desc = base_node->GetOpDesc();
  const auto base_input = base_desc->MutableInputDesc(static_cast<uint32_t>(base_input_index));
  FUSION_TURBO_NOTNULL(base_input, nullptr);

  /* 1. Create new operator, OpDesc and Node. */
  const auto op = ge::OperatorFactory::CreateOperator(op_name.c_str(), op_type.c_str());
  if (op.IsEmpty()) {
    GELOGE(FAILED, "[FusionTurbo][InstNodeBefore]Cannot find this op %s in op factory.", op_type.c_str());
    return nullptr;
  }

  const auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
  auto ret_node = graph_->AddNode(op_desc);

  const auto base_in_anchor = base_node->GetInDataAnchor(base_input_index);
  FUSION_TURBO_NOTNULL(base_in_anchor, nullptr);
  const auto peer_out_anchor = base_in_anchor->GetPeerOutAnchor();
  /* 2. Update Output desc using base node's successor node. */
  if (op_desc->UpdateOutputDesc(static_cast<uint32_t>(output_index), *base_input) != ge::GRAPH_SUCCESS) {
    GELOGE(FAILED, "[FusionTurbo][InstNodeBefore]Failed to update output %d of node %s", output_index, op_name.c_str());
    goto failed_process;
  }

  if (peer_out_anchor != nullptr) {
    const auto peer_out_index = peer_out_anchor->GetIdx();
    const auto peer_output = peer_out_anchor->GetOwnerNode()->GetOpDesc()->MutableOutputDesc(
        static_cast<uint32_t>(peer_out_index));
    FUSION_TURBO_NOTNULL(peer_output, nullptr);
    /* 3. Update input desc using base node's father node. */
    if (op_desc->UpdateInputDesc(static_cast<uint32_t>(input_index), *peer_output) != ge::GRAPH_SUCCESS) {
      GELOGE(FAILED, "[FusionTurbo][InstNodeBefore]Failed to update input %d of node %s", input_index, op_name.c_str());
      goto failed_process;
    }

    /* 4.1. Insert new op into graph and between peer-out and base-in anchors. */
    if (ge::GraphUtils::InsertNodeBefore(base_in_anchor, ret_node, static_cast<uint32_t>(input_index),
                                         static_cast<uint32_t>(output_index)) != ge::GRAPH_SUCCESS) {
      goto failed_process;
    }
  } else {
    GELOGD("Input %d of base node %s does not have peer out node.", base_input_index, base_node->GetName().c_str());
    /* 4.2. Just insert new op before base-in anchor. */
    FUSION_TURBO_NOTNULL(ret_node, nullptr);
    const auto out_anchor = ret_node->GetOutDataAnchor(output_index);
    FUSION_TURBO_NOTNULL(out_anchor, nullptr);
    if (ge::GraphUtils::AddEdge(out_anchor, base_in_anchor) != ge::GRAPH_SUCCESS) {
      goto failed_process;
    }
  }
  GELOGD("Succeed inserting %s before %s.", op_name.c_str(), base_node->GetName().c_str());
  return ret_node;

failed_process:
  graph_->RemoveNode(ret_node);
  return nullptr;
}

ge::NodePtr FusionTurbo::InsertNodeAfter(const string &op_name, const string &op_type, const ge::NodePtr &base_node,
                                         const int32_t &base_output_index, const int32_t &input_index,
                                         const int32_t &output_index) const {
  FUSION_TURBO_NOTNULL(base_node, nullptr);
  const auto base_desc = base_node->GetOpDesc();
  const auto base_output = base_desc->MutableOutputDesc(static_cast<uint32_t>(base_output_index));
  FUSION_TURBO_NOTNULL(base_output, nullptr);

  const auto base_out_anchor = base_node->GetOutDataAnchor(base_output_index);
  FUSION_TURBO_NOTNULL(base_out_anchor, nullptr);
  auto peer_in_anchors = base_out_anchor->GetPeerInDataAnchors();

  /* 1. Create new operator, OpDesc and Node. */
  const auto op = ge::OperatorFactory::CreateOperator(op_name.c_str(), op_type.c_str());
  if (op.IsEmpty()) {
    GELOGE(FAILED, "[FusionTurbo][InstNodeAfter]Cannot find this op %s in op factory.", op_type.c_str());
    return nullptr;
  }
  const auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
  auto ret_node = graph_->AddNode(op_desc);

  /* 2. Update input desc using base_node. */
  if (op_desc->UpdateInputDesc(static_cast<uint32_t>(input_index), *base_output) != ge::GRAPH_SUCCESS) {
    GELOGE(FAILED, "[FusionTurbo][InstNodeAfter]Failed to update input %d of node %s", input_index, op_name.c_str());
    goto failed_process;
  }

  if (!peer_in_anchors.empty()) {
    /* 3. Update output desc by peer input. */
    const auto peer_in_anchor = peer_in_anchors.at(0);
    const auto peer_in_index = peer_in_anchor->GetIdx();
    const auto peer_node = peer_in_anchor->GetOwnerNode();
    const auto peer_input = peer_node->GetOpDesc()->MutableInputDesc(static_cast<uint32_t>(peer_in_index));
    if (op_desc->UpdateOutputDesc(static_cast<uint32_t>(output_index), *peer_input) != ge::GRAPH_SUCCESS) {
      GELOGE(FAILED, "[FusionTurbo][InstNodeAfter]Failed to update output %d of node %s",
             output_index, op_name.c_str());
      goto failed_process;
    }

    /* 4.1. Insert new op between base-out anchor and every peer-in anchor. */
    const auto peer_in_anchors_vec = std::vector<ge::InDataAnchorPtr>(peer_in_anchors.begin(), peer_in_anchors.end());
    if (ge::GraphUtils::InsertNodeAfter(base_out_anchor, peer_in_anchors_vec, ret_node,
                                        static_cast<uint32_t>(input_index),
                                        static_cast<uint32_t>(output_index)) != ge::GRAPH_SUCCESS) {
      GELOGE(FAILED, "[FusionTurbo][InstNodeAfter]Failed to insert node after output %d of node %s",
             base_output_index, base_node->GetName().c_str());
      goto failed_process;
    }
  } else {
    GELOGD("Output %d of base node %s does not have a peer in the nodes.", base_output_index, base_node->GetName().c_str());
    FUSION_TURBO_NOTNULL(ret_node, nullptr);
    const auto in_anchor = ret_node->GetInDataAnchor(input_index);
    /* 4.2. Just insert new op after base-out anchor. */
    if (ge::GraphUtils::AddEdge(base_out_anchor, in_anchor) != ge::GRAPH_SUCCESS) {
      GELOGE(FAILED, "[FusionTurbo][InstNodeAfter]Failed to add edge between %d of %s and %d of %s",
             base_output_index, base_node->GetName().c_str(), input_index, op_name.c_str());
      goto failed_process;
    }
  }
  GELOGD("Succeed inserting %s after %s.", op_name.c_str(), base_node->GetName().c_str());
  return ret_node;
failed_process:
  graph_->RemoveNode(ret_node);
  return nullptr;
}

/* parent_node -> child_node */
static Status HandleTensorUpdate(const ge::NodePtr &parent_node, const ge::NodePtr &child_node,
                                 const uint32_t parent_index, const uint32_t child_index,
                                 const bool update_child) {
  if (update_child) {
    const auto parent_out_tensor_desc = parent_node->GetOpDesc()->MutableOutputDesc(parent_index);
    FUSION_TURBO_NOTNULL(parent_out_tensor_desc, FAILED);
    if (child_node->GetOpDesc()->UpdateInputDesc(child_index, *parent_out_tensor_desc) != ge::GRAPH_SUCCESS) {
      GELOGE(FAILED, "[FusionTurbo][LinkInput]Failed to update input %u of node %s",
             child_index, child_node->GetName().c_str());
      return FAILED;
    }
  } else {
    const auto child_in_tensor_desc = child_node->GetOpDesc()->MutableInputDesc(child_index);
    FUSION_TURBO_NOTNULL(child_in_tensor_desc, FAILED);
    if (parent_node->GetOpDesc()->UpdateOutputDesc(parent_index, *child_in_tensor_desc) != ge::GRAPH_SUCCESS) {
      GELOGE(FAILED, "[FusionTurbo][LinkOutput]Failed to update output %d of node %s",
             parent_index, parent_node->GetName().c_str());
      return FAILED;
    }
  }
  return SUCCESS;
}

Status FusionTurbo::LinkInput(Relations &input_relations,
                              const ge::NodePtr &dst_node,
                              const TensorUptType &update_tensor) {
  FUSION_TURBO_NOTNULL(dst_node, PARAM_INVALID);
  const auto dst_op_desc = dst_node->GetOpDesc();
  const auto &in_relations = input_relations.GetInRelations();
  if (in_relations.empty()) {
    GELOGD("dst_node %s's input relations is empty.", dst_node->GetName().c_str());
    return PARAM_INVALID;
  }

  const auto dst_input_size = dst_node->GetAllInDataAnchorsSize();
  for (const auto &relation : in_relations) {
    const auto dst_in_index = static_cast<uint32_t>(relation.first);
    if (dst_in_index >= dst_input_size) {
      GELOGW("Dst input index %u is larger than dst node %s's input size %u.",
             dst_in_index, dst_node->GetName().c_str(), dst_input_size);
      continue;
    }

    if (relation.second.empty()) {
      continue;
    }

    const auto src_node = relation.second.at(0).node;
    const auto src_out_index = relation.second.at(0).index;
    FUSION_TURBO_NOTNULL(src_node, PARAM_INVALID);
    const auto out_anchor = src_node->GetOutDataAnchor(src_out_index);
    FUSION_TURBO_NOTNULL(out_anchor, PARAM_INVALID);
    /* 1. Update tensor descs. We assume the input desc of src node is correct. */
    if (update_tensor == UPDATE_THIS) {
      (void)HandleTensorUpdate(src_node, dst_node, static_cast<uint32_t>(src_out_index), dst_in_index, true);
    } else if (update_tensor == UPDATE_PEER) {
      (void)HandleTensorUpdate(src_node, dst_node, static_cast<uint32_t>(src_out_index), dst_in_index, false);
    } else {
      // do nothing
    }
    /* 2. Link anchors. */
    const auto dst_in_anchor = dst_node->GetInDataAnchor(static_cast<int32_t>(dst_in_index));
    if (ge::GraphUtils::AddEdge(out_anchor, dst_in_anchor) != ge::GRAPH_SUCCESS) {
      return FAILED;
    }
    GELOGD("SuccessFully link input %s %d ---> %s %d.", src_node->GetName().c_str(), src_out_index,
           dst_node->GetName().c_str(), dst_in_index);
  }
  return SUCCESS;
}

Status FusionTurbo::LinkOutput(Relations &output_relations, const ge::NodePtr &src_node,
                               const TensorUptType &update_tensor) {
  FUSION_TURBO_NOTNULL(src_node, PARAM_INVALID);
  const auto dst_op_desc = src_node->GetOpDesc();
  const auto &out_relations = output_relations.GetOutRelations();
  if (out_relations.empty()) {
    GELOGD("src_node %s's output relations is empty.", src_node->GetName().c_str());
    return PARAM_INVALID;
  }

  const auto src_op_desc = src_node->GetOpDesc();
  const auto src_output_size = src_node->GetAllOutDataAnchorsSize();

  for (auto &relation : out_relations) {
    const auto src_out_index = static_cast<uint32_t>(relation.first);
    if (src_out_index >= src_output_size) {
      GELOGW("Source output index %u is larger than src node %s's output size %u.",
             src_out_index, src_node->GetName().c_str(), src_output_size);
      continue;
    }

    if (relation.second.empty()) {
      continue;
    }

    for (const auto &ele: relation.second) {
      const auto dst_node = ele.node;
      const auto dst_index = ele.index;
      if (dst_node == nullptr) {
        continue;
      }

      /* 1. Update tensor descs. */
      if (update_tensor == UPDATE_THIS) {
        (void)HandleTensorUpdate(src_node, dst_node, src_out_index, static_cast<uint32_t>(dst_index), false);
      } else if (update_tensor == UPDATE_PEER) {
        (void)HandleTensorUpdate(src_node, dst_node, src_out_index, static_cast<uint32_t>(dst_index), true);
      } else {
        // do nothing
      }
      /* 2. Link all peer in anchors. */
      const auto in_anchor = dst_node->GetInDataAnchor(dst_index);
      FUSION_TURBO_NOTNULL(in_anchor, PARAM_INVALID);
      const auto peer_out = in_anchor->GetPeerOutAnchor();
      if (peer_out != nullptr) {
        GELOGD("Dst node %s's input %d already has a peer output [%s].", dst_node->GetName().c_str(), dst_index,
               peer_out->GetOwnerNode()->GetName().c_str());
        in_anchor->UnlinkAll();
      }

      const auto src_out_anchor = src_node->GetOutDataAnchor(static_cast<int32_t>(src_out_index));
      if (ge::GraphUtils::AddEdge(src_out_anchor, in_anchor) != ge::GRAPH_SUCCESS) {
        return FAILED;
      }
      GELOGD("SuccessFully link output %s %d ---> %s %d.", src_node->GetName().c_str(), src_out_index,
             dst_node->GetName().c_str(), dst_index);
    }
  }
  return SUCCESS;
}


ge::NodePtr FusionTurbo::GetPeerOutNode(const ge::NodePtr &node, const int32_t &this_node_input_index) {
  FUSION_TURBO_NOTNULL(node, nullptr);
  const auto input_anchor = node->GetInDataAnchor(this_node_input_index);
  FUSION_TURBO_NOTNULL(input_anchor, nullptr);
  const auto peer_out_anchor = input_anchor->GetPeerOutAnchor();
  FUSION_TURBO_NOTNULL(peer_out_anchor, nullptr);
  return peer_out_anchor->GetOwnerNode();
}

std::vector<ge::NodePtr> FusionTurbo::GetPeerInNodes(const ge::NodePtr &node, const int32_t &this_node_output_index) {
  std::vector<ge::NodePtr> ret;
  FUSION_TURBO_NOTNULL(node, ret);
  const auto output_anchor = node->GetOutDataAnchor(this_node_output_index);
  FUSION_TURBO_NOTNULL(output_anchor, ret);
  const auto peer_in_anchors = output_anchor->GetPeerInDataAnchors();
  for (const auto& ele : peer_in_anchors) {
    (void)ret.emplace_back(ele->GetOwnerNode());
  }

  return ret;
}

bool FusionTurbo::CheckConnected(const ge::NodePtr &node1, const ge::NodePtr &node2, const int32_t &index1) {
  FUSION_TURBO_NOTNULL(node1, false);
  FUSION_TURBO_NOTNULL(node2, false);
  if (index1 == -1) {
    const auto all_output_of_node1 = node1->GetOutDataNodes();
    for (const auto &out_node : all_output_of_node1) {
      return out_node == node2;
    }
  } else {
    auto peer_in_nodes = GetPeerInNodes(node1, index1);
    return (std::find(peer_in_nodes.begin(), peer_in_nodes.end(), node2) != peer_in_nodes.end());
  }
  return false;
}

Status FusionTurbo::UpdateInputByPeer(const ge::NodePtr &node, const int32_t &index,
                                      const ge::NodePtr &peer_node, const int32_t &peer_index) const {
  FUSION_TURBO_NOTNULL(node, PARAM_INVALID);
  FUSION_TURBO_NOTNULL(peer_node, PARAM_INVALID);

  const auto peer_output_desc = peer_node->GetOpDesc()->MutableOutputDesc(static_cast<uint32_t>(peer_index));
  FUSION_TURBO_NOTNULL(peer_output_desc, PARAM_INVALID);

  const auto input_desc = node->GetOpDesc()->MutableInputDesc(static_cast<uint32_t>(index));
  FUSION_TURBO_NOTNULL(input_desc, PARAM_INVALID);

  *input_desc = *peer_output_desc;

  return SUCCESS;
}

Status FusionTurbo::UpdateOutputByPeer(const ge::NodePtr &node, const int32_t &index,
                                       const ge::NodePtr &peer_node, const int32_t &peer_index) const {
  FUSION_TURBO_NOTNULL(node, PARAM_INVALID);
  FUSION_TURBO_NOTNULL(peer_node, PARAM_INVALID);

  const auto peer_input_desc = peer_node->GetOpDesc()->MutableInputDesc(static_cast<uint32_t>(peer_index));
  FUSION_TURBO_NOTNULL(peer_input_desc, PARAM_INVALID);

  const auto output_desc = node->GetOpDesc()->MutableOutputDesc(static_cast<uint32_t>(index));
  FUSION_TURBO_NOTNULL(output_desc, PARAM_INVALID);

  *output_desc = *peer_input_desc;
  return SUCCESS;
}

bool FusionTurbo::IsUnknownShape(const ge::NodePtr &node, const int32_t &index, const bool &is_input) {
  ge::GeTensorDescPtr tensor;
  if (is_input) {
    tensor = node->GetOpDesc()->MutableInputDesc(static_cast<uint32_t>(index));
  } else {
    tensor = node->GetOpDesc()->MutableOutputDesc(static_cast<uint32_t>(index));
  }
  FUSION_TURBO_NOTNULL(tensor, false);
  const auto &shape = tensor->MutableShape();
  return shape.IsUnknownShape();
}

bool FusionTurbo::IsUnknownOriShape(const ge::NodePtr &node, const int32_t &index, const bool &is_input) {
  ge::GeTensorDescPtr tensor;
  if (is_input) {
    tensor = node->GetOpDesc()->MutableInputDesc(static_cast<uint32_t>(index));
  } else {
    tensor = node->GetOpDesc()->MutableOutputDesc(static_cast<uint32_t>(index));
  }
  FUSION_TURBO_NOTNULL(tensor, false);
  const auto &shape = tensor->GetOriginShape();
  return shape.IsUnknownShape();
}

Status FusionTurbo::TransferOutCtrlEdges(const std::vector<ge::NodePtr> &nodes,
                                         const ge::NodePtr &new_node) {
  FUSION_TURBO_NOTNULL(new_node, FAILED);
  for (const auto &node : nodes) {
    if (node == nullptr) {
      continue;
    }
    const auto peer_in_ctrl_nodes = node->GetOutControlNodes();
    if (peer_in_ctrl_nodes.empty()) {
      continue;
    }

    for (const auto &in_node : peer_in_ctrl_nodes) {
      if (new_node == in_node) {
        GELOGD("Out Ctrl: Avoid same source and destination %s.", new_node->GetName().c_str());
        continue;
      }
      (void)ge::GraphUtils::AddEdge(new_node->GetOutControlAnchor(), in_node->GetInControlAnchor());
    }
  }
  return SUCCESS;
}

Status FusionTurbo::TransferInCtrlEdges(const std::vector<ge::NodePtr> &nodes,
                                        const ge::NodePtr &new_node) {
  FUSION_TURBO_NOTNULL(new_node, FAILED);
  for (const auto &node : nodes) {
    if (node == nullptr) {
      continue;
    }
    const auto peer_out_ctrl_nodes = node->GetInControlNodes();
    if (peer_out_ctrl_nodes.empty()) {
      continue;
    }

    for (const auto &out_node : peer_out_ctrl_nodes) {
      if (out_node == new_node) {
        GELOGD("In Ctrl: avoid same source and destination %s.", new_node->GetName().c_str());
        continue;
      }
      if (ge::GraphUtils::AddEdge(out_node->GetOutControlAnchor(), new_node->GetInControlAnchor()) !=
          ge::GRAPH_SUCCESS) {
        return FAILED;
      }
    }
  }
  return SUCCESS;
}

ge::NodePtr FusionTurbo::MultiInOne(const string &node_name, const string &node_type,
                                    Relations &input_relations,
                                    Relations &output_relations,
                                    const std::vector<ge::NodePtr> &old_nodes,
                                    const bool &remove_old) {
  auto node = AddNodeOnly(node_name, node_type);
  if (MultiInOne(node, input_relations, output_relations, old_nodes, remove_old) != SUCCESS) {
    (void)graph_->RemoveNode(node);
    return nullptr;
  }
  return node;
}

Status FusionTurbo::MultiInOne(const ge::NodePtr &new_node,
                               Relations &input_relations,
                               Relations &output_relations,
                               const std::vector<ge::NodePtr> &old_nodes,
                               const bool &remove_old) {
  FUSION_TURBO_NOTNULL(new_node, FAILED);
  GELOGD("Merged multiple nodes into %s.", new_node->GetName().c_str());
  /* Check params. */
  const auto &in_ori_relaitons = input_relations.GetRelations();
  if (in_ori_relaitons.size() > new_node->GetAllInDataAnchorsSize()) {
    GELOGE(FAILED, "[FusionTurbo][MultiInOne][ChkInput]Input relation size %zu is larger than %s's input size %u.",
           in_ori_relaitons.size(), new_node->GetName().c_str(), new_node->GetAllInDataAnchorsSize());
    return FAILED;
  }
  const auto &out_ori_relaitons = output_relations.GetRelations();
  if (out_ori_relaitons.size() > new_node->GetAllOutDataAnchorsSize()) {
    GELOGE(FAILED, "[FusionTurbo][MultiInOne][ChkOutput]Output relation size %zu is larger than %s's output size %u.",
           out_ori_relaitons.size(), new_node->GetName().c_str(), new_node->GetAllOutDataAnchorsSize());
    return FAILED;
  }

  /* Link data edges. */
  if (LinkInput(input_relations, new_node, UPDATE_THIS) != SUCCESS) {
    GELOGE(FAILED, "[FusionTurbo][MultiInOne][LnkIn]Failed to link input for node %s.", new_node->GetName().c_str());
    return FAILED;
  }

  if (output_relations.GetOutRelations().empty()) {
    GELOGD("[FusionTurbo][MultiInOne][LnkOut] Output relations is empty, skip for node %s.",
           new_node->GetName().c_str());
  } else if (LinkOutput(output_relations, new_node, UPDATE_THIS) != SUCCESS) {
    GELOGE(FAILED, "[FusionTurbo][MultiInOne][LnkOut]Failed to link output for node %s.", new_node->GetName().c_str());
    return FAILED;
  } else {
    // No return value expected.
  }

  /* Link control edges. */
  if (TransferInCtrlEdges(old_nodes, new_node) != SUCCESS) {
    return FAILED;
  }

  if (TransferOutCtrlEdges(old_nodes, new_node) != SUCCESS) {
    return FAILED;
  }

  if (remove_old) {
    for (auto &old_node : old_nodes) {
      (void)RemoveNodeOnly(old_node);
    }
  }
  return SUCCESS;
}

bool FusionTurbo::HasInControl(const ge::NodePtr &node) {
  FUSION_TURBO_NOTNULL(node, false);
  const auto in_control_anchor = node->GetInControlAnchor();
  for (const auto &peer_out_control_anchor : in_control_anchor->GetPeerOutControlAnchors()) {
    if (peer_out_control_anchor->GetOwnerNode() != nullptr) {
      return true;
    }
  }
  return false;
}

bool FusionTurbo::HasOutControl(const ge::NodePtr &node) {
  FUSION_TURBO_NOTNULL(node, false);
  const auto out_control_anchor = node->GetOutControlAnchor();
  for (const auto &peer_in_control_anchor : out_control_anchor->GetPeerInControlAnchors()) {
    if (peer_in_control_anchor->GetOwnerNode() != nullptr) {
      return true;
    }
  }
  return false;
}

bool FusionTurbo::HasOutData(const ge::NodePtr &node) {
  FUSION_TURBO_NOTNULL(node, false);
  const auto out_data_anchors = node->GetAllOutDataAnchors();
  for (const auto &out_anchor : out_data_anchors) {
    for (const auto &peer_in_data_anchor : out_anchor->GetPeerInDataAnchors()) {
      if (peer_in_data_anchor->GetOwnerNode() != nullptr) {
        return true;
      }
    }
  }
  return false;
}

bool FusionTurbo::HasControl(const ge::NodePtr &node) {
  return HasInControl(node) || HasOutControl(node);
}

Status FusionTurbo::MoveDataOutputUp(const ge::NodePtr &node, int32_t index) {
  const NodeIndex subgraph_node = FusionTurboUtils::GetPeerOutPair(node, index);
  FUSION_TURBO_NOTNULL(subgraph_node.node, FAILED);

  uint32_t subgraph_output_size = subgraph_node.node->GetAllOutDataAnchorsSize();
  const auto node_pair_peer_out_anchor = subgraph_node.node->GetOutDataAnchor(subgraph_node.index);
  ge::OutDataAnchorPtr out_link_anchor = node_pair_peer_out_anchor;

  // for multi outputs, first output move to current node output index, others need add new output anchor
  for (size_t node_outanchor_index = 0; node_outanchor_index < node->GetAllOutDataAnchorsSize();
       ++node_outanchor_index) {
    if (node_outanchor_index != 0) {
      (void)subgraph_node.node->GetOpDesc()->AddOutputDesc(node->GetOpDesc()->GetOutputDesc(
          static_cast<uint32_t>(node_outanchor_index)));
      out_link_anchor = subgraph_node.node->GetOutDataAnchor(static_cast<int32_t>(subgraph_output_size));
      ++subgraph_output_size;
    } else {
      FUSION_TURBO_NOTNULL(out_link_anchor, FAILED);
      (void)subgraph_node.node->GetOpDesc()->UpdateOutputDesc(static_cast<uint32_t>(out_link_anchor->GetIdx()),
          node->GetOpDesc()->GetOutputDesc(static_cast<uint32_t>(node_outanchor_index)));
    }
    const auto node_outanchor = node->GetOutDataAnchor(static_cast<int32_t>(node_outanchor_index));
    FUSION_TURBO_NOTNULL(node_outanchor, FAILED);
    for (auto &peer_in_anchor : node_outanchor->GetPeerInDataAnchors()) {
      if (peer_in_anchor->Unlink(node_outanchor) != ge::GRAPH_SUCCESS) {
        return FAILED;
      }
      if (peer_in_anchor->LinkFrom(out_link_anchor) != ge::GRAPH_SUCCESS) {
        return FAILED;
      }
    }
  }
  return SUCCESS;
}

static ge::NodePtr AddSubGraphDataWithIndex(const ge::ComputeGraphPtr &graph, const ge::GeTensorDesc &tensor_desc,
                                            const int32_t node_input_size) {
  const std::string data_name = "Data_" + std::to_string(node_input_size);
  auto data_node = FusionTurbo::AddNodeOnly(*graph, data_name, "Data");
  FUSION_TURBO_NOTNULL(data_node, nullptr);
  auto op_desc = data_node->GetOpDesc();
  (void)ge::AttrUtils::SetInt(op_desc, ge::ATTR_NAME_PARENT_NODE_INDEX, node_input_size);
  (void)op_desc->UpdateInputDesc(0U, tensor_desc);
  (void)op_desc->UpdateOutputDesc(0U, tensor_desc);
  return data_node;
}

static ge::NodePtr FindSubgraphData(const ge::ComputeGraphPtr &graph, const int32_t index) {
  ge::NodePtr pair_data_node = nullptr;
  for (auto &tmp_node : graph->GetDirectNode()) {
    int64_t ref_i;
    if ((tmp_node->GetType() == "Data") &&
        (ge::AttrUtils::GetInt(tmp_node->GetOpDesc(), ge::ATTR_NAME_PARENT_NODE_INDEX, ref_i)) &&
        (ref_i == static_cast<int64_t>(index))) {
      pair_data_node = tmp_node;
      break;
    }
  }
  return pair_data_node;
}

static int32_t GetNetOutputTensorIndex(const ge::NodePtr &node, const int32_t index) {
  int32_t tensor_index = 0;
  for (uint32_t input_index = 0; input_index < node->GetOpDesc()->GetAllInputsSize(); ++input_index) {
    int64_t parent_index = -1;
    const auto input_desc = node->GetOpDesc()->MutableInputDesc(input_index);
    (void)ge::AttrUtils::GetInt(input_desc, ge::ATTR_NAME_PARENT_NODE_INDEX, parent_index);
    if (parent_index == index) {
      tensor_index = static_cast<int32_t>(input_index);
      break;
    }
  }

  return tensor_index;
}

static Status MoveDataInputUpToSubgraph(const ge::NodePtr &node, const int32_t index, Relations &input_relations) {
  const NodeIndex subgraph_node = FusionTurboUtils::GetPeerOutPair(node, index);
  FUSION_TURBO_NOTNULL(subgraph_node.node, FAILED);
  const auto subgraph = ge::NodeUtils::GetSubgraph(*subgraph_node.node, 0);
  FUSION_TURBO_NOTNULL(subgraph, FAILED);
  const auto netout_node = subgraph->FindFirstNodeMatchType(kNetOutput);
  FUSION_TURBO_NOTNULL(netout_node, FAILED);

  const auto netout_tensor_index = GetNetOutputTensorIndex(netout_node, subgraph_node.index);
  uint32_t subgraph_node_input_size = subgraph_node.node->GetAllInDataAnchorsSize();

  /* for current node multi inputs, subgraph at move direction record the netout peer anchor
  ** other inputs need add new data node and record input
  */
  for (uint32_t node_inanchor_index = 0; node_inanchor_index < node->GetAllInDataAnchorsSize(); ++node_inanchor_index) {
    const auto node_inanchor = node->GetInDataAnchor(static_cast<int32_t>(node_inanchor_index));
    FUSION_TURBO_NOTNULL(node_inanchor, FAILED);
    const auto out_data_anchor = node_inanchor->GetPeerOutAnchor();
    if (out_data_anchor == nullptr) {
      continue;
    }
    (void)out_data_anchor->Unlink(node_inanchor);
    if (node_inanchor_index == static_cast<uint32_t>(index)) {
      (void)input_relations.Add(static_cast<int32_t>(node_inanchor_index), {netout_node, netout_tensor_index, PEER});
      continue;
    }
    if (ge::NodeUtils::AppendInputAnchor(subgraph_node.node, subgraph_node_input_size + 1) != ge::GRAPH_SUCCESS) {
      return FAILED;
    }
    FUSION_TURBO_NOTNULL(subgraph_node.node->GetInDataAnchor(static_cast<int32_t>(subgraph_node_input_size)), FAILED);
    if (subgraph_node.node->GetInDataAnchor(static_cast<int32_t>(subgraph_node_input_size))->LinkFrom(out_data_anchor)
        != ge::GRAPH_SUCCESS) {
      return FAILED;
    }

    const ge::NodePtr data_node = AddSubGraphDataWithIndex(subgraph, node->GetOpDesc()->GetInputDesc(node_inanchor_index),
                                                           static_cast<int32_t>(subgraph_node_input_size));
    (void)input_relations.Add(static_cast<int32_t>(node_inanchor_index), {data_node, 0});
    ++subgraph_node_input_size;
  }
  return SUCCESS;
}

Status FusionTurbo::GraphNodeUpMigration(const ge::NodePtr &node,
                                         const int32_t index) {
  if (HasControl(node)) {
    GELOGD("[FusionTurbo][GraphNodeUpMigration] node:%s has control anchors, cannot move", node->GetName().c_str());
    return NOT_CHANGED;
  }
  const NodeIndex pre_node_index = FusionTurboUtils::GetPeerOutPair(node, index);
  FUSION_TURBO_NOTNULL(pre_node_index.node, FAILED);
  const auto subgraph = ge::NodeUtils::GetSubgraph(*pre_node_index.node, 0);
  FUSION_TURBO_NOTNULL(subgraph, FAILED);
  const auto netout_node = subgraph->FindFirstNodeMatchType(kNetOutput);
  FUSION_TURBO_NOTNULL(netout_node, FAILED);
  const auto netout_tensor_index = GetNetOutputTensorIndex(netout_node, pre_node_index.index);

  /* move all data output up to parent subgraph node, clear current op all output */
  if (MoveDataOutputUp(node, index) != SUCCESS) {
    GELOGE(FAILED, "[FusionTurbo][GraphNodeUpMigration][MoveDataOutputUp] Failed to relink output for node:%s",
           node->GetName().c_str());
    return FAILED;
  }

  /* move all data input up to parent subgraph node, and record the input relationship for node to added in subgraph */
  Relations input_relations;
  if (MoveDataInputUpToSubgraph(node, index, input_relations) != SUCCESS) {
    GELOGE(FAILED, "[FusionTurbo][GraphNodeUpMigration][MoveDataInputUpToSubgraph] Failed to relink output for node:%s",
           node->GetName().c_str());
    return FAILED;
  }

  (void)RemoveNodeOnly(node);

  Relations output_relations(0, {netout_node, netout_tensor_index});
  (void)BreakInput(netout_node, {netout_tensor_index});

  const auto node_in_subgraph = subgraph->AddNode(node->GetOpDesc());
  FUSION_TURBO_NOTNULL(node_in_subgraph, FAILED);
  if (LinkInput(input_relations, node_in_subgraph, UPDATE_NONE) != SUCCESS) {
    GELOGE(FAILED, "[FusionTurbo][GraphNodeUpMigration][LnkIn] Failed to link input for node:%s",
           node_in_subgraph->GetName().c_str());
    return FAILED;
  }

  if (LinkOutput(output_relations, node_in_subgraph, UPDATE_NONE) != SUCCESS) {
    GELOGE(FAILED, "[FusionTurbo][GraphNodeUpMigration][LnkOut] Failed to link input for node:%s",
           node_in_subgraph->GetName().c_str());
    return FAILED;
  }
  return SUCCESS;
}

static Status MoveDataInputDownToSubgraph(const ge::NodePtr &node, const int32_t index, Relations &input_relations) {
  const NodeIndex out_node_index = FusionTurboUtils::GetPeerInFirstPair(node, index);
  FUSION_TURBO_NOTNULL(out_node_index.node, FAILED);
  const auto subgraph = ge::NodeUtils::GetSubgraph(*out_node_index.node, 0U);
  FUSION_TURBO_NOTNULL(subgraph, FAILED);
  const ge::NodePtr pair_data_node = FindSubgraphData(subgraph, out_node_index.index);
  FUSION_TURBO_NOTNULL(pair_data_node, FAILED);

  uint32_t subgraph_node_input_size = out_node_index.node->GetAllInDataAnchorsSize();
  ge::InDataAnchorPtr linkin_anchor = out_node_index.node->GetInDataAnchor(out_node_index.index);
  FUSION_TURBO_NOTNULL(linkin_anchor, FAILED);
  linkin_anchor->UnlinkAll();

  // for multi inputs, first input connect to current node data in subgraph, others need create new data node
  for (uint32_t node_inanchor_index = 0; node_inanchor_index < node->GetAllInDataAnchorsSize(); ++node_inanchor_index) {
    const auto input_tensor_desc = node->GetOpDesc()->GetInputDesc(node_inanchor_index);
    if (node_inanchor_index != 0) {
      if (ge::NodeUtils::AppendInputAnchor(out_node_index.node, subgraph_node_input_size + 1) != ge::GRAPH_SUCCESS) {
        return FAILED;
      }
      linkin_anchor = out_node_index.node->GetInDataAnchor(static_cast<int32_t>(subgraph_node_input_size));
      subgraph_node_input_size++;
    } else {
      (void)out_node_index.node->GetOpDesc()->UpdateInputDesc(static_cast<uint32_t>(linkin_anchor->GetIdx()),
          input_tensor_desc);
    }
    const auto node_inanchor = node->GetInDataAnchor(static_cast<int32_t>(node_inanchor_index));
    FUSION_TURBO_NOTNULL(node_inanchor, FAILED);
    const auto peer_out_anchor = node_inanchor->GetPeerOutAnchor();
    FUSION_TURBO_NOTNULL(peer_out_anchor, FAILED);
    if (peer_out_anchor->Unlink(node_inanchor) != ge::GRAPH_SUCCESS) {
      return FAILED;
    }
    if (peer_out_anchor->LinkTo(linkin_anchor) != ge::GRAPH_SUCCESS) {
      return FAILED;
    }
    ge::NodePtr data_node(pair_data_node);
    if (node_inanchor_index != 0) {
      data_node = AddSubGraphDataWithIndex(subgraph, input_tensor_desc,
                                           static_cast<int32_t>(subgraph_node_input_size) - 1);
    }
    (void)input_relations.Add(static_cast<int32_t>(node_inanchor_index), {data_node, 0});
  }
  return SUCCESS;
}

Status FusionTurbo::GraphNodeDownMigration(const ge::NodePtr &node,
                                           const int32_t index) {
  if ((node->GetOutDataNodesSize() != 1) || (HasControl(node))) {
    GELOGD("[FusionTurbo][GraphNodeDownMigration] Node: %s has multiple outputs or contains control nodes, cannot migrate",
           node->GetName().c_str());
    return NOT_CHANGED;
  }
  const NodeIndex out_node_index = FusionTurboUtils::GetPeerInFirstPair(node, index);
  FUSION_TURBO_NOTNULL(out_node_index.node, FAILED);
  const auto subgraph = ge::NodeUtils::GetSubgraph(*out_node_index.node, 0U);
  FUSION_TURBO_NOTNULL(subgraph, FAILED);
  const ge::NodePtr pair_data_node = FindSubgraphData(subgraph, out_node_index.index);
  FUSION_TURBO_NOTNULL(pair_data_node, FAILED);

  /* move data input down to subgraph node, and record input relationship for node to be added in subgraph */
  Relations input_relations;
  if (MoveDataInputDownToSubgraph(node, index, input_relations) != SUCCESS) {
    GELOGE(FAILED, "[FusionTurbo][GraphNodeDownMigration][MoveDataInputDownSubgraph] Failed to link input for node:%s",
           node->GetName().c_str());
    return FAILED;
  }
  (void)BreakOutput(node, {index});

  Relations output_relations(0, {pair_data_node, 0, PEER});
  (void)BreakOutput(pair_data_node, {0});
  (void)RemoveNodeOnly(node);

  const auto node_in_subgraph = subgraph->AddNode(node->GetOpDesc());
  FUSION_TURBO_NOTNULL(node_in_subgraph, FAILED);
  if (LinkInput(input_relations, node_in_subgraph, UPDATE_NONE) != SUCCESS) {
    GELOGE(FAILED, "[FusionTurbo][GraphNodeDownMigration][LnkIn] Failed to link input for node:%s",
           node_in_subgraph->GetName().c_str());
    return FAILED;
  }
  if (LinkOutput(output_relations, node_in_subgraph, UPDATE_NONE) != SUCCESS) {
    GELOGE(FAILED, "[FusionTurbo][GraphNodeDownMigration][LnkOut] Failed to link input for node:%s",
           node_in_subgraph->GetName().c_str());
    return FAILED;
  }
  return SUCCESS;
}

NodeIndex FusionTurbo::GetPeerInFirstPair(const ge::NodePtr &node, int32_t index) {
  return FusionTurboUtils::GetPeerInFirstPair(node, index);
}

NodeIndex FusionTurbo::GetPeerOutPair(const ge::NodePtr &node, int32_t index) {
  return FusionTurboUtils::GetPeerOutPair(node, index);
}
}
