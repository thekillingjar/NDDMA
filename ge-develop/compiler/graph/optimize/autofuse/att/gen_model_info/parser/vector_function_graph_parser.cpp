/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "vector_function_graph_parser.h"
#include "base/att_const_values.h"
#include "common/checker.h"
#include "ascir_ops.h"
#include "base_types_printer.h"
#include "ascir/meta/ascir_ops_utils.h"
namespace att {
namespace {
constexpr ge::char_t kSubGraphName[] = "sub_graph_name";
constexpr ge::char_t kInputPrefix[] = "_input_";
constexpr ge::char_t kOutputPrefix[] = "_output_";
}
using namespace ge::ops;
using namespace ge::ascir_op;
ge::Status VectorFunctionGraphParser::ParseNodeInfos(NodeInfo &node_info) {
  GE_ASSERT_SUCCESS(ParseInputTensors(node_info), "Parse input tensors failed, node name:%s, %s.",
                    asc_node_->GetNamePtr(), asc_node_->GetTypePtr());
  GE_ASSERT_SUCCESS(ParseOutputTensors(node_info), "Parse output tensors failed, node name:%s, %s.",
                    asc_node_->GetNamePtr(), asc_node_->GetTypePtr());
  return ge::SUCCESS;
}

ge::Status VectorFunctionGraphParser::GetVectorizedAxes(const TensorPtr &tensor,
                                                        const ge::AscTensorAttr &tensor_attr) const {
  std::unordered_map<size_t, size_t> axis_id_to_index;
  for (size_t i = 0UL; i < tensor_attr.axis.size(); i++) {
    axis_id_to_index[tensor_attr.axis[i]] = i;
  }
  for (auto &vectorized_axis_id : tensor_attr.vectorized_axis) {
    auto axis_index_iter = axis_id_to_index.find(vectorized_axis_id);
    GE_ASSERT_TRUE(axis_index_iter != axis_id_to_index.end(),
                   "Vectorized axis[%ld] not in tensor axis, tensor info is %s.", vectorized_axis_id,
                   tensor->ToString().c_str());
    const auto vectorized_axis_index = axis_index_iter->second;
    const auto vectorized_axis_repeat = tensor_attr.repeats[vectorized_axis_index];
    const auto vectorized_axis_stride = tensor_attr.strides[vectorized_axis_index];
    tensor->repeat.emplace_back(vectorized_axis_repeat);
    tensor->stride.emplace_back(vectorized_axis_stride);
  }
  return ge::SUCCESS;
}

ge::Status VectorFunctionGraphParser::ParseTensorInfo(const ge::AscTensorAttr &attr, const TensorPtr &tensor,
                                                      size_t index) {
  GE_ASSERT_SUCCESS(GetVectorizedAxes(tensor, attr), "Get vectorized axis tensor size[%s] failed, graph name[%s].",
                    tensor->name.c_str(), graph_.GetName().c_str());
  tensor->owner_node = asc_node_.get();
  tensor->data_type = BaseTypeUtils::DtypeToStr(attr.dtype);
  tensor->data_type_size = ge::GetSizeByDataType(attr.dtype);
  // Vector function默认都放在UB
  tensor->loc = HardwareDef::UB;
  GELOGD("Get node [%s] input[%zu] datatype [%d] name[%s], tensor info is %s", asc_node_->GetNamePtr(), index,
         static_cast<int32_t>(attr.dtype), tensor->name.c_str(), tensor->ToString().c_str());
  return ge::SUCCESS;
}

ge::Status VectorFunctionGraphParser::ParseInputTensors(NodeInfo &node_info) {
  for (size_t in_id = 0U; in_id < asc_node_->inputs.Size(); in_id++) {
    TensorPtr tensor = std::make_shared<Tensor>();
    GE_ASSERT_NOTNULL(tensor, "Create tensor failed, node name:%s, %s.", asc_node_->GetNamePtr(),
                      asc_node_->GetTypePtr());
    tensor->name = asc_node_->GetName() + kInputPrefix + std::to_string(in_id);
    GE_ASSERT_SUCCESS(ParseTensorInfo(asc_node_->inputs[in_id].attr, tensor, in_id),
                      "Parse tensor info failed, node name:%s, %s.", asc_node_->GetNamePtr(), asc_node_->GetTypePtr());
    node_info.inputs.emplace_back(tensor);
  }
  return ge::SUCCESS;
}

ge::Status VectorFunctionGraphParser::ParseOutputTensors(NodeInfo &node_info) {
  for (size_t out_id = 0U; out_id < asc_node_->outputs().size(); out_id++) {
    TensorPtr tensor = std::make_shared<Tensor>();
    GE_ASSERT_NOTNULL(tensor, "Create tensor failed, node name:%s, %s.", asc_node_->GetNamePtr(),
                      asc_node_->GetTypePtr());
    tensor->name = asc_node_->GetName() + kOutputPrefix + std::to_string(out_id);
    GE_ASSERT_SUCCESS(ParseTensorInfo(asc_node_->outputs[out_id].attr, tensor, out_id),
                      "Parse tensor info failed, node name:%s, %s.", asc_node_->GetNamePtr(), asc_node_->GetTypePtr());
    node_info.outputs.emplace_back(tensor);
  }
  return ge::SUCCESS;
}

ge::Status VectorFunctionGraphParser::Parse() {
  GE_ASSERT_NOTNULL(asc_node_);
  // 仅解析VectorFunc节点
  if (asc_node_->GetTypePtr() != kVectorFunc) {
    return ge::SUCCESS;
  }
  GELOGD("Node:%s, %s, sub_graph_name start to parse subgraph", asc_node_->GetNamePtr(), asc_node_->GetTypePtr());
  const std::string *graph_name = ge::AttrUtils::GetStr(asc_node_->GetOpDescBarePtr(), kSubGraphName);
  GE_ASSERT_NOTNULL(graph_name, "Get sub graph name failed, vf node:%s", asc_node_->GetNamePtr());
  ge::AscGraph sub_graph("vf_sub_graph");
  GE_ASSERT_SUCCESS(graph_.FindSubGraph(*graph_name, sub_graph), "Get sub_graph failed, vf node:%s, sub_graph_name:%s",
                    asc_node_->GetNamePtr(), graph_name);
  GELOGI("VF node:%s, sub_graph_name:%s", asc_node_->GetNamePtr(), graph_name->c_str());
  for (auto node : sub_graph.GetAllNodes()) {
    // subgraph上的Load api直接使用Tpipe上保存的UB tensor, 因此vf子图上Data节点的输出Tensor不必保存在tensor manager中.
    if (IsOps<Output>(node) || IsOps<Data>(node)) {
      continue;
    }
    NodeInfo node_info;
    node_info.node_type = node->GetType();
    node_info.name = node->GetName();
    GE_ASSERT_SUCCESS(ParseNodeInfos(node_info), "Parse node infos failed, node name:%s, %s.",
                      node->GetNamePtr(), node->GetTypePtr());
    nodes_infos_.emplace_back(node_info);
  }
  return ge::SUCCESS;
}
}