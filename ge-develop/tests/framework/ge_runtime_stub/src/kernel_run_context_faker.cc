/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "faker/kernel_run_context_facker.h"
#include "graph/compute_graph.h"
#include "exe_graph/lowering/bg_kernel_context_extend.h"
#include "exe_graph/runtime/tiling_context.h"
#include "graph/utils/graph_utils.h"

namespace gert {
KernelRunContextHolder BuildKernelRunContext(size_t input_num, size_t output_num) {
  return KernelRunContextFaker().KernelIONum(input_num, output_num).Build();
}
KernelRunContextFaker &KernelRunContextFaker::KernelIONum(size_t input_num, size_t output_num) {
  kernel_input_num_ = input_num;
  kernel_output_num_ = output_num;
  return *this;
}
KernelRunContextFaker &KernelRunContextFaker::NodeIoNum(size_t input_num, size_t output_num) {
  node_input_num_ = input_num;
  node_output_num_ = output_num;
  node_input_tds_.resize(input_num);
  node_output_tds_.resize(output_num);
  return *this;
}
KernelRunContextFaker &KernelRunContextFaker::IrInputNum(size_t input_num) {
  ir_instance_num_ = std::vector<uint32_t>(input_num, 1);
  return *this;
}
KernelRunContextFaker &KernelRunContextFaker::IrInstanceNum(std::vector<uint32_t> instance_num) {
  ir_instance_num_ = std::move(instance_num);
  return *this;
}
ge::NodePtr KernelRunContextFaker::FakeNode(ge::ComputeGraphPtr &graph) const {
  auto op_desc = std::make_shared<ge::OpDesc>(node_name_, node_type_);
  size_t input_index = 0;
  for (size_t ir_index = 0; ir_index < ir_instance_num_.size(); ++ir_index) {
    auto ir_ins_num = ir_instance_num_[ir_index];
    auto prefix = "x_" + std::to_string(ir_index) + "_";
    op_desc->AppendIrInput(prefix, ge::kIrInputDynamic);
    for (size_t i = 0; i < ir_ins_num; ++i, ++input_index) {
      auto td = ge::GeTensorDesc();
      if (node_input_tds_.size() > input_index) {
        td.SetOriginFormat(node_input_tds_[input_index].GetOriginFormat());
        td.SetFormat(node_input_tds_[input_index].GetStorageFormat());
        td.SetDataType(node_input_tds_[input_index].GetDataType());
        td.SetOriginDataType(node_input_tds_[input_index].GetDataType());
      }
      op_desc->AddInputDesc(prefix + std::to_string(i), td);
    }
  }
  for (size_t i = 0; i < node_output_num_; ++i) {
    auto td = ge::GeTensorDesc();
    if (node_output_tds_.size() > i) {
      td.SetOriginFormat(node_output_tds_[i].GetOriginFormat());
      td.SetFormat(node_output_tds_[i].GetStorageFormat());
      td.SetDataType(node_output_tds_[i].GetDataType());
      td.SetOriginDataType(node_output_tds_[i].GetDataType());
    }
    op_desc->AddOutputDesc("y" + std::to_string(i), td);
  }
  for (const auto &attr : attrs_) {
    op_desc->AppendIrAttrName(attr.first);
    op_desc->SetAttr(attr.first, attr.second);
  }

  auto fake_node = graph->AddNode(op_desc);
  // fake peer inputs of fake node
  auto data_op_desc = std::make_shared<ge::OpDesc>("data", "Data");
  data_op_desc->AddInputDesc("x", ge::GeTensorDesc());
  data_op_desc->AddOutputDesc("y", ge::GeTensorDesc());
  auto data_node = graph->AddNode(data_op_desc);
  ge::AttrUtils::SetInt(data_op_desc, "index", 0);
  auto data_out_anchor = data_node->GetOutDataAnchor(0);
  for (auto &anchor : fake_node->GetAllInDataAnchors()) {
    ge::GraphUtils::AddEdge(data_out_anchor, anchor);
  }
  for (auto &anchor : fake_node->GetAllOutDataAnchors()) {
    ge::GraphUtils::AddEdge(data_out_anchor, anchor);
  }
  return fake_node;
}
KernelRunContextHolder KernelRunContextFaker::Build() const {
  KernelRunContextHolder holder;
  size_t size = sizeof(KernelRunContext) + sizeof(AsyncAnyValue *) * (kernel_input_num_ + kernel_output_num_);
  holder.context_holder = std::unique_ptr<uint8_t[]>(new uint8_t[size]);
  memset(holder.context_holder.get(), 0xff, size);
  holder.value_holder.resize(kernel_input_num_ + kernel_output_num_);

  size_t extend_info_size;
  auto graph = std::make_shared<ge::ComputeGraph>("tmp");  // 续命
  auto fake_node = FakeNode(graph);
  holder.compute_node_info_holder = bg::CreateComputeNodeInfo(fake_node, holder.buffer_pool, extend_info_size);
  auto c_node_info = reinterpret_cast<ComputeNodeInfo *>(holder.compute_node_info_holder.get());
  c_node_info->SetNodeName(holder.buffer_pool.GetBufById(reinterpret_cast<size_t>(c_node_info->GetNodeName())));
  c_node_info->SetNodeType(holder.buffer_pool.GetBufById(reinterpret_cast<size_t>(c_node_info->GetNodeType())));

  holder.kernel_name_holder = kernel_name_;
  holder.kernel_type_holder = kernel_type_;
  holder.kernel_extend_info_holder = std::unique_ptr<uint8_t[]>(new uint8_t[sizeof(KernelExtendInfo)]);
  auto extend_info = reinterpret_cast<KernelExtendInfo *>(holder.kernel_extend_info_holder.get());
  extend_info->SetKernelName(holder.kernel_name_holder.c_str());
  extend_info->SetKernelType(holder.kernel_type_holder.c_str());

  holder.context = reinterpret_cast<KernelRunContext *>(holder.context_holder.get());
  holder.context->input_size = kernel_input_num_;
  holder.context->output_size = kernel_output_num_;
  holder.context->compute_node_info = holder.compute_node_info_holder.get();
  holder.context->kernel_extend_info = extend_info;
  holder.context->output_start = &(holder.context->values[holder.context->input_size]);

  for (size_t i = 0; i < kernel_input_num_ + kernel_output_num_; ++i) {
    holder.context->values[i] = reinterpret_cast<AsyncAnyValue *>(&holder.value_holder[i]);
  }

  if (inputs_.size() == kernel_input_num_) {
    for (size_t i = 0; i < inputs_.size(); ++i) {
      holder.value_holder[i].Set(inputs_[i], nullptr);
    }
  }
  if (outputs_.size() == kernel_output_num_) {
    for (size_t i = 0; i < outputs_.size(); ++i) {
      holder.value_holder[i + kernel_input_num_].Set(outputs_[i], nullptr);
    }
  }
  return holder;
}
KernelRunContextFaker &KernelRunContextFaker::NodeInputTd(int32_t index, ge::DataType dt, ge::Format origin_format,
                                                          ge::Format storage_format) {
  node_input_tds_[index].SetDataType(dt);
  node_input_tds_[index].SetOriginFormat(origin_format);
  node_input_tds_[index].SetStorageFormat(storage_format);
  return *this;
}
KernelRunContextFaker &KernelRunContextFaker::NodeOutputTd(int32_t index, ge::DataType dt, ge::Format origin_format,
                                                           ge::Format storage_format) {
  node_output_tds_[index].SetDataType(dt);
  node_output_tds_[index].SetOriginFormat(origin_format);
  node_output_tds_[index].SetStorageFormat(storage_format);
  return *this;
}
KernelRunContextFaker &KernelRunContextFaker::Inputs(std::vector<void *> inputs) {
  kernel_input_num_ = inputs.size();
  inputs_ = std::move(inputs);
  return *this;
}
KernelRunContextFaker &KernelRunContextFaker::Outputs(std::vector<void *> outputs) {
  kernel_output_num_ = outputs.size();
  outputs_ = std::move(outputs);
  return *this;
}
KernelRunContextFaker &KernelRunContextFaker::NodeAttrs(
    std::vector<std::pair<std::string, ge::AnyValue>> keys_to_value) {
  attrs_ = std::move(keys_to_value);
  return *this;
}
KernelRunContextFaker &KernelRunContextFaker::KernelName(std::string name) {
  kernel_name_ = std::move(name);
  return *this;
}
KernelRunContextFaker &KernelRunContextFaker::KernelType(std::string kernel_type) {
  kernel_type_ = std::move(kernel_type);
  return *this;
}
KernelRunContextFaker &KernelRunContextFaker::NodeName(std::string name) {
  node_name_ = std::move(name);
  return *this;
}
KernelRunContextFaker &KernelRunContextFaker::NodeType(std::string node_type) {
  node_type_ = std::move(node_type);
  return *this;
}

InferShapeContextFaker &InferShapeContextFaker::NodeIoNum(size_t input_num, size_t output_num) {
  base_faker_.KernelIONum(input_num + kInputsAppendEnd, output_num);
  base_faker_.NodeIoNum(input_num, output_num);
  return *this;
}
InferShapeContextFaker &InferShapeContextFaker::InputShapes(std::vector<void *> input_shapes) {
  std::vector<void *> inputs(std::move(input_shapes));
  inputs.push_back(nullptr);  // infershape func
  base_faker_.Inputs(std::move(inputs));
  return *this;
}
InferShapeContextFaker &InferShapeContextFaker::OutputShapes(std::vector<void *> output_shapes) {
  base_faker_.Outputs(std::move(output_shapes));
  return *this;
}
KernelRunContextHolder InferShapeContextFaker::Build() const {
  return base_faker_.Build();
}
TilingContextFaker &TilingContextFaker::NodeIoNum(size_t input_num, size_t output_num) {
  base_faker_.KernelIONum(input_num + output_num + kInputsAppendEnd, outputs_.size());
  base_faker_.NodeIoNum(input_num, output_num);
  return *this;
}
TilingContextFaker &TilingContextFaker::FallibleContext() {
  outputs_.resize(TilingContext::kOutputNum + 1);
  return *this;
}
TilingContextFaker &TilingContextFaker::InputShapes(std::vector<gert::StorageShape *> input_shapes) {
  input_shapes_ = std::move(input_shapes);
  UpdateInputs();
  return *this;
}
TilingContextFaker &TilingContextFaker::OutputShapes(std::vector<gert::StorageShape *> output_shapes) {
  output_shapes_ = std::move(output_shapes);
  UpdateInputs();
  return *this;
}
TilingContextFaker &TilingContextFaker::CompileInfo(void *compile_info) {
  compile_info_ = compile_info;
  UpdateInputs();
  return *this;
}
TilingContextFaker &TilingContextFaker::TilingData(void *tiling_data) {
  outputs_[TilingContext::kOutputTilingData] = tiling_data;
  base_faker_.Outputs(outputs_);
  return *this;
}
TilingContextFaker &TilingContextFaker::Workspace(ContinuousVector *workspace) {
  outputs_[TilingContext::kOutputWorkspace] = workspace;
  base_faker_.Outputs(outputs_);
  return *this;
}
KernelRunContextHolder TilingContextFaker::Build() const {
  return base_faker_.Build();
}
void TilingContextFaker::UpdateInputs() {
  std::vector<void *> inputs;
  for (const auto input_shape : input_shapes_) {
    inputs.push_back(input_shape);
  }
  for (const auto output_shape : output_shapes_) {
    inputs.push_back(output_shape);
  }
  inputs.push_back(compile_info_);         // kInputsCompileInfo
  inputs.push_back(fwk_data_);  // kInputsTilingFunc
  inputs.push_back(deterministic_);
  inputs.push_back(deterministic_level_);
  base_faker_.Inputs(std::move(inputs));
}
TilingContextFaker &TilingContextFaker::TilingFwkData(void* fwk_data) {
  fwk_data_ = fwk_data;
  UpdateInputs();
  return *this;
}
TilingContextFaker &TilingContextFaker::Deterministic(void* deterministic) {
  deterministic_ = deterministic;
  UpdateInputs();
  return *this;
}
}  // namespace gert
