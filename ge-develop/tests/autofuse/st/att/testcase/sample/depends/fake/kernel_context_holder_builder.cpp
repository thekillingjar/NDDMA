/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "kernel_context_holder_builder.h"
#include <vector>
#include <string>
#include "graph/utils/graph_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "node_faker.h"
namespace att {
namespace {
void SetSpaceRegistry(const std::vector<std::pair<ge::AscendString, ge::AnyValue>> &private_attrs_) {
  auto space_registry = std::make_shared<gert::OpImplSpaceRegistryV2>();
  auto funcs = space_registry->CreateOrGetOpImpl("Bar");
  funcs->max_tiling_data_size = 100;
  funcs->host_inputs = 1;
  funcs->tiling_dependency = 1;
  funcs->tiling_dependency_placements = 1;
  funcs->unique_private_attrs.insert("test");
  ge::AscendString additional_output("additional_output");
  funcs->private_attrs = private_attrs_;
  gert::DefaultOpImplSpaceRegistryV2::GetInstance().SetSpaceRegistry(space_registry);
}
}
KernelContextHolderBuilder &KernelContextHolderBuilder::AddInput(const InOutput &input) {
  inputs_.emplace_back(input);
  return *this;
}

KernelContextHolderBuilder &KernelContextHolderBuilder::AddOutput(const InOutput &output) {
  outputs_.emplace_back(output);
  return *this;
}

KernelContextHolderBuilder &KernelContextHolderBuilder::SetTilingData(const size_t size) {
  tiling_data_ = std::move(gert::TilingData::CreateCap(size));
  return *this;
}

KernelContextHolderBuilder &KernelContextHolderBuilder::SetWorkSpace(const size_t size) {
  this->workspace_size_ = std::move(gert::ContinuousVector::Create<size_t>(size));
  return *this;
}

KernelContextHolderBuilder &KernelContextHolderBuilder::SetCompileInfo(const size_t size) {
  this->compile_info_ = std::make_unique<uint8_t[]>(size);
  return *this;
}

KernelContextHolderBuilder &KernelContextHolderBuilder::SetPlatformInfo() {
  platform_info_.reset(new fe::PlatFormInfos());
  return *this;
}

KernelContextHolderBuilder &KernelContextHolderBuilder::AddPrivateAtt(
    const std::pair<ge::AscendString, ge::AnyValue> &attr) {
  private_attrs_.emplace_back(attr);
  return *this;
}

gert::KernelContextHolder KernelContextHolderBuilder::Build() {
  tiling_ctx_builder_ = std::move(std::make_unique<gert::TilingContextBuilder>());
  SetSpaceRegistry(private_attrs_);
  uint64_t input_num = inputs_.size();
  uint64_t output_num = outputs_.size();
  std::vector<std::string> names;
  for (uint64_t i = 0UL; i < input_num; ++i) {
    names.emplace_back("x_" + std::to_string(i));
  }
  std::vector<std::string> output_names;
  for (uint64_t i = 0UL; i < output_num; ++i) {
    output_names.emplace_back("y_" + std::to_string(i));
  }
  auto foo_node =
      ComputeNodeFaker().NameAndType("foo", "Foo").IoNum(1, input_num).InputNames({"x"}).OutputNames(names).Build();
  auto bar_node = ComputeNodeFaker()
      .NameAndType("bar", "Bar")
      .IoNum(input_num, output_num)
      .InputNames(names)
      .OutputNames(output_names)
      .Build();
  ge::OpDescPtr op_desc = bar_node->GetOpDesc();
  for (uint64_t i = 0UL; i < input_num; ++i) {
    ge::GraphUtils::AddEdge(foo_node->GetOutDataAnchor(i), bar_node->GetInDataAnchor(i));
    op_desc->MutableInputDesc(i)->SetDataType(inputs_[i].dtype);
    op_desc->MutableInputDesc(i)->SetShape(inputs_[i].shape);
    op_desc->MutableInputDesc(i)->SetOriginShape(inputs_[i].shape);
    op_desc->MutableInputDesc(i)->SetFormat(inputs_[i].format);
  }
  for (uint64_t i = 0UL; i < output_num; ++i) {
    op_desc->MutableOutputDesc(i)->SetDataType(outputs_[i].dtype);
    op_desc->MutableOutputDesc(i)->SetShape(outputs_[i].shape);
    op_desc->MutableOutputDesc(i)->SetOriginShape(outputs_[i].shape);
    op_desc->MutableOutputDesc(i)->SetFormat(outputs_[i].format);
  }
  auto op = ge::OpDescUtils::CreateOperatorFromNode(bar_node->shared_from_this());
  return tiling_ctx_builder_->CompileInfo(compile_info_.get())
      .PlatformInfo(platform_info_.get())
      .Workspace(reinterpret_cast<gert::ContinuousVector *>(workspace_size_.get()))
      .TilingData(tiling_data_.get())
      .Build(op);
}
}