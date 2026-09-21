/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "faker/node_faker.h"
#include "graph/compute_graph.h"
#include "common/checker.h"
namespace gert {
using namespace ge;
ComputeNodeFaker &ComputeNodeFaker::IoNum(size_t input_num, size_t output_num, ge::DataType data_type) {
  inputs_desc_.resize(input_num, ge::GeTensorDesc(GeShape({10,10,10,10}), ge::FORMAT_ND, data_type));
  outputs_desc_.resize(output_num, ge::GeTensorDesc(GeShape({10,10,10,10}), ge::FORMAT_ND, data_type));
  return *this;
}
ge::NodePtr ComputeNodeFaker::Build() {
  auto op_desc = std::make_shared<OpDesc>(name_, type_);
  GE_ASSERT_NOTNULL(op_desc);
  for (size_t i = 0U; i < inputs_desc_.size(); ++i) {
    auto &input_desc = inputs_desc_[i];
    if (i < input_names_.size()) {
      op_desc->AddInputDesc(input_names_[i], input_desc);
      op_desc->AppendIrInput(input_names_[i], ge::kIrInputRequired);
    } else {
      op_desc->AddInputDesc(input_desc);
    }
  }

  for (size_t i = 0U; i < outputs_desc_.size(); ++i) {
    auto &desc = outputs_desc_[i];
    if (i < output_names_.size()) {
      op_desc->AddOutputDesc(output_names_[i], desc);
      op_desc->AppendIrOutput(desc.GetName(), ge::kIrOutputRequired);
    } else {
      op_desc->AddOutputDesc(desc);
    }
  }

  for (const auto &attr : attr_keys_to_value_) {
    op_desc->SetAttr(attr.first, attr.second);
  }

  return graph_->AddNode(op_desc);
}
ComputeNodeFaker &ComputeNodeFaker::NameAndType(std::string name, std::string type) {
  name_ = std::move(name);
  type_ = std::move(type);
  return *this;
}
ComputeNodeFaker &ComputeNodeFaker::InputNames(vector<std::string> names) {
  if (names.size() != inputs_desc_.size()) {
    throw std::invalid_argument("The size of names and input num not match");
  }
  input_names_ = std::move(names);
  return *this;
}
ComputeNodeFaker &ComputeNodeFaker::OutputNames(vector<std::string> names) {
  if (names.size() != outputs_desc_.size()) {
    throw std::invalid_argument("The size of names and output num not match");
  }
  output_names_ = std::move(names);
  return *this;
}
}
