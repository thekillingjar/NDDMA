/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "attribute_group/attr_group_symbolic_desc.h"
#include "common/checker.h"
#include "proto/ge_ir.pb.h"

namespace ge {

graphStatus SymbolicDescAttr::Serialize(proto::AttrGroupDef &attr_group_def) {
  auto tensor_attr_group = attr_group_def.mutable_tensor_attr_group();
  GE_ASSERT_NOTNULL(tensor_attr_group);
  tensor_attr_group->clear_origin_symbol_shape();
  for (const auto &ori_shape : symbolic_tensor.GetOriginSymbolShape().GetDims()) {
    tensor_attr_group->add_origin_symbol_shape(ori_shape.Str().get());
  }
  tensor_attr_group->clear_symbolic_value();
  if (symbolic_tensor.GetSymbolicValue() != nullptr) {
    for (const auto &symbol_value : *symbolic_tensor.GetSymbolicValue()) {
      tensor_attr_group->add_symbolic_value(symbol_value.Str().get());
    }
  }
  return GRAPH_SUCCESS;
}

graphStatus SymbolicDescAttr::Deserialize(const proto::AttrGroupDef &attr_group_def, AttrHolder *attr_holder) {
  (void) attr_holder;
  const auto& tensor_attr_group = attr_group_def.tensor_attr_group();
  this->symbolic_tensor.MutableOriginSymbolShape().Clear();
  for (const auto &ori_sh : tensor_attr_group.origin_symbol_shape()) {
    this->symbolic_tensor.MutableOriginSymbolShape().AppendDim(ge::Expression::Parse(ori_sh.c_str()));
  }
  if (!tensor_attr_group.symbolic_value().empty()) {
    auto symbol_value_ptr = ComGraphMakeUnique<std::vector<ge::Expression>>();
    if (symbol_value_ptr != nullptr) {
      for (const auto &symbol_value : tensor_attr_group.symbolic_value()) {
        symbol_value_ptr->push_back(ge::Expression::Parse(symbol_value.c_str()));
      }
      this->symbolic_tensor.SetSymbolicValue(std::move(symbol_value_ptr));
    }
  }
  return GRAPH_SUCCESS;
}

std::unique_ptr<AttrGroupsBase> SymbolicDescAttr::Clone() {
  std::unique_ptr<AttrGroupsBase> attr = ComGraphMakeUnique<SymbolicDescAttr>(*this);
  GE_ASSERT_NOTNULL(attr);
  return attr;
}
} // namespace ge
