/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_COMPILER_GRAPH_EAGER_STYLE_GRAPH_BUILDER_ESB_TENSOR_H_
#define AIR_CXX_COMPILER_GRAPH_EAGER_STYLE_GRAPH_BUILDER_ESB_TENSOR_H_
#include "graph/node.h"
#include "graph/symbolizer/symbolic.h"
#include "attribute_group/attr_group_symbolic_desc.h"
class EsbGraph;
class EsbTensor {
 public:
  // 注意，这是个内部类，调用着需要保证传入的`producer`不为空,`index`合法
  EsbTensor(EsbGraph &owner, ge::NodePtr producer, int32_t index)
      : owner_graph_(owner), producer_(std::move(producer)), producer_out_index_(index) {}

  ge::Status SetOriginShape(const ge::GeShape &shape) {
    auto td = GetTd();
    GE_ASSERT_NOTNULL(td);
    td->SetOriginShape(shape);
    return ge::SUCCESS;
  }

  ge::Status SetStorageShape(const ge::GeShape &shape) {
    auto td = GetTd();
    GE_ASSERT_NOTNULL(td);
    td->SetShape(shape);
    return ge::SUCCESS;
  }

  ge::Status SetShape(const ge::GeShape &shape) {
    GE_ASSERT_SUCCESS(SetOriginShape(shape));
    GE_ASSERT_SUCCESS(SetStorageShape(shape));
    return ge::SUCCESS;
  }

  ge::Status SetOriginSymbolShape(const std::vector<ge::Expression> &shape) {
    const auto td = GetTd();
    GE_ASSERT_NOTNULL(td);
    const auto attr = td->GetOrCreateAttrsGroup<ge::SymbolicDescAttr>();
    GE_ASSERT_NOTNULL(attr);
    attr->symbolic_tensor.MutableOriginSymbolShape().MutableDims() = shape;
    return ge::SUCCESS;
  }
  ge::Status SetInputOriginSymbolShape(const std::vector<ge::Expression> &shape) {
    const auto input_td = GetInputTd();
    GE_ASSERT_NOTNULL(input_td);
    const auto input_attr = input_td->GetOrCreateAttrsGroup<ge::SymbolicDescAttr>();
    GE_ASSERT_NOTNULL(input_attr);
    input_attr->symbolic_tensor.MutableOriginSymbolShape().MutableDims() = shape;
    return ge::SUCCESS;
  }
  ge::Status SetSymbolShape(const std::vector<ge::Expression> &shape) {
    GE_ASSERT_SUCCESS(SetOriginSymbolShape(shape));
    return ge::SUCCESS;
  }
  ge::Status SetInputSymbolShape(const std::vector<ge::Expression> &shape) {
    GE_ASSERT_SUCCESS(SetInputOriginSymbolShape(shape));
    return ge::SUCCESS;
  }
  ge::Status SetSymbolShape(const char *const *shape_str, const int64_t shape_str_num) {
    std::vector<ge::Expression> shape;
    shape.reserve(shape_str_num);
    for (int64_t i = 0; i < shape_str_num; ++i) {
      shape.emplace_back(ge::Expression::Parse(shape_str[i]));
    }
    return SetSymbolShape(shape);
  }
  ge::Status SetInputSymbolShape(const char *const *shape_str, const int64_t shape_str_num) {
    std::vector<ge::Expression> shape;
    shape.reserve(shape_str_num);
    for (int64_t i = 0; i < shape_str_num; ++i) {
      shape.emplace_back(ge::Expression::Parse(shape_str[i]));
    }
    return SetInputSymbolShape(shape);
  }
  ge::NodePtr GetProducer() const {
    return producer_;
  }

  EsbGraph &GetOwner() {
    return owner_graph_;
  }
  ge::OutDataAnchorPtr GetAnchor() {
    return producer_->GetOutDataAnchor(producer_out_index_);
  }

  EsbGraph &GetOwnerGraph() const {
    return owner_graph_;
  }

 private:
  ge::GeTensorDescPtr GetTd() {
    return producer_->GetOpDesc()->MutableOutputDesc(producer_out_index_);
  }
  ge::GeTensorDescPtr GetInputTd() {
    return producer_->GetOpDesc()->MutableInputDesc(producer_out_index_);
  }

 private:
  EsbGraph &owner_graph_;
  ge::NodePtr producer_;
  int32_t producer_out_index_;
};

#endif  // AIR_CXX_COMPILER_GRAPH_EAGER_STYLE_GRAPH_BUILDER_ESB_TENSOR_H_
