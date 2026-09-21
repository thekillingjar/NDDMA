/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "es_c_tensor_holder.h"
#include "common/checker.h"
/****************************************************/
// 这部分头文件是符号化能力引入的对内头文件，符号化能力暂时无法对外
#include "graph/symbolizer/symbolic.h"
#include "attribute_group/attr_group_symbolic_desc.h"
#include "graph/utils/node_adapter.h"
/****************************************************/

struct EsCTensorHolder::EsCTensorHolderImpl {
 public:
  EsCTensorHolderImpl(EsCGraphBuilder &owner, const ge::GNode &producer, int32_t index)
      : owner_graph_builder_(owner), producer_(producer), producer_out_index_(index) {}

  // 符号化相关能力因为没有对外接口，暂时使用内部接口；后续提供对外接口后，这部分整改掉
  ge::Status SetOriginSymbolShape(const char *const *shape_str, const int64_t shape_str_num) {
    GELOGW("This interface has no compatibility guarantees for now; use with caution!");
    std::vector<ge::Expression> shape;
    shape.reserve(shape_str_num);
    for (int64_t i = 0; i < shape_str_num; ++i) {
      shape.emplace_back(ge::Expression::Parse(shape_str[i]));
    }
    const auto td = GetInnerTd(producer_, producer_out_index_);
    GE_ASSERT_NOTNULL(td);
    const auto attr = td->GetOrCreateAttrsGroup<ge::SymbolicDescAttr>();
    GE_ASSERT_NOTNULL(attr);
    attr->symbolic_tensor.MutableOriginSymbolShape().MutableDims() = shape;
    return ge::SUCCESS;
  }
  ge::Status SetFormat(const ge::Format format) {
    GE_ASSERT_SUCCESS(SetOriginFormat(format));
    GE_ASSERT_SUCCESS(SetStorageFormat(format));
    return ge::SUCCESS;
  }
  ge::Status SetShape(const ge::Shape &shape) {
    GE_ASSERT_SUCCESS(SetOriginShape(shape));
    GE_ASSERT_SUCCESS(SetStorageShape(shape));
    return ge::SUCCESS;
  }
  ge::Status SetDataType(const ge::DataType data_type) {
    auto td = GetTd();
    td.SetDataType(data_type);
    UpdateTd(td);
    return ge::SUCCESS;
  }
  ge::Status SetOriginFormat(const ge::Format format) {
    auto td = GetTd();
    td.SetOriginFormat(format);
    UpdateTd(td);
    return ge::SUCCESS;
  }
  ge::Status SetStorageFormat(const ge::Format format) {
    auto td = GetTd();
    td.SetFormat(format);
    UpdateTd(td);
    return ge::SUCCESS;
  }
  ge::Status SetOriginShape(const ge::Shape &shape) {
    auto td = GetTd();
    td.SetOriginShape(shape);
    UpdateTd(td);
    return ge::SUCCESS;
  }
  ge::Status SetStorageShape(const ge::Shape &shape) {
    auto td = GetTd();
    td.SetShape(shape);
    UpdateTd(td);
    return ge::SUCCESS;
  }
  ge::GNode &GetProducer() {
    return producer_;
  }
  int32_t GetOutIndex() const {
    return producer_out_index_;
  }
  EsCGraphBuilder &GetOwnerBuilder() {
    return owner_graph_builder_;
  }

 private:
  ge::GeTensorDescPtr GetInnerTd(const ge::GNode &node, int32_t output_index) const {
    auto ge_node = ge::NodeAdapter::GNode2Node(node);
    GE_ASSERT_NOTNULL(ge_node);
    auto op_desc = ge_node->GetOpDesc();
    GE_ASSERT_NOTNULL(op_desc);
    return op_desc->MutableOutputDesc(output_index);
  }
  ge::TensorDesc GetTd() {
    ge::TensorDesc td;
    (void)producer_.GetOutputDesc(producer_out_index_, td);
    return td;
  }
  void UpdateTd(const ge::TensorDesc &td) {
    (void)producer_.UpdateOutputDesc(producer_out_index_, td);
  }

  EsCGraphBuilder &owner_graph_builder_;
  ge::GNode producer_;
  int32_t producer_out_index_;
};

EsCTensorHolder::EsCTensorHolder(EsCGraphBuilder &owner, const ge::GNode &producer, int32_t index) {
  // new失败说明业务无法正常进行，主动抛出异常
  impl_ = std::make_unique<EsCTensorHolderImpl>(owner, producer, index);
}
EsCTensorHolder::~EsCTensorHolder() = default;
EsCTensorHolder::EsCTensorHolder(EsCTensorHolder &&) noexcept = default;
EsCTensorHolder &EsCTensorHolder::operator=(EsCTensorHolder &&) noexcept = default;

// 符号化相关能力因为没有对外接口，暂时使用内部接口；后续提供对外接口后，这部分整改掉
ge::Status EsCTensorHolder::SetOriginSymbolShape(const char *const *shape_str, const int64_t shape_str_num) {
  return impl_->SetOriginSymbolShape(shape_str, shape_str_num);
}
ge::Status EsCTensorHolder::SetFormat(const ge::Format format) {
  return impl_->SetFormat(format);
}
ge::Status EsCTensorHolder::SetShape(const ge::Shape &shape) {
  return impl_->SetShape(shape);
}
ge::Status EsCTensorHolder::SetDataType(const ge::DataType data_type) {
  return impl_->SetDataType(data_type);
}
ge::Status EsCTensorHolder::SetOriginFormat(const ge::Format format) {
  return impl_->SetOriginFormat(format);
}
ge::Status EsCTensorHolder::SetStorageFormat(const ge::Format format) {
  return impl_->SetStorageFormat(format);
}
ge::Status EsCTensorHolder::SetOriginShape(const ge::Shape &shape) {
  return impl_->SetOriginShape(shape);
}
ge::Status EsCTensorHolder::SetStorageShape(const ge::Shape &shape) {
  return impl_->SetStorageShape(shape);
}
ge::GNode &EsCTensorHolder::GetProducer() {
  return impl_->GetProducer();
}
int32_t EsCTensorHolder::GetOutIndex() const {
  return impl_->GetOutIndex();
}
EsCGraphBuilder &EsCTensorHolder::GetOwnerBuilder() {
  return impl_->GetOwnerBuilder();
}
