/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "arg_desc_info_impl.h"
#include "graph_metadef/graph/debug/ge_util.h"
#include "common/checker.h"

namespace ge {
ArgDescInfoImpl::ArgDescInfoImpl(ArgDescType arg_type, int32_t ir_index, bool is_folded)
    : arg_type_(arg_type), ir_index_(ir_index), is_folded_(is_folded) {}

ArgDescInfoImplPtr ArgDescInfoImpl::CreateCustomValue(uint64_t custom_value) {
  auto impl_ptr = ComGraphMakeUnique<ArgDescInfoImpl>();
  GE_ASSERT_NOTNULL(impl_ptr);
  impl_ptr->arg_type_ = ArgDescType::kCustomValue;
  impl_ptr->custom_value_ = custom_value;
  return impl_ptr;
}

ArgDescInfoImplPtr ArgDescInfoImpl::CreateHiddenInput(HiddenInputSubType hidden_type) {
  auto impl_ptr = ComGraphMakeUnique<ArgDescInfoImpl>();
  GE_ASSERT_NOTNULL(impl_ptr);
  impl_ptr->arg_type_ = ArgDescType::kHiddenInput;
  impl_ptr->hidden_type_ = hidden_type;
  return impl_ptr;
}

ArgDescType ArgDescInfoImpl::GetType() const {
  return arg_type_;
}
uint64_t ArgDescInfoImpl::GetCustomValue() const {
  return custom_value_;
}
graphStatus ArgDescInfoImpl::SetCustomValue(uint64_t custom_value) {
  GE_ASSERT_TRUE(arg_type_ == ArgDescType::kCustomValue,
      "Only ArgDescType::kCustomValue arg desc info can set custom value");
  custom_value_ = custom_value;
  return SUCCESS;
}
HiddenInputSubType ArgDescInfoImpl::GetHiddenInputSubType() const {
  return hidden_type_;
}
graphStatus ArgDescInfoImpl::SetHiddenInputSubType(HiddenInputSubType hidden_type) {
  GE_ASSERT_TRUE(arg_type_ == ArgDescType::kHiddenInput,
      "Only ArgDescType::kHiddenInput arg desc info can set hidden input sub type");
  hidden_type_ = hidden_type;
  return SUCCESS;
}
int32_t ArgDescInfoImpl::GetIrIndex() const {
  return ir_index_;
}

void ArgDescInfoImpl::SetIrIndex(int32_t ir_index) {
  ir_index_ = ir_index;
}

bool ArgDescInfoImpl::IsFolded() const {
  return is_folded_;
}
void ArgDescInfoImpl::SetFolded(bool is_folded) {
  is_folded_ = is_folded;
}
void ArgDescInfoImpl::SetInnerArgType(AddrType inner_arg_type) {
  inner_arg_type_ = inner_arg_type;
}
AddrType ArgDescInfoImpl::GetInnerArgType() const {
  return inner_arg_type_;
}
}