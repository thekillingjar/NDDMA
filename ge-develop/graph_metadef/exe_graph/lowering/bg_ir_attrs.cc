/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "exe_graph/lowering/bg_ir_attrs.h"

#include <cstring>
#include <securec.h>
#include "framework/common/debug/ge_log.h"
#include "graph/utils/math_util.h"
#include "graph/def_types.h"
#include "graph/types.h"
#include "common/checker.h"
#include "graph_metadef/graph/debug/ge_util.h"

#include "exe_graph/runtime/tensor.h"
#include "graph_metadef/base/attr/attrs_to_buffer.h"

namespace gert {
namespace bg {
namespace {
void GeShapeToGertShape(const ge::GeShape &ge_shape, gert::Shape &gert_shape) {
  gert_shape.SetDimNum(ge_shape.GetDimNum());
  for (size_t i = 0; i < ge_shape.GetDimNum(); ++i) {
    gert_shape.SetDim(i, ge_shape.GetDim(i));
  }
}

size_t GetGeTensorSize(const ge::GeTensor &tensor) {
  auto dt = tensor.GetTensorDesc().GetDataType();
  if (dt == ge::DT_STRING) {
    return tensor.GetData().GetSize();
  }
  auto shape_size = tensor.GetTensorDesc().GetShape().GetShapeSize();
  return static_cast<size_t>(ge::GetSizeInBytes(shape_size, dt));
}
bool AppendTensorAttr(const ge::AnyValue &attr, std::vector<std::vector<uint8_t>> &attrs) {
  auto val = attr.Get<ge::GeTensor>();
  GE_ASSERT_NOTNULL(val);
  auto &tensor_desc = val->GetTensorDesc();
  auto shape_size = tensor_desc.GetShape().GetShapeSize();
  if (shape_size < 0) {
    GELOGE(ge::PARAM_INVALID, "Failed to append tensor attr, shape size less than 0");
    return false;
  }
  size_t total_size;
  size_t tensor_size = GetGeTensorSize(*val);
  auto tensor_holder = Tensor::CreateFollowing(val->GetTensorDesc().GetDataType(), tensor_size, total_size);
  GE_ASSERT_NOTNULL(tensor_holder);
  auto tensor = ge::PtrToPtr<uint8_t, Tensor>(tensor_holder.get());
  GeShapeToGertShape(tensor_desc.GetShape(), tensor->MutableStorageShape());
  GeShapeToGertShape(tensor_desc.GetOriginShape(), tensor->MutableOriginShape());
  tensor->SetOriginFormat(tensor_desc.GetOriginFormat());
  tensor->SetStorageFormat(tensor_desc.GetFormat());
  if (total_size < sizeof(Tensor)) {
    GELOGE(ge::PARAM_INVALID, "total_size[%zu] < size of Tensor[%zu]", total_size, sizeof(Tensor));
    return false;
  }
  const auto copy_len = total_size - sizeof(Tensor);
  if (copy_len != 0U) {
    GE_CHECK_GE(val->GetData().size(), total_size - sizeof(Tensor));
    const auto ret_copy = ge::GeMemcpy(tensor->GetData<uint8_t>(), total_size - sizeof(Tensor),
        val->GetData().GetData(), total_size - sizeof(Tensor));
    GE_ASSERT_TRUE((ret_copy == ge::SUCCESS), "memcpy_s failed, copy size is %zu", (total_size - sizeof(Tensor)));
  }

  std::vector<uint8_t> buf(total_size);
  const auto ret = ge::GeMemcpy(buf.data(), total_size, tensor_holder.get(), total_size);
  GE_ASSERT_TRUE((ret == ge::SUCCESS), "memcpy_s failed, copy size is %zu", total_size);
  attrs.emplace_back(std::move(buf));
  return true;
}

bool AppendAttrTensor(const ge::AnyValue &attr, std::vector<std::vector<uint8_t>> &attrs) {
  switch (attr.GetValueType()) {
    case ge::AnyValue::VT_TENSOR:
      return AppendTensorAttr(attr, attrs);
    default:
      return AppendAttr(attr, attrs);
  }
}
}  // namespace

bool GetAllIrAttrs(const ge::NodePtr &node, std::vector<std::vector<uint8_t>> &runtime_attrs) {
  const auto op_desc = node->GetOpDescBarePtr();
  GE_ASSERT_NOTNULL(op_desc);
  const auto &all_attrs = ge::AttrUtils::GetAllAttrs(op_desc);
  const auto &ir_attr_names = op_desc->GetIrAttrNames();
  for (const auto &attr_name : ir_attr_names) {
    const std::map<std::string, ge::AnyValue>::const_iterator &iter = all_attrs.find(attr_name);
    if (iter == all_attrs.cend()) {
      runtime_attrs.clear();
      GELOGI("Can not find the IR attr %s from node %s(%s), clear all attrs",
             attr_name.c_str(), node->GetNamePtr(), node->GetTypePtr());
      return true;
    }
    GE_ASSERT_TRUE(AppendAttrTensor(iter->second, runtime_attrs));
  }
  return true;
}

std::unique_ptr<uint8_t[]> CreateAttrBuffer(const ge::NodePtr &node, size_t &size) {
  return CreateAttrBuffer(node, {}, size);
}

std::unique_ptr<uint8_t[]> CreateAttrBuffer(const ge::NodePtr &node,
                                            const std::vector<ge::AnyValue> &runtime_attrs_list,
                                            size_t &size) {
  std::vector<std::vector<uint8_t>> runtime_attrs;
  GE_ASSERT_TRUE(GetAllIrAttrs(node, runtime_attrs));
  for (auto &runtime_attr : runtime_attrs_list) {
    AppendAttrTensor(runtime_attr, runtime_attrs);
  }
  return CreateAttrBuffer(runtime_attrs, size);
}

std::unique_ptr<uint8_t[]> CreateAttrBufferWithoutIr(const ge::NodePtr &node,
                                                     const std::vector<ge::AnyValue> &runtime_attrs_list,
                                                     size_t &size) {
  (void)node;
  std::vector<std::vector<uint8_t>> runtime_attrs;
  for (auto &runtime_attr : runtime_attrs_list) {
    AppendAttrTensor(runtime_attr, runtime_attrs);
  }
  return CreateAttrBuffer(runtime_attrs, size);
}
}  // namespace bg
}  // namespace gert
