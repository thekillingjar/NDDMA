/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph/attr_value.h"
#include "graph_metadef/graph/debug/ge_util.h"
#include "graph/ge_attr_value.h"
#include "graph/ge_tensor.h"
#include "graph/type_utils.h"
#include "graph/utils/tensor_adapter.h"
#include "common/checker.h"

#define ATTR_VALUE_SET_GET_IMP(type)                 \
  graphStatus AttrValue::GetValue(type &val) const { \
    if (impl != nullptr) {                           \
      return impl->geAttrValue_.GetValue<type>(val); \
    }                                                \
    return GRAPH_FAILED;                             \
  }

#define ATTR_VALUE_SET_ATTR_IMP(type)                                 \
  graphStatus AttrValue::SetAttrValue(const type &attr_value) const { \
    GE_ASSERT_NOTNULL(impl);                                          \
    return impl->geAttrValue_.SetValue(attr_value);                   \
  }

#define ATTR_VALUE_GET_ATTR_IMP(type)                           \
  graphStatus AttrValue::GetAttrValue(type &attr_value) const { \
    GE_ASSERT_NOTNULL(impl);                                    \
    return impl->geAttrValue_.GetValue(attr_value);             \
  }

namespace ge {
GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY AttrValue::AttrValue() {
  impl = ComGraphMakeShared<AttrValueImpl>();
}

ATTR_VALUE_SET_GET_IMP(AttrValue::STR)
ATTR_VALUE_SET_GET_IMP(AttrValue::INT)
ATTR_VALUE_SET_GET_IMP(AttrValue::FLOAT)

// 使用宏生成基本类型的 SetAttrValue 和 GetAttrValue 函数
ATTR_VALUE_SET_ATTR_IMP(int64_t)
ATTR_VALUE_GET_ATTR_IMP(int64_t)
ATTR_VALUE_SET_ATTR_IMP(float32_t)
ATTR_VALUE_GET_ATTR_IMP(float32_t)
ATTR_VALUE_SET_ATTR_IMP(bool)
ATTR_VALUE_GET_ATTR_IMP(bool)
ATTR_VALUE_SET_ATTR_IMP(ge::DataType)
ATTR_VALUE_GET_ATTR_IMP(ge::DataType)

// 使用宏生成容器类型的 SetAttrValue 和 GetAttrValue 函数
ATTR_VALUE_SET_ATTR_IMP(std::vector<int64_t>)
ATTR_VALUE_GET_ATTR_IMP(std::vector<int64_t>)
ATTR_VALUE_SET_ATTR_IMP(std::vector<float32_t>)
ATTR_VALUE_GET_ATTR_IMP(std::vector<float32_t>)
ATTR_VALUE_SET_ATTR_IMP(std::vector<bool>)
ATTR_VALUE_GET_ATTR_IMP(std::vector<bool>)
ATTR_VALUE_SET_ATTR_IMP(std::vector<std::vector<int64_t>>)
ATTR_VALUE_GET_ATTR_IMP(std::vector<std::vector<int64_t>>)
ATTR_VALUE_SET_ATTR_IMP(std::vector<ge::DataType>)
ATTR_VALUE_GET_ATTR_IMP(std::vector<ge::DataType>)

graphStatus AttrValue::GetValue(AscendString &val) {
  std::string val_get;
  const auto status = GetValue(val_get);
  if (status != GRAPH_SUCCESS) {
    return status;
  }
  val = AscendString(val_get.c_str());
  return GRAPH_SUCCESS;
}

// 特殊处理 AscendString 类型
graphStatus AttrValue::SetAttrValue(const AscendString &attr_value) const {
  GE_ASSERT_NOTNULL(impl);
  return impl->geAttrValue_.SetValue(std::string(attr_value.GetString()));
}

graphStatus AttrValue::GetAttrValue(AscendString &attr_value) const {
  GE_ASSERT_NOTNULL(impl);
  std::string str_value;
  GE_ASSERT_GRAPH_SUCCESS(impl->geAttrValue_.GetValue(str_value));

  attr_value = AscendString(str_value.c_str());
  return GRAPH_SUCCESS;
}

// 特殊处理 std::vector<AscendString> 类型
graphStatus AttrValue::SetAttrValue(const std::vector<AscendString> &attr_values) const {
  GE_ASSERT_NOTNULL(impl);

  std::vector<std::string> str_values;
  for (const auto &value : attr_values) {
    str_values.emplace_back(value.GetString());
  }
  return impl->geAttrValue_.SetValue(str_values);
}

graphStatus AttrValue::GetAttrValue(std::vector<AscendString> &attr_values) const {
  GE_ASSERT_NOTNULL(impl);
  std::vector<std::string> str_values;
  GE_ASSERT_GRAPH_SUCCESS(impl->geAttrValue_.GetValue(str_values));
  attr_values.clear();
  for (const auto &value : str_values) {
    attr_values.emplace_back(value.c_str());
  }
  return GRAPH_SUCCESS;
}

// 特殊处理 Tensor 类型
graphStatus AttrValue::SetAttrValue(const Tensor &attr_value) const {
  GE_ASSERT_NOTNULL(impl);
  return impl->geAttrValue_.SetValue(TensorAdapter::AsGeTensor(attr_value));
}

graphStatus AttrValue::GetAttrValue(Tensor &attr_value) const {
  GE_ASSERT_NOTNULL(impl);
  GeTensor ge_tensor;
  GE_ASSERT_GRAPH_SUCCESS(impl->geAttrValue_.GetValue(ge_tensor));
  attr_value = TensorAdapter::AsTensor(ge_tensor);
  return GRAPH_SUCCESS;
}

// 特殊处理 std::vector<Tensor> 类型
graphStatus AttrValue::SetAttrValue(const std::vector<Tensor> &attr_value) const {
  GE_ASSERT_NOTNULL(impl);
  std::vector<GeTensor> ge_tensors;
  for (const auto &tensor : attr_value) {
    ge_tensors.emplace_back(TensorAdapter::AsGeTensor(tensor));
  }
  return impl->geAttrValue_.SetValue(ge_tensors);
}

graphStatus AttrValue::GetAttrValue(std::vector<Tensor> &attr_value) const {
  GE_ASSERT_NOTNULL(impl);
  std::vector<GeTensor> ge_tensors;
  GE_ASSERT_GRAPH_SUCCESS(impl->geAttrValue_.GetValue(ge_tensors));
  attr_value.clear();
  for (const auto &ge_tensor : ge_tensors) {
    attr_value.emplace_back(TensorAdapter::AsTensor(ge_tensor));
  }
  return GRAPH_SUCCESS;
}
}  // namespace ge
