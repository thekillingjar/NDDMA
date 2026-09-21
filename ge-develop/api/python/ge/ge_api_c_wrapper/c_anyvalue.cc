/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
// 直接使用 AttrValue 类来实现属性 set 和 get 的底层逻辑
// 对外的报错体现到py文件中
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include "graph/attr_value.h"
#include "ge_api_c_wrapper_utils.h"
#include "graph/ge_attr_value.h"

namespace {
// 获取标量值的通用函数
template <typename T>
ge::graphStatus GetScalar(const ge::AttrValue *attr_value, T *out) {
  if (attr_value == nullptr || out == nullptr) {
    return ge::GRAPH_FAILED;
  }
  return attr_value->GetAttrValue(*out);
}

// 获取POD类型列表的通用函数
template <typename T>
T *GetListPOD(const ge::AttrValue *attr_value, int64_t *size) {
  if (attr_value == nullptr || size == nullptr) {
    return nullptr;
  }
  std::vector<T> v;
  if ((attr_value->GetAttrValue(v) != ge::GRAPH_SUCCESS) || (v.empty())) {
    return nullptr;
  }
  *size = static_cast<int64_t>(v.size());
  return ge::c_wrapper::MallocCopy(v);
}

// 设置标量值的通用函数
template <typename T>
ge::graphStatus SetScalar(void *av, T value) {
  auto *attr_value = reinterpret_cast<ge::AttrValue *>(av);
  if (attr_value == nullptr) {
    return ge::GRAPH_FAILED;
  }
  return attr_value->SetAttrValue(value);
}

// 设置POD类型列表的通用函数
template <typename T>
ge::graphStatus SetListPOD(void *av, const T *data, int64_t size) {
  auto *attr_value = reinterpret_cast<ge::AttrValue *>(av);
  if ((attr_value == nullptr) || (size < 0)) {
    return ge::GRAPH_FAILED;
  }

  std::vector<T> v;
  if ((size > 0) && (data != nullptr)) {
    v.assign(data, data + static_cast<size_t>(size));
  }
  return attr_value->SetAttrValue(v);
}
}  // namespace

#ifdef __cplusplus
extern "C" {
#endif

// 创建与销毁
void *GeApiWrapper_AttrValue_Create() {
  return reinterpret_cast<void *>(new (std::nothrow) ge::AttrValue());
}

void GeApiWrapper_AttrValue_Destroy(void *av) {
  delete reinterpret_cast<ge::AttrValue *>(av);
}

// 标量值释放函数（Python使用）
void GeApiWrapper_FreeString(char *p) {
  ge::c_wrapper::SafeFree(p);
}

// 列表值释放函数（Python使用）
void GeApiWrapper_FreeListFloat(float *p) {
  ge::c_wrapper::SafeFree(p);
}

void GeApiWrapper_FreeListBool(bool *p) {
  ge::c_wrapper::SafeFree(p);
}

void GeApiWrapper_FreeListInt(int64_t *p) {
  ge::c_wrapper::SafeFree(p);
}

void GeApiWrapper_FreeListDataType(ge::DataType *p) {
  ge::c_wrapper::SafeFree(p);
}

// 需要配合MallocCopyListString使用
void GeApiWrapper_FreeListString(char **p) {
  ge::c_wrapper::SafeFree(p);
}

// 值类型探测
int32_t GeApiWrapper_AttrValue_GetValueType(const void *av) {
  const auto *attr_value = reinterpret_cast<const ge::AttrValue *>(av);
  if (attr_value == nullptr) {
    return -1;
  }
  return static_cast<int32_t>(attr_value->impl->MutableAnyValue().GetValueType());
}

// 标量值操作 - 直接使用模板函数
ge::graphStatus GeApiWrapper_AttrValue_SetFloat(void *av, float value) {
  return SetScalar<float>(av, value);
}

ge::graphStatus GeApiWrapper_AttrValue_GetFloat(const void *av, float *value) {
  const auto *attr_value = reinterpret_cast<const ge::AttrValue *>(av);
  return GetScalar<float>(attr_value, value);
}

ge::graphStatus GeApiWrapper_AttrValue_SetBool(void *av, bool value) {
  return SetScalar<bool>(av, value);
}

ge::graphStatus GeApiWrapper_AttrValue_GetBool(const void *av, bool *value) {
  const auto *attr_value = reinterpret_cast<const ge::AttrValue *>(av);
  return GetScalar<bool>(attr_value, value);
}

ge::graphStatus GeApiWrapper_AttrValue_SetInt(void *av, int64_t value) {
  return SetScalar<int64_t>(av, value);
}

ge::graphStatus GeApiWrapper_AttrValue_GetInt(const void *av, int64_t *value) {
  const auto *attr_value = reinterpret_cast<const ge::AttrValue *>(av);
  return GetScalar<int64_t>(attr_value, value);
}

ge::graphStatus GeApiWrapper_AttrValue_SetDataType(void *av, ge::DataType value) {
  return SetScalar<ge::DataType>(av, value);
}

ge::graphStatus GeApiWrapper_AttrValue_GetDataType(const void *av, ge::DataType *value) {
  const auto *attr_value = reinterpret_cast<const ge::AttrValue *>(av);
  return GetScalar<ge::DataType>(attr_value, value);
}

// 列表值操作 - 直接使用模板函数
ge::graphStatus GeApiWrapper_AttrValue_SetListFloat(void *av, const float *data, int64_t size) {
  return SetListPOD<float>(av, data, size);
}

float *GeApiWrapper_AttrValue_GetListFloat(const void *av, int64_t *size) {
  const auto *attr_value = reinterpret_cast<const ge::AttrValue *>(av);
  return GetListPOD<float>(attr_value, size);
}

ge::graphStatus GeApiWrapper_AttrValue_SetListInt(void *av, const int64_t *data, int64_t size) {
  return SetListPOD<int64_t>(av, data, size);
}

int64_t *GeApiWrapper_AttrValue_GetListInt(const void *av, int64_t *size) {
  const auto *attr_value = reinterpret_cast<const ge::AttrValue *>(av);
  return GetListPOD<int64_t>(attr_value, size);
}

ge::graphStatus GeApiWrapper_AttrValue_SetListDataType(void *av, const ge::DataType *data, int64_t size) {
  return SetListPOD<ge::DataType>(av, data, size);
}

ge::DataType *GeApiWrapper_AttrValue_GetListDataType(const void *av, int64_t *size) {
  const auto *attr_value = reinterpret_cast<const ge::AttrValue *>(av);
  return GetListPOD<ge::DataType>(attr_value, size);
}

// 字符串特殊处理
char *GeApiWrapper_AttrValue_GetString(const void *av) {
  const auto *attr_value = reinterpret_cast<const ge::AttrValue *>(av);
  if (attr_value == nullptr) {
    return nullptr;
  }
  ge::AscendString str_val;
  if (attr_value->GetAttrValue(str_val) != ge::GRAPH_SUCCESS) {
    return nullptr;
  }
  return ge::c_wrapper::MallocCopyString(str_val.GetString());
}

ge::graphStatus GeApiWrapper_AttrValue_SetString(void *av, const char *value) {
  auto *attr_value = reinterpret_cast<ge::AttrValue *>(av);
  if (attr_value == nullptr || value == nullptr) {
    return ge::GRAPH_FAILED;
  }
  ge::AscendString str_val(value);
  return attr_value->SetAttrValue(str_val);
}

// bool列表特殊处理（因为std::vector<bool>的特殊性）
bool *GeApiWrapper_AttrValue_GetListBool(const void *av, int64_t *size) {
  const auto *attr_value = reinterpret_cast<const ge::AttrValue *>(av);
  if (attr_value == nullptr || size == nullptr) {
    return nullptr;
  }
  std::vector<bool> v;
  if (attr_value->GetAttrValue(v) != ge::GRAPH_SUCCESS || v.empty()) {
    return nullptr;
  }
  *size = static_cast<int64_t>(v.size());
  bool *buf = static_cast<bool *>(std::malloc(sizeof(bool) * static_cast<size_t>(*size)));
  if (buf == nullptr) {
    return nullptr;
  }
  for (int64_t i = 0; i < *size; ++i) {
    buf[i] = v[static_cast<size_t>(i)];
  }
  return buf;
}

ge::graphStatus GeApiWrapper_AttrValue_SetListBool(void *av, const bool *data, int64_t size) {
  auto *attr_value = reinterpret_cast<ge::AttrValue *>(av);
  if (attr_value == nullptr || size < 0) {
    return ge::GRAPH_FAILED;
  }

  std::vector<bool> v;
  if (size > 0 && data != nullptr) {
    v.reserve(static_cast<size_t>(size));
    for (int64_t i = 0; i < size; ++i) {
      v.push_back(data[i]);
    }
  }
  return attr_value->SetAttrValue(v);
}

// 字符串列表特殊处理
char **GeApiWrapper_AttrValue_GetListString(const void *av, int64_t *size) {
  const auto *attr_value = reinterpret_cast<const ge::AttrValue *>(av);
  if (attr_value == nullptr || size == nullptr) {
    return nullptr;
  }
  std::vector<ge::AscendString> v;
  if (attr_value->GetAttrValue(v) != ge::GRAPH_SUCCESS || v.empty()) {
    return nullptr;
  }
  return ge::c_wrapper::MallocCopyListString(v, size);
}

ge::graphStatus GeApiWrapper_AttrValue_SetListString(void *av, const char *const *data, int64_t size) {
  auto *attr_value = reinterpret_cast<ge::AttrValue *>(av);
  if (attr_value == nullptr || size < 0) {
    return ge::GRAPH_FAILED;
  }
  std::vector<ge::AscendString> v;
  if (size > 0 && data != nullptr) {
    v.reserve(static_cast<size_t>(size));
    for (int64_t i = 0; i < size; ++i) {
      v.emplace_back(data[i] == nullptr ? "" : data[i]);
    }
  }
  return attr_value->SetAttrValue(v);
}

#ifdef __cplusplus
}
#endif
