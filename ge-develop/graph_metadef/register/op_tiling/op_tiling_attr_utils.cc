/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "register/op_tiling_attr_utils.h"
#include <vector>
#include <functional>
#include "framework/common/debug/ge_log.h"
#include "op_tiling/op_tiling_utils.h"
#include "common/util/tiling_utils.h"

namespace optiling {
template<typename T>
class AttrDataImpl : public AttrData {
public:
  explicit AttrDataImpl(std::vector<T> &value) : data_value_(std::move(value)) {}
  ~AttrDataImpl() override {}
  size_t GetSize() const override {
    return data_value_.size() * sizeof(T);
  }
  const std::uint8_t *GetData() override {
    if (data_value_.empty()) {
      return nullptr;
    } else {
      return reinterpret_cast<const uint8_t *>(data_value_.data());
    }
  }

private:
  std::vector<T> data_value_;
};

enum class AttrDataType {
  BOOL,
  STRING,
  INT32,
  UINT32,
  FLOAT32,
  FLOAT16,
  BFLOAT16,
  LIST_BOOL,
  LIST_STRING,
  LIST_INT32,
  LIST_UINT32,
  LIST_FLOAT32,
  LIST_FLOAT16,
  LIST_LIST_INT32
};

static const std::map<std::string, AttrDataType> kAttrDataTypeMap {
  {"bool", AttrDataType::BOOL},
  {"str", AttrDataType::STRING},
  {"string", AttrDataType::STRING},
  {"int", AttrDataType::INT32},
  {"int32", AttrDataType::INT32},
  {"uint", AttrDataType::UINT32},
  {"uint32", AttrDataType::UINT32},
  {"float", AttrDataType::FLOAT32},
  {"float32", AttrDataType::FLOAT32},
  {"float16", AttrDataType::FLOAT16},
  {"bfloat16", AttrDataType::BFLOAT16},
  {"list_int", AttrDataType::LIST_INT32},
  {"list_int32", AttrDataType::LIST_INT32},
  {"list_uint", AttrDataType::LIST_UINT32},
  {"list_uint32", AttrDataType::LIST_UINT32},
  {"list_float16", AttrDataType::LIST_FLOAT16},
  {"list_float", AttrDataType::LIST_FLOAT32},
  {"list_float32", AttrDataType::LIST_FLOAT32}
};

static const std::vector<AttrDataType> kValidSrcDTypeList {
  AttrDataType::BOOL,
  AttrDataType::STRING,
  AttrDataType::INT32,
  AttrDataType::FLOAT32,
  AttrDataType::LIST_INT32,
  AttrDataType::LIST_FLOAT32
};

static const std::vector<AttrDataType> kValidDstDTypeList {
  AttrDataType::UINT32,
  AttrDataType::INT32,
  AttrDataType::FLOAT32,
  AttrDataType::FLOAT16,
  AttrDataType::BFLOAT16,
  AttrDataType::LIST_INT32,
  AttrDataType::LIST_UINT32,
  AttrDataType::LIST_FLOAT16,
  AttrDataType::LIST_FLOAT32
};

static const uint32_t kBitsOfByte = 8;

inline uint32_t GenerateAttrFuncKey(const AttrDataType attr_dtype) {
  return ((static_cast<uint32_t>(attr_dtype) & 0xFFU) << kBitsOfByte) | (static_cast<uint32_t>(attr_dtype) & 0xFFU);
}

inline uint32_t GenerateAttrFuncKey(const AttrDataType src_dtype, const AttrDataType dest_dtype) {
  return ((static_cast<uint32_t>(src_dtype) & 0xFFU) << kBitsOfByte) | (static_cast<uint32_t>(dest_dtype) & 0xFFU);
}

class AttrDataManager;
using GetOpAttrValueFunc = std::function<AttrDataPtr(AttrDataManager *, const ge::Operator &, const char *)>;

class AttrDataManager {
public:
  AttrDataManager(const AttrDataManager &) = delete;
  AttrDataManager &operator=(const AttrDataManager &) = delete;
  static AttrDataManager& Instance() {
    static AttrDataManager attr_data_manager;
    return attr_data_manager;
  }

  bool VerifyAttrDtype(const AttrDataType src_dtype, const AttrDataType dst_dtype) {
    const uint32_t func_key = GenerateAttrFuncKey(src_dtype, dst_dtype);
    const auto iter = attr_func_.find(func_key);
    return iter != attr_func_.end();
  }

  AttrDataPtr GetOpAttrValue(const ge::Operator &op, const char *attr_name, const AttrDataType src_dtype,
                             const AttrDataType dst_dtype) {
    const uint32_t func_key = GenerateAttrFuncKey(src_dtype, dst_dtype);
    const auto iter = attr_func_.find(func_key);
    if (iter == attr_func_.end()) {
      return nullptr;
    }
    return iter->second(this, op, attr_name);
  }

private:
  AttrDataManager() {}
  ~AttrDataManager() {}
  template<typename T, bool IsList = false, typename std::enable_if<!IsList, bool>::type = true>
  AttrDataPtr GetAttrValue(const ge::Operator &op, const char *attr_name) const {
    T attr_value;
    if (op.GetAttr(attr_name, attr_value) != ge::GRAPH_SUCCESS) {
      GELOGW("Failed to retrieve attribute [%s] from op.", attr_name);
      return nullptr;
    }
    std::vector<T> attr_vec;
    attr_vec.push_back(attr_value);
    AttrDataPtr attr_data_ptr = nullptr;
    OP_TILING_MAKE_SHARED(attr_data_ptr = std::make_shared<AttrDataImpl<T>>(attr_vec), return nullptr);
    return attr_data_ptr;
  }

  template<typename T, bool IsList = false, typename std::enable_if<IsList, bool>::type = true>
  AttrDataPtr GetAttrValue(const ge::Operator &op, const char *attr_name) const {
    std::vector<T> attr_vec;
    if (op.GetAttr(attr_name, attr_vec) != ge::GRAPH_SUCCESS) {
      GELOGW("Failed to retrieve attribute [%s] from op.", attr_name);
      return nullptr;
    }
    if (attr_vec.empty()) {
      GELOGW("The vector value of attribute [%s] is empty.", attr_name);
      return nullptr;
    }
    AttrDataPtr attr_data_ptr = nullptr;
    OP_TILING_MAKE_SHARED(attr_data_ptr = std::make_shared<AttrDataImpl<T>>(attr_vec), return nullptr);
    return attr_data_ptr;
  }

  AttrDataPtr GetBoolAttrValue(const ge::Operator &op, const char *attr_name) const {
    bool attr_value = false;
    if (op.GetAttr(attr_name, attr_value) != ge::GRAPH_SUCCESS) {
      GELOGW("Failed to retrieve attribute [%s] from op.", attr_name);
      return nullptr;
    }

    std::vector<int8_t> attr_vec;
    attr_vec.push_back(static_cast<int8_t>(attr_value));

    AttrDataPtr attr_data_ptr = nullptr;
    OP_TILING_MAKE_SHARED(attr_data_ptr = std::make_shared<AttrDataImpl<int8_t>>(attr_vec), return nullptr);
    return attr_data_ptr;
  }

  AttrDataPtr GetStrAttrValue(const ge::Operator &op, const char *attr_name) const {
    std::string attr_value;
    if (op.GetAttr(attr_name, attr_value) != ge::GRAPH_SUCCESS) {
      GELOGW("Failed to retrieve attribute [%s] from op.", attr_name);
      return nullptr;
    }

    std::vector<char> attr_vec;
    for (const char &c : attr_value) {
      attr_vec.push_back(c);
    }

    AttrDataPtr attr_data_ptr = nullptr;
    OP_TILING_MAKE_SHARED(attr_data_ptr = std::make_shared<AttrDataImpl<char>>(attr_vec), return nullptr);
    return attr_data_ptr;
  }

  AttrDataPtr GetIntAttrValueAndToUint(const ge::Operator &op, const char *attr_name) const {
    int32_t attr_value = 0;
    if (op.GetAttr(attr_name, attr_value) != ge::GRAPH_SUCCESS) {
      GELOGW("Failed to retrieve attribute [%s] from op.", attr_name);
      return nullptr;
    }

    std::vector<uint32_t> attr_vec;
    attr_vec.push_back(static_cast<uint32_t>(attr_value));

    AttrDataPtr attr_data_ptr = nullptr;
    OP_TILING_MAKE_SHARED(attr_data_ptr = std::make_shared<AttrDataImpl<uint32_t>>(attr_vec), return nullptr);
    return attr_data_ptr;
  }

  AttrDataPtr GetListIntAttrValueAndToListUint(const ge::Operator &op, const char *attr_name) const {
    std::vector<int32_t> attr_value;
    if (op.GetAttr(attr_name, attr_value) != ge::GRAPH_SUCCESS) {
      GELOGW("Failed to retrieve attribute [%s] from op.", attr_name);
      return nullptr;
    }

    if (attr_value.empty()) {
      GELOGW("The vector value of attribute [%s] is empty.", attr_name);
      return nullptr;
    }

    std::vector<uint32_t> attr_vec;
    for (const int32_t &int_value : attr_value) {
      attr_vec.push_back(static_cast<uint32_t>(int_value));
    }

    AttrDataPtr attr_data_ptr = nullptr;
    OP_TILING_MAKE_SHARED(attr_data_ptr = std::make_shared<AttrDataImpl<uint32_t>>(attr_vec), return nullptr);
    return attr_data_ptr;
  }

  AttrDataPtr GetFloatAttrValueAndToFp16(const ge::Operator &op, const char *attr_name) const {
    float attr_value = 0.0F;
    if (op.GetAttr(attr_name, attr_value) != ge::GRAPH_SUCCESS) {
      GELOGW("Failed to retrieve attribute [%s] from op.", attr_name);
      return nullptr;
    }

    std::vector<uint16_t> attr_vec;
    attr_vec.push_back(Float32ToFloat16(attr_value));

    AttrDataPtr attr_data_ptr = nullptr;
    OP_TILING_MAKE_SHARED(attr_data_ptr = std::make_shared<AttrDataImpl<uint16_t>>(attr_vec), return nullptr);
    return attr_data_ptr;
  }

  AttrDataPtr GetFloatAttrValueAndToBf16(const ge::Operator &op, const char *attr_name) const {
    float attr_value = 0.0F;
    if (op.GetAttr(attr_name, attr_value) != ge::GRAPH_SUCCESS) {
      GELOGW("Failed to retrieve attribute [%s] from op.", attr_name);
      return nullptr;
    }

    std::vector<uint16_t> attr_vec;
    attr_vec.push_back(Float32ToBfloat16(attr_value));

    AttrDataPtr attr_data_ptr = nullptr;
    OP_TILING_MAKE_SHARED(attr_data_ptr = std::make_shared<AttrDataImpl<uint16_t>>(attr_vec), return nullptr);
    return attr_data_ptr;
  }

  AttrDataPtr GetFloatAttrValueAndToInt(const ge::Operator &op, const char *attr_name) const {
    float attr_value = 0.0F;
    if (op.GetAttr(attr_name, attr_value) != ge::GRAPH_SUCCESS) {
      GELOGW("Failed to retrieve attribute [%s] from op.", attr_name);
      return nullptr;
    }

    std::vector<int32_t> attr_vec;
    attr_vec.push_back(static_cast<int32_t>(attr_value));

    AttrDataPtr attr_data_ptr = nullptr;
    OP_TILING_MAKE_SHARED(attr_data_ptr = std::make_shared<AttrDataImpl<int32_t>>(attr_vec), return nullptr);
    return attr_data_ptr;
  }

  AttrDataPtr GetListFloatAttrValueAndToListFp16(const ge::Operator &op, const char *attr_name) const {
    std::vector<float> attr_value;
    if (op.GetAttr(attr_name, attr_value) != ge::GRAPH_SUCCESS) {
      GELOGW("Failed to retrieve attribute [%s] from op.", attr_name);
      return nullptr;
    }

    if (attr_value.empty()) {
      GELOGW("The vector value of attribute [%s] is empty.", attr_name);
      return nullptr;
    }

    std::vector<uint16_t> attr_vec;
    for (const float &fp_value : attr_value) {
      attr_vec.push_back(Float32ToFloat16(fp_value));
    }

    AttrDataPtr attr_data_ptr = nullptr;
    OP_TILING_MAKE_SHARED(attr_data_ptr = std::make_shared<AttrDataImpl<uint16_t>>(attr_vec), return nullptr);
    return attr_data_ptr;
  }

  AttrDataPtr GetListFloatAttrValueAndToListInt(const ge::Operator &op, const char *attr_name) const {
    std::vector<float> attr_value;
    if (op.GetAttr(attr_name, attr_value) != ge::GRAPH_SUCCESS) {
      GELOGW("Failed to retrieve attribute [%s] from op.", attr_name);
      return nullptr;
    }

    if (attr_value.empty()) {
      GELOGW("The vector value of attribute [%s] is empty.", attr_name);
      return nullptr;
    }

    std::vector<int32_t> attr_vec;
    for (const float &fp_value : attr_value) {
      attr_vec.push_back(static_cast<int32_t>(fp_value));
    }

    AttrDataPtr attr_data_ptr = nullptr;
    OP_TILING_MAKE_SHARED(attr_data_ptr = std::make_shared<AttrDataImpl<int32_t>>(attr_vec), return nullptr);
    return attr_data_ptr;
  }

  static const std::map<uint32_t, GetOpAttrValueFunc> attr_func_;
};

const std::map<uint32_t, GetOpAttrValueFunc> AttrDataManager::attr_func_ = {
    {GenerateAttrFuncKey(AttrDataType::BOOL), &AttrDataManager::GetBoolAttrValue},
    {GenerateAttrFuncKey(AttrDataType::INT32), &AttrDataManager::GetAttrValue<int32_t>},
    {GenerateAttrFuncKey(AttrDataType::FLOAT32), &AttrDataManager::GetAttrValue<float>},
    {GenerateAttrFuncKey(AttrDataType::LIST_INT32), &AttrDataManager::GetAttrValue<int32_t, true>},
    {GenerateAttrFuncKey(AttrDataType::LIST_FLOAT32), &AttrDataManager::GetAttrValue<float, true>},
    {GenerateAttrFuncKey(AttrDataType::STRING), &AttrDataManager::GetStrAttrValue},
    {GenerateAttrFuncKey(AttrDataType::INT32, AttrDataType::UINT32),
     &AttrDataManager::GetIntAttrValueAndToUint},
    {GenerateAttrFuncKey(AttrDataType::LIST_INT32, AttrDataType::LIST_UINT32),
     &AttrDataManager::GetListIntAttrValueAndToListUint},
    {GenerateAttrFuncKey(AttrDataType::FLOAT32, AttrDataType::FLOAT16),
     &AttrDataManager::GetFloatAttrValueAndToFp16},
    {GenerateAttrFuncKey(AttrDataType::FLOAT32, AttrDataType::BFLOAT16),
     &AttrDataManager::GetFloatAttrValueAndToBf16},
    {GenerateAttrFuncKey(AttrDataType::LIST_FLOAT32, AttrDataType::LIST_FLOAT16),
     &AttrDataManager::GetListFloatAttrValueAndToListFp16},
    {GenerateAttrFuncKey(AttrDataType::FLOAT32, AttrDataType::INT32),
     &AttrDataManager::GetFloatAttrValueAndToInt},
    {GenerateAttrFuncKey(AttrDataType::LIST_FLOAT32, AttrDataType::LIST_INT32),
     &AttrDataManager::GetListFloatAttrValueAndToListInt}
};

ge::graphStatus GetOperatorAttrValue(const ge::Operator &op, const char *attr_name, const char *attr_dtype,
                                     AttrDataPtr &attr_data_ptr, const char *target_dtype) {
  ge::AscendString op_name;
  ge::AscendString op_type;
  (void)op.GetName(op_name);
  (void)op.GetOpType(op_type);
  if (attr_name == nullptr || attr_dtype == nullptr) {
    GE_LOGE("Attribute name or attribute data type is null for op [%s].", op_name.GetString());
    return ge::GRAPH_FAILED;
  }

  GELOGD("Begin to retrieve attribute [%s] of data type [%s] from op [%s, %s].",
         attr_name, attr_dtype, op_name.GetString(), op_type.GetString());
  const std::string attr_dtype_str = attr_dtype;
  auto iter = kAttrDataTypeMap.find(attr_dtype_str);
  if (iter == kAttrDataTypeMap.end()) {
    GELOGW("Attr data type [%s] for attribute [%s] is not supported.", attr_dtype, attr_name);
    return ge::GRAPH_FAILED;
  }
  const AttrDataType src_dtype = iter->second;
  if (std::find(kValidSrcDTypeList.begin(), kValidSrcDTypeList.end(), src_dtype) == kValidSrcDTypeList.end()) {
    GELOGW("Attr data type [%s] for attribute [%s] is not supported.", attr_dtype, attr_name);
    return ge::GRAPH_FAILED;
  }
  AttrDataType dst_dtype = src_dtype;
  if (target_dtype != nullptr) {
    GELOGD("Attempting to retrieve attribute [%s] and transform its value from [%s] to [%s].",
           attr_name, attr_dtype, target_dtype);
    const std::string target_dtype_str = target_dtype;
    iter = kAttrDataTypeMap.find(target_dtype_str);
    if (iter == kAttrDataTypeMap.end()) {
      GELOGW("Target attr data type[%s] of attr[%s] is not supported.", target_dtype, attr_name);
      return ge::GRAPH_FAILED;
    }
    dst_dtype = iter->second;
    if (dst_dtype != src_dtype) {
      if (std::find(kValidDstDTypeList.begin(), kValidDstDTypeList.end(), dst_dtype) == kValidDstDTypeList.end()) {
        GELOGW("Target attr data type[%s] of attr[%s] is not supported.", target_dtype, attr_name);
        return ge::GRAPH_FAILED;
      }
      if (!AttrDataManager::Instance().VerifyAttrDtype(src_dtype, dst_dtype)) {
        GELOGW("Get attr[%s] and transform from [%s] to [%s] is not supported.",
               attr_name, attr_dtype, target_dtype);
        return ge::GRAPH_FAILED;
      }
    }
  }
  attr_data_ptr = AttrDataManager::Instance().GetOpAttrValue(op, attr_name, src_dtype, dst_dtype);
  GELOGD("Finished getting attr [%s] of data type [%s] from op [%s, %s].",
         attr_name, attr_dtype, op_name.GetString(), op_type.GetString());
  return attr_data_ptr == nullptr ? ge::GRAPH_FAILED : ge::GRAPH_SUCCESS;
}
}  // namespace optiling
