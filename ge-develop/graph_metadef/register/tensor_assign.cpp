/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <complex>
#include <limits>
#include <map>
#include <memory>
#include "securec.h"
#include "framework/common/debug/ge_log.h"
#include "graph_metadef/graph/debug/ge_util.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/type_utils.h"
#include "graph/utils/type_utils_inner.h"
#include "graph/utils/attr_utils.h"
#include "register/register_error_codes.h"
#include "graph/types.h"
#include "graph/def_types.h"
#include "graph_metadef/graph/debug/ge_util.h"
#include "register/tensor_assign.h"

namespace domi {
namespace {
using GeTensorDesc = ge::GeTensorDesc;
using GeShape = ge::GeShape;
using domi::tensorflow::TensorProto;
using google::protobuf::int32;
using google::protobuf::int64;
const char_t *const kOriginElementNumAttrName = "origin_element_num";
const std::map<uint32_t, ge::DataType> data_type_map = {
    {domi::tensorflow::DataType::DT_FLOAT, ge::DataType::DT_FLOAT},
    {domi::tensorflow::DataType::DT_HALF, ge::DataType::DT_FLOAT16},
    {domi::tensorflow::DataType::DT_INT8, ge::DataType::DT_INT8},
    {domi::tensorflow::DataType::DT_INT16, ge::DataType::DT_INT16},
    {domi::tensorflow::DataType::DT_UINT16, ge::DataType::DT_UINT16},
    {domi::tensorflow::DataType::DT_UINT8, ge::DataType::DT_UINT8},
    {domi::tensorflow::DataType::DT_INT32, ge::DataType::DT_INT32},
    {domi::tensorflow::DataType::DT_INT64, ge::DataType::DT_INT64},
    {domi::tensorflow::DataType::DT_UINT32, ge::DataType::DT_UINT32},
    {domi::tensorflow::DataType::DT_UINT64, ge::DataType::DT_UINT64},
    {domi::tensorflow::DataType::DT_BOOL, ge::DataType::DT_BOOL},
    {domi::tensorflow::DataType::DT_DOUBLE, ge::DataType::DT_DOUBLE},
    {domi::tensorflow::DataType::DT_COMPLEX32, ge::DataType::DT_COMPLEX32},
    {domi::tensorflow::DataType::DT_COMPLEX64, ge::DataType::DT_COMPLEX64},
    {domi::tensorflow::DataType::DT_QINT8, ge::DataType::DT_INT8},
    {domi::tensorflow::DataType::DT_QUINT8, ge::DataType::DT_UINT8},
    {domi::tensorflow::DataType::DT_QINT32, ge::DataType::DT_INT32},
    {domi::tensorflow::DataType::DT_QINT16, ge::DataType::DT_INT16},
    {domi::tensorflow::DataType::DT_QUINT16, ge::DataType::DT_UINT16},
    {domi::tensorflow::DataType::DT_COMPLEX128, ge::DataType::DT_COMPLEX128},
    {domi::tensorflow::DataType::DT_RESOURCE, ge::DataType::DT_RESOURCE},
    {domi::tensorflow::DataType::DT_BFLOAT16, ge::DataType::DT_BF16},
    {domi::tensorflow::DataType::DT_STRING, ge::DataType::DT_STRING},
    {domi::tensorflow::DataType::DT_FLOAT_REF, ge::DataType::DT_FLOAT},
    {domi::tensorflow::DataType::DT_DOUBLE_REF, ge::DataType::DT_DOUBLE},
    {domi::tensorflow::DataType::DT_INT32_REF, ge::DataType::DT_INT32},
    {domi::tensorflow::DataType::DT_INT8_REF, ge::DataType::DT_INT8},
    {domi::tensorflow::DataType::DT_UINT8_REF, ge::DataType::DT_UINT8},
    {domi::tensorflow::DataType::DT_INT16_REF, ge::DataType::DT_INT16},
    {domi::tensorflow::DataType::DT_UINT16_REF, ge::DataType::DT_UINT16},
    {domi::tensorflow::DataType::DT_COMPLEX32_REF, ge::DataType::DT_COMPLEX32},
    {domi::tensorflow::DataType::DT_COMPLEX64_REF, ge::DataType::DT_COMPLEX64},
    {domi::tensorflow::DataType::DT_QINT8_REF, ge::DataType::DT_INT8},
    {domi::tensorflow::DataType::DT_QUINT8_REF, ge::DataType::DT_UINT8},
    {domi::tensorflow::DataType::DT_QINT32_REF, ge::DataType::DT_INT32},
    {domi::tensorflow::DataType::DT_QINT16_REF, ge::DataType::DT_INT16},
    {domi::tensorflow::DataType::DT_QUINT16_REF, ge::DataType::DT_UINT16},
    {domi::tensorflow::DataType::DT_COMPLEX128_REF, ge::DataType::DT_COMPLEX128},
    {domi::tensorflow::DataType::DT_RESOURCE_REF, ge::DataType::DT_RESOURCE},
    {domi::tensorflow::DataType::DT_BFLOAT16_REF, ge::DataType::DT_FLOAT16},
    {domi::tensorflow::DataType::DT_UINT32_REF, ge::DataType::DT_UINT32},
    {domi::tensorflow::DataType::DT_UINT64_REF, ge::DataType::DT_UINT64},
    {domi::tensorflow::DataType::DT_INT64_REF, ge::DataType::DT_INT64},
    {domi::tensorflow::DataType::DT_BOOL_REF, ge::DataType::DT_BOOL},
    {domi::tensorflow::DataType::DT_HALF_REF, ge::DataType::DT_FLOAT16},
    {domi::tensorflow::DataType::DT_STRING_REF, ge::DataType::DT_STRING},
    {domi::tensorflow::DataType::DT_VARIANT, ge::DataType::DT_VARIANT},
};
}  // namespace

ge::DataType TensorAssign::ConvertTensorflowDataType(const uint32_t tf_data_type) {
  const auto search = data_type_map.find(tf_data_type);
  if (search != data_type_map.end()) {
    return search->second;
  } else {
    return ge::DataType::DT_UNDEFINED;
  }
}

bool TensorAssign::CheckBoolVal(const tensorflow::DataType data_type) {
  return ((data_type == tensorflow::DT_BOOL) || (data_type == tensorflow::DT_BOOL_REF));
}

bool TensorAssign::CheckHalfVal(const tensorflow::DataType data_type) {
  return ((data_type == tensorflow::DT_HALF) || (data_type == tensorflow::DT_BFLOAT16) ||
          (data_type == tensorflow::DT_HALF_REF) || (data_type == tensorflow::DT_BFLOAT16_REF));
}

bool TensorAssign::CheckFloatVal(const tensorflow::DataType data_type) {
  return ((data_type == tensorflow::DT_FLOAT) || (data_type == tensorflow::DT_FLOAT_REF));
}

bool TensorAssign::CheckDoubleVal(const tensorflow::DataType data_type) {
  return ((data_type == tensorflow::DT_DOUBLE) || (data_type == tensorflow::DT_DOUBLE_REF));
}

bool TensorAssign::CheckComplex32Val(const tensorflow::DataType data_type) {
  return ((data_type == tensorflow::DT_COMPLEX32) || (data_type == tensorflow::DT_COMPLEX32_REF));
}

bool TensorAssign::CheckComplex64Val(const tensorflow::DataType data_type) {
  return ((data_type == tensorflow::DT_COMPLEX64) || (data_type == tensorflow::DT_COMPLEX64_REF));
}

bool TensorAssign::CheckComplex128Val(const tensorflow::DataType data_type) {
  return ((data_type == tensorflow::DT_COMPLEX128) || (data_type == tensorflow::DT_COMPLEX128_REF));
}

bool TensorAssign::CheckStringVal(const tensorflow::DataType data_type) {
  return ((data_type == tensorflow::DT_STRING) || (data_type == tensorflow::DT_STRING_REF));
}

bool TensorAssign::CheckByte(const tensorflow::DataType data_type) {
  return ((data_type == tensorflow::DT_UINT8) || (data_type == tensorflow::DT_INT8) ||
          (data_type == tensorflow::DT_QINT8) || (data_type == tensorflow::DT_QUINT8) ||
          (data_type == tensorflow::DT_UINT8_REF) || (data_type == tensorflow::DT_INT8_REF) ||
          (data_type == tensorflow::DT_QINT8_REF) || (data_type == tensorflow::DT_QUINT8_REF));
}

bool TensorAssign::CheckDoubleByte(const tensorflow::DataType data_type) {
  return ((data_type == tensorflow::DT_INT16) || (data_type == tensorflow::DT_UINT16) ||
          (data_type == tensorflow::DT_QINT16) || (data_type == tensorflow::DT_QUINT16) ||
          (data_type == tensorflow::DT_INT16_REF) || (data_type == tensorflow::DT_UINT16_REF) ||
          (data_type == tensorflow::DT_QINT16_REF) || (data_type == tensorflow::DT_QUINT16_REF));
}

bool TensorAssign::CheckSignedFourByte(const tensorflow::DataType data_type) {
  return ((data_type == tensorflow::DT_INT32) || (data_type == tensorflow::DT_QINT32) ||
          (data_type == tensorflow::DT_INT32_REF) || (data_type == tensorflow::DT_QINT32_REF));
}

bool TensorAssign::CheckUnsignedFourByte(const tensorflow::DataType data_type) {
  return ((data_type == tensorflow::DT_UINT32) || (data_type == tensorflow::DT_UINT32_REF));
}

bool TensorAssign::CheckSignedEightByte(const tensorflow::DataType data_type) {
  return ((data_type == tensorflow::DT_INT64) || (data_type == tensorflow::DT_INT64_REF));
}

bool TensorAssign::CheckUnsignedEightByte(const tensorflow::DataType data_type) {
  return ((data_type == tensorflow::DT_UINT64) || (data_type == tensorflow::DT_UINT64_REF));
}

Status TensorAssign::GetDoubleByteVal(const int64_t val_size, const google::protobuf::RepeatedField<int32> &val_vector,
                                      const int64_t count, GeTensorPtr &weight) {
  GE_CHECK_NOTNULL(weight);
  const bool zerosLike = ((count != val_size) && (val_size == 1));
  std::vector<uint16_t> addr(static_cast<uint64_t>(count));
  if (val_size == 0) {  // addr has been zero initialized
    (void)weight->SetData(ge::PtrToPtr<uint16_t, uint8_t>(addr.data()), static_cast<size_t>(count) * sizeof(uint16_t));
    return SUCCESS;
  }
  if (!zerosLike) {
    const int64_t minCount = (count > val_size) ? val_size : count;
    for (int64_t i = 0; i < minCount; i++) {
      GE_ASSERT_EQ(ge::IntegerChecker<int32_t>::Compat(i), true);
      addr[static_cast<uint64_t>(i)] = static_cast<uint16_t>(val_vector.Get(static_cast<int32_t>(i)));
    }
    const int64_t value_index = minCount - 1;
    GE_ASSERT_EQ(ge::IntegerChecker<int32_t>::Compat(value_index), true);
    for (int64_t i = minCount; i < count; i++) {
      addr[static_cast<uint64_t>(i)] = static_cast<uint16_t>(val_vector.Get(static_cast<int32_t>(value_index)));
    }
  } else {
    for (int64_t i = 0; i < count; i++) {
      addr[static_cast<uint64_t>(i)] = static_cast<uint16_t>(val_vector.Get(0));
    }
  }
  (void)weight->SetData(ge::PtrToPtr<uint16_t, uint8_t>(addr.data()), static_cast<size_t>(count) * sizeof(uint16_t));
  return SUCCESS;
}

Status TensorAssign::GetByteVal(const int64_t val_size, const google::protobuf::RepeatedField<int32> &val_vector,
                                const int64_t count, GeTensorPtr &weight) {
  GE_CHECK_NOTNULL(weight);
  const bool zerosLike = ((count != val_size) && (val_size == 1));
  std::vector<uint8_t> addr(static_cast<uint64_t>(count));
  if (val_size == 0) {  // addr has been zero initialized
    (void)weight->SetData(addr.data(), static_cast<size_t>(count) * sizeof(uint8_t));
    return SUCCESS;
  }
  if (!zerosLike) {
    const int64_t minCount = (count > val_size) ? val_size : count;
    for (int64_t i = 0; i < minCount; i++) {
      GE_ASSERT_EQ(ge::IntegerChecker<int32_t>::Compat(i), true);
      addr[static_cast<uint64_t>(i)] = static_cast<uint8_t>(val_vector.Get(static_cast<int32_t>(i)));
    }
    const int64_t value_index = minCount - 1;
    GE_ASSERT_EQ(ge::IntegerChecker<int32_t>::Compat(value_index), true);
    for (int64_t i = minCount; i < count; i++) {
      addr[static_cast<uint64_t>(i)] = static_cast<uint8_t>(val_vector.Get(static_cast<int32_t>(value_index)));
    }
  } else {
    for (int64_t i = 0; i < count; i++) {
      addr[static_cast<uint64_t>(i)] = static_cast<uint8_t>(val_vector.Get(0));
    }
  }
  (void)weight->SetData(addr.data(), static_cast<size_t>(count) * sizeof(uint8_t));
  return SUCCESS;
}

Status TensorAssign::GetStringVal(const int64_t val_size,
                                  const google::protobuf::RepeatedPtrField<std::string> &val_vector,
                                  const int64_t count, GeTensorPtr &weight) {
  GE_CHECK_NOTNULL(weight);
  const bool flag = ((count != val_size) && (val_size == 1));
  size_t total_size = 0U;
  if (!flag) {
    const int64_t min_count = (count > val_size) ? val_size : count;
    for (int64_t i = 0; i < min_count; i++) {
      // extra 16 bytes store head of string
      // extra 1 byte store '\0'
      GE_ASSERT_EQ(ge::IntegerChecker<int32_t>::Compat(i), true);
      total_size += (val_vector[static_cast<int32_t>(i)].size() + sizeof(ge::StringHead) + 1U);
    }
    total_size += (static_cast<size_t>(count) - static_cast<size_t>(min_count)) * (sizeof(ge::StringHead) + 1U);
    std::vector<uint8_t> addr(total_size);
    ge::StringHead *const string_head = ge::PtrToPtr<uint8_t, ge::StringHead>(addr.data());
    // front 16 bytes store head of each string
    auto raw_data = ge::PtrAdd<uint8_t>(addr.data(), total_size + 1U,
                                        static_cast<size_t>(count) * sizeof(ge::StringHead));
    GE_ASSERT_TRUE(count > 0);
    GE_ASSERT_EQ(ge::TypeUtilsInner::CheckUint64MulOverflow(static_cast<uint64_t>(count),
                                                            static_cast<uint32_t>(sizeof(ge::StringHead))),
                 false);
    uint64_t ptr_size = static_cast<uint64_t>(count) * sizeof(ge::StringHead);
    for (int64_t i = 0; i < count; ++i) {
      ge::PtrAdd<ge::StringHead>(string_head, static_cast<size_t>(count) + 1U,
                                 static_cast<size_t>(i))->addr = static_cast<int64_t>(ptr_size);
      if (i < val_size) {
        GE_ASSERT_EQ(ge::IntegerChecker<int32_t>::Compat(i), true);
        const string &str = val_vector.Get(static_cast<int32_t>(i));
        ge::PtrAdd<ge::StringHead>(string_head, static_cast<size_t>(count) + 1U,
                                   static_cast<size_t>(i))->len = static_cast<int64_t>(str.size());
        CHECK_FALSE_EXEC(memcpy_s(raw_data, str.size() + 1U, str.c_str(), str.size() + 1U) == EOK,
                         GELOGW("[GetStringVal][Copy] memcpy failed"));
        raw_data = ge::PtrAdd<uint8_t>(raw_data, total_size + 1U, str.size() + 1U);
        ptr_size += (str.size() + 1U);
      } else {
        ge::PtrAdd<ge::StringHead>(string_head, static_cast<size_t>(count) + 1U, static_cast<size_t>(i))->len = 0;
        raw_data = ge::PtrAdd<uint8_t>(raw_data, total_size + 1U, 1U);
        ptr_size += 1U;
      }
    }
    (void)weight->SetData(ge::PtrToPtr<uint8_t, const uint8_t>(addr.data()), total_size);
  } else {
    const string &str = val_vector.Get(0);
    // extra 16 bytes store head of string
    // extra 1 byte store '\0'
    total_size = (str.size() + sizeof(ge::StringHead) + 1U) * static_cast<size_t>(count);
    std::vector<uint8_t> addr(total_size);
    // front 16 bytes store head of each string
    ge::StringHead *const string_head = ge::PtrToPtr<uint8_t, ge::StringHead>(addr.data());
    auto raw_data = ge::PtrAdd<uint8_t>(addr.data(), total_size + 1U,
                                        static_cast<size_t>(count) * sizeof(ge::StringHead));
    GE_ASSERT_TRUE(count > 0);
    GE_ASSERT_EQ(ge::TypeUtilsInner::CheckUint64MulOverflow(static_cast<uint64_t>(count),
                                                            static_cast<uint32_t>(sizeof(ge::StringHead))),
                 false);
    uint64_t ptr_size = static_cast<uint64_t>(count) * sizeof(ge::StringHead);
    for (int64_t i = 0; i < count; ++i) {
      ge::PtrAdd<ge::StringHead>(string_head, static_cast<size_t>(count) + 1U,
                                 static_cast<size_t>(i))->addr = static_cast<int64_t>(ptr_size);
      ge::PtrAdd<ge::StringHead>(string_head, static_cast<size_t>(count) + 1U,
                                 static_cast<size_t>(i))->len = static_cast<int64_t>(str.size());
      const bool b = memcpy_s(raw_data, str.size() + 1U, str.c_str(), str.size() + 1U) == EOK;
      if (!b) {
        GELOGW("[GetStringVal][Copy] memcpy failed");
      }
      raw_data = ge::PtrAdd<uint8_t>(raw_data, total_size + 1U, str.size() + 1U);
      ptr_size += (str.size() + 1U);
    }
    (void)weight->SetData(ge::PtrToPtr<uint8_t, const uint8_t>(addr.data()), total_size);
  }
  return SUCCESS;
}

static Status GetComplex32Val(const int64_t val_size, const google::protobuf::RepeatedField<int32> &val_vector,
                              const int64_t count, GeTensorPtr &weight) {
  // val_size must be even, and complex value should be an integer multiple of 2
  GE_ASSERT_TRUE((val_size % kComplexWidth) == 0, "complex value should be an integer multiple of 2.");
  const std::unique_ptr<uint16_t[]> addr = ge::ComGraphMakeUnique<uint16_t[]>(static_cast<size_t>(count));
  GE_CHECK_NOTNULL(addr);
  // Complex numbers are made up of real and imaginary numbers
  const bool zerosLike = ((count != val_size) && (val_size == 2));
  if (!zerosLike) {
    GE_ASSERT_TRUE(val_size <= count);
    for (size_t i = 0UL; i < static_cast<size_t>(val_size); i++) {
      addr[i] = static_cast<uint16_t>(val_vector.Get(static_cast<int32_t>(i)));
    }
    const int64_t value_r = val_size - 1;
    GE_ASSERT_EQ(ge::IntegerChecker<int32_t>::Compat(value_r), true);
    // val_vector format is real value, complex value..., here is getting the corresponding value.
    // real value and complex value are stored spaced apart, so use 2 and 1 to store in the correct addr.
    const int64_t value_l = val_size - kComplexWidth;
    GE_ASSERT_EQ(ge::IntegerChecker<int32_t>::Compat(value_l), true);
    for (int64_t i = val_size; i < count; i += kComplexWidth) {
      addr[static_cast<size_t>(i)] = static_cast<uint16_t>(val_vector.Get(static_cast<int32_t>(value_l)));
      addr[static_cast<size_t>(i) + 1UL] = static_cast<uint16_t>(val_vector.Get(static_cast<int32_t>(value_r)));
    }
  } else {
    for (int64_t i = 0; i < count; i += kComplexWidth) {
      addr[static_cast<size_t>(i)] = static_cast<uint16_t>(val_vector.Get(0));
      addr[static_cast<size_t>(i) + 1UL] = static_cast<uint16_t>(val_vector.Get(1));
    }
  }
  (void)weight->SetData(ge::PtrToPtr<uint16_t, uint8_t>(addr.get()), static_cast<size_t>(count) * sizeof(uint16_t));
  return SUCCESS;
}

void TensorAssign::SetGeTensorWeightData(const TensorProto &tensor, const int64_t val_size,
                                         const int64_t count, GeTensorPtr &weight) {
  const tensorflow::DataType data_type = tensor.dtype();
  constexpr int64_t kNumElementOfComplex = 2;
  if (CheckFloatVal(data_type)) {
    (void)GetVal(val_size, tensor.float_val(), count, weight);
  } else if (CheckComplex32Val(data_type)) {
    (void)GetComplex32Val(val_size, tensor.icomplex_val(), count * kNumElementOfComplex, weight);
  } else if (CheckComplex64Val(data_type)) {
    (void)GetVal(val_size, tensor.scomplex_val(), count * kNumElementOfComplex, weight, true);
  } else if (CheckSignedFourByte(data_type)) {
    (void)GetVal(val_size, tensor.int_val(), count, weight);
  } else if (CheckUnsignedFourByte(data_type)) {
    (void)GetVal(val_size, tensor.uint32_val(), count, weight);
  } else if (CheckSignedEightByte(data_type)) {
    (void)GetVal(val_size, tensor.int64_val(), count, weight);
  } else if (CheckUnsignedEightByte(data_type)) {
    (void)GetVal(val_size, tensor.uint64_val(), count, weight);
  } else if (CheckBoolVal(data_type)) {
    (void)GetVal(val_size, tensor.bool_val(), count, weight);
  } else if (CheckStringVal(data_type)) {
    (void)GetStringVal(val_size, tensor.string_val(), count, weight);
  } else if (CheckHalfVal(data_type)) {
    (void)GetDoubleByteVal(val_size, tensor.half_val(), count, weight);
  } else if (CheckDoubleByte(data_type)) {
    (void)GetDoubleByteVal(val_size, tensor.int_val(), count, weight);
  } else if (CheckByte(data_type)) {
    (void)GetByteVal(val_size, tensor.int_val(), count, weight);
  } else if (CheckDoubleVal(data_type)) {
    (void)GetVal(val_size, tensor.double_val(), count, weight);
  } else if (CheckComplex128Val(data_type)) {
    (void)GetVal(val_size, tensor.dcomplex_val(), count * kNumElementOfComplex, weight, true);
  } else {
    GELOGI("data_type:%s.", DataType_Name(data_type).c_str());
  }
}

void TensorAssign::SetWeightData(const tensorflow::DataType data_type, const int64_t count,
                                 const std::string &tensor_content, GeTensorPtr &weight) {
  if (weight == nullptr) {
    GE_LOGE("weight is nullptr.");
    return;
  }
  GELOGD("Set data from tensor_content, count = %ld, data_type = %s.",
         count, DataType_Name(data_type).c_str());
  const auto tensor_content_data = tensor_content.data();
  const bool is_four_byte =
      CheckSignedFourByte(data_type) || CheckUnsignedFourByte(data_type) || CheckComplex32Val(data_type);
  const bool is_double_byte = CheckHalfVal(data_type) || CheckDoubleByte(data_type);
  const bool is_eight_byte = CheckSignedEightByte(data_type) || CheckUnsignedEightByte(data_type);
  if (CheckByte(data_type)) {
    (void)weight->SetData(ge::PtrToPtr<char, uint8_t>(tensor_content_data),
                          static_cast<size_t>(count) * sizeof(uint8_t));
  } else if (CheckBoolVal(data_type)) {
    (void)weight->SetData(ge::PtrToPtr<char, uint8_t>(tensor_content_data),
                          static_cast<size_t>(count) * sizeof(bool));
  } else if (is_double_byte) {
    (void)weight->SetData(ge::PtrToPtr<char, uint8_t>(tensor_content_data),
                          static_cast<size_t>(count) * sizeof(uint16_t));
  } else if (is_four_byte) {
    (void)weight->SetData(ge::PtrToPtr<char, uint8_t>(tensor_content_data),
                          static_cast<size_t>(count) * sizeof(uint32_t));
  } else if (is_eight_byte) {
    (void)weight->SetData(ge::PtrToPtr<char, uint8_t>(tensor_content_data),
                          static_cast<size_t>(count) * sizeof(uint64_t));
  } else if (CheckDoubleVal(data_type)) {
    (void)weight->SetData(ge::PtrToPtr<char, uint8_t>(tensor_content_data),
                          static_cast<size_t>(count) * sizeof(double));
  } else if (CheckComplex128Val(data_type)) {
    (void)weight->SetData(ge::PtrToPtr<char, uint8_t>(tensor_content_data),
                          static_cast<size_t>(count) * sizeof(std::complex<double>));
  } else if (CheckComplex64Val(data_type)) {
    (void)weight->SetData(ge::PtrToPtr<char, uint8_t>(tensor_content_data),
                          static_cast<size_t>(count) * sizeof(std::complex<float>));
  } else if (CheckStringVal(data_type)) {
    if (ge::TypeUtilsInner::CheckUint64MulOverflow(static_cast<uint64_t>(count),
                                                   static_cast<uint32_t>(sizeof(ge::StringHead)))) {
      GELOGE(ge::FAILED, "count multiply StringHead is overflow uint64, count: %u", static_cast<uint64_t>(count));
      return;
    }
    std::string weight_content;
    if (count > 0) {
      // each byte of top count bytes is each string length
      weight_content = tensor_content.substr(static_cast<uint64_t>(count));
    }
    const size_t total_size = weight_content.size() + static_cast<size_t>(count) * (sizeof(ge::StringHead) + 1U);
    std::vector<uint8_t> addr(total_size);
    ge::StringHead *const string_head = ge::PtrToPtr<uint8_t, ge::StringHead>(addr.data());
    auto raw_data =
        ge::PtrAdd<uint8_t>(addr.data(), total_size + 1U, static_cast<size_t>(count) * sizeof(ge::StringHead));
    uint64_t ptr_size = static_cast<uint64_t>(count) * sizeof(ge::StringHead);
    size_t str_start_index = 0U;
    for (int64_t i = 0; i < count; ++i) {
      ge::PtrAdd<ge::StringHead>(string_head, static_cast<size_t>(count) + 1U, static_cast<size_t>(i))->addr =
          static_cast<int64_t>(ptr_size);
      const size_t str_len = static_cast<size_t>(tensor_content.at(static_cast<size_t>(i)));
      const string &str = weight_content.substr(str_start_index, str_len);
      str_start_index += str_len;
      ge::PtrAdd<ge::StringHead>(string_head, static_cast<size_t>(count) + 1U, static_cast<size_t>(i))->len =
          static_cast<int64_t>(str.size());
      CHECK_FALSE_EXEC(memcpy_s(raw_data, str.size() + 1U, str.c_str(), str.size() + 1U) == EOK,
                       GELOGW("[SetWeight][Copy] memcpy failed"));
      raw_data = ge::PtrAdd<uint8_t>(raw_data, total_size + 1U, str.size() + 1U);
      ptr_size += static_cast<uint64_t>(str.size()) + 1U;
    }
    (void)weight->SetData(ge::PtrToPtr<uint8_t, const uint8_t>(addr.data()), total_size);
  } else {
    (void)weight->SetData(ge::PtrToPtr<char, uint8_t>(tensor_content_data),
                          static_cast<size_t>(count) * sizeof(float));
  }
}

Status TensorAssign::SetGeTensor(const TensorProto &tensor, GeTensorPtr &weight) {
  GE_CHECK_NOTNULL(weight);
  std::map<tensorflow::DataType, int64_t> datatype_val_size_map = {
      {tensorflow::DT_FLOAT, tensor.float_val().size()},
      {tensorflow::DT_INT32, tensor.int_val().size()},
      {tensorflow::DT_INT64, tensor.int64_val().size()},
      {tensorflow::DT_BOOL, tensor.bool_val().size()},
      {tensorflow::DT_HALF, tensor.half_val().size()},
      {tensorflow::DT_INT8, tensor.int_val().size()},
      {tensorflow::DT_UINT8, tensor.int_val().size()},
      {tensorflow::DT_INT16, tensor.int_val().size()},
      {tensorflow::DT_UINT16, tensor.int_val().size()},
      {tensorflow::DT_DOUBLE, tensor.double_val().size()},
      {tensorflow::DT_STRING, tensor.string_val().size()},
      {tensorflow::DT_QINT8, tensor.int_val().size()},
      {tensorflow::DT_QINT16, tensor.int_val().size()},
      {tensorflow::DT_QINT32, tensor.int_val().size()},
      {tensorflow::DT_QUINT8, tensor.int_val().size()},
      {tensorflow::DT_QUINT16, tensor.int_val().size()},
      {tensorflow::DT_COMPLEX32, tensor.icomplex_val().size()},
      {tensorflow::DT_COMPLEX64, tensor.scomplex_val().size()},
      {tensorflow::DT_COMPLEX128, tensor.dcomplex_val().size()},
      {tensorflow::DT_BFLOAT16, tensor.half_val().size()},
      {tensorflow::DT_UINT32, tensor.uint32_val().size()},
      {tensorflow::DT_UINT64, tensor.uint64_val().size()},
      {tensorflow::DT_RESOURCE, tensor.resource_handle_val().size()},
      {tensorflow::DT_VARIANT, tensor.variant_val().size()},
      {tensorflow::DT_FLOAT_REF, tensor.float_val().size()},
      {tensorflow::DT_INT32_REF, tensor.int_val().size()},
      {tensorflow::DT_INT64_REF, tensor.int64_val().size()},
      {tensorflow::DT_BOOL_REF, tensor.bool_val().size()},
      {tensorflow::DT_HALF_REF, tensor.half_val().size()},
      {tensorflow::DT_INT8_REF, tensor.int_val().size()},
      {tensorflow::DT_UINT8_REF, tensor.int_val().size()},
      {tensorflow::DT_INT16_REF, tensor.int_val().size()},
      {tensorflow::DT_UINT16_REF, tensor.int_val().size()},
      {tensorflow::DT_DOUBLE_REF, tensor.double_val().size()},
      {tensorflow::DT_STRING_REF, tensor.string_val().size()},
      {tensorflow::DT_QINT8_REF, tensor.int_val().size()},
      {tensorflow::DT_QINT16_REF, tensor.int_val().size()},
      {tensorflow::DT_QINT32_REF, tensor.int_val().size()},
      {tensorflow::DT_QUINT8_REF, tensor.int_val().size()},
      {tensorflow::DT_QUINT16_REF, tensor.int_val().size()},
      {tensorflow::DT_COMPLEX32_REF, tensor.icomplex_val().size()},
      {tensorflow::DT_COMPLEX64_REF, tensor.scomplex_val().size()},
      {tensorflow::DT_COMPLEX128_REF, tensor.dcomplex_val().size()},
      {tensorflow::DT_BFLOAT16_REF, tensor.half_val().size()},
      {tensorflow::DT_UINT32_REF, tensor.uint32_val().size()},
      {tensorflow::DT_UINT64_REF, tensor.uint64_val().size()},
      {tensorflow::DT_RESOURCE_REF, tensor.resource_handle_val().size()},
      {tensorflow::DT_VARIANT_REF, tensor.variant_val().size()},
  };
  const tensorflow::DataType data_type = tensor.dtype();
  int64_t datatype_val_size = 0;

  const auto iter = datatype_val_size_map.find(data_type);
  if (iter != datatype_val_size_map.cend()) {
    datatype_val_size = iter->second;
  } else {
    GE_CHECK_GE(data_type, 0);
    GE_LOGE("datatype:%s not support.", DataType_Name(data_type).c_str());
    return FAILED;
  }

  std::vector<int64_t> shape_vec;
  // There is tensor shape, get the dimension
  int64_t count = 1;
  GE_IF_BOOL_EXEC(
      tensor.has_tensor_shape(), const tensorflow::TensorShapeProto &tensor_shape = tensor.tensor_shape();
      for (int32_t i = 0; i < tensor_shape.dim_size(); i++) {
        const tensorflow::TensorShapeProto_Dim &shape_dim = tensor_shape.dim(i);
        shape_vec.push_back(shape_dim.size());
        const int64_t dim = shape_vec[static_cast<size_t>(i)];
        // tensorflow support weights shape [0],have no weights
        if (dim < 0) {
          GELOGE(FAILED, "Dim size invalid");
          return FAILED;
        }
        if ((count != 0) && (dim >= (std::numeric_limits<int64_t>::max() / count))) {
          GELOGE(FAILED, "Dim size exceeds INT64_MAX");
          return FAILED;
        }
        count *= dim;
      });
  const GeShape shape(shape_vec);
  GeTensorDesc tmp_desc = weight->GetTensorDesc();
  tmp_desc.SetShape(shape);

  // Fixed input ND
  tmp_desc.SetFormat(ge::Format::FORMAT_ND);
  tmp_desc.SetOriginFormat(ge::Format::FORMAT_ND);

  weight->SetTensorDesc(tmp_desc);

  if (datatype_val_size > 0 || ((datatype_val_size == 0) && (count > 0) && tensor.tensor_content().empty())) {
    SetGeTensorWeightData(tensor, datatype_val_size, count, weight);
    const int64_t origin_element_num = static_cast<int64_t>(datatype_val_size);
    GE_CHK_BOOL_EXEC(ge::AttrUtils::SetInt(weight->MutableTensorDesc(), kOriginElementNumAttrName, origin_element_num),
                     return FAILED, "Set origin element num failed.");
  } else if (!tensor.tensor_content().empty()) {
    const auto &tensor_content = tensor.tensor_content();
    SetWeightData(data_type, count, tensor_content, weight);
  } else {
    if (count == 0) {
      GELOGI("Empty tensor, has no data.");
      return SUCCESS;
    }
    GE_LOGE("value Attr tensor should have val() or tensor_content");
    return FAILED;
  }

  return SUCCESS;
}

Status TensorAssign::SetGeTensorDataType(const int64_t data_type, GeTensorPtr &weight) {
  GE_CHECK_NOTNULL(weight);
  GeTensorDesc tmp_desc = weight->GetTensorDesc();
  tmp_desc.SetDataType(static_cast<ge::DataType>(data_type));
  weight->SetTensorDesc(tmp_desc);
  return SUCCESS;
}
}  // namespace domi
