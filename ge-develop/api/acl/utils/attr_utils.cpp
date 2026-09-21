/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "attr_utils.h"
#include <cmath>
#include "securec.h"

#include "mmpa/mmpa_api.h"
#include "common/log_inner.h"
#include "hash_utils.h"

namespace {
constexpr float32_t FLOAT_DELTA = 1e-6F;
constexpr uint32_t FP16_MAX_EXP = 0x001FU;
constexpr uint32_t FP32_SIGN_INDEX = 31U;
constexpr uint32_t FP32_MAN_LEN = 23U;
constexpr uint32_t FP32_MAX_MAN = 0x7FFFFFU;
constexpr uint32_t FP16_MAN_HIDE_BIT = 0x0400U;
constexpr uint32_t FP32_EXP_BIAS = 127U;
constexpr uint32_t FP16_MAN_MASK = 0x03FFU;
constexpr uint32_t FP16_MAN_LEN = 10U;
constexpr uint32_t FP16_EXP_BIAS = 15U;

#define FP16_EXTRAC_SIGN(x)            (((x) >> 15U) & 1U)
#define FP16_EXTRAC_EXP(x)             (((x) >> 10U) & FP16_MAX_EXP)
#define FP16_EXTRAC_MAN(x)             ((((x) >> 0U) & 0x3FFU) |          \
                                       ((((((x) >> 10U) & 0x1FU) > 0U) ? 1U : 0U) * 0x400U))
#define FP32_CONSTRUCTOR(s, e, m)        (((s) << FP32_SIGN_INDEX) |      \
                                          ((e) << FP32_MAN_LEN) |         \
                                          ((m) & FP32_MAX_MAN))

union TypeUnion {
  float32_t fVal;
  uint32_t uVal;
};

static void ExtractFP16(const uint16_t val, uint16_t *const s, int16_t *const e, uint16_t *const m)
{
  // 1.Extract
  *s = FP16_EXTRAC_SIGN(val);
  *e = static_cast<int16_t>(FP16_EXTRAC_EXP(val));
  *m = FP16_EXTRAC_MAN(val);

  // Denormal
  if ((*e) == 0) {
    *e = 1;
  }
}

float32_t Fp16ToFloat(const uint16_t val)
{
  uint16_t hfSign;
  uint16_t hfMan;
  int16_t hfExp;
  ExtractFP16(val, &hfSign, &hfExp, &hfMan);

  while ((hfMan != 0U) && ((hfMan & FP16_MAN_HIDE_BIT) == 0U)) {
    hfMan <<= 1U;
    hfExp--;
  }

  uint32_t eRet;
  uint32_t mRet;
  if (hfMan == 0U) {
    eRet = 0U;
    mRet = 0U;
  } else {
    eRet = static_cast<uint32_t>(hfExp + static_cast<int16_t>(FP32_EXP_BIAS - FP16_EXP_BIAS));
    mRet = static_cast<uint32_t>(hfMan & FP16_MAN_MASK);
    mRet = mRet << (FP32_MAN_LEN - FP16_MAN_LEN);
  }

  const uint32_t sRet = hfSign;
  TypeUnion u;
  u.uVal = FP32_CONSTRUCTOR(sRet, eRet, mRet);
  const auto ret = u.fVal;
  return ret;
}

template <typename T>
std::string ScalarAttrToString(const ge::GeAttrValue &attrVal)
{
    std::stringstream ss;
    T val{};
    (void)attrVal.GetValue<T>(val);
    ss << val;
    return ss.str();
}

template <>
std::string ScalarAttrToString<std::string>(const ge::GeAttrValue &attrVal)
{
    std::string val;
    (void)attrVal.GetValue<std::string>(val);
    return val;
}

template <>
std::string ScalarAttrToString<bool>(const ge::GeAttrValue &attrVal)
{
    bool val = false;
    (void)attrVal.GetValue<bool>(val);
    return val ? "True" : "False";
}

template <typename T>
std::string ListAttrToString(const ge::GeAttrValue &attrVal)
{
    std::vector<T> values;
    (void)attrVal.GetValue<std::vector<T>>(values);
    return acl::StringUtils::VectorToString(values);
}

template <>
std::string ListAttrToString<bool>(const ge::GeAttrValue &attrVal)
{
    std::stringstream ss;
    std::vector<bool> values;
    (void)attrVal.GetValue<std::vector<bool>>(values);
    ss << '[';
    const auto size = values.size();
    for (size_t i = 0U; i < size; ++i) {
        if (values[i]) {
            ss << "True";
        } else {
            ss << "False";
        }
        if (i != (size - 1U)) {
            ss << ", ";
        }
    }
    ss << ']';
    return ss.str();
}

template <typename T>
std::string ListListAttrToString(const ge::GeAttrValue &attrVal)
{
    std::stringstream ss;
    ss << '[';
    std::vector<std::vector<T>> values;
    (void)attrVal.GetValue<std::vector<std::vector<T>>>(values);
    return acl::StringUtils::VectorToString(values);
}

template <typename T>
void ListAttrToStringForDigest(std::string &buffer, const ge::GeAttrValue &attrVal)
{
    std::vector<T> values;
    (void)attrVal.GetValue<std::vector<T>>(values);
    for (const T &str : values) {
        buffer += std::to_string(str);
        buffer.push_back(',');
    }
}

template <typename T>
void ListAttrToDigest(size_t &seed, const ge::GeAttrValue &attrVal)
{
    std::vector<T> values;
    (void)attrVal.GetValue<std::vector<T>>(values);
    for (const T &str : values) {
        acl::hash_utils::HashCombine(seed, str);
    }
}

template <typename T>
void GetAttrValueAndGenHash(size_t &seed, const ge::GeAttrValue &val, const T &init_value)
{
    T value = init_value;
    (void)val.GetValue<T>(value);
    acl::hash_utils::HashCombine(seed, value);
}

template<typename T>
void ListListAttrToStringForDigest(std::string &buffer, const ge::GeAttrValue &attrVal)
{
    std::vector<std::vector<T>> values;
    (void)attrVal.GetValue<std::vector<std::vector<T>>>(values);
    for (auto &subVec : values) {
        for (auto &val : subVec) {
            buffer += std::to_string(val);
            buffer.push_back(',');
        }
        buffer.push_back('|');
    }
}

template<typename T>
void ListListAttrToDigest(size_t &seed, const ge::GeAttrValue &attrVal)
{
    std::vector<std::vector<T>> values;
    (void)attrVal.GetValue<std::vector<std::vector<T>>>(values);
    for (auto &subVec : values) {
        for (auto &val : subVec) {
            acl::hash_utils::HashCombine(seed, val);
        }
    }
}

template<typename T>
bool AttrScalarValueEquals(const ge::GeAttrValue &lhs, const ge::GeAttrValue &rhs)
{
    T lhsValue {};
    T rhsValue {};
    (void)lhs.GetValue<T>(lhsValue);
    (void)rhs.GetValue<T>(rhsValue);
    return lhsValue == rhsValue;
}

template<>
bool AttrScalarValueEquals<float32_t>(const ge::GeAttrValue &lhs, const ge::GeAttrValue &rhs)
{
    float32_t lhsValue = 0.0F;
    float32_t rhsValue = 0.0F;
    (void)lhs.GetValue<float32_t>(lhsValue);
    (void)rhs.GetValue<float32_t>(rhsValue);
    if (std::isnan(lhsValue) && std::isnan(rhsValue)) {
        return true;
    }
    if (std::isinf(lhsValue) && std::isinf(rhsValue)) {
        return true;
    }
    return fabsf(lhsValue - rhsValue) <= FLOAT_DELTA;
}

template<typename T>
bool AttrListValueEquals(const ge::GeAttrValue &lhs, const ge::GeAttrValue &rhs)
{
    std::vector<T> lhsValue;
    std::vector<T> rhsValue;
    (void)(lhs.GetValue<std::vector<T>>(lhsValue));
    (void)(rhs.GetValue<std::vector<T>>(rhsValue));
    return lhsValue == rhsValue;
}

template<>
bool AttrListValueEquals<float32_t>(const ge::GeAttrValue &lhs, const ge::GeAttrValue &rhs)
{
    std::vector<float32_t> lhsValue;
    std::vector<float32_t> rhsValue;
    (void)lhs.GetValue<std::vector<float32_t>>(lhsValue);
    (void)rhs.GetValue<std::vector<float32_t>>(rhsValue);
    return acl::attr_utils::IsListFloatEquals(lhsValue, rhsValue);
}

template<typename T>
bool AttrListListValueEquals(const ge::GeAttrValue &lhs, const ge::GeAttrValue &rhs)
{
    std::vector<std::vector<T>> lhsValue;
    std::vector<std::vector<T>> rhsValue;
    (void)lhs.GetValue<std::vector<std::vector<T>>>(lhsValue);
    (void)rhs.GetValue<std::vector<std::vector<T>>>(rhsValue);
    return lhsValue == rhsValue;
}

template<typename T>
bool CheckValExact(const std::vector<T> &valueRange, const std::vector<T> &data)
{
    if (valueRange.size() != data.size()) {
        ACL_LOG_DEBUG("input data size [%zu] must be equal to value range size [%zu]", data.size(), valueRange.size());
        return false;
    }
    for (size_t i = 0U; i < valueRange.size(); ++i) {
        if (valueRange[i] != data[i]) {
            return false;
        }
    }
    return true;
}

template<typename T1, typename T2>
void SetInputValue(const aclDataBuffer *const dataBuffer, std::vector<T1> &inputData)
{
    for (size_t i = 0U; i < (dataBuffer->length / sizeof(T2)); ++i) {
        inputData.push_back(static_cast<T1>(*(reinterpret_cast<const T2 *>(dataBuffer->data) + i)));
    }
    return;
}

template<typename T>
bool CheckValRange(const std::vector<std::vector<T>> &valueRange, const std::vector<T> &data)
{
    if (valueRange.size() != data.size()) {
        ACL_LOG_DEBUG("input data size [%zu] must be equal to value range size [%zu]", data.size(), valueRange.size());
        return false;
    }
    for (size_t i = 0U; i < valueRange.size(); ++i) {
        // 2 is range size
        if (valueRange[i].size() != 2U) {
            ACL_LOG_WARN("range size must be 2");
            return false;
        }
        if ((valueRange[i][0U] > data[i]) || (valueRange[i][1U] < data[i])) {
            return false;
        }
    }
    return true;
}
} // namespace

namespace acl {
namespace attr_utils {
std::string GeAttrValueToString(const ge::GeAttrValue &val)
{
    switch (val.GetValueType()) {
        case ge::GeAttrValue::VT_STRING:
            return ScalarAttrToString<std::string>(val);
        case ge::GeAttrValue::VT_BOOL:
            return ScalarAttrToString<bool>(val);
        case ge::GeAttrValue::VT_INT:
            return ScalarAttrToString<int64_t>(val);
        case ge::GeAttrValue::VT_FLOAT:
            return ScalarAttrToString<float32_t>(val);
        case ge::GeAttrValue::VT_DATA_TYPE:
            return ScalarAttrToString<ge::GeAttrValue::DATA_TYPE>(val);
        case ge::GeAttrValue::VT_LIST_STRING:
            return ListAttrToString<std::string>(val);
        case ge::GeAttrValue::VT_LIST_BOOL:
            return ListAttrToString<bool>(val);
        case ge::GeAttrValue::VT_LIST_INT:
            return ListAttrToString<int64_t>(val);
        case ge::GeAttrValue::VT_LIST_FLOAT:
            return ListAttrToString<float32_t>(val);
        case ge::GeAttrValue::VT_LIST_DATA_TYPE:
            return ListAttrToString<ge::GeAttrValue::DATA_TYPE>(val);
        case ge::GeAttrValue::VT_LIST_LIST_INT:
            return ListListAttrToString<int64_t>(val);
        default:
            std::stringstream ss;
            ss << "Unknown type ";
            ss << val.GetValueType();
            return ss.str();
    }
}

void GeAttrValueToDigest(size_t &seed, const ge::GeAttrValue &val)
{
    switch (val.GetValueType()) {
        case ge::GeAttrValue::VT_BOOL:
            GetAttrValueAndGenHash<bool>(seed, val, false);
            break;
        case ge::GeAttrValue::VT_STRING:
            GetAttrValueAndGenHash<std::string>(seed, val, "");
            break;
        case ge::GeAttrValue::VT_INT:
            GetAttrValueAndGenHash<int64_t>(seed, val, 0L);
            break;
        case ge::GeAttrValue::VT_DATA_TYPE:
            GetAttrValueAndGenHash<ge::GeAttrValue::DATA_TYPE>(seed, val, {});
            break;
        case ge::GeAttrValue::VT_FLOAT:
            GetAttrValueAndGenHash<float32_t>(seed, val, 0.0f);
            break;
        case ge::GeAttrValue::VT_LIST_STRING:
            ListAttrToDigest<std::string>(seed, val);
            break;
        case ge::GeAttrValue::VT_LIST_BOOL:
            ListAttrToDigest<bool>(seed, val);
            break;
        case ge::GeAttrValue::VT_LIST_INT:
            ListAttrToDigest<int64_t>(seed, val);
            break;
        case ge::GeAttrValue::VT_LIST_DATA_TYPE:
            ListAttrToDigest<ge::GeAttrValue::DATA_TYPE>(seed, val);
            break;
        case ge::GeAttrValue::VT_LIST_FLOAT:
            ListAttrToDigest<float32_t>(seed, val);
            break;
        case ge::GeAttrValue::VT_LIST_LIST_INT:
            ListListAttrToDigest<int64_t>(seed, val);
            break;
        default:
            break;
    }
}

std::string AttrMapToString(const std::map<std::string, ge::GeAttrValue> &attrMap)
{
    std::stringstream ss;
    ss << "{";
    const size_t size = attrMap.size();
    size_t cnt = 0U;
    for (auto &attr : attrMap) {
        ss << attr.first << " = " << GeAttrValueToString(attr.second);
        cnt++;
        if (cnt != size) {
            ss << ", ";
        }
    }
    ss << "}";
    return ss.str();
}

size_t AttrMapToDigest(const std::map<std::string, ge::GeAttrValue> &attrMap)
{
    if (attrMap.empty()) {
        return 0U;
    }

    size_t digest = 0U;
    for (auto &attr : attrMap) {
        GeAttrValueToDigest(digest, attr.second);
    }

    return digest;
}

bool IsListFloatEquals(const std::vector<float32_t> &lhsValue, const std::vector<float32_t> &rhsValue)
{
    if (lhsValue.size() != rhsValue.size()) {
        return false;
    }

    for (size_t i = 0U; i < lhsValue.size(); ++i) {
        const float32_t val1 = lhsValue[i];
        const float32_t val2 = rhsValue[i];
        if (std::isnan(val1) && std::isnan(val2)) {
            continue;
        }
        if (std::isinf(val1) && std::isinf(val2)) {
            continue;
        }
        if (fabsf(val1 - val2) > FLOAT_DELTA) {
            return false;
        }
    }

    return true;
}

bool AttrValueEquals(const ge::GeAttrValue &lhs, const ge::GeAttrValue &rhs)
{
    const auto type = lhs.GetValueType();
    if (rhs.GetValueType() != type) {
        ACL_LOG_INFO("type mismatch, lhs type = %d, rhs type = %d",
            static_cast<int32_t>(type), static_cast<int32_t>(rhs.GetValueType()));
        return false;
    }

    switch (type) {
        case ge::GeAttrValue::VT_STRING:
            return AttrScalarValueEquals<std::string>(lhs, rhs);
        case ge::GeAttrValue::VT_BOOL:
            return AttrScalarValueEquals<bool>(lhs, rhs);
        case ge::GeAttrValue::VT_INT:
            return AttrScalarValueEquals<int64_t>(lhs, rhs);
        case ge::GeAttrValue::VT_FLOAT:
            return AttrScalarValueEquals<float32_t>(lhs, rhs);
        case ge::GeAttrValue::VT_DATA_TYPE:
            return AttrScalarValueEquals<ge::GeAttrValue::DATA_TYPE>(lhs, rhs);
        case ge::GeAttrValue::VT_LIST_STRING:
            return AttrListValueEquals<std::string>(lhs, rhs);
        case ge::GeAttrValue::VT_LIST_BOOL:
            return AttrListValueEquals<bool>(lhs, rhs);
        case ge::GeAttrValue::VT_LIST_INT:
            return AttrListValueEquals<int64_t>(lhs, rhs);
        case ge::GeAttrValue::VT_LIST_FLOAT:
            return AttrListValueEquals<float32_t>(lhs, rhs);
        case ge::GeAttrValue::VT_LIST_DATA_TYPE:
            return AttrListValueEquals<ge::GeAttrValue::DATA_TYPE>(lhs, rhs);
        case ge::GeAttrValue::VT_LIST_LIST_INT:
            return AttrListListValueEquals<int64_t>(lhs, rhs);
        default:
            ACL_LOG_INNER_ERROR("[Check][Type]Unknown type %d", static_cast<int32_t>(type));
            return false;
    }
}

static bool GetInputData(const aclDataBuffer *const dataBuffer, const aclDataType dataType,
    std::vector<int64_t> &inputIntData, std::vector<float32_t> &inputFloatData)
{
    switch (dataType) {
        case ACL_FLOAT:
            SetInputValue<float32_t, float32_t>(dataBuffer, inputFloatData);
            break;
        case ACL_FLOAT16:
            for (size_t i = 0U; i < (dataBuffer->length / sizeof(aclFloat16)); ++i) {
                inputFloatData.push_back(
                    Fp16ToFloat(*(static_cast<const aclFloat16 *>(dataBuffer->data) + i)));
            }
            break;
        case ACL_INT8:
            SetInputValue<int64_t, int8_t>(dataBuffer, inputIntData);
            break;
        case ACL_UINT8:
            SetInputValue<int64_t, uint8_t>(dataBuffer, inputIntData);
            break;
        case ACL_INT16:
            SetInputValue<int64_t, int16_t>(dataBuffer, inputIntData);
            break;
        case ACL_UINT16:
            SetInputValue<int64_t, uint16_t>(dataBuffer, inputIntData);
            break;
        case ACL_INT32:
            SetInputValue<int64_t, int32_t>(dataBuffer, inputIntData);
            break;
        case ACL_UINT32:
            SetInputValue<int64_t, uint32_t>(dataBuffer, inputIntData);
            break;
        case ACL_INT64:
            SetInputValue<int64_t, int64_t>(dataBuffer, inputIntData);
            break;
        case ACL_UINT64:
            SetInputValue<int64_t, uint64_t>(dataBuffer, inputIntData);
            break;
        default:
            ACL_LOG_WARN("unsupported type: %d", dataType);
            return false;
    }
    ACL_LOG_INFO("Get input data success, dt is %d, int type size is %zu, float type size is %zu",
                 dataType, inputIntData.size(), inputFloatData.size());
    return true;
}

static bool CheckIntValueRange(const std::map<AttrRangeType, ge::GeAttrValue> &valueRange,
    const std::vector<int64_t> &inputIntData)
{
    // Check range
    const auto it = valueRange.find(AttrRangeType::RANGE_TYPE);
    if (it != valueRange.end()) {
        std::vector<std::vector<int64_t>> valRangeInt;
        if (it->second.GetValue<std::vector<std::vector<int64_t>>>(valRangeInt) == ge::GRAPH_SUCCESS) {
            return CheckValRange(valRangeInt, inputIntData);
        } else {
            std::vector<int64_t> tmpInt;
            if (it->second.GetValue<std::vector<int64_t>>(tmpInt) == ge::GRAPH_SUCCESS) {
                valRangeInt.push_back(tmpInt);
                return CheckValRange(valRangeInt, inputIntData);
            } else {
                ACL_LOG_WARN("can not find listlist or list data struct");
                return false;
            }
        }
    }
    return true;
}

static bool CheckFloatValueRange(const std::map<AttrRangeType, ge::GeAttrValue> &valueRange,
    const std::vector<float32_t> &inputFloatData)
{
    // Check range
    const auto it = valueRange.find(AttrRangeType::RANGE_TYPE);
    if (it != valueRange.end()) {
        std::vector<std::vector<float32_t>> valRangeFloat;
        if (it->second.GetValue<std::vector<std::vector<float32_t>>>(valRangeFloat) == ge::GRAPH_SUCCESS) {
            return CheckValRange(valRangeFloat, inputFloatData);
        } else {
            std::vector<float32_t> tmpFloat;
            if (it->second.GetValue<std::vector<float32_t>>(tmpFloat) == ge::GRAPH_SUCCESS) {
                valRangeFloat.push_back(tmpFloat);
                return CheckValRange(valRangeFloat, inputFloatData);
            }
        }
    }
    return true;
}

static bool CheckGeTensorValue(const std::map<AttrRangeType, ge::GeAttrValue> &valueRange,
                               const aclDataBuffer *const dataBuffer)
{
    const auto it = valueRange.find(AttrRangeType::VALUE_TYPE);
    if (it == valueRange.end()) {
        return true;
    }
    const auto geTensor = const_cast<ge::GeAttrValue *>(&(it->second))->MutableGet<ge::GeTensor>();
    if (geTensor != nullptr) {
        void *const geDataPtr = reinterpret_cast<void *>(geTensor->MutableData().data());
        const size_t dataSize = geTensor->MutableData().size();
        auto *const inputDataPtr = dataBuffer->data;
        if ((dataSize >= dataBuffer->length) && (memcmp(geDataPtr, inputDataPtr, dataBuffer->length) == 0)) {
            return true;
        }
    }
    return false;
}

static bool IsSameRange(const std::map<AttrRangeType, ge::GeAttrValue> &value1,
                        const std::map<AttrRangeType, ge::GeAttrValue> &value2)
{
    const auto it1 = value1.find(AttrRangeType::RANGE_TYPE);
    const auto it2 = value2.find(AttrRangeType::RANGE_TYPE);
    if ((it1 == value1.end()) && (it2 == value2.end())) {
        return true;
    }
    if ((it1 != value1.end()) && (it2 != value2.end())) {
        std::vector<std::vector<int64_t>> valRangeInt1;
        std::vector<std::vector<int64_t>> valRangeInt2;
        if ((it1->second.GetValue<std::vector<std::vector<int64_t>>>(valRangeInt1) == ge::GRAPH_SUCCESS) &&
            (it1->second.GetValue<std::vector<std::vector<int64_t>>>(valRangeInt2) == ge::GRAPH_SUCCESS)) {
            return (valRangeInt1 == valRangeInt2);
        }
    }

    return false;
}

static bool IsSameValue(const std::map<AttrRangeType, ge::GeAttrValue> &value1,
                        const std::map<AttrRangeType, ge::GeAttrValue> &value2)
{
    const auto it1 = value1.find(AttrRangeType::VALUE_TYPE);
    const auto it2 = value2.find(AttrRangeType::VALUE_TYPE);
    if ((it1 == value1.end()) && (it2 == value2.end())) {
        return true;
    }
    if ((it1 != value1.end()) && (it2 != value2.end())) {
        const auto geTensor1 = (&(it1->second))->Get<ge::GeTensor>();
        const auto geTensor2 = (&(it2->second))->Get<ge::GeTensor>();
        if ((geTensor1 != nullptr) && (geTensor2 != nullptr)) {
            const void *const geDataPtr1 = static_cast<const void *>(geTensor1->GetData().data());
            const void *const geDataPtr2 = static_cast<const void *>(geTensor2->GetData().data());
            const size_t dataSize1 = geTensor1->GetData().size();
            const size_t dataSize2 = geTensor2->GetData().size();
            if ((dataSize1 == dataSize2) && (memcmp(geDataPtr1, geDataPtr2, dataSize1) == 0)) {
                return true;
            }
        }
    }
    return false;
}

bool IsSameValueRange(const std::map<AttrRangeType, ge::GeAttrValue> &valueRange1,
                      const std::map<AttrRangeType, ge::GeAttrValue> &valueRange2)
{
    if (valueRange1.size() != valueRange2.size()) {
        ACL_LOG_INFO("IsSameValueRange return fasle, size of valueRange1 is %zu, ize of valueRange2 is %zu",
            valueRange1.size(), valueRange1.size());
        return false;
    }

    if (!IsSameValue(valueRange1, valueRange2)) {
        ACL_LOG_INFO("Value of valueRange1 mismatch value of valueRange1");
        return false;
    }
    if (!IsSameRange(valueRange1, valueRange2)) {
        ACL_LOG_INFO("Range of valueRange1 mismatch range of valueRange1");
        return false;
    }
    return true;
}

bool ValueRangeCheck(const std::map<AttrRangeType, ge::GeAttrValue> &valueRange,
                     const aclDataBuffer *const dataBuffer, const aclDataType dataType)
{
    // value is not nullptr
    if (dataBuffer->data == nullptr) {
        return true;
    }
    // check value first
    if (!CheckGeTensorValue(valueRange, dataBuffer)) {
        return false;
    }
    std::vector<int64_t> inputIntData;
    std::vector<float32_t> inputFloatData;
    if (!GetInputData(dataBuffer, dataType, inputIntData, inputFloatData)) {
        return false;
    }
    // check value range
    if (!inputIntData.empty()) {
        return CheckIntValueRange(valueRange, inputIntData);
    }
    if (!inputFloatData.empty()) {
        return CheckFloatValueRange(valueRange, inputFloatData);
    }
    return true;
}

bool OpAttrEquals(const aclopAttr *const lhs, const aclopAttr *const rhs)
{
    if (lhs == rhs) {
        return true;
    }
    // inner logic, must not be nullptr
    if ((lhs == nullptr) || (rhs == nullptr)) {
        return false;
    }
    const size_t lhsAttrNum = lhs->Attrs().size();
    const size_t rhsAttrNum = rhs->Attrs().size();
    if (lhsAttrNum != rhsAttrNum) {
        return false;
    }
    // attr num of lhs and rhs is not 0, so attr is not nullptr
    const auto &lhsAttrs = lhs->Attrs();
    for (const auto &iter : rhs->Attrs()) {
        const std::string &attrName = iter.first;
        const ge::GeAttrValue &attrValue = iter.second;

        const auto it = lhsAttrs.find(attrName);
        if (it == lhsAttrs.end()) {
            return false;
        }
        // no overloaded operator != for GeAttrValue
        if (!attr_utils::AttrValueEquals(it->second, attrValue)) {
            return false;
        }
    }
    const auto constStrLeft = lhs->GetConstBuf();
    const auto constStrRight = rhs->GetConstBuf();
    if (constStrLeft != constStrRight) {
        return false;
    }
    return true;
}

uint64_t GetCurrentTimestamp()
{
    static uint64_t timeStamp = 0UL;
    ++timeStamp;
    return timeStamp;
}

static bool ConstToAttr(const vector<aclTensorDesc> &tensorDesc,
                        std::vector<std::string> &constStr)
{
    for (auto &desc : tensorDesc) {
        if (desc.isConst) {
            if ((desc.constDataBuf != nullptr) && (desc.constDataLen <= 0U)) {
                ACL_LOG_INNER_ERROR("[Check][constDataBuf]constDataBuf is not nullptr and dataLen is <= 0");
                return false;
            }
            if ((desc.constDataBuf == nullptr) && (desc.constDataLen > 0U)) {
                ACL_LOG_INNER_ERROR("[Check][constDataBuf]constDataBuf is nullptr and dataLen is > 0");
                return false;
            }
            std::string constBufStr;
            (void)constBufStr.assign(reinterpret_cast<const char_t *>(desc.constDataBuf.get()), desc.constDataLen);
            constStr.emplace_back(constBufStr);
        }
    }
    return true;
}

bool SaveConstToAttr(OpModelDef &modelDef)
{
    std::vector<std::string> constStr;
    bool ret = ConstToAttr(modelDef.inputDescArr, constStr);
    if (!ret) {
        ACL_LOG_INNER_ERROR("[Check][InputTenspr]inputTenspr get const dataLen failed");
        return false;
    }
    ret = ConstToAttr(modelDef.outputDescArr, constStr);
    if (!ret) {
        ACL_LOG_INNER_ERROR("[Check][OutputTensor]outputTensor get const dataLen failed");
        return false;
    }
    // opAttr is not nullptr
    for (size_t i = 0U; i < constStr.size(); ++i) {
        modelDef.opAttr.EmplaceConstBuf(constStr[i]);
    }
    return true;
}

static bool ConstToAttr(const int32_t tensorNum,
                        const aclTensorDesc *const tensorDesc[],
                        std::vector<std::string> &constStr)
{
    for (int32_t i = 0; i < tensorNum; ++i) {
        const aclTensorDesc *const desc = tensorDesc[i];
        if (desc->isConst) {
            if ((desc->constDataBuf != nullptr) && (desc->constDataLen <= 0U)) {
                ACL_LOG_INNER_ERROR("[Check][constDataBuf]constDataBuf is not nullptr and dataLen is <= 0");
                return false;
            }
            if ((desc->constDataBuf == nullptr) && (desc->constDataLen > 0U)) {
                ACL_LOG_INNER_ERROR("[Check][constDataBuf]constDataBuf is nullptr and dataLen is > 0");
                return false;
            }
            std::string constBufStr =
                std::string(reinterpret_cast<const char_t *>(desc->constDataBuf.get()), desc->constDataLen);
            constStr.emplace_back(constBufStr);
        }
    }
    return true;
}

bool SaveConstToAttr(const AclOp &opDesc, aclopAttr *const opAttr)
{
    std::vector<std::string> constStr;
    bool ret = ConstToAttr(opDesc.numInputs, opDesc.inputDesc, constStr);
    if (!ret) {
        ACL_LOG_INNER_ERROR("[Check][InputTenspr]inputTenspr get const dataLen failed");
        return false;
    }
    ret = ConstToAttr(opDesc.numOutputs, opDesc.outputDesc, constStr);
    if (!ret) {
        ACL_LOG_INNER_ERROR("[Check][OutputTensor]outputTensor get const dataLen failed");
        return false;
    }
    // opAttr is not nullptr
    opAttr->ClearConstBuf();
    for (size_t i = 0U; i < constStr.size(); ++i) {
        opAttr->EmplaceConstBuf(constStr[i]);
    }
    return true;
}
} // namespace attr_utils
} // acl
