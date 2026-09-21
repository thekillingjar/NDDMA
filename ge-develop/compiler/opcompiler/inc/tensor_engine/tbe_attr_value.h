/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ATC_OPCOMPILER_INC_TENSOR_ENGINE_TBE_ATTR_VALUE_H_
#define ATC_OPCOMPILER_INC_TENSOR_ENGINE_TBE_ATTR_VALUE_H_

#include <vector>
#include "tensor_engine/fusion_types.h"

namespace te {
class TbeAttrValue {
public:
    TbeAttrValue() : dtype_(ATTR_INT8)
    {
    }

    ~TbeAttrValue()
    {
    }

    TbeAttrValue(const std::string &name, const int8_t value):name_(name), int8_value_(value), dtype_(ATTR_INT8)
    {
    }

    TbeAttrValue(const std::string &name, const uint8_t value):name_(name), uint8_value_(value), dtype_(ATTR_UINT8)
    {
    }

    TbeAttrValue(const std::string &name, const int16_t value):name_(name), int16_value_(value), dtype_(ATTR_INT16)
    {
    }

    TbeAttrValue(const std::string &name, const uint16_t value):name_(name), uint16_value_(value), dtype_(ATTR_UINT16)
    {
    }

    TbeAttrValue(const std::string &name, const int32_t value):name_(name), int32_value_(value), dtype_(ATTR_INT32)
    {
    }

    TbeAttrValue(const std::string &name, const uint32_t value):name_(name), uint32_value_(value), dtype_(ATTR_UINT32)
    {
    }

    TbeAttrValue(const std::string &name, const int64_t value):name_(name), int64_value_(value), dtype_(ATTR_INT64)
    {
    }

    TbeAttrValue(const std::string &name, const uint64_t value):name_(name), uint64_value_(value), dtype_(ATTR_UINT64)
    {
    }

    TbeAttrValue(const std::string &name, const float value):name_(name), float_value_(value), dtype_(ATTR_FLOAT32)
    {
    }

    TbeAttrValue(const std::string &name, const double value):name_(name), double_value_(value), dtype_(ATTR_DOUBLE)
    {
    }

    TbeAttrValue(const std::string &name, const bool value):name_(name), bool_value_(value), dtype_(ATTR_BOOL)
    {
    }

    TbeAttrValue(const std::string &name, const std::string &value):name_(name), str_value_(value), dtype_(ATTR_STR)
    {
    }

    TbeAttrValue(const std::string &name, const std::vector<int8_t> &value)
        : name_(name), list_int8_value_(value), dtype_(ATTR_LIST_INT8)
    {
    }

    TbeAttrValue(const std::string &name, const std::vector<uint8_t> &value)
        : name_(name), list_uint8_value_(value), dtype_(ATTR_LIST_UINT8)
    {
    }

    TbeAttrValue(const std::string &name, const std::vector<int16_t> &value)
        : name_(name), list_int16_value_(value), dtype_(ATTR_LIST_INT16)
    {
    }

    TbeAttrValue(const std::string &name, const std::vector<uint16_t> &value)
        : name_(name), list_uint16_value_(value), dtype_(ATTR_LIST_UINT16)
    {
    }

    TbeAttrValue(const std::string &name, const std::vector<int32_t> &value)
        : name_(name), list_int32_value_(value), dtype_(ATTR_LIST_INT32)
    {
    }

    TbeAttrValue(const std::string &name, const std::vector<uint32_t> &value)
        : name_(name), list_uint32_value_(value), dtype_(ATTR_LIST_UINT32)
    {
    }

    TbeAttrValue(const std::string &name, const std::vector<int64_t> &value)
        : name_(name), list_int64_value_(value), dtype_(ATTR_LIST_INT64)
    {
    }

    TbeAttrValue(const std::string &name, const std::vector<uint64_t> &value)
        : name_(name), list_uint64_value_(value), dtype_(ATTR_LIST_UINT64)
    {
    }

    TbeAttrValue(const std::string &name, const std::vector<float> &value)
        : name_(name), list_float_value_(value), dtype_(ATTR_LIST_FLOAT32)
    {
    }

    TbeAttrValue(const std::string &name, const std::vector<double> &value)
        : name_(name), list_double_value_(value), dtype_(ATTR_LIST_DOUBLE)
    {
    }

    TbeAttrValue(const std::string &name, const std::vector<bool> &value)
        : name_(name), list_bool_value_(value), dtype_(ATTR_LIST_BOOL)
    {
    }

    TbeAttrValue(const std::string &name, const std::vector<std::string> &value)
        : name_(name), list_str_value_(value), dtype_(ATTR_LIST_STR)
    {
    }

    TbeAttrValue(const std::string &name, const std::vector<std::vector<int64_t>> &value)
        : name_(name), list_list_int64_value_(value), dtype_(ATTR_LIST_LIST_INT64)
    {
    }

    TbeAttrValue(const std::string &name, const std::vector<std::vector<float>> &value)
        : name_(name), list_list_float_value_(value), dtype_(ATTR_LIST_LIST_FLOAT)
    {
    }

    const ATTR_DTYPE& GetType() const
    {
        return dtype_;
    }

    const std::string& GetName() const
    {
        return name_;
    }

    void GetValue(int8_t& value) const
    {
        value = int8_value_;
    }

    void GetValue(uint8_t& value) const
    {
        value = uint8_value_;
    }

    void GetValue(int16_t& value) const
    {
        value = int16_value_;
    }

    void GetValue(uint16_t& value) const
    {
        value = uint16_value_;
    }

    void GetValue(int32_t& value) const
    {
        value = int32_value_;
    }

    void GetValue(uint32_t& value) const
    {
        value = uint32_value_;
    }

    void GetValue(int64_t& value) const
    {
        value = int64_value_;
    }

    void GetValue(uint64_t& value) const
    {
        value = uint64_value_;
    }

    void GetValue(float& value) const
    {
        value = float_value_;
    }

    void GetValue(double& value) const
    {
        value = double_value_;
    }

    void GetValue(bool& value) const
    {
        value = bool_value_;
    }

    void GetValue(std::string& value) const
    {
        value = str_value_;
    }

    void GetValue(std::vector<int8_t>& value) const
    {
        value = list_int8_value_;
    }

    void GetValue(std::vector<uint8_t>& value) const
    {
        value = list_uint8_value_;
    }

    void GetValue(std::vector<int16_t>& value) const
    {
        value = list_int16_value_;
    }

    void GetValue(std::vector<uint16_t>& value) const
    {
        value = list_uint16_value_;
    }

    void GetValue(std::vector<int32_t>& value) const
    {
        value = list_int32_value_;
    }

    void GetValue(std::vector<uint32_t>& value) const
    {
        value = list_uint32_value_;
    }

    void GetValue(std::vector<int64_t>& value) const
    {
        value = list_int64_value_;
    }

    void GetValue(std::vector<uint64_t>& value) const
    {
        value = list_uint64_value_;
    }

    void GetValue(std::vector<float>& value) const
    {
        value = list_float_value_;
    }

    void GetValue(std::vector<double>& value) const
    {
        value = list_double_value_;
    }

    void GetValue(std::vector<bool>& value) const
    {
        value = list_bool_value_;
    }

    void GetValue(std::vector<std::string>& value) const
    {
        value = list_str_value_;
    }

    void GetValue(std::vector<std::vector<int64_t>>& value) const
    {
        value = list_list_int64_value_;
    }

    void GetValue(std::vector<std::vector<float>>& value) const
    {
        value = list_list_float_value_;
    }

    ATTR_SHAPETYPE GetSType() const
    {
        return stype_;
    }

    void SetSType(const ATTR_SHAPETYPE& stype)
    {
        stype_ = stype;
    }

    void SetAttrSupAllFlag(const bool &isAttrSupportAllValue)
    {
        isAttrSupportAllValue_ = isAttrSupportAllValue;
    }

    bool GetAttrSupAllFlag() const
    {
        return isAttrSupportAllValue_;
    }

    void SetAttrHasDefaultValFlag(const bool &isAttrHasDefaultValue)
    {
        isAttrHasDefaultValue_ = isAttrHasDefaultValue;
    }

    bool GetAttrHasDefaultValFlag() const
    {
        return isAttrHasDefaultValue_;
    }

    void SetIsDefaultValue(const bool &isDefaultValue)
    {
        isDefaultValue_ = isDefaultValue;
    }

    bool GetIsDefaultValue() const
    {
        return isDefaultValue_;
    }

    void SetIsRequiredAttr(const bool &isRequiredAttr)
    {
        isRequiredAttr_ = isRequiredAttr;
    }

    bool GetIsRequiredAttr() const
    {
        return isRequiredAttr_;
    }

    bool operator==(TbeAttrValue& rObject);

private:
    // need to adapt operator== func while adding new variable
    std::string name_;
    int8_t   int8_value_{0};
    uint8_t  uint8_value_{0};
    int16_t  int16_value_{0};
    uint16_t uint16_value_{0};
    int32_t  int32_value_{0};
    uint32_t uint32_value_{0};
    int64_t  int64_value_{0};
    uint64_t uint64_value_{0};
    float float_value_{0.0};
    double double_value_{0.0};
    bool bool_value_{false};
    std::string str_value_;

    std::vector<int8_t>   list_int8_value_;
    std::vector<uint8_t>  list_uint8_value_;
    std::vector<int16_t>  list_int16_value_;
    std::vector<uint16_t> list_uint16_value_;
    std::vector<int32_t>  list_int32_value_;
    std::vector<uint32_t> list_uint32_value_;
    std::vector<int64_t>  list_int64_value_;
    std::vector<uint64_t> list_uint64_value_;
    std::vector<float>    list_float_value_;
    std::vector<double>   list_double_value_;
    std::vector<bool>     list_bool_value_;
    std::vector<std::string> list_str_value_;
    std::vector<std::vector<int64_t>> list_list_int64_value_;
    std::vector<std::vector<float>> list_list_float_value_;

    ATTR_DTYPE dtype_{ATTR_FLOAT32};
    ATTR_SHAPETYPE stype_{ATTR_SHAPE_TUPLE};

    // binary infos
    bool isAttrSupportAllValue_ = false;
    bool isAttrHasDefaultValue_ = false;
    bool isDefaultValue_ = false;
    bool isRequiredAttr_ = true;
};
}
#endif  // ATC_OPCOMPILER_INC_TENSOR_ENGINE_TBE_ATTR_VALUE_H_