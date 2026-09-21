/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "attr_value_impl.h"
#include "common/udf_log.h"

namespace FlowFunc {
int32_t AttrValueImpl::GetVal(AscendString &value) const {
    if (proto_attr_.value_case() != ff::udf::AttrValue::kS) {
        UDF_LOG_ERROR("attr type mismatch, value case=%d is not kS=%d",
            static_cast<int32_t>(proto_attr_.value_case()), static_cast<int32_t>(ff::udf::AttrValue::kS));
        return FLOW_FUNC_ERR_ATTR_TYPE_MISMATCH;
    }
    value = AscendString(proto_attr_.s().c_str(), proto_attr_.s().size());
    return FLOW_FUNC_SUCCESS;
}

int32_t AttrValueImpl::GetVal(std::vector<AscendString> &value) const {
    if (!proto_attr_.has_array()) {
        UDF_LOG_ERROR("proto is not has array, value case=%d, kArray=%d",
            static_cast<int32_t>(proto_attr_.value_case()), static_cast<int32_t>(ff::udf::AttrValue::kArray));
        return FLOW_FUNC_ERR_ATTR_TYPE_MISMATCH;
    }
    auto &array = proto_attr_.array();
    value.reserve(static_cast<uint32_t>(array.s_size()));
    for (const auto &str : array.s()) {
        value.emplace_back(AscendString(str.c_str(), str.size()));
    }
    return FLOW_FUNC_SUCCESS;
}

int32_t AttrValueImpl::GetVal(int64_t &value) const {
    if (proto_attr_.value_case() != ff::udf::AttrValue::kI) {
        UDF_LOG_ERROR("attr type mismatch, value case=%d is not kI=%d",
            static_cast<int32_t>(proto_attr_.value_case()), static_cast<int32_t>(ff::udf::AttrValue::kI));
        return FLOW_FUNC_ERR_ATTR_TYPE_MISMATCH;
    }
    value = proto_attr_.i();
    return FLOW_FUNC_SUCCESS;
}

int32_t AttrValueImpl::GetVal(std::vector<int64_t> &value) const {
    if (!proto_attr_.has_array()) {
        UDF_LOG_ERROR("proto is not has array, value case=%d, kArray=%d",
            static_cast<int32_t>(proto_attr_.value_case()), static_cast<int32_t>(ff::udf::AttrValue::kArray));
        return FLOW_FUNC_ERR_ATTR_TYPE_MISMATCH;
    }
    auto &array = proto_attr_.array();
    value.reserve(static_cast<uint32_t>(array.i_size()));
    for (auto num : array.i()) {
        value.emplace_back(num);
    }
    return FLOW_FUNC_SUCCESS;
}

int32_t AttrValueImpl::GetVal(std::vector<std::vector<int64_t>> &value) const {
    if (!proto_attr_.has_list_list_i()) {
        UDF_LOG_ERROR("proto is not has array, value case=%d, kArray=%d",
            static_cast<int32_t>(proto_attr_.value_case()), static_cast<int32_t>(ff::udf::AttrValue::kArray));
        return FLOW_FUNC_ERR_ATTR_TYPE_MISMATCH;
    }
    auto &array = proto_attr_.list_list_i();
    value.reserve(static_cast<uint32_t>(array.list_i_size()));
    for (auto &list_int : array.list_i()) {
        std::vector<int64_t> vec;
        vec.reserve(static_cast<uint32_t>(list_int.i_size()));
        for (auto i : list_int.i()) {
            vec.emplace_back(i);
        }
        value.emplace_back(vec);
    }
    return FLOW_FUNC_SUCCESS;
}

int32_t AttrValueImpl::GetVal(float &value) const {
    if (proto_attr_.value_case() != ff::udf::AttrValue::kF) {
        UDF_LOG_ERROR("attr type mismatch, value case=%d is not kF=%d",
            static_cast<int32_t>(proto_attr_.value_case()), static_cast<int32_t>(ff::udf::AttrValue::kF));
        return FLOW_FUNC_ERR_ATTR_TYPE_MISMATCH;
    }
    value = proto_attr_.f();
    return FLOW_FUNC_SUCCESS;
}

int32_t AttrValueImpl::GetVal(std::vector<float> &value) const {
    if (!proto_attr_.has_array()) {
        UDF_LOG_ERROR("proto is not has array, value case=%d, kArray=%d",
            static_cast<int32_t>(proto_attr_.value_case()), static_cast<int32_t>(ff::udf::AttrValue::kArray));
        return FLOW_FUNC_ERR_ATTR_TYPE_MISMATCH;
    }
    auto &array = proto_attr_.array();
    value.reserve(static_cast<uint32_t>(array.f_size()));
    for (auto f_num : array.f()) {
        value.emplace_back(f_num);
    }
    return FLOW_FUNC_SUCCESS;
}

int32_t AttrValueImpl::GetVal(bool &value) const {
    if (proto_attr_.value_case() != ff::udf::AttrValue::kB) {
        UDF_LOG_ERROR("attr type mismatch, value case=%d is not kB=%d",
            static_cast<int32_t>(proto_attr_.value_case()), static_cast<int32_t>(ff::udf::AttrValue::kB));
        return FLOW_FUNC_ERR_ATTR_TYPE_MISMATCH;
    }
    value = proto_attr_.b();
    return FLOW_FUNC_SUCCESS;
}

int32_t AttrValueImpl::GetVal(std::vector<bool> &value) const {
    if (!proto_attr_.has_array()) {
        UDF_LOG_ERROR("proto is not has array, value case=%d, kArray=%d",
            static_cast<int32_t>(proto_attr_.value_case()), static_cast<int32_t>(ff::udf::AttrValue::kArray));
        return FLOW_FUNC_ERR_ATTR_TYPE_MISMATCH;
    }
    auto &array = proto_attr_.array();
    value.reserve(static_cast<uint32_t>(array.b_size()));
    for (auto bool_num : array.b()) {
        value.emplace_back(bool_num);
    }
    return FLOW_FUNC_SUCCESS;
}

int32_t AttrValueImpl::GetVal(TensorDataType &value) const {
    if (proto_attr_.value_case() != ff::udf::AttrValue::kType) {
        UDF_LOG_ERROR("attr type mismatch, value case=%d is not kType=%d",
            static_cast<int32_t>(proto_attr_.value_case()), static_cast<int32_t>(ff::udf::AttrValue::kType));
        return FLOW_FUNC_ERR_ATTR_TYPE_MISMATCH;
    }
    value = static_cast<TensorDataType>(proto_attr_.type());
    return FLOW_FUNC_SUCCESS;
}

int32_t AttrValueImpl::GetVal(std::vector<TensorDataType> &value) const {
    if (!proto_attr_.has_array()) {
        UDF_LOG_ERROR("proto is not has array, value case=%d, kArray=%d",
            static_cast<int32_t>(proto_attr_.value_case()), static_cast<int32_t>(ff::udf::AttrValue::kArray));
        return FLOW_FUNC_ERR_ATTR_TYPE_MISMATCH;
    }
    auto &array = proto_attr_.array();
    value.reserve(static_cast<uint32_t>(array.type_size()));
    for (auto type : array.type()) {
        value.emplace_back(static_cast<TensorDataType>(type));
    }
    return FLOW_FUNC_SUCCESS;
}
}