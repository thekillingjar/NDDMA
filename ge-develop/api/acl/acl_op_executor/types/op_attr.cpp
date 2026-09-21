/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "op_attr.h"
#include <sstream>
#include "graph/ge_attr_value.h"
#include "utils/attr_utils.h"
#include "common/prof_api_reg.h"
#include "single_op/acl_op_executor_impl.h"

aclopAttr *aclopCreateAttrImpl()
{
    ACL_PROFILING_REG(acl::AclProfType::AclopCreateAttr);
    auto *const attr = new(std::nothrow) aclopAttr();
    if (attr == nullptr) {
        ACL_LOG_INNER_ERROR("[Check][Attr]opAttr memory apply failed");
        return nullptr;
    }
    return attr;
}

void aclopDestroyAttrImpl(const aclopAttr *attr)
{
    ACL_PROFILING_REG(acl::AclProfType::AclopDestroyAttr);
    ACL_DELETE_AND_SET_NULL(attr);
}

aclError aclopSetAttrBoolImpl(aclopAttr *attr, const char *attrName, uint8_t attrValue)
{
    ACL_REQUIRES_NOT_NULL_WITH_INPUT_REPORT(attr);
    return attr->SetAttr(attrName, static_cast<bool>(attrValue));
}

aclError aclopSetAttrIntImpl(aclopAttr *attr, const char *attrName, int64_t attrValue)
{
    ACL_REQUIRES_NOT_NULL_WITH_INPUT_REPORT(attr);
    return attr->SetAttr(attrName, attrValue);
}

aclError aclopSetAttrFloatImpl(aclopAttr *attr, const char *attrName, float attrValue)
{
    ACL_REQUIRES_NOT_NULL_WITH_INPUT_REPORT(attr);
    return attr->SetAttr(attrName, attrValue);
}

aclError aclopSetAttrStringImpl(aclopAttr *attr, const char *attrName, const char *attrValue)
{
    ACL_REQUIRES_NOT_NULL_WITH_INPUT_REPORT(attr);
    ACL_REQUIRES_NOT_NULL_WITH_INPUT_REPORT(attrValue);
    return attr->SetAttr(attrName, std::string(attrValue));
}

aclError aclopSetAttrDataTypeImpl(aclopAttr *attr, const char *attrName, aclDataType attrValue)
{
    ACL_REQUIRES_NOT_NULL_WITH_INPUT_REPORT(attr);
    ge::DataType dt = ge::DT_UNDEFINED;
    if (attrValue != ACL_DT_UNDEFINED) {
        dt = static_cast<ge::DataType>(attrValue);
    }
    return attr->SetAttr(attrName, dt);
}

aclError aclopSetAttrListDataTypeImpl(aclopAttr *attr, const char *attrName, int numValues, const aclDataType values[])
{
    ACL_REQUIRES_NOT_NULL_WITH_INPUT_REPORT(attr);
    ACL_REQUIRES_NOT_NULL_WITH_INPUT_REPORT(attrName);
    if (numValues > 0) {
        ACL_REQUIRES_NOT_NULL_WITH_INPUT_REPORT(values);
    }
    std::vector<ge::DataType> dtVec;
    for (int32_t i = 0; i < numValues; ++i) {
        ge::DataType dt = ge::DT_UNDEFINED;
        if (values[i] != ACL_DT_UNDEFINED) {
            dt = static_cast<ge::DataType>(values[i]);
        }
        dtVec.push_back(dt);
    }
    return attr->SetAttr(attrName, numValues, dtVec.data());
}

aclError aclopSetAttrListBoolImpl(aclopAttr *attr, const char *attrName, int numValues, const uint8_t *values)
{
    ACL_REQUIRES_NOT_NULL_WITH_INPUT_REPORT(attr);
    const auto *const boolValues = static_cast<const bool *>(static_cast<const void *>(values));
    return attr->SetAttr(attrName, numValues, boolValues);
}

aclError aclopSetAttrListIntImpl(aclopAttr *attr, const char *attrName, int numValues, const int64_t *values)
{
    ACL_REQUIRES_NOT_NULL_WITH_INPUT_REPORT(attr);
    return attr->SetAttr(attrName, numValues, values);
}

aclError aclopSetAttrListFloatImpl(aclopAttr *attr, const char *attrName, int numValues, const float *values)
{
    ACL_REQUIRES_NOT_NULL_WITH_INPUT_REPORT(attr);
    return attr->SetAttr(attrName, numValues, values);
}

aclError aclopSetAttrListStringImpl(aclopAttr *attr, const char *attrName, int numValues, const char **values)
{
    ACL_REQUIRES_NOT_NULL_WITH_INPUT_REPORT(attr);
    ACL_REQUIRES_OK(acl::array_utils::CheckPtrArray(numValues, values));
    std::vector<std::string> strValues;
    for (int32_t i = 0; i < numValues; ++i) {
        strValues.emplace_back(std::string(values[i]));
    }
    return attr->SetAttr(attrName, strValues);
}

aclError aclopSetAttrListListIntImpl(aclopAttr *attr,
                                 const char *attrName,
                                 int numLists,
                                 const int *numValues,
                                 const int64_t *const values[])
{
    ACL_REQUIRES_NOT_NULL_WITH_INPUT_REPORT(attr);
    ACL_REQUIRES_NOT_NULL_WITH_INPUT_REPORT(attrName);
    ACL_REQUIRES_NOT_NULL_WITH_INPUT_REPORT(numValues);
    ACL_REQUIRES_OK(acl::array_utils::CheckPtrArray(numLists, values));
    std::vector<std::vector<int64_t>> valueVec;
    for (int32_t i = 0; i < numLists; ++i) {
        std::vector<int64_t> vec;
        for (int32_t j = 0; j < numValues[i]; ++j) {
            vec.emplace_back(values[i][j]);
        }
        valueVec.emplace_back(vec);
    }

    return attr->SetAttr(attrName, valueVec);
}

aclopAttr::aclopAttr(const aclopAttr &opAttr)
{
    this->attrs_ = opAttr.attrs_;
    this->digest_ = opAttr.digest_;
    this->constDataBuf_ = opAttr.constDataBuf_;
}

std::string aclopAttr::DebugString() const
{
    return acl::attr_utils::AttrMapToString(attrs_);
}

void aclopAttr::UpdateDigest()
{
    digest_ = acl::attr_utils::AttrMapToDigest(attrs_);
}

size_t aclopAttr::GetDigest() const
{
    if (digest_ != 0UL) {
        return digest_;
    }

    return acl::attr_utils::AttrMapToDigest(attrs_);
}

bool aclopAttr::HasAttr(const char_t *const attrName) const
{
    if (attrName != nullptr) {
        const auto it = attrs_.find(std::string(attrName));
        if (it != attrs_.end()) {
            ACL_LOG_INFO("Find attr [%s] from attrs", attrName);
            return true;
        }
    }
    return false;
}
