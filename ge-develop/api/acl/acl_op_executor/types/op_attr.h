/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ACL_TYPES_OP_ATTR_H_
#define ACL_TYPES_OP_ATTR_H_

#include "graph/op_desc.h"
#include "common/log_inner.h"
#include "utils/array_utils.h"

struct ACL_FUNC_VISIBILITY aclopAttr {
    aclopAttr() = default;
    aclopAttr(const aclopAttr &opAttr);
    aclopAttr& operator=(const aclopAttr &opAttr) = delete;

    ~aclopAttr() = default;

    inline const std::map<std::string, ge::GeAttrValue> &Attrs() const
    {
        return attrs_;
    }

    inline const std::map<std::string, ge::GeAttrValue> &EmplaceAttr(const std::string &str, ge::GeAttrValue val)
    {
        (void)attrs_.emplace(str, val);
        return attrs_;
    }

    inline void ClearConstBuf()
    {
        constDataBuf_.clear();
    }

    inline void EmplaceConstBuf(std::string &str)
    {
        constDataBuf_.emplace_back(str);
    }

    inline const std::vector<std::string> &GetConstBuf() const
    {
        return constDataBuf_;
    }

    template<typename T>
    aclError SetAttr(const char_t *const attrName, const T val)
    {
        ACL_REQUIRES_NOT_NULL_WITH_INPUT_REPORT(attrName);
        const auto attrVal = ge::GeAttrValue::CreateFrom<T>(val);
        attrs_[std::string(attrName)] = attrVal;
        return ACL_SUCCESS;
    }

    template<typename T>
    aclError SetAttr(const char_t *const attrName, const int32_t numValues, const T *const values)
    {
        ACL_REQUIRES_NOT_NULL_WITH_INPUT_REPORT(attrName);
        if (numValues > 0) {
            ACL_REQUIRES_NOT_NULL_WITH_INPUT_REPORT(values);
        }
        std::vector<T> valueVec;
        for (int32_t i = 0; i < numValues; ++i) {
            valueVec.push_back(values[i]);
        }

        const auto attrValues = ge::GeAttrValue::CreateFrom<std::vector<T>>(valueVec);
        attrs_[std::string(attrName)] = attrValues;
        return ACL_SUCCESS;
    }

    void UpdateDigest();

    size_t GetDigest() const;

    std::string DebugString() const;

    bool HasAttr(const char_t *const attrName) const;

private:
    std::map<std::string, ge::GeAttrValue> attrs_;
    std::vector<std::string> constDataBuf_;
    mutable size_t digest_ = 0U;
};
#endif // ACL_TYPES_OP_ATTR_H_
