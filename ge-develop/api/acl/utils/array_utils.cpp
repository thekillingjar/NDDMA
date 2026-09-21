/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "array_utils.h"
#include <map>
#include <vector>
#include <set>

namespace acl {
namespace array_utils {
bool IsAllTensorEmpty(const int32_t size, const aclTensorDesc *const *const arr)
{
    if (size == 0) {
        return false;
    }

    for (int32_t idx = 0; idx < size; ++idx) {
        bool flag = false;
        for (size_t idy = 0U; idy < arr[idx]->dims.size(); ++idy) {
            if (arr[idx]->dims[idy] == 0) {
                flag = true;
                break;
            }
        }

        if (!flag) {
            return false;
        }
    }

    return true;
}

bool IsAllTensorEmpty(const int32_t size, const aclDataBuffer *const *const arr)
{
    if (size == 0) {
        return false;
    }

    for (int32_t idx = 0; idx < size; ++idx) {
        if (arr[idx]->length > 0U) {
            return false;
        }
    }

    return true;
}

bool GetDynamicInputIndex(const int32_t size, const aclTensorDesc *const *const arr, DynamicInputIndexPair &indexPair)
{
    indexPair.first.clear();
    indexPair.second.clear();
    auto &startVec = indexPair.first;
    auto &endVec = indexPair.second;
    if (size == 0) {
        return true;
    }

    std::set<std::string> attrNameSet;
    int32_t start = -1;
    int32_t ended = -1;
    std::string lastName;
    for (int32_t i = 0; i < size; ++i) {
        const std::string curName = arr[i]->dynamicInputName;
        if ((curName.size() == 0U) && (lastName.size() == 0U)) {
            continue;
        }

        if (curName != lastName) {
            // current name exist before
            if (attrNameSet.find(curName) != attrNameSet.end()) {
                ACL_LOG_ERROR("[Check][DynamicInputName]DynamicInputName[%s] is duplicated to other input name", curName.c_str());
                acl::AclErrorLogManager::ReportInputError(acl::INVALID_PARAM_MSG,
                    std::vector<const char *>({"param", "value", "reason"}),
                    std::vector<const char *>({"dynamicInputName", curName.c_str(),
                    "DynamicInputName is duplicated to other input name."}));
                return false;
            }
            // valid attr name
            if (lastName.size() > 0U) {
                startVec.emplace_back(start);
                endVec.emplace_back(ended);
                (void)attrNameSet.insert(lastName);
            }

            if (curName.size() > 0U) {
                start = i;
            }
        }
        ended = i; // refresh ended index
        lastName = curName; // refresh lastName
    }

    if (lastName.size() > 0U) {
        startVec.emplace_back(start);
        endVec.emplace_back(ended);
    }

    return true;
}

aclError CheckDataBufferArry(const int32_t size, const aclDataBuffer *const *const arr)
{
    if (size == 0) {
        return ACL_SUCCESS;
    }

    ACL_REQUIRES_NOT_NULL_WITH_INPUT_REPORT(arr);
    for (int32_t idx = 0; idx < size; ++idx) {
        ACL_REQUIRES_NOT_NULL_WITH_INPUT_REPORT(arr[idx]);
        if ((arr[idx]->data == nullptr) && (arr[idx]->length > 0U)) {
            ACL_LOG_ERROR("[Check][data]data of element at index[%d] while size is larger than 0", idx);
            const std::string errMsg = acl::AclErrorLogManager::FormatStr("data of element at index[%d]"
                "while size is larger than 0", idx);
            acl::AclErrorLogManager::ReportInputError(acl::INVALID_PARAM_MSG,
                std::vector<const char *>({"param", "value", "reason"}),
                std::vector<const char *>({"data", "nullptr", errMsg.c_str()}));
            return ACL_ERROR_INVALID_PARAM;
        }
    }

    return ACL_SUCCESS;
}

aclError IsHostMemTensorDesc(const int32_t size, const aclTensorDesc *const *const arr)
{
    if (size == 0) {
        return ACL_SUCCESS;
    }
    ACL_REQUIRES_NOT_NULL(arr);
    for (int32_t idx = 0; idx < size; ++idx) {
        if ((!arr[idx]->IsConstTensor()) && (arr[idx]->IsHostMemTensor())) {
            ACL_LOG_INNER_ERROR("[Check][HostMemTensorDesc]PlaceMent of element at index %d is hostMem", idx);;
            return ACL_ERROR_INVALID_PARAM;
        }
    }

    return ACL_SUCCESS;
}
} // namespace array_utils
} // acl
