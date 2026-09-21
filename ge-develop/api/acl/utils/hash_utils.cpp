/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "hash_utils.h"

namespace acl {
namespace hash_utils {
// use aclTensorDesc to calculate a hash seed according to HashCombine function
aclError GetTensorDescHash(const int32_t num, const aclTensorDesc *const descArr[], size_t &seed)
{
    if ((num > 0) && (descArr == nullptr)) {
        ACL_LOG_ERROR("[Check][Params] param descArr must not be null");
        return ACL_ERROR_FAILURE;
    }
    HashCombine(seed, num);
    for (int32_t i = 0; i < num; ++i) {
        ACL_REQUIRES_NOT_NULL(descArr[i]);
        HashCombine(seed, static_cast<int32_t>(descArr[i]->dataType));
        HashCombine(seed, static_cast<int32_t>(descArr[i]->format));
        if (descArr[i]->storageFormat != ACL_FORMAT_UNDEFINED) {
            HashCombine(seed, static_cast<int32_t>(descArr[i]->storageFormat));
        }
        for (auto &dim : descArr[i]->dims) {
            HashCombine(seed, dim);
        }
        for (auto &shapeRange : descArr[i]->shapeRange) {
            HashCombine(seed, shapeRange.first);
            HashCombine(seed, shapeRange.second);
        }
        HashCombine(seed, descArr[i]->isConst);
        HashCombine(seed, static_cast<int32_t>(descArr[i]->memtype));
    }
    return ACL_SUCCESS;
}

aclError GetTensorDescHashDynamic(const int32_t num, const aclTensorDesc *const descArr[], size_t &seed)
{
    if ((num > 0) && (descArr == nullptr)) {
        ACL_LOG_ERROR("[Check][Params] param descArr must not be null");
        return ACL_ERROR_FAILURE;
    }
    HashCombine(seed, num);
    for (int32_t i = 0; i < num; ++i) {
        ACL_REQUIRES_NOT_NULL(descArr[i]);
        HashCombine(seed, static_cast<int32_t>(descArr[i]->dataType));
        HashCombine(seed, static_cast<int32_t>(descArr[i]->format));
        if (descArr[i]->storageFormat != ACL_FORMAT_UNDEFINED) {
            HashCombine(seed, static_cast<int32_t>(descArr[i]->storageFormat));
        }
        HashCombine(seed, descArr[i]->isConst);
        HashCombine(seed, static_cast<int32_t>(descArr[i]->memtype));
    }
    return ACL_SUCCESS;
}

aclError GetAclOpHash(const AclOp &aclOp, const aclopAttr *const opAttr, const size_t attrDigest, size_t &seed)
{
    // Init seed to maintain consistency between Insert and Get
    seed = 0U;

    HashCombine(seed, aclOp.opType);

    ACL_REQUIRES_OK(GetTensorDescHash(aclOp.numInputs, aclOp.inputDesc, seed));

    ACL_REQUIRES_OK(GetTensorDescHash(aclOp.numOutputs, aclOp.outputDesc, seed));

    HashCombine(seed, attrDigest);
    // need use tmp opAttr, because aclop.opAttr may not be matched
    if (opAttr != nullptr) {
        for (auto &constBuf : opAttr->GetConstBuf()) {
            HashCombine(seed, constBuf);
        }
    }
    return ACL_SUCCESS;
}

aclError GetAclOpHashDynamic(const AclOp &aclOp, const aclopAttr *const opAttr, const size_t attrDigest,
                             size_t &seed, const uint64_t seq)
{
    // Init seed to maintain consistency between Insert and Get
    seed = 0U;

    HashCombine(seed, aclOp.opType);
    HashCombine(seed, seq);

    ACL_REQUIRES_OK(GetTensorDescHashDynamic(aclOp.numInputs, aclOp.inputDesc, seed));

    ACL_REQUIRES_OK(GetTensorDescHashDynamic(aclOp.numOutputs, aclOp.outputDesc, seed));

    HashCombine(seed, attrDigest);
    // need use tmp opAttr, because aclop.opAttr may not be matched
    if (opAttr != nullptr) {
        for (auto &constBuf : opAttr->GetConstBuf()) {
            HashCombine(seed, constBuf);
        }
    }
    return ACL_SUCCESS;
}

} // namespace hash_utils
} // namespace acl