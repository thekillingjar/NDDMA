/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ACL_UTILS_HASH_UTILS_H
#define ACL_UTILS_HASH_UTILS_H
#include "acl/acl_base.h"
#include "utils/string_utils.h"
#include "utils/attr_utils.h"
#include "types/op_attr.h"
#include "types/acl_op_inner.h"
#include "common/log_inner.h"

namespace acl {
namespace hash_utils {

template <class T>
inline void HashCombine(size_t &seed, const T& v)
{
    const std::hash<T> hasher;
    seed ^= hasher(v) + 0x9E3779B9U + (seed << 6U) + (seed >> 2U);
}

inline void HashCombine(size_t &seed, const ge::DataType &v)
{
    const std::hash<int32_t> hasher;
    seed ^= hasher(static_cast<int32_t>(v)) + 0x9E3779B9U + (seed << 6U) + (seed >> 2U);
}

aclError GetTensorDescHash(const int32_t num, const aclTensorDesc *const descArr[], size_t &seed);

aclError GetAclOpHash(const AclOp &aclOp, const aclopAttr *const opAttr, const size_t attrDigest, size_t &seed);

aclError GetTensorDescHashDynamic(const int32_t num, const aclTensorDesc *const descArr[], size_t &seed);

aclError GetAclOpHashDynamic(const AclOp &aclOp, const aclopAttr *const opAttr,
                             const size_t attrDigest, size_t &seed, const uint64_t seq);

template<typename T>
bool CheckModelAndAttrMatch(const AclOp &aclOp, const aclopAttr *const opAttr, const T &entry)
{
    if (entry == nullptr) {
        return false;
    }
    if (aclOp.opType != entry->opType) {
        return false;
    }

    if (aclOp.numInputs != static_cast<int32_t>(entry->inputDescArr.size())) {
        return false;
    }

    for (size_t i = 0U; i < static_cast<size_t>(aclOp.numInputs); ++i) {
        if (!(entry->inputDescArr[i] == aclOp.inputDesc[i])) {
            return false;
        }
    }

    if (aclOp.numOutputs != static_cast<int32_t>(entry->outputDescArr.size())) {
        return false;
    }

    for (size_t i = 0U; i < static_cast<size_t>(aclOp.numOutputs); ++i) {
        if (!(entry->outputDescArr[i] == aclOp.outputDesc[i])) {
            return false;
        }
    }

    if (!attr_utils::OpAttrEquals(opAttr, &(entry->opAttr))) {
        return false;
    }
    return true;
}

template<typename T>
bool CheckModelAndAttrMatchDynamic(const AclOp &aclOp, const aclopAttr *const opAttr, const T &entry,
    const uint64_t seq)
{
    if (entry == nullptr) {
        return false;
    }
    if (aclOp.opType != entry->opType) {
        return false;
    }

    if (seq != entry->seq) {
        return false;
    }

    if (aclOp.numInputs != static_cast<int32_t>(entry->inputDescArr.size())) {
        return false;
    }

    for (size_t i = 0U; i < static_cast<size_t>(aclOp.numInputs); ++i) {
        if (!(entry->inputDescArr[i].IsSameTensor(aclOp.inputDesc[i]))) {
            return false;
        }
    }

    if (aclOp.numOutputs != static_cast<int32_t>(entry->outputDescArr.size())) {
        return false;
    }

    for (size_t i = 0U; i < static_cast<size_t>(aclOp.numOutputs); ++i) {
        if (!(entry->outputDescArr[i].IsSameTensor(aclOp.outputDesc[i]))) {
            return false;
        }
    }

    if (!attr_utils::OpAttrEquals(opAttr, &(entry->opAttr))) {
        return false;
    }
    return true;
}

} // namespace hash_utils
} // namespace acl

#endif // ACL_UTILS_HASH_UTILS_H
