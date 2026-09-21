/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ACL_UTILS_ACL_OP_MAP_H
#define ACL_UTILS_ACL_OP_MAP_H

#include <map>
#include <mutex>
#include <vector>
#include <climits>
#include "types/acl_op_inner.h"
#include "utils/attr_utils.h"
#include "utils/hash_utils.h"
#include <chrono>
#include <unordered_map>

namespace acl {
template<typename T>
class AclOpMap {
public:
    aclError Insert(const AclOp &op, const T &entry, T &agingT, bool &isDeduplicate);
    aclError InsertDynamic(const AclOp &op, const T &entry, T &agingT, bool &isDeduplicate);

    aclError Get(const AclOp &op, T &entry, const bool needUpdateTimestamp = false);
    aclError GetDynamic(const AclOp &op, T &entry, const uint64_t seq, const bool needUpdateTimestamp = false);

    void SetMaxOpNum(const uint64_t maxNum)
    {
        maxOpNum = maxNum;
    }

private:
    aclError Aging(T &agingT);
    void Updatetimestamp(T &entry) const;
    bool Deduplicate(const AclOp &op, const aclopAttr *const attr, const T &entry,
                     const size_t seed, const bool isDynamic);
    aclError AddMemAndAging(const T &entry, T &agingT, const size_t seed);
    bool CheckValueRange(const AclOp &op, const T &entry) const;
    bool HasSameValueRange(const AclOp &op, const T &entry) const;
    using HashMap = std::unordered_map<size_t, std::vector<T>>;

    HashMap hashMap_;
    mutable std::mutex mutex_;
    uint64_t cnt{0U};
    uint64_t maxOpNum{DEFAULT_MAX_OPQUEUE_NUM};
};

template<typename T>
aclError AclOpMap<T>::Aging(T &agingT)
{
    // Find the min timestampe in hashMap
    uint64_t timestampMin = static_cast<uint64_t>(ULONG_MAX);
    typename HashMap::iterator itHashMapMin;
    typename std::vector<T>::iterator itVectorMin;

    bool found = false;
    for (auto hashMapIter = hashMap_.begin(); hashMapIter != hashMap_.end(); ++hashMapIter) {
        for (auto vecIter = hashMapIter->second.begin(); vecIter != hashMapIter->second.end(); ++vecIter) {
            if ((*vecIter)->timestamp < timestampMin) {
                timestampMin = (*vecIter)->timestamp;
                itHashMapMin = hashMapIter;
                itVectorMin = vecIter;
                found = true;
            }
        }
    }

    if (!found) {
        ACL_LOG_DEBUG("AclOpMap::Aging IN, can not find minimum value");
        return ACL_SUCCESS;
    }
    ACL_LOG_INFO("AclOpMap::Aging IN, type = %s, digest = %zu, timestamp is %lu",
        (*itVectorMin)->opType.c_str(), itHashMapMin->first, timestampMin);
    agingT = *itVectorMin;
    // aging model in hashMap with min timestamp
    (void)itHashMapMin->second.erase(itVectorMin);
    --cnt;
    // after aging model in hashMap, the model vector maybe empty, delete the seed in hashMap
    if (itHashMapMin->second.empty()) {
        ACL_LOG_INFO("AclOpMap::After delete model, hash map empty while seed is %zu. delete seed in HashMap",
            itHashMapMin->first);
        (void)hashMap_.erase(itHashMapMin);
    }

    return ACL_SUCCESS;
}

template<typename T>
void AclOpMap<T>::Updatetimestamp(T &entry) const
{
    if (entry->timestamp == static_cast<uint64_t>(ULLONG_MAX)) {
        return;
    }
    entry->timestamp = attr_utils::GetCurrentTimestamp();
}

template<typename T>
bool AclOpMap<T>::HasSameValueRange(const AclOp &op, const T &entry) const
{
    if ((static_cast<size_t>(op.numInputs) != entry->inputDescArr.size()) ||
        (static_cast<size_t>(op.numOutputs) != entry->outputDescArr.size())) {
        return false;
    }
    for (size_t i = 0U; i < entry->inputDescArr.size(); ++i) {
        ACL_LOG_INFO("the input [%zu] needs to check value range", i);
        if (!attr_utils::IsSameValueRange(entry->inputDescArr[i].valueRange, op.inputDesc[i]->valueRange)) {
            return false;
        }
    }
    for (size_t i = 0U; i < entry->outputDescArr.size(); ++i) {
        ACL_LOG_INFO("the output [%zu] needs to check value range", i);
        if (!attr_utils::IsSameValueRange(entry->outputDescArr[i].valueRange, op.outputDesc[i]->valueRange)) {
            return false;
        }
    }
    return true;
}

template<typename T>
bool AclOpMap<T>::Deduplicate(const AclOp &op, const aclopAttr *const attr, const T &entry,
                              const size_t seed, const bool isDynamic)
{
    const auto iter = hashMap_.find(seed);
    if (iter != hashMap_.end()) {
        for (auto vecIter = iter->second.begin(); vecIter != iter->second.end(); ++vecIter) {
            if (isDynamic) {
                if (!HasSameValueRange(op, *vecIter)) {
                    continue;
                }
                if (hash_utils::CheckModelAndAttrMatchDynamic(op, attr, *vecIter, entry->seq)) {
                    Updatetimestamp(*vecIter);
                    ACL_LOG_DEBUG("Find same dynamic op_desc in Hashmap, seed %zu", seed);
                    return true;
                }
            } else {
                if (hash_utils::CheckModelAndAttrMatch(op, attr, *vecIter)) {
                    Updatetimestamp(*vecIter);
                    ACL_LOG_DEBUG("Find same static op_desc in Hashmap, seed %zu", seed);
                    return true;
                }
            }
        }
    }
    return false;
}

template<typename T>
aclError AclOpMap<T>::AddMemAndAging(const T &entry, T &agingT, const size_t seed)
{
    hashMap_[seed].emplace_back(entry);
    ACL_LOG_INFO("AclOpMap::Insert op into HashMap success, seed = %zu", seed);

    ++cnt;
    if ((entry->timestamp == static_cast<uint64_t>(ULLONG_MAX)) || (cnt <= maxOpNum)) {
        ACL_LOG_INFO("AclOpMap::AddMemAndAging in, cnt is %lu, maxOpNum is %lu, no need aging", cnt, maxOpNum);
        return ACL_SUCCESS;
    }
    ACL_LOG_INFO("AclOpMap::time stamp is %lu, cnt is %lu, maxOpNum is %lu, start aging",
        entry->timestamp, cnt, maxOpNum);
    return Aging(agingT);
}

template<typename T>
bool AclOpMap<T>::CheckValueRange(const AclOp &op, const T &entry) const
{
    for (size_t i = 0U; i < entry->inputDescArr.size(); ++i) {
        if ((entry->inputDescArr[i].IsHostMemTensor()) && (!entry->inputDescArr[i].valueRange.empty())) {
            ACL_LOG_INFO("the input [%zu] needs to check value range", i);
            if (!attr_utils::ValueRangeCheck(entry->inputDescArr[i].valueRange,
                                             op.inputs[i], entry->inputDescArr[i].dataType)) {
                ACL_LOG_DEBUG("ValueRangeCheck input is not match");
                return false;
            }
        }
    }
    for (size_t i = 0U; i < entry->outputDescArr.size(); ++i) {
        if ((entry->outputDescArr[i].IsHostMemTensor()) && (!entry->outputDescArr[i].valueRange.empty())) {
            ACL_LOG_INFO("the output [%zu] needs to check value range", i);
            if (!attr_utils::ValueRangeCheck(entry->outputDescArr[i].valueRange,
                                             op.outputs[i], entry->outputDescArr[i].dataType)) {
                ACL_LOG_DEBUG("ValueRangeCheck output is not match");
                return false;
            }
        }
    }
    return true;
}

template<typename T>
aclError AclOpMap<T>::Insert(const AclOp &op, const T &entry, T &agingT, bool &isDeduplicate)
{
    ACL_LOG_DEBUG("AclOpMap::Insert IN, op = %s", op.DebugString().c_str());
    size_t digest = 0U;
    auto opAttr = op.opAttr;
    aclopAttr emptyAttr;
    if (opAttr != nullptr) {
        if (!attr_utils::SaveConstToAttr(op, const_cast<aclopAttr *>(opAttr))) {
            ACL_LOG_ERROR("[Save][ConstData]save const data buffer to attr fail");
            return ACL_ERROR_FAILURE;
        }
        digest = attr_utils::AttrMapToDigest(opAttr->Attrs());
    } else {
        if (!attr_utils::SaveConstToAttr(op, &emptyAttr)) {
            ACL_LOG_ERROR("[Save][ConstData]save const data buffer to attr fail");
            return ACL_ERROR_FAILURE;
        }
        digest = attr_utils::AttrMapToDigest(emptyAttr.Attrs());
        opAttr = &emptyAttr;
    }
    size_t seed = 0U;
    if (hash_utils::GetAclOpHash(op, opAttr, digest, seed) != ACL_SUCCESS) {
        ACL_LOG_ERROR("[Check][GetAclOpHash]GetAclOpHash failed, seed = %zu, op = %s",
            seed, op.DebugString().c_str());
        return ACL_ERROR_FAILURE;
    }
    // Lock
    {
        const std::lock_guard<std::mutex> lk(mutex_);
        isDeduplicate = Deduplicate(op, opAttr, entry, seed, false);
        if (!isDeduplicate) {
            ACL_REQUIRES_OK(AddMemAndAging(entry, agingT, seed));
        }
        ACL_LOG_INFO("AclOpMap::Insert success, seed = %zu, op = %s", seed, op.DebugString().c_str());
    }
    return ACL_SUCCESS;
}

template<typename T>
aclError AclOpMap<T>::InsertDynamic(const AclOp &op, const T &entry, T &agingT, bool &isDeduplicate)
{
    ACL_LOG_DEBUG("AclOpMap::Insert IN, op = %s", op.DebugString().c_str());
    size_t digest = 0U;
    auto opAttr = op.opAttr;
    aclopAttr emptyAttr;
    if (opAttr != nullptr) {
        if (!attr_utils::SaveConstToAttr(op, const_cast<aclopAttr *>(opAttr))) {
            ACL_LOG_ERROR("[Save][ConstData]save const data buffer to attr fail");
            return ACL_ERROR_FAILURE;
        }
        digest = attr_utils::AttrMapToDigest(opAttr->Attrs());
    } else {
        if (!attr_utils::SaveConstToAttr(op, &emptyAttr)) {
            ACL_LOG_ERROR("[Save][ConstData]save const data buffer to attr fail");
            return ACL_ERROR_FAILURE;
        }
        digest = attr_utils::AttrMapToDigest(emptyAttr.Attrs());
        opAttr = &emptyAttr;
    }
    size_t seed = 0U;
    if (hash_utils::GetAclOpHashDynamic(op, opAttr, digest, seed, entry->seq) != ACL_SUCCESS) {
        ACL_LOG_ERROR("[Check][GetAclOpHash]GetAclOpHash failed, seed = %zu, op = %s",
            seed, op.DebugString().c_str());
        return ACL_ERROR_FAILURE;
    }
    // Lock
    {
        const std::lock_guard<std::mutex> lk(mutex_);
        isDeduplicate = Deduplicate(op, opAttr, entry, seed, true);
        if (!isDeduplicate) {
            ACL_REQUIRES_OK(AddMemAndAging(entry, agingT, seed));
        }
        ACL_LOG_INFO("AclOpMap::Insert success, seed = %zu, op = %s", seed, op.DebugString().c_str());
    }
    return ACL_SUCCESS;
}

template<typename T>
aclError AclOpMap<T>::Get(const AclOp &op, T &entry, const bool needUpdateTimestamp)
{
    auto opAttr = op.opAttr;
    size_t digest = 0U;
    aclopAttr emptyAttr;
    if (opAttr != nullptr) {
        digest = op.opAttr->GetDigest();
        if (!attr_utils::SaveConstToAttr(op, const_cast<aclopAttr *>(opAttr))) {
            ACL_LOG_ERROR("[Save][ConstData]save const data buffer to attr fail");
            return ACL_ERROR_INVALID_PARAM;
        }
    } else {
        if (!attr_utils::SaveConstToAttr(op, &emptyAttr)) {
            ACL_LOG_ERROR("[Save][ConstData]save const data buffer to attr fail");
            return ACL_ERROR_INVALID_PARAM;
        }
        opAttr = &emptyAttr;
    }
    size_t seed = 0U;
    ACL_REQUIRES_OK(hash_utils::GetAclOpHash(op, opAttr, digest, seed));
    const std::lock_guard<std::mutex> lk(mutex_);
    const auto iter = hashMap_.find(seed);
    if (iter == hashMap_.end()) {
        return ACL_ERROR_OP_NOT_FOUND;
    }
    // Cyclically traversing the model size >= 1 with same seed
    for (auto modelVecIter = iter->second.begin(); modelVecIter != iter->second.end(); ++ modelVecIter) {
        if (CheckValueRange(op, *modelVecIter)) {
            if (hash_utils::CheckModelAndAttrMatch(op, opAttr, *modelVecIter)) {
                ACL_LOG_INFO("Get aclOp from aclOpMap success! seed = %zu, aclOp = %s", seed,
                    op.DebugString().c_str());
                entry = *modelVecIter;
                if (needUpdateTimestamp) {
                    Updatetimestamp(*modelVecIter);
                }
                return ACL_SUCCESS;
            }
        }
    }
    ACL_LOG_DEBUG("Get aclOp from aclOpMap failed due to CheckValueRange failed! seed = %zu, aclOp = %s",
        seed, op.DebugString().c_str());
    return ACL_ERROR_OP_NOT_FOUND;
}

template<typename T>
aclError AclOpMap<T>::GetDynamic(const AclOp &op, T &entry, const uint64_t seq, const bool needUpdateTimestamp)
{
    auto opAttr = op.opAttr;
    size_t digest = 0U;
    aclopAttr emptyAttr;
    if (opAttr != nullptr) {
        digest = op.opAttr->GetDigest();
        if (!attr_utils::SaveConstToAttr(op, const_cast<aclopAttr *>(opAttr))) {
            ACL_LOG_ERROR("[Save][ConstData]save const data buffer to attr fail");
            return ACL_ERROR_INVALID_PARAM;
        }
    } else {
        if (!attr_utils::SaveConstToAttr(op, &emptyAttr)) {
            ACL_LOG_ERROR("[Save][ConstData]save const data buffer to attr fail");
            return ACL_ERROR_INVALID_PARAM;
        }
        opAttr = &emptyAttr;
    }
    size_t seed = 0U;
    ACL_REQUIRES_OK(hash_utils::GetAclOpHashDynamic(op, opAttr, digest, seed, seq));
    const std::lock_guard<std::mutex> lk(mutex_);
    const auto iter = hashMap_.find(seed);
    if (iter == hashMap_.end()) {
        return ACL_ERROR_OP_NOT_FOUND;
    }
    // Cyclically traversing the model size >= 1 with same seed
    for (auto modelVecIter = iter->second.begin(); modelVecIter != iter->second.end(); ++ modelVecIter) {
        if (CheckValueRange(op, *modelVecIter)) {
            if (hash_utils::CheckModelAndAttrMatchDynamic(op, opAttr, *modelVecIter, seq)) {
                ACL_LOG_INFO("Get aclOp from aclOpMap success! seed = %zu, aclOp = %s", seed,
                    op.DebugString().c_str());
                entry = *modelVecIter;
                if (needUpdateTimestamp) {
                    Updatetimestamp(*modelVecIter);
                }
                return ACL_SUCCESS;
            }
        }
    }
    ACL_LOG_DEBUG("Get aclOp from aclOpMap failed due to CheckValueRange failed! seed = %zu, aclOp = %s",
        seed, op.DebugString().c_str());
    return ACL_ERROR_OP_NOT_FOUND;
}

} // namespace acl

#endif // ACL_UTILS_ACL_OP_MAP_H
