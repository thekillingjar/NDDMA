/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "op_model_cache.h"

#include "framework/common/util.h"
#include "executor/ge_executor.h"

namespace acl {
aclError OpModelCache::GetOpModel(const OpModelDef &modelDef, OpModel &operModel)
{
    const auto &key = modelDef.opModelId;
    {
        const std::lock_guard<std::recursive_mutex> locker(mutex_);
        const auto iter = cachedModels_.find(key);
        if (iter != cachedModels_.end()) {
            operModel = iter->second;
            return ACL_SUCCESS;
        }
    }
    ACL_LOG_INNER_ERROR("[Get][OpModel]GetOpModel fail, modelPath = %lu", modelDef.opModelId);
    return ACL_ERROR_FAILURE;
}

std::shared_ptr<std::mutex> OpModelCache::GetCacheMutex(const uint64_t id)
{
    const std::lock_guard<std::recursive_mutex> locker(mutex_);
    const auto &iter = cachedModels_.find(id);
    if (iter == cachedModels_.end()) {
        ACL_LOG_INNER_ERROR("get mutex failed, key is %lu", id);
        return nullptr;
    }
    return iter->second.mtx;
}

aclError OpModelCache::Add(const uint64_t opId, OpModel &operModel)
{
    ACL_LOG_INFO("start to execute OpModelCache::Add, opId = %lu", opId);
    operModel.opModelId = opId;
    const std::lock_guard<std::recursive_mutex> locker(mutex_);
    auto mtx = std::make_shared<std::mutex>();
    ACL_REQUIRES_NOT_NULL(mtx);
    auto ret = cachedModels_.insert(std::make_pair(opId, operModel));
    ACL_REQUIRES_TRUE(ret.second, ACL_ERROR_FAILURE, "ACL inner Error: OpModelCache Add has same opId!");
    ret.first->second.mtx = std::move(mtx);
    return ACL_SUCCESS;
}

aclError OpModelCache::Delete(const OpModelDef &modelDef, const bool isDynamic)
{
    ACL_LOG_INFO("start to execute OpModelCache::Delete, modelId = %lu", modelDef.opModelId);
    const uint64_t key = modelDef.opModelId;
    const std::lock_guard<std::recursive_mutex> locker(mutex_);
    const auto it = cachedModels_.find(key);
    if (it != cachedModels_.end()) {
        const uint64_t opId = it->second.opModelId;
        (void)cachedModels_.erase(it);
        ACL_LOG_INFO("start to unload single op resource %lu", opId);
        if (isDynamic) {
            return static_cast<int32_t>(ge::GeExecutor::UnloadDynamicSingleOp(opId));
        } else {
            return static_cast<int32_t>(ge::GeExecutor::UnloadSingleOp(opId));
        }
    }
    return ACL_SUCCESS;
}

aclError OpModelCache::CreateCachedExecutor(std::shared_ptr<gert::StreamExecutor> &streamExecutor, rtStream_t stream,
                                            const gert::ModelExecuteArg &arg, gert::ModelV2Executor *&executor)
{
    const std::lock_guard<std::recursive_mutex> locker(mutex_);
    executor = streamExecutor->GetOrCreateLoaded(stream, arg);
    ACL_REQUIRES_NOT_NULL(executor);
    return ACL_SUCCESS;
}

std::shared_ptr<gert::StreamExecutor> OpModelCache::GetRT2Executor(const uint64_t id)
{
    const std::lock_guard<std::recursive_mutex> locker(mutex_);
    const auto &iter = cachedModels_.find(id);
    if (iter == cachedModels_.end()) {
        ACL_LOG_INNER_ERROR("GetRT2Executor failed, key is %lu", id);
        return nullptr;
    }
    return iter->second.executor;
}

aclError OpModelCache::UpdateCachedExecutor(const uint64_t &id,
                                            const std::shared_ptr<gert::StreamExecutor> &executor)
{
    const std::lock_guard<std::recursive_mutex> locker(mutex_);
    const auto &iter = cachedModels_.find(id);
    if (iter == cachedModels_.end()) {
        ACL_LOG_INNER_ERROR("search model cache faild when update runtime v2 stream executor, key is %lu",
                            id);
        return ACL_ERROR_FAILURE;
    }
    iter->second.executor = executor;
    return ACL_SUCCESS;
}

aclError OpModelCache::CleanCachedExecutor(rtStream_t stream)
{
    const std::lock_guard<std::recursive_mutex> locker(mutex_);
    for (auto it = cachedModels_.begin(); it != cachedModels_.end(); ++it) {
        if (it->second.executor != nullptr) {
            (void)it->second.executor->Erase(stream);
        }
    }
    return ACL_SUCCESS;
}

aclError OpModelCache::UnloadCachedModelData(const uint64_t &id)
{
    const std::lock_guard<std::recursive_mutex> locker(mutex_);
    const auto &iter = cachedModels_.find(id);
    if (iter == cachedModels_.end()) {
        ACL_LOG_INNER_ERROR("search model cache faild when remove model data, key is %lu", id);
        return ACL_ERROR_FAILURE;
    }
    iter->second.data = nullptr;
    iter->second.size = 0U;
    return ACL_SUCCESS;
}

void OpModelCache::CleanCachedModels() noexcept
{
    cachedModels_.clear();
}
} // namespace acl
