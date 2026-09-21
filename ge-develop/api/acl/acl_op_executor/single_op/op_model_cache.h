/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ACL_OP_EXEC_OP_MODEL_CACHE_H_
#define ACL_OP_EXEC_OP_MODEL_CACHE_H_

#include <unordered_map>
#include <mutex>

#include "types/op_model.h"
#include "framework/runtime/model_v2_executor.h"
#include "framework/runtime/stream_executor.h"

namespace acl {
class OpModelCache {
public:
    OpModelCache() = default;

    ~OpModelCache() = default;

    aclError GetOpModel(const OpModelDef &modelDef, OpModel &operModel);

    std::shared_ptr<std::mutex> GetCacheMutex(const uint64_t id);

    aclError Add(const uint64_t opId, OpModel &operModel);

    aclError Delete(const OpModelDef &modelDef, const bool isDynamic);

    aclError CreateCachedExecutor(std::shared_ptr<gert::StreamExecutor> &streamExecutor, rtStream_t stream,
                                  const gert::ModelExecuteArg &arg, gert::ModelV2Executor *&executor);

    std::shared_ptr<gert::StreamExecutor> GetRT2Executor(const uint64_t id);

    aclError UpdateCachedExecutor(const uint64_t &id, const std::shared_ptr<gert::StreamExecutor> &executor);

    aclError UnloadCachedModelData(const uint64_t &id);

    aclError CleanCachedExecutor(rtStream_t stream);

    void CleanCachedModels() noexcept;

private:
    std::unordered_map<uint64_t, OpModel> cachedModels_;
    std::recursive_mutex mutex_;
};
} // namespace acl

#endif // ACL_OP_EXEC_OP_MODEL_CACHE_H_
