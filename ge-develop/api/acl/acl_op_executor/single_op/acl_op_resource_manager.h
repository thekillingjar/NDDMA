/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ASCEND_ACL_OP_RESOURCE_MANAGER_H
#define ASCEND_ACL_OP_RESOURCE_MANAGER_H

#include <map>
#include <mutex>
#include <memory>
#include <string>
#include <vector>
#include <unordered_map>
#include <atomic>
#include "acl/acl_base.h"
#include "graph/ge_attr_value.h"
#include "mmpa/mmpa_api.h"
#include "types/op_model.h"
#include "types/acl_op_inner.h"
#include "single_op/op_model_cache.h"
#include "single_op/compile/op_compiler.h"
#include "utils/acl_op_map.h"
#include "utils/string_utils.h"
#include "framework/runtime/mem_allocator.h"
#include "framework/runtime/model_v2_executor.h"
#include "framework/runtime/stream_executor.h"
#include "ge/ge_allocator.h"

namespace acl {
class ACL_FUNC_VISIBILITY AclOpResourceManager {
public:
    ~AclOpResourceManager();

    static AclOpResourceManager &GetInstance()
    {
        static AclOpResourceManager instance;
        return instance;
    }

    // executor
    bool IsRuntimeV2Enable(bool isModel) const
    {
        return isModel ? enableRuntimeV2ForModel_ : enableRuntimeV2ForSingleOp_;
    }

    static void *GetKeyByStreamOrDefaultStream(const aclrtStream stream);

    // op_model
    aclError LoadAllModels(const std::string &modelDir);

    aclError LoadModelFromMem(const void *const model,
                              const size_t modelSize,
                              const bool isStatic = false);

    aclError LoadModelFromSharedMem(const std::shared_ptr<void> &model,
                                    const size_t modelSize,
                                    AclOp *aclOp,
                                    const bool isStatic = false);

    aclError GetOpModel(AclOp &aclOp);

    aclError MatchOpModel(const AclOp &aclOp, OpModel &opModel, bool &isDynamic);

    aclError UpdateRT2Executor(const uint64_t id, const std::shared_ptr<gert::StreamExecutor> &executor);

    std::shared_ptr<std::mutex> GetCacheMutex(const uint64_t id);

    std::shared_ptr<gert::StreamExecutor> GetRT2Executor(const uint64_t id);

    aclError UnloadModelData(const uint64_t id);

    aclError CleanRT2Executor(rtStream_t stream);

    aclError HandleMaxOpQueueConfig(const char_t *const configBuffer);

    aclError SetHostMemToConst(const AclOp &aclopHostMemToConst, bool &isExistConst) const;

    static aclError SetTensorConst(aclTensorDesc *const desc, const aclDataBuffer *const dataBuffer);

    void SetMaxOpNum(const uint64_t maxOpNum);

    void HandleReleaseSourceByStream(aclrtStream stream, aclrtStreamState state, void *args);
    void HandleReleaseSourceByDevice(int32_t deviceId, aclrtDeviceState state, void *args) const;
    void UpdateAllocatorsForOp(const void * const cacheKey, std::shared_ptr<gert::Allocators> &allocators);

    aclError CreateRT2Executor(std::shared_ptr<gert::StreamExecutor> &streamExecutor, rtStream_t stream,
                               const gert::ModelExecuteArg &arg, gert::ModelV2Executor *&executor);

private:
    AclOpResourceManager();

    // op_model
    using ModelMap = AclOpMap<std::shared_ptr<OpModelDef>>;

    aclError RegisterModel(OpModelDef &&modelConfig,
                           ModelMap &opModelDefs,
                           const bool isDynamic,
                           bool &isRegistered,
                           const bool isStaticRegister = true);

    aclError ReadModelDefs(const std::string &configPath,
                           std::vector<OpModelDef> &configList);

    aclError BuildOpModel(AclOp &aclOp);

    static bool OmFileFilterFn(const std::string &fileName);

    static bool IsDynamicOpModel(const OpModelDef &modelDef);

    static bool IsDynamicOpModel(const AclOp &aclOp);

    aclError MatchStaticOpModel(const AclOp &aclOp, OpModel &opModel, bool &isDynamic, bool &isNeedMatchDymaic);
    aclError MatchDynamicOpModel(const AclOp &aclOp, OpModel &opModel, bool &isDynamic);

    void CleanAllocatorsForOp(const void * const cacheKey);

    void GetRuntimeV2Env();
private:
    bool enableRuntimeV2ForModel_ = true;
    bool enableRuntimeV2ForSingleOp_ = true;

    // op_model
    ModelMap opModels_;
    OpModelCache modelCache_;
    std::recursive_mutex streamAllocatorMutex_;
    std::map<const void *, std::shared_ptr<gert::Allocators>> streamAllocators_; // use for extend life-cycle
};
}

#endif // ASCEND_ACL_RESOURCE_MANAGER_H
