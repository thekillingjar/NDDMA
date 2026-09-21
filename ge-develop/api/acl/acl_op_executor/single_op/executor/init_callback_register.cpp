/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "acl/acl_rt.h"
#include "error_codes_inner.h"
#include "common/log_inner.h"
#include "single_op/acl_op_resource_manager.h"

namespace {
void HandleReleaseSourceByStream(aclrtStream stream, aclrtStreamState state, void *args)
{
    acl::AclOpResourceManager::GetInstance().HandleReleaseSourceByStream(stream, state, args);
}
}

namespace acl {
// --------------------------------initialize----------------------------------------------------------------------
aclError AclOpExecutorInitCallbackFunc(const char *configBuffer, size_t bufferSize, void *userData)
{
    (void) userData;
    ACL_LOG_INFO("start to enter AclOpExecutorInitCallbackFunc");

    if ((configBuffer != nullptr) && (bufferSize != 0UL)) {
      // config max_opqueue_num
        ACL_LOG_INFO("set max_opqueue_num in aclInit");
        auto ret = acl::AclOpResourceManager::GetInstance().HandleMaxOpQueueConfig(configBuffer);
        if (ret != ACL_SUCCESS) {
            ACL_LOG_ERROR("[Process][QueueConfig]process HandleMaxOpQueueConfig failed");
            return ret;
        }
        ACL_LOG_INFO("set HandleMaxOpQueueConfig success");
    }
    return ACL_SUCCESS;
}
__attribute__((constructor)) aclError RegAclOpExecutorInitCallback()
{
    return aclInitCallbackRegister(ACL_REG_TYPE_ACL_OP_EXECUTOR, AclOpExecutorInitCallbackFunc, nullptr);
}
__attribute__((destructor)) aclError UnRegAclOpExecutorInitCallback()
{
    return aclInitCallbackUnRegister(ACL_REG_TYPE_ACL_OP_EXECUTOR, AclOpExecutorInitCallbackFunc);
}

aclError AclOpResourceInitCallbackFunc(const char *configBuffer, size_t bufferSize, void *userData)
{
    (void)configBuffer;
    (void)bufferSize;
    (void)userData;
    ACL_LOG_INFO("start to enter AclOpResourceInitCallbackFunc");
    // register ge release function by stream
    auto aclErr = aclrtRegStreamStateCallback("ACL_MODULE_STREAM_OP", &HandleReleaseSourceByStream, nullptr);
    if (aclErr != ACL_SUCCESS) {
        ACL_LOG_ERROR("register release function by stream to runtime failed, ret:%d", aclErr);
        return aclErr;
    }
    return ACL_SUCCESS;
}
__attribute__((constructor)) aclError RegResourceInitCallback()
{
    return aclInitCallbackRegister(ACL_REG_TYPE_OTHER, AclOpResourceInitCallbackFunc, nullptr);
}
__attribute__((destructor)) aclError UnRegResourceInitCallback()
{
    return aclInitCallbackUnRegister(ACL_REG_TYPE_OTHER, AclOpResourceInitCallbackFunc);
}

// --------------------------- finalize --------------------------------------------------
aclError AclOpResourceFinalizeCallbackFunc(void *userData)
{
    (void)userData;
    ACL_LOG_INFO("start to enter AclOpResourceFinalizeCallbackFunc");
    // unregister ge release function by stream
    auto aclErr = aclrtRegStreamStateCallback("ACL_MODULE_STREAM_OP", nullptr, nullptr);
    if (aclErr != ACL_SUCCESS) {
        ACL_LOG_ERROR("unregister release function by stream to runtime failed, ret:%d", aclErr);
        return aclErr;
    }
    return ACL_SUCCESS;
}
__attribute__((constructor)) aclError RegResourceFinalizeCallback()
{
    return aclFinalizeCallbackRegister(ACL_REG_TYPE_OTHER, AclOpResourceFinalizeCallbackFunc, nullptr);
}
__attribute__((destructor)) aclError UnRegResourceFinalizeCallback()
{
    return aclFinalizeCallbackUnRegister(ACL_REG_TYPE_OTHER, AclOpResourceFinalizeCallbackFunc);
}
}