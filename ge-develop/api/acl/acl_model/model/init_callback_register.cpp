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
#include "executor/ge_executor.h"
#include "acl_resource_manager.h"
#include "error_codes_inner.h"
#include "json_parser.h"
#include "log_inner.h"

namespace {
void HandleReleaseSourceByDevice(int32_t deviceId, aclrtDeviceState state, void *args)
{
    acl::AclResourceManager::GetInstance().HandleReleaseSourceByDevice(deviceId, state, args);
}

void HandleReleaseSourceByStream(aclrtStream stream, aclrtStreamState state, void *args)
{
    acl::AclResourceManager::GetInstance().HandleReleaseSourceByStream(stream, state, args);
}
}

namespace acl {
// --------------------------------initialize----------------------------------------------------------------------
aclError AclMdlInitCallbackFunc(const char *configStr, size_t len, void *userData)
{
    (void)configStr;
    (void)len;
    (void)userData;
    ACL_LOG_INFO("start to enter AclMdlInitCallbackFunc");
    // init GeExecutor
    ge::GeExecutor executor;
    ACL_LOG_INFO("call ge interface executor.Initialize");
    auto geRet = executor.Initialize();
    ACL_REQUIRES_CALL_GE_OK(geRet, "[Init][Geexecutor]init ge executor failed, ge errorCode = %u", geRet);
    return ACL_SUCCESS;
}
__attribute__((constructor)) aclError RegAclMdlInitCallback()
{
    return aclInitCallbackRegister(ACL_REG_TYPE_ACL_MODEL, AclMdlInitCallbackFunc, nullptr);
}
__attribute__((destructor)) aclError UnRegAclMdlInitCallback()
{
    return aclInitCallbackUnRegister(ACL_REG_TYPE_ACL_MODEL, AclMdlInitCallbackFunc);
}

aclError ResourceInitCallbackFunc(const char *configStr, size_t len, void *userData)
{
    (void)configStr;
    (void)len;
    (void)userData;
    ACL_LOG_INFO("start to enter ResourceInitCallbackFunc");
    // register ge release function by stream
    auto aclErr = aclrtRegStreamStateCallback("ACL_MODULE_STREAM_MODEL", &HandleReleaseSourceByStream, nullptr);
    if (aclErr != ACL_ERROR_NONE) {
        ACL_LOG_ERROR("register release function by stream to runtime failed, ret:%d", aclErr);
        return aclErr;
    }

    // register ge release function by device
    aclErr = aclrtRegDeviceStateCallback("ACL_MODULE_DEVICE", &HandleReleaseSourceByDevice, nullptr);
    if (aclErr != ACL_ERROR_NONE) {
        ACL_LOG_ERROR("register release function by device to runtime failed, ret:%d", aclErr);
        return aclErr;
    }
    return ACL_SUCCESS;
}
__attribute__((constructor)) aclError RegResourceInitCallback()
{
    return aclInitCallbackRegister(ACL_REG_TYPE_OTHER, ResourceInitCallbackFunc, nullptr);
}
__attribute__((destructor)) aclError UnRegResourceInitCallback()
{
    return aclInitCallbackUnRegister(ACL_REG_TYPE_OTHER, ResourceInitCallbackFunc);
}

// --------------------------------finalize----------------------------------------------------------------------
aclError AclMdlFinalizeCallbackFunc(void *userData)
{
    (void)userData;
    ACL_LOG_INFO("start to enter AclMdlFinalizeCallbackFunc");
    // Finalize GeExecutor
    ge::GeExecutor executor;
    const ge::Status geRet = executor.Finalize();
    if (geRet != ge::SUCCESS) {
        ACL_LOG_ERROR("[Finalize][Ge]finalize ge executor failed, ge errorCode = %u", geRet);
        return ACL_GET_ERRCODE_GE(geRet);
    }
    return ACL_SUCCESS;
}
__attribute__((constructor)) aclError RegAclMdlFinalizeCallback()
{
    return aclFinalizeCallbackRegister(ACL_REG_TYPE_ACL_MODEL, AclMdlFinalizeCallbackFunc, nullptr);
}
__attribute__((destructor)) aclError UnRegAclMdlFinalizeCallback()
{
    return aclFinalizeCallbackUnRegister(ACL_REG_TYPE_ACL_MODEL, AclMdlFinalizeCallbackFunc);
}

aclError ResourceFinalizeCallbackFunc(void *userData)
{
    (void)userData;
    ACL_LOG_INFO("start to enter ResourceFinalizeCallbackFunc");
    // unregister ge release function by stream
    auto aclErr = aclrtRegStreamStateCallback("ACL_MODULE_STREAM_MODEL", nullptr, nullptr);
    if (aclErr != ACL_ERROR_NONE) {
        ACL_LOG_ERROR("unregister release function by stream to runtime failed, ret:%d", aclErr);
        return aclErr;
    }

    // unregister ge release function by device
    aclErr = aclrtRegDeviceStateCallback("ACL_MODULE_DEVICE", nullptr, nullptr);
    if (aclErr != ACL_ERROR_NONE) {
        ACL_LOG_ERROR("unregister release function by device to runtime failed, ret:%d", aclErr);
        return aclErr;
    }
    return ACL_SUCCESS;
}
__attribute__((constructor)) aclError RegResourceFinalizeCallback()
{
    return aclFinalizeCallbackRegister(ACL_REG_TYPE_OTHER, ResourceFinalizeCallbackFunc, nullptr);
}
__attribute__((destructor)) aclError UnRegResourceFinalizeCallback()
{
    return aclFinalizeCallbackUnRegister(ACL_REG_TYPE_OTHER, ResourceFinalizeCallbackFunc);
}
}