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
#include "common/common_inner.h"

namespace acl {
GeFinalizeCallback aclGeFinalizeCallback = nullptr;
GeFinalizeCallback GetGeFinalizeCallback()
{
    return aclGeFinalizeCallback;
}

void SetGeFinalizeCallback(const GeFinalizeCallback func)
{
    aclGeFinalizeCallback = func;
}

aclError AclOpCompilerFinalizeCallbackFunc(void *userData)
{
    (void)userData;
    ACL_LOG_INFO("start to enter AclOpCompilerFinalizeCallbackFunc");
    auto geFinalizeCbFunc = acl::GetGeFinalizeCallback();
    if (geFinalizeCbFunc != nullptr) {
        const auto ret = geFinalizeCbFunc();
        if (ret != ACL_SUCCESS) {
            ACL_LOG_ERROR("[Finalize][Ge]ge finalize failed, errorCode = %d", static_cast<int32_t>(ret));
        }
    }
    return ACL_SUCCESS;
}
__attribute__((constructor)) aclError RegAclOpCompilerFinalizeCallback()
{
    return aclFinalizeCallbackRegister(ACL_REG_TYPE_ACL_OP_COMPILER, AclOpCompilerFinalizeCallbackFunc, nullptr);
}
__attribute__((destructor)) aclError UnRegAclOpCompilerFinalizeCallback()
{
    return aclFinalizeCallbackUnRegister(ACL_REG_TYPE_ACL_OP_COMPILER, AclOpCompilerFinalizeCallbackFunc);
}
}