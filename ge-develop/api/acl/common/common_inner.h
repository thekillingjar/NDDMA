/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ACL_COMMON_INNER_API_H_
#define ACL_COMMON_INNER_API_H_

#include <map>
#include "acl/acl_base.h"
#include "acl/acl_rt.h"
#include "log_inner.h"

namespace acl {
    constexpr size_t SOC_VERSION_LEN = 128U;

    using GeFinalizeCallback = aclError (*)();

    ACL_FUNC_VISIBILITY void SetGeFinalizeCallback(const GeFinalizeCallback func);

    ACL_FUNC_VISIBILITY aclError aclMallocMemInner(void **devPtr,
                                                   const size_t size,
                                                   bool isPadding,
                                                   const aclrtMemMallocPolicy policy,
                                                   const uint16_t moduleId);

    bool GetAclInitFlag();

    uint64_t GetAclInitRefCount();

    void SetCastHasTruncateAttr(const bool hasTruncate);

    bool GetIfCastHasTruncateAttr();

    ACL_FUNC_VISIBILITY void SetGlobalCompileFlag(const int32_t flag);

    ACL_FUNC_VISIBILITY int32_t GetGlobalCompileFlag();

    ACL_FUNC_VISIBILITY void SetGlobalJitCompileFlag(const int32_t flag);

    ACL_FUNC_VISIBILITY int32_t GetGlobalJitCompileFlag();

    void aclGetMsgCallback(const char_t *msg, uint32_t len);

    int32_t UpdateOpSystemRunCfg(void *cfgAddr, uint32_t cfgLen);

    aclError HandleDefaultDeviceAndStackSize(const char_t *const configPath);
}

#endif // ACL_COMMON_INNER_API_H_
