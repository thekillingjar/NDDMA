/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ACL_MODEL_CONFIG_API_H_
#define ACL_MODEL_CONFIG_API_H_
#include <stdbool.h>
#include "acl/acl_mdl.h"
#include "runtime/mem.h"
#define ACL_MDL_LOAD_TYPE_SIZET_BIT (0x1 << ACL_MDL_LOAD_TYPE_SIZET)
#define ACL_MDL_PATH_PTR_BIT        (0x1 << ACL_MDL_PATH_PTR)
#define ACL_MDL_MEM_ADDR_PTR_BIT    (0x1 << ACL_MDL_MEM_ADDR_PTR)
#define ACL_MDL_MEM_SIZET_BIT       (0x1 << ACL_MDL_MEM_SIZET)

bool CheckMdlConfigHandle(const aclmdlConfigHandle *const handle);
aclError GetMemTypeFromPolicy(aclrtMemMallocPolicy policy, rtMemType_t *type);
#endif // ACL_MODEL_CONFIG_API_H_
