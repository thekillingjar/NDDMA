/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "stub.h"
#include "runtime/stream.h"
#include "mmpa/mmpa_api.h"

using namespace ge;

void ForceUsingMmpa()
{
    (void)mmSleep(1);
}

RunContext CreateContext()
{
    RunContext context;
    context.dataMemSize = 10000;
    context.dataMemBase = (uint8_t *) (intptr_t) 1000;
    context.sessionId = 10000011011;

    return context;
}

void DestroyContext(RunContext& context) {}