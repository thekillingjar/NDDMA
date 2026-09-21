/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "acl_stub.h"
#include "adx_datadump_server.h"
#include "adump_api.h"
#include "awatchdog.h"

extern "C" int AdxDataDumpServerInit();
extern "C" int AdxDataDumpServerUnInit();

int aclStub::AdxDataDumpServerInit()
{
    return 0;
}

int aclStub::AdxDataDumpServerUnInit()
{
    return 0;
}

int32_t aclStub::AdumpSetDumpConfig(Adx::DumpType dumpType, const Adx::DumpConfig &dumpConfig)
{
    return 0;
}

bool aclStub::AdumpIsDumpEnable(Adx::DumpType dumpType)
{
    return true;
}

int AdxDataDumpServerInit()
{
    return MockFunctionTest::aclStubInstance().AdxDataDumpServerInit();
}

int AdxDataDumpServerUnInit()
{
    return MockFunctionTest::aclStubInstance().AdxDataDumpServerUnInit();
}

AwdHandle AwdCreateThreadWatchdog(uint32_t moduleId, uint32_t timeout, AwatchdogCallbackFunc callback)
{
    (void)moduleId;
    (void)timeout;
    (void)callback;
    AwdThreadWatchdog *hdl = new AwdThreadWatchdog();
    return (AwdHandle)hdl;
}

void AwdDestroyThreadWatchdog(AwdHandle handle)
{
    AwdThreadWatchdog *dog = (AwdThreadWatchdog *)handle;
    delete dog;
}

namespace Adx {
    int32_t AdumpSetDumpConfig(DumpType dumpType, const DumpConfig &dumpConfig)
    {
        return MockFunctionTest::aclStubInstance().AdumpSetDumpConfig(dumpType, dumpConfig);
    }

    bool AdumpIsDumpEnable(DumpType dumpType)
    {
        return MockFunctionTest::aclStubInstance().AdumpIsDumpEnable(dumpType);
    }
}

