/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ops_kernel_info_store_stub.h"
#include <mutex>
#include <unistd.h>

#include "config/ops_json_file.h"

using namespace ge;
using namespace std;

namespace aicpu {
KernelInfoPtr OpskernelInfoStoreStub::instance_ = nullptr;

KernelInfoPtr OpskernelInfoStoreStub::Instance()
{
    static once_flag flag;
    call_once(flag, [&]() {
        instance_.reset(new (nothrow) OpskernelInfoStoreStub);
    });
    return instance_;
}

bool OpskernelInfoStoreStub::ReadOpInfoFromJsonFile()
{
    char *buffer;
    buffer = getcwd(NULL, 0);
    printf("pwd====%s\n", buffer);

    string jsonPath(buffer);
    free(buffer);
    jsonPath += "/tests/engines/cpueng/stub/config/tf_kernel.json";
    bool ret = OpsJsonFile::Instance().ParseUnderPath(jsonPath, op_info_json_file_);
    return ret;
}

// FACTORY_KERNELINFO_CLASS_KEY(OpskernelInfoStoreStub, "stub")
}
