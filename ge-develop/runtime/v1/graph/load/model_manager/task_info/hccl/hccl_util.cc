/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <dlfcn.h>
#include "common/debug/ge_log.h"
#include "hccl_util.h"

namespace ge {

HcclDllHcomMgr &HcclDllHcomMgr::GetInstance()
{
    static HcclDllHcomMgr mgr;
    return mgr;
}

HcclDllHcomMgr::HcclDllHcomMgr()
{
    GELOGI("hccl_HcomGetCcuTaskInfo_func load start.");
    dll_handle = dlopen("libhccl_v2.so", RTLD_LAZY);
    if (dll_handle == nullptr) {
        GELOGI("hccl_HcomGetCcuTaskInfo_func load fail: libhccl_v2.so no found");
        return;
    }
    hccl_HcomGetCcuTaskInfo_func = (HcclResult(*)(const std::string &group, void *tilingData, void *ccuTaskGroup))dlsym(dll_handle, "HcomGetCcuTaskInfo");
}

HcclResult HcclDllHcomMgr::HcomGetCcuTaskInfoFunc(const std::string &group, void *tilingData, void *ccuTaskGroup)
{
    GELOGI("[HcclDllHcomMgr][HcomGetCcuTaskInfoFunc] group[%s] tilingData[%p] ccuTaskGroup[%p]",group.c_str(),
            tilingData, ccuTaskGroup);
    if(hccl_HcomGetCcuTaskInfo_func == nullptr)
    {
        GELOGI("hccl_HcomGetCcuTaskInfo_func no found");
        return HCCL_E_PTR;
    }
    return hccl_HcomGetCcuTaskInfo_func(group, tilingData, ccuTaskGroup);
}

}