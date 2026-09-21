/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef __HCCL_DLL_HCOM_MGR_H__
#define __HCCL_DLL_HCOM_MGR_H__

#include <functional>
#include <hccl/hccl_types.h>
#include "hccl/base.h"

namespace ge {

class HcclDllHcomMgr
{
public:
    static HcclDllHcomMgr &GetInstance();
    HcclResult HcomGetCcuTaskInfoFunc(const std::string &group, void *tilingData, void *ccuTaskGroup);
    
private:
    HcclDllHcomMgr();

    void *dll_handle{nullptr};
    std::function<HcclResult(const std::string &group, void *tilingData, void *ccuTaskGroup)> hccl_HcomGetCcuTaskInfo_func{nullptr};
};

} // namespace ge

#endif