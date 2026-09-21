/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FLOW_FUNC_DRV_MANAGER_H
#define FLOW_FUNC_DRV_MANAGER_H

#include<cstdint>

namespace FlowFunc {

class FlowFuncDrvManager {
public:
    static FlowFuncDrvManager &Instance();

    int32_t Init() const;

    void Finalize() const;

    int32_t WaitBindHostPid(uint32_t wait_timeout) const;

private:
    int32_t InitMemGroup() const;

    int32_t InitBuff() const;

    int32_t CreateSchedGroup() const;
};
}
#endif // FLOW_FUNC_DRV_MANAGER_H_H
