/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FLOW_FUNC_BALANCE_CONFIG_H
#define FLOW_FUNC_BALANCE_CONFIG_H

#include <cstdint>
#include <vector>
#include "flow_func_defines.h"

namespace FlowFunc {
enum class AffinityPolicy : int32_t {
    NO_AFFINITY = 0,     // no affinity
    ROW_AFFINITY = 1,    // row affinity
    COL_AFFINITY = 2,    // col affinity
};
#pragma pack(1)     // 1 byte alignment
struct BalanceWeight {
    int32_t rowNum = 0;
    int32_t colNum = 0;
    const int32_t *matrix = nullptr;  // pointer life cycle must be more than BalanceWeight. null means all value is 1
};
#pragma pack()      // Cancel 1 byte alignment

class FLOW_FUNC_VISIBILITY BalanceConfig {
public:
    BalanceConfig() = default;

    virtual ~BalanceConfig() = default;

    virtual void SetAffinityPolicy(AffinityPolicy affinityPolicy) = 0;

    virtual AffinityPolicy GetAffinityPolicy() const = 0;

    virtual void SetBalanceWeight(const BalanceWeight &balanceWeight) = 0;

    virtual const BalanceWeight &GetBalanceWeight() const = 0;

    /**
     * @brief set data pos.
     * each element of dataPos is the flow msg position in balance weight matrix.
     * pair first is row index, pair second is col index.
     * @param dataPos position in balance weight matrix.
     */
    virtual void SetDataPos(const std::vector<std::pair<int32_t, int32_t>> &dataPos) = 0;

    virtual const std::vector<std::pair<int32_t, int32_t>> &GetDataPos() const = 0;
};
}
#endif // FLOW_FUNC_BALANCE_CONFIG_H
