/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FLOW_FUNC_BALANCE_CONFIG_IMPL_H
#define FLOW_FUNC_BALANCE_CONFIG_IMPL_H

#include "flow_func/balance_config.h"

namespace FlowFunc {
class BalanceConfigImpl : public BalanceConfig {
public:
    BalanceConfigImpl() = default;

    ~BalanceConfigImpl() override = default;

    void SetAffinityPolicy(AffinityPolicy affinity_policy) override {
        affinity_policy_ = affinity_policy;
    }

    AffinityPolicy GetAffinityPolicy() const override {
        return affinity_policy_;
    }

    void SetBalanceWeight(const BalanceWeight &balance_weight) override {
        balance_weight_ = balance_weight;
    }

    const BalanceWeight &GetBalanceWeight() const override {
        return balance_weight_;
    }

    /**
     * @brief set data pos.
     * each element of data_pos is the flow msg position in balance weight matrix.
     * pair first is row index, pair second is col index.
     * @param data_pos position in balance weight matrix.
     */
    void SetDataPos(const std::vector<std::pair<int32_t, int32_t>> &data_pos) override {
        data_pos_ = data_pos;
    }

    const std::vector<std::pair<int32_t, int32_t>> &GetDataPos() const override
    {
        return data_pos_;
    }

private:
    AffinityPolicy affinity_policy_ = AffinityPolicy::NO_AFFINITY;
    BalanceWeight balance_weight_;
    std::vector<std::pair<int32_t, int32_t>> data_pos_;
};
}
#endif // FLOW_FUNC_BALANCE_CONFIG_IMPL_H
