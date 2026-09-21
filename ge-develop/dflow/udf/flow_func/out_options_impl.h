/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FLOW_FUNC_OUT_OPTIONS_IMPL_H
#define FLOW_FUNC_OUT_OPTIONS_IMPL_H

#include "flow_func/balance_config.h"
#include "balance_config_impl.h"

namespace FlowFunc {
class OutOptionsImpl {
public:
    BalanceConfig *MutableBalanceConfig() {
        has_balance_config_ = true;
        return &balance_config_;
    }

    const BalanceConfig *GetBalanceConfig() const {
        if (has_balance_config_) {
            return &balance_config_;
        } else {
            return nullptr;
        }
    }

private:
    bool has_balance_config_ = false;
    BalanceConfigImpl balance_config_;
};
}
#endif // FLOW_FUNC_OUT_OPTIONS_IMPL_H
