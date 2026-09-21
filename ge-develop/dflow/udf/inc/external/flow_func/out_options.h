/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FLOW_FUNC_OUT_OPTIONS_H
#define FLOW_FUNC_OUT_OPTIONS_H

#include <memory>
#include "balance_config.h"

namespace FlowFunc {
class OutOptionsImpl;

class FLOW_FUNC_VISIBILITY OutOptions {
public:
    OutOptions();

    ~OutOptions();

    /**
     * @brief will get or create BalanceConfig
     * @return BalanceConfig
     */
    BalanceConfig *MutableBalanceConfig();

    /**
     * @brief get BalanceConfig
     * @return BalanceConfig null means not balance config
     */
    const BalanceConfig *GetBalanceConfig() const;

private:
    std::shared_ptr<OutOptionsImpl> impl_;
};
}
#endif // FLOW_FUNC_OUT_OPTIONS_H
