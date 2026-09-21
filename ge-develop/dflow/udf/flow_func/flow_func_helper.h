/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FLOW_FUNC_HELPER_H
#define FLOW_FUNC_HELPER_H

#include "flow_func/balance_config.h"

namespace FlowFunc {
class FLOW_FUNC_VISIBILITY FlowFuncHelper {
public:
    static bool IsBalanceConfigValid(const BalanceConfig &config);

    static void CalcRouteLabelAndDataLabel(const BalanceConfig &config, const std::pair<int32_t, int32_t> &pos,
        uint32_t &data_label, uint32_t &route_label);
private:
    static bool CheckMatrixValid(int32_t row_num, int32_t col_num);
};
}
#endif // FLOW_FUNC_HELPER_H
