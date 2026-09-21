/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "default_reg_func_v2.h"

namespace ge {
namespace ascir {
std::vector<std::unique_ptr<ge::TmpBufDesc>> CalcTanhTmpSizeV2(const ge::AscNode &node)
{
    constexpr uint32_t TANH_HALF_CALC_PROC = 4;
    constexpr uint32_t TANH_FLOAT_CALC_PROC = 1;
    constexpr uint32_t TANH_ONE_REPEAT_BYTE_SIZE = 256;
    auto node_inputs = node.inputs;
    GE_ASSERT_TRUE(node_inputs.Size() > 0, "Node %s[%s] inputs size is 0.", node.GetTypePtr(), node.GetNamePtr());
    uint32_t calcTmpBuf =
        TANH_ONE_REPEAT_BYTE_SIZE * (node_inputs[0].attr.dtype == ge::DT_FLOAT16 ? TANH_HALF_CALC_PROC : TANH_FLOAT_CALC_PROC);
    GELOGD("Node %s[%s] temp buffer size: %u", node.GetTypePtr(), node.GetNamePtr(), calcTmpBuf);
    Expression TmpSize = ge::Symbol(calcTmpBuf);
    ge::TmpBufDesc desc = {TmpSize, -1};
    std::vector<std::unique_ptr<ge::TmpBufDesc>> tmpBufDescs;
    tmpBufDescs.emplace_back(std::make_unique<ge::TmpBufDesc>(desc));
    return tmpBufDescs;
}
}
}