/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef INC_REGISTER_OP_TILING_H
#define INC_REGISTER_OP_TILING_H

#include "graph/debug/ge_attr_define.h"
#include "graph/node.h"
#include "register/op_tiling_registry.h"

namespace optiling {
extern "C" ge::graphStatus OpParaCalculateV2(const ge::Operator &op, OpRunInfoV2 &run_info);
extern "C" ge::graphStatus OpAtomicCalculateV2(const ge::Node &node, OpRunInfoV2 &run_info);
extern "C" ge::graphStatus OpFftsPlusCalculate(const ge::Operator &op, std::vector<OpRunInfoV2> &op_run_info);
}  // namespace optiling
#endif  // INC_REGISTER_OP_TILING_H
