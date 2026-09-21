/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_RUNTIME_V2_SEQUENCE_OP_H
#define AIR_CXX_RUNTIME_V2_SEQUENCE_OP_H
#include <cstdint>
#include "graph/types.h"
#include "graph/node.h"

namespace gert {
void SetOpInfo(const ge::OpDescPtr &op_desc, ge::Format format, ge::DataType dt,
    std::initializer_list<int64_t> shape);
}  // namespace gert
#endif  // AIR_CXX_RUNTIME_V2_SEQUENCE_OP_H
