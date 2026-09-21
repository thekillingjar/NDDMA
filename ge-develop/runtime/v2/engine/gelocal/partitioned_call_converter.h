/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ASCEND_PARTITIONED_CALL_CONVERTER_H
#define ASCEND_PARTITIONED_CALL_CONVERTER_H

#include "graph/node.h"
#include "register/node_converter_registry.h"
namespace gert {
LowerResult LoweringPartitionedCall(const ge::NodePtr &node, const LowerInput &lower_input);
}  // namespace gert

#endif  // ASCEND_PARTITIONED_CALL_CONVERTER_H
