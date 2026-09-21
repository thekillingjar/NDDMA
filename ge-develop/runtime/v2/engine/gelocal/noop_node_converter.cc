/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "register/node_converter_registry.h"
#include "graph/node.h"

namespace gert {
LowerResult LoweringNoOpNode(const ge::NodePtr &node, const LowerInput &input) {
  (void)node;
  (void)input;
  return {};
}
// NoOp算子无数据输入和输出，只有控制输入和输出，这里按照空实现处理
// NoOp算子的控制作用实现逻辑在lowering框架中体现，遍历上游控制节点，继承节点的ordered_holder
REGISTER_NODE_CONVERTER("NoOp", LoweringNoOpNode);
REGISTER_NODE_CONVERTER("ControlTrigger", LoweringNoOpNode);
}  // namespace gert
