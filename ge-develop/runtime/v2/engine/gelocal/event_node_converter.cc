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
LowerResult LoweringEventNode(const ge::NodePtr &node, const LowerInput &input) {
  (void)node;
  (void)input;
  return LowerResult{};
}
// Send/Recv算子产生于静态图中流拆分
// 若出现在动态根图的静态子图中，则由静态子图执行来承载
// 若出现在静态根图中，且该静态根图走rt2，因为单算子下发都在一条流上
// 且send/recv算子无数据输入和输出，只有控制输入和输出，这里按照空实现处理
REGISTER_NODE_CONVERTER("Send", LoweringEventNode);
REGISTER_NODE_CONVERTER("Recv", LoweringEventNode);
}  // namespace gert
