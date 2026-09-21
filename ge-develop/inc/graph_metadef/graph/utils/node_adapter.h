/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef INC_GRAPH_UTILS_NODE_ADAPTER_H_
#define INC_GRAPH_UTILS_NODE_ADAPTER_H_

#include "graph/gnode.h"
#include "graph/node.h"

namespace ge {
class GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY NodeAdapter {
 public:
  static GNode Node2GNode(const NodePtr &node);
  static NodePtr GNode2Node(const GNode &node);
  static GNodePtr Node2GNodePtr(const NodePtr &node);
};
}  // namespace ge
#endif  // INC_GRAPH_UTILS_NODE_ADAPTER_H_
