/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_GRAPH_BUILD_MEMORY_CHECKER_NODE_CHECKER_REGISTER_H_
#define GE_GRAPH_BUILD_MEMORY_CHECKER_NODE_CHECKER_REGISTER_H_
#include <vector>
#include <functional>
#include "graph/graph.h"
#include "ge_common/ge_api_types.h"
#include "small_vector.h"

namespace ge {
enum SpecailNodeType {
  kSpecialNodeTypeContinuousInput,
  kSpecialNodeTypeContinuousOutput,
  kSpecialNodeTypeNoPaddingContinuousInput,
  kSpecialNodeTypeNoPaddingContinuousOutput,
  kSpecialNodeTypeMax
};

using SpecailNodeTypes = SmallVector<SpecailNodeType, static_cast<size_t>(kSpecialNodeTypeMax)>;

const char *SpecialNodeTypeStr(const SpecailNodeType &type);

struct NodeCheckerParam {
  const ComputeGraphPtr &graph;
  const Node *const node;
  const SpecailNodeType type;
};

using SpecailNodeCheckerFunc = Status (*)(const NodeCheckerParam &context);

class NodeCheckerRegister {
 public:
  NodeCheckerRegister() = default;
  ~NodeCheckerRegister() = default;
  NodeCheckerRegister(const SpecailNodeType &type, const SpecailNodeCheckerFunc &func);

  NodeCheckerRegister(NodeCheckerRegister &&other) = default;
  NodeCheckerRegister(const NodeCheckerRegister &other) = default;
  NodeCheckerRegister &operator=(const NodeCheckerRegister &other) = default;
  NodeCheckerRegister &operator=(NodeCheckerRegister &&other) = default;

  static std::array<SpecailNodeCheckerFunc, static_cast<size_t>(SpecailNodeType::kSpecialNodeTypeMax)> funcs;
};

SpecailNodeCheckerFunc GetSpecialNodeChecker(const SpecailNodeType &type);

#define REGISTER_SPECIAL_NODE_CHECKER(type, func) static auto g_var_##type##func = NodeCheckerRegister(type, func)
}  // namespace ge

#endif  // GE_GRAPH_BUILD_MEMORY_CHECKER_NODE_CHECKER_REGISTER_H_
