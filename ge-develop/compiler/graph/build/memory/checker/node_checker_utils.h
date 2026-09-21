/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_GRAPH_BUILD_MEMORY_CHECKER_NODE_CHECKER_UTILS_H_
#define GE_GRAPH_BUILD_MEMORY_CHECKER_NODE_CHECKER_UTILS_H_
#include "ge_common/ge_api_types.h"
#include "ge_common/ge_api_error_codes.h"
#include "node_checker_register.h"
#include "graph/node.h"

namespace ge {
class NodeCheckerUtils {
 public:
  static Status AlignMemSize(int64_t &size);
  static std::string NodeName(const Node *const node);
  static SpecailNodeTypes GetSpecialNodeTypes(const Node *const node);
  static Status GetOutputOffset(const OpDesc *const op_desc, const int32_t out_index, int64_t &offset);
  static Status GetStrideForContinuousInput(const Node *const peer_node, const int32_t out_index, int64_t &stride);
  static Status GetStrideForNoPaddingContinuousInput(const Node *const node, const Node *const peer_node,
                                                     const int32_t out_index, int64_t &stride);
  static Status GetStrideForNoPaddingContinuousOutput(const Node *const node, const int32_t out_index, int64_t &stride);
  static Status ErrorLogAllInputs(const NodeCheckerParam &param);
  static Status ErrorLogAllOutputs(const NodeCheckerParam &param);
  static Status CheckNoPaddingContinuousInputNodeAttrs(const Node *const node);
  static Status CheckNoPaddingContinuousOutputNodeAttrs(const Node *const node);
  static Status CheckPhonyConcatOutputSize(const Node *const node);
  static Status CheckPhonySplitInputSize(const Node *const node);
  static Status GetOutputSize(const Node *const node, const int32_t index, int64_t &mem_size);
  static Status GetInputSize(const Node *const node, const int32_t index, int64_t &mem_size);
};
}  // namespace ge

#endif  // GE_GRAPH_BUILD_MEMORY_CHECKER_NODE_CHECKER_UTILS_H_
