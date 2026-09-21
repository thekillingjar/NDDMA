/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_GRAPH_PASSES_TRANSPOSE_TRANSDATA_PASS_H_
#define GE_GRAPH_PASSES_TRANSPOSE_TRANSDATA_PASS_H_

#include "graph/passes/base_pass.h"

namespace ge {
class TransposeTransDataPass : public BaseNodePass {
 public:
  Status Run(NodePtr &node) override;
 private:
  Status FusionTranspose(NodePtr &node);
  Status CheckInOutDataAnchorValid(const NodePtr &node, uint32_t input_data_num,
                                   uint32_t output_data_num) const;
  Status RemoveTranspose(const NodePtr &node);
  bool FusionIfNeed(const OpDescPtr &op_desc, const NodePtr &node) const;
  void CopyInputEdges(NodePtr &origin_node, NodePtr &new_node) const;
};
}  // namespace ge
#endif  // GE_GRAPH_PASSES_TRANSPOSE_TRANSDATA_PASS_H_

