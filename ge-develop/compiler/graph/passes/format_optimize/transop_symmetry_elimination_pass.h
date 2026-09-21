/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_SYMMETRY_ELIMINATION_PASS_H
#define GE_SYMMETRY_ELIMINATION_PASS_H

#include "graph/passes/base_pass.h"

namespace ge {
class TransOpSymmetryEliminationPass : public BaseNodePass {
 public:
  Status Run(NodePtr &node) override;

 private:
  /// Judge whether the node can be offset
  /// 1.both are transform op
  /// 2.is symmetry position
  /// 3.satisfy precision loss
  /// @param node
  /// @return True or False
  static bool CheckCanBeEliminated(const ge::NodePtr &src_node, const InDataAnchorPtr &dst_in_anchor);

  /// two transform nodes can be offset only when the front node's input is
  /// consistent with the back one's output
  /// @param src_node: the front node
  /// @param dst_node: the back node
  /// @return True or False, whether can be offset or not
  static bool DescAreSymmetry(const NodePtr &src_node, const NodePtr &dst_node);

  /// two transdata nodes can be offset only when the front node's input is
  /// consistent with the back one's output
  /// @param src_node: the front node
  /// @param dst_node: the back node
  /// @return True or False, whether can be offset or not
  static bool IsTransdataMemLayoutSymmetry(const NodePtr &src_node, const NodePtr &dst_node);

  /// get the number of unknown shape of node
  /// @param node_desc: node to be checked
  /// @return  0 , is not dynamic shape; UNKNOWN_DIM_NUM , all dims are unknown; n , n > 0 , has n dims unknown
  static int32_t GetUnknownDimsNum(const GeTensorDesc& node_desc);

  /// judge after two transposed op transform the raw data will be the same
  /// @param src_node: first transposed op
  /// @param dst_node: second transposed op
  /// @return True or False, same or not
  static bool JudgeTransposeDBack2Raw(const NodePtr &src_node, const NodePtr &dst_node);

  /// two transform nodes can be offset like A->T1->T2->B
  /// 1.unlink T1->T2
  /// 2.link A->T2
  /// 3.copy in-control/data-in-control from T1->T2
  /// 4.isolateAndDelete T2, it will re-pass all in and out node
  /// then we get  A->B  . Leave T1 to prune pass.
  ///               ->T1
  /// @param src_node: the front node
  /// @param src_out_anchor: the front node out anchor
  /// @param dst_node: the back node
  /// @param dst_in_anchor: the back node in anchor
  /// @return SUCCESS or Fail, whether
  Status EliminateTransOp(NodePtr &src_node, const OutDataAnchorPtr &src_out_anchor, NodePtr &dst_node,
                          const InDataAnchorPtr &dst_in_anchor);

  Status RemoveTransOpWithoutOutput(NodePtr &pre_node, NodePtr &trans_node);
};
}  // namespace ge

#endif  // GE_SYMMETRY_ELIMINATION_PASS_H
