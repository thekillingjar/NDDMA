/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef DIM1_TRANSPOSE_TO_SQUEEZE_PASS_H
#define DIM1_TRANSPOSE_TO_SQUEEZE_PASS_H

#include "framework/common/debug/log.h"
#include "graph/passes/base_pass.h"
#include "attribute_group/attr_group_symbolic_desc.h"

namespace ge {
///
/// Define a useless transpose op as transposing value but not changes the actual value layout,
/// for example, transposing shape (1, 16, 32) to (16, 1, 32).
/// This Pass replaces the useless transposes to squeeze (to squeeze out the 1's dims)
/// and unsqueeze(to get 1's dims back to the wanted place)
///
/// 定义无用transpose为未实际修改数据内存排布的transpose，如将(1, 16, 32) 转置为 (16, 1, 32)
/// 该Pass识别无用transpose并把此类transpose替换为一个squeeze和一个unsqueeze
///
class Dim1TransposeToSqueezePass : public BaseNodePass {
 public:
  Status Run(NodePtr &node) override;

 private:
  int64_t GetBlockSize() const;
  bool ShouldIgnoreOp(const NodePtr &node) const;
  Status ReplaceTransposeToSqueeze(const GeTensorDescPtr &tensor, const std::vector<int64_t> &squeeze_axis,
                                   const std::vector<int64_t> &unsqueeze_axis, NodePtr &transpose_node);
  Status DeleteTranspose(NodePtr &node);
  OpDescPtr CreateOpDescPtr(const NodePtr &node, const string &op_type, const GeTensorDesc &input_desc_x,
                            const GeTensorDesc &output_desc, const std::vector<int64_t> &axis_value_vec) const;
  static bool IsUselessTransposeByShape(const GeTensorDescPtr &input_x_desc, const std::vector<int64_t> &perm,
                                        std::vector<int64_t> &squeeze_axis, std::vector<int64_t> &unsqueeze_axis,
                                        std::vector<int64_t> &shape);
  static bool IsUselessTransposeBySymbolicShape(const GeTensorDescPtr &input_x_desc, const std::vector<int64_t> &perm,
                                                std::vector<int64_t> &squeeze_axis,
                                                std::vector<int64_t> &unsqueeze_axis,
                                                std::vector<int64_t> &shape_index);
  Status SetShapeForSymbolic(const GeTensorDescPtr &input_x_desc, const std::vector<int64_t> &shape_index,
                             const GeTensorDescPtr &tensor, std::vector<int64_t> &squeeze_output_shape) const;
};
}  // namespace ge

#endif  // DIM1_TRANSPOSE_TO_SQUEEZE_PASS_H
