/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_GRAPH_PASSES_CONSTANT_CLIP_PASS_H_
#define GE_GRAPH_PASSES_CONSTANT_CLIP_PASS_H_

#include "graph/passes/base_pass.h"

namespace ge {
/* Pass Explaination:
 *
 * float weight value may be infinite from high bit to low bit when const link cast(fp32->fp16) on ConstantFoldingPass
 * this pass need to be processed before ConstantFoldingPass to infinite this
 * by insert clipbyvalue(outputs is acquired by inputs limited beween min and max value) between const and cast
 *
 *    const
 *      |
 *     cast
 *
 * when after process, it will be like as follows:
 *
 *   const const_min const_max
 *     \      |         /
 *        cplibyvalue
 *            |
 *           cast
 */
class ConstantClipPass : public BaseNodePass {
 public:
  Status Run(NodePtr& node) override;

 private:
  Status CheckWeightInfinite(const ConstGeTensorPtr &const_tensor, const DataType &dst_dt, bool &is_infinite) const;

  Status InsertClipByValueBetweenCastAndConst(const NodePtr &node);

  NodePtr CreateClipMinMaxNode(const NodePtr &node, const bool &is_min) const;

  NodePtr CreateClipNode(const NodePtr &node);
};
}  // namespace ge
#endif  // GE_GRAPH_PASSES_CONSTANT_CLIP_PASS_H_
