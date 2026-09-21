/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_NODE_SHAPE_INFER_H
#define AIR_CXX_NODE_SHAPE_INFER_H

#include <vector>
#include "node_shape_propagator.h"
#include "ge/ge_api_error_codes.h"
#include "graph/types.h"
#include "graph/op_desc.h"

namespace ge {
namespace hybrid {
struct NodeShapeInfer : NodeShapePropagator {

  const std::string &NodeName() const {
    return node_name;
  }

  const std::string &NodeType() const {
    return node_type;
  }

  inline bool IsShapeInFuture() const {
    return (shape_inference_type == DEPEND_SHAPE_RANGE) || (shape_inference_type == DEPEND_COMPUTE);
  }

  Status CalcOutputTensorSizes(const bool fallback_with_range = false) const;
  Status OnNodeDone() const;

  bool IsInputShapeStatic(const int32_t index) const;

public:
  std::string node_name;
  std::string node_type;
  int32_t group = -1;
  bool is_dynamic = false;
  bool is_output_shape_static = true;
  bool is_need_force_infershape = false;
  OpDesc *op_desc;
  UnknowShapeOpType shape_inference_type = DEPEND_IN_SHAPE;

protected:
  std::vector<bool> is_input_shape_static_;
};
} // namespace hybrid
} // namespace ge
#endif  // AIR_CXX_NODE_SHAPE_INFER_H
