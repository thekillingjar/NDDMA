/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_RUNTIME_V2_GRAPH_BUILDER_BG_INFER_SHAPE_RANGE_H_
#define AIR_CXX_RUNTIME_V2_GRAPH_BUILDER_BG_INFER_SHAPE_RANGE_H_
#include "exe_graph/lowering/value_holder.h"
#include "graph/node.h"
#include "graph/debug/ge_attr_define.h"
#include "exe_graph/lowering/lowering_global_data.h"

namespace gert {
namespace bg {
struct ShapeRangeInferenceResult {
 public:
  ShapeRangeInferenceResult();
  ShapeRangeInferenceResult(size_t outputs_num, std::vector<ValueHolderPtr> &out_holder);
  std::vector<ValueHolderPtr> GetAllMaxShapes();
  std::vector<ValueHolderPtr> GetAllShapeRanges();
  std::vector<ValueHolderPtr> GetAllMinShapes();
  bool IsSuccess() const;
  void SetErrorStatus();
  static struct ShapeRangeInferenceResult ErrorResult();

 private:
  HyperStatus status_;
  std::vector<ValueHolderPtr> all_outputs_;
  size_t compute_node_output_num_;
};

ShapeRangeInferenceResult InferShapeRange(const ge::NodePtr &node, const std::vector<ValueHolderPtr> &input_shapes,
                                          const LoweringGlobalData &global_data);
std::vector<ValueHolderPtr> InferMaxShape(const ge::NodePtr &node, const std::vector<ValueHolderPtr> &input_shapes,
                                          const LoweringGlobalData &global_data);
}
}
#endif // AIR_CXX_RUNTIME_V2_GRAPH_BUILDER_BG_INFER_SHAPE_RANGE_H_
