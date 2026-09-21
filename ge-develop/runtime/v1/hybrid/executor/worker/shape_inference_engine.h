/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_HYBRID_EXECUTOR_INFERSHAPE_SHAPE_INFERENCE_ENGINE_H_
#define GE_HYBRID_EXECUTOR_INFERSHAPE_SHAPE_INFERENCE_ENGINE_H_

#include "hybrid/executor/hybrid_execution_context.h"
#include "hybrid/executor/subgraph_context.h"

namespace ge {
namespace hybrid {
class ShapeInferenceEngine {
 public:
  ShapeInferenceEngine(GraphExecutionContext *const execution_context, const bool force_infer_shape);
  ~ShapeInferenceEngine() = default;
  void Config(SubgraphContext *const subgraph_context);
  Status InferShape(const NodeState &node_state) const;
  Status InitInferShapes(const GraphItem *const graph_item,
                         const std::vector<TensorValue> &inputs,
                         const std::vector<ConstGeTensorDescPtr> &input_desc) const;
  bool IsForceInferShape() const { return force_infer_shape_; }

private:
  bool force_infer_shape_;

private:
  Status PropagateOutputs(const NodeItem *const node_item, const TensorValue *const tensor,
                          const GeTensorDesc *const tensor_desc) const;
  Status UpdateShapeAndValue(const NodeItem *const node_item, const TensorValue *const tensor,
                             const GeTensorDesc *const tensor_desc) const;
  Status DoInferShape(const NodeState &node_state) const;
  Status PropagateOutputShapes(const NodeState &node_state) const;
  Status InferShapeForSubgraph(const NodeItem &node_item, const FusedSubgraph &fused_subgraph) const;
  Status UpdatePeerNodeShape(const Node &node) const;
  Status SetDependingTensor(const FusedSubgraph &fused_subgraph) const;

  GraphExecutionContext *execution_context_;
  SubgraphContext *subgraph_context_{nullptr};
};
}  // namespace hybrid
}  // namespace ge
#endif // GE_HYBRID_EXECUTOR_INFERSHAPE_SHAPE_INFERENCE_ENGINE_H_
