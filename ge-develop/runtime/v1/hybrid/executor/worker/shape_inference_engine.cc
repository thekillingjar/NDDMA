/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "hybrid/executor/worker/shape_inference_engine.h"
#include "graph/runtime_inference_context.h"
#include "graph/shape_refiner.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/tensor_utils.h"
#include "common/math/math_util.h"
#include "common/profiling/profiling_properties.h"
#include "common/profiling/profiling_manager.h"
#include "common/profiling_definitions.h"
#include "hybrid/node_executor/node_executor.h"
#include "hybrid/model/infer/shape_utils.h"

namespace ge {
namespace {
constexpr int32_t kDataInputIndex = 0;
}

namespace hybrid {
ShapeInferenceEngine::ShapeInferenceEngine(GraphExecutionContext *const execution_context, const bool force_infer_shape)
    : force_infer_shape_(force_infer_shape),
      execution_context_(execution_context) {}

void ShapeInferenceEngine::Config(SubgraphContext *const subgraph_context) {
  this->subgraph_context_ = subgraph_context;
}

Status ShapeInferenceEngine::InferShape(const NodeState &node_state) const {
  HYBRID_CHK_STATUS_RET(DoInferShape(node_state),
                        "[Invoke][DoInferShape] failed for [%s(%s)].",
                        node_state.GetName().c_str(), node_state.GetType().c_str());
  PROFILING_START(node_state.GetProfilingIndex(), profiling::kInferShapePropgate);
  HYBRID_CHK_STATUS_RET(PropagateOutputShapes(node_state),
                        "[Invoke][PropagateOutputShapes] failed for [%s(%s)].",
                        node_state.GetName().c_str(), node_state.GetType().c_str());
  PROFILING_END(node_state.GetProfilingIndex(), profiling::kInferShapePropgate);
  return SUCCESS;
}


Status ShapeInferenceEngine::InitInferShapes(const GraphItem *const graph_item, const std::vector<TensorValue> &inputs,
                                             const std::vector<ConstGeTensorDescPtr> &input_desc) const {
  // Number of inputs of parent node should be greater or equal than that of subgraph
  const auto input_nodes = graph_item->GetInputNodes();
  if (inputs.size() < input_nodes.size()) {
    GELOGE(INTERNAL_ERROR,
           "[Check][Size][%s] Number of inputs [%zu] is not sufficient for subgraph which needs [%zu] inputs.",
           graph_item->GetName().c_str(), inputs.size(), input_nodes.size());
    REPORT_INNER_ERR_MSG("E19999",
                       "[%s] Number of inputs [%zu] is not sufficient for subgraph which needs [%zu] inputs.",
                       graph_item->GetName().c_str(), inputs.size(), input_nodes.size());
    return INTERNAL_ERROR;
  }

  for (size_t i = 0UL; i < input_nodes.size(); ++i) {
    auto &input_node = input_nodes[i];
    if (input_node == nullptr) {
      GELOGD("[%s] Input[%zu] is not needed by subgraph, skip it.", graph_item->GetName().c_str(), i);
      continue;
    }

    auto &input_tensor = inputs[i];
    GELOGD("[%s] Set input tensor[%zu] to inputs with index = %d, tensor = %s",
           graph_item->GetName().c_str(),
           i,
           input_node->input_start,
           input_tensor.DebugString().c_str());

    const GeTensorDesc *tensor_desc = nullptr;
    // tensor_desc may be nullptr when static shape input
    if (i < input_desc.size()) {
      tensor_desc = input_desc[i].get();
    }
    if (input_node->has_observer || graph_item->HasCtrlFlowOp()) {
      GELOGD("[%s] Start to update input[%zu] for subgraph data node.", graph_item->GetName().c_str(), i);
      GE_CHK_STATUS_RET(UpdateShapeAndValue(input_node, &input_tensor, tensor_desc),
                        "[Update][ShapeAndValue] Failed to update shape and value for data node.");
    } else {
      GELOGD("[%s] Start to propagate input[%zu] value from data node.", graph_item->GetName().c_str(), i);
      GE_CHK_STATUS_RET(PropagateOutputs(input_node, &input_tensor, tensor_desc),
                        "[Propagate][Outputs] Failed to propagate outputs for data node.");
    }
  }

  GELOGD("[%s] Done setting inputs.", graph_item->GetName().c_str());
  return SUCCESS;
}

Status ShapeInferenceEngine::PropagateOutputs(const NodeItem *const node_item, const TensorValue *const tensor,
                                              const GeTensorDesc *const tensor_desc) const {
  for (int32_t i = 0; i < node_item->num_outputs; ++i) {
    auto &output_nodes = node_item->outputs[static_cast<size_t>(i)];

    for (auto &dst_input_index_and_node : output_nodes) {
      const auto dst_input_idx = dst_input_index_and_node.first;
      auto &dst_node_item = dst_input_index_and_node.second;
      GE_CHECK_NOTNULL(dst_node_item);
      GE_CHK_STATUS_RET(subgraph_context_->SetInput(*dst_node_item, dst_input_idx, *tensor),
                        "[Invoke][SetInput] failed for node[%s] propagate input value to node[%s].",
                        node_item->NodeName().c_str(), dst_node_item->NodeName().c_str());

      if (node_item->is_output_shape_static) {
        continue;
      }
      GE_CHECK_NOTNULL(tensor_desc);
      GE_CHECK_NOTNULL(dst_node_item->op_desc->MutableInputDesc(static_cast<uint32_t>(dst_input_idx)));
      GE_CHK_STATUS_RET_NOLOG(ShapeUtils::CopyShapeAndTensorSize(*tensor_desc,
          *(dst_node_item->op_desc->MutableInputDesc(static_cast<uint32_t>(dst_input_idx)))));
    }
  }

  const auto node_state = subgraph_context_->GetNodeState(node_item);
  GE_CHECK_NOTNULL(node_state);
  node_state->SetSkipSchedule(true);
  return SUCCESS;
}

Status ShapeInferenceEngine::UpdateShapeAndValue(const NodeItem *const node_item, const TensorValue *const tensor,
                                                 const GeTensorDesc *const tensor_desc) const {
  GE_CHK_STATUS_RET(subgraph_context_->SetInput(*node_item, kDataInputIndex, *tensor),
                    "[Invoke][SetInput] failed for node[%s] to set input tensor", node_item->NodeName().c_str());

  if (IsForceInferShape() || node_item->is_dynamic) {
    GELOGD("Start to update node [%s] for subgraph data node.", node_item->NodeName().c_str());
    GE_CHECK_NOTNULL(tensor_desc);
    const auto node_state = subgraph_context_->GetNodeState(node_item);
    GE_CHECK_NOTNULL(node_state);
    const auto op_desc = node_item->GetOpDesc();
    GE_CHECK_NOTNULL(op_desc);
    GE_CHK_STATUS_RET_NOLOG(ShapeUtils::CopyShapeAndTensorSize(*tensor_desc, *op_desc->MutableInputDesc(0U)));
    const auto output_desc = op_desc->MutableOutputDesc(static_cast<uint32_t>(kDataInputIndex));
    GE_CHECK_NOTNULL(output_desc);
    output_desc->SetShape(tensor_desc->GetShape());
    output_desc->SetOriginShape(tensor_desc->GetOriginShape());
    node_state->SetSkipInferShape(true);
  }
  return SUCCESS;
}

Status ShapeInferenceEngine::DoInferShape(const NodeState &node_state) const {
  // Wait for "const input nodes" if node's shape inference function requires any.
  // Even if output shape is static, there are cases that the const-input will be used in OpTiling and Execution
  GE_CHK_STATUS_RET_NOLOG(node_state.AwaitDependShapes(*execution_context_));
  GE_CHK_STATUS_RET_NOLOG(node_state.AwaitInputTensors(*execution_context_));

  auto &node_item = node_state.GetNodeItem();
  if (node_item.is_output_shape_static && (!node_item.is_need_force_infershape)) {
    return SUCCESS;
  }

  const auto &guard = node_item.MutexGuard("DoInferShape");
  if (node_item.fused_subgraph != nullptr) {
    GE_CHK_STATUS_RET_NOLOG(InferShapeForSubgraph(node_item, *node_item.fused_subgraph));
    GE_CHK_STATUS_RET_NOLOG(node_item.CalcOutputTensorSizes());
    return SUCCESS;
  }
  (void)guard;

  // Skip shape inference for node of type DEPEND_COMPUTE
  if (node_item.shape_inference_type == DEPEND_COMPUTE) {
    GELOGD("[%s] Skipping node with unknown shape type DEPEND_COMPUTE", node_item.NodeName().c_str());
    return SUCCESS;
  }

  // Do shape inference
  // Skipping infer shape of input node.
  GELOGD("[%s] Start to invoke InferShapeAndType", node_item.NodeName().c_str());
  if (!node_state.MaySkipShapeInference()) {
    Operator *const op = node_state.GetOperator(execution_context_->stage_id);
    GE_CHECK_NOTNULL(op);
    RECORD_SHAPE_INFERENCE_EVENT(execution_context_, node_item.NodeName().c_str(), "[InferShapeAndType] Start");
    PROFILING_START(node_state.GetProfilingIndex(), profiling::kInferShape);
    GE_CHK_STATUS_RET(ShapeRefiner::InferShapeAndTypeForRunning(node_item.node, *op, true),
        "[Invoke][InferShapeAndType] for %s failed.", node_item.NodeName().c_str());
    PROFILING_END(node_state.GetProfilingIndex(), profiling::kInferShape);
    RECORD_SHAPE_INFERENCE_EVENT(execution_context_, node_item.NodeName().c_str(), "[InferShapeAndType] End");
  }

  // update output tensor sizes after shape inference
  // error if shape is still unknown and not of type DEPEND_SHAPE_RANGE
  RECORD_COMPILE_EVENT(execution_context_, node_item.NodeName().c_str(), "[CalcOpRunningParam] Start");
  GE_CHK_STATUS_RET_NOLOG(node_item.CalcOutputTensorSizes(node_item.shape_inference_type == DEPEND_SHAPE_RANGE));
  RECORD_COMPILE_EVENT(execution_context_, node_item.NodeName().c_str(), "[CalcOpRunningParam] End");

  GELOGD("[%s] [HybridTrace] After shape inference. Node = %s",
         node_item.NodeName().c_str(),
         node_item.DebugString().c_str());

  GELOGD("[%s] InferShapeAndType finished successfully.", node_item.NodeName().c_str());
  return SUCCESS;
}

Status ShapeInferenceEngine::PropagateOutputShapes(const NodeState &node_state) const {
  const auto &node_item = node_state.GetNodeItem();
  if (node_item.is_output_shape_static) {
    return SUCCESS;
  }

  RECORD_SHAPE_INFERENCE_EVENT(execution_context_, node_item.NodeName().c_str(), "[PropagateOutputShapes] Start");
  if (!node_item.IsShapeInFuture()) {
    GE_CHK_STATUS_RET_NOLOG(const_cast<NodeItem&>(node_item).DoPropagate());
  }
  RECORD_SHAPE_INFERENCE_EVENT(execution_context_, node_item.NodeName().c_str(), "[PropagateOutputShapes] End");
  GELOGD("[%s] Propagating output shapes finished successfully.", node_item.NodeName().c_str());
  return SUCCESS;
}

Status ShapeInferenceEngine::InferShapeForSubgraph(const NodeItem &node_item,
                                                   const FusedSubgraph &fused_subgraph) const {
  GELOGD("[%s] Start to infer shape by fused subgraph", node_item.NodeName().c_str());
  for (auto &it : fused_subgraph.input_mapping) {
    const auto parent_tensor_desc = node_item.MutableInputDesc(it.first);
    GE_CHECK_NOTNULL(parent_tensor_desc);
    GELOGD("Start to update shape by input[%d]", it.first);
    GELOGD("Update shape to [%s]", parent_tensor_desc->GetShape().ToString().c_str());
    GELOGD("Update original shape to [%s]", parent_tensor_desc->GetOriginShape().ToString().c_str());
    for (auto &tensor_desc : it.second) {
      GE_CHECK_NOTNULL(tensor_desc);
      tensor_desc->SetShape(parent_tensor_desc->GetShape());
      tensor_desc->SetOriginShape(parent_tensor_desc->GetOriginShape());
    }
  }

  GE_CHK_STATUS_RET(SetDependingTensor(fused_subgraph), "[Set][DependingTensor] Failed for set depending tensor.");
  for (size_t i = 0; i < fused_subgraph.nodes.size(); ++i) {
    GELOGD("[%s] Start to invoke InferShapeAndType", fused_subgraph.nodes[i]->GetName().c_str());
    PROFILING_START(fused_subgraph.name_to_prof_indexes[i], profiling::kInferShape);
    GE_CHK_STATUS_RET(ShapeRefiner::InferShapeAndType(fused_subgraph.nodes[i]));
    PROFILING_END(fused_subgraph.name_to_prof_indexes[i], profiling::kInferShape);
    GELOGD("[%s] Done invoking InferShapeAndType", fused_subgraph.nodes[i]->GetName().c_str());
    GE_CHK_STATUS_RET(UpdatePeerNodeShape(*fused_subgraph.nodes[i]),
        "[Update][PeerNodeShape] failed for [%s].", fused_subgraph.nodes[i]->GetName().c_str());
  }

  for (auto &it : fused_subgraph.output_mapping) {
    const int32_t parent_output_idx = it.first;
    GELOGD("Update parent output[%d]", parent_output_idx);
    const auto &input_desc = it.second;
    GE_CHECK_NOTNULL(input_desc);
    const auto parent_output_tensor_desc = node_item.MutableOutputDesc(parent_output_idx);
    GE_CHECK_NOTNULL(parent_output_tensor_desc);
    GELOGD("Update shape to [%s]", input_desc->GetShape().ToString().c_str());
    GELOGD("Update original shape to [%s]", input_desc->GetOriginShape().ToString().c_str());
    parent_output_tensor_desc->SetOriginShape(input_desc->GetOriginShape());
    parent_output_tensor_desc->SetShape(input_desc->GetShape());
  }

  GELOGD("[%s] Done shape inference by subgraph successfully.", node_item.NodeName().c_str());
  return SUCCESS;
}

Status ShapeInferenceEngine::UpdatePeerNodeShape(const Node &node) const {
  const auto op_desc = node.GetOpDesc();
  GE_CHECK_NOTNULL(op_desc);
  for (const auto &out_anchor : node.GetAllOutDataAnchors()) {
    const auto output_tensor = op_desc->MutableOutputDesc(static_cast<uint32_t>(out_anchor->GetIdx()));
    for (const auto &peer_anchor : out_anchor->GetPeerInDataAnchors()) {
      const auto peer_node = peer_anchor->GetOwnerNode();
      GE_CHECK_NOTNULL(peer_node);
      const auto peer_op_desc = peer_node->GetOpDesc();
      GE_CHECK_NOTNULL(peer_op_desc);
      const auto peer_input_desc = peer_op_desc->MutableInputDesc(static_cast<uint32_t>(peer_anchor->GetIdx()));
      if (peer_input_desc == nullptr) {
        GELOGE(GRAPH_FAILED, "[Call][MutableInputDesc] for %s(%s) return nullptr.",
               peer_op_desc->GetName().c_str(), peer_op_desc->GetType().c_str());
        REPORT_INNER_ERR_MSG("E19999", "%s(%s) call MutableInputDesc return nullptr.",
                          peer_op_desc->GetName().c_str(), peer_op_desc->GetType().c_str());
        continue;
      }

      GELOGI("Peer input op desc name is %s, need to flush: shape size is %zu, datatype is %d, original datatype is %d",
             peer_anchor->GetOwnerNode()->GetOpDesc()->GetName().c_str(),
             output_tensor->GetShape().GetDimNum(), output_tensor->GetDataType(),
             output_tensor->GetOriginDataType());
      peer_input_desc->SetOriginShape(output_tensor->GetOriginShape());
      peer_input_desc->SetShape(output_tensor->GetShape());
      GELOGI("Peer input op desc name is %s, shape size is %zu, datatype is %d, original datatype is %d",
             peer_anchor->GetOwnerNode()->GetOpDesc()->GetName().c_str(),
             peer_input_desc->GetShape().GetDimNum(), peer_input_desc->GetDataType(),
             peer_input_desc->GetOriginDataType());
    }
  }
  return SUCCESS;
}

Status ShapeInferenceEngine::SetDependingTensor(const FusedSubgraph &fused_subgraph) const {
  if (!fused_subgraph.data_dependencies.empty()) {
    for (auto &dep : fused_subgraph.data_dependencies) {
      const auto src_node_id = dep.first.first;
      const auto src_output_idx = dep.first.second;
      const auto dst_op_desc = dep.second.first;
      const auto dst_input_idx = dep.second.second;
      const auto dst_tensor_desc = dst_op_desc->MutableInputDesc(static_cast<uint32_t>(dst_input_idx));
      GE_CHECK_NOTNULL(dst_tensor_desc);
      GeTensorPtr tensor = nullptr;
      if (execution_context_->runtime_context_.GetTensor(src_node_id, src_output_idx, tensor) != GRAPH_SUCCESS) {
        REPORT_INNER_ERR_MSG("E19999", "Failed to get tensor, node_id = %" PRId64 ", output_idx = %d.",
                          src_node_id, src_output_idx);
        GELOGE(INTERNAL_ERROR, "[Get][Tensor]Failed to get tensor, node_id = %ld, output_idx = %d.",
               src_node_id, src_output_idx);
        return INTERNAL_ERROR;
      }
      (void)AttrUtils::SetTensor(dst_tensor_desc, ATTR_NAME_VALUE, tensor);
    }
  }
  return SUCCESS;
}
}  // namespace hybrid
}  // namespace ge
