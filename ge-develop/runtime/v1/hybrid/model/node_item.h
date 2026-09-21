/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_HYBRID_MODEL_NODE_ITEM_H_
#define GE_HYBRID_MODEL_NODE_ITEM_H_

#include <mutex>
#include <vector>
#include "framework/memory/memory_api.h"
#include "graph/node.h"
#include "graph/op_desc.h"
#include "graph/utils/node_utils.h"
#include "graph/runtime_inference_context.h"

#include "hybrid/common/tensor_value.h"
#include "hybrid/model/infer/node_shape_infer.h"

namespace ge {
namespace hybrid {
class NodeTask;
class NodeExecutor;
struct GraphExecutionContext;
class SubgraphContext;

struct FusedSubgraph {
  std::map<int32_t, std::vector<GeTensorDescPtr>> input_mapping;
  std::map<int32_t, GeTensorDescPtr> output_mapping;
  // [ {{src_node_id, output_idx}, {dst_op_desc, input_idx} ]
  std::vector<std::pair<std::pair<int64_t, int32_t>, std::pair<OpDesc *, int32_t>>> data_dependencies;
  std::vector<NodePtr> nodes;
  std::vector<int64_t> name_to_prof_indexes;
  ComputeGraphPtr graph;
};

bool IsControlFlowV2Op(const std::string &op_type);

bool IsFftsGraphNode(const OpDesc &op_desc);

bool IsFftsKernelNode(const OpDesc &op_desc);

class OptionalMutexGuard {
 public:
  OptionalMutexGuard(std::mutex *const mutex, const std::string &name);
  ~OptionalMutexGuard();

 private:
  std::mutex *mu_{nullptr};
  std::string name_;
};

// for caching static information across execution
struct NodeItem : NodeShapeInfer {
  ~NodeItem() = default;
  static Status Create(const NodePtr &node_ptr, std::unique_ptr<NodeItem> &node_item);

  OpDescPtr GetOpDesc() const {
    return node->GetOpDesc();
  }

  GeTensorDescPtr MutableOutputDesc(const int32_t index) const;

  Status UpdateInputDesc(const int32_t index, const GeTensorDesc &tensor_desc) const;

  GeTensorDescPtr MutableInputDesc(const int32_t index) const;

  Status GetInputDesc(const int32_t index, GeTensorDesc &tensor_desc) const;

  Status GetCanonicalInputIndex(const uint32_t index, int32_t &canonical_index) const;

  bool IsNoOp() const;

  bool IsControlFlowV2Op() const {
    return is_ctrl_flow_v2_op_;
  }

  bool IsControlFlowOp() const {
    return is_ctrl_flow_op_;
  }

  bool IsMergeOp() const {
    return is_merge_op_;
  }

  bool IsEnterOp() const {
    return kEnterOpTypes.count(node_type) > 0U;
  }

  bool IsExitOp() const {
    return kExitOpTypes.count(node_type) > 0U;
  }

  bool IsNextIterationOp() const {
    return kNextIterationOpTypes.count(node_type) > 0UL;
  }

  bool IsHcclOp() const;

  bool IsFftsSubNode() const {
    return is_ffts_sub_node_;
  }

  bool IsInputEnterFeedNode(const NodePtr &node) const ;

  void SetToDynamic();

  void SetDataSend(NodeItem *const node_item, int32_t anchor_index);
  void SetCtrlSend(NodeItem *const node_item, const uint32_t switch_index);
  void SetMergeCtrl(NodeItem *const node_item, const uint32_t merge_index);
  size_t GetMergeCtrl(const uint32_t merge_index) const;

  OptionalMutexGuard MutexGuard(const std::string &name) const {
    return OptionalMutexGuard(copy_mu_.get(), name + "_" + node_name);
  }

  std::string DebugString() const;

  NodePtr node;
  int64_t node_id = -1;
  int32_t num_inputs = 0;
  int32_t num_outputs = 0;
  int32_t input_start = -1;
  int32_t output_start = -1;
  bool has_observer = false;
  bool has_optional_inputs = false;
  std::vector<MemStorageType> output_mem_types_;
  std::vector<ge::NodePtr> dependents_for_shape_inference;
  std::vector<ge::NodePtr> dependents_for_execution;
  std::set<int32_t> to_const_output_id_list;
  // src_output_id, dst_anchor_id, dst_node
  std::vector<std::vector<std::pair<int32_t, NodeItem *>>> outputs;

  // for linked drive
  bool is_root_node_ = false;
  bool is_ctrl_flow_v2_op_ = false;
  bool is_ctrl_flow_op_ = false;
  bool is_merge_op_ = false;
  bool is_enter_active_ = false;
  bool is_ffts_sub_node_ = false;
  int64_t frame_index_ = -1;
  int64_t parent_frame_ = -1;
  std::set<const NodeItem *> root_ctrl_;  // Recv ctrl from root node
  std::map<const NodeItem *, std::set<int32_t>> root_data_;  // Recv data from root node
  std::set<const NodeItem *> enter_ctrl_; // Recv ctrl from Enter node
  std::map<const NodeItem *, std::set<int32_t>> enter_data_; // Recv data from Enter node
  std::set<const NodeItem *> data_send_;  // Send data notify to
  std::map<const NodeItem *, int32_t> data_recv_;  // Recv data notify from
  std::set<const NodeItem *> ctrl_send_;  // Send ctrl notify to
  std::set<const NodeItem *> ctrl_recv_;  // Recv ctrl notify from
  std::vector<std::set<const NodeItem *>> switch_groups_;  // Send ctrl notify to

  std::shared_ptr<NodeTask> kernel_task;
  std::unique_ptr<FusedSubgraph> fused_subgraph;
  const NodeExecutor *node_executor = nullptr;
  std::map<int32_t, ge::NodePtr> ref_outputs;
  std::map<int32_t, int32_t> reuse_inputs;
  std::map<int32_t, int32_t> reuse_outputs;
  int32_t num_static_input_shapes = 0;
  bool is_profiling_report = false;
  // Indexes which will skip checking the amount of input data
  std::unordered_set<int32_t> skip_sufficiency_of_input_check_;

 private:
  explicit NodeItem(NodePtr node_ptr);
  Status Init();
  Status InitInputsAndOutputs();
  void ResolveOptionalInputs();
  Status ResolveDynamicState();
  Status ResolveStaticInputsAndOutputs();
  void ResolveUnknownShapeType();
  void ResolveFftsPlusStatus();
  Status ResolveOutputMemType();
  GeTensorDescPtr DoGetInputDesc(const int32_t index) const;

  std::vector<uint32_t> input_desc_indices_;
  std::shared_ptr<std::mutex> copy_mu_;
  mutable std::mutex mu_;
};
}  // namespace hybrid
}  // namespace ge

#endif // GE_HYBRID_MODEL_NODE_ITEM_H_
