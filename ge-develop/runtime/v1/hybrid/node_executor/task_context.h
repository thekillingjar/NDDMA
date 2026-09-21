/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_HYBRID_KERNEL_TASK_CONTEXT_H_
#define GE_HYBRID_KERNEL_TASK_CONTEXT_H_

#include <map>
#include <mutex>
#include <vector>
#include "common/dump/dump_properties.h"
#include "common/dump/exception_dumper.h"
#include "hybrid/common/tensor_value.h"
#include "hybrid/common/npu_memory_allocator.h"
#include "hybrid/executor/node_state.h"
#include "hybrid/executor/rt_callback_manager.h"
#include "hybrid/model/node_item.h"

namespace ge {
namespace hybrid {
class TaskContext {
 public:
  static std::unique_ptr<TaskContext> Create(NodeState *const node_state, SubgraphContext *const subgraph_context);

  ~TaskContext();
  void Reset();
  int32_t NumInputs() const;
  int32_t NumOutputs() const;
  int32_t NumWorkspaces() const;
  const NodeItem &GetNodeItem() const;
  NodeState *GetNodeState() const;
  const char_t *GetNodeName() const;
  TensorValue *MutableInput(const int32_t idx) const;
  ConstGeTensorDescPtr GetInputDesc(const int32_t index) const;
  Status GetInputDesc(const int32_t index, GeTensorDesc &tensor_desc) const;
  ConstGeTensorDescPtr GetOutputDesc(const int32_t index) const;
  GeTensorDescPtr MutableInputDesc(const int32_t index) const;
  GeTensorDescPtr MutableOutputDesc(const int32_t index) const;
  Status UpdateInputDesc(const int32_t index, const GeTensorDesc &tensor_desc) const;
  bool NeedCallback() const;
  void ReleaseInput(const int32_t index);
  void ReleaseOutput(const int32_t index);
  void ReleaseAllOutput();
  void ReleaseWorkspace();
  void ReleaseAllMem();
  const TensorValue *GetInput(const int32_t idx) const;
  const TensorValue *GetOutput(const int32_t idx) const;
  TensorValue *MutableOutput(const int32_t idx) const;
  TensorValue *GetVariable(const std::string &name) const;
  rtStream_t GetStream() const;

  void NodeDone();
  void OnError(const Status error) const;

  Status SetOutput(const int32_t index, const TensorValue &tensor_in) const;
  bool HasAllocated(const int32_t idx) const;
  Status AllocateOutput(const int32_t idx,
                        const GeTensorDesc &tensor_desc_in,
                        TensorValue **const tensor_out,
                        const AllocationAttr * const attr = nullptr) const;
  Status AllocateOutputs(AllocationAttr *const attr = nullptr) const;
  Status AllocateWorkspaces();
  Status AllocateWorkspace(const size_t alloc_size, void *&alloc_buffer, void *const ori_addr = nullptr);

  bool IsTraceEnabled() const;

  bool IsDumpEnabled() const;

  const DumpProperties& GetDumpProperties() const;

  GraphExecutionContext *GetExecutionContext() const {
    return execution_context_;
  }

  void *MutableWorkspace(const int32_t idx) const;

  Status RegisterCallback(const std::function<void()> &callback_fun) const;
  Status TryExecuteCallback(const std::function<void()> &callback_fun) const;

  Status PropagateOutputs() const;

  void SetStatus(const Status stat);

  uint32_t GetTaskId() const;

  uint32_t GetStreamId() const;
  
  uint32_t *MutableTaskId() {
    return &task_id_;
  }

  uint32_t *MutableStreamId() {
    return &stream_id_;
  }

  void SetOverFlow(const bool over_flow);
  bool IsOverFlow() const;

  Status Synchronize() const;

  bool IsForceInferShape() const;
  void SetForceInferShape(const bool force_infer_shape);

  const std::vector<TaskDescInfo>& GetProfilingTaskDescInfo() const { return task_desc_info; }
  Status SaveProfilingTaskDescInfo(const std::string &task_type,  const uint32_t block_dim,
                                   const std::string &op_type);
  void ClearProfilingTaskDescInfo() { task_desc_info.clear(); }

  bool SkipSufficiencyOfInputCheck(const int32_t index) const {
    return skip_sufficiency_of_input_check_.count(index) != 0ULL;
  }

  ExtraOpInfo &MutableExtraOpInfo() {
    return extra_op_info_;
  }

  TaskContext(GraphExecutionContext *const execution_context,
              NodeState *const node_state,
              SubgraphContext *const subgraph_context);
  void SetContextHandle(void *const handle) { handle_ = handle; }
  void *GetContextHandle() const { return handle_; }
 private:
  void *handle_ = nullptr;
  static std::string TensorDesc2String(const GeTensorDesc &desc);
  Status AllocateTensor(const GeTensorDesc &tensor_desc_in, TensorValue &tensor_out,
                        const AllocationAttr * const attr) const;

  NodeState *const node_state_;
  const NodeItem *const node_item_;
  GraphExecutionContext *const execution_context_;
  SubgraphContext *const subgraph_context_;
  TensorValue *inputs_start_;
  TensorValue *outputs_start_;
  Status status_ = SUCCESS;
  std::vector<void *> workspaces_;
  bool force_infer_shape_ = false;
  uint32_t task_id_ = 0U;
  uint32_t stream_id_ = 0U;
  std::vector<TaskDescInfo> task_desc_info;
  bool is_over_flow_ = false;
  // Indexes which will skip checking the amount of input data
  const std::unordered_set<int32_t> skip_sufficiency_of_input_check_;
  // dynamic info related
  ExtraOpInfo extra_op_info_;
};
}  // namespace hybrid
}  // namespace ge
#endif // GE_HYBRID_KERNEL_TASK_CONTEXT_H_
