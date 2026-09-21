/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_SINGLE_OP_SINGLE_OP_MODEL_H_
#define GE_SINGLE_OP_SINGLE_OP_MODEL_H_

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "framework/common/helper/model_helper.h"
#include "single_op/single_op.h"
#include "single_op/single_op_impl.h"
#include "single_op/stream_resource.h"
#include "single_op/task/op_task.h"

namespace ge {
struct SingleOpModelParam {
  RuntimeParam runtime_param;
  std::map<uintptr_t, int32_t> addr_mapping_;
  int64_t core_type = 0;
  bool graph_is_dynamic = false;
  fe::PlatFormInfos platform_infos;
  std::shared_ptr<gert::OpImplSpaceRegistryV2Array> space_registries_;
};

class SingleOpModel {
 public:
  SingleOpModel(const std::string &model_name, const void *const model_data, const uint32_t model_size);
  ~SingleOpModel() = default;

  Status Init();
  Status BuildOp(StreamResource &resource, SingleOpImpl &single_op);
  Status BuildDynamicOp(StreamResource &resource, DynamicSingleOpImpl &single_op);

 private:
  Status InitModel();
  Status LoadRootGraph();
  Status ParseInputsAndOutputs();
  Status SetInputsAndOutputs(SingleOpImpl &single_op);
  Status InitModelMem(StreamResource &resource);
  Status InitOverflowAddr(StreamResource &resource) const;
  Status ParseInputNode(const OpDescPtr &op_desc);
  void ParseOutputNode(const OpDescPtr &op_desc);

  Status BuildTaskList(StreamResource &stream_resource, SingleOpImpl &single_op);
  Status BuildTaskListForDynamicOp(StreamResource &stream_resource, DynamicSingleOpImpl &single_op);
  Status BuildKernelAndExTask(const domi::TaskDef &task_def, OpTask *&op_task, StreamResource &stream_resource);
  Status BuildTEKernelAndTask(const domi::TaskDef &task_def, OpTask *&op_task, StreamResource &stream_resource);

  Status BuildKernelTask(const domi::TaskDef &task_def, TbeOpTask **const task, StreamResource &stream_resource);
  Status BuildAtomicTask(const domi::TaskDef &task_def, AtomicAddrCleanOpTask **const task,
                         const StreamResource &stream_resource);
  Status BuildKernelExTask(const domi::KernelExDef &kernel_def, AiCpuTask **const task, const uint64_t kernel_id);
  Status BuildCpuKernelTask(const domi::KernelDef &kernel_def, AiCpuCCTask **const task, const uint64_t kernel_id);

  Status BuildMixL2KernelTask(const domi::TaskDef &task_def, OpTask *&op_task, StreamResource &stream_resource);
  Status ParseOpModelParams();
  void ParseArgTable(OpTask *const task, SingleOpImpl &op);
  void SetHostMemTensorAndNode(DynamicSingleOpImpl &single_op) const;
  Status NeedHybridModel(bool &need_hybrid_model);
  Status ParseTasks();
  Status SetHostMemNode(std::vector<NodePtr> &node_with_hostmem);
  void SetHostMemInputFlagToTask(const OpDescPtr &op_desc, OpTask &task) const;
  Status MallocWeight(StreamResource &resource);

  std::map<NodePtr, std::vector<domi::TaskDef>> node_tasks_;
  std::set<NodePtr> built_nodes_;
  std::string model_name_;
  uint32_t model_id_ = 0U;
  const void *ori_model_data_;
  uint32_t ori_model_size_;

  ModelHelper model_helper_;
  GeModelPtr root_ge_model_;
  ComputeGraphPtr root_graph_;

  std::map<uint32_t, NodePtr> op_list_;
  std::map<int32_t, NodePtr> op_with_hostmem_;
  SingleOpModelParam model_params_;

  std::vector<ptrdiff_t> input_offset_list_;
  std::vector<size_t> input_sizes_;
  std::vector<ptrdiff_t> output_offset_list_;
  std::vector<size_t> output_sizes_;
  std::vector<OpDescPtr> data_ops_;
  OpDescPtr netoutput_op_;
  bool has_weight_ = false;
};
}  // namespace ge

#endif  // GE_SINGLE_OP_SINGLE_OP_MODEL_H_
