/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_RUNTIME_DEPLOY_EXECUTOR_CPU_SCHED_MODEL_BUILDER_H_
#define AIR_RUNTIME_DEPLOY_EXECUTOR_CPU_SCHED_MODEL_BUILDER_H_

#include <atomic>
#include <vector>
#include "executor/cpu_sched_model.h"
#include "framework/common/ge_types.h"
#include "graph/ge_context.h"
#include "graph/ge_tensor.h"

namespace ge {
class CpuSchedModelBuilder {
 public:
  explicit CpuSchedModelBuilder(CpuSchedModel &model);
  virtual ~CpuSchedModelBuilder() = default;

  virtual Status Build();
  void SetModelId(uint32_t model_id);
  void SetAicpuStreamId(int32_t stream_id);
  void SetModelQueueParam(const ModelQueueParam &model_queue_param);
  void AddInputQueue(QueueAttrs queue_attrs, uintptr_t mbuf_addr);
  void AddOutputQueue(QueueAttrs queue_attrs, uintptr_t mbuf_addr);
  void AddInputClientQueue(QueueAttrs queue_attrs, uintptr_t mbuf_addr);
  void AddOutputClientQueue(QueueAttrs queue_attrs, uintptr_t mbuf_addr);
  void SetAlignAttributes(uint32_t align_interval, const std::vector<uint32_t> &align_offsets);
  void SetGlobalStep(uint64_t global_step);
  void SetInputBufferAddrs(const std::vector<void *> &input_buf_addresses);
  void SetOutputBufferAddrs(const std::vector<void *> &output_buf_addresses);
  void SetInputTensor(const std::vector<GeTensorDesc> &input_tensor_descs);
  void SetOutputTensor(const std::vector<GeTensorDesc> &output_tensor_descs);
  void SetIsHost(const bool is_host) { is_host_ = is_host; }
  void SetInputAlignAttrs(const InputAlignAttrs &input_align_attrs) { input_align_attrs_ = input_align_attrs; }

 protected:
  void AddDequeueTasks(uint32_t stream_id);
  Status AddMarkStepTask(uint32_t stream_id, bool is_head);
  void AddEnqueueTasks(uint32_t stream_id);
  void AddActivateTask(uint32_t stream_id);
  void AddWaitEndGraph(uint32_t stream_id);
  void AddModelRepeat(uint32_t stream_id);
  void AddQueueOpTask(const char_t *kernel_name, uint32_t queue_id, uintptr_t mbuf_addr, uint32_t stream_id);
  void AddQueueBuffOpTask(const char_t *kernel_name, const QueueAttrs &queue_attrs,
                          uintptr_t mbuf_addr, uint32_t stream_id);
  void AddBatchDequeueOpTask(uint32_t stream_id);
  void AddBatchDequeueBuffOpTask(uint32_t stream_id);
  void AddGatherDequeueTask(uint32_t stream_id);
  void AddDequeueTasksByAttrs(uint32_t stream_id);

  uint8_t *NewTask(const char_t *kernel_name, size_t param_size, uint32_t stream_id);
  uint32_t GenerateStreamId() const;

 protected:
  CpuSchedModel &model_;

  bool has_align_attr_ = false;
  uint32_t align_interval_ = 0;
  bool is_host_ = false;
  std::vector<uint32_t> align_offsets_;
  std::vector<std::pair<QueueAttrs, uintptr_t>> input_local_queue_infos_; // executor and queue both in host or device
  std::vector<std::pair<QueueAttrs, uintptr_t>> output_local_queue_infos_; // executor and queue both in host or device
  std::vector<std::pair<QueueAttrs, uintptr_t>> input_client_queue_infos_;  // executor on host and queue in device
  std::vector<std::pair<QueueAttrs, uintptr_t>> output_client_queue_infos_; // executor on host and queue in device
  std::vector<void *> input_buf_addresses_;
  std::vector<void *> output_buf_addresses_;
  const GeTensorDesc *input_tensor_descs_{};
  const GeTensorDesc *output_tensor_descs_{};
  uint32_t task_id_gen_ = 0;
  ModelQueueParam model_queue_param_;
  uint64_t global_step_ = 0UL;
  int32_t aicpu_stream_id_ = 0;
  InputAlignAttrs input_align_attrs_{};
};
} // namespace ge

#endif  // AIR_RUNTIME_DEPLOY_EXECUTOR_CPU_SCHED_MODEL_BUILDER_H_
