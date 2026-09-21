/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_RUNTIME_DEPLOY_EXECUTOR_NPU_SCHED_MODEL_LOADER_H
#define AIR_RUNTIME_DEPLOY_EXECUTOR_NPU_SCHED_MODEL_LOADER_H

#include <mutex>
#include "ge/ge_api_types.h"
#include "common/plugin/ge_make_unique_util.h"
#include "executor/sched_task_info.h"
#include "acl/acl.h"

namespace ge {
using SchedTaskInfoPtr = std::shared_ptr<SchedTaskInfo>;

class NpuSchedModelLoader {
 public:
  NpuSchedModelLoader() = default;
  virtual ~NpuSchedModelLoader() noexcept = default;
  GE_DELETE_ASSIGN_AND_COPY(NpuSchedModelLoader);
  Status LoadModel(const ModelQueueParam &model_queue_param, uint32_t &runtime_model_id);
  Status UnloadModel();
  void SetModelId(const uint32_t model_id);
  void SetDeviceId(int32_t device_id);
  void SetAlignAttributes(const uint32_t align_interval, const std::vector<uint32_t> &align_offsets);
  void SetOutputTensorSizes(const std::vector<int64_t> &output_tensor_sizes);
  void SetOutputDynamicFlags(const std::vector<uint32_t> &output_dynamic_flags);
  void SetOutputQueueIds(const std::vector<uint32_t> &output_queue_ids);
  void SetInputDynamicFlags(const std::vector<bool> &input_dynamic_flags);
  void SetOutputStaticTensorDescs(const std::vector<RuntimeTensorDesc> &output_static_tensor_descs);
  void SetGlobalStep(const uint64_t global_step);
  void SetInputAlignAttrs(const InputAlignAttrs &input_align_attrs) { input_align_attrs_ = input_align_attrs; }
  void SetEnablePostProcessV2Flag(const bool enable);
  void SetSkipMarkStep(bool skip_mark_step);
  uint32_t GetReqMsgQueueId() const;
  uint32_t GetRespMsgQueueId() const;

 private:
  Status SetModelConfig() const;
  Status CreateSchedTasks();
  Status DistributeTasks() const;
  Status DistributeEndGraph();
  Status BindInputQueue(const aclrtStream stream);
  Status CreateModelBatchDequeueTask(const aclrtStream stream, const std::vector<uint32_t> &queue_ids,
                                     const uint32_t align_interval, const std::vector<uint32_t> &align_offsets,
                                     std::vector<uint64_t> &mbuf_addrs);
  Status CreateModelGatherDequeueTask(const aclrtStream stream, const std::vector<QueueAttrs> &queues,
                                      std::vector<uint64_t> &mbuf_addrs);
  Status UpdateFusionInputsMbufAddr(const std::vector<uint64_t> &unique_input_mbuf_addrs);
  Status CreatePrepareDynamicInputOutputTask(const aclrtStream stream);
  Status CreateEnqueueTask(const aclrtStream stream, const uint32_t queue_id, const uint64_t mbuf_addr);
  uint32_t GenerateNotifyId() const;
  Status CreateNotifyRecordTask(const aclrtStream stream, const uint32_t notify_id);
  Status CreateNotifyWaitTask(const aclrtStream stream, const uint32_t notify_id);
  Status CreateStreamRepeatTask(const aclrtStream stream);
  Status CreateZeroCopyTask(const aclrtStream stream, const std::vector<uint64_t> &src_addrs,
                            std::vector<uint64_t> &dst_addrs);
  Status CreatePostprocessDynamicOutputTask(const aclrtStream stream);
  Status CreateDequeueTask(const aclrtStream stream, const uint32_t queue_id, uint64_t &mbuf_addr);
  Status BindOutputQueue(const aclrtStream stream);
  Status CreateModelBatchEnqueueTask(const aclrtStream stream, const std::vector<uint32_t> &queue_ids,
                                     const std::vector<uint64_t> &mbuf_addrs);
  Status CreateMarkStepTask(const aclrtStream stream);
  Status UnbindStreams();
  Status ReleaseTasks();
  Status CreateMsgQueues();
  Status CreateQueue(const int32_t device_id, const std::string &name, uint32_t &queue_id);
  Status DestroyQueue(const int32_t device_id, const uint32_t queue_id) const;
  Status EnsureQueueResourceInitialized(const int32_t device_id);

 private:
  aclmdlRI rt_model_handle_ = nullptr;
  uint32_t model_id_ = 0U;
  uint32_t runtime_model_id_ = 0U;
  aclrtStream rt_entry_stream_ = nullptr;
  aclrtStream rt_next_stream_ = nullptr;
  aclrtStream rt_fake_stream_ = nullptr;
  std::vector<SchedTaskInfoPtr> sched_tasks_;
  bool has_align_attr_ = false;
  bool enable_post_process_v2_ = false;
  uint32_t align_interval_ = 0U;
  std::vector<uint32_t> align_offsets_;
  ModelQueueParam model_queue_param_;
  std::vector<uint64_t> input_mbuf_addrs_;
  std::vector<uint64_t> output_mbuf_addrs_;
  std::vector<int64_t> output_tensor_sizes_;
  std::vector<uint32_t> output_queue_ids_;
  std::vector<uint32_t> input_dynamic_flags_;
  std::vector<uint32_t> output_dynamic_flags_;
  RuntimeTensorDesc *output_static_tensor_descs_ = nullptr;
  size_t output_static_tensor_num_ = 0UL;
  std::vector<uint64_t> postproc_input_mbuf_addrs_;
  std::vector<uint64_t> postproc_output_mbuf_addrs_;
  uint64_t req_msg_mbuf_addr_ = 0UL;
  uint32_t req_msg_queue_id_ = 0U;
  uint64_t resp_msg_mbuf_addr_ = 0UL;
  uint32_t resp_msg_queue_id_ = 0U;
  uint64_t global_step_ = 0UL;
  InputAlignAttrs input_align_attrs_{};
  bool skip_mark_step_ = false;
  int32_t device_id_ = 0;
  std::mutex queue_res_init_mutex_;
  bool queue_res_init_flag_ = false;
};
}
#endif  // AIR_RUNTIME_DEPLOY_EXECUTOR_NPU_SCHED_MODEL_LOADER_H
