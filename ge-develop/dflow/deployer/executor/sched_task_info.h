/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_RUNTIME_DEPLOY_EXECUTOR_SCHED_TASK_INFO_H
#define AIR_RUNTIME_DEPLOY_EXECUTOR_SCHED_TASK_INFO_H

#include <cstdint>
#include "ge/ge_api_types.h"
#include "framework/common/ge_types.h"
#include "framework/common/runtime_tensor_desc.h"
#include "acl/acl.h"

namespace ge {
struct AicpuNotifyKernelArgs {
  volatile uint32_t notify_id;  // notify id
  volatile uint16_t ret_code;
};

struct StreamRepeatKernelArgs {
  uint32_t model_id;   // model id
  uint32_t stream_id;  // stream id
};

struct BatchQueueInfoKernelArgs {
  uint32_t queue_num;           // queue number
  uint32_t align_interval;      // align interval
  uint64_t align_offsets_addr;  // address of uint32_t array witch size is inputs_num and value is align offset
  uint64_t queue_ids_addr;      // address of uint32_t array witch size is inputs_num and value is queue id
  uint64_t mbuf_addrs_addr;     // address of uint64_t array witch size is inputs_num and value is mbuf addr
};

struct QueueInfoKernelArgs {
  uint32_t queue_id;   // queue id
  uint64_t mbuf_addr;  // mbuf addr
};

struct AddrMapInfoKernelArgs {
  uint32_t addr_num{0U};        // address number
  uint64_t src_addr_list{0UL};  // address of uint64_t array witch size is addr_num and value is src addr
  uint64_t dst_addr_list{0UL};  // address of uint64_t array witch size is addr_num and value is dst addr
};

struct MarkStepKernelArgs {
  uint32_t group_total_count{1U};
  uint32_t group_index{0U};
  uint32_t group_policy{0U};   // load balance policy
  uint64_t step_id_addr{0UL};  // current step id addr
  uint64_t reserved[32]{0UL};
  char_t dump_step[1024]{'\0'};
};

#pragma pack(push, 1)
struct PrepareDynamicInputOutputKernelArgs {
  uint32_t inputs_num;                 // inputs number
  uint32_t outputs_num;                // outputs number
  uint64_t input_dynamic_flags_addr;   // address of uint32_t array witch size is inputs_num and value is dynamic flag
  uint64_t output_tensor_sizes_addr;   // address of int64_t array witch size is outputs_num and value is tensor size
  uint64_t input_mbuf_addrs_addr;      // address of uint64_t array witch size is inputs_num and value is mbuf addr
  uint64_t output_mbuf_addrs_addr;     // address of uint64_t array witch size is outputs_num and value is mbuf addr
  uint64_t input_fusion_offsets_addr;  // address of uint32_t array witch size is inputs_num and value is fusion offset
  uint64_t req_msg_mbuf_addr;          // request message mbuf addr
};

struct GatherDequeueParam {
  uint32_t input_nums;
  int32_t inputs_align_timeout;
  uint32_t inputs_align_max_cache_num;
  uint32_t inputs_align_drop_out;
  uint64_t queue_ids_addr;  // uint32_t
  uint64_t mbuf_addrs_addr; // uintptr
  uint64_t queue_device_ids_addr;  // uint32
  uint64_t queue_device_type_addr; // uint32 0 NPU 1 CPU
};

struct PostprocessDynamicOutputKernelArgs {
  uint32_t inputs_num;                 // inputs number
  uint32_t outputs_num;                // outputs number
  uint64_t resp_msg_mbuf_addr;         // response message mbuf addr
  uint64_t input_mbuf_addrs_addr;      // address of uint64_t array witch size is inputs_num and value is mbuf addr,
                                       // used for release input mbuf, mbuf need unique
  uint64_t output_mbuf_addrs_addr;     // address of uint64_t array witch size is outputs_num and value is mbuf addr
  uint64_t output_dynamic_flags_addr;  // address of uint32_t array witch size is outputs_num and value is dynamic flag
  uint64_t output_static_tensor_desc_addr;  // address of uint64_t array witch value is static output tensor desc and
                                            // size is static outputs num
};
#pragma pack(pop)

class SchedTaskInfo {
 public:
  explicit SchedTaskInfo(aclrtStream const stream);
  virtual ~SchedTaskInfo() noexcept = default;
  virtual Status Distribute() = 0;
  virtual Status Release();

 protected:
  Status LaunchCpuKernel(const char *kernel_name) const;
  void *args_ = nullptr;
  uint64_t args_size_ = 0UL;
  aclrtStream stream_ = nullptr;

 private:
  SchedTaskInfo &operator=(const SchedTaskInfo &) & = delete;
  SchedTaskInfo(const SchedTaskInfo &) = delete;
};

class SchedTaskModelDequeue : public SchedTaskInfo {
 public:
  explicit SchedTaskModelDequeue(aclrtStream const stream) : SchedTaskInfo(stream) {}
  ~SchedTaskModelDequeue() override = default;
  Status Init(const uint32_t queue_id, uint64_t &mbuf_addr);
  Status Distribute() override;
};

class SchedTaskModelEnqueue : public SchedTaskInfo {
 public:
  explicit SchedTaskModelEnqueue(aclrtStream const stream) : SchedTaskInfo(stream) {}
  ~SchedTaskModelEnqueue() override = default;
  Status Init(const uint32_t queue_id, const uint64_t mbuf_addr);
  Status Distribute() override;
};

class SchedTaskModelBatchDequeue : public SchedTaskInfo {
 public:
  explicit SchedTaskModelBatchDequeue(aclrtStream const stream) : SchedTaskInfo(stream) {}
  ~SchedTaskModelBatchDequeue() override = default;
  Status Init(const std::vector<uint32_t> &queue_ids, const uint32_t align_interval,
              const std::vector<uint32_t> &align_offsets, std::vector<uint64_t> &mbuf_addrs);
  Status Distribute() override;
};

class SchedTaskModelGatherDequeue : public SchedTaskInfo {
 public:
  explicit SchedTaskModelGatherDequeue(aclrtStream const stream) : SchedTaskInfo(stream) {}
  ~SchedTaskModelGatherDequeue() override = default;
  Status Init(const std::vector<QueueAttrs> &queues, const InputAlignAttrs &input_align_attrs,
             std::vector<uint64_t> &mbuf_addrs);
  Status Distribute() override;
};

class SchedTaskPrepareDynamicInputOutput : public SchedTaskInfo {
 public:
  explicit SchedTaskPrepareDynamicInputOutput(aclrtStream const stream) : SchedTaskInfo(stream) {}
  ~SchedTaskPrepareDynamicInputOutput() override = default;
  Status Init(const std::vector<uint32_t> &input_dynamic_flags, const std::vector<uint64_t> &input_mbuf_addrs,
              const std::vector<int32_t> &input_fusion_offsets, const std::vector<int64_t> &output_tensor_sizes,
              std::vector<uint64_t> &output_mbuf_addrs, uint64_t &req_msg_mbuf_addr, const bool enable_v2 = false);
  Status Distribute() override;
 private:
  bool enable_v2_ = false;
};

class SchedTaskModelBatchEnqueue : public SchedTaskInfo {
 public:
  explicit SchedTaskModelBatchEnqueue(aclrtStream const stream) : SchedTaskInfo(stream) {};
  ~SchedTaskModelBatchEnqueue() override = default;
  Status Init(const std::vector<uint32_t> &queue_ids, const std::vector<uint64_t> &mbuf_addrs);
  Status Distribute() override;
};

class SchedTaskPostprocessDynamicOutput : public SchedTaskInfo {
 public:
  explicit SchedTaskPostprocessDynamicOutput(aclrtStream const stream) : SchedTaskInfo(stream) {};
  ~SchedTaskPostprocessDynamicOutput() override = default;
  Status Init(const uint64_t resp_msg_mbuf_addr, const std::vector<uint64_t> &input_mbuf_addrs,
              const std::vector<uint64_t> &output_mbuf_addrs, const std::vector<uint32_t> &output_dynamic_flags,
              const RuntimeTensorDesc *output_static_tensor_descs, const size_t output_static_tensor_num,
              const bool enable_v2 = false);
  Status Distribute() override;
 private:
  bool enable_v2_ = false;
};

class SchedTaskNotifyWait : public SchedTaskInfo {
 public:
  explicit SchedTaskNotifyWait(aclrtStream const stream) : SchedTaskInfo(stream) {};
  ~SchedTaskNotifyWait() override = default;
  Status Init(const uint32_t notify_id);
  Status Distribute() override;
};

class SchedTaskNotifyRecord : public SchedTaskNotifyWait {
 public:
  explicit SchedTaskNotifyRecord(aclrtStream const stream) : SchedTaskNotifyWait(stream) {};
  ~SchedTaskNotifyRecord() override = default;
  Status Distribute() override;
};

class SchedTaskZeroCopy : public SchedTaskInfo {
 public:
  explicit SchedTaskZeroCopy(aclrtStream const stream) : SchedTaskInfo(stream) {};
  ~SchedTaskZeroCopy() override = default;
  Status Init(const std::vector<uint64_t> &src_addrs, std::vector<uint64_t> &dst_addrs);
  Status Distribute() override;
};

class SchedTaskStreamRepeat : public SchedTaskInfo {
 public:
  explicit SchedTaskStreamRepeat(aclrtStream const stream) : SchedTaskInfo(stream) {};
  ~SchedTaskStreamRepeat() override = default;
  Status Init(const uint32_t model_id);
  Status Distribute() override;
};

class SchedTaskMarkStep : public SchedTaskInfo {
 public:
  explicit SchedTaskMarkStep(aclrtStream const stream) : SchedTaskInfo(stream) {};
  ~SchedTaskMarkStep() override = default;
  Status Init(const uint32_t group_total_count, const uint32_t group_index, const uint32_t group_policy,
              const std::string &dump_step, const uint64_t step_id_addr);
  Status Distribute() override;
};
}
#endif  // AIR_RUNTIME_DEPLOY_EXECUTOR_SCHED_TASK_INFO_H
