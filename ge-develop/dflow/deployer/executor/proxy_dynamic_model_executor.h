/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_RUNTIME_DEPLOY_EXECUTOR_PROXY_DYNAMIC_MODEL_EXECUTOR_H_
#define AIR_RUNTIME_DEPLOY_EXECUTOR_PROXY_DYNAMIC_MODEL_EXECUTOR_H_

#include "executor/dynamic_model_executor.h"
#include "executor/npu_sched_model_loader.h"

namespace ge {
class ProxyDynamicModelExecutor : public DynamicModelExecutor {
 public:
  ProxyDynamicModelExecutor();
  ~ProxyDynamicModelExecutor() override = default;
  Status Initialize() override;
  void Finalize() override;
  Status ClearModel(const int32_t clear_type) override;
  Status ExceptionNotify(uint32_t type, uint64_t trans_id) override;
  uint32_t GetRuntimeModelId() const;
 protected:
  Status LoadWithAicpuSd(const ComputeGraphPtr &root_graph, const ModelQueueParam &model_queue_param) override;
  Status UnloadFromAicpuSd() override;
  Status CheckInputs() override;
  Status PrepareInputs(std::vector<DataBuffer> &model_inputs) override;
  Status PrepareOutputs(std::vector<DataBuffer> &model_outputs) override;
  Status UpdateOutputs(std::vector<DataBuffer> &model_outputs) override;
  void UpdateFusionInputsAddr() override;
  Status PublishOutputWithoutExecute() override;
  void PublishErrorOutput(Status ret) override;
  Status StartDispatcherThread();
  virtual void Dispatcher();
  Status PrepareMsgMbuf(rtMbufPtr_t &req_msg_mbuf, rtMbufPtr_t &resp_msg_mbuf) const;
  Status DoDequeue(const int32_t device_id, const uint32_t queue_id,
                   rtMbufPtr_t &mbuf, const int32_t timeout) const;
  virtual Status DequeueMbuf(const int32_t device_id, const uint32_t queue_id,
                             rtMbufPtr_t &mbuf, const int32_t timeout) const;
  Status EnqueueMbuf(const int32_t device_id, const uint32_t queue_id,
                     rtMbufPtr_t mbuf, const int32_t timeout) const;
  virtual Status OnInputsReady(rtMbufPtr_t req_msg_mbuf, rtMbufPtr_t resp_msg_mbuf);
  Status OnModelExecuted(const Status status, rtMbufPtr_t req_msg_mbuf, rtMbufPtr_t resp_msg_mbuf) const;
 private:
  Status SetNpuModelLoaderOutputInfo();

  int32_t req_msg_queue_id_ = -1;   // -1: invalid queue id
  int32_t resp_msg_queue_id_ = -1;  // -1: invalid queue id
  std::thread dispatcher_thread_;
  std::atomic_bool dispatcher_running_flag_{};
  NpuSchedModelLoader loader_;
  uint64_t req_inputs_len_ = 0UL;   // inputs total len in req mbuf
  uint64_t req_outputs_len_ = 0UL;  // outputs total len in req mbuf
  uint64_t resp_total_len_ = 0UL;   // resp total len
  uint32_t runtime_model_id_ = 0U;
  std::string model_name_;
};
}
#endif  // AIR_RUNTIME_DEPLOY_EXECUTOR_PROXY_DYNAMIC_MODEL_EXECUTOR_H_
