/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef EXECUTOR_GRAPH_LOAD_MODEL_MANAGER_AICPU_RESOURCES_H_
#define EXECUTOR_GRAPH_LOAD_MODEL_MANAGER_AICPU_RESOURCES_H_

#include <mutex>

#include "ge/ge_api_types.h"
#include "common/plugin/ge_make_unique_util.h"
#include "graph/node.h"
#include "framework/common/debug/log.h"

namespace ge {
class AiCpuResources {
 public:
  AiCpuResources() = default;
  ~AiCpuResources() noexcept;
  GE_DELETE_ASSIGN_AND_COPY(AiCpuResources);

  static Status Resize(std::vector<uint8_t> &task_args, size_t args_size) {
    try {
      task_args.resize(args_size);
    } catch (...) {
      GELOGE(FAILED, "Failed to resize task args to %zu", args_size);
      return FAILED;
    }
    return SUCCESS;
  }
  static const std::string &ResourceTypeQueue();
  static const std::string &ResourceTypeChannel();
  static const std::string &ResourceTypeVdecChannel();
  Status AllocateChannelResource(const OpDescPtr &op_desc,
                                 const int32_t rt_stream_id);
  Status AllocateQueueResource(const OpDescPtr &op_desc,
                               const NamedAttrs &resource_attr,
                               int32_t &input_idx,
                               uint32_t &queue_id);
  Status AllocateVdecChannelResource(const OpDescPtr &op_desc,
                                     const int32_t rt_stream_id);
  Status GetOrCreateQueue(const std::string &queue_name, const uint32_t queue_depth, uint32_t &queue_id);
  void ReleaseResources();

  struct AiCpuModelConfig {
    int32_t version;
    uint32_t model_id;
    uint32_t runtime_model_id;
    int32_t abnormal_exist;
    int32_t abnormal_enqueue;
    int32_t req_msg_queue;
    int32_t resp_msg_queue;
    int8_t rsv[36];
  };
  struct AiCpuModelShapeConfig {
    int32_t version;
    uint32_t model_id;
    uint32_t runtime_model_id;
    uint32_t data_len;
    uint64_t data_device_addr;
    int8_t rsv[40];
  };

  Status SetModelConfig(const AiCpuModelConfig &config) const;
  Status SetStaticModelShapeConfig(const AiCpuModelShapeConfig &config,
                                   const std::vector<InputOutputDescInfo> &input_desc_list);
  bool GetStaticModelShapeConfigRet() const;

 private:
  static Status CreateQueue(const std::string &name, const uint32_t depth, uint32_t &queue_id);
  static Status DestroyQueue(const uint32_t queue_id);

  static Status BuildCreateQueueTask(const uintptr_t queue_id_dev,
                                     const std::string &name,
                                     const uint32_t depth,
                                     std::vector<uint8_t> &task_args);
  static Status BuildDestroyQueueTask(const uint32_t queue_id, std::vector<uint8_t> &task_args);

  static Status CreateChannel(const int32_t rt_stream_id);
  static Status DestroyChannel(const int32_t rt_stream_id);
  static Status BuildCreateChannelTask(const int32_t rt_stream_id, std::vector<uint8_t> &task_args);
  static Status BuildDestroyChannelTask(const int32_t rt_stream_id, std::vector<uint8_t> &task_args);

  static Status CreateVdecChannel(const int32_t rt_stream_id);
  static Status BuildCreateVdecChannelTask(const int32_t rt_stream_id, std::vector<uint8_t> &task_args);
  static Status DestroyVdecChannel(const int32_t rt_stream_id);
  static Status BuildDestroyVdecChannelTask(const int32_t rt_stream_id, std::vector<uint8_t> &task_args);

  static Status BuildModelConfigTask(const AiCpuModelConfig &config, std::vector<uint8_t> &task_args);
  static Status BuildModelShapeConfigTask(const AiCpuModelShapeConfig &config, std::vector<uint8_t> &task_args);

  static Status ExecuteKernel(const char_t *const so_name,
                              const std::string &kernel_name,
                              const std::vector<uint8_t> &task_args);
  static Status ExecuteKernel(const std::string &kernel_name, const std::vector<uint8_t> &task_args);

  std::mutex mu_;
  std::map<std::string, uint32_t> aicpu_queues_;
  std::set<int32_t> aicpu_channels_;
  std::set<int32_t> aicpu_vdec_channels_;
  bool static_model_shape_config_result_ = false;
};
}  // namespace ge
#endif  // EXECUTOR_GRAPH_LOAD_MODEL_MANAGER_AICPU_RESOURCES_H_
