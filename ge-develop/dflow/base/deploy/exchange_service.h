/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef BASE_EXEC_RUNTIME_DEPLOY_EXCHANGE_SERVICE_H_
#define BASE_EXEC_RUNTIME_DEPLOY_EXCHANGE_SERVICE_H_

#include <functional>
#include <memory>
#include "ge/ge_api_error_codes.h"
#include "graph/ge_tensor.h"
#include "framework/common/debug/ge_log.h"
#include "runtime/rt.h"

namespace ge {
// bit 0
constexpr uint8_t kNullDataFlagBit = 1U;
// bit 1 means whether take
constexpr uint8_t kCustomTransIdFlagBit = 1U << 1;
constexpr size_t kMaxUserDataSize = 64U;
struct DeployQueueAttr {
  uint32_t queue_id;
  int32_t device_id;
  int32_t device_type;
  uint32_t global_logic_id;
  std::string DebugString() const {
    return "queue:" + std::to_string(queue_id) +
           ", device_id:" + std::to_string(device_id) +
           ", device_type:" + std::to_string(device_type);
  }
  std::string GetKey() const {
    return std::to_string(queue_id) + "_" + std::to_string(device_id) + "_" +
           std::to_string(device_type);
  }
  bool operator < (const DeployQueueAttr &other) const {
    return DebugString() < other.DebugString();
  }
};
struct MemQueueAttr {
  uint32_t depth;
  uint32_t work_mode;
  bool is_client = false;
  bool overwrite = false;
};
/// Interfaces for data exchange operations
class ExchangeService {
 public:
  using FillFunc = std::function<Status(void *buffer, size_t size)>;
  struct MsgInfo {
    uint64_t trans_id;
    uint16_t version;
    uint16_t msg_type;
    int32_t ret_code;
    uint64_t start_time;
    uint64_t end_time;
    uint32_t flags;
    uint8_t data_flag;  // 0 bit is null data flag, 1 is null data, 0 is not null data
    char rsv0[3];
    int32_t worker_id;
    int32_t step_id;
    char rsv[8];
    uint32_t data_label;
    uint32_t route_label;
  };
  struct ControlInfo {
    bool end_of_sequence_flag = false;
    int32_t timeout = 0;
    size_t skip_size = 0U;
    MsgInfo *msg_info = nullptr;
    bool is_shared_input = false;
    int8_t user_data[kMaxUserDataSize] = {};
    bool print_error_flag = true;  // true: print error when queue is full
  };
  struct BuffInfo {
    void *addr;
    size_t len;
  };
  ExchangeService() = default;
  ExchangeService(const ExchangeService &) = delete;
  ExchangeService &operator=(const ExchangeService &) = delete;
  virtual ~ExchangeService() = default;

  Status CreateQueue(const int32_t device_id,
                     const std::string &name,
                     const uint32_t depth,
                     const uint32_t work_mode,
                     uint32_t &queue_id) {
    MemQueueAttr mem_queue_attr{};
    mem_queue_attr.depth = depth;
    mem_queue_attr.work_mode = work_mode;
    mem_queue_attr.overwrite = false;
    return CreateQueue(device_id, name, mem_queue_attr, queue_id);
  }

  virtual Status CreateQueue(const int32_t device_id,
                             const std::string &name,
                             const MemQueueAttr &mem_queue_attr,
                             uint32_t &queue_id) = 0;
  virtual Status DestroyQueue(const int32_t device_id, const uint32_t queue_id) = 0;
  virtual Status Enqueue(const int32_t device_id, const uint32_t queue_id, const void *const data,
                         const size_t size, const ControlInfo &control_info) = 0;
  virtual Status Enqueue(int32_t device_id, uint32_t queue_id, size_t size, rtMbufPtr_t m_buf,
                         const ControlInfo &control_info) = 0;
  virtual Status Enqueue(const int32_t device_id, const uint32_t queue_id, const size_t size,
                         const FillFunc &fill_func, const ControlInfo &control_info) = 0;
  virtual Status Enqueue(const int32_t device_id, const uint32_t queue_id, const std::vector<BuffInfo> &buffs,
                         const ControlInfo &control_info) = 0;
  virtual Status EnqueueMbuf(int32_t device_id, uint32_t queue_id, rtMbufPtr_t m_buf, int32_t timeout) = 0;
  virtual Status Dequeue(const int32_t device_id, const uint32_t queue_id, void *const data, const size_t size,
                         ControlInfo &control_info) = 0;
  virtual Status DequeueMbufTensor(const int32_t device_id, const uint32_t queue_id,
                                   std::shared_ptr<AlignedPtr> &aligned_ptr, const size_t size,
                                   ControlInfo &control_info) = 0;
  virtual Status DequeueTensor(const int32_t device_id, const uint32_t queue_id, GeTensor &tensor,
                               ControlInfo &control_info) = 0;
  virtual Status DequeueMbuf(int32_t device_id, uint32_t queue_id, rtMbufPtr_t *m_buf, int32_t timeout) = 0;
  virtual void ResetQueueInfo(const int32_t device_id, const uint32_t queue_id) = 0;
};
}
#endif  // BASE_EXEC_RUNTIME_DEPLOY_EXCHANGE_SERVICE_H_
