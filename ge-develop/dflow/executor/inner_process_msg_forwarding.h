/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef EXECUTOR_GRAPH_LOAD_MODEL_MANAGER_INNER_PROCESS_MSG_FORWARDING_H_
#define EXECUTOR_GRAPH_LOAD_MODEL_MANAGER_INNER_PROCESS_MSG_FORWARDING_H_
#include <thread>
#include <map>
#include <mutex>
#include "ge/ge_api_types.h"
#include "dflow/base/deploy/exchange_service.h"
#include "proto/task.pb.h"

namespace ge {
enum class StatusQueueMsgType : uint32_t {
    STATUS = 0,
    EXCEPTION = 1,
    INVALID = 2
};
class InnerProcessMsgForwarding {
 public:
  InnerProcessMsgForwarding() = default;
  ~InnerProcessMsgForwarding();
  Status Initialize(const std::vector<DeployQueueAttr> &status_input_queue_attrs);
  void Finalize();
  void Start();
  Status RegisterCallBackFunc(const StatusQueueMsgType &msg_type,
                              const std::function<Status(const domi::SubmodelStatus&)> &func);
  
 private:
  void Run();

  std::mutex func_mutex_;
  std::map<StatusQueueMsgType, std::function<Status(const domi::SubmodelStatus&)>>  callbacks_;
  ExchangeService *exchange_service_ = nullptr;
  std::atomic_bool run_flag_{false};
  std::thread run_thread_;
  std::vector<DeployQueueAttr> status_input_queue_attrs_; // 状态接受队列
};
}  // namespace ge
#endif  // EXECUTOR_GRAPH_LOAD_MODEL_MANAGER_INNER_PROCESS_MSG_FORWARDING_H_
