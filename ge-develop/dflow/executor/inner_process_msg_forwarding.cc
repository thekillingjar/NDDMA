/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "inner_process_msg_forwarding.h"
#include "dflow/base/exec_runtime/execution_runtime.h"
#include "common/compile_profiling/ge_call_wrapper.h"

namespace ge {
namespace {
constexpr int32_t kDequeueMbufWaitTime = 3000;
}

InnerProcessMsgForwarding::~InnerProcessMsgForwarding() {
  Finalize();
}

Status InnerProcessMsgForwarding::Initialize(const std::vector<DeployQueueAttr> &status_input_queue_attrs)
{
  const auto execution_runtime = ExecutionRuntime::GetInstance();
  GE_CHECK_NOTNULL(execution_runtime);
  exchange_service_ = &execution_runtime->GetExchangeService();
  GE_CHECK_NOTNULL(exchange_service_);
  if (status_input_queue_attrs.empty()) {
    GELOGE(FAILED, "Status queue size can not be 0.");
    return FAILED;
  }
  status_input_queue_attrs_ = status_input_queue_attrs;
  return SUCCESS;
}

void InnerProcessMsgForwarding::Finalize() {
  run_flag_ = false;
  if (run_thread_.joinable()) {
    run_thread_.join();
  }
  const std::lock_guard<std::mutex> lock(func_mutex_);
  callbacks_.clear();
}

void InnerProcessMsgForwarding::Start() {
  run_flag_ = true;
  run_thread_ = std::thread([this]() {
    SET_THREAD_NAME(pthread_self(), "ge_innermsgfwd");
    Run();
  });
}

void InnerProcessMsgForwarding::Run() {
  while (run_flag_) {
    const uint32_t queue_id = status_input_queue_attrs_[0].queue_id;
    const int32_t device_id = status_input_queue_attrs_[0].device_id;
    GELOGD("Start to dequeue from status queue, device_id=%u, queue_id = %u.", device_id, queue_id);
    rtMbufPtr_t m_buf = nullptr;
    auto ret = exchange_service_->DequeueMbuf(device_id, queue_id, &m_buf, kDequeueMbufWaitTime);
    if (ret != SUCCESS) {
      GELOGW("Forwarding module dequeue mbuf failed, device_id=%u, queue_id = %u", device_id, queue_id);
      continue;
    }

    GELOGD("Inner process message forwarding dequeue message success");
    GE_MAKE_GUARD(m_buf, [m_buf]() { (void)rtMbufFree(m_buf); });
    void *buffer_addr = nullptr;
    uint64_t buffer_size = 0;
    if (rtMbufGetBuffAddr(m_buf, &buffer_addr) != RT_ERROR_NONE) {
      GELOGE(FAILED, "Get mbuf buffer addr failed, device_id=%u, queue_id = %u", device_id, queue_id);
      break;
    }
    if (rtMbufGetBuffSize(m_buf, &buffer_size) != RT_ERROR_NONE) {
      GELOGE(FAILED, "Get mbuf buffer size failed, device_id=%u, queue_id = %u", device_id, queue_id);
      break;
    }
    domi::SubmodelStatus sub_model_status;
    google::protobuf::io::ArrayInputStream stream(buffer_addr, static_cast<int32_t>(buffer_size));
    if (!sub_model_status.ParseFromZeroCopyStream(&stream)) {
      GELOGW("Proto message in status queue parser failed.");
      continue;
    }
    auto msg_type = static_cast<StatusQueueMsgType>(sub_model_status.msg_type());
    {
      const std::lock_guard<std::mutex> lock(func_mutex_);
      const auto iter = callbacks_.find(msg_type);
      if (iter == callbacks_.cend()) {
        GELOGW("Inner process message forwarding callback function not found, msg_type:%u",
               sub_model_status.msg_type());
        continue;
      }
      if(iter->second(sub_model_status) != SUCCESS) {
        GELOGW("Execute call back func failed for message type:%u.", sub_model_status.msg_type());
        continue;
      }
    }
    GELOGI("Call callback function for msg_type:%u finished.", sub_model_status.msg_type());
  }
  GELOGI("Inner process message forwarding thread exit.");
}

Status InnerProcessMsgForwarding::RegisterCallBackFunc(const StatusQueueMsgType& msg_type,
    const std::function<Status(const domi::SubmodelStatus&)> &func)
{
  const std::lock_guard<std::mutex> lock(func_mutex_);
  if (callbacks_.find(msg_type) == callbacks_.end()) {
    callbacks_[msg_type] = func;
    return SUCCESS;
  }
  GELOGE(FAILED, "RegisterCallBackFunc failed, msg_type:%u has already been registered",
         static_cast<uint32_t>(msg_type));
  return FAILED;
}
}
