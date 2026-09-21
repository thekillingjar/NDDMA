/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_RUNTIME_HETEROGENEOUS_COMMON_MESSAGE_HANDLE_MESSAGE_CLIENT_H
#define AIR_RUNTIME_HETEROGENEOUS_COMMON_MESSAGE_HANDLE_MESSAGE_CLIENT_H

#include <mutex>
#include <thread>
#include <condition_variable>
#include "proto/deployer.pb.h"
#include "common/config/device_debug_config.h"
#include "ge/ge_api_error_codes.h"

namespace ge {
template <class Request, class Response>
class MessageClient {
 public:
  explicit MessageClient(int32_t device_id, bool parallel_send = false);
  virtual ~MessageClient();
  Status CreateMessageQueue(const std::string &name_suffix, uint32_t &request_qid, uint32_t &response_qid,
                            bool is_client = false);
  Status Initialize(int32_t pid, const std::function<Status(void)> &get_stat_func, bool waiting_rsp = true);
  Status NotifyFinalize() const;
  void Stop();
  Status Finalize();
  virtual Status SendRequestWithoutResponse(const Request &request);
  // when parallel_send = true, response need set the same message id as the request message id;
  // when parallel_send = false, default message id is 0
  virtual Status SendRequest(const Request &request, Response &response, int64_t timeout = -1);

 protected:
  virtual Status WaitForProcessInitialized();
  // timeout: 单位s
  virtual Status WaitResponseWithMessageId(Response &response, uint64_t message_id = 0UL, int64_t timeout = -1);
  virtual Status WaitResponse(Response &response, int64_t timeout = -1);
  virtual void DequeueMessageThread();
  virtual Status DequeueMessage(std::shared_ptr<Response> &response);

 private:
  Status InitMessageQueue() const;
  void SetMessageId(Request &request);

  volatile bool running_ = false;
  int32_t pid_ = -1;
  int32_t device_id_ = -1;
  bool parallel_send_ = false;
  std::mutex mu_;
  std::condition_variable response_cv_;
  uint32_t req_msg_queue_id_ = UINT32_MAX;
  uint32_t rsp_msg_queue_id_ = UINT32_MAX;
  std::function<Status(void)> get_stat_func_;
  std::thread wait_rsp_thread_;
  std::map<uint64_t, std::shared_ptr<Response>> responses_received_;
  std::atomic<uint64_t> message_id_{0UL};
};

using DeployerMessageClient = MessageClient<deployer::DeployerRequest, deployer::DeployerResponse>;
using ExecutorMessageClient = MessageClient<deployer::ExecutorRequest, deployer::ExecutorResponse>;
}  // namespace ge

#endif  // AIR_RUNTIME_HETEROGENEOUS_COMMON_MESSAGE_HANDLE_MESSAGE_CLIENT_H
