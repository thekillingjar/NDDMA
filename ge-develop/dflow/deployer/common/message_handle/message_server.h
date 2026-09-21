/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_RUNTIME_HETEROGENEOUS_COMMON_MESSAGE_HANDLE_MESSAGE_SERVER_H
#define AIR_RUNTIME_HETEROGENEOUS_COMMON_MESSAGE_HANDLE_MESSAGE_SERVER_H

#include "proto/deployer.pb.h"
#include "ge/ge_api_error_codes.h"

namespace ge {
class MessageServer {
 public:
  MessageServer(int32_t device_id, uint32_t req_msg_queue_id, uint32_t rsp_msg_queue_id)
      : device_id_(device_id), req_msg_queue_id_(req_msg_queue_id), rsp_msg_queue_id_(rsp_msg_queue_id) {}
  virtual ~MessageServer() = default;

  Status Initialize() const;

  void Finalize() const;

  template <class Request>
  Status WaitRequest(Request &request, bool &is_finalize) const;

  template <class Response>
  Status SendResponse(const Response &response) const;

 private:
  Status InitMessageQueue() const;

  int32_t device_id_ = -1;
  uint32_t req_msg_queue_id_ = UINT32_MAX;
  uint32_t rsp_msg_queue_id_ = UINT32_MAX;
};
}  // namespace ge

#endif  // AIR_RUNTIME_HETEROGENEOUS_COMMON_MESSAGE_HANDLE_MESSAGE_SERVER_H
