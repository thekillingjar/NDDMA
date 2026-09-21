/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "common/message_handle/message_server.h"
#include "common/utils/rts_api_utils.h"
#include "common/data_flow/queue/heterogeneous_exchange_service.h"
#include "graph_metadef/common/ge_common/util.h"

namespace ge {
namespace {
constexpr int32_t kDefaultTimeout = 10 * 1000;  // 10s
constexpr uint16_t kMsgTypeFinalize = 2U;
using HService = HeterogeneousExchangeService;
}  // namespace

Status MessageServer::Initialize() const {
  GE_CHK_STATUS_RET_NOLOG(RtsApiUtils::MemQueueInit(device_id_));
  GE_CHK_STATUS_RET(InitMessageQueue(), "Failed to init message queue");
  GELOGI("[Initialize] message server success");
  return SUCCESS;
}

void MessageServer::Finalize() const {
  (void)HService::GetInstance().Finalize();
}

Status MessageServer::InitMessageQueue() const {
  GE_CHK_STATUS_RET(RtsApiUtils::MemQueueAttach(0, req_msg_queue_id_, kDefaultTimeout),
                    "[Attach][ReqMsgQueue] failed, queue_id = %u", req_msg_queue_id_);
  GE_CHK_STATUS_RET(RtsApiUtils::MemQueueAttach(0, rsp_msg_queue_id_, kDefaultTimeout),
                    "[Attach][RspMsgQueue] failed, queue_id = %u", rsp_msg_queue_id_);
  GELOGD("[Init][MessageQueue] succeeded, request queue_id = %u, response queue_id = %u", req_msg_queue_id_,
         rsp_msg_queue_id_);
  return SUCCESS;
}

template <class Request>
Status MessageServer::WaitRequest(Request &request, bool &is_finalize) const {
  rtMbufPtr_t mbuf = nullptr;
  // max wait 1s per time.
  constexpr int32_t kWaitRequestPerTime = 1 * 1000;
  auto ret = HService::GetInstance().DequeueMbuf(device_id_, req_msg_queue_id_, &mbuf, kWaitRequestPerTime);
  if (ret == RT_ERROR_TO_GE_STATUS(ACL_ERROR_RT_QUEUE_EMPTY)) {
    GELOGD("No message was received");
    return ret;
  }
  GE_MAKE_GUARD(mbuf, [mbuf]() { GE_CHK_RT(rtMbufFree(mbuf)); });
  ExchangeService::ControlInfo control_info{};
  ExchangeService::MsgInfo msg_info{};
  control_info.msg_info = &msg_info;
  GE_CHK_STATUS_RET(HService::CheckResult(mbuf, control_info), "Failed to check msg head");
  if (control_info.msg_info->msg_type == kMsgTypeFinalize) {
    is_finalize = true;
    return SUCCESS;
  }
  void *buffer_addr = nullptr;
  uint64_t buffer_size = 0;
  GE_CHK_STATUS_RET_NOLOG(RtsApiUtils::MbufGetBufferAddr(mbuf, &buffer_addr));
  GE_CHK_STATUS_RET_NOLOG(RtsApiUtils::MbufGetBufferSize(mbuf, buffer_size));
  google::protobuf::io::ArrayInputStream stream(buffer_addr, static_cast<int32_t>(buffer_size));
  GE_CHK_BOOL_RET_STATUS(request.ParseFromZeroCopyStream(&stream), FAILED, "[Parse][Message] failed");
  GELOGD("[Wait][Request] succeeded");
  return SUCCESS;
}

template <class Response>
Status MessageServer::SendResponse(const Response &response) const {
  auto rsp_size = response.ByteSizeLong();
  ExchangeService::FillFunc fill_func = [&response](void *buffer, size_t size) {
    GE_CHK_BOOL_RET_STATUS(response.SerializeToArray(buffer, static_cast<int32_t>(size)), FAILED,
                           "SerializeToArray failed");
    return SUCCESS;
  };
  ExchangeService::ControlInfo control_info{};
  control_info.timeout = kDefaultTimeout;
  GE_CHK_STATUS_RET(HService::GetInstance().Enqueue(device_id_, rsp_msg_queue_id_, rsp_size, fill_func, control_info),
                    "Failed to enqueue response");
  GELOGD("[Send][Response] succeeded");
  return SUCCESS;
}

template Status MessageServer::WaitRequest(deployer::DeployerRequest &, bool &) const;
template Status MessageServer::WaitRequest(deployer::ExecutorRequest &, bool &) const;
template Status MessageServer::SendResponse(const deployer::DeployerResponse &) const;
template Status MessageServer::SendResponse(const deployer::ExecutorResponse &) const;
}  // namespace ge
