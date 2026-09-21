/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "common/message_handle/message_client.h"
#include "common/utils/rts_api_utils.h"
#include "common/util/mem_utils.h"
#include "common/data_flow/queue/heterogeneous_exchange_service.h"
#include "common/compile_profiling/ge_call_wrapper.h"
#include "acl/acl.h"
#include "common/df_chk.h"

namespace ge {
namespace {
constexpr int64_t kDefaultTimeout = 1200;       // s
constexpr int32_t kDequeueTimeout = 3000;       // ms
constexpr int32_t kDequeueTimeoutInSec = 3;       // s
constexpr int32_t kEnqueueTimeout = 30 * 1000;  // ms
constexpr int64_t kDefaultRetryTimes = 400;
constexpr uint32_t kMsgQueueDepth = 3U;
constexpr uint16_t kMsgTypeFinalize = 2U;
using HService = HeterogeneousExchangeService;
}  // namespace

template <class Request, class Response>
MessageClient<Request, Response>::MessageClient(int32_t device_id, bool parallel_send)
    : device_id_(device_id), parallel_send_(parallel_send) {}

template <class Request, class Response>
MessageClient<Request, Response>::~MessageClient() {
  (void)Finalize();
}

template <class Request, class Response>
void MessageClient<Request, Response>::Stop() {
  running_ = false;
}

template <class Request, class Response>
Status MessageClient<Request, Response>::Finalize() {
  if (pid_ == 0) {
    return SUCCESS;
  }

  running_ = false;
  if (parallel_send_ && wait_rsp_thread_.joinable()) {
    wait_rsp_thread_.join();
  }
  (void)HService::GetInstance().DestroyQueue(device_id_, req_msg_queue_id_);
  (void)HService::GetInstance().DestroyQueue(device_id_, rsp_msg_queue_id_);
  req_msg_queue_id_ = UINT32_MAX;
  rsp_msg_queue_id_ = UINT32_MAX;
  pid_ = 0;
  responses_received_.clear();
  return SUCCESS;
}

template <class Request, class Response>
Status MessageClient<Request, Response>::NotifyFinalize() const {
  ExchangeService::ControlInfo control_info{};
  control_info.timeout = kEnqueueTimeout;
  ExchangeService::MsgInfo msg_info{};
  msg_info.msg_type = kMsgTypeFinalize;
  control_info.msg_info = &msg_info;
  uint8_t data = 0U;
  size_t data_size = sizeof(data);
  GE_CHK_STATUS_RET(HService::GetInstance().Enqueue(device_id_, req_msg_queue_id_, &data, data_size, control_info),
                    "Failed to enqueue request");
  return SUCCESS;
}

template <class Request, class Response>
Status MessageClient<Request, Response>::CreateMessageQueue(const std::string &name_suffix, uint32_t &request_qid,
                                                            uint32_t &response_qid, bool is_client) {
  if (is_client) {
    DF_CHK_ACL_RET(aclrtSetDevice(device_id_));
  }
  MemQueueAttr mem_queue_attr{};
  mem_queue_attr.depth = kMsgQueueDepth;
  mem_queue_attr.work_mode = RT_MQ_MODE_PULL;
  mem_queue_attr.is_client = is_client;
  const std::string req_msg_queue_name = "request_queue." + name_suffix;
  GE_CHK_STATUS_RET(HService::GetInstance().CreateQueue(device_id_, req_msg_queue_name, mem_queue_attr, request_qid),
                    "[Create][Request] message queue failed, queue name %s.", req_msg_queue_name.c_str());

  const std::string rsp_msg_queue_name = "response_queue." + name_suffix;
  const auto ret = HService::GetInstance().CreateQueue(device_id_, rsp_msg_queue_name, mem_queue_attr, response_qid);
  if (ret != SUCCESS) {
    GELOGE(FAILED, "[Create][Reponse] message queue failed, queue name %s.", rsp_msg_queue_name.c_str());
    (void)HService::GetInstance().DestroyQueue(device_id_, request_qid);
    return FAILED;
  }
  req_msg_queue_id_ = request_qid;
  rsp_msg_queue_id_ = response_qid;
  GELOGI("[Create][Message] queue success, request qid:%u [%s], response qid:%u [%s]", request_qid,
         req_msg_queue_name.c_str(), response_qid, rsp_msg_queue_name.c_str());
  return SUCCESS;
}

template <class Request, class Response>
Status MessageClient<Request, Response>::Initialize(int32_t pid, const std::function<Status(void)> &get_stat_func,
                                                    bool waiting_rsp) {
  pid_ = pid;
  get_stat_func_ = get_stat_func;
  GE_CHECK_NOTNULL(get_stat_func_);
  GE_CHK_STATUS_RET_NOLOG(InitMessageQueue());
  running_ = true;
  if (parallel_send_) {
    wait_rsp_thread_ = std::thread(&MessageClient::DequeueMessageThread, this);
  }
  if (waiting_rsp) {
    GE_CHK_STATUS_RET_NOLOG(WaitForProcessInitialized());
  }
  GELOGI("[Initialize] message client succeeded, pid:%d, device_id:%d", pid_, device_id_);
  return SUCCESS;
}

template <class Request, class Response>
Status MessageClient<Request, Response>::InitMessageQueue() const {
  rtMemQueueShareAttr_t req_queue_attr{};
  req_queue_attr.read = 1;
  GE_CHK_RT_RET(rtMemQueueGrant(device_id_, req_msg_queue_id_, pid_, &req_queue_attr));
  rtMemQueueShareAttr_t rsp_queue_attr{};
  rsp_queue_attr.write = 1;
  GE_CHK_RT_RET(rtMemQueueGrant(device_id_, rsp_msg_queue_id_, pid_, &rsp_queue_attr));
  return SUCCESS;
}

template <class Request, class Response>
Status MessageClient<Request, Response>::DequeueMessage(std::shared_ptr<Response> &response) {
  rtMbufPtr_t mbuf = nullptr;
  auto ret = HService::GetInstance().DequeueMbuf(device_id_, rsp_msg_queue_id_, &mbuf, kDequeueTimeout);
  if (ret == SUCCESS) {
    GE_MAKE_GUARD(mbuf, [mbuf]() { (void)rtMbufFree(mbuf); });
    void *buffer_addr = nullptr;
    uint64_t buffer_size = 0;
    GE_CHK_STATUS_RET_NOLOG(RtsApiUtils::MbufGetBufferAddr(mbuf, &buffer_addr));
    GE_CHK_STATUS_RET_NOLOG(RtsApiUtils::MbufGetBufferSize(mbuf, buffer_size));
    google::protobuf::io::ArrayInputStream stream(buffer_addr, static_cast<int32_t>(buffer_size));
    auto rsp = MakeShared<Response>();
    GE_CHECK_NOTNULL(rsp);
    GE_CHK_BOOL_RET_STATUS(rsp->ParseFromZeroCopyStream(&stream), FAILED, "Failed to parse message from proto");
    response = rsp;
  }
  return ret;
}

template <class Request, class Response>
void MessageClient<Request, Response>::DequeueMessageThread() {
  SET_THREAD_NAME(pthread_self(), "ge_dpl_msgc");
  while (running_) {
    std::shared_ptr<Response> response;
    auto ret = DequeueMessage(response);
    if (ret == SUCCESS) {
      uint64_t message_id = response->message_id();
      std::lock_guard<std::mutex> lk(mu_);
      responses_received_[message_id] = response;
      response_cv_.notify_all();
      GELOGD("[Dequeue][Message] success, queue id:%u, message id:%lu", rsp_msg_queue_id_, message_id);
      continue;
    }
    if (ret != RT_ERROR_TO_GE_STATUS(ACL_ERROR_RT_QUEUE_EMPTY)) {
      GELOGE(FAILED, "Failed to dequeue message");
      return;
    }
  }
  GEEVENT("Dequeue Message thread end");
}

template <class Request, class Response>
Status MessageClient<Request, Response>::WaitForProcessInitialized() {
  Response response;
  if (parallel_send_) {
    GE_CHK_STATUS_RET(WaitResponseWithMessageId(response), "Failed to wait response");
  } else {
    GE_CHK_STATUS_RET(WaitResponse(response), "Failed to wait response");
  }
  GE_CHK_BOOL_RET_STATUS(response.error_code() == SUCCESS, FAILED, "[Wait][Response] failed, error_message = %s",
                         response.error_message().c_str());
  GEEVENT("[Wait][Process] initialized succeeded, pid:%d", pid_);
  return SUCCESS;
}

template <class Request, class Response>
void MessageClient<Request, Response>::SetMessageId(Request &request) {
  request.set_message_id(message_id_++);
}

template <class Request, class Response>
Status MessageClient<Request, Response>::SendRequestWithoutResponse(const Request &request) {
  auto req_size = request.ByteSizeLong();
  ExchangeService::FillFunc fill_func = [&request](void *buffer, size_t size) {
    GE_CHK_BOOL_RET_STATUS(request.SerializeToArray(buffer, static_cast<int32_t>(size)), FAILED,
                           "SerializeToArray failed, request size:%zu", size);
    return SUCCESS;
  };
  ExchangeService::ControlInfo control_info{};
  control_info.timeout = kEnqueueTimeout;
  GE_CHK_STATUS_RET(HService::GetInstance().Enqueue(device_id_, req_msg_queue_id_, req_size, fill_func, control_info),
                    "Failed to enqueue request");
  GELOGD("[Enqueue][Request] without response succeeded, queue id:%u", req_msg_queue_id_);
  return SUCCESS;
}

template <class Request, class Response>
Status MessageClient<Request, Response>::SendRequest(const Request &request, Response &response, int64_t timeout) {
  if (parallel_send_) {
    // only grpc deployer requests are sent in parallel now
    SetMessageId(const_cast<Request &>(request));
  }
  auto req_size = request.ByteSizeLong();
  ExchangeService::FillFunc fill_func = [&request, &response](void *buffer, size_t size) {
    if (request.SerializeToArray(buffer, static_cast<int32_t>(size))) {
      return SUCCESS;
    }
    response.set_error_code(FAILED);
    response.set_error_message("SerializeToArray failed");
    GELOGE(FAILED, "SerializeToArray failed");
    return FAILED;
  };
  ExchangeService::ControlInfo control_info{};
  control_info.timeout = kEnqueueTimeout;
  GE_CHK_STATUS_RET(HService::GetInstance().Enqueue(device_id_, req_msg_queue_id_, req_size, fill_func, control_info),
                    "Failed to enqueue request");
  GELOGD("[Enqueue][Request] succeeded, queue id:%u, message id:%lu", req_msg_queue_id_, request.message_id());
  if (parallel_send_) {
    GE_CHK_STATUS_RET(WaitResponseWithMessageId(response, request.message_id(), timeout),
                      "Failed to wait response, message id:%lu", request.message_id());
  } else {
    GE_CHK_STATUS_RET(WaitResponse(response, timeout), "Failed to wait response");
  }
  GELOGD("[Dequeue][Response] succeeded, queue id:%u, message id:%lu", rsp_msg_queue_id_, response.message_id());
  return SUCCESS;
}

template <class Request, class Response>
Status MessageClient<Request, Response>::WaitResponseWithMessageId(Response &response, uint64_t message_id,
                                                                   int64_t timeout) {
  const int64_t rsp_timeout = (timeout == -1) ? kDefaultTimeout : timeout;
  std::unique_lock<std::mutex> lk(mu_);
  GE_CHK_STATUS_RET(get_stat_func_(), "Process already exit");
  response_cv_.wait_for(lk, std::chrono::seconds(rsp_timeout), [this, message_id] {
    return (!running_) || (responses_received_.find(message_id) != responses_received_.cend());
  });
  if (responses_received_.find(message_id) != responses_received_.cend()) {
    GELOGD("[Wait][Message] success, queue id:%u, message id:%lu", rsp_msg_queue_id_, message_id);
    response = std::move(*responses_received_.at(message_id));
    responses_received_.erase(message_id);
    return SUCCESS;
  }
  GELOGE(FAILED, "Wait response timeout, wait time = %ld s", rsp_timeout);
  return FAILED;
}

template <class Request, class Response>
Status MessageClient<Request, Response>::WaitResponse(Response &response, int64_t timeout) {
  std::shared_ptr<Response> rsp;
  const int64_t retry_times = (timeout == -1) ? kDefaultRetryTimes : std::max(timeout / kDequeueTimeoutInSec, 1L);
  for (int32_t i = 0; i < retry_times; ++i) {
    GE_CHK_STATUS_RET(get_stat_func_(), "Process already exit");
    GE_CHK_BOOL_RET_STATUS(running_, FAILED, "Wait response failed as stopped");
    auto ret = DequeueMessage(rsp);
    if (ret == SUCCESS) {
      response = *rsp;
      return SUCCESS;
    }
    GE_CHK_BOOL_RET_STATUS(ret == RT_ERROR_TO_GE_STATUS(ACL_ERROR_RT_QUEUE_EMPTY), FAILED, "[Dequeue][Message] failed");
    GELOGD("Response message queue is empty, retry time = %d.", i);
  }
  GELOGE(FAILED, "Wait response timeout, wait time = %d ms.", retry_times * kDequeueTimeout);
  return FAILED;
}

template class MessageClient<deployer::DeployerRequest, deployer::DeployerResponse>;
template class MessageClient<deployer::ExecutorRequest, deployer::ExecutorResponse>;
}  // namespace ge
