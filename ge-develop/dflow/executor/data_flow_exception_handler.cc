/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "data_flow_exception_handler.h"
#include "dflow/base/deploy/exchange_service.h"
#include "common/compile_profiling/ge_call_wrapper.h"

namespace ge {
namespace {
constexpr const char *kModelIOExceptionScope = "";
constexpr size_t kMaxExceptionCacheNum = 1024;
}  // namespace
DataFlowExceptionHandler::DataFlowExceptionHandler(
    std::function<void(const UserExceptionNotify &)> exception_notify_callback)
    : exception_notify_callback_(std::move(exception_notify_callback)) {}

DataFlowExceptionHandler::~DataFlowExceptionHandler() {
  Finalize();
}

Status DataFlowExceptionHandler::Initialize(InnerProcessMsgForwarding &process_forwarding) {
  if (exception_notify_callback_ == nullptr) {
    return SUCCESS;
  }
  process_thread_ = std::thread([this]() {
    SET_THREAD_NAME(pthread_self(), "ge_dpl_exhd");
    Process();
  });
  auto func = [this](const domi::SubmodelStatus &request) {
    return NotifyException(request.exception());
  };
  return process_forwarding.RegisterCallBackFunc(StatusQueueMsgType::EXCEPTION, func);
}

void DataFlowExceptionHandler::Finalize() {
  process_queue_.Stop();
  if (process_thread_.joinable()) {
    process_thread_.join();
  }
}

void DataFlowExceptionHandler::Process() {
  GELOGI("process exception thread start.");
  domi::DataFlowException data_flow_exception{};
  while (process_queue_.Pop(data_flow_exception)) {
    ProcessException(data_flow_exception);
  }
  GELOGI("process exception thread exit.");
}

Status DataFlowExceptionHandler::NotifyException(const domi::DataFlowException &data_flow_exception) {
  GELOGI("receive exception, trans_id=%" PRIu64 ", scope=%s, exception_code=%d, user_context_id=%" PRIu64,
         data_flow_exception.trans_id(), data_flow_exception.scope().c_str(), data_flow_exception.exception_code(),
         data_flow_exception.user_context_id());
  GE_CHK_BOOL_RET_STATUS(process_queue_.Push(data_flow_exception), INTERNAL_ERROR,
                         "Failed to enqueue exception, trans_id=%" PRIu64
                         ", scope=%s, exception_code=%d, user_context_id=%" PRIu64 ".",
                         data_flow_exception.trans_id(), data_flow_exception.scope().c_str(),
                         data_flow_exception.exception_code(), data_flow_exception.user_context_id());
  return SUCCESS;
}

void DataFlowExceptionHandler::ProcessException(const domi::DataFlowException &data_flow_exception) {
  uint64_t trans_id = data_flow_exception.trans_id();
  const auto &scope = data_flow_exception.scope();
  {
    std::lock_guard<std::mutex> guard(mt_for_all_exception_);
    if (all_exceptions_.find({trans_id, scope}) != all_exceptions_.cend()) {
      GELOGI("receive repeat exception, trans_id=%" PRIu64 ", scope=%s", trans_id, scope.c_str());
      return;
    }
    if (all_exceptions_.size() >= kMaxExceptionCacheNum) {
      auto &expired = all_exceptions_.cbegin()->second;
      GELOGI("over max, expire the oldest exception, trans_id=%" PRIu64 ", scope=%s, "
             "exception_code=%d, user_context_id=%" PRIu64 ".",
             expired.trans_id(), expired.scope().c_str(), expired.exception_code(), expired.user_context_id());
      NotifyModelIO(expired, kExceptionTypeExpired);
      NotifyExecutor(expired, kExceptionTypeExpired);
      (void)all_exceptions_.erase(all_exceptions_.begin());
    }
    all_exceptions_[{trans_id, scope}] = data_flow_exception;
  }
  GELOGI("notify exception, trans_id=%" PRIu64 ", scope=%s, exception_code=%d, user_context_id=%" PRIu64 ".",
         data_flow_exception.trans_id(), data_flow_exception.scope().c_str(), data_flow_exception.exception_code(),
         data_flow_exception.user_context_id());
  NotifyModelIO(data_flow_exception, kExceptionTypeOccured);
  NotifyExecutor(data_flow_exception, kExceptionTypeOccured);
}

bool DataFlowExceptionHandler::IsModelIoIgnoreTransId(uint64_t trans_id) {
  std::lock_guard<std::mutex> guard(mt_for_model_io_);
  return model_io_exception_all_.find(trans_id) != model_io_exception_all_.cend();
}

bool DataFlowExceptionHandler::TakeWaitModelIoException(DataFlowInfo &info) {
  uint64_t trans_id = 0;
  {
    std::lock_guard<std::mutex> guard(mt_for_model_io_);
    if (model_io_exception_wait_report_.empty()) {
      return false;
    }
    trans_id = *(model_io_exception_wait_report_.cbegin());
    (void)model_io_exception_wait_report_.erase(trans_id);
  }
  {
    std::lock_guard<std::mutex> guard(mt_for_all_exception_);
    auto find_ret = all_exceptions_.find({trans_id, kModelIOExceptionScope});
    if (find_ret == all_exceptions_.cend()) {
      GELOGW("model io exception can't found, trans_id=%" PRIu64, trans_id);
      return false;
    }
    const auto &exception_info = find_ret->second;
    const auto &exception_context = exception_info.exception_context();
    const auto *exception_context_buf = exception_context.c_str();
    size_t exception_context_len = exception_context.size();
    if (exception_context_len >= sizeof(ExchangeService::MsgInfo)) {
      const auto *msg_info = reinterpret_cast<const ExchangeService::MsgInfo *>(
          exception_context_buf + (exception_context_len - sizeof(ExchangeService::MsgInfo)));
      info.SetStartTime(msg_info->start_time);
      info.SetEndTime(msg_info->end_time);
      info.SetFlowFlags(msg_info->flags);
    }
    if (exception_context_len >= kMaxUserDataSize) {
      (void)info.SetUserData(exception_context_buf, kMaxUserDataSize);
    }
    GELOGI("find model io exception, trans_id=%" PRIu64 ", exception_code=%d, user_context_id=%" PRIu64 ".", trans_id,
           exception_info.exception_code(), exception_info.user_context_id());
  }
  return true;
}

void DataFlowExceptionHandler::NotifyModelIO(const domi::DataFlowException &data_flow_exception, uint32_t type) {
  uint64_t trans_id = data_flow_exception.trans_id();
  const auto &scope = data_flow_exception.scope();
  if (scope != kModelIOExceptionScope) {
    return;
  }
  {
    std::lock_guard<std::mutex> guard(mt_for_model_io_);
    if (type == kExceptionTypeOccured) {
      model_io_exception_all_.emplace(trans_id);
      model_io_exception_wait_report_.emplace(trans_id);
    } else {
      model_io_exception_all_.erase(trans_id);
      model_io_exception_wait_report_.erase(trans_id);
    }
    for (const auto &callback : model_io_callback_list_) {
      callback(trans_id, type);
    }
  }
}

void DataFlowExceptionHandler::NotifyExecutor(const domi::DataFlowException &data_flow_exception, uint32_t type) {
  UserExceptionNotify notify{};
  notify.type = type;
  notify.trans_id = data_flow_exception.trans_id();
  notify.scope = data_flow_exception.scope();
  notify.exception_code = data_flow_exception.exception_code();
  notify.user_context_id = data_flow_exception.user_context_id();
  notify.exception_context = data_flow_exception.exception_context().c_str();
  notify.exception_context_len = data_flow_exception.exception_context().size();
  exception_notify_callback_(notify);
}
void DataFlowExceptionHandler::RegisterModelIoExpTransIdCallback(const ModelIoExpTransIdCallbackFunc &callback) {
  std::lock_guard<std::mutex> guard(mt_for_model_io_);
  model_io_callback_list_.emplace_back(callback);
}
}  // namespace ge
