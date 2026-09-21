/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "executor_message.h"

namespace hccl {
using namespace std;
ExecutorMessage::ExecutorMessage()
    : isGatherAlltoAll_(false), statusCallback_(nullptr), messageType_(ExecutorMessageType::MSG_OPBASED) {}
ExecutorMessage::~ExecutorMessage() {}
void ExecutorMessage::SetOperationMessage(const HcomOperation_t &opInfo) {
  hcomOperationInfo_ = opInfo;
  messageType_ = ExecutorMessageType::MSG_OPBASED;
}

ExecutorMessageType ExecutorMessage::GetMessageType() {
  return messageType_;
}

HcomOperation_t ExecutorMessage::GetOperationMessage() const {
  return hcomOperationInfo_;
}

void ExecutorMessage::SetAlltoAllMessage(const HcomAllToAllVParams &opInfo, bool isGatherAlltoAll) {
  hcomAlltoAllInfo_ = opInfo;
  isGatherAlltoAll_ = isGatherAlltoAll;
  messageType_ = ExecutorMessageType::MSG_ALLTOALL;
}

HcomAllToAllVParams ExecutorMessage::GetAlltoAllMessage() const {
  return hcomAlltoAllInfo_;
}

void ExecutorMessage::SetAlltoAllVCMessage(const HcomAllToAllVCParams &opInfo) {
  hcomAlltoAllVCInfo_ = opInfo;
  messageType_ = ExecutorMessageType::MSG_ALLTOALLVC;
}

HcomAllToAllVCParams ExecutorMessage::GetAlltoAllVCMessage() const {
  return hcomAlltoAllVCInfo_;
}

void ExecutorMessage::SetStatusCallback(StatusCallback callBack) {
  statusCallback_ = callBack;
}

StatusCallback ExecutorMessage::GetStatusCallback() const {
  return statusCallback_;
}
}  // namespace hccl