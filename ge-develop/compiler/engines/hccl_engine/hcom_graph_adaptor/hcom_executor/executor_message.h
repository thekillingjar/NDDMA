/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef EXECUTOR_MESSAGE_H
#define EXECUTOR_MESSAGE_H

#include <mutex>
#include <map>
#include <queue>
#include <string>
#include <functional>
#include "hccl/hccl_types.h"
#include "hccl/base.h"

namespace hccl {
using HcomOperation_t = struct HcomOperation;
using HcomRemoteOperation_t = struct HcomRemoteOperation;
using StatusCallback = std::function<void(HcclResult status)>;
using HcclRtEvent = void *;

struct RemoteAccessInfo {
  std::string remoteAccessType;
};

enum class ExecutorMessageType {
  MSG_OPBASED = 0,
  MSG_REMOTE_ACCESS = 1,
  MSG_ALLTOALL = 2,
  MSG_ALLTOALLVC = 3,
  MSG_REMOTE_OP = 4
};

class ExecutorMessage {
 public:
  ExecutorMessage();
  ~ExecutorMessage();

  ExecutorMessageType GetMessageType();

  void SetOperationMessage(const HcomOperation_t &opInfo);
  HcomOperation_t GetOperationMessage() const;

  void SetAlltoAllMessage(const HcomAllToAllVParams &opInfo, bool isGatherAlltoAll = false);
  HcomAllToAllVParams GetAlltoAllMessage() const;

  void SetAlltoAllVCMessage(const HcomAllToAllVCParams &opInfo);
  HcomAllToAllVCParams GetAlltoAllVCMessage() const;

  void SetStatusCallback(StatusCallback callBack);
  StatusCallback GetStatusCallback() const;

  bool isGatherAlltoAll_;

 private:
  HcomOperation_t hcomOperationInfo_;
  HcomRemoteOperationParams hcomRemoteOperationInfo_;
  RemoteAccessInfo remoteAccessInfo_;
  HcomAllToAllVParams hcomAlltoAllInfo_;
  HcomAllToAllVCParams hcomAlltoAllVCInfo_;
  StatusCallback statusCallback_;
  ExecutorMessageType messageType_;
};
}  // namespace hccl

#endif