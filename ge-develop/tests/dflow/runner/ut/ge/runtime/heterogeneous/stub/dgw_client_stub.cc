/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "queue_schedule/dgw_client.h"

namespace bqs {
DgwClient::DgwClient(const uint32_t deviceId) {

}

std::shared_ptr<DgwClient> DgwClient::GetInstance(const uint32_t deviceId) {
  if (deviceId > 10000) {
    return nullptr;
  }
  std::shared_ptr<DgwClient> ptr = std::make_shared<DgwClient>(deviceId);
  return ptr;
}

std::shared_ptr<DgwClient> DgwClient::GetInstance(const uint32_t deviceId, const pid_t pid) {
  if (deviceId > 10000) {
    return nullptr;
  }
  std::shared_ptr<DgwClient> ptr = std::make_shared<DgwClient>(deviceId);
  return ptr;
}

std::shared_ptr<DgwClient> DgwClient::GetInstance(const uint32_t deviceId, const pid_t pid, const bool proxy) {
  if (deviceId > 10000) {
    return nullptr;
  }
  std::shared_ptr<DgwClient> ptr = std::make_shared<DgwClient>(deviceId);
  return ptr;
}

int32_t DgwClient::Initialize(const uint32_t dgwPid,
                              const std::string procSign,
                              const bool isProxy,
                              const int32_t timeout) {
  return 0;
}

int32_t DgwClient::CreateHcomHandle(const std::string &rankTable,
                                    const int32_t rankId,
                                    const void *const reserve,
                                    uint64_t &handle,
                                    const int32_t timeout) {
  handle = 0U;
  return 0;
}

int32_t DgwClient::DestroyHcomHandle(const uint64_t handle, const int32_t timeout) {
  if (handle > 10000) {
    return 1;
  }
  return 0;
}


int32_t DgwClient::UpdateConfig(ConfigInfo &cfgInfo,
                                std::vector<int32_t> &cfgRets,
                                const int32_t timeout) {
  return 0;
}

int32_t DgwClient::WaitConfigEffect(const int32_t rsv, const int32_t timeout) {
  return 0;
}
}
