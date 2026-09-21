/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_RUNTIME_HETEROGENEOUS_DEPLOY_DEVICE_ABNORMAL_STATUS_HANDLER_H_
#define AIR_RUNTIME_HETEROGENEOUS_DEPLOY_DEVICE_ABNORMAL_STATUS_HANDLER_H_

#include <set>
#include <map>
#include <vector>
#include <mutex>
#include "runtime/rt.h"

namespace ge {
class DeviceAbnormalStatusHandler {
 public:
  static DeviceAbnormalStatusHandler &Instance();
  void Initialize() const;
  void Finalize() const;
  void AbnormalCallback(const rtExceptionInfo_t &exceptionInfo);
  void HandleDeviceAbnormal(std::map<uint32_t, std::vector<uint32_t>> &abnormal_device_info);

 private:
  DeviceAbnormalStatusHandler() = default;
  ~DeviceAbnormalStatusHandler() = default;
  std::mutex oom_mutex_;
  std::set<uint32_t> dev_oom_set_;
};
}  // namespace ge
#endif  // AIR_RUNTIME_HETEROGENEOUS_DEPLOY_DEVICE_ABNORMAL_STATUS_HANDLER_H_
