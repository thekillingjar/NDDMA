/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "deploy/abnormal_status_handler/device_abnormal_status_handler.h"
#include "common/checker.h"
#include "acl/acl.h"

namespace ge {
namespace {
constexpr const char *kDeviceAbnormalRegModuleName = "deployer_dev_abnormal";
}
DeviceAbnormalStatusHandler &DeviceAbnormalStatusHandler::Instance() {
  static DeviceAbnormalStatusHandler inst;
  return inst;
}

void DeviceAbnormalStatusHandler::Initialize() const {
  rtError_t rt_error =
      rtRegTaskFailCallbackByModule(kDeviceAbnormalRegModuleName, [](rtExceptionInfo_t *exceptionInfo) {
        if (exceptionInfo != nullptr) {
          DeviceAbnormalStatusHandler::Instance().AbnormalCallback(*exceptionInfo);
        }
      });
  if (rt_error == RT_ERROR_NONE) {
    GEEVENT("Register callback by module[%s] success", kDeviceAbnormalRegModuleName);
  } else {
    GELOGE(RT_FAILED, "Register callback by module[%s] failed, rt_error=%u", kDeviceAbnormalRegModuleName, rt_error);
  }
}

void DeviceAbnormalStatusHandler::Finalize() const {
  // callback is null means unreg
  rtError_t rt_error = rtRegTaskFailCallbackByModule(kDeviceAbnormalRegModuleName, nullptr);
  if (rt_error == RT_ERROR_NONE) {
    GEEVENT("Unregister callback by module[%s] success", kDeviceAbnormalRegModuleName);
  } else {
    GELOGE(RT_FAILED, "Unregister callback by module[%s] failed, rt_error=%u", kDeviceAbnormalRegModuleName, rt_error);
  }
}

void DeviceAbnormalStatusHandler::AbnormalCallback(const rtExceptionInfo_t &exceptionInfo) {
  GELOGD("receive callback, deviceid=%u, retcode=%u", exceptionInfo.deviceid, exceptionInfo.retcode);
  // now just handle oom error
  if (exceptionInfo.retcode == ACL_ERROR_RT_DEVICE_OOM) {
    std::lock_guard<std::mutex> lock(oom_mutex_);
    dev_oom_set_.emplace(exceptionInfo.deviceid);
    GELOGE(FAILED, "Device OOM: deviceId=%u, retCode=%u", exceptionInfo.deviceid, exceptionInfo.retcode);
  }
}

void DeviceAbnormalStatusHandler::HandleDeviceAbnormal(
    std::map<uint32_t, std::vector<uint32_t>> &abnormal_device_info) {
  std::lock_guard<std::mutex> lock(oom_mutex_);
  for (uint32_t oom_dev_id : dev_oom_set_) {
    abnormal_device_info[oom_dev_id].emplace_back(ACL_ERROR_RT_DEVICE_OOM);
  }
  dev_oom_set_.clear();
}
}  // namespace ge
