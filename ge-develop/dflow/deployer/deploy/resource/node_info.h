/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_RUNTIME_HETEROGENEOUS_DEPLOY_RESOURCE_NODE_INFO_H_
#define AIR_RUNTIME_HETEROGENEOUS_DEPLOY_RESOURCE_NODE_INFO_H_

#include <cstdint>
#include <vector>
#include "deploy/resource/device_info.h"

namespace ge {
class NodeInfo {
 public:
  NodeInfo() = default;
  explicit NodeInfo(bool is_local);
  void AddDeviceInfo(const DeviceInfo &device_info);
  const std::vector<DeviceInfo> &GetDeviceList() const;
  bool IsLocal() const;
  void SetAddress(const std::string &address);
  void SetClientId(int64_t client_id);
  std::string DebugString() const;

 private:
  std::vector<DeviceInfo> device_list_;
  bool is_local_;
  std::string type_;
  std::string address_;
  int64_t client_id_ = -1;
};
}  // namespace ge

#endif  // AIR_RUNTIME_HETEROGENEOUS_DEPLOY_RESOURCE_NODE_INFO_H_
