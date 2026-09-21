/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "deploy/resource/device_info.h"

namespace ge {
DeviceInfo::DeviceInfo(int32_t node_id, int32_t device_id) noexcept
    : node_id_(node_id),
      device_id_(device_id),
      phy_device_id_(device_id),
      device_index_(device_id) {
}

DeviceInfo::DeviceInfo(int32_t node_id, DeviceType device_type, int32_t device_id) noexcept
    : node_id_(node_id),
      device_type_(device_type),
      device_id_(device_id),
      phy_device_id_(device_id),
      device_index_(device_type == NPU ? device_id : -1) {
}

int32_t DeviceInfo::GetNodeId() const {
  return node_id_;
}

int32_t DeviceInfo::GetDeviceId() const {
  return device_id_;
}

const std::string &DeviceInfo::GetHostIp() const {
  return host_ip_;
}

void DeviceInfo::SetHostIp(const std::string &host_ip) {
  host_ip_ = host_ip;
}

const std::string &DeviceInfo::GetDeviceIp() const {
  return device_ip_;
}

void DeviceInfo::SetDeviceIp(const std::string &device_ip) {
  device_ip_ = device_ip;
}

int32_t DeviceInfo::GetDgwPort() const {
  return dgw_port_;
}

void DeviceInfo::SetDgwPort(int32_t dgw_port) {
  dgw_port_ = dgw_port;
}

DeviceType DeviceInfo::GetDeviceType() const {
  return device_type_;
}
}  // namespace ge
