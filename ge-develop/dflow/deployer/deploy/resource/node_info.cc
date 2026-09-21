/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "node_info.h"

namespace ge {
NodeInfo::NodeInfo(bool is_local) : is_local_(is_local) {
  if (is_local) {
    type_ = "local";
  } else {
    type_ = "remote";
  }
}

void NodeInfo::AddDeviceInfo(const DeviceInfo &device_info) {
  device_list_.emplace_back(device_info);
}

const std::vector<DeviceInfo> &NodeInfo::GetDeviceList() const {
  return device_list_;
}

bool NodeInfo::IsLocal() const {
  return is_local_;
}

void NodeInfo::SetAddress(const std::string &address) {
  address_ = address;
}

void NodeInfo::SetClientId(int64_t client_id) {
  client_id_ = client_id;
}

std::string NodeInfo::DebugString() const {
  std::string node_info = "node_type = " + type_  + ", client_id = " + std::to_string(client_id_);
  if (!is_local_) {
    node_info += (", addr = " + address_);
  }
  return node_info;
}
}  // namespace ge
