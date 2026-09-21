/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "deploy/resource/deployer_port_distributor.h"
#include "common/debug/log.h"

namespace ge {
namespace {
constexpr int32_t kMinPort = 1024;
constexpr int32_t kMaxPort = 65535;
}

DeployerPortDistributor &DeployerPortDistributor::GetInstance() {
  static DeployerPortDistributor instance;
  return instance;
}

void DeployerPortDistributor::Finalize() {
  ipaddr_allocated_port_range_.clear();
  GELOGI("DeployerPortDistributor finalize");
}

Status DeployerPortDistributor::ParsePortRange(const std::string &port_range_str,
                                               std::pair<int32_t, int32_t> &port_range) {
  auto normalized_port_range = StringUtils::ReplaceAll(port_range_str, " ", "");
  GE_CHK_BOOL_RET_STATUS(!normalized_port_range.empty(), PARAM_INVALID,
                         "Port range[%s] is empty after normalized.", port_range_str.c_str());
  auto ports = ge::StringUtils::Split(normalized_port_range, '~');
  GE_CHK_BOOL_RET_STATUS(ports.size() == 2UL, PARAM_INVALID,
                         "Invalid format of port range:%s, must be begin~end", port_range_str.c_str());
  try {
    port_range.first = std::stoi(ports[0]);
    port_range.second = std::stoi(ports[1]);
  } catch (...) {
    GELOGE(FAILED, "Port of range[%s] is illegal.", port_range_str.c_str());
    return FAILED;
  }
  GE_CHK_BOOL_RET_STATUS((port_range.first <= port_range.second) &&
                         (port_range.first >= kMinPort) &&
                         (port_range.second <= kMaxPort),
                         PARAM_INVALID,
                         "Invalid value of port range:%s, must be %d <= begin <= end <= %d",
                         port_range_str.c_str(), kMinPort, kMaxPort);
  return SUCCESS;
}

Status DeployerPortDistributor::AllocatePortFromAvailableRange(const std::string &ipaddr,
                                                               const std::pair<int32_t, int32_t> &available_port_range,
                                                               int32_t &port) {
  auto it = ipaddr_allocated_port_range_.find(ipaddr);
  if (it == ipaddr_allocated_port_range_.end()) {
    port = available_port_range.first;
    ipaddr_allocated_port_range_[ipaddr] = std::make_pair(port, port);
    GELOGI("Allocate port[%d] successfully, ipaddr = %s, available_ports = [%d, %d].",
           port, ipaddr.c_str(), available_port_range.first, available_port_range.second);
    return SUCCESS;
  }
  auto &allocated_port_range = it->second;
  GE_CHK_BOOL_RET_STATUS(allocated_port_range.second < available_port_range.second,
                         PARAM_INVALID,
                         "No available port left, ipaddr = %s, "
                         "available_port_range = [%d, %d], allocated_port_range = [%d, %d]",
                         ipaddr.c_str(), available_port_range.first, available_port_range.second,
                         allocated_port_range.first, allocated_port_range.second);
  allocated_port_range.second = allocated_port_range.second + 1;
  port = allocated_port_range.second;
  GELOGI("Allocate port[%d] successfully, ipaddr = %s, available_ports = [%d, %d], allocated port range = [%d, %d].",
         port, ipaddr.c_str(), available_port_range.first, available_port_range.second,
         allocated_port_range.first, allocated_port_range.second);
  return SUCCESS;
}

Status DeployerPortDistributor::AllocatePort(const std::string &ipaddr,
                                             const std::string &available_ports,
                                             int32_t &port) {
  std::pair<int32_t, int32_t> available_port_range;
  GE_CHK_STATUS_RET(ParsePortRange(available_ports, available_port_range), "Failed to parse port range");
  GE_CHK_STATUS_RET(AllocatePortFromAvailableRange(ipaddr, available_port_range, port), "Failed to allocate port");
  return SUCCESS;
}
}  // namespace ge
