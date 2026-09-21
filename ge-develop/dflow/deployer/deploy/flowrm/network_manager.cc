/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "deploy/flowrm/network_manager.h"
#include <regex>
#include <sys/types.h>
#include <netinet/in.h>
#include "common/ge_common/debug/log.h"
#include "base/err_msg.h"

namespace ge {
namespace {
constexpr uint32_t kMaxPortNum = 128U;
constexpr uint32_t kMaxPort = 65535U;
}

Status NetworkManager::BindMainPort() {
  DeployerConfig host_information;
  GE_CHK_STATUS_RET_NOLOG(GetHostInfo(host_information));

  std::string ports_range = host_information.host_info.data_panel.available_ports;
  auto ports = ge::StringUtils::Split(ports_range, '~');
  if (ports.size() != 2UL) { // available ports string has 2 numbers, which represent start and end port
    REPORT_INNER_ERR_MSG("E19999", "[Invaild][Port] ports_range[%s] is illegal.", ports_range.c_str());
    GELOGE(FAILED, "[Invaild][Port] ports_range[%s] is illegal.", ports_range.c_str());
    return FAILED;
  }
  std::regex port_reg("[1-9]\\d*|0");
  bool start_reg = std::regex_match(ports[0], port_reg);
  bool end_reg = std::regex_match(ports[1], port_reg);
  if (!(start_reg && end_reg)) {
    REPORT_INNER_ERR_MSG("E19999", "[Invaild][Port] start_port[%s] or end_port[%s] is illegal.",
                       ports[0].c_str(), ports[1].c_str());
    GELOGE(FAILED, "[Invaild][Port] start_port[%s] or end_port[%s] is illegal.", ports[0].c_str(), ports[1].c_str());
    return FAILED;
  }

  uint32_t start_port = 0U;
  uint32_t end_port = 0U;
  try {
    start_port = std::stoi(ports[0]);
    end_port = std::stoi(ports[1]);
  } catch (...) {
    REPORT_INNER_ERR_MSG("E19999", "[Invaild][Port] start_port[%s] or end_port[%s] is illegal.",
                       ports[0].c_str(), ports[1].c_str());
    GELOGE(FAILED, "[Invaild][Port] start_port[%s] or end_port[%s] is illegal.", ports[0].c_str(), ports[1].c_str());
    return FAILED;
  }
  if ((end_port <= start_port) || (start_port > kMaxPort) || (end_port > kMaxPort)) {
    REPORT_INNER_ERR_MSG("E19999", "[Invaild][Port] start_port[%u] is larger than end_port[%u].", start_port, end_port);
    GELOGE(FAILED, "[Invaild][Port] start_port[%u] is larger than end_port[%u].", start_port, end_port);
    return FAILED;
  }

  uint32_t num_segments = (end_port - start_port) / kMaxPortNum;
  uint32_t remainder = (end_port - start_port) % kMaxPortNum;
  if (remainder != 0U) {
    num_segments = num_segments + 1;
  }

  for (uint32_t i = 0U; i < num_segments; i++) {
    int32_t main_port = start_port + i * kMaxPortNum;
    Status res = TryToBindPort(main_port);
    if (res == SUCCESS) {
      main_port_ = main_port;
      GELOGD("Bind main port[%d] success.", main_port_);
      return SUCCESS;
    } else {
      GELOGW("Can not bind main port[%d], it may bind by other process, continue.", main_port);
      continue;
    }
  }

  REPORT_INNER_ERR_MSG("E19999", "[Bind][Port]All main port can not be bind, all prots in data panel can not be used.");
  GELOGE(FAILED, "[Bind][Port]All main port can not be bind, all prots in data panel can not be used.");
  return FAILED;
}

Status NetworkManager::GetDataPanelPort(int32_t &port) {
  if (socket_fd_ == -1) {
    socket_fd_ = socket(AF_INET, SOCK_STREAM, 0);
    GE_CHK_BOOL_RET_STATUS(socket_fd_ != -1, FAILED, "[Get][Fd] Get socket fd error");
    GE_CHK_STATUS_RET(BindMainPort(), "Failed to bind main port");
    GELOGI("[Bind][Port] Bind main port success, port = %d", main_port_);
  }
  port = static_cast<int32_t>(main_port_) + static_cast<int32_t>(PortType::kDataGw);
  return SUCCESS;
}

Status NetworkManager::TryToBindPort(uint32_t port) const {
  struct sockaddr_in addr;
  addr.sin_family = AF_INET;
  addr.sin_port = htons(port);
  addr.sin_addr.s_addr = INADDR_ANY;
  int32_t res = bind(socket_fd_, reinterpret_cast<sockaddr *>(&addr), sizeof(struct sockaddr));
  if (res == -1) {
    // 2P/2*2PG, multi process bind port will print
    GELOGW("[Bind][Port]Bind port[%d] failed.", port);
    return FAILED;
  }
  return SUCCESS;
}

Status NetworkManager::GetHostInfo(DeployerConfig &host_info) const {
  host_info = Configurations::GetInstance().GetHostInformation();
  return SUCCESS;
}

Status NetworkManager::GetDataPanelIp(std::string& host_ip) const {
  DeployerConfig host_information;
  GE_CHK_STATUS_RET_NOLOG(GetHostInfo(host_information));
  host_ip = host_information.host_info.data_panel.ipaddr;
  return SUCCESS;
}

std::string NetworkManager::GetCtrlPanelIp() const {
  return Configurations::GetInstance().GetHostInformation().host_info.ctrl_panel.ipaddr;
}

std::string NetworkManager::GetCtrlPanelPorts() const {
  return Configurations::GetInstance().GetHostInformation().host_info.ctrl_panel.available_ports;
}

Status NetworkManager::Initialize() {
  return SUCCESS;
}

Status NetworkManager::Finalize() {
  if (socket_fd_ == -1) {
    return SUCCESS;
  }

  int32_t res = close(socket_fd_);
  socket_fd_ = -1;
  if (res == -1) {
    GELOGE(FAILED, "[Close][Socket] Close socket failed.");
    return FAILED;
  }
  return SUCCESS;
}
}  // namespace ge
