/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_RUNTIME_HETEROGENEOUS_FLOWRM_NETWORK_MANAGER_H_
#define AIR_RUNTIME_HETEROGENEOUS_FLOWRM_NETWORK_MANAGER_H_

#include "common/config/configurations.h"
#include "ge/ge_api_error_codes.h"
#include "common/config/json_parser.h"

namespace ge {
enum class PortType {
  kDataGw = 1
};

class NetworkManager {
 public:
  static NetworkManager &GetInstance() {
    static NetworkManager instance;
    return instance;
  }

  /*
   *  @ingroup ge
   *  @brief   initialize all port
   *  @return: SUCCESS or FAILED
   */
  static Status Initialize();

  /*
   *  @ingroup ge
   *  @brief   finalize all socket fd
   *  @param   [in]  None
   *  @return  SUCCESS or FAILED
   */
  Status Finalize();

  /*
   *  @ingroup ge
   *  @brief   get data panel ip
   *  @return: ip address
   */
  Status GetDataPanelIp(std::string& host_ip) const;

  /*
   *  @ingroup ge
   *  @brief   get port by type
   *  @return: port number
   */
  Status GetDataPanelPort(int32_t &port);

  /*
   *  @ingroup ge
   *  @brief   get ctrl panel ip
   *  @return: ctrl ip address
   */
  std::string GetCtrlPanelIp() const;

  /*
   *  @ingroup ge
   *  @brief   get ctrl panel ports
   *  @return: ctrl port range
   */
  std::string GetCtrlPanelPorts() const;

 private:
  NetworkManager() = default;
  ~NetworkManager() = default;

  /*
   *  @ingroup ge
   *  @brief   bind main port
   *  @return: SUCCESS or FAILED
   */
  Status BindMainPort();

  /*
   *  @ingroup ge
   *  @brief   try to bind port
   *  @param   [in]  uint32_t
   *  @return: SUCCESS or FAILED
   */
  Status TryToBindPort(uint32_t port) const;

  /*
   *  @ingroup ge
   *  @brief   get host json info
   *  @param   [in]  DeployerConfig &
   *  @return: SUCCESS or FAILED
   */
  Status GetHostInfo(DeployerConfig &host_info) const;

  uint32_t main_port_ = 0U;
  uint32_t datagw_port_ = 0U;
  int32_t socket_fd_ = -1;
};
}  // namespace ge
#endif  // AIR_RUNTIME_HETEROGENEOUS_FLOWRM_NETWORK_MANAGER_H_
