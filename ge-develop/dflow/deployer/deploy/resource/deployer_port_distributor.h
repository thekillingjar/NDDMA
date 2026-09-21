/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_RUNTIME_HETEROGENEOUS_DEPLOY_RESOURCE_DEPLOYER_PORT_DISTRIBUTOR_H_
#define AIR_RUNTIME_HETEROGENEOUS_DEPLOY_RESOURCE_DEPLOYER_PORT_DISTRIBUTOR_H_

#include <map>
#include <string>
#include "ge/ge_api_types.h"

namespace ge {
class DeployerPortDistributor {
 public:
  static DeployerPortDistributor &GetInstance();
  void Finalize();
  Status AllocatePort(const std::string &ipaddr, const std::string &available_ports, int32_t &port);

 private:
  static Status ParsePortRange(const std::string &port_range_str,
                               std::pair<int32_t, int32_t> &port_range);
  Status AllocatePortFromAvailableRange(const std::string &ipaddr,
                                        const std::pair<int32_t, int32_t> &available_port_range,
                                        int32_t &port);

  // key:ipaddr, value[first:allocated port range begin, second:allocated port range end]
  std::map<std::string, std::pair<int32_t, int32_t>> ipaddr_allocated_port_range_;
};
}  // namespace ge

#endif  // AIR_RUNTIME_HETEROGENEOUS_DEPLOY_RESOURCE_DEPLOYER_PORT_DISTRIBUTOR_H_
