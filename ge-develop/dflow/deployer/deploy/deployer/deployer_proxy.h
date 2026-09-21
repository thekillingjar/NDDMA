/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_RUNTIME_HETEROGENEOUS_DEPLOY_DEPLOYER_DEPLOYER_PROXY_H_
#define AIR_RUNTIME_HETEROGENEOUS_DEPLOY_DEPLOYER_DEPLOYER_PROXY_H_

#include <map>
#include <memory>
#include "proto/deployer.pb.h"
#include "common/config/configurations.h"
#include "deploy/resource/node_info.h"
#include "deploy/deployer/deployer.h"

namespace ge {
class DeployerProxy {
 public:
  static DeployerProxy &GetInstance();

  Status Initialize(const std::vector<NodeConfig> &node_config_list);

  Status Initialize(const NodeConfig &node_config);

  void Finalize();
  int32_t NumNodes() const;

  const NodeInfo *GetNodeInfo(int32_t node_id) const;

  Status GetNodeStat() const;

  Status GetDeviceAbnormalCode() const;

  Status SendRequest(int32_t node_id, deployer::DeployerRequest &request, deployer::DeployerResponse &response);
 private:
  DeployerProxy() = default;
  ~DeployerProxy();
  static std::unique_ptr<Deployer> CreateDeployer(const NodeConfig &node_config, const bool with_heartbeat = true);

  mutable std::mutex mu_;
  // all deployers: resource.json + clusterCfg
  std::vector<std::unique_ptr<Deployer>> deployers_;
  // key: ipaddr of ctrl panel, value: node id
  std::map<std::string, int32_t> node_ip_2_node_ids_;
};
}  // namespace ge

#endif  // AIR_RUNTIME_HETEROGENEOUS_DEPLOY_DEPLOYER_DEPLOYER_PROXY_H_
