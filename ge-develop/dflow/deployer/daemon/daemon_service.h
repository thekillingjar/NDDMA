/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_RUNTIME_HETEROGENEOUS_DAEMON_DAEMON_SERVICE_H_
#define AIR_RUNTIME_HETEROGENEOUS_DAEMON_DAEMON_SERVICE_H_

#include "ge/ge_api_types.h"
#include "proto/deployer.pb.h"
#include "daemon/daemon_client_manager.h"
#include "deploy/rpc/deployer_server.h"

namespace ge {
class DaemonService {
 public:
  virtual ~DaemonService() = default;
  Status Initialize();
  void Finalize();
  Status Process(const std::string &peer_uri,
                 const deployer::DeployerRequest &request,
                 deployer::DeployerResponse &response);

 protected:
  virtual Status InitClientManager();

 private:
  void ProcessInitRequest(const std::string &peer_uri,
                          const deployer::DeployerRequest &request,
                          deployer::DeployerResponse &response);

  void ProcessDisconnectRequest(const std::string &peer_uri,
                                const deployer::DeployerRequest &request,
                                deployer::DeployerResponse &response);

  void ProcessHeartbeatRequest(const deployer::DeployerRequest &request,
                               deployer::DeployerResponse &response);

  void ProcessDeployRequest(const deployer::DeployerRequest &request,
                            deployer::DeployerResponse &response);

  bool GetClient(int64_t client_id, DeployerDaemonClient **client, deployer::DeployerResponse &response);

  static Status VerifyInitRequest(const std::string &peer_uri,
                                  const deployer::DeployerRequest &request,
                                  deployer::DeployerResponse &response);

  static Status VerifySignData(const deployer::DeployerRequest &request);

  static Status VerifyIpaddr(const std::string &peer_uri);

  static void SetSupportFlowgwMerged(deployer::DeployerResponse &response);

  std::unique_ptr<DaemonClientManager> client_manager_;
};

class DeployerDaemonService : public DeployerSpi {
 public:
  Status Initialize() override;
  void Finalize() override;
  Status Process(const std::string &peer_uri,
                 const deployer::DeployerRequest &request,
                 deployer::DeployerResponse &response) override;

 private:
  std::unique_ptr<DaemonService> daemon_service_;
};
} // namespace ge

#endif  // AIR_RUNTIME_HETEROGENEOUS_DAEMON_DAEMON_SERVICE_H_
