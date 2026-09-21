/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_RUNTIME_HETEROGENEOUS_DEPLOY_DEPLOYER_DEPLOYER_H_
#define AIR_RUNTIME_HETEROGENEOUS_DEPLOY_DEPLOYER_DEPLOYER_H_

#include <string>
#include <memory>
#include <thread>
#include "proto/deployer.pb.h"
#include "ge/ge_api_error_codes.h"
#include "common/config/configurations.h"
#include "deploy/deployer/deploy_context.h"
#include "deploy/resource/node_info.h"
#include "deploy/resource/device_info.h"
#include "deploy/rpc/deployer_client.h"

namespace ge {
class Deployer {
 public:
  Deployer() = default;
  virtual ~Deployer() = default;
  virtual const NodeInfo &GetNodeInfo() const = 0;
  virtual Status Process(deployer::DeployerRequest &request, deployer::DeployerResponse &response) = 0;

  virtual Status Initialize() = 0;
  virtual Status Finalize() = 0;
  virtual Status GetDevStat() = 0;
  virtual void AddAbnormalNodeConfig() { return; }
  virtual void AddAbnormalDeviceInfo(int32_t device_id, int32_t device_type) {
    (void) device_id;
    (void) device_type;
    return;
  }
  void FormatAndAddAbnormalDeviceInfo(int32_t node_id, int32_t device_id, int32_t device_type);
  void ParseRsponse(deployer::DeployerResponse &response);
  void ParseAbnormalDevice(const deployer::DeployerResponse &response);
  Status GetDeviceAbnormalCode();
 private:
  std::mutex dev_abnormal_mutex_;
  std::map<uint32_t, std::vector<uint32_t>> dev_abnormal_info_;
};

class LocalDeployer : public Deployer {
 public:
  explicit LocalDeployer(bool with_heartbeat = true);
  ~LocalDeployer() override {
    LocalDeployer::Finalize();
  };
  Status Initialize() override;
  const NodeInfo &GetNodeInfo() const override;
  Status Finalize() override;
  Status GetDevStat() override;

  void AddAbnormalDeviceInfo(int32_t device_id, int32_t device_type);
  Status Process(deployer::DeployerRequest &request, deployer::DeployerResponse &response) override;
 private:
  void Keepalive();
  NodeInfo node_info_;
  DeployContext &local_context_;
  std::atomic_bool is_running_{true};
  std::mutex mu_cv_;
  std::thread keepalive_thread_;
  bool with_heartbeat_ = true;
};

class RemoteDeployer : public Deployer {
 public:
  explicit RemoteDeployer(NodeConfig node_config);
  Status Initialize() override;
  Status Finalize() override;
  Status Process(deployer::DeployerRequest &request, deployer::DeployerResponse &response) override;

  const NodeInfo &GetNodeInfo() const override;
  Status GetDevStat() override {
    return dev_status_;
  }
 protected:
  virtual std::unique_ptr<DeployerClient> CreateClient();

 private:
  Status Connect();
  Status Disconnect();
  Status Process(deployer::DeployerRequest &request,
                 deployer::DeployerResponse &response,
                 const int64_t timeout_sec,
                 const int32_t retry_times);
  void AddAbnormalNodeConfig();
  void AddAbnormalDeviceInfo(int32_t device_id, int32_t device_type);
  void SendHeartbeat(int32_t max_retry_times = 0);
  void Keepalive();
  Status InitRequest(deployer::DeployerRequest &request);
  Status InitNodeInfoByDeviceList();
  Status InitNodeInfoByChipCount();
  void UpdateNodeInfo(const ::deployer::InitResponse &init_response);
  Status SendInitRequest(const deployer::DeployerRequest request,
                         deployer::DeployerResponse &response);
  void SetDevStat(Status status) {
    dev_status_ = status;
  }

  NodeConfig node_config_;
  NodeInfo node_info_;
  std::unique_ptr<DeployerClient> client_;
  std::thread keepalive_thread_;
  std::atomic_bool keep_alive_{true};
  std::mutex mu_cv_;
  std::mutex mu_init_;
  bool connected_ = false;
  Status connected_status_ = FAILED;
  int64_t client_id_ = 0;
  Status dev_status_ = SUCCESS;
  bool exception_{false};
};
}  // namespace ge

#endif  // AIR_RUNTIME_HETEROGENEOUS_DEPLOY_DEPLOYER_DEPLOYER_H_
