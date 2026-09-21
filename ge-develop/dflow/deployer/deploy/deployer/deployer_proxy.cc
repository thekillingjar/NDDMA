/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "common/debug/log.h"
#include "common/util.h"
#include "common/plugin/ge_make_unique_util.h"
#include "deploy/resource/resource_manager.h"
#include "deploy/deployer/deployer_proxy.h"

namespace ge {
DeployerProxy::~DeployerProxy() {
  try {
    Finalize();
  } catch (const std::exception &e) {
    GELOGE(FAILED, "Exception occurred in ~DeployerProxy: %s", e.what());
  } catch (...) {
    GELOGE(FAILED, "Unknown exception occurred in ~DeployerProxy");
  }
}

DeployerProxy &DeployerProxy::GetInstance() {
  static DeployerProxy instance;
  return instance;
}

Status DeployerProxy::Initialize(const std::vector<NodeConfig> &node_config_list) {
  std::lock_guard<std::mutex> lk(mu_);
  deployers_.clear();
  for (size_t i = 0U; i < node_config_list.size(); ++i) {
    const auto &node_config = node_config_list[i];
    auto deployer = CreateDeployer(node_config);
    GE_CHECK_NOTNULL(deployer);
    GE_CHK_STATUS_RET(deployer->Initialize(), "Failed to initialize deployer, id = %zu", i);
    deployers_.emplace_back(std::move(deployer));
    auto node_key = node_config.ipaddr + ":" + std::to_string(node_config.port);
    node_ip_2_node_ids_.emplace(std::make_pair(node_key, static_cast<int32_t>(i)));
    GELOGI("Deployer initialized successfully, node_id = %zu, node key = %s", i, node_key.c_str());
  }
  return SUCCESS;
}

Status DeployerProxy::Initialize(const NodeConfig &node_config) {
  std::lock_guard<std::mutex> lk(mu_);
  deployers_.clear();
  auto deployer = CreateDeployer(node_config, false);
  GE_CHECK_NOTNULL(deployer);
  GE_CHK_STATUS_RET(deployer->Initialize(), "Failed to initialize deployer.");
  deployers_.emplace_back(std::move(deployer));
  GELOGI("Deployer initialized successfully.");
  return SUCCESS;
}

void DeployerProxy::Finalize() {
  std::lock_guard<std::mutex> lk(mu_);
  std::vector<std::future<Status>> fut_rets;
  for (auto &deployer : deployers_) {
    auto fut = std::async(std::launch::async, &Deployer::Finalize, std::move(deployer));
    fut_rets.emplace_back(std::move(fut));
  }
  for (const auto &fut : fut_rets) {
    fut.wait();
  }
  deployers_.clear();
  node_ip_2_node_ids_.clear();
  GELOGI("Deployer proxy finalized");
}

std::unique_ptr<Deployer> DeployerProxy::CreateDeployer(const NodeConfig &node_config,
                                                        const bool with_heartbeat) {
  if (node_config.is_local) {
    return MakeUnique<LocalDeployer>(with_heartbeat);
  } else {
    return MakeUnique<RemoteDeployer>(node_config);
  }
}

Status DeployerProxy::SendRequest(int32_t node_id,
                                  deployer::DeployerRequest &request,
                                  deployer::DeployerResponse &response) {
  Deployer *deployer = nullptr;
  {
    std::lock_guard<std::mutex> lk(mu_);
    if (static_cast<size_t>(node_id) >= deployers_.size()) {
      GELOGE(PARAM_INVALID,
             "device id out of range, node id = %d, num_deployers = %zu",
             node_id,
             deployers_.size());
      return PARAM_INVALID;
    }
    deployer = deployers_[node_id].get();
  }

  GE_CHK_STATUS_RET(deployer->Process(request, response),
                    "Failed to send request, node_id = %d, request = %s, %s.",
                    node_id, request.DebugString().c_str(), deployer->GetNodeInfo().DebugString().c_str());
  GE_CHK_STATUS(response.error_code(),
                "Failed to process request, node_id = %d, request type = %d, error code = %u, %s.",
                node_id, request.type(), response.error_code(), deployer->GetNodeInfo().DebugString().c_str());
  return SUCCESS;
}

int32_t DeployerProxy::NumNodes() const {
  std::lock_guard<std::mutex> lk(mu_);
  return static_cast<int32_t>(deployers_.size());
}

const NodeInfo *DeployerProxy::GetNodeInfo(int32_t node_id) const {
  std::lock_guard<std::mutex> lk(mu_);
  if (static_cast<size_t>(node_id) >= deployers_.size()) {
    GELOGE(PARAM_INVALID,
           "device id out of range, node id = %d, num_deployers = %zu",
           node_id,
           deployers_.size());
    return nullptr;
  }
  return &deployers_[node_id]->GetNodeInfo();
}

Status DeployerProxy::GetNodeStat() const {
  for (size_t deployer_index = 0U; deployer_index < deployers_.size(); deployer_index++) {
    Status stat = deployers_[deployer_index]->GetDevStat();
    if (stat != SUCCESS) {
      return stat;
    }
  }
  return SUCCESS;
}

Status DeployerProxy::GetDeviceAbnormalCode() const {
  for (size_t deployer_index = 0U; deployer_index < deployers_.size(); deployer_index++) {
    Status error_code = deployers_[deployer_index]->GetDeviceAbnormalCode();
    if (error_code != SUCCESS) {
      return error_code;
    }
  }
  return SUCCESS;
}
}  // namespace ge
