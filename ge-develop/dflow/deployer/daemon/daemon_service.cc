/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "daemon/daemon_service.h"
#include <fstream>
#include <regex>
#include "ge/ge_api_error_codes.h"
#include "base/err_msg.h"
#include "common/util.h"
#include "dflow/base/utils/process_utils.h"
#include "common/config/configurations.h"
#include "daemon/daemon_client_manager.h"
#include "common/utils/rts_api_utils.h"
#include "deploy/deployer/deployer_service_impl.h"
#include "deploy/deployer/deployer_proxy.h"
#include "deploy/resource/resource_manager.h"
#include "deploy/resource/deployer_port_distributor.h"
#include "deploy/deployer/deployer_authentication.h"

namespace ge {
namespace {
const std::string kProtocolTypeRdma = "RDMA";
}
Status DaemonService::Process(const std::string &peer_uri,
                              const deployer::DeployerRequest &request,
                              deployer::DeployerResponse &response) {
  auto request_type = request.type();
  if (request_type == deployer::kInitRequest) {
    ProcessInitRequest(peer_uri, request, response);
  } else if (request_type == deployer::kDisconnect) {
    ProcessDisconnectRequest(peer_uri, request, response);
  } else if (request_type == deployer::kHeartbeat) {
    ProcessHeartbeatRequest(request, response);
  } else {
    ProcessDeployRequest(request, response);
  }
  return SUCCESS;
}

Status DaemonService::VerifyIpaddr(const std::string &peer_uri) {
  const auto &remote_configs = Configurations::GetInstance().GetRemoteNodeConfigs();
  if (remote_configs.empty()) {
    GELOGI("Without remote config, no need to verify ipaddr.");
    return SUCCESS;
  }

  for (const auto &config : remote_configs) {
    if (peer_uri.find(config.ipaddr) != std::string::npos) {
      return SUCCESS;
    }
  }
  GELOGE(FAILED, "Failed to match ipaddr with remote configs, remote size = %zu.", remote_configs.size());
  return FAILED;
}

Status DaemonService::VerifySignData(const deployer::DeployerRequest &request) {
  const auto &local_node = Configurations::GetInstance().GetLocalNode();
  if (local_node.auth_lib_path.empty()) {
    GELOGI("No need to verify data for deployer");
    return SUCCESS;
  }
  const auto &init_request = request.init_request();
  auto data = std::to_string(request.type());
  return DeployerAuthentication::GetInstance().AuthVerify(data, init_request.sign_data());
}

Status DaemonService::VerifyInitRequest(const std::string &peer_uri,
                                        const deployer::DeployerRequest &request,
                                        deployer::DeployerResponse &response) {
  auto init_request = const_cast<deployer::DeployerRequest &>(request).mutable_init_request();
  GE_MAKE_GUARD(init_request, [&init_request]() {
    auto sign_data = init_request->mutable_sign_data();
    sign_data->replace(sign_data->begin(), sign_data->end(), "");
  });
  std::string error_msg;
  GE_DISMISSABLE_GUARD(failed_process, ([&error_msg, &response]() {
    response.set_error_code(FAILED);
    response.set_error_message(error_msg);
  }));
  error_msg = "Failed to verify ipaddr.";
  GE_CHK_STATUS_RET(VerifyIpaddr(peer_uri), "%s", error_msg.c_str());

  error_msg = "Failed to verify sign data.";
  GE_CHK_STATUS_RET_NOLOG(VerifySignData(request));
  GE_DISMISS_GUARD(failed_process);
  return SUCCESS;
}

void DaemonService::SetSupportFlowgwMerged(deployer::DeployerResponse &response) {
  const auto &local_node = Configurations::GetInstance().GetLocalNode();
  bool support_flowgw_merged = false;  // default not support merged
  if (local_node.protocol == kProtocolTypeRdma) {
    support_flowgw_merged = false;
  } else {
    int32_t support_merged = 0;
    (void) aclrtGetDeviceCapability(0, ACL_FEATURE_SYSTEM_MEMQ_EVENT_CROSS_DEV,
                                    &support_merged);
    support_flowgw_merged = (support_merged == 1);
  }
  GEEVENT("Support flowgw merged = %d.", static_cast<int32_t>(support_flowgw_merged));
  response.mutable_init_response()->set_support_flowgw_merged(support_flowgw_merged);
}

void DaemonService::ProcessInitRequest(const std::string &peer_uri,
                                       const deployer::DeployerRequest &request,
                                       deployer::DeployerResponse &response) {
  GEEVENT("[Process][Request] init request start.");
  auto res = VerifyInitRequest(peer_uri, request, response);
  if (res != SUCCESS) {
    GELOGW("[Create][Client] Verify init request failed.");
    return;
  }

  std::map<std::string, std::string> deployer_envs;
  for (const auto &env : request.init_request().envs()) {
    deployer_envs.emplace(env.first, env.second);
  }
  int64_t client_id = 0;
  res = client_manager_->CreateAndInitClient(peer_uri, deployer_envs, client_id);
  if (res != SUCCESS) {
    REPORT_INNER_ERR_MSG("E19999", "Create client failed.");
    GELOGE(FAILED, "[Create][Client] Create client failed.");
    response.set_error_code(FAILED);
    response.set_error_message("Create client id failed.");
    return;
  }

  int32_t dev_count = 1;
  int32_t offset = 0;
  client_manager_->GenDgwPortOffset(dev_count, offset);
  GEEVENT("[Process][Request] init request success, client_id = %ld, "
          "address = %s, device count = %d, dwg port offset = %d.",
          client_id, peer_uri.c_str(), dev_count, offset);
  response.mutable_init_response()->set_client_id(client_id);
  response.mutable_init_response()->set_dev_count(dev_count);
  response.mutable_init_response()->set_dgw_port_offset(offset);
  SetSupportFlowgwMerged(response);
}

void DaemonService::ProcessDisconnectRequest(const std::string &peer_uri,
                                             const deployer::DeployerRequest &request,
                                             deployer::DeployerResponse &response) {
  int64_t client_id = request.client_id();
  GEEVENT("[Process][Request] disconnect request start, client_id = %ld, addr = %s.", client_id, peer_uri.c_str());
  DeployerDaemonClient *client = nullptr;
  if (GetClient(client_id, &client, response)) {
    (void)client_manager_->CloseClient(client_id);
  }
  GEEVENT("[Process][Request] disconnect request succeeded, client_id = %ld, addr = %s.", client_id, peer_uri.c_str());
}

void DaemonService::ProcessHeartbeatRequest(const deployer::DeployerRequest &request,
                                            deployer::DeployerResponse &response) {
  int64_t client_id = request.client_id();
  DeployerDaemonClient *client = nullptr;
  if (GetClient(client_id, &client, response)) {
    client->SetIsExecuting(true);
    client->OnHeartbeat();
    client->SetIsExecuting(false);
    (void)client->ProcessHeartbeatRequest(request, response);
  }
}

Status DaemonService::InitClientManager() {
  client_manager_ = MakeUnique<DaemonClientManager>();
  GE_CHECK_NOTNULL(client_manager_);
  GE_CHK_STATUS_RET(client_manager_->Initialize(), "Failed to initialize ClientManager");
  return SUCCESS;
}

bool DaemonService::GetClient(int64_t client_id, DeployerDaemonClient **client, deployer::DeployerResponse &response) {
  *client = client_manager_->GetClient(client_id);
  if (*client != nullptr) {
    return true;
  }

  REPORT_INNER_ERR_MSG("E19999", "Get client[%ld] failed.", client_id);
  GELOGE(FAILED, "[Get][Client] Get client[%ld] failed.", client_id);
  response.set_error_code(FAILED);
  response.set_error_message("Not exist client id");
  return false;
}

void DaemonService::ProcessDeployRequest(const deployer::DeployerRequest &request,
                                         deployer::DeployerResponse &response) {
  int64_t client_id = request.client_id();
  DeployerDaemonClient *client = nullptr;
  if (GetClient(client_id, &client, response)) {
    GELOGI("[Process][Request] process deploy request begin, client_id = %ld, type = %s.", client_id,
           deployer::DeployerRequestType_Name(request.type()).c_str());
    GE_CHK_STATUS(client->ProcessDeployRequest(request, response),
                  "[Process][Request] process deploy request failed, client_id = %ld, type = %s", client_id,
                  deployer::DeployerRequestType_Name(request.type()).c_str());
    GE_CHK_STATUS(response.error_code(),
                  "[Process][Request] check response failed, client_id = %ld, type = %s, error code = %u", client_id,
                  deployer::DeployerRequestType_Name(request.type()).c_str(), response.error_code());
    GELOGI("[Process][Request] process deploy request end, client_id = %ld, type = %s, ret = %u.", client_id,
           deployer::DeployerRequestType_Name(request.type()).c_str(), response.error_code());
  }
}

Status DaemonService::Initialize() {
  const auto &local_node = Configurations::GetInstance().GetLocalNode();
  GE_CHK_STATUS_RET(DeployerAuthentication::GetInstance().Initialize(local_node.auth_lib_path),
                    "Failed to deployer auth.");
  GE_CHK_STATUS_RET(DeployerProxy::GetInstance().Initialize(local_node), "Failed to initialize device proxy.");
  ResourceManager::GetInstance().ClearWorkingDir();
  GE_CHK_STATUS_RET(ResourceManager::GetInstance().Initialize(), "Failed to initialize ResourceManager");
  GE_CHK_STATUS_RET(InitClientManager(), "Failed to initialize ClientManager");
  GELOGI("DaemonService initialized successfully");
  return SUCCESS;
}

void DaemonService::Finalize() {
  DeployerProxy::GetInstance().Finalize();
  ResourceManager::GetInstance().Finalize();
  ResourceManager::GetInstance().ClearWorkingDir();
  client_manager_->Finalize();
  DeployerPortDistributor::GetInstance().Finalize();
  DeployerAuthentication::GetInstance().Finalize();
  GELOGI("DaemonService finalized successfully");
}

Status DeployerDaemonService::Initialize() {
  daemon_service_ = MakeUnique<DaemonService>();
  GE_CHECK_NOTNULL(daemon_service_);
  return daemon_service_->Initialize();
}

void DeployerDaemonService::Finalize() {
  daemon_service_->Finalize();
}

Status DeployerDaemonService::Process(const std::string &peer_uri,
                                      const deployer::DeployerRequest &request,
                                      deployer::DeployerResponse &response) {
  return daemon_service_->Process(peer_uri, request, response);
}
} // namespace ge
