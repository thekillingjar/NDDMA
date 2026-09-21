/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "deploy/deployer/deployer.h"
#include "mmpa/mmpa_api.h"
#include "common/debug/log.h"
#include "common/plugin/ge_make_unique_util.h"
#include "common/config/device_debug_config.h"
#include "common/subprocess/subprocess_manager.h"
#include "deploy/deployer/deployer_service_impl.h"
#include "deploy/flowrm/network_manager.h"
#include "deploy/resource/deployer_port_distributor.h"
#include "deploy/deployer/deployer_authentication.h"
#include "common/compile_profiling/ge_call_wrapper.h"

namespace ge {
namespace {
constexpr int32_t kHeartbeatInterval = 5000; // millisconds
constexpr int32_t kDataGwPortBase = 16666;
constexpr int32_t kDataGwPortMaxOffset = 256;
constexpr int32_t kHeartbeatTryMaxNum = 10;
constexpr int32_t kInvalidRankId = -1;
constexpr uint32_t kInitTryWaitInterval = 1000; // millisconds
}  // namespace

void Deployer::FormatAndAddAbnormalDeviceInfo(int32_t node_id, int32_t device_id, int32_t device_type) {
  GELOGI("ParseRsponse: node_id=%d, device id=%d, device type=%d", node_id, device_id, device_type);
  auto &deploy_context = DeployContext::LocalContext();
  std::lock_guard<std::mutex> lk(deploy_context.GetAbnormalHeartbeatInfoMu());
  DeployPlan::DeviceInfo device_info = DeployPlan::DeviceInfo(device_type, node_id, device_id);
  deploy_context.AddAbnormalDeviceInfo(device_info);
}

void Deployer::ParseAbnormalDevice(const deployer::DeployerResponse &response) {
  std::unique_lock<std::mutex> lk(dev_abnormal_mutex_);
  dev_abnormal_info_.clear();
  const auto &abnormal_device = response.heartbeat_response().abnormal_device();
  if (abnormal_device.empty()) {
    GELOGI("node[%s] has no abnormal device info", GetNodeInfo().DebugString().c_str());
    return;
  }

  for (const auto &abnormal : abnormal_device) {
    std::vector<uint32_t> error_codes(abnormal.second.error_code().cbegin(), abnormal.second.error_code().cend());
    GELOGE(FAILED, "node[%s] device[%u] is abnormal, error code[%s]", GetNodeInfo().DebugString().c_str(),
           abnormal.first, ToString(error_codes).c_str());
    dev_abnormal_info_[abnormal.first] = std::move(error_codes);
  }
}

Status Deployer::GetDeviceAbnormalCode() {
  std::unique_lock<std::mutex> lk(dev_abnormal_mutex_);
  for (const auto &abnormal_info : dev_abnormal_info_) {
    if (!abnormal_info.second.empty()) {
      GELOGE(FAILED, "node[%s] device[%u] is abnormal, error code[%s]", GetNodeInfo().DebugString().c_str(),
             abnormal_info.first, ToString(abnormal_info.second).c_str());
      return *(abnormal_info.second.cbegin());
    }
  }
  return SUCCESS;
}

void Deployer::ParseRsponse(deployer::DeployerResponse &response) {
  ParseAbnormalDevice(response);
  if (response.heartbeat_response().abnormal_type() == kAbnormalTypeNode) {
    AddAbnormalNodeConfig();
    return;
  } else if (response.heartbeat_response().abnormal_type() == kAbnormalTypeDevice) {
    for (int32_t i = 0; i < response.heartbeat_response().device_status_size(); ++i) {
      const auto &device_status = response.heartbeat_response().device_status(i);
      if (device_status.device_id() >= 0) { // flowgw_client进程异常
        AddAbnormalDeviceInfo(device_status.device_id(), device_status.device_type());
      }
    }
    return;
  } else if (response.heartbeat_response().abnormal_type() == kAbnormalTypeModelInstance) {
    auto &deploy_context = DeployContext::LocalContext();
    const auto abnormal_submodel_instance_name = response.heartbeat_response().abnormal_submodel_instance_name();
    for (const auto &submodel_instances : abnormal_submodel_instance_name) {
      for (auto &submodel_instance : submodel_instances.second.submodel_instance_name()) {
        std::lock_guard<std::mutex> lk(deploy_context.GetAbnormalHeartbeatInfoMu());
        deploy_context.AddAbnormalSubmodelInstanceName(submodel_instances.first, submodel_instance.first);
        GELOGI("ParseRsponse: root model id=%u, anormal model instance is %s",
            submodel_instances.first, submodel_instance.first.c_str());
      }
    }
  }
}

LocalDeployer::LocalDeployer(bool with_heartbeat)
    : node_info_(true), local_context_(DeployContext::LocalContext()), with_heartbeat_(with_heartbeat) {}

const NodeInfo &LocalDeployer::GetNodeInfo() const {
  return node_info_;
}

Status LocalDeployer::Initialize() {
  const auto &node_config = Configurations::GetInstance().GetLocalNode();
  for (size_t i = 0U; i < node_config.device_list.size(); ++i) {
    const auto &device_config = node_config.device_list[i];
    DeviceInfo device_info;
    device_info.SetDeviceId(device_config.device_id);
    device_info.SetDeviceType(device_config.device_type);
    device_info.SetResourceType(device_config.resource_type);
    device_info.SetPhyDeviceId(device_config.phy_device_id);
    device_info.SetHcomDeviceId(device_config.hcom_device_id);
    device_info.SetDeviceIndex(device_config.device_index);
    device_info.SetDeviceIp(device_config.ipaddr);
    device_info.SetSupportHcom(device_config.support_hcom);
    device_info.SetOsId(device_config.os_id);
    device_info.SetAvailablePorts(node_config.available_ports);
    int32_t port = -1;
    if (device_config.device_type == CPU && node_config.need_port_preemption) {
      GE_CHK_STATUS_RET(NetworkManager::GetInstance().GetDataPanelPort(port),
                        "Failed to allocate port");
    } else {
      GE_CHK_STATUS_RET(DeployerPortDistributor::GetInstance().AllocatePort(node_config.ipaddr,
                                                                            node_config.available_ports,
                                                                            port),
                        "Failed to allocate port");
    }
    device_info.SetDgwPort(port);
    device_info.SetHostIp(node_config.ipaddr);
    device_info.SetNodePort(node_config.port);
    device_info.SetNodeMeshIndex(node_config.node_mesh_index);
    device_info.SetSupportFlowgw(device_config.support_flowgw);
    device_info.SetSuperDeviceId(device_config.super_device_id);

    node_info_.AddDeviceInfo(device_info);
    GELOGD("Local deployer add device successfully, device_id = %d.", device_config.device_id);
  }

  auto context_name = "local_context_" + std::to_string(mmGetPid());
  local_context_.SetName(context_name);
  local_context_.Initialize();
  is_running_ = true;
  if (with_heartbeat_) {
    keepalive_thread_ = std::thread([this]() { Keepalive(); });
  }
  GELOGD("Local device initialized successfully.");
  return SUCCESS;
}

void LocalDeployer::Keepalive() {
  SET_THREAD_NAME(pthread_self(), "ge_dpl_kalv1");
  GELOGI("Keepalive task started");
  while (is_running_) {
    std::unique_lock<std::mutex> lk(mu_cv_);
    constexpr int32_t kSleepInterval = 100; // millisconds
    constexpr int32_t kSleepCnt = kHeartbeatInterval / kSleepInterval;
    for (int32_t i = 0; (i < kSleepCnt) && is_running_; i++) {
      std::this_thread::sleep_for(std::chrono::milliseconds(kSleepInterval));
    }
    const Status ret = GetDevStat();
    if (ret != SUCCESS) {
      GELOGW("Local process status abnormal");
    }
  }
  GELOGI("Keepalive task ended.");
}

Status LocalDeployer::Finalize() {
  if (!is_running_) {
    return SUCCESS;
  }
  local_context_.Finalize();
  is_running_ = false;
  if (with_heartbeat_ && keepalive_thread_.joinable()) {
    keepalive_thread_.join();
  }
  GELOGI("Local device finalized successfully.");
  return SUCCESS;
}

void LocalDeployer::AddAbnormalDeviceInfo(int32_t device_id, int32_t device_type) {
  GELOGI("LocalDeployer ParseRsponse: device id=%d, device type=%d", device_id, device_type);
  FormatAndAddAbnormalDeviceInfo(0, device_id, device_type);
}

Status LocalDeployer::GetDevStat() {
  deployer::DeployerRequest request;
  request.set_type(deployer::kHeartbeat);
  deployer::DeployerResponse response;
  (void) DeployerServiceImpl::GetInstance().Process(local_context_, request, response);
  ParseRsponse(response);
  if (response.error_code() != SUCCESS) {
    return FAILED;
  }
  return SUCCESS;
}

Status LocalDeployer::Process(deployer::DeployerRequest &request, deployer::DeployerResponse &response) {
  return DeployerServiceImpl::GetInstance().Process(local_context_, request, response);
}

RemoteDeployer::RemoteDeployer(NodeConfig node_config)
    : node_config_(std::move(node_config)), node_info_(false) {}

Status RemoteDeployer::InitNodeInfoByDeviceList() {
  for (size_t i = 0U; i < node_config_.device_list.size(); ++i) {
    const auto &device_config = node_config_.device_list[i];
    DeviceInfo device_info;
    device_info.SetDeviceId(device_config.device_id);
    device_info.SetDeviceType(device_config.device_type);
    device_info.SetDeviceIp(device_config.ipaddr);
    device_info.SetResourceType(device_config.resource_type);
    device_info.SetPhyDeviceId(device_config.phy_device_id);
    device_info.SetHcomDeviceId(device_config.hcom_device_id);
    device_info.SetDeviceIndex(device_config.device_index);
    device_info.SetSupportHcom(device_config.support_hcom);
    device_info.SetOsId(device_config.os_id);
    device_info.SetAvailablePorts(node_config_.available_ports);
    int32_t port = -1;
    GE_CHK_STATUS_RET(DeployerPortDistributor::GetInstance().AllocatePort(node_config_.ipaddr,
                                                                          node_config_.available_ports,
                                                                          port),
                      "Failed to allocate port");
    device_info.SetDgwPort(port);
    device_info.SetHostIp(node_config_.ipaddr);
    device_info.SetNodePort(node_config_.port);
    device_info.SetNodeMeshIndex(node_config_.node_mesh_index);
    device_info.SetSupportFlowgw(device_config.support_flowgw);
    device_info.SetSuperDeviceId(device_config.super_device_id);
    node_info_.AddDeviceInfo(device_info);
    GELOGD("Remote deployer add device by device list successfully, device_id = %d.", device_config.device_id);
  }
  return SUCCESS;
}

Status RemoteDeployer::InitNodeInfoByChipCount() {
  if ((node_info_.GetDeviceList().size() != 0UL) &&
      (node_config_.chip_count != 0U)) {
    GELOGE(FAILED, "It is not supported to set chip count when device list detail info is existed.");
    return FAILED;
  }
  for (uint32_t i = 0U; i < node_config_.chip_count; ++i) {
    DeviceInfo device_info;
    device_info.SetHostIp(node_config_.ipaddr);
    device_info.SetNodePort(node_config_.port);
    device_info.SetDeviceId(i);
    device_info.SetDeviceIp(node_config_.ipaddr);
    device_info.SetResourceType(node_config_.resource_type);
    device_info.SetPhyDeviceId(i);
    device_info.SetHcomDeviceId(i);
    device_info.SetDeviceIndex(i);
    device_info.SetAvailablePorts(node_config_.available_ports);
    int32_t port = -1;
    GE_CHK_STATUS_RET(DeployerPortDistributor::GetInstance().AllocatePort(node_config_.ipaddr,
                                                                          node_config_.available_ports,
                                                                          port),
                      "Failed to allocate port");
    device_info.SetDgwPort(port);
    device_info.SetNodeMeshIndex(node_config_.node_mesh_index);
    device_info.SetSupportFlowgw(true);
    device_info.SetSupportHcom(true);
    node_info_.AddDeviceInfo(device_info);
    GELOGD("Remote deployer add device by chip count successfully, device_id = %d", i);
  }
  return SUCCESS;
}

Status RemoteDeployer::Initialize() {
  const std::string rpc_address = node_config_.ipaddr + ":" + std::to_string(node_config_.port);
  GELOGI("Initialize remote deploy client started, address = %s", rpc_address.c_str());
  node_info_.SetAddress(rpc_address);
  client_ = CreateClient();
  GE_CHECK_NOTNULL(client_);
  GE_CHK_STATUS_RET(client_->Initialize(rpc_address), "Failed to create rpc client");
  if (node_config_.lazy_connect) {
    GE_CHK_STATUS_RET(InitNodeInfoByDeviceList(), "Failed to init node info.");
    GE_CHK_STATUS_RET_NOLOG(InitNodeInfoByChipCount());
    GELOGI("Initialize remote deploy node info successfully");
    return SUCCESS;
  }

  GE_CHK_STATUS_RET(Connect(), "Failed to connect to remote deployer");
  GELOGI("Initialize remote deploy client successfully, address = %s, client = %ld", rpc_address.c_str(), client_id_);
  return SUCCESS;
}

Status RemoteDeployer::InitRequest(deployer::DeployerRequest &request) {
  request.set_type(deployer::kInitRequest);
  auto init_request = request.mutable_init_request();

  if (!node_config_.auth_lib_path.empty()) {
    std::string sign_data;
    GE_MAKE_GUARD(sign_data, [&sign_data]() { sign_data.replace(sign_data.begin(), sign_data.end(), ""); });
    auto type = std::to_string(request.type());
    GE_CHK_STATUS_RET(DeployerAuthentication::GetInstance().AuthSign(type, sign_data),
                     "Failed to auth sign.");
    init_request->set_sign_data(sign_data);
  }

  std::map<std::string, std::string> log_envs;
  DevMaintenanceCfgUtils::GetLogEnvs(log_envs);
  for (const auto &log_env : log_envs) {
    init_request->mutable_envs()->insert({log_env.first, log_env.second});
  }
  return SUCCESS;
}

Status RemoteDeployer::SendInitRequest(const deployer::DeployerRequest request,
                                       deployer::DeployerResponse &response) {
  Status ret = SUCCESS;
  const int32_t kTryTimes = 61;
  // max wait 60s
  for (int32_t i = 0; i < kTryTimes; ++i) {
    if (i != 0) {
      (void) mmSleep(kInitTryWaitInterval);  // mmSleep max wait 1s
    }
    GEEVENT("[Send][Request] send init request, try time = %d, address = %s.", i + 1, node_config_.ipaddr.c_str());
    GE_CHECK_NOTNULL(client_.get());
    ret = client_->SendRequest(request, response);
    if (ret == SUCCESS && response.error_code() == SUCCESS) {
      break;
    }
  }
  return ret;
}

Status RemoteDeployer::Connect() {
  std::lock_guard<std::mutex> lk(mu_init_);
  if (connected_) {
    return connected_status_;
  }
  connected_ = true;
  deployer::DeployerRequest request;
  GE_CHK_STATUS_RET_NOLOG(InitRequest(request));
  GE_MAKE_GUARD(request, [&request]() {
    auto sign_data = request.mutable_init_request()->mutable_sign_data();
    sign_data->replace(sign_data->begin(), sign_data->end(), "");
  });
  deployer::DeployerResponse response;
  auto status = SendInitRequest(request, response);
  if (status != SUCCESS) {
    GELOGE(FAILED, "Failed to send init request");
    REPORT_PREDEFINED_ERR_MSG("E13027", std::vector<const char_t *>({"address"}),
        std::vector<const char_t *>({(node_config_.ipaddr + ":" + std::to_string(node_config_.port)).c_str()}));
    return status;
  }

  auto error_code = response.error_code();
  GE_CHK_STATUS_RET(error_code, "[Check][Response]Check response failed. error code =%u, error message=%s",
                    error_code, response.error_message().c_str());
  client_id_ = response.init_response().client_id();
  UpdateNodeInfo(response.init_response());
  keepalive_thread_ = std::thread([this]() {
    Keepalive();
  });
  connected_status_ = SUCCESS;
  GEEVENT("[Send][Request] add client succeeded, client_id = %ld, address = %s.",
          client_id_, node_config_.ipaddr.c_str());
  return SUCCESS;
}

Status RemoteDeployer::Disconnect() {
  GE_IF_BOOL_EXEC(exception_, return SUCCESS);
  std::lock_guard<std::mutex> lk(mu_init_);
  if (!connected_) {
    return SUCCESS;
  }
  deployer::DeployerRequest request;
  request.set_type(deployer::kDisconnect);
  request.set_client_id(client_id_);
  deployer::DeployerResponse response;
  // It takes up to 60s for the device to respond to the request
  constexpr int64_t kTimeout = 70;
  GE_CHK_STATUS_RET(client_->SendRequest(request, response, kTimeout),
                    "[Send][Request] Send disconnect request failed, response info:%s",
                    response.error_message().c_str());
  connected_ = false;
  connected_status_ = FAILED;
  return SUCCESS;
}

void RemoteDeployer::Keepalive() {
  SET_THREAD_NAME(pthread_self(), "ge_dpl_kalv2");
  GELOGI("Keepalive task started.");
  while (keep_alive_) {
    std::unique_lock<std::mutex> lk(mu_cv_);
    constexpr int32_t kSleepInterval = 100; // millisconds
    constexpr int32_t kSleepCnt = kHeartbeatInterval / kSleepInterval;
    for (int32_t i = 0; (i < kSleepCnt) && keep_alive_; i++) {
      std::this_thread::sleep_for(std::chrono::milliseconds(kSleepInterval));
    }
    if (keep_alive_) {
      SendHeartbeat(kHeartbeatTryMaxNum);
    }
  }
  GELOGI("Keepalive task ended.");
}

Status RemoteDeployer::Finalize() {
  if (!keep_alive_) {
    return SUCCESS;
  }
  GELOGI("Start to finalize remote deployer, remote address = %s, port = %d.",
         node_config_.ipaddr.c_str(),
         node_config_.port);
  keep_alive_ = false;
  if (keepalive_thread_.joinable()) {
    keepalive_thread_.join();
  }
  GE_CHK_STATUS_RET_NOLOG(Disconnect());
  return SUCCESS;
}

void RemoteDeployer::UpdateNodeInfo(const deployer::InitResponse &init_response) {
  if (!node_config_.lazy_connect) {
    auto dev_count = init_response.dev_count();
    auto support_flowgw_merged = init_response.support_flowgw_merged();
    int32_t offset = init_response.dgw_port_offset() % kDataGwPortMaxOffset;
    for (int32_t i = 0; i < dev_count; ++i) {
      DeviceInfo device_info;
      device_info.SetHostIp(node_config_.ipaddr);
      device_info.SetNodePort(node_config_.port);
      device_info.SetDeviceId(i);
      device_info.SetDeviceIp(node_config_.ipaddr);
      device_info.SetOsId(0);
      device_info.SetSupportHcom(true);
      device_info.SetResourceType(node_config_.resource_type);
      device_info.SetPhyDeviceId(i);
      auto hcom_device_id = support_flowgw_merged ? 0 : i;
      device_info.SetHcomDeviceId(hcom_device_id);
      device_info.SetDeviceIndex(i);
      device_info.SetDgwPort(kDataGwPortBase + offset + i);
      device_info.SetNodeMeshIndex(node_config_.node_mesh_index);
      device_info.SetAvailablePorts(node_config_.available_ports);
      device_info.SetSupportFlowgw(true);
      node_info_.AddDeviceInfo(device_info);
    }
  }
  node_info_.SetClientId(client_id_);
}

const NodeInfo &RemoteDeployer::GetNodeInfo() const {
  return node_info_;
}

std::unique_ptr<DeployerClient> RemoteDeployer::CreateClient() {
  return MakeUnique<DeployerClient>();
}

Status RemoteDeployer::Process(deployer::DeployerRequest &request, deployer::DeployerResponse &response) {
  GE_IF_BOOL_EXEC(exception_, return SUCCESS);
  GE_CHK_STATUS_RET(Connect(), "Failed to connect to remote deployer.");
  request.set_client_id(client_id_);
  return client_->SendRequest(request, response);
}

Status RemoteDeployer::Process(deployer::DeployerRequest &request,
                               deployer::DeployerResponse &response,
                               const int64_t timeout_sec,
                               const int32_t retry_times) {
  GE_IF_BOOL_EXEC(exception_, return SUCCESS);
  GE_CHK_STATUS_RET(Connect(), "Failed to connect to remote deployer.");
  std::future<Status> future = std::async(std::launch::async, [this, &request, &response, timeout_sec, retry_times]() {
    request.set_client_id(client_id_);
    Status ret = client_->SendRequestWithRetry(request, response, timeout_sec, retry_times);
    return ret;
  });
  constexpr int64_t kWaitSec = 5;
  while (true) {
    std::future_status status = future.wait_for(std::chrono::seconds(timeout_sec + kWaitSec));
    if (status == std::future_status::timeout) {
      GELOGW("Send request timeout, request type=%d.", static_cast<int32_t>(request.type()));
      continue;
    }
    break;
  }
  return future.get();
}

void RemoteDeployer::AddAbnormalDeviceInfo(int32_t device_id, int32_t device_type) {
  GELOGI("RemoteDeployer ParseRsponse: device id=%d, device type=%d", device_id, device_type);
  FormatAndAddAbnormalDeviceInfo(node_config_.node_id, device_id, device_type);
}

void RemoteDeployer::AddAbnormalNodeConfig() {
  GELOGI("RemoteDeployer ParseRsponse: node id=%d", node_config_.node_id);
  auto &deploy_context = DeployContext::LocalContext();
  std::lock_guard<std::mutex> lk(deploy_context.GetAbnormalHeartbeatInfoMu());
  deploy_context.AddAbnormalNodeConfig(node_config_);
}

void RemoteDeployer::SendHeartbeat(int32_t max_retry_times) {
  deployer::DeployerRequest request;
  request.set_type(deployer::kHeartbeat);
  deployer::DeployerResponse response;
  constexpr int64_t timeout_sec = 15;
  auto proc_ret = Process(request, response, timeout_sec, max_retry_times);
  // when response.error_code() == STOP, the device process state can be restored
  Status status = (proc_ret != SUCCESS || response.error_code() == FAILED) ? FAILED : SUCCESS;
  SetDevStat(status);
  if (proc_ret != SUCCESS) {
    // 心跳丢失看做对端整个node损坏
    response.mutable_heartbeat_response()->set_abnormal_type(kAbnormalTypeNode);
    GELOGW("Send heartbeat request failed, %s.", node_info_.DebugString().c_str());
  } else if (response.error_code() != SUCCESS) {
    GELOGW("Process on device is abnormal, %s, response info:%s",
           node_info_.DebugString().c_str(), response.error_message().c_str());
  } else {
    GELOGI("Success to send heartbeat to client_id[%ld], %s.", client_id_, node_info_.DebugString().c_str());
  }
  GE_IF_BOOL_EXEC(proc_ret != SUCCESS, exception_ = true);
  ParseRsponse(response);
}
}  // namespace ge
