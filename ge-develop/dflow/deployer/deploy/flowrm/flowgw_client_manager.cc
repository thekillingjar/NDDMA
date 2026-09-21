/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "deploy/flowrm/flowgw_client_manager.h"
#include "framework/common/debug/log.h"
#include "common/subprocess/subprocess_manager.h"

namespace ge {
FlowGwClient *FlowGwClientManager::CreateClient(int32_t device_id,
                                                int32_t device_type,
                                                const std::vector<int32_t> &res_ids,
                                                bool is_proxy) const {
  return new (std::nothrow)FlowGwClient(device_id, device_type, res_ids, is_proxy);
}

FlowGwClient *FlowGwClientManager::GetOrCreateClient(int32_t device_id,
                                                     int32_t device_type,
                                                     const std::vector<int32_t> &res_ids,
                                                     bool is_proxy) {
  auto key = std::make_pair(device_id, device_type);
  {
    std::lock_guard<std::mutex> lock(mutex_);
    const auto &it = device_to_client_index_.find(key);
    if (it != device_to_client_index_.cend()) {
      return clients_[it->second].get();
    }
  }
  GELOGI("Begin to create client, device_id = %d, device_type = %d, res_ids = %s.",
         device_id, device_type, ToString(res_ids).c_str());
  auto client = CreateClient(device_id, device_type, res_ids, is_proxy);
  if (client == nullptr) {
    GELOGE(FAILED, "Failed to create flowgw client.");
    return nullptr;
  }
  GE_DISMISSABLE_GUARD(client, ([client]() { delete client; }));
  const auto ret = client->Initialize();
  if (ret != SUCCESS) {
    GELOGE(FAILED, "Failed to initialize flowgw client.");
    return nullptr;
  }
  // will be freed by Finalize
  GE_DISMISS_GUARD(client);
  std::lock_guard<std::mutex> lock(mutex_);
  clients_.emplace_back(std::move(std::unique_ptr<FlowGwClient>(client)));
  for (auto res_id : res_ids) {
    auto res_key = std::make_pair(res_id, device_type);
    device_to_client_index_[res_key] = clients_.size() - 1U;
  }
  device_to_client_index_[key] = clients_.size() - 1U;
  GEEVENT("FlowGw client initialized successfully, device_id = %d, device_type = %d, res_ids = %s.",
          device_id, device_type, ToString(res_ids).c_str());
  return clients_.back().get();
}

FlowGwClient *FlowGwClientManager::GetClient(int32_t device_id, int32_t device_type) {
  auto key = std::make_pair(device_id, device_type);
  const std::lock_guard<std::mutex> lock(mutex_);
  const auto &it = device_to_client_index_.find(key);
  if (it != device_to_client_index_.cend()) {
    GELOGI("Get client successfully, device_id = %d, device_type = %d.", device_id, device_type);
    return clients_[it->second].get();
  }
  GELOGE(FAILED, "Failed to get flowgw client, device_id = %d, device_type = %d.", device_id, device_type);
  return nullptr;
}

Status FlowGwClientManager::GetHcomHandle(int32_t device_id,
                                          int32_t device_type,
                                          uint64_t &handle) {
  auto client = GetClient(device_id, device_type);
  GE_CHECK_NOTNULL(client);
  return client->GetOrCreateHcomHandle(handle);
}

Status FlowGwClientManager::WaitAllClientConfigEffect() {
  GEEVENT("Wait all client config effect begin.");
  const std::lock_guard<std::mutex> lock(mutex_);
  for (const auto &client : clients_) {
    GE_CHK_STATUS_RET(client->WaitConfigEffect(), "Failed to wait client effect, device_id = %u, device_type = %d.",
                      client->GetDeviceId(), client->GetDeviceType());
  }
  GEEVENT("Wait all client config effect success.");
  return SUCCESS;
}

Status FlowGwClientManager::CreateFlowGwGroup(
    int32_t device_id,
    int32_t device_type,
    const std::vector<const ExchangeEndpoint *> &endpoint_list,
    int32_t &group_id) {
  auto client = GetClient(device_id, device_type);
  GE_CHECK_NOTNULL(client);
  GE_CHK_STATUS_RET(client->CreateFlowGwGroup(endpoint_list, group_id), "Failed to create flowgw group.");
  return SUCCESS;
}

Status FlowGwClientManager::DestroyFlowGwGroup(int32_t device_id,
                                               int32_t device_type,
                                               int32_t group_id) {
  auto client = GetClient(device_id, device_type);
  GE_CHECK_NOTNULL(client);
  GE_CHK_STATUS_RET(client->DestroyFlowGwGroup(group_id), "Failed to destroy flowgw group.");
  return SUCCESS;
}

Status FlowGwClientManager::SelectTargetClientIndex(const FlowGwClient::RouteDeviceInfo &route_device_info,
                                                    size_t &client_index) {
  int32_t target_device_id = route_device_info.src_device_id;
  int32_t target_device_type = route_device_info.src_device_type;
  if ((route_device_info.src_device_type != route_device_info.dst_device_type) &&
      (route_device_info.src_device_type == static_cast<int32_t>(NPU))) {
    target_device_id = route_device_info.dst_device_id;
    target_device_type = route_device_info.dst_device_type;
  }
  const std::lock_guard<std::mutex> lock(mutex_);
  const auto &it =  device_to_client_index_.find(std::make_pair(target_device_id, target_device_type));
  GE_CHK_BOOL_RET_STATUS(it != device_to_client_index_.cend(), FAILED,
                         "Failed to get target client, src device_id = %d, src device_type = %d, "
                         "dst device_id = %d, dst device_type = %d, "
                         "target device_id = %d, target device_type = %d",
                         route_device_info.src_device_id, route_device_info.src_device_type,
                         route_device_info.dst_device_id, route_device_info.dst_device_type,
                         target_device_id, target_device_type);
  client_index = it->second;
  return SUCCESS;
}

Status FlowGwClientManager::BindQueues(
    const std::vector<std::pair<const ExchangeEndpoint *,
                                const ExchangeEndpoint *>> &queue_routes) {
  GELOGI("[Bind][Routes], routes size = %zu", queue_routes.size());
  std::map<size_t, std::vector<std::pair<const ExchangeEndpoint *,
                                         const ExchangeEndpoint *>>> index_to_routes;
  for (size_t i = 0U; i < queue_routes.size(); ++i) {
    FlowGwClient::RouteDeviceInfo device_info = {};
    device_info.src_device_id = queue_routes[i].first->device_id;
    device_info.dst_device_id = queue_routes[i].second->device_id;
    device_info.src_device_type = queue_routes[i].first->device_type;
    device_info.dst_device_type = queue_routes[i].second->device_type;
    size_t client_index = 0U;
    GE_CHK_STATUS_RET(SelectTargetClientIndex(device_info, client_index),
                      "Failed to select target client index.");
    index_to_routes[client_index].emplace_back(queue_routes[i]);
  }

  for (const auto &it : index_to_routes) {
    auto client = clients_[it.first].get();
    GE_CHK_STATUS_RET(client->BindQueues(it.second), "Failed to bind routes.");
  }
  return SUCCESS;
}

Status FlowGwClientManager::UnbindQueues(
    const std::vector<std::pair<const ExchangeEndpoint *,
                                const ExchangeEndpoint *>> &queue_routes) {
  GELOGI("[Unbind][Routes], routes size = %zu", queue_routes.size());
  std::map<size_t, std::vector<std::pair<const ExchangeEndpoint *,
                                         const ExchangeEndpoint *>>> index_to_routes;
  for (size_t i = 0U; i < queue_routes.size(); ++i) {
    FlowGwClient::RouteDeviceInfo device_info = {};
    device_info.src_device_id = queue_routes[i].first->device_id;
    device_info.dst_device_id = queue_routes[i].second->device_id;
    device_info.src_device_type = queue_routes[i].first->device_type;
    device_info.dst_device_type = queue_routes[i].second->device_type;
    size_t client_index = 0U;
    GE_CHK_STATUS_RET(SelectTargetClientIndex(device_info, client_index),
                      "Failed to select target client index.");
    index_to_routes[client_index].emplace_back(queue_routes[i]);
  }

  for (const auto &it : index_to_routes) {
    auto client = clients_[it.first].get();
    GE_CHK_STATUS_RET(client->UnbindQueues(it.second), "Failed to unbind routes.");
  }
  return SUCCESS;
}

Status FlowGwClientManager::UpdateExceptionRoutes(
    const std::vector<std::pair<const ExchangeEndpoint *,
                                const ExchangeEndpoint *>> &exception_routes,
    std::map<int32_t, std::shared_ptr<ExchangeEndpoint>> &endpoints) {
  std::map<size_t, std::vector<std::pair<const ExchangeEndpoint *,
                                         const ExchangeEndpoint *>>> index_to_routes;
  for (size_t i = 0U; i < exception_routes.size(); ++i) {
    FlowGwClient::RouteDeviceInfo exception_device_info = {};
    exception_device_info.src_device_id = exception_routes[i].first->device_id;
    exception_device_info.dst_device_id = exception_routes[i].second->device_id;
    exception_device_info.src_device_type = exception_routes[i].first->device_type;
    exception_device_info.dst_device_type = exception_routes[i].second->device_type;
    size_t client_index = 0U;
    GE_CHK_STATUS_RET(SelectTargetClientIndex(exception_device_info, client_index),
                      "[UpdateExceptionRoutes] Failed to select target client index.");
    GELOGI("[UpdateExceptionRoutes] find exception route client index, src endpoint = %s, dst endpoint = %s,"
           " client_index = %d.",
           exception_routes[i].first->DebugString().c_str(), exception_routes[i].second->DebugString().c_str(),
           client_index);
    index_to_routes[client_index].emplace_back(exception_routes[i]);
  }

  auto parallel_num = index_to_routes.size();
  std::vector<std::future<Status>> route_fut_rets;
  ThreadPool pool("ge_dpl_uer", parallel_num, false);
  for (const auto &it : index_to_routes) {
    auto client = clients_[it.first].get();
    auto &device_exception_routes = it.second;
    auto fut = pool.commit([this, client, &device_exception_routes, &endpoints]() -> Status {
      GE_CHK_STATUS_RET_NOLOG(client->UpdateExceptionRoutes(device_exception_routes, endpoints));
      return SUCCESS;
    });
    route_fut_rets.emplace_back(std::move(fut));
  }

  for (auto &fut : route_fut_rets) {
    if (fut.get() != SUCCESS) {
      return FAILED;
    }
  }
  return SUCCESS;
}

Status FlowGwClientManager::Finalize() {
  GEEVENT("Finalize begin.");
  const std::lock_guard<std::mutex> lock(mutex_);
  std::vector<std::thread> threads;
  for (const auto &client : clients_) {
    threads.emplace_back([&client]() {
      GE_CHK_STATUS(client->DestroyHcomHandle(), "Failed to destroy hcom handle.");
      (void) client->Finalize();
    });
  }
  for (auto &th : threads) {
    if (th.joinable()) {
      th.join();
    }
  }
  clients_.clear();
  device_to_client_index_.clear();
  GEEVENT("Finalize success.");
  return SUCCESS;
}

void FlowGwClientManager::ResponseDeviceInfoFormat(deployer::DeployerResponse &response,
    const std::unique_ptr<FlowGwClient> &client) const {
  auto dev_status = response.mutable_heartbeat_response()->add_device_status();
  dev_status->set_device_id(client->GetDeviceId());
  dev_status->set_device_type(client->GetDeviceType());
  return;
}

void FlowGwClientManager::ResponseErrorInfoFormat(deployer::DeployerResponse &response) const {
  response.mutable_heartbeat_response()->set_abnormal_type(kAbnormalTypeDevice);
  response.set_error_code(FAILED);
  response.set_error_message("Queue schedule abnormal.");
  return;
}

void FlowGwClientManager::GetFlowGwStatus(deployer::DeployerResponse &response) {
  const std::lock_guard<std::mutex> lock(mutex_);
  for (auto &client : clients_) {
    auto status = client->GetSubProcStat();
    if ((status == ProcStatus::NORMAL) || (status == ProcStatus::INVALID)) {
      continue;
    } else if ((status == ProcStatus::EXITED) || (status == ProcStatus::STOPPED)) {
      const std::string status_str = status == ProcStatus::EXITED ? "exited" : "stopped";
      GELOGW("Queue schedule %s, device id[%d], device type[%d].", status_str.c_str(),
          client->GetDeviceId(), client->GetDeviceType());
      ResponseDeviceInfoFormat(response, client);
      client->SetExceptionFlag();
    } else {
      // do nothing
    }
  }
  if (response.heartbeat_response().device_status_size() > 0) {
    ResponseErrorInfoFormat(response);
  }
  return;
}

Status FlowGwClientManager::ConfigSchedInfoToDataGw(const uint32_t device_id, const int32_t device_type,
                                                    const bool is_proxy, const int32_t input_indice,
                                                    const uint32_t input, const uint32_t output,
                                                    const uint32_t root_model_id) {
  auto client = GetClient(device_id, device_type);
  GE_CHECK_NOTNULL(client);
  GE_CHK_STATUS_RET(client->ConfigSchedInfoToDataGw(device_id, input_indice, input, output, root_model_id, is_proxy),
                    "Failed to config sched info to datagw.");
  return SUCCESS;
}
}  // namespace ge
