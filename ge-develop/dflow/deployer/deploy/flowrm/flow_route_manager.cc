/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "deploy/flowrm/flow_route_manager.h"
#include "common/data_flow/queue/heterogeneous_exchange_service.h"
#include "common/debug/log.h"

namespace ge {
namespace {
using HService = HeterogeneousExchangeService;
}

Status FlowRouteManager::AddRoute(const ExchangeRoute &route, const FlowRouteType &route_type, int64_t &route_id) {
  GE_CHK_STATUS_RET_NOLOG(GenerateRouteId(route_id));
  std::lock_guard<std::mutex> lk(mu_);
  exchange_routes_[route_type][route_id] = route;
  return SUCCESS;
}

Status FlowRouteManager::GenerateRouteId(int64_t &route_id) {
  std::lock_guard<std::mutex> lk(mu_);
  if (route_id_gen_ < 0) {
    GELOGE(FAILED, "Run out of route id");  // will not happen
    return FAILED;
  }
  route_id = route_id_gen_++;
  return SUCCESS;
}

void FlowRouteManager::UndeployRoute(const FlowRouteType &route_type,
                                     int64_t route_id,
                                     FlowGwClientManager &client_manager) {
  std::lock_guard<std::mutex> lk(mu_);
  auto it = exchange_routes_.find(route_type);
  if (it != exchange_routes_.end()) {
    auto route_it = it->second.find(route_id);
    if (route_it != it->second.end()) {
      (void) HeterogeneousExchangeDeployer::Undeploy(HService::GetInstance(), route_it->second, client_manager);
      exchange_routes_[route_type].erase(route_it);
      return;
    }
  }
}

void FlowRouteManager::UpdateExceptionRoutes(const FlowRouteType &route_type,
                                             int64_t route_id,
                                             FlowGwClientManager &client_manager,
                                             const std::vector<FlowGwClient::ExceptionDeviceInfo> &devices) {
  std::lock_guard<std::mutex> lk(mu_);
  auto it = exchange_routes_.find(route_type);
  if (it != exchange_routes_.end()) {
    auto route_it = it->second.find(route_id);
    if (route_it != it->second.end()) {
      (void) HeterogeneousExchangeDeployer::UpdateExceptionRoutes(route_it->second,
                                                                  client_manager, devices);
      return;
    }
  }
}

const ExchangeRoute *FlowRouteManager::QueryRoute(const FlowRouteType &route_type, int64_t route_id) {
  std::lock_guard<std::mutex> lk(mu_);
  auto it = exchange_routes_.find(route_type);
  if (it != exchange_routes_.end()) {
    auto route_it = it->second.find(route_id);
    if (route_it != it->second.end()) {
      return &route_it->second;
    }
  }
  return nullptr;
}
}  // namespace ge
