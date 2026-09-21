/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_RUNTIME_HETEROGENEOUS_FLOWRM_FLOW_ROUTE_MANAGER_H_
#define AIR_RUNTIME_HETEROGENEOUS_FLOWRM_FLOW_ROUTE_MANAGER_H_

#include <map>
#include <mutex>
#include "deploy/flowrm/heterogeneous_exchange_deployer.h"

namespace ge {
enum class FlowRouteType {
  kModelFlowRoute = 1,
  kTransferFlowRoute = 2
};

class FlowRouteManager {
 public:
  static FlowRouteManager &GetInstance() {
    static FlowRouteManager instance;
    return instance;
  }

  Status AddRoute(const ExchangeRoute &route, const FlowRouteType &route_type, int64_t &route_id);

  void UndeployRoute(const FlowRouteType &route_type, int64_t route_id, FlowGwClientManager &client_manager);

  const ExchangeRoute *QueryRoute(const FlowRouteType &route_type, int64_t route_id);

  void UpdateExceptionRoutes(const FlowRouteType &route_type, int64_t route_id,
                             FlowGwClientManager &client_manager,
                             const std::vector<FlowGwClient::ExceptionDeviceInfo> &devices);
 private:
  Status GenerateRouteId(int64_t &route_id);
  std::mutex mu_;
  int64_t route_id_gen_ = 0;
  // key: route type, sub_key: route id
  std::map<FlowRouteType, std::map<int64_t, ExchangeRoute>> exchange_routes_;
};
}  // namespace ge

#endif  // AIR_RUNTIME_HETEROGENEOUS_FLOWRM_FLOW_ROUTE_MANAGER_H_
