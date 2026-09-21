/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_RUNTIME_HETEROGENEOUS_FLOWRM_FLOWGW_CLIENT_MANAGER_H_
#define AIR_RUNTIME_HETEROGENEOUS_FLOWRM_FLOWGW_CLIENT_MANAGER_H_

#include <string>
#include <vector>
#include <mutex>
#include "ge/ge_api_types.h"
#include "ge/ge_api_error_codes.h"
#include "aicpu/queue_schedule/dgw_client.h"
#include "aicpu/queue_schedule/qs_client.h"
#include "deploy/flowrm/flowgw_client.h"

namespace ge {
class FlowGwClientManager {
 public:
  FlowGwClientManager() = default;
  virtual ~FlowGwClientManager() = default;

  FlowGwClient *GetOrCreateClient(int32_t device_id,
                                  int32_t device_type,
                                  const std::vector<int32_t> &res_ids,
                                  bool is_proxy);

  FlowGwClient *GetClient(int32_t device_id, int32_t device_type);

  Status GetHcomHandle(int32_t device_id, int32_t device_type, uint64_t &handle);

  Status WaitAllClientConfigEffect();

  Status CreateFlowGwGroup(int32_t device_id,
                           int32_t device_type,
                           const std::vector<const ExchangeEndpoint *> &endpoint_list,
                           int32_t &group_id);

  Status DestroyFlowGwGroup(int32_t device_id,
                            int32_t device_type,
                            int32_t group_id);

  Status Finalize();

  Status BindQueues(const std::vector<std::pair<const ExchangeEndpoint *,
                                                const ExchangeEndpoint *>> &queue_routes);

  Status UnbindQueues(const std::vector<std::pair<const ExchangeEndpoint *,
                                                  const ExchangeEndpoint *>> &queue_routes);

  void GetFlowGwStatus(deployer::DeployerResponse &response);

  void ResponseDeviceInfoFormat(deployer::DeployerResponse &response,
      const std::unique_ptr<FlowGwClient> &client) const;

  void ResponseErrorInfoFormat(deployer::DeployerResponse &response) const;

  Status ConfigSchedInfoToDataGw(const uint32_t device_id, const int32_t device_type, const bool is_proxy,
                                 const int32_t input_indice, const uint32_t input, const uint32_t output,
                                 const uint32_t root_model_id);

  Status UpdateExceptionRoutes(const std::vector<std::pair<const ExchangeEndpoint *,
                               const ExchangeEndpoint *>> &exception_routes,
                               std::map<int32_t, std::shared_ptr<ExchangeEndpoint>> &endpoints);

  const std::vector<std::unique_ptr<FlowGwClient>>& GetClients() const {
    return clients_;
  }
 protected:
  virtual FlowGwClient *CreateClient(int32_t device_id,
                                     int32_t device_type,
                                     const std::vector<int32_t> &res_ids,
                                     bool is_proxy) const;

 private:
  Status SelectTargetClientIndex(const FlowGwClient::RouteDeviceInfo &route_device_info,
                                 size_t &client_index);

  std::mutex mutex_;
  std::vector<std::unique_ptr<FlowGwClient>> clients_;
  // key:device_id + device_type, value:client index
  std::map<std::pair<int32_t, int32_t>, size_t> device_to_client_index_;
};
}
#endif // AIR_RUNTIME_HETEROGENEOUS_FLOWRM_FLOWGW_CLIENT_MANAGER_H_
