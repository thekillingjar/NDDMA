/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_RUNTIME_HETEROGENEOUS_FLOWRM_HETEROGENEOUS_EXCHANGE_DEPLOYER_H_
#define AIR_RUNTIME_HETEROGENEOUS_FLOWRM_HETEROGENEOUS_EXCHANGE_DEPLOYER_H_

#include <cstdint>
#include <vector>
#include "dflow/base/deploy/exchange_service.h"
#include "aicpu/queue_schedule/qs_client.h"
#include "aicpu/queue_schedule/dgw_client.h"
#include "ge/ge_api_error_codes.h"
#include "proto/deployer.pb.h"
#include "deploy/flowrm/flowgw_client_manager.h"

namespace ge {
class ExchangeRoute {
 public:
  Status GetQueueId(int32_t queue_index, uint32_t &queue_id) const;
  Status GetQueueIds(const std::vector<int32_t> &queue_indices, std::vector<uint32_t> &queue_ids) const;
  void GetQueueIds(std::vector<uint32_t> &queue_ids) const;
  Status GetQueueAttr(int32_t queue_index, DeployQueueAttr &queue_attr) const;
  Status GetQueueAttrs(const std::vector<int32_t> &queue_indices,
                       std::vector<DeployQueueAttr> &queue_attrs) const;
  void GetQueueAttrs(std::vector<DeployQueueAttr> &queue_attrs) const;
  Status GetFusionOffset(int32_t queue_index, int32_t &fusion_offset) const;
  Status GetFusionOffsets(const std::vector<int32_t> &queue_indices,
                          std::vector<int32_t> &fusion_offsets) const;
  uint32_t GetModelId(int32_t index) const {
    ExchangeEndpoint *point = MutableEndpoint(index);
    return point == nullptr ? 0U : point->model_id;
  }
  bool IsProxyQueue(int32_t queue_index) const;

 private:
  friend class HeterogeneousExchangeDeployer;

  ExchangeEndpoint *MutableEndpoint(int32_t index, ExchangeEndpointType expect_type) const;
  ExchangeEndpoint *MutableEndpoint(int32_t index, int32_t expect_type) const;
  ExchangeEndpoint *MutableEndpoint(int32_t index) const;

  // key: queue index, value: queue_handle
  mutable std::map<int32_t, std::shared_ptr<ExchangeEndpoint>> endpoints_;
  std::vector<std::pair<const ExchangeEndpoint *,
                        const ExchangeEndpoint *>> queue_routes_;
  std::map<int32_t, std::vector<const ExchangeEndpoint *>> groups_;
};

class HeterogeneousExchangeDeployer {
 public:
  HeterogeneousExchangeDeployer(ExchangeService &exchange_service,
                         deployer::FlowRoutePlan route_plan,
                         FlowGwClientManager &client_manager);
  virtual ~HeterogeneousExchangeDeployer();

  Status PreDeploy();
  Status Deploy(ExchangeRoute &deployed, bool wait_config_effect = false);
  static Status Undeploy(ExchangeService &exchange_service,
                         const ExchangeRoute &deployed,
                         FlowGwClientManager &client_manager);
  const ExchangeRoute *GetRoute() const;
  ExchangeRoute *MutableRoute();
  static Status UpdateExceptionRoutes(ExchangeRoute &deployed,
                                      FlowGwClientManager &client_manager,
                                      const std::vector<FlowGwClient::ExceptionDeviceInfo> &exception_devices);

 private:
  Status CreateHcomHandles();
  Status CreateExchangeEndpoints();
  Status GetSrcEndpointIndices(std::set<int32_t> &src_endpoint_indices);
  Status CreateGroupTags(int32_t group_index,
                         const deployer::EndpointDesc &endpoint_desc,
                         ExchangeEndpoint &endpoint);

  Status CreateGroups(const std::vector<int32_t> &group_indices);
  Status BindEndpoints(const std::vector<deployer::EndpointBinding> &bindings);
  std::vector<deployer::EndpointBinding> GetBindingsBeforeLoad() const;
  std::vector<deployer::EndpointBinding> GetBindingsAfterLoad() const;
  std::vector<deployer::EndpointBinding> GetAllBindings() const;

  virtual Status BindRoute(const std::vector<std::pair<const ExchangeEndpoint *,
                                                       const ExchangeEndpoint *>> &queue_routes);
  static Status UnbindRoute(const std::vector<std::pair<const ExchangeEndpoint *,
                                                        const ExchangeEndpoint *>> &queue_routes,
                            FlowGwClientManager &client_manager);
  virtual Status CreateGroup(int32_t group_index, ExchangeEndpoint &group_endpoint);
  const deployer::EndpointDesc *GetEndpointDesc(int32_t index);
  Status DoCreateQueue(const deployer::QueueDesc &queue_desc,
                       uint32_t work_mode,
                       ExchangeEndpoint &endpoint) const;
  Status DoCreateTag(const deployer::TagDesc &tag_desc, ExchangeEndpoint &endpoint) const;

  static bool CheckExceptionEndpoint(const ExchangeEndpoint *endpoint,
                                     const std::vector<FlowGwClient::ExceptionDeviceInfo> &devices);

  static bool CheckExceptionBind(ExchangeEndpoint *endpoint,
                                 const std::vector<FlowGwClient::ExceptionDeviceInfo> &devices,
                                 std::vector<const ExchangeEndpoint *> &del_endpoints, ExchangeRoute &deployed);
  void UpdateCommonInfo(ExchangeEndpoint &endpoint, const deployer::EndpointDesc &endpoint_desc);
  static void DeleteExceptionGroup(ExchangeRoute &deployed, const std::vector<FlowGwClient::ExceptionDeviceInfo> &devices);
  ExchangeService &exchange_service_;
  deployer::FlowRoutePlan route_plan_;
  bool pre_deployed_ = false;
  ExchangeRoute deploying_;
  FlowGwClientManager &client_manager_;
};
}  // namespace ge
#endif  // AIR_RUNTIME_HETEROGENEOUS_FLOWRM_HETEROGENEOUS_EXCHANGE_DEPLOYER_H_
