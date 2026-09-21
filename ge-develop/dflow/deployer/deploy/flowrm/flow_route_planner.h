/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_RUNTIME_HETEROGENEOUS_DEPLOY_FLOWRM_FLOW_ROUTE_PLANNER_H_
#define AIR_RUNTIME_HETEROGENEOUS_DEPLOY_FLOWRM_FLOW_ROUTE_PLANNER_H_

#include "ge/ge_api_error_codes.h"
#include "dflow/base/deploy/deploy_planner.h"
#include "deploy/deployer/deploy_state.h"
#include "proto/deployer.pb.h"

namespace ge {
struct PlanAttrs {
  uint32_t root_model_id = 0;
  bool is_dynamic_sched = false;
  bool keep_out_of_order = false;
};
class FlowRoutePlanner {
 public:
  static Status ResolveFlowRoutePlans(DeployState &deploy_state);
  static Status ResolveFlowRoutePlan(const DeployPlan &deploy_plan,
                                     int32_t node_id,
                                     deployer::FlowRoutePlan &flow_route_plan,
                                     const PlanAttrs &plan_attrs = {});
  static void PrintFlowRoutePlan(const deployer::FlowRoutePlan &flow_route_plan);
  static void PrintEndpointDesc(const deployer::FlowRoutePlan &flow_route_plan,
                                const deployer::EndpointDesc &endpoint_desc,
                                std::set<int32_t> &printed);
 private:
  static Status ResolveEndpoints(const DeployPlan &deploy_plan,
                                 int32_t node_id,
                                 deployer::FlowRoutePlan &flow_route_plan,
                                 const PlanAttrs &plan_attrs);
  static Status FillGroupEndpoint(const DeployPlan &deploy_plan,
                                  int32_t group_index, bool keep_out_of_order,
                                  deployer::EndpointDesc &endpoint_desc);
  static Status ResolveBindings(const DeployPlan &deploy_plan,
                                int32_t node_id,
                                deployer::FlowRoutePlan &flow_route_plan,
                                const uint32_t root_model_id);
  static Status AddFlowRoutePlanBindings(const DeployPlan &deploy_plan,
                                         int32_t node_id,
                                         std::vector<DeployPlan::DeviceInfo> &devices,
                                         deployer::FlowRoutePlan &flow_route_plan,
                                         std::map<std::string, std::set<size_t>> &relative_tag_indices);
  static Status AddGroupEntries(const DeployPlan &deploy_plan,
                                int32_t index,
                                std::vector<DeployPlan::DeviceInfo> &devices,
                                std::map<std::string, std::set<size_t>> &relative_tag_indices);
  static uint32_t GetTagIdByName(const std::string &tag_name);
  static Status GetDeviceHcomInfo(const DeployPlan::DeviceInfo &device_info, uint32_t &rank_id);
  static Status GetDeviceRankId(const DeployPlan::DeviceInfo &device_info, uint32_t &rank_id);

  static void SetEndpointType(const DeployPlan::QueueInfo &endpoint_info, deployer::EndpointDesc &endpoint_desc);
  static Status AssignTagInfo(const DeployPlan::DeviceInfo &device_info, deployer::TagDesc &tag_desc,
                              const std::string &tag_name, uint32_t index);
};
}  // namespace ge
#endif  // AIR_RUNTIME_HETEROGENEOUS_DEPLOY_FLOWRM_FLOW_ROUTE_PLANNER_H_
