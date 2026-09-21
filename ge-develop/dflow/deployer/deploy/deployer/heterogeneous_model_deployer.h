/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_RUNTIME_HETEROGENEOUS_DEPLOY_DEPLOYER_HETEROGENEOUS_MODEL_DEPLOYER_H_
#define AIR_RUNTIME_HETEROGENEOUS_DEPLOY_DEPLOYER_HETEROGENEOUS_MODEL_DEPLOYER_H_

#include "dflow/inc/data_flow/model/flow_model.h"
#include "deploy/deployer/deploy_state.h"
#include "deploy/deployer/deploy_context.h"

namespace ge {
class HeterogeneousModelDeployer {
 public:
  static Status DeployModel(DeployContext &deploy_context, DeployState &deploy_state);
  static void UndeployModel(const std::set<int32_t> &deployed_node_ids, uint32_t root_model_id);
  static Status ClearNodelExceptionData(uint32_t node_id, uint32_t model_id,
                                        const std::vector<DeployPlan::DeviceInfo> &device_infos,
                                        const int32_t type);
  static Status UpdateRemoteProfiling(const bool is_prof_start, const std::string &prof_data,
                                      const std::map<uint32_t, std::set<int32_t>> &model_id_to_nodes);
 private:
  static Status DoDeployModel(DeployContext &deploy_context, DeployState &deploy_state);
  static Status DoDeployModelWithFlow(DeployContext &deploy_context, DeployState &deploy_state);
  static Status BuildDeployPlan(DeployState &deploy_state);
  static Status LoadSubmodels(DeployContext &deploy_context, DeployState &deploy_state);
  static Status LoadLocalModel(DeployContext &deploy_context, uint32_t root_model_id);
  static Status LoadRemoteModel(int32_t node_id, uint32_t root_model_id);
  static Status DoUndeployModel(int32_t node_id, uint32_t root_model_id);
  static void BuildModelAttrs(const FlowModelPtr &flow_model, DeployPlan &deploy_plan);
};
}  // namespace ge

#endif  // AIR_RUNTIME_HETEROGENEOUS_DEPLOY_DEPLOYER_HETEROGENEOUS_MODEL_DEPLOYER_H_
