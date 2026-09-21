/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef RUNTIME_DEPLOY_HETEROGENEOUS_DEPLOY_PLANNER_H_
#define RUNTIME_DEPLOY_HETEROGENEOUS_DEPLOY_PLANNER_H_

#include "dflow/base/deploy/deploy_planner.h"
#include "dflow/inc/data_flow/model/flow_model.h"

namespace ge {
class HeterogeneousDeployPlanner : public DeployPlannerBase {
 public:
  HeterogeneousDeployPlanner() = default;
  HeterogeneousDeployPlanner(const FlowModelPtr &flow_model,
                      std::vector<DeployPlan::DeviceInfo> device_list) noexcept;

  HeterogeneousDeployPlanner(std::vector<FlowModelPtr> models,
                      const ModelRelation *root_model_relation,
                      std::vector<DeployPlan::DeviceInfo> device_list) noexcept;
  static void GetValidLogicDeviceId(std::string &device_id);
  Status BuildTransferPlan(const DeployPlan::DeviceInfo &local_device,
                           const std::vector<DeployPlan::DeviceInfo> &target_devices,
                           DeployPlan &deploy_plan);

 protected:
  Status PrepareModelsAndRelation(ModelRelation &model_relation) override;
  void SelectHeadAndTailDevice(DeployPlan::DeviceInfo &device_info) override;

 private:
  Status CreateTransferInfo(const DeployPlan::DeviceInfo &src_device_info,
                            const std::vector<DeployPlan::DeviceInfo> &dst_device_infos);
  Status MergeModels(std::map<std::string, PneModelPtr> &name_to_models,
                     ModelRelation &model_relation);
  Status PrepareForSingleFlowModel(std::map<std::string, PneModelPtr> &name_to_models,
                                   ModelRelation &model_relation);
  Status UnfoldSubModel(const ModelRelation::ModelEndpointInfo &model_endpoint_info,
                        const FlowModel &model,
                        std::map<std::string, PneModelPtr> &name_to_models);
  Status SelectTargetDevice(const std::map<std::string, PneModelPtr> &name_to_models,
                            std::map<std::string, std::vector<std::pair<DeployPlan::DeviceInfo,
                                                                        bool>>> &target_devices);
  Status AssignDevices(const std::string &model_name,
                       const PneModelPtr &pne_model,
                       const std::string &engine_type,
                       const std::vector<DeployPlan::DeviceInfo> &device_list,
                       std::map<std::string, std::vector<std::pair<DeployPlan::DeviceInfo,
                                                                   bool>>> &target_devices);
  static void NormalizeLogicalDeviceId(std::vector<std::string> &logical_device_ids);
  static Status GetLogicalDeviceId(const PneModel &pne_model,
                                   const std::string &engine_type,
                                   std::vector<std::string> &logical_device_ids);
  static Status GetRedundantLogicalDeviceId(const PneModel &pne_model,
                                            std::vector<std::string> &logical_device_ids);
  static Status GetLogicalDeviceIdFromOption(std::string &logical_device_id);
  static Status ParseLogicalDeviceIds(const std::string &logical_device_id_str,
                                      std::vector<std::string> &logical_device_ids);
  Status ReindexDevices();
  static Status GetDeployEngineType(const PneModel &pne_model, std::string &deploy_type);
  Status PrepareTargetDevices(std::map<std::string, PneModelPtr> &name_to_models,
                              ModelRelation &model_relation);
  void RecordDeviceDeployedModelNum(const DeployPlan::DeviceInfo &device_info);
  static Status CheckAssignedDevice(const std::string &model_name,
                                    const std::string &engine_type,
                                    const std::string &device_index,
                                    const DeployPlan::DeviceInfo &device_info);

  void AddDynamicSchedRelation(std::map<std::string, PneModelPtr> &model_instance_to_models,
                               ModelRelation &resolved_model_relation);
  void AddDynamicSchedModel(std::map<std::string, PneModelPtr> &model_instance_to_models,
                            ModelRelation &resolved_model_relation,
                            std::map<std::string, ModelRelation::ModelEndpointInfo> &device_endpoint_info,
                            std::vector<DeployPlan::DeviceInfo> &flowgw_device_infos);
  Status AssignSubmodelsQueueDeviceInfo(const std::string &model_name,
                                        DeployPlan::SubmodelInfo &submodel_info) const;
  bool HasDeployedModelOnDevice(const DeployPlan::DeviceInfo &device_info) const;

  const std::vector<FlowModelPtr> models_;
  const ModelRelation *root_model_relation_{};
  std::vector<DeployPlan::DeviceInfo> device_list_;
  ModelRelation merged_model_relation_;
  std::map<std::string, DeployPlan::DeviceInfo> index_to_devices_;
  std::map<DeployPlan::DeviceInfo, uint32_t> device_deployed_model_num_;
};
}  // namespace ge
#endif  // RUNTIME_DEPLOY_HETEROGENEOUS_DEPLOY_PLANNER_H_
