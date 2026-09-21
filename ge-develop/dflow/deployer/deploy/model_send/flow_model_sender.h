/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_RUNTIME_HETEROGENEOUS_DEPLOY_MODEL_SEND_FLOW_MODEL_SENDER_H_
#define AIR_RUNTIME_HETEROGENEOUS_DEPLOY_MODEL_SEND_FLOW_MODEL_SENDER_H_

#include <map>
#include "proto/deployer.pb.h"
#include "ge_common/ge_api_types.h"
#include "dflow/base/deploy/deploy_planner.h"
#include "common/config/device_debug_config.h"
#include "deploy/deployer/deploy_state.h"

namespace ge {
class FlowModelSender {
 public:
  FlowModelSender() = default;
  ~FlowModelSender();

  static Status DeployDevMaintenanceCfg(const DeployState &deploy_state);

  static Status TransferDeployPlan(const DeployState &deploy_state);

  static Status TransferFlowRoutePlan(const DeployState &deploy_state);

  static Status TransferSubmodels(DeployState &deploy_state);

  static Status TransferModel(int32_t node_id,
                              const DeployState &deploy_state,
                              const PneModelPtr &model,
                              const ModelBufferData model_buff);

  Status DeployRemoteVarManager(DeployState &deploy_state);
  Status DeployRemoteVarManager(const std::map<std::string,
                                               std::vector<const DeployPlan::SubmodelInfo *>> &models);
  static Status TransferDataGwDeployPlan(DeployState &deploy_state);

 private:
  struct SendInfo {
    uint64_t session_id;
    int32_t node_id;
    std::vector<int32_t> device_ids;
  };

  static Status GetDeviceInfo(int32_t node_id, int32_t device_id, int32_t device_type,
                              DeployPlan::DeviceInfo &deploy_device_info);

  Status TransferPreDeploy(const SendInfo &send_info,
                           ExchangeRoute &local_route,
                           deployer::FlowRoutePlan &remote_plan) const;
  static Status TransferFile(int32_t target_node_id, const std::string &path, const void *content, size_t size);

  static Status SerializeModel(const PneModelPtr &model, ModelBufferData &model_buff);

  static Status GetAllRelatedVarManager(
      const DeployPlan::DeviceInfo &device_info, const std::vector<const DeployPlan::SubmodelInfo *> &submodels,
      std::map<int32_t, std::set<uint64_t>> &sessions,
      std::map<int32_t, std::map<uint64_t, std::map<OpDescPtr, std::set<int32_t>>>> &node_need_transfer_memory,
      std::map<int32_t, std::set<int32_t>> &device_ids);

  Status TransferFileConstants(
      const std::map<int32_t, std::set<int32_t>> &device_ids,
      const std::map<int32_t, std::map<uint64_t, std::map<OpDescPtr, std::set<int32_t>>>> &node_need_transfer_memory);

  Status CopyOneWeightToTransfer(const SendInfo &send_info,
                                 std::istream &input_stream,
                                 int64_t file_constant_size,
                                 const OpDescPtr &op_desc,
                                 const std::set<int32_t> &devices);

  Status TransferWeightWithQ(std::istream &input_stream,
                             int64_t file_constant_size,
                             const DeployQueueAttr &queue_attr) const;

  static Status CreateInputStream(const std::string &constant_file_path, size_t offset,
                                  std::unique_ptr<std::istream> &in_stream);

  static Status DownloadDevMaintenanceCfg(int32_t dev_id);
  static Status DeployDevCfg(int32_t dev_id, DeviceDebugConfig::ConfigType conf_type);
  static std::string GetModelFilePath(const DeployState &deploy_state, const std::string &model_name);
  static Status GetSavedFilePath(const DeployPlan::SubmodelInfo &submodel_info, const std::string &model_file_path,
                                 std::string &saved_model_path);
  static void BuildDeployPlanOptions(const DeployState &deploy_state,
                                     deployer::UpdateDeployPlanRequest &request);
  static Status BuildUpdateDeployPlanRequest(
      const DeployState &deploy_state,
      const DeployPlan::DeviceInfo &target_device,
      std::vector<deployer::SubmodelDesc> &submodel_descs,
      deployer::DeployerRequest &request);
  static void AddDynamicSchedInfo(const DeployState &deploy_state, const std::string &model_instance_name,
                                  deployer::SubmodelDesc &submodel_desc);
  static Status BuildSubmodelDescs(const DeployState &deploy_state,
                                   std::map<std::string, std::vector<deployer::SubmodelDesc>> &submodel_descs,
                                   std::map<std::string, const DeployPlan::DeviceInfo *> &devices_used);

  Status GetOrCreateFlowRoutePlan(const SendInfo &send_info, deployer::FlowRoutePlan &remote_route);

  static bool CacheLocalModel(const DeployPlan::SubmodelInfo &submodel);

  static Status GetOpFileInfo(const OpDescPtr &op_desc,
                              const std::map<std::string, std::string> &file_id_to_path,
                              std::string &file_path,
                              size_t &offset,
                              size_t &length);
  static Status SendDatagwSchedInfo(std::map<std::string, const DeployPlan::DeviceInfo *> &datagw_devices_used,
                                    std::map<std::string, deployer::DeployerRequest> &datagw_sched_infos);
  std::map<int32_t, deployer::FlowRoutePlan> node_id_to_plan_;
  DeployPlan::DeviceInfo head_device_;
};
}  // namespace ge

#endif  // AIR_RUNTIME_HETEROGENEOUS_DEPLOY_MODEL_SEND_FLOW_MODEL_SENDER_H_
