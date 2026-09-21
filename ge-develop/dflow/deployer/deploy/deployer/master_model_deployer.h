/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef RUNTIME_DEPLOY_HETEROGENEOUS_MODEL_DEPLOYER_H_
#define RUNTIME_DEPLOY_HETEROGENEOUS_MODEL_DEPLOYER_H_

#include "ge/ge_ir_build.h"
#include "common/config/device_debug_config.h"
#include "dflow/base/deploy/model_deployer.h"
#include "dflow/base/deploy/deploy_planner.h"
#include "graph_metadef/common/plugin/plugin_manager.h"
#include "common/config/device_debug_config.h"
#include "common/data_flow/route/rank_table_builder.h"
#include "ge/ge_ir_build.h"
#include "deploy/flowrm/heterogeneous_exchange_deployer.h"
#include "deploy/deployer/deployer_proxy.h"
#include "deploy/abnormal_status_handler/abnormal_status_handler.h"

namespace ge {
class MasterModelDeployer : public ModelDeployer {
 public:
  MasterModelDeployer();
  ~MasterModelDeployer() override = default;

  Status Initialize(const std::map<std::string, std::string> &options);

  Status Finalize();

  Status DeployModel(const FlowModelPtr &flow_model, DeployResult &deploy_result) override;
  Status Undeploy(const uint32_t model_id) override;

  Status UpdateProfilingInfo(const bool is_prof_start) override;

  Status GetDeviceMeshIndex(const int32_t device_id, std::vector<int32_t> &node_mesh_index) override;
  Status GetValidLogicDeviceId(std::string &device_id) override;
 private:
  using ConstSubmodelInfoPtr = const DeployPlan::SubmodelInfo *;
  struct DeployedModel {
    uint32_t model_id = UINT32_MAX;
    // key: device_id, value: model_name

    std::vector<DeployPlan::DeviceInfo> deployed_remote_devices;
    // key: model_name, (key: model_instance_name, value: device_info)
    DeployPlan::ModelDeployInfo model_deploy_infos;
    std::set<int32_t> deployed_remote_nodes;
    int64_t route_id = -1;
  };

  struct SendInfo {
    uint64_t session_id;
    int32_t device_id;
    int32_t sub_device_id;
  };

  struct DeployFlowGwInfo {
    std::string rank_table;
    int32_t rank_id;
    std::vector<int32_t> res_ids;
    const DeviceInfo *device_info;
    bool profiling_on;
  };

  static Status GetDynamicSchedModelIoQueueIds(const DeployPlan &deploy_plan,
                                               const ExchangeRoute &route,
                                               DeployResult &deploy_result);
  static Status GetModelIoQueueAttrs(const DeployPlan &deploy_plan,
                                     const ExchangeRoute &route,
                                     DeployResult &deploy_result);
  static void UndeployModel(DeployedModel &deployed_model);
  Status InitFlowGwInfo();
  Status PrepareFlowGwInfos(const std::string &rank_table_str,
                            std::map<std::string, DeployFlowGwInfo> &deploy_flowgw_infos) const;
  Status InitProcessResourceRequest(const DeployFlowGwInfo &flowgw_info,
                                    const std::string &remote_group_cache_config) const;
  int32_t GetRankTableOrder();
  Status CreateRankTable(HcomRankTable &rank_table);
  static Status GetReplicaNum(const DeployState &state, size_t &replica_num);
  Status SetDeployResult(const DeployState &state, DeployResult &deploy_result);
  static void SetTrimmingModelInstanceNames(
      const std::map<std::string, std::vector<std::string>> &org_model_instance_names,
      std::vector<std::unordered_set<std::string>> &processed_model_instance_names);
  static Status GetBroadcastInputQueueAttrs(const DeployPlan &deploy_plan,
                                            const ExchangeRoute &route,
                                            std::vector<std::vector<DeployQueueAttr>> &broadcast_input_queue_attrs);
  void AbnormalStatusMonitorInitialize();
  void AbnormalStatusMonitorFinalize();
  Status DeployStateUpdate(DeployState &deploy_state, uint32_t root_model_id, const FlowModelPtr &flow_model);
  void IncDeployingRootModelNum();
  void DecreaseDeployingRootModelNum();
  void DelCallback(const uint32_t root_model_id);
  DeployPlan::AbnormalStatusCallbackInfo* GetAbnormalStatusCallbackInfo();
  void SetDynamicSchedFlag(bool flag);
  void AddDeployedModelInfo(uint32_t root_model_id);
  void DelDeployedModelInfo(uint32_t root_model_id);
  Status NotifyException(uint32_t root_model_id, const UserExceptionNotify &user_exception_notify);
  static Status GetRemoteGroupCacheConfig(std::string &remote_group_cache_config);

  std::mutex mu_;
  std::map<uint32_t, DeployedModel> deployed_models_;
  std::atomic<std::uint32_t> model_id_gen_{};
  bool flow_gw_inited_ = false;
  mutable std::mutex mu_for_init_;
  AbnormalStatusHandler abnormal_status_handler_;
};
}  // namespace ge
#endif  // RUNTIME_DEPLOY_HETEROGENEOUS_MODEL_DEPLOYER_H_
