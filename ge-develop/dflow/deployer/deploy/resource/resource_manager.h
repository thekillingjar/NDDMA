/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_RUNTIME_HETEROGENEOUS_DEPLOY_RESOURCE_RESOURCE_MANAGER_H_
#define AIR_RUNTIME_HETEROGENEOUS_DEPLOY_RESOURCE_RESOURCE_MANAGER_H_

#include "dflow/inc/data_flow/model/flow_model.h"
#include "deploy/deployer/deployer_proxy.h"
#include "dflow/base/deploy/deploy_planner.h"
#include "common/config/configurations.h"
#include "dflow/base/model/model_deploy_resource.h"

namespace ge {
class ResourceManager {
 public:
  static ResourceManager &GetInstance();

  Status Initialize();

  void Finalize();

  const DeviceInfo *GetDeviceInfo(int32_t node_id, int32_t device_id, int32_t type) const;

  const std::vector<DeviceInfo> &GetDeviceInfoList() const;

  const std::vector<DeviceInfo> &GetNpuDeviceInfoList() const;

  const DeviceInfo *GetHeadNpuDeviceInfo(int32_t node_id) const;

  Status AllocateResources(const FlowModelPtr &flow_model, DeployPlan &deploy_plan) const;

  int32_t GetLocalNodeId() const;

  static void ClearWorkingDir();
  const std::map<std::string, std::vector<DeviceInfo>> &GetDeviceIp2DevicesMap() const;
  const ModelCompileResource &GetCompileResource() const { return compile_resource_; }
 private:
  std::vector<const DeviceInfo *> GetAvailableDevices() const;

  // npu and cpu resources from resource.json
  std::vector<DeviceInfo> device_info_list_;
  // npu resources from resource.json
  std::vector<DeviceInfo> npu_device_info_list_;
  std::map<int32_t, std::map<int32_t, std::map<DeviceType, const DeviceInfo *>>> device_info_map_;
  int32_t local_node_id_ = -1;
  // mapping of device ip and device infos
  std::map<std::string, std::vector<DeviceInfo>> device_ip_2_devices_;
  ModelCompileResource compile_resource_;
};
}  // namespace ge

#endif  // AIR_RUNTIME_HETEROGENEOUS_DEPLOY_RESOURCE_RESOURCE_MANAGER_H_
