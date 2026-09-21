/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "deploy/resource/resource_manager.h"
#include "deploy/deployer/deployer_proxy.h"
#include "common/debug/log.h"
#include "dflow/base/utils/process_utils.h"
#include "graph/ge_context.h"
#include "deploy/resource/resource_allocator.h"

namespace ge {
ResourceManager &ResourceManager::GetInstance() {
  static ResourceManager instance;
  return instance;
}

Status ResourceManager::Initialize() {
  device_info_list_.clear();
  npu_device_info_list_.clear();
  auto num_nodes = DeployerProxy::GetInstance().NumNodes();
  GELOGI("Get deployer number success, node num = %d.", num_nodes);
  for (int32_t node_id = 0; node_id < num_nodes; ++node_id) {
    const auto *node_info = DeployerProxy::GetInstance().GetNodeInfo(node_id);
    GE_CHECK_NOTNULL(node_info);
    GELOGI("Node[%d] has device size = %zu.", node_id, node_info->GetDeviceList().size());
    for (const auto &device_info : node_info->GetDeviceList()) {
      auto new_device = device_info;
      new_device.SetNodeId(node_id);
      device_info_list_.emplace_back(new_device);
      if (new_device.GetDeviceType() == NPU) {
        npu_device_info_list_.emplace_back(new_device);
        device_ip_2_devices_[new_device.GetDeviceIp()].emplace_back(new_device);
      }
      GELOGI("Get device info success, device info = %s", new_device.DebugString().c_str());
    }

    if (node_info->IsLocal()) {
      local_node_id_ = node_id;
      GELOGI("Get local node_id[%d] success.", node_id);
    }
  }
  GE_CHECK_GE(local_node_id_, 0);
  for (const auto &device_info : device_info_list_) {
    device_info_map_[device_info.GetNodeId()][device_info.GetDeviceId()][device_info.GetDeviceType()] = &device_info;
    if (device_info.GetDeviceType() == CPU) {
      compile_resource_.host_resource_type = device_info.GetResourceType();
      GELOGD("Record host resource type %s", compile_resource_.host_resource_type.c_str());
    } else {
      compile_resource_.logic_dev_id_to_res_type[device_info.ToIndex()] = device_info.GetResourceType();
      GELOGD("Record device logic id %s and resource type %s", device_info.ToIndex().c_str(),
             device_info.GetResourceType().c_str());
    }
  }
  return SUCCESS;
}

const std::vector<DeviceInfo> &ResourceManager::GetDeviceInfoList() const {
  return device_info_list_;
}

const std::vector<DeviceInfo> &ResourceManager::GetNpuDeviceInfoList() const {
  return npu_device_info_list_;
}

void ResourceManager::Finalize() {
  device_info_list_.clear();
  npu_device_info_list_.clear();
  device_info_map_.clear();
  compile_resource_.host_resource_type.clear();
  compile_resource_.logic_dev_id_to_res_type.clear();
  GELOGI("ResourceManager finalize");
}

const DeviceInfo *ResourceManager::GetDeviceInfo(int32_t node_id,
                                                 int32_t device_id,
                                                 int32_t type) const {
  auto node_it = device_info_map_.find(node_id);
  if (node_it == device_info_map_.cend()) {
    GELOGE(PARAM_INVALID, "Invalid node id: %d", node_id);
    return nullptr;
  }

  auto dev_it = node_it->second.find(device_id);
  if (dev_it == node_it->second.cend()) {
    GELOGE(PARAM_INVALID, "Invalid device id: %d, node_id = %d", device_id, node_id);
    return nullptr;
  }

  DeviceType device_type = static_cast<DeviceType>(type);
  auto type_it = dev_it->second.find(device_type);
  if (type_it == dev_it->second.cend()) {
    GELOGE(PARAM_INVALID, "Invalid device type: %d, device id: %d, node_id = %d",
           type, device_id, node_id);
    return nullptr;
  }
  return type_it->second;
}

const DeviceInfo *ResourceManager::GetHeadNpuDeviceInfo(int32_t node_id) const {
  auto node_it = device_info_map_.find(node_id);
  if (node_it == device_info_map_.cend()) {
    GELOGE(PARAM_INVALID, "Invalid node id: %d", node_id);
    return nullptr;
  }

  for (const auto &dev_it : node_it->second) {
    for (const auto &type_it : dev_it.second) {
      auto type = type_it.first;
      if (type == NPU) {
        return type_it.second;
      }
    }
  }
  return nullptr;
}

std::vector<const DeviceInfo *> ResourceManager::GetAvailableDevices() const {
  std::vector<const DeviceInfo *> available_devices;
  for (const auto &device_info : device_info_list_) {
    available_devices.emplace_back(&device_info);
  }
  return available_devices;
}

Status ResourceManager::AllocateResources(const FlowModelPtr &flow_model, DeployPlan &deploy_plan) const {
  auto available_devices = GetAvailableDevices();
  ResourceAllocator allocator;
  GE_CHK_STATUS_RET(allocator.AllocateResources(flow_model, compile_resource_, available_devices, deploy_plan),
                    "[FlowModel: %s] Failed to allocate resources", flow_model->GetModelName().c_str());
  GELOGI("[FlowModel: %s] allocate resources successfully", flow_model->GetModelName().c_str());
  return SUCCESS;
}

void ResourceManager::ClearWorkingDir() {
  const auto &working_dir = Configurations::GetInstance().GetDeployResDir();
  if (!working_dir.empty()) {
    std::string clear_dir_cmd = "rm -rf ";
    clear_dir_cmd += working_dir + "/*";
    GELOGI("Clear working dir: %s, execute: %s", working_dir.c_str(), clear_dir_cmd.c_str());
    (void) ProcessUtils::System(clear_dir_cmd);
  }
}

int32_t ResourceManager::GetLocalNodeId() const {
  return local_node_id_;
}

const std::map<std::string, std::vector<DeviceInfo>> &ResourceManager::GetDeviceIp2DevicesMap() const {
  return device_ip_2_devices_;
}
}  // namespace ge
