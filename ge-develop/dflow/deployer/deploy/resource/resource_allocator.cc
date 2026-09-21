/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "deploy/resource/resource_allocator.h"
#include "deploy/resource/resource_manager.h"

namespace ge {
Status ResourceAllocator::AllocateResources(const FlowModelPtr &flow_model,
                                            const ModelCompileResource &compile_resource,
                                            const std::vector<const DeviceInfo *> &device_list,
                                            DeployPlan &deploy_plan) {
  SetCheckCompileResource(flow_model);
  // cpu and npu resource info
  std::vector<DeployPlan::DeviceInfo> device_info_list;
  std::set<std::string> device_index_set;
  for (const auto device_info : device_list) {
    bool filter_flag = true;
    GE_CHK_STATUS_RET_NOLOG(CheckAndFilterByCompileResource(device_info, filter_flag, compile_resource));
    if (filter_flag) {
      device_info_list.emplace_back(DeployPlan::DeviceInfo(device_info->GetDeviceType(),
                                                           device_info->GetNodeId(),
                                                           device_info->GetDeviceId()));
      device_index_set.insert(device_info->ToIndex());
    }
  }
  GE_CHK_STATUS_RET_NOLOG(CheckAfterFilterDevice(device_index_set));

  GE_CHK_STATUS_RET(UpdateDeviceInfo(device_info_list), "Failed to update device info list.");
  GE_CHK_STATUS_RET(HeterogeneousDeployPlanner(flow_model, device_info_list).BuildPlan(deploy_plan),
                    "[FlowModel: %s] Failed to build DeployPlan.", flow_model->GetModelName().c_str());
  GELOGD("[FlowModel: %s] deploy plan built successfully", flow_model->GetModelName().c_str());
  return SUCCESS;
}

Status ResourceAllocator::UpdateDeviceInfo(std::vector<DeployPlan::DeviceInfo> &device_info_list) {
  for (auto &plan_device_info : device_info_list) {
    auto device_info = ResourceManager::GetInstance().GetDeviceInfo(plan_device_info.GetNodeId(),
                                                                    plan_device_info.GetDeviceId(),
                                                                    plan_device_info.GetType());
    GE_CHECK_NOTNULL(device_info);
    plan_device_info.SetOsId(device_info->GetOsId());
    plan_device_info.SetHcomDeviceId(device_info->GetHcomDeviceId());
  }
  return SUCCESS;
}

void ResourceAllocator::SetCheckCompileResource(const FlowModelPtr &flow_model) {
  compile_resource_ = flow_model->GetCompileResource();
  if (compile_resource_ == nullptr) {
    GELOGI("compile resource is nullptr");
  }
  if (compile_resource_ == nullptr || ((compile_resource_ != nullptr) &&
      compile_resource_->host_resource_type.empty() && compile_resource_->logic_dev_id_to_res_type.empty())) {
    GELOGI("Need't to check compile resource info");
    return;
  }
  std::map<std::string, std::string> valid_dev_to_res_type;
  for (const auto &dev_id_to_res_type : compile_resource_->logic_dev_id_to_res_type) {
    string valid_device_id = dev_id_to_res_type.first;
    HeterogeneousDeployPlanner::GetValidLogicDeviceId(valid_device_id);
    valid_dev_to_res_type[valid_device_id] = dev_id_to_res_type.second;
  }
  compile_resource_->logic_dev_id_to_res_type = std::move(valid_dev_to_res_type);
  is_need_to_check_ = true;
  GELOGI("It is necessary to check compile resource info");
}

Status ResourceAllocator::CheckAndFilterByCompileResource(const DeviceInfo *device_info, bool &filter_flag,
                                                          const ModelCompileResource &curr_res_info) const {
  filter_flag = true;
  if (!is_need_to_check_) {
    return SUCCESS;
  }
  GE_CHECK_NOTNULL(compile_resource_);
  GE_CHECK_NOTNULL(device_info);
  if (device_info->GetDeviceType() == DeviceType::CPU) {
    if (curr_res_info.host_resource_type != compile_resource_->host_resource_type) {
      GELOGE(FAILED, "Current host resource %s is not same as compile stage host resource %s",
             curr_res_info.host_resource_type.c_str(), compile_resource_->host_resource_type.c_str());
      return FAILED;
    }
    GELOGI("Host resource type %s is same in compile stage and deploy stage.",
           curr_res_info.host_resource_type.c_str());
  }
  if (device_info->GetDeviceType() == DeviceType::NPU) {
    const std::string device_index = device_info->ToIndex();
    const auto iter_deploy = curr_res_info.logic_dev_id_to_res_type.find(device_index);
    if (iter_deploy == curr_res_info.logic_dev_id_to_res_type.cend()) {
      GELOGE(FAILED, "Can not find device index %s in resource manager", device_index.c_str());
      return FAILED;
    }
    const auto iter_compile = compile_resource_->logic_dev_id_to_res_type.find(device_index);
    if (iter_compile == compile_resource_->logic_dev_id_to_res_type.cend()) {
      GELOGI("Skip to add device %s to available device list result of can not find in compile resource.",
             device_index.c_str());
      filter_flag = false;
      return SUCCESS;
    }
    if (iter_deploy->second != iter_compile->second) {
      GELOGE(FAILED, "Current device resource %s is not same as compile stage device resource %s for logic device %s",
             iter_deploy->second.c_str(), iter_compile->second.c_str(), device_index.c_str());
      return FAILED;
    }
    GELOGI("Device %s, type:%s is same in both compile resource and current resource config.",
           device_index.c_str(), iter_deploy->second.c_str());
  }
  return SUCCESS;
}

Status ResourceAllocator::CheckAfterFilterDevice(const std::set<std::string> &device_index) const {
  if (!is_need_to_check_) {
    return SUCCESS;
  }
  GE_CHECK_NOTNULL(compile_resource_);
  for (const auto &logic_dev_id : compile_resource_->logic_dev_id_to_res_type) {
    if (device_index.count(logic_dev_id.first) == 0) {
      GELOGE(FAILED, "Can not fount logic device id %s from compile resource in curent env.",
             logic_dev_id.first.c_str());
      return FAILED;
    }
  }
  return SUCCESS;
}
}  // namespace ge
