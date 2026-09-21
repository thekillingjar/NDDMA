/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_RUNTIME_HETEROGENEOUS_DEPLOY_RESOURCE_RESOURCE_ALLOCATOR_H_
#define AIR_RUNTIME_HETEROGENEOUS_DEPLOY_RESOURCE_RESOURCE_ALLOCATOR_H_
#include <set>
#include "deploy/resource/heterogeneous_deploy_planner.h"
#include "deploy/resource/device_info.h"
#include "dflow/base/model/model_deploy_resource.h"

namespace ge {
class ResourceAllocator {
 public:
  Status AllocateResources(const FlowModelPtr &flow_model,
                           const ModelCompileResource &compile_resource,
                           const std::vector<const DeviceInfo *> &device_list,
                           DeployPlan &deploy_plan);
 private:
  void SetCheckCompileResource(const FlowModelPtr &flow_model);
  Status CheckAndFilterByCompileResource(const DeviceInfo *device_info, bool &filter_flag,
                                         const ModelCompileResource &curr_res_info) const;
  Status CheckAfterFilterDevice(const std::set<std::string> &device_index) const;
  static Status UpdateDeviceInfo(std::vector<DeployPlan::DeviceInfo> &device_info_list);
  bool is_need_to_check_ = false;
  std::shared_ptr<ModelCompileResource> compile_resource_;
};
}  // namespace ge

#endif  // AIR_RUNTIME_HETEROGENEOUS_DEPLOY_RESOURCE_RESOURCE_ALLOCATOR_H_
