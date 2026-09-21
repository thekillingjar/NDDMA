/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_HOST_RESOURCE_MANAGER_H
#define AIR_CXX_HOST_RESOURCE_MANAGER_H

#include "graph/host_resource/host_resource.h"
#include "graph/detail/attributes_holder.h"
#include "ge/ge_api_types.h"

namespace ge {
class HostResourceManager {
 public:
  HostResourceManager() = default;
  virtual const HostResource *GetResource(const std::shared_ptr<AttrHolder> &attr_holder, int64_t type) const = 0;
  virtual Status AddResource(const std::shared_ptr<AttrHolder> &attr_holder, int64_t type,
                                  std::shared_ptr<HostResource> &host_resource) = 0;
  virtual Status TakeoverResources(const std::shared_ptr<AttrHolder> &attr_holder) = 0;
  virtual ~HostResourceManager() = default;

  HostResourceManager(const HostResourceManager &host_resource_mgr) = delete;
  HostResourceManager(const HostResourceManager &&host_resource_mgr) = delete;
  HostResourceManager &operator=(const HostResourceManager &host_resource_mgr) = delete;
  HostResourceManager &operator=(HostResourceManager &&host_resource_mgr) = delete;
};
}  // namespace ge
#endif  // AIR_CXX_HOST_RESOURCE_MANAGER_H
