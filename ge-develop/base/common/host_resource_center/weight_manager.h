/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_WEIGHT_MANAGER_H
#define AIR_CXX_WEIGHT_MANAGER_H

#include "framework/common/host_resource_center/host_resource_manager.h"
#include "graph/ge_attr_value.h"

namespace ge {
class WeightResource : public HostResource {
 public:
  explicit WeightResource(const ConstGeTensorPtr &weight) : HostResource(), weight_(weight) {}
  const GeTensor *GetWeight() const {
    return weight_.get();
  }

 private:
  ConstGeTensorPtr weight_;
};

class WeightManager : public HostResourceManager {
 public:
  WeightManager() = default;
  const HostResource *GetResource(const std::shared_ptr<AttrHolder> &attr_holder, int64_t type) const override;
  Status AddResource(const std::shared_ptr<AttrHolder> &attr_holder, int64_t type,
                          std::shared_ptr<HostResource> &host_resource) override;
  Status TakeoverResources(const std::shared_ptr<AttrHolder> &attr_holder) override;

  ~WeightManager() override = default;

  WeightManager(const WeightManager &host_resource_mgr) = delete;
  WeightManager(const WeightManager &&host_resource_mgr) = delete;
  WeightManager &operator=(const WeightManager &host_resource_mgr) = delete;
  WeightManager &operator=(WeightManager &&host_resource_mgr) = delete;

 private:
  std::unordered_map<std::shared_ptr<AttrHolder>, WeightResource> op_desc_2_weights_;
};
} // namespace ge
#endif  // AIR_CXX_WEIGHT_MANAGER_H
