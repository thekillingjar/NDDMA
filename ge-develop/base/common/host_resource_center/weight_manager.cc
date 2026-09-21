/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "weight_manager.h"
#include "graph/op_desc.h"
#include "graph/utils/constant_utils.h"
#include "common/checker.h"
namespace ge {
const HostResource *WeightManager::GetResource(const std::shared_ptr<AttrHolder> &attr_holder, int64_t type) const {
  (void)type;
  const auto &weight_iter = op_desc_2_weights_.find(attr_holder);
  if (weight_iter == op_desc_2_weights_.cend()) {
    return nullptr;
  }
  return &(weight_iter->second);
}

Status WeightManager::AddResource(const std::shared_ptr<AttrHolder> &attr_holder, int64_t type,
                                       std::shared_ptr<HostResource> &host_resource) {
  (void)attr_holder;
  (void)type;
  (void)host_resource;
  return FAILED;
}

Status WeightManager::TakeoverResources(const std::shared_ptr<AttrHolder> &attr_holder) {
  auto op_desc = std::dynamic_pointer_cast<OpDesc>(attr_holder);
  GE_ASSERT_NOTNULL(op_desc);
  if (!ConstantUtils::IsRealConst(op_desc)) {
    return SUCCESS;
  }
  ConstGeTensorPtr weight;
  GE_ASSERT_TRUE(ConstantUtils::GetWeight(op_desc, 0, weight));
  GE_ASSERT_NOTNULL(weight);
  WeightResource weight_resource(weight);
  GE_ASSERT_TRUE(op_desc_2_weights_.emplace(attr_holder, std::move(weight_resource)).second,
                 "Already has resource key is %ld, failed to takeover weigh on %s. Check node with same id in graph.",
                 op_desc->GetId(), op_desc->GetNamePtr());
  GELOGD("Take over weight resource on node %s.", op_desc->GetNamePtr());
  return SUCCESS;
}
} // namespace ge