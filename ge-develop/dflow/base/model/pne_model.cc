/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "dflow/inc/data_flow/model/pne_model.h"
#include "graph/debug/ge_attr_define.h"
#include "graph_metadef/common/ge_common/util.h"

namespace ge {
Status PneModel::SetLogicDeviceId(const std::string &logic_device_id) {
  const std::lock_guard<std::mutex> lk(pne_model_mutex_);
  if (!submodels_.empty()) {
    for (const auto &submodel : submodels_) {
      (void)submodel.second->SetLogicDeviceId(logic_device_id);
    }
    return SUCCESS;
  }
  GE_CHECK_NOTNULL(root_graph_);
  GE_CHK_BOOL_RET_STATUS(AttrUtils::SetStr(root_graph_, ATTR_NAME_DEPLOY_DEVICE_ID, logic_device_id), FAILED,
                         "Failed to set attribute: %s", ATTR_NAME_DEPLOY_DEVICE_ID.c_str());
  GELOGI("Model[%s] set logical device = %s.", GetModelName().c_str(), logic_device_id.c_str());
  return SUCCESS;
}

std::string PneModel::GetLogicDeviceId() const {
  std::string logical_device_id;
  const std::lock_guard<std::mutex> lk(pne_model_mutex_);
  if (!submodels_.empty()) {
    return logical_device_id;
  }
  if (root_graph_ != nullptr) {
    if (AttrUtils::GetStr(root_graph_, ATTR_NAME_DEPLOY_DEVICE_ID, logical_device_id)) {
      GELOGI("Model[%s] has logical device = %s.", root_graph_->GetName().c_str(), logical_device_id.c_str());
    }
  }
  return logical_device_id;
}

Status PneModel::SetRedundantLogicDeviceId(const std::string &logic_device_id) {
  const std::lock_guard<std::mutex> lk(pne_model_mutex_);
  if (!submodels_.empty()) {
    for (const auto &submodel : submodels_) {
      (void)submodel.second->SetRedundantLogicDeviceId(logic_device_id);
    }
    return SUCCESS;
  }

  GE_CHECK_NOTNULL(root_graph_);
  GE_CHK_BOOL_RET_STATUS(AttrUtils::SetStr(root_graph_, ATTR_NAME_REDUNDANT_DEPLOY_DEVICE_ID, logic_device_id), FAILED,
                         "Failed to set attribute: %s", ATTR_NAME_REDUNDANT_DEPLOY_DEVICE_ID.c_str());
  GELOGI("Model[%s] set redundant logical device = %s.", GetModelName().c_str(), logic_device_id.c_str());
  return SUCCESS;
}

std::string PneModel::GetRedundantLogicDeviceId() const {
  std::string logical_device_id;
  const std::lock_guard<std::mutex> lk(pne_model_mutex_);
  if (!submodels_.empty()) {
    return logical_device_id;
  }
  if (root_graph_ != nullptr) {
    if (AttrUtils::GetStr(root_graph_, ATTR_NAME_REDUNDANT_DEPLOY_DEVICE_ID, logical_device_id)) {
      GELOGI("Model[%s] has redundant logical device = %s.", root_graph_->GetName().c_str(), logical_device_id.c_str());
    }
  }
  return logical_device_id;
}
}  // namespace ge