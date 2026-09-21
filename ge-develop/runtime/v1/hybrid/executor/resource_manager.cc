/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "hybrid/executor/resource_manager.h"
#include "hybrid/node_executor/ge_local/data_flow_kernels.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/type_utils.h"
#include "graph/debug/ge_attr_define.h"
#include "framework/common/types.h"

namespace ge {
namespace hybrid {
namespace {
const std::unordered_set<std::string> kDataFlowSources = {
    STACK
};
const std::unordered_set<std::string> kDataFlowOperations = {
    STACKPUSH,
    STACKPOP,
    STACKCLOSE
};
inline bool IsDataFlowSource(const std::string &op_type) {
  return kDataFlowSources.count(op_type) != 0UL;
}
inline bool IsDataFlowOperations(const std::string &op_type) {
  return kDataFlowOperations.count(op_type) != 0UL;
}
inline bool IsDataFlowOps(const std::string &op_type) {
  return (IsDataFlowSource(op_type)) || (IsDataFlowOperations(op_type));
}
}
Status ResourceManager::Init(const GraphItem *const graph_item) {
  GE_CHECK_NOTNULL(graph_item);
  graph_item_ = graph_item;
  GE_CHK_STATUS_RET(InitDataFlowResource(), "Failed to init data flow resource, graph:%s.",
                    graph_item_->GetName().c_str());
  return SUCCESS;
}

DataFlowResourcePtr ResourceManager::GetDataFlowResource(const int64_t handle) const {
  const std::lock_guard<std::mutex> lock(mutex_);
  const auto iter = data_flow_resources_.find(handle);
  if (iter == data_flow_resources_.end()) {
    return nullptr;
  }
  return iter->second;
}

DataFlowKernelBasePtr ResourceManager::GetDataFlowKernel(const std::string &type) const {
  const std::lock_guard<std::mutex> lock(mutex_);
  const auto iter = data_flow_kernels_.find(type);
  if (iter == data_flow_kernels_.end()) {
    return nullptr;
  }
  return iter->second;
}

void ResourceManager::ClearDataFlowResources() {
  const std::lock_guard<std::mutex> lock(mutex_);
  for (const auto &res : data_flow_resources_) {
    res.second->Clear();
  }
}

Status ResourceManager::InitDataFlowResource() {
  for (const auto node_item : graph_item_->GetAllNodes()) {
    if (!IsDataFlowOps(node_item->NodeType())) {
      continue;
    }
    int64_t data_flow_handle;
    if (!(AttrUtils::GetInt(node_item->GetOpDesc(), ATTR_NAME_DATA_FLOW_HANDLE, data_flow_handle))) {
      REPORT_INNER_ERR_MSG("E19999", "Failed to get handle for [%s(%s)].",
                        node_item->NodeName().c_str(), node_item->NodeType().c_str());
      GELOGE(INTERNAL_ERROR, "[Get][Attr] Failed for [%s(%s)].",
             node_item->NodeName().c_str(), node_item->NodeType().c_str());
      return INTERNAL_ERROR;
    }
    if (data_flow_kernels_.count(node_item->NodeType()) == 0UL) {
      const auto kernel = DataFlowKernelFactory::GetInstance().CreateKernel(node_item->NodeType());
      if (kernel == nullptr) {
        REPORT_INNER_ERR_MSG("E19999", "Create data flow kernel failed for [%s(%s)], handle:%" PRId64 ".",
                          node_item->NodeName().c_str(), node_item->NodeType().c_str(), data_flow_handle);
        GELOGE(INTERNAL_ERROR, "[Create][Kernel] failed for [%s(%s)], handle:%ld.",
               node_item->NodeName().c_str(), node_item->NodeType().c_str(), data_flow_handle);
        return INTERNAL_ERROR;
      }
      data_flow_kernels_[node_item->NodeType()] = kernel;
    }
    if (data_flow_resources_.count(data_flow_handle) == 0UL) {
      const auto res = MakeShared<DataFlowResource>();
      if (res == nullptr) {
        REPORT_INNER_ERR_MSG("E19999", "Create res failed for [%s(%s)], handle:%" PRId64 ".",
                          node_item->NodeName().c_str(), node_item->NodeType().c_str(), data_flow_handle);
        GELOGE(INTERNAL_ERROR, "[Create][Res] failed for [%s(%s)], handle:%ld.",
               node_item->NodeName().c_str(), node_item->NodeType().c_str(), data_flow_handle);
        return INTERNAL_ERROR;
      }
      data_flow_resources_[data_flow_handle] = res;
    }
    if (IsDataFlowSource(node_item->NodeType())) {
      int64_t max_size = -1;
      if (AttrUtils::GetInt(node_item->GetOpDesc(), ATTR_NAME_DATA_FLOW_MAX_SIZE, max_size)) {
        max_size = (max_size > 0) ? max_size : std::numeric_limits<int64_t>::max();
        data_flow_resources_[data_flow_handle]->SetMaxSize(max_size);
        data_flow_resources_[data_flow_handle]->SetMaxSizeConst(true);
        GELOGD("Init data flow source max size[%ld], handle[%ld]", max_size, data_flow_handle);
      }
    }
    GELOGD("Init data flow for [%s(%s)], handle:%ld.",
           node_item->NodeName().c_str(), node_item->NodeType().c_str(), data_flow_handle);
  }
  return SUCCESS;
}
}  // namespace hybrid
}  // namespace ge
