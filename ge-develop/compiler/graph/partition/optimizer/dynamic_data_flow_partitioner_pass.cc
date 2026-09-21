/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph/partition/optimizer/dynamic_data_flow_partitioner_pass.h"
#include "framework/common/util.h"
#include "framework/common/types.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/type_utils.h"

namespace ge {
namespace {
  const std::unordered_set<std::string> kDataFlowPartitionerDataFlowOps = {STACK, STACKPUSH, STACKPOP, STACKCLOSE};
  inline bool IsDataFlowOps(const std::string &op_type) {
    return (kDataFlowPartitionerDataFlowOps.count(op_type) != 0UL);
  }
} // namespace

Status DynamicDataFlowPartitionerPass::Run(const ge::ComputeGraphPtr &graph,
                                           const std::vector<std::shared_ptr<BaseCluster>> &sorted_unique_clusters,
                                           bool &result) {
  /* If the handle and the corresponding resource  ops container are empty, execute */
  if (data_flow_ops_.empty()) {
    GE_CHK_STATUS_RET(GetDataFlowOpMapping(graph), "[GetDataFlowOpMapping] failed");
  }
  GE_CHK_STATUS_RET(GetDataFlowOpAttr(sorted_unique_clusters), "[GetDataFlowOpAttr] failed");
  GE_CHK_STATUS_RET(GetDataFlowOpNeedProcess(result), "[GetDataFlowOpNeedProcess] failed");
  if (result) {
    GE_CHK_STATUS_RET(GetUnknownDataFlowOp(), "[GetUnknownDataFlowOp] failed");
    GE_CHK_STATUS_RET(MarkDataFlowOpAttr(graph), "[MarkDataFlowOpAttr] failed");
  } else {
    data_flow_ops_.clear();
  }
  ClearDataFlowsource();
  return SUCCESS;
}

Status DynamicDataFlowPartitionerPass::GetDataFlowOpMapping(const ge::ComputeGraphPtr &graph) {
  for (const auto &node : graph->GetAllNodes()) {
    GE_CHECK_NOTNULL(node);
    int64_t handle;
    if (AttrUtils::GetInt(node->GetOpDesc(), ATTR_NAME_DATA_FLOW_HANDLE, handle)) {
      (void)data_flow_ops_[handle].insert(node);
      GELOGD("Get data flow node %s, handle %ld is success.", node->GetName().c_str(), handle);
    }
  }
  return SUCCESS;
}

Status DynamicDataFlowPartitionerPass::GetDataFlowOpAttr(
    const std::vector<std::shared_ptr<BaseCluster>> &sorted_unique_clusters) {
  for (const auto &cluster : sorted_unique_clusters) {
    for (const auto &node : cluster->Nodes()) {
      const auto op_type = node->GetType();
      if (IsDataFlowOps(op_type)) {
        const auto cast_cluster = std::dynamic_pointer_cast<DynamicShapeCluster>(cluster);
        GE_CHK_BOOL_RET_STATUS(cast_cluster != nullptr, PARAM_INVALID,
                               "[Cast][Cluster] to DynamicShapeCluster failed.");
        if (cast_cluster->IsUnknownShape()) {
          data_flow_ops_attr_[node] = true;
        } else {
          data_flow_ops_attr_[node] = false;
        }
      }
    }
  }
  return SUCCESS;
}

Status DynamicDataFlowPartitionerPass::GetDataFlowOpNeedProcess(bool &result) {
  bool is_need_process = false;
  for (const auto &resop : data_flow_ops_) {
    if (resop.second.size() <= 1) {
      continue;
    }
    const auto iter = data_flow_ops_attr_.find(*(resop.second.begin()));
    if (iter == data_flow_ops_attr_.end()) {
      continue;
    }
    bool srcop_attr = iter->second;
    for (const auto &dstop : resop.second) {
      const auto dstop_iter = data_flow_ops_attr_.find(dstop);
      if (dstop_iter == data_flow_ops_attr_.end()) {
        continue;
      }
      bool dstop_attr = dstop_iter->second;
      if (srcop_attr != dstop_attr) {
        is_need_process = true;
        data_flow_handle_.insert(resop.first);
      }
    }
  }
  result = is_need_process;
  return SUCCESS;
}

Status DynamicDataFlowPartitionerPass::GetUnknownDataFlowOp() {
  for (const auto &op_handle : data_flow_handle_) {
    if (data_flow_ops_.find(op_handle) != data_flow_ops_.end()) {
      for (const auto &dst_op : data_flow_ops_.find(op_handle)->second) {
        unknown_data_flow_ops_.insert(dst_op);
        GELOGD("Unknown data flow handle %ld, node %s.", op_handle, dst_op->GetName().c_str());
      }
    }
  }
  return SUCCESS;
}

Status DynamicDataFlowPartitionerPass::MarkDataFlowOpAttr(const ge::ComputeGraphPtr &graph) const {
  for (auto &node : graph->GetAllNodes()) {
    GE_CHECK_NOTNULL(node);
    const auto op_type = node->GetType();
    if (IsDataFlowOps(op_type)) {
      if (unknown_data_flow_ops_.find(node) != unknown_data_flow_ops_.end()) {
        auto desc = node->GetOpDesc();
        GE_CHECK_NOTNULL(desc);
        (void)AttrUtils::SetBool(desc, "_force_unknown_shape", true);
      }
    }
  }

  return SUCCESS;
}

void DynamicDataFlowPartitionerPass::ClearDataFlowsource() {
  data_flow_handle_.clear();
  unknown_data_flow_ops_.clear();
  data_flow_ops_attr_.clear();
}
} // namespace ge
