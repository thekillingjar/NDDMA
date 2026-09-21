/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "generate_cmo_type_base.h"
#include <string>
#include "common/fe_log.h"
#include "common/configuration.h"
#include "common/fe_op_info_common.h"
#include "common/math_util.h"
#include "common/aicore_util_types.h"
#include "common/aicore_util_attr_define.h"
#include "common/graph/fe_graph_utils.h"
#include "platform/platform_info.h"
#include "graph/utils/tensor_utils.h"
#include "graph/utils/op_desc_utils.h"

namespace fe {
const std::string kAssociatedAttr = "AssociatedTask_";
const uint64_t kDefaultL2CacheSize = 8388608; // 8*1024*1024

GenerateCMOTypeBase::GenerateCMOTypeBase() {
}

void GenerateCMOTypeBase::AddToNodeCmoAttr(const ge::OpDescPtr &op_desc,
                                           const std::string &cmo_type,
                                           const std::vector<CmoAttr> &attr_vec) const {
  map<std::string, std::vector<CmoAttr>> cmo{};
  cmo = op_desc->TryGetExtAttr(kOpExtattrNameCmo, cmo);
  const auto iter = cmo.find(cmo_type);
  if (iter == cmo.end()) {
    cmo.emplace(std::make_pair(cmo_type, attr_vec));
  } else {
    iter->second.insert(iter->second.cend(), attr_vec.cbegin(), attr_vec.cend());
  }
  (void)op_desc->SetExtAttr(kOpExtattrNameCmo, cmo);

  // for GE cut stream
  int32_t associate_task_count = 0;
  associate_task_count = op_desc->TryGetExtAttr(kAssociatedAttr, associate_task_count);
  associate_task_count += static_cast<int32_t>(attr_vec.size());
  if (associate_task_count < 0) {
    associate_task_count = INT32_MAX;
  }
  (void)op_desc->SetExtAttr(kAssociatedAttr, associate_task_count);
}

bool GenerateCMOTypeBase::CheckParentOpIsAiCore(const ge::InDataAnchorPtr &in_anchor) const {
  auto peer_out_anchor = in_anchor->GetPeerOutAnchor();
  if (peer_out_anchor == nullptr) {
    return false;
  }

  auto peer_out_node = peer_out_anchor->GetOwnerNode();
  if (peer_out_node == nullptr || peer_out_node->GetOpDesc() == nullptr) {
    return false;
  }

  return IsAiCoreOp(peer_out_node);
}

bool GenerateCMOTypeBase::ReadIsLifeCycleEnd(const ge::NodePtr &node, const ge::InDataAnchorPtr &in_anchor) const {
  auto idx        = in_anchor->GetIdx();
  auto op_desc    = node->GetOpDesc();
  auto input_desc = op_desc->MutableInputDesc(static_cast<uint32_t>(idx));
  if (input_desc == nullptr) {
    return false;
  }
  return IsLifeCycleEnd(*node, input_desc, idx);
}

bool GenerateCMOTypeBase::IsPeerOutNoTaskNode(const ge::InDataAnchorPtr &in_anchor) const {
  auto peer_out_anchor = in_anchor->GetPeerOutAnchor();
  if (peer_out_anchor == nullptr) {
    return false;
  }

  auto peer_out_node = peer_out_anchor->GetOwnerNode();
  if (peer_out_node == nullptr || peer_out_node->GetOpDesc() == nullptr) {
    return false;
  }

  if (IsNoTaskOp(peer_out_node)) {
    return true;
  }

  for (auto &tmp_node : peer_out_node->GetOutAllNodes()) {
    if (IsNoTaskOp(tmp_node)) {
      return true;
    }
  }
  return false;
}

void GenerateCMOTypeBase::CalcTotalTensorSize(const ge::GeTensorDescPtr &tensor_desc,
                                              int64_t &total_tensor_size) const {
  int64_t tensor_size = 0;
  if (ge::TensorUtils::GetSize(*tensor_desc, tensor_size) != ge::GRAPH_SUCCESS) {
    return;
  }
  total_tensor_size += tensor_size;
  if (total_tensor_size < 0) {
    total_tensor_size = INT64_MAX;
  }
  return;
}

int64_t GenerateCMOTypeBase::GetInputTensorSize(const ge::OpDescPtr &op_desc) const {
  const size_t inputs_size = op_desc->GetAllInputsSize();
  int64_t total_tensor_size = 0;
  for (size_t i = 0; i < inputs_size; i++) {
    ge::GeTensorDescPtr tensor_desc = op_desc->MutableInputDesc(static_cast<uint32_t>(i));
    if (tensor_desc == nullptr) {
      FE_LOGW("paramater tensor_desc should not be nullptr at index(%zu)", i);
      continue;
    }
    CalcTotalTensorSize(tensor_desc, total_tensor_size);
  }
  return total_tensor_size;
}

int64_t GenerateCMOTypeBase::GetOutputTensorSize(const ge::OpDescPtr &op_desc) const {
  const size_t outputs_size = op_desc->GetOutputsSize();
  int64_t total_tensor_size = 0;
  for (size_t i = 0; i < outputs_size; i++) {
    ge::GeTensorDescPtr tensor_desc = op_desc->MutableOutputDesc(static_cast<uint32_t>(i));
    if (tensor_desc == nullptr) {
      FE_LOGW("paramater tensor_desc should not be nullptr at index(%zu)", i);
      continue;
    }
    CalcTotalTensorSize(tensor_desc, total_tensor_size);
  }
  return total_tensor_size;
}

int64_t GenerateCMOTypeBase::GetWorkSpaceSize(const ge::OpDescPtr &op_desc) const {
  int64_t total_workspace_size = 0;
  const std::vector<int64_t> v_workspace_size = op_desc->GetWorkspaceBytes();
  for (const auto workspace_bytes : v_workspace_size) {
    total_workspace_size += workspace_bytes;
  }
  return total_workspace_size;
}

bool GenerateCMOTypeBase::CheckAndGetWeightSize(const ge::GeTensorDescPtr &weight, int64_t &tensor_size) const {
  if (weight != nullptr && ge::TensorUtils::GetSize(*weight, tensor_size) != ge::GRAPH_SUCCESS) {
    return false;
  }
  return (kPrefetchMin <= tensor_size && tensor_size < kPrefetchMax);
}

int64_t GenerateCMOTypeBase::GetWeightSize(const ge::NodePtr &node) const {
  int64_t total_weight_size = 0;
  for (const auto &in_data_anchor : node->GetAllInDataAnchors()) {
    ge::NodePtr tmp_node = nullptr;
    if (FeGraphUtils::IsPeerOutWeight(node.get(), in_data_anchor->GetIdx(), tmp_node)) {
      int64_t tensor_size = 0;
      auto weight = node->GetOpDesc()->MutableInputDesc(in_data_anchor->GetIdx());
      if (CheckAndGetWeightSize(weight, tensor_size)) {
        FE_ADD_OVERFLOW(total_weight_size, tensor_size, total_weight_size);
      }
    }
  }
  return total_weight_size;
}

uint64_t GenerateCMOTypeBase::GetCacheSize() const {
  static uint64_t cache_size = 0;
  if (cache_size == 0) {
    PlatformInfo platform_info;
    OptionalInfo opti_info;
    if (PlatformInfoManager::Instance().GetPlatformInfoWithOutSocVersion(platform_info, opti_info) != SUCCESS) {
      FE_LOGW("Cannot get l2 cache size from platform info, using default value:%lu", kDefaultL2CacheSize);
      cache_size = kDefaultL2CacheSize;
    } else {
      cache_size = platform_info.soc_info.l2_size;
    }
  }
  return cache_size;
}
} // namespace fe
