/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "framework/common/host_resource_center/tiling_resource_manager.h"
#include "common/checker.h"
#include "register/op_tiling_info.h"
#include "graph/debug/ge_attr_define.h"
#include "common/util/mem_utils.h"
#include "graph/compute_graph.h"

namespace {
constexpr uint64_t kBitFlagType = 0xFFFFFFFFFFFFFFUL;
constexpr uint64_t kBit8Shift = 56UL;
constexpr uint8_t kBit8Mask = 0xFFU;
const std::string kOpTilingId = "_op_tiling_ids";
}  // namespace
namespace ge {
const HostResource *TilingResourceManager::GetResource(const std::shared_ptr<AttrHolder> &attr_holder,
                                                       int64_t type) const {
  auto op_desc = std::dynamic_pointer_cast<OpDesc>(attr_holder);
  GE_ASSERT_NOTNULL(op_desc, "The type of the attr holder  does not match the tiling resource.");

  if (use_op_id_as_key_) {
    // Compatible with old models, use op_id as key.
    const TilingId id = GenTilingId(*op_desc, static_cast<TilingResourceType>(type));
    const auto iter = key_to_resources_.find(id);
    if (iter != key_to_resources_.end()) {
      return iter->second.get();
    }
    return nullptr;
  }

  std::vector<int64_t> tiling_ids;
  (void)AttrUtils::GetListInt(op_desc, kOpTilingId, tiling_ids);
  for (const int64_t tiling_id : tiling_ids) {
    const auto iter = key_to_resources_.find(tiling_id);
    const int64_t type_flag = static_cast<int64_t>((static_cast<uint64_t>(tiling_id) >> kBit8Shift) & kBit8Mask);
    if ((iter != key_to_resources_.end()) && (type_flag == type)) {
      return iter->second.get();
    }
  }
  GELOGW("Tiling result for [%s] with type [%ld] is not found.", op_desc->GetNamePtr(), type);
  return nullptr;
}

Status TilingResourceManager::AddResource(const std::shared_ptr<AttrHolder> &attr_holder, int64_t type,
                                          std::shared_ptr<HostResource> &host_resource) {
  auto op_desc = std::dynamic_pointer_cast<OpDesc>(attr_holder);
  GE_ASSERT_NOTNULL(op_desc, "The type of the attr holder  does not match the tiling resource.");
  auto tiling_resource = std::dynamic_pointer_cast<TilingResource>(host_resource);
  GE_ASSERT_NOTNULL(tiling_resource, "Resource type is invalid.");
  const TilingId id = GenTilingId(static_cast<TilingResourceType>(type));
  key_to_resources_[id] = host_resource;
  resource_to_keys_[tiling_resource->GetOpRunInfo().get()].push_back(id);
  std::vector<int64_t> tiling_ids;
  (void)AttrUtils::GetListInt(op_desc, kOpTilingId, tiling_ids);
  tiling_ids.push_back(id);
  GE_ASSERT_TRUE(AttrUtils::SetListInt(op_desc, kOpTilingId, tiling_ids), "Failed to set tiling ids attr to node [%s].",
                 op_desc->GetNamePtr());
  GELOGD("Take over tiling resource on node [%s] type [%ld].", op_desc->GetNamePtr(), type);
  return SUCCESS;
}

Status TilingResourceManager::TakeoverResources(const OpDescPtr &op_desc) {
  std::shared_ptr<optiling::utils::OpRunInfo> tmp_op_run_info;
  std::shared_ptr<optiling::utils::OpRunInfo> tmp_atomic_op_run_info;
  tmp_op_run_info = op_desc->TryGetExtAttr(ATTR_NAME_OP_RUN_INFO, tmp_op_run_info);
  tmp_atomic_op_run_info = op_desc->TryGetExtAttr(ATTR_NAME_ATOMIC_OP_RUN_INFO, tmp_atomic_op_run_info);

  if (tmp_op_run_info != nullptr) {
    std::shared_ptr<HostResource> host_resource = MakeShared<TilingResource>(tmp_op_run_info);
    GE_ASSERT_NOTNULL(host_resource);
    GE_ASSERT_SUCCESS(AddResource(op_desc, static_cast<int64_t>(TilingResourceType::kAiCore), host_resource));
    GELOGI("find %s has aicore tiling info", op_desc->GetNamePtr());
  }

  if (tmp_atomic_op_run_info != nullptr) {
    std::shared_ptr<HostResource> host_resource = MakeShared<TilingResource>(tmp_atomic_op_run_info);
    GE_ASSERT_NOTNULL(host_resource);
    GE_ASSERT_SUCCESS(AddResource(op_desc, static_cast<int64_t>(TilingResourceType::kAtomic), host_resource));
    GELOGI("find %s has atomic tiling info", op_desc->GetNamePtr());
  }
  return SUCCESS;
}

Status TilingResourceManager::TakeoverResources(const std::shared_ptr<AttrHolder> &attr_holder) {
  auto op_desc = std::dynamic_pointer_cast<OpDesc>(attr_holder);
  GE_ASSERT_NOTNULL(op_desc, "The type of the attr holder  does not match the tiling resource.");
  if (op_desc->GetType() == "SuperKernel") {
    GELOGI("find sk node %s", op_desc->GetNamePtr());
    ComputeGraphPtr sub_graph = nullptr;
    sub_graph = op_desc->TryGetExtAttr("_sk_sub_graph", sub_graph);
    if (sub_graph != nullptr) {
      GELOGI("takeover resource in sk %s sub_graph %s", op_desc->GetNamePtr(), sub_graph->GetName().c_str());
      for (auto &sub_node : sub_graph->GetDirectNode()) {
        GE_ASSERT_NOTNULL(sub_node);
        auto sub_op_desc = sub_node->GetOpDesc();
        GE_ASSERT_NOTNULL(sub_op_desc);
        GE_ASSERT_SUCCESS(TakeoverResources(sub_op_desc));
      }
      GE_ASSERT_TRUE(AttrUtils::SetGraph(op_desc, "_sk_sub_graph", sub_graph));
    }
  } else {
    GE_ASSERT_SUCCESS(TakeoverResources(op_desc));
  }
  return SUCCESS;
}

TilingId TilingResourceManager::GenTilingId(const OpDesc &op_desc, const TilingResourceType tiling_type) {
  size_t res = static_cast<size_t>(op_desc.GetId());
  res &= kBitFlagType;
  res |= (static_cast<uint64_t>(tiling_type) << kBit8Shift);
  return static_cast<TilingId>(res);
}

TilingId TilingResourceManager::GenTilingId(const TilingResourceType tiling_type) const {
  size_t res = key_to_resources_.size();
  res &= kBitFlagType;
  res |= (static_cast<uint64_t>(tiling_type) << kBit8Shift);
  return static_cast<TilingId>(res);
}

Status TilingResourceManager::RecoverResource(const TilingId tiling_id,
                                              const std::shared_ptr<HostResource> &host_resource) {
  auto tiling_resource = std::dynamic_pointer_cast<TilingResource>(host_resource);
  GE_ASSERT_NOTNULL(tiling_resource, "Resource type is invalid.");
  auto iter = key_to_resources_.find(tiling_id);
  if (iter == key_to_resources_.end()) {
    key_to_resources_[tiling_id] = host_resource;
    resource_to_keys_[tiling_resource->GetOpRunInfo().get()].push_back(tiling_id);
  }

  return SUCCESS;
}

}  // namespace ge