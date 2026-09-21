/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "framework/runtime/model_rt_var_manager.h"
#include "framework/common/debug/ge_log.h"
#include "ge/ge_error_codes.h"
#include "common/checker.h"
#include "framework/common/types.h"
#include "graph/node.h"
#include "graph/utils/tensor_utils.h"
#include "graph/manager/mem_manager.h"
#include "graph/manager/active_memory_allocator.h"

namespace gert {
static void GeShapeAsRtShape(const ge::GeShape &ge_shape, gert::Shape &gert_shape) {
  gert_shape.SetDimNum(ge_shape.GetDims().size());
  for (size_t i = 0UL; i < gert_shape.GetDimNum(); ++i) {
    gert_shape.SetDim(i, ge_shape.GetDim(i));
  }
}

std::shared_ptr<ModelRtVarManager> ModelRtVarManager::Instance(const uint64_t session_id) {
  return RtVarManagerPool::Instance().GetVarManager(session_id);
}

ge::Status ModelRtVarManager::Init(const uint64_t device_id, const uint64_t logic_var_base,
                                   const int64_t total_var_size, void* external_var_addr, uint64_t external_var_size) {
  if (inited) {
    return ge::SUCCESS;
  }
  std::shared_ptr<ge::VarManager> var_manager = ge::VarManager::Instance(session_id_);
  GE_ASSERT_NOTNULL(var_manager);
  if (!var_manager->IsVarResourceInited()) {
    GE_ASSERT_SUCCESS(
        var_manager->Init(static_cast<int32_t>(ge::SessionVersion::ClOUD_VERSION), session_id_, device_id, 0U));
    var_manager->SetVarMemLogicBase(logic_var_base);
    var_manager->SetVarMemMaxSize(total_var_size);
    var_manager->SetExternalVar(external_var_addr, external_var_size);
    GELOGI("Reinit var manager in session:[%" PRIu64 "], logic_base:[%" PRIu64 "], total_size:[%" PRId64 "], "
           "external_var_addr %p, external_var_size %lu",
      session_id_, logic_var_base, total_var_size, external_var_addr, external_var_size);
  }
  if (!var_manager->HasMemoryManager()) {
    var_manager->SetMemManager(&ge::MemManager::Instance());
  }
  GELOGI("[Init] Variable mem auto malloc, no need to malloc var max size mem.");
  const auto page_size = ge::VarManager::IsVariableUse1gHugePage() ? ge::kDrv1GPageSize : ge::kDrvPageSize;
  auto allocator = ge::SessionMemAllocator<ge::ExpandableActiveMemoryAllocator>::Instance().GetMemAllocator(
      session_id_, device_id, RT_MEMORY_HBM, page_size);
  (void)var_manager->InitExpandableMemoryAllocator(allocator);
  inited = true;
  return ge::SUCCESS;
}

ge::Status ModelRtVarManager::RestoreDeviceVariables(const std::vector<ge::NodePtr> &variables, const uint32_t graph_id,
                                                     const uint32_t device_id, const bool need_collect) {
  GE_ASSERT_TRUE(inited);
  std::shared_ptr<ge::VarManager> var_manager = ge::VarManager::Instance(session_id_);
  GE_ASSERT_NOTNULL(var_manager);
  for (const auto &node : variables) {
    auto op_desc = node->GetOpDesc();
    GE_ASSERT_NOTNULL(op_desc);
    if ((op_desc->GetType() != ge::VARIABLE) && (op_desc->GetType() != ge::CONSTANTOP)) {
      continue;
    }
    const auto &var_desc = op_desc->GetOutputDesc(0U);
    if (!var_manager->IsVarExist(op_desc->GetName(), var_desc)) {
      GE_ASSERT_SUCCESS(var_manager->RestoreVarMem(op_desc->GetName(), op_desc, var_desc, RT_MEMORY_HBM),
                        "Failed to restore var [%s] to var_manager, session:[%" PRIu64 "].", op_desc->GetNamePtr(),
                        session_id_);
      var_manager->SetAllocatedGraphId(node->GetName(), graph_id);
      GELOGD("Variable [%s] has been recovered by graph_id [%u].", op_desc->GetName().c_str(), graph_id);
    }
    auto iter = name_to_var_info_.find(op_desc->GetName());
    if (!need_collect || iter != name_to_var_info_.end()) {
      continue;
    }
    auto &var_info = name_to_var_info_[op_desc->GetName()];
    GeShapeAsRtShape(var_desc.GetOriginShape(), var_info.shape_info.MutableOriginShape());
    GeShapeAsRtShape(var_desc.GetShape(), var_info.shape_info.MutableStorageShape());
    uint8_t *var_logic{nullptr};
    GE_ASSERT_SUCCESS(var_manager->GetVarAddr(op_desc->GetName(), var_desc, var_logic));
    var_info.var_addr = var_manager->GetVarMemoryAddr(var_logic, RT_MEMORY_HBM, device_id);
    int64_t tensor_size{0};
    (void)ge::TensorUtils::GetSize(var_desc, tensor_size);
    GE_ASSERT_TRUE(tensor_size > 0);
    var_info.var_size = static_cast<uint64_t>(tensor_size);
    GELOGD("Variable [%s] has been collected in session [%u], addr:[%" PRIx64 "], size:[%" PRId64 "]",
      op_desc->GetNamePtr(), session_id_, var_info.var_addr, var_info.var_size);
  }
  return ge::SUCCESS;
}

ge::Status ModelRtVarManager::GetVarShapeAndMemory(const std::string &id, StorageShape &shape,
                                                   TensorData &memory) const {
  GE_ASSERT_TRUE(inited);
  const auto iter = name_to_var_info_.find(id);
  GE_ASSERT_TRUE(iter != name_to_var_info_.end(), "Variable [%s] is not found.", id.c_str());
  shape = iter->second.shape_info;
  memory.SetPlacement(iter->second.placement);
  memory.SetAddr(iter->second.var_addr, nullptr);
  memory.SetSize(iter->second.var_size);
  return ge::SUCCESS;
}

RtVarManagerPool &RtVarManagerPool::Instance() {
  static RtVarManagerPool var_manager_pool;
  return var_manager_pool;
}

std::shared_ptr<ModelRtVarManager> RtVarManagerPool::GetVarManager(const uint64_t session_id) {
  const std::lock_guard<std::mutex> lg(var_manager_mutex_);
  const auto &iter = session_id_to_var_manager_.find(session_id);
  if (iter != session_id_to_var_manager_.end()) {
    return iter->second;
  }
  const std::shared_ptr<ModelRtVarManager> var_manager = ge::MakeShared<ModelRtVarManager>(session_id);
  GE_ASSERT_NOTNULL(var_manager);
  session_id_to_var_manager_[session_id] = var_manager;
  return var_manager;
}

void RtVarManagerPool::RemoveRtVarManager(const uint64_t session_id) {
  std::shared_ptr<ModelRtVarManager> var_manager = nullptr;
  {
    const std::lock_guard<std::mutex> lock(var_manager_mutex_);
    const auto it = session_id_to_var_manager_.find(session_id);
    if (it != session_id_to_var_manager_.end()) {
      var_manager = it->second;
      (void)session_id_to_var_manager_.erase(it);
    }
  }

  if (var_manager != nullptr) {
    var_manager->Destroy();
  }
}

}  // namespace gert