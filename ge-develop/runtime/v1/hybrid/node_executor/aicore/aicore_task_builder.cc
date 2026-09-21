/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "hybrid/node_executor/aicore/aicore_task_builder.h"
#include "framework/common/debug/log.h"
#include "hybrid/node_executor/aicore/aicore_node_executor.h"
#include "hybrid/node_executor/aicpu/aicpu_node_executor.h"
#include "hybrid/node_executor/controlop/control_op_executor.h"
#include "common/plugin/ge_make_unique_util.h"
#include "graph/bin_cache/node_bin_selector_factory.h"

namespace ge {
namespace hybrid {
namespace {
const char_t *const kPartiallySupported = "partially_supported";
}

AiCoreTaskBuilder::AiCoreTaskBuilder(const NodePtr &node, const std::vector<domi::TaskDef> &task_defs,
                                     const HybridModel &model, AiCoreNodeTask &aicore_node_task)
    : node_(node),
      op_desc_(node->GetOpDesc()),
      task_defs_(task_defs),
      model_(model),
      aicore_node_task_(aicore_node_task) {}

Status AiCoreTaskBuilder::InitTaskDef() {
  for (size_t i = 0U; i < task_defs_.size(); ++i) {
    const domi::TaskDef &task_def = task_defs_[i];
    const auto task_type = static_cast<ModelTaskType>(task_def.type());
    if ((task_type == ModelTaskType::MODEL_TASK_KERNEL) || (task_type == ModelTaskType::MODEL_TASK_ALL_KERNEL)) {
      const auto &context = (task_type == ModelTaskType::MODEL_TASK_KERNEL) ? task_def.kernel().context() :
          task_def.kernel_with_handle().context();
      const auto kernel_type = static_cast<ccKernelType>(context.kernel_type());
      if (kernel_type == ccKernelType::TE) {
        aicore_task_defs_.emplace_back(task_def);
      } else if ((kernel_type == ccKernelType::AI_CPU) || (kernel_type == ccKernelType::CUST_AI_CPU)) {
        aicpu_task_defs_.emplace_back(&task_def);
      } else {
        GELOGE(ACL_ERROR_GE_OP_KERNEL_TYPE_INVALID,
               "[Check][KernelType]Only TBE, AI_CPU, CUST_AI_CPU kernel are supported, but got %d",
               static_cast<int32_t>(kernel_type));
        REPORT_INNER_ERR_MSG("E19999",
            "Init taskdef fail for %d not supported, Only TBE, AI_CPU, CUST_AI_CPU kernel are supported.",
            static_cast<int32_t>(kernel_type));
        return ACL_ERROR_GE_OP_KERNEL_TYPE_INVALID;
      }
    } else if (task_type == ModelTaskType::MODEL_TASK_KERNEL_EX) {
      aicpu_task_defs_.emplace_back(&task_def);
    } else {
      GELOGE(INTERNAL_ERROR, "Only AI_CORE and AI_CPU kernel for aicore task, but got task_type %d",
          static_cast<int32_t>(task_type));
      REPORT_INNER_ERR_MSG("E19999",
          "Only AI_CORE and AI_CPU kernel for aicore task, but got task_type %d",
          static_cast<int32_t>(task_type));
      return INTERNAL_ERROR;
    }
  }
  return SUCCESS;
}

Status AiCoreTaskBuilder::LoadAtomicWorkspace() {
  std::map<std::string, std::map<int64_t, int64_t>> workspace_info;
  workspace_info = op_desc_->TryGetExtAttr(EXT_ATTR_ATOMIC_WORKSPACE_INFO, workspace_info);
  if (!workspace_info.empty()) {
    return SUCCESS;
  }
  GeAttrValue::NAMED_ATTRS workspaces;

  if (!AttrUtils::GetNamedAttrs(op_desc_, EXT_ATTR_ATOMIC_WORKSPACE_INFO, workspaces)) {
    return SUCCESS;
  }
  std::vector<int64_t> value;
  const std::string &op_name = op_desc_->GetName();
  (void)AttrUtils::GetListInt(workspaces, op_name, value);
  if (value.empty()) {
    return SUCCESS;
  }
  std::map<int64_t, int64_t> index_offset;
  for (size_t i = 0U; i < (value.size() - 1U); i += 2U) {  // two sets of vector, parsing the key value of the map
    index_offset[value[i]] = value[i + 1U];
  }
  workspace_info[op_name] = index_offset;
  if (!op_desc_->SetExtAttr(EXT_ATTR_ATOMIC_WORKSPACE_INFO, workspace_info)) {
    GELOGE(INTERNAL_ERROR, "[Set][Attr:%s]fail for node:%s.",
           EXT_ATTR_ATOMIC_WORKSPACE_INFO.c_str(), op_desc_->GetName().c_str());
    REPORT_INNER_ERR_MSG("E19999", "Set Attr:%s fail for node:%s.",
                       EXT_ATTR_ATOMIC_WORKSPACE_INFO.c_str(), op_desc_->GetName().c_str());
    return INTERNAL_ERROR;
  }
  return SUCCESS;
}

Status AiCoreTaskBuilder::BuildTask() {
  GE_CHECK_NOTNULL(op_desc_);
  GE_CHK_STATUS_RET(InitTaskDef());
  GE_CHK_STATUS_RET(LoadAtomicWorkspace(),
                    "[LoadAtomicWorkSpace]failed for [%s(%s)].",
                    op_desc_->GetName().c_str(), op_desc_->GetType().c_str());
  if (aicore_task_defs_.size() > kNumTaskWithAtomicAddrCleanTask) {
    GELOGE(INTERNAL_ERROR, "[Check][Size][%s(%s)] At most %zu task was supported, but got %zu",
           op_desc_->GetName().c_str(), op_desc_->GetType().c_str(),
           kNumTaskWithAtomicAddrCleanTask, aicore_task_defs_.size());
    REPORT_INNER_ERR_MSG("E19999", "[%s(%s)] At most %zu task was supported, but got %zu, check invalid.",
                       op_desc_->GetName().c_str(), op_desc_->GetType().c_str(),
                       kNumTaskWithAtomicAddrCleanTask, aicore_task_defs_.size());
    return INTERNAL_ERROR;
  }

  const bool is_single_op = model_.IsSingleOp();
  if (ExpectAtomicAddrCleanTask()) {
    if (aicore_task_defs_.size() != kNumTaskWithAtomicAddrCleanTask) {
      GELOGE(INTERNAL_ERROR, "[Check][Size][%s(%s)] AtomicAddrClean task was expected:%zu, but got %zu task_defs",
             op_desc_->GetName().c_str(), op_desc_->GetType().c_str(),
             kNumTaskWithAtomicAddrCleanTask, aicore_task_defs_.size());
      REPORT_INNER_ERR_MSG("E19999", "[%s(%s)] AtomicAddrClean task was expected:%zu, but got %zu task_defs,",
                         op_desc_->GetName().c_str(), op_desc_->GetType().c_str(),
                         kNumTaskWithAtomicAddrCleanTask, aicore_task_defs_.size());
      return INTERNAL_ERROR;
    }

    GELOGD("[%s] Build AtomicAddrClean task.", op_desc_->GetName().c_str());
    auto atomic_task = MakeUnique<AtomicAddrCleanOpTask>();
    GE_CHECK_NOTNULL(atomic_task);
    atomic_task->SetModelId(model_.GetModelId());
    atomic_task->SetSingleOp(is_single_op);
    void *const atomic_overflow_addr = model_.GetOverflowAddr();
    atomic_task->SetOverflowAddr(atomic_overflow_addr);
    GE_CHK_STATUS_RET(atomic_task->SetPlatformInfo(node_, model_.GetPlatformInfo()),
                      "Set platform info to atomic clean task failed.");
    GE_CHK_STATUS_RET(atomic_task->Init(node_, aicore_task_defs_.front()),
                      "[Invoke][AtomicAddrCleanOpTask::Init] failed for [%s(%s)].",
                      op_desc_->GetName().c_str(), op_desc_->GetType().c_str());
    aicore_node_task_.tasks_.emplace_back(std::move(atomic_task));
  }

  if (!aicore_task_defs_.empty()) {
    // build aicore task
    auto aicore_task = MakeUnique<AiCoreOpTask>();
    GE_CHECK_NOTNULL(aicore_task);
    aicore_task->SetModelId(model_.GetModelId());
    aicore_task->SetSingleOp(is_single_op);
    void *const aicore_overflow_addr = model_.GetOverflowAddr();
    aicore_task->SetOverflowAddr(aicore_overflow_addr);

    GE_CHK_STATUS_RET(aicore_task->Init(node_, aicore_task_defs_.back()),
                      "[Invoke][AiCoreOpTask::Init] failed for [%s(%s)].", op_desc_->GetName().c_str(),
                      op_desc_->GetType().c_str());
    GE_CHK_STATUS_RET(aicore_task->SetPlatformInfo(node_, model_.GetPlatformInfo()), "Set platform info failed.");
    aicore_node_task_.tasks_.emplace_back(std::move(aicore_task));
  }

  const auto selector = NodeBinSelectorFactory::GetInstance().GetNodeBinSelector(model_.GetNodeBinMode());
  if (selector == nullptr) {
    GELOGE(ACL_ERROR_GE_INTERNAL_ERROR, "Can not find the handler for bin mode %d", model_.GetNodeBinMode());
    return FAILED;
  }

  const auto ret = selector->Initialize();
  if (ret != SUCCESS) {
    return ret;
  }

  aicore_node_task_.SetBinSelector(selector);
  return SUCCESS;
}

Status AiCoreTaskBuilder::LoadAicpuTask() {
  bool partially_supported = false;
  (void)AttrUtils::GetBool(op_desc_, kPartiallySupported, partially_supported);
  if (partially_supported) {
    if (aicpu_task_defs_.empty()) {
      GELOGW("The aicpu task with partially_supported task was not found.");
      return SUCCESS;
    }

    const auto node_item = model_.GetNodeItem(node_);
    GE_CHECK_NOTNULL(node_item);
    const auto &task_def = *(aicpu_task_defs_[0U]);
    std::unique_ptr<AicpuNodeTaskBase> aicpu_task;
    if (static_cast<ModelTaskType>(task_def.type()) == ModelTaskType::MODEL_TASK_KERNEL_EX) {
      aicpu_task = MakeUnique<AicpuTfNodeTask>(node_item, task_def);
    } else if (static_cast<ModelTaskType>(task_def.type()) == ModelTaskType::MODEL_TASK_KERNEL) {
      aicpu_task = MakeUnique<AicpuNodeTask>(node_item, task_def);
    } else {
      REPORT_INNER_ERR_MSG("E19999", "Node[%s] task type=%u is not supported by aicpu node executor.",
                         node_->GetName().c_str(), task_def.type());
      GELOGE(UNSUPPORTED, "Node[%s] task type=%u is not supported by aicpu node executor.",
             node_->GetName().c_str(), task_def.type());
      return UNSUPPORTED;
    }
    GE_CHK_BOOL_RET_STATUS(aicpu_task != nullptr, MEMALLOC_FAILED,
                           "Load task for node %s failed.", node_->GetName().c_str());
    GE_CHK_STATUS_RET(aicpu_task->Init(model_), "Node[%s] task init failed.", node_->GetName().c_str());
    aicore_node_task_.aicpu_task_ = std::move(aicpu_task);
  }
  return SUCCESS;
}

Status AiCoreTaskBuilder::LoadFusedTask() {
  if (!op_desc_->HasAttr("_original_fusion_graph")) {
    GELOGD("Node %s is not fused node. Skip load fused task.", op_desc_->GetName().c_str());
    return SUCCESS;
  }
  const auto node_item = model_.GetNodeItem(node_);
  GE_CHECK_NOTNULL(node_item);
  GE_CHECK_NOTNULL(node_item->fused_subgraph);
  std::unique_ptr<FusedOpNodeTask> fused_task = MakeUnique<FusedOpNodeTask>();
  GE_CHECK_NOTNULL(fused_task);
  GE_CHK_STATUS_RET(fused_task->Init(node_, model_), "Node[%s] origin_fused_task init failed.",
                    op_desc_->GetName().c_str());
  aicore_node_task_.fused_task_ = std::move(fused_task);
  return SUCCESS;
}

bool AiCoreTaskBuilder::ExpectAtomicAddrCleanTask() const {
  if (op_desc_->HasAttr(ATOMIC_ATTR_OUTPUT_INDEX)) {
    GELOGD("[%s] Node has ATOMIC_ATTR_OUTPUT_INDEX", op_desc_->GetName().c_str());
    return true;
  }
  std::map<std::string, std::map<int64_t, int64_t>> workspace_info;
  workspace_info = op_desc_->TryGetExtAttr(EXT_ATTR_ATOMIC_WORKSPACE_INFO, workspace_info);
  return !workspace_info.empty();
}
}  // namespace hybrid
}  // namespace ge
