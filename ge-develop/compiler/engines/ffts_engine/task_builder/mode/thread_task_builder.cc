/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "task_builder/mode/thread_task_builder.h"
#include "common/sgt_slice_type.h"
#include "common/aicore_util_attr_define.h"
#include "graph/debug/ge_attr_define.h"
#include "inc/ffts_type.h"
namespace ffts {
namespace {
  const std::vector<std::string> kCtxTypeAttrs = {kAttrAICpuCtxDef, kAttrDsaCtxDef};
}
TheadTaskBuilder::TheadTaskBuilder() {}

TheadTaskBuilder::~TheadTaskBuilder() {}

void TheadTaskBuilder::SetModeType(const ModeType &type) {
  mode_type_ = type;
}

Status TheadTaskBuilder::GetNodeContextTypeByNode(const ge::NodePtr &node, TaskBuilderType &task_builder_type) const {
  ge::OpDescPtr op_desc = node->GetOpDesc();
  if (op_desc->HasAttr(kAttrAICoreCtxType)) {
    int64_t ctx_type = static_cast<int64_t>(TaskBuilderType::EN_TASK_TYPE_RESERVED);
    (void)ge::AttrUtils::GetInt(op_desc, kAttrAICoreCtxType, ctx_type);
    task_builder_type = TaskBuilderType(ctx_type);
    return SUCCESS;
  }
  if (kHCCLOpType.count(op_desc->GetType()) > 0) {
    task_builder_type = TaskBuilderType::EN_TASK_TYPE_COLLECTION_COMMICATE;
    return SUCCESS;
  }
  for (auto const &ctx_type_attr : kCtxTypeAttrs) {
    FftsPlusCtxDefPtr ctx_def_ptr = nullptr;
    ctx_def_ptr = op_desc->TryGetExtAttr(ctx_type_attr, ctx_def_ptr);
    if (ctx_def_ptr == nullptr) {
      continue;
    }
    ffts::ThreadSliceMapPtr slice_info_ptr = nullptr;
    slice_info_ptr = op_desc->TryGetExtAttr(ffts::kAttrSgtStructInfo, slice_info_ptr);
    if (slice_info_ptr == nullptr) {
      continue;
    }
    auto innerkey = ctx_type_attr;
    if (slice_info_ptr->thread_mode == kAutoMode) {
      innerkey += kAutoSuffix;
    }
    auto iter = kDsaOrAicpuMap.find(innerkey);
    if (iter != kDsaOrAicpuMap.end()) {
      task_builder_type = iter->second;
      return SUCCESS;
    }
  }
  FftsPlusCtxDefPtr runtime_control_ctx_def =  nullptr;
  runtime_control_ctx_def = op_desc->TryGetExtAttr("FFTS_PLUS_TASK_DEF", runtime_control_ctx_def);
  FFTS_LOGD("Node[%s] rts def:%lx.", node->GetName().c_str(), runtime_control_ctx_def);
  if (runtime_control_ctx_def != nullptr) {
    task_builder_type = (mode_type_ == ModeType::MANUAL_MODE_TYPE) ? TaskBuilderType::EN_TASK_TYPE_RUNTIME_CONTROL :
        TaskBuilderType::EN_TASK_TYPE_RUNTIME_CONTROL_AUTO;
    return SUCCESS;
  }
  FFTS_LOGE("Node [%s] of type [%s] cannot obtain context type.", node->GetName().c_str(), node->GetType().c_str());
  return FAILED;
}

FFTSPlusTaskBuilderPtr TheadTaskBuilder::GetTaskBuilder(TaskBuilderType task_builder_type) {
  switch (task_builder_type) {
    case TaskBuilderType::EN_TASK_TYPE_AIC_AIV:
      return aic_aiv_task_builder_ptr_;
    case TaskBuilderType::EN_TASK_TYPE_AIC_AIV_AUTO:
     return aic_aiv_auto_task_builder_ptr_;
    case TaskBuilderType::EN_TASK_TYPE_AIC_AIV_DYNAMIC:
      return aic_aiv_dynamic_task_builder_ptr_;
    case TaskBuilderType::EN_TASK_TYPE_MIX_AIC_AIV_DYNAMIC:
      return mix_aic_aiv_dynamic_task_builder_ptr_;
    case TaskBuilderType::EN_TASK_TYPE_MIX_AIC_AIV:
      return mix_aic_aiv_task_builder_ptr_;
    case TaskBuilderType::EN_TASK_TYPE_MIX_AIC_AIV_AUTO:
      return mix_aic_aiv_auto_task_builder_ptr_;
    case TaskBuilderType::EN_TASK_TYPE_MIX_L2_AIC_AIV:
      return mix_L2_task_builder_ptr_;
    case TaskBuilderType::EN_TASK_TYPE_COLLECTION_COMMICATE:
      return collection_ops_task_builder_ptr_;
    case TaskBuilderType::EN_TASK_TYPE_AICPU:
      return aicpu_task_builder_ptr_;
    case TaskBuilderType::EN_TASK_TYPE_AICPU_AUTO:
      return aicpu_auto_task_builder_ptr_;
    case TaskBuilderType::EN_TASK_TYPE_RUNTIME_CONTROL:
      return runtime_ops_task_builder_ptr_;
    case TaskBuilderType::EN_TASK_TYPE_RUNTIME_CONTROL_AUTO:
      return runtime_ops_auto_task_builder_ptr_;
    case TaskBuilderType::EN_TASK_TYPE_DSA:
      return dsa_ops_task_builder_ptr_;
    default:
      return nullptr;
  }
}

Status TheadTaskBuilder::GenerateDataTaskDef(const ge::NodePtr &node, domi::FftsPlusTaskDef *ffts_plus_task_def,
                                             const ModeType &mode_type) const {
  FFTS_LOGD("Current FFT mode type is %d.", static_cast<int>(mode_type));
  if (mode_type == ModeType::MANUAL_MODE_TYPE) {
    PrefetchTaskBuilder prefetch;
    OutTaskBuilder invalid(CACHE_OPERATION::INVALIDATE);
    OutTaskBuilder write_back(CACHE_OPERATION::WRITE_BACK);
    if (prefetch.GenManualDataCtxDef(node, ffts_plus_task_def) != SUCCESS) {
      return FAILED;
    }
    if (invalid.GenManualDataCtxDef(node, ffts_plus_task_def) != SUCCESS) {
      return FAILED;
    }
    if (write_back.GenManualDataCtxDef(node, ffts_plus_task_def) != SUCCESS) {
      return FAILED;
    }
  } else if (mode_type == ModeType::AUTO_MODE_TYPE) {
    PrefetchAutoTaskBuilder prefetch_auto;
    OutAutoTaskBuilder invalid_auto(CACHE_OPERATION::INVALIDATE);
    OutAutoTaskBuilder write_back_auto(CACHE_OPERATION::WRITE_BACK);
    if (prefetch_auto.GenAutoDataCtxDef(node, ffts_plus_task_def) != SUCCESS) {
      return FAILED;
    }
    if (invalid_auto.GenAutoDataCtxDef(node, ffts_plus_task_def) != SUCCESS) {
      return FAILED;
    }
    if (write_back_auto.GenAutoDataCtxDef(node, ffts_plus_task_def) != SUCCESS) {
      return FAILED;
    }
  } else if (mode_type == ModeType::DYNAMIC_MODE_TYPE) {
    PrefetchDynamicTaskBuilder prefetch_dyn;
    OutDynamicTaskBuilder invalid_dyn(CACHE_OPERATION::INVALIDATE);
    OutDynamicTaskBuilder write_back_dyn(CACHE_OPERATION::WRITE_BACK);
    if (prefetch_dyn.GenDynamicDataCtxDef(node, ffts_plus_task_def) != SUCCESS) {
      return FAILED;
    }
    if (invalid_dyn.GenDynamicDataCtxDef(node, ffts_plus_task_def) != SUCCESS) {
      return FAILED;
    }
    if (write_back_dyn.GenDynamicDataCtxDef(node, ffts_plus_task_def) != SUCCESS) {
      return FAILED;
    }
  }
  return SUCCESS;
}

bool TheadTaskBuilder::IsNoCtx(const ge::NodePtr &node) const {
  ge::OpDescPtr op_desc = node->GetOpDesc();
  if (NO_NEED_GEN_TASK_OP_TYPE.count(op_desc->GetType()) != 0) {
    return true;
  }
  if (CONTROL_OP_V2_TYPE.count(op_desc->GetType()) != 0) {
    return true;
  }
  bool no_task = false;
  (void)ge::AttrUtils::GetBool(op_desc, ge::ATTR_NAME_NOTASK, no_task);
  if (no_task) {
    return true;
  }
  return false;
}

Status TheadTaskBuilder::GenFftsPlusContextIdWithMemSet(std::vector<ge::NodePtr> &pre_sub_graph_nodes,
                                                        std::vector<ge::NodePtr> &memset_nodes,
                                                        ge::ComputeGraph &sgt_graph) const {
  ge::NodePtr memset_node = nullptr;
  for (auto &node : sgt_graph.GetDirectNode()) {
    FFTS_CHECK_NOTNULL(node);
    memset_node = nullptr;
    memset_node = node->GetOpDesc()->TryGetExtAttr(fe::ATTR_NAME_MEMSET_NODE, memset_node);
    if (memset_node == nullptr) {
      continue;
    }
    if (memset_node->GetType() == fe::MEMSET_OP_TYPE) {
      FFTS_LOGD("Get MemSet Node: %s, type: %s from Node: %s.", memset_node->GetName().c_str(),
                memset_node->GetType().c_str(), node->GetName().c_str());
      (void)ge::AttrUtils::SetBool(memset_node->GetOpDesc(), kOnlyAddContext, true);
      // duplicate MemSet node removal
      auto iter = find(memset_nodes.begin(), memset_nodes.end(), memset_node);
      if (iter != memset_nodes.end()) {
        continue;
      }
      pre_sub_graph_nodes.emplace_back(memset_node);
      memset_nodes.emplace_back(memset_node);
    }
  }
  return SUCCESS;
}
}  // namespace ffts
