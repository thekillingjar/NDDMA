/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aic_aiv_manual_task_builder.h"
#include "inc/ffts_utils.h"
#include "graph/debug/ge_attr_define.h"

namespace ffts {
AICAIVTaskBuilder::AICAIVTaskBuilder() {}

AICAIVTaskBuilder::~AICAIVTaskBuilder() {}

Status AICAIVTaskBuilder::GenContextDef(const ge::NodePtr &node, domi::FftsPlusTaskDef *ffts_plus_task_def) {
  FFTS_LOGD("AIC AIV task builder genContextDef begin, node name:%s, node type:%s.", node->GetName().c_str(),
            node->GetType().c_str());
  auto op_desc = node->GetOpDesc();
  vector<FftsPlusComCtx_t> sub_ffts_plus_context;
  GenFftsPlusTaskCommonInfo(node, sub_ffts_plus_context);
  FftsPlusCtxDefPtr ctx_def_ptr = nullptr;
  ctx_def_ptr = op_desc->TryGetExtAttr(kAttrAICoreCtxDef, ctx_def_ptr);
  FFTS_CHECK_NOTNULL(ctx_def_ptr);
  domi::FftsPlusAicAivCtxDef *aicore_ctx_def = ctx_def_ptr->mutable_aic_aiv_ctx();
  FFTS_CHECK_NOTNULL(aicore_ctx_def);
  domi::FftsPlusCtxDef *ffts_plus_ctx_def = ffts_plus_task_def->add_ffts_plus_ctx();
  FFTS_CHECK_NOTNULL(ffts_plus_ctx_def);
  std::string core_type;
  (void)ge::AttrUtils::GetStr(op_desc, ge::ATTR_NAME_CUBE_VECTOR_CORE_TYPE, core_type);
  if (core_type.empty()) {
    return FAILED;
  }
  if (core_type == kCoreTypeAIC) {
    ffts_plus_ctx_def->set_context_type(RT_CTX_TYPE_AICORE);
  } else {
    ffts_plus_ctx_def->set_context_type(RT_CTX_TYPE_AIV);
  }
  ffts_plus_ctx_def->set_op_index(op_desc->GetId());

  uint32_t context_id = 0;
  if (ge::AttrUtils::GetInt(op_desc, kContextId, context_id)) {
    ffts_plus_ctx_def->set_context_id(context_id);
  }
  FFTS_LOGD("GenContextDef aicaiv manual context_id: %u, node_type: %s, name: %s, context_type: %u, op_index: %u.",
            context_id, node->GetType().c_str(), node->GetName().c_str(), ffts_plus_ctx_def->context_type(),
            ffts_plus_ctx_def->op_index());
  domi::FftsPlusAicAivCtxDef *aic_aiv_ctx_def = ffts_plus_ctx_def->mutable_aic_aiv_ctx();
  FFTS_CHECK_NOTNULL(aic_aiv_ctx_def);
  FillContextData(aicore_ctx_def, aic_aiv_ctx_def);

  if (sub_ffts_plus_context.empty()) {
    return FAILED;
  }
  aic_aiv_ctx_def->set_pred_cnt(sub_ffts_plus_context[0].pred_cnt);
  aic_aiv_ctx_def->set_pred_cnt_init(sub_ffts_plus_context[0].pred_cnt);
  aic_aiv_ctx_def->set_aten(0);
  aic_aiv_ctx_def->set_successor_num(0);

  uint32_t thread_id = 0;
  if (ge::AttrUtils::GetInt(op_desc, kThreadId, thread_id)) {
    aic_aiv_ctx_def->set_thread_id(thread_id);
    FFTS_LOGD("GenContextDef threadid nodetype: %s, name: %s, context_id: %u, thread_id: %u.", node->GetType().c_str(),
              node->GetName().c_str(), context_id, thread_id);
  } else {
    FFTS_LOGD("GenContextDef threadid nodetype:%s, name:%s, context_id:%u, get no thread id.", node->GetType().c_str(),
              node->GetName().c_str(), context_id);
  }

  aic_aiv_ctx_def->set_thread_window_size(kDefaultManualWindowSize); // not used yet
  FFTS_LOGD("GenContextDef threadid nodetype:%s, name:%s, defaultManualWindowSize:%u.", node->GetType().c_str(),
            node->GetName().c_str(), kDefaultManualWindowSize);

  (void)ge::AttrUtils::SetListInt(op_desc, kSuccList, sub_ffts_plus_context[0].succ_list);

  uint32_t addr_size = aic_aiv_ctx_def->task_addr_size();
  uint32_t cur_addr_size = ffts_plus_task_def->addr_size();
  ffts_plus_task_def->set_addr_size(cur_addr_size + addr_size);
  FFTS_LOGD("GenContextDef nodetype:%s, name:%s, total_addr_size:%u", node->GetType().c_str(),
            node->GetName().c_str(), ffts_plus_task_def->addr_size());
  return SUCCESS;
}
}  // namespace ffts
