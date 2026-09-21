/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aicpu_manual_task_builder.h"
#include "inc/ffts_utils.h"

namespace ffts {
AicpuTaskBuilder::AicpuTaskBuilder() {}

AicpuTaskBuilder::~AicpuTaskBuilder() {}

Status AicpuTaskBuilder::GenContextDef(const ge::NodePtr &node, domi::FftsPlusTaskDef *ffts_plus_task_def) {
  FFTS_LOGD("AiCpu TaskBuilder::GenContextDef begin, node name:%s, node type:%s.", node->GetName().c_str(),
            node->GetType().c_str());
  auto op_desc = node->GetOpDesc();

  vector<FftsPlusComCtx_t> sub_ffts_plus_context;
  GenFftsPlusTaskCommonInfo(node, sub_ffts_plus_context);
  FftsPlusCtxDefPtr ctx_def_ptr = nullptr;
  ctx_def_ptr = op_desc->TryGetExtAttr("_ffts_plus_aicpu_ctx_def", ctx_def_ptr);
  FFTS_CHECK_NOTNULL(ctx_def_ptr);
  domi::FftsPlusAicpuCtxDef *aicpu_ctx_def_ptr = ctx_def_ptr->mutable_aicpu_ctx();
  FFTS_CHECK_NOTNULL(aicpu_ctx_def_ptr);
  domi::FftsPlusCtxDef *ffts_plus_ctx_def = ffts_plus_task_def->add_ffts_plus_ctx();
  FFTS_CHECK_NOTNULL(ffts_plus_ctx_def);
  ffts_plus_ctx_def->set_context_type(RT_CTX_TYPE_AICPU);
  ffts_plus_ctx_def->set_op_index(op_desc->GetId());

  uint32_t context_id = 0;
  if (ge::AttrUtils::GetInt(op_desc, kContextId, context_id)) {
    ffts_plus_ctx_def->set_context_id(context_id);
  }
  FFTS_LOGD("GenContextDef aicpu manual context_id:%u, nodetype:%s, name:%s, context_type:%u, op_index:%u",
            context_id, node->GetType().c_str(), node->GetName().c_str(),
            ffts_plus_ctx_def->context_type(), ffts_plus_ctx_def->op_index());
  domi::FftsPlusAicpuCtxDef *aicpu_ctx_def = ffts_plus_ctx_def->mutable_aicpu_ctx();
  FFTS_CHECK_NOTNULL(aicpu_ctx_def);
  if (FillContextData(aicpu_ctx_def_ptr, aicpu_ctx_def) != SUCCESS) {
    FFTS_LOGE("FillContextData failed. Op[%s, optype[%s]]",
              op_desc->GetName().c_str(), op_desc->GetType().c_str());
    return FAILED;
  }
  if (sub_ffts_plus_context.empty()) {
    REPORT_FFTS_ERROR("[AicpuTaskBuilder][GenContextDef] Node[name=%s, type=%s] sub ffts plus context is empty.",
                      op_desc->GetName().c_str(), op_desc->GetType().c_str());
    return FAILED;
  }
  aicpu_ctx_def->set_pred_cnt(sub_ffts_plus_context[0].pred_cnt);
  aicpu_ctx_def->set_pred_cnt_init(sub_ffts_plus_context[0].pred_cnt);
  aicpu_ctx_def->set_aten(0);
  aicpu_ctx_def->set_successor_num(0);

  uint32_t args_size = aicpu_ctx_def_ptr->kernel().args_size();
  uint32_t addr_size = args_size / sizeof(uintptr_t);
  addr_size = (args_size % sizeof(uintptr_t) == 0) ? addr_size : addr_size + 1;
  uint32_t cur_size = ffts_plus_task_def->addr_size();
  ffts_plus_task_def->set_addr_size(cur_size + addr_size);
  FFTS_LOGD("AICpu add addr size[%u] with value[%u].", addr_size, cur_size);
  (void)ge::AttrUtils::SetListInt(op_desc, kSuccList, sub_ffts_plus_context[0].succ_list);
  return SUCCESS;
}
}  // namespace ffts
