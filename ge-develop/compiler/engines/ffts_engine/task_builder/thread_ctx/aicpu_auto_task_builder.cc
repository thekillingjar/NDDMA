/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aicpu_auto_task_builder.h"
#include "inc/ffts_utils.h"

namespace ffts {
AicpuAutoTaskBuilder::AicpuAutoTaskBuilder() {}

AicpuAutoTaskBuilder::~AicpuAutoTaskBuilder() {}

Status AicpuAutoTaskBuilder::GenContextDef(const ge::NodePtr &node, domi::FftsPlusTaskDef *ffts_plus_task_def) {
  FFTS_LOGD("AiCpu auto TaskBuilder::GenContextDef begin, node name:%s, node type:%s.", node->GetName().c_str(),
            node->GetType().c_str());
  auto op_desc = node->GetOpDesc();

  vector<FftsPlusComCtx_t> sub_ffts_plus_context;
  GenFftsPlusTaskCommonInfo(node, sub_ffts_plus_context);
  FftsPlusCtxDefPtr ctx_def_ptr = nullptr;
  ctx_def_ptr = op_desc->TryGetExtAttr("_ffts_plus_aicpu_ctx_def", ctx_def_ptr);
  FFTS_CHECK_NOTNULL(ctx_def_ptr);
  domi::FftsPlusAicpuCtxDef *aicpu_ctx_def_ptr = ctx_def_ptr->mutable_aicpu_ctx();
  FFTS_CHECK_NOTNULL(aicpu_ctx_def_ptr);
  for (size_t i = 0; i < sub_ffts_plus_context.size(); i++) {
    domi::FftsPlusCtxDef *ffts_plus_ctx_def = ffts_plus_task_def->add_ffts_plus_ctx();
    FFTS_CHECK_NOTNULL(ffts_plus_ctx_def);
    ffts_plus_ctx_def->set_context_type(RT_CTX_TYPE_AICPU);
    ffts_plus_ctx_def->set_op_index(op_desc->GetId());

    vector<uint32_t> context_id_list;
    if (ge::AttrUtils::GetListInt(op_desc, kAutoCtxIdList, context_id_list)) {
      if (context_id_list.size() != sub_ffts_plus_context.size()) {
        REPORT_FFTS_ERROR("[FftsPlusGenTask][GenCtxDef][AicpuAuto] Generating context definition failed: "
                          "the size of context_id_list is not equal to the size of sub_ffts_plus_context.");
        return FAILED;
      }
      ffts_plus_ctx_def->set_context_id(context_id_list[i]);
    }
    FFTS_LOGD("GenContextDef nodetype: %s, name: %s, context_type: %u, op_index: %u",
              node->GetType().c_str(), node->GetName().c_str(),
              ffts_plus_ctx_def->context_type(), ffts_plus_ctx_def->op_index());
    domi::FftsPlusAicpuCtxDef *aicpu_ctx_def = ffts_plus_ctx_def->mutable_aicpu_ctx();
    FFTS_CHECK_NOTNULL(aicpu_ctx_def);
    if (FillContextData(aicpu_ctx_def_ptr, aicpu_ctx_def) != SUCCESS) {
      REPORT_FFTS_ERROR("[FftsPlusGenTask][GenCtxDef][AicpuAuto] FillContextData failed for Op[%s], Optype[%s]",
                        op_desc->GetName().c_str(), op_desc->GetType().c_str());
      return FAILED;
    }
    if (sub_ffts_plus_context.empty()) {
      REPORT_FFTS_ERROR("[FftsPlusGenTask][GenCtxDef][AicpuAuto] Node[name=%s,type=%s] sub ffts plus context is empty.",
                        op_desc->GetName().c_str(), op_desc->GetType().c_str());
      return FAILED;
    }
    aicpu_ctx_def->set_pred_cnt(sub_ffts_plus_context[i].pred_cnt);
    aicpu_ctx_def->set_pred_cnt_init(sub_ffts_plus_context[i].pred_cnt);
    aicpu_ctx_def->set_aten(1);
    aicpu_ctx_def->set_successor_num(0);
    aicpu_ctx_def->set_thread_id(static_cast<uint32_t>(i));
  }
  return SUCCESS;
}
}  // namespace ffts
