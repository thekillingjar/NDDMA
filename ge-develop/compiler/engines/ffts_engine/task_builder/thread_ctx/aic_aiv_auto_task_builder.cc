/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aic_aiv_auto_task_builder.h"
#include "inc/ffts_utils.h"

namespace ffts {
AICAIVAutoTaskBuilder::AICAIVAutoTaskBuilder() {}

AICAIVAutoTaskBuilder::~AICAIVAutoTaskBuilder() {}

Status AICAIVAutoTaskBuilder::GenContextDef(const ge::NodePtr &node, domi::FftsPlusTaskDef *ffts_plus_task_def) {
  FFTS_LOGD("AIC AIV auto task builder genContextDef begin, node name:%s, node type:%s.", node->GetName().c_str(),
            node->GetType().c_str());
  auto op_desc = node->GetOpDesc();
  vector<FftsPlusComCtx_t> sub_ffts_plus_context;
  GenFftsPlusTaskCommonInfo(node, sub_ffts_plus_context);
  FftsPlusCtxDefPtr ctx_def_ptr = nullptr;
  ctx_def_ptr = op_desc->TryGetExtAttr(kAttrAICoreCtxDef, ctx_def_ptr);
  FFTS_CHECK_NOTNULL(ctx_def_ptr);
  domi::FftsPlusAicAivCtxDef *aicore_ctx_def = ctx_def_ptr->mutable_aic_aiv_ctx();
  FFTS_CHECK_NOTNULL(aicore_ctx_def);
  uint32_t addr_size = 0;
  uint32_t thread_dim = 0;

  ThreadSliceMapPtr slice_info_ptr = nullptr;
  slice_info_ptr = op_desc->TryGetExtAttr(kAttrSgtStructInfo, slice_info_ptr);
  FFTS_CHECK_NOTNULL(slice_info_ptr);
  size_t window_size = sub_ffts_plus_context.size();
  for (size_t i = 0; i < window_size; ++i) {
    domi::FftsPlusCtxDef *ffts_plus_ctx_def = ffts_plus_task_def->add_ffts_plus_ctx();
    FFTS_CHECK_NOTNULL(ffts_plus_ctx_def);

    vector<string> thread_core_type;
    (void)ge::AttrUtils::GetListStr(op_desc, kAttrThreadCoreType, thread_core_type);
    if (thread_core_type.empty()) {
      return FAILED;
    }
    if (thread_core_type[0] == kCoreTypeAIC) {
      ffts_plus_ctx_def->set_context_type(RT_CTX_TYPE_AICORE);
    } else {
      ffts_plus_ctx_def->set_context_type(RT_CTX_TYPE_AIV);
    }

    vector<uint32_t> auto_ctx_id_list;
    if (ge::AttrUtils::GetListInt(op_desc, kAutoCtxIdList, auto_ctx_id_list) && auto_ctx_id_list.size() ==
        sub_ffts_plus_context.size()) {
      ffts_plus_ctx_def->set_context_id(auto_ctx_id_list[i]);
    }
    ffts_plus_ctx_def->set_op_index(op_desc->GetId());

    FFTS_LOGD("GenContextDef nodetype:%s, name:%s, context_type:%u, op_index:%u, windowsize:%u",
              node->GetType().c_str(), node->GetName().c_str(), ffts_plus_ctx_def->context_type(),
              ffts_plus_ctx_def->op_index(), slice_info_ptr->parallel_window_size);
    domi::FftsPlusAicAivCtxDef *aic_aiv_ctx_def = ffts_plus_ctx_def->mutable_aic_aiv_ctx();
    FFTS_CHECK_NOTNULL(aic_aiv_ctx_def);
    FillContextData(aicore_ctx_def, aic_aiv_ctx_def);

    /* GE fill next node context's base_addr need this
     * node(a) has windown size context_a, generate context_a has same base_addr_a.
     * next continue node(b) has windown size context_b,
     * context_b's base_ddr_b = base_addr_a + a_size(memory for context_a)
     * FE set save_task_addr at node(a) last context for GE fill contexts(generate by b)'s base_addr_b */
    if (i == sub_ffts_plus_context.size() - 1) {
      aic_aiv_ctx_def->set_save_task_addr(1);
    } else {
      aic_aiv_ctx_def->set_save_task_addr(0);
    }

    aic_aiv_ctx_def->set_thread_id(i);
    aic_aiv_ctx_def->set_pred_cnt(sub_ffts_plus_context[i].pred_cnt);
    aic_aiv_ctx_def->set_pred_cnt_init(sub_ffts_plus_context[i].pred_cnt);
    aic_aiv_ctx_def->set_successor_num(0);
    aic_aiv_ctx_def->set_thread_window_size(slice_info_ptr->parallel_window_size); // not used yet

    addr_size = aic_aiv_ctx_def->task_addr_size();
    thread_dim = aic_aiv_ctx_def->thread_dim();
  }

  /* cur_addr_size: total context addr size in sqe
   * addr_size: single thread addr_size
   * GE memory request size is the size of all threads, this addr_size for GE */
  uint32_t cur_addr_size = ffts_plus_task_def->addr_size();
  ffts_plus_task_def->set_addr_size(cur_addr_size + addr_size * thread_dim);

  (void)ge::AttrUtils::SetInt(node->GetOpDesc(), kFFTSThreadDim, thread_dim);
  (void)ge::AttrUtils::SetInt(node->GetOpDesc(), kFFTSWindowSize, window_size);
  FFTS_LOGD("GenContextDef nodetype:%s, name:%s, total_addr_size:%u", node->GetType().c_str(),
            node->GetName().c_str(), ffts_plus_task_def->addr_size());
  return SUCCESS;
}
}  // namespace ffts
