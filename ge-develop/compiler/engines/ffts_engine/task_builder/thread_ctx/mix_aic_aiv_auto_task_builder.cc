/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "mix_aic_aiv_auto_task_builder.h"
#include "inc/ffts_utils.h"

namespace ffts {
static const vector<std::string> kMixPrefixs = { "_mix_aic", "_mix_aiv" };

MixAICAIVAutoTaskBuilder::MixAICAIVAutoTaskBuilder() {}

MixAICAIVAutoTaskBuilder::~MixAICAIVAutoTaskBuilder() {}

Status MixAICAIVAutoTaskBuilder::GenContextDef(const ge::NodePtr &node, domi::FftsPlusTaskDef *ffts_plus_task_def) {
  auto op_desc = node->GetOpDesc();
  vector<FftsPlusComCtx_t> sub_ffts_plus_context;
  GenFftsPlusTaskCommonInfo(node, sub_ffts_plus_context);
  FftsPlusCtxDefPtr ctx_def_ptr = nullptr;
  ctx_def_ptr = op_desc->TryGetExtAttr(kAttrAICoreCtxDef, ctx_def_ptr);
  FFTS_CHECK_NOTNULL(ctx_def_ptr);
  domi::FftsPlusMixAicAivCtxDef *aicore_ctx_def = ctx_def_ptr->mutable_mix_aic_aiv_ctx();
  FFTS_CHECK_NOTNULL(aicore_ctx_def);
  uint32_t addr_size = 0;
  uint32_t thread_dim = 0;
  for (size_t i = 0; i < sub_ffts_plus_context.size(); ++i) {
    domi::FftsPlusCtxDef *ffts_plus_ctx_def = ffts_plus_task_def->add_ffts_plus_ctx();
    FFTS_CHECK_NOTNULL(ffts_plus_ctx_def);

    vector<string> thread_core_type;
    (void)ge::AttrUtils::GetListStr(op_desc, kAttrThreadCoreType, thread_core_type);
    FFTS_CHECK(thread_core_type.empty(), FFTS_LOGW("thread_core_type is empty."), return FAILED);
    uint32_t context_type = (thread_core_type[0] == kCoreTypeMixAIC) ? RT_CTX_TYPE_MIX_AIC : RT_CTX_TYPE_MIX_AIV;
    ffts_plus_ctx_def->set_context_type(context_type);

    vector<uint32_t> auto_ctx_id_list;
    if (ge::AttrUtils::GetListInt(op_desc, kAutoCtxIdList, auto_ctx_id_list) &&
        auto_ctx_id_list.size() == sub_ffts_plus_context.size()) {
      ffts_plus_ctx_def->set_context_id(auto_ctx_id_list[i]);
    }
    ffts_plus_ctx_def->set_op_index(op_desc->GetId());

    FFTS_LOGD("GenContextDef nodetype:%s, name:%s, context_type:%u, op_index:%u", node->GetType().c_str(),
              node->GetName().c_str(), ffts_plus_ctx_def->context_type(), ffts_plus_ctx_def->op_index());

    domi::FftsPlusMixAicAivCtxDef *mix_aic_aiv_ctx_def = ffts_plus_ctx_def->mutable_mix_aic_aiv_ctx();
    FFTS_CHECK_NOTNULL(mix_aic_aiv_ctx_def);
    FillContextData(aicore_ctx_def, mix_aic_aiv_ctx_def);

    /* GE fill next node context's base_addr need this
     * node(a) has windown size context_a, generate context_a has same base_addr_a.
     * next continue node(b) has windown size context_b,
     * context_b's base_ddr_b = base_addr_a + a_size(memory for context_a)
     * FE set save_task_addr at node(a) last context for GE fill contexts(generate by b)'s base_addr_b */
    uint32_t save_task_addr = (i == sub_ffts_plus_context.size() - 1) ? 1 : 0;
    mix_aic_aiv_ctx_def->set_save_task_addr(save_task_addr);

    mix_aic_aiv_ctx_def->set_thread_id(i);
    mix_aic_aiv_ctx_def->set_pred_cnt(sub_ffts_plus_context[i].pred_cnt);
    mix_aic_aiv_ctx_def->set_pred_cnt_init(sub_ffts_plus_context[i].pred_cnt);
    mix_aic_aiv_ctx_def->set_successor_num(0);
    addr_size = mix_aic_aiv_ctx_def->task_addr_size();
    thread_dim = mix_aic_aiv_ctx_def->thread_dim();

    if (AddAdditionalArgs(op_desc, ffts_plus_task_def, 1) != SUCCESS) {
      REPORT_FFTS_ERROR("[MixAICAIVTaskBuilder] Node [%s] failed to add additional arguments.", node->GetNamePtr());
      return FAILED;
    }
  }

  /* cur_addr_size: total context addr size in sqe
   * addr_size: single thread addr_size
   * GE memory request size is the size of all threads, this addr_size for GE */
  ffts_plus_task_def->set_addr_size(ffts_plus_task_def->addr_size() + addr_size * thread_dim);
  return SUCCESS;
}

Status MixAICAIVAutoTaskBuilder::AddAdditionalArgs(ge::OpDescPtr &op_desc, domi::FftsPlusTaskDef *ffts_plus_task_def,
                                                   const size_t &ctx_num) const {
  FFTS_CHECK_NOTNULL(op_desc);
  // modeInArgsFirstField
  uint32_t mode = 0;
  uint32_t data_type = 0;
  (void) ge::AttrUtils::GetInt(op_desc, kModeInArgsFirstField, mode);
  bool inter_core_sync = false;
  (void)ge::AttrUtils::GetBool(op_desc, kAttrIntercoreSync, inter_core_sync);
  if (mode == 1 || inter_core_sync) {
    domi::AdditionalDataDef *additional_data_def = ffts_plus_task_def->add_additional_data();
    FFTS_CHECK_NOTNULL(additional_data_def);
    uint32_t context_id = 0;
    (void) ge::AttrUtils::GetInt(op_desc, kContextId, context_id);
    additional_data_def->set_data_type(data_type);
    additional_data_def->add_context_id(context_id);
    if (ctx_num > 1) {
      (void) ge::AttrUtils::GetInt(op_desc, "_default_context_id", context_id);
      additional_data_def->add_context_id(context_id);
    }
  }
  return SUCCESS;
}
}  // namespace ffts
