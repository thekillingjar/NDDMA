/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "mix_aic_aiv_dynamic_task_builder.h"
#include "inc/ffts_utils.h"
#include "graph/debug/ge_attr_define.h"

namespace ffts {
static const vector<std::string> kMixPrefixs = { "_mix_aic", "_mix_aiv" };

MixAICAIVDynamicTaskBuilder::MixAICAIVDynamicTaskBuilder() {}

MixAICAIVDynamicTaskBuilder::~MixAICAIVDynamicTaskBuilder() {}

Status MixAICAIVDynamicTaskBuilder::GenContextDef(const ge::NodePtr &node, domi::FftsPlusTaskDef *ffts_plus_task_def) {
  FFTS_LOGD("Generate dynamic mix begin, node name:%s, node type:%s.", node->GetName().c_str(),
            node->GetType().c_str());
  auto op_desc = node->GetOpDesc();
  vector<FftsPlusComCtx_t> sub_ffts_plus_context;
  GenFftsPlusTaskCommonInfo(node, sub_ffts_plus_context);
  FftsPlusCtxDefPtr ctx_def_ptr = nullptr;
  ctx_def_ptr = op_desc->TryGetExtAttr(kAttrAICoreCtxDef, ctx_def_ptr);
  FFTS_CHECK_NOTNULL(ctx_def_ptr);
  domi::FftsPlusMixAicAivCtxDef *aicore_ctx_def = ctx_def_ptr->mutable_mix_aic_aiv_ctx();
  FFTS_CHECK_NOTNULL(aicore_ctx_def);
  string core_type;
  rtFftsPlusContextType_t ctx_type = RT_CTX_TYPE_MIX_AIC;
  (void)ge::AttrUtils::GetStr(op_desc, ge::ATTR_NAME_CUBE_VECTOR_CORE_TYPE, core_type);
  if (core_type == kCoreTypeMixAIV) {
    ctx_type = RT_CTX_TYPE_MIX_AIV;
  }
  vector<uint32_t> auto_ctx_id_list;
  (void)ge::AttrUtils::GetListInt(op_desc, kAutoCtxIdList, auto_ctx_id_list);
  if (auto_ctx_id_list.size() != sub_ffts_plus_context.size()) {
    FFTS_LOGE("Mix context size [%zu] does not equal %zu.", auto_ctx_id_list.size(), sub_ffts_plus_context.size());
    return FAILED;
  }
  for (size_t i = 0; i < sub_ffts_plus_context.size(); ++i) {
    domi::FftsPlusCtxDef *ffts_plus_ctx_def = ffts_plus_task_def->add_ffts_plus_ctx();
    FFTS_CHECK_NOTNULL(ffts_plus_ctx_def);
    ffts_plus_ctx_def->set_context_type(ctx_type);
    ffts_plus_ctx_def->set_context_id(auto_ctx_id_list[i]);
    ffts_plus_ctx_def->set_op_index(op_desc->GetId());
    FFTS_LOGD("Generated dynamic mix for context_id:%u with op_index:%ld.", auto_ctx_id_list[i], op_desc->GetId());
    domi::FftsPlusMixAicAivCtxDef *mix_aic_aiv_ctx_def = ffts_plus_ctx_def->mutable_mix_aic_aiv_ctx();
    FFTS_CHECK_NOTNULL(mix_aic_aiv_ctx_def);
    FillContextData(aicore_ctx_def, mix_aic_aiv_ctx_def);
    mix_aic_aiv_ctx_def->set_thread_id(i);
    mix_aic_aiv_ctx_def->set_pred_cnt(sub_ffts_plus_context[i].pred_cnt);
    mix_aic_aiv_ctx_def->set_pred_cnt_init(sub_ffts_plus_context[i].pred_cnt);
    mix_aic_aiv_ctx_def->set_successor_num(0);
  }
  if (AddAdditionalArgs(op_desc, ffts_plus_task_def, auto_ctx_id_list) != SUCCESS) {
    REPORT_FFTS_ERROR("[MixAICAIVTaskBuilder] [AddAdditionalArgs] Failed to add additional args for node [%s].",
                      op_desc->GetName().c_str());
    return FAILED;
  }
  return SUCCESS;
}

Status MixAICAIVDynamicTaskBuilder::AddAdditionalArgs(ge::OpDescPtr &op_desc,
                                                      domi::FftsPlusTaskDef *ffts_plus_task_def,
                                                      vector<uint32_t> &auto_ctx_id_list) const {
  FFTS_CHECK_NOTNULL(op_desc);
  uint32_t mode = 0;
  uint32_t data_type = 0;
  (void) ge::AttrUtils::GetInt(op_desc, kModeInArgsFirstField, mode);
  bool inter_core_sync = false;
  (void)ge::AttrUtils::GetBool(op_desc, kAttrIntercoreSync, inter_core_sync);
  FFTS_LOGD("Mix node [%s] mode: %u, sync flag: %d.", op_desc->GetName().c_str(), mode, inter_core_sync);
  if (mode == 1 || inter_core_sync) {
    domi::AdditionalDataDef *additional_data_def = ffts_plus_task_def->add_additional_data();
    FFTS_CHECK_NOTNULL(additional_data_def);
    additional_data_def->set_data_type(data_type);
    if (auto_ctx_id_list.empty()) {
      FFTS_LOGW("Mix ctx id list is empty.");
      return FAILED;
    }
    additional_data_def->add_context_id(auto_ctx_id_list[0]);
  }
  return SUCCESS;
}
}  // namespace ffts
