/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aic_aiv_dynamic_task_builder.h"
#include <securec.h>
#include <string>
#include "inc/ffts_log.h"
#include "inc/ffts_utils.h"
#include "inc/ffts_type.h"
#include "inc/ffts_error_codes.h"
#include "common/sgt_slice_type.h"
#include "framework/common/ge_types.h"
#include "graph/utils/node_utils.h"
#include "graph/debug/ge_attr_define.h"
#include "rt_error_codes.h"
#include "runtime/rt_model.h"
#include "runtime/mem.h"

namespace ffts {
AICAIVDynamicTaskBuilder::AICAIVDynamicTaskBuilder() {}

AICAIVDynamicTaskBuilder::~AICAIVDynamicTaskBuilder() {}

Status AICAIVDynamicTaskBuilder::GenContextDef(const ge::NodePtr &node, domi::FftsPlusTaskDef *ffts_plus_task_def) {
  FFTS_LOGD("AIC AIV dynamic task generate begin, node:%s, type:%s.", node->GetName().c_str(), node->GetType().c_str());
  auto op_desc = node->GetOpDesc();
  vector<FftsPlusComCtx_t> sub_ffts_plus_context;
  GenFftsPlusTaskCommonInfo(node, sub_ffts_plus_context);
  FftsPlusCtxDefPtr ctx_def_ptr = nullptr;
  ctx_def_ptr = op_desc->TryGetExtAttr(kAttrAICoreCtxDef, ctx_def_ptr);
  FFTS_CHECK_NOTNULL(ctx_def_ptr);
  domi::FftsPlusAicAivCtxDef *aicore_ctx_def = ctx_def_ptr->mutable_aic_aiv_ctx();
  FFTS_CHECK_NOTNULL(aicore_ctx_def);
  vector<uint32_t> auto_ctx_id_list;
  (void)ge::AttrUtils::GetListInt(op_desc, kAutoCtxIdList, auto_ctx_id_list);
  if (auto_ctx_id_list.size() != sub_ffts_plus_context.size()) {
    FFTS_LOGE("AICAIV Context Id size: %zu does not match context num: %zu.", auto_ctx_id_list.size(),
              sub_ffts_plus_context.size());
    return FAILED;
  }
  std::string core_type;
  (void)ge::AttrUtils::GetStr(op_desc, ge::ATTR_NAME_CUBE_VECTOR_CORE_TYPE, core_type);
  if (core_type.empty()) {
    return FAILED;
  }
  for (size_t i = 0; i < sub_ffts_plus_context.size(); ++i) {
    domi::FftsPlusCtxDef *ffts_plus_ctx_def = ffts_plus_task_def->add_ffts_plus_ctx();
    FFTS_CHECK_NOTNULL(ffts_plus_ctx_def);
    if (core_type == kCoreTypeAIC) {
      ffts_plus_ctx_def->set_context_type(RT_CTX_TYPE_AICORE);
    } else {
      ffts_plus_ctx_def->set_context_type(RT_CTX_TYPE_AIV);
    }
    ffts_plus_ctx_def->set_op_index(op_desc->GetId());
    ffts_plus_ctx_def->set_context_id(auto_ctx_id_list[i]);
    FFTS_LOGD("Gen dynamic mode type:%s, name:%s, context_type:%u, op_index:%u", node->GetType().c_str(),
              node->GetName().c_str(), ffts_plus_ctx_def->context_type(), ffts_plus_ctx_def->op_index());
    domi::FftsPlusAicAivCtxDef *aic_aiv_ctx_def = ffts_plus_ctx_def->mutable_aic_aiv_ctx();
    FFTS_CHECK_NOTNULL(aic_aiv_ctx_def);
    FillContextData(aicore_ctx_def, aic_aiv_ctx_def);
    if (i == sub_ffts_plus_context.size() - 1) {
      aic_aiv_ctx_def->set_save_task_addr(1);
    } else {
      aic_aiv_ctx_def->set_save_task_addr(0);
    }
    aic_aiv_ctx_def->set_pred_cnt(sub_ffts_plus_context[i].pred_cnt);
    aic_aiv_ctx_def->set_pred_cnt_init(sub_ffts_plus_context[i].pred_cnt);
    aic_aiv_ctx_def->set_successor_num(0);
    aic_aiv_ctx_def->set_thread_id(i);
  }
  FFTS_LOGD("GenContextDef nodetype:%s, name:%s.", node->GetType().c_str(), node->GetName().c_str());
  return SUCCESS;
}
}
