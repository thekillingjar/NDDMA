/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "runtime_ops_auto_task_builder.h"
#include "inc/ffts_utils.h"
#include "common/opskernel/ops_kernel_info_types.h"
#include "runtime/base.h"
#include "common/resource_def.h"
namespace ffts {
RuntimeOpsAutoTaskBuilder::RuntimeOpsAutoTaskBuilder() {}
RuntimeOpsAutoTaskBuilder::~RuntimeOpsAutoTaskBuilder() {}
Status RuntimeOpsAutoTaskBuilder::GenContextDef(const ge::NodePtr &node, domi::FftsPlusTaskDef *ffts_plus_task_def) {
  FFTS_LOGD("Gen auto runtime context begin, node name:%s, node type:%s.", node->GetName().c_str(),
            node->GetType().c_str());
  auto op_desc = node->GetOpDesc();
  ThreadSliceMapPtr slice_info_ptr = nullptr;
  slice_info_ptr = node->GetOpDesc()->TryGetExtAttr(kAttrSgtStructInfo, slice_info_ptr);
  if (slice_info_ptr == nullptr) {
    FFTS_LOGD("Setting temporary auto-thread slicing information for node [%s].", node->GetName().c_str());
    FFTS_MAKE_SHARED(slice_info_ptr = std::make_shared<ThreadSliceMap>(), return FAILED);
    slice_info_ptr->thread_mode = kAutoMode;
    slice_info_ptr->parallel_window_size = kDefaultWindowSize;
    node->GetOpDesc()->SetExtAttr(kAttrSgtStructInfo, slice_info_ptr);
  }
  vector<FftsPlusComCtx_t> sub_ffts_plus_context;
  GenFftsPlusTaskCommonInfo(node, sub_ffts_plus_context);
  if (sub_ffts_plus_context.empty()) {
    FFTS_LOGE("FFTS plus context is empty.");
    return FAILED;
  }
  FftsPlusCtxDefPtr ctx_def_ptr =  nullptr;
  ctx_def_ptr = op_desc->TryGetExtAttr("FFTS_PLUS_TASK_DEF", ctx_def_ptr);
  FFTS_CHECK_NOTNULL(ctx_def_ptr);

  rtFftsPlusContextType_t context_type = RT_CTX_TYPE_AICORE;
  if (JudgeContextType(node, context_type) != SUCCESS) {
    return FAILED;
  }
  FFTS_LOGD("Judge node [%s] context type: %lx.", node->GetName().c_str(), context_type);
  vector<uint32_t> auto_ctx_id_list;
  ge::AttrUtils::GetListInt(op_desc, kAutoCtxIdList, auto_ctx_id_list);
  if (auto_ctx_id_list.size() != sub_ffts_plus_context.size()) {
    FFTS_LOGE("Context list size [%zu] is not equal to context size [%zu].", auto_ctx_id_list.size(),
              sub_ffts_plus_context.size());
    return FAILED;
  }
  for (size_t i = 0; i < sub_ffts_plus_context.size(); ++i) {
    domi::FftsPlusCtxDef *ffts_plus_ctx_def = ffts_plus_task_def->add_ffts_plus_ctx();
    FFTS_CHECK_NOTNULL(ffts_plus_ctx_def);
    ffts_plus_ctx_def->set_op_index(op_desc->GetId());
    ffts_plus_ctx_def->set_context_id(auto_ctx_id_list[i]);
    ffts_plus_ctx_def->set_context_type(context_type);
    Status status = FAILED;
    switch (context_type) {
      case RT_CTX_TYPE_CASE_SWITCH:
        status = FillCaseSwitchContext(node, ffts_plus_ctx_def, ctx_def_ptr, sub_ffts_plus_context, i);
        break;
      case RT_CTX_TYPE_LABEL:
        status = FillLabelContext(op_desc, ffts_plus_ctx_def, sub_ffts_plus_context, i);
        break;
      case RT_CTX_TYPE_SDMA:
        status = FillSdmaContext(op_desc, ffts_plus_ctx_def, ctx_def_ptr, sub_ffts_plus_context, i);
        break;
      case RT_CTX_TYPE_COND_SWITCH:
        status = FillCondSwitchContext(node, ffts_plus_ctx_def, ctx_def_ptr, sub_ffts_plus_context, i);
        break;
      default:
        status = FAILED;
        break;
    }
    if (status != SUCCESS) {
      FFTS_LOGE("GenContextTaskDef failed. Operation: [%s], Type: [%s].", op_desc->GetName().c_str(), op_desc->GetType().c_str());
      return status;
    }
  }
  return SUCCESS;
}
}  // namespace ffts
