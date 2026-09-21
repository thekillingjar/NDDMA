/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "prefetch_dynamic_task_builder.h"

namespace ffts {
PrefetchDynamicTaskBuilder::PrefetchDynamicTaskBuilder()
    : DataTaskBuilder(CACHE_OPERATION::PREFETCH) {}

PrefetchDynamicTaskBuilder::~PrefetchDynamicTaskBuilder() {}

Status PrefetchDynamicTaskBuilder::FillDynamicDataCtx(const size_t &in_anchor_index, const ge::NodePtr &node,
                                                      domi::FftsPlusTaskDef *ffts_plus_task_def,
                                                      const rtFftsPlusContextType_t &context_type,
                                                      const vector<uint32_t> &context_id_list) {
  (void)in_anchor_index;
  FFTS_LOGD("Node [%s] starts to fill unknown shape prefetch data context.", node->GetName().c_str());
  std::vector<std::vector<int64_t>> data_prefetch_ctx_id_list;
  (void)ge::AttrUtils::GetListListInt(node->GetOpDesc(), "_data_prefetch_ctx_id_list", data_prefetch_ctx_id_list);
  for (size_t i = 0; i < context_id_list.size(); i++) {
    domi::FftsPlusCtxDef *ffts_plus_ctx_def = ffts_plus_task_def->add_ffts_plus_ctx();
    FFTS_CHECK_NOTNULL(ffts_plus_ctx_def);
    ffts_plus_ctx_def->set_context_type(context_type);
    domi::FftsPlusDataCtxDef *data_ctx_def = ffts_plus_ctx_def->mutable_data_ctx();
    FFTS_CHECK_NOTNULL(data_ctx_def);
    data_ctx_def->set_aten(kAutoMode);
    data_ctx_def->set_atm(kAutoMode);
    data_ctx_def->set_cnt_init(1);
    data_ctx_def->set_cnt(1);
    data_ctx_def->set_orig_consumer_counter(1);
    data_ctx_def->set_run_consumer_counter(1);
    data_ctx_def->set_thread_id(i);
    UpdateSrcSlotAndPfBm(ffts_plus_task_def, context_id_list[i]);

    if (data_prefetch_ctx_id_list.size() != context_id_list.size()) {
      vector<int64_t> data_prefetch_ctx_id = {ffts_plus_task_def->ffts_plus_ctx_size() - 1};
      data_prefetch_ctx_id_list.push_back(data_prefetch_ctx_id);
    } else {
      vector<int64_t> data_prefetch_ctx_id = data_prefetch_ctx_id_list[i];
      data_prefetch_ctx_id.push_back(ffts_plus_task_def->ffts_plus_ctx_size() - 1);
      data_prefetch_ctx_id_list[i] = data_prefetch_ctx_id;
    }
  }
  (void)ge::AttrUtils::SetListListInt(node->GetOpDesc(), "_data_prefetch_ctx_id_list", data_prefetch_ctx_id_list);
  return SUCCESS;
}
}
