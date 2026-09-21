/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "out_dynamic_task_builder.h"

namespace ffts {
const size_t kDynamicWindowSize = 4;

OutDynamicTaskBuilder::OutDynamicTaskBuilder() {}

OutDynamicTaskBuilder::OutDynamicTaskBuilder(CACHE_OPERATION operation)
    : DataTaskBuilder(operation) {}

OutDynamicTaskBuilder::~OutDynamicTaskBuilder() {}

void OutDynamicTaskBuilder::RecordNewCtxIdList(vector<uint32_t> &data_ctx_id_list,
                                               vector<vector<int64_t>> &write_back_invalid_ctx_id_list,
                                               domi::FftsPlusTaskDef *ffts_plus_task_def,
                                               const rtFftsPlusContextType_t &context_type,
                                               uint32_t const_cnt) const {
  for (size_t i = 0; i < kDynamicWindowSize; i++) {
    domi::FftsPlusCtxDef *ffts_plus_ctx_def = ffts_plus_task_def->add_ffts_plus_ctx();
    FFTS_CHECK(ffts_plus_ctx_def == nullptr, FFTS_LOGW("ffts_plus_ctx_def is nullptr"), return);
    ffts_plus_ctx_def->set_context_type(context_type);
    domi::FftsPlusDataCtxDef *data_ctx_def = ffts_plus_ctx_def->mutable_data_ctx();
    FFTS_CHECK(data_ctx_def == nullptr, FFTS_LOGW("data_ctx_def is a nullptr"), return);
    data_ctx_def->set_aten(kAutoMode);
    data_ctx_def->set_atm(kAutoMode);
    data_ctx_def->set_cnt(const_cnt);
    data_ctx_def->set_cnt_init(const_cnt);
    data_ctx_def->set_orig_consumer_counter(const_cnt);
    data_ctx_def->set_run_consumer_counter(const_cnt);
    data_ctx_def->set_thread_id(i);
    FFTS_LOGD("Set data const count:%u", const_cnt);
    vector<int64_t> pre_ctx_id = write_back_invalid_ctx_id_list[i];
    pre_ctx_id.push_back(ffts_plus_task_def->ffts_plus_ctx_size() - 1);
    data_ctx_id_list.push_back(ffts_plus_task_def->ffts_plus_ctx_size() - 1);
    write_back_invalid_ctx_id_list[i] = pre_ctx_id;
  }
}

Status OutDynamicTaskBuilder::FillDynamicDataCtx(const size_t &out_anchor_index, const ge::NodePtr &node,
                                                 domi::FftsPlusTaskDef *ffts_plus_task_def,
                                                 const rtFftsPlusContextType_t &context_type,
                                                 const vector<uint32_t> &context_id_list) {
  FFTS_LOGD("node[%s] starts to fill unknown shape write_back with invalid data context.", node->GetName().c_str());
  if (operation_ == CACHE_OPERATION::WRITE_BACK) {
    vector<vector<int64_t>> write_back_ctx_id_list(kDynamicWindowSize, vector<int64_t>(0, 0));
    (void)ge::AttrUtils::GetListListInt(node->GetOpDesc(), "_write_back_ctx_id_list", write_back_ctx_id_list);

    vector<uint32_t> data_ctx_id_list;
    RecordNewCtxIdList(data_ctx_id_list, write_back_ctx_id_list, ffts_plus_task_def, context_type, 1);

    if (context_id_list.size() != kDynamicWindowSize) {
      REPORT_FFTS_ERROR("[DataTaskBuilder][GenContextDef][FillDynamicInvalidAndWriteBackCtx] Node: %s context_id_list "
                        "size is not equal to kDynamicWindowSize: %zu.", node->GetName().c_str(), kDynamicWindowSize);
      return FAILED;
    }

    for (size_t i = 0; i < kDynamicWindowSize; i++) {
      UpdateSuccList(data_ctx_id_list[i], context_id_list[i], ffts_plus_task_def, i, true);
    }

    (void)ge::AttrUtils::SetListListInt(node->GetOpDesc(), "_write_back_ctx_id_list", write_back_ctx_id_list);
  } else {
    // generate succ_list
    vector<uint32_t> succ_list;
    uint32_t const_cnt = 0;
    GetSuccessorContextId(out_anchor_index, node, succ_list, const_cnt);
    vector<vector<int64_t>> invalid_ctx_id_list(kDynamicWindowSize, vector<int64_t>(0, 0));
    (void)ge::AttrUtils::GetListListInt(node->GetOpDesc(), "_invalid_ctx_id_list", invalid_ctx_id_list);
    vector<uint32_t> data_ctx_id_list;
    RecordNewCtxIdList(data_ctx_id_list, invalid_ctx_id_list, ffts_plus_task_def, context_type, const_cnt);
    for (size_t i = 0; i < kDynamicWindowSize; i++) {
      for (size_t j = 0; j < succ_list.size(); j++) {
        UpdateSuccList(data_ctx_id_list[i], succ_list[j] + i, ffts_plus_task_def, i, true);
      }
    }

    (void)ge::AttrUtils::SetListListInt(node->GetOpDesc(), "_invalid_ctx_id_list", invalid_ctx_id_list);
  }
  return SUCCESS;
}
}
