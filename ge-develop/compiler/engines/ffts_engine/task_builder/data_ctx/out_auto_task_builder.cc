/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "out_auto_task_builder.h"

namespace ffts {
OutAutoTaskBuilder::OutAutoTaskBuilder() {}

OutAutoTaskBuilder::OutAutoTaskBuilder(CACHE_OPERATION operation)
    : DataTaskBuilder(operation) {}

OutAutoTaskBuilder::~OutAutoTaskBuilder() {}

Status OutAutoTaskBuilder::UptSuccListOfRelatedNode(const ge::NodePtr &node, const std::vector<uint32_t> &succ_list,
                                                    const size_t &window_id,
                                                    int32_t &data_ctx_id,
                                                    domi::FftsPlusTaskDef *ffts_plus_task_def) const {
  /* If original context size is 9 and after generate this out data context,
   * the total size is 10. So the context id of this out data context is 10 - 1 = 9; */
  data_ctx_id = ffts_plus_task_def->ffts_plus_ctx_size() - 1;
  FFTS_LOGD("Data context id is %d.", data_ctx_id);

  if (operation_ == CACHE_OPERATION::WRITE_BACK) {
    auto op_desc = node->GetOpDesc();
    vector<uint32_t> context_id_list;
    (void)ge::AttrUtils::GetListInt(op_desc, kAutoCtxIdList, context_id_list);
    if (context_id_list.size() > window_id) {
      UpdateSuccList(data_ctx_id, context_id_list[window_id], ffts_plus_task_def, window_id, true);
    }
  } else {
    for (auto target_id : succ_list) {
      UpdateSuccList(data_ctx_id, target_id + window_id, ffts_plus_task_def, window_id, true);
    }
  }
  return SUCCESS;
}

Status OutAutoTaskBuilder::FillAutoDataCtx(size_t out_anchor_index, const ge::NodePtr &node,
                                           const std::vector<DataContextParam> &params,
                                           domi::FftsPlusTaskDef *ffts_plus_task_def,
                                           domi::FftsPlusDataCtxDef *data_ctx_def, const size_t &window_id) {
  auto op_desc = node->GetOpDesc();
  vector<uint32_t> context_id_list;
  if (!ge::AttrUtils::GetListInt(op_desc, kAutoCtxIdList, context_id_list)) {
    REPORT_FFTS_ERROR("[GenTsk][DataTsk][FillAutoCtxt] Node [%s] cannot obtain the context ID list.", node->GetNamePtr());
    return FAILED;
  }
  FFTS_CHECK_NOTNULL(data_ctx_def);
  data_ctx_def->set_aten(kAutoMode);
  data_ctx_def->set_atm(kAutoMode);

  uint32_t cons_cnt = 0;
  std::vector<uint32_t> succ_list;
  if (operation_ == CACHE_OPERATION::WRITE_BACK) {
    /* In write back case, the write back data context can only be
     * executed when the only producer has already been executed. */
    data_ctx_def->set_cnt_init(1);
    data_ctx_def->set_cnt(1);
    FFTS_LOGD("Fill write back for node %s, consumer count is 1.", node->GetName().c_str());
  } else {
    /* In invalidate case, the invalidate data context can only be
     * executed when all the consumers have already been executed. */
    GetSuccessorContextId(out_anchor_index, node, succ_list, cons_cnt);
    data_ctx_def->set_cnt_init(cons_cnt);
    data_ctx_def->set_cnt(cons_cnt);
    data_ctx_def->set_orig_consumer_counter(cons_cnt);
    data_ctx_def->set_run_consumer_counter(cons_cnt);
    FFTS_LOGD("Invalidate fill for node %s, consumer count is %u.", node->GetName().c_str(), cons_cnt);
  }
  int32_t data_ctx_id = -1;
  UptSuccListOfRelatedNode(node, succ_list, window_id, data_ctx_id, ffts_plus_task_def);

  vector<int64_t> output_addrs;
  if (!ge::AttrUtils::GetListInt(op_desc, "output_addrs", output_addrs)) {
    REPORT_FFTS_ERROR("[GenTsk][DataTsk][FillAutoCtxt] Node [%s] attribute 'output_addrs' is empty.", node->GetNamePtr());
    return FAILED;
  }

  if (out_anchor_index >= output_addrs.size()) {
    REPORT_FFTS_ERROR("[GenTsk][DataTsk][FillAutoCtxt]Node[%s] out anchor %zu must be smaller than the size of output_addrs %zu.",
                      node->GetNamePtr(), out_anchor_index, output_addrs.size());
    return FAILED;
  }
  data_ctx_def->set_addr_base(output_addrs[out_anchor_index]);

  ThreadSliceMapPtr slice_info_ptr = nullptr;
  slice_info_ptr = op_desc->TryGetExtAttr(kAttrSgtStructInfo, slice_info_ptr);
  FFTS_CHECK_NOTNULL(slice_info_ptr);
  data_ctx_def->set_thread_dim(slice_info_ptr->slice_instance_num);
  data_ctx_def->set_thread_id(window_id);

  if (operation_ == CACHE_OPERATION::INVALIDATE &&
      GenInvalidSuccListWithMemReuse(node, out_anchor_index, ffts_plus_task_def, data_ctx_id, window_id) != SUCCESS) {
    REPORT_FFTS_ERROR("[GenTsk][DataTsk][FillCtxt] Node [%s] failed to generate GenInvalidSuccList.", node->GetNamePtr());
    return FAILED;
  }
  FillAutoThreadingParam(params, data_ctx_def, slice_info_ptr->slice_instance_num);
  FFTS_LOGI("[Auto mode]Finish gen %d context for node %s.", static_cast<int>(operation_), node->GetName().c_str());
  return SUCCESS;
}

Status OutAutoTaskBuilder::UpdateInvalidCtxWithMemReuse(const ge::NodePtr &node, int &data_ctx_id,
                                                        const size_t &window_id,
                                                        domi::FftsPlusTaskDef *ffts_plus_task_def) {
  auto reuse_op_desc = node->GetOpDesc();
  vector<uint32_t> context_id_list;
  if (!ge::AttrUtils::GetListInt(reuse_op_desc, kAutoCtxIdList, context_id_list)) {
    FFTS_LOGD("[GenTsk][DataTsk][FillCtxt][node %s, type %s] has no context id list.",
              node->GetName().c_str(), node->GetType().c_str());
    return FAILED;
  }

  if (context_id_list.size() > window_id) {
    UpdateSuccList(context_id_list[window_id], data_ctx_id, ffts_plus_task_def, window_id, true);
    UpdatePreCnt(context_id_list[window_id], ffts_plus_task_def, 1);
  }
  return SUCCESS;
}
}
