/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "prefetch_auto_task_builder.h"

namespace ffts {
PrefetchAutoTaskBuilder::PrefetchAutoTaskBuilder()
    : DataTaskBuilder(CACHE_OPERATION::PREFETCH) {}

PrefetchAutoTaskBuilder::~PrefetchAutoTaskBuilder() {}

Status PrefetchAutoTaskBuilder::FillAutoDataCtx(size_t in_anchor_index, const ge::NodePtr &node,
                                                const std::vector<DataContextParam> &params,
                                                domi::FftsPlusTaskDef *ffts_plus_task_def,
                                                domi::FftsPlusDataCtxDef *data_ctx_def, const size_t &window_id) {
  FFTS_LOGD("Fill prefetch context for node %s, input %zu.", node->GetName().c_str(), in_anchor_index);
  auto op_desc = node->GetOpDesc();
  vector<uint32_t> context_id_list;
  if (!ge::AttrUtils::GetListInt(op_desc, kAutoCtxIdList, context_id_list)) {
    REPORT_FFTS_ERROR("[GenTsk][PrefetchTsk][FillCtxt][node %s, type %s] Cannot get context id for this node.",
                      op_desc->GetName().c_str(), op_desc->GetType().c_str());
    return FAILED;
  }
  FFTS_CHECK_NOTNULL(data_ctx_def);
  data_ctx_def->set_aten(kAutoMode);
  data_ctx_def->set_atm(kAutoMode);

  /*
   * prefetch context does not need the dependency table and it
   * is executed by the hardware instead of the MCU.
   * We just init it with a non-zero number in case the MCU will
   * execute it immediately. */
  uint32_t cons_cnt = 1;
  data_ctx_def->set_cnt(cons_cnt);
  data_ctx_def->set_cnt_init(cons_cnt);
  /* Only do the prefetch when orig_consumer_counter is not equal to
   * the run_consumer_counter. */
  data_ctx_def->set_orig_consumer_counter(cons_cnt);
  data_ctx_def->set_run_consumer_counter(cons_cnt);

  /* when we do prefetch we need to get the peer output addr */
  uint64_t addr_base = 0;
  if (GetAddrBase(in_anchor_index, node, addr_base) != SUCCESS) {
    return FAILED;
  }
  data_ctx_def->set_addr_base(addr_base);

  if (params.empty()) {
    REPORT_FFTS_ERROR("[GenTsk][PrefetchTsk][FillCtxt][node %s, type %s] Parameters are empty.",
                      op_desc->GetName().c_str(), op_desc->GetType().c_str());
    return FAILED;
  }

  ThreadSliceMapPtr slice_info_ptr = nullptr;
  slice_info_ptr = op_desc->TryGetExtAttr(kAttrSgtStructInfo, slice_info_ptr);
  FFTS_CHECK_NOTNULL(slice_info_ptr);
  data_ctx_def->set_thread_id(window_id);
  data_ctx_def->set_thread_dim(slice_info_ptr->slice_instance_num);
  FillAutoThreadingParam(params, data_ctx_def, slice_info_ptr->slice_instance_num);
  if (context_id_list.size() <= window_id) {
    return FAILED;
  }
  return UpdateSrcSlotAndPfBm(ffts_plus_task_def, context_id_list[window_id]);
}

}
