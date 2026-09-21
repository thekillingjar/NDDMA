/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "cache_persistent_manual_task_builder.h"
#include "common/sgt_slice_type.h"
#include "inc/ffts_utils.h"
#include "inc/ffts_type.h"

namespace ffts {
const int64_t kBytesInMB = 1048576;
inline int64_t DivisionCeiling(int64_t dividend, int64_t divisor) {
  if (divisor == 0) {
    return 0;
  } else {
    int64_t tmp_divisor = divisor - 1;
    if (CheckInt64AddOverflow(dividend, tmp_divisor) == SUCCESS) {
      dividend = dividend + tmp_divisor;
    }
    return dividend / divisor;
  }
}

Status CachePersistTaskBuilder::GenContextDef(const ge::NodePtr &node, domi::FftsPlusTaskDef *ffts_plus_task_def) {
  FFTS_CHECK_NOTNULL(node);
  return GenContextDef(*node.get(), ffts_plus_task_def);
}

Status CachePersistTaskBuilder::GenContextDef(const ge::Node &node, domi::FftsPlusTaskDef *ffts_plus_task_def) const {
  FFTS_CHECK_NOTNULL(ffts_plus_task_def);
  ge::OpDescPtr op_desc = node.GetOpDesc();
  uint32_t persist_id;
  if (!ge::AttrUtils::GetInt(op_desc, kCachePersist, persist_id) ||
      persist_id > kMaxPersistNum) {
    return FAILED;
  }

  FFTS_LOGD("Graph '%s' needs to perform cache persistence with ID %u.", node.GetName().c_str(), persist_id);

  domi::FftsPlusCtxDef *ffts_plus_ctx_def = ffts_plus_task_def->add_ffts_plus_ctx();
  FFTS_CHECK_NOTNULL(ffts_plus_ctx_def);
  ffts_plus_ctx_def->set_context_type(RT_CTX_TYPE_PERSISTENT_CACHE);
  domi::FftsPlusCachePersistCtxDef *cp_ctx_def = ffts_plus_ctx_def->mutable_cache_persist_ctx();
  FFTS_CHECK_NOTNULL(cp_ctx_def);

  cp_ctx_def->set_pred_cnt(0);
  cp_ctx_def->set_pred_cnt_init(0);
  cp_ctx_def->set_aten(0);

  cp_ctx_def->set_persistent_en(1);
  cp_ctx_def->set_persistent_id(persist_id);

  int64_t cache_persist_size = 0;
  (void)ge::AttrUtils::GetInt(op_desc, kCachePersistSize, cache_persist_size);
  int64_t size_in_mb = DivisionCeiling(cache_persist_size, kBytesInMB);
  /* size_in_mb should not larger than the maximum of uint16.
   * Because in the real context, we only reserve 16 bits
   * for cache_persist_size. */
  if (size_in_mb > UINT16_MAX) {
    REPORT_FFTS_ERROR("[CachePersistTaskBuilder][GenContextDef] Persist size %ld is larger than"
                      "the max of uint16(65536).", size_in_mb);
    return FAILED;
  }
  FFTS_LOGD("Persist id and size for graph %s are %u and %ld.", op_desc->GetName().c_str(), persist_id,
            cache_persist_size);
  cp_ctx_def->set_persistent_size(static_cast<uint32_t>(size_in_mb));
  return SUCCESS;
}
}
