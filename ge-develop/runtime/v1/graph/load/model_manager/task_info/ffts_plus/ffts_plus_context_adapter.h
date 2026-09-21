/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_FFTS_PLUS_CONTEXT_ADAPTER_H
#define AIR_CXX_FFTS_PLUS_CONTEXT_ADAPTER_H

#include "ge/ge_api_error_codes.h"

#include "runtime/rt.h"
#include "proto/task.pb.h"
#include "graph/load/model_manager/task_info/task_info.h"
#include "graph/load/model_manager/task_info/ffts_plus/ffts_plus_args_helper.h"
#include "graph/load/model_manager/davinci_model.h"
#include "hybrid/node_executor/aicpu/aicpu_ext_info_handler.h"
#include "common/op_tiling/op_tiling_rt2.h"

namespace ge {
class FftsPlusContextAdapter {
 public:
  // ge不感知字段解析
  static Status ParseCtxByType(rtFftsPlusContextType_t ctx_type, const domi::FftsPlusCtxDef &task_def,
                               rtFftsPlusComCtx_t *const com_ctx);

  static void InitFftsPlusSqe(const domi::FftsPlusSqeDef &sqe_def, rtFftsPlusSqe_t &sqe);

 private:
  static Status ParseAicAivCtx(const domi::FftsPlusCtxDef &task_def, rtFftsPlusComCtx_t *const com_ctx);

  static Status ParseMixAicAivCtx(const domi::FftsPlusCtxDef &task_def, rtFftsPlusComCtx_t *const com_ctx);

  static Status ParseAicpuCtx(const domi::FftsPlusCtxDef &task_def, rtFftsPlusComCtx_t *const com_ctx);

  static Status ParseDsaCtx(const domi::FftsPlusCtxDef &task_def, rtFftsPlusComCtx_t *const com_ctx);

  static Status ParseNotifyCtx(const domi::FftsPlusCtxDef &task_def, rtFftsPlusComCtx_t *const com_ctx);

  static Status ParsePersistentCacheCtx(const domi::FftsPlusCtxDef &task_def, rtFftsPlusComCtx_t *const com_ctx);

  static Status ParseDataCtx(const domi::FftsPlusCtxDef &task_def, rtFftsPlusComCtx_t *const com_ctx);

  static Status ParseSdmaCtx(const domi::FftsPlusCtxDef &task_def, rtFftsPlusComCtx_t *const com_ctx);

  static Status ParseCondSwitchCtx(const domi::FftsPlusCtxDef &task_def, rtFftsPlusComCtx_t *const com_ctx);

  static Status ParseCaseSwitchCtx(const domi::FftsPlusCtxDef &task_def, rtFftsPlusComCtx_t *const com_ctx);

  static Status ParseCaseDefaultCtx(const domi::FftsPlusCtxDef &task_def, rtFftsPlusComCtx_t *const com_ctx);

  static Status ParseCaseCtx(const domi::FftsPlusCtxDef &task_def, rtFftsPlusComCtx_t *const com_ctx);

  static Status ParseAtStartCtx(const domi::FftsPlusCtxDef &task_def, rtFftsPlusComCtx_t *const com_ctx);

  static Status ParseAtEndCtx(const domi::FftsPlusCtxDef &task_def, rtFftsPlusComCtx_t *const com_ctx);

  static Status ParseLabelCtx(const domi::FftsPlusCtxDef &task_def, rtFftsPlusComCtx_t *const com_ctx);

  static Status ParseWriteValueCtx(const domi::FftsPlusCtxDef &task_def, rtFftsPlusComCtx_t *const com_ctx);
};
}  // namespace ge

#endif  // AIR_CXX_FFTS_PLUS_CONTEXT_ADAPTER_H
