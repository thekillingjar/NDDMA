/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef INC_REGISTER_FFTS_PLUS_ENGINE_UPDATE_H_
#define INC_REGISTER_FFTS_PLUS_ENGINE_UPDATE_H_
#include "graph/utils/graph_utils.h"
#include "graph/utils/tensor_utils.h"
#include "runtime/rt_ffts_plus.h"
#include "common/sgt_slice_type.h"
namespace ffts {
class FFTSPlusEngineUpdate {
public:
  FFTSPlusEngineUpdate();
  ~FFTSPlusEngineUpdate();
  static bool UpdateCommonCtx(ge::ComputeGraphPtr &sgt_graph, rtFftsPlusTaskInfo_t &task_info);
  static ThreadSliceMapDyPtr slice_info_ptr_;
};
};
#endif  // INC_REGISTER_FFTS_PLUS_ENGINE_UPDATE_H_
