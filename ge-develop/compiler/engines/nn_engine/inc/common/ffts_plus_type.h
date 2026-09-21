/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_COMPILER_GRAPHCOMPILER_ENGINES_NNENG_INC_COMMON_FFTS_PLUS_TYPE_H
#define AIR_COMPILER_GRAPHCOMPILER_ENGINES_NNENG_INC_COMMON_FFTS_PLUS_TYPE_H

#include "common/sgt_slice_type.h"

namespace fe {
inline bool OpIsAutoThread(ffts::ThreadSliceMapPtr slice_info_ptr)
{
  if ((slice_info_ptr != nullptr) && (slice_info_ptr->thread_mode == 1)
      && (!slice_info_ptr->input_tensor_slice.empty())) {
    return true;
  }
  return false;
}

struct TickCacheMap {
  std::vector<int32_t> src_out_of_graph_input_index;
  std::map<int32_t, uint8_t> input_cache_table;
  std::map<int32_t, uint8_t> output_cache_table;
};

enum class CACHE_OPERATION {
  PREFETCH = 0,
  INVALIDATE = 1,
  WRITE_BACK = 2,
  CACHE_OPERATION_BOTTOM = 3
};
}
#endif // AIR_COMPILER_GRAPHCOMPILER_ENGINES_NNENG_INC_COMMON_FFTS_PLUS_TYPE_H
