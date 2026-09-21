/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "format_selector/builtin/broadcast/format_process/broadcast_enhanced_process_5hd.h"
#include "common/fe_op_info_common.h"

namespace fe {
bool BroadcastEnhancedProcess5HD::CheckAllNonScalarInputs(const FormatProccessArgs &args) {
  std::vector<ge::GeShape> input_shapes = args.origin_info_ptr->input_shapes;
  // each shape should be 4D
  for (size_t i = 0; i < input_shapes.size(); i++) {
    if (!CheckOriginShapeDimNum(input_shapes[i], DIM_DEFAULT_SIZE)) {
      FE_LOGD("The dim_num of the input_shape[%zu] value[%s] is < %zu.",
              i, GetShapeDims(input_shapes[i]).c_str(),
              DIM_DEFAULT_SIZE);
      return false;
    }
  }
  return true;
}

REGISTER_FORMAT_PROCESS(BroadcastEnhancedProcess5HD, OP_PATTERN_BROADCAST_ENHANCED, FORMAT_NC1HWC0,
                        ge::FORMAT_NC1HWC0);
}  // namespace fe
