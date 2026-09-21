/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "format_selector/builtin/broadcast/format_process/broadcast_process_6hd.h"
#include "common/fe_op_info_common.h"

namespace fe {
bool BroadcastProcess6HD::CheckPartNonScalarInputs(const FormatProccessInputArg &input_arg) {
  auto input_format = input_arg.input_format_;
  auto input_dtype = input_arg.input_dtype_;
  auto input_shape = input_arg.input_shape_;
  // shape should be 5D
  if (CheckOriginShapeDimNum(input_shape, DIMENSION_NUM_FIVE)) {
    int64_t dim_value = 0;
    GetDimValue(C_AXIS_NAME, input_format, input_shape, dim_value);
    // axis C should be C0 aligned
    if (!IsAxisC0Aligned(input_dtype, dim_value)) {
      FE_LOGD("Axis C [%ld] is not 16 aligned, input_formats=[%d], inputDtype=[%d], inputShape=[%s].", dim_value,
              input_format, input_dtype, GetShapeDims(input_shape).c_str());
      return false;
    }
  } else {
    FE_LOGD("The dim_num of the input_shape=[%s] is < %u.", GetShapeDims(input_shape).c_str(), DIMENSION_NUM_FIVE);
    return false;
  }
  return true;
}

bool BroadcastProcess6HD::CheckAllNonScalarInputs(const FormatProccessArgs &args) {
  auto input_formats_b = args.origin_info_ptr->input_formats;
  auto input_shapes = args.origin_info_ptr->input_shapes;
  // each shape should be 5D
  for (size_t i = 0; i < input_shapes.size(); i++) {
    if (!CheckOriginShapeDimNum(input_shapes[i], DIMENSION_NUM_FIVE)) {
      FE_LOGD("The dim_num of the input_shape[%zu] value[%s] is < %u.", i, GetShapeDims(input_shapes[i]).c_str(),
              DIMENSION_NUM_FIVE);
      return false;
    }
  }

  // axis c should not need broadcast, if need, not support 6HD
  if (CheckAxisNeedBroadcast(C_AXIS_NAME, input_formats_b, input_shapes)) {
    FE_LOGD("Axis C needs to broadcast.");
    return false;
  }
  return true;
}

REGISTER_FORMAT_PROCESS(BroadcastProcess6HD, OP_PATTERN_BROADCAST, FORMAT_NDC1HWC0, ge::FORMAT_NDC1HWC0);
REGISTER_FORMAT_PROCESS(BroadcastProcess6HD, OP_PATTERN_BROADCAST_ENHANCED, FORMAT_NDC1HWC0, ge::FORMAT_NDC1HWC0);
}  // namespace fe
