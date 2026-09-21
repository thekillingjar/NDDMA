/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "format_selector/builtin/broadcast/format_process/broadcast_process_6d.h"
#include "common/fe_op_info_common.h"

namespace fe {
bool BroadcastProcess6D::CheckPartNonScalarInputs(const FormatProccessInputArg &input_arg) {
  auto input_format = input_arg.input_format_;
  auto input_dtype = input_arg.input_dtype_;
  auto input_shape = input_arg.input_shape_;

  // shape should be 4D
  if (CheckOriginShapeDimNum(input_shape, DIM_DEFAULT_SIZE)) {
    int64_t dim_value_a = 0;
    // axis C should be C0 aligned
    GetDimValue(C_AXIS_NAME, input_format, input_shape, dim_value_a);
    if (!IsAxisC0Aligned(input_dtype, dim_value_a)) {
      FE_LOGD("Axis C [%ld] is not C0 aligned, input_formats=[%d], inputDtype=[%d], inputShape=[%s].", dim_value_a,
              input_format, input_dtype, GetShapeDims(input_shape).c_str());
      return false;
    }

    // axis N should be NI aligned
    GetDimValue(N_AXIS_NAME, input_format, input_shape, dim_value_a);
    int64_t n = dim_value_a;
    if (!IsAxisAligned(n, NI)) {
      FE_LOGD("Axis N [%ld] is not 16 aligned, input_format=[%d], input_dtype=[%d], inputShape=[%s].", n, input_format,
              input_dtype, GetShapeDims(input_shape).c_str());
      return false;
    }
  } else {
    FE_LOGD("The dim_num of the input_shape=[%s] is < %zu.", GetShapeDims(input_shape).c_str(), DIM_DEFAULT_SIZE);
    return false;
  }
  return true;
}

bool BroadcastProcess6D::CheckAllNonScalarInputs(const FormatProccessArgs &args) {
  auto current_input_formats = args.origin_info_ptr->input_formats;
  auto input_shapes = args.origin_info_ptr->input_shapes;
  for (size_t i = 0; i < input_shapes.size(); i++) {
    // each shape should be 4D
    if (!CheckOriginShapeDimNum(input_shapes[i], DIM_DEFAULT_SIZE)) {
      FE_LOGD("The dim_num of the input_shapes[%zu] value[%s] is < %zu.", i, GetShapeDims(input_shapes[i]).c_str(),
              DIM_DEFAULT_SIZE);
      return false;
    }
  }
  // axis C&N should not need broadcast, if need, not support 6D
  if (CheckAxisNeedBroadcast(C_AXIS_NAME, current_input_formats, input_shapes)) {
    FE_LOGD("Axis C needs to broadcast.");
    return false;
  }
  if (CheckAxisNeedBroadcast(N_AXIS_NAME, current_input_formats, input_shapes)) {
    FE_LOGD("Axis N needs to broadcast.");
    return false;
  }
  return true;
}

REGISTER_FORMAT_PROCESS(BroadcastProcess6D, OP_PATTERN_BROADCAST, FORMAT_C1HWNCoC0, ge::FORMAT_C1HWNCoC0);
REGISTER_FORMAT_PROCESS(BroadcastProcess6D, OP_PATTERN_BROADCAST_ENHANCED, FORMAT_C1HWNCoC0, ge::FORMAT_C1HWNCoC0);
}  // namespace fe
