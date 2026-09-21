/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "format_selector/builtin/broadcast/format_process/broadcast_process_fractal_z.h"
#include "common/fe_op_info_common.h"

namespace fe {
bool BroadcastProcessFractalZ::CheckPartNonScalarInputs(const FormatProccessInputArg &input_arg) {
  auto input_format = input_arg.input_format_;
  auto input_dtype = input_arg.input_dtype_;
  auto input_shape = input_arg.input_shape_;
  auto propagat_primary_format = input_arg.propagat_primary_format_;
  auto propagat_sub_format = input_arg.propagat_sub_format_;

  // shape should be 4D
  if (CheckOriginShapeDimNum(input_shape, DIM_DEFAULT_SIZE)) {
    int64_t dim_value = 0;
    // axis C should be C0 aligned
    GetDimValue(C_AXIS_NAME, input_format, input_shape, dim_value);
    if (!IsAxisC0Aligned(input_dtype, dim_value)) {
      FE_LOGD("Axis C [%ld] is not C0 aligned, input_formats=[%d], inputDtype=[%d], inputShape=[%s].", dim_value,
              input_format, input_dtype, GetShapeDims(input_shape).c_str());
      return false;
    }
    if (input_shape.IsUnknownShape()) {
      FE_LOGD("The dim_value of the input_shape[%s] is dynamic.", GetShapeDims(input_shape).c_str());
      return false;
    }
    // axis N should be NI aligned
    GetDimValue(N_AXIS_NAME, input_format, input_shape, dim_value);
    int64_t n = dim_value;
    if (propagat_primary_format == ge::FORMAT_FRACTAL_Z && propagat_sub_format > 1) {
      n /= propagat_sub_format;
      FE_LOGD("n=[%ld]: the dim_value is %ld, propagat_sub_format is %d.", n, dim_value, propagat_sub_format);
    }
    if (!IsAxisAligned(n, NI)) {
      FE_LOGD("Axis N [%ld] is not 16 aligned, input_format[%d], input_dtype[%d], inputShape=[%s].", n, input_format,
              input_dtype, GetShapeDims(input_shape).c_str());
      return false;
    }
  } else {
    FE_LOGD("The dim_num of the input_shape=[%s] is < %zu.", GetShapeDims(input_shape).c_str(), DIM_DEFAULT_SIZE);
    return false;
  }
  return true;
}

bool BroadcastProcessFractalZ::CheckAllNonScalarInputs(const FormatProccessArgs &args) {
  auto input_formats_a = args.origin_info_ptr->input_formats;
  auto input_shapes = args.origin_info_ptr->input_shapes;

  // each shape should be 4D
  for (size_t i = 0; i < input_shapes.size(); i++) {
    if (!CheckOriginShapeDimNum(input_shapes[i], DIM_DEFAULT_SIZE)) {
      FE_LOGD("The dim_num of the input_shape[%zu] value[%s] is < %zu.", i, GetShapeDims(input_shapes[i]).c_str(),
              DIM_DEFAULT_SIZE);
      return false;
    }
    if (input_shapes[i].IsUnknownShape()) {
      FE_LOGD("The dim_value of the input_shape[%zu] value[%s] is dynamic.", i, GetShapeDims(input_shapes[i]).c_str());
      return false;
    }
  }

  // axis c should not need broadcast, if need, not support FRACTAL_Z
  if (CheckAxisNeedBroadcast(C_AXIS_NAME, input_formats_a, input_shapes)) {
    FE_LOGD("Axis C needs to broadcast.");
    return false;
  }

  // axis n should not need broadcast, if need, not support FRACTAL_Z
  if (CheckAxisNeedBroadcast(N_AXIS_NAME, input_formats_a, input_shapes)) {
    FE_LOGD("Axis N needs to broadcast.");
    return false;
  }
  return true;
}

REGISTER_FORMAT_PROCESS(BroadcastProcessFractalZ, OP_PATTERN_BROADCAST, FORMAT_FRACTAL_Z, ge::FORMAT_FRACTAL_Z);
REGISTER_FORMAT_PROCESS(BroadcastProcessFractalZ, OP_PATTERN_BROADCAST_ENHANCED, FORMAT_FRACTAL_Z,
                        ge::FORMAT_FRACTAL_Z);
}  // namespace fe
