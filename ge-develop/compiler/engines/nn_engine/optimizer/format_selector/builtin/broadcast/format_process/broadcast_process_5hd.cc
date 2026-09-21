/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "format_selector/builtin/broadcast/format_process/broadcast_process_5hd.h"
#include "common/fe_op_info_common.h"

namespace fe {
bool BroadcastProcess5HD::CheckPartNonScalarInputs(const FormatProccessInputArg &input_arg) {
  auto input_format = input_arg.input_format_;
  auto input_dtype = input_arg.input_dtype_;
  auto input_shape = input_arg.input_shape_;
  // shape should be 4D
  if (!CheckOriginShapeDimNum(input_shape, DIM_DEFAULT_SIZE)) {
    FE_LOGD("The dim_num of the input_shape=[%s] is < %zu.", GetShapeDims(input_shape).c_str(), DIM_DEFAULT_SIZE);
    return false;
  }
  int64_t dim_value = 0;
  GetDimValue(C_AXIS_NAME, input_format, input_shape, dim_value);
  // axis c should be C0 aligned
  if (!IsAxisC0Aligned(input_dtype, dim_value)) {
    FE_LOGD("Axis C [%ld] is not 16 aligned, input_formats=[%d], inputDtype=[%d], inputShape=[%s].", dim_value,
            input_format, input_dtype, GetShapeDims(input_shape).c_str());
    return false;
  }
  return true;
}

bool BroadcastProcess5HD::CheckAllNonScalarInputs(const FormatProccessArgs &args) {
  auto input_formats = args.origin_info_ptr->input_formats;
  auto current_input_shapes = args.origin_info_ptr->input_shapes;

  // each shape should be 4D
  for (size_t i = 0; i < current_input_shapes.size(); i++) {
    if (!CheckOriginShapeDimNum(current_input_shapes[i], DIM_DEFAULT_SIZE)) {
      FE_LOGD("The dim_num of the input_shape[%zu] value[%s] is < %zu.",
              i, GetShapeDims(current_input_shapes[i]).c_str(),
              DIM_DEFAULT_SIZE);
      return false;
    }
  }

  // axis c should not need broadcast, if need, not support 5HD
  if (CheckAxisNeedBroadcast(C_AXIS_NAME, input_formats, current_input_shapes)) {
    FE_LOGD("Axis C needs to broadcast.");
    return false;
  }
  return true;
}

REGISTER_FORMAT_PROCESS(BroadcastProcess5HD, OP_PATTERN_BROADCAST, FORMAT_NC1HWC0, ge::FORMAT_NC1HWC0);
}  // namespace fe
