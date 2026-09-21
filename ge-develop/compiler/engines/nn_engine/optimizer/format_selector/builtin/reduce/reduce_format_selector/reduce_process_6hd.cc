/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "format_selector/builtin/reduce/reduce_format_selector/reduce_process_6hd.h"

namespace fe {
Status ReduceProcess6HD::Process(const ge::OpDesc &op_desc, const FormatProccessArgs &args,
                                 FormatProccessResult &result) {
  OriginInfoPtr origin_info_ptr = args.origin_info_ptr;
  vector<ge::Format> input_formats = origin_info_ptr->input_formats;
  vector<ge::GeShape> input_shapes = origin_info_ptr->input_shapes;
  vector<ge::Format> output_formats = origin_info_ptr->output_formats;
  string op_name = op_desc.GetName();
  string op_type = op_desc.GetType();

  // 1. check origin formats and shapes
  if (!CheckOriginFormatAndShape(input_formats, output_formats, input_shapes, DIMENSION_NUM_FIVE)) {
    FE_LOGD("Op[name=%s,type=%s]: check origin format and shape not successfully.", op_name.c_str(), op_type.c_str());
    return FAILED;
  }

  // 2. check the axis attribute of the op_desc
  bool is_reduce_c = CheckContainReduceAxis(op_desc, input_formats, input_shapes, C_AXIS_NAME);
  if (is_reduce_c) {
    FE_LOGD("reduce axis contains C is not allowed for NDC1HWC0.");
    return FAILED;
  }

  // 3. if not contain reduce_c, input/output cannot contain ND
  if (!is_reduce_c && (!CheckNdFormat(input_formats) || !CheckNdFormat(output_formats))) {
    FE_LOGD("The input/output origin formats contains ND");
    return false;
  }

  // 4. genareate result
  GenerateFormats(input_shapes.size(), origin_info_ptr->output_shapes.size(),
                  {args.support_format}, {args.support_format}, result);
  return SUCCESS;
}

REGISTER_FORMAT_PROCESS(ReduceProcess6HD, OP_PATTERN_REDUCE, FORMAT_NDC1HWC0, ge::FORMAT_NDC1HWC0);
}  // namespace fe
