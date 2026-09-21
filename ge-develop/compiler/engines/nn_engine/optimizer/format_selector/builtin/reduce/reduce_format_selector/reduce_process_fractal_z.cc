/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "format_selector/builtin/reduce/reduce_format_selector/reduce_process_fractal_z.h"
#include "common/fe_inner_attr_define.h"

namespace fe {
Status ReduceProcessFractalZ::Process(const ge::OpDesc &op_desc, const FormatProccessArgs &args,
                                      FormatProccessResult &result) {
  OriginInfoPtr origin_info_ptr = args.origin_info_ptr;
  vector<ge::Format> input_formats = origin_info_ptr->input_formats;
  vector<ge::GeShape> input_shapes = origin_info_ptr->input_shapes;
  vector<ge::Format> output_formats = origin_info_ptr->output_formats;
  string op_name = op_desc.GetName();
  string op_type = op_desc.GetType();

  // 1. check origin formats and shapes; if support format is 6HD,
  // we need to check whether the inputs shape is 5D.
  size_t min_dim = DIM_DEFAULT_SIZE;
  if (args.support_format == ge::FORMAT_FRACTAL_Z_3D) {
    min_dim = DIMENSION_NUM_FIVE;
  }

  if (!CheckOriginFormatAndShape(input_formats, output_formats, input_shapes, min_dim)) {
    FE_LOGD("Op[name=%s,type=%s]: check origin format and shape not successfully.", op_name.c_str(), op_type.c_str());
    return FAILED;
  }

  // 2. check the axis attribute of the op_desc
  if (!CheckOpDescAxisAttr(op_desc, input_formats, input_shapes)) {
    FE_LOGD("Op[name=%s,type=%s]: check the axis attribute not successfully.", op_name.c_str(), op_type.c_str());
    return FAILED;
  }

  // 3. genareate result
  GenerateFormats(input_shapes.size(), origin_info_ptr->output_shapes.size(), {args.support_format},
                  {args.support_format}, result);
  return SUCCESS;
}

bool ReduceProcessFractalZ::CheckOriginFormatAndShape(const vector<ge::Format> &input_formats,
                                                      const vector<ge::Format> &output_formats,
                                                      const vector<ge::GeShape> &shapes, const size_t &dim) {
  if (!CheckOriginShapesDimNum(shapes, dim)) {
    FE_LOGD("The size of the origin shapes is < %zu.", dim);
    return false;
  }

  if (!CheckOriginFormatsIdentifiable(input_formats)) {
    FE_LOGD("The origin formats are not identifiable.");
    return false;
  }

  if (!CheckNdFormat(input_formats) || !CheckNdFormat(output_formats)) {
    FE_LOGD("The input/output origin formats contains ND");
    return false;
  }

  return true;
}

bool ReduceProcessFractalZ::CheckOpDescAxisAttr(const ge::OpDesc &op_desc, const vector<ge::Format> &formats,
                                                const vector<ge::GeShape> &shapes) {
  string op_name = op_desc.GetName();
  string op_type = op_desc.GetType();

  // 1. keep_dims must be true
  bool keep_dim = false;
  if (!ge::AttrUtils::GetBool(op_desc, KEEP_DIMS, keep_dim) || !keep_dim) {
    FE_LOGD("Op[name=%s,type=%s]: the attribute keep_dims is not true.", op_name.c_str(), op_type.c_str());
    return false;
  }

  // 2. reduce axis can not be C
  if (CheckContainReduceAxis(op_desc, formats, shapes, C_AXIS_NAME)) {
    FE_LOGD("Op[name=%s,type=%s]: reduce axis contains C.", op_name.c_str(), op_type.c_str());
    return false;
  }

  // 3. reduce axis can not be N
  if (CheckContainReduceAxis(op_desc, formats, shapes, N_AXIS_NAME)) {
    FE_LOGD("Op[name=%s,type=%s]: reduce axis contains N.", op_name.c_str(), op_type.c_str());
    return false;
  }
  return true;
}

REGISTER_FORMAT_PROCESS(ReduceProcessFractalZ, OP_PATTERN_REDUCE, FORMAT_FRACTAL_Z, ge::FORMAT_FRACTAL_Z);
REGISTER_FORMAT_PROCESS(ReduceProcessFractalZ, OP_PATTERN_REDUCE, FORMAT_C1HWNCoC0, ge::FORMAT_C1HWNCoC0);
REGISTER_FORMAT_PROCESS(ReduceProcessFractalZ, OP_PATTERN_REDUCE, FORMAT_FRACTAL_Z_3D, ge::FORMAT_FRACTAL_Z_3D);
}  // namespace fe
