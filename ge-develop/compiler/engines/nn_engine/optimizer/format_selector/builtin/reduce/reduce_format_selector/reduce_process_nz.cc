/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "format_selector/builtin/reduce/reduce_format_selector/reduce_process_nz.h"
#include "common/util/op_info_util.h"
#include "common/fe_op_info_common.h"

namespace fe {
namespace {
const int64_t kIntervalValue = 16;
}

Status ReduceProcessNz::Process(const ge::OpDesc &op_desc, const FormatProccessArgs &args,
                                FormatProccessResult &result) {
  OriginInfoPtr origin_info_ptr = args.origin_info_ptr;
  vector<ge::Format> input_formats = origin_info_ptr->input_formats;
  vector<ge::DataType> input_dtypes = origin_info_ptr->input_dtypes;
  vector<ge::GeShape> input_shapes = origin_info_ptr->input_shapes;
  vector<ge::Format> output_formats = origin_info_ptr->output_formats;
  size_t input_shapes_size = input_shapes.size();
  string op_name = op_desc.GetName();
  string op_type = op_desc.GetType();

  if (!CheckOriginFormatAndShape(input_formats, output_formats, input_shapes, MINIMUM_NZ_SHAPE_DIM_NUM)) {
    FE_LOGD("Op[name=%s,type=%s]: check origin format and shape not successfully.", op_name.c_str(), op_type.c_str());
    return FAILED;
  }

  // 2. check the axis attribute of the op_desc
  bool is_reduce_last = CheckContainReduceAxis(op_desc, input_formats, input_shapes, LAST_AXIS_NAME);
  bool is_reduce_lastbutone = CheckContainReduceAxis(op_desc, input_formats, input_shapes, LASTBUTONE_AXIS_NAME);

  vector<ge::Format> input_support_format;
  vector<ge::Format> output_support_format;
  ge::Format support_format = args.support_format;
  // don't reduce the last axis and lastbutone axis: NZ->NZ
  if (!is_reduce_last && !is_reduce_lastbutone) {
    input_support_format.emplace_back(support_format);
    output_support_format.emplace_back(support_format);
  } else {
    // the last axis and the lastbutone axis is aligned: NZ->ND
    if (CheckNzAxisIsAligned(op_desc, input_dtypes, input_shapes, is_reduce_last, is_reduce_lastbutone)) {
      input_support_format.emplace_back(support_format);
      output_support_format.emplace_back(ge::FORMAT_ND);
    }
  }

  // 3. generate the result
  if (input_support_format.empty()) {
    return FAILED;
  }
  GenerateFormats(input_shapes_size, origin_info_ptr->output_shapes.size(), input_support_format, output_support_format,
                  result);
  return SUCCESS;
}  // namespace fe

bool ReduceProcessNz::CheckOriginFormatAndShape(const vector<ge::Format> &input_formats,
                                                const vector<ge::Format> &output_formats,
                                                const vector<ge::GeShape> &shapes,
                                                const size_t &dim) {
  (void)input_formats;
  (void)output_formats;
  if (!CheckOriginShapesDimNum(shapes, dim)) {
    FE_LOGD("The shape size is < %zu.", dim);
    return false;
  }
  return true;
}

bool ReduceProcessNz::CheckNzAxisIsAligned(const ge::OpDesc &op_desc, const vector<ge::DataType> &dtypes,
                                           const vector<ge::GeShape> &shapes, const bool &is_reduce_last,
                                           const bool &is_reduce_lastbutone) const {
  string op_name = op_desc.GetName();
  string op_type = op_desc.GetType();
  size_t size = shapes.size();

  for (size_t i = 0; i != size; ++i) {
    if (is_reduce_last || is_reduce_lastbutone) {
      int64_t dim_num = static_cast<int64_t>(shapes[i].GetDimNum());
      int64_t last_dim_value = shapes[i].GetDim(dim_num - 1);
      int64_t lastbutone_dim_value = shapes[i].GetDim(dim_num - 2);
      if (!IsAxisC0Aligned(dtypes[i], last_dim_value) || !IsAxisAligned(lastbutone_dim_value, kIntervalValue)) {
        return false;
      }
    }
  }
  return true;
}

REGISTER_FORMAT_PROCESS(ReduceProcessNz, OP_PATTERN_REDUCE, FORMAT_FRACTAL_NZ, ge::FORMAT_FRACTAL_NZ);
}  // namespace fe
