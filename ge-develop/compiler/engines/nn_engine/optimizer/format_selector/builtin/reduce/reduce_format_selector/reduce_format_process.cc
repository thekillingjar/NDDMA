/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "format_selector/builtin/reduce/reduce_format_selector/reduce_format_process.h"
#include "common/fe_type_utils.h"

namespace fe {
bool ReduceFormatProcess::CheckOriginFormatAndShape(const vector<ge::Format> &input_formats,
                                                    const vector<ge::Format> &output_formats,
                                                    const vector<ge::GeShape> &shapes,
                                                    const size_t &dim) {
  (void)output_formats;
  if (!CheckAccuracyOriginShapesDimNum(shapes, dim)) {
    FE_LOGD("The size of the origin shapes is != %zu.", dim);
    return false;
  }

  if (!CheckOriginFormatsIdentifiable(input_formats)) {
    FE_LOGD("The origin formats are not identifiable.");
    return false;
  }

  return true;
}

bool ReduceFormatProcess::CheckContainReduceAxis(const ge::OpDesc &op_desc, const vector<ge::Format> &formats,
                                                 const vector<ge::GeShape> &shapes, const string &axis_name) {
  string op_name = op_desc.GetName();
  string op_type = op_desc.GetType();
  size_t size = formats.size();

  if (shapes.size() != size) {
    return false;
  }

  for (size_t i = 0; i < size; ++i) {
    vector<int64_t> axis_index_vec;
    if (AxisUtil::GetOriginAxisAttribute(op_desc, shapes[i], axis_index_vec) != SUCCESS) {
      FE_LOGW("Op[%s,optype[%s]]: can not get origin axis attribute.", op_name.c_str(), op_type.c_str());
      return true;
    }

    int64_t axis_tobe_check = -1;
    if (GetOriginAxisIndexByName(formats[i], shapes[i], axis_name, axis_tobe_check) != SUCCESS) {
      FE_LOGW("Op[%s,optype[%s]]: can not get %s!", op_name.c_str(), op_type.c_str(), axis_name.c_str());
      return true;
    }

    for (const auto &axis_index : axis_index_vec) {
      if (axis_tobe_check == axis_index) {
        return true;
      }
    }
  }
  return false;
}

bool ReduceFormatProcess::CheckNdFormat(const vector<ge::Format>& formats) {
  for (auto format : formats) {
    FE_LOGD("The format %s.", ge::TypeUtils::FormatToSerialString(format).c_str());
    if (format == ge::FORMAT_ND) {
      return false;
    }
  }
  return true;
}

Status ReduceFormatProcess::GetOriginAxisIndexByName(const ge::Format &format, const ge::GeShape &shape,
                                                     const string &axis_name, int64_t &axis_index) {
  if (axis_name == N_AXIS_NAME) {
    axis_index = GetAxisIndexByFormat(format, N_AXIS_NAME);
  } else if (axis_name == C_AXIS_NAME) {
    axis_index = GetAxisIndexByFormat(format, C_AXIS_NAME);
  } else if (axis_name == LAST_AXIS_NAME) {
    axis_index = shape.GetDimNum() - 1;
  } else if (axis_name == LASTBUTONE_AXIS_NAME) {
    axis_index = shape.GetDimNum() - 2;
  }
  if (axis_index < 0) {
    return FAILED;
  }

  FE_LOGD("origin axis index by axis_name %s is: %ld.", axis_name.c_str(), axis_index);
  return SUCCESS;
}

void ReduceFormatProcess::GenerateFormats(const size_t &in_shape_size, const size_t &out_shape_size,
                                          const vector<ge::Format> &support_in_formats,
                                          const vector<ge::Format> &support_out_formats, FormatProccessResult &result)
                                          const {
  for (const auto &support_format : support_in_formats) {
    vector<ge::Format> tmp(in_shape_size, support_format);
    result.input_format_res.emplace_back(tmp);
    result.input_subformat_res.emplace_back(SUPPORT_ALL_SUB_FORMAT);
  }

  for (const auto &support_format : support_out_formats) {
    vector<ge::Format> tmp(out_shape_size, support_format);
    result.output_format_res.emplace_back(tmp);
    result.output_subformat_res.emplace_back(SUPPORT_ALL_SUB_FORMAT);
  }
}
}  // namespace fe
