/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "format_selector/builtin/broadcast/format_process/broadcast_format_process.h"
#include "common/fe_op_info_common.h"
#include "graph/utils/type_utils.h"

namespace fe {
namespace {
bool IsFormatValid(ge::Format &first_idtf_format, const ge::Format &current_format, const ge::GeShape &current_shape) {
  bool is_scalar = IsScalarShape(current_shape);
  if (current_format == ge::FORMAT_ND) {
    return is_scalar;
  } else {
    if (is_scalar) {
      return true;
    } else {
      if (first_idtf_format == ge::FORMAT_RESERVED) {
        first_idtf_format = current_format;
        return true;
      } else {
        return first_idtf_format == current_format;
      }
    }
  }
}
} // namespace

Status BroadcastFormatProcess::Process(const ge::OpDesc &op_desc, const FormatProccessArgs &args,
                                       FormatProccessResult &result) {
  auto support_format = args.support_format;
  FE_CHECK_NOTNULL(args.origin_info_ptr);
  auto input_formats = args.origin_info_ptr->input_formats;
  auto input_dtypes = args.origin_info_ptr->input_dtypes;
  auto input_shapes = args.origin_info_ptr->input_shapes;
  auto output_shapes = args.origin_info_ptr->output_shapes;
  auto propagat_primary_format = args.propagat_primary_format;
  auto propagat_sub_format = args.propagat_sub_format;

  auto op_name = op_desc.GetName();
  auto op_type = op_desc.GetType();

  // 1. check the origin format
  if (!CheckOriginFormat(input_formats, input_shapes)) {
    FE_LOGW("Op[name%s,optype[%s]]: all inputs are different in format.", op_name.c_str(), op_type.c_str());
    return FAILED;
  }

  // 2. check the origin shape
  size_t input_size = op_desc.GetAllInputsSize();
  size_t output_size = op_desc.GetOutputsSize();
  std::set<size_t> scalar_input_index;
  GetScalarInputIndex(input_shapes, scalar_input_index);
  // check whether shapes of all inputs are the same
  if (CheckOriginShape(input_shapes)) {
    FE_LOGD("Op[name=%s,type=%s]: all input_descs have the same origin shape.", op_name.c_str(), op_type.c_str());
    /* 2.1 For 6HD, we need the original shape to be 5D although the two
     * shape is same. The reason is there is no padding mechanism from less
     * than 5D to 5D. */
    size_t dim_value = 0;
    if (Check6HDShape(input_shapes, support_format, dim_value) != SUCCESS) {
      FE_LOGD("Op[name=%s,type=%s]: dim size %zu should > 5 for 6HD.", op_name.c_str(), op_type.c_str(), dim_value);
      return SUCCESS;
    }
    // check whether has scalar shape
    ge::Format same_format = (scalar_input_index.empty() ? support_format : ge::FORMAT_ND);
    InsertFormatVec(input_size, same_format, result.input_format_res);
    InsertFormatVec(output_size, same_format, result.output_format_res);
    InsertSubformatVec(support_format, propagat_sub_format, result.input_subformat_res, result.output_subformat_res);
    return SUCCESS;
  }

  bool is_need_check_flag = propagat_sub_format > GROUPS_DEFAULT_VALUE &&
      std::find(FE_GROUP_RELA_FORMAT_VECTOR.begin(), FE_GROUP_RELA_FORMAT_VECTOR.end(), support_format) !=
      FE_GROUP_RELA_FORMAT_VECTOR.end();
  if (is_need_check_flag && CheckNewShapeSupportBroadcast(op_desc, input_formats, input_dtypes, input_shapes,
        support_format, propagat_sub_format)) {
    FE_LOGD("Op[name=%s,type=%s]: new shape with sub_format[%u] support broadcast.", op_name.c_str(), op_type.c_str(),
            propagat_sub_format);
    result.input_subformat_res.emplace_back(propagat_sub_format);
    result.output_subformat_res.emplace_back(propagat_sub_format);
  }
  // 3. has scalar inputs
  if (!scalar_input_index.empty()) {
    vector<ge::Format> input_support_formats;
    FE_CHECK((input_shapes.size() != input_dtypes.size()), FE_LOGE("Shape and type size not equal."), return FAILED);
    for (size_t i = 0; i < input_shapes.size(); ++i) {
      if (scalar_input_index.count(i) == 1) {
        FE_LOGD("Op[name=%s,type=%s]: input %zu is scalar.", op_name.c_str(), op_type.c_str(), i);
        input_support_formats.emplace_back(ge::FORMAT_ND);
        continue;
      }

      FormatProccessInputArg input_arg(input_formats[i], input_dtypes[i], input_shapes[i], propagat_primary_format,
                                       propagat_sub_format);
      if (!CheckPartNonScalarInputs(input_arg)) {
        FE_LOGD("Op[name=%s,type=%s]: check non-scalar input [%zu] not successfully.", op_name.c_str(), op_type.c_str(), i);
        return FAILED;
      }
      FE_LOGD("Op[name=%s,type=%s]: input %zu is non-scalar.", op_name.c_str(), op_type.c_str(), i);
      input_support_formats.emplace_back(support_format);
    }

    result.input_format_res.emplace_back(input_support_formats);
    vector<ge::Format> output_support_formats;
    GenerateOutputFormats(output_shapes, support_format, output_support_formats);
    result.output_format_res.emplace_back(output_support_formats);
    return SUCCESS;
  }

  // 4. has no scalar inputs
  if (!CheckAllNonScalarInputs(args)) {
    FE_LOGD("Op[name=%s,type=%s]: check all non-scalar inputs not successfully.", op_name.c_str(), op_type.c_str());
    return FAILED;
  }

  InsertFormatVec(input_size, support_format, result.input_format_res);
  vector<ge::Format> out_support_formats;
  GenerateOutputFormats(output_shapes, support_format, out_support_formats);
  result.output_format_res.emplace_back(out_support_formats);
  return SUCCESS;
}

bool BroadcastFormatProcess::CheckOriginFormat(const std::vector<ge::Format> &input_formats,
                                               const vector<ge::GeShape> &input_shapes) {
  // formats must be identifiable and same
  if (input_formats.empty()) {
    FE_LOGD("The input_formats is empty.");
    return false;
  }

  if (input_shapes.empty()) {
    FE_LOGD("The input shapes is empty.");
    return false;
  }
  if (input_shapes.size() != input_formats.size()) {
    FE_LOGD("Sizes of input shapes and format are not equal.");
    return false;
  }
  ge::Format first_format = ge::FORMAT_RESERVED;
  for (size_t i = 0; i < input_shapes.size(); ++i) {
    if (!IsFormatValid(first_format, input_formats[i], input_shapes[i])) {
      FE_LOGD("The format %d is different from the first format %d.", input_formats[i], first_format);
      return false;
    }
  }

  /* If there is no (non-scalar && non-ND) tensors, the first_format will not
   * be set, we just return false. */
  return !(first_format == ge::FORMAT_RESERVED);
}

bool BroadcastFormatProcess::CheckOriginShape(const std::vector<ge::GeShape> &shapes) {
  // shape must be same
  if (shapes.empty()) {
    return false;
  }
  ge::GeShape first_shape = shapes.front();
  for (auto iter = shapes.begin(); iter != shapes.end(); iter++) {
    if (!IsSameShape(first_shape, *iter)) {
      return false;
    }
  }
  return true;
}

// If we need to support format 6HD, we need to make sure the shape
// is 5D.
Status BroadcastFormatProcess::Check6HDShape(const std::vector<ge::GeShape> &shapes, const ge::Format &supprt_format,
                                             size_t &dim_value) const {
  if (!(supprt_format == ge::FORMAT_NDC1HWC0 || supprt_format == ge::FORMAT_FRACTAL_Z_3D)) {
    return SUCCESS;
  }
  if (shapes.empty()) {
    REPORT_FE_ERROR("[GraphOpt][GenBuiltInFmt][ChkUbSizeEnHDShp] The parameter[shapes] is empty.");
    return FAILED;
  }

  /* !!!!Attention!!!!: Because now this function is invoked when every input
   * shape is the same, so here only need to judge the first input shape. */
  ge::GeShape first_shape = shapes.front();
  dim_value = first_shape.GetDimNum();
  if (dim_value < DIMENSION_NUM_FIVE) {
    return FAILED;
  }
  return SUCCESS;
}

void BroadcastFormatProcess::GetScalarInputIndex(const std::vector<ge::GeShape> &shapes,
                                                 std::set<size_t> &scalar_input_index) const {
  for (size_t i = 0; i < shapes.size(); ++i) {
    if (IsScalarShape(shapes[i])) {
      scalar_input_index.emplace(i);
    }
  }
}

void BroadcastFormatProcess::GetDimValue(const std::string &dim, const ge::Format &format, const ge::GeShape &shape,
                                         int64_t &dim_value) const {
  (void)GetDimValueByFormatAndShape(format, shape, dim, dim_value);
}

bool BroadcastFormatProcess::CheckAxisNeedBroadcast(const std::string &dim, const std::vector<ge::Format> &formats,
                                                    const std::vector<ge::GeShape> &shapes) const {
  if (formats.empty()) {
    REPORT_FE_ERROR(
        "[GraphOpt][GenBuiltInFmt][ChkAxisNeedBrdct] The parameter[formats] is empty, no need to do broadcast.");
    return false;
  }

  if (shapes.empty()) {
    REPORT_FE_ERROR(
        "[GraphOpt][GenBuiltInFmt][ChkAxisNeedBrdct] The parameter[shapes] is empty, no need to do broadcast.");
    return false;
  }

  int64_t input_desc_num = formats.size();
  if (input_desc_num == 1) {
    FE_LOGD("The size of formats is 1, no need to do broadcast.");
    return false;
  }

  int64_t shape_num = shapes.size();
  if (input_desc_num != shape_num) {
    REPORT_FE_ERROR("[GraphOpt][GenBuiltInFmt][ChkAxisNeedBrdct] The size of shape and format are not equal.");
    return false;
  }

  int64_t dim_value = 0;
  GetDimValue(dim, formats.front(), shapes.front(), dim_value);

  for (int64_t i = 1; i < input_desc_num; ++i) {
    int64_t tmp_dim_value = 0;
    GetDimValue(dim, formats[i], shapes[i], tmp_dim_value);
    if (dim_value == SHAPE_UNKNOWN_DIM && tmp_dim_value == SHAPE_UNKNOWN_DIM) {
      FE_LOGD("Both Axis[%s] of two shape is -1, need to broadcast.", dim.c_str());
      return true;
    }
    if (dim_value != tmp_dim_value) {
      FE_LOGD("Axis[%s] is not equal(dim_value=%ld, tmp_dim_value=[%ld]), need to broadcast.", dim.c_str(), dim_value,
              tmp_dim_value);
      return true;
    }
  }
  FE_LOGD("Axis[%s] is equal, no need to broadcast.", dim.c_str());
  return false;
}

void BroadcastFormatProcess::GenerateOutputFormats(const vector<ge::GeShape> &output_shapes, const ge::Format &format,
                                                   vector<ge::Format> &output_formats) const {
  for (auto shape : output_shapes) {
    if (IsScalarShape(shape)) {
      output_formats.push_back(ge::FORMAT_ND);
    } else {
      output_formats.push_back(format);
    }
  }
}

void BroadcastFormatProcess::InsertFormatVec(const size_t &size, const ge::Format &format,
                                             vector<vector<ge::Format>> &formats) const {
  vector<ge::Format> format_vec(size, format);
  formats.emplace_back(format_vec);
}

void BroadcastFormatProcess::InsertSubformatVec(const ge::Format &format, const uint32_t &sub_format,
                                                vector<uint32_t> &input_subformat_res,
                                                vector<uint32_t> &output_subformat_res) const {
    if (sub_format <= 1 && std::find(FE_GROUP_RELA_FORMAT_VECTOR.begin(), FE_GROUP_RELA_FORMAT_VECTOR.end(),
        format) == FE_GROUP_RELA_FORMAT_VECTOR.end()) {
      return;
    }
    input_subformat_res.emplace_back(sub_format);
    output_subformat_res.emplace_back(sub_format);
}

ge::GeShape BroadcastFormatProcess::GetNewShapeWithNewFormat(const ge::OpDesc &op_desc, const ge::GeShape &old_shape,
    const ge::Format &old_format, const ge::Format &new_format,
    const ge::DataType &current_data_type, const int32_t &group) const {
  ge::GeShape new_shape;
  int64_t hidden_size = 1;
  int64_t input_size = 1;
  int64_t state_size = -1;
  (void)ge::AttrUtils::GetInt(op_desc, "hidden_size", hidden_size);
  (void)ge::AttrUtils::GetInt(op_desc, "input_size", input_size);
  (void)ge::AttrUtils::GetInt(op_desc, "state_size", state_size);
  CalcShapeExtraAttr extra_attr = {hidden_size, input_size, state_size};
  ShapeAndFormat shape_and_format_info = {old_shape, new_shape, old_format, new_format,
                                          current_data_type, group, extra_attr};
  Status ret = GetShapeAccordingToFormat(shape_and_format_info);
  if (ret != SUCCESS) {
    FE_LOGW("[GraphOpt][ChkShpWithSubfmtSupBrdcst] Old format is %s, new format is %s, old dimension is %ld.",
            ge::TypeUtils::FormatToSerialString(old_format).c_str(),
            ge::TypeUtils::FormatToSerialString(new_format).c_str(), old_shape.GetDimNum());
    return old_shape;
  }
  return new_shape;
}

bool BroadcastFormatProcess::CheckShapeWithSubformatSupportBroadcast(vector<ge::GeShape> &shapes) const {
  size_t shape_size = shapes.size();
  if (shape_size <= 1) {
    FE_LOGD("[GraphOpt][GenBuiltInFmt][ChkShpWithSubfmtSupBrdcst] shape_size %zu invalid.", shape_size);
    return false;
  }
  size_t dim_num = shapes[0].GetDimNum();
  for (size_t i = 1; i < shape_size; i++) {
    size_t tmp_dim_num = shapes[i].GetDimNum();
    if (tmp_dim_num != dim_num) {
      FE_LOGD("[ChkShpWithSubfmtSupBrdcst] shape_dims size[%zu], tmp_shape_dims size[%zu], not support broadcast.",
              dim_num, tmp_dim_num);
      return false;
    }
  }
  return true;
}

bool BroadcastFormatProcess::CheckNewShapeSupportBroadcast(const ge::OpDesc &op_desc,
                                                           const vector<ge::Format> &input_formats,
                                                           const vector<ge::DataType> &input_dtypes,
                                                           const vector<ge::GeShape> &input_shapes,
                                                           const ge::Format &new_format,
                                                           uint32_t sub_format) const {
  vector<ge::GeShape> new_shapes;
  bool is_no_need_check_flag = input_formats.empty() || input_dtypes.empty() || input_shapes.empty();
  if (is_no_need_check_flag) {
    FE_LOGD("[GraphOpt][GenBuiltInFmt][ChkNewShpSupotBrdcst] input args empty.");
    return false;
  }
  auto input_desc_num = input_formats.size();
  auto input_dtypes_num = input_dtypes.size();
  auto input_shapes_num = input_shapes.size();
  if (input_desc_num != input_dtypes_num || input_desc_num != input_shapes_num) {
    FE_LOGE("[ChkNewShpSupotBrdcst] node[%s, %s] size of shape and format and dtype are not equal.",
            op_desc.GetName().c_str(), op_desc.GetName().c_str());
    return false;
  }
  for (size_t i = 0; i < input_desc_num; i++) {
    if (IsScalarShape(input_shapes[i])) {
      new_shapes.emplace_back(ge::GeShape({1}));
      continue;
    }
    ge::GeShape new_shape = GetNewShapeWithNewFormat(op_desc, input_shapes[i], input_formats[i], new_format,
                                                     input_dtypes[i], sub_format);
    new_shapes.emplace_back(new_shape);
  }
  if (!CheckShapeWithSubformatSupportBroadcast(new_shapes)) {
    FE_LOGD("[GraphOpt][GenBuiltInFmt][ChkNewShpSupotBrdcst] node[%s, %s] with sub_format[%u] not support broadcast",
            op_desc.GetName().c_str(), op_desc.GetName().c_str(), sub_format);
    return false;
  }
  return true;
}
}  // namespace fe
