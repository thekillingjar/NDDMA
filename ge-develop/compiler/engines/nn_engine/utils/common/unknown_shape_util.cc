/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "common/unknown_shape_util.h"
#include "common/aicore_util_attr_define.h"
#include "common/fe_inner_attr_define.h"
#include "common/op_info_common.h"
#include "graph/ge_context.h"
#include "register/graph_optimizer/fusion_common/unknown_shape_utils.h"
#include "transfer_range_according_to_format.h"
#include "common/string_utils.h"

namespace fe {
std::string ShapeRangeToStr(const std::vector<std::pair<int64_t, int64_t>> &shape_range) {
  std::string s;
  s += "[";
  for (size_t i = 0; i < shape_range.size(); i++) {
    if (i != shape_range.size() - 1) {
      s = s + "[" + std::to_string(shape_range[i].first) + "," + std::to_string(shape_range[i].second) + "], ";
    } else {
      s = s + "[" + std::to_string(shape_range[i].first) + "," + std::to_string(shape_range[i].second) + "]";
    }
  }
  s += "]";
  return s;
}

std::vector<std::pair<int64_t, int64_t>> GetShapeRange(const ge::GeTensorDesc &tensor_desc) {
  std::vector<std::pair<int64_t, int64_t>> shape_range;
  auto status = tensor_desc.GetShapeRange(shape_range);
  if (status != ge::GRAPH_SUCCESS) {
    FE_LOGW("Shape range is not legal.");
    shape_range.clear();
    return shape_range;
  }
  return shape_range;
}

std::vector<std::pair<int64_t, int64_t>> GetOriginShapeRange(const ge::GeTensorDesc &tensor_desc) {
  std::vector<std::pair<int64_t, int64_t>> shape_range;
  auto status = tensor_desc.GetOriginShapeRange(shape_range);
  if (status != ge::GRAPH_SUCCESS) {
    FE_LOGW("Origin Shape range is not legal.");
    shape_range.clear();
    return shape_range;
  }
  return shape_range;
}

std::vector<std::pair<int64_t, int64_t>> GetValueRange(const ge::GeTensorDesc &tensor_desc) {
  std::vector<std::pair<int64_t, int64_t>> value_range;
  if (tensor_desc.GetValueRange(value_range) != ge::GRAPH_SUCCESS) {
    FE_LOGW("[SubGraphOpt][SetTbeTensor][GetValueRange] Failed to get value range from GE.");
    value_range.clear();
    return value_range;
  }
  return value_range;
}

std::vector<std::pair<int64_t, int64_t>> GetAlignShapeRange(
    const std::vector<std::pair<int64_t, int64_t>> &ori_shape_range, const ge::GeShape &shape) {
  if (ori_shape_range.empty()) {
    return ori_shape_range;
  }
  if (ori_shape_range.size() == shape.GetDimNum()) {
    FE_LOGD("Size of range: %zu, is equal to size of shape %zu.", ori_shape_range.size(), shape.GetDimNum());
    return ori_shape_range;
  }
  std::vector<std::pair<int64_t, int64_t>> shape_range(shape.GetDimNum());
  size_t range_index = 0;
  for (size_t i = 0; i < shape.GetDimNum(); i++) {
    int64_t dim = shape.GetDim(i);
    if (dim < 0) {  // unknown shape
      if (ori_shape_range.size() - 1 < range_index) {
        FE_LOGW("Size of shape: %zu, is less than %zu.", ori_shape_range.size(), range_index);
        return ori_shape_range;
      }
      shape_range[i] =
          std::pair<int64_t, int64_t>(ori_shape_range[range_index].first, ori_shape_range[range_index].second);
      range_index++;
    } else if (dim == 0) {
      shape_range[i] = std::pair<int64_t, int64_t>(1, 1);
    } else {
      shape_range[i] = std::pair<int64_t, int64_t>(dim, dim);
    }
  }
  return shape_range;
}

Status CalcShapeRange(const ge::OpDescPtr &op_desc, const ge::Format &final_format,
                      const ge::GeShape &dimension_expanded_ori_shape, ge::GeTensorDesc &tensor_desc) {
  auto original_format = tensor_desc.GetOriginFormat();
  auto primary_format = static_cast<ge::Format>(ge::GetPrimaryFormat(final_format));
  if (original_format == primary_format) {
    return SUCCESS;
  }

  if (UnknownShapeUtils::IsUnknownShapeOp(*op_desc)) {
    /* update shape range for unknown op */
    vector<std::pair<int64_t, int64_t>> new_range_shape;
    vector<std::pair<int64_t, int64_t>> ori_shape_range;
    (void)tensor_desc.GetShapeRange(ori_shape_range);
    FE_LOGD("Before Aligned shape and range are %s, %s.",
            StringUtils::IntegerVecToString(dimension_expanded_ori_shape.GetDims()).c_str(),
            ShapeRangeToStr(ori_shape_range).c_str());
    vector<std::pair<int64_t, int64_t>> old_shape_range =
        GetAlignShapeRange(ori_shape_range, dimension_expanded_ori_shape);
    FE_LOGD("Aligned shape and range are %s, %s.",
            StringUtils::IntegerVecToString(dimension_expanded_ori_shape.GetDims()).c_str(),
            ShapeRangeToStr(old_shape_range).c_str());
    transformer::RangeAndFormat range_and_format_info = {dimension_expanded_ori_shape, old_shape_range, new_range_shape,
                                                         original_format, final_format, tensor_desc.GetDataType()};
    bool ret = transformer::RangeTransferAccordingToFormat::GetRangeAccordingToFormat(
        op_desc, range_and_format_info);
    if (!ret) {
      REPORT_FE_ERROR(
          "[UnknownShapeUtil][CalcShpRange] Failed to get shape range. old format is %s, new format is %s",
          ge::TypeUtils::FormatToSerialString(original_format).c_str(),
          ge::TypeUtils::FormatToSerialString(primary_format).c_str());
      return FAILED;
    }

    if (tensor_desc.SetShapeRange(range_and_format_info.new_range) != ge::GRAPH_SUCCESS) {
      REPORT_FE_ERROR("[UnknownShapeUtil][CalcShpRange] Set shape range of op[name:%s,type:%s] failed.",
                      op_desc->GetName().c_str(), op_desc->GetType().c_str());
      return FAILED;
    }
    FE_LOGD("Set shape range of op[name:%s,type:%s] successfully. old format is %u, "
            "new format is %u. old range is %s, new range is %s.",
        op_desc->GetName().c_str(), op_desc->GetType().c_str(), original_format, primary_format,
        ShapeRangeToStr(old_shape_range).c_str(), ShapeRangeToStr(range_and_format_info.new_range).c_str());
  }
  return SUCCESS;
}

bool IsShapeGeneralizedMode() {
  std::string shape_generalized_str;
  if (ge::GetContext().GetOption("ge.shape_generalized", shape_generalized_str) != ge::GRAPH_SUCCESS) {
    FE_LOGD("[SubGraphOpt][Compile]There is no shape_generalized flag in ge options");
    return false;
  }

  FE_LOGD("[SubGraphOpt][Compile]Get shape_generalized flag from ge is %s.",
          shape_generalized_str.c_str());
  if (shape_generalized_str == kDefaultTrueStr) {
    return true;
  }

  return false;
}

bool IsFuzzBuild() {
  std::string build_mode;
  if (ge::GetContext().GetOption(ge::SHAPE_GENERALIZED_BUILD_MODE, build_mode) != ge::GRAPH_SUCCESS) {
    FE_LOGD("[SubGraphOpt][Compile]There is no fuzz_build flag in ge options");
    return false;
  }

  FE_LOGD("[SubGraphOpt][Compile]Get fuzz_build flag from ge, build mode is %s.", build_mode.c_str());
  if (build_mode == SHAPE_GENERALIZED) {
    return true;
  }

  return false;
}

bool IsFuzzBuildOp(const ge::OpDesc &op_desc) {
  bool support_dyn_shape = false;
  if (IsFuzzBuild() &&
      ge::AttrUtils::GetBool(op_desc, ATTR_NAME_SUPPORT_DYNAMIC_SHAPE, support_dyn_shape) && support_dyn_shape) {
    return true;
  } else {
    return false;
  }
}
}  // namespace fe
