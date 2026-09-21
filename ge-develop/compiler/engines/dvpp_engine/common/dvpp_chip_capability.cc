/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "dvpp_chip_capability.h"
#include "util/dvpp_constexpr.h"
#include "util/dvpp_define.h"
#include "util/dvpp_log.h"
#include "util/util.h"

namespace dvpp {
bool CheckSupportedDims(const DvppOpInfo& dvpp_op_info,
    const std::vector<int64_t>& dims, ge::Format data_format,
    std::string& unsupported_reason) {
  // 对于动态shape -2 跳过校验
  DVPP_CHECK_IF_THEN_DO(dims[kNum0] == kDynamicShape, return true);

  // 针对 FORMAT_ND 需要推测实际 FORMAT
  bool ret = InferDataFormatND(dims, data_format, unsupported_reason);
  DVPP_CHECK_IF_THEN_DO(!ret, return false);

  // 获取 n c width height
  int64_t n = 0;
  int64_t c = 0;
  int64_t width = 0;
  int64_t height = 0;
  GetDimsValueByDataFormat(dims, data_format, n, c, width, height);

  // N 对于动态dim -1 跳过校验
  DVPP_CHECK_IF_THEN_DO((n != kNum1) && (n != kDynamicDim),
      unsupported_reason = "N[" + std::to_string(n) + "] should be 1";
      return false);

  // C 对于动态dim -1 跳过校验
  DVPP_CHECK_IF_THEN_DO((c != kNum1) && (c != kNum3) && (c != kDynamicDim),
      unsupported_reason = "C[" + std::to_string(c) + "] should be 3";
      return false);

  // width 对于动态dim -1 跳过校验
  ret = IsInRange(width, dvpp_op_info.widthMin, dvpp_op_info.widthMax);
  DVPP_CHECK_IF_THEN_DO((width != kDynamicDim) && (!ret),
      unsupported_reason = "width[" + std::to_string(width) +
          "] should be in range[" + std::to_string(dvpp_op_info.widthMin) +
          "," + std::to_string(dvpp_op_info.widthMax) + "]";
      return false);

  // height 对于动态dim -1 跳过校验
  ret = IsInRange(height, dvpp_op_info.heightMin, dvpp_op_info.heightMax);
  DVPP_CHECK_IF_THEN_DO((height != kDynamicDim) && (!ret),
      unsupported_reason = "height[" + std::to_string(height) +
          "] should be in range[" + std::to_string(dvpp_op_info.heightMin) +
          "," + std::to_string(dvpp_op_info.heightMax) + "]";
      return false);

  return true;
}

bool CheckSupportedInput(const DvppOpInfo& dvpp_op_info,
    const ge::NodePtr& node, std::string& unsupported_reason,
    const uint32_t index) {
  auto op_desc_ptr = node->GetOpDesc();
  auto input = op_desc_ptr->GetInputDesc(index);
  auto input_dims = input.GetShape().GetDims();
  DVPP_CHECK_IF_THEN_DO(input_dims.empty(),
      unsupported_reason = "op[" + op_desc_ptr->GetType() + "] input[" +
                           std::to_string(index) + "] dims is empty";
      DVPP_ENGINE_LOG_EVENT("%s", unsupported_reason.c_str());
      return false);

  auto input_data_format = input.GetFormat();
  bool ret = CheckSupportedDims(
      dvpp_op_info, input_dims, input_data_format, unsupported_reason);
  DVPP_CHECK_IF_THEN_DO(!ret,
      unsupported_reason = "op[" + op_desc_ptr->GetType() + "] input[" +
                           std::to_string(index) + "] " + unsupported_reason;
      DVPP_ENGINE_LOG_EVENT("%s", unsupported_reason.c_str());
      return false);

  return true;
}

bool CheckSupportedOutput(const DvppOpInfo& dvpp_op_info,
    const ge::NodePtr& node, std::string& unsupported_reason,
    const uint32_t index) {
  auto op_desc_ptr = node->GetOpDesc();
  auto output = op_desc_ptr->GetOutputDesc(index);
  auto output_dims = output.GetShape().GetDims();
  DVPP_CHECK_IF_THEN_DO(output_dims.empty(),
      unsupported_reason = "op[" + op_desc_ptr->GetType() + "] output[" +
                           std::to_string(index) + "] dims is empty";
      DVPP_ENGINE_LOG_EVENT("%s", unsupported_reason.c_str());
      return false);

  auto output_data_format = output.GetFormat();
  bool ret = CheckSupportedDims(
      dvpp_op_info, output_dims, output_data_format, unsupported_reason);
  DVPP_CHECK_IF_THEN_DO(!ret,
      unsupported_reason = "op[" + op_desc_ptr->GetType() + "] output[" +
                           std::to_string(index) + "] " + unsupported_reason;
      DVPP_ENGINE_LOG_EVENT("%s", unsupported_reason.c_str());
      return false);

  return true;
}


bool GetPaddings(const ge::OpDescPtr& op_desc_ptr, const std::string& attr_name,
    std::vector<std::vector<int64_t>>& paddings_array) {
  bool ret = ge::AttrUtils::GetListListInt(op_desc_ptr, attr_name,
      paddings_array);
  DVPP_CHECK_IF_THEN_DO(!ret,
      DVPP_ENGINE_LOG_EVENT(
          "Call GetListListInt function failed to get op[%s] attr[%s]",
          op_desc_ptr->GetType().c_str(), attr_name.c_str());
      return false);

  size_t dims = paddings_array.size();
  DVPP_CHECK_IF_THEN_DO((dims != kNum4),
      DVPP_ENGINE_LOG_EVENT(
          "Get paddings dims %zu should be 4, op[%s] attr[%s]",
          dims, op_desc_ptr->GetType().c_str(), attr_name.c_str());
      return false);

  for (size_t i = 0; i < dims; ++i) {
    size_t len = paddings_array[i].size();
    DVPP_CHECK_IF_THEN_DO((len != kNum2),
        DVPP_ENGINE_LOG_EVENT(
            "Get each dims value num %zu should be 2, op[%s] attr[%s]",
            len, op_desc_ptr->GetType().c_str(), attr_name.c_str());
        return false);
    for (size_t j = 0; j < len; ++j) {
      DVPP_ENGINE_LOG_DEBUG("paddings[%zu][%zu] = %ld", i, j,
          paddings_array[i][j]);
    }
  }

  return true;
}

bool GetPaddingDimIndex(const ge::OpDescPtr& op_desc_ptr, uint32_t& hPos,
    uint32_t& wPos, uint32_t& cPos) {
  size_t input_size = op_desc_ptr->GetInputsSize();
  DVPP_CHECK_IF_THEN_DO((input_size == kNum0),
      DVPP_ENGINE_LOG_EVENT(
          "op[%s] input num %zu should be bigger than 0",
          op_desc_ptr->GetType().c_str(), input_size);
      return false);
  auto input_desc = op_desc_ptr->GetInputDesc(0);
  auto data_format = static_cast<ge::Format>(
      ge::GetPrimaryFormat(static_cast<int32_t>(input_desc.GetFormat())));

  hPos = kNum1; // NHWC
  wPos = kNum2;
  cPos = kNum3;
  if (data_format == ge::FORMAT_NCHW) {
    cPos = kNum1;
    hPos = kNum2;
    wPos = kNum3;
  }
  return true;
}

bool CheckAttrPaddingsForPadV3D(const ge::OpDescPtr& op_desc_ptr,
    const std::string& attr_name, const std::string& check_value) {
  // 1. 获取用户配置的paddings二维数组
  std::vector<std::vector<int64_t>> paddings_array;
  bool ret = GetPaddings(op_desc_ptr, attr_name, paddings_array);
  DVPP_CHECK_IF_THEN_DO(!ret, return false);

  // 2. 获取padding宽度范围限制
  std::vector<int64_t> result;
  SplitSequence(check_value, kTilde, result);
  DVPP_CHECK_IF_THEN_DO((result.size() != kNum2),
      DVPP_ENGINE_LOG_EVENT(
          "Get ListListInt limit value num %zu should be 2, op[%s] attr[%s]",
          result.size(), op_desc_ptr->GetType().c_str(), attr_name.c_str());
      return false);

  // 3. 根据输入格式获取padding通道顺序，从而校验各个维度的padding宽度
  uint32_t hPos = kNum0;
  uint32_t wPos = kNum0;
  uint32_t cPos = kNum0;
  ret = GetPaddingDimIndex(op_desc_ptr, hPos, wPos, cPos);
  DVPP_CHECK_IF_THEN_DO(!ret, return false);

  // N, C维度，只支持padding 0
  ret = (paddings_array[kNum0][kNum0] == kNum0) &&
        (paddings_array[kNum0][kNum1] == kNum0) &&
        (paddings_array[cPos][kNum0] == kNum0) &&
        (paddings_array[cPos][kNum1] == kNum0);
  DVPP_CHECK_IF_THEN_DO((!ret),
      DVPP_ENGINE_LOG_EVENT(
          "ListListInt paddings, dim N and C only support 0, op[%s], attr[%s]",
          op_desc_ptr->GetType().c_str(), attr_name.c_str());
      return false);

  // H, W维度，支持padding宽度在ini文件中的范围以内
  ret = (paddings_array[hPos][kNum0] >= result[kNum0]) &&
        (paddings_array[hPos][kNum0] <= result[kNum1]) &&
        (paddings_array[hPos][kNum1] >= result[kNum0]) &&
        (paddings_array[hPos][kNum1] <= result[kNum1]) &&
        (paddings_array[wPos][kNum0] >= result[kNum0]) &&
        (paddings_array[wPos][kNum0] <= result[kNum1]) &&
        (paddings_array[wPos][kNum1] >= result[kNum0]) &&
        (paddings_array[wPos][kNum1] <= result[kNum1]);
  DVPP_CHECK_IF_THEN_DO((!ret),
      DVPP_ENGINE_LOG_EVENT(
          "paddings, dim H and W only support [%ld, %ld], op[%s], attr[%s]",
          result[kNum0], result[kNum1], op_desc_ptr->GetType().c_str(),
          attr_name.c_str());
      return false);
  return true;
}

bool CheckAttrForPadV3D(const DvppOpInfo& dvpp_op_info,
    const ge::NodePtr& node, std::string& unsupported_reason) {
  auto op_desc_ptr = node->GetOpDesc();
  auto op_desc_attrs = op_desc_ptr->GetAllAttrs();
  for (auto& attr : dvpp_op_info.attrs) {
    std::string attr_name = attr.first;
    DVPP_ENGINE_LOG_DEBUG("attr_name = %s", attr_name.c_str());
    if (attr_name != "paddings") {
      continue;
    }
    const auto iter = op_desc_attrs.find(attr_name);
    DVPP_CHECK_IF_THEN_DO((iter == op_desc_attrs.end()),
        unsupported_reason = "op[" + op_desc_ptr->GetType() +
            "]: does not have attr[" + attr_name + "]";
        DVPP_ENGINE_LOG_EVENT("%s", unsupported_reason.c_str());
        return false);

    // check attr value
    const auto ge_value_type = iter->second.GetValueType();
    std::string attr_value = attr.second.value;
    DVPP_CHECK_IF_THEN_DO((ge_value_type !=
        ge::GeAttrValue::ValueType::VT_LIST_LIST_INT),
        unsupported_reason = "do not support op[" + op_desc_ptr->GetType() +
            "] attr[" + attr_name + "] value";
        DVPP_ENGINE_LOG_EVENT("%s", unsupported_reason.c_str());
        return false);

    return CheckAttrPaddingsForPadV3D(op_desc_ptr, attr_name, attr_value);
  }
  return true;
}

bool CheckAttrsForRotate(const ge::NodePtr& node,
    std::string& unsupported_reason) {
  auto op_desc_ptr = node->GetOpDesc();
  auto op_desc_attrs = op_desc_ptr->GetAllAttrs();
  const string center_attr = "center";
  const auto iter = op_desc_attrs.find(center_attr);
  DVPP_CHECK_IF_THEN_DO((iter == op_desc_attrs.end()),
      unsupported_reason = "op[" + op_desc_ptr->GetType() +
          "]: does not have center attr.";
      DVPP_ENGINE_LOG_EVENT("%s", unsupported_reason.c_str());
      return false);

  std::vector<int64_t> center{};
  DVPP_CHECK_IF_THEN_DO(!ge::AttrUtils::GetListInt(op_desc_ptr, center_attr,
      center),
      DVPP_ENGINE_LOG_EVENT(
          "Call GetListInt function failed to get op[%s] attr[%s]",
          op_desc_ptr->GetType().c_str(), center_attr.c_str());
      return false);

  DVPP_CHECK_IF_THEN_DO((!center.empty() && center.size() != 2),
      DVPP_ENGINE_LOG_EVENT("op[%s] attr[%s] size should be 0 or 2.",
          op_desc_ptr->GetType().c_str(), center_attr.c_str());
      return false);

  return true;
}

// 只需校验输入输出的普通算子通过这个校验即可
bool CommonOpCheck(const DvppOpInfo& dvpp_op_info,
    const ge::NodePtr& node, std::string& unsupported_reason) {
  // check input_images
  bool ret = CheckSupportedInput(dvpp_op_info, node, unsupported_reason, kNum0);
  DVPP_CHECK_IF_THEN_DO(!ret, return false);

  // check output_y
  ret = CheckSupportedOutput(dvpp_op_info, node, unsupported_reason, kNum0);
  DVPP_CHECK_IF_THEN_DO(!ret, return false);

  return true;
}

// 无需校验的算子通过这个校验即可
bool NoNeedToCheck(const DvppOpInfo& dvpp_op_info,
    const ge::NodePtr& node, std::string& unsupported_reason) {
  // 这里算子输入format为NHWC 会被优化为HWC 且H和W均为动态shape的-1 所以无需校验
  (void)dvpp_op_info;
  (void)node;
  (void)unsupported_reason;
  return true;
}

// Crop
bool Crop(const DvppOpInfo& dvpp_op_info,
    const ge::NodePtr& node, std::string& unsupported_reason) {
  // check attr
  auto op_desc_ptr = node->GetOpDesc();
  auto data_format = op_desc_ptr->GetInputDesc(0).GetFormat();
  int64_t attr_axis_value = 0;
  DVPP_CHECK_IF_THEN_DO(!ge::AttrUtils::GetInt(
      op_desc_ptr, "axis", attr_axis_value),
      DVPP_REPORT_INNER_ERR_MSG("get op[%s] attr axis failed.",
                              op_desc_ptr->GetType().c_str());
      return false);

  if (data_format == ge::FORMAT_NHWC) {
    DVPP_CHECK_IF_THEN_DO(attr_axis_value != kNum1,
        unsupported_reason = "op[" + op_desc_ptr->GetType() +
                             "] only support axis be 1 when NHWC";
        return false);
  } else { // data_format == ge::FORMAT_NCHW, 其他情况走不到这里
    DVPP_CHECK_IF_THEN_DO(attr_axis_value != kNum2,
        unsupported_reason = "op[" + op_desc_ptr->GetType() +
                             "] only support axis be 2 when NCHW";
        return false);
  }

  // check input_x
  bool ret = CheckSupportedInput(dvpp_op_info, node, unsupported_reason, kNum0);
  DVPP_CHECK_IF_THEN_DO(!ret, return false);

  // check output_y
  ret = CheckSupportedOutput(dvpp_op_info, node, unsupported_reason, kNum0);
  DVPP_CHECK_IF_THEN_DO(!ret, return false);

  return true;
}

// PadV3D
bool PadV3D(const DvppOpInfo& dvpp_op_info,
    const ge::NodePtr& node, std::string& unsupported_reason) {
  // check input_x
  bool ret = CheckSupportedInput(dvpp_op_info, node, unsupported_reason, kNum0);
  DVPP_CHECK_IF_THEN_DO(!ret, return false);

  // check output_y
  ret = CheckSupportedOutput(dvpp_op_info, node, unsupported_reason, kNum0);
  DVPP_CHECK_IF_THEN_DO(!ret, return false);

  // check padding attr
  ret = CheckAttrForPadV3D(dvpp_op_info, node, unsupported_reason);
  DVPP_CHECK_IF_THEN_DO(!ret, return false);

  return true;
}

// ReverseV2
bool ReverseV2(const DvppOpInfo& dvpp_op_info,
    const ge::NodePtr& node, std::string& unsupported_reason) {
  // 校验input_x 无需校验output_y 无法校验axis
  return CheckSupportedInput(dvpp_op_info, node, unsupported_reason, kNum0);
}

// Rotate
bool Rotate(const DvppOpInfo& dvpp_op_info,
    const ge::NodePtr& node, std::string& unsupported_reason) {
  // check input_x
  bool ret = CheckSupportedInput(dvpp_op_info, node, unsupported_reason, kNum0);
  DVPP_CHECK_IF_THEN_DO(!ret, return false);

  // check output_y
  ret = CheckSupportedOutput(dvpp_op_info, node, unsupported_reason, kNum0);
  DVPP_CHECK_IF_THEN_DO(!ret, return false);

  ret = CheckAttrsForRotate(node, unsupported_reason);
  DVPP_CHECK_IF_THEN_DO(!ret, return false);

  return true;
}

using FuntionType =
    std::function<bool(const DvppOpInfo&, const ge::NodePtr&, std::string&)>;
const std::map<std::string, FuntionType> kDvppChipCapability = {
  {"AdjustBrightness", CommonOpCheck},
  {"AdjustBrightnessV2", CommonOpCheck},
  {"AdjustContrast", CommonOpCheck},
  {"AdjustContrastWithMean", CommonOpCheck},
  {"AdjustHue", CommonOpCheck},
  {"AdjustSaturation", CommonOpCheck},
  {"AdjustSaturationV2", CommonOpCheck},
  {"Crop", Crop},
  {"CropAndResize", CommonOpCheck},
  {"CropAndResizeV2", CommonOpCheck},
  {"DecodeAndCropJpeg", NoNeedToCheck},
  {"DecodeJpeg", NoNeedToCheck},
  {"GaussianBlur", CommonOpCheck},
  {"ImgCrop", CommonOpCheck},
  {"ImgToTensor", CommonOpCheck},
  {"NormalizeV2", CommonOpCheck},
  {"PadV3D", PadV3D},
  {"Resize", CommonOpCheck},
  {"ResizeV2", CommonOpCheck},
  {"ResizeBicubic", CommonOpCheck},
  {"ResizeBilinearV2", CommonOpCheck},
  {"ResizeNearestNeighborV2", CommonOpCheck},
  {"ReverseV2", ReverseV2},
  {"RgbToGrayscale", CommonOpCheck},
  {"Rotate", Rotate},
  {"WarpAffineV2", CommonOpCheck},
  {"WarpPerspective", CommonOpCheck},
  {"YUVToRGB", NoNeedToCheck}
};

bool DvppChipCapability::CheckSupported(const DvppOpInfo& dvpp_op_info,
    const ge::NodePtr& node, std::string& unsupported_reason) {
  std::string op_type = node->GetOpDesc()->GetType();
  auto iter = kDvppChipCapability.find(op_type);
  DVPP_CHECK_IF_THEN_DO(iter == kDvppChipCapability.end(),
      unsupported_reason =
          "dvpp chip capability does not support op[" + op_type + "]";
      DVPP_ENGINE_LOG_EVENT("%s", unsupported_reason.c_str());
      return false);

  // 动态shape下 当前无法确认的参数 默认支持
  auto check_function = iter->second;
  bool ret = check_function(dvpp_op_info, node, unsupported_reason);
  DVPP_CHECK_IF_THEN_DO(!ret,
      DVPP_ENGINE_LOG_EVENT("%s", unsupported_reason.c_str()));
  return ret;
}
} // namespace dvpp
