/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "transfer_shape_according_to_format.h"
#include "transfer_shape_according_to_format_ext.h"
#include <algorithm>
#include "axis_constants.h"
#include "graph/utils/attr_utils.h"
#include "transfer_shape_utils.h"

namespace transformer {
namespace {
  const std::string kAttrHiddenSize = "hidden_size";
  const std::string kAttrInputSize = "input_size";
  const std::string kAttrStateSize = "state_size";
  const int64_t kM0DefaultVal = 16;

  void GeShapeToRtShape(const ge::GeShape &ge_shape, gert::Shape &rt_shape) {
    rt_shape.SetDimNum(0);
    for (size_t i = 0; i < ge_shape.GetDimNum(); ++i) {
      rt_shape.AppendDim(ge_shape.GetDim(i));
    }
  }

  void RtShapeToGeShape(const gert::Shape &rt_shape, ge::GeShape &ge_shape) {
    ge_shape.SetDimNum(0);
    for (size_t i = 0; i < rt_shape.GetDimNum(); ++i) {
      ge_shape.AppendDim(rt_shape.GetDim(i));
    }
  }
}

ShapeTransferAccordingToFormat::ShapeTransferAccordingToFormat() {}

bool ShapeTransferAccordingToFormat::GetShapeAccordingToFormat(const ge::OpDescPtr &op_desc,
                                                               ShapeAndFormat &shapeAndFormatInfo) {
  if (shapeAndFormatInfo.oldShape.IsUnknownDimNum()) {
    return true;
  }
  gert::Shape shape;
  GeShapeToRtShape(shapeAndFormatInfo.oldShape, shape);
  ExtAxisValue ext_axis;
  InitExtAxisValue(op_desc, ext_axis);
  bool ret = TransferShapeUtils::TransferShape(shapeAndFormatInfo.oldFormat, shapeAndFormatInfo.newFormat,
                                               shapeAndFormatInfo.currentDataType, ext_axis, shape);
  RtShapeToGeShape(shape, shapeAndFormatInfo.oldShape);
  return ret;
}

bool ShapeTransferAccordingToFormat::GetShapeAccordingToFormat(const ExtAxisOpValue &op_value,
                                                               ShapeAndFormat &shapeAndFormatInfo) {
  if (shapeAndFormatInfo.oldShape.IsUnknownDimNum()) {
    return true;
  }
  gert::Shape shape;
  GeShapeToRtShape(shapeAndFormatInfo.oldShape, shape);
  ExtAxisValue ext_axis;
  InitExtAxisValue(op_value, ext_axis);
  bool ret = TransferShapeUtils::TransferShape(shapeAndFormatInfo.oldFormat, shapeAndFormatInfo.newFormat,
                                               shapeAndFormatInfo.currentDataType, ext_axis, shape);
  RtShapeToGeShape(shape, shapeAndFormatInfo.oldShape);
  return ret;
}

bool ShapeTransferAccordingToFormat::GetShapeAccordingToFormat(ShapeAndFormat &shapeAndFormatInfo) {
  if (shapeAndFormatInfo.oldShape.IsUnknownDimNum()) {
    return true;
  }
  gert::Shape shape;
  GeShapeToRtShape(shapeAndFormatInfo.oldShape, shape);
  ExtAxisValue ext_axis = {shapeAndFormatInfo.extra_attr.input_size, shapeAndFormatInfo.extra_attr.hidden_size,
                           shapeAndFormatInfo.extra_attr.state_size, kM0DefaultVal};
  bool ret = TransferShapeUtils::TransferShape(shapeAndFormatInfo.oldFormat, shapeAndFormatInfo.newFormat,
                                               shapeAndFormatInfo.currentDataType, ext_axis, shape);
  RtShapeToGeShape(shape, shapeAndFormatInfo.oldShape);
  return ret;
}

bool ShapeTransferAccordingToFormat::TransferShape(const ge::Format &origin_format, const ge::Format &format,
                                                   const ge::DataType &data_type, const ExtAxisValue &ext_axis,
                                                   ge::GeShape &shape) {
  gert::Shape rt_shape;
  GeShapeToRtShape(shape, rt_shape);
  bool ret = TransferShapeUtils::TransferShape(origin_format, format, data_type, ext_axis, rt_shape);
  RtShapeToGeShape(rt_shape, shape);
  return ret;
}

bool ShapeTransferAccordingToFormat::TransferShape(const ge::Format &origin_format, const ge::Format &format,
                                                   const ge::DataType &data_type, const ExtAxisValue &ext_axis,
                                                   const ge::GeShape &origin_shape, ge::GeShape &shape) {
  gert::Shape rt_origin_shape;
  GeShapeToRtShape(origin_shape, rt_origin_shape);
  gert::Shape rt_shape;
  GeShapeToRtShape(shape, rt_shape);
  bool ret = TransferShapeUtils::TransferShape(origin_format, format, data_type, ext_axis, rt_origin_shape, rt_shape);
  RtShapeToGeShape(rt_shape, shape);
  return ret;
}

bool ShapeTransferAccordingToFormat::TransferShape(const ge::Format &origin_format, const ge::Format &format,
                                                   const ge::DataType &data_type, gert::Shape &shape,
                                                   const ge::OpDescPtr op_desc, const fe::PlatFormInfos *platform_infos_ptr) {
  ExtAxisValue ext_axis;
  InitExtAxisValue(op_desc, ext_axis);
  return TransferShapeUtils::TransferShape(origin_format, format, data_type, ext_axis, shape, platform_infos_ptr);
}

bool ShapeTransferAccordingToFormat::TransferShape(const ge::Format &origin_format, const ge::Format &format,
                                                   const ge::DataType &data_type, gert::Shape &shape,
                                                   const ExtAxisOpValue &op_value,
                                                   const fe::PlatFormInfos *platform_infos_ptr) {
  ExtAxisValue ext_axis;
  InitExtAxisValue(op_value, ext_axis);
  return TransferShapeUtils::TransferShape(origin_format, format, data_type, ext_axis, shape, platform_infos_ptr);
}

bool ShapeTransferAccordingToFormat::TransferShape(const ge::Format &origin_format, const ge::Format &format,
                                                   const ge::DataType &data_type, const gert::Shape &origin_shape,
                                                   gert::Shape &shape, const ge::OpDescPtr op_desc) {
  ExtAxisValue ext_axis;
  InitExtAxisValue(op_desc, ext_axis);
  return TransferShapeUtils::TransferShape(origin_format, format, data_type, ext_axis, origin_shape, shape);
}

bool ShapeTransferAccordingToFormat::TransferShape(const ge::Format &origin_format, const ge::Format &format,
                                                   const ge::DataType &data_type, const gert::Shape &origin_shape,
                                                   gert::Shape &shape, const ExtAxisOpValue &op_value) {
  ExtAxisValue ext_axis;
  InitExtAxisValue(op_value, ext_axis);
  return TransferShapeUtils::TransferShape(origin_format, format, data_type, ext_axis, origin_shape, shape);
}

void ShapeTransferAccordingToFormat::InitExtAxisValue(const ge::OpDescPtr &op_desc, ExtAxisValue &ext_axis) {
  int64_t input_size = 1;
  int64_t hidden_size = 1;
  int64_t state_size = -1;
  if (op_desc != nullptr) {
    (void)ge::AttrUtils::GetInt(op_desc, kAttrInputSize, input_size);
    (void)ge::AttrUtils::GetInt(op_desc, kAttrHiddenSize, hidden_size);
    (void)ge::AttrUtils::GetInt(op_desc, kAttrStateSize, state_size);
  }

  ext_axis[EXT_INDEX_INPUT_SIZE] = input_size;
  ext_axis[EXT_INDEX_HIDDEN_SIZE] = hidden_size;
  ext_axis[EXT_INDEX_STATE_SIZE] = state_size;
  ext_axis[EXT_INDEX_M0_VAL] = kM0DefaultVal;
}

void ShapeTransferAccordingToFormat::InitExtAxisValue(const ExtAxisOpValue &op_value, ExtAxisValue &ext_axis) {
  ext_axis[EXT_INDEX_INPUT_SIZE] = op_value[EXT_INDEX_INPUT_SIZE];
  ext_axis[EXT_INDEX_HIDDEN_SIZE] = op_value[EXT_INDEX_HIDDEN_SIZE];
  ext_axis[EXT_INDEX_STATE_SIZE] = op_value[EXT_INDEX_STATE_SIZE];
  ext_axis[EXT_INDEX_M0_VAL] = kM0DefaultVal;
}

bool ShapeTransferAccordingToFormat::InitPlatformInfo() {
  return TransferShapeUtils::InitPlatformInfo();
}

int64_t ShapeTransferAccordingToFormat::GetC0ByDtype(const ge::DataType &data_type) {
  return TransferShapeUtils::GetC0ByDtype(data_type);
}
int64_t ShapeTransferAccordingToFormat::GetM0ByDtype(const ge::DataType &data_type) {
  return TransferShapeUtils::GetM0ByDtype(data_type);
}
int64_t ShapeTransferAccordingToFormat::GetN0ByDtype(const ge::DataType &data_type) {
  return TransferShapeUtils::GetN0ByDtype(data_type);
}

bool ShapeTransferAccordingToFormat::GetAlignedShape(const AlignShapeInfo &align_shape_info, gert::Shape &aligned_shape) {
  return TransferShapeUtils::GetAlignedShape(align_shape_info, aligned_shape);
}

bool ShapeTransferAccordingToFormat::TransferDims(const TransferDimsInfo &transfer_dims_info, AxisIndexMapping &axis_index_mapping) {
  return TransferShapeUtils::TransferDims(transfer_dims_info, axis_index_mapping);
}

bool ShapeTransferAccordingToFormatExt::TransferShape(const ge::Format &origin_format, const ge::Format &format,
                                                      const ge::DataType &data_type, gert::Shape &shape,
                                                      const ExtAxisOpValue &op_value,
                                                      const fe::PlatFormInfos *platform_infos_ptr) {
  return ShapeTransferAccordingToFormat::TransferShape(origin_format, format, data_type, shape, op_value,
                                                       platform_infos_ptr);
}

bool ShapeTransferAccordingToFormatExt::TransferShape(const ge::Format &origin_format, const ge::Format &format,
                                                      const ge::DataType &data_type, const gert::Shape &origin_shape,
                                                      gert::Shape &shape, const ExtAxisOpValue &op_value) {
  return ShapeTransferAccordingToFormat::TransferShape(origin_format, format, data_type, origin_shape,
                                                       shape, op_value);
}
} // namespace transformer
