/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef COMMON_UTILS_TRANSFORMER_INC_TRANSFER_SHAPE_ACCORDING_TO_FORMAT_H_
#define COMMON_UTILS_TRANSFORMER_INC_TRANSFER_SHAPE_ACCORDING_TO_FORMAT_H_

#include <memory>
#include "graph/types.h"
#include "graph/ge_tensor.h"
#include "graph/op_desc.h"
#include "platform/platform_info.h"
#include "transfer_def.h"
#include "transfer_shape_utils.h"

namespace transformer {
struct CalcShapeExtraAttr {
  int64_t hidden_size;
  int64_t input_size;
  int64_t state_size;
};

struct ShapeAndFormatInfo {
  ge::GeShape &oldShape;
  const ge::Format &oldFormat;
  const ge::Format &newFormat;
  const ge::DataType &currentDataType;
  CalcShapeExtraAttr extra_attr;
  ShapeAndFormatInfo(ge::GeShape &old_shape, const ge::Format &old_format, const ge::Format &new_format,
                     const ge::DataType &data_type)
                     : oldShape(old_shape), oldFormat(old_format), newFormat(new_format), currentDataType(data_type),
                       extra_attr({1, 1, -1}) {}
};

using ShapeAndFormat = struct ShapeAndFormatInfo;

class ShapeTransferAccordingToFormat {
 public:
  ShapeTransferAccordingToFormat();

  ~ShapeTransferAccordingToFormat() {};

  ShapeTransferAccordingToFormat(const ShapeTransferAccordingToFormat&) = delete;

  ShapeTransferAccordingToFormat &operator=(const ShapeTransferAccordingToFormat&) = delete;

  static bool GetShapeAccordingToFormat(ShapeAndFormat &shapeAndFormatInfo);

  // deprecated ATTRIBUTED_DEPRECATED(static bool GetShapeAccordingToFormat(const ExtAxisOpValue &, ShapeAndFormat &))
  static bool GetShapeAccordingToFormat(const ge::OpDescPtr &op_desc, ShapeAndFormat &shapeAndFormatInfo);

  static bool GetShapeAccordingToFormat(const ExtAxisOpValue &op_value, ShapeAndFormat &shapeAndFormatInfo);

  static bool TransferShape(const ge::Format &origin_format, const ge::Format &format, const ge::DataType &data_type,
                            const ExtAxisValue &ext_axis, ge::GeShape &shape);

  static bool TransferShape(const ge::Format &origin_format, const ge::Format &format, const ge::DataType &data_type,
                            const ExtAxisValue &ext_axis, const ge::GeShape &origin_shape, ge::GeShape &shape);

  /* deprecated ATTRIBUTED_DEPRECATED(static bool TransferShape(const ge::Format &, const ge::Format &, const ge::DataType &,
                        gert::Shape &, const ExtAxisOpValue &,
                        const fe::PlatFormInfos *)) */
  static bool TransferShape(const ge::Format &origin_format, const ge::Format &format, const ge::DataType &data_type,
                            gert::Shape &shape, const ge::OpDescPtr op_desc = nullptr,
                            const fe::PlatFormInfos *platform_infos_ptr = nullptr);
  
  static bool TransferShape(const ge::Format &origin_format, const ge::Format &format, const ge::DataType &data_type,
                            gert::Shape &shape, const ExtAxisOpValue &op_value,
                            const fe::PlatFormInfos *platform_infos_ptr = nullptr);

  /* deprecated ATTRIBUTED_DEPRECATED(static bool TransferShape(const ge::Format &, const ge::Format &, const ge::DataType &,
                        const gert::Shape &, gert::Shape &, const ExtAxisOpValue &)) */
  static bool TransferShape(const ge::Format &origin_format, const ge::Format &format, const ge::DataType &data_type,
                            const gert::Shape &origin_shape, gert::Shape &shape,
                            const ge::OpDescPtr op_desc = nullptr);

  static bool TransferShape(const ge::Format &origin_format, const ge::Format &format, const ge::DataType &data_type,
                            const gert::Shape &origin_shape, gert::Shape &shape, const ExtAxisOpValue &op_value);

  // deprecated ATTRIBUTED_DEPRECATED(static void InitExtAxisValue(const ExtAxisOpValue &, ExtAxisValue &))
  static void InitExtAxisValue(const ge::OpDescPtr &op_desc, ExtAxisValue &ext_axis);

  static void InitExtAxisValue(const ExtAxisOpValue &op_value, ExtAxisValue &ext_axis);

  static bool InitPlatformInfo();
  static int64_t GetC0ByDtype(const ge::DataType &data_type);
  static int64_t GetM0ByDtype(const ge::DataType &data_type);
  static int64_t GetN0ByDtype(const ge::DataType &data_type);
  static bool GetAlignedShape(const AlignShapeInfo &align_shape_info, gert::Shape &aligned_shape);
  static bool TransferDims(const TransferDimsInfo &transfer_dims_info, AxisIndexMapping &axis_index_mapping);
};
} // namespace transformer
#endif  // COMMON_UTILS_TRANSFORMER_INC_TRANSFER_SHAPE_ACCORDING_TO_FORMAT_H_
