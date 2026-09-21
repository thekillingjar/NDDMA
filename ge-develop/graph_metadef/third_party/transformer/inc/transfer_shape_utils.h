/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef TRANSFORMER_INC_TRANSFER_SHAPE_UTILS_H_
#define TRANSFORMER_INC_TRANSFER_SHAPE_UTILS_H_
 
#include <array>
#include "platform/platform_info.h"
#include "axis_util.h"
#include "transfer_def.h"

namespace transformer {
enum class TransferShapeType {
  ND_TO_ND = 0,
  ND_TO_NZ,
  FULL_SIZE,
  NOT_FULL_SIZE,
  INVALID
};

class TransferShapeUtils {
 public:
  TransferShapeUtils() {}
  ~TransferShapeUtils() {}
  static bool InitPlatformInfo(const fe::PlatFormInfos *platform_infos_ptr = nullptr);
  static bool TransferShape(const ge::Format &origin_format, const ge::Format &format, const ge::DataType &data_type,
                            const ExtAxisValue &ext_axis, gert::Shape &shape,
                            const fe::PlatFormInfos *platforminfos = nullptr);
  static bool TransferShape(const ge::Format &origin_format, const ge::Format &format, const ge::DataType &data_type,
                            const ExtAxisValue &ext_axis, const gert::Shape &origin_shape, gert::Shape &shape,
                            const fe::PlatFormInfos *platforminfos = nullptr);
  static int64_t GetC0ByDtype(const ge::DataType &data_type);
  static int64_t GetM0ByDtype(const ge::DataType &data_type);
  static int64_t GetN0ByDtype(const ge::DataType &data_type);
  static bool GetAlignedShape(const AlignShapeInfo &align_shape_info, gert::Shape &aligned_shape);
  static bool TransferDims(const TransferDimsInfo &transfer_dims_info, AxisIndexMapping &axis_index_mapping);

 private:
  static bool InitM0K0CO(const fe::PlatFormInfos *platform_infos);
  static bool TransferShapeByFormat(const ge::Format &primary_format, const AxisValue &axis_value,
                                    gert::Shape &shape);
  static bool TransferShapeByAxisValue(const ge::Format &primary_format, const AxisValue &axis_value,
                                       gert::Shape &shape);
  static bool TransferShapeByOriginShape(const ge::Format &primary_format, const int64_t &c0, const int64_t &m0,
                                         const ExtAxisValue &ext_axis, const gert::Shape &origin_shape,
                                         gert::Shape &shape);
  static bool TransferShapeByFormatIndex(const ge::Format &origin_format, const ge::Format &format, const int64_t &c0,
                                         const gert::Shape &origin_shape, gert::Shape &shape);
  static bool IsNeedTransferShape(const ge::Format &origin_format, const ge::Format &format, const gert::Shape &shape);
  static bool CheckInputParam(const ge::Format &origin_format, const ge::Format &primary_format,
                              const ge::DataType &data_type);
  static bool IsNeedAxisValue(const ge::Format &format, const size_t &origin_dim_size);
  static int64_t GetC0Value(const ge::DataType &data_type, const ge::Format &format);

  /* ----------Below is the function of getting new shape by axis value---------------------- */
  static bool GetNCHWShapeByAxisValue(const AxisValue &axis_value, gert::Shape &shape);

  static bool GetNHWCShapeByAxisValue(const AxisValue &axis_value, gert::Shape &shape);

  static bool GetHWCNShapeByAxisValue(const AxisValue &axis_value, gert::Shape &shape);

  static bool GetCHWNShapeByAxisValue(const AxisValue &axis_value, gert::Shape &shape);

  static bool GetNDHWCShapeByAxisValue(const AxisValue &axis_value, gert::Shape &shape);

  static bool GetNCDHWShapeByAxisValue(const AxisValue &axis_value, gert::Shape &shape);

  static bool GetDHWCNShapeByAxisValue(const AxisValue &axis_value, gert::Shape &shape);

  static bool GetDHWNCShapeByAxisValue(const AxisValue &axis_value, gert::Shape &shape);

  static bool GetNC1HWC0ShapeByAxisValue(const AxisValue &axis_value, gert::Shape &shape);

  static bool GetC1HWC0ShapeByAxisValue(const AxisValue &axis_value, gert::Shape &shape);

  static bool GetNDC1HWC0ShapeByAxisValue(const AxisValue &axis_value, gert::Shape &shape);

  static bool GetC1HWNCoC0ShapeByAxisValue(const AxisValue &axis_value, gert::Shape &shape);

  static bool GetNzShapeByAxisValue(const AxisValue &axis_value, gert::Shape &shape);

  static bool GetFzShapeByAxisValue(const AxisValue &axis_value, gert::Shape &shape);

  static bool GetFz3DShapeByAxisValue(const AxisValue &axis_value, gert::Shape &shape);

  static bool GetFz3DTransposeShapeByAxisValue(const AxisValue &axis_value, gert::Shape &shape);

  static bool GetFzLstmShapeByAxisValue(const AxisValue &axis_value, gert::Shape &shape);

  static bool GetFzC04ShapeByAxisValue(const AxisValue &axis_value, gert::Shape &shape);

  static bool GetFznRNNShapeByAxisValue(const AxisValue &axis_value, gert::Shape &shape);

  static bool GetNDRNNShapeByAxisValue(const AxisValue &axis_value, gert::Shape &shape);

  /* ----------Below is the function of getting new shape by origin shape---------------------- */
  static bool GetNCHWShape(const FormatIndex& format_index, const gert::Shape &origin_shape, gert::Shape &shape);

  static bool GetNHWCShape(const FormatIndex& format_index, const gert::Shape &origin_shape, gert::Shape &shape);

  static bool GetHWCNShape(const FormatIndex& format_index, const gert::Shape &origin_shape, gert::Shape &shape);

  static bool GetCHWNShape(const FormatIndex& format_index, const gert::Shape &origin_shape, gert::Shape &shape);

  static bool GetNDHWCShape(const FormatIndex& format_index, const gert::Shape &origin_shape, gert::Shape &shape);

  static bool GetNCDHWShape(const FormatIndex& format_index, const gert::Shape &origin_shape, gert::Shape &shape);

  static bool GetDHWCNShape(const FormatIndex& format_index, const gert::Shape &origin_shape, gert::Shape &shape);

  static bool GetDHWNCShape(const FormatIndex& format_index, const gert::Shape &origin_shape, gert::Shape &shape);

  static bool GetNC1HWC0Shape(const FormatIndex& format_index, const int64_t &c0, const gert::Shape &origin_shape,
                              gert::Shape &shape);

  static bool GetC1HWC0Shape(const FormatIndex& format_index, const int64_t &c0, const gert::Shape &origin_shape,
                             gert::Shape &shape);

  static bool GetNDC1HWC0Shape(const FormatIndex& format_index, const int64_t &c0, const gert::Shape &origin_shape,
                               gert::Shape &shape);

  static bool GetC1HWNCoC0Shape(const FormatIndex& format_index, const int64_t &c0, const gert::Shape &origin_shape,
                                gert::Shape &shape);

  static bool GetFractalNzShape(const int64_t &c0, const int64_t &m0,
                                const gert::Shape &origin_shape, gert::Shape &shape);

  static bool GetFractalZShape(const int64_t &c0, const gert::Shape &origin_shape, gert::Shape &shape);

  static bool GetFractalZShape(const FormatIndex& format_index, const int64_t &c0, const int64_t &group,
                               const gert::Shape &origin_shape, gert::Shape &shape);

  static bool GetFractalZ3DShape(const FormatIndex& format_index, const int64_t &c0, const int64_t &group,
                                 const gert::Shape &origin_shape, gert::Shape &shape);

  static bool GetFractalZ3DTransposeShape(const FormatIndex& format_index, const int64_t &c0,
                                          const gert::Shape &origin_shape, gert::Shape &shape);

  static bool GetFractalZLstmShape(const FormatIndex& format_index, const gert::Shape &origin_shape,
                                   gert::Shape &shape);

  static bool GetFractalZC04Shape(const FormatIndex& format_index, const int64_t &c0, const gert::Shape &origin_shape,
                                  gert::Shape &shape);

  static bool GetFractalZnRnnShape(const ExtAxisValue &ext_axis, const int64_t &c0, const gert::Shape &origin_shape,
                                   gert::Shape &shape);

  static bool GetNdRnnBiasShape(const ExtAxisValue &ext_axis, const int64_t &c0, const gert::Shape &origin_shape,
                                gert::Shape &shape);

  static bool GetNYUVShape(gert::Shape &shape);

  static bool GetFzWinoShapeByAxisValue(const AxisValue &axis_value, gert::Shape &shape);

  static bool GetFractalZWinoShape(const FormatIndex& format_index, const int64_t &c0,
                                   const gert::Shape &origin_shape, gert::Shape &shape);

  static TransferShapeType GetTransferShapeType(const ge::Format &src_format, const ge::Format &dst_format,
                                                const gert::Shape &src_shape);

  static bool GetNdToNdAlignedShape(const AlignShapeInfo &align_shape_info, gert::Shape &aligned_shape);

  static bool GetNdToNzAlignedShape(const AlignShapeInfo &align_shape_info, gert::Shape &aligned_shape);

  static bool GetFullSizeAlignedShape(const AlignShapeInfo &align_shape_info, gert::Shape &aligned_shape);

  static bool GetNotFullSizeAlignedShape(const AlignShapeInfo &align_shape_info, gert::Shape &aligned_shape);

  static bool GetNdToNdAxisIndexMapping(const TransferDimsInfo &transfer_dims_info,
                                        AxisIndexMapping &axis_index_mapping);

  static bool GetNdToNzAxisIndexMapping(const TransferDimsInfo &transfer_dims_info,
                                        AxisIndexMapping &axis_index_mapping);

  static bool GetFullSizeAxisIndexMapping(const TransferDimsInfo &transfer_dims_info,
                                          AxisIndexMapping &axis_index_mapping);

  static bool GetNotFullSizeAxisIndexMapping(const TransferDimsInfo &transfer_dims_info,
                                             AxisIndexMapping &axis_index_mapping);

  static std::array<uint32_t, static_cast<size_t>(ge::DataType::DT_MAX)> m0_list_;
  static std::array<uint32_t, static_cast<size_t>(ge::DataType::DT_MAX)> k0_list_;
  static std::array<uint32_t, static_cast<size_t>(ge::DataType::DT_MAX)> n0_list_;
  static const std::map<TransferShapeType, GetAlignedShapeFunc> get_aligned_shape_func_map;
  static const std::map<TransferShapeType, TransferDimsFunc> transfer_dims_func_map;
};
}
#endif  // TRANSFORMER_INC_TRANSFER_SHAPE_UTILS_H_
 