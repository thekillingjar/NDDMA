/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef COMMON_UTILS_TRANSFORMER_INC_TRANSFER_SHAPE_ACCORDING_TO_FORMAT_EXT_H_
#define COMMON_UTILS_TRANSFORMER_INC_TRANSFER_SHAPE_ACCORDING_TO_FORMAT_EXT_H_

#include <memory>
#include "graph/types.h"
#include "transfer_def.h"

namespace transformer {
class ShapeTransferAccordingToFormatExt {
 public:
  ShapeTransferAccordingToFormatExt();

  ~ShapeTransferAccordingToFormatExt() {};

  ShapeTransferAccordingToFormatExt(const ShapeTransferAccordingToFormatExt&) = delete;

  ShapeTransferAccordingToFormatExt &operator=(const ShapeTransferAccordingToFormatExt&) = delete;

  static bool TransferShape(const ge::Format &origin_format, const ge::Format &format, const ge::DataType &data_type,
                            gert::Shape &shape, const ExtAxisOpValue &op_value,
                            const fe::PlatFormInfos *platform_infos_ptr = nullptr);

  static bool TransferShape(const ge::Format &origin_format, const ge::Format &format, const ge::DataType &data_type,
                          const gert::Shape &origin_shape, gert::Shape &shape, const ExtAxisOpValue &op_value);
};
} // namespace transformer
#endif  // COMMON_UTILS_TRANSFORMER_INC_TRANSFER_SHAPE_ACCORDING_TO_FORMAT_EXT_H_