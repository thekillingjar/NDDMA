/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef TRANSFORMER_INC_TRANSFER_DEF_H_
#define TRANSFORMER_INC_TRANSFER_DEF_H_

#include <memory.h>
#include <array>
#include "exe_graph/runtime/shape.h"

namespace transformer {

const size_t ORIGIN_FORMAT_DIM_SIZE = 5;
const size_t EXT_AXIS_SIZE = 4;
const size_t EXT_AXIS_OP_SIZE = 3;


struct AlignShapeInfo {
  ge::Format src_format;
  ge::Format dst_format;
  gert::Shape src_shape;
  ge::DataType data_type;
  int64_t reshape_type_mask;
};

struct TransferDimsInfo {
  ge::Format src_format;
  ge::Format dst_format;
  gert::Shape src_shape;
  int64_t reshape_type_mask;
};

struct AxisIndexMapping {
  std::vector<std::vector<int32_t>> src_to_dst_transfer_dims;
  std::vector<std::vector<int32_t>> dst_to_src_transfer_dims;
};

using GetAlignedShapeFunc = std::function<bool(const AlignShapeInfo &, gert::Shape &)>;
using TransferDimsFunc = std::function<bool(const TransferDimsInfo &, AxisIndexMapping &)>;

using FormatIndex = std::array<size_t, ORIGIN_FORMAT_DIM_SIZE>;
using ExtAxisValue = std::array<int64_t, EXT_AXIS_SIZE>;
using ExtAxisOpValue = std::array<int64_t, EXT_AXIS_OP_SIZE>;

}  // namespace transformer
#endif  // TRANSFORMER_INC_TRANSFER_DEF_H_
