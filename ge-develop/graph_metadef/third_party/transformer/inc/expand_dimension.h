/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef COMMON_UTILS_TRANSFORMER_INC_EXPAND_DIMENSION_H_
#define COMMON_UTILS_TRANSFORMER_INC_EXPAND_DIMENSION_H_

#include <string>
#include <vector>
#include "graph/types.h"
#include "graph/ge_tensor.h"
#include "exe_graph/runtime/shape.h"
#include "transfer_def.h"

namespace transformer {
 /* Pad dimension according to reshape type */
bool ExpandDimension(const std::string &op_type, const ge::Format &original_format, const ge::Format &final_format,
                     const uint32_t &tensor_index, const std::string &reshape_type, ge::GeShape &shape);

bool ExpandRangeDimension(const std::string &op_type, const ge::Format &original_format, const ge::Format &final_format,
                          const uint32_t &tensor_index, const std::string &reshape_type,
                          std::vector<std::pair<int64_t, int64_t>> &ranges);

class ExpandDimension {
 public:
  ExpandDimension();
  ~ExpandDimension();

  static int64_t GenerateReshapeType(const ge::Format &origin_format, const ge::Format &format,
                                     const size_t &origin_dim_size, const std::string &reshape_type);
  static bool GenerateReshapeType(const ge::Format &origin_format, const ge::Format &format,
                                  const size_t &origin_dim_size, const std::string &reshape_type,
                                  int64_t &reshape_type_mask);
  static bool GenerateReshapeTypeByMask(const ge::Format &origin_format, const size_t &origin_dim_size,
                                        const int64_t &reshape_type_mask, std::string &reshape_type,
                                        std::string &failed_reason);
  static void ExpandDims(const int64_t &reshape_type, ge::GeShape &shape);
  static void ExpandDims(const int64_t &reshape_type, const ge::GeShape &origin_shape, ge::GeShape &shape);
  static void ExpandDims(const int64_t &reshape_type, gert::Shape &shape);
  static void ExpandDims(const int64_t &reshape_type, const gert::Shape &origin_shape, gert::Shape &shape);
  static bool GetDefaultReshapeType(const ge::Format &origin_format, const size_t &origin_dim_size,
                                    std::string &reshape_type);
  static int32_t GetAxisIndexByName(char ch, const ge::Format &format);
  static int64_t GetReshapeAxicValue(const int64_t &reshape_type_mask,
                                     const ge::GeShape &shape, int32_t axis_index);
  static int64_t GetReshapeAxicValueByName(const int64_t &reshape_type_mask, char ch,
                                           const ge::GeShape &shape, const ge::Format &format);
  static bool GetFormatFullSize(const ge::Format &format, size_t &full_size);
 private:
  static bool IsNeedExpand(const ge::Format &origin_format, const ge::Format &format,
                           const size_t &origin_dim_size, const size_t &full_size, const std::string &reshape_type);
  static bool IsReshapeTypeValid(const ge::Format &origin_format, const size_t &origin_dim_size,
                                 const std::string &reshape_type);
};
} // namespace transformer
#endif // COMMON_UTILS_TRANSFORMER_INC_EXPAND_DIMENSION_H_
 