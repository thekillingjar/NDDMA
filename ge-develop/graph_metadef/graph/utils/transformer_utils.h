/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef COMMON_GRAPH_UTILS_TRANSFORMER_UTILS_H_
#define COMMON_GRAPH_UTILS_TRANSFORMER_UTILS_H_
#include <string>
#include <map>

#include "graph/types.h"
#include "graph/op_desc.h"
#include "graph/ge_tensor.h"
#include "graph/small_vector.h"
#include "graph/ascend_limits.h"

namespace ge {

class NodeShapeTransUtils {
 public:
  bool Init();
  bool CatchFormatAndShape();
  bool UpdateFormatAndShape();

  explicit NodeShapeTransUtils(const OpDescPtr op_desc) : op_desc_(op_desc), in_num_(0U), out_num_(0U) {
  }

  ~NodeShapeTransUtils() {
  }

 private:
  SmallVector<Format, kDefaultMaxInputNum> map_format_in_;
  SmallVector<Format, kDefaultMaxInputNum> map_ori_format_in_;
  SmallVector<DataType, kDefaultMaxInputNum> map_dtype_in_;
  SmallVector<Format, kDefaultMaxOutputNum> map_format_out_;
  SmallVector<Format, kDefaultMaxOutputNum> map_ori_format_out_;
  SmallVector<DataType, kDefaultMaxOutputNum> map_dtype_out_;

  OpDescPtr op_desc_;
  size_t in_num_;
  size_t out_num_;
};
}  // namespace ge
#endif  // COMMON_GRAPH_UTILS_TRANSFORMER_UTILS_H_
