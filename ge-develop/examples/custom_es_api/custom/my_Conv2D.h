/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_COMPILER_GRAPH_EAGER_STYLE_GRAPH_BUILDER_MY_OP_Conv2D_CPP_H_
#define AIR_CXX_COMPILER_GRAPH_EAGER_STYLE_GRAPH_BUILDER_MY_OP_Conv2D_CPP_H_
#include <utility>
#include "es_tensor_holder.h"
#include "es_graph_builder.h"
#include "my_Conv2D_c.h"
namespace ge {
namespace es {

inline EsTensorHolder MyConv2D(EsTensorHolder &x, ge::Format x_format, EsTensorHolder &filter,
         ge::Format filter_format, const EsTensorHolder &bias,
         const EsTensorHolder &offset_w, const std::vector<int64_t> &strides,
         const std::vector<int64_t> &pads,
         const std::vector<int64_t> &dilations = {1, 1, 1, 1},
         int64_t groups = 1, const char *data_format = "NHWC",
         int64_t offset_x = 0, const char *my_attr = "my_attr") {
  auto out = MyEsConv2D(
      x.GetCTensorHolder(), x_format, filter.GetCTensorHolder(), filter_format, bias.GetCTensorHolder(),
      offset_w.GetCTensorHolder(), strides.data(),
      static_cast<int64_t>(strides.size()), pads.data(),
      static_cast<int64_t>(pads.size()), dilations.data(),
      static_cast<int64_t>(dilations.size()), groups, data_format, offset_x, my_attr);
  return out;
}
}  // namespace es
}  // namespace ge
#endif
