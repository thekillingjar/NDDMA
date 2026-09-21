/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_COMPILER_GRAPH_EAGER_STYLE_GRAPH_BUILDER_MY_OP_IdentityN_CPP_H_
#define AIR_CXX_COMPILER_GRAPH_EAGER_STYLE_GRAPH_BUILDER_MY_OP_IdentityN_CPP_H_
#include <utility>
#include "es_tensor_holder.h"
#include "es_graph_builder.h"
#include "my_IdentityN_c.h"
namespace ge {
namespace es {

/**
 * @note user needs to provide following inputs for dynamic output numbers:
 *   y_num: dynamic output number of y
 */
inline std::vector<EsTensorHolder> MyIdentityN(const std::vector<EsTensorHolder> &x) {
  auto esb_x = TensorsToEsCTensorHolders(x);
  auto out = MyEsIdentityN(esb_x.data(), static_cast<int64_t>(esb_x.size()));
  std::vector<EsTensorHolder> y_dynamic_outs;
  y_dynamic_outs.reserve(out.y_num);
  for (int64_t dyn_idx = 0; dyn_idx < out.y_num; ++dyn_idx) {
    y_dynamic_outs.emplace_back(out.y[dyn_idx]);
  }
  return {y_dynamic_outs};
}
}  // namespace es
}  // namespace ge
#endif
