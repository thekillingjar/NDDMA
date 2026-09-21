/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_RUNTIME_V2_OP_IMPL_MAX_POOL_V3_IMPL_H_
#define AIR_CXX_RUNTIME_V2_OP_IMPL_MAX_POOL_V3_IMPL_H_
#include <cstdint>
#include <vector>
#include "register/op_compile_info_base.h"

namespace gert {
struct MaxPoolV3Param {
  int32_t tiling_mode = 0;
  int32_t act_core_num = 0;
  int32_t one_core_ele = 0;
  int32_t last_core_ele = 0;
  int32_t input_h = 0;
  int32_t input_w = 0;
  int32_t output_h = 0;
  int32_t output_w = 0;
  int32_t pad_h = 0;
  int32_t pad_w = 0;
  int32_t pad_t = 0;
  int32_t pad_b = 0;
  int32_t pad_l = 0;
  int32_t pad_r = 0;
  int32_t c_factor = 1;
  int32_t h_factor = 1;
  int32_t w_factor = 1;
  int32_t one_core_loop_num = 0;
  int32_t one_core_loop_left = 0;
  int32_t last_core_loop_num = 0;
  int32_t last_core_loop_left = 0;
  int32_t n_c1 = 0;
};

struct MaxPoolV3CompileInfo : public optiling::CompileInfoBase {
  int32_t ub_ele = 0;
  int32_t core_num = 1;
  int32_t ksize_h = 1;
  int32_t ksize_w = 1;
  int32_t strides_h = 1;
  int32_t strides_w = 1;
  int32_t padding = 0;    // SAME
  int32_t ceil_mode = 0;  // floor
  int32_t pad_top = 0;
  int32_t pad_bottom = 0;
  int32_t pad_left = 0;
  int32_t pad_right = 0;
  int32_t global = 0;
};
}  // namespace gert
#endif  // AIR_CXX_RUNTIME_V2_OP_IMPL_MAX_POOL_V3_IMPL_H_
