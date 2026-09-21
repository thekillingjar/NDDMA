/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef COMMON_UTILS_TRANSFORMER_INC_AXIS_CONSTANTS_H_
#define COMMON_UTILS_TRANSFORMER_INC_AXIS_CONSTANTS_H_

#include <cstdint>
#include <cstddef>
#include <set>
#include "graph/types.h"

namespace transformer {
extern const size_t DIM_SIZE_TWO;
extern const size_t DIM_SIZE_FOUR;
extern const size_t DIM_SIZE_FIVE;
extern const size_t DIM_SIZE_SIX;

extern const size_t EXT_INDEX_INPUT_SIZE;
extern const size_t EXT_INDEX_HIDDEN_SIZE;
extern const size_t EXT_INDEX_STATE_SIZE;
extern const size_t EXT_INDEX_M0_VAL;

extern const int32_t AXIS_NCHW_DIM_N;
extern const int32_t AXIS_NCHW_DIM_C;
extern const int32_t AXIS_NCHW_DIM_H;
extern const int32_t AXIS_NCHW_DIM_W;

extern const int32_t AXIS_NHWC_DIM_N;
extern const int32_t AXIS_NHWC_DIM_H;
extern const int32_t AXIS_NHWC_DIM_W;
extern const int32_t AXIS_NHWC_DIM_C;

extern const int32_t AXIS_HWCN_DIM_H;
extern const int32_t AXIS_HWCN_DIM_W;
extern const int32_t AXIS_HWCN_DIM_C;
extern const int32_t AXIS_HWCN_DIM_N;

extern const int32_t AXIS_CHWN_DIM_C;
extern const int32_t AXIS_CHWN_DIM_H;
extern const int32_t AXIS_CHWN_DIM_W;
extern const int32_t AXIS_CHWN_DIM_N;

extern const int32_t NDHWC_DIM_N;
extern const int32_t NDHWC_DIM_D;
extern const int32_t NDHWC_DIM_H;
extern const int32_t NDHWC_DIM_W;
extern const int32_t NDHWC_DIM_C;

extern const int32_t NCDHW_DIM_N;
extern const int32_t NCDHW_DIM_C;
extern const int32_t NCDHW_DIM_D;
extern const int32_t NCDHW_DIM_H;
extern const int32_t NCDHW_DIM_W;

extern const int32_t DHWCN_DIM_D;
extern const int32_t DHWCN_DIM_H;
extern const int32_t DHWCN_DIM_W;
extern const int32_t DHWCN_DIM_C;
extern const int32_t DHWCN_DIM_N;

extern const int32_t DHWNC_DIM_D;
extern const int32_t DHWNC_DIM_H;
extern const int32_t DHWNC_DIM_W;
extern const int32_t DHWNC_DIM_N;
extern const int32_t DHWNC_DIM_C;

extern const int32_t AXIS_NC1HWC0_DIM_N;
extern const int32_t AXIS_NC1HWC0_DIM_C1;
extern const int32_t AXIS_NC1HWC0_DIM_H;
extern const int32_t AXIS_NC1HWC0_DIM_W;
extern const int32_t AXIS_NC1HWC0_DIM_C0;

extern const int32_t AXIS_C1HWNCoC0_DIM_C1;
extern const int32_t AXIS_C1HWNCoC0_DIM_H;
extern const int32_t AXIS_C1HWNCoC0_DIM_W;
extern const int32_t AXIS_C1HWNCoC0_DIM_N;
extern const int32_t AXIS_C1HWNCoC0_DIM_Co;

const std::set<ge::Format> kFormatNZSet = {ge::FORMAT_FRACTAL_NZ, ge::FORMAT_FRACTAL_NZ_C0_2,
                                           ge::FORMAT_FRACTAL_NZ_C0_4, ge::FORMAT_FRACTAL_NZ_C0_8,
                                           ge::FORMAT_FRACTAL_NZ_C0_16, ge::FORMAT_FRACTAL_NZ_C0_32};

} // namespace transformer

#endif // COMMON_UTILS_TRANSFORMER_INC_AXIS_CONSTANTS_H_
