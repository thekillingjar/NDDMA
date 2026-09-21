/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_RUNTIME_V2_OP_IMPL_DYNAMIC_RNN_IMPL_H_
#define AIR_CXX_RUNTIME_V2_OP_IMPL_DYNAMIC_RNN_IMPL_H_
#include <cstdint>
#include <vector>
#include "register/op_compile_info_base.h"

namespace gert {
struct DynamicRnnV3Param {
  int32_t sequenceLength;
  int32_t dynamicRnnBatch;
  int32_t chequeIndex;
};

struct DynamicRNNV3CompileInfo : public optiling::CompileInfoBase {
  std::vector<std::vector<int64_t>> tune_shape_list;  // todo 这玩意儿可否用固定长度的二维数组搞定？
};
}  // namespace gert
#endif  // AIR_CXX_RUNTIME_V2_OP_IMPL_DYNAMIC_RNN_IMPL_H_
