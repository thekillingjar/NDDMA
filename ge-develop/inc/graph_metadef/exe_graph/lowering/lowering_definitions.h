/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_RUNTIME_V2_LOWERING_LOWERING_DEFINITIONS_H_
#define AIR_CXX_RUNTIME_V2_LOWERING_LOWERING_DEFINITIONS_H_
#include "graph/types.h"

namespace gert {
constexpr const ge::char_t *kLoweringInputInfo = "_lowering_input_info";
constexpr const ge::char_t *kLoweringResult = "_lowering_result";
constexpr const ge::char_t *kLoweringTensorResult = "_lowering_tensor_result";
constexpr const ge::char_t *kLoweringHostTensorResult = "_lowering_host_tensor_result";
}  // namespace gert
#endif  // AIR_CXX_RUNTIME_V2_LOWERING_LOWERING_DEFINITIONS_H_
