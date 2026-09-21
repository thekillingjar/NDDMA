/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FUSION_ENGINE_OPTIMIZER_COMMON_UTIL_CONSTANTS_H_
#define FUSION_ENGINE_OPTIMIZER_COMMON_UTIL_CONSTANTS_H_

#include "common/aicore_util_constants.h"

namespace fe {
/* type of AI core and vector core */
constexpr char AI_CORE_TYPE[] = "AiCore";
constexpr char VECTOR_CORE_TYPE[] = "VectorCore";
constexpr char CUBE_CORE_TYPE[] = "CubeCore";
constexpr char TBE_L2_MODE[] = "l2_mode";

// fusion switch
constexpr char const *UB_FUSION = "UBFusion";
constexpr char const *GRAPH_FUSION = "GraphFusion";
// min coefficients of reduce op which needed ub size
const int REDUCE_COEFFICIENTS = 2;
}  // namespace fe
#endif  // FUSION_ENGINE_OPTIMIZER_COMMON_UTIL_CONSTANTS_H_
