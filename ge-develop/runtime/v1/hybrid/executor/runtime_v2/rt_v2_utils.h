/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_HYBRID_EXECUTOR_RT_V2_UTILS_H_
#define GE_HYBRID_EXECUTOR_RT_V2_UTILS_H_

#include "exe_graph/runtime/tensor.h"
#include "graph/utils/type_utils.h"
#include "graph/ge_tensor.h"

namespace ge {
namespace hybrid {
std::string DebugString(const gert::Shape &shape);

std::string DebugString(const gert::TensorPlacement &placement);

std::string DebugString(const ge::Placement &placement);

std::string DebugString(const gert::Tensor &tensor, bool show_rt_attr = true);

void DimsAsShape(const std::vector<int64_t> &dims, gert::Shape &shape);

void SmallVecDimsAsRtShape(const SmallVector<int64_t, kDefaultDimsNum> &dims, gert::Shape &shape);

void RtShapeAsGeShape(const gert::Shape &shape, ge::GeShape &ge_shape);

void GeShapeAsRtShape(const ge::GeShape &ge_shape, gert::Shape &gert_shape);

std::string DebugString(const GeShape &shape);

std::string DebugString(const GeTensorDesc &desc);
}  // namespace hybrid
}  // namespace ge

#endif  // GE_HYBRID_EXECUTOR_RT_V2_UTILS_H_
