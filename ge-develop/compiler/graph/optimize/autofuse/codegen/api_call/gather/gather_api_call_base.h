/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ASCGEN_GATHER_API_CALL_BASE_H
#define ASCGEN_GATHER_API_CALL_BASE_H
#include <sstream>
#include "codegen_kernel.h"

namespace gather_base {
using namespace codegen;

std::string CalGatherOuterAxesSize(const std::vector<ascir::AxisId> &param_outer_axes,
                                   const std::vector<ascir::AxisId> &indices_axes, size_t index, const TPipe &tpipe);
std::string CalGatherOuterAxesIndex(std::string outer_axis_offset, const std::vector<ascir::AxisId> &param_outer_axes,
                                    const std::vector<ascir::AxisId> &indices_axes, const TPipe &tpipe);
bool IsAxisInParamAxes(ascir::AxisId axis_id, const std::vector<ascir::AxisId> &param_axes, const TPipe &tpipe);
std::string CalGatherOuterAxisOffset(const std::vector<ascir::AxisId> &current_axis,
                                     const std::vector<ascir::AxisId> &param_inner_axes,
                                     ascir::AxisId tile_inner_axis_id, const TPipe &tpipe);
void CollectParamOuterAndInnerAxes(const std::vector<ascir::AxisId> &param_axis, ascir::AxisId gather_axis_id,
                                   std::vector<ascir::AxisId> &param_outer_axes,
                                   std::vector<ascir::AxisId> &param_inner_axes);
std::string CalGatherParamOffset(const std::vector<ascir::AxisId> &param_axis, std::string indices_value,
                                 ascir::AxisId gather_axis_id, const Axis &inner_vectorized_axis, const TPipe &tpipe);
std::string CalGatherIndicesAxesSize(const std::vector<ascir::AxisId> &indices_axes, const TPipe &tpipe);
std::string CalGatherOuterSize(const std::vector<ascir::AxisId> &param_axis, ascir::AxisId gather_axis_id, const TPipe &tpipe);
std::string CalGatherInnerSize(const std::vector<ascir::AxisId> &param_axis, ascir::AxisId gather_axis_id, const TPipe &tpipe);
std::string CalGatherSize(const std::vector<ascir::AxisId> &param_axis, const TPipe &tpipe);
}  // namespace codegen
#endif  // ASCGEN_GATHER_API_CALL_BASE_H