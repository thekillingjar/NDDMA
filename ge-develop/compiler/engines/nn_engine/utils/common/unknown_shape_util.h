/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FUSION_ENGINE_UTILS_COMMON_UNKNOWN_SHAPE_UTIL_H_
#define FUSION_ENGINE_UTILS_COMMON_UNKNOWN_SHAPE_UTIL_H_

#include <map>
#include <string>
#include <vector>
#include "common/fe_inner_error_codes.h"
#include "common/fe_log.h"
#include "common/fe_utils.h"
#include "common/range_format_transfer/transfer_range_according_to_format.h"
#include "ops_store/op_kernel_info.h"

namespace fe {
/*
 *  @ingroup fe
 *  @brief   get sting of shape range.
 *  @param   [in]  input or output tensor.
 *  @return  shape range
 */
std::string ShapeRangeToStr(const std::vector<std::pair<int64_t, int64_t>> &shape_range);

/*
 *  @ingroup fe
 *  @brief   get shape range of unknown shape op.
 *  @param   [in]  input or output tensor.
 *  @return  shape range
 */
std::vector<std::pair<int64_t, int64_t>> GetShapeRange(const ge::GeTensorDesc &tensor_desc);

std::vector<std::pair<int64_t, int64_t>> GetOriginShapeRange(const ge::GeTensorDesc &tensor_desc);

std::vector<std::pair<int64_t, int64_t>> GetValueRange(const ge::GeTensorDesc &tensor_desc);
/*
 *  @ingroup fe
 *  @brief   get new shape range aligned to shape.
 *  @param   [in]  input or output tensor.
 *  @return  shape range
 */
std::vector<std::pair<int64_t, int64_t>> GetAlignShapeRange(
    const std::vector<std::pair<int64_t, int64_t>> &ori_shape_range, const ge::GeShape &shape);

Status CalcShapeRange(const ge::OpDescPtr &op_desc, const ge::Format &final_format,
                      const ge::GeShape &dimension_expanded_ori_shape, ge::GeTensorDesc &tensor_desc);

/*
 *  @ingroup fe
 *  @brief   check whether fuzzy build.
 *  @param   [in]  nothing.
 *  @return  true: is fuzzy build; false: is not fuzzy build
 */
bool IsShapeGeneralizedMode();

bool IsFuzzBuild();

bool IsFuzzBuildOp(const ge::OpDesc &op_desc);
}  // namespace fe

#endif  // FUSION_ENGINE_UTILS_COMMON_UNKNOWN_SHAPE_UTIL_H_
