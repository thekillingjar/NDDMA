/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FUSION_ENGINE_UTILS_COMMON_WEIGHT_COMPRESS_FLAG_H_
#define FUSION_ENGINE_UTILS_COMMON_WEIGHT_COMPRESS_FLAG_H_

#include <map>
#include <memory>
#include <string>
#include <vector>
#include "common/fe_inner_error_codes.h"
#include "common/fe_utils.h"
#include "common/format/axis_util.h"
#include "common/util/op_info_util.h"
#include "graph/compute_graph.h"
namespace fe {
enum class WEIGHCOMPRESSINNERFLAG {
  DISABLE_COMPRESS_FLAG = 0,
  WEIGHT_COMPRESS_FLAG = 1,
  FOUR_TO_TWO_FLAG = 2
};
constexpr char const *kWeightSparseFourToTwo = "weight_sparse_4_2";

extern WEIGHCOMPRESSINNERFLAG JudgeIsSparsityFlag();
extern bool ReadWeightConfig();
extern Status TranseWeight2SparsityFZShapeAndFormat(const ge::OpDescPtr &conv_desc,
                                                    const ge::GeShape &src_shape,
                                                    const ge::Format &src_format,
                                                    const ge::DataType &src_type,
                                                    ge::GeShape &dst_shape,
                                                    const std::pair<bool, bool> &flag);
extern Status SetSparsityCompressWeightShape(const ge::OpDescPtr &conv_desc, const ge::GeTensorDescPtr &tensor_desc_ptr,
                                             const std::pair<bool, bool> &flag);

Status RefreshConvShapeForSpasity(const ge::OpDescPtr &conv_desc);

Status RefreshCompressShapeForSpasity(const ge::OpDescPtr &conv_desc, const ge::OpDescPtr &compress_opdesc);

ge::GeShape ReshapeConvWeight(const ge::GeShape &src_shape,
                              const ge::Format &src_format);
}  // namespace fe

#endif
