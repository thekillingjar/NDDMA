/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef __ASCENDC_API_TRANSPOSE_BASE_TYPE_H__
#define __ASCENDC_API_TRANSPOSE_BASE_TYPE_H__

#include <vector>
#include <map>

#include <stdint.h>

enum class AutoFuseTransposeType : uint8_t {
  TRANSPOSE_ND2ND_ONLY = 0,
  TRANSPOSE_ND2ND_102 = 1,
  TRANSPOSE_ND2ND_0213 = 2,
  TRANSPOSE_ND2ND_2103 = 3,
  TRANSPOSE_ND2ND_021 = 4,
  TRANSPOSE_ND2ND_210 = 5,
  TRANSPOSE_ND2ND_0321 = 6,
  TRANSPOSE_INVALID = 7
};
using Permutation = std::vector<uint32_t>;
struct PermuteParam {
  AutoFuseTransposeType true_transpose_type;
  std::vector<std::vector<uint32_t>> potential_axis_idx;
};
// permutation类型转换表
const std::map<Permutation, PermuteParam> kPermutationTable = {
    {{1, 0}, {AutoFuseTransposeType::TRANSPOSE_ND2ND_ONLY, {}}},
    {{0, 2, 1}, {AutoFuseTransposeType::TRANSPOSE_ND2ND_021, {}}},
    {{1, 0, 2}, {AutoFuseTransposeType::TRANSPOSE_ND2ND_102, {}}},
    {{1, 2, 0}, {AutoFuseTransposeType::TRANSPOSE_ND2ND_ONLY, {{1, 2}}}},
    {{2, 0, 1}, {AutoFuseTransposeType::TRANSPOSE_ND2ND_ONLY, {{0, 1}}}},
    {{2, 1, 0}, {AutoFuseTransposeType::TRANSPOSE_ND2ND_210, {}}},
    {{0, 1, 3, 2}, {AutoFuseTransposeType::TRANSPOSE_ND2ND_021, {{0, 1}}}},
    {{0, 2, 1, 3}, {AutoFuseTransposeType::TRANSPOSE_ND2ND_0213, {}}},
    {{0, 2, 3, 1}, {AutoFuseTransposeType::TRANSPOSE_ND2ND_021, {{2, 3}}}},
    {{0, 3, 1, 2}, {AutoFuseTransposeType::TRANSPOSE_ND2ND_021, {{1, 2}}}},
    {{0, 3, 2, 1}, {AutoFuseTransposeType::TRANSPOSE_ND2ND_0321, {}}},
    {{1, 0, 2, 3}, {AutoFuseTransposeType::TRANSPOSE_ND2ND_102, {{2, 3}}}},
    {{1, 2, 0, 3}, {AutoFuseTransposeType::TRANSPOSE_ND2ND_102, {{1, 2}}}},
    {{1, 2, 3, 0}, {AutoFuseTransposeType::TRANSPOSE_ND2ND_ONLY, {{1, 2, 3}}}},
    {{2, 0, 1, 3}, {AutoFuseTransposeType::TRANSPOSE_ND2ND_102, {{0, 1}}}},
    {{2, 1, 0, 3}, {AutoFuseTransposeType::TRANSPOSE_ND2ND_2103, {}}},
    {{2, 3, 0, 1}, {AutoFuseTransposeType::TRANSPOSE_ND2ND_ONLY, {{2, 3}, {0, 1}}}},
    {{2, 3, 1, 0}, {AutoFuseTransposeType::TRANSPOSE_ND2ND_210, {{2, 3}}}},
    {{3, 0, 1, 2}, {AutoFuseTransposeType::TRANSPOSE_ND2ND_ONLY, {{0, 1, 2}}}},
    {{3, 1, 2, 0}, {AutoFuseTransposeType::TRANSPOSE_ND2ND_210, {{1, 2}}}},
    {{3, 2, 0, 1}, {AutoFuseTransposeType::TRANSPOSE_ND2ND_210, {{0, 1}}}},
};

#endif  // __ASCENDC_API_TRANSPOSE_BASE_TYPE_H__