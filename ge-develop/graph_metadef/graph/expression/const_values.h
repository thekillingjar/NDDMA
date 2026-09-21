/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GRAPH_EXPRESSION_CONST_VALUE_H_
#define GRAPH_EXPRESSION_CONST_VALUE_H_

#include <string>
#include <cstdint>
#include "graph/symbolizer/symbolic.h"

namespace ge {
namespace sym {
const uint32_t kNumOne = 1u;
const uint32_t kNumTwo = 2u;
const size_t kBinaryOpFirstArgIdx = 0u;
const size_t kBinaryOpSecondArgIdx = 1u;
const std::string kNullStr = "";
const int32_t kMinusOne = -1;
const int32_t kConstOne = 1;
const int32_t kConstTwo = 2;
const size_t kSizeNumTwo = 2u;
const size_t kIndexOne = 1u;
const size_t kIndexTwo = 2u;
const int32_t kBaseTwo = 2;
const int32_t kMinDimLength = 1;
const Symbol kSymbolOne = ge::Symbol(1, "sym_one");
const Symbol kSymbolZero = ge::Symbol(0, "sym_zero");

// options
const std::string kOutputFilePath = "output_file_path";
const std::string kTilingDataTypeName = "tiling_data_type_name";
const std::string kGenExtraInfo = "gen_extra_info";
const std::string kDumpDebugInfo = "dump_debug_info";
const std::string kGenTilingDataDef = "gen_tiling_data_def";
const std::string kWithTilingContext = "with_tiling_context";
const std::string kDefaultFilePath = "./";
const std::string kDefaultTilingDataTypeName = "TilingData";
const std::string kIsTrue = "1";
const std::string kIsFalse = "0";
const std::string kTilingFuncIdentify = "TilingFunc";
const std::string kDefaultTilingDataFileName = "tiling_data.h";
const std::string kDefaultTilingFuncFileName = "tiling_func.cpp";
}  // namespace sym
}  // namespace ge

#endif  // GRAPH_EXPRESSION_CONST_VALUE_H_