/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef SAMPLEPROJECT_ATT_SAMPLE_TESTS_TILING_ST_UTILS_H_
#define SAMPLEPROJECT_ATT_SAMPLE_TESTS_TILING_ST_UTILS_H_
#include <cstdio>
#include <vector>
#include <string>
#include "graph/ascendc_ir/ascendc_ir_core/ascendc_ir.h"
#include "common/checker.h"
namespace att {
constexpr char kTilingDataEntrance[] = "tiling_data";
constexpr char kTilingContextEntrance[] = "tiling_context";
constexpr char kAttToolScence[] = "tool";
constexpr char kAutofuseScence[] = "autofuse";
struct InputArgs {
  std::string output_path{"./"};
  std::string tiling_entrance{"tiling_data"};
  std::string op_name{"OpTest"};
  std::string scence{"tool"};
};
class GenExecTilingUtils {
 public:
  static ge::Status GenExecFunc(std::vector<ge::AscGraph> &graphs, const InputArgs &input_args);

 private:
  static ge::Status GenInputJson(std::vector<ge::AscGraph> &graphs, const std::string &output_path);
  static ge::Status GenExecFunc(std::vector<ge::AscGraph> &graphs, const std::string &op_name,
                                const std::string &output_path);
  static ge::Status GenContextEntranceExecFunc(std::vector<ge::AscGraph> &graphs, const std::string &op_name,
                                               const std::string &output_path);
};
}  // namespace att

#endif  // SAMPLEPROJECT_ATT_SAMPLE_TESTS_TILING_ST_UTILS_H_
