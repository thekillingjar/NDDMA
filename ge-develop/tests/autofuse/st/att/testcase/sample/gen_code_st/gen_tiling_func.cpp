/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <iostream>
#include <fstream>
#include <iostream>
#include <sstream>
#include <cstdlib>
#include "base/att_const_values.h"
#include "custom_ascend_graph.h"
#include "graph/graph.h"
#include "gen_tiling_impl.h"
#include "utils/gen_exec_tiling_utils.h"
#include "user_input_parser.h"
#include "common/checker.h"
#include "../utils/graph_construct_utils.h"
using namespace att;
namespace {
constexpr int32_t kMaxArgsSize = 5;
constexpr int32_t kTilingEntranceId = 1;
constexpr int32_t kOutputPathId = 2;
constexpr int32_t kOpNameId = 3;
constexpr int32_t kSenceId = 4;
const std::set<std::string> kValidTilingEntrance = {kTilingDataEntrance, kTilingContextEntrance};
const std::set<std::string> kValidScence = {kAttToolScence, kAutofuseScence};
}
bool CheckTilingEntranceValid(const std::string &tiling_entrace) {
  return kValidTilingEntrance.find(tiling_entrace) != kValidTilingEntrance.cend();
}
bool CheckSenceValid(const std::string &scence) {
  return kValidScence.find(scence) != kValidScence.cend();
}

std::string ValidRangeStr(const std::set<std::string> &valid) {
  std::string valid_tiling_entrace_range_str;
  for (const auto &iter : valid) {
    valid_tiling_entrace_range_str.append(iter).append(",");
  }
  return valid_tiling_entrace_range_str;
}

int ParseInputArgs(int32_t argc, char *argv[], InputArgs &input_args, std::map<std::string, std::string> &options) {
  GE_ASSERT_TRUE(CheckTilingEntranceValid(argv[1]), "Check tiling entrance failed, argv[1]=%s(valid range is %s)",
                 argv[1], ValidRangeStr(kValidTilingEntrance).c_str());
  if (argc < (kTilingEntranceId + 1)) {
    return 0;
  }
  input_args.tiling_entrance = argv[kTilingEntranceId];
  if (argc < (kOutputPathId + 1)) {
    return 0;
  }
  input_args.output_path = argv[kOutputPathId];
  if (input_args.tiling_entrance == kTilingContextEntrance) {
    options["with_tiling_context"] = "1";
    options["gen_extra_info"] = "1";
  }
  if (argc < (kOpNameId + 1)) {
    return 0;
  }
  input_args.op_name = argv[kOpNameId];
  if (strlen(argv[kSenceId]) == 0) {
    input_args.scence = "tool";
  } else {
    input_args.scence = argv[kSenceId];
  }
  GE_ASSERT_TRUE(CheckSenceValid(input_args.scence), "Check scence failed, scence=%s(valid range is %s)",
                 argv[4], ValidRangeStr(kValidScence).c_str());
  return 0;
}
std::string RemoveAutoFuseTilingHeadGuards(const std::string &input) {
  std::istringstream iss(input);
  std::ostringstream oss;
  std::string line;
  const std::string guard_token = "__AUTOFUSE_TILING_FUNC_COMMON_H__";

  while (std::getline(iss, line)) {
    // 如果当前行不包含 guard_token，则保留
    if (line.find(guard_token) == std::string::npos) {
      oss << line << "\n";
    }
  }

  return oss.str();
}

void CombineTilings(const std::map<std::string, std::string> &tilings, std::string &result) {
  const std::string tiling_head = "TilingHead";  // TilingHead作为开头拼接其他文件
  const std::string tiling_data = "TilingData";  // 要排除的 TilingData 子串
  result += RemoveAutoFuseTilingHeadGuards(tilings.at(tiling_head));  // 删除头文件的宏保护，cpp文件不需要
  const std::string include_str = "#include \"autofuse_tiling_func_common.h\"";

  // 遍历所有非 TilingHead 和 TilingData 的条目，去掉第一行后拼接
  for (const auto &[key, value] : tilings) {
    if (key == tiling_head || key.find(tiling_data) != std::string::npos) {
      continue;
    }

    // 查找并跳过第一行头文件行
    size_t include_pos = value.find(include_str);
    if (include_pos != std::string::npos) {
      // 找到 include 行，跳过它，并去掉后面的换行符
      size_t content_start = include_pos + include_str.length();
      while (content_start < value.size() && (value[content_start] == '\n' || value[content_start] == '\r')) {
        content_start++;
      }
      result += value.substr(content_start);
    } else {
      // 如果没有 include 行，直接拼接整个内容
      result += value;
    }

    if (!result.empty() && result.back() != '\n') {
      result += '\n';
    }
  }
}
// arg1: tiling mode:[tiling_data, tiling_context]
// arg2: output_path: default is "./"
// arg3: op_name: default is "OpTest"
int main(int32_t argc, char *argv[]) {
  std::string op_name = "OpTest";
  GE_ASSERT_TRUE(argc <= kMaxArgsSize, "Get argc[%d] is invalid, valid range is [0, %d].", argc, kMaxArgsSize);
  // user graph define
  std::vector<ge::AscGraph> graphs;
  std::map<std::string, std::string> options;
  GE_ASSERT_SUCCESS(GenerateAscGraphs(graphs));
  GeneratorAttOptions(options);
  // default options
  options["solver_type"] = "AxesReorder";
  // gen tiling st
  InputArgs input_args;
  GE_ASSERT_SUCCESS(ParseInputArgs(argc, argv, input_args, options));
  int32_t size = 0;
  for (const auto &graph : graphs) {
    for (const auto &node : graph.GetAllNodes()) {
      size++;
    }
  }
  GraphConstructUtils::UpdateGraphsVectorizedStride(graphs);
  GE_ASSERT_TRUE(size != 0,
                 "Please implement the Ascend graph construction first, then proceed with the gen code compilation.");
  // gen tiling code
  if (input_args.scence == kAttToolScence) {
    GELOGI("Enter att tool scence.\n");
    GE_ASSERT_TRUE(GenTilingImpl(input_args.op_name, graphs, options), "Gen tiling implement failed.");
  } else if (input_args.scence == kAutofuseScence) {
    GELOGI("Enter att autofuse scence.\n");
    // for gen tiling data
    options.emplace(kGenTilingDataDef, "1");
    GE_ASSERT_TRUE(GenTilingImpl(input_args.op_name, graphs, options), "Gen tiling implement failed.");
    // for gen tiling func
    ascir::FusedScheduledResult fused_schedule_result;
    std::vector<ascir::ScheduledResult> scheduled_results;
    for (const auto &graph : graphs) {
      ascir::ScheduleGroup schedule_group;
      schedule_group.impl_graphs.emplace_back(graph);
      ascir::ScheduledResult scheduled_result;
      scheduled_result.schedule_groups.emplace_back(schedule_group);
      scheduled_results.emplace_back(scheduled_result);
    }
    fused_schedule_result.node_idx_to_scheduled_results.emplace_back(scheduled_results);
    std::map<std::string, std::string> tiling_funcs;
    GE_ASSERT_TRUE(GenTilingImplAutoFuseV3(input_args.op_name, fused_schedule_result, options, tiling_funcs, true),
                   "Gen tiling implement for autofuse failed.");
    for (const auto &[key, value] : tiling_funcs) {
      if (key == "TilingHead") {
        std::ofstream oss;
        oss.open("autofuse_tiling_func_common.h", std::ios::out);
        oss << "#include \"" << input_args.op_name << "_tiling_data.h\"\n";
        oss << value;
        oss.close();
      } else if ((key == "TilingData") || (key.find("TilingData") != std::string::npos)) {
        // doning nothing,在上面做过处理了
      } else {
        std::ofstream oss;
        oss.open(input_args.op_name + "_" + key + "_" + "tiling_func.cpp", std::ios::out);
        oss << value;
        oss.close();
      }
    }
  }
  GELOGI("Gen tiling implement successfully.\n");
  GE_ASSERT_SUCCESS(GenExecTilingUtils::GenExecFunc(graphs, input_args));
  return 0;
}  // namespace att
