/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef TESTS_AUTOFUSE_ST_ATT_TESTCASE_COMMON_TEST_COMMON_UTILS_H_
#define TESTS_AUTOFUSE_ST_ATT_TESTCASE_COMMON_TEST_COMMON_UTILS_H_

#include <string>
#include <map>
#include <vector>
#include <fstream>
#include <iostream>
#include "base/base_types.h"
#include "base/att_const_values.h"
#include "gen_model_info.h"
#include "gen_tiling_impl.h"
#include "autofuse_config/auto_fuse_config.h"
#include "graph_construct_utils.h"
#include "common_gen_utils.h"
#include "tiling_code_generator.h"

namespace att {
namespace test_common {

// 验证tiling生成结果
inline bool ValidateTilingResult(const std::map<std::string, std::string>& tiling_funcs) {
  if (tiling_funcs.empty()) {
    return false;
  }
  bool found_solver_config = false;
  for (const auto &func : tiling_funcs) {
    if (func.second.find("corenum_threshold") != std::string::npos) {
      found_solver_config = true;
      break;
    }
  }
  return found_solver_config;
}

inline void PrintTilingDebugInfo(const std::map<std::string, std::string>& tiling_funcs) {
  std::cout << "Reduce split penalty ST test completed successfully." << std::endl;
  std::cout << "\n=== Generated tiling function count: " << tiling_funcs.size() << " ===" << std::endl;
  for (const auto &func : tiling_funcs) {
    std::cout << "Function: " << func.first << ", length: " << func.second.length() << std::endl;

    bool has_solver = func.second.find("solver") != std::string::npos;
    bool has_run = func.second.find("solver.Run") != std::string::npos;
    bool has_trade_off = func.second.find("trade_off") != std::string::npos;
    bool has_core_num_ratio = func.second.find("core_num_ratio") != std::string::npos;

    std::cout << "  Contains 'solver': " << (has_solver ? "YES" : "NO") << std::endl;
    std::cout << "  Contains 'solver.Run': " << (has_run ? "YES" : "NO") << std::endl;
    std::cout << "  Contains 'trade_off': " << (has_trade_off ? "YES" : "NO") << std::endl;
    std::cout << "  Contains 'core_num_ratio': " << (has_core_num_ratio ? "YES" : "NO") << std::endl;

    if (has_run) {
      size_t pos = func.second.find("solver.Run");
      if (pos != std::string::npos) {
        size_t start = (pos > 200) ? pos - 200 : 0;
        size_t end = (pos + 300 < func.second.length()) ? pos + 300 : func.second.length();
        std::cout << "\n  Solver.Run context:\n" << func.second.substr(start, end - start) << std::endl;
      }
    }

    if (func.first == "TilingHead" && func.second.length() > 3000) {
      std::cout << "\n=== TilingHead (first 3000 chars) ===" << std::endl;
      std::cout << func.second.substr(0, 3000) << "..." << std::endl;
    }
  }
  std::cout << "=== End of tiling code analysis ===" << std::endl;
}

// 构建单个测试用例的schedule result
inline ge::Status ConstructSingleCaseForReduceSplitPenalty(std::vector<ascir::ScheduledResult> &schedule_results) {
  ascir::ScheduleGroup schedule_group1;
  ascir::ScheduledResult scheduled_result1;
  ge::AscGraph graph_0("ReduceSplitPenalty");
  GE_ASSERT_EQ(att::GraphConstructUtils::BuildConcatGroupAscendGraphS0S1ReduceMultiTiling(graph_0), ge::SUCCESS);
  graph_0.SetTilingKey(0U);
  schedule_group1.impl_graphs.emplace_back(graph_0);
  GraphConstructUtils::UpdateGraphsVectorizedStride(schedule_group1.impl_graphs);
  scheduled_result1.schedule_groups.emplace_back(schedule_group1);
  schedule_results.emplace_back(scheduled_result1);
  return ge::SUCCESS;
}

// 生成tiling实现
inline ge::Status GenTilingImpl(std::vector<ascir::ScheduledResult> &schedule_results) {
  std::map<std::string, std::string> options;
  std::map<std::string, std::string> tiling_funcs;
  std::string op_name = "ReduceSplitPenalty";

  // 设置tiling选项
  options.emplace(kGenConfigType, "AxesReorder");
  options.emplace("enable_score_func", "1");

  // 创建fused scheduled result
  ascir::FusedScheduledResult fused_scheduled_result;
  fused_scheduled_result.node_idx_to_scheduled_results.emplace_back(schedule_results);

  // 生成tiling实现
  auto res = GenTilingImplAutoFuseV3(op_name, fused_scheduled_result, options, tiling_funcs, true);

  // 拼接tiling函数
  std::string tiling_func;
  att::test::CombineTilings(tiling_funcs, tiling_func);

  // 写入tiling函数到文件
  std::ofstream oss;
  oss.open("ReduceSplitPenalty_tiling_func.cpp", std::ios::out);
  oss << "#include \"ReduceSplitPenalty_tiling_data.h\"\n";
  oss << tiling_func;
  oss.close();

  GE_ASSERT_EQ(res, true);

  // 生成tiling data
  TilingCodeGenerator generator;
  TilingCodeGenConfig generator_config;
  std::map<std::string, std::string> tiling_res;
  FusedParsedScheduleResult all_model_infos;
  GetModelInfoMap(fused_scheduled_result, options, all_model_infos);

  generator_config.type = TilingImplType::HIGH_PERF;
  generator_config.tiling_data_type_name = options[ge::sym::kTilingDataTypeName];
  generator_config.gen_tiling_data = true;
  generator_config.gen_extra_infos = false;
  GE_ASSERT_EQ(generator.GenTilingCode(op_name, all_model_infos, generator_config, tiling_res), ge::SUCCESS);

  // 写入tiling data到文件
  oss.open("ReduceSplitPenalty_tiling_data.h", std::ios::out);
  oss << tiling_res["ReduceSplitPenaltyTilingData"];
  oss.close();

  // 复制必要的stub文件
  auto ret = std::system(
    std::string("cp ").append(TOP_DIR).append("/tests/autofuse/st/att/testcase/op_log.h ./ -f").c_str());
  // 创建tiling和register目录并拷贝文件
  (void)std::system("mkdir -p ./tiling ./register");
  ret = std::system(
    std::string("cp ").append(TOP_DIR).append("/tests/autofuse/st/att/testcase/stub/platform_ascendc.h ./tiling/ -f").c_str());
  ret = std::system(
    std::string("cp ").append(TOP_DIR).append("/tests/autofuse/st/att/testcase/stub/tiling_api.h ./tiling/ -f").c_str());
  ret = std::system(
    std::string("cp ").append(TOP_DIR).append("/tests/autofuse/st/att/testcase/stub/tiling_context.h ./tiling/ -f").c_str());
  ret = std::system(
    std::string("cp ").append(TOP_DIR).append("/tests/autofuse/st/att/testcase/stub/tilingdata_base.h ./register/ -f").c_str());
  GE_ASSERT_EQ(ret, 0);
  return ge::SUCCESS;
}

// 通用的 tiling 代码生成和写入逻辑
inline ge::Status GenTilingCodeToFile(
    const std::string &op_name,
    const std::map<std::string, std::string> &tiling_funcs,
    const ascir::FusedScheduledResult &fused_scheduled_result,
    const std::map<std::string, std::string> &options,
    const std::string &tiling_data_name) {
  // 写入 tiling 函数
  std::ofstream oss;
  oss.open(op_name + "_tiling_func.cpp", std::ios::out);
  oss << "#include \"" << op_name << "_tiling_data.h\"\n";

  std::string tiling_func;
  att::test::CombineTilings(tiling_funcs, tiling_func);
  oss << tiling_func;
  oss.close();

  // 生成 tiling data
  TilingCodeGenerator generator;
  TilingCodeGenConfig generator_config;
  std::map<std::string, std::string> tiling_res;
  FusedParsedScheduleResult all_model_infos;
  GetModelInfoMap(fused_scheduled_result, options, all_model_infos);

  generator_config.type = TilingImplType::HIGH_PERF;
  generator_config.tiling_data_type_name = options.at(ge::sym::kTilingDataTypeName);
  generator_config.gen_tiling_data = true;
  generator_config.gen_extra_infos = false;
  GE_ASSERT_EQ(generator.GenTilingCode(op_name, all_model_infos, generator_config, tiling_res), ge::SUCCESS);

  // 写入 tiling data
  oss.open(op_name + "_tiling_data.h", std::ios::out);
  oss << tiling_res[tiling_data_name];
  oss.close();

  return ge::SUCCESS;
}

// 创建 tiling main 函数模板字符串
inline std::string CreateTilingMainFunc(
    const std::string& tiling_data_type,
    const std::string& block_dim,
    const std::string& ub_size,
    const std::vector<std::pair<std::string, std::string>>& params = {},
    const std::string& error_msg = "tiling func execute failed.") {
  std::ostringstream oss;
  oss << R"(
#include <iostream>
#include ")";
  oss << tiling_data_type << R"(_tiling_data.h"
using namespace optiling;

void PrintResult()" << tiling_data_type << R"(TilingData& tilingData) {
  std::cout << "====================================================" << std::endl;
  auto tiling_key = tilingData.get_tiling_key();
  std::cout << "get_tiling_key"<< " = " << tiling_key << std::endl;
  std::cout << "====================================================" << std::endl;
}

int main() {
  )" << tiling_data_type << R"(TilingData tilingData;
  tilingData.set_block_dim()"<< block_dim << R"();
  tilingData.set_ub_size()"<< ub_size << R"();
)";
  for (const auto &param : params) {
    oss << "  tilingData.set_" << param.first << "(" << param.second << ");\n";
  }
  oss << R"(  if (GetTiling(tilingData)) {
    PrintResult(tilingData);
  } else {
    std::cout << ")" << error_msg << R"(" << std::endl;
    return -1;
  }
  return 0;
}
)";
  return oss.str();
}

// 运行基本的 tiling 测试验证
inline void RunBasicTilingTest(
    const std::string& op_name,
    const std::vector<ascir::ScheduledResult>& schedule_results,
    const std::string& config_type = "AxesReorder") {
  std::map<std::string, std::string> options;
  options.emplace(kGenConfigType, config_type);
  ascir::FusedScheduledResult fused_scheduled_result;
  fused_scheduled_result.node_idx_to_scheduled_results.emplace_back(
      const_cast<std::vector<ascir::ScheduledResult>&>(schedule_results));

  std::map<std::string, std::string> tiling_funcs;
  auto res = GenTilingImplAutoFuseV3(op_name, fused_scheduled_result, options, tiling_funcs, true);
  EXPECT_EQ(res, true);
  EXPECT_EQ(ValidateTilingResult(tiling_funcs), true);
  PrintTilingDebugInfo(tiling_funcs);
}

// 清理测试用例生成的临时文件和stub拷贝
inline void CleanupTestArtifacts() {
  // 删除stub相关目录和文件
  system("rm -rf ./stub ./tiling ./register ./graph ./lib ./kernel_tiling");
  // 删除公共文件
  system("rm -f ./op_log.h ./autofuse_tiling_func_common.h");
  // 删除日志文件
  system("rm -f *.log");
  // 删除生成的二进制文件
  system("rm -f ./tiling_func_main ./tiling_func_main_concat ./tiling_func_main_transpose ./tiling_func_main_softmax");
  // 删除生成的tiling data和func文件
  system("rm -f ./*_tiling_data.h ./*_tiling_func.cpp ./tiling_func_main_*.cpp");
}

}  // namespace test_common
}  // namespace att

#endif  // TESTS_AUTOFUSE_ST_ATT_TESTCASE_COMMON_TEST_COMMON_UTILS_H_
