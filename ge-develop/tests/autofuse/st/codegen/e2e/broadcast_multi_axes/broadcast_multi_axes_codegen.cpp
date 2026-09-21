/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <fstream>
#include <gtest/gtest.h>
#include <exception>
#include <filesystem>
#include "codegen.h"
#include "e2e_common.h"

#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include "e2e_broadcast.h"

std::vector<std::string> splitString(const std::string& input, char delimiter) {
  std::vector<std::string> result;
  std::stringstream ss(input);
  std::string token;

  while (std::getline(ss, token, delimiter)) {
      result.push_back(token);
  }

  return result;
}

class BroadcastMultiAxesST : public testing::Test {
};

TEST_F(BroadcastMultiAxesST, BroadcastMultiAxesCodegen) {

  bool gen_success = true;
  std::string tilig_stub = R"(
    #define REGISTER_TILING_DEFAULT(tiling)
    #define GET_TILING_DATA(t, tiling)  AutofuseTilingData t = *(AutofuseTilingData*)tiling;
    )";
  ge::AscGraph graph_multi_axis("broadcast_multi_axes");
  std::vector<ge::AscGraph> impl_graphs_multi_axis;
  ConstructMultiAxisGraph(graph_multi_axis, impl_graphs_multi_axis, {false,false,true,false,true});

  std::cout<<"ATT_SO_NAME="<<ATT_SO_NAME<<std::endl;
  std::cout<<"KERNEL_SRC_LIST="<<KERNEL_SRC_LIST<<std::endl;
  std::vector<std::string> parts = splitString(KERNEL_SRC_LIST, ':');
  std::string kernel_src_file_name = parts[0];
  std::string tiling_src_file_name = parts[1];
  std::string tiling_data_src_file_name = parts[2];

  try {
    auto codegen = codegen::Codegen(codegen::CodegenOptions{
         // lib + 用例文件夹名 + _gen_tiling.so
        .tiling_lib_path = ATT_SO_NAME, .tiling_lib_codegen_symbol = "CodegenTiling", .using_att_calc_qbt_size = false});
    std::fstream kernel_file(kernel_src_file_name, std::ios::out);
    std::fstream tiling_file(tiling_src_file_name, std::ios::out);
    std::fstream tiling_data_file(tiling_data_src_file_name, std::ios::out);

    ascir::ScheduledResult schedule_result;
    std::vector<ascir::ScheduledResult> schedule_results{schedule_result};
    ascir::FusedScheduledResult fused_schedule_result;
    fused_schedule_result.fused_graph_name = ge::AscendString("broadcast_multi_axes");
    fused_schedule_result.node_idx_to_scheduled_results.push_back(schedule_results);
    InitScheduleResultsByImplGraphs(impl_graphs_multi_axis, fused_schedule_result);
    codegen::CodegenResult result;
    EXPECT_EQ(codegen.Generate(fused_schedule_result, result), 0);
    kernel_file << tilig_stub << RemoveSubDirInclude(result.kernel);
    tiling_file << result.tiling;
    tiling_data_file << result.tiling_data;
  }
  catch (...) {
    gen_success = false;
  }

  EXPECT_EQ(gen_success, true);

  std::string tilig_stub_bab = R"(
    #define REGISTER_TILING_DEFAULT(tiling)
    #define GET_TILING_DATA(t, tiling)  AutofuseTilingData t = *(AutofuseTilingData*)tiling;
    )";
  ge::AscGraph graph_multi_axis_bab("broadcast_multi_axes_bab");
  std::vector<ge::AscGraph> impl_graphs_multi_axis_bab;
  ConstructMultiAxisGraph(graph_multi_axis_bab, impl_graphs_multi_axis_bab, {false,true,false,false,true});

  std::string kernel_src_file_name_bab = "broadcast_multi_axes_kernel_bab.cpp";

  try {
    auto codegen_bab = codegen::Codegen(codegen::CodegenOptions{
         // lib + 用例文件夹名 + _gen_tiling.so
        .tiling_lib_path = ATT_SO_NAME, .tiling_lib_codegen_symbol = "CodegenTiling", .using_att_calc_qbt_size = false});
    std::fstream kernel_file_bab(kernel_src_file_name_bab, std::ios::out);

    ascir::ScheduledResult schedule_result_bab;
    std::vector<ascir::ScheduledResult> schedule_results_bab{schedule_result_bab};
    ascir::FusedScheduledResult fused_schedule_result_bab;
    fused_schedule_result_bab.fused_graph_name = ge::AscendString("broadcast_multi_axes_bab");
    fused_schedule_result_bab.node_idx_to_scheduled_results.push_back(schedule_results_bab);
    InitScheduleResultsByImplGraphs(impl_graphs_multi_axis_bab, fused_schedule_result_bab);
    codegen::CodegenResult result_bab;
    EXPECT_EQ(codegen_bab.Generate(fused_schedule_result_bab, result_bab), 0);
    kernel_file_bab << tilig_stub_bab << RemoveSubDirInclude(result_bab.kernel);
  }
  catch (...) {
    gen_success = false;
  }
}
