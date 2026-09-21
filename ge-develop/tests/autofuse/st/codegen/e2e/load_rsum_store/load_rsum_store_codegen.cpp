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
#include "e2e_rsum.h"
#include "e2e_common.h"

#include <iostream>
#include <vector>
#include <string>
#include <sstream>

using namespace ascir;

std::vector<std::string> splitString(const std::string& input, char delimiter) {
    std::vector<std::string> result;
    std::stringstream ss(input);
    std::string token;

    while (std::getline(ss, token, delimiter)) {
        result.push_back(token);
    }

    return result;
}

class LoadRsumStoreSt : public testing::Test {
};

TEST_F(LoadRsumStoreSt, reduceSumTest) {
  bool gen_success = true;

  ge::AscGraph test_graph("load_rsum_store");
  std::string tilig_stub = R"(
#define REGISTER_TILING_DEFAULT(tiling)
#define GET_TILING_DATA(t, tiling)  AutofuseTilingData t = *(AutofuseTilingData*)tiling;
)";
  LoadRsumStore_BeforeAutofuse(test_graph, ge::DT_FLOAT);
  LoadRsumStore_AfterAutofuse(test_graph, ge::DT_FLOAT);

  std::vector<ge::AscGraph> test_impl_graphs = {ge::AscGraph("load_rsum_store_general_0_nil_0_nil")};
  test_impl_graphs[0].CopyFrom(test_graph);

  std::cout<<"ATT_SO_NAME="<<ATT_SO_NAME<<std::endl;
  std::cout<<"KERNEL_SRC_LIST="<<KERNEL_SRC_LIST<<std::endl;
  std::vector<std::string> parts = splitString(KERNEL_SRC_LIST, ':');
  std::string kernel_src_file_name = parts[0];      // load_abs_store_kernel.cpp
  std::string tiling_src_file_name = parts[1];      // load_abs_store_tiling.cpp
  std::string tiling_data_src_file_name = parts[2]; // autofuse_tiling_data.h

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
    fused_schedule_result.fused_graph_name = ge::AscendString("load_rsum_store");
    fused_schedule_result.node_idx_to_scheduled_results.push_back(schedule_results);
    InitScheduleResultsByImplGraphs(test_impl_graphs, fused_schedule_result);
    codegen::CodegenResult result;
    EXPECT_EQ(codegen.Generate(fused_schedule_result, result), 0);
    kernel_file << tilig_stub << RemoveSubDirInclude(result.kernel);
    tiling_file << result.tiling;
    tiling_data_file << result.tiling_data;
  }
  catch (...) {
    gen_success = false;
  }

  // Add reducesum int32
  ge::AscGraph test_graph_int32("load_rsum_store_int32");
  std::string tilig_stub_int32 = R"(
#define REGISTER_TILING_DEFAULT(tiling)
#define GET_TILING_DATA(t, tiling)  load_rsum_store_int32TilingData t = *(load_rsum_store_int32TilingData*)tiling;
)";
  LoadRsumStore_BeforeAutofuse(test_graph_int32, ge::DT_INT32);
  LoadRsumStore_AfterAutofuse(test_graph_int32, ge::DT_INT32);

  std::vector<ge::AscGraph> test_impl_graphs_int32 = {ge::AscGraph("load_rsum_store_int32_general_0_nil_0_nil")};
  test_impl_graphs_int32[0].CopyFrom(test_graph_int32);

  std::string kernel_src_file_name_int32 = "load_rsum_store_int32_kernel.cpp";
  std::string tiling_src_file_name_int32 = "load_rsum_store_int32_tiling.cpp";
  std::string tiling_data_src_file_name_int32 = "autofuse_tiling_data.h";

  try {
    auto codegen_int32 = codegen::Codegen(codegen::CodegenOptions{
         // lib + 用例文件夹名 + _gen_tiling.so
        .tiling_lib_path = ATT_SO_NAME, .tiling_lib_codegen_symbol = "CodegenTiling", .using_att_calc_qbt_size = false});
    std::fstream kernel_file_int32(kernel_src_file_name_int32, std::ios::out);
    std::fstream tiling_file_int32(tiling_src_file_name_int32, std::ios::out);
    std::fstream tiling_data_file_int32(tiling_data_src_file_name_int32, std::ios::out);

    ascir::ScheduledResult schedule_result_int32;
    std::vector<ascir::ScheduledResult> schedule_results_int32{schedule_result_int32};
    ascir::FusedScheduledResult fused_schedule_result_int32;
    fused_schedule_result_int32.fused_graph_name = ge::AscendString("load_rsum_store_int32");
    fused_schedule_result_int32.node_idx_to_scheduled_results.push_back(schedule_results_int32);
    InitScheduleResultsByImplGraphs(test_impl_graphs_int32, fused_schedule_result_int32);
    codegen::CodegenResult result_int32;
    EXPECT_EQ(codegen_int32.Generate(fused_schedule_result_int32, result_int32), 0);
    kernel_file_int32 << tilig_stub_int32 << RemoveSubDirInclude(result_int32.kernel);
    tiling_file_int32 << result_int32.tiling;
    tiling_data_file_int32 << result_int32.tiling_data;
  }
  catch (...) {
    gen_success = false;
  }

    EXPECT_EQ(gen_success, true);
}
