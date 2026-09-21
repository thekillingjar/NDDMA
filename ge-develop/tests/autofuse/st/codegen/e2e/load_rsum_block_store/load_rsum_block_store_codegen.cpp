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

class LoadRsumBlkStoreSt : public testing::Test {
};

TEST_F(LoadRsumBlkStoreSt, reduceSumTest) {
  bool gen_success = true;
  // Add reducesum int32 no tileouter axis
  ge::AscGraph test_graph("load_rsum_store_block");
  std::string tilig_stub = R"(
#define REGISTER_TILING_DEFAULT(tiling)
#define GET_TILING_DATA(t, tiling)  AutofuseTilingData t = *(AutofuseTilingData*)tiling;
)";
  LoadRsumStore_BeforeAutofuse(test_graph, ge::DT_FLOAT);
  LoadRsumUbStore_AfterAutofuse(test_graph, ge::DT_FLOAT);

  std::vector<ge::AscGraph> test_impl_graphs = {ge::AscGraph("load_rsum_store_block_general_0_nil_0_nil")};
  test_impl_graphs[0].CopyFrom(test_graph);

  std::string kernel_src_file_name_block = "load_rsum_block_store_float_kernel.cpp";
  std::string tiling_src_file_name_block = "load_rsum_block_store_float_tiling.cpp";
  std::string tiling_data_src_file_name_block = "autofuse_tiling_data.h";

  try {
    auto codegen = codegen::Codegen(codegen::CodegenOptions{
         // lib + 用例文件夹名 + _gen_tiling.so
        .tiling_lib_path = ATT_SO_NAME, .tiling_lib_codegen_symbol = "CodegenTiling", .using_att_calc_qbt_size = false});
    std::fstream kernel_file(kernel_src_file_name_block, std::ios::out);
    std::fstream tiling_file(tiling_src_file_name_block, std::ios::out);
    std::fstream tiling_data_file(tiling_data_src_file_name_block, std::ios::out);

    ascir::ScheduledResult schedule_result;
    std::vector<ascir::ScheduledResult> schedule_results{schedule_result};
    ascir::FusedScheduledResult fused_schedule_result;
    fused_schedule_result.fused_graph_name = ge::AscendString("load_rsum_store_block");
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

    EXPECT_EQ(gen_success, true);

  // Add reducesum int32 no tileouter axis
  ge::AscGraph test_graph_int32_block("load_rsum_store_int32_block");
  LoadRsumStore_BeforeAutofuse(test_graph_int32_block, ge::DT_INT32);
  LoadRsumUbStore_AfterAutofuse(test_graph_int32_block, ge::DT_INT32);

  std::vector<ge::AscGraph> test_impl_graphs_int32_blk = {ge::AscGraph("load_rsum_store_int32_block_general_0_nil_0_nil")};
  test_impl_graphs_int32_blk[0].CopyFrom(test_graph_int32_block);

  std::string kernel_src_file_name_int32_block = "load_rsum_block_store_kernel.cpp";
  std::string tiling_src_file_name_int32_block = "load_rsum_block_store_tiling.cpp";

  try {
    auto codegen_int32_blk = codegen::Codegen(codegen::CodegenOptions{
         // lib + 用例文件夹名 + _gen_tiling.so
        .tiling_lib_path = ATT_SO_NAME, .tiling_lib_codegen_symbol = "CodegenTiling", .using_att_calc_qbt_size = false});
    std::fstream kernel_file_int32_blk(kernel_src_file_name_int32_block, std::ios::out);

    ascir::ScheduledResult schedule_result_int32_blk;
    std::vector<ascir::ScheduledResult> schedule_results_int32_blk{schedule_result_int32_blk};
    ascir::FusedScheduledResult fused_schedule_result_int32_blk;
    fused_schedule_result_int32_blk.fused_graph_name = ge::AscendString("load_rsum_store_int32_block");
    fused_schedule_result_int32_blk.node_idx_to_scheduled_results.push_back(schedule_results_int32_blk);
    InitScheduleResultsByImplGraphs(test_impl_graphs_int32_blk, fused_schedule_result_int32_blk);
    codegen::CodegenResult result_int32_blk;
    EXPECT_EQ(codegen_int32_blk.Generate(fused_schedule_result_int32_blk, result_int32_blk), 0);
    kernel_file_int32_blk << tilig_stub << RemoveSubDirInclude(result_int32_blk.kernel);
  }
  catch (...) {
    gen_success = false;
  }

    EXPECT_EQ(gen_success, true);
}
