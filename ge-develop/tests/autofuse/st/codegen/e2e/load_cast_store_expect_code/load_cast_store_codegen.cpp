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

#include "codegen.h"
#include "e2e_load_cast_store.h"
#include "e2e_common.h"

std::vector<std::string> splitString(const std::string& input, char delimiter) {
  std::vector<std::string> result;
  std::stringstream ss(input);
  std::string token;

  while (std::getline(ss, token, delimiter)) {
      result.push_back(token);
  }

  return result;
}

class LoadCastStoreTest : public testing::Test {
};

TEST_F(LoadCastStoreTest, LoadCastStoreCodegen) {

  bool gen_success= true;
  ge::AscGraph test_graph("load_cast_store");
  std::string tilig_stub = R"(
#define REGISTER_TILING_DEFAULT(tiling)
#define GET_TILING_DATA(t, tiling)  AutofuseTilingData t = *(AutofuseTilingData*)tiling;
)";
  LoadCastStore_BeforeAutofuse(test_graph, ge::DT_FLOAT, ge::DT_INT16);
  LoadCastStore_AfterInferOutput(test_graph, ge::DT_FLOAT, ge::DT_INT16);

  std::vector<ge::AscGraph> test_impl_graphs = {ge::AscGraph("load_cast_store_general_0_nil_0_nil")};
  test_impl_graphs[0].CopyFrom(test_graph);
  LoadCastStore_AfterGetApiInfo(test_impl_graphs[0]);
  LoadCastStore_AfterScheduler(test_impl_graphs[0]);
  LoadCastStore_AfterQueBufAlloc(test_impl_graphs[0]);

  std::vector<std::string> parts = splitString(KERNEL_SRC_LIST, ':');
  std::string kernel_src_file_name = parts[0];      // load_cast_store_kernel.cpp
  std::string tiling_src_file_name = parts[1];      // load_cast_store_tiling.cpp
  std::string tiling_data_src_file_name = parts[2]; // autofuse_tiling_data.h

  try {
    auto codegen = codegen::Codegen(codegen::CodegenOptions{
        .tiling_lib_path = ATT_SO_NAME, .tiling_lib_codegen_symbol = "CodegenTiling", .using_att_calc_qbt_size = false});

    std::fstream kernel_file(kernel_src_file_name, std::ios::out);
    std::fstream tiling_file(tiling_src_file_name, std::ios::out);
    std::fstream tiling_data_file(tiling_data_src_file_name, std::ios::out);

    ascir::ScheduledResult schedule_result;
    std::vector<ascir::ScheduledResult> schedule_results{schedule_result};
    ascir::FusedScheduledResult fused_schedule_result;
    fused_schedule_result.fused_graph_name = ge::AscendString("load_cast_store");
    fused_schedule_result.node_idx_to_scheduled_results.push_back(schedule_results);
    InitScheduleResultsByImplGraphs(test_impl_graphs, fused_schedule_result);
    codegen::CodegenResult result;
    EXPECT_EQ(codegen.Generate(fused_schedule_result, result),0);
    kernel_file << tilig_stub << RemoveSubDirInclude(result.kernel);
    tiling_file << result.tiling;
    tiling_data_file << result.tiling_data;
  }
  catch (...) {
    gen_success = false;
  }

  EXPECT_EQ(gen_success, true);

  ge::AscGraph test_graph_uint8("load_cast_store_uint8");
  std::string tilig_stub_uint8 = R"(
#define REGISTER_TILING_DEFAULT(tiling)
#define GET_TILING_DATA(t, tiling)  AutofuseTilingData t = *(AutofuseTilingData*)tiling;
)";
  LoadCastStore_BeforeAutofuse(test_graph_uint8, ge::DT_UINT8, ge::DT_FLOAT);
  LoadCastStore_AfterInferOutput(test_graph_uint8, ge::DT_UINT8, ge::DT_FLOAT);

  std::vector<ge::AscGraph> test_impl_graphs_uint8 = {ge::AscGraph("load_cast_store_uint8_general_0_nil_0_nil")};
  test_impl_graphs_uint8[0].CopyFrom(test_graph_uint8);
  LoadCastStore_AfterGetApiInfo(test_impl_graphs_uint8[0]);
  LoadCastStore_AfterScheduler(test_impl_graphs_uint8[0]);
  LoadCastStore_AfterQueBufAlloc(test_impl_graphs_uint8[0]);

  std::string kernel_src_file_name_uint8 = "load_cast_store_uint8_kernel.cpp";      // load_cast_store_kernel.cpp
  std::string tiling_src_file_name_uint8 = "load_cast_store_uint8_tiling.cpp";      // load_cast_store_tiling.cpp
  std::string tiling_data_src_file_name_uint8 = "autofuse_tiling_data.h"; // autofuse_tiling_data.h

  try {
    auto codegen_uint8 = codegen::Codegen(codegen::CodegenOptions{
        .tiling_lib_path = ATT_SO_NAME, .tiling_lib_codegen_symbol = "CodegenTiling", .using_att_calc_qbt_size = false});

    std::fstream kernel_file_uint8(kernel_src_file_name_uint8, std::ios::out);
    std::fstream tiling_data_file_uint8(tiling_data_src_file_name_uint8, std::ios::out);

    ascir::ScheduledResult schedule_result_uint8;
    std::vector<ascir::ScheduledResult> schedule_results_uint8{schedule_result_uint8};
    ascir::FusedScheduledResult fused_schedule_result_uint8;
    fused_schedule_result_uint8.fused_graph_name = ge::AscendString("load_cast_store_uint8");
    fused_schedule_result_uint8.node_idx_to_scheduled_results.push_back(schedule_results_uint8);
    InitScheduleResultsByImplGraphs(test_impl_graphs_uint8, fused_schedule_result_uint8);
    codegen::CodegenResult result_uint8;
    EXPECT_EQ(codegen_uint8.Generate(fused_schedule_result_uint8, result_uint8),0);
    kernel_file_uint8 << tilig_stub_uint8 << RemoveSubDirInclude(result_uint8.kernel);
    tiling_data_file_uint8 << result_uint8.tiling_data;
  }
  catch (...) {
    gen_success = false;
  }

  EXPECT_EQ(gen_success, true);
}
