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

#include "ascir_utils.h"
#include "codegen.h"
#include "e2e_common.h"
#include "e2e_multi_vectorize_axis.h"

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

class LoadMultiAxisStoreTest : public testing::Test {
};

TEST_F(LoadMultiAxisStoreTest, LoadMultiAxisStoreCodegen) {

  bool gen_success = true;
  ge::AscGraph graph("load_multi_axis_store");
  std::string tilig_stub = R"(
#define REGISTER_TILING_DEFAULT(tiling)
#define GET_TILING_DATA(t, tiling)  AutofuseTilingData t = *(AutofuseTilingData*)tiling;
)";
  LoadMultiVectorizeAxisStore_BeforeAutofuse(graph, false);

  std::vector<ge::AscGraph> impl_graph = {ge::AscGraph("load_multi_axis_store_general_0_nil_0_nil")};
  impl_graph[0].CopyFrom(graph);

  LoadMultiVectorizeAxisStore_AfterAutofuse(impl_graph[0], false);

  std::cout << utils::DebugImplGraphStr(impl_graph[0]) << std::endl;

  std::vector<std::string> parts = splitString(KERNEL_SRC_LIST, ':');
  std::string kernel_src_file_name = parts[0];      // load_multi_axis_store_kernel.cpp
  std::string tiling_src_file_name = parts[1];      // load_multi_axis_store_tiling.cpp
  std::string tiling_data_src_file_name = parts[2]; // autofuse_tiling_data.h

  try {
    codegen::Codegen c(codegen::CodegenOptions{
        .tiling_lib_path = ATT_SO_NAME,
        .tiling_lib_codegen_symbol = "CodegenTiling",
        .using_att_calc_qbt_size = false,
    });
    ascir::ScheduledResult schedule_result;
    std::vector<ascir::ScheduledResult> schedule_results{schedule_result};
    ascir::FusedScheduledResult fused_schedule_result;
    fused_schedule_result.fused_graph_name = ge::AscendString("load_multi_axis_store");
    fused_schedule_result.node_idx_to_scheduled_results.push_back(schedule_results);
    InitScheduleResultsByImplGraphs({impl_graph}, fused_schedule_result);
    codegen::CodegenResult result;
    EXPECT_EQ(c.Generate(fused_schedule_result, result), 0);

    std::fstream kernel_file(kernel_src_file_name, std::ios::out);
    std::fstream tiling_file(tiling_src_file_name, std::ios::out);
    std::fstream tiling_data_file(tiling_data_src_file_name, std::ios::out);

    kernel_file << tilig_stub << RemoveSubDirInclude(result.kernel);
    tiling_file << result.tiling;
    tiling_data_file << result.tiling_data;
  }
  catch (...) {
    gen_success = false;
  }

  EXPECT_EQ(gen_success, true);
}
