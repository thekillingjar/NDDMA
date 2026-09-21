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

#include "codegen.h"
#include "e2e_constant_load_le_store.h"
#include "e2e_common.h"

int main() {
  ge::AscGraph test_graph("constant_load_le_store");
  std::string tilig_stub = R"(
#define REGISTER_TILING_DEFAULT(tiling)
#define GET_TILING_DATA(t, tiling)  AutofuseTilingData t = *(AutofuseTilingData*)tiling;
)";
  std::vector<ge::AscGraph> test_impl_graphs;
  ConstantLoadLeStore_AfterAutofuse(test_graph, test_impl_graphs);

  auto codegen = codegen::Codegen(codegen::CodegenOptions{
      .tiling_lib_path = "./libtest_constant_load_le_store_codegen_tiling_gen.so", .tiling_lib_codegen_symbol = "CodegenTiling", .using_att_calc_qbt_size = false});

  std::fstream kernel_file("constant_load_le_store_kernel.cpp", std::ios::out);
  std::fstream tiling_file("constant_load_le_store_tiling.cpp", std::ios::out);
  std::fstream tiling_data_file("autofuse_tiling_data.h", std::ios::out);

  ascir::ScheduledResult schedule_result;
  std::vector<ascir::ScheduledResult> schedule_results{schedule_result};
  ascir::FusedScheduledResult fused_schedule_result;
  fused_schedule_result.fused_graph_name = ge::AscendString("constant_load_le_store");
  fused_schedule_result.node_idx_to_scheduled_results.push_back(schedule_results);
  InitScheduleResultsByImplGraphs(test_impl_graphs, fused_schedule_result);
  codegen::CodegenResult result;
  codegen.Generate(fused_schedule_result, result);
  kernel_file << tilig_stub << RemoveSubDirInclude(result.kernel);
  tiling_file << result.tiling;
  tiling_data_file << result.tiling_data;
}
