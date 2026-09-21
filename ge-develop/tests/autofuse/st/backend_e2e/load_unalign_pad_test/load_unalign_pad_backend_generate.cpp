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
#include "optimize.h"
#include "share_graph.h"
#include "backend_common.h"
#include "ascir_ops.h"

#include <iostream>
#include <vector>
#include <string>
#include <sstream>

class TestBackendLoadUnalignPadE2e : public testing::Test {
 protected:
  static void CreatePadAscGraph(ge::AscGraph &graph, const std::vector<std::string> &dim_sizes) {
    ge::ascir_op::Data x1("pad_data0", graph);
    x1.ir_attr.SetIndex(0);
    x1.y.dtype = ge::DT_INT32;
    x1.attr.tmp_buffers = {{{ge::Symbol(32*32), -1}, ge::MemAttr(), 0}};

    ge::ascir_op::Data x2("pad_data1", graph);
    x2.ir_attr.SetIndex(1);
    x2.y.dtype = ge::DT_INT32;
    x2.attr.tmp_buffers = {{{ge::Symbol(32*32), -1}, ge::MemAttr(), 1}};

    ge::ascir_op::Load x1Local("load0");
    x1Local.x = x1.y;
    x1Local.y.dtype =  ge::DT_INT32;

    ge::ascir_op::Load x2Local("load1");
    x2Local.x = x2.y;
    x2Local.y.dtype =  ge::DT_INT32;

    ge::ascir_op::Add pad("add");
    pad.x1 = x1Local.y;
    pad.x2 = x2Local.y;
    pad.y.dtype = ge::DT_INT32;

    ge::ascir_op::Store x_out("store");
    x_out.x = pad.y;
    x_out.y.dtype = ge::DT_INT32;

    ge::ascir_op::Output y("output");
    y.x = x_out.y;
    y.ir_attr.SetIndex(0);
    y.y.dtype = ge::DT_INT32;

    ConstructVVAscGraphAxisInfo(graph, dim_sizes);
    auto pad_node = graph.FindNode("add");
    auto size = pad_node->attr.sched.axis.size();
    auto repeats = pad_node->outputs()[0]->attr.repeats;
    auto strides = pad_node->outputs()[0]->attr.strides;
    for (int i = dim_sizes.size() - 2; i >= 0; i--) {
      strides[i] = ge::sym::Mul(repeats[i + 1], strides[i + 1]);
    }
    pad_node->outputs()[0]->attr.strides = strides;
    pad_node->outputs()[0]->attr.repeats = repeats;
    auto store_node = graph.FindNode("store");
    strides[dim_sizes.size() - 1] = ge::Symbol(1);
    for (int i = dim_sizes.size() - 2; i >= 0; i--) {
      strides[i] = ge::sym::Mul(repeats[i + 1], strides[i + 1]);
    }
    store_node->outputs()[0]->attr.strides = strides;
    store_node->outputs()[0]->attr.repeats = repeats;
  }

  static void ConstructVVAscGraphAxisInfo(ge::AscGraph &graph, const std::vector<std::string> &dim_sizes) {
    std::vector<int64_t> axis;
    std::vector<ge::Expression> repeats;
    std::vector<ge::Expression> strides;
    auto ONE = ge::Symbol(2);

    // 构造符号、轴信息
    for (size_t i = 0; i < dim_sizes.size(); i++) {
      ge::Symbol sym_s;
      if (dim_sizes[i][0] == 's') {
        sym_s = ge::Symbol(dim_sizes[i].c_str());
      } else {
        sym_s = ge::Symbol(std::atoi(dim_sizes[i].c_str()));
      }
      std::string sym_str = "s" + std::to_string(i);
      std::string axis_str = "z" + std::to_string(i);
      auto aixs_z = graph.CreateAxis(axis_str.c_str(), sym_s);
      axis.push_back(aixs_z.id);
      repeats.push_back(sym_s);
      strides.push_back(ONE);
    }
    // 计算每个轴的stride
    for (int i = dim_sizes.size() - 2; i >= 0; i--) {
      strides[i] = ge::sym::Mul(repeats[i + 1], strides[i + 1]);
    }
    // 将原始轴信息设置到图中所有节点上
    for (auto node : graph.GetAllNodes()) {
      node->attr.sched.axis = axis;
      for (auto output_attr : node->outputs()) {
        output_attr->attr.axis = axis;
        output_attr->attr.repeats = repeats;
        output_attr->attr.strides = strides;
      }
    }
  }
};

TEST_F(TestBackendLoadUnalignPadE2e, PadE2eCodegen) {
  bool gen_success = true;
  std::string tilig_stub = R"(
#define REGISTER_TILING_DEFAULT(tiling)
#define GET_TILING_DATA(t, tiling)  AutofuseTilingData t = *(AutofuseTilingData*)tiling;
)";

  ge::AscGraph graph("load_unalign_pad_test");
  CreatePadAscGraph(graph, {"100", "51"});
  std::cout<<"KERNEL_SRC_LIST="<<KERNEL_SRC_LIST<<std::endl;
  std::vector<std::string> parts = splitString(KERNEL_SRC_LIST, ':');
  std::string kernel_src_file_name = parts[0];      // add_abs_test_tiling.cpp
  std::string tiling_src_file_name = parts[1];      // add_abs_test_kernel.cpp
  std::string tiling_data_src_file_name = parts[2]; // autofuse_tiling_data.h

  try {
    optimize::Optimizer optimizer(optimize::OptimizerOptions{});
    codegen::Codegen codegen(codegen::CodegenOptions{});

    std::fstream kernel_file(kernel_src_file_name, std::ios::out);
    std::fstream tiling_file(tiling_src_file_name, std::ios::out);
    std::fstream tiling_data_file(tiling_data_src_file_name, std::ios::out);

    std::vector<::ascir::ScheduledResult> schedule_results;
    ascir::FusedScheduledResult fused_schedule_result;
    fused_schedule_result.node_idx_to_scheduled_results.push_back(schedule_results);
    EXPECT_EQ(optimizer.Optimize(graph, fused_schedule_result), 0);
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
}
