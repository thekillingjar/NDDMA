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
#include <string>
#include <functional>
#include <regex>
#include "gtest/gtest.h"
#include "gen_model_info.h"
#include "ascir_ops.h"
#include "tiling_code_generator.h"
#include "api_tiling_gen/gen_api_tiling.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/mem_utils.h"
#include "graph/ascendc_ir/ascendc_ir_dump_utils.h"
#include "gen_tiling_impl.h"

using namespace ge::ascir_op;
namespace ascir {
using namespace ge;
}
using namespace att;
using AscGraph = ge::AscGraph;

class TestGenAddLayerNormalModelInfoE2E : public ::testing::Test {
 public:
  static void TearDownTestCase() {
    std::cout << "Test end." << std::endl;
  }
  static void SetUpTestCase() {
    std::cout << "Test begin." << std::endl;
  }
  void SetUp() override {
    // Code here will be called immediately after the constructor (right
    // before each test).
    system("rm -rf tiling_func/");
  }

  void TearDown() override {
    // Code here will be called immediately after each test (right
    // before the destructor).
  }
};
namespace ge {
using namespace ascir::cg;
constexpr int64_t ID_NONE = -1;  // 取多少？
constexpr ge::DataType dtypes[3] = {ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16};

/*
for aBO
  for aBIO
    for aBII
      for r
        load x1
        load x2
        load bias
        CalcMean
        CalcRstd
        Store X
        Store mean
        Load beta
        Load gamma
        CalcRstd
        Store rstd
        CalcY
        Store y
*/
void Add_Layer_Norm_Normal_BeforeAutofuseNewCg(AscGraph &graph, bool bias_broad_cast, bool is_addtional_output, int dtype_index) {
  auto ONE = ge::sym::kSymbolOne;
  auto ZERO = ge::sym::kSymbolZero;
  // 定义轴的大小
  auto A = ge::Symbol("A");
  auto R = ge::Symbol("R");
  auto BL = ge::Symbol(8, "BL");
  // 定义轴
  auto a = graph.CreateAxis("A", A);
  auto r = graph.CreateAxis("R", R);
  auto bl = graph.CreateAxis("BL", BL);
  // 切分轴
  auto [aBO, aBI] = graph.BlockSplit(a.id, "nbi", "nbo");      // AB Ab
  auto [aBIO, aBII] = graph.TileSplit(aBI->id, "nii", "nio");  // AbT Abt
  aBI->align = ge::Symbol(1);
  aBII->align = ge::Symbol(1);
  /*
  // 是否需要输出x
  if (is_addtional_output) {
    graph.CreateOptionalAtt("additional_output", 1, 1, 1);
  } else {
    graph.CreateOptionalAtt("additional_output", 1, 0, 0);
  }
  */
  // 创建输入节点
  auto x1 = graph.CreateContiguousData("x1", dtypes[dtype_index], {a, r}, {{0, -2}, {-1}});
  auto x2 = graph.CreateContiguousData("x2", dtypes[dtype_index], {a, r}, {{0, -2}, {-1}});
  auto gamma = graph.CreateContiguousData("gamma", dtypes[dtype_index], {r});
  auto beta = graph.CreateContiguousData("beta", dtypes[dtype_index], {r});
  auto bias = bias_broad_cast ? graph.CreateContiguousData("bias", dtypes[dtype_index], {r})
                              : graph.CreateContiguousData("bias", dtypes[dtype_index], {a, r}, {{0, -2}, {-1}});
  auto one = TbufData("one", graph, ge::DT_FLOAT, {bl.id}, {BL}, {ONE});
  // 构建schedule图
  LOOP(*aBO) {
    LOOP(*aBIO) {
      auto x1Local = Load("x1Local", x1).TQue(Position::kPositionVecIn, 1, 1);
      auto x2Local = Load("x2Local", x2).TQue(Position::kPositionVecIn, 1, 1);
      auto biasLocal = Load("biasLocal", bias).TQue(Position::kPositionVecIn, 1, 1);
      auto [mean, xOut, xFp32] = CalcMean("mean", x1Local, x2Local, biasLocal, r.id);
      mean.TQue(Position::kPositionVecOut, 1, 1);
      xOut.TQue(Position::kPositionVecOut, 1, 1);
      xFp32.TQue(Position::kPositionVecOut, 1, 1);
      auto mean_out = Store("mean_out", mean);
      auto [xSubMean, rtsd] = CalcRstd("rstd", xFp32, mean, one);
      xSubMean.TQue(Position::kPositionVecOut, 1, 1);
      rtsd.TQue(Position::kPositionVecOut, 1, 1);
      auto rstd_out = Store("rstd_out", rtsd);
      auto betaLocal = Load("betaLocal", beta).TQue(Position::kPositionVecIn, 1, 1);
      auto gammaLocal = Load("gammaLocal", gamma).TQue(Position::kPositionVecIn, 1, 1);
      auto y = CalcY("y", xSubMean, betaLocal, gammaLocal, rtsd).Use(xSubMean);
      auto y_out = Store("y_out", y);
      // 图的输出
      if (is_addtional_output) {
        auto x_out = Store("x_out", xOut);
        auto buf1 = Output("buf1", x_out);
      }
      auto buf2 = Output("buf2", mean_out);
      auto buf3 = Output("buf3", rstd_out);
      auto buf = Output("buf", y_out);
    }
  }
}

/*
for aBO
  for aBI
    for rO
      for rI
        load x1
        load x2
        load bias
        CalcMean
        Store X
        Load beta
        Load gamma
        CalcRstd
        Store mean
        Store rstd
        CalcY
        Store y
*/
void Add_Layer_Norm_Slice_BeforeAutofuseNewCg(AscGraph &graph, bool bias_broad_cast, bool is_addtional_output, int dtype_index) {
  auto ONE = ge::sym::kSymbolOne;
  auto ZERO = ge::sym::kSymbolZero;
  // 定义轴的大小
  auto A = ge::Symbol("A");
  auto R = ge::Symbol("R");
  // 定义轴
  auto a = graph.CreateAxis("A", A);
  auto r = graph.CreateAxis("R", R);
  // 轴变化
  auto [aBO, aBI] = graph.BlockSplit(a.id, "sbi", "sbo");
  auto [rO, rI] = graph.TileSplit(r.id, "sii", "sio");
  aBI->align = ge::Symbol(1);
  rI->align = ge::Symbol(16);
  /*
  // 是否需要输出x
  if (is_addtional_output) {
    graph.CreateOptionalAtt("additional_output", 1, 1, 1);
  } else {
    graph.CreateOptionalAtt("additional_output", 1, 0, 0);
  }
  */
  auto x1 = graph.CreateContiguousData("x1", dtypes[dtype_index], {a, r}, {{0, -2}, {-1}});
  auto x2 = graph.CreateContiguousData("x2", dtypes[dtype_index], {a, r}, {{0, -2}, {-1}});
  auto gamma = graph.CreateContiguousData("gamma", dtypes[dtype_index], {r});
  auto beta = graph.CreateContiguousData("beta", dtypes[dtype_index], {r});
  auto bias = bias_broad_cast ? graph.CreateContiguousData("bias", dtypes[dtype_index], {r})
                              : graph.CreateContiguousData("bias", dtypes[dtype_index], {a, r}, {{0, -2}, {-1}});

  LOOP(*aBO) {
    LOOP(*aBI) {
      VectorizedOutTensor xFp32({rO->id, rI->id});
      VectorizedOutTensor xSubMean({rO->id, rI->id});
      VectorizedOutTensor betaLocal({rO->id, rI->id});
      VectorizedOutTensor gammaLocal({rO->id, rI->id});
      VectorizedOutTensor y({rO->id, rI->id});
      LOOP(*rO) {
        auto x1Local = Load("x1Local", x1).TQue(Position::kPositionVecIn, 1, 1);
        auto x2Local = Load("x2Local", x2).TQue(Position::kPositionVecIn, 1, 1);
        auto biasLocal = Load("biasLocal", bias).TQue(Position::kPositionVecIn, 1, 1);
        auto [mean, xOut, xFp32_tmp] = CalcMeanSlice("mean", x1Local, x2Local, biasLocal, rI->id);
        xFp32 = std::move(xFp32_tmp);
        mean.TQue(Position::kPositionVecOut, 1, 1);
        xOut.TQue(Position::kPositionVecOut, 1, 1);
        static_cast<AscOpOutput>(xFp32).TQue(Position::kPositionVecOut, 1, 1);
        auto mean_out = Store("mean_out", mean);
        auto [xSubMean_tmp, rtsd] = CalcRstdSlice("rstd", static_cast<AscOpOutput>(xFp32), mean);
        xSubMean = std::move(xSubMean_tmp);
        static_cast<AscOpOutput>(xSubMean).Use(static_cast<AscOpOutput>(xFp32));
        rtsd.TQue(Position::kPositionVecOut, 1, 1);
        auto rstd_out = Store("rstd_out", rtsd);
        betaLocal = static_cast<AscOpOutput &&>(Load("betaLocal", beta).TQue(Position::kPositionVecIn, 1, 1));
        gammaLocal = static_cast<AscOpOutput &&>(Load("gammaLocal", gamma).TQue(Position::kPositionVecIn, 1, 1));
        y = static_cast<AscOpOutput &&>(CalcY("y", static_cast<AscOpOutput>(xSubMean),
                                              static_cast<AscOpOutput>(betaLocal), static_cast<AscOpOutput>(gammaLocal),
                                              rtsd)
                                            .Use(static_cast<AscOpOutput>(xSubMean)));
        auto y_out = Store("y_out", static_cast<AscOpOutput>(y));
        // 图的输出
        if (is_addtional_output) {
          auto x_out = Store("x_out", xOut);
          auto buf1 = Output("buf1", x_out);
        }
        auto buf2 = Output("buf2", mean_out);
        auto buf3 = Output("buf3", rstd_out);
        auto buf = Output("buf", y_out);
      }
    }
  }
}

/*
for aBO
  for aBI
    for rO
      for rI
        load x1
        load x2
        load bias
        CalcMean
        Store X
        Load beta
        Load gamma
        CalcRstd
        Store mean
        Store rstd
        CalcY
        Store y
*/
void Add_Layer_Norm_Welford_BeforeAutofuseNewCg(AscGraph &graph, bool bias_broad_cast, bool is_addtional_output, int dtype_index) {
  auto ONE = ge::sym::kSymbolOne;
  auto ZERO = ge::sym::kSymbolZero;
  // 定义轴的大小
  auto A = ge::Symbol("A");
  auto R = ge::Symbol("R");
  // 定义轴
  auto a = graph.CreateAxis("A", A);
  auto r = graph.CreateAxis("R", R);
  // 轴变化
  auto [aBO, aBI] = graph.BlockSplit(a.id, "wbi", "wbo");
  auto [rO, rI] = graph.TileSplit(r.id, "wii", "wio");
  aBI->align = ge::Symbol(1);
  rI->align = ge::Symbol(1);
  /*
  // 是否需要输出x
  if (is_addtional_output) {
    graph.CreateOptionalAtt("additional_output", 1, 1, 1);
  } else {
    graph.CreateOptionalAtt("additional_output", 1, 0, 0);
  }
  */
  auto x1 = graph.CreateContiguousData("x1", dtypes[dtype_index], {a, r}, {{0, -2}, {-1}});
  auto x2 = graph.CreateContiguousData("x2", dtypes[dtype_index], {a, r}, {{0, -2}, {-1}});
  auto gamma = graph.CreateContiguousData("gamma", dtypes[dtype_index], {r});
  auto beta = graph.CreateContiguousData("beta", dtypes[dtype_index], {r});
  auto bias = bias_broad_cast ? graph.CreateContiguousData("bias", dtypes[dtype_index], {r})
                              : graph.CreateContiguousData("bias", dtypes[dtype_index], {a, r}, {{0, -2}, {-1}});

  LOOP(*aBO) {
    VectorizedOutTensor workspace({aBI->id, rO->id, rI->id});
    LOOP(*aBI) {
      LOOP(*rO) {
        auto x1Local = Load("x1Local", x1).TQue(Position::kPositionVecIn, 1, 1);
        auto x2Local = Load("x2Local", x2).TQue(Position::kPositionVecIn, 1, 1);
        auto biasLocal = Load("biasLocal", bias).TQue(Position::kPositionVecIn, 1, 1);
        auto [xOut, xFp32, m, v] = VFWelfordPart1Update("part1", x1Local, x2Local, biasLocal);
        xOut.TQue(Position::kPositionVecOut, 1, 1);
        xFp32.TQue(Position::kPositionVecOut, 1, 1);
        m.TQue(Position::kPositionVecOut, 1, 1);
        v.TQue(Position::kPositionVecOut, 1, 1);
        auto x_fp32 = WorkspaceWithInput("x_fp32_workspace", Store("x_fp32_out", xFp32));
        workspace = std::move(x_fp32);

        auto [mean, rtsd] = VFWelfordPart1Finalize("part1Final", m, v, rI->id, rI->id);
        mean.TQue(Position::kPositionVecOut, 1, 1);
        rtsd.TQue(Position::kPositionVecOut, 1, 1);
        auto mean_out = Store("mean_out", mean);
        auto rstd_out = Store("rstd_out", rtsd);
        auto x32 = Load("x32", static_cast<AscOpOutput>(workspace)).Use(v);
        auto betaLocal = Load("betaLocal", beta).Use(x2Local);
        auto gammaLocal = Load("gammaLocal", gamma).Use(biasLocal);
        auto y = VFCalcYWelford("y", x32, betaLocal, gammaLocal).Use(m);
        auto y_out = Store("y_out", y);
        // 图的输出
        if (is_addtional_output) {
          auto x_out = Store("x_out", xOut);
          auto buf1 = Output("buf1", x_out);
        }
        auto buf2 = Output("buf2", mean_out);
        auto buf3 = Output("buf3", rstd_out);
        auto buf = Output("buf", y_out);
      }
    }
  }
}
}  // namespace ge

// 看护一二三阶段构图合一流程
// 36个tiling图对应61 + 64 + 64=189行代码左右
TEST_F(TestGenAddLayerNormalModelInfoE2E, test_36_graphs_origin) {
  std::vector<ascir::AscGraph> graphs;
  const std::function<void(AscGraph &graph, bool bias_broad_cast, bool is_addtional_output, int dtype_index)> func_array[3] = 
    {ge::Add_Layer_Norm_Normal_BeforeAutofuseNewCg,
     ge::Add_Layer_Norm_Slice_BeforeAutofuseNewCg,
     ge::Add_Layer_Norm_Welford_BeforeAutofuseNewCg};
  const int func_key[3] = {0, 1, 5};
  for (int func_index = 0; func_index < 3; func_index++) {
    for (int bias_brc_index = 1; bias_brc_index < 3; bias_brc_index++) {
      for (int is_addtional_output_index = 0; is_addtional_output_index < 2; is_addtional_output_index++) {
        for (int dtype_index = 1; dtype_index < 4; dtype_index++) {
          std::string name = std::string("graph_") + std::to_string(func_index) + "_" + std::to_string(bias_brc_index) +
            "_" + std::to_string(is_addtional_output_index);
          ascir::AscGraph graph(name.c_str());
          graph.SetTilingKey(1000 * dtype_index + 100 * is_addtional_output_index + 10 * func_key[func_index] + bias_brc_index);
          func_array[func_index](graph, (bias_brc_index == 2), (is_addtional_output_index == 1), (dtype_index - 1));
          graphs.emplace_back(graph);
        }
      }
    }
  }

  std::map<std::string, std::string> options;
  options["output_file_path"] = "./";
  options["gen_extra_info"] = "1";
  options["with_tiling_context"] = "1";
  options["duration_level"] = "1";
  options["solver_type"] = "HighPerf";
  options[kDumpDebugInfo] = "./";
  auto ret = std::system("rm -rf ./*_tiling_data.h");
  EXPECT_EQ(ret, 0);
  EXPECT_EQ(GenTilingImpl("AddLayerNorm", graphs, options), true);
  ret = std::system(std::string("cp -r ").append(ST_DIR).append("/testcase/tiling_func ./ -f").c_str());
  EXPECT_EQ(ret, 0);
  ret = std::system(std::string("cp ").append(ST_DIR).append("/testcase/create_struct.cpp ./ -f").c_str());
  EXPECT_EQ(ret, 0);
  ret = std::system("g++ -std=c++17 create_struct.cpp -o create_struct");
  EXPECT_EQ(ret, 0);
  ret = std::system("./create_struct");
  EXPECT_EQ(ret, 0);
  std::cout << "gen struct success!" << std::endl;
  ret = std::system(
      std::string("cp ").append(ST_DIR).append("/testcase/op_log.h ./tiling_func/include/ -f").c_str());
  EXPECT_EQ(ret, 0);

  ret = std::system("cp ./AddLayerNorm_*_tiling_func.cpp ./tiling_func/src/ -f");
  EXPECT_EQ(ret, 0);
  ret = std::system("cp ./AddLayerNorm_tiling_data.h ./tiling_func/include/ -f");
  EXPECT_EQ(ret, 0);
  ret = std::system("cp ./struct_info.h ./tiling_func/include/ -f");
  EXPECT_EQ(ret, 0);
  std::cout << "cp success!" << std::endl;

  std::string run_bash = "cd ./tiling_func/ && bash ./build.sh";
#ifdef ASCEND_INSTALL_PATH
  std::string path(ASCEND_INSTALL_PATH);
  run_bash += " " + path;
#endif
  run_bash.append(" ").append(TOP_DIR);
  std::cout << run_bash << std::endl;
  ret = std::system(run_bash.c_str());
  EXPECT_EQ(ret, 0);
  std::cout << "run bash success!" << std::endl;
}

TEST_F(TestGenAddLayerNormalModelInfoE2E, test_36_graphs_decision_tree) {
  std::vector<ascir::AscGraph> graphs;
  const std::function<void(AscGraph &graph, bool bias_broad_cast, bool is_addtional_output, int dtype_index)> func_array[3] = 
    {ge::Add_Layer_Norm_Normal_BeforeAutofuseNewCg,
     ge::Add_Layer_Norm_Slice_BeforeAutofuseNewCg,
     ge::Add_Layer_Norm_Welford_BeforeAutofuseNewCg};
  const int func_key[3] = {0, 1, 5};
  for (int func_index = 0; func_index < 3; func_index++) {
    for (int bias_brc_index = 1; bias_brc_index < 3; bias_brc_index++) {
      for (int is_addtional_output_index = 0; is_addtional_output_index < 2; is_addtional_output_index++) {
        for (int dtype_index = 1; dtype_index < 4; dtype_index++) {
          std::string name = std::string("graph_") + std::to_string(func_index) + "_" + std::to_string(bias_brc_index) +
            "_" + std::to_string(is_addtional_output_index);
          ascir::AscGraph graph(name.c_str());
          graph.SetTilingKey(1000 * dtype_index + 100 * is_addtional_output_index + 10 * func_key[func_index] + bias_brc_index);
          func_array[func_index](graph, (bias_brc_index == 2), (is_addtional_output_index == 1), (dtype_index - 1));
          graphs.emplace_back(graph);
        }
      }
    }
  }

  std::map<std::string, std::string> options;
  options["output_file_path"] = "./";
  options["gen_extra_info"] = "1";
  options["with_tiling_context"] = "1";
  options["duration_level"] = "1";
  options["solver_type"] = "HighPerf";
  options[kOpenDT] = kIsTrue;
  options[kDTDebug] = kIsTrue;
  options[kDumpDebugInfo] = "./";
  auto ret = std::system(std::string("cp -r ").append(ST_DIR).append("/testcase/decision_tree/config_files/add_layer_norm_e2e_36.json ./ -f").c_str());
  EXPECT_EQ(ret, 0);
  options[kDTConfigPath] = "./add_layer_norm_e2e_36.json";

  ret = std::system("rm -rf ./*_tiling_data.h");
  EXPECT_EQ(ret, 0);
  EXPECT_EQ(GenTilingImpl("AddLayerNorm", graphs, options), true);
  ret = std::system(std::string("cp -r ").append(ST_DIR).append("/testcase/tiling_func ./ -f").c_str());
  EXPECT_EQ(ret, 0);
  ret = std::system(std::string("cp ").append(ST_DIR).append("/testcase/create_struct.cpp ./ -f").c_str());
  EXPECT_EQ(ret, 0);
  ret = std::system("g++ -std=c++17 create_struct.cpp -o create_struct");
  EXPECT_EQ(ret, 0);
  ret = std::system("./create_struct");
  EXPECT_EQ(ret, 0);
  std::cout << "gen struct success!" << std::endl;
  ret = std::system(
      std::string("cp ").append(ST_DIR).append("/testcase/op_log.h ./tiling_func/include/ -f").c_str());
  EXPECT_EQ(ret, 0);

  ret = std::system("cp ./AddLayerNorm_*_tiling_func.cpp ./tiling_func/src/ -f");
  EXPECT_EQ(ret, 0);
  ret = std::system("cp ./AddLayerNorm_decision_tree.h ./tiling_func/src/ -f");
  EXPECT_EQ(ret, 0);
  ret = std::system("cp ./AddLayerNorm_tiling_data.h ./tiling_func/include/ -f");
  EXPECT_EQ(ret, 0);
  ret = std::system("cp ./struct_info.h ./tiling_func/include/ -f");
  EXPECT_EQ(ret, 0);
  std::cout << "cp success!" << std::endl;

  std::string run_bash = "cd ./tiling_func/ && bash ./build.sh";
#ifdef ASCEND_INSTALL_PATH
  std::string path(ASCEND_INSTALL_PATH);
  run_bash += " " + path;
#endif
  run_bash.append(" ").append(TOP_DIR);
  std::cout << run_bash << std::endl;
  ret = std::system(run_bash.c_str());
  EXPECT_EQ(ret, 0);
  std::cout << "run bash success!" << std::endl;
}

TEST_F(TestGenAddLayerNormalModelInfoE2E, case_axes_reorder) {
  std::vector<ascir::AscGraph> graphs;
  const std::function<void(AscGraph &graph, bool bias_broad_cast, bool is_addtional_output, int dtype_index)> func_array[3] = 
    {ge::Add_Layer_Norm_Normal_BeforeAutofuseNewCg,
     ge::Add_Layer_Norm_Slice_BeforeAutofuseNewCg,
     ge::Add_Layer_Norm_Welford_BeforeAutofuseNewCg};
  const int func_key[3] = {0, 1, 5};
  for (int func_index = 0; func_index < 3; func_index++) {
    for (int bias_brc_index = 1; bias_brc_index < 3; bias_brc_index++) {
      for (int is_addtional_output_index = 0; is_addtional_output_index < 2; is_addtional_output_index++) {
        for (int dtype_index = 1; dtype_index < 4; dtype_index++) {
          std::string name = std::string("graph_") + std::to_string(func_index) + "_" + std::to_string(bias_brc_index) +
            "_" + std::to_string(is_addtional_output_index);
          ascir::AscGraph graph(name.c_str());
          graph.SetTilingKey(1000 * dtype_index + 100 * is_addtional_output_index + 10 * func_key[func_index] + bias_brc_index);
          func_array[func_index](graph, (bias_brc_index == 2), (is_addtional_output_index == 1), (dtype_index - 1));
          graphs.emplace_back(graph);
        }
      }
    }
  }

  std::map<std::string, std::string> options;
  options["output_file_path"] = "./";
  options["gen_extra_info"] = "1";
  options["with_tiling_context"] = "1";
  options["duration_level"] = "1";
  options["solver_type"] = "AxesReorder";
  auto ret = std::system("rm -rf ./*_tiling_data.h");
  EXPECT_EQ(ret, 0);
  EXPECT_EQ(GenTilingImpl("AddLayerNorm", graphs, options), true);
  ret = std::system(std::string("cp -r ").append(ST_DIR).append("/testcase/tiling_func ./ -f").c_str());
  EXPECT_EQ(ret, 0);
  ret = std::system(std::string("cp ").append(ST_DIR).append("/testcase/create_struct.cpp ./ -f").c_str());
  EXPECT_EQ(ret, 0);
  ret = std::system("g++ -std=c++17 create_struct.cpp -o create_struct");
  EXPECT_EQ(ret, 0);
  ret = std::system("./create_struct");
  EXPECT_EQ(ret, 0);
  std::cout << "gen struct success!" << std::endl;
  ret = std::system(
      std::string("cp ").append(ST_DIR).append("/testcase/op_log.h ./tiling_func/include/ -f").c_str());
  EXPECT_EQ(ret, 0);

  ret = std::system("cp ./AddLayerNorm_*_tiling_func.cpp ./tiling_func/src/ -f");
  EXPECT_EQ(ret, 0);
  ret = std::system("cp ./AddLayerNorm_tiling_data.h ./tiling_func/include/ -f");
  EXPECT_EQ(ret, 0);
  ret = std::system("cp ./struct_info.h ./tiling_func/include/ -f");
  EXPECT_EQ(ret, 0);
  std::cout << "cp success!" << std::endl;

  std::string run_bash = "cd ./tiling_func/ && bash ./build.sh";
#ifdef ASCEND_INSTALL_PATH
  std::string path(ASCEND_INSTALL_PATH);
  run_bash += " " + path;
#endif
  run_bash.append(" ").append(TOP_DIR);
  std::cout << run_bash << std::endl;
  ret = std::system(run_bash.c_str());
  EXPECT_EQ(ret, 0);
  std::cout << "run bash success!" << std::endl;
}