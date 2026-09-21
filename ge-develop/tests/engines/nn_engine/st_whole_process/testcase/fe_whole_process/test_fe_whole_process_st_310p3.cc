/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>
#include "register/graph_optimizer/fusion_common/fusion_turbo.h"
#define protected public
#define private public
#include "framework/fe_running_env/include/fe_running_env.h"
#include "graph_optimizer/fe_graph_optimizer.h"

#include "ge_graph_dsl/graph_dsl.h"
#include "ge_graph_dsl/assert/graph_assert.h"
#include "graph_constructor/graph_constructor.h"
#undef protected
#undef private
#include "stub/te_fusion_api.h"

using namespace std;

namespace fe {
class STestFeWholeProcess310P3: public testing::Test
{
 protected:
  static void SetUpTestCase() {
    auto &fe_env = fe_env::FeRunningEnv::Instance();
    std::map<string, string> new_options;
    new_options[ge::PRECISION_MODE] = fe::FORCE_FP16;
    new_options[ge::SOC_VERSION] = "Ascend310P3";

    new_options[ge::AICORE_NUM] = "8";
    const char *soc_version = "Ascend310P3";
    rtSetSocVersion(soc_version);
    fe_env.InitRuningEnv(new_options);
  }

  static void TearDownTestCase() {
    te::TeStub::GetInstance().SetImpl(std::make_shared<te::TeStubApi>());
    auto &fe_env = fe_env::FeRunningEnv::Instance();
    fe_env.FinalizeEnv();
  }
};

TEST_F(STestFeWholeProcess310P3, test_mul_single_op_fp16) {
  auto &fe_env = fe_env::FeRunningEnv::Instance();
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("fe_st");
  ge::GeShape original_shape = ge::GeShape({3, 12, 5, 6});
  GraphConstructor test(graph, "", ge::FORMAT_NCHW, ge::DT_FLOAT, original_shape);

  test.AddOpDesc("Data_0", "Data", 1, 1)
      .AddOpDesc("Data_1", "Data", 1, 1)
      .SetInput("Mul_0:0", "Data_0")
      .SetInput("Mul_0:1", "Data_1")
      .SetInput("NetOutput:0", "Mul_0:0");
  EXPECT_EQ(fe_env.Run(graph, std::map<std::string, std::string>{}), fe::SUCCESS);
  EXPECT_EQ(graph->GetDirectNode().size(), 7);
}

/*
 *                                                          graph_1

┌────────┐  (0,1)   ┌────────┐  (0,0)   ┌──────────────┐
│ const1 │ ───────> │ matmul │ ───────> │  netoutput   │
└────────┘          └────────┘          └──────────────┘
                      ∧        (0,0)
                      └───────────────────────────────────────────────────────────────────────────────────────────┐
                                                                                                                  │
┌────────┐  (0,0)   ┌────────┐  (0,0)   ┌──────────────┐  (0,0)   ┌─────────┐  (0,0)   ┌─────────────┐  (0,0)   ┌────────┐
│ data0  │ ───────> │ quant  │ ───────> │    conv2d    │ ───────> │ dequant │ ───────> │ reduce_mean │ ───────> │ quant1 │
└────────┘          └────────┘          └──────────────┘          └─────────┘          └─────────────┘          └────────┘
                                          ∧
                                          │ (0,1)
                                          │
                                        ┌──────────────┐
                                        │ const_conv2d │
                                        └──────────────┘
 */
/* Data->Conv2D->Quant->Relu->MatMul->NetOutput */
TEST_F(STestFeWholeProcess310P3, test_matmul_quant) {
  /* This case uses real compilation environment. */
  te::TeStub::GetInstance().SetImpl(std::make_shared<te::RealTimeCompilation>());
  auto &fe_env = fe_env::FeRunningEnv::Instance();
  const auto data0 = OP_CFG(DATA)
      .TensorDesc(ge::FORMAT_NHWC, ge::DT_FLOAT, {8, 10, 10, 1536});

  const auto const_conv2d = OP_CFG(CONSTANT)
      .TensorDesc(ge::FORMAT_HWCN, ge::DT_INT8, {1, 1, 1536, 2048});

  std::vector<int64_t> strides = {1, 1, 1, 1};
  std::vector<int64_t> dilations = {1, 1, 1, 1};
  std::vector<int64_t> pads = {0, 0, 0, 0};
  const auto conv2d = OP_CFG(CONV2D)
      .InCnt(2).OutCnt(1)
      .Attr("strides", strides)
      .Attr("pads", pads)
      .Attr("data_format", "NHWC")
      .Attr("groups", 1)
      .Attr("offset_x", 1)
      .Attr("dilations", dilations)
      .TensorDesc(ge::FORMAT_NHWC, ge::DT_FLOAT, {}).Build("conv2d");

  const auto matmul = OP_CFG("MatMulV2")
      .InCnt(2).OutCnt(1)
      .Attr("transpose_x1", false)
      .Attr("transpose_x2", false)
      .Attr("offset_x", 0)
      .TensorDesc(ge::FORMAT_ND, ge::DT_FLOAT, {}).Build("matmul");

  const auto reduce_mean = OP_CFG("ReduceMean")
      .InCnt(1).OutCnt(1)
      .Attr("keep_dims", false)
      .Attr("noop_with_empty_axes", true)
      .TensorDesc(ge::FORMAT_ND, ge::DT_FLOAT, {});

  float scale = 1.0;
  float offset = 0.0;
  const auto quant = OP_CFG(ASCEND_QUANT)
      .Attr("scale", scale)
      .Attr("offset", offset)
      .InCnt(1).OutCnt(1)
      .TensorDesc(ge::FORMAT_ND, ge::DT_FLOAT, {});

  const auto quant1 = OP_CFG(ASCEND_QUANT)
      .Attr("scale", scale)
      .Attr("offset", offset)
      .InCnt(1).OutCnt(1)
      .TensorDesc(ge::FORMAT_ND, ge::DT_FLOAT, {});

  const auto dequant = OP_CFG(ASCEND_DEQUANT)
      .Attr("sqrt_mode", false)
      .Attr("relu_flag", false)
      .InCnt(1).OutCnt(1)
      .TensorDesc(ge::FORMAT_ND, ge::DT_FLOAT, {});

  const auto const1 = OP_CFG(CONSTANT)
      .TensorDesc(ge::FORMAT_NHWC, ge::DT_INT8, {2048, 32});

  DEF_GRAPH(graph_1) {
      CHAIN(NODE("data0", data0)->NODE("quant", quant)->NODE(conv2d));
      CHAIN(NODE("const_conv2d", const_conv2d)->EDGE(0, 1)->NODE(conv2d));
      CHAIN(NODE(conv2d)->NODE("dequant", dequant)->NODE("reduce_mean", reduce_mean)->
            NODE("quant1", quant1)->NODE(matmul));
      CHAIN(NODE("const1", const1)->EDGE(0, 1)->NODE(matmul)->NODE("netoutput", NETOUTPUT));
  };

  auto graph = ge::ToComputeGraph(graph_1);
  for (auto &node : graph->GetDirectNode()) {
    auto op = node->GetOpDesc();
    if (node->GetName() == "conv2d") {
      auto input1 = node->GetOpDesc()->MutableInputDesc(1);
      input1->SetOriginFormat(ge::FORMAT_HWCN);
      input1->SetFormat(ge::FORMAT_HWCN);
    }
    if (node->GetName() == "dequant") {
      unique_ptr<uint64_t[]> data(new(std::nothrow) uint64_t[2048]);
      ge::GeTensorDesc weight(ge::GeShape({2048}), ge::FORMAT_NHWC, ge::DT_UINT64);
      WeightInfo w(weight, data.get());
      fe::FusionTurbo ft(graph);
      ft.AddWeight(node, w);
    }

    if (node->GetName() == "reduce_mean") {
      unique_ptr<int32_t[]> data(new(std::nothrow) int32_t[2]);
      data[0] = 1;
      data[1] = 2;
      ge::GeTensorDesc weight(ge::GeShape({2}), ge::FORMAT_ND, ge::DT_INT32);
      WeightInfo w(weight, data.get());
      fe::FusionTurbo ft(graph);
      ft.AddWeight(node, w);
    }
  }

  DUMP_GRAPH_WHEN("OptimizeOriginalGraph_FeDistHeavyFormatAfter", "PreRunAfterBuild");
  EXPECT_EQ(fe_env.Run(graph, std::map<std::string, std::string>{}, true), fe::SUCCESS);
  EXPECT_EQ(graph->GetDirectNode().size(), 19);
  CHECK_GRAPH(OptimizeOriginalGraph_FeDistHeavyFormatAfter) {
    EXPECT_EQ(graph->GetDirectNodesSize(), 14);
    for (auto &node : graph->GetDirectNode()) {
      if (node->GetName() == "quant1") {
        auto op = node->GetOpDesc();
        auto input = op->MutableInputDesc(0);
        ASSERT_NE(input, nullptr);
        EXPECT_EQ(input->GetFormat(), ge::FORMAT_FRACTAL_NZ);
        EXPECT_EQ(input->GetDataType(), ge::DT_FLOAT16);
        EXPECT_EQ(input->GetShape(), ge::GeShape({128, 1, 16, 16}));
        auto output = op->MutableOutputDesc(0);
        ASSERT_NE(output, nullptr);
        EXPECT_EQ(output->GetFormat(), ge::FORMAT_FRACTAL_NZ);
        EXPECT_EQ(output->GetDataType(), ge::DT_INT8);
        EXPECT_EQ(output->GetShape(), ge::GeShape({64, 1, 16, 32}));
      }
    }
  };

  CHECK_GRAPH(PreRunAfterBuild) {
    for (auto &node : graph->GetDirectNode()) {
      if (node->GetName() == "quant1") {
        auto op = node->GetOpDesc();
        auto input = op->MutableInputDesc(0);
        ASSERT_NE(input, nullptr);
        EXPECT_EQ(input->GetFormat(), ge::FORMAT_FRACTAL_NZ);
        EXPECT_EQ(input->GetDataType(), ge::DT_FLOAT16);
        EXPECT_EQ(input->GetShape(), ge::GeShape({128, 1, 16, 16}));
        auto output = op->MutableOutputDesc(0);
        ASSERT_NE(output, nullptr);
        EXPECT_EQ(output->GetFormat(), ge::FORMAT_FRACTAL_NZ);
        EXPECT_EQ(output->GetDataType(), ge::DT_INT8);
        EXPECT_EQ(output->GetShape(), ge::GeShape({64, 1, 16, 32}));
        int64_t tbe_kernel_size = 0;
        string kernel_name;
        string magic;
        int64_t block_dim = 0;
        ge::AttrUtils::GetInt(op, "_tbeKernelSize", tbe_kernel_size);
        ge::AttrUtils::GetStr(op, "_tbe_kernel_name_for_load", kernel_name);
        ge::AttrUtils::GetInt(op, "tvm_blockdim", block_dim);
        ge::AttrUtils::GetStr(op, "tvm_magic", magic);
        EXPECT_EQ(tbe_kernel_size, 1448);
        EXPECT_EQ(kernel_name,
                  "te_ascendquant_a1d2ba3e017fb47c5c72b1b7a8731b478e57f76066bdeef3c625e801ec18bbdc_1__kernel0");
        EXPECT_EQ(block_dim, 8);
        EXPECT_EQ(magic, "RT_DEV_BINARY_MAGIC_ELF");
      }
    }
  };
}

TEST_F(STestFeWholeProcess310P3, test_unsortedsegmentsum) {
  auto &fe_env = fe_env::FeRunningEnv::Instance();
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("fe_st");
  string network_path = fe_env::FeRunningEnv::GetNetworkPath("unsortedsegmentsum.txt");
  bool state = ge::GraphUtils::LoadGEGraph(network_path.c_str(), graph);
  ASSERT_EQ(state, true);
  EXPECT_EQ(graph->GetName(), "unsortedsegmentsum_L2_xlsx_16_31_139_16_fp16_int32_0d0_1d0_0d0_100d0_u_ND_16_xfun_0_stream.pb");
  EXPECT_EQ(graph->GetDirectNode().size(), 4);
  std::map<string, string> session_options;
  EXPECT_EQ(fe_env.Run(graph, session_options), fe::SUCCESS);
  fe_env.Run(graph, session_options);
}
}
