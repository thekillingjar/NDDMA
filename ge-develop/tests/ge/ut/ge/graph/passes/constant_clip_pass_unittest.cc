/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <vector>
#include <climits>
#include <float.h>
#include <gtest/gtest.h>

#include "macro_utils/dt_public_scope.h"
#include "graph/passes/feature/constant_clip_pass.h"
#include "graph_builder_utils.h"
#include "macro_utils/dt_public_unscope.h"

#include "graph/graph.h"
#include "graph/utils/op_desc_utils.h"
#include "common/ge_inner_error_codes.h"
#include "common/types.h"
#include "graph/ge_local_context.h"

using namespace std;
using namespace testing;
using namespace ge;

constexpr float FP16_MAX_VALUE = 65504;
constexpr float FP16_MIN_VALUE = -65504;
constexpr float BF16_MAX_VALUE = 3.38E38;
constexpr float BF16_MIN_VALUE = -3.38E38;
constexpr float FLOAT_DELTA = 1e-6F;

class UtestGraphPassesConstantClipPass : public testing::Test {
protected:
  void SetUp() {
    std::map<std::string, std::string> options{{"ge.is_weight_clip", "1"}};
    GetThreadLocalContext().SetGlobalOption(options);
  }
  void TearDown() {
    std::map<std::string, std::string> options{{"ge.is_weight_clip", "0"}};
    GetThreadLocalContext().SetGlobalOption(options);
    for (auto &name_to_pass : names_to_pass) {
      delete name_to_pass.second;
    }
  }

  NamesToPass names_to_pass;
};

//    const
//      |
//     cast
// change to be
// ******************************************
//   const const_min const_max
//     \      |         /
//        cplibyvalue
//            |
//           cast
namespace {
ComputeGraphPtr BuildGraph1(DataType src_dt, DataType dst_dt, double value) {
  auto builder = ut::GraphBuilder("g1");
  auto const_node = builder.AddNode("const1", CONSTANT, 0, 1, FORMAT_ND, src_dt, {});
  auto cast = builder.AddNode("cast", CAST, 1, 1, FORMAT_ND, src_dt, {});
  cast->GetOpDesc()->MutableOutputDesc(0)->SetDataType(dst_dt);
  GeTensorDesc weight_desc(GeShape(), FORMAT_ND, src_dt);

  GeTensorPtr tensor = nullptr;
  if (src_dt == DT_DOUBLE) {
    double value_tmp = value;
    tensor = std::make_shared<GeTensor>(weight_desc, (uint8_t *) (&value_tmp), sizeof(value_tmp));
  } else {
    float value_tmp = static_cast<float>(value);
    tensor = std::make_shared<GeTensor>(weight_desc, (uint8_t *) (&value_tmp), sizeof(value_tmp));
  }
  OpDescUtils::SetWeights(const_node, {tensor});

  builder.AddDataEdge(const_node, 0, cast, 0);
  return builder.GetGraph();
}

ComputeGraphPtr BuildGraphNoConst(DataType src_dt, DataType dst_dt) {
  auto builder = ut::GraphBuilder("g1");
  auto data_node = builder.AddNode("data", DATA, 0, 1, FORMAT_ND, src_dt, {});
  auto cast = builder.AddNode("cast", CAST, 1, 1, FORMAT_ND, src_dt, {});
  cast->GetOpDesc()->MutableOutputDesc(0)->SetDataType(dst_dt);

  builder.AddDataEdge(data_node, 0, cast, 0);
  return builder.GetGraph();
}

ComputeGraphPtr BuildGraphConstNoValue(DataType src_dt, DataType dst_dt) {
  auto builder = ut::GraphBuilder("g1");
  auto const_node = builder.AddNode("const1", CONSTANT, 0, 1, FORMAT_ND, src_dt, {});
  auto cast = builder.AddNode("cast", CAST, 1, 1, FORMAT_ND, src_dt, {});
  cast->GetOpDesc()->MutableOutputDesc(0)->SetDataType(dst_dt);
  builder.AddDataEdge(const_node, 0, cast, 0);
  return builder.GetGraph();
}

TEST_F(UtestGraphPassesConstantClipPass, success_fp16_fp32_max_const_no_value) {
  ge::ComputeGraphPtr graph = BuildGraphConstNoValue(DT_FLOAT16, DT_FLOAT);
  EXPECT_EQ(graph->GetAllNodes().size(), 2);

  names_to_pass.push_back( {"ConstantClipPass", new ConstantClipPass});
  GEPass pass(graph);
  EXPECT_EQ(pass.Run(names_to_pass), SUCCESS);
  EXPECT_EQ(graph->GetAllNodes().size(), 2);
}

TEST_F(UtestGraphPassesConstantClipPass, success_fp16_fp32_max) {
  float max_val = 65505;
  ge::ComputeGraphPtr graph = BuildGraph1(DT_FLOAT16, DT_FLOAT, max_val);
  EXPECT_EQ(graph->GetAllNodes().size(), 2);

  names_to_pass.push_back( {"ConstantClipPass", new ConstantClipPass});
  GEPass pass(graph);
  EXPECT_EQ(pass.Run(names_to_pass), SUCCESS);
  EXPECT_EQ(graph->GetAllNodes().size(), 2);

  std::map<std::string, std::string> options{{"ge.is_weight_clip", "1"}};
  GetThreadLocalContext().SetGlobalOption(options);

  EXPECT_EQ(pass.Run(names_to_pass), SUCCESS);
  EXPECT_EQ(graph->GetAllNodes().size(), 2);
}

TEST_F(UtestGraphPassesConstantClipPass, success_FP32_int32_max) {
  float max_val = 65505;
  ge::ComputeGraphPtr graph = BuildGraph1(DT_FLOAT, DT_INT32, max_val);
  EXPECT_EQ(graph->GetAllNodes().size(), 2);

  names_to_pass.push_back( {"ConstantClipPass", new ConstantClipPass});
  GEPass pass(graph);
  EXPECT_EQ(pass.Run(names_to_pass), SUCCESS);
  EXPECT_EQ(graph->GetAllNodes().size(), 2);
}

TEST_F(UtestGraphPassesConstantClipPass, success_fp32_fp16_not_inf) {
  float max_val = 65503;
  ge::ComputeGraphPtr graph = BuildGraph1(DT_FLOAT, DT_FLOAT16, max_val);
  EXPECT_EQ(graph->GetAllNodes().size(), 2);
  names_to_pass.push_back( {"ConstantClipPass", new ConstantClipPass});
  GEPass pass(graph);
  EXPECT_EQ(pass.Run(names_to_pass), SUCCESS);
  EXPECT_EQ(graph->GetAllNodes().size(), 2);
  auto clip_node = graph->FindNode("cast_const_clip");
  EXPECT_EQ(clip_node, nullptr);
}

TEST_F(UtestGraphPassesConstantClipPass, success_fp32_fp16_not_link_const) {
  ge::ComputeGraphPtr graph = BuildGraphNoConst(DT_FLOAT, DT_FLOAT16);
  EXPECT_EQ(graph->GetAllNodes().size(), 2);
  names_to_pass.push_back( {"ConstantClipPass", new ConstantClipPass});
  GEPass pass(graph);
  EXPECT_EQ(pass.Run(names_to_pass), SUCCESS);
  EXPECT_EQ(graph->GetAllNodes().size(), 2);
  auto clip_node = graph->FindNode("cast_const_clip");
  EXPECT_EQ(clip_node, nullptr);
}

TEST_F(UtestGraphPassesConstantClipPass, success_fp32_fp16_max) {
  float max_val = 65505;
  ge::ComputeGraphPtr graph = BuildGraph1(DT_FLOAT, DT_FLOAT16, max_val);
  EXPECT_EQ(graph->GetAllNodes().size(), 2);
  names_to_pass.push_back({"ConstantClipPass", new ConstantClipPass});
  GEPass pass(graph);
  EXPECT_EQ(pass.Run(names_to_pass), SUCCESS);
  EXPECT_EQ(graph->GetAllNodes().size(), 5);
  auto clip_node = graph->FindNode("cast_const_clip");
  EXPECT_NE(clip_node, nullptr);
  auto in_nodes = clip_node->GetInNodes();
  EXPECT_EQ(in_nodes.size(), 3);

  auto const_min = in_nodes.at(1);
  EXPECT_NE(const_min, nullptr);
  auto min_weights = OpDescUtils::GetWeights(const_min);
  EXPECT_EQ(min_weights.size(), 1);
  auto min_weight = min_weights[0];
  EXPECT_NE(min_weight, nullptr);
  DataType const_dt = min_weight->GetTensorDesc().GetDataType();
  EXPECT_EQ(const_dt, DT_FLOAT);
  const float *const min_data = reinterpret_cast<const float *>(min_weight->GetData().GetData());
  bool min_equal = (fabsf(*min_data - FP16_MIN_VALUE) <= FLOAT_DELTA);
  EXPECT_TRUE(min_equal);

  auto const_max = in_nodes.at(2);
  EXPECT_NE(const_max, nullptr);
  auto max_weights = OpDescUtils::GetWeights(const_max);
  EXPECT_EQ(max_weights.size(), 1);
  auto max_weight = max_weights[0];
  EXPECT_NE(max_weight, nullptr);
  const_dt = max_weight->GetTensorDesc().GetDataType();
  EXPECT_EQ(const_dt, DT_FLOAT);
  const float *const max_data = reinterpret_cast<const float *>(max_weight->GetData().GetData());
  bool max_equal = (fabsf(*max_data - FP16_MAX_VALUE) <= FLOAT_DELTA);
  EXPECT_TRUE(max_equal);
}

TEST_F(UtestGraphPassesConstantClipPass, success_fp32_fp16_min) {
  float min_val = -65505;
  ge::ComputeGraphPtr graph = BuildGraph1(DT_FLOAT, DT_FLOAT16, min_val);
  EXPECT_EQ(graph->GetAllNodes().size(), 2);
  names_to_pass.push_back( {"ConstantClipPass", new ConstantClipPass});
  GEPass pass(graph);
  EXPECT_EQ(pass.Run(names_to_pass), SUCCESS);
  EXPECT_EQ(graph->GetAllNodes().size(), 5);
  auto clip_node = graph->FindNode("cast_const_clip");
  EXPECT_NE(clip_node, nullptr);
  auto in_nodes = clip_node->GetInNodes();
  EXPECT_EQ(in_nodes.size(), 3);

  auto const_min = in_nodes.at(1);
  EXPECT_NE(const_min, nullptr);
  auto min_weights = OpDescUtils::GetWeights(const_min);
  EXPECT_EQ(min_weights.size(), 1);
  auto min_weight = min_weights[0];
  EXPECT_NE(min_weight, nullptr);
  DataType const_dt = min_weight->GetTensorDesc().GetDataType();
  EXPECT_EQ(const_dt, DT_FLOAT);
  const float *const min_data = reinterpret_cast<const float *>(min_weight->GetData().GetData());
  bool min_equal = (fabsf(*min_data - FP16_MIN_VALUE) <= FLOAT_DELTA);
  EXPECT_TRUE(min_equal);

  auto const_max = in_nodes.at(2);
  EXPECT_NE(const_max, nullptr);
  auto max_weights = OpDescUtils::GetWeights(const_max);
  EXPECT_EQ(max_weights.size(), 1);
  auto max_weight = max_weights[0];
  EXPECT_NE(max_weight, nullptr);
  const_dt = max_weight->GetTensorDesc().GetDataType();
  EXPECT_EQ(const_dt, DT_FLOAT);
  const float *const max_data = reinterpret_cast<const float *>(max_weight->GetData().GetData());
  bool max_equal = (fabsf(*max_data - FP16_MAX_VALUE) <= FLOAT_DELTA);
  EXPECT_TRUE(max_equal);
}

TEST_F(UtestGraphPassesConstantClipPass, success_fp64_fp16_max) {
  float max_val = 65505;
  ge::ComputeGraphPtr graph = BuildGraph1(DT_DOUBLE, DT_FLOAT16, max_val);
  EXPECT_EQ(graph->GetAllNodes().size(), 2);
  names_to_pass.push_back( {"ConstantClipPass", new ConstantClipPass});
  GEPass pass(graph);
  EXPECT_EQ(pass.Run(names_to_pass), SUCCESS);
  EXPECT_EQ(graph->GetAllNodes().size(), 5);
  auto clip_node = graph->FindNode("cast_const_clip");
  EXPECT_NE(clip_node, nullptr);
  auto in_nodes = clip_node->GetInNodes();
  EXPECT_EQ(in_nodes.size(), 3);

  auto const_min = in_nodes.at(1);
  EXPECT_NE(const_min, nullptr);
  auto min_weights = OpDescUtils::GetWeights(const_min);
  EXPECT_EQ(min_weights.size(), 1);
  auto min_weight = min_weights[0];
  EXPECT_NE(min_weight, nullptr);
  DataType const_dt = min_weight->GetTensorDesc().GetDataType();
  EXPECT_EQ(const_dt, DT_DOUBLE);
  const double *const min_data = reinterpret_cast<const double *>(min_weight->GetData().GetData());
  bool min_equal = (fabsf(*min_data - FP16_MIN_VALUE) <= FLOAT_DELTA);
  EXPECT_TRUE(min_equal);

  auto const_max = in_nodes.at(2);
  EXPECT_NE(const_max, nullptr);
  auto max_weights = OpDescUtils::GetWeights(const_max);
  EXPECT_EQ(max_weights.size(), 1);
  auto max_weight = max_weights[0];
  EXPECT_NE(max_weight, nullptr);
  const_dt = max_weight->GetTensorDesc().GetDataType();
  EXPECT_EQ(const_dt, DT_DOUBLE);
  const double *const max_data = reinterpret_cast<const double *>(max_weight->GetData().GetData());
  bool max_equal = (fabsf(*max_data - FP16_MAX_VALUE) <= FLOAT_DELTA);
  EXPECT_TRUE(max_equal);
}

TEST_F(UtestGraphPassesConstantClipPass, success_fp64_fp16_min) {
  float min_val = -65505;
  ge::ComputeGraphPtr graph = BuildGraph1(DT_DOUBLE, DT_FLOAT16, min_val);
  EXPECT_EQ(graph->GetAllNodes().size(), 2);
  names_to_pass.push_back( {"ConstantClipPass", new ConstantClipPass});
  GEPass pass(graph);
  EXPECT_EQ(pass.Run(names_to_pass), SUCCESS);
  EXPECT_EQ(graph->GetAllNodes().size(), 5);
  auto clip_node = graph->FindNode("cast_const_clip");
  EXPECT_NE(clip_node, nullptr);
  auto in_nodes = clip_node->GetInNodes();
  EXPECT_EQ(in_nodes.size(), 3);

  auto const_min = in_nodes.at(1);
  EXPECT_NE(const_min, nullptr);
  auto min_weights = OpDescUtils::GetWeights(const_min);
  EXPECT_EQ(min_weights.size(), 1);
  auto min_weight = min_weights[0];
  EXPECT_NE(min_weight, nullptr);
  DataType const_dt = min_weight->GetTensorDesc().GetDataType();
  EXPECT_EQ(const_dt, DT_DOUBLE);
  const double *const min_data = reinterpret_cast<const double *>(min_weight->GetData().GetData());
  bool min_equal = (fabsf(*min_data - FP16_MIN_VALUE) <= FLOAT_DELTA);
  EXPECT_TRUE(min_equal);

  auto const_max = in_nodes.at(2);
  EXPECT_NE(const_max, nullptr);
  auto max_weights = OpDescUtils::GetWeights(const_max);
  EXPECT_EQ(max_weights.size(), 1);
  auto max_weight = max_weights[0];
  EXPECT_NE(max_weight, nullptr);
  const_dt = max_weight->GetTensorDesc().GetDataType();
  EXPECT_EQ(const_dt, DT_DOUBLE);
  const double *const max_data = reinterpret_cast<const double *>(max_weight->GetData().GetData());
  bool max_equal = (fabsf(*max_data - FP16_MAX_VALUE) <= FLOAT_DELTA);
  EXPECT_TRUE(max_equal);
}

TEST_F(UtestGraphPassesConstantClipPass, success_fp32_bf16_max) {
  float max_val = FLT_MAX;
  ge::ComputeGraphPtr graph = BuildGraph1(DT_FLOAT, DT_BF16, max_val);
  EXPECT_EQ(graph->GetAllNodes().size(), 2);
  names_to_pass.push_back( {"ConstantClipPass", new ConstantClipPass});
  GEPass pass(graph);
  EXPECT_EQ(pass.Run(names_to_pass), SUCCESS);
  EXPECT_EQ(graph->GetAllNodes().size(), 5);
  auto clip_node = graph->FindNode("cast_const_clip");
  EXPECT_NE(clip_node, nullptr);
  auto in_nodes = clip_node->GetInNodes();
  EXPECT_EQ(in_nodes.size(), 3);

  auto const_min = in_nodes.at(1);
  EXPECT_NE(const_min, nullptr);
  auto min_weights = OpDescUtils::GetWeights(const_min);
  EXPECT_EQ(min_weights.size(), 1);
  auto min_weight = min_weights[0];
  EXPECT_NE(min_weight, nullptr);
  DataType const_dt = min_weight->GetTensorDesc().GetDataType();
  EXPECT_EQ(const_dt, DT_FLOAT);
  const float *const min_data = reinterpret_cast<const float *>(min_weight->GetData().GetData());
  bool min_equal = (fabsf(*min_data - BF16_MIN_VALUE) <= FLOAT_DELTA);
  EXPECT_TRUE(min_equal);

  auto const_max = in_nodes.at(2);
  EXPECT_NE(const_max, nullptr);
  auto max_weights = OpDescUtils::GetWeights(const_max);
  EXPECT_EQ(max_weights.size(), 1);
  auto max_weight = max_weights[0];
  EXPECT_NE(max_weight, nullptr);
  const_dt = max_weight->GetTensorDesc().GetDataType();
  EXPECT_EQ(const_dt, DT_FLOAT);
  const float *const max_data = reinterpret_cast<const float *>(max_weight->GetData().GetData());
  bool max_equal = (fabsf(*max_data - BF16_MAX_VALUE) <= FLOAT_DELTA);
  EXPECT_TRUE(max_equal);
}

TEST_F(UtestGraphPassesConstantClipPass, success_fp32_bf16_min) {
  float min_val = -FLT_MAX;
  ge::ComputeGraphPtr graph = BuildGraph1(DT_FLOAT, DT_BF16, min_val);
  EXPECT_EQ(graph->GetAllNodes().size(), 2);
  names_to_pass.push_back( {"ConstantClipPass", new ConstantClipPass});
  GEPass pass(graph);
  EXPECT_EQ(pass.Run(names_to_pass), SUCCESS);
  EXPECT_EQ(graph->GetAllNodes().size(), 5);
  auto clip_node = graph->FindNode("cast_const_clip");
  EXPECT_NE(clip_node, nullptr);
  auto in_nodes = clip_node->GetInNodes();
  EXPECT_EQ(in_nodes.size(), 3);

  auto const_min = in_nodes.at(1);
  EXPECT_NE(const_min, nullptr);
  auto min_weights = OpDescUtils::GetWeights(const_min);
  EXPECT_EQ(min_weights.size(), 1);
  auto min_weight = min_weights[0];
  EXPECT_NE(min_weight, nullptr);
  DataType const_dt = min_weight->GetTensorDesc().GetDataType();
  EXPECT_EQ(const_dt, DT_FLOAT);
  const float *const min_data = reinterpret_cast<const float *>(min_weight->GetData().GetData());
  bool min_equal = (fabsf(*min_data - BF16_MIN_VALUE) <= FLOAT_DELTA);
  EXPECT_TRUE(min_equal);

  auto const_max = in_nodes.at(2);
  EXPECT_NE(const_max, nullptr);
  auto max_weights = OpDescUtils::GetWeights(const_max);
  EXPECT_EQ(max_weights.size(), 1);
  auto max_weight = max_weights[0];
  EXPECT_NE(max_weight, nullptr);
  const_dt = max_weight->GetTensorDesc().GetDataType();
  EXPECT_EQ(const_dt, DT_FLOAT);
  const float *const max_data = reinterpret_cast<const float *>(max_weight->GetData().GetData());
  bool max_equal = (fabsf(*max_data - BF16_MAX_VALUE) <= FLOAT_DELTA);
  EXPECT_TRUE(max_equal);
}

TEST_F(UtestGraphPassesConstantClipPass, success_fp64_fp32_max) {
  double max_val = DBL_MAX;
  ge::ComputeGraphPtr graph = BuildGraph1(DT_DOUBLE, DT_FLOAT, max_val);
  EXPECT_EQ(graph->GetAllNodes().size(), 2);
  names_to_pass.push_back( {"ConstantClipPass", new ConstantClipPass});
  GEPass pass(graph);
  EXPECT_EQ(pass.Run(names_to_pass), SUCCESS);
  EXPECT_EQ(graph->GetAllNodes().size(), 5);
  auto clip_node = graph->FindNode("cast_const_clip");
  EXPECT_NE(clip_node, nullptr);
  auto in_nodes = clip_node->GetInNodes();
  EXPECT_EQ(in_nodes.size(), 3);

  auto const_min = in_nodes.at(1);
  EXPECT_NE(const_min, nullptr);
  auto min_weights = OpDescUtils::GetWeights(const_min);
  EXPECT_EQ(min_weights.size(), 1);
  auto min_weight = min_weights[0];
  EXPECT_NE(min_weight, nullptr);
  DataType const_dt = min_weight->GetTensorDesc().GetDataType();
  EXPECT_EQ(const_dt, DT_DOUBLE);
  const double *const min_data = reinterpret_cast<const double *>(min_weight->GetData().GetData());
  bool min_equal = (fabsf(*min_data + FLT_MAX) <= FLOAT_DELTA);
  EXPECT_TRUE(min_equal);

  auto const_max = in_nodes.at(2);
  EXPECT_NE(const_max, nullptr);
  auto max_weights = OpDescUtils::GetWeights(const_max);
  EXPECT_EQ(max_weights.size(), 1);
  auto max_weight = max_weights[0];
  EXPECT_NE(max_weight, nullptr);
  const_dt = max_weight->GetTensorDesc().GetDataType();
  EXPECT_EQ(const_dt, DT_DOUBLE);
  const double *const max_data = reinterpret_cast<const double *>(max_weight->GetData().GetData());
  bool max_equal = (fabsf(*max_data - FLT_MAX) <= FLOAT_DELTA);
  EXPECT_TRUE(max_equal);
}

TEST_F(UtestGraphPassesConstantClipPass, success_fp64_fp32_min) {
  float min_val = -DBL_MAX;
  ge::ComputeGraphPtr graph = BuildGraph1(DT_DOUBLE, DT_FLOAT, min_val);
  EXPECT_EQ(graph->GetAllNodes().size(), 2);
  names_to_pass.push_back( {"ConstantClipPass", new ConstantClipPass});
  GEPass pass(graph);
  EXPECT_EQ(pass.Run(names_to_pass), SUCCESS);
  EXPECT_EQ(graph->GetAllNodes().size(), 5);
  auto clip_node = graph->FindNode("cast_const_clip");
  EXPECT_NE(clip_node, nullptr);
  auto in_nodes = clip_node->GetInNodes();
  EXPECT_EQ(in_nodes.size(), 3);

  auto const_min = in_nodes.at(1);
  EXPECT_NE(const_min, nullptr);
  auto min_weights = OpDescUtils::GetWeights(const_min);
  EXPECT_EQ(min_weights.size(), 1);
  auto min_weight = min_weights[0];
  EXPECT_NE(min_weight, nullptr);
  DataType const_dt = min_weight->GetTensorDesc().GetDataType();
  EXPECT_EQ(const_dt, DT_DOUBLE);
  const double *const min_data = reinterpret_cast<const double *>(min_weight->GetData().GetData());
  bool min_equal = (fabsf(*min_data + FLT_MAX) <= FLOAT_DELTA);
  EXPECT_TRUE(min_equal);

  auto const_max = in_nodes.at(2);
  EXPECT_NE(const_max, nullptr);
  auto max_weights = OpDescUtils::GetWeights(const_max);
  EXPECT_EQ(max_weights.size(), 1);
  auto max_weight = max_weights[0];
  EXPECT_NE(max_weight, nullptr);
  const_dt = max_weight->GetTensorDesc().GetDataType();
  EXPECT_EQ(const_dt, DT_DOUBLE);
  const double *const max_data = reinterpret_cast<const double *>(max_weight->GetData().GetData());
  bool max_equal = (fabsf(*max_data - FLT_MAX) <= FLOAT_DELTA);
  EXPECT_TRUE(max_equal);
}
}
