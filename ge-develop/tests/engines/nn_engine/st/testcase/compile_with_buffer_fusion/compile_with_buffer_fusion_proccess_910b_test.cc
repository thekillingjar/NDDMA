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

#include "fe_llt_utils.h"
#include "common/fe_op_info_common.h"
#include "compute_graph.h"
#include "graph/utils/attr_utils.h"
#include "itf_handler/itf_handler.h"
#include "graph_optimizer/buffer_fusion/buffer_fusion_pass_base.h"

#define protected public
#define private public
#include "fusion_manager/fusion_manager.h"
#undef private
#undef protected

using namespace ge;
namespace fe {
class CompileWithBufferFusionProcess910BTest : public testing::Test {
 protected:
  static void SetUpTestCase() {
    cout << "CompileWithBufferFusionProcess910BTest TearDown" << endl;
    map<string, string> options;
    options.emplace(ge::SOC_VERSION, "Ascend910B1");
    InitWithOptions(options);
    FEGraphOptimizerPtr graph_optimizer_ptr = FusionManager::Instance(AI_CORE_NAME).graph_opt_;
    EXPECT_EQ(graph_optimizer_ptr->Initialize(options, nullptr), SUCCESS);
  }

  static void TearDownTestCase() {
    cout << "CompileWithBufferFusionProcess910BTest TearDown" << endl;
    Finalize();
  }

  /**
   *    Data \                                                  Const \                  Const \
   * Data -> Add -> Cast -> Square -> ReduceMeanD -> Add -> Sqrt -> RealDiv -> Cast -> Mul -> Mul
   *          \                                 Const /                                /
   *           ------------------------------------------------------------------------
   */
  ge::ComputeGraphPtr CreateGraphWithType(const int32_t &type) {
    vector<int64_t> dims = {3, 4, 5, 6};
    ge::GeShape shape(dims);
    ge::GeTensorDesc tensor_desc_fp32(shape, ge::FORMAT_NCHW, ge::DT_FLOAT);
    tensor_desc_fp32.SetOriginShape(shape);
    tensor_desc_fp32.SetOriginDataType(ge::DT_FLOAT);
    tensor_desc_fp32.SetOriginFormat(ge::FORMAT_NCHW);

    ge::GeTensorDesc tensor_desc_fp16(shape, ge::FORMAT_NCHW, ge::DT_FLOAT16);
    tensor_desc_fp16.SetOriginShape(shape);
    tensor_desc_fp16.SetOriginDataType(ge::DT_FLOAT);
    tensor_desc_fp16.SetOriginFormat(ge::FORMAT_NCHW);

    OpDescPtr data1_op = std::make_shared<OpDesc>("data1", "PlaceHolder");
    OpDescPtr data2_op = std::make_shared<OpDesc>("data2", "PlaceHolder");
    OpDescPtr const1_op = std::make_shared<OpDesc>("const1", "Const");
    OpDescPtr const2_op = std::make_shared<OpDesc>("const2", "Const");
    OpDescPtr const3_op = std::make_shared<OpDesc>("const3", "Const");
    OpDescPtr cast1_op = std::make_shared<OpDesc>("cast1", "Cast");
    OpDescPtr cast2_op = std::make_shared<OpDesc>("cast2", "Cast");
    OpDescPtr add1_op = std::make_shared<OpDesc>("add1", "Add");
    OpDescPtr add2_op = std::make_shared<OpDesc>("add2", "Add");
    OpDescPtr square_op = std::make_shared<OpDesc>("square", "Square");
    OpDescPtr mean_op = std::make_shared<OpDesc>("reduce_mean", "ReduceMeanD");
    OpDescPtr sqrt_op = std::make_shared<OpDesc>("sqrt", "Sqrt");
    OpDescPtr div_op = std::make_shared<OpDesc>("real_div", "RealDiv");
    OpDescPtr mul1_op = std::make_shared<OpDesc>("mul1", "Mul");
    OpDescPtr mul2_op = std::make_shared<OpDesc>("mul2", "Mul");
    OpDescPtr end_op = std::make_shared<OpDesc>("end", "End");

    data1_op->AddOutputDesc(tensor_desc_fp32);
    data2_op->AddOutputDesc(tensor_desc_fp32);
    const1_op->AddOutputDesc(tensor_desc_fp16);
    const2_op->AddOutputDesc(tensor_desc_fp16);
    const3_op->AddOutputDesc(tensor_desc_fp32);
    add1_op->AddInputDesc(tensor_desc_fp32);
    add1_op->AddInputDesc(tensor_desc_fp32);
    add1_op->AddOutputDesc(tensor_desc_fp32);
    cast1_op->AddInputDesc(tensor_desc_fp32);
    cast1_op->AddOutputDesc(tensor_desc_fp16);
    square_op->AddInputDesc(tensor_desc_fp16);
    square_op->AddOutputDesc(tensor_desc_fp16);
    mean_op->AddInputDesc(tensor_desc_fp16);
    mean_op->AddOutputDesc(tensor_desc_fp16);
    add2_op->AddInputDesc(tensor_desc_fp16);
    add2_op->AddInputDesc(tensor_desc_fp16);
    add2_op->AddOutputDesc(tensor_desc_fp16);
    sqrt_op->AddInputDesc(tensor_desc_fp16);
    sqrt_op->AddOutputDesc(tensor_desc_fp16);
    div_op->AddInputDesc(tensor_desc_fp16);
    div_op->AddInputDesc(tensor_desc_fp16);
    div_op->AddOutputDesc(tensor_desc_fp16);
    cast2_op->AddInputDesc(tensor_desc_fp16);
    cast2_op->AddOutputDesc(tensor_desc_fp32);
    mul1_op->AddInputDesc(tensor_desc_fp32);
    mul1_op->AddInputDesc(tensor_desc_fp32);
    mul1_op->AddOutputDesc(tensor_desc_fp32);
    mul2_op->AddInputDesc(tensor_desc_fp32);
    mul2_op->AddInputDesc(tensor_desc_fp32);
    mul2_op->AddOutputDesc(tensor_desc_fp32);
    end_op->AddInputDesc(tensor_desc_fp32);

    AttrUtils::SetInt(add1_op, "_fe_imply_type", 6);
    AttrUtils::SetInt(add2_op, "_fe_imply_type", 6);
    AttrUtils::SetInt(cast1_op, "_fe_imply_type", 6);
    AttrUtils::SetInt(cast2_op, "_fe_imply_type", 6);
    AttrUtils::SetInt(square_op, "_fe_imply_type", 6);
    AttrUtils::SetInt(mean_op, "_fe_imply_type", 6);
    AttrUtils::SetInt(sqrt_op, "_fe_imply_type", 6);
    AttrUtils::SetInt(div_op, "_fe_imply_type", 6);
    AttrUtils::SetInt(mul1_op, "_fe_imply_type", 6);
    AttrUtils::SetInt(mul2_op, "_fe_imply_type", 6);
    AttrUtils::SetInt(add1_op, ge::ATTR_NAME_IMPLY_TYPE, static_cast<int64_t>(domi::ImplyType::TVM));
    AttrUtils::SetInt(add2_op, ge::ATTR_NAME_IMPLY_TYPE, static_cast<int64_t>(domi::ImplyType::TVM));
    AttrUtils::SetInt(cast1_op, ge::ATTR_NAME_IMPLY_TYPE, static_cast<int64_t>(domi::ImplyType::TVM));
    AttrUtils::SetInt(cast2_op, ge::ATTR_NAME_IMPLY_TYPE, static_cast<int64_t>(domi::ImplyType::TVM));
    AttrUtils::SetInt(square_op, ge::ATTR_NAME_IMPLY_TYPE, static_cast<int64_t>(domi::ImplyType::TVM));
    AttrUtils::SetInt(mean_op, ge::ATTR_NAME_IMPLY_TYPE, static_cast<int64_t>(domi::ImplyType::TVM));
    AttrUtils::SetInt(sqrt_op, ge::ATTR_NAME_IMPLY_TYPE, static_cast<int64_t>(domi::ImplyType::TVM));
    AttrUtils::SetInt(div_op, ge::ATTR_NAME_IMPLY_TYPE, static_cast<int64_t>(domi::ImplyType::TVM));
    AttrUtils::SetInt(mul1_op, ge::ATTR_NAME_IMPLY_TYPE, static_cast<int64_t>(domi::ImplyType::TVM));
    AttrUtils::SetInt(mul2_op, ge::ATTR_NAME_IMPLY_TYPE, static_cast<int64_t>(domi::ImplyType::TVM));

    AttrUtils::SetInt(cast1_op, "dst_type", 1);
    AttrUtils::SetInt(cast2_op, "dst_type", 0);
    AttrUtils::SetListInt(mean_op, "axes", {1});

    ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
    NodePtr data1_node = graph->AddNode(data1_op);
    NodePtr data2_node = graph->AddNode(data2_op);
    NodePtr const1_node = graph->AddNode(const1_op);
    NodePtr const2_node = graph->AddNode(const2_op);
    NodePtr const3_node = graph->AddNode(const3_op);
    NodePtr add1_node = graph->AddNode(add1_op);
    NodePtr add2_node = graph->AddNode(add2_op);
    NodePtr cast1_node = graph->AddNode(cast1_op);
    NodePtr cast2_node = graph->AddNode(cast2_op);
    NodePtr square_node = graph->AddNode(square_op);
    NodePtr sqrt_node = graph->AddNode(sqrt_op);
    NodePtr mean_node = graph->AddNode(mean_op);
    NodePtr div_node = graph->AddNode(div_op);
    NodePtr mul1_node = graph->AddNode(mul1_op);
    NodePtr mul2_node = graph->AddNode(mul2_op);
    NodePtr end_node = graph->AddNode(end_op);

    GraphUtils::AddEdge(data1_node->GetOutDataAnchor(0), add1_node->GetInDataAnchor(0));
    GraphUtils::AddEdge(data2_node->GetOutDataAnchor(0), add1_node->GetInDataAnchor(1));
    GraphUtils::AddEdge(add1_node->GetOutDataAnchor(0), cast1_node->GetInDataAnchor(0));
    GraphUtils::AddEdge(cast1_node->GetOutDataAnchor(0), square_node->GetInDataAnchor(0));
    GraphUtils::AddEdge(square_node->GetOutDataAnchor(0), mean_node->GetInDataAnchor(0));
    GraphUtils::AddEdge(mean_node->GetOutDataAnchor(0), add2_node->GetInDataAnchor(0));
    GraphUtils::AddEdge(const1_node->GetOutDataAnchor(0), add2_node->GetInDataAnchor(1));
    GraphUtils::AddEdge(add2_node->GetOutDataAnchor(0), sqrt_node->GetInDataAnchor(0));
    GraphUtils::AddEdge(sqrt_node->GetOutDataAnchor(0), div_node->GetInDataAnchor(0));
    GraphUtils::AddEdge(const2_node->GetOutDataAnchor(0), div_node->GetInDataAnchor(1));
    GraphUtils::AddEdge(div_node->GetOutDataAnchor(0), cast2_node->GetInDataAnchor(0));
    GraphUtils::AddEdge(cast2_node->GetOutDataAnchor(0), mul1_node->GetInDataAnchor(0));
    GraphUtils::AddEdge(add1_node->GetOutDataAnchor(0), mul1_node->GetInDataAnchor(1));
    GraphUtils::AddEdge(mul1_node->GetOutDataAnchor(0), mul2_node->GetInDataAnchor(0));
    GraphUtils::AddEdge(const3_node->GetOutDataAnchor(0), mul2_node->GetInDataAnchor(1));
    GraphUtils::AddEdge(mul2_node->GetOutDataAnchor(0), end_node->GetInDataAnchor(0));

    switch (type) {
      case 1:
        AttrUtils::SetBool(add1_op, "_fusion_check_fail", false);
        break;
      case 2:
        AttrUtils::SetBool(add1_op, "_fusion_check_fail", true);
        break;
      case 3:
        break;
    }

    graph->TopologicalSorting();
    AttrUtils::SetStr(graph, ge::ATTR_NAME_SESSION_GRAPH_ID, "11_22");
    return graph;
  }
};

class STAddRmsNormFusionPass : public BufferFusionPassBase {
 public:
  STAddRmsNormFusionPass() {}
  ~STAddRmsNormFusionPass() override {}

 protected:
  vector<BufferFusionPattern*> DefinePatterns() override {
    vector<BufferFusionPattern *> patterns;
    const string pass_name = "AddRmsNormFusionPattern1";
    BufferFusionPattern *pattern = new (std::nothrow) BufferFusionPattern(pass_name);
    if (pattern == nullptr) {
      return patterns;
    }

    pattern->AddOpDesc("add_0", {"Add"}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT,
                      TBE_PATTERN_GROUPID_INVALID, ONLY_SUPPORT_STATIC, true)
        .AddOpDesc("cast_0", {CAST}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT,
                   TBE_PATTERN_GROUPID_INVALID, ONLY_SUPPORT_STATIC, true)
        .AddOpDesc("square", {"Square"}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT,
                   TBE_PATTERN_GROUPID_INVALID, ONLY_SUPPORT_STATIC, true)
        .AddOpDesc("mean", {"ReduceMeanD"}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT,
                   TBE_PATTERN_GROUPID_INVALID, ONLY_SUPPORT_STATIC, true)
        .AddOpDesc("add_1", {"Add"}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT,
                   TBE_PATTERN_GROUPID_INVALID, ONLY_SUPPORT_STATIC, true)
        .AddOpDesc("sqrt", {"Sqrt"}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT,
                   TBE_PATTERN_GROUPID_INVALID, ONLY_SUPPORT_STATIC, true)
        .AddOpDesc("div", {"RealDiv"}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT,
                   TBE_PATTERN_GROUPID_INVALID, ONLY_SUPPORT_STATIC, true)
        .AddOpDesc("cast_1", {"Cast"}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT,
                   TBE_PATTERN_GROUPID_INVALID, ONLY_SUPPORT_STATIC, true)
        .AddOpDesc("mul_0", {"Mul"}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT,
                   TBE_PATTERN_GROUPID_INVALID, ONLY_SUPPORT_STATIC, true)
        .AddOpDesc("mul_1", {"Mul"}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT,
                   TBE_PATTERN_GROUPID_INVALID, ONLY_SUPPORT_STATIC, true)
        .SetHead({"add_0"})
        .SetOutputs("add_0", {"cast_0", "mul_0"}, TBE_OUTPUT_BRANCH_MULTI, false, true)
        .SetOutputs("cast_0", {"square"}, TBE_OUTPUT_BRANCH_SINGLE)
        .SetOutputs("square", {"mean"}, TBE_OUTPUT_BRANCH_SINGLE)
        .SetOutputs("mean", {"add_1"}, TBE_OUTPUT_BRANCH_SINGLE)
        .SetOutputs("add_1", {"sqrt"}, TBE_OUTPUT_BRANCH_SINGLE, true)
        .SetOutputs("sqrt", {"div"}, TBE_OUTPUT_BRANCH_SINGLE)
        .SetOutputs("div", {"cast_1"}, TBE_OUTPUT_BRANCH_SINGLE, true)
        .SetOutputs("cast_1", {"mul_0"}, TBE_OUTPUT_BRANCH_SINGLE)
        .SetOutputs("mul_0", {"mul_1"}, TBE_OUTPUT_BRANCH_SINGLE)
        .SetOutputs("mul_1", {}, TBE_OUTPUT_BRANCH_SINGLE, true, true);
    patterns.push_back(pattern);
    return patterns;
  }
};
REG_BUFFER_FUSION_PASS("STAddRmsNormFusionPass", BUILT_IN_AI_CORE_BUFFER_FUSION_PASS,
                       STAddRmsNormFusionPass, ENABLE_FUSION_CHECK);

TEST_F(CompileWithBufferFusionProcess910BTest, buffer_fusion_check_case1) {
  FEGraphOptimizerPtr graph_optimizer = FusionManager::Instance(AI_CORE_NAME).graph_opt_;
  EXPECT_NE(graph_optimizer, nullptr);
  ge::ComputeGraphPtr graph_ptr = CreateGraphWithType(1);
  Status ret = graph_optimizer->OptimizeGraphInit(*graph_ptr);
  EXPECT_EQ(ret, SUCCESS);
  ret = graph_optimizer->OptimizeFusedGraph(*graph_ptr);
  // EXPECT_EQ(ret, SUCCESS);
  // EXPECT_EQ(graph_ptr->GetDirectNodesSize(), 7);
  size_t node_count = 0;
  for (const ge::NodePtr &node : graph_ptr->GetDirectNode()) {
    if (node == nullptr) {
      continue;
    }
    if (node->GetType() != "PlaceHolder" && node->GetType() != "Const" && node->GetType() != "End") {
      node_count++;
    }
  }
  // EXPECT_EQ(node_count, 1);
}

TEST_F(CompileWithBufferFusionProcess910BTest, buffer_fusion_check_case2) {
  FEGraphOptimizerPtr graph_optimizer = FusionManager::Instance(AI_CORE_NAME).graph_opt_;
  EXPECT_NE(graph_optimizer, nullptr);
  ge::ComputeGraphPtr graph_ptr = CreateGraphWithType(2);
  Status ret = graph_optimizer->OptimizeGraphInit(*graph_ptr);
  EXPECT_EQ(ret, SUCCESS);
  ret = graph_optimizer->OptimizeFusedGraph(*graph_ptr);
  // EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(graph_ptr->GetDirectNodesSize() > 7, true);
}
}
