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

#include "graph/compute_graph.h"
#include "graph/graph.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"

#include "register/graph_optimizer/graph_fusion/fusion_pass_manager/fusion_pass_registry.h"
#include "register/graph_optimizer/graph_fusion/graph_fusion_pass_base.h"
#include "register/graph_optimizer/fusion_common/pattern_fusion_base_pass.h"
#include "register/pass_option_utils.h"
#include "framework/common/debug/ge_log.h"
#include "graph/ge_local_context.h"
#include "register/optimization_option_registry.h"

using namespace testing;
using namespace ge;
using namespace fe;

namespace fe {
namespace graph_fusion_reg_v2 {

class UTestGraphFusionPassReg : public testing::Test {
 public:

 protected:

  void SetUp() {
    oopt.Initialize({}, {});
  }

  void TearDown() {
  }

  ge::OptimizationOption &oopt = ge::GetThreadLocalContext().GetOo();

  const std::unordered_map<std::string, ge::OoInfo> &registered_opt_table =
      ge::OptionRegistry::GetInstance().GetRegisteredOptTable();
};


const char *kOpTypeCast = "Cast";
const char *kOpTypeRelu = "Relu";

const char *kPatternCast0 = "Cast0";
const char *kPatternCast1 = "Cast1";
const char *kPatternRelu = "Relu";
#define UT_CHECK(cond, log_func, return_expr) \
  do {                                        \
    if (cond) {                               \
      log_func;                               \
      return_expr;                            \
    }                                         \
  } while (0)

#define UT_CHECK_NOTNULL(val)                           \
  do {                                                  \
    if ((val) == nullptr) {                             \
      GELOGD("Parameter[%s] must not be null.", #val); \
      return fe::PARAM_INVALID;                         \
    }                                                   \
  } while (0)

string pass_name_test = "CastCastFusionPass";
string pass_name_test1 = "CastCastFusionPass1";
class TestPass : public fe::PatternFusionBasePass {
  using Mapping = std::map<const std::shared_ptr<OpDesc>, std::vector<ge::NodePtr>, fe::CmpKey>;
 protected:

  vector<FusionPattern *> DefinePatterns() override {
    vector<FusionPattern *> patterns;

    FusionPattern *pattern = new(std::nothrow) FusionPattern("CastCastFusionPass");
    UT_CHECK(pattern == nullptr, GELOGD(" Fail to create a new pattern object."),
             return patterns);
    pattern->AddOpDesc(kPatternCast0, {kOpTypeCast})
        .AddOpDesc(kPatternRelu, {kOpTypeRelu})
        .AddOpDesc(kPatternCast1, {kOpTypeCast})
        .SetInputs(kPatternRelu, {kPatternCast0})
        .SetInputs(kPatternCast1, {kPatternRelu})
        .SetOutput(kPatternCast1);

    patterns.push_back(pattern);

    return patterns;
  }

  Status Fusion(ge::ComputeGraph &graph, Mapping &mapping, vector<ge::NodePtr> &new_nodes) override {
    FusionPattern pattern("CastCastFusionPass");
    DumpMapping(pattern, mapping);

    CheckGraphCycle(graph);
    ge::NodePtr cast_Node0 = GetNodeFromMapping(kPatternCast0, mapping);
    CheckOpSupported(cast_Node0);
    CheckOpSupported(cast_Node0->GetOpDesc());
    CheckAccuracySupported(cast_Node0);

    UT_CHECK(cast_Node0 == nullptr, GELOGD("cast_Node0 is null,fusion failed."),
             return NOT_CHANGED);
    ge::OpDescPtr cast_desc0 = cast_Node0->GetOpDesc();
    UT_CHECK(cast_desc0 == nullptr, GELOGD("cast_Node0's Desc is null,fusion failed."),
             return NOT_CHANGED);

    ge::NodePtr relu_Node = GetNodeFromMapping(kPatternRelu, mapping);
    UT_CHECK(relu_Node == nullptr, GELOGD("relu_Node is null,fusion failed."),
             return NOT_CHANGED);
    ge::OpDescPtr relu_desc = relu_Node->GetOpDesc();
    UT_CHECK(cast_desc0 == nullptr, GELOGD("relu_Node's Desc is null,fusion failed."),
             return NOT_CHANGED);

    auto relu_input = relu_desc->MutableInputDesc(0);
    UT_CHECK_NOTNULL(relu_input);
    auto relu_input_desc_dtype = relu_input->GetDataType();

    auto relu_output = relu_desc->MutableOutputDesc(0);
    UT_CHECK_NOTNULL(relu_output);
    auto relu_output_desc_dtype = relu_output->GetDataType();
    if (relu_input_desc_dtype != DT_FLOAT || relu_output_desc_dtype != DT_FLOAT) {
      GELOGD("Relu node [%s]'s input dtype or output dtype is unsuitable", relu_desc->GetName().c_str());
      return NOT_CHANGED;
    }

    ge::NodePtr cast_Node1 = GetNodeFromMapping(kPatternCast1, mapping);
    UT_CHECK(cast_Node1 == nullptr, GELOGD("cast_Node1 is null,fusion failed."),
             return NOT_CHANGED);
    ge::OpDescPtr cast_desc1 = cast_Node1->GetOpDesc();
    UT_CHECK(cast_desc0 == nullptr, GELOGD("cast_Node1's Desc is null,fusion failed."),
             return NOT_CHANGED);

    auto cast0_input = cast_desc0->MutableInputDesc(0);
    UT_CHECK_NOTNULL(cast0_input);
    DataType cast0_in_d_type = cast0_input->GetDataType();
    auto cast1_output = cast_desc1->MutableOutputDesc(0);
    UT_CHECK_NOTNULL(cast1_output);
    DataType cast1_out_d_type = cast1_output->GetDataType();
    if (cast0_in_d_type != cast1_out_d_type) {
      GELOGD("Cast Node0 [%s] input data type is not equal to Cast Node1 [%s] output data type ",
             cast_Node0->GetName().c_str(), cast_Node1->GetName().c_str());
      return NOT_CHANGED;
    }

    auto cast0_out_data_anchor = cast_Node0->GetOutDataAnchor(0);
    UT_CHECK_NOTNULL(cast0_out_data_anchor);
    if (cast0_out_data_anchor->GetPeerInDataAnchors().size() > 1) {
      GELOGD("The first output anchor of Cast node[%s] has more than one peer in anchor.",
             cast_Node0->GetName().c_str());
      return NOT_CHANGED;
    }

    auto relu_out_data_anchor = relu_Node->GetOutDataAnchor(0);
    UT_CHECK_NOTNULL(relu_out_data_anchor);
    if (relu_out_data_anchor->GetPeerInDataAnchors().size() > 1) {
      for (auto node : relu_Node->GetOutAllNodes()) {
        if (node->GetType() != "Cast") {
          GELOGD("The  output anchor of Relu node has not Cast node,name is [%s] Type is [%s].",
                 node->GetName().c_str(), node->GetType().c_str());
          return NOT_CHANGED;
        }
        auto node_desc = node->GetOpDesc();
        UT_CHECK_NOTNULL(node_desc);
        auto in_dtype = node_desc->MutableInputDesc(0)->GetDataType();
        auto out_dtype = node_desc->MutableOutputDesc(0)->GetDataType();
        if (in_dtype != DT_FLOAT || out_dtype != DT_FLOAT16) {
          GELOGD("The Cast node [%s]'s indatatype is not equal to DT_FLOAT or outdatatype is not equal to DT_FLOAT16.",
                 node->GetName().c_str());
          return NOT_CHANGED;
        }
      }
    }

    ge::ComputeGraphPtr graphPtr = relu_Node->GetOwnerComputeGraph();
    UT_CHECK_NOTNULL(graphPtr);
    if (GraphUtils::IsolateNode(cast_Node0, {0}) != GRAPH_SUCCESS) {
      GELOGD("Isolate op:%s(%s) failed", cast_Node0->GetName().c_str(), cast_Node0->GetType().c_str());
      return FAILED;
    }
    if (GraphUtils::RemoveNodeWithoutRelink(graphPtr, cast_Node0) != GRAPH_SUCCESS) {
      GELOGD("[Remove][Node] %s, type:%s without relink in graph:%s failed",
             cast_Node0->GetName().c_str(), cast_Node0->GetType().c_str(), graph.GetName().c_str());
      return FAILED;
    }
    for (auto inAnchor : relu_out_data_anchor->GetPeerInDataAnchors()) {
      auto node = inAnchor->GetOwnerNode();
      UT_CHECK_NOTNULL(node);
      if (GraphUtils::IsolateNode(node, {0}) != GRAPH_SUCCESS) {
        GELOGD("Isolate op:%s(%s) failed", node->GetName().c_str(), node->GetType().c_str());
        return FAILED;
      }
      if (GraphUtils::RemoveNodeWithoutRelink(graphPtr, node) != GRAPH_SUCCESS) {
        GELOGD("[Remove][Node] %s, type:%s without relink in graph:%s failed",
               node->GetName().c_str(), node->GetType().c_str(), graph.GetName().c_str());
        return FAILED;
      }
    }
    relu_desc->MutableInputDesc(0)->SetDataType(cast0_in_d_type);
    relu_desc->MutableOutputDesc(0)->SetDataType(cast1_out_d_type);
    new_nodes.push_back(relu_Node);
    return SUCCESS;
  }
};

TEST_F(UTestGraphFusionPassReg, test_case_01) {
  REG_PASS(pass_name_test, GRAPH_FUSION_PASS_TYPE_RESERVED, TestPass, 0);
  REG_PASS("", BUILT_IN_GRAPH_PASS, TestPass, 1);
  REG_PASS(pass_name_test, BUILT_IN_GRAPH_PASS, TestPass, 2);
  REG_PASS(pass_name_test, BUILT_IN_GRAPH_PASS, TestPass, 3);
  std::map<string, FusionPassRegistry::PassDesc> pass_desc =
      FusionPassRegistry::GetInstance().GetPassDesc(SECOND_ROUND_BUILT_IN_GRAPH_PASS);
  EXPECT_EQ(pass_desc.size(), 0);
  pass_desc =
      FusionPassRegistry::GetInstance().GetPassDesc(BUILT_IN_GRAPH_PASS);
  EXPECT_EQ(pass_desc.size(), 1);

  EXPECT_EQ(pass_desc[pass_name_test].attr, 3);

  auto create_fn1 = pass_desc[pass_name_test].create_fn;
  auto pattern_fusion_base_pass_ptr = std::unique_ptr<PatternFusionBasePass>(
      dynamic_cast<PatternFusionBasePass *>(create_fn1()));
  auto patterns = pattern_fusion_base_pass_ptr->DefinePatterns();
  EXPECT_EQ(patterns.size(), 1);
  for (auto pattern : patterns) {
    if (pattern != nullptr) {
      delete pattern;
      pattern = nullptr;
    }
  }
}

TEST_F(UTestGraphFusionPassReg, test_compile_level_case_01) {
  auto pass_desc = FusionPassRegistry::GetInstance().GetPassDesc(BUILT_IN_GRAPH_PASS);
  EXPECT_EQ(pass_desc.size(), 1);

  bool enable_flag = true;
  auto ret = ge::PassOptionUtils::CheckIsPassEnabled(pass_name_test, enable_flag);
  EXPECT_NE(ret, ge::GRAPH_SUCCESS);

  REG_PASS(pass_name_test, BUILT_IN_GRAPH_PASS, TestPass, COMPILE_LEVEL_O0 | COMPILE_LEVEL_O1 | COMPILE_LEVEL_O2);
  REG_PASS(pass_name_test1, BUILT_IN_GRAPH_PASS, TestPass, COMPILE_LEVEL_O3);

  std::map<std::string, std::string> ge_options = {{ge::OO_LEVEL, "O1"},
                                                   {"ge.constLifecycle", "graph"},
                                                   {"ge.oo.test_graph_fusion", "true"},
                                                   {"ge.oo.test_graph_fusion_add_relu", "false"}};
  EXPECT_EQ(oopt.Initialize(ge_options, registered_opt_table), GRAPH_SUCCESS);

  pass_desc = FusionPassRegistry::GetInstance().GetPassDesc(BUILT_IN_GRAPH_PASS);
  EXPECT_EQ(pass_desc.size(), 2);
  enable_flag = true;
  ret = ge::PassOptionUtils::CheckIsPassEnabled(pass_name_test, enable_flag);
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  EXPECT_TRUE(enable_flag == true);

  pass_desc = FusionPassRegistry::GetInstance().GetPassDesc(BUILT_IN_GRAPH_PASS);
  EXPECT_EQ(pass_desc.size(), 2);
  enable_flag = true;
  ret = ge::PassOptionUtils::CheckIsPassEnabled(pass_name_test1, enable_flag);
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  EXPECT_EQ(enable_flag, false);
}

}
}
