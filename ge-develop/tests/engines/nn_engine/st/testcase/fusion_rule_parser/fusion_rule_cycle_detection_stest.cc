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
#include <fstream>
#include "fe_llt_utils.h"

#define protected public
#define private   public

#include "graph_constructor.h"
#include "fusion_rule_manager/fusion_rule_manager.h"
#include "common/configuration.h"
#include "ops_store/sub_op_info_store.h"
#include "ops_store/ops_kernel_manager.h"
#include "common/fe_op_info_common.h"
#include "graph/operator_factory_impl.h"
#include "fusion_rule_manager/fusion_cycle_detector.h"
#include "fusion_config_manager/fusion_config_parser.h"
#include "graph_optimizer/graph_fusion/fusion_pattern.h"
using namespace testing;
using namespace fe;
using namespace std;
using namespace nlohmann;

class STEST_FUSION_RULE_CYCLE_DETECTION : public testing::Test {
 protected:
  void SetUp() {
    ops_kernel_info_store_ptr_ = std::make_shared<FEOpsKernelInfoStore>(fe::AI_CORE_NAME);

    FEOpsStoreInfo tbe_custom {
        2,
        "tbe-custom",
        EN_IMPL_CUSTOM_TBE,
        GetCodeDir() + "/tests/engines/nn_engine/ut/testcase/fusion_engine/ops_kernel_store/fe_config/fusion_rule_manager",
        GetCodeDir() + "/tests/engines/nn_engine/ut/testcase/fusion_engine/ops_kernel_store/fe_config/fusion_rule_manager",
        false,
        false,
        false};

    vector<FEOpsStoreInfo> store_info;
    store_info.emplace_back(tbe_custom);
    Configuration::Instance(AI_CORE_NAME).ops_store_info_vector_ = (store_info);
    OpsKernelManager::Instance(AI_CORE_NAME).Finalize();
    map<string, string> options;
    ops_kernel_info_store_ptr_->Initialize(options);

    string file_path =
        GetCodeDir() + "/tests/engines/nn_engine/ut/testcase/fusion_engine/fusion_rule_parser/cycle_detection.json";
    frm_ = std::make_shared<FusionRuleManager>(ops_kernel_info_store_ptr_);
    ori_opp_path_ = Configuration::Instance(AI_CORE_NAME).ascend_ops_path_;
    Configuration::Instance(AI_CORE_NAME).ascend_ops_path_ = "";
    ori_path_ = Configuration::Instance(AI_CORE_NAME).content_map_[path_key_];
    Configuration::Instance(AI_CORE_NAME).content_map_[path_key_] = file_path;
    ori_path_built_in_ = Configuration::Instance(AI_CORE_NAME).content_map_[path_key_built_in_];
    Configuration::Instance(AI_CORE_NAME).content_map_[path_key_built_in_] = file_path;
    frm_->Initialize(AI_CORE_NAME);
  }

  void TearDowm() {
    Configuration::Instance(AI_CORE_NAME).content_map_[path_key_] = ori_path_;
    Configuration::Instance(AI_CORE_NAME).content_map_[path_key_built_in_] = ori_path_built_in_;
    Configuration::Instance(AI_CORE_NAME).ascend_ops_path_ = ori_opp_path_;
  }
 private:
  FEOpsKernelInfoStorePtr ops_kernel_info_store_ptr_;
  std::shared_ptr<FusionRuleManager> frm_;
  string ori_path_;
  string ori_path_built_in_;
  string ori_opp_path_;
  const string path_key_ = "fusionrulemgr.aicore.customfilepath";
  const string path_key_built_in_ = "fusionrulemgr.aicore.graphfilepath";
};

void BuildGraph01(ge::ComputeGraphPtr &graph) {
  ge::GeShape original_shape = ge::GeShape({3, 12, 5, 6});
  GraphConstructor test(graph, "", ge::FORMAT_NHWC, ge::DT_FLOAT,
                        original_shape);

  test.AddOpDesc("neg", "Neg", 3, 1)
      .SetInputs({"Data_1", "Data_2", "Data_3"})

      .AddOpDesc("mul1", "Mul", 2, 1)
      .SetInputs({"neg:0", "Data_4"})

      .AddOpDesc("other", "Other", 1, 1)
      .SetInputs({"mul1:1"})

      .AddOpDesc("other1", "Other", 1, 1)
      .SetInputs({"Data_5"})

      .AddOpDesc("mul2", "Mul", 2, 1)
      .SetInputs({"mul1:0", "other1:0"})

      .AddOpDesc("out1", "NetOutput", 1, 0)
      .SetInputs({"mul2:0"});

  test.DumpGraph(graph);
}

void BuildGraph02(ge::ComputeGraphPtr &graph) {
  ge::GeShape original_shape = ge::GeShape({3, 12, 5, 6});
  GraphConstructor test(graph, "", ge::FORMAT_NHWC, ge::DT_FLOAT,
                        original_shape);

  test.AddOpDesc("neg", "Neg", 2, 1)
      .SetInputs({"Data_1", "Data_2", "Data_3"})

      .AddOpDesc("mul1", "Mul", 2, 1)
      .SetInputs({"neg:0", "Data_4"})

      .AddOpDesc("other", "Other", 1, 1)
      .SetInputs({"mul1:1"})

      .AddOpDesc("other1", "Other", 1, 1)
      .SetInputs({"other:0"})

      .AddOpDesc("mul2", "Mul", 2, 1)
      .SetInputs({"mul1:0", "other1:0"})

      .AddOpDesc("out1", "NetOutput", 1, 0)
      .SetInputs({"mul2:0"});

  test.DumpGraph(graph);
}


void BuildGraph03(ge::ComputeGraphPtr &graph) {
  ge::GeShape original_shape = ge::GeShape({3, 12, 5, 6});
  GraphConstructor test(graph, "", ge::FORMAT_NHWC, ge::DT_FLOAT,
                        original_shape);

  test.AddOpDesc("neg", "Neg", 2, 1)
      .SetInputs({"Data_1", "Data_2","Data_3"})

      .AddOpDesc("mul1", "Mul", 2, 1)
      .SetInputs({"neg:0", "Data_4"})

      .AddOpDesc("other1", "Other", 1, 1)
      .SetInputs({"mul1:1"})

      .AddOpDesc("other2", "Other", 1, 1)
      .SetInputs({"other1:0"})

      .AddOpDesc("other3", "Other", 1, 1)
      .SetInputs({"other2:0"})

      .AddOpDesc("other4", "Other", 1, 1)
      .SetInputs({"other3:0"})

      .AddOpDesc("other5", "Other", 1, 1)
      .SetInputs({"other4:0"})

      .AddOpDesc("mul2", "Mul", 2, 1)
      .SetInputs({"mul1:0", "other5:0"})

      .AddOpDesc("out", "NetOutput", 1, 0)
      .SetInputs({"mul2:0"});
  test.DumpGraph(graph);
}

void BuildGraph04(ge::ComputeGraphPtr &graph) {
  ge::GeShape original_shape = ge::GeShape({3, 12, 5, 6});
  GraphConstructor test(graph, "", ge::FORMAT_NHWC, ge::DT_FLOAT,
                        original_shape);

  test.AddOpDesc("neg", "Neg", 3, 1)
      .SetInputs({"Data_1", "Data_2", "Data_3"})

      .AddOpDesc("mul2", "Mul", 2, 1)
      .SetInputs({"neg:0", "Data4"})

      .AddOpDesc("other", "Other", 1, 1)
      .SetInputs({"mul2:0"})

      .AddOpDesc("other1", "Other", 1, 1)
      .SetInputs({"other:0"})

      .AddOpDesc("other2", "Other", 1, 0)
      .SetInputs({"other1:0"})

      .AddOpDesc("mul1", "Mul", 2, 1)
      .SetInputs({"neg:0", "Data5"})

      .AddOpDesc("out1", "NetOutput", 1, 0)
      .SetInputs({"mul1:0"});

  test.DumpGraph(graph);
}

void BuildGraph04_1(ge::ComputeGraphPtr &graph) {
  ge::GeShape original_shape = ge::GeShape({3, 12, 5, 6});
  GraphConstructor test(graph, "", ge::FORMAT_NHWC, ge::DT_FLOAT,
                        original_shape);

  test.AddOpDesc("neg", "Neg", 3, 1)
      .SetInputs({"Data_1", "Data_2", "Data_3"})

      .AddOpDesc("mul2", "Mul", 2, 1)
      .SetInputs({"neg:0", "Data4"})

      .AddOpDesc("other", "Other", 1, 1)
      .SetInputs({"mul2:0"})

      .AddOpDesc("other1", "Other", 1, 1)
      .SetInputs({"other:0"})

      .AddOpDesc("other2", "Other", 1, 0)
      .SetInputs({"other1:0"})

      .AddOpDesc("mul1", "Mul", 2, 1)
      .SetInputs({"neg:0", "Data5"})

      .AddOpDesc("out1", "NetOutput", 1, 0)
      .SetInputs({"mul1:0"})

      .SetInputs("out1:-1", {"mul1:-1"})
      .SetInputs("neg:-1", {"Data_1:-1"});

  test.DumpGraph(graph);
}

void BuildGraph05(ge::ComputeGraphPtr &graph) {
  ge::GeShape original_shape = ge::GeShape({3, 12, 5, 6});
  GraphConstructor test(graph, "", ge::FORMAT_NHWC, ge::DT_FLOAT,
                        original_shape);

  test.AddOpDesc("neg", "Neg", 3, 1)
      .SetInputs({"Data_1", "Data_2", "Data_3"})

      .AddOpDesc("mul2", "Mul", 2, 1)
      .SetInputs({"neg:0", "Data4"})

      .AddOpDesc("other", "Other", 1, 1)
      .SetInputs({"mul2:0"})

      .AddOpDesc("other1", "Other", 1, 1)
      .SetInputs({"other:0"})

      .AddOpDesc("other2", "Other", 1, 0)
      .SetInputs({"other1:0"})

      .AddOpDesc("Mul1", "Mul", 2, 1)
      .SetInputs({"neg:0", "Data5"})

      .AddOpDesc("out1", "NetOutput", 1, 0)
      .SetInputs({"Mul1:0"});

  test.DumpGraph(graph);
}


void BuildGraph06(ge::ComputeGraphPtr &graph) {
  ge::GeShape original_shape = ge::GeShape({3, 12, 5, 6});
  GraphConstructor test(graph, "", ge::FORMAT_NHWC, ge::DT_FLOAT,
                        original_shape);

  test.AddOpDesc("neg", "Neg", 3, 1)
      .SetInputs({"Data_1", "Data_2", "Data_3"})

      .AddOpDesc("mul2", "Mul", 2, 1)
      .SetInputs({"neg:0", "Data4"})

      .AddOpDesc("other", "Other", 1, 1)
      .SetInputs({"mul2:0"})

      .AddOpDesc("other1", "Other", 1, 1)
      .SetInputs({"other:0"})

      .AddOpDesc("other2", "Other", 1, 0)
      .SetInputs({"other1:0"})

      .AddOpDesc("mul1", "Mul", 2, 1)
      .SetInputs({"neg:0", "other1:0"})

      .AddOpDesc("out1", "NetOutput", 1, 0)
      .SetInputs({"mul1:0"});

  test.DumpGraph(graph);
}

void BuildGraph07(ge::ComputeGraphPtr &graph) {
  ge::GeShape original_shape = ge::GeShape({3, 12, 5, 6});
  GraphConstructor test(graph, "", ge::FORMAT_NHWC, ge::DT_FLOAT,
                        original_shape);

  test.AddOpDesc("lng", "LayerNormGrad", 3, 1)
      .SetInputs({"Data_1", "Data_2", "Data_3"})

      .AddOpDesc("mul2", "Mul", 2, 1)
      .SetInputs({"lng:0", "Data4"})

      .AddOpDesc("other", "Other", 1, 1)
      .SetInputs({"mul2:0"})

      .AddOpDesc("other1", "Other", 1, 1)
      .SetInputs({"other:0"})

      .AddOpDesc("other2", "Other", 1, 0)
      .SetInputs({"other1:0"})

      .AddOpDesc("mul1", "Mul", 2, 1)
      .SetInputs({"lng:0", "Data5"})

      .AddOpDesc("out1", "NetOutput", 1, 0)
      .SetInputs({"mul1:0"})

      .SetInputs("out1:-1", {"mul1:-1"})
      .SetInputs("lng:-1", {"Data_1:-1"});

  test.DumpGraph(graph);
}

TEST_F(STEST_FUSION_RULE_CYCLE_DETECTION, test_cycle_01) {
  ge:ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("test");
  BuildGraph01(graph);
  frm_->RunGraphFusionRuleByType(*graph, RuleType::CUSTOM_GRAPH_RULE, "LayerNormGradFusionRule");

  size_t fusion_node_count = 0;
  size_t original_node_cout = 0;
  for (auto &node : graph->GetDirectNode()) {
    if (node->GetType() == "LayerNormGrad") {
      fusion_node_count++;
    }
    if (node->GetName() == "neg" ||
        node->GetName() == "mul1" ||
        node->GetName() == "mul2") {
      original_node_cout++;
    }
  }
  EXPECT_EQ(fusion_node_count, 0);
  EXPECT_EQ(original_node_cout, 3);
}

TEST_F(STEST_FUSION_RULE_CYCLE_DETECTION, test_cycle_02) {
  ge:ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("test");
  BuildGraph02(graph);
  frm_->RunGraphFusionRuleByType(*graph, RuleType::CUSTOM_GRAPH_RULE, "LayerNormGradFusionRule");
  size_t fusion_node_count = 0;
  size_t original_node_cout = 0;
  for (auto &node : graph->GetDirectNode()) {
    if (node->GetType() == "LayerNormGrad") {
      fusion_node_count++;
    }
    if (node->GetName() == "neg" ||
        node->GetName() == "mul1" ||
        node->GetName() == "mul2") {
      original_node_cout++;
    }
  }
  EXPECT_EQ(fusion_node_count, 0);
  EXPECT_EQ(original_node_cout, 3);
}

TEST_F(STEST_FUSION_RULE_CYCLE_DETECTION, test_cycle_03) {
  ge:ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("test");
  BuildGraph03(graph);
  frm_->RunGraphFusionRuleByType(*graph, RuleType::CUSTOM_GRAPH_RULE, "LayerNormGradFusionRule");
  size_t fusion_node_count = 0;
  size_t original_node_cout = 0;
  for (auto &node : graph->GetDirectNode()) {
    if (node->GetType() == "LayerNormGrad") {
      fusion_node_count++;
    }
    if (node->GetName() == "neg" ||
        node->GetName() == "mul1" ||
        node->GetName() == "mul2") {
      original_node_cout++;
    }
  }
  EXPECT_EQ(fusion_node_count, 0);
  EXPECT_EQ(original_node_cout, 3);
}

TEST_F(STEST_FUSION_RULE_CYCLE_DETECTION, test_cycle_04) {
  ge:ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("test");
  BuildGraph04(graph);
  frm_->RunGraphFusionRuleByType(*graph, RuleType::CUSTOM_GRAPH_RULE, "LayerNormGradFusionRule2");
  size_t fusion_node_count = 0;
  size_t original_node_cout = 0;
  for (auto &node : graph->GetDirectNode()) {
    if (node->GetType() == "LayerNormGrad") {
      fusion_node_count++;
    }
    if (node->GetName() == "neg" ||
        node->GetName() == "mul1" ||
        node->GetName() == "mul2") {
      original_node_cout++;
    }
  }
  EXPECT_EQ(fusion_node_count, 0);
  EXPECT_EQ(original_node_cout, 3);
}

TEST_F(STEST_FUSION_RULE_CYCLE_DETECTION, test_cycle_05) {
  ge:ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("test");
  BuildGraph06(graph);
  frm_->RunGraphFusionRuleByType(*graph, RuleType::CUSTOM_GRAPH_RULE, "LayerNormGradFusionRule2");
  size_t fusion_node_count = 0;
  size_t original_node_cout = 0;
  for (auto &node : graph->GetDirectNode()) {
    if (node->GetType() == "LayerNormGrad") {
      fusion_node_count++;
    }
    if (node->GetName() == "neg" ||
        node->GetName() == "mul1" ||
        node->GetName() == "mul2") {
      original_node_cout++;
    }
  }
  graph->TopologicalSorting();
  EXPECT_EQ(fusion_node_count, 0);
  EXPECT_EQ(original_node_cout, 3);
}


TEST_F(STEST_FUSION_RULE_CYCLE_DETECTION, coverage_01) {
  ge:ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("test");
  BuildGraph05(graph);
  frm_->RunGraphFusionRuleByType(*graph, RuleType::CUSTOM_GRAPH_RULE, "LayerNormGradFusionRule2");
  size_t fusion_node_count = 0;
  size_t original_node_cout = 0;
  for (auto &node : graph->GetDirectNode()) {
    if (node->GetType() == "LayerNormGrad") {
      fusion_node_count++;
    }
    if (node->GetName() == "neg" ||
        node->GetName() == "Mul1" ||
        node->GetName() == "mul2") {
      original_node_cout++;
    }
  }
  EXPECT_EQ(fusion_node_count, 0);
  EXPECT_EQ(original_node_cout, 3);
}

TEST_F(STEST_FUSION_RULE_CYCLE_DETECTION, coverage_02) {
  ge:ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("test");
  BuildGraph04(graph);
  for (ge::NodePtr &node : graph->GetDirectNode()) {
    if (node->GetName() == "mul1" ||
        node->GetName() == "mul2") {
      auto op_desc = node->GetOpDesc();
      ge::GeAttrValue attr_value;
      attr_value.SetValue(1);
      op_desc->SetAttr("test", attr_value);
    }
  }

  frm_->RunGraphFusionRuleByType(*graph, RuleType::CUSTOM_GRAPH_RULE, "LayerNormGradFusionRule3");
  size_t fusion_node_count = 0;
  size_t original_node_cout = 0;
  for (auto &node : graph->GetDirectNode()) {
    if (node->GetType() == "LayerNormGrad") {
      fusion_node_count++;
    }
    if (node->GetName() == "neg" ||
        node->GetName() == "Mul1" ||
        node->GetName() == "mul2") {
      original_node_cout++;
    }
  }
  EXPECT_EQ(fusion_node_count, 0);
  EXPECT_EQ(original_node_cout, 2);
}

TEST_F(STEST_FUSION_RULE_CYCLE_DETECTION, coverage_03) {
  ge:ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("test");
  BuildGraph04(graph);
  for (ge::NodePtr &node : graph->GetDirectNode()) {
    if (node->GetName() == "mul1" ||
        node->GetName() == "Data_1") {
      auto output_node = node->GetOutDataNodes().at(0);
      ge::GraphUtils::AddEdge(node->GetOutControlAnchor(),
                              output_node->GetInControlAnchor());
    }
  }

  frm_->RunGraphFusionRuleByType(*graph, RuleType::CUSTOM_GRAPH_RULE, "LayerNormGradFusionRule2WithCtrl");
  size_t fusion_node_count = 0;
  size_t original_node_cout = 0;
  for (auto &node : graph->GetDirectNode()) {
    if (node->GetType() == "LayerNormGrad") {
      fusion_node_count++;
    }
    if (node->GetName() == "neg" ||
        node->GetName() == "mul1" ||
        node->GetName() == "mul2") {
      original_node_cout++;
    }
  }
  EXPECT_EQ(fusion_node_count, 0);
  EXPECT_EQ(original_node_cout, 3);
}

TEST_F(STEST_FUSION_RULE_CYCLE_DETECTION, coverage_03_1) {
  ge:ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("test");
  BuildGraph04_1(graph);

  frm_->RunGraphFusionRuleByType(*graph, RuleType::CUSTOM_GRAPH_RULE, "LayerNormGradFusionRule2WithCtrl");
  size_t fusion_node_count = 0;
  size_t original_node_cout = 0;
  for (auto &node : graph->GetDirectNode()) {
    if (node->GetType() == "LayerNormGrad") {
      fusion_node_count++;
    }
    if (node->GetName() == "neg" ||
        node->GetName() == "mul1" ||
        node->GetName() == "mul2") {
      original_node_cout++;
    }
  }
  EXPECT_EQ(fusion_node_count, 0);
  EXPECT_EQ(original_node_cout, 3);
}

TEST_F(STEST_FUSION_RULE_CYCLE_DETECTION, coverage_03_2) {
  ge:ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("test");
  BuildGraph04_1(graph);

  frm_->RunGraphFusionRuleByType(*graph, RuleType::CUSTOM_GRAPH_RULE, "LayerNormGradFusionRule2WithCtrl2");
  size_t fusion_node_count = 0;
  size_t original_node_cout = 0;
  for (auto &node : graph->GetDirectNode()) {
    if (node->GetType() == "LayerNormGrad") {
      fusion_node_count++;
    }
    if (node->GetName() == "neg" ||
        node->GetName() == "mul1" ||
        node->GetName() == "mul2") {
      original_node_cout++;
    }
  }
  EXPECT_EQ(fusion_node_count, 0);
  EXPECT_EQ(original_node_cout, 3);
}

TEST_F(STEST_FUSION_RULE_CYCLE_DETECTION, coverage_04) {
  ge:ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("test");
  BuildGraph04(graph);

  frm_->RunGraphFusionRuleByType(*graph, RuleType::CUSTOM_GRAPH_RULE, "LayerNormGradFusionRule4");
  size_t fusion_node_count = 0;
  size_t original_node_cout = 0;
  for (auto &node : graph->GetDirectNode()) {
    if (node->GetType() == "LayerNormGrad" ||
        node->GetType() == "Add") {
      fusion_node_count++;
    }
    if (node->GetName() == "neg" ||
        node->GetName() == "mul1" ||
        node->GetName() == "mul2") {
      original_node_cout++;
    }
  }
  EXPECT_EQ(fusion_node_count, 0);
  EXPECT_EQ(original_node_cout, 3);
}

ge::graphStatus ReduceMeanInferShape(ge::Operator &op) {
  auto output = op.GetOutputDesc(0);
  output.SetShape(ge::Shape({3, 12, 5, 6}));
  op.UpdateOutputDesc("y", output);

  return ge::GRAPH_SUCCESS;
}

ge::graphStatus LayerNormGradInferShape(ge::Operator &op) {
  auto output = op.GetOutputDesc(0);
  output.SetShape(ge::Shape({3, 12, 5, 6}));
  op.UpdateOutputDesc("pd_x", output);
  op.UpdateOutputDesc("pd_gamma", output);
  op.UpdateOutputDesc("pd_beta", output);

  return ge::GRAPH_SUCCESS;
}
ge::graphStatus WrongLayerNormGradInferShape(ge::Operator &op) {
  return ge::GRAPH_SUCCESS;
}

TEST_F(STEST_FUSION_RULE_CYCLE_DETECTION, coverage_05) {
  ge:ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("test");
  BuildGraph04(graph);
  ge::OperatorFactoryImpl::operator_infershape_funcs_ =
      std::make_shared<std::map<std::string, ge::InferShapeFunc>>();
  ge::OperatorFactoryImpl::operator_infershape_funcs_->emplace("ReduceMean", ReduceMeanInferShape);
  ge::OperatorFactoryImpl::operator_infershape_funcs_->emplace("LayerNormGrad", LayerNormGradInferShape);

  frm_->RunGraphFusionRuleByType(*graph, RuleType::CUSTOM_GRAPH_RULE, "LayerNormGradFusionRule5");
  size_t fusion_node_count = 0;
  size_t original_node_cout = 0;
  for (auto &node : graph->GetDirectNode()) {
    if (node->GetType() == "LayerNormGrad" ||
        node->GetType() == "ReduceMean") {
      fusion_node_count++;
    }
    if (node->GetName() == "neg" ||
        node->GetName() == "mul1" ||
        node->GetName() == "mul2") {
      original_node_cout++;
    }
  }
  // EXPECT_EQ(fusion_node_count, 0);
  EXPECT_EQ(original_node_cout, 3);
}

TEST_F(STEST_FUSION_RULE_CYCLE_DETECTION, coverage_05_false) {
  ge:ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("test");
  BuildGraph04(graph);
  ge::OperatorFactoryImpl::operator_infershape_funcs_ =
      std::make_shared<std::map<std::string, ge::InferShapeFunc>>();
  ge::OperatorFactoryImpl::operator_infershape_funcs_->emplace("ReduceMean", ReduceMeanInferShape);
  ge::OperatorFactoryImpl::operator_infershape_funcs_->emplace("LayerNormGrad", WrongLayerNormGradInferShape);

  frm_->RunGraphFusionRuleByType(*graph, RuleType::CUSTOM_GRAPH_RULE, "LayerNormGradFusionRule5");
  size_t fusion_node_count = 0;
  size_t original_node_cout = 0;
  for (auto &node : graph->GetDirectNode()) {
    if (node->GetType() == "LayerNormGrad" ||
        node->GetType() == "ReduceMean") {
      fusion_node_count++;
    }
    if (node->GetName() == "neg" ||
        node->GetName() == "mul1" ||
        node->GetName() == "mul2") {
      original_node_cout++;
    }
  }
  // EXPECT_EQ(fusion_node_count, 0);
  // EXPECT_EQ(original_node_cout, 3);
}

TEST_F(STEST_FUSION_RULE_CYCLE_DETECTION, coverage_06) {
  fe::FusionCycleDetector detector;
  ge::ComputeGraph graph("test");
  std::map<const std::shared_ptr<fe::FusionPattern::OpDesc>, std::vector<ge::NodePtr>, fe::CmpKey> mapping;
  std::vector<ge::NodePtr> new_nodes;

  EXPECT_EQ(detector.Fusion(graph, mapping, new_nodes), fe::SUCCESS);
  detector.DefinePatterns();
}


TEST_F(STEST_FUSION_RULE_CYCLE_DETECTION, coverage_07) {
  ge:ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("test");
  BuildGraph07(graph);

  frm_->RunGraphFusionRuleByType(*graph, RuleType::CUSTOM_GRAPH_RULE, "LayerNormGradFusionRule2WithCtrl3");
  size_t fusion_node_count = 0;
  size_t original_node_cout = 0;
  for (auto &node : graph->GetDirectNode()) {
    if (node->GetType() == "LayerNormGrad") {
      fusion_node_count++;
    }
    if (node->GetName() == "lng" ||
        node->GetName() == "mul1" ||
        node->GetName() == "mul2") {
      original_node_cout++;
    }
  }
  EXPECT_EQ(fusion_node_count, 1);
  EXPECT_EQ(original_node_cout, 1);
}

TEST_F(STEST_FUSION_RULE_CYCLE_DETECTION, coverage_08) {
  FusionConfigParser parser(AI_CORE_NAME);
  std::map<std::string, std::string> error_key_map;

  string empty_file;
  std::map<string, bool> old_fusion_switch_map;
  EXPECT_EQ(parser.LoadOldFormatFusionSwitchFile(empty_file, old_fusion_switch_map), fe::FILE_NOT_EXIST);

  string file2 = GetCodeDir() + "/tests/engines/nn_engine/st/testcase/fusion_config_manager/builtin_config/old_fusion_config.json";
  EXPECT_EQ(parser.LoadOldFormatFusionSwitchFile(file2, old_fusion_switch_map), fe::SUCCESS);

  string file3 = GetCodeDir() + "/tests/engines/nn_engine/st/testcase/fusion_config_manager/builtin_config/old_fusion_config1.json";
  EXPECT_EQ(parser.LoadOldFormatFusionSwitchFile(file3, old_fusion_switch_map), fe::FAILED);

  string file4 = GetCodeDir() + "/tests/engines/nn_engine/st/testcase/fusion_config_manager/builtin_config/old_fusion_config2.json";
  EXPECT_EQ(parser.LoadOldFormatFusionSwitchFile(file4, old_fusion_switch_map), fe::FAILED);
}
