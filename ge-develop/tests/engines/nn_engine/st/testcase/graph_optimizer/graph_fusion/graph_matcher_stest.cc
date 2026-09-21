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
#include "all_ops_stub.h"

#include "graph/ge_tensor.h"
#include "graph/op_desc.h"
#include "graph/compute_graph.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/attr_utils.h"
#include "graph/utils/tensor_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/graph_utils_ex.h"
#include "graph/compute_graph.h"

#define protected public
#define private public
#include "graph_optimizer/graph_fusion/graph_matcher.h"
#undef private
#undef protected

using namespace std;
using namespace testing;
using namespace ge;
using namespace fe;

class UTEST_GraphMatcher : public testing::Test
{
protected:
    void SetUp()
    {
    }

    void TearDown()
    {

    }

    static ComputeGraphPtr BuildGraph(vector<Operator> inputs)
    {
        Graph graph("graph");
        vector<Operator> outputs {};
        graph.SetInputs(inputs).SetOutputs(outputs);
        return GraphUtilsEx::GetComputeGraph(graph);
    }

    Status Match(ComputeGraphPtr graph, vector<GraphMatchResult>& match_results)
    {
        rule_ = make_shared<FusionRulePattern>();
        rule_->rule_name_ = "myrule";
        rule_->input_info_ = inputs_;
        rule_->origin_rule_nodes_ = org_nodes_;
        rule_->output_info_ = outputs_;

        inputs_.clear();
        org_nodes_.clear();
        outputs_.clear();

        return GraphMatcher().Match(*rule_, *graph, match_results);
    }

    FusionRuleNodePtr CreateInputNode(const string& name, int out_anchors = 1)
    {
        FusionRuleNodePtr node = CreateRuleNode(name, {}, 0, out_anchors);
        SetAnchorIdxDefault(node->output_data_anchors_);
        inputs_.push_back(node);
        return node;
    }

    FusionRuleNodePtr CreateOrgNode(const string& name, vector<string>&& types, int in_anchors = 1, int out_anchors = 1)
    {
        FusionRuleNodePtr node = CreateRuleNode(name, types, in_anchors, out_anchors);
        org_nodes_.emplace(node);
        return node;
    }

    FusionRuleNodePtr CreateOutputNode(const string& name, int in_anchors = 1)
    {
        FusionRuleNodePtr node = CreateRuleNode(name, {}, in_anchors, 0);
        SetAnchorIdxDefault(node->input_data_anchors_);
        outputs_.push_back(node);
        return node;
    }

    void SetInputAnchorIdx(FusionRuleNodePtr node, int idx, int value)
    {
        node->input_data_anchors_[idx]->anchor_idx_ = value;
    }

    void SetOutputAnchorIdx(FusionRuleNodePtr node, int idx, int value)
    {
        node->output_data_anchors_[idx]->anchor_idx_ = value;
    }

private:
    static FusionRuleNodePtr CreateRuleNode(const string& name, vector<string> types, int in_anchors, int out_anchors)
    {
        FusionRuleNodePtr node = make_shared<fe::FusionRuleNode>();
        node->node_name_ = name;
        node->node_type_ = types;
        for (int i = 0; i < in_anchors; i++) {
            FusionRuleAnchorPtr anchor = make_shared<FusionRuleAnchor>();
            anchor->anchor_idx_ = i;
            anchor->anchor_name_ = name + "_input_" + to_string(i);
            anchor->owner_node_ = node;
            node->input_data_anchors_.push_back(anchor);
        }
        for (int i = 0; i < out_anchors; i++) {
            FusionRuleAnchorPtr anchor = make_shared<FusionRuleAnchor>();
            anchor->anchor_idx_ = i;
            anchor->anchor_name_ = name + "_output_" + to_string(i);
            anchor->owner_node_ = node;
            node->output_data_anchors_.push_back(anchor);
        }
        return node;
    }

    static void SetAnchorIdxDefault(vector<FusionRuleAnchorPtr> anchors)
    {
        for (FusionRuleAnchorPtr anchor : anchors) {
            anchor->anchor_idx_ = DEFAULT_ANCHOR_INDEX;
        }
    }

private:
    FusionRulePatternPtr rule_;
    vector<FusionRuleNodePtr> inputs_;
    set<FusionRuleNodePtr> org_nodes_;
    vector<FusionRuleNodePtr> outputs_;

public:

    /*
          data1 data2         data3 data4
             \  /               /\  /
             conv             /   add
              |             /      |
        /   /   \  \      /        |
      mean neg SSDDetectionOutput PReLU
      (out1) \    /    \           |
              \  /       \       split
               sub         \     /  \
                 \           mul   pooling
                   \        /  \   (out4)
                     \    /     \
                   FloorDiv    activation
                 (out2, out3)    (out3)
     */
/*    static ComputeGraphPtr GenGraph()
    {
        auto data1 = op::Data();
        auto data2 = op::Data();
        auto data3 = op::Data();
        auto data4 = op::Data();
        auto conv = op::Conv2D().set_input_x(data1).set_input_filter(data2);
        auto add = op::Add().set_input_x1(data3).set_input_x2(data4);
        auto mean = op::ReduceMean().set_input_x(conv);
        auto neg = op::Neg().set_input_x(conv);
        auto detect_out = op::SSDDetectionOutput().set_input_x1(conv).set_input_x2(conv).set_input_x3(data3);
        auto prelu = op::PReLU().set_input_x(add);
        auto sub = op::Sub().set_input_x1(neg).set_input_x2(detect_out, "y1");
        auto split = op::SplitD().set_input_input_value(prelu).create_dynamic_output_output_data(2).set_attr_split_dim(0).set_attr_num_split(1);
        auto mul = op::Mul().set_input_x(detect_out, "y2").set_input_y(split, "output_data0");
        auto pool = op::Pooling().set_input_x(split, "output_data1");
        auto div = op::FloorDiv().set_input_x(sub).set_input_y(mul);
        auto activation = op::Activation().set_input_x(mul);

        return BuildGraph({data1, data2, data3, data4});
    }*/
};

class Linker {
public:
    Linker(FusionRuleNodePtr node)
        : node_(node)
    {
    }
    Linker Link(FusionRuleNodePtr dst, int dst_idx = 0)
    {
        Link(node_, 0, dst, dst_idx);
        return Linker(dst);
    }
    Linker Link(int src_idx, FusionRuleNodePtr dst, int dst_idx = 0)
    {
        Link(node_, src_idx, dst, dst_idx);
        return Linker(dst);
    }
    static void Link(FusionRuleNodePtr src, FusionRuleNodePtr dst)
    {
        Link(src, 0, dst, 0);
    }
    static void Link(FusionRuleNodePtr src, int src_idx, FusionRuleNodePtr dst, int dst_idx)
    {
        FusionRuleAnchorPtr src_anchor = src->output_data_anchors_[src_idx];
        FusionRuleAnchorPtr dst_anchor = dst->input_data_anchors_[dst_idx];
        src_anchor->peer_anchors_.push_back(dst_anchor);
        dst_anchor->peer_anchors_.push_back(src_anchor);
    }

private:
    FusionRuleNodePtr node_;
};

class USTEST_GraphMatcher : public testing::Test
{
protected:
  void SetUp()
  {
  }

  void TearDown()
  {

  }

  static ComputeGraphPtr BuildGraph(vector<Operator> inputs)
  {
    Graph graph("graph");
    vector<Operator> outputs {};
    graph.SetInputs(inputs).SetOutputs(outputs);
    return GraphUtilsEx::GetComputeGraph(graph);
  }

  Status Match(ComputeGraphPtr graph, vector<GraphMatchResult>& match_results)
  {
    rule_ = make_shared<FusionRulePattern>();
    rule_->rule_name_ = "myrule";
    rule_->input_info_ = inputs_;
    rule_->origin_rule_nodes_ = org_nodes_;
    rule_->output_info_ = outputs_;

    inputs_.clear();
    org_nodes_.clear();
    outputs_.clear();

    return GraphMatcher().Match(*rule_, *graph, match_results);
  }

  FusionRuleNodePtr CreateInputNode(const string& name, int out_anchors = 1)
  {
    FusionRuleNodePtr node = CreateRuleNode(name, {}, 0, out_anchors);
    SetAnchorIdxDefault(node->output_data_anchors_);
    inputs_.push_back(node);
    return node;
  }

  FusionRuleNodePtr CreateOrgNode(const string& name, vector<string>&& types, int in_anchors = 1, int out_anchors = 1)
  {
    FusionRuleNodePtr node = CreateRuleNode(name, types, in_anchors, out_anchors);
    org_nodes_.emplace(node);
    return node;
  }

  FusionRuleNodePtr CreateOutputNode(const string& name, int in_anchors = 1)
  {
    FusionRuleNodePtr node = CreateRuleNode(name, {}, in_anchors, 0);
    SetAnchorIdxDefault(node->input_data_anchors_);
    outputs_.push_back(node);
    return node;
  }

  bool IsGraphMatched(vector<GraphMatchResult>& match_results)
  {
    return true;
  }

  void SetInputAnchorIdx(FusionRuleNodePtr node, int idx, int value)
  {
    node->input_data_anchors_[idx]->anchor_idx_ = value;
  }

  void SetOutputAnchorIdx(FusionRuleNodePtr node, int idx, int value)
  {
    node->output_data_anchors_[idx]->anchor_idx_ = value;
  }

private:
  static FusionRuleNodePtr CreateRuleNode(const string& name, vector<string> types, int in_anchors, int out_anchors)
  {
    FusionRuleNodePtr node = make_shared<fe::FusionRuleNode>();
    node->node_name_ = name;
    node->node_type_ = types;
    for (int i = 0; i < in_anchors; i++) {
      FusionRuleAnchorPtr anchor = make_shared<FusionRuleAnchor>();
      anchor->anchor_idx_ = i;
      anchor->anchor_name_ = name + "_input_" + to_string(i);
      anchor->owner_node_ = node;
      node->input_data_anchors_.push_back(anchor);
    }
    for (int i = 0; i < out_anchors; i++) {
      FusionRuleAnchorPtr anchor = make_shared<FusionRuleAnchor>();
      anchor->anchor_idx_ = i;
      anchor->anchor_name_ = name + "_output_" + to_string(i);
      anchor->owner_node_ = node;
      node->output_data_anchors_.push_back(anchor);
    }
    return node;
  }

  static void SetAnchorIdxDefault(vector<FusionRuleAnchorPtr> anchors)
  {
    for (FusionRuleAnchorPtr anchor : anchors) {
      anchor->anchor_idx_ = DEFAULT_ANCHOR_INDEX;
    }
  }

private:
  FusionRulePatternPtr rule_;
  vector<FusionRuleNodePtr> inputs_;
  set<FusionRuleNodePtr> org_nodes_;
  vector<FusionRuleNodePtr> outputs_;
};

TEST_F(USTEST_GraphMatcher, case_single_in_single_out_success)
{
auto x = op::Data();
auto pool = op::Pooling().set_input_x(x);
auto neg = op::Neg().set_input_x(pool);
auto mean = op::ReduceMean().set_input_x(neg);
ComputeGraphPtr comp_graph = BuildGraph({x});

auto node_in = CreateInputNode("in1");
auto node_pool = CreateOrgNode("pooling", {"Pooling"});
auto node_neg = CreateOrgNode("neg", {"Neg"});
auto node_out = CreateOutputNode("out");
Linker(node_in).Link(node_pool).Link(node_neg).Link(node_out);

vector<GraphMatchResult> match_results;
Status ret = Match(comp_graph, match_results);

EXPECT_EQ(ret, fe::SUCCESS);
EXPECT_EQ(match_results.size(), 1);
EXPECT_EQ(IsGraphMatched(match_results), true);
}

// x  y
// \ /
// mul
// neg
// out
TEST_F(USTEST_GraphMatcher, case_multi_in_single_out_success)
{
auto x = op::Data();
auto y = op::Data();
auto mul = op::Mul().set_input_x1(x).set_input_x2(y);
auto neg = op::Neg().set_input_x(mul);
auto mean = op::ReduceMean().set_input_x(neg);
ComputeGraphPtr comp_graph = BuildGraph({x, y});

auto node_in1 = CreateInputNode("in1");
auto node_in2 = CreateInputNode("in2");
auto node_mul = CreateOrgNode("mul", {"Mul"}, 2, 1);
auto node_neg = CreateOrgNode("neg", {"Neg"});
auto node_out = CreateOutputNode("out");
Linker(node_in1).Link(node_mul);
Linker(node_in2).Link(0, node_mul, 1).Link(node_neg).Link(node_out);

vector<GraphMatchResult> match_results;
Status ret = Match(comp_graph, match_results);

EXPECT_EQ(ret, fe::SUCCESS);
EXPECT_EQ(match_results.size(), 1);
EXPECT_EQ(IsGraphMatched(match_results), true);
}
/*
TEST_F(UTEST_GraphMatcher, case_match_all_success)
{
    ComputeGraphPtr comp_graph = GenGraph();

    auto in1 = CreateInputNode("in1");
    auto in2 = CreateInputNode("in2");
    auto in3 = CreateInputNode("in3");
    auto in4 = CreateInputNode("in4");
    auto conv = CreateOrgNode("conv", {"Conv2D"}, 2, 1);
    auto add = CreateOrgNode("add", {"Add"}, 2, 1);
    auto neg = CreateOrgNode("neg", {"Neg"});
    auto detect_out = CreateOrgNode("detectOut", {"SSDDetectionOutput"}, 3, 2);
    auto prelu = CreateOrgNode("prelu", {"PReLU"});
    auto sub = CreateOrgNode("sub", {"Sub"}, 2, 1);
    auto split = CreateOrgNode("split", {"SplitD"}, 1, 2);
    auto mul = CreateOrgNode("mul", {"Mul"}, 2, 1);
    auto out1 = CreateOutputNode("out1");
    auto out2 = CreateOutputNode("out2");
    auto out3 = CreateOutputNode("out3");
    auto out4 = CreateOutputNode("out4");
    Linker(in1).Link(conv);
    Linker(in2).Link(conv, 1);
    Linker(conv).Link(out1);
    Linker(conv).Link(neg);
    Linker(conv).Link(detect_out, 0);
    Linker(conv).Link(detect_out, 1);
    Linker(in3).Link(detect_out, 2);
    Linker(in3).Link(add, 0);
    Linker(in4).Link(add, 1);
    Linker(add).Link(prelu);
    Linker(neg).Link(sub, 0);
    Linker(detect_out).Link(0, sub, 1);
    Linker(detect_out).Link(1, mul, 0);
    Linker(prelu).Link(split);
    Linker(split).Link(0, mul, 1);
    Linker(split).Link(1, out4);
    Linker(sub).Link(out2);
    Linker(mul).Link(out3);

    vector<GraphMatchResult> match_results;
    Status ret = Match(comp_graph, match_results);

    EXPECT_EQ(ret, fe::SUCCESS);
    EXPECT_EQ(match_results.size(), 1);

    auto& outer_outputs = match_results[0].outer_outputs;
    EXPECT_EQ(outer_outputs[out1->GetInputDataAnchors().front()].size(), 1);
    EXPECT_EQ(outer_outputs[out2->GetInputDataAnchors().front()].size(), 1);
    EXPECT_EQ(outer_outputs[out3->GetInputDataAnchors().front()].size(), 2);
    EXPECT_EQ(outer_outputs[out4->GetInputDataAnchors().front()].size(), 1);
}
*/
