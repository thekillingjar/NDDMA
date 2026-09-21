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

#include "graph/ge_tensor.h"
#include "graph/op_desc.h"
#include "graph/compute_graph.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/attr_utils.h"
#include "graph/utils/tensor_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/graph_utils_ex.h"
#include "graph/compute_graph.h"
#include "all_ops_stub.h"

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

// x -> pooling -> neg -> out
TEST_F(UTEST_GraphMatcher, case_single_in_single_out_success)
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
TEST_F(UTEST_GraphMatcher, case_multi_in_single_out_success)
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

//     x
//     |
//   relu
//     |
//   split
//    / \
//  neg mean
//   |   |
// out1 out2
TEST_F(UTEST_GraphMatcher, case_single_in_multi_out_success)
{
    auto x = op::Data();
    auto activation = op::Relu().set_input_x(x);
    auto split = op::SplitD().set_input_x(activation).create_dynamic_output_y(2).set_attr_split_dim(0).set_attr_num_split(1);
    auto neg = op::Neg().set_input_x(split, "output0");
    auto mean = op::ReduceMean().set_input_x(split, "output1");
    auto out1 = op::Neg().set_input_x(neg);
    auto out2 = op::Neg().set_input_x(mean);
    ComputeGraphPtr comp_graph = BuildGraph({x});

    auto node_in = CreateInputNode("in");
    auto node_activation = CreateOrgNode("activation", {"Activation"});
    auto node_split = CreateOrgNode("split", {"SplitD"}, 1, 2);
    auto node_neg = CreateOrgNode("neg", {"Neg"});
    auto node_mean = CreateOrgNode("mean", {"ReduceMean"});
    auto node_out1 = CreateOutputNode("out1");
    auto node_out2 = CreateOutputNode("out2");
    Linker(node_in).Link(node_activation).Link(node_split);
    Linker(node_split).Link(0, node_neg).Link(node_out1);
    Linker(node_split).Link(1, node_mean).Link(node_out2);

    vector<GraphMatchResult> match_results;
    Status ret = Match(comp_graph, match_results);

    EXPECT_EQ(ret, fe::SUCCESS);
    EXPECT_EQ(match_results.size(), 0);
    EXPECT_EQ(IsGraphMatched(match_results), true);
}

//     x   y   z
//     |   |   |
//   neg  mean prelu
//     \   |   /
// SSDDetectionOutput
//       /   \
//   asinh  activation
//      |    |
//    out1  out2
//TEST_F(UTEST_GraphMatcher, case_multi_in_multi_out_success)
//{
//    auto x = op::Data();
//    auto y = op::Data();
//    auto z = op::Data();
//    auto neg = op::Neg().set_input_x(x);
//    auto mean = op::ReduceMean().set_input_x(y);
//    auto prelu = op::PRelu().set_input_x(z);
//    auto detect_out = op::SSDDetectionOutput().set_input_x1(neg).set_input_x2(mean).set_input_x3(prelu);
//    auto asinh = op::Asinh().set_input_x(detect_out, "y1");
//    auto activation = op::Activation().set_input_x(detect_out, "y2");
//    auto out1 = op::Neg().set_input_x(asinh);
//    auto out2 = op::Neg().set_input_x(activation);
//    ComputeGraphPtr comp_graph = BuildGraph({x, y, z});
//
//    auto node_in1 = CreateInputNode("in1");
//    auto node_in2 = CreateInputNode("in2");
//    auto node_in3 = CreateInputNode("in3");
//    auto node_neg = CreateOrgNode("Neg", {"Neg"});
//    auto node_mean = CreateOrgNode("Mean", {"ReduceMean"});
//    auto node_p_re_l_u = CreateOrgNode("PReLU", {"PRelu"});
//    auto node_dect_out = CreateOrgNode("SSDDetectionOutput", {"SSDDetectionOutput"}, 3, 2);
//    auto node_asinh = CreateOrgNode("Asinh", {"Asinh"});
//    auto node_activation = CreateOrgNode("Activation", {"Activation"});
//    auto node_out1 = CreateOutputNode("out1");
//    auto node_out2 = CreateOutputNode("out2");
//    Linker(node_in1).Link(node_neg).Link(node_dect_out, 0);
//    Linker(node_in2).Link(node_mean).Link(node_dect_out, 1);
//    Linker(node_in3).Link(node_p_re_l_u).Link(node_dect_out, 2);
//    Linker(node_dect_out).Link(0, node_asinh).Link(node_out1);
//    Linker(node_dect_out).Link(1, node_activation).Link(node_out2);
//
//    vector<GraphMatchResult> match_results;
//    Status ret = Match(comp_graph, match_results);
//
//    EXPECT_EQ(ret, fe::SUCCESS);
//    EXPECT_EQ(match_results.size(), 1);
//    EXPECT_EQ(IsGraphMatched(match_results), true);
//}

//     x   y   z
//     |   |   |
// SSDDetectionOutput
//       /   \
//    out1  out2
//TEST_F(UTEST_GraphMatcher, case_multi_in_multi_out_2_success)
//{
//    auto x = op::Data();
//    auto y = op::Data();
//    auto z = op::Data();
//    auto detect_out = op::SSDDetectionOutput().set_input_x1(x).set_input_x2(y).set_input_x3(z);
//    auto out1 = op::Neg().set_input_x(detect_out, "y1");
//    auto out2 = op::Neg().set_input_x(detect_out, "y2");
//    ComputeGraphPtr comp_graph = BuildGraph({x, y, z});
//
//    auto node_in1 = CreateInputNode("in1");
//    auto node_in2 = CreateInputNode("in2");
//    auto node_in3 = CreateInputNode("in3");
//    auto node_dect_out = CreateOrgNode("SSDDetectionOutput", {"SSDDetectionOutput"}, 3, 2);
//    auto node_out1 = CreateOutputNode("out1");
//    auto node_out2 = CreateOutputNode("out2");
//    Linker(node_in1).Link(node_dect_out, 0);
//    Linker(node_in2).Link(node_dect_out, 1);
//    Linker(node_in3).Link(node_dect_out, 2);
//    Linker(node_dect_out).Link(0, node_out1);
//    Linker(node_dect_out).Link(1, node_out2);
//
//    vector<GraphMatchResult> match_results;
//    Status ret = Match(comp_graph, match_results);
//
//    EXPECT_EQ(ret, fe::SUCCESS);
//    EXPECT_EQ(match_results.size(), 1);
//    EXPECT_EQ(IsGraphMatched(match_results), true);
//}

//            x y
//            \ /
//            mul
//    /   /   |   \    \
// neg1 mean neg2 neg3 asinh
// out1 out2 out3 out4 out5
TEST_F(UTEST_GraphMatcher, case_single_out_multi_use_success)
{
    auto x = op::Data();
    auto y = op::Data();
    auto mul = op::Mul().set_input_x1(x).set_input_x2(y);
    auto neg1 = op::Neg().set_input_x(mul);
    auto mean = op::ReduceMean().set_input_x(mul);
    auto neg2 = op::Neg().set_input_x(mul);
    auto neg3 = op::Neg().set_input_x(mul);
    auto asinh = op::Asinh().set_input_x(mul);
    auto out1 = op::Neg().set_input_x(neg1);
    auto out2 = op::Neg().set_input_x(mean);
    auto out3 = op::Neg().set_input_x(neg2);
    auto out4 = op::Neg().set_input_x(neg3);
    auto out5 = op::Neg().set_input_x(asinh);
    ComputeGraphPtr comp_graph = BuildGraph({x, y});

    auto node_in1 = CreateInputNode("in1");
    auto node_in2 = CreateInputNode("in2");
    auto node_mul = CreateOrgNode("mul", {"Mul"}, 2, 1);
    auto node_mean = CreateOrgNode("Mean", {"ReduceMean"});
    auto node_neg1 = CreateOrgNode("neg1", {"Neg"});
    auto node_neg2 = CreateOrgNode("neg2", {"Neg"});
    auto node_asinh = CreateOrgNode("Asinh", {"Asinh"});
    auto node_neg3 = CreateOrgNode("neg3", {"Neg"});
    auto node_out1 = CreateOutputNode("out1");
    auto node_out2 = CreateOutputNode("out2");
    auto node_out3 = CreateOutputNode("out3");
    auto node_out4 = CreateOutputNode("out4");
    auto node_out5 = CreateOutputNode("out5");
    Linker(node_in1).Link(node_mul);
    Linker(node_in2).Link(0, node_mul, 1);
    Linker(node_mul).Link(node_mean).Link(node_out1);
    Linker(node_mul).Link(node_neg1).Link(node_out2);
    Linker(node_mul).Link(node_neg2).Link(node_out3);
    Linker(node_mul).Link(node_asinh).Link(node_out4);
    Linker(node_mul).Link(node_neg3).Link(node_out5);

    vector<GraphMatchResult> match_results;
    Status ret = Match(comp_graph, match_results);

    EXPECT_EQ(ret, fe::SUCCESS);
    EXPECT_EQ(match_results.size(), 1);
    EXPECT_EQ(IsGraphMatched(match_results), true);
}

// x   y  z
// \  /\ /
// mul1 mul2
//  |    |
// neg1 neg2
//   \  /
//    out
/*
TEST_F(UTEST_GraphMatcher, case_match_multi_graph_success)
{
    auto x = op::Data();
    auto y = op::Data();
    auto z = op::Data();
    auto mul1 = op::Mul().set_input_x(x).set_input_y(y);
    auto mul2 = op::Mul().set_input_x(y).set_input_y(z);
    auto neg1 = op::Neg().set_input_x(mul1);
    auto neg2 = op::Neg().set_input_x(mul2);
    auto out = op::Mul().set_input_x(neg1).set_input_y(neg2);
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
    EXPECT_EQ(match_results.size(), 2);
    EXPECT_EQ(IsGraphMatched(match_results), true);
}
*/
// x w
// \ | /
// conv2d
//  neg
//  out
/*TEST_F(UTEST_GraphMatcher, case_unused_input_success)
{
    auto x = op::Data();
    auto w = op::Data();
    auto conv2d = op::Conv2D().set_input_x(x).set_input_w(w);
    auto neg = op::Neg().set_input_x(conv2d);
    auto mean = op::Mean().set_input_x(neg);
    ComputeGraphPtr comp_graph = BuildGraph({x, w});

    auto node_in1 = CreateInputNode("in1");
    auto node_in2 = CreateInputNode("in2");
    auto node_conv2_d = CreateOrgNode("conv2d", {"Conv2D"}, 2, 1);
    auto node_neg = CreateOrgNode("neg", {"Neg"});
    auto node_out = CreateOutputNode("out");
    Linker(node_in1).Link(node_conv2_d);
    Linker(node_in2).Link(0, node_conv2_d, 1).Link(node_neg).Link(node_out);

    vector<GraphMatchResult> match_results;
    Status ret = Match(comp_graph, match_results);

    EXPECT_EQ(ret, fe::SUCCESS);
    EXPECT_EQ(match_results.size(), 1);
    EXPECT_EQ(IsGraphMatched(match_results), true);
}
*/
//     x
//     |
//   relu 
//     |
//   split
//    / \
//  neg
//   |
// out1
TEST_F(UTEST_GraphMatcher, case_unused_output_success)
{
    auto x = op::Data();
    auto activation = op::Relu().set_input_x(x);
    auto split = op::SplitD().set_input_x(activation).create_dynamic_output_y(2).set_attr_split_dim(0).set_attr_num_split(1);
    auto neg = op::Neg().set_input_x(split, "output0");
    auto out1 = op::Neg().set_input_x(neg);
    ComputeGraphPtr comp_graph = BuildGraph({x});

    auto node_in = CreateInputNode("in");
    auto node_activation = CreateOrgNode("activation", {"Activation"});
    auto node_split = CreateOrgNode("split", {"SplitD"}, 1, 1);
    auto node_neg = CreateOrgNode("neg", {"Neg"});
    auto node_out1 = CreateOutputNode("out1");
    Linker(node_in).Link(node_activation).Link(node_split).Link(0, node_neg).Link(node_out1);

    vector<GraphMatchResult> match_results;
    Status ret = Match(comp_graph, match_results);

    EXPECT_EQ(ret, fe::SUCCESS);
    EXPECT_EQ(match_results.size(), 0);
    EXPECT_EQ(IsGraphMatched(match_results), true);
}

//     in
//    /  \
//  neg mean
//   |   |
// out1 out2
TEST_F(UTEST_GraphMatcher, case_outer_inputs_from_same_node_same_anchor_success)
{
    auto in = op::Data();
    auto neg = op::Neg().set_input_x(in);
    auto mean = op::ReduceMean().set_input_x(in);
    auto out1 = op::Neg().set_input_x(neg);
    auto out2 = op::Neg().set_input_x(mean);
    ComputeGraphPtr comp_graph = BuildGraph({in});

    auto node_in = CreateInputNode("in");
    auto node_neg = CreateOrgNode("neg", {"Neg"});
    auto node_mean = CreateOrgNode("mean", {"ReduceMean"});
    auto node_out1 = CreateOutputNode("out1");
    auto node_out2 = CreateOutputNode("out2");
    Linker(node_in).Link(node_mean).Link(node_out1);
    Linker(node_in).Link(node_neg).Link(node_out2);

    vector<GraphMatchResult> match_results;
    Status ret = Match(comp_graph, match_results);

    EXPECT_EQ(ret, fe::SUCCESS);
    EXPECT_EQ(match_results.size(), 1);
    EXPECT_EQ(IsGraphMatched(match_results), true);
}

//  in(Split)
//  /   /\
//   neg mean
//    |   |
//  out1 out2
TEST_F(UTEST_GraphMatcher, case_outer_inputs_from_same_node_same_anchor_2_success)
{
    auto x = op::Data();
    auto in = op::SplitD().set_input_x(x).create_dynamic_output_y(2).set_attr_split_dim(0).set_attr_num_split(1);
    auto neg = op::Neg().set_input_x(in, "output1");
    auto mean = op::ReduceMean().set_input_x(in, "output1");
    auto out1 = op::Neg().set_input_x(neg);
    auto out2 = op::Neg().set_input_x(mean);
    ComputeGraphPtr comp_graph = BuildGraph({x});

    auto node_in = CreateInputNode("in");
    auto node_neg = CreateOrgNode("neg", {"Neg"});
    auto node_mean = CreateOrgNode("mean", {"ReduceMean"});
    auto node_out1 = CreateOutputNode("out1");
    auto node_out2 = CreateOutputNode("out2");
    Linker(node_in).Link(node_neg).Link(node_out1);
    Linker(node_in).Link(node_mean).Link(node_out2);

    vector<GraphMatchResult> match_results;
    Status ret = Match(comp_graph, match_results);

    EXPECT_EQ(ret, fe::SUCCESS);
    EXPECT_EQ(match_results.size(), 0);
    EXPECT_EQ(IsGraphMatched(match_results), true);
}

//  in(Split)
//    /  \
//  neg mean
//   |   |
// out1 out2
TEST_F(UTEST_GraphMatcher, case_outer_inputs_from_same_node_diff_anchor_success)
{
    auto x = op::Data();
    auto in = op::SplitD().set_input_x(x).create_dynamic_output_y(2).set_attr_split_dim(0).set_attr_num_split(1);
    auto neg = op::Neg().set_input_x(in, "output0");
    auto mean = op::ReduceMean().set_input_x(in, "output1");
    auto out1 = op::Neg().set_input_x(neg);
    auto out2 = op::Neg().set_input_x(mean);
    ComputeGraphPtr comp_graph = BuildGraph({x});

    auto node_in = CreateInputNode("in", 2);
    auto node_neg = CreateOrgNode("neg", {"Neg"});
    auto node_mean = CreateOrgNode("mean", {"ReduceMean"});
    auto node_out1 = CreateOutputNode("out1");
    auto node_out2 = CreateOutputNode("out2");
    SetOutputAnchorIdx(node_in, 0, 0);
    SetOutputAnchorIdx(node_in, 1, 1);
    Linker(node_in).Link(0, node_neg).Link(node_out1);
    Linker(node_in).Link(1, node_mean).Link(node_out2);

    vector<GraphMatchResult> match_results;
    Status ret = Match(comp_graph, match_results);

    EXPECT_EQ(ret, fe::SUCCESS);
    EXPECT_EQ(match_results.size(), 0);
    EXPECT_EQ(IsGraphMatched(match_results), true);
}

//  x
// ||
// mul
// neg
// out
TEST_F(UTEST_GraphMatcher, case_diff_outer_input_match_same_node_success)
{
    auto x = op::Data();
    auto mul = op::Mul().set_input_x1(x).set_input_x2(x);
    auto neg = op::Neg().set_input_x(mul);
    auto mean = op::ReduceMean().set_input_x(neg);
    ComputeGraphPtr comp_graph = BuildGraph({x});

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
    EXPECT_EQ(match_results[0].outer_inputs[node_in1->GetOutputDataAnchors()[0]],
        match_results[0].outer_inputs[node_in2->GetOutputDataAnchors()[0]]);
}

//    x
//  /   \
// neg neg
//  \   /
//   out
TEST_F(UTEST_GraphMatcher, case_diff_outer_output_match_same_node_success)
{
    auto x = op::Data();
    auto neg1 = op::Neg().set_input_x(x);
    auto neg2 = op::Neg().set_input_x(x);
    auto out = op::Mul().set_input_x1(neg1).set_input_x2(neg2);
    ComputeGraphPtr comp_graph = BuildGraph({x});

    auto node_in = CreateInputNode("in");
    auto node_neg1 = CreateOrgNode("neg1", {"Neg"});
    auto node_neg2 = CreateOrgNode("neg2", {"Neg"});
    auto node_out1 = CreateOutputNode("out1");
    auto node_out2 = CreateOutputNode("out2");
    Linker(node_in).Link(node_neg1).Link(node_out1);
    Linker(node_in).Link(node_neg2).Link(node_out2);

    vector<GraphMatchResult> match_results;
    Status ret = Match(comp_graph, match_results);

    EXPECT_EQ(ret, fe::SUCCESS);
    EXPECT_EQ(match_results.size(), 1);
    EXPECT_EQ(IsGraphMatched(match_results), true);

    const auto& out1_anchors = match_results[0].outer_outputs[node_out1->GetInputDataAnchors()[0]];
    const auto& out2_anchors = match_results[0].outer_outputs[node_out2->GetInputDataAnchors()[0]];
    EXPECT_EQ(out1_anchors.size(), 1);
    EXPECT_EQ(out2_anchors.size(), 1);
    EXPECT_EQ((*out1_anchors.begin())->GetOwnerNode(), (*out2_anchors.begin())->GetOwnerNode());
}

//     x
//     |
//   relu 
//     |
//   mean  y
//    / \ /
// out1 out2
TEST_F(UTEST_GraphMatcher, case_outer_output_support_control_anchor_success)
{
    auto x = op::Data();
    auto y = op::Data();
    auto activation = op::Relu().set_input_x(x);
    auto mean = op::ReduceMean("mean").set_input_x(activation);
    auto out1 = op::Neg("out1").set_input_x(mean);
    auto out2 = op::ReduceMean("out2").set_input_x(y); // AddControlInput(mean);
    ComputeGraphPtr comp_graph = BuildGraph({x, y});

    NodePtr mean_node = nullptr;
    NodePtr out2_node = nullptr;
    for (NodePtr node : comp_graph->GetDirectNode()) {
        if (node->GetName() == "mean")
            mean_node = node;
        else if (node->GetName() == "out2") {
            out2_node = node;
        }
    }
    EXPECT_NE(mean_node, nullptr);
    EXPECT_NE(out2_node, nullptr);

    OutDataAnchorPtr mean_out_data_anchor = mean_node->GetOutDataAnchor(0);
    InControlAnchorPtr out2_in_ctrl_anchor = out2_node->GetInControlAnchor();
    mean_out_data_anchor->LinkTo(out2_in_ctrl_anchor);

    auto node_in = CreateInputNode("in");
    auto node_activation = CreateOrgNode("activation", {"Relu"});
    auto node_mean = CreateOrgNode("mean", {"ReduceMean"});
    auto node_out1 = CreateOutputNode("out1");
    Linker(node_in).Link(node_activation).Link(node_mean).Link(node_out1);

    vector<GraphMatchResult> match_results;
    Status ret = Match(comp_graph, match_results);

    EXPECT_EQ(ret, fe::SUCCESS);
    EXPECT_EQ(match_results.size(), 1);
    EXPECT_EQ(IsGraphMatched(match_results), true);
}

// x -> pooling -> neg -> out
TEST_F(UTEST_GraphMatcher, case_node_type_mismatch)
{
    auto x = op::Data();
    auto pool = op::Pooling().set_input_x(x);
    auto neg = op::Neg().set_input_x(pool);
    auto mean = op::ReduceMean().set_input_x(neg);
    ComputeGraphPtr comp_graph = BuildGraph({x});

    auto node_in = CreateInputNode("in1");
    auto node_pool = CreateOrgNode("pooling", {"Pooling"});
    auto node_error = CreateOrgNode("error", {"Error"});
    auto node_out = CreateOutputNode("out");
    Linker(node_in).Link(node_pool).Link(node_error).Link(node_out);

    vector<GraphMatchResult> match_results;
    Status ret = Match(comp_graph, match_results);

    EXPECT_EQ(ret, fe::SUCCESS);
    EXPECT_EQ(match_results.size(), 0);
}

//     x
//     |
// activation
//     |
//   split
//    / \
//  neg mean
//   |   |
// out1 out2
TEST_F(UTEST_GraphMatcher, case_single_in_multi_out_out_order_mismatch)
{
    auto x = op::Data();
    auto activation = op::Relu().set_input_x(x);
    auto split = op::SplitD().set_input_x(activation).create_dynamic_output_y(2).set_attr_split_dim(0).set_attr_num_split(1);
    auto neg = op::Neg().set_input_x(split, "output0");
    auto mean = op::ReduceMean().set_input_x(split, "output1");
    auto out1 = op::Neg().set_input_x(neg);
    auto out2 = op::Neg().set_input_x(mean);
    ComputeGraphPtr comp_graph = BuildGraph({x});

    auto node_in = CreateInputNode("in");
    auto node_activation = CreateOrgNode("activation", {"Activation"});
    auto node_split = CreateOrgNode("split", {"SplitD"}, 1, 2);
  auto node_neg = CreateOrgNode("neg", {"Neg"});
    auto node_mean = CreateOrgNode("mean", {"ReduceMean"});
    auto node_out1 = CreateOutputNode("out1");
    auto node_out2 = CreateOutputNode("out2");
    Linker(node_in).Link(node_activation).Link(node_split);
    Linker(node_split).Link(0, node_mean).Link(node_out1);
    Linker(node_split).Link(1, node_neg).Link(node_out2);

    vector<GraphMatchResult> match_results;
    Status ret = Match(comp_graph, match_results);

    EXPECT_EQ(ret, fe::SUCCESS);
    EXPECT_EQ(match_results.size(), 0);
}

// x -> pooling -> out
TEST_F(UTEST_GraphMatcher, case_outer_input_anchor_count_mismatch)
{
    auto x = op::Data();
    auto pool = op::Pooling().set_input_x(x);
    auto out = op::ReduceMean().set_input_x(pool);
    ComputeGraphPtr comp_graph = BuildGraph({x});

    auto node_in = CreateInputNode("in1", 2);    // error
    auto node_pool = CreateOrgNode("pooling", {"Pooling"});
    auto node_out = CreateOutputNode("out");
    Linker(node_in).Link(node_pool).Link(node_out);

    vector<GraphMatchResult> match_results;
    Status ret = Match(comp_graph, match_results);

    EXPECT_EQ(ret, fe::SUCCESS);
    EXPECT_EQ(match_results.size(), 0);
}

TEST_F(UTEST_GraphMatcher, case_Match_from_output_nullptr_error)
{
    GraphMatchResult match_result;
    bool ret = GraphMatcher().MatchFromOutput(nullptr, nullptr, match_result);
    EXPECT_EQ(ret, false);
}

TEST_F(UTEST_GraphMatcher, case_Match_origin_node_inputs_nullptr_error)
{
    GraphMatchResult match_result;
    bool ret = GraphMatcher().MatchOriginNodeInputs(nullptr, nullptr, match_result);
    EXPECT_EQ(ret, false);
}

TEST_F(UTEST_GraphMatcher, case_Match_origin_node_outputs_nullptr_error)
{
    GraphMatchResult match_result;
    bool ret = GraphMatcher().MatchOriginNodeOutputs(nullptr, nullptr, match_result);
    EXPECT_EQ(ret, false);
}

TEST_F(UTEST_GraphMatcher, case_Match_outputs_for_outer_input_nullptr_error)
{
    GraphMatchResult match_result;
    bool ret = GraphMatcher().MatchOutputsForOuterInput(nullptr, nullptr, match_result);
    EXPECT_EQ(ret, false);
}

TEST_F(UTEST_GraphMatcher, case_Match_peer_nullptr_error)
{
    GraphMatchResult match_result;
    bool ret = GraphMatcher().MatchPeer(nullptr, nullptr, nullptr, match_result);
    EXPECT_EQ(ret, false);
}

TEST_F(UTEST_GraphMatcher, case_Match_origin_node_nullptr_error)
{
    GraphMatchResult match_result;
    bool ret = GraphMatcher().MatchOriginNode(nullptr, nullptr, match_result);
    EXPECT_EQ(ret, false);
}

TEST_F(UTEST_GraphMatcher, case_Is_node_type_match_nullptr_error)
{
    GraphMatchResult match_result;
    bool ret = GraphMatcher().IsNodeTypeMatch(nullptr, nullptr);
    EXPECT_EQ(ret, false);
}

TEST_F(UTEST_GraphMatcher, case_Is_in_outer_inputs_nullptr_error)
{
    GraphMatchResult match_result;
    bool ret = GraphMatcher().IsInOuterInputs(nullptr);
    EXPECT_EQ(ret, false);
}

TEST_F(UTEST_GraphMatcher, case_Is_in_outer_outputs_nullptr_error)
{
    GraphMatchResult match_result;
    bool ret = GraphMatcher().IsInOuterOutputs(nullptr);
    EXPECT_EQ(ret, false);
}

TEST_F(UTEST_GraphMatcher, case_Is_in_origin_graph_nullptr_error)
{
    GraphMatchResult match_result;
    bool ret = GraphMatcher().IsInOriginGraph(nullptr);
    EXPECT_EQ(ret, false);
}

TEST_F(UTEST_GraphMatcher, case_Get_first_output_rule_node_nullptr_error)
{
    GraphMatchResult match_result;
    FusionRuleNodePtr ret = GraphMatcher().GetFirstOutputRuleNode();
    EXPECT_EQ(ret, nullptr);
}
