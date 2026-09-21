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

#define protected public
#define private   public

#include "common/debug/log.h"
#include "common/types.h"
#include "graph/graph.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/graph_utils_ex.h"
#include "graph/utils/attr_utils.h"
#include "graph/types.h"
#include "graph/ge_tensor.h"
#include "graph/op_desc.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/tensor_utils.h"
#include "fusion_rule_manager/fusion_rule_data/fusion_rule_pattern.h"
#include "ops_kernel_store/fe_ops_kernel_info_store.h"
#include "graph_optimizer/graph_fusion/graph_replace.h"
#include "adapter/tbe_adapter/tbe_op_store_adapter.h"
#undef protected
#undef private
#include "graph/compute_graph.h"
#include "all_ops_stub.h"

using namespace std;
using namespace fe;
using namespace ge;


using FEOpsKernelInfoStorePtr = std::shared_ptr<FEOpsKernelInfoStore>;

class UTEST_fusion_engine_graph_replace_unittest :public testing::Test {
protected:
    void SetUp()
    {
    }
    void TearDown()
    {

    }

protected:
    /*
        ���ܣ���������ͼ
              ����ͼ�ṹ���£�
                                        data1  weight
                                          |    /
                                          |   /
                                          |  /
                                        matmul bias
                                          |    /
                                          |   /
                                       BiasAdd
                                          |
                                          |
                                        Square1
                                          |  data2
                                          | /  |
                                         Mul   |
                                          |    |
                                          |    |
                                         Sum--Mul
                                               |
                                               |
                                              Square2
*/
    ge::ComputeGraphPtr CreateGraph()
    {
        Graph graph("testGraph");
        auto data1 = op::Data("data1").set_attr_index(0);
        auto weight = op::Data("weight").set_attr_index(1);
        auto bias = op::Data("bias").set_attr_index(2);
        auto mat_mul = op::MatMul("matMul").set_input_x1(data1).set_input_x2(weight);
        auto bias_add = op::BiasAdd("biasAdd").set_input_x(mat_mul).set_input_bias(bias);
        auto square_1 = op::Square("square_1").set_input_x(bias_add);
        auto data2 = op::Data("data2").set_attr_index(3);
        auto mul1 = op::Mul("mul1").set_input_x1(square_1).set_input_x2(data2);
        auto sum = op::ReduceSumD("reduceSum").set_input_x(mul1).set_attr_axes({-1}).set_attr_keep_dims(true);
        auto mul2 = op::Mul("mul2").set_input_x1(sum).set_input_x2(data2);
        auto square_2 = op::Square("square_2").set_input_x(mul2);
        vector<Operator> inputs {data1, weight, bias, data2};
        vector<Operator> outputs {square_2};
        graph.SetInputs(inputs).SetOutputs(outputs);
        GraphUtilsEx::GetComputeGraph(graph);
        auto compute_graph = GraphUtilsEx::GetComputeGraph(graph);
        for (auto node : compute_graph->GetDirectNode()) {
            if (node->GetType() == "BiasAdd") {
                ge::AttrUtils::SetStr(*node->GetOpDesc().get(), "data_format", "NCHW");
            }
        }
        return compute_graph;
    }
  static ComputeGraphPtr SquareSumV1()
  {
    auto data1 = op::Data("Input0");
    auto square0 = op::Square("Square0").set_input_x(data1);
    auto sum0 = op::ReduceSum("Sum0").set_input_x(square0);
    auto data2 = op::Data("DataOut0").set_input_x(sum0);
    Graph graph("graph_SquareSumV1");
    vector<Operator> outputs {};
    graph.SetInputs({data1}).SetOutputs(outputs);

    ComputeGraphPtr tmp_graph = GraphUtilsEx::GetComputeGraph(graph);
    ge::NodePtr sq = tmp_graph->FindNode("Square0");
    string stream_label = "All_Cast_t";
    ge::AttrUtils::SetStr(sq->GetOpDesc(), "_stream_label", stream_label);
    return tmp_graph;
  }

    /*
        �����ںϹ���1��matmul+bias_add�ںϳ�һ��matmul��biad_add��������Ϊmatmul�ĵ���������
 */
    vector<FusionRulePatternPtr> CreateFusionRulePatternOne()
    {
        vector<FusionRulePatternPtr> fusion_patterns;
        FusionRulePatternPtr fusion_rule_pattern1 = make_shared<FusionRulePattern>();
        fusion_rule_pattern1->rule_name_ = "test_fusion_one";

        FusionRuleNodePtr data_node1 = CreateFusionRuleNode("data1", {"Data"}, {}, {0}, {});
        FusionRuleNodePtr weight_node = CreateFusionRuleNode("weight", {"Data"}, {}, {0}, {});
        FusionRuleNodePtr bias_node = CreateFusionRuleNode("bias", {"Data"}, {}, {0}, {});
        // ��������FusionRuleNode
        fusion_rule_pattern1->input_info_.push_back(data_node1);
        fusion_rule_pattern1->input_info_.push_back(weight_node);
        fusion_rule_pattern1->input_info_.push_back(bias_node);

        // matmul rule node
        FusionRuleNodePtr matmul_node = CreateFusionRuleNode("matMul", {"MatMul"}, {0, 1}, {0}, {});
        Link(data_node1, 0, matmul_node, 0);
        Link(weight_node, 0, matmul_node, 1);
        // add2 rule node
        FusionRuleNodePtr bias_add_node = CreateFusionRuleNode("biasAdd", {"BiasAdd"}, {0, 1}, {0}, {});
        Link(matmul_node, 0, bias_add_node, 0);
        Link(bias_node, 0, bias_add_node, 1);
        // �����ں�ǰFusionRuleNode
        fusion_rule_pattern1->origin_rule_nodes_.insert(matmul_node);
        fusion_rule_pattern1->origin_rule_nodes_.insert(bias_add_node);
        // �������node
        FusionRuleNodePtr square_node1 = CreateFusionRuleNode("square_1", {"Square"}, {0}, {}, {});
        Link(bias_add_node, 0, square_node1, 0);
        fusion_rule_pattern1->output_info_.push_back(square_node1);
        // �����ںϺ�FusionRuleNode
        // attributes
        map<string, FusionRuleAttrValuePtr> attributes = {};
        ge::GeAttrValue attr_value1 = ge::GeAttrValue();
        attr_value1.SetValue(true);
        ge::GeAttrValue attr_value2 = ge::GeAttrValue();
        FusionRuleAttr rule_node_attr;
        FusionRuleAttrValuePtr  attrvalue_ptr1 = CreateAttribute(false,  attr_value1, rule_node_attr);
        rule_node_attr.node_name = "node2";
        rule_node_attr.attr_name = "has_bias";
        FusionRuleAttrValuePtr  attrvalue_ptr2 = CreateAttribute(true, attr_value2, rule_node_attr);
        attributes["has_matmul"] = attrvalue_ptr1;
        attributes["has_bias"] = attrvalue_ptr2;
        attrvalue_ptr2->SetOwnerNode(matmul_node);
        // Fusion node
        FusionRuleNodePtr fusion_matmul_node = CreateFusionRuleNode("matMul", {"MatMul"}, {0, 1, 2}, {0},
                                                            attributes);
        Link(fusion_matmul_node, 0, square_node1, 0);
        Link(data_node1, 0, fusion_matmul_node, 0);
        Link(weight_node, 0, fusion_matmul_node, 1);
        Link(bias_node, 0, fusion_matmul_node, 2);
        // fusion matmul rule node
        fusion_rule_pattern1->fusion_rule_nodes_.insert(fusion_matmul_node);

        fusion_patterns.push_back(fusion_rule_pattern1);
        return fusion_patterns;
    }
    /*
        ��������FusionRuleNode
 */
    FusionRuleNodePtr CreateFusionRuleNode(const string &node_name,
                                           const vector<string> &node_types,
                                           vector<int> inputs_anchor_indxs,
                                           vector<int> output_anchor_indexs,
                                           const map<string, FusionRuleAttrValuePtr> &attributes)
    {
        FusionRuleNodePtr node = make_shared<fe::FusionRuleNode>();
        node->node_name_ = node_name;
        node->node_type_ = node_types;
        for (size_t i = 0; i < inputs_anchor_indxs.size(); ++i) {
          int index = inputs_anchor_indxs[i];
          string anchor_name = node_name + "_input_" + to_string(index);
          auto input_anchor = CreateAnchor(index, anchor_name, node, {});
          node->input_data_anchors_.push_back(input_anchor);
        }
        for (size_t i = 0; i < output_anchor_indexs.size(); ++i) {
          int index = output_anchor_indexs[i];
          string anchor_name = node_name + "_output_" + to_string(index);
          auto output_anchor = CreateAnchor(index, anchor_name, node, {});
          node->output_data_anchors_.push_back(output_anchor);
        }
        node->attributes_ = attributes;
        return node;
    }
    /*
        ����anchor
 */
    FusionRuleAnchorPtr CreateAnchor(int anchor_idx,
                                     const string &anchor_name,
                                     FusionRuleNodePtr owner_node,
                                     const vector<FusionRuleAnchorPtr> &peer_anchors)
    {
        FusionRuleAnchorPtr anchor = make_shared<fe::FusionRuleAnchor>();
        anchor->anchor_idx_ = anchor_idx;
        anchor->anchor_name_ = anchor_name;
        anchor->owner_node_ = owner_node;
        for (size_t i = 0; i < peer_anchors.size(); ++i) {
            anchor->peer_anchors_.emplace_back(peer_anchors[i]);
        }
        for (size_t i = 0; i < anchor->peer_anchors_.size(); ++i) {
            auto peer_anchor = anchor->peer_anchors_[i].lock();
            peer_anchor->peer_anchors_.emplace_back(anchor);
        }
        return anchor;
    }
    /*
        ����attribute
 */
    FusionRuleAttrValuePtr CreateAttribute(bool is_fusion_rule_attr, ge::GeAttrValue &attr_value,
                                           FusionRuleAttr &rule_node_attr)
    {
        FusionRuleAttrValuePtr attribute = make_shared<FusionRuleAttrValue>();
        attribute->is_fusion_rule_attr_ = is_fusion_rule_attr;
        attribute->fix_value_attr_ = attr_value;
        attribute->rule_node_attr_ = rule_node_attr;
        return attribute;
    }
    void Link(FusionRuleNodePtr src, int src_idx, FusionRuleNodePtr dst, int dst_idx)
    {
        FusionRuleAnchorPtr src_anchor = src->output_data_anchors_[src_idx];
        FusionRuleAnchorPtr dst_anchor = dst->input_data_anchors_[dst_idx];
        src_anchor->peer_anchors_.push_back(dst_anchor);
        dst_anchor->peer_anchors_.push_back(src_anchor);
    }
    /*
        ����ƥ�䵽����ͼ
    */
    void CreateMappings(const vector<FusionRulePatternPtr> &patterns,
                        vector<GraphMatchResult> &match_results, ge::ComputeGraph& graph)
    {
        auto nodes = graph.GetAllNodes();
        map<string, ge::NodePtr> nodes_dict = {};
        GraphMatchResult match_result;
        for (auto i = 0; i < nodes.size(); ++i) {
            nodes_dict[(nodes.at(i))->GetName()] = nodes.at(i);
        }

        for (size_t i = 0; i < patterns.size(); ++i) {
            const shared_ptr<FusionRulePattern> pattern = patterns.at(i);
            FusionRuleNodeMapping mapping;
            const vector<FusionRuleNodePtr> &input_info = pattern->GetInputInfo();
            const vector<FusionRuleNodePtr> &output_info = pattern->GetOutputInfo();
            const set<FusionRuleNodePtr> &origin_rule_node = pattern->GetOriginRuleNodes();
            const set<FusionRuleNodePtr> &fusion_rule_nodes = pattern->GetFusionRuleNodes();
            for (size_t j = 0; j < input_info.size(); ++j) {
                auto input = input_info[j];
                mapping[input] = nodes_dict[input->GetNodeName()];
            }
            for (size_t j = 0; j < output_info.size(); ++j) {
                auto output = output_info[j];
                mapping[output] = nodes_dict[output->GetNodeName()];
            }
            for (auto iter = origin_rule_node.begin(); iter != origin_rule_node.end(); ++iter) {
                auto origin = *iter;
                mapping[origin] = nodes_dict[origin->GetNodeName()];
            }
            for (auto iter = fusion_rule_nodes.begin(); iter != fusion_rule_nodes.end(); ++iter) {
                FusionRuleNodePtr fusion_rule_node = *iter;
                const vector<FusionRuleAnchorPtr>& input_anchors =
                    fusion_rule_node->GetInputDataAnchors();
                for (auto input : input_anchors) {
                    const vector<FusionRuleAnchorPtr>& peer_anchors = input->GetPeerAnchors();
                    FusionRuleNodePtr peer_node = peer_anchors[0]->GetOwnerNode();
                    if (mapping.find(peer_node) == mapping.end()) {
                        continue;
                    }
                    ge::NodePtr node = mapping[peer_node];
                    int anchor_index = peer_anchors[0]->GetAnchorIdx();
                    match_result.outer_inputs[peer_anchors[0]] = node->GetOutDataAnchor(anchor_index);
                }
                auto output_anchors = fusion_rule_node->GetOutputDataAnchors();
                for (auto output : output_anchors) {
                    const vector<FusionRuleAnchorPtr>& peer_anchors = output->GetPeerAnchors();
                    for (auto peer_anchor : peer_anchors) {
                        FusionRuleNodePtr peer_node = peer_anchor->GetOwnerNode();
                        if (mapping.find(peer_node) == mapping.end()) {
                            continue;
                        }
                        ge::NodePtr node = mapping[peer_node];
                        int anchor_index = peer_anchor->GetAnchorIdx();
                        auto data_anchor = node->GetInDataAnchor(anchor_index);
                        if (match_result.outer_outputs.find(peer_anchor) == match_result.outer_outputs.end()) {
                            match_result.outer_outputs[peer_anchor] = {};
                        }
                        match_result.outer_outputs[peer_anchor].emplace(data_anchor);
                    }
                }
            }
            match_result.origin_nodes = mapping;
            match_results.push_back(match_result);
        }
    }
void UpadteMappings(GraphMatchResult &match_result) {
  string pattern_name = "test";
  std::map<FusionRuleAnchorPtr, ge::AnchorPtr> origin_outer_inputs;
  for (auto &anchor_map : match_result.outer_inputs) {
    auto peer_in_anchor = anchor_map.second->GetFirstPeerAnchor();
    origin_outer_inputs.emplace(anchor_map.first, peer_in_anchor);
  }
  // update outer_outputs anchors
  std::map<FusionRuleAnchorPtr, std::set<ge::AnchorPtr>> origin_outer_outputs;
  for (auto &anchor_map : match_result.outer_outputs) {
    std::set<ge::AnchorPtr> anchor_set = anchor_map.second;
    std::set<ge::AnchorPtr> origin_anchor_set;
    for (auto &anchor : anchor_set) {
      auto peer_out_anchor = anchor->GetFirstPeerAnchor();
      origin_anchor_set.insert(peer_out_anchor);
    }

    origin_outer_outputs.insert(make_pair(anchor_map.first, origin_anchor_set));
  }
  match_result.origin_outer_inputs = origin_outer_inputs;
  match_result.origin_outer_outputs = origin_outer_outputs;
}
    /*
    �����ںϹ���2��mul+sum+mul�ںϳ�softmax_grad
     */
    vector<FusionRulePatternPtr> CreateFusionRulePatternTwo()
    {
        vector<FusionRulePatternPtr> fusion_patterns;
        FusionRulePatternPtr fusion_rule_pattern = make_shared<FusionRulePattern>();
        fusion_rule_pattern->rule_name_ = "test_fusion_two";

        // Add1 rule node
        FusionRuleNodePtr data_node2 = CreateFusionRuleNode("data2", {"Data"}, {}, {0}, {});
        FusionRuleNodePtr square_node1 = CreateFusionRuleNode("square_1", {"Square"}, {}, {0}, {});
        // ��������FusionRuleNode
        fusion_rule_pattern->input_info_.push_back(data_node2);
        fusion_rule_pattern->input_info_.push_back(square_node1);

        // mul1 rule node
        FusionRuleNodePtr mul_node1 = CreateFusionRuleNode("mul1", {"Mul"}, {0, 1}, {0}, {});
        FusionRuleNodePtr sum_node = CreateFusionRuleNode("reduceSum", {"ReduceSumD"}, {0}, {0}, {});
        FusionRuleNodePtr mul_node2 = CreateFusionRuleNode("mul2", {"Mul"}, {0, 1}, {0}, {});
        Link(data_node2, 0, mul_node1, 1);
        Link(data_node2, 0, mul_node2, 1);
        Link(square_node1, 0, mul_node1, 0);
        Link(mul_node1, 0, sum_node, 0);
        Link(sum_node, 0, mul_node2, 0);
        // �����ں�ǰFusionRuleNode
        fusion_rule_pattern->origin_rule_nodes_.insert(mul_node1);
        fusion_rule_pattern->origin_rule_nodes_.insert(sum_node);
        fusion_rule_pattern->origin_rule_nodes_.insert(mul_node2);

        // �������node
        FusionRuleNodePtr square_node2 = CreateFusionRuleNode("square_2", {"Square"}, {0}, {}, {});
        Link(mul_node2, 0, square_node2, 0);
        fusion_rule_pattern->output_info_.push_back(square_node2);

        // �����ںϺ�FusionRuleNode
        // attributes
        map<string, FusionRuleAttrValuePtr> attributes = {};

        // Fusion node
        FusionRuleNodePtr fusion_node = CreateFusionRuleNode("node4", {"SoftmaxGrad"}, {0, 1}, {0},
                                                            attributes);
        Link(square_node1, 0, fusion_node, 0);
        Link(data_node2, 0, fusion_node, 1);
        Link(fusion_node, 0, square_node2, 0);
        fusion_rule_pattern->fusion_rule_nodes_.insert(fusion_node);

        fusion_patterns.push_back(fusion_rule_pattern);

        return fusion_patterns;
    }

    /*
     pattern three
    */
   vector<FusionRulePatternPtr> CreateFusionRulePatternThree()
    {
        vector<FusionRulePatternPtr> fusion_patterns;
        FusionRulePatternPtr fusion_rule_pattern = make_shared<FusionRulePattern>();
        fusion_rule_pattern->rule_name_ = "test_fusion_two";

        // Add1 rule node
        FusionRuleNodePtr data_node2 = CreateFusionRuleNode("data2", {"Data"}, {}, {0}, {});
        FusionRuleNodePtr square_node1 = CreateFusionRuleNode("square_1", {"Square"}, {}, {0}, {});
        // ��������FusionRuleNode
        fusion_rule_pattern->input_info_.push_back(data_node2);
        fusion_rule_pattern->input_info_.push_back(square_node1);

        // mul1 rule node
        FusionRuleNodePtr mul_node1 = CreateFusionRuleNode("mul1", {"Mul"}, {0, 1}, {0}, {});
        FusionRuleNodePtr sum_node = CreateFusionRuleNode("reduceSum", {"ReduceSumD"}, {0}, {0}, {});
        FusionRuleNodePtr mul_node2 = CreateFusionRuleNode("mul2", {"Mul"}, {0, 1}, {0}, {});
        Link(data_node2, 0, mul_node1, 1);
        Link(data_node2, 0, mul_node2, 1);
        Link(square_node1, 0, mul_node1, 0);
        Link(mul_node1, 0, sum_node, 0);
        Link(sum_node, 0, mul_node2, 0);
        // �����ں�ǰFusionRuleNode
        fusion_rule_pattern->origin_rule_nodes_.insert(mul_node1);
        fusion_rule_pattern->origin_rule_nodes_.insert(sum_node);
        fusion_rule_pattern->origin_rule_nodes_.insert(mul_node2);

        // �������node
        FusionRuleNodePtr square_node2 = CreateFusionRuleNode("square_2", {"Square"}, {0}, {}, {});
        Link(mul_node2, 0, square_node2, 0);
        fusion_rule_pattern->output_info_.push_back(square_node2);

        // �����ںϺ�FusionRuleNode
        // attributes
        map<string, FusionRuleAttrValuePtr> attributes = {};

        // Fusion node
        FusionRuleNodePtr fusion_node = CreateFusionRuleNode("node4", {"SoftmaxGrad"}, {0, 1}, {0},
                                                            attributes);
        FusionRuleNodePtr fusion_node2 = CreateFusionRuleNode("node5", {"Sub"}, {0, 1}, {0},
                                                            attributes);
        Link(square_node1, 0, fusion_node, 0);
        Link(data_node2, 0, fusion_node, 1);
        Link(fusion_node, 0, fusion_node2, 0);
        Link(data_node2, 0, fusion_node2, 1);
        Link(fusion_node2, 0, square_node2, 0);
        fusion_rule_pattern->fusion_rule_nodes_.insert(fusion_node);
        fusion_rule_pattern->fusion_rule_nodes_.insert(fusion_node2);
        fusion_patterns.push_back(fusion_rule_pattern);

        return fusion_patterns;
    }
};

TEST_F(UTEST_fusion_engine_graph_replace_unittest, replace_update_specattr)
{
  ComputeGraphPtr graph = SquareSumV1();
  ComputeGraphPtr graph1 = SquareSumV1();

  ge::NodePtr sq = graph->FindNode("Square0");
  ge::NodePtr sm = graph->FindNode("Sum0");
  ge::NodePtr sq1 = graph1->FindNode("Square0");
  ge::NodePtr sm1 = graph1->FindNode("Sum0");

  FusionRuleNodeMapping origin_sub_graph;
  origin_sub_graph.insert(make_pair(nullptr, sq));
  origin_sub_graph.insert(make_pair(nullptr, sm));

  FusionRuleNodeMapping fusion_sub_graph;
  fusion_sub_graph.insert(make_pair(nullptr, sm1));
  fusion_sub_graph.insert(make_pair(nullptr, sq1));

  shared_ptr<FEOpsKernelInfoStore> ops_kernel_info_store_ptr = std::make_shared<FEOpsKernelInfoStore>(fe::AI_CORE_NAME);
  GraphReplace graph_replace(ops_kernel_info_store_ptr);

  graph_replace.UpdateSpecialAttr(origin_sub_graph, fusion_sub_graph);
  string sl;
  ge::AttrUtils::GetStr(sm1->GetOpDesc(), "_stream_label", sl);

  EXPECT_EQ("All_Cast_t", sl);
}
