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
#include "framework/common/debug/ge_log.h"
/**
 *      Input                            Input
 *        |                                |
 *      switch                           Switch
 *     /     \                          /     \
 *    A      B                         A      B
 *     \    /           ->             |      |
 *      Merge                        Cast    Cast
 *        |                            \     /
 *       Cast                           Merge
 *        |                               |
 *     NetOutput                       NetOutPut
 */
namespace fe {
using Mapping = std::map<const std::shared_ptr<ge::OpDesc>, std::vector<ge::NodePtr>, fe::CmpKey>;
class SwapMergeCastFusionTestPass : public PatternFusionBasePass {
 protected:
  vector<FusionPattern *> DefinePatterns() override;

  vector<FusionPattern *> DefineInnerPatterns() override;

  Status Fusion(ge::ComputeGraph &graph, Mapping &mapping, vector<ge::NodePtr> &new_nodes) override;

 private:
  Status VerifyNodes(const ge::NodePtr &merge_node, ge::NodePtr &cast_node, ge::NodePtr &netout_node) const;

  Status RelinkMergeNode(const ge::NodePtr &merge_node, const ge::NodePtr &cast_node, const ge::NodePtr &netout_node);

  Status AddCastNodeBeforeMergeNode(const ge::NodePtr &merge_node, ge::OpDescPtr &cast_op_desc,
                                    ge::ComputeGraph &graph);
};

static const string SWAPMERGECAST_PASS_NAME = "SwapMergeCastFusionPass";
static const string PATTERN_MERGE = "Pattern_Merge";
static const string PATTERN_CAST = "Pattern_Cast";
static const string PATTERN_RELU = "Pattern_Relu";
static const string OP_TYPE_MERGE = "Merge";
static const string OP_TYPE_CAST = "Cast";
static const string OP_TYPE_RELU = "Relu";
static const string OP_TYPE_NETOUTPUT = "NetOutput";

vector<FusionPattern *> SwapMergeCastFusionTestPass::DefinePatterns() {
  vector<FusionPattern *> patterns;

  FusionPattern *pattern = new(std::nothrow) FusionPattern("SwapMergeCastFusionPattern");

  pattern->AddOpDesc(PATTERN_MERGE, {OP_TYPE_MERGE})
      .AddOpDesc(PATTERN_CAST, {OP_TYPE_CAST})
      .SetInputs(PATTERN_CAST, {PATTERN_MERGE})
      .SetOutput(PATTERN_CAST);

  patterns.push_back(pattern);

  return patterns;
}

vector<FusionPattern *> SwapMergeCastFusionTestPass::DefineInnerPatterns() {
  vector<FusionPattern *> patterns;

  FusionPattern *pattern = new(std::nothrow) FusionPattern("SwapMergeCastFusionInnerPattern");

  pattern->AddOpDesc(PATTERN_RELU, {OP_TYPE_RELU})
      .AddOpDesc(PATTERN_CAST, {OP_TYPE_CAST})
      .AddOpDesc(PATTERN_MERGE, {OP_TYPE_MERGE})
      .SetInputs(PATTERN_CAST, {PATTERN_RELU})
      .SetInputs(PATTERN_MERGE, {PATTERN_CAST})
      .SetOutput(PATTERN_MERGE);

  patterns.push_back(pattern);
  return patterns;
}

Status SwapMergeCastFusionTestPass::Fusion(ge::ComputeGraph &graph, Mapping &mapping, vector<ge::NodePtr> &new_nodes) {
  ge::NodePtr merge_node = GetNodeFromMapping(PATTERN_MERGE, mapping);
  ge::NodePtr cast_node = GetNodeFromMapping(PATTERN_CAST, mapping);
  CheckOpSupported(merge_node);
  ge::NodePtr netout_node = nullptr;
  Status verify_status = VerifyNodes(merge_node, cast_node, netout_node);
  if (verify_status != SUCCESS) {
    return verify_status;
  }

  // unlink cast node and link merge node to netoutput node
  Status status = RelinkMergeNode(merge_node, cast_node, netout_node);
  if (status != SUCCESS) {
    return status;
  }

  // add cast node for each input data anchor of merge node
  ge::OpDescPtr cast_op_desc = cast_node->GetOpDesc();
  status = AddCastNodeBeforeMergeNode(merge_node, cast_op_desc, graph);
  if (status != SUCCESS) {
    return status;
  }

  if (graph.RemoveNode(cast_node) != ge::GRAPH_SUCCESS) {
    return FAILED;
  }

  return SUCCESS;
}

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
      GE_LOGE("Parameter[%s] must not be null.", #val); \
      return fe::PARAM_INVALID;                         \
    }                                                   \
  } while (0)


Status
SwapMergeCastFusionTestPass::AddCastNodeBeforeMergeNode(const ge::NodePtr &merge_node,
                                                        ge::OpDescPtr &cast_op_desc,
                                                        ge::ComputeGraph &graph) {
  ge::OpDescPtr merge_op_desc = merge_node->GetOpDesc();
  ge::DataType cast_out_d_type = cast_op_desc->MutableOutputDesc(0)->GetDataType();
  merge_op_desc->MutableOutputDesc(0)->SetDataType(cast_out_d_type);

  size_t input_size = merge_op_desc->GetAllInputsSize();
  for (size_t i = 0; i < input_size; i++) {
    ge::InDataAnchorPtr in_data_anchor = merge_node->GetInDataAnchor(i);
    if (in_data_anchor == nullptr || in_data_anchor->GetPeerOutAnchor() == nullptr) {
      GELOGD("InData Anchor[%zu] of merge node[%s] is not linked.", i, merge_node->GetName().c_str());
      continue;
    }

    // update data Type of each input tensor desc of merge node
    ge::GeTensorDescPtr in_data_desc = merge_op_desc->MutableInputDesc(i);
    if (in_data_desc == nullptr) {
      GELOGD("In data desc[%zu] is null.", i);
      continue;
    }
    in_data_desc->SetDataType(cast_out_d_type);

    // copy cast op desc and update the shape of input and output
    ge::OpDescPtr new_cast_op_desc = ge::OpDescUtils::CopyOpDesc(cast_op_desc);
    UT_CHECK(new_cast_op_desc == nullptr,
             GE_LOGE("[GraphOpt][SwapMrgCastFus][AddCastNd] Fail to copy op desc for cast node[%s].",
                             cast_op_desc->GetName().c_str()),
             return FAILED);

    new_cast_op_desc->SetName(cast_op_desc->GetName() + std::to_string(i));
    new_cast_op_desc->MutableInputDesc(0)->SetShape(in_data_desc->GetShape());
    new_cast_op_desc->MutableOutputDesc(0)->SetShape(in_data_desc->GetShape());

    ge::NodePtr new_cast_node = graph.AddNode(new_cast_op_desc);
    UT_CHECK(new_cast_node == nullptr,
             GE_LOGE("[GraphOpt][SwapMrgCastFus][AddCastNd] Fail to add cast node[%s] to graph.",
                             new_cast_op_desc->GetName().c_str()),
             return FAILED);

    ge::OutDataAnchorPtr out_data_anchor = in_data_anchor->GetPeerOutAnchor();
    UT_CHECK_NOTNULL(out_data_anchor);
    // unlink the indata anchor of merge node
    in_data_anchor->UnlinkAll();
    if (ge::GraphUtils::AddEdge(out_data_anchor, new_cast_node->GetInDataAnchor(0)) != ge::GRAPH_SUCCESS) {
      GE_LOGE("[GraphOpt][SwapMrgCastFus][AddCastNd] Fail to link in_data_anchor of cast node[%s].",
                      new_cast_node->GetName().c_str());
      return FAILED;
    }
    if (ge::GraphUtils::AddEdge(new_cast_node->GetOutDataAnchor(0), in_data_anchor) != ge::GRAPH_SUCCESS) {
      GE_LOGE(
          "[GraphOpt][SwapMrgCastFus][AddCastNd] Fail to link in_data_anchor[%zu] of merge node[%s]"
          " with cast node.",
          i, merge_node->GetName().c_str());
      return FAILED;
    }
  }

  return SUCCESS;
}

Status SwapMergeCastFusionTestPass::RelinkMergeNode(const ge::NodePtr &merge_node, const ge::NodePtr &cast_node,
                                                const ge::NodePtr &netout_node) {
  ge::InDataAnchorPtr netout_in_data_anchor = cast_node->GetOutDataAnchor(0)->GetPeerInDataAnchors().at(0);
  cast_node->GetInDataAnchor(0)->UnlinkAll();
  cast_node->GetOutDataAnchor(0)->UnlinkAll();

  // if cast node has in control anchors, then link them to netoutputnode
  if (cast_node->GetInControlAnchor() != nullptr) {
    if (cast_node->GetInControlAnchor()->GetPeerOutControlAnchors().size() > 0 &&
        netout_node->GetInControlAnchor() != nullptr) {
      for (ge::OutControlAnchorPtr out_control_anchor : cast_node->GetInControlAnchor()->GetPeerOutControlAnchors()) {
        if (ge::GraphUtils::AddEdge(out_control_anchor, netout_node->GetInControlAnchor()) != ge::GRAPH_SUCCESS) {
          GE_LOGE(
              "[GraphOpt][SwapMrgCastFus][RelkMrgNd] Fail to link control edge between netoutput node[%s]"
              " and peer out control anchor of cast node[%s].",
              netout_node->GetName().c_str(), cast_node->GetName().c_str());
          return FAILED;
        }
      }
    }
    cast_node->GetInControlAnchor()->UnlinkAll();
  }

  // usually cast node do not have any output control anchor
  // if cast node has output control anchors, unlink them
  if (cast_node->GetOutControlAnchor() != nullptr) {
    cast_node->GetOutControlAnchor()->UnlinkAll();
  }

  if (ge::GraphUtils::AddEdge(merge_node->GetOutDataAnchor(0), netout_in_data_anchor) != ge::GRAPH_SUCCESS) {
    GE_LOGE("[GraphOpt][SwapMrgCastFus][RelkMrgNd] Fail to link the output data anchor of merge node[%s].",
                    merge_node->GetName().c_str());
    return FAILED;
  }

  return SUCCESS;
}

Status SwapMergeCastFusionTestPass::VerifyNodes(const ge::NodePtr &merge_node,
                                            ge::NodePtr &cast_node, ge::NodePtr &netout_node) const {
  UT_CHECK(merge_node == nullptr, GE_LOGE("[GraphOpt][SwapMrgCastFus][VerifyNd] Merge node is nullptr."),
           return PARAM_INVALID);

  UT_CHECK(cast_node == nullptr, GE_LOGE("[GraphOpt][SwapMrgCastFus][VerifyNd] Cast node is nullptr."),
           return PARAM_INVALID);

  // merge node has two outputs, first output must has only one peer in anchor
  if (merge_node->GetOutDataAnchor(0)->GetPeerInDataAnchors().size() > 1) {
    GELOGD(
        "The first output anchor of Merge node[%s]"
        " has more than one peer in anchor.",
        merge_node->GetName().c_str());
    return NOT_CHANGED;
  }

  // cast node must have only one output node
  if (cast_node->GetOutDataNodesSize() != 1) {
    GELOGD("Cast node[%s] has more than one out data nodes.", cast_node->GetName().c_str());
    return NOT_CHANGED;
  }

  netout_node = cast_node->GetOutDataNodes().at(0);
  UT_CHECK_NOTNULL(netout_node);
  if (netout_node->GetType() != OP_TYPE_NETOUTPUT) {
    GELOGD("The next node of cast node[%s] is not NetOutput.", cast_node->GetName().c_str());
    return NOT_CHANGED;
  }

  return SUCCESS;
}

using namespace ge;
using namespace fe;

class UTESTGraphFusionPass2 : public testing::Test {
 protected:
  void SetUp() {
  }

  void TearDown() {

  }

 protected:
  static ComputeGraphPtr CreateSwapMergeCastGraph1() {
    ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test1");
    ge::OpDescPtr op_desc_switch = std::make_shared<OpDesc>("switch", "Switch");
    ge::OpDescPtr op_desc_relu1 = std::make_shared<OpDesc>("relu1", "Relu");
    ge::OpDescPtr op_desc_relu2 = std::make_shared<OpDesc>("relu2", "Relu");
    ge::OpDescPtr op_desc_merge = std::make_shared<OpDesc>("merge", "Merge");
    ge::OpDescPtr op_desc_cast = std::make_shared<OpDesc>("cast", "Cast");
    ge::OpDescPtr op_desc_netoutput = std::make_shared<OpDesc>("netoutput", "NetOutput");
    ge::OpDescPtr op_desc_other = std::make_shared<OpDesc>("other", "Other");

    //add descriptor
    vector<int64_t> dim_a = {8, 4, 16, 16};
    GeShape shape_a(dim_a);
    GeTensorDesc tensor_desc_a(shape_a);
    tensor_desc_a.SetFormat(FORMAT_NCHW);
    tensor_desc_a.SetOriginFormat(FORMAT_NCHW);
    tensor_desc_a.SetDataType(DT_FLOAT16);
    tensor_desc_a.SetOriginDataType(DT_FLOAT);

    vector<int64_t> dim_b = {1, 4, 64, 64};
    GeShape shape_b(dim_b);
    GeTensorDesc tensor_desc_b(shape_b);
    tensor_desc_b.SetFormat(FORMAT_NCHW);
    tensor_desc_b.SetOriginFormat(FORMAT_NCHW);
    tensor_desc_b.SetDataType(DT_FLOAT16);
    tensor_desc_b.SetOriginDataType(DT_FLOAT);

    vector<int64_t> dim_c = {8, 4, 64, 64};
    GeShape shape_c(dim_c);
    GeTensorDesc tensor_desc_c(shape_c);
    tensor_desc_c.SetFormat(FORMAT_NCHW);
    tensor_desc_c.SetOriginFormat(FORMAT_NCHW);
    tensor_desc_c.SetDataType(DT_FLOAT16);
    tensor_desc_c.SetOriginDataType(DT_FLOAT);

    GeTensorDesc tensor_desc_d(shape_c);
    tensor_desc_d.SetFormat(FORMAT_NCHW);
    tensor_desc_d.SetOriginFormat(FORMAT_NCHW);
    tensor_desc_d.SetDataType(DT_FLOAT);
    tensor_desc_d.SetOriginDataType(DT_FLOAT);

    vector<int64_t> dim_e;
    GeShape shape_e(dim_e);
    GeTensorDesc tensor_desc_e(shape_e);
    tensor_desc_e.SetFormat(FORMAT_ND);
    tensor_desc_e.SetOriginFormat(FORMAT_ND);
    tensor_desc_e.SetDataType(DT_INT32);
    tensor_desc_e.SetOriginDataType(DT_INT32);

    op_desc_switch->AddOutputDesc(tensor_desc_a);
    op_desc_switch->AddOutputDesc(tensor_desc_b);

    op_desc_relu1->AddInputDesc(tensor_desc_a);
    op_desc_relu1->AddOutputDesc(tensor_desc_a);

    op_desc_relu2->AddInputDesc(tensor_desc_b);
    op_desc_relu2->AddOutputDesc(tensor_desc_b);

    op_desc_merge->AddInputDesc(tensor_desc_a);
    op_desc_merge->AddInputDesc(tensor_desc_b);
    op_desc_merge->AddOutputDesc(tensor_desc_c);
    op_desc_merge->AddOutputDesc(tensor_desc_e);

    op_desc_other->AddInputDesc(tensor_desc_e);

    op_desc_cast->AddInputDesc(tensor_desc_c);
    op_desc_cast->AddOutputDesc(tensor_desc_d);

    op_desc_netoutput->AddInputDesc(tensor_desc_d);

    ge::NodePtr node_switch = graph->AddNode(op_desc_switch);
    ge::NodePtr node_relu1 = graph->AddNode(op_desc_relu1);
    ge::NodePtr node_relu2 = graph->AddNode(op_desc_relu2);
    ge::NodePtr node_merge = graph->AddNode(op_desc_merge);
    ge::NodePtr node_cast = graph->AddNode(op_desc_cast);
    ge::NodePtr node_netoutput = graph->AddNode(op_desc_netoutput);
    ge::NodePtr node_other = graph->AddNode(op_desc_other);

    ge::GraphUtils::AddEdge(node_switch->GetOutDataAnchor(0), node_relu1->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(node_switch->GetOutDataAnchor(1), node_relu2->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(node_relu1->GetOutDataAnchor(0), node_merge->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(node_relu2->GetOutDataAnchor(0), node_merge->GetInDataAnchor(1));
    ge::GraphUtils::AddEdge(node_merge->GetOutDataAnchor(0), node_cast->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(node_merge->GetOutDataAnchor(1), node_other->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(node_cast->GetOutDataAnchor(0), node_netoutput->GetInDataAnchor(0));

    return graph;
  }

  static ComputeGraphPtr CreateSwapMergeCastGraph2() {
    ComputeGraphPtr graph = CreateSwapMergeCastGraph1();
    ge::OpDescPtr op_desc_some = std::make_shared<OpDesc>("some_node", "Some");
    vector<int64_t> dim = {8, 4, 64, 64};
    GeShape shape(dim);
    GeTensorDesc tensor_desc(shape);
    tensor_desc.SetFormat(FORMAT_NCHW);
    tensor_desc.SetOriginFormat(FORMAT_NCHW);
    tensor_desc.SetDataType(DT_FLOAT16);
    tensor_desc.SetOriginDataType(DT_FLOAT);
    op_desc_some->AddInputDesc(tensor_desc);
    op_desc_some->AddOutputDesc(tensor_desc);

    ge::NodePtr node_some = graph->AddNode(op_desc_some);

    for (ge::NodePtr node : graph->GetDirectNode()) {
      if (node->GetType() == "Merge") {
        ge::GraphUtils::AddEdge(node->GetOutDataAnchor(0), node_some->GetInDataAnchor(0));
      }
    }
    return graph;
  }

  static ComputeGraphPtr CreateSwapMergeCastGraph3() {
    ComputeGraphPtr graph = CreateSwapMergeCastGraph1();
    ge::OpDescPtr op_desc_some = std::make_shared<OpDesc>("some_node", "Some");
    vector<int64_t> dim = {8, 4, 64, 64};
    GeShape shape(dim);
    GeTensorDesc tensor_desc(shape);
    tensor_desc.SetFormat(FORMAT_NCHW);
    tensor_desc.SetOriginFormat(FORMAT_NCHW);
    tensor_desc.SetDataType(DT_FLOAT);
    tensor_desc.SetOriginDataType(DT_FLOAT);
    op_desc_some->AddInputDesc(tensor_desc);
    op_desc_some->AddOutputDesc(tensor_desc);

    ge::NodePtr node_some = graph->AddNode(op_desc_some);

    for (ge::NodePtr node : graph->GetDirectNode()) {
      if (node->GetType() == "Cast") {
        ge::GraphUtils::AddEdge(node->GetOutDataAnchor(0), node_some->GetInDataAnchor(0));
      }
    }
    return graph;
  }

  static ComputeGraphPtr CreateSwapMergeCastGraph4() {
    ComputeGraphPtr graph = CreateSwapMergeCastGraph1();

    for (ge::NodePtr node : graph->GetDirectNode()) {
      if (node->GetType() == "NetOutput") {
        node->GetOpDesc()->SetType("NetOut");
      }
    }
    return graph;
  }

  static ComputeGraphPtr CreateSwapMergeCastGraph5() {
    ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test1");
    ge::OpDescPtr op_desc_switch = std::make_shared<OpDesc>("switch", "Switch");
    ge::OpDescPtr op_desc_relu1 = std::make_shared<OpDesc>("relu1", "Relu");
    ge::OpDescPtr op_desc_relu2 = std::make_shared<OpDesc>("relu2", "Relu");
    ge::OpDescPtr op_desc_merge = std::make_shared<OpDesc>("merge", "Merge");
    ge::OpDescPtr op_desc_cast = std::make_shared<OpDesc>("cast", "Cast");
    ge::OpDescPtr op_desc_netoutput = std::make_shared<OpDesc>("netoutput", "NetOutput");
    ge::OpDescPtr op_desc_other = std::make_shared<OpDesc>("other", "Other");

    //add descriptor
    vector<int64_t> dim_a = {8, 4, 16, 16};
    GeShape shape_a(dim_a);
    GeTensorDesc tensor_desc_a(shape_a);
    tensor_desc_a.SetFormat(FORMAT_NCHW);
    tensor_desc_a.SetOriginFormat(FORMAT_NCHW);
    tensor_desc_a.SetDataType(DT_FLOAT16);
    tensor_desc_a.SetOriginDataType(DT_FLOAT);

    vector<int64_t> dim_b = {1, 4, 64, 64};
    GeShape shape_b(dim_b);
    GeTensorDesc tensor_desc_b(shape_b);
    tensor_desc_b.SetFormat(FORMAT_NCHW);
    tensor_desc_b.SetOriginFormat(FORMAT_NCHW);
    tensor_desc_b.SetDataType(DT_FLOAT16);
    tensor_desc_b.SetOriginDataType(DT_FLOAT);

    vector<int64_t> dim_c = {8, 4, 64, 64};
    GeShape shape_c(dim_c);
    GeTensorDesc tensor_desc_c(shape_c);
    tensor_desc_c.SetFormat(FORMAT_NCHW);
    tensor_desc_c.SetOriginFormat(FORMAT_NCHW);
    tensor_desc_c.SetDataType(DT_FLOAT16);
    tensor_desc_c.SetOriginDataType(DT_FLOAT);

    GeTensorDesc tensor_desc_d(shape_c);
    tensor_desc_d.SetFormat(FORMAT_NCHW);
    tensor_desc_d.SetOriginFormat(FORMAT_NCHW);
    tensor_desc_d.SetDataType(DT_FLOAT);
    tensor_desc_d.SetOriginDataType(DT_FLOAT);

    vector<int64_t> dim_e;
    GeShape shape_e(dim_e);
    GeTensorDesc tensor_desc_e(shape_e);
    tensor_desc_e.SetFormat(FORMAT_ND);
    tensor_desc_e.SetOriginFormat(FORMAT_ND);
    tensor_desc_e.SetDataType(DT_INT32);
    tensor_desc_e.SetOriginDataType(DT_INT32);

    op_desc_switch->AddOutputDesc(tensor_desc_a);
    op_desc_switch->AddOutputDesc(tensor_desc_b);

    op_desc_relu1->AddInputDesc(tensor_desc_a);
    op_desc_relu1->AddOutputDesc(tensor_desc_a);

    op_desc_relu2->AddInputDesc(tensor_desc_b);
    op_desc_relu2->AddOutputDesc(tensor_desc_b);

    op_desc_merge->AddInputDesc(tensor_desc_a);
    op_desc_merge->AddInputDesc(tensor_desc_b);
    op_desc_merge->AddOutputDesc(tensor_desc_c);
    op_desc_merge->AddOutputDesc(tensor_desc_e);

    op_desc_other->AddInputDesc(tensor_desc_e);

    op_desc_cast->AddInputDesc(tensor_desc_c);
    op_desc_cast->AddOutputDesc(tensor_desc_d);

    op_desc_netoutput->AddInputDesc(tensor_desc_d);

    ge::NodePtr node_switch = graph->AddNode(op_desc_switch);
    ge::NodePtr node_relu1 = graph->AddNode(op_desc_relu1);
    ge::NodePtr node_relu2 = graph->AddNode(op_desc_relu2);
    ge::NodePtr node_merge = graph->AddNode(op_desc_merge);
    ge::NodePtr node_cast = graph->AddNode(op_desc_cast);
    ge::NodePtr node_netoutput = graph->AddNode(op_desc_netoutput);
    ge::NodePtr node_other = graph->AddNode(op_desc_other);

    ge::GraphUtils::AddEdge(node_switch->GetOutDataAnchor(0), node_relu1->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(node_switch->GetOutDataAnchor(1), node_relu2->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(node_relu1->GetOutDataAnchor(0), node_merge->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(node_relu2->GetOutDataAnchor(0), node_merge->GetInDataAnchor(1));
    ge::GraphUtils::AddEdge(node_merge->GetOutDataAnchor(0), node_cast->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(node_merge->GetOutDataAnchor(1), node_other->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(node_cast->GetOutDataAnchor(0), node_netoutput->GetInDataAnchor(0));

    ge::GraphUtils::AddEdge(node_relu1->GetOutControlAnchor(), node_cast->GetInControlAnchor());
    ge::GraphUtils::AddEdge(node_relu2->GetOutControlAnchor(), node_cast->GetInControlAnchor());
    return graph;
  }

  static ComputeGraphPtr CreateSwapMergeCastGraph6() {
    ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test1");
    ge::OpDescPtr op_desc_switch1 = std::make_shared<OpDesc>("switch1", "Switch");
    ge::OpDescPtr op_desc_relu1 = std::make_shared<OpDesc>("relu1", "Relu");
    ge::OpDescPtr op_desc_relu2 = std::make_shared<OpDesc>("relu2", "Relu");
    ge::OpDescPtr op_desc_merge1 = std::make_shared<OpDesc>("merge1", "Merge");
    ge::OpDescPtr op_desc_cast1 = std::make_shared<OpDesc>("cast1", "Cast");
    ge::OpDescPtr op_desc_other1 = std::make_shared<OpDesc>("other1", "Other");

    ge::OpDescPtr op_desc_switch2 = std::make_shared<OpDesc>("switch2", "Switch");
    ge::OpDescPtr op_desc_relu3 = std::make_shared<OpDesc>("relu3", "Relu");
    ge::OpDescPtr op_desc_relu4 = std::make_shared<OpDesc>("relu4", "Relu");
    ge::OpDescPtr op_desc_merge2 = std::make_shared<OpDesc>("merge2", "Merge");
    ge::OpDescPtr op_desc_cast2 = std::make_shared<OpDesc>("cast2", "Cast");
    ge::OpDescPtr op_desc_other2 = std::make_shared<OpDesc>("other2", "Other");

    ge::OpDescPtr op_desc_netoutput = std::make_shared<OpDesc>("netoutput", "NetOutput");

    //add descriptor
    vector<int64_t> dim_a = {8, 4, 16, 16};
    GeShape shape_a(dim_a);
    GeTensorDesc tensor_desc_a(shape_a);
    tensor_desc_a.SetFormat(FORMAT_NCHW);
    tensor_desc_a.SetOriginFormat(FORMAT_NCHW);
    tensor_desc_a.SetDataType(DT_FLOAT16);
    tensor_desc_a.SetOriginDataType(DT_FLOAT);

    vector<int64_t> dim_b = {1, 4, 64, 64};
    GeShape shape_b(dim_b);
    GeTensorDesc tensor_desc_b(shape_b);
    tensor_desc_b.SetFormat(FORMAT_NCHW);
    tensor_desc_b.SetOriginFormat(FORMAT_NCHW);
    tensor_desc_b.SetDataType(DT_FLOAT16);
    tensor_desc_b.SetOriginDataType(DT_FLOAT);

    vector<int64_t> dim_c = {8, 4, 64, 64};
    GeShape shape_c(dim_c);
    GeTensorDesc tensor_desc_c(shape_c);
    tensor_desc_c.SetFormat(FORMAT_NCHW);
    tensor_desc_c.SetOriginFormat(FORMAT_NCHW);
    tensor_desc_c.SetDataType(DT_FLOAT16);
    tensor_desc_c.SetOriginDataType(DT_FLOAT);

    GeTensorDesc tensor_desc_d(shape_c);
    tensor_desc_d.SetFormat(FORMAT_NCHW);
    tensor_desc_d.SetOriginFormat(FORMAT_NCHW);
    tensor_desc_d.SetDataType(DT_FLOAT);
    tensor_desc_d.SetOriginDataType(DT_FLOAT);

    vector<int64_t> dim_e;
    GeShape shape_e(dim_e);
    GeTensorDesc tensor_desc_e(shape_e);
    tensor_desc_e.SetFormat(FORMAT_ND);
    tensor_desc_e.SetOriginFormat(FORMAT_ND);
    tensor_desc_e.SetDataType(DT_INT32);
    tensor_desc_e.SetOriginDataType(DT_INT32);

    op_desc_switch1->AddOutputDesc(tensor_desc_a);
    op_desc_switch1->AddOutputDesc(tensor_desc_b);

    op_desc_switch2->AddOutputDesc(tensor_desc_a);
    op_desc_switch2->AddOutputDesc(tensor_desc_b);

    op_desc_relu1->AddInputDesc(tensor_desc_a);
    op_desc_relu1->AddOutputDesc(tensor_desc_a);
    op_desc_relu2->AddInputDesc(tensor_desc_b);
    op_desc_relu2->AddOutputDesc(tensor_desc_b);

    op_desc_relu3->AddInputDesc(tensor_desc_a);
    op_desc_relu3->AddOutputDesc(tensor_desc_a);
    op_desc_relu4->AddInputDesc(tensor_desc_b);
    op_desc_relu4->AddOutputDesc(tensor_desc_b);

    op_desc_merge1->AddInputDesc(tensor_desc_a);
    op_desc_merge1->AddInputDesc(tensor_desc_b);
    op_desc_merge1->AddOutputDesc(tensor_desc_c);
    op_desc_merge1->AddOutputDesc(tensor_desc_e);

    op_desc_merge2->AddInputDesc(tensor_desc_a);
    op_desc_merge2->AddInputDesc(tensor_desc_b);
    op_desc_merge2->AddOutputDesc(tensor_desc_c);
    op_desc_merge2->AddOutputDesc(tensor_desc_e);

    op_desc_other1->AddInputDesc(tensor_desc_e);

    op_desc_other2->AddInputDesc(tensor_desc_e);

    op_desc_cast1->AddInputDesc(tensor_desc_c);
    op_desc_cast1->AddOutputDesc(tensor_desc_d);

    op_desc_cast2->AddInputDesc(tensor_desc_c);
    op_desc_cast2->AddOutputDesc(tensor_desc_d);

    op_desc_netoutput->AddInputDesc(tensor_desc_d);
    op_desc_netoutput->AddInputDesc(tensor_desc_d);

    ge::NodePtr node_switch1 = graph->AddNode(op_desc_switch1);
    ge::NodePtr node_relu1 = graph->AddNode(op_desc_relu1);
    ge::NodePtr node_relu2 = graph->AddNode(op_desc_relu2);
    ge::NodePtr node_merge1 = graph->AddNode(op_desc_merge1);
    ge::NodePtr node_cast1 = graph->AddNode(op_desc_cast1);
    ge::NodePtr node_other1 = graph->AddNode(op_desc_other1);

    ge::NodePtr node_switch2 = graph->AddNode(op_desc_switch2);
    ge::NodePtr node_relu3 = graph->AddNode(op_desc_relu3);
    ge::NodePtr node_relu4 = graph->AddNode(op_desc_relu4);
    ge::NodePtr node_merge2 = graph->AddNode(op_desc_merge2);
    ge::NodePtr node_cast2 = graph->AddNode(op_desc_cast2);
    ge::NodePtr node_other2 = graph->AddNode(op_desc_other2);

    ge::NodePtr node_netoutput = graph->AddNode(op_desc_netoutput);

    ge::GraphUtils::AddEdge(node_switch1->GetOutDataAnchor(0), node_relu1->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(node_switch1->GetOutDataAnchor(1), node_relu2->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(node_relu1->GetOutDataAnchor(0), node_merge1->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(node_relu2->GetOutDataAnchor(0), node_merge1->GetInDataAnchor(1));
    ge::GraphUtils::AddEdge(node_merge1->GetOutDataAnchor(0), node_cast1->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(node_merge1->GetOutDataAnchor(1), node_other1->GetInDataAnchor(0));

    ge::GraphUtils::AddEdge(node_switch2->GetOutDataAnchor(0), node_relu3->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(node_switch2->GetOutDataAnchor(1), node_relu4->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(node_relu3->GetOutDataAnchor(0), node_merge2->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(node_relu4->GetOutDataAnchor(0), node_merge2->GetInDataAnchor(1));
    ge::GraphUtils::AddEdge(node_merge2->GetOutDataAnchor(0), node_cast2->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(node_merge2->GetOutDataAnchor(1), node_other2->GetInDataAnchor(0));

    ge::GraphUtils::AddEdge(node_cast1->GetOutDataAnchor(0), node_netoutput->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(node_cast2->GetOutDataAnchor(0), node_netoutput->GetInDataAnchor(1));

    return graph;
  }
};

TEST_F(UTESTGraphFusionPass2, UTESTGraphFusionPass2_1) {
  ComputeGraphPtr graph = CreateSwapMergeCastGraph1();

  SwapMergeCastFusionTestPass pass;
  Status status = fe::FAILED;


  pass.SetName("test");
  status = pass.Run(*graph);

  EXPECT_EQ(fe::SUCCESS, status);
  vector<int64_t> dim_a = {8, 4, 16, 16};
  vector<int64_t> dim_b = {1, 4, 64, 64};
  vector<int64_t> dim_c = {8, 4, 64, 64};
  for (ge::NodePtr node : graph->GetDirectNode()) {
    ge::OpDescPtr op_desc = node->GetOpDesc();
    if (op_desc->GetType() == "Merge") {
      EXPECT_EQ(op_desc->MutableInputDesc(0)->GetDataType(), DT_FLOAT);
      EXPECT_EQ(op_desc->MutableInputDesc(0)->MutableShape().GetDims(), dim_a);

      EXPECT_EQ(op_desc->MutableInputDesc(1)->GetDataType(), DT_FLOAT);
      EXPECT_EQ(op_desc->MutableInputDesc(1)->MutableShape().GetDims(), dim_b);

      EXPECT_EQ(op_desc->MutableOutputDesc(0)->GetDataType(), DT_FLOAT);
      EXPECT_EQ(op_desc->MutableOutputDesc(0)->MutableShape().GetDims(), dim_c);
    }
    if (op_desc->GetType() == "Cast") {
      EXPECT_EQ(op_desc->MutableInputDesc(0)->GetDataType(), DT_FLOAT16);
      EXPECT_EQ(op_desc->MutableOutputDesc(0)->GetDataType(), DT_FLOAT);
    }
    if (op_desc->GetName() == "relu1") {
      ge::NodePtr node_cast = node->GetOutDataAnchor(0)->GetPeerInDataAnchors().at(0)->GetOwnerNode();
      EXPECT_EQ(node_cast->GetType(), "Cast");
      ge::OpDescPtr op_desc_cast = node_cast->GetOpDesc();
      EXPECT_EQ(op_desc_cast->MutableInputDesc(0)->MutableShape().GetDims(), dim_a);
      EXPECT_EQ(op_desc_cast->MutableOutputDesc(0)->MutableShape().GetDims(), dim_a);
    }
    if (op_desc->GetName() == "relu2") {
      ge::NodePtr node_cast = node->GetOutDataAnchor(0)->GetPeerInDataAnchors().at(0)->GetOwnerNode();
      EXPECT_EQ(node_cast->GetType(), "Cast");
      ge::OpDescPtr op_desc_cast = node_cast->GetOpDesc();
      EXPECT_EQ(op_desc_cast->MutableInputDesc(0)->MutableShape().GetDims(), dim_b);
      EXPECT_EQ(op_desc_cast->MutableOutputDesc(0)->MutableShape().GetDims(), dim_b);
    }
  }
}

TEST_F(UTESTGraphFusionPass2, UTESTGraphFusionPass2_2) {
  ComputeGraphPtr graph = CreateSwapMergeCastGraph2();

  Status status = fe::FAILED;
  SwapMergeCastFusionTestPass pass;

  pass.SetName("test");
  status = pass.Run(*graph);


  EXPECT_EQ(fe::NOT_CHANGED, status);
}

TEST_F(UTESTGraphFusionPass2, UTESTGraphFusionPass2_3) {
  ComputeGraphPtr graph = CreateSwapMergeCastGraph3();

  Status status = fe::FAILED;

  SwapMergeCastFusionTestPass pass;

  pass.SetName("test");
  status = pass.Run(*graph);

  EXPECT_EQ(fe::NOT_CHANGED, status);
}

TEST_F(UTESTGraphFusionPass2, UTESTGraphFusionPass2_4) {
  ComputeGraphPtr graph = CreateSwapMergeCastGraph4();

  Status status = fe::FAILED;
  SwapMergeCastFusionTestPass pass;

  pass.SetName("test");
  status = pass.Run(*graph);
  EXPECT_EQ(fe::NOT_CHANGED, status);
}

TEST_F(UTESTGraphFusionPass2, UTESTGraphFusionPass2_5) {
  ComputeGraphPtr graph = CreateSwapMergeCastGraph5();
  Status status = fe::FAILED;

  SwapMergeCastFusionTestPass pass;

  pass.SetName("test");
  status = pass.Run(*graph);


  EXPECT_EQ(fe::SUCCESS, status);
  vector<int64_t> dim_a = {8, 4, 16, 16};
  vector<int64_t> dim_b = {1, 4, 64, 64};
  vector<int64_t> dim_c = {8, 4, 64, 64};
  for (ge::NodePtr node : graph->GetDirectNode()) {
    ge::OpDescPtr op_desc = node->GetOpDesc();
    if (op_desc->GetType() == "Merge") {
      EXPECT_EQ(op_desc->MutableInputDesc(0)->GetDataType(), DT_FLOAT);
      EXPECT_EQ(op_desc->MutableInputDesc(0)->MutableShape().GetDims(), dim_a);

      EXPECT_EQ(op_desc->MutableInputDesc(1)->GetDataType(), DT_FLOAT);
      EXPECT_EQ(op_desc->MutableInputDesc(1)->MutableShape().GetDims(), dim_b);

      EXPECT_EQ(op_desc->MutableOutputDesc(0)->GetDataType(), DT_FLOAT);
      EXPECT_EQ(op_desc->MutableOutputDesc(0)->MutableShape().GetDims(), dim_c);
    }
    if (op_desc->GetType() == "Cast") {
      EXPECT_EQ(op_desc->MutableInputDesc(0)->GetDataType(), DT_FLOAT16);
      EXPECT_EQ(op_desc->MutableOutputDesc(0)->GetDataType(), DT_FLOAT);
    }
    if (op_desc->GetName() == "relu1") {
      ge::NodePtr node_cast = node->GetOutDataAnchor(0)->GetPeerInDataAnchors().at(0)->GetOwnerNode();
      EXPECT_EQ(node_cast->GetType(), "Cast");
      ge::OpDescPtr op_desc_cast = node_cast->GetOpDesc();
      EXPECT_EQ(op_desc_cast->MutableInputDesc(0)->MutableShape().GetDims(), dim_a);
      EXPECT_EQ(op_desc_cast->MutableOutputDesc(0)->MutableShape().GetDims(), dim_a);
      EXPECT_EQ(node->GetOutControlAnchor()->GetPeerInControlAnchors().at(0)->GetOwnerNode()->GetType(), "NetOutput");
    }
    if (op_desc->GetName() == "relu2") {
      ge::NodePtr node_cast = node->GetOutDataAnchor(0)->GetPeerInDataAnchors().at(0)->GetOwnerNode();
      EXPECT_EQ(node_cast->GetType(), "Cast");
      ge::OpDescPtr op_desc_cast = node_cast->GetOpDesc();
      EXPECT_EQ(op_desc_cast->MutableInputDesc(0)->MutableShape().GetDims(), dim_b);
      EXPECT_EQ(op_desc_cast->MutableOutputDesc(0)->MutableShape().GetDims(), dim_b);
      EXPECT_EQ(node->GetOutControlAnchor()->GetPeerInControlAnchors().at(0)->GetOwnerNode()->GetType(), "NetOutput");
    }
  }
}


TEST_F(UTESTGraphFusionPass2, UTESTGraphFusionPass2_6) {
  ComputeGraphPtr graph = CreateSwapMergeCastGraph6();

  Status status = fe::FAILED;

  SwapMergeCastFusionTestPass pass;

  pass.SetName("test");
  status = pass.Run(*graph);


  EXPECT_EQ(fe::SUCCESS, status);
  vector<int64_t> dim_a = {8, 4, 16, 16};
  vector<int64_t> dim_b = {1, 4, 64, 64};
  vector<int64_t> dim_c = {8, 4, 64, 64};
  for (ge::NodePtr node : graph->GetDirectNode()) {
    ge::OpDescPtr op_desc = node->GetOpDesc();
    if (op_desc->GetType() == "Merge") {
      EXPECT_EQ(op_desc->MutableInputDesc(0)->GetDataType(), DT_FLOAT);
      EXPECT_EQ(op_desc->MutableInputDesc(0)->MutableShape().GetDims(), dim_a);

      EXPECT_EQ(op_desc->MutableInputDesc(1)->GetDataType(), DT_FLOAT);
      EXPECT_EQ(op_desc->MutableInputDesc(1)->MutableShape().GetDims(), dim_b);

      EXPECT_EQ(op_desc->MutableOutputDesc(0)->GetDataType(), DT_FLOAT);
      EXPECT_EQ(op_desc->MutableOutputDesc(0)->MutableShape().GetDims(), dim_c);

      EXPECT_EQ(node->GetOutDataNodes().at(0)->GetType(), "NetOutput");
    }
    if (op_desc->GetType() == "Cast") {
      EXPECT_EQ(op_desc->MutableInputDesc(0)->GetDataType(), DT_FLOAT16);
      EXPECT_EQ(op_desc->MutableOutputDesc(0)->GetDataType(), DT_FLOAT);
    }
    if (op_desc->GetName() == "relu1" || op_desc->GetName() == "relu3") {
      ge::NodePtr node_cast = node->GetOutDataAnchor(0)->GetPeerInDataAnchors().at(0)->GetOwnerNode();
      EXPECT_EQ(node_cast->GetType(), "Cast");
      ge::OpDescPtr op_desc_cast = node_cast->GetOpDesc();
      EXPECT_EQ(op_desc_cast->MutableInputDesc(0)->MutableShape().GetDims(), dim_a);
      EXPECT_EQ(op_desc_cast->MutableOutputDesc(0)->MutableShape().GetDims(), dim_a);
    }
    if (op_desc->GetName() == "relu2" || op_desc->GetName() == "relu4") {
      ge::NodePtr node_cast = node->GetOutDataAnchor(0)->GetPeerInDataAnchors().at(0)->GetOwnerNode();
      EXPECT_EQ(node_cast->GetType(), "Cast");
      ge::OpDescPtr op_desc_cast = node_cast->GetOpDesc();
      EXPECT_EQ(op_desc_cast->MutableInputDesc(0)->MutableShape().GetDims(), dim_b);
      EXPECT_EQ(op_desc_cast->MutableOutputDesc(0)->MutableShape().GetDims(), dim_b);
    }
  }
}

TEST_F(UTESTGraphFusionPass2, get_and_match_inner_pattern) {
  ComputeGraphPtr graph = CreateSwapMergeCastGraph6();
  Status status = fe::FAILED;
  SwapMergeCastFusionTestPass pass;
  pass.SetName("test");
  status = pass.Run(*graph);
  EXPECT_EQ(fe::SUCCESS, status);

  const auto &inner_patterns = pass.GetInnerPatterns();
  EXPECT_EQ(inner_patterns.size(), 1);

  bool inner_pattern_check_flag = false;
  for (ge::NodePtr node : graph->GetDirectNode()) {
    ge::OpDescPtr op_desc = node->GetOpDesc();
    if (op_desc->GetType() == "Merge") {
      std::shared_ptr<FusionPattern::OpDesc> output_op_desc = inner_patterns.at(0)->GetOutput();
      ASSERT_NE(output_op_desc, nullptr);
      fe::PatternFusionBasePass::Mapping mapping;
      inner_pattern_check_flag = pass.MatchFromOutput(node, output_op_desc, mapping);
      EXPECT_EQ(inner_pattern_check_flag, true);
    }
  }
}
}
