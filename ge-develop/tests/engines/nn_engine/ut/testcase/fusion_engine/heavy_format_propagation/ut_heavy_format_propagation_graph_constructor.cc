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
#include <memory>
#include "fe_llt_utils.h"
#include "common/util/op_info_util.h"

#define private public
#define protected public
#include "graph/utils/graph_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/attr_utils.h"
#include "adapter/common/op_store_adapter_manager.h"
#include "adapter/tbe_adapter/tbe_op_store_adapter.h"
#include "graph_optimizer/heavy_format_propagation/heavy_format_propagation.h"

#include "graph_optimizer/shape_format_transfer/trans_node_manager/trans_node_manager.h"
#include "graph/debug/ge_attr_define.h"
#include "common/configuration.h"
#include "../../../../graph_constructor/graph_constructor.h"
#include "ops_store/ops_kernel_manager.h"
using namespace std;
using namespace ge;
using namespace fe;


using TransNodeManagerPtr = std::shared_ptr<TransNodeManager>;
using HeavyFormatPropagationPtr = std::shared_ptr<HeavyFormatPropagation>;
class STEST_fusion_engine_heavy_format_distribution_graph_constructor : public testing::Test {
 protected:

  void SetUp() {
    std::map<std::string, std::string> options;
    fe_ops_kernel_info_store_ptr_ = make_shared<fe::FEOpsKernelInfoStore>();
    FEOpsStoreInfo heavy_op_info{
        6,
        "tbe-builtin",
        EN_IMPL_HW_TBE,
        GetCodeDir() + "/tests/engines/nn_engine/ut/testcase/fusion_engine/ops_kernel_store/fe_config/heavy_opinfo",
        "",
        false,
        false,
        false};

    vector<FEOpsStoreInfo> store_info;
    store_info.emplace_back(heavy_op_info);
    Configuration::Instance(fe::AI_CORE_NAME).ops_store_info_vector_ = (store_info);
    OpsKernelManager::Instance(AI_CORE_NAME).Finalize();

    fe_ops_kernel_info_store_ptr_->Initialize(options);

    reflection_builder_ptr_ = std::make_shared<ge::RefRelations>();
  }

  void TearDown() {

  }
  shared_ptr<fe::FEOpsKernelInfoStore> fe_ops_kernel_info_store_ptr_;
  RefRelationsPtr reflection_builder_ptr_;

  static void CreateThreeGraphWithL2LossAndAddN_NewMethod(ComputeGraphPtr graph) {
    /* conv2_d_back_prop_filter(Fragz)      Conv2D(NC1HWC0)
     *          |                       /
     *        a.m.(NCHW)          L2Loss (NCHW)
     *               \           /
     *                 AddN(NCHW)
     *  After distribution, the input and output of a.m. will become Fragz */
    GraphConstructor test(graph);
    test.SetInput("am1", "Conv2DBackPropFilter")
        .SetInput("AddN", "am1")
        .SetInput("AddN", "L2Loss")
        .SetInput("L2Loss", "Conv2D");

    GraphConstructor::DumpGraph(graph);
  }

  static void CreateThreeGraphWithL2LossAndAddN_NewMethod_1(ComputeGraphPtr graph) {
    /* conv2_d_back_prop_filter(Fragz)      Conv2D(NC1HWC0)
     *          |                       /
     *        a.m.(NCHW)          L2Loss (NCHW)
     *               \           /
     *                 AddN(NCHW)
     *  After distribution, the input and output of a.m. will become Fragz */
    GraphConstructor test(graph);
    test.SetInput("am1`23fdsf1_", "Conv2DBackPropFilter")
        .SetInput("am1`23fdsf1_:2", "am1")
        .SetInput("am1`23fdsf1_:3", "L2Loss")
        .SetInput("L2Loss", "Conv2D");

    GraphConstructor::DumpGraph(graph);
  }

  static Status CheckGraphAtStage1(ComputeGraphPtr graph) {
    for (auto& node : graph->GetDirectNode()) {
      if (node->GetName() == "am1") {
        EXPECT_EQ("{[], }", GraphConstructor::GetInputString(node));
      }
      if (node->GetName() == "am2_0") {
        EXPECT_EQ("{}", GraphConstructor::GetInputString(node));
      }
      if (node->GetName() == "am2") {
        EXPECT_EQ("{[am2_0], }", GraphConstructor::GetInputString(node));
      }
      if (node->GetName() == "am2_2") {
        EXPECT_EQ("{[], }", GraphConstructor::GetInputString(node));
      }
      if (node->GetName() == "am2_") {
        EXPECT_EQ("{[am2_2], }", GraphConstructor::GetInputString(node));
      }
      if (node->GetName() == "_2_2") {
        EXPECT_EQ("{[am2_2], }", GraphConstructor::GetInputString(node));
      }
    }
    return fe::SUCCESS;
  }

  static Status CheckGraphAtStage2(ComputeGraphPtr graph) {
    for (auto& node : graph->GetDirectNode()) {
      if (node->GetName() == "am1") {
        EXPECT_EQ("{[], }", GraphConstructor::GetInputString(node));
      }
      if (node->GetName() == "am2_0") {
        EXPECT_EQ("{}", GraphConstructor::GetInputString(node));
      }
      if (node->GetName() == "am2") {
        EXPECT_EQ("{[am2_0], [], [], }", GraphConstructor::GetInputString(node));
      }
      if (node->GetName() == "am2:") {
        EXPECT_EQ("{[am2], }", GraphConstructor::GetInputString(node));
      }
      if (node->GetName() == "am2_2") {
        EXPECT_EQ("{[], }", GraphConstructor::GetInputString(node));
      }
      if (node->GetName() == "am2_") {
        EXPECT_EQ("{[am2_2], }", GraphConstructor::GetInputString(node));
      }
      if (node->GetName() == "_2_2") {
        EXPECT_EQ("{[am2_2], }", GraphConstructor::GetInputString(node));
      }
      if (node->GetName() == ":a") {
        EXPECT_EQ("{[], [], [am2_2], }", GraphConstructor::GetInputString(node));
      }
    }
    return fe::SUCCESS;
  }

  static Status CheckGraphAtStage3(ComputeGraphPtr graph) {
    for (auto& node : graph->GetDirectNode()) {
      if (node->GetName() == "am2_0") {
        EXPECT_EQ("{}", GraphConstructor::GetInputString(node));
      }
      if (node->GetName() == "am2") {
        EXPECT_EQ("{[am2_0], [], [], }", GraphConstructor::GetInputString(node));
      }
      if (node->GetName() == "am2:") {
        EXPECT_EQ("{[am2], }", GraphConstructor::GetInputString(node));
      }
      if (node->GetName() == "am2_2") {
        EXPECT_EQ("{[], }", GraphConstructor::GetInputString(node));
      }
      if (node->GetName() == "am2_") {
        EXPECT_EQ("{[am2_2], }", GraphConstructor::GetInputString(node));
      }
      if (node->GetName() == "_2_2") {
        EXPECT_EQ("{[am2_2], }", GraphConstructor::GetInputString(node));
      }
      if (node->GetName() == ":a") {
        EXPECT_EQ("{[], [], [am2_2], }", GraphConstructor::GetInputString(node));
      }
      if (node->GetName() == "am1") {
        EXPECT_EQ("{[am2], [am2], [am2], [Conv2DBackPropFilter], }", GraphConstructor::GetInputString(node));
      }
      if (node->GetName() == "123") {
        EXPECT_EQ("{}", GraphConstructor::GetInputString(node));
      }
      if (node->GetName() == "am1:0_0") {
        EXPECT_EQ("{[123], }", GraphConstructor::GetInputString(node));
      }
      if (node->GetName() == "am1:_0") {
        EXPECT_EQ("{[123], }", GraphConstructor::GetInputString(node));
      }
      if (node->GetName() == "am1_0") {
        EXPECT_EQ("{[123], }", GraphConstructor::GetInputString(node));
      }
      if (node->GetName() == "o_conv2DBackPropFilter_1") {
        EXPECT_EQ("{[123], }", GraphConstructor::GetInputString(node));
      }
    }
    return fe::SUCCESS;
  }

  static void CreateThreeGraphWithL2LossAndAddN_NewMethod_BasicTest(ComputeGraphPtr graph) {
    /* Only Contains underline and colon is missing */
    GraphConstructor test(graph);
    test.SetInput("am1_", "Conv2DBackPropFilter_a") // node name is am1 and illegal
        .SetInput("am2_2", "am2_2")  // self loop exists, illegal
        .SetInput("am2__", "am2_2") // node name is am2_ and am2_2
        .SetInput("am2", "am2_0") // node name is am2 and am2_0, they are different nodes
        .SetInput("_2", "am2_2") // node name is "" (illegal) and am2_2
        .SetInput("_2_2", "am2_2"); // node name is "_2_2", type is _2_ (legal) and am2_2
    CheckGraphAtStage1(graph);
    GraphConstructor::DumpGraph(graph);

    /* Only Contains colon */
    test.SetInput("am1:", "Conv2DBackPropFilter:a") // node name is am1 and illegal
        .SetInput("am2:2", "am2:1")  // self loop exists, illegal
        .SetInput("am2::", "am2:2") // node name is am2: and am2(output anchor 2)
        .SetInput("am2", "am2:0") // node name is am2 and am2, they are same node
        .SetInput(":2", "am2:2") // node name is "" (illegal) and am2
        .SetInput(":a:2", "am2_2"); // node name is ":a", type is :a (legal) and am2_2
    CheckGraphAtStage2(graph);
    GraphConstructor::DumpGraph(graph);

    /* Contains underline and colon */
    test.SetInput("am1_:", "am2:"); //index is 0 and 0 for node am1 and am2
    test.SetInput("am1_:", "am2:"); //index is 1 and 1 for node am1 and am2
    test.SetInput("am1_:", "am2:"); //index is 2 and 3 for node am1 and am2, the index 2 of node am2 is occupied
    test.SetInput("am1_:", "Conv2DBackPropFilter_:"); // node name is am1 tensor 0, conv2_d_back_prop_filter tensor 0
    test.SetInput("am1_0:", "o_conv2DBackPropFilter_1:0"); // node name is am1 tensor0, and -_conv2DBackPropFilter_1 tensor 0
    test.SetInput("am1:0_0", "123:0"); // am1:0_0 is legal, the optype is am1:0
    test.SetInput("am1:_0", "123:0"); // am1:_0 is legal, the optype is am1:
    test.SetInput("am1_0:3.0", "123:0"); // am1:0_3.0 is illegal, the char after colon should be integer
    test.SetInput("am1_0:0", "123:0"); // first optype is am1, remove the edge between
    test.SetInput("o_conv2DBackPropFilter_1", "123"); // node name is am1 tensor0, and -_conv2DBackPropFilter_1 tensor 0
    CheckGraphAtStage3(graph);
    GraphConstructor::DumpGraph(graph);
  }
  static void GetNode(const ge::ComputeGraphPtr& graph,
               const string& name, ge::NodePtr& node_out) {
    for (auto& node : graph->GetDirectNode()) {
      if (node->GetName() == "conv2d") {
        node_out = node;
      }
    }
  }
  static void CreateThreeGraphWithL2LossAndAddN_1_NEW(ComputeGraphPtr graph) {
    ge::GeShape original_shape = GeShape({3, 4, 5, 6});
    GraphConstructor test(graph, "", ge::FORMAT_NCHW, ge::DT_FLOAT16, original_shape);

    /* If the peer out anchor is empty, we can set the input and output 0's
     * tensor desc without :index*/
    test.AddOpDesc("conv2d", "Conv2D")
        .AddOpDesc("addn", "AddN", 0, 1)
        .AddOpDesc("l2loss", "L2Loss")
        .AddOpDesc("addn", "Conv2D")
        .AddOpDesc("am", "ApplyMomentum")
        .AddOpDesc("conv2dback", "Conv2DBackpropInput")
        .SetInput("conv2d", "", ge::FORMAT_FRACTAL_Z) // set input 0 of conv2d as FORMAT_FRACTAL_Z
        .SetInput("conv2d", "", ge::FORMAT_NC1HWC0); // set input 0 of conv2d as FORMAT_NC1HWC0

    ge::NodePtr conv_node;
    GetNode(graph, "conv2d", conv_node);
    ASSERT_NE(conv_node, nullptr);
    ge::OpDescPtr conv_opdesc = conv_node->GetOpDesc();
    EXPECT_EQ(1, conv_opdesc->GetInputsSize());
    auto input_desc0 = conv_opdesc->MutableInputDesc(0);
    EXPECT_EQ(input_desc0->GetFormat(), FORMAT_NC1HWC0);

    test.SetInput("conv2d:1", "", ge::FORMAT_NC1HWC0);

    test.SetInput("conv2d", "", ge::FORMAT_NC1HWC0) // set input 0 of conv2d as FORMAT_FRACTAL_Z
        .SetInput("conv2d", "", ge::FORMAT_FRACTAL_Z); // set input 0 of conv2d as FORMAT_NC1HWC0

    auto input_desc1 = conv_opdesc->MutableInputDesc(1);
    input_desc0 = conv_opdesc->MutableInputDesc(0);
    EXPECT_EQ(input_desc0->GetFormat(), FORMAT_FRACTAL_Z);
    EXPECT_EQ(input_desc1->GetFormat(), FORMAT_NC1HWC0);

    test.SetInput("conv2d:0", "", ge::FORMAT_NC1HWC0) // set input 0 of conv2d as FORMAT_FRACTAL_Z
        .SetInput("conv2d:1", "", ge::FORMAT_FRACTAL_Z); // set input 1 of conv2d as FORMAT_NC1HWC0

    input_desc1 = conv_opdesc->MutableInputDesc(1);
    input_desc0 = conv_opdesc->MutableInputDesc(0);
    EXPECT_EQ(input_desc0->GetFormat(), FORMAT_NC1HWC0);
    EXPECT_EQ(input_desc1->GetFormat(), FORMAT_FRACTAL_Z);

    /* conv2_d_back_prop_filter(Fragz)      Conv2D(NC1HWC0)
   *          |                       /
   *        a.m.(NCHW)          L2Loss (NCHW)
   *               \           /
   *                 AddN(NCHW)
   *  After distribution, the input and output of a.m. will become Fragz */
    vector<int64_t> dims_l2_loss_out = {};
    test.SetInput("l2loss", ge::FORMAT_NCHW, "conv2d", ge::FORMAT_NC1HWC0)
        .SetInput("addn:1", "l2loss", dims_l2_loss_out)
        .SetInput("addn:0", "am")

        .SetInput("conv2dback", "", ge::FORMAT_FRACTAL_Z)
        .SetInput("conv2dback:1", "", ge::FORMAT_NC1HWC0)
        .SetInput("am", ge::FORMAT_NCHW, "conv2dback", ge::FORMAT_FRACTAL_Z)
        .SetInput("am:4", "")
        .SetInput("conv2d", "", ge::FORMAT_NC1HWC0);
  }

  static void CreateThreeGraphWithL2LossAndAddN_1(ComputeGraphPtr graph) {
    /* conv2_d_back_prop_filter(Fragz)      Conv2D(NC1HWC0)
     *          |                       /
     *        a.m.(NCHW)          L2Loss (NCHW)
     *               \           /
     *                 AddN(NCHW)
     *  After distribution, the input and output of a.m. will become Fragz */
    OpDescPtr conv_o_p = std::make_shared<OpDesc>("conv2d", "Conv2D");
    GeTensorDesc conv_tensor_desc(GeShape({3, 1, 5, 6, 16}), ge::FORMAT_NC1HWC0, ge::DT_FLOAT16);
    conv_tensor_desc.SetOriginShape(GeShape({3, 4, 5, 6}));
    conv_tensor_desc.SetOriginFormat(ge::FORMAT_NCHW);
    GeTensorDesc conv_tensor_desc_weight(GeShape({30, 1, 16, 16}), ge::FORMAT_FRACTAL_Z, ge::DT_FLOAT16);
    conv_tensor_desc_weight.SetOriginShape(GeShape({3, 4, 5, 6}));
    conv_tensor_desc_weight.SetOriginFormat(ge::FORMAT_NCHW);
    conv_o_p->AddInputDesc(conv_tensor_desc);
    conv_o_p->AddInputDesc(conv_tensor_desc_weight);
    conv_o_p->AddOutputDesc(conv_tensor_desc);
    auto conv_node = graph->AddNode(conv_o_p);
    ge::AttrUtils::SetInt(conv_o_p, FE_IMPLY_TYPE, 6);

    OpDescPtr l2_loss_op = std::make_shared<OpDesc>("l2loss", "L2Loss");
    GeTensorDesc l2_loss_tensor_desc(GeShape({3, 4, 5, 6}), ge::FORMAT_NCHW, ge::DT_FLOAT16);
    l2_loss_tensor_desc.SetOriginShape(GeShape({3, 4, 5, 6}));
    l2_loss_tensor_desc.SetOriginFormat(ge::FORMAT_NCHW);
    GeTensorDesc l2_loss_out_tensor_desc(GeShape(), ge::FORMAT_NCHW, ge::DT_FLOAT16);
    l2_loss_out_tensor_desc.SetOriginShape(GeShape());
    l2_loss_out_tensor_desc.SetOriginFormat(ge::FORMAT_NCHW);
    l2_loss_op->AddInputDesc(l2_loss_tensor_desc);
    l2_loss_op->AddOutputDesc(l2_loss_out_tensor_desc);
    auto l2loss_node = graph->AddNode(l2_loss_op);
    ge::AttrUtils::SetInt(l2_loss_op, FE_IMPLY_TYPE, 6);

    OpDescPtr add_no_p = std::make_shared<OpDesc>("addn", "AddN");
    add_no_p->AddInputDesc(l2_loss_tensor_desc);
    add_no_p->AddInputDesc(l2_loss_out_tensor_desc);
    add_no_p->AddOutputDesc(l2_loss_tensor_desc);
    auto addn_Node = graph->AddNode(add_no_p);
    ge::AttrUtils::SetInt(add_no_p, FE_IMPLY_TYPE, 6);

    OpDescPtr apply_momentum_op = std::make_shared<OpDesc>("am", "ApplyMomentum");
    GeTensorDesc am_tensor_desc(GeShape({3, 4, 5, 6}), ge::FORMAT_NCHW, ge::DT_FLOAT16);
    am_tensor_desc.SetOriginShape(GeShape({3, 4, 5, 6}));
    am_tensor_desc.SetOriginFormat(ge::FORMAT_NCHW);
    apply_momentum_op->AddOutputDesc(am_tensor_desc);
    for (uint32_t i = 0;  i < 5; i++) {
      apply_momentum_op->AddInputDesc(am_tensor_desc);
    }
    auto am_node = graph->AddNode(apply_momentum_op);
    ge::AttrUtils::SetInt(apply_momentum_op, FE_IMPLY_TYPE, 6);

    OpDescPtr conv_back_o_p = std::make_shared<OpDesc>("conv2dback", "Conv2DBackpropInput");
    conv_back_o_p->AddInputDesc(conv_tensor_desc_weight);
    conv_back_o_p->AddInputDesc(conv_tensor_desc);
    conv_back_o_p->AddOutputDesc(conv_tensor_desc_weight);
    auto conv_back_node = graph->AddNode(conv_back_o_p);
    ge::AttrUtils::SetInt(conv_back_o_p, FE_IMPLY_TYPE, 6);

    GraphUtils::AddEdge(conv_node->GetOutDataAnchor(0), l2loss_node->GetInDataAnchor(0));
    GraphUtils::AddEdge(l2loss_node->GetOutDataAnchor(0), addn_Node->GetInDataAnchor(1));
    GraphUtils::AddEdge(am_node->GetOutDataAnchor(0), addn_Node->GetInDataAnchor(0));
    GraphUtils::AddEdge(conv_back_node->GetOutDataAnchor(0), am_node->GetInDataAnchor(0));
  }

  static void Conv2D_ReduceSumD(ComputeGraphPtr graph) {
    /*      Conv2D(NC1HWC0)
     *          |
     *       ReduceSumD (NCHW)
     *          |
     *       ReduceSumD(NCHW)
     *  After distribution, the input and output of a.m. will become Fragz */
    OpDescPtr conv_o_p = std::make_shared<OpDesc>("conv2d", "Conv2D");
    GeTensorDesc conv_tensor_desc(GeShape({3, 1, 5, 6, 16}), ge::FORMAT_NC1HWC0, ge::DT_FLOAT16);
    conv_tensor_desc.SetOriginShape(GeShape({3, 4, 5, 6}));
    conv_tensor_desc.SetOriginFormat(ge::FORMAT_NCHW);
    GeTensorDesc conv_tensor_desc_weight(GeShape({30, 1, 16, 16}), ge::FORMAT_FRACTAL_Z, ge::DT_FLOAT16);
    conv_tensor_desc_weight.SetOriginShape(GeShape({3, 4, 5, 6}));
    conv_tensor_desc_weight.SetOriginFormat(ge::FORMAT_NCHW);
    conv_o_p->AddInputDesc(conv_tensor_desc);
    conv_o_p->AddInputDesc(conv_tensor_desc_weight);
    conv_o_p->AddOutputDesc(conv_tensor_desc);
    auto conv_node = graph->AddNode(conv_o_p);
    ge::AttrUtils::SetInt(conv_o_p, FE_IMPLY_TYPE, 6);

    OpDescPtr reduce_sum_d = std::make_shared<OpDesc>("reducesumd", "ReduceSumD");
    GeTensorDesc reduce_in_tensor_desc(GeShape({3, 4, 5, 6}), ge::FORMAT_NCHW, ge::DT_FLOAT16);
    reduce_in_tensor_desc.SetOriginShape(GeShape({3, 4, 5, 6}));
    reduce_in_tensor_desc.SetOriginFormat(ge::FORMAT_NCHW);
    GeTensorDesc reduce_out_tensor_desc(GeShape({3, 4, 5, 6}), ge::FORMAT_NCHW, ge::DT_FLOAT16);
    reduce_out_tensor_desc.SetOriginShape(GeShape({3, 4, 5, 6}));
    reduce_out_tensor_desc.SetOriginFormat(ge::FORMAT_NCHW);
    reduce_sum_d->AddInputDesc(reduce_in_tensor_desc);
    reduce_sum_d->AddOutputDesc(reduce_out_tensor_desc);
    auto reduce_node = graph->AddNode(reduce_sum_d);
    ge::AttrUtils::SetInt(reduce_sum_d, FE_IMPLY_TYPE, 6);

    GraphUtils::AddEdge(conv_node->GetOutDataAnchor(0), reduce_node->GetInDataAnchor(0));
  }

};



TEST_F(STEST_fusion_engine_heavy_format_distribution_graph_constructor, test1) {

  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  CreateThreeGraphWithL2LossAndAddN_NewMethod(graph);
  EXPECT_EQ(graph->GetName(), "test");

}

TEST_F(STEST_fusion_engine_heavy_format_distribution_graph_constructor, test2) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  CreateThreeGraphWithL2LossAndAddN_NewMethod_BasicTest(graph);
  EXPECT_EQ(graph->GetName(), "test");

}

TEST_F(STEST_fusion_engine_heavy_format_distribution_graph_constructor, test3) {
  ComputeGraphPtr graph1 = std::make_shared<ComputeGraph>("test1");
  ComputeGraphPtr graph2 = std::make_shared<ComputeGraph>("test2");
  CreateThreeGraphWithL2LossAndAddN_1(graph1);

  CreateThreeGraphWithL2LossAndAddN_1_NEW(graph2);
  GraphConstructor::DumpGraph(graph1);
  FE_LOGI("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++");
  GraphConstructor::DumpGraph(graph2);

  EXPECT_EQ(GraphConstructor::CompareGraph(graph1, graph2), true);
}

TEST_F(STEST_fusion_engine_heavy_format_distribution_graph_constructor,
       distribute_to_reduce) {
  ComputeGraphPtr graph1 = std::make_shared<ComputeGraph>("test1");
  Conv2D_ReduceSumD(graph1);
  HeavyFormatPropagationPtr HeavyFormatPropagator = std::make_shared<HeavyFormatPropagation>(AI_CORE_NAME, reflection_builder_ptr_);
  HeavyFormatPropagator->Initialize();
  Status ret = HeavyFormatPropagator->PropagateHeavyFormat(*(graph1.get()));
  for(auto node : graph1->GetDirectNode()) {
    if (node->GetType() == "ReduceSumD") {
      auto opdesc = node->GetOpDesc();
      {
        auto input =opdesc->GetInputDesc(0);
        EXPECT_EQ(ge::GetPrimaryFormat(input.GetFormat()), ge::FORMAT_NC1HWC0);
        EXPECT_EQ(input.GetDataType(), ge::DT_FLOAT16);
      }
      {
        auto output =opdesc->GetOutputDesc(0);
        EXPECT_EQ(output.GetFormat(), ge::FORMAT_NCHW);
        EXPECT_EQ(output.GetDataType(), ge::DT_FLOAT16);
      }
    }
  }
}

//TEST_F(STEST_fusion_engine_heavy_format_distribution_graph_constructor,
//       stop_distributing_from_non5d_shape_of_format_ndc1hwc0) {
//   /*        Cosh(NDHWC, {3,4,5,6} 4D)
//    *          |
//    *       Conv3D (NDC1HWC0, original format NDHWC, original shape {3,4,5,6,1})
//    *          |
//    *       Cosh(NDHWC, {3,4,5,6} 4D)
//    *  After distribution, the input and output of Cosh will not become
//    *  NDC1HWC0 because the shape size of Cosh is not 5*/
//  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test2");
//  ge::GeShape original_shape = GeShape({3, 4, 5, 6, 1});
//  vector<int64_t> cosh_dims = {3, 4, 5, 6 };
//  GraphConstructor test(graph, "", ge::FORMAT_NDHWC, ge::DT_FLOAT16, original_shape);
//  test.SetInput("Cosh_0", "", {3, 4, 5, 6 })
//
//      .SetInput("Conv3D:0", "Cosh_0", {3, 4, 5, 6 }, SOURCE)
//      .SetInput("Conv3D:1", "");
//
//  test.SetInput("Cosh_0", "Conv3D", {3, 4, 5, 6 }, DESTINATION);
//  test.Judge(fe_ops_kernel_info_store_ptr_);
//  HeavyFormatPropagationPtr HeavyFormatPropagator = std::make_shared<HeavyFormatPropagation>(AI_CORE_NAME, reflection_builder_ptr_);
//  HeavyFormatPropagator->Initialize();
//  Status ret = HeavyFormatPropagator->PropagateHeavyFormat(*(graph.get()));
//
//
//  for(auto node : graph->GetDirectNode()) {
//    if (node->GetType() == "Cosh_0") {
//      auto opdesc = node->GetOpDesc();
//      {
//        auto input =opdesc->GetInputDesc(0);
//        EXPECT_EQ(input.GetFormat(), ge::FORMAT_NDHWC);
//        EXPECT_EQ(input.GetDataType(), ge::DT_FLOAT16);
//        EXPECT_EQ(input.GetShape().GetDims(), cosh_dims);
//      }
//      {
//        auto output =opdesc->GetOutputDesc(0);
//        EXPECT_EQ(output.GetFormat(), ge::FORMAT_NDHWC);
//        EXPECT_EQ(output.GetDataType(), ge::DT_FLOAT16);
//        EXPECT_EQ(output.GetShape().GetDims(), cosh_dims);
//      }
//    }
//
//    if (node->GetType() == "Cosh_1") {
//      auto opdesc = node->GetOpDesc();
//      {
//        auto input =opdesc->GetInputDesc(0);
//        EXPECT_EQ(input.GetFormat(), ge::FORMAT_NDHWC);
//        EXPECT_EQ(input.GetDataType(), ge::DT_FLOAT16);
//        EXPECT_EQ(input.GetShape().GetDims(), cosh_dims);
//      }
//      {
//        auto output =opdesc->GetOutputDesc(0);
//        EXPECT_EQ(output.GetFormat(), ge::FORMAT_NDHWC);
//        EXPECT_EQ(output.GetDataType(), ge::DT_FLOAT16);
//        EXPECT_EQ(output.GetShape().GetDims(), cosh_dims);
//      }
//    }
//
//    if (node->GetType() == "Conv2D") {
//      auto opdesc = node->GetOpDesc();
//      {
//        auto input =opdesc->GetInputDesc(0);
//        EXPECT_EQ(input.GetFormat(), ge::FORMAT_NDHWC);
//        EXPECT_EQ(input.GetDataType(), ge::DT_FLOAT16);
//      }
//      {
//        auto output =opdesc->GetOutputDesc(0);
//        EXPECT_EQ(output.GetFormat(), ge::FORMAT_NDHWC);
//        EXPECT_EQ(output.GetDataType(), ge::DT_FLOAT16);
//      }
//    }
//  }
//}
//
//TEST_F(STEST_fusion_engine_heavy_format_distribution_graph_constructor,
//       keep_distributing_from_5d_shape_of_format_ndc1hwc0) {
//  /*        Cosh(NDHWC, {3,4,5,6,7} 5D)
//   *          |
//   *       Conv3D (NDC1HWC0, original format NDHWC, original shape {3,4,5,6,1})
//   *          |
//   *       Cosh(NDHWC, {3,4,5,6,7} 5D)
//   *  After distribution, the input and output of Cosh will not become
//   *  NDC1HWC0 because the shape size of Cosh is not 5*/
//  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test2");
//  ge::GeShape original_shape = GeShape({3, 4, 5, 6, 1});
//  vector<int64_t> cosh_dims = {3,4,5,6,7};
//  vector<int64_t> cosh_dims6_h_d = {3,4,1,5,6,16};
//  GraphConstructor test(graph, "", ge::FORMAT_NDHWC, ge::DT_FLOAT16, original_shape);
//  test.SetInput("Cosh_0", "", cosh_dims)
//
//      .SetInput("Conv3D:0", "Cosh_0", cosh_dims, SOURCE)
//      .SetInput("Conv3D:1", "");
//
//  test.SetInput("Cosh_0", "Conv3D", cosh_dims, DESTINATION);
//  test.Judge(fe_ops_kernel_info_store_ptr_);
//
//  HeavyFormatPropagationPtr HeavyFormatPropagator = std::make_shared<HeavyFormatPropagation>(AI_CORE_NAME, reflection_builder_ptr_);
//  HeavyFormatPropagator->Initialize();
//  Status ret = HeavyFormatPropagator->PropagateHeavyFormat(*(graph.get()));
//
//  for(auto node : graph->GetDirectNode()) {
//    if (node->GetType() == "Cosh_0") {
//      auto opdesc = node->GetOpDesc();
//      {
//        auto input =opdesc->GetInputDesc(0);
//        EXPECT_EQ(input.GetFormat(), ge::FORMAT_NDC1HWC0);
//        EXPECT_EQ(input.GetDataType(), ge::DT_FLOAT16);
//        EXPECT_EQ(input.GetShape().GetDims(), cosh_dims6_h_d);
//      }
//      {
//        auto output =opdesc->GetOutputDesc(0);
//        EXPECT_EQ(output.GetFormat(), ge::FORMAT_NDC1HWC0);
//        EXPECT_EQ(output.GetDataType(), ge::DT_FLOAT16);
//        EXPECT_EQ(output.GetShape().GetDims(), cosh_dims6_h_d);
//      }
//    }
//
//    if (node->GetType() == "Cosh_1") {
//      auto opdesc = node->GetOpDesc();
//      {
//        auto input =opdesc->GetInputDesc(0);
//        EXPECT_EQ(input.GetFormat(), ge::FORMAT_NDC1HWC0);
//        EXPECT_EQ(input.GetDataType(), ge::DT_FLOAT16);
//        EXPECT_EQ(input.GetShape().GetDims(), cosh_dims6_h_d);
//      }
//      {
//        auto output =opdesc->GetOutputDesc(0);
//        EXPECT_EQ(output.GetFormat(), ge::FORMAT_NDC1HWC0);
//        EXPECT_EQ(output.GetDataType(), ge::DT_FLOAT16);
//        EXPECT_EQ(output.GetShape().GetDims(), cosh_dims6_h_d);
//      }
//    }
//
//    if (node->GetType() == "Conv3D") {
//      auto opdesc = node->GetOpDesc();
//      {
//        auto input =opdesc->GetInputDesc(0);
//        EXPECT_EQ(input.GetFormat(), ge::FORMAT_NDC1HWC0);
//        EXPECT_EQ(input.GetDataType(), ge::DT_FLOAT16);
//      }
//      {
//        auto output =opdesc->GetOutputDesc(0);
//        EXPECT_EQ(output.GetFormat(), ge::FORMAT_NDC1HWC0);
//        EXPECT_EQ(output.GetDataType(), ge::DT_FLOAT16);
//      }
//    }
//  }
//}
