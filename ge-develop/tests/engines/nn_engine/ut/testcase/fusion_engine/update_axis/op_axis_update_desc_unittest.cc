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

#include "common/util/op_info_util.h"
#include "common/fe_inner_attr_define.h"
#include "fe_llt_utils.h"
#define private public
#define protected public
#include "graph/utils/graph_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/attr_utils.h"
#include "adapter/common/op_store_adapter_manager.h"
#include "adapter/tbe_adapter/tbe_op_store_adapter.h"
#include "graph/debug/ge_attr_define.h"
#include "common/configuration.h"
#include "ops_store/ops_kernel_manager.h"
#include "graph_optimizer/op_axis_update/op_axis_update_desc.h"

using namespace std;
using namespace ge;
using namespace fe;

using OpAxisUpdateDescPtr = std::shared_ptr<OpAxisUpdateDesc>;
class UTEST_fusion_engine_update_axis : public testing::Test {
protected:

  void SetUp() {

    FEOpsStoreInfo reduce_op_info{
            6,
            "tbe-builtin",
            EN_IMPL_HW_TBE,
            GetCodeDir() + "/tests/engines/nn_engine/ut/testcase/fusion_engine/format_selector/fe_config/tbe_dynamic_opinfo",
            "",
            false,
            false,
            false};

    vector<FEOpsStoreInfo> store_info;
    store_info.emplace_back(reduce_op_info);
    Configuration::Instance(fe::AI_CORE_NAME).ops_store_info_vector_ = (store_info);
    OpsKernelManager::Instance(AI_CORE_NAME).Finalize();
    OpsKernelManager::Instance(AI_CORE_NAME).Initialize();
  }

  void TearDown() {

  }


protected:

  static void CreateReduceNodesGraph(ComputeGraphPtr graph) {
    /*   Data         Const
     *   |     /                \
     *  conv (Fz)         a.m. (NCHW)  a.m.3,4,5,6(NCHW)  a.m.10(NCHW)
     *                          |         / / / /     \     /
     *                         a.m.2 (NCHW)          d.w.conv1(6d)
     *                        /   |   \                          \
     *                    a.m.7  a.m.8 a.m.9    a.m.12(NCHW)    a.m.11(NCHW)
     *                                  \      /
     *                                d.w.reduce(sd)
     *                                    |
     *                                  a.m.13(NCHW) */
    OpDescPtr const_op = std::make_shared<OpDesc>("const", "Const");
    GeTensorDesc const_tensor_desc(GeShape({3, 4, 5, 6}), ge::FORMAT_NCHW, ge::DT_FLOAT16);
    const_tensor_desc.SetOriginShape(GeShape({3, 4, 5, 6}));
    const_tensor_desc.SetOriginFormat(ge::FORMAT_NCHW);
    const_op->AddOutputDesc(const_tensor_desc);
    const_op->AddInputDesc(const_tensor_desc);
    auto const_node = graph->AddNode(const_op);

    OpDescPtr data_op = std::make_shared<OpDesc>("data", "Data");
    GeTensorDesc data_tensor_desc(GeShape({3, 4, 5, 6}), ge::FORMAT_NCHW, ge::DT_FLOAT16);
    data_tensor_desc.SetOriginShape(GeShape({3, 4, 5, 6}));
    data_tensor_desc.SetOriginFormat(ge::FORMAT_NCHW);
    data_op->AddOutputDesc(data_tensor_desc);
    data_op->AddInputDesc(data_tensor_desc);
    auto data_node = graph->AddNode(data_op);

    OpDescPtr reduce_o_p = std::make_shared<OpDesc>("reduce1", "ReduceOp");
    GeTensorDesc conv_tensor_desc(GeShape({3, 1, 5, 6, 16}), ge::FORMAT_NC1HWC0, ge::DT_FLOAT16);
    conv_tensor_desc.SetOriginShape(GeShape({3, 4, 5, 6}));
    conv_tensor_desc.SetOriginFormat(ge::FORMAT_NCHW);
    reduce_o_p->AddInputDesc(conv_tensor_desc);
    reduce_o_p->AddOutputDesc(conv_tensor_desc);
    auto reduce_node = graph->AddNode(reduce_o_p);
    ge::AttrUtils::SetInt(reduce_o_p, FE_IMPLY_TYPE, 6);
    ge::AttrUtils::SetBool(reduce_o_p, KEEP_DIMS, true);
    ge::AttrUtils::SetListInt(reduce_o_p, AXES_ATTR_NAME, {-1, -2, -3});

    OpDescPtr reduce_o_p2 = std::make_shared<OpDesc>("reduce2", "ReduceOp");
    GeTensorDesc conv_tensor_desc2(GeShape({3, 1, 5, 6, 16, 16}), ge::FORMAT_C1HWNCoC0, ge::DT_FLOAT16);
    conv_tensor_desc2.SetOriginShape(GeShape({3, 4, 5, 6}));
    conv_tensor_desc2.SetOriginFormat(ge::FORMAT_NHWC);
    reduce_o_p2->AddInputDesc(conv_tensor_desc2);
    reduce_o_p2->AddOutputDesc(conv_tensor_desc2);
    auto reduce_node2 = graph->AddNode(reduce_o_p2);
    ge::AttrUtils::SetInt(reduce_o_p2, FE_IMPLY_TYPE, 6);
    ge::AttrUtils::SetBool(reduce_o_p2, KEEP_DIMS, true);
    ge::AttrUtils::SetListInt(reduce_o_p2, AXES_ATTR_NAME, {-2, -3});

    OpDescPtr reduce_o_p3 = std::make_shared<OpDesc>("reduce3", "ReduceOp");
    int64_t group = 2;
    GeTensorDesc conv_tensor_desc3(GeShape({24, 1, 16, 16}), static_cast<ge::Format>(ge::GetFormatFromSub(ge::FORMAT_FRACTAL_Z, group)), ge::DT_FLOAT16);
    conv_tensor_desc3.SetOriginShape(GeShape({3, 4, 16, 16}));
    conv_tensor_desc3.SetOriginFormat(ge::FORMAT_HWCN);
    reduce_o_p3->AddInputDesc(conv_tensor_desc3);
    reduce_o_p3->AddOutputDesc(conv_tensor_desc3);
    auto reduce_node3 = graph->AddNode(reduce_o_p3);
    ge::AttrUtils::SetInt(reduce_o_p3, FE_IMPLY_TYPE, 6);
    ge::AttrUtils::SetBool(reduce_o_p3, KEEP_DIMS, true);
    ge::AttrUtils::SetListInt(reduce_o_p3, AXES_ATTR_NAME, {0, 1});

    OpDescPtr reduce_o_p4 = std::make_shared<OpDesc>("reduce4", "ReduceOp");
    GeTensorDesc conv_tensor_desc4(GeShape({3, 1, 5, 6, 16}), ge::FORMAT_FRACTAL_NZ, ge::DT_FLOAT16);
    conv_tensor_desc4.SetOriginShape(GeShape({3, 4, 5, 6}));
    conv_tensor_desc4.SetOriginFormat(ge::FORMAT_CHWN);
    reduce_o_p4->AddInputDesc(conv_tensor_desc4);
    reduce_o_p4->AddOutputDesc(conv_tensor_desc4);
    auto reduce_node4 = graph->AddNode(reduce_o_p4);
    ge::AttrUtils::SetInt(reduce_o_p4, FE_IMPLY_TYPE, 6);
    ge::AttrUtils::SetBool(reduce_o_p4, KEEP_DIMS, true);
    ge::AttrUtils::SetListInt(reduce_o_p4, AXES_ATTR_NAME, {-2, -3});

    OpDescPtr reduce_o_p5 = std::make_shared<OpDesc>("reduce5", "ReduceOp");
    GeTensorDesc conv_tensor_desc5(GeShape({48, 1, 16, 16}), static_cast<ge::Format>(ge::GetFormatFromSub(ge::FORMAT_FRACTAL_Z_3D, group)), ge::DT_FLOAT16);
    conv_tensor_desc5.SetOriginShape(GeShape({2, 3, 4, 16, 16}));
    conv_tensor_desc5.SetOriginFormat(ge::FORMAT_DHWCN);
    reduce_o_p5->AddInputDesc(conv_tensor_desc5);
    reduce_o_p5->AddOutputDesc(conv_tensor_desc5);
    auto reduce_node5 = graph->AddNode(reduce_o_p5);
    ge::AttrUtils::SetInt(reduce_o_p5, FE_IMPLY_TYPE, 6);
    ge::AttrUtils::SetBool(reduce_o_p5, KEEP_DIMS, true);
    ge::AttrUtils::SetListInt(reduce_o_p5, AXES_ATTR_NAME, {0, 1, 2});

    GraphUtils::AddEdge(data_node->GetOutDataAnchor(0), reduce_node->GetInDataAnchor(0));
    GraphUtils::AddEdge(data_node->GetOutDataAnchor(0), reduce_node2->GetInDataAnchor(0));
    GraphUtils::AddEdge(data_node->GetOutDataAnchor(0), reduce_node3->GetInDataAnchor(0));
    GraphUtils::AddEdge(data_node->GetOutDataAnchor(0), reduce_node4->GetInDataAnchor(0));
    GraphUtils::AddEdge(data_node->GetOutDataAnchor(0), reduce_node5->GetInDataAnchor(0));
  }

  static void CreateReduceNodesGraph2(ComputeGraphPtr graph) {
    OpDescPtr data_op = std::make_shared<OpDesc>("data", "Data");
    GeTensorDesc data_tensor_desc(GeShape({3, 4, 5, 6}), ge::FORMAT_NCHW, ge::DT_FLOAT16);
    data_tensor_desc.SetOriginShape(GeShape({3, 4, 5, 6}));
    data_tensor_desc.SetOriginFormat(ge::FORMAT_NCHW);
    data_op->AddOutputDesc(data_tensor_desc);
    data_op->AddInputDesc(data_tensor_desc);
    auto data_node = graph->AddNode(data_op);

    OpDescPtr reduce_o_p = std::make_shared<OpDesc>("reduce1", "ReduceOp");
    GeTensorDesc conv_tensor_desc(GeShape({3, 4, 1, 5, 6, 16}), ge::FORMAT_NDC1HWC0, ge::DT_FLOAT16);
    conv_tensor_desc.SetOriginShape(GeShape({3, 4, 5, 6, 7}));
    conv_tensor_desc.SetOriginFormat(ge::FORMAT_NDHWC);
    reduce_o_p->AddInputDesc(conv_tensor_desc);
    reduce_o_p->AddOutputDesc(conv_tensor_desc);
    auto reduce_node = graph->AddNode(reduce_o_p);
    ge::AttrUtils::SetInt(reduce_o_p, FE_IMPLY_TYPE, 6);
    ge::AttrUtils::SetBool(reduce_o_p, KEEP_DIMS, true);
    ge::AttrUtils::SetListInt(reduce_o_p, AXES_ATTR_NAME, {-1, -2, -3});

    GraphUtils::AddEdge(data_node->GetOutDataAnchor(0), reduce_node->GetInDataAnchor(0));
  }

  static void CreateReduceNodesGraph3(ComputeGraphPtr graph) {
    OpDescPtr data_op = std::make_shared<OpDesc>("data", "Data");
    GeTensorDesc data_tensor_desc(GeShape({3, 4, 5, 6}), ge::FORMAT_NCHW, ge::DT_FLOAT16);
    data_tensor_desc.SetOriginShape(GeShape({3, 4, 5, 6}));
    data_tensor_desc.SetOriginFormat(ge::FORMAT_NCHW);
    data_op->AddOutputDesc(data_tensor_desc);
    data_op->AddInputDesc(data_tensor_desc);
    auto data_node = graph->AddNode(data_op);

    OpDescPtr reduce_o_p = std::make_shared<OpDesc>("reduce1", "ReduceOp");
    GeTensorDesc conv_tensor_desc(GeShape({3, 4, 1, 5, 6, 16}), ge::FORMAT_NDC1HWC0, ge::DT_FLOAT16);
    conv_tensor_desc.SetOriginShape(GeShape({3, 4, 5, 6, 7}));
    conv_tensor_desc.SetOriginFormat(ge::FORMAT_NDHWC);
    reduce_o_p->AddInputDesc(conv_tensor_desc);
    reduce_o_p->AddOutputDesc(conv_tensor_desc);
    auto reduce_node = graph->AddNode(reduce_o_p);
    ge::AttrUtils::SetInt(reduce_o_p, FE_IMPLY_TYPE, 6);
    ge::AttrUtils::SetBool(reduce_o_p, KEEP_DIMS, true);
    ge::AttrUtils::SetListInt(reduce_o_p, AXES_ATTR_NAME, {-4, -5});

    GraphUtils::AddEdge(data_node->GetOutDataAnchor(0), reduce_node->GetInDataAnchor(0));
  }

  static void CreateReduceNodesGraph4NCDHW(ComputeGraphPtr graph) {
    OpDescPtr data_op = std::make_shared<OpDesc>("data", "Data");
    GeTensorDesc data_tensor_desc(GeShape({3, 4, 5, 6, 7}), ge::FORMAT_NCDHW, ge::DT_FLOAT16);
    data_tensor_desc.SetOriginShape(GeShape({3, 4, 5, 6, 7}));
    data_tensor_desc.SetOriginFormat(ge::FORMAT_NCDHW);
    data_op->AddOutputDesc(data_tensor_desc);
    data_op->AddInputDesc(data_tensor_desc);
    auto data_node = graph->AddNode(data_op);

    OpDescPtr reduce_o_p = std::make_shared<OpDesc>("reduce1", "ReduceOp");
    GeTensorDesc conv_tensor_desc(GeShape({3, 4, 1, 5, 6, 16}), ge::FORMAT_NDC1HWC0, ge::DT_FLOAT16);
    conv_tensor_desc.SetOriginShape(GeShape({3, 4, 5, 6, 7}));
    conv_tensor_desc.SetOriginFormat(ge::FORMAT_NCDHW);
    reduce_o_p->AddInputDesc(conv_tensor_desc);
    reduce_o_p->AddOutputDesc(conv_tensor_desc);
    auto reduce_node = graph->AddNode(reduce_o_p);
    ge::AttrUtils::SetInt(reduce_o_p, FE_IMPLY_TYPE, 6);
    ge::AttrUtils::SetBool(reduce_o_p, KEEP_DIMS, true);
    ge::AttrUtils::SetListInt(reduce_o_p, AXES_ATTR_NAME, {-4, -5});

    GraphUtils::AddEdge(data_node->GetOutDataAnchor(0), reduce_node->GetInDataAnchor(0));
  }

  static void CreateGraphForNz(ComputeGraphPtr graph) {
      OpDescPtr reduce_o_p = std::make_shared<OpDesc>("reduce1", "ReduceOp");
      ge::AttrUtils::SetInt(reduce_o_p, FE_IMPLY_TYPE, 6);
      ge::AttrUtils::SetBool(reduce_o_p, KEEP_DIMS, true);
      ge::AttrUtils::SetListInt(reduce_o_p, AXES_ATTR_NAME, {0, -1, -2});

      GeTensorDesc input_tensor_desc(GeShape({3, 4, 1, 1, 16, 16}), ge::FORMAT_FRACTAL_NZ, ge::DT_FLOAT16);
      input_tensor_desc.SetOriginShape(GeShape({3, 4, 16, 16}));
      input_tensor_desc.SetOriginFormat(ge::FORMAT_ND);

      GeTensorDesc output_tensor_desc(GeShape({1, 4, 1, 1}), ge::FORMAT_FRACTAL_NZ, ge::DT_FLOAT16);
      output_tensor_desc.SetOriginShape(GeShape({3, 4, 1, 1}));
      output_tensor_desc.SetOriginFormat(ge::FORMAT_ND);
      reduce_o_p->AddInputDesc(input_tensor_desc);
      reduce_o_p->AddOutputDesc(output_tensor_desc);
      graph->AddNode(reduce_o_p);
  }

  static void CreateNoNeedUpdateGraph(ComputeGraphPtr graph) {
    OpDescPtr reduce_o_p = std::make_shared<OpDesc>("reduce1", "ReduceOp");
    ge::AttrUtils::SetInt(reduce_o_p, FE_IMPLY_TYPE, 6);
    ge::AttrUtils::SetBool(reduce_o_p, KEEP_DIMS, true);
    ge::AttrUtils::SetListInt(reduce_o_p, AXES_ATTR_NAME, {0, -1, -2});

    GeTensorDesc input_tensor_desc(GeShape({3, 4, 1, 1, 16, 16}), ge::FORMAT_FRACTAL_NZ, ge::DT_FLOAT16);
    input_tensor_desc.SetOriginShape(GeShape({3, 4, 16, 16}));
    input_tensor_desc.SetOriginFormat(ge::FORMAT_FRACTAL_NZ);

    GeTensorDesc output_tensor_desc(GeShape({1, 4, 1, 1}), ge::FORMAT_FRACTAL_NZ, ge::DT_FLOAT16);
    output_tensor_desc.SetOriginShape(GeShape({3, 4, 1, 1}));
    output_tensor_desc.SetOriginFormat(ge::FORMAT_FRACTAL_NZ);
    reduce_o_p->AddInputDesc(input_tensor_desc);
    reduce_o_p->AddOutputDesc(output_tensor_desc);
    graph->AddNode(reduce_o_p);
  }
};

TEST_F(UTEST_fusion_engine_update_axis, update_axis_success) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  CreateReduceNodesGraph(graph);

  OpAxisUpdateDescPtr op_axis_update_desc_ptr = std::make_shared<OpAxisUpdateDesc>(AI_CORE_NAME);
  Status ret = op_axis_update_desc_ptr->UpdateAxis(*(graph.get()));
  for (auto node : graph->GetDirectNode()) {
    auto op_desc = node->GetOpDesc();
    if (node->GetName() == "reduce3") {
      vector<int64_t> expect_shape{2, 3, 4, 1, 16, 16};
      EXPECT_EQ(expect_shape, op_desc->GetInputDesc(0).GetShape().GetDims());
      vector<int64_t> expect_axis{1, 2};
      vector<int64_t> axis_vec;
      ge::AttrUtils::GetListInt(op_desc, AXES_ATTR_NAME, axis_vec);
      EXPECT_EQ(expect_axis, axis_vec);
    }
    if (node->GetName() == "reduce5") {
      vector<int64_t> expect_shape{2, 2, 3, 4, 1, 16, 16};
      EXPECT_EQ(expect_shape, op_desc->GetInputDesc(0).GetShape().GetDims());
      vector<int64_t> expect_axis{1, 2, 3};
      vector<int64_t> axis_vec;
      ge::AttrUtils::GetListInt(op_desc, AXES_ATTR_NAME, axis_vec);
      EXPECT_EQ(expect_axis, axis_vec);
    }
  }
  EXPECT_EQ(fe::SUCCESS, ret);
}


TEST_F(UTEST_fusion_engine_update_axis, update_axis_success_6_h_d)
{
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  CreateReduceNodesGraph2(graph);

  OpAxisUpdateDescPtr op_axis_update_desc_ptr = std::make_shared<OpAxisUpdateDesc>(AI_CORE_NAME);

  Status ret = op_axis_update_desc_ptr->UpdateAxis(*(graph.get()));
  EXPECT_EQ(fe::SUCCESS, ret);
}


TEST_F(UTEST_fusion_engine_update_axis, update_axis_success_6_h_d_2)
{
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  CreateReduceNodesGraph3(graph);

  OpAxisUpdateDescPtr op_axis_update_desc_ptr = std::make_shared<OpAxisUpdateDesc>(AI_CORE_NAME);

  Status ret = op_axis_update_desc_ptr->UpdateAxis(*(graph.get()));
  EXPECT_EQ(fe::SUCCESS, ret);
}

TEST_F(UTEST_fusion_engine_update_axis, update_axis_success_6_h_d_3)
{
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  CreateReduceNodesGraph4NCDHW(graph);
  OpAxisUpdateDescPtr op_axis_update_desc_ptr = std::make_shared<OpAxisUpdateDesc>(AI_CORE_NAME);
  Status ret = op_axis_update_desc_ptr->UpdateAxis(*(graph.get()));
  EXPECT_EQ(fe::SUCCESS, ret);
  for (auto node : graph->GetDirectNode()) {
    if (node->GetType() == "ReduceOp") {
      std::vector<int64_t> axis_new_value;
      (void)ge::AttrUtils::GetListInt(node->GetOpDesc(), AXES_ATTR_NAME, axis_new_value);
      EXPECT_EQ(axis_new_value.at(0), 2);
      EXPECT_EQ(axis_new_value.at(1), 5);
      EXPECT_EQ(axis_new_value.at(2), 0);
    }
  }
}

TEST_F(UTEST_fusion_engine_update_axis, nz_success) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  CreateGraphForNz(graph);

  OpAxisUpdateDescPtr op_axis_update_desc_ptr = std::make_shared<OpAxisUpdateDesc>(AI_CORE_NAME);
  Status ret = op_axis_update_desc_ptr->UpdateAxis(*(graph.get()));
  EXPECT_EQ(fe::SUCCESS, ret);
  for (auto node : graph->GetDirectNode()) {
    if (node->GetType() == "ReduceOp") {
      std::vector<int64_t> result;
      (void)ge::AttrUtils::GetListInt(node->GetOpDesc(), AXES_ATTR_NAME, result);
      EXPECT_EQ(result.at(0), 0);
      EXPECT_EQ(result.at(1), 2);
      EXPECT_EQ(result.at(2), 5);
      EXPECT_EQ(result.at(3), 3);
      EXPECT_EQ(result.at(4), 4);
    }
  }
}

TEST_F(UTEST_fusion_engine_update_axis, format_notchanged) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  CreateNoNeedUpdateGraph(graph);

  OpAxisUpdateDescPtr op_axis_update_desc_ptr = std::make_shared<OpAxisUpdateDesc>(AI_CORE_NAME);
  Status ret = op_axis_update_desc_ptr->UpdateAxis(*(graph.get()));
  EXPECT_EQ(fe::SUCCESS, ret);
  for (auto node : graph->GetDirectNode()) {
    if (node->GetType() == "ReduceOp") {
      std::vector<int64_t> result;
      (void)ge::AttrUtils::GetListInt(node->GetOpDesc(), AXES_ATTR_NAME, result);
      EXPECT_EQ(result.at(0), 0);
      EXPECT_EQ(result.at(1), -1);
      EXPECT_EQ(result.at(2), -2);
    }
  }
}
