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
#define private public
#include "graph_optimizer/graph_fusion/fusion_pass_manager/builtin_pass/strided_slice_d_to_split_fusion_pass.h"
#include "graph/compute_graph.h"
#include "pass_manager.h"
#include "ops_kernel_info_store_stub.h"
#include <vector>

#define protected public
#define private public
#include "fusion_manager/fusion_manager.h"
#undef private
#undef protected

using namespace testing;
using namespace ge;
using namespace fe;

namespace fe {
class UTEST_fusion_engine_strided_slice_d_to_split_pass : public testing::Test
{
protected:
  void SetUp()
  {
    ops_kernel_info_store_ = std::make_shared<OpsKernelInfoStoreStub>();
  }
  void TearDown() {}
  std::shared_ptr<ge::OpsKernelInfoStore> ops_kernel_info_store_;

  static ge::ComputeGraphPtr CreateTestGraph1(std::vector<int64_t> dims = {2, 4, 9, 16}, Format format = FORMAT_NCHW,
      std::vector<int64_t> begins = {0, 0, 0, 0}, std::vector<int64_t> ends = {1, 4, 9, 16},
      std::vector<int64_t> strides = {1, 1, 1, 1}, int64_t new_axis_mask = 0, int64_t shrink_axis_mask = 0,
      int64_t begin_mask = 0, int64_t end_mask = 0, int64_t ellipsis_mask = 0, bool supported_flag = true) {
    ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
    OpDescPtr data = std::make_shared<OpDesc>("const", DATA);
    OpDescPtr stridedsliced = std::make_shared<OpDesc>("stridedsliced", STRIDEDSLICED);
    OpDescPtr relu = std::make_shared<OpDesc>("relu", RELU);
    OpDescPtr netoutput = std::make_shared<OpDesc>("netoutput", "NetOutput");

    // add descriptor
    GeShape shape(dims);
    GeTensorDesc tensor_desc1(shape, format, ge::DT_FLOAT);
    tensor_desc1.SetOriginFormat(format);
    tensor_desc1.SetOriginDataType(ge::DT_FLOAT);
    tensor_desc1.SetOriginShape(shape);

    data->AddOutputDesc(tensor_desc1);
    stridedsliced->AddInputDesc(tensor_desc1);

    std::vector<int64_t> out_dims = {1, 4, 9, 16};
    GeShape out_shape(out_dims);
    GeTensorDesc tensor_desc2(out_shape, format, ge::DT_FLOAT);
    tensor_desc2.SetOriginFormat(format);
    tensor_desc2.SetOriginDataType(ge::DT_FLOAT);
    tensor_desc2.SetOriginShape(out_shape);

    stridedsliced->AddOutputDesc(tensor_desc2);
    relu->AddInputDesc(tensor_desc2);
    relu->AddOutputDesc(tensor_desc2);
    netoutput->AddInputDesc(tensor_desc2);

    // create nodes
    NodePtr data_node = graph->AddNode(data);
    NodePtr strided_slice_d_node = graph->AddNode(stridedsliced);
    NodePtr relu_node = graph->AddNode(relu);
    NodePtr netoutput_node = graph->AddNode(netoutput);

    // link edge
    ge::GraphUtils::AddEdge(data_node->GetOutDataAnchor(0),
                            strided_slice_d_node->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(strided_slice_d_node->GetOutDataAnchor(0),
                            relu_node->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(relu_node->GetOutDataAnchor(0),
                            netoutput_node->GetInDataAnchor(0));

    // set attr
    (void)ge::AttrUtils::SetListInt(strided_slice_d_node->GetOpDesc(), "begin", begins);
    (void)ge::AttrUtils::SetListInt(strided_slice_d_node->GetOpDesc(), "end", ends);
    (void)ge::AttrUtils::SetListInt(strided_slice_d_node->GetOpDesc(), "strides", strides);
    (void)ge::AttrUtils::SetInt(strided_slice_d_node->GetOpDesc(), "begin_mask", begin_mask);
    (void)ge::AttrUtils::SetInt(strided_slice_d_node->GetOpDesc(), "end_mask", end_mask);
    (void)ge::AttrUtils::SetInt(strided_slice_d_node->GetOpDesc(), "ellipsis_mask", ellipsis_mask);
    (void)ge::AttrUtils::SetInt(strided_slice_d_node->GetOpDesc(), "new_axis_mask", new_axis_mask);
    (void)ge::AttrUtils::SetInt(strided_slice_d_node->GetOpDesc(), "shrink_axis_mask", shrink_axis_mask);

    if (!supported_flag) {
      (void)ge::AttrUtils::SetBool(strided_slice_d_node->GetOpDesc(), "supported_flag", supported_flag);
    }
    return graph;
  }

  static ge::ComputeGraphPtr CreateTestGraph2() {
    ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
    OpDescPtr data = std::make_shared<OpDesc>("const", DATA);
    OpDescPtr stridedsliced = std::make_shared<OpDesc>("stridedsliced", STRIDEDSLICED);
    OpDescPtr relu1 = std::make_shared<OpDesc>("relu1", RELU);
    OpDescPtr relu2 = std::make_shared<OpDesc>("relu2", RELU);
    OpDescPtr concat = std::make_shared<OpDesc>("concat", CONCATV2D);
    OpDescPtr netoutput = std::make_shared<OpDesc>("netoutput", NETOUTPUT);

    // add descriptor
    ge::GeShape shape1({2,4,9,16});
    GeTensorDesc tensor_desc1(shape1, ge::FORMAT_NCHW, ge::DT_FLOAT);
    tensor_desc1.SetOriginFormat(ge::FORMAT_NCHW);
    tensor_desc1.SetOriginDataType(ge::DT_FLOAT);
    tensor_desc1.SetOriginShape(shape1);

    data->AddOutputDesc(tensor_desc1);
    stridedsliced->AddInputDesc(tensor_desc1);
    concat->AddOutputDesc(tensor_desc1);
    netoutput->AddInputDesc(tensor_desc1);

    ge::GeShape shape2({1,4,9,16});
    GeTensorDesc tensor_desc2(shape2, ge::FORMAT_NCHW, ge::DT_FLOAT);
    tensor_desc2.SetOriginFormat(ge::FORMAT_NCHW);
    tensor_desc2.SetOriginDataType(ge::DT_FLOAT);
    tensor_desc2.SetOriginShape(shape2);

    stridedsliced->AddOutputDesc(tensor_desc2);
    relu1->AddInputDesc(tensor_desc2);
    relu1->AddOutputDesc(tensor_desc2);
    relu2->AddInputDesc(tensor_desc2);
    relu2->AddOutputDesc(tensor_desc2);
    concat->AddInputDesc(tensor_desc2);
    concat->AddInputDesc(tensor_desc2);

    // create nodes
    NodePtr data_node = graph->AddNode(data);
    NodePtr strided_slice_d_node = graph->AddNode(stridedsliced);
    NodePtr relu1_node = graph->AddNode(relu1);
    NodePtr relu2_node = graph->AddNode(relu2);
    NodePtr concat_node = graph->AddNode(concat);
    NodePtr netoutput_node = graph->AddNode(netoutput);

    // link edge
    ge::GraphUtils::AddEdge(data_node->GetOutDataAnchor(0),
                            strided_slice_d_node->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(strided_slice_d_node->GetOutDataAnchor(0),
                            relu1_node->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(strided_slice_d_node->GetOutDataAnchor(0),
                            relu2_node->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(relu1_node->GetOutDataAnchor(0),
                            concat_node->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(relu2_node->GetOutDataAnchor(0),
                            concat_node->GetInDataAnchor(1));
    ge::GraphUtils::AddEdge(concat_node->GetOutDataAnchor(0),
                            netoutput_node->GetInDataAnchor(0));

    // set attr
    std::vector<int64_t> begins = {0, 0, 0, 0};
    ge::AttrUtils::SetListInt(strided_slice_d_node->GetOpDesc(), "begin", begins);
    std::vector<int64_t> ends = {1, 4, 9, 16};
    ge::AttrUtils::SetListInt(strided_slice_d_node->GetOpDesc(), "end", ends);
    std::vector<int64_t> strides = {1, 1, 1, 1};
    ge::AttrUtils::SetListInt(strided_slice_d_node->GetOpDesc(), "strides", strides);
    return graph;
  }

  static ge::ComputeGraphPtr CreateTestGraph3(std::vector<int64_t> begins = {0, 0, 0, 0},
      std::vector<int64_t> ends = {1, 4, 9, 16}, std::vector<int64_t> strides = {1, 2, 3, 4},
      int64_t new_axis_mask = 0, int64_t shrink_axis_mask = 0,
      int64_t begin_mask = 0, int64_t end_mask = 0, int64_t ellipsis_mask = 0) {
    ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_strid_slice");
    OpDescPtr data = std::make_shared<OpDesc>("const", DATA);
    OpDescPtr stridedsliced = std::make_shared<OpDesc>("stridedsliced", "StridedSliceD");
    OpDescPtr relu = std::make_shared<OpDesc>("relu", RELU);
    OpDescPtr netoutput = std::make_shared<OpDesc>("netoutput", "NetOutput");

    // add descriptor
    ge::GeShape shape1({2,4,9,16});
    GeTensorDesc tensor_desc1(shape1, ge::FORMAT_NCHW, ge::DT_FLOAT);
    tensor_desc1.SetOriginFormat(ge::FORMAT_NCHW);
    tensor_desc1.SetOriginDataType(ge::DT_FLOAT);
    tensor_desc1.SetOriginShape(shape1);

    data->AddOutputDesc(tensor_desc1);
    stridedsliced->AddInputDesc(tensor_desc1);

    ge::GeShape shape2({1,4,9,16});
    GeTensorDesc tensor_desc2(shape2, ge::FORMAT_NCHW, ge::DT_FLOAT);
    tensor_desc2.SetOriginFormat(ge::FORMAT_NCHW);
    tensor_desc2.SetOriginDataType(ge::DT_FLOAT);
    tensor_desc2.SetOriginShape(shape2);

    stridedsliced->AddOutputDesc(tensor_desc2);
    relu->AddInputDesc(tensor_desc2);
    relu->AddOutputDesc(tensor_desc2);
    netoutput->AddInputDesc(tensor_desc2);

    // create nodes
    NodePtr data_node = graph->AddNode(data);
    NodePtr strided_slice_d_node = graph->AddNode(stridedsliced);
    NodePtr relu_node = graph->AddNode(relu);
    NodePtr netoutput_node = graph->AddNode(netoutput);

    // link edge
    ge::GraphUtils::AddEdge(data_node->GetOutDataAnchor(0),
                            strided_slice_d_node->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(strided_slice_d_node->GetOutDataAnchor(0),
                            relu_node->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(relu_node->GetOutDataAnchor(0),
                            netoutput_node->GetInDataAnchor(0));

    // set attr
    (void)ge::AttrUtils::SetListInt(strided_slice_d_node->GetOpDesc(), "begin", begins);
    (void)ge::AttrUtils::SetListInt(strided_slice_d_node->GetOpDesc(), "end", ends);
    (void)ge::AttrUtils::SetListInt(strided_slice_d_node->GetOpDesc(), "strides", strides);
    (void)ge::AttrUtils::SetInt(strided_slice_d_node->GetOpDesc(), "begin_mask", begin_mask);
    (void)ge::AttrUtils::SetInt(strided_slice_d_node->GetOpDesc(), "end_mask", end_mask);
    (void)ge::AttrUtils::SetInt(strided_slice_d_node->GetOpDesc(), "ellipsis_mask", ellipsis_mask);
    (void)ge::AttrUtils::SetInt(strided_slice_d_node->GetOpDesc(), "new_axis_mask", new_axis_mask);
    (void)ge::AttrUtils::SetInt(strided_slice_d_node->GetOpDesc(), "shrink_axis_mask", shrink_axis_mask);
    return graph;
  }

  static ge::ComputeGraphPtr CreateTestGraph4() {
    ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
    OpDescPtr data = std::make_shared<OpDesc>("const", DATA);
    OpDescPtr stridedsliced = std::make_shared<OpDesc>("stridedsliced", "StridedSliceD");
    OpDescPtr relu1 = std::make_shared<OpDesc>("relu1", RELU);
    OpDescPtr relu2 = std::make_shared<OpDesc>("relu2", RELU);
    OpDescPtr concat = std::make_shared<OpDesc>("concat", CONCATV2D);
    OpDescPtr netoutput = std::make_shared<OpDesc>("netoutput", NETOUTPUT);

    // add descriptor
    ge::GeShape shape1({2,4,9,16});
    GeTensorDesc tensor_desc1(shape1, ge::FORMAT_NCHW, ge::DT_FLOAT);
    tensor_desc1.SetOriginFormat(ge::FORMAT_NCHW);
    tensor_desc1.SetOriginDataType(ge::DT_FLOAT);
    tensor_desc1.SetOriginShape(shape1);

    data->AddOutputDesc(tensor_desc1);
    stridedsliced->AddInputDesc(tensor_desc1);
    concat->AddOutputDesc(tensor_desc1);
    netoutput->AddInputDesc(tensor_desc1);

    ge::GeShape shape2({1,4,9,16});
    GeTensorDesc tensor_desc2(shape2, ge::FORMAT_NCHW, ge::DT_FLOAT);
    tensor_desc2.SetOriginFormat(ge::FORMAT_NCHW);
    tensor_desc2.SetOriginDataType(ge::DT_FLOAT);
    tensor_desc2.SetOriginShape(shape2);

    stridedsliced->AddOutputDesc(tensor_desc2);
    relu1->AddInputDesc(tensor_desc2);
    relu1->AddOutputDesc(tensor_desc2);
    relu2->AddInputDesc(tensor_desc2);
    relu2->AddOutputDesc(tensor_desc2);
    concat->AddInputDesc(tensor_desc2);
    concat->AddInputDesc(tensor_desc2);

    // create nodes
    NodePtr data_node = graph->AddNode(data);
    NodePtr strided_slice_d_node = graph->AddNode(stridedsliced);
    NodePtr relu1_node = graph->AddNode(relu1);
    NodePtr relu2_node = graph->AddNode(relu2);
    NodePtr concat_node = graph->AddNode(concat);
    NodePtr netoutput_node = graph->AddNode(netoutput);

    // link edge
    ge::GraphUtils::AddEdge(data_node->GetOutDataAnchor(0),
                            strided_slice_d_node->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(strided_slice_d_node->GetOutDataAnchor(0),
                            relu1_node->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(strided_slice_d_node->GetOutDataAnchor(0),
                            relu2_node->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(relu1_node->GetOutDataAnchor(0),
                            concat_node->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(relu2_node->GetOutDataAnchor(0),
                            concat_node->GetInDataAnchor(1));
    ge::GraphUtils::AddEdge(concat_node->GetOutDataAnchor(0),
                            netoutput_node->GetInDataAnchor(0));

    // set attr
    std::vector<int64_t> begins = {0, 0, 0, 0};
    ge::AttrUtils::SetListInt(strided_slice_d_node->GetOpDesc(), "begin", begins);
    std::vector<int64_t> ends = {1, 4, 9, 16};
    ge::AttrUtils::SetListInt(strided_slice_d_node->GetOpDesc(), "end", ends);
    std::vector<int64_t> strides = {1, 2, 3, 4};
    ge::AttrUtils::SetListInt(strided_slice_d_node->GetOpDesc(), "strides", strides);
    return graph;
  }

  static ge::ComputeGraphPtr CreateTestGraph5(std::vector<int64_t> begins = {0, 0, 0, 0},
      std::vector<int64_t> ends = {1, 4, 9, 16}, std::vector<int64_t> strides = {1, 2, 3, 4},
      int64_t new_axis_mask = 1, int64_t shrink_axis_mask = 1,
      int64_t begin_mask = 1, int64_t end_mask = 1, int64_t ellipsis_mask = 1) {
    ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_strid_slice");
    OpDescPtr data = std::make_shared<OpDesc>("const", DATA);
    OpDescPtr stridedsliced = std::make_shared<OpDesc>("stridedsliced", "StridedSliceD");
    OpDescPtr relu = std::make_shared<OpDesc>("relu", RELU);
    OpDescPtr netoutput = std::make_shared<OpDesc>("netoutput", "NetOutput");

    // add descriptor
    ge::GeShape shape1({2,4,9,16});
    GeTensorDesc tensor_desc1(shape1, ge::FORMAT_NCHW, ge::DT_FLOAT);
    tensor_desc1.SetOriginFormat(ge::FORMAT_NCHW);
    tensor_desc1.SetOriginDataType(ge::DT_FLOAT);
    tensor_desc1.SetOriginShape(shape1);

    data->AddOutputDesc(tensor_desc1);
    stridedsliced->AddInputDesc(tensor_desc1);

    ge::GeShape shape2({1,4,9,16});
    GeTensorDesc tensor_desc2(shape2, ge::FORMAT_NCHW, ge::DT_FLOAT);
    tensor_desc2.SetOriginFormat(ge::FORMAT_NCHW);
    tensor_desc2.SetOriginDataType(ge::DT_FLOAT);
    tensor_desc2.SetOriginShape(shape2);

    stridedsliced->AddOutputDesc(tensor_desc2);
    relu->AddInputDesc(tensor_desc2);
    relu->AddOutputDesc(tensor_desc2);
    netoutput->AddInputDesc(tensor_desc2);

    // create nodes
    NodePtr data_node = graph->AddNode(data);
    NodePtr strided_slice_d_node = graph->AddNode(stridedsliced);
    NodePtr relu_node = graph->AddNode(relu);
    NodePtr netoutput_node = graph->AddNode(netoutput);

    // link edge
    ge::GraphUtils::AddEdge(data_node->GetOutDataAnchor(0),
                            strided_slice_d_node->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(strided_slice_d_node->GetOutDataAnchor(0),
                            relu_node->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(relu_node->GetOutDataAnchor(0),
                            netoutput_node->GetInDataAnchor(0));

    // set attr
    (void)ge::AttrUtils::SetListInt(strided_slice_d_node->GetOpDesc(), "begin", begins);
    (void)ge::AttrUtils::SetListInt(strided_slice_d_node->GetOpDesc(), "end", ends);
    (void)ge::AttrUtils::SetListInt(strided_slice_d_node->GetOpDesc(), "strides", strides);
    (void)ge::AttrUtils::SetInt(strided_slice_d_node->GetOpDesc(), "begin_mask", begin_mask);
    (void)ge::AttrUtils::SetInt(strided_slice_d_node->GetOpDesc(), "end_mask", end_mask);
    (void)ge::AttrUtils::SetInt(strided_slice_d_node->GetOpDesc(), "ellipsis_mask", ellipsis_mask);
    (void)ge::AttrUtils::SetInt(strided_slice_d_node->GetOpDesc(), "new_axis_mask", new_axis_mask);
    (void)ge::AttrUtils::SetInt(strided_slice_d_node->GetOpDesc(), "shrink_axis_mask", shrink_axis_mask);
    return graph;
  }

  static ge::ComputeGraphPtr CreateTestGraph6() {
    ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
    OpDescPtr relu1 = std::make_shared<OpDesc>("relu1", RELU);
    OpDescPtr relu2 = std::make_shared<OpDesc>("relu2", RELU);
    OpDescPtr concat = std::make_shared<OpDesc>("concat", CONCATV2D);
    OpDescPtr stridedsliced = std::make_shared<OpDesc>("stridedsliced", STRIDEDSLICED);
    OpDescPtr netoutput = std::make_shared<OpDesc>("netoutput", NETOUTPUT);

    // add descriptor
    ge::GeShape shape1({1,4,9,16});
    GeTensorDesc tensor_desc1(shape1, ge::FORMAT_NCHW, ge::DT_FLOAT);
    tensor_desc1.SetOriginFormat(ge::FORMAT_NCHW);
    tensor_desc1.SetOriginDataType(ge::DT_FLOAT);
    tensor_desc1.SetOriginShape(shape1);

    relu1->AddOutputDesc(tensor_desc1);
    relu2->AddOutputDesc(tensor_desc1);
    concat->AddInputDesc(tensor_desc1);
    concat->AddInputDesc(tensor_desc1);
    stridedsliced->AddOutputDesc(tensor_desc1);
    netoutput->AddInputDesc(tensor_desc1);

    ge::GeShape shape2({2,4,9,16});
    GeTensorDesc tensor_desc2(shape2, ge::FORMAT_NCHW, ge::DT_FLOAT);
    tensor_desc2.SetOriginFormat(ge::FORMAT_NCHW);
    tensor_desc2.SetOriginDataType(ge::DT_FLOAT);
    tensor_desc2.SetOriginShape(shape2);

    concat->AddOutputDesc(tensor_desc2);
    stridedsliced->AddInputDesc(tensor_desc2);

    // create nodes
    NodePtr relu1_node = graph->AddNode(relu1);
    NodePtr relu2_node = graph->AddNode(relu2);
    NodePtr concat_node = graph->AddNode(concat);
    NodePtr strided_slice_d_node = graph->AddNode(stridedsliced);
    NodePtr netoutput_node = graph->AddNode(netoutput);

    // link edge
    ge::GraphUtils::AddEdge(relu1_node->GetOutDataAnchor(0),
                            concat_node->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(relu2_node->GetOutDataAnchor(0),
                            concat_node->GetInDataAnchor(1));
    ge::GraphUtils::AddEdge(concat_node->GetOutDataAnchor(0),
                            strided_slice_d_node->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(strided_slice_d_node->GetOutDataAnchor(0),
                            netoutput_node->GetInDataAnchor(0));
    // set attr
    std::vector<int64_t> begins = {0, 0, 0, 0};
    ge::AttrUtils::SetListInt(strided_slice_d_node->GetOpDesc(), "begin", begins);
    std::vector<int64_t> ends = {1, 4, 9, 16};
    ge::AttrUtils::SetListInt(strided_slice_d_node->GetOpDesc(), "end", ends);
    std::vector<int64_t> strides = {1, 1, 1, 1};
    ge::AttrUtils::SetListInt(strided_slice_d_node->GetOpDesc(), "strides", strides);
    return graph;
  }
};

//unknown_shape
TEST_F(UTEST_fusion_engine_strided_slice_d_to_split_pass, not_changed_01)
{
  std::vector<int64_t> dims = {2, -1, 9, 16};
  ComputeGraphPtr graph = CreateTestGraph1(dims);
  StridedSliceDToSplitFusionPass pass;
  vector<GraphPass*> passes = {&pass};
  Status status = PassManager::Run(*graph, passes, ops_kernel_info_store_);
  EXPECT_EQ(status, fe::NOT_CHANGED);

  std::vector<std::string> op_type_vec = {DATA, STRIDEDSLICED, RELU, NETOUTPUT};
  size_t index = 0;
  for (auto &node : graph->GetDirectNode()) {
    EXPECT_EQ(node->GetType(), op_type_vec.at(index));
    index++;
  }
}

// new_axis_mask != 0
TEST_F(UTEST_fusion_engine_strided_slice_d_to_split_pass, not_changed_02)
{
  std::vector<int64_t> dims = {2, 4, 9, 16};
  Format format = FORMAT_NCHW;
  std::vector<int64_t> begins = {0, 0, 0, 0};
  std::vector<int64_t> ends = {1, 4, 9, 16};
  std::vector<int64_t> strides = {1, 1, 1, 1};
  int64_t new_axis_mask = 1;
  ComputeGraphPtr graph = CreateTestGraph1(dims, format, begins, ends, strides, new_axis_mask);
  StridedSliceDToSplitFusionPass pass;
  vector<GraphPass*> passes = {&pass};
  Status status = PassManager::Run(*graph, passes, ops_kernel_info_store_);
  EXPECT_EQ(status, fe::NOT_CHANGED);

  std::vector<std::string> op_type_vec = {DATA, STRIDEDSLICED, RELU, NETOUTPUT};
  size_t index = 0;
  for (auto &node : graph->GetDirectNode()) {
    EXPECT_EQ(node->GetType(), op_type_vec.at(index));
    index++;
  }
}

// shrink_axis_mask != 0
TEST_F(UTEST_fusion_engine_strided_slice_d_to_split_pass, not_changed_03)
{
  std::vector<int64_t> dims = {2, 4, 9, 16};
  Format format = FORMAT_NCHW;
  std::vector<int64_t> begins = {0, 0, 0, 0};
  std::vector<int64_t> ends = {1, 4, 9, 16};
  std::vector<int64_t> strides = {1, 1, 1, 1};
  int64_t new_axis_mask = 0;
  int64_t shrink_axis_mask = 1;
  ComputeGraphPtr graph = CreateTestGraph1(dims, format, begins, ends, strides, new_axis_mask, shrink_axis_mask);
  StridedSliceDToSplitFusionPass pass;
  vector<GraphPass*> passes = {&pass};
  Status status = PassManager::Run(*graph, passes, ops_kernel_info_store_);
  EXPECT_EQ(status, fe::NOT_CHANGED);

  std::vector<std::string> op_type_vec = {DATA, STRIDEDSLICED, RELU, NETOUTPUT};
  size_t index = 0;
  for (auto &node : graph->GetDirectNode()) {
    EXPECT_EQ(node->GetType(), op_type_vec.at(index));
    index++;
  }
}

// begin size != end size
TEST_F(UTEST_fusion_engine_strided_slice_d_to_split_pass, not_changed_04)
{
  std::vector<int64_t> dims = {2, 4, 9, 16};
  Format format = FORMAT_NCHW;
  std::vector<int64_t> begins = {0, 0, 0};
  std::vector<int64_t> ends = {1, 4, 9, 16};
  ComputeGraphPtr graph = CreateTestGraph1(dims, format, begins, ends);
  StridedSliceDToSplitFusionPass pass;
  vector<GraphPass*> passes = {&pass};
  Status status = PassManager::Run(*graph, passes, ops_kernel_info_store_);
  EXPECT_EQ(status, fe::NOT_CHANGED);

  std::vector<std::string> op_type_vec = {DATA, STRIDEDSLICED, RELU, NETOUTPUT};
  size_t index = 0;
  for (auto &node : graph->GetDirectNode()) {
    EXPECT_EQ(node->GetType(), op_type_vec.at(index));
    index++;
  }
}

// begin size > shape size
TEST_F(UTEST_fusion_engine_strided_slice_d_to_split_pass, not_changed_05)
{
  std::vector<int64_t> dims = {2, 4, 9, 16};
  Format format = FORMAT_NCHW;
  std::vector<int64_t> begins = {0, 0, 0, 0, 0};
  ComputeGraphPtr graph = CreateTestGraph1(dims, format, begins);
  StridedSliceDToSplitFusionPass pass;
  vector<GraphPass*> passes = {&pass};
  Status status = PassManager::Run(*graph, passes, ops_kernel_info_store_);
  EXPECT_EQ(status, fe::NOT_CHANGED);

  std::vector<std::string> op_type_vec = {DATA, STRIDEDSLICED, RELU, NETOUTPUT};
  size_t index = 0;
  for (auto &node : graph->GetDirectNode()) {
    EXPECT_EQ(node->GetType(), op_type_vec.at(index));
    index++;
  }
}

// stride != 1
TEST_F(UTEST_fusion_engine_strided_slice_d_to_split_pass, not_changed_06)
{
  std::vector<int64_t> dims = {2, 4, 9, 16};
  Format format = FORMAT_NCHW;
  std::vector<int64_t> begins = {0, 0, 0, 0};
  std::vector<int64_t> ends = {1, 4, 9, 16};
  std::vector<int64_t> strides = {2, 1, 1, 1};
  ComputeGraphPtr graph = CreateTestGraph1(dims, format, begins, ends, strides);
  StridedSliceDToSplitFusionPass pass;
  vector<GraphPass*> passes = {&pass};
  Status status = PassManager::Run(*graph, passes, ops_kernel_info_store_);
  EXPECT_EQ(status, fe::NOT_CHANGED);

  std::vector<std::string> op_type_vec = {DATA, STRIDEDSLICED, RELU, NETOUTPUT};
  size_t index = 0;
  for (auto &node : graph->GetDirectNode()) {
    EXPECT_EQ(node->GetType(), op_type_vec.at(index));
    index++;
  }
}

// ellipsis_mask < 0
TEST_F(UTEST_fusion_engine_strided_slice_d_to_split_pass, not_changed_07)
{
  std::vector<int64_t> dims = {2, 4, 9, 16};
  Format format = FORMAT_NCHW;
  std::vector<int64_t> begins = {0, 0, 0, 0};
  std::vector<int64_t> ends = {1, 4, 9, 16};
  std::vector<int64_t> strides = {1, 1, 1, 1};
  int64_t new_axis_mask = 0;
  int64_t shrink_axis_mask = 0;
  int64_t begin_mask = 0;
  int64_t end_mask = 0;
  int64_t ellipsis_mask = -1;
  ComputeGraphPtr graph = CreateTestGraph1(dims, format, begins, ends, strides,
                                           new_axis_mask, shrink_axis_mask, begin_mask, end_mask, ellipsis_mask);
  StridedSliceDToSplitFusionPass pass;
  vector<GraphPass*> passes = {&pass};
  Status status = PassManager::Run(*graph, passes, ops_kernel_info_store_);
  EXPECT_EQ(status, fe::NOT_CHANGED);

  std::vector<std::string> op_type_vec = {DATA, STRIDEDSLICED, RELU, NETOUTPUT};
  size_t index = 0;
  for (auto &node : graph->GetDirectNode()) {
    EXPECT_EQ(node->GetType(), op_type_vec.at(index));
    index++;
  }
}

// ellipsis_mask has multiple non-zero bits
TEST_F(UTEST_fusion_engine_strided_slice_d_to_split_pass, not_changed_08)
{
  std::vector<int64_t> dims = {2, 4, 9, 16};
  Format format = FORMAT_NCHW;
  std::vector<int64_t> begins = {0, 0, 0, 0};
  std::vector<int64_t> ends = {1, 4, 9, 16};
  std::vector<int64_t> strides = {1, 1, 1, 1};
  int64_t new_axis_mask = 0;
  int64_t shrink_axis_mask = 0;
  int64_t begin_mask = 0;
  int64_t end_mask = 0;
  int64_t ellipsis_mask = 3;
  ComputeGraphPtr graph = CreateTestGraph1(dims, format, begins, ends, strides,
                                           new_axis_mask, shrink_axis_mask, begin_mask, end_mask, ellipsis_mask);
  StridedSliceDToSplitFusionPass pass;
  vector<GraphPass*> passes = {&pass};
  Status status = PassManager::Run(*graph, passes, ops_kernel_info_store_);
  EXPECT_EQ(status, fe::NOT_CHANGED);

  std::vector<std::string> op_type_vec = {DATA, STRIDEDSLICED, RELU, NETOUTPUT};
  size_t index = 0;
  for (auto &node : graph->GetDirectNode()) {
    EXPECT_EQ(node->GetType(), op_type_vec.at(index));
    index++;
  }
}

// num_split > 63
TEST_F(UTEST_fusion_engine_strided_slice_d_to_split_pass, not_changed_09)
{
  std::vector<int64_t> dims = {2, 4, 9, 160};
  Format format = FORMAT_NCHW;
  std::vector<int64_t> begins = {0, 0, 0, 0};
  std::vector<int64_t> ends = {1, 4, 9, 1};
  ComputeGraphPtr graph = CreateTestGraph1(dims, format, begins, ends);
  FEGraphOptimizerPtr graph_optimizer_ptr = FusionManager::Instance(AI_CORE_NAME).graph_opt_;
  StridedSliceDToSplitFusionPass pass;
  vector<GraphPass*> passes = {&pass};
  Status status = PassManager::Run(*graph, passes, ops_kernel_info_store_);
  EXPECT_EQ(status, fe::NOT_CHANGED);

  std::vector<std::string> op_type_vec = {DATA, STRIDEDSLICED, RELU, NETOUTPUT};
  size_t index = 0;
  for (auto &node : graph->GetDirectNode()) {
    EXPECT_EQ(node->GetType(), op_type_vec.at(index));
    index++;
  }
}

// unaligned
TEST_F(UTEST_fusion_engine_strided_slice_d_to_split_pass, not_changed_10)
{
  std::vector<int64_t> dims = {2, 4, 9, 16};
  Format format = FORMAT_NCHW;
  std::vector<int64_t> begins = {0, 0, 0, 0};
  std::vector<int64_t> ends = {2, 4, 2, 16};
  ComputeGraphPtr graph = CreateTestGraph1(dims, format, begins, ends);
  StridedSliceDToSplitFusionPass pass;
  vector<GraphPass*> passes = {&pass};
  Status status = PassManager::Run(*graph, passes, ops_kernel_info_store_);
  EXPECT_EQ(status, fe::NOT_CHANGED);

  std::vector<std::string> op_type_vec = {DATA, STRIDEDSLICED, RELU, NETOUTPUT};
  size_t index = 0;
  for (auto &node : graph->GetDirectNode()) {
    EXPECT_EQ(node->GetType(), op_type_vec.at(index));
    index++;
  }
}

// unaligned
TEST_F(UTEST_fusion_engine_strided_slice_d_to_split_pass, not_changed_11)
{
  std::vector<int64_t> dims = {2, 4, 9, 16};
  Format format = FORMAT_NCHW;
  std::vector<int64_t> begins = {0, 0, 2, 0};
  std::vector<int64_t> ends = {2, 4, 9, 16};
  ComputeGraphPtr graph = CreateTestGraph1(dims, format, begins, ends);
  StridedSliceDToSplitFusionPass pass;
  vector<GraphPass*> passes = {&pass};
  Status status = PassManager::Run(*graph, passes, ops_kernel_info_store_);
  EXPECT_EQ(status, fe::NOT_CHANGED);

  std::vector<std::string> op_type_vec = {DATA, STRIDEDSLICED, RELU, NETOUTPUT};
  size_t index = 0;
  for (auto &node : graph->GetDirectNode()) {
    EXPECT_EQ(node->GetType(), op_type_vec.at(index));
    index++;
  }
}

// multi-dims slice
TEST_F(UTEST_fusion_engine_strided_slice_d_to_split_pass, not_changed_12)
{
  std::vector<int64_t> dims = {2, 4, 9, 16};
  Format format = FORMAT_NCHW;
  std::vector<int64_t> begins = {0, 0, 0, 0};
  std::vector<int64_t> ends = {1, 2, 3, 4};
  ComputeGraphPtr graph = CreateTestGraph1(dims, format, begins, ends);
  StridedSliceDToSplitFusionPass pass;
  vector<GraphPass*> passes = {&pass};
  Status status = PassManager::Run(*graph, passes, ops_kernel_info_store_);
  EXPECT_EQ(status, fe::NOT_CHANGED);

  std::vector<std::string> op_type_vec = {DATA, STRIDEDSLICED, RELU, NETOUTPUT};
  size_t index = 0;
  for (auto &node : graph->GetDirectNode()) {
    EXPECT_EQ(node->GetType(), op_type_vec.at(index));
    index++;
  }
}

// not support
TEST_F(UTEST_fusion_engine_strided_slice_d_to_split_pass, not_changed_13)
{
  std::vector<int64_t> dims = {2, 4, 9, 16};
  Format format = FORMAT_RESERVED;
  std::vector<int64_t> begins = {0, 0, 0, 0};
  std::vector<int64_t> ends = {1, 4, 9, 16};
  std::vector<int64_t> strides = {1, 1, 1, 1};
  int64_t new_axis_mask = 0;
  int64_t shrink_axis_mask = 0;
  int64_t begin_mask = 0;
  int64_t end_mask = 0;
  int64_t ellipsis_mask = -1;
  ComputeGraphPtr graph = CreateTestGraph1(dims, format, begins, ends, strides,
                                           new_axis_mask, shrink_axis_mask, begin_mask, end_mask, ellipsis_mask, false);
  StridedSliceDToSplitFusionPass pass;
  vector<GraphPass*> passes = {&pass};
  Status status = PassManager::Run(*graph, passes, ops_kernel_info_store_);
  EXPECT_EQ(status, fe::NOT_CHANGED);

  std::vector<std::string> op_type_vec = {DATA, STRIDEDSLICED, RELU, NETOUTPUT};
  size_t index = 0;
  for (auto &node : graph->GetDirectNode()) {
    EXPECT_EQ(node->GetType(), op_type_vec.at(index));
    index++;
  }
}

TEST_F(UTEST_fusion_engine_strided_slice_d_to_split_pass, not_changed_14)
{
  ComputeGraphPtr graph = CreateTestGraph6();
  StridedSliceDToSplitFusionPass pass;
  vector<GraphPass*> passes = {&pass};
  Status status = PassManager::Run(*graph, passes, ops_kernel_info_store_);
  EXPECT_EQ(status, fe::NOT_CHANGED);

  std::vector<std::string> op_type_vec = {RELU, RELU, CONCATV2D, STRIDEDSLICED, NETOUTPUT};
  size_t index = 0;
  for (auto &node : graph->GetDirectNode()) {
    EXPECT_EQ(node->GetType(), op_type_vec.at(index));
    index++;
  }
}

TEST_F(UTEST_fusion_engine_strided_slice_d_to_split_pass, succ_01)
{
  ComputeGraphPtr graph = CreateTestGraph1();
  StridedSliceDToSplitFusionPass pass;
  vector<GraphPass*> passes = {&pass};
  Status status = PassManager::Run(*graph, passes, ops_kernel_info_store_);
  EXPECT_EQ(status, fe::SUCCESS);

  std::vector<std::string> op_type_vec = {DATA, RELU, NETOUTPUT, SPLITD};
  size_t index = 0;
  for (auto &node : graph->GetDirectNode()) {
    EXPECT_EQ(node->GetType(), op_type_vec.at(index));
    index++;
  }
}

TEST_F(UTEST_fusion_engine_strided_slice_d_to_split_pass, succ_02)
{
  std::vector<int64_t> dims = {2, 4, 9, 16};
  Format format = FORMAT_NCHW;
  std::vector<int64_t> begins = {0, 0, 0, 0};
  std::vector<int64_t> ends = {2, 1, 9, 16};
  ComputeGraphPtr graph = CreateTestGraph1(dims, format, begins, ends);
  StridedSliceDToSplitFusionPass pass;
  vector<GraphPass*> passes = {&pass};
  Status status = PassManager::Run(*graph, passes, ops_kernel_info_store_);
  EXPECT_EQ(status, fe::SUCCESS);

  std::vector<std::string> op_type_vec = {DATA, RELU, NETOUTPUT, SPLITD};
  size_t index = 0;
  for (auto &node : graph->GetDirectNode()) {
    EXPECT_EQ(node->GetType(), op_type_vec.at(index));
    index++;
  }
}

TEST_F(UTEST_fusion_engine_strided_slice_d_to_split_pass, succ_03)
{
  std::vector<int64_t> dims = {2, 4, 9, 16};
  Format format = FORMAT_NCHW;
  std::vector<int64_t> begins = {0, 0, 0};
  std::vector<int64_t> ends = {2, 1, 9};
  std::vector<int64_t> strides = {1, 1, 1};
  ComputeGraphPtr graph = CreateTestGraph1(dims, format, begins, ends, strides);
  StridedSliceDToSplitFusionPass pass;
  vector<GraphPass*> passes = {&pass};
  Status status = PassManager::Run(*graph, passes, ops_kernel_info_store_);
  EXPECT_EQ(status, fe::SUCCESS);

  std::vector<std::string> op_type_vec = {DATA, RELU, NETOUTPUT, SPLITD};
  size_t index = 0;
  for (auto &node : graph->GetDirectNode()) {
    EXPECT_EQ(node->GetType(), op_type_vec.at(index));
    index++;
  }
}

TEST_F(UTEST_fusion_engine_strided_slice_d_to_split_pass, succ_04)
{
  std::vector<int64_t> dims = {2, 4, 9, 16};
  Format format = FORMAT_NCHW;
  std::vector<int64_t> begins = {0, 0, 0, 0};
  std::vector<int64_t> ends = {2, 1, 9, 0};
  std::vector<int64_t> strides = {1, 1, 1, 1};
  int64_t new_axis_mask = 0;
  int64_t shrink_axis_mask = 0;
  int64_t begin_mask = 0;
  int64_t end_mask = 0;
  int64_t ellipsis_mask = 1;
  ComputeGraphPtr graph = CreateTestGraph1(dims, format, begins, ends, strides,
                                           new_axis_mask, shrink_axis_mask, begin_mask, end_mask, ellipsis_mask);
  StridedSliceDToSplitFusionPass pass;
  vector<GraphPass*> passes = {&pass};
  Status status = PassManager::Run(*graph, passes, ops_kernel_info_store_);
  EXPECT_EQ(status, fe::SUCCESS);

  std::vector<std::string> op_type_vec = {DATA, RELU, NETOUTPUT, SPLITD};
  size_t index = 0;
  for (auto &node : graph->GetDirectNode()) {
    EXPECT_EQ(node->GetType(), op_type_vec.at(index));
    index++;
  }
}

TEST_F(UTEST_fusion_engine_strided_slice_d_to_split_pass, succ_05)
{
  std::vector<int64_t> dims = {2, 4, 9, 16};
  Format format = FORMAT_NCHW;
  std::vector<int64_t> begins = {0, 0, 0, 1};
  std::vector<int64_t> ends = {2, 1, 0, 0};
  std::vector<int64_t> strides = {1, 1, 1, 1};
  int64_t new_axis_mask = 0;
  int64_t shrink_axis_mask = 0;
  int64_t begin_mask = 1;
  int64_t end_mask = 3;
  ComputeGraphPtr graph = CreateTestGraph1(dims, format, begins, ends, strides,
                                           new_axis_mask, shrink_axis_mask, begin_mask, end_mask);
  StridedSliceDToSplitFusionPass pass;
  vector<GraphPass*> passes = {&pass};
  Status status = PassManager::Run(*graph, passes, ops_kernel_info_store_);
  EXPECT_EQ(status, fe::SUCCESS);

  std::vector<std::string> op_type_vec = {DATA, RELU, NETOUTPUT, SPLITD};
  size_t index = 0;
  for (auto &node : graph->GetDirectNode()) {
    EXPECT_EQ(node->GetType(), op_type_vec.at(index));
    index++;
  }
}

TEST_F(UTEST_fusion_engine_strided_slice_d_to_split_pass, succ_06)
{
  std::vector<int64_t> dims = {2, 4, 9, 16};
  Format format = FORMAT_NCHW;
  std::vector<int64_t> begins = {0, 0, 0, 0};
  std::vector<int64_t> ends = {2, 1, 9, 100};
  ComputeGraphPtr graph = CreateTestGraph1(dims, format, begins, ends);
  StridedSliceDToSplitFusionPass pass;
  vector<GraphPass*> passes = {&pass};
  Status status = PassManager::Run(*graph, passes, ops_kernel_info_store_);
  EXPECT_EQ(status, fe::SUCCESS);

  std::vector<std::string> op_type_vec = {DATA, RELU, NETOUTPUT, SPLITD};
  size_t index = 0;
  for (auto &node : graph->GetDirectNode()) {
    EXPECT_EQ(node->GetType(), op_type_vec.at(index));
    index++;
  }
}

TEST_F(UTEST_fusion_engine_strided_slice_d_to_split_pass, succ_07)
{
  std::vector<int64_t> dims = {2, 4, 9, 16};
  Format format = FORMAT_NCHW;
  std::vector<int64_t> begins = {-2, -4, 0, 0};
  std::vector<int64_t> ends = {2, -2, 9, 16};
  ComputeGraphPtr graph = CreateTestGraph1(dims, format, begins, ends);
  StridedSliceDToSplitFusionPass pass;
  vector<GraphPass*> passes = {&pass};
  Status status = PassManager::Run(*graph, passes, ops_kernel_info_store_);
  EXPECT_EQ(status, fe::SUCCESS);

  std::vector<std::string> op_type_vec = {DATA, RELU, NETOUTPUT, SPLITD};
  size_t index = 0;
  for (auto &node : graph->GetDirectNode()) {
    EXPECT_EQ(node->GetType(), op_type_vec.at(index));
    index++;
  }
}

TEST_F(UTEST_fusion_engine_strided_slice_d_to_split_pass, succ_08)
{
  ComputeGraphPtr graph = CreateTestGraph2();
  StridedSliceDToSplitFusionPass pass;
  vector<GraphPass*> passes = {&pass};
  Status status = PassManager::Run(*graph, passes, ops_kernel_info_store_);
  EXPECT_EQ(status, fe::SUCCESS);

  std::vector<std::string> op_type_vec = {DATA, RELU, RELU, CONCATV2D, NETOUTPUT, SPLITD};
  size_t index = 0;
  for (auto &node : graph->GetDirectNode()) {
    EXPECT_EQ(node->GetType(), op_type_vec.at(index));
    index++;
  }
}

TEST_F(UTEST_fusion_engine_strided_slice_d_to_split_pass, fail_01)
{
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  OpDescPtr stridedsliced = std::make_shared<OpDesc>("stridedsliced", STRIDEDSLICED);
  OpDescPtr concat = std::make_shared<OpDesc>("concat", CONCATV2D);
  std::vector<int64_t> dims = {2, 4, 9, 16};
  Format format = FORMAT_NCHW;
  GeShape shape(dims);
  GeTensorDesc tensor_desc1(shape, format, ge::DT_FLOAT);
  tensor_desc1.SetOriginFormat(format);
  tensor_desc1.SetOriginDataType(ge::DT_FLOAT);
  tensor_desc1.SetOriginShape(shape);

  std::vector<int64_t> out_dims = {1, 4, 9, 16};
  GeShape out_shape(out_dims);
  GeTensorDesc tensor_desc2(out_shape, format, ge::DT_FLOAT);
  tensor_desc2.SetOriginFormat(format);
  tensor_desc2.SetOriginDataType(ge::DT_FLOAT);
  tensor_desc2.SetOriginShape(out_shape);

  stridedsliced->AddInputDesc(tensor_desc1);
  stridedsliced->AddOutputDesc(tensor_desc2);

  concat->AddInputDesc(tensor_desc1);
  concat->AddInputDesc(tensor_desc1);
  concat->AddOutputDesc(tensor_desc2);

  // create nodes
  NodePtr concat_node = graph->AddNode(concat);
  NodePtr strided_slice_d_node = graph->AddNode(stridedsliced);
  ge::GraphUtils::AddEdge(concat_node->GetOutDataAnchor(0), strided_slice_d_node->GetInDataAnchor(0));

  StridedSliceDToSplitFusionPass pass;
  SplitInfo split_info;
  bool ret = pass.CheckCommonCondition(strided_slice_d_node, stridedsliced, split_info);
  EXPECT_EQ(ret, false);
}
}
