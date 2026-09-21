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

#include "macro_utils/dt_public_scope.h"
#include "graph/passes/standard_optimize/constant_folding/dimension_adjust_pass.h"

#include "common/debug/log.h"
#include "common/debug/memory_dumper.h"
#include "common/ge_inner_error_codes.h"
#include "common/op/ge_op_utils.h"
#include "common/types.h"
#include "graph/types.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "host_kernels/kernel.h"
#include "host_kernels/kernel_factory.h"
#include "macro_utils/dt_public_unscope.h"

using namespace std;
using namespace testing;

namespace ge {
class TestExpandDimKernel : public Kernel {
 public:
  Status Compute(const NodePtr &node_ptr) const override {
    return SUCCESS;
  }
};
REGISTER_COMPUTE_NODE_KERNEL(EXPANDDIMS, TestExpandDimKernel);
class TestExpandDimKernelNotChange : public Kernel {
 public:
  Status Compute(const NodePtr &node_ptr) const override {
    return NOT_CHANGED;
  }
};

class UtestGraphPassesDimensionAdjustPass : public testing::Test {
 protected:
  void SetUp() {}

  void TearDown() {
    KernelFactory::Instance().creator_map_.clear();
  }
};

namespace ut {
class GraphBuilder {
 public:
  explicit GraphBuilder(const std::string &name) { graph_ = std::make_shared<ComputeGraph>(name); }
  NodePtr AddNode(const std::string &name, const std::string &type, int in_cnt, int out_cnt,
                  Format format = FORMAT_NCHW, DataType data_type = DT_FLOAT,
                  std::vector<int64_t> shape = {1, 1, 224, 224});
  NodePtr AddNode(const std::string &name, const std::string &type,
                  std::initializer_list<std::string> input_names,
                  std::initializer_list<std::string> output_names,
                  Format format = FORMAT_NCHW, DataType data_type = DT_FLOAT,
                  std::vector<int64_t> shape = {1, 1, 224, 224});
  void AddDataEdge(const NodePtr &src_node, int src_idx, const NodePtr &dst_node, int dst_idx);
  void AddControlEdge(const NodePtr &src_node, const NodePtr &dst_node);
  ComputeGraphPtr GetGraph() {
    graph_->TopologicalSorting();
    return graph_;
  }

 private:
  ComputeGraphPtr graph_;
};
}  // namespace ut

namespace {
  const char* AddNNo = "AddNNo";
  const char* ShapeYes = "ShapeYes";

  /*
  *
  *      netoutput1
  *         |
  *       shapeYes1
  *        |
  *      addnNo1
  */
  ComputeGraphPtr BuildGraph1() {
    auto builder = ut::GraphBuilder("test");
    auto addnNo1 = builder.AddNode("addnNo1", AddNNo, 2, 1);
    auto shapeYes1 = builder.AddNode("shapeYes1", ShapeYes, 1, 1);
    auto netoutput1 = builder.AddNode("netoutput1", NETOUTPUT, 1, 0);

    builder.AddDataEdge(addnNo1, 0, shapeYes1, 0);
    builder.AddDataEdge(shapeYes1, 0, netoutput1, 0);
    return builder.GetGraph();
  }
}

TEST_F(UtestGraphPassesDimensionAdjustPass, succ) {
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("default");

  ge::OpDescPtr data_op_desc = std::make_shared<ge::OpDesc>("data", CONSTANTOP);
  int64_t dims_size = 1;
  vector<int64_t> data_vec = {1, 2, 3};
  for_each(data_vec.begin(), data_vec.end(), [&](int64_t &data) { dims_size *= data; });
  vector<int32_t> data_value_vec(dims_size, 1);
  GeTensorDesc data_tensor_desc(GeShape(data_vec), FORMAT_NCHW, DT_INT32);
  GeTensorPtr data_tensor = std::make_shared<GeTensor>(data_tensor_desc, (uint8_t *)data_value_vec.data(),
                                                  data_value_vec.size() * sizeof(int32_t));
  OpDescUtils::SetWeights(data_op_desc, data_tensor);
  data_op_desc->AddOutputDesc(data_tensor_desc);
  NodePtr data_node = graph->AddNode(data_op_desc);
  data_node->Init();

  // add dim node
  ge::OpDescPtr dim_op_desc = std::make_shared<ge::OpDesc>("dim", CONSTANTOP);
  vector<int32_t> dim_value_vec = {0};
  GeTensorDesc dim_tensor_desc(ge::GeShape(), FORMAT_NCHW, DT_INT32);
  GeTensorPtr dim_tensor =
      std::make_shared<GeTensor>(dim_tensor_desc, (uint8_t *)dim_value_vec.data(), dim_value_vec.size() * sizeof(int32_t));
  OpDescUtils::SetWeights(dim_op_desc, dim_tensor);
  dim_op_desc->AddOutputDesc(dim_tensor_desc);
  NodePtr dim_node = graph->AddNode(dim_op_desc);
  dim_node->Init();

  // add expanddims node
  OpDescPtr expanddims_op_desc = std::make_shared<OpDesc>("Expanddims", EXPANDDIMS);
  vector<int64_t> expanddims_vec = {1, 1, 2, 3};
  GeTensorDesc expanddims_tensor_desc(ge::GeShape(expanddims_vec), FORMAT_NCHW, DT_INT32);
  GeTensorPtr expanddims_tensor = std::make_shared<GeTensor>(expanddims_tensor_desc, (uint8_t *)data_value_vec.data(),
                                                        data_value_vec.size() * sizeof(int32_t));
  OpDescUtils::SetWeights(expanddims_op_desc, expanddims_tensor);
  expanddims_op_desc->AddInputDesc(data_tensor_desc);
  expanddims_op_desc->AddInputDesc(dim_tensor_desc);
  expanddims_op_desc->AddOutputDesc(expanddims_tensor_desc);
  NodePtr op_node = graph->AddNode(expanddims_op_desc);
  op_node->Init();

  // add output node
  OpDescPtr netoutput_op_desc = std::make_shared<OpDesc>("NetOutput", "NetOutput");
  netoutput_op_desc->AddInputDesc(expanddims_tensor_desc);
  NodePtr netoutput_node = graph->AddNode(netoutput_op_desc);
  netoutput_node->Init();

  // add edge
  GraphUtils::AddEdge(data_node->GetOutDataAnchor(0), op_node->GetInDataAnchor(0));
  GraphUtils::AddEdge(dim_node->GetOutDataAnchor(0), op_node->GetInDataAnchor(1));
  GraphUtils::AddEdge(op_node->GetOutDataAnchor(0), netoutput_node->GetInDataAnchor(0));

  std::shared_ptr<DimensionAdjustPass> pass = std::make_shared<DimensionAdjustPass>();
  NamesToPass names_to_passes;
  EXPECT_EQ(4, graph->GetDirectNodesSize());
  ge::Status ret = pass->Run(op_node);
  EXPECT_EQ(SUCCESS, ret);
  EXPECT_EQ(2, op_node->GetOwnerComputeGraph()->GetDirectNodesSize());
}

TEST_F(UtestGraphPassesDimensionAdjustPass, input_node_is_nullptr) {
  std::shared_ptr<DimensionAdjustPass> pass = std::make_shared<DimensionAdjustPass>();
  ge::NodePtr node = nullptr;
  ge::Status ret = pass->Run(node);
  EXPECT_EQ(PARAM_INVALID, ret);
}

TEST_F(UtestGraphPassesDimensionAdjustPass, node_op_desc_is_nullptr) {
  NodePtr op_node = std::make_shared<Node>(nullptr, nullptr);

  std::shared_ptr<DimensionAdjustPass> pass = std::make_shared<DimensionAdjustPass>();
  ge::Status ret = pass->Run(op_node);
  EXPECT_EQ(PARAM_INVALID, ret);
}

TEST_F(UtestGraphPassesDimensionAdjustPass, node_get_original_type_failed) {
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("default");
  OpDescPtr expanddim_op_desc = std::make_shared<OpDesc>("Expanddims", FRAMEWORKOP);
  NodePtr op_node = std::make_shared<Node>(expanddim_op_desc, graph);

  std::shared_ptr<DimensionAdjustPass> pass = std::make_shared<DimensionAdjustPass>();
  ge::Status ret = pass->Run(op_node);
  EXPECT_NE(ret, ge::SUCCESS);
}

TEST_F(UtestGraphPassesDimensionAdjustPass, node_not_register_op) {
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("default");
  OpDescPtr expanddim_op_desc = std::make_shared<OpDesc>("Expanddims", FRAMEWORKOP);
  AttrUtils::SetStr(expanddim_op_desc, ATTR_NAME_FRAMEWORK_ORIGINAL_TYPE, "expanddims_fake");
  NodePtr op_node = std::make_shared<Node>(expanddim_op_desc, graph);

  std::shared_ptr<DimensionAdjustPass> pass = std::make_shared<DimensionAdjustPass>();
  ge::Status ret = pass->Run(op_node);
  EXPECT_EQ(SUCCESS, ret);
}

TEST_F(UtestGraphPassesDimensionAdjustPass, node_compute_failed) {
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("default");
  OpDescPtr expanddim_op_desc = std::make_shared<OpDesc>("Expanddims", EXPANDDIMS);
  NodePtr op_node = std::make_shared<Node>(expanddim_op_desc, graph);

  std::shared_ptr<DimensionAdjustPass> pass = std::make_shared<DimensionAdjustPass>();
  ge::Status ret = pass->Run(op_node);
  EXPECT_EQ(SUCCESS, ret);
}

TEST_F(UtestGraphPassesDimensionAdjustPass, AddIdentityNodeToGraph) {
  ge::ComputeGraphPtr graph = nullptr;
  ge::ComputeGraphPtr graph1 = std::make_shared<ge::ComputeGraph>("default");
  std::shared_ptr<DimensionAdjustPass> pass = std::make_shared<DimensionAdjustPass>();

  int64_t dims_size = 1;
  vector<int64_t> data_vec = {5};
  for_each(data_vec.begin(), data_vec.end(), [&](int64_t &data) { dims_size *= data; });
  vector<int32_t> data_value_vec(dims_size, 1);
  GeTensorDesc data_tensor_desc(GeShape(data_vec), FORMAT_NCHW, DT_INT32);
  auto ret = pass->AddIdentityNodeToGraph("name", data_tensor_desc, graph);
  EXPECT_EQ(ret, nullptr);

  ret = pass->AddIdentityNodeToGraph("name", data_tensor_desc, graph1);
  EXPECT_NE(ret, nullptr);
}

TEST_F(UtestGraphPassesDimensionAdjustPass, DimAdjWithUnknowShape) {
  ge::ComputeGraphPtr graph = BuildGraph1();
  auto op_node = graph->FindNode("shapeYes1");
  auto op_desc = op_node->GetOpDesc();
  std::vector<int64_t> dims = {-1, 224, 224, 3};
  op_desc->MutableOutputDesc(0)->SetShape(GeShape(dims));
  std::shared_ptr<DimensionAdjustPass> pass = make_shared<DimensionAdjustPass>();
  ge::Status ret = pass->Run(op_node);
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(graph->GetAllNodesSize(), 3);
}

TEST_F(UtestGraphPassesDimensionAdjustPass, Don_Not_Constat_Folding) {
  ge::ComputeGraphPtr graph = BuildGraph1();
  auto op_node = graph->FindNode("shapeYes1");
  (void)AttrUtils::SetBool(op_node->GetOpDesc(), ATTR_NAME_DO_NOT_CONSTANT_FOLDING, true);
  std::shared_ptr<DimensionAdjustPass> pass = make_shared<DimensionAdjustPass>();
  EXPECT_EQ(pass->Run(op_node), SUCCESS);
  EXPECT_EQ(graph->GetAllNodesSize(), 3);
  EXPECT_NE(graph->FindNode("shapeYes1"), nullptr);
}
}  // namespace ge
