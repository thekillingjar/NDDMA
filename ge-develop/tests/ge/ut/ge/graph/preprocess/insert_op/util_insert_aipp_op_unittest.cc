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
#include <gmock/gmock.h>
#include <vector>

#include "common/ge_inner_error_codes.h"
#include "common/types.h"
#include "common/util.h"
#include "ge_graph_dsl/graph_dsl.h"
#include "graph/debug/ge_attr_define.h"
#include "common/types.h"
#include "graph/passes/graph_builder_utils.h"

#include "macro_utils/dt_public_scope.h"
#include "graph/preprocess/insert_op/insert_aipp_op_util.h"
#include "proto/insert_op.pb.h"
#include "graph/utils/attr_utils.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/passes/graph_builder_utils.h"
#include "framework/omg/omg_inner_types.h"
#include "graph/preprocess/graph_prepare.h"
#include "graph/preprocess/insert_op/aipp_op.h"
#include "ge_running_env/path_utils.h"
#include "macro_utils/dt_public_unscope.h"

using namespace std;
using namespace testing;

namespace ge {

class UtestUtilInsertAippOp : public testing::Test {
 protected:
  void SetUp() {
  }
  void TearDown() {
  }
};

namespace {
ComputeGraphPtr MakeFunctionGraph(const std::string &func_node_name, const std::string &func_node_type) {
  DEF_GRAPH(g1) {
    CHAIN(NODE("_arg_0", DATA)->NODE(func_node_name, func_node_type)->NODE("Node_Output", NETOUTPUT));
    CHAIN(NODE("_arg_1", DATA)->NODE(func_node_name));
    CHAIN(NODE("_arg_2", AIPP)->NODE(func_node_name));
  };
  return ToComputeGraph(g1);
}

ComputeGraphPtr MakeSubGraph(const std::string &prefix) {
  DEF_GRAPH(g2, prefix.c_str()) {
    auto data_0 = OP_CFG(DATA).Attr(ATTR_NAME_PARENT_NODE_INDEX, 0);
    auto data_1 = OP_CFG(DATA).Attr(ATTR_NAME_PARENT_NODE_INDEX, 1);
    auto data_2 = OP_CFG(DATA).Attr(ATTR_NAME_PARENT_NODE_INDEX, 2);
    auto conv_0 = OP_CFG(CONV2D).Attr(ATTR_NAME_IMPLY_TYPE, static_cast<int64_t>(domi::ImplyType::TVM));
    auto relu_0 = OP_CFG(RELU).Attr(ATTR_NAME_IMPLY_TYPE, static_cast<int64_t>(domi::ImplyType::AI_CPU));
    auto add_0 = OP_CFG(ADD).Attr(ATTR_NAME_IMPLY_TYPE, static_cast<int64_t>(domi::ImplyType::AI_CPU));
    CHAIN(NODE(prefix + "_arg_0", data_0)
              ->EDGE(0, 0)
              ->NODE(prefix + "Conv2D", conv_0)
              ->EDGE(0, 0)
              ->NODE(prefix + "Relu", relu_0)
              ->EDGE(0, 0)
              ->NODE(prefix + "Add", add_0)
              ->EDGE(0, 0)
              ->NODE(prefix + "Node_Output", NETOUTPUT));
    CHAIN(NODE(prefix + "_arg_1", data_1)->EDGE(0, 1)->NODE(prefix + "Conv2D", conv_0));
    CHAIN(NODE(prefix + "_arg_2", data_2)->EDGE(0, 1)->NODE(prefix + "Add", add_0));
  };
  return ToComputeGraph(g2);
}

ComputeGraphPtr MakeSubGraph1(const std::string &prefix) {
  DEF_GRAPH(g2, prefix.c_str()) {
    auto data_0 = OP_CFG(DATA).Attr(ATTR_NAME_PARENT_NODE_INDEX, 0);
    auto data_1 = OP_CFG(DATA).Attr(ATTR_NAME_PARENT_NODE_INDEX, 1);
    auto data_2 = OP_CFG(DATA).Attr(ATTR_NAME_PARENT_NODE_INDEX, 2);
    auto conv_0 = OP_CFG(CONV2D).Attr(ATTR_NAME_IMPLY_TYPE, static_cast<int64_t>(domi::ImplyType::TVM));
    auto relu_0 = OP_CFG(RELU).Attr(ATTR_NAME_IMPLY_TYPE, static_cast<int64_t>(domi::ImplyType::AI_CPU));
    auto add_0 = OP_CFG(ADD).Attr(ATTR_NAME_IMPLY_TYPE, static_cast<int64_t>(domi::ImplyType::AI_CPU));
    auto aipp_0 = OP_CFG(AIPP).Attr(ATTR_NAME_IMPLY_TYPE, static_cast<int64_t>(domi::ImplyType::TVM));
    CHAIN(NODE(prefix + "_arg_0", data_0)
              ->EDGE(0, 0)
              ->NODE(prefix + "Conv2D", conv_0)
              ->EDGE(0, 0)
              ->NODE(prefix + "Relu", relu_0)
              ->EDGE(0, 0)
              ->NODE(prefix + "Add", add_0)
              ->EDGE(0, 0)
              ->NODE(prefix + "Node_Output", NETOUTPUT));
    CHAIN(NODE(prefix + "_arg_1", data_1)->EDGE(0, 1)->NODE(prefix + "Conv2D", conv_0));
    CHAIN(NODE(prefix + "_arg_2", data_2)->EDGE(0, 1)->NODE(prefix + "AIPP", aipp_0)->EDGE(0, 1)->NODE(prefix + "Add", add_0));
  };
  return ToComputeGraph(g2);
}

}  // namespace

class NodeBuilder {
 public:
  NodeBuilder(const std::string &name, const std::string &type) { op_desc_ = std::make_shared<OpDesc>(name, type); }

  NodeBuilder &AddInputDesc(std::initializer_list<int64_t> shape, ge::Format format = FORMAT_NCHW,
                            ge::DataType data_type = DT_FLOAT) {
    op_desc_->AddInputDesc(CreateTensorDesc(shape, format, data_type)->Clone());
    return *this;
  }

  NodeBuilder &AddOutputDesc(std::initializer_list<int64_t> shape, ge::Format format = FORMAT_NCHW,
                             ge::DataType data_type = DT_FLOAT) {
    op_desc_->AddOutputDesc(CreateTensorDesc(shape, format, data_type)->Clone());
    return *this;
  }

  ge::NodePtr Build(const ge::ComputeGraphPtr &graph) { return graph->AddNode(op_desc_); }

 private:
  ge::GeTensorDescPtr CreateTensorDesc(std::initializer_list<int64_t> shape, ge::Format format = FORMAT_NCHW,
                                       ge::DataType data_type = DT_FLOAT) {
    GeShape ge_shape{std::vector<int64_t>(shape)};
    ge::GeTensorDescPtr tensor_desc = std::make_shared<ge::GeTensorDesc>();
    tensor_desc->SetShape(ge_shape);
    tensor_desc->SetFormat(format);
    tensor_desc->SetDataType(data_type);
    return tensor_desc;
  }

  ge::OpDescPtr op_desc_;
};

OpDescPtr CreateOpDesc(const std::string name, const std::string type, uint32_t input_num, uint32_t output_num) {
  OpDescPtr op_desc = std::shared_ptr<OpDesc>(new (std::nothrow) OpDesc(name, type));
  if (op_desc == nullptr) {
    return nullptr;
  }
  for (uint32_t i = 0; i < input_num; i++) {
    op_desc->AddInputDesc(GeTensorDesc());
  }
  for (uint32_t i = 0; i < output_num; i++) {
    op_desc->AddOutputDesc(GeTensorDesc());
  }
  return op_desc;
}

TEST_F(UtestUtilInsertAippOp, test_Init) {
    InsertAippOpUtil &instance = InsertAippOpUtil::Instance();
    auto ret = instance.Init();
    ASSERT_EQ(ret, SUCCESS);
}

TEST_F(UtestUtilInsertAippOp, test_Parse) {
    InsertAippOpUtil &instance = InsertAippOpUtil::Instance();
    instance.Init();
    GraphManagerOptions options;
    auto ret = instance.Parse(options);
    ASSERT_EQ(ret, SUCCESS);
    options.insert_op_file = GetAirPath() + "/test_aipp_test_parse_UT11/";
    ret = instance.Parse(options);
    ASSERT_NE(ret, SUCCESS);

    options.insert_op_file = GetAirPath() + "/tests/ge/st/config_file/aipp_conf/aipp_conf_check.cfg";
    ret = instance.Parse(options);
    ASSERT_EQ(ret, SUCCESS);

    options.dynamic_image_size = "1";
    instance.insert_op_conf_->add_aipp_op();
    ret = instance.Parse(options);
    ASSERT_NE(ret, SUCCESS);

    options.insert_op_file = GetAirPath() + "/tests/ge/st/config_file/aipp_conf/aipp_static.cfg";
    ret = instance.Parse(options);
    ASSERT_NE(ret, SUCCESS);
}

TEST_F(UtestUtilInsertAippOp, test_Insert_Aipp_Ops) {
    InsertAippOpUtil &instance = InsertAippOpUtil::Instance();
    ge::ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
    const ge::NodePtr data1 = NodeBuilder("data1", DATA)
                            .AddOutputDesc({1}, FORMAT_NCHW, DT_FLOAT)
                            .AddOutputDesc({1}, FORMAT_NCHW, DT_FLOAT)
                            .Build(graph);
    ge::AttrUtils::SetInt(data1->GetOpDesc(), ATTR_NAME_INDEX, 0);
    domi::AippOpParams params;
    std::string aipp_cfg_path = "/root/";
    auto ret = instance.InsertAippOps(graph, aipp_cfg_path);
    ASSERT_EQ(ret, SUCCESS);
}

TEST_F(UtestUtilInsertAippOp, test_Check_Input_Name_Position_Not_Repeat) {
    InsertAippOpUtil &instance = InsertAippOpUtil::Instance();
    ge::ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
    std::string aipp_cfg_path = "/root/";
    auto ret = instance.InsertAippOps(graph, aipp_cfg_path);
    ASSERT_EQ(ret, SUCCESS);
    instance.insert_op_conf_->InsertNewOps::add_aipp_op();
    instance.insert_op_conf_->InsertNewOps::add_aipp_op();
    instance.insert_op_conf_->InsertNewOps::add_aipp_op();
    ret = instance.CheckInputNamePositionNotRepeat();
    ASSERT_NE(ret, SUCCESS);
}

TEST_F(UtestUtilInsertAippOp, test_Check_Input_Rank_Position_No_Repeat) {
  InsertAippOpUtil &instance = InsertAippOpUtil::Instance();
    ge::ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
    std::string aipp_cfg_path = "/root/";
    auto ret = instance.InsertAippOps(graph, aipp_cfg_path);
    ASSERT_EQ(ret, SUCCESS);
    instance.insert_op_conf_->InsertNewOps::add_aipp_op();
    instance.insert_op_conf_->InsertNewOps::add_aipp_op();
    instance.insert_op_conf_->InsertNewOps::add_aipp_op();
    ret = instance.CheckInputRankPositionNoRepeat();
    ASSERT_NE(ret, SUCCESS);
}

TEST_F(UtestUtilInsertAippOp, test_Check_Position_Not_Repeat) {
    InsertAippOpUtil &instance = InsertAippOpUtil::Instance();
    ge::ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
    std::string aipp_cfg_path = "/root/";
    auto ret = instance.InsertAippOps(graph, aipp_cfg_path);
    ASSERT_EQ(ret, SUCCESS);
    instance.insert_op_conf_->InsertNewOps::add_aipp_op();
    instance.insert_op_conf_->InsertNewOps::add_aipp_op();
    instance.insert_op_conf_->InsertNewOps::add_aipp_op();
    ret = instance.CheckPositionNotRepeat();
    ASSERT_NE(ret, SUCCESS);
}

TEST_F(UtestUtilInsertAippOp, test_Get_Aipp_Params) {
    InsertAippOpUtil &instance = InsertAippOpUtil::Instance();
    ge::ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
    const ge::NodePtr data1 = NodeBuilder("data1", DATA)
                            .AddOutputDesc({1}, FORMAT_NCHW, DT_FLOAT)
                            .AddOutputDesc({1}, FORMAT_NCHW, DT_FLOAT)
                            .Build(graph);
    ge::AttrUtils::SetInt(data1->GetOpDesc(), ATTR_NAME_INDEX, 0);
    const std::unique_ptr<domi::AippOpParams> aippParams;
    std::string aipp_cfg_path = "/root/";
    auto ret = instance.InsertAippOps(graph, aipp_cfg_path);
    ASSERT_EQ(ret, SUCCESS);
    ret = instance.GetAippParams(aippParams, data1);
    ASSERT_NE(ret, SUCCESS);
}

TEST_F(UtestUtilInsertAippOp, test_Find_Max_Size_Node) {
    InsertAippOpUtil &instance = InsertAippOpUtil::Instance();
    ge::ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
    const ge::NodePtr data1 = NodeBuilder("data1", DATA)
                            .AddOutputDesc({1}, FORMAT_NCHW, DT_FLOAT)
                            .AddOutputDesc({1}, FORMAT_NCHW, DT_FLOAT)
                            .Build(graph);
    ge::AttrUtils::SetInt(data1->GetOpDesc(), ATTR_NAME_INDEX, 0);
    std::map<uint32_t, GeTensorDescPtr> aipp_inputs;
    std::map<uint32_t, int64_t> max_sizes;
    std::string aipp_cfg_path = "/root/";
    auto ret = instance.InsertAippOps(graph, aipp_cfg_path);
    ASSERT_EQ(ret, SUCCESS);
    ret = instance.FindMaxSizeNode(graph, data1, max_sizes, aipp_inputs);
    ASSERT_EQ(ret, SUCCESS);
}

TEST_F(UtestUtilInsertAippOp, test_Update_Case_Node) {
    InsertAippOpUtil &instance = InsertAippOpUtil::Instance();
    ge::ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
    const ge::NodePtr data1 = NodeBuilder("case1", CASE)
                            .AddOutputDesc({1}, FORMAT_NCHW, DT_FLOAT)
                            .AddOutputDesc({1}, FORMAT_NCHW, DT_FLOAT)
                            .Build(graph);
    ge::AttrUtils::SetInt(data1->GetOpDesc(), ATTR_NAME_INDEX, 0);
    std::string aipp_cfg_path = "/root/";
    auto ret = instance.InsertAippOps(graph, aipp_cfg_path);
    ASSERT_EQ(ret, SUCCESS);
    ret = instance.UpdateCaseNode(graph, data1);
    ASSERT_EQ(ret, SUCCESS);
}

TEST_F(UtestUtilInsertAippOp, test_Update_Prev_Node_By_Aipp) {
    InsertAippOpUtil &instance = InsertAippOpUtil::Instance();
    ge::ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
    ge::NodePtr data1 = NodeBuilder("data1", DATA)
                      .AddInputDesc({1}, FORMAT_NCHW, DT_FLOAT)
                      .AddOutputDesc({1}, FORMAT_NCHW, DT_FLOAT)
                      .AddOutputDesc({1}, FORMAT_NCHW, DT_FLOAT)
                      .Build(graph);
    ge::AttrUtils::SetInt(data1->GetOpDesc(), ATTR_NAME_INDEX, 0);
    std::string aipp_cfg_path = "/root/";
    auto ret = instance.InsertAippOps(graph, aipp_cfg_path);
    ASSERT_EQ(ret, SUCCESS);
    auto input_desc = data1->GetOpDesc()->MutableInputDesc(0);
    ge::TensorUtils::SetSize(*input_desc, 100);
    std::vector<int32_t> input_dims;
    input_dims.push_back(100);
    AttrUtils::SetListInt(input_desc, ATTR_NAME_INPUT_ORIGIN_SIZE, input_dims);
    ret = instance.UpdatePrevNodeByAipp(data1);
    ASSERT_NE(ret, SUCCESS);
}

TEST_F(UtestUtilInsertAippOp, test_Update_Data_By_SwitchN) {
    InsertAippOpUtil &instance = InsertAippOpUtil::Instance();
    ge::ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
    ge::NodePtr data1 = NodeBuilder("data1", DATA)
                       .AddOutputDesc({1}, FORMAT_NCHW, DT_FLOAT)
                       .AddOutputDesc({1}, FORMAT_NCHW, DT_FLOAT)
                       .Build(graph);
    ge::AttrUtils::SetInt(data1->GetOpDesc(), ATTR_NAME_INDEX, 0);
    auto output_desc = data1->GetOpDesc()->MutableOutputDesc(0);
    auto output_desc1 = data1->GetOpDesc()->MutableOutputDesc(0);
    ge::TensorUtils::SetSize(*output_desc, 100);
    ge::TensorUtils::SetSize(*output_desc1, 100);
    ge::NodePtr switchN = NodeBuilder("switchN1", SWITCHN)
                         .AddOutputDesc({1}, FORMAT_NCHW, DT_FLOAT)
                         .AddOutputDesc({1}, FORMAT_NCHW, DT_FLOAT)
                         .Build(graph);
    std::string aipp_cfg_path = "/root/";
    auto ret = instance.InsertAippOps(graph, aipp_cfg_path);
    auto output_desc2 = switchN->GetOpDesc()->MutableOutputDesc(0);
    auto output_desc3 = switchN->GetOpDesc()->MutableOutputDesc(0);
    ASSERT_EQ(ret, SUCCESS);
}

TEST_F(UtestUtilInsertAippOp, test_Update_Multi_Batch_Input_Dims) {
    InsertAippOpUtil &instance = InsertAippOpUtil::Instance();
    ge::ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
    std::string aipp_cfg_path = "/root/";
    ge::Format format = FORMAT_NCHW;
    auto ret = instance.InsertAippOps(graph, aipp_cfg_path);
    ASSERT_EQ(ret, SUCCESS);
    ge::OpDescPtr data_opdesc1 = std::make_shared<ge::OpDesc>("test", "data");;
    data_opdesc1->AddInputDesc("x", GeTensorDesc(GeShape({1, 16, 16, 16}), FORMAT_NCHW));
    std::vector<int32_t> input_dims;
    input_dims.push_back(1);
    AttrUtils::SetListInt(data_opdesc1, ATTR_MBATCH_ORIGIN_INPUT_DIMS, input_dims);
    instance.UpdateMultiBatchInputDims(data_opdesc1, format);
    input_dims.push_back(3);
    input_dims.push_back(224);
    input_dims.push_back(224);
    ge::OpDescPtr data_opdesc2 = std::make_shared<ge::OpDesc>("test1", "data");;
    data_opdesc2->AddInputDesc("x", GeTensorDesc(GeShape({1, 16, 16, 16}), FORMAT_NCHW));
    data_opdesc2->AddOutputDesc("y", GeTensorDesc(GeShape({1, 16, 16, 16}), FORMAT_NCHW));
    AttrUtils::SetListInt(data_opdesc2, ATTR_MBATCH_ORIGIN_INPUT_DIMS, input_dims);
    instance.UpdateMultiBatchInputDims(data_opdesc2, format);
    ge::OpDescPtr data_opdesc3 = std::make_shared<ge::OpDesc>("test1", "data");;
    data_opdesc2->AddInputDesc("x", GeTensorDesc(GeShape({1, 16, 16, 16}), FORMAT_HWCN));
    data_opdesc2->AddOutputDesc("y", GeTensorDesc(GeShape({1, 16, 16, 16}), FORMAT_HWCN));
    AttrUtils::SetListInt(data_opdesc3, ATTR_MBATCH_ORIGIN_INPUT_DIMS, input_dims);
    instance.UpdateMultiBatchInputDims(data_opdesc3, format);
}

TEST_F(UtestUtilInsertAippOp, test_Get_All_Aipps) {
    InsertAippOpUtil &instance = InsertAippOpUtil::Instance();
    ge::ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
    const NodePtr data1 = NodeBuilder("data1", DATA)
                            .AddOutputDesc({1}, FORMAT_NCHW, DT_FLOAT)
                            .AddOutputDesc({1}, FORMAT_NCHW, DT_FLOAT)
                            .Build(graph);
    ge::AttrUtils::SetInt(data1->GetOpDesc(), ATTR_NAME_INDEX, 0);
    const NodePtr switchN = NodeBuilder("switchN1", SWITCHN)
                            .AddOutputDesc({1}, FORMAT_NCHW, DT_FLOAT)
                            .AddOutputDesc({1}, FORMAT_NCHW, DT_FLOAT)
                            .Build(graph);
    const NodePtr case1 = NodeBuilder("case1", CASE)
                            .AddOutputDesc({1}, FORMAT_NCHW, DT_FLOAT)
                            .AddOutputDesc({1}, FORMAT_NCHW, DT_FLOAT)
                            .Build(graph);
    const NodePtr aipp1 = NodeBuilder("aipp1", AIPP)
                            .AddOutputDesc({1}, FORMAT_NCHW, DT_FLOAT)
                            .AddOutputDesc({1}, FORMAT_NCHW, DT_FLOAT)
                            .Build(graph);
    std::string aipp_cfg_path = "/root/";
    std::vector<NodePtr> aipps;
    aipps.push_back(data1);
    aipps.push_back(switchN);
    aipps.push_back(case1);
    aipps.push_back(aipp1);
    auto ret = instance.InsertAippOps(graph, aipp_cfg_path);
    ASSERT_EQ(ret, SUCCESS);
    ret = instance.GetAllAipps(data1, switchN, aipps);
    ASSERT_EQ(ret, SUCCESS);
    ret = instance.GetAllAipps(data1, case1, aipps);
    ASSERT_EQ(ret, SUCCESS);
    ret = instance.GetAllAipps(data1, aipp1, aipps);
    ASSERT_EQ(ret, SUCCESS);
}

TEST_F(UtestUtilInsertAippOp, test_Update_Data_Node_ByAipp) {
    InsertAippOpUtil &instance = InsertAippOpUtil::Instance();
    ge::ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
    const NodePtr data1 = NodeBuilder("data1", DATA)
                            .AddOutputDesc({1}, FORMAT_NCHW, DT_FLOAT)
                            .AddOutputDesc({1}, FORMAT_NCHW, DT_FLOAT)
                            .Build(graph);
    ge::AttrUtils::SetInt(data1->GetOpDesc(), ATTR_NAME_INDEX, 0);
    const NodePtr switchN = NodeBuilder("switchN1", SWITCHN)
                            .AddOutputDesc({1}, FORMAT_NCHW, DT_FLOAT)
                            .AddOutputDesc({1}, FORMAT_NCHW, DT_FLOAT)
                            .Build(graph);
    const NodePtr case1 = NodeBuilder("case1", CASE)
                            .AddOutputDesc({1}, FORMAT_NCHW, DT_FLOAT)
                            .AddOutputDesc({1}, FORMAT_NCHW, DT_FLOAT)
                            .Build(graph);
    const NodePtr aipp1 = NodeBuilder("aipp1", AIPP)
                            .AddOutputDesc({1}, FORMAT_NCHW, DT_FLOAT)
                            .AddOutputDesc({1}, FORMAT_NCHW, DT_FLOAT)
                            .Build(graph);
    std::string aipp_cfg_path = "/root/";
    auto ret = instance.InsertAippOps(graph, aipp_cfg_path);
    ASSERT_EQ(ret, SUCCESS);
    ret = instance.UpdateDataNodeByAipp(graph);
    ASSERT_NE(ret, SUCCESS);
}

TEST_F(UtestUtilInsertAippOp, test_Record_AIPP_Info_To_Data) {
    InsertAippOpUtil &instance = InsertAippOpUtil::Instance();
    ge::GraphPrepare graph_prepare;
    ge::ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
    const NodePtr data0 = NodeBuilder("data0", DATA)
                            .AddOutputDesc({1}, FORMAT_NCHW, DT_FLOAT)
                            .AddOutputDesc({1}, FORMAT_NCHW, DT_FLOAT)
                            .Build(graph);
    ge::AttrUtils::SetInt(data0->GetOpDesc(), ATTR_NAME_INDEX, 0);
    const NodePtr switchN = NodeBuilder("switchN1", SWITCHN)
                            .AddOutputDesc({1}, FORMAT_NCHW, DT_FLOAT)
                            .AddOutputDesc({1}, FORMAT_NCHW, DT_FLOAT)
                            .Build(graph);
    const NodePtr case1 = NodeBuilder("case1", CASE)
                            .AddOutputDesc({1}, FORMAT_NCHW, DT_FLOAT)
                            .AddOutputDesc({1}, FORMAT_NCHW, DT_FLOAT)
                            .Build(graph);
    const NodePtr aipp1 = NodeBuilder("aipp1", AIPP)
                            .AddOutputDesc({1}, FORMAT_NCHW, DT_FLOAT)
                            .AddOutputDesc({1}, FORMAT_NCHW, DT_FLOAT)
                            .Build(graph);
    NodePtr func_node = graph->AddNode(CreateOpDesc("case", CASE, 2, 1));
    NodePtr func_node1 = graph->AddNode(CreateOpDesc("aipp0", AIPP, 2, 1));
    NodePtr data_node_0 = graph->AddNode(CreateOpDesc("data_0", DATA, 1, 1));
    NodePtr data_node_1 = graph->AddNode(CreateOpDesc("data_1", DATA, 1, 1));
    EXPECT_EQ(GraphUtils::AddEdge(data_node_0->GetOutDataAnchor(0), func_node->GetInDataAnchor(0)), GRAPH_SUCCESS);
    EXPECT_EQ(GraphUtils::AddEdge(data_node_1->GetOutDataAnchor(0), func_node->GetInDataAnchor(1)), GRAPH_SUCCESS);
    EXPECT_EQ(GraphUtils::AddEdge(data_node_0->GetOutDataAnchor(0), func_node1->GetInDataAnchor(0)), GRAPH_SUCCESS);
    EXPECT_EQ(GraphUtils::AddEdge(data_node_1->GetOutDataAnchor(0), func_node1->GetInDataAnchor(1)), GRAPH_SUCCESS);
    std::string subgraph_name_1 = "instance_branch_1";
    ComputeGraphPtr subgraph_1 = std::make_shared<ComputeGraph>(subgraph_name_1);
    subgraph_1->SetParentNode(func_node);
    subgraph_1->SetParentGraph(graph);
    std::string subgraph_name_2 = "instance_branch_2";
    ComputeGraphPtr subgraph_2 = std::make_shared<ComputeGraph>(subgraph_name_2);
    subgraph_2->SetParentNode(func_node1);
    subgraph_2->SetParentGraph(graph);
    std::string aipp_cfg_path = "/root/";
    std::map<uint32_t, GeTensorDescPtr> aipp_inputs;
    //aipp_inputs.insert(1, data_node_0->GetOpDesc()->GetOutputDescPtr());
    std::map<uint32_t, int64_t> max_sizes;
    auto ret = instance.InsertAippOps(graph, aipp_cfg_path);
    ASSERT_NE(ret, SUCCESS);
    ret = instance.RecordAIPPInfoToData(graph);
    ASSERT_EQ(ret, SUCCESS);
}

TEST_F(UtestUtilInsertAippOp, test_Get_Input_Out_put_Info) {
    InsertAippOpUtil &instance = InsertAippOpUtil::Instance();
    ge::GraphPrepare graph_prepare;
    ge::ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
    NodePtr data0 = NodeBuilder("data0", DATA)
                            .AddOutputDesc({1}, FORMAT_NCHW, DT_FLOAT)
                            .AddOutputDesc({1}, FORMAT_NCHW, DT_FLOAT)
                            .Build(graph);
    ge::AttrUtils::SetInt(data0->GetOpDesc(), ATTR_NAME_INDEX, 0);
    NodePtr aipp1 = NodeBuilder("aipp1", AIPP)
                            .AddOutputDesc({1}, FORMAT_NCHW, DT_FLOAT)
                            .AddOutputDesc({1}, FORMAT_NCHW, DT_FLOAT)
                            .Build(graph);
    std::string aipp_cfg_path = "/root/";
    std::map<uint32_t, GeTensorDescPtr> aipp_inputs;
    std::map<uint32_t, int64_t> max_sizes;
    auto ret = instance.InsertAippOps(graph, aipp_cfg_path);
    ASSERT_EQ(ret, SUCCESS);
    string input = "data0";
    string output = "aipp1";
    ret = instance.GetInputOutputInfo(data0, aipp1, input, output);
    ASSERT_EQ(ret, SUCCESS);
}

TEST_F(UtestUtilInsertAippOp, test_Set_Model_Input_Dims) {
    InsertAippOpUtil &instance = InsertAippOpUtil::Instance();
    ge::GraphPrepare graph_prepare;
    ge::ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
    NodePtr data0 = NodeBuilder("data0", DATA)
                            .AddOutputDesc({1}, FORMAT_NCHW, DT_FLOAT)
                            .AddOutputDesc({1}, FORMAT_NCHW, DT_FLOAT)
                            .Build(graph);
    ge::AttrUtils::SetInt(data0->GetOpDesc(), ATTR_NAME_INDEX, 0);
    NodePtr aipp1 = NodeBuilder("aipp1", AIPP)
                            .AddOutputDesc({1}, FORMAT_NCHW, DT_FLOAT)
                            .AddOutputDesc({1}, FORMAT_NCHW, DT_FLOAT)
                            .Build(graph);
    std::string aipp_cfg_path = "/root/";
    std::vector<int32_t> input_dims;
    input_dims.push_back(1);
    AttrUtils::SetListInt(aipp1->GetOpDesc(), ATTR_NAME_INPUT_DIMS, input_dims);
    AttrUtils::SetListInt(data0->GetOpDesc(), ATTR_MBATCH_ORIGIN_INPUT_DIMS, input_dims);
    std::map<uint32_t, GeTensorDescPtr> aipp_inputs;
    std::map<uint32_t, int64_t> max_sizes;
    auto ret = instance.InsertAippOps(graph, aipp_cfg_path);
    ASSERT_EQ(ret, SUCCESS);
    string input = "data0";
    string output = "aipp1";
    ret = instance.SetModelInputDims(data0, aipp1);
    ASSERT_EQ(ret, SUCCESS);
}

TEST_F(UtestUtilInsertAippOp, test_Set_Only_Model_Input_Dims) {
    InsertAippOpUtil &instance = InsertAippOpUtil::Instance();
    ge::GraphPrepare graph_prepare;
    ge::ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
    NodePtr data0 = NodeBuilder("data0", DATA)
        .AddOutputDesc({1}, FORMAT_NCHW, DT_FLOAT)
        .AddOutputDesc({1}, FORMAT_NCHW, DT_FLOAT)
        .Build(graph);
    ge::AttrUtils::SetInt(data0->GetOpDesc(), ATTR_NAME_INDEX, 0);
    NodePtr aipp1 = NodeBuilder("aipp1", AIPP)
        .AddOutputDesc({1}, FORMAT_NCHW, DT_FLOAT)
        .AddOutputDesc({1}, FORMAT_NCHW, DT_FLOAT)
        .Build(graph);
    std::string aipp_cfg_path = "/root/";
    std::vector<int32_t> input_dims;
    input_dims.push_back(1);
    AttrUtils::SetListInt(aipp1->GetOpDesc(), ATTR_NAME_INPUT_DIMS, input_dims);
    std::map<uint32_t, GeTensorDescPtr> aipp_inputs;
    std::map<uint32_t, int64_t> max_sizes;
    auto ret = instance.InsertAippOps(graph, aipp_cfg_path);
    ASSERT_EQ(ret, SUCCESS);
    string input = "data0";
    string output = "aipp1";
    ret = instance.SetModelInputDims(data0, aipp1);
    ASSERT_EQ(ret, SUCCESS);
}

TEST_F(UtestUtilInsertAippOp, test_Get_Data_Related_Node) {
    InsertAippOpUtil &instance = InsertAippOpUtil::Instance();
    ge::GraphPrepare graph_prepare;
    ge::ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
    NodePtr data0 = NodeBuilder("data0", DATA)
                            .AddOutputDesc({1}, FORMAT_NCHW, DT_FLOAT)
                            .AddOutputDesc({1}, FORMAT_NCHW, DT_FLOAT)
                            .Build(graph);
    ge::AttrUtils::SetInt(data0->GetOpDesc(), ATTR_NAME_INDEX, 0);
    NodePtr aipp1 = NodeBuilder("aipp1", AIPP)
                            .AddOutputDesc({1}, FORMAT_NCHW, DT_FLOAT)
                            .AddOutputDesc({1}, FORMAT_NCHW, DT_FLOAT)
                            .Build(graph);
    std::string aipp_cfg_path = "/root/";
    std::vector<int32_t> input_dims;
    input_dims.push_back(1);
    std::map<NodePtr, std::set<NodePtr>> data_next_node_map;
    AttrUtils::SetListInt(aipp1->GetOpDesc(), ATTR_NAME_INPUT_DIMS, input_dims);
    AttrUtils::SetListInt(data0->GetOpDesc(), ATTR_MBATCH_ORIGIN_INPUT_DIMS, input_dims);
    std::vector<string> inputs = {};
    NamedAttrs aipp_attr;
    aipp_attr.SetAttr("aipp_mode", GeAttrValue::CreateFrom<int64_t>(domi::AippOpParams_AippMode_dynamic));
    aipp_attr.SetAttr("related_input_rank", GeAttrValue::CreateFrom<int64_t>(0));
    aipp_attr.SetAttr("max_src_image_size", GeAttrValue::CreateFrom<int64_t>(2048));
    aipp_attr.SetAttr("support_rotation", GeAttrValue::CreateFrom<int64_t>(1));
    AttrUtils::SetNamedAttrs(data0->GetOpDesc(), ATTR_NAME_AIPP, aipp_attr);
    std::map<uint32_t, GeTensorDescPtr> aipp_inputs;
    std::map<uint32_t, int64_t> max_sizes;
    auto ret = instance.InsertAippOps(graph, aipp_cfg_path);
    ASSERT_EQ(ret, SUCCESS);
    string input = "data0";
    string output = "aipp1";
    ret = instance.GetDataRelatedNode(data0, data_next_node_map);
    ASSERT_EQ(ret, SUCCESS);
}

TEST_F(UtestUtilInsertAippOp, test_Find_Max_Size_Node_3) {
    InsertAippOpUtil &instance = InsertAippOpUtil::Instance();
    std::map<uint32_t, GeTensorDescPtr> aipp_inputs;
    std::map<uint32_t, int64_t> max_sizes;
    std::string func_node_name = "PartitionedCall_0";
    const auto &root_graph = MakeFunctionGraph(func_node_name, PARTITIONEDCALL);
    const auto &sub_graph = MakeSubGraph("sub_graph_0/");
    ut::GraphBuilder::AddPartitionedCall(root_graph, func_node_name, sub_graph);
    auto ret = instance.FindMaxSizeNode(root_graph, root_graph->FindNode(func_node_name), max_sizes, aipp_inputs);
    ASSERT_EQ(ret, SUCCESS);
    std::vector<NodePtr> aipps;
    auto case_0 = root_graph->FindNode("Case");
    auto switchn_0 = root_graph->FindNode("SwitchN");
    auto conv2d = root_graph->FindNode("Conv2D");
    auto netout_put = root_graph->FindNode("Node_Output");
    aipps.push_back(case_0);
    aipps.push_back(switchn_0);
    aipps.push_back(conv2d);
    aipps.push_back(netout_put);
    ret = instance.GetAllAipps(netout_put, case_0, aipps);
    ASSERT_NE(ret, SUCCESS);
    ret = instance.GetAllAipps(netout_put, switchn_0, aipps);
    ASSERT_NE(ret, SUCCESS);
    ret = instance.GetAllAipps(netout_put, conv2d, aipps);
}

TEST_F(UtestUtilInsertAippOp, test_Find_Max_Size_Node_4) {
    InsertAippOpUtil &instance = InsertAippOpUtil::Instance();
    std::map<uint32_t, GeTensorDescPtr> aipp_inputs;
    std::map<uint32_t, int64_t> max_sizes;
    std::string func_node_name = "test";
    const auto &root_graph = MakeFunctionGraph(func_node_name, PARTITIONEDCALL);
    ge::ut::GraphBuilder builder("sub_graph_0");
    auto data1 = builder.AddNode("sub_graph_0/data1", "Data", 1, 1);
    auto data2 = builder.AddNode("sub_graph_0/data2", "Data", 1, 1);
    auto aipp = builder.AddNode("sub_graph_0/aipp", "aipp", 2, 1);
    auto netoutput = builder.AddNode("sub_graph_0/Node_Output", "NetOutput", 1, 0);

    aipp->GetOpDesc()->AddInputDesc(ge::GeTensorDesc());
    aipp->GetOpDesc()->AddInputDesc(ge::GeTensorDesc());
    aipp->GetOpDesc()->AddOutputDesc(ge::GeTensorDesc());
    data1->GetOpDesc()->AddOutputDesc(GeTensorDesc(GeShape(std::vector<int64_t>{3,224,224}), FORMAT_NCHW, DT_FLOAT));
    data1->GetOpDesc()->UpdateOutputDesc(0, GeTensorDesc(GeShape(std::vector<int64_t>{3,224,224}), FORMAT_NCHW, DT_FLOAT));
    data2->GetOpDesc()->AddOutputDesc(GeTensorDesc(GeShape(std::vector<int64_t>{2}), FORMAT_NCHW, DT_INT64));
    data2->GetOpDesc()->UpdateOutputDesc(0, GeTensorDesc(GeShape(std::vector<int64_t>{2}), FORMAT_NCHW, DT_INT64));

    builder.AddDataEdge(data1, 0, aipp, 0);
    builder.AddDataEdge(data2, 0, aipp, 1);
    builder.AddDataEdge(aipp, 0, netoutput, 0);
    const auto &sub_graph = builder.GetGraph();
    ut::GraphBuilder::AddPartitionedCall(root_graph, func_node_name, sub_graph);
    auto ret = instance.FindMaxSizeNode(root_graph, root_graph->FindNode(func_node_name), max_sizes, aipp_inputs);
    ASSERT_EQ(ret, SUCCESS);
}

TEST_F(UtestUtilInsertAippOp, test_Get_All_Aipp2) {
    InsertAippOpUtil &instance = InsertAippOpUtil::Instance();
    std::map<uint32_t, GeTensorDescPtr> aipp_inputs;
    std::map<uint32_t, int64_t> max_sizes;
    std::string func_node_name = "CASE_0";
    const auto &root_graph = MakeFunctionGraph(func_node_name, CASE);
    const auto &sub_graph = MakeSubGraph1("sub_graph_0/");
    ut::GraphBuilder::AddPartitionedCall(root_graph, func_node_name, sub_graph);
    std::string func_node_name1 = "Switch_0";
    const auto &root_graph1 = MakeFunctionGraph(func_node_name1, SWITCHN);
    const auto &sub_graph1 = MakeSubGraph1("sub_graph_0/");
    ut::GraphBuilder::AddPartitionedCall(root_graph1, func_node_name1, sub_graph1);
    std::vector<NodePtr> aipps;
    auto case_0 = root_graph->FindNode(func_node_name);
    auto switchn_0 = root_graph->FindNode(func_node_name1);
    auto conv2d_0 = root_graph->FindNode("sub_graph_0/Conv2D");
    auto relu_0 = root_graph->FindNode("sub_graph_0/Relu");
    aipps.push_back(case_0);
    aipps.push_back(switchn_0);
    aipps.push_back(conv2d_0);
    aipps.push_back(relu_0);
    auto ret = instance.GetAllAipps(case_0, case_0, aipps);
    ASSERT_EQ(ret, SUCCESS);
}

TEST_F(UtestUtilInsertAippOp, test_UpdateCaseNode1) {
    InsertAippOpUtil &instance = InsertAippOpUtil::Instance();
    std::map<uint32_t, GeTensorDescPtr> aipp_inputs;
    std::map<uint32_t, int64_t> max_sizes;
    std::string func_node_name = "test";
    const auto &root_graph = MakeFunctionGraph(func_node_name, PARTITIONEDCALL);
    ge::ut::GraphBuilder builder("sub_graph_0");
    auto data1 = builder.AddNode("sub_graph_0/data1", "Data", 1, 1);
    auto data2 = builder.AddNode("sub_graph_0/data2", "Data", 1, 1);
    auto aipp = builder.AddNode("sub_graph_0/aipp", "Aipp", 2, 1);
    auto netoutput = builder.AddNode("sub_graph_0/Node_Output", "NetOutput", 1, 0);

    auto ge_tensor_desc = GeTensorDesc(GeShape(std::vector<int64_t>{3,224,224}), FORMAT_NCHW, DT_FLOAT);
    aipp->GetOpDesc()->AddInputDesc(ge_tensor_desc);
    ge::TensorUtils::SetSize(*aipp->GetOpDesc()->MutableInputDesc(0), 10000);
    aipp->GetOpDesc()->AddInputDesc(ge_tensor_desc);
    aipp->GetOpDesc()->AddOutputDesc(ge_tensor_desc);
    data1->GetOpDesc()->AddOutputDesc(ge_tensor_desc);
    data1->GetOpDesc()->UpdateOutputDesc(0, ge_tensor_desc);
    (void)ge::AttrUtils::SetInt(data1->GetOpDesc(), ge::ATTR_NAME_PARENT_NODE_INDEX, 0);
    data2->GetOpDesc()->AddOutputDesc(ge_tensor_desc);
    data2->GetOpDesc()->UpdateOutputDesc(0, ge_tensor_desc);

    builder.AddDataEdge(data1, 0, aipp, 0);
    builder.AddDataEdge(data2, 0, aipp, 1);
    builder.AddDataEdge(aipp, 0, netoutput, 0);
    const auto &sub_graph = builder.GetGraph();
    ut::GraphBuilder::AddPartitionedCall(root_graph, func_node_name, sub_graph);
    const auto call_node = root_graph->FindNode(func_node_name);
    ASSERT_NE(call_node, nullptr);
    auto ret = instance.UpdateCaseNode(root_graph, call_node);
    ASSERT_EQ(ret, SUCCESS);
}

TEST_F(UtestUtilInsertAippOp, test_CheckPositionNotRepeat1) {
  InsertAippOpUtil &instance = InsertAippOpUtil::Instance();
  instance.insert_op_conf_.reset((new (std::nothrow) domi::InsertNewOps()));
  auto ret = instance.CheckPositionNotRepeat();
  ASSERT_EQ(ret, SUCCESS);
}

TEST_F(UtestUtilInsertAippOp, test_UpdatePrevNodeByAipp) {
  InsertAippOpUtil &instance = InsertAippOpUtil::Instance();
  ge::ut::GraphBuilder builder("graph");
  auto data1 = builder.AddNode("data1", "Data", 1, 1);
  auto data2 = builder.AddNode("data2", "Data", 1, 1);
  auto aipp = builder.AddNode("aipp", "aipp", 2, 1);
  auto netoutput = builder.AddNode("Node_Output", "NetOutput", 1, 0);

  builder.AddDataEdge(data1, 0, aipp, 0);
  builder.AddDataEdge(data2, 0, aipp, 1);
  builder.AddDataEdge(aipp, 0, netoutput, 0);
  ComputeGraphPtr computeGraph = builder.GetGraph();
  AippOp aipp_op;
  aipp->GetOpDesc()->AddInputDesc(ge::GeTensorDesc());
  aipp->GetOpDesc()->AddInputDesc(ge::GeTensorDesc());
  aipp->GetOpDesc()->AddOutputDesc(ge::GeTensorDesc());
  data1->GetOpDesc()->AddOutputDesc(GeTensorDesc(GeShape(std::vector<int64_t>{224,224,3}), FORMAT_NHWC, DT_FLOAT));
  data1->GetOpDesc()->UpdateOutputDesc(0, GeTensorDesc(GeShape(std::vector<int64_t>{224,224,3}), FORMAT_ND, DT_FLOAT));
  data2->GetOpDesc()->AddOutputDesc(GeTensorDesc(GeShape(std::vector<int64_t>{2}), FORMAT_NHWC, DT_INT64));
  data2->GetOpDesc()->UpdateOutputDesc(0, GeTensorDesc(GeShape(std::vector<int64_t>{2}), FORMAT_ND, DT_INT64));

  EXPECT_EQ(instance.UpdatePrevNodeByAipp(aipp), FAILED);
}

TEST_F(UtestUtilInsertAippOp, test_UpdateDataNodeByAipp1) {
  InsertAippOpUtil &instance = InsertAippOpUtil::Instance();
  std::string func_node_name = "PartitionedCall_0";
  const auto &root_graph = MakeFunctionGraph(func_node_name, PARTITIONEDCALL);

  ge::ut::GraphBuilder builder("graph");
  auto data1 = builder.AddNode("data1", "Data", 1, 1);
  auto data2 = builder.AddNode("data2", "Case", 1, 1);
  auto data3 = builder.AddNode("data3", "Data", 1, 1);
  auto aipp = builder.AddNode("aipp", "aipp", 2, 1);
  auto netoutput = builder.AddNode("Node_Output", "NetOutput", 1, 0);
  AttrUtils::SetStr(data1->GetOpDesc(), "mbatch-switch-name", "data3");
  AttrUtils::SetInt(data2->GetOpDesc(), "_batch_num", 1);
  aipp->GetOpDesc()->AddInputDesc(ge::GeTensorDesc());
  aipp->GetOpDesc()->AddInputDesc(ge::GeTensorDesc());
  aipp->GetOpDesc()->AddOutputDesc(ge::GeTensorDesc());
  data1->GetOpDesc()->AddOutputDesc(GeTensorDesc(GeShape(std::vector<int64_t>{224,224,3}), FORMAT_NHWC, DT_FLOAT));
  data1->GetOpDesc()->UpdateOutputDesc(0, GeTensorDesc(GeShape(std::vector<int64_t>{224,224,3}), FORMAT_ND, DT_FLOAT));
  data2->GetOpDesc()->AddOutputDesc(GeTensorDesc(GeShape(std::vector<int64_t>{2}), FORMAT_NHWC, DT_INT64));
  data2->GetOpDesc()->UpdateOutputDesc(0, GeTensorDesc(GeShape(std::vector<int64_t>{2}), FORMAT_ND, DT_INT64));

  builder.AddDataEdge(data1, 0, aipp, 0);
  builder.AddDataEdge(data2, 0, aipp, 1);
  builder.AddDataEdge(aipp, 0, netoutput, 0);
  const auto &sub_graph = builder.GetGraph();

  ut::GraphBuilder::AddPartitionedCall(root_graph, func_node_name, sub_graph);
  EXPECT_EQ(instance.UpdateDataNodeByAipp(sub_graph), SUCCESS);
}

TEST_F(UtestUtilInsertAippOp, test_UpdateDataNodeByAipp2) {
  InsertAippOpUtil &instance = InsertAippOpUtil::Instance();

  ge::ut::GraphBuilder builder("graph");
  auto data1 = builder.AddNode("data1", "Data", 1, 1);
  auto data2 = builder.AddNode("data2", "Data", 1, 1);
  auto aipp = builder.AddNode("aipp", "aipp", 2, 1);
  auto netoutput = builder.AddNode("Node_Output", "NetOutput", 1, 0);
  builder.AddDataEdge(data1, 0, aipp, 0);
  builder.AddDataEdge(data2, 0, aipp, 1);
  builder.AddDataEdge(aipp, 0, netoutput, 0);

  data1->GetOpDesc()->AddOutputDesc(GeTensorDesc(GeShape(std::vector<int64_t>{224,224,3}), FORMAT_NHWC, DT_FLOAT));
  data1->GetOpDesc()->UpdateOutputDesc(0, GeTensorDesc(GeShape(std::vector<int64_t>{224,224,3}), FORMAT_ND, DT_FLOAT));
  data2->GetOpDesc()->AddOutputDesc(GeTensorDesc(GeShape(std::vector<int64_t>{2}), FORMAT_NHWC, DT_INT64));
  data2->GetOpDesc()->UpdateOutputDesc(0, GeTensorDesc(GeShape(std::vector<int64_t>{2}), FORMAT_ND, DT_INT64));
  GeTensorDesc tensor1(GeShape(std::vector<int64_t>{1,224,224,3}), FORMAT_NHWC, DT_FLOAT);
  TensorUtils::SetSize(tensor1, 602112);
  aipp->GetOpDesc()->AddInputDesc(tensor1);
  GeTensorDesc tensor2(GeShape(std::vector<int64_t>{2}), FORMAT_ND, DT_INT64);
  TensorUtils::SetSize(tensor2, 8);
  aipp->GetOpDesc()->AddInputDesc(tensor2);
  aipp->GetOpDesc()->AddOutputDesc(tensor1);
  const auto &sub_graph = builder.GetGraph();

  const GeTensorDescPtr &input1 = aipp->GetOpDesc()->MutableInputDesc(0);
  ge::TensorUtils::SetSize(*input1, 602112);
  const GeTensorDescPtr &input2 = aipp->GetOpDesc()->MutableInputDesc(1);
  ge::TensorUtils::SetSize(*input2, 8);
  const GeTensorDescPtr &output1 = aipp->GetOpDesc()->MutableOutputDesc(0);
  ge::TensorUtils::SetSize(*output1, 75264);

  EXPECT_EQ(instance.UpdatePrevNodeByAipp(aipp), SUCCESS);
}

TEST_F(UtestUtilInsertAippOp, test_UpdateDataNodeByAipp3) {
  InsertAippOpUtil &instance = InsertAippOpUtil::Instance();

  ge::ut::GraphBuilder builder("graph");
  auto data1 = builder.AddNode("data1", "Data", 1, 1);
  auto data2 = builder.AddNode("data2", "Data", 1, 1);
  auto aipp = builder.AddNode("aipp", "SwitchN", 2, 2);
  auto netoutput = builder.AddNode("Node_Output", "NetOutput", 2, 0);
  builder.AddDataEdge(data1, 0, aipp, 0);
  builder.AddDataEdge(data2, 0, aipp, 1);
  builder.AddDataEdge(aipp, 0, netoutput, 0);
  builder.AddDataEdge(aipp, 1, netoutput, 1);
  NamedAttrs aipp_attr;
  AttrUtils::SetNamedAttrs(data1->GetOpDesc(), "aipp", aipp_attr);

  data1->GetOpDesc()->AddOutputDesc(GeTensorDesc(GeShape(std::vector<int64_t>{224,224,3}), FORMAT_NHWC, DT_FLOAT));
  data1->GetOpDesc()->UpdateOutputDesc(0, GeTensorDesc(GeShape(std::vector<int64_t>{224,224,3}), FORMAT_ND, DT_FLOAT));
  data2->GetOpDesc()->AddOutputDesc(GeTensorDesc(GeShape(std::vector<int64_t>{2}), FORMAT_NHWC, DT_INT64));
  data2->GetOpDesc()->UpdateOutputDesc(0, GeTensorDesc(GeShape(std::vector<int64_t>{2}), FORMAT_ND, DT_INT64));
  GeTensorDesc tensor1(GeShape(std::vector<int64_t>{1,224,224,3}), FORMAT_NHWC, DT_FLOAT);
  TensorUtils::SetSize(tensor1, 602112);
  aipp->GetOpDesc()->AddInputDesc(tensor1);
  GeTensorDesc tensor2(GeShape(std::vector<int64_t>{2}), FORMAT_ND, DT_INT64);
  TensorUtils::SetSize(tensor2, 8);
  aipp->GetOpDesc()->AddInputDesc(tensor2);
  aipp->GetOpDesc()->AddOutputDesc(tensor1);
  aipp->GetOpDesc()->AddOutputDesc(tensor2);
  const auto &sub_graph = builder.GetGraph();

  const GeTensorDescPtr &input1 = aipp->GetOpDesc()->MutableInputDesc(0);
  ge::TensorUtils::SetSize(*input1, 602112);
  const GeTensorDescPtr &input2 = aipp->GetOpDesc()->MutableInputDesc(1);
  ge::TensorUtils::SetSize(*input2, 8);
  const GeTensorDescPtr &output1 = aipp->GetOpDesc()->MutableOutputDesc(0);
  ge::TensorUtils::SetSize(*output1, 602112);
  const GeTensorDescPtr &output2 = aipp->GetOpDesc()->MutableOutputDesc(1);
  ge::TensorUtils::SetSize(*output2, 8);
  EXPECT_EQ(instance.RecordAIPPInfoToData(sub_graph), SUCCESS);
}

} // namespace ge
