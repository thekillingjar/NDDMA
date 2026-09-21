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
#include "fe_llt_utils.h"
#define private public
#define protected public
#include "adapter/common/op_store_adapter_manager.h"
#include "adapter/tbe_adapter/tbe_op_store_adapter.h"
#include "graph/utils/attr_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/ge_local_context.h"
#include "graph_optimizer/op_judge/format_and_dtype/op_format_dtype_judge.h"
#include "graph_optimizer/op_judge/format_and_dtype/strategy/matcher/dtype/op_dtype_rise_matcher.h"
#include "graph_optimizer/op_judge/format_and_dtype/strategy/matcher/format/op_format_matcher.h"
#include "graph_optimizer/op_judge/imply_type/op_impl_type_judge.h"

#include "../../../../graph_constructor/graph_builder_utils.h"
#include "common/configuration.h"
#include "graph/debug/ge_attr_define.h"
#include "graph_optimizer/shape_format_transfer/trans_node_manager/trans_node_manager.h"
#include "ops_store/ops_kernel_manager.h"
using namespace std;
using namespace ge;
using namespace fe;
using OpImplTypeJudgePtr = std::shared_ptr<OpImplTypeJudge>;
using OpFormatDtypeJudgePtr = std::shared_ptr<OpFormatDtypeJudge>;
using OpDtypeRiseMatcherPtr = std::shared_ptr<OpDtypeRiseMatcher>;
using OpFormatMatcherPtr = std::shared_ptr<OpFormatMatcher>;

using TransNodeManagerPtr = std::shared_ptr<TransNodeManager>;

class UTEST_fusion_engine_op_judge_function_op : public testing::Test {
protected:
  void SetUp() {
    std::map<std::string, std::string> options;
    fe_ops_kernel_info_store_ptr_ = make_shared<fe::FEOpsKernelInfoStore>();

    FEOpsStoreInfo tbe_custom {6, "tbe-custom", EN_IMPL_HW_TBE,
                              GetCodeDir() + "/tests/engines/nn_engine/ut/testcase/fusion_engine/ops_kernel_store/fe_config/tbe_custom_opinfo",
                              ""};
    vector<FEOpsStoreInfo> store_info;

    store_info.emplace_back(tbe_custom);
    Configuration::Instance(fe::AI_CORE_NAME).ops_store_info_vector_ = (store_info);
    ge::GetThreadLocalContext().graph_options_[ge::PRECISION_MODE] = ALLOW_MIX_PRECISION;
    OpsKernelManager::Instance(AI_CORE_NAME).Finalize();

    fe_ops_kernel_info_store_ptr_->Initialize(options);

    reflection_builder_ptr_ = std::make_shared<ge::RefRelations>();
    op_impl_type_judge_ptr_ =
        std::make_shared<OpImplTypeJudge>(AI_CORE_NAME, fe_ops_kernel_info_store_ptr_);
    op_format_dtype_judge_ptr_ = std::make_shared<OpFormatDtypeJudge>(AI_CORE_NAME, reflection_builder_ptr_);
    op_format_dtype_judge_ptr_->Initialize();
  }

  void TearDown() {}

  static fe::Status AddSubgraphInstance(ge::NodePtr funtion_node_ptr,
                                        const int &sub_index,
                                        const std::string &sub_name) {
    FE_CHECK_NOTNULL(funtion_node_ptr);
    FE_CHECK_NOTNULL(funtion_node_ptr->GetOpDesc());
    funtion_node_ptr->GetOpDesc()->AddSubgraphName(sub_name);
    funtion_node_ptr->GetOpDesc()->SetSubgraphInstanceName(sub_index, sub_name);
    return fe::SUCCESS;
  }

  /*
   *          netoutput
   *          |        \
   *        add          add123
   *      /     \        /     \
   * data1(x)data2(y)  data1(x) data2(y)
   */
  static ge::ComputeGraphPtr BuildIfSubGraph(const std::string &name) {
    ut::ComputeGraphBuilder builder(name);
    auto data1 = builder.AddNode(name + "data1", fe::DATA, 1, 1, ge::FORMAT_ND,
                                 ge::DT_FLOAT);
    auto data2 = builder.AddNode(name + "data2", fe::DATA, 1, 1, ge::FORMAT_ND,
                                 ge::DT_FLOAT);
    auto add = builder.AddNode(name + "add1", "Add", 2, 1, ge::FORMAT_ND,
                               ge::DT_FLOAT);
    auto add123 = builder.AddNode(name + "add2", "Add123", 2, 1, ge::FORMAT_ND,
                                  ge::DT_FLOAT);
    auto netoutput = builder.AddNode(name + "netoutput", fe::NETOUTPUT, 2, 2,
                                     ge::FORMAT_ND, ge::DT_FLOAT);

    ge::AttrUtils::SetInt(data1->GetOpDesc(), ge::ATTR_NAME_PARENT_NODE_INDEX,
                          static_cast<int>(1));
    ge::AttrUtils::SetInt(data2->GetOpDesc(), ge::ATTR_NAME_PARENT_NODE_INDEX,
                          static_cast<int>(2));
    ge::AttrUtils::SetInt(netoutput->GetOpDesc()->MutableInputDesc(0),
                          ge::ATTR_NAME_PARENT_NODE_INDEX, static_cast<int>(0));
    ge::AttrUtils::SetInt(netoutput->GetOpDesc()->MutableInputDesc(1),
                          ge::ATTR_NAME_PARENT_NODE_INDEX, static_cast<int>(1));

    builder.AddDataEdge(data1, 0, add, 0);
    builder.AddDataEdge(data2, 0, add, 1);
    builder.AddDataEdge(data1, 0, add123, 0);
    builder.AddDataEdge(data2, 0, add123, 1);
    builder.AddDataEdge(add, 0, netoutput, 0);
    builder.AddDataEdge(add123, 0, netoutput, 1);
    return builder.GetGraph();
  }

  /*
   *      netoutput
   *         ||
   *         if2(if2_sub1,if2_sub2)
   *    /     |      \
   *  const data1(x) data2(y)
   */
  static ge::ComputeGraphPtr BuildContainIfSubGraph(ComputeGraphPtr main_graph, const std::string &name) {
    ut::ComputeGraphBuilder builder(name);
    auto condition = builder.AddNode(name + "condition", fe::CONSTANT, 0, 1,
                                 ge::FORMAT_ND, ge::DT_FLOAT);
    auto data1 = builder.AddNode(name + "data1", fe::DATA, 1, 1, ge::FORMAT_ND,
                                 ge::DT_FLOAT);
    auto data2 = builder.AddNode(name + "data2", fe::DATA, 1, 1, ge::FORMAT_ND,
                                 ge::DT_FLOAT);
    auto if2 =
        builder.AddNode(name + "if2", "If", 3, 2, ge::FORMAT_ND, ge::DT_FLOAT);
    auto netoutput = builder.AddNode(name + "netoutput", fe::NETOUTPUT, 2, 2,
                                     ge::FORMAT_ND, ge::DT_FLOAT);
    ge::AttrUtils::SetInt(data1->GetOpDesc(), ge::ATTR_NAME_PARENT_NODE_INDEX,
                          static_cast<int>(1));
    ge::AttrUtils::SetInt(data2->GetOpDesc(), ge::ATTR_NAME_PARENT_NODE_INDEX,
                          static_cast<int>(2));
    ge::AttrUtils::SetInt(netoutput->GetOpDesc()->MutableInputDesc(0),
                          ge::ATTR_NAME_PARENT_NODE_INDEX, static_cast<int>(0));
    ge::AttrUtils::SetInt(netoutput->GetOpDesc()->MutableInputDesc(1),
                          ge::ATTR_NAME_PARENT_NODE_INDEX, static_cast<int>(1));

    builder.AddDataEdge(condition, 0, if2, 0);
    builder.AddDataEdge(data1, 0, if2, 1);
    builder.AddDataEdge(data2, 0, if2, 2);
    builder.AddDataEdge(if2, 0, netoutput, 0);
    builder.AddDataEdge(if2, 1, netoutput, 1);

    auto sub_graph = builder.GetGraph();
    vector<std::string> sub_names = {"if2_sub1", "if2_sub2"};
    AddIfSubGraph(main_graph, sub_graph, name + "if2", sub_names);
    return sub_graph;
  }

  static void AddIfSubGraph(ge::ComputeGraphPtr main_graph,
                            ge::ComputeGraphPtr parent_graph,
                            const std::string &fuction_name,
                            const vector<std::string> &sub_names) {
    ge::NodePtr funtion_node_ptr = parent_graph->FindNode(fuction_name);
    for (int i = 0; i != sub_names.size(); ++i) {
      string sub_name = sub_names[i];
      auto sub = BuildIfSubGraph(sub_name);
      sub->SetParentGraph(parent_graph);
      sub->SetParentNode(funtion_node_ptr);
      AddSubgraphInstance(funtion_node_ptr, i, sub_name);
      main_graph->AddSubgraph(sub_name, sub);
    }
  }

  /*
   *            netoutput
   *                ||
   *            if(sub1, sub2)
   *          /        \       \
   *        /         square  square123
   *      /              \        \
   * const(condition) data1(x)  data2(y)
   */
  static ge::ComputeGraphPtr BuildMainGraphWithIf() {
    ut::ComputeGraphBuilder builder("main_graph");
    auto data1 =
        builder.AddNode("data1", fe::DATA, 1, 1, ge::FORMAT_ND, ge::DT_FLOAT);
    auto data2 =
        builder.AddNode("data2", fe::DATA, 1, 1, ge::FORMAT_ND, ge::DT_FLOAT);
    auto condition = builder.AddNode("condition", fe::CONSTANT, 0, 1,
                                     ge::FORMAT_ND, ge::DT_FLOAT);
    auto square =
        builder.AddNode("square", "Square", 1, 1, ge::FORMAT_ND, ge::DT_FLOAT);
    auto square123 = builder.AddNode("square123", "Square123", 1, 1,
                                     ge::FORMAT_ND, ge::DT_FLOAT);

    auto if1 = builder.AddNode("if", "If", 3, 2, ge::FORMAT_ND, ge::DT_FLOAT);
    auto netoutput1 = builder.AddNode("netoutput", fe::NETOUTPUT, 2, 2,
                                      ge::FORMAT_ND, ge::DT_FLOAT);

    builder.AddDataEdge(condition, 0, if1, 0);
    builder.AddDataEdge(data1, 0, square, 0);
    builder.AddDataEdge(data2, 0, square123, 0);
    builder.AddDataEdge(square, 0, if1, 1);
    builder.AddDataEdge(square123, 0, if1, 2);

    builder.AddDataEdge(if1, 0, netoutput1, 0);
    builder.AddDataEdge(if1, 1, netoutput1, 1);

    auto main_graph = builder.GetGraph();
    vector<std::string> sub_names = {"sub1", "sub2"};
    AddIfSubGraph(main_graph, main_graph, "if", sub_names);
    return main_graph;
  }

  /*
   *                netoutput
   *                   ||
   *            if1(if1_sub1,if1_sub2)
   *            /    \        \
   *           /     square  square123
   *          /         \         \
   * const(condition1) data1(x)  data2(y)
   */
  static ge::ComputeGraphPtr BuildMainGraphWithIfContainIf() {
    ut::ComputeGraphBuilder builder("main_graph");
    auto condition1 = builder.AddNode("condition1", fe::CONSTANT, 0, 1,
                                      ge::FORMAT_ND, ge::DT_FLOAT);
    auto data1 =
        builder.AddNode("data1", fe::DATA, 1, 1, ge::FORMAT_ND, ge::DT_FLOAT);
    auto data2 =
        builder.AddNode("data2", fe::DATA, 1, 1, ge::FORMAT_ND, ge::DT_FLOAT);
    auto square =
        builder.AddNode("square", "Square", 1, 1, ge::FORMAT_ND, ge::DT_FLOAT);
    auto square123 = builder.AddNode("square123", "Square123", 1, 1,
                                     ge::FORMAT_ND, ge::DT_FLOAT);

    auto if1 = builder.AddNode("if1", "If", 3, 2, ge::FORMAT_ND, ge::DT_FLOAT);
    auto netoutput = builder.AddNode("netoutput", "netoutput", 2, 2,
                                     ge::FORMAT_ND, ge::DT_FLOAT);

    builder.AddDataEdge(condition1, 0, if1, 0);
    builder.AddDataEdge(data1, 0, square, 0);
    builder.AddDataEdge(data2, 0, square123, 0);
    builder.AddDataEdge(square, 0, if1, 1);
    builder.AddDataEdge(square123, 0, if1, 2);
    builder.AddDataEdge(if1, 0, netoutput, 0);
    builder.AddDataEdge(if1, 1, netoutput, 1);

    auto main_graph = builder.GetGraph();
    auto sub1 = BuildContainIfSubGraph(main_graph, "if1_sub1");
    sub1->SetParentGraph(main_graph);
    sub1->SetParentNode(if1);
    AddSubgraphInstance(if1, 0, "if1_sub1");
    main_graph->AddSubgraph("if1_sub1", sub1);

    auto sub2 = BuildIfSubGraph("if1_sub2");
    sub2->SetParentGraph(main_graph);
    sub2->SetParentNode(if1);
    AddSubgraphInstance(if1, 1, "if1_sub2");
    main_graph->AddSubgraph("if1_sub2", sub2);

    return main_graph;
  }

  /*
   *      netoutput
   *          |
   *        greater123
   *        |      \
   *    data2(x)   add
   *             /   \
   *         const1   data3(y)
   */
  static ge::ComputeGraphPtr BuildWhileCondSubGraph(const std::string name) {
    ut::ComputeGraphBuilder builder(name);
    auto data2 = builder.AddNode(name + "data2", fe::DATA, 1, 1);
    auto const1 = builder.AddNode(name + "const1", fe::CONSTANT, 1, 1);
    auto data3 = builder.AddNode(name + "data3", fe::DATA, 1, 1);
    auto add = builder.AddNode(name + "add", "Add", 2, 1);
    auto greater = builder.AddNode(name + "greater", "Greater123", 2, 1);
    auto netoutput = builder.AddNode(name + "netoutput", fe::NETOUTPUT, 1, 1);

    ge::AttrUtils::SetInt(data2->GetOpDesc(), ge::ATTR_NAME_PARENT_NODE_INDEX,
                          static_cast<int>(1));
    ge::AttrUtils::SetInt(data3->GetOpDesc(), ge::ATTR_NAME_PARENT_NODE_INDEX,
                          static_cast<int>(2));

    builder.AddDataEdge(const1, 0, add, 0);
    builder.AddDataEdge(data3, 0, add, 1);

    builder.AddDataEdge(data2, 0, greater, 0);
    builder.AddDataEdge(add, 0, greater, 1);

    builder.AddDataEdge(greater, 0, netoutput, 0);
    return builder.GetGraph();
  }

  /*
   *               netoutput
   *             /     |    \
   *          add    relu123 memcpyasync
   *         /   \     |        \
   *   data1(n)const1 data2(x) data3(y)
   */
  static ge::ComputeGraphPtr BuildWhileBodySubGraph(const std::string name) {
    ut::ComputeGraphBuilder builder(name);
    auto data1 = builder.AddNode(name + "data1", fe::DATA, 1, 1);
    auto data2 = builder.AddNode(name + "data2", fe::DATA, 1, 1);
    auto data3 = builder.AddNode(name + "data3", fe::DATA, 1, 1);
    auto const1 = builder.AddNode(name + "const1", fe::CONSTANT, 1, 1);

    auto add = builder.AddNode(name + "add", "Add", 2, 1);
    auto relu = builder.AddNode(name + "relu", "Relu123", 1, 1);
    auto memcpy_async =
        builder.AddNode(name + "memcpyAsync", "MemcpyAsync", 1, 1);
    auto netoutput = builder.AddNode(name + "netoutput", fe::NETOUTPUT, 3, 3);

    ge::AttrUtils::SetInt(data1->GetOpDesc(), ge::ATTR_NAME_PARENT_NODE_INDEX,
                          static_cast<int>(0));
    ge::AttrUtils::SetInt(data2->GetOpDesc(), ge::ATTR_NAME_PARENT_NODE_INDEX,
                          static_cast<int>(1));
    ge::AttrUtils::SetInt(data3->GetOpDesc(), ge::ATTR_NAME_PARENT_NODE_INDEX,
                          static_cast<int>(2));
    ge::AttrUtils::SetInt(netoutput->GetOpDesc()->MutableInputDesc(0),
                          ge::ATTR_NAME_PARENT_NODE_INDEX, static_cast<int>(0));
    ge::AttrUtils::SetInt(netoutput->GetOpDesc()->MutableInputDesc(1),
                          ge::ATTR_NAME_PARENT_NODE_INDEX, static_cast<int>(1));
    ge::AttrUtils::SetInt(netoutput->GetOpDesc()->MutableInputDesc(2),
                          ge::ATTR_NAME_PARENT_NODE_INDEX, static_cast<int>(2));

    builder.AddDataEdge(data1, 0, add, 0);
    builder.AddDataEdge(const1, 0, add, 1);
    builder.AddDataEdge(data2, 0, relu, 0);
    builder.AddDataEdge(data3, 0, memcpy_async, 0);

    builder.AddDataEdge(add, 0, netoutput, 0);
    builder.AddDataEdge(relu, 0, netoutput, 1);
    builder.AddDataEdge(memcpy_async, 0, netoutput, 2);
    return builder.GetGraph();
  }

  /*
   *       netoutput
   *        |   |  \
   *      add1  |  add2
   *      /  \  |  /   \
   * const1   while   const2
   *        /   |   \
   *      /  square  square123
   *    /       |       \
   * data1(n) data2(x) data3(y)
   */
  static ge::ComputeGraphPtr BuildMainGraphWithWhile() {
    ut::ComputeGraphBuilder builder("main_graph");
    auto data1 = builder.AddNode("n", fe::DATA, 1, 1);
    auto data2 = builder.AddNode("x", fe::DATA, 1, 1);
    auto data3 = builder.AddNode("y", fe::DATA, 1, 1);

    auto while1 = builder.AddNode("while", "While", 3, 3);
    auto netoutput = builder.AddNode("netoutput", fe::NETOUTPUT, 2, 2);
    auto square = builder.AddNode("square", "Square", 1, 1);
    auto square123 = builder.AddNode("square123", "Square123", 1, 1);

    auto const1 = builder.AddNode("const1", fe::CONSTANT, 1, 1);
    auto const2 = builder.AddNode("const2", fe::CONSTANT, 1, 1);
    auto add1 = builder.AddNode("add1", "Add", 2, 2);
    auto add2 = builder.AddNode("add2", "Add", 2, 2);

    builder.AddDataEdge(data1, 0, while1, 0);
    builder.AddDataEdge(data2, 0, square, 0);
    builder.AddDataEdge(data3, 0, square123, 0);
    builder.AddDataEdge(square, 0, while1, 1);
    builder.AddDataEdge(square123, 0, while1, 2);

    builder.AddDataEdge(const1, 0, add1, 0);
    builder.AddDataEdge(while1, 0, add1, 1);
    builder.AddDataEdge(add1, 0, netoutput, 0);

    builder.AddDataEdge(while1, 1, netoutput, 1);

    builder.AddDataEdge(while1, 2, add2, 0);
    builder.AddDataEdge(const2, 0, add2, 1);
    builder.AddDataEdge(add2, 0, netoutput, 2);

    auto main_graph = builder.GetGraph();
    auto sub1 = BuildWhileCondSubGraph("sub1");
    sub1->SetParentGraph(main_graph);
    sub1->SetParentNode(while1);
    AddSubgraphInstance(while1, 0, "sub1");
    main_graph->AddSubgraph("sub1", sub1);

    auto sub2 = BuildWhileBodySubGraph("sub2");
    sub2->SetParentGraph(main_graph);
    sub2->SetParentNode(while1);
    AddSubgraphInstance(while1, 1, "sub2");
    main_graph->AddSubgraph("sub2", sub2);
    return main_graph;
  }

  static fe::Status CheckIfResult(const ge::ComputeGraphPtr &main_graph) {
    bool is_updated = true;
    ge::DataType if_cond_dtype = ge::DT_FLOAT;
    ge::DataType if_data1_dtype = ge::DT_FLOAT16;
    ge::DataType if_data2_dtype = ge::DT_FLOAT;
    ge::DataType if_output0_dtype = ge::DT_FLOAT16;
    ge::DataType if_output1_dtype = ge::DT_FLOAT;

    auto if_node = main_graph->FindNode("if");
    FE_CHECK_NOTNULL(if_node);
    auto if_op_desc = if_node->GetOpDesc();
    FE_CHECK_NOTNULL(if_op_desc);

    // if.cond
    auto if_cond_input_desc = if_op_desc->GetInputDesc(0);
    EXPECT_EQ(ge::AttrUtils::HasAttr(if_cond_input_desc, ATTR_NAME_DTYPE_IS_UPDATED), !is_updated);
    EXPECT_EQ(if_cond_input_desc.GetDataType(), if_cond_dtype);

    // if.data1
    auto if_data1_input_desc = if_op_desc->GetInputDesc(1);
    EXPECT_EQ(ge::AttrUtils::HasAttr(if_data1_input_desc, ATTR_NAME_DTYPE_IS_UPDATED),
              is_updated);
    EXPECT_EQ(if_data1_input_desc.GetDataType(), if_data1_dtype);

    // if.data2
    auto if_data2_input_desc = if_op_desc->GetInputDesc(2);
    EXPECT_EQ(ge::AttrUtils::HasAttr(if_data2_input_desc, ATTR_NAME_DTYPE_IS_UPDATED),
              is_updated);
    EXPECT_EQ(if_data2_input_desc.GetDataType(), if_data2_dtype);

    // if.netoutput
    auto if_output_desc0 = if_op_desc->GetOutputDesc(0);
    EXPECT_EQ(ge::AttrUtils::HasAttr(if_output_desc0, ATTR_NAME_DTYPE_IS_UPDATED),
              is_updated);
    EXPECT_EQ(if_output_desc0.GetDataType(), if_output0_dtype);
    auto if_output_desc1 = if_op_desc->GetOutputDesc(1);
    EXPECT_EQ(ge::AttrUtils::HasAttr(if_output_desc1, ATTR_NAME_DTYPE_IS_UPDATED),
              is_updated);
    EXPECT_EQ(if_output_desc1.GetDataType(), if_output1_dtype);

    // subgraph
    for (const auto &sub_graph : main_graph->GetAllSubgraphs()) {
      auto sub_graph_name = sub_graph->GetName();
      // data1
      auto data1 = sub_graph->FindNode(sub_graph_name + "data1");
      FE_CHECK_NOTNULL(data1);
      FE_CHECK_NOTNULL(data1->GetOpDesc());
      auto data1_output_desc = data1->GetOpDesc()->GetOutputDesc(0);
      EXPECT_EQ(ge::AttrUtils::HasAttr(data1_output_desc, ATTR_NAME_DTYPE_IS_UPDATED),
                is_updated);
      EXPECT_EQ(data1_output_desc.GetDataType(), if_data1_dtype);

      // data2
      auto data2 = sub_graph->FindNode(sub_graph_name + "data2");
      FE_CHECK_NOTNULL(data2);
      FE_CHECK_NOTNULL(data2->GetOpDesc());
      auto data2_output_desc = data2->GetOpDesc()->GetOutputDesc(0);
      EXPECT_EQ(ge::AttrUtils::HasAttr(data2_output_desc,
                                       ATTR_NAME_DTYPE_IS_UPDATED),
                is_updated);
      EXPECT_EQ(data2_output_desc.GetDataType(), if_data2_dtype);

      // netoutput
      auto netoutput = sub_graph->FindNode(sub_graph_name + "netoutput");
      FE_CHECK_NOTNULL(netoutput);
      auto net_output_op_desc = netoutput->GetOpDesc();
      FE_CHECK_NOTNULL(net_output_op_desc);
      auto netoutput0 = net_output_op_desc->GetInputDesc(0);
      EXPECT_EQ(ge::AttrUtils::HasAttr(netoutput0,ATTR_NAME_DTYPE_IS_UPDATED),
                is_updated);
      EXPECT_EQ(netoutput0.GetDataType(), if_output0_dtype);
      auto netoutput1 =net_output_op_desc->GetInputDesc(1);
      EXPECT_EQ(ge::AttrUtils::HasAttr(netoutput1,ATTR_NAME_DTYPE_IS_UPDATED),
                is_updated);
      EXPECT_EQ(netoutput1.GetDataType(), if_output1_dtype);
    }
    return fe::SUCCESS;
  }

  static fe::Status CheckWhileResult(const ge::ComputeGraphPtr &main_graph) {
    bool is_updated = true;
    ge::DataType while_data1_dtype = ge::DT_FLOAT;
    ge::DataType while_data2_dtype = ge::DT_FLOAT16;
    ge::DataType while_data3_dtype = ge::DT_FLOAT;
    ge::DataType while_output0_dtype = while_data1_dtype;
    ge::DataType while_output1_dtype = while_data2_dtype;
    ge::DataType while_output2_dtype = while_data3_dtype;
    auto while_node = main_graph->FindNode("while");
    FE_CHECK_NOTNULL(while_node);
    auto while_op_desc = while_node->GetOpDesc();
    FE_CHECK_NOTNULL(while_op_desc);

    // while.data1
    auto while_data1_input_desc = while_op_desc->GetInputDesc(0);
    EXPECT_EQ(ge::AttrUtils::HasAttr(while_data1_input_desc, ATTR_NAME_DTYPE_IS_UPDATED), is_updated);
    EXPECT_EQ(while_data1_input_desc.GetDataType(), while_data1_dtype);
    // while.data2
    auto while_data2_input_desc = while_op_desc->GetInputDesc(1);
    EXPECT_EQ(ge::AttrUtils::HasAttr(while_data2_input_desc, ATTR_NAME_DTYPE_IS_UPDATED), is_updated);
    EXPECT_EQ(while_data2_input_desc.GetDataType(), while_data2_dtype);
    // while.data3
    auto while_data3_input_desc = while_op_desc->GetInputDesc(2);
    EXPECT_EQ(ge::AttrUtils::HasAttr(while_data3_input_desc, ATTR_NAME_DTYPE_IS_UPDATED), is_updated);
    EXPECT_EQ(while_data3_input_desc.GetDataType(), while_data3_dtype);

    // while.netouput
    auto while_output_desc0 = while_op_desc->GetOutputDesc(0);
    EXPECT_EQ(ge::AttrUtils::HasAttr(while_output_desc0, ATTR_NAME_DTYPE_IS_UPDATED), is_updated);
    EXPECT_EQ(while_output_desc0.GetDataType(), while_output0_dtype);
    auto while_output_desc1 = while_op_desc->GetOutputDesc(1);
    EXPECT_EQ(ge::AttrUtils::HasAttr(while_output_desc1, ATTR_NAME_DTYPE_IS_UPDATED), is_updated);
    EXPECT_EQ(while_output_desc1.GetDataType(), while_output1_dtype);
    auto while_output_desc2 = while_op_desc->GetOutputDesc(2);
    EXPECT_EQ(ge::AttrUtils::HasAttr(while_output_desc2, ATTR_NAME_DTYPE_IS_UPDATED), is_updated);
    EXPECT_EQ(while_output_desc2.GetDataType(), while_output2_dtype);

    // condition graph
    auto cond_subgraph = main_graph->GetSubgraph("sub1");
    auto cond_subgraph_name = cond_subgraph->GetName();
    auto cond_data2 = cond_subgraph->FindNode(cond_subgraph_name + "data2");
    FE_CHECK_NOTNULL(cond_data2);
    FE_CHECK_NOTNULL(cond_data2->GetOpDesc());
    auto cond_data2_output_desc = cond_data2->GetOpDesc()->GetOutputDesc(0);
    EXPECT_EQ(ge::AttrUtils::HasAttr(cond_data2_output_desc, ATTR_NAME_DTYPE_IS_UPDATED), is_updated);
    EXPECT_EQ(cond_data2_output_desc.GetDataType(), while_data2_dtype);

    auto cond_data3 = cond_subgraph->FindNode(cond_subgraph_name + "data3");
    FE_CHECK_NOTNULL(cond_data3);
    FE_CHECK_NOTNULL(cond_data3->GetOpDesc());
    auto cond_data3_output_desc = cond_data3->GetOpDesc()->GetOutputDesc(0);
    EXPECT_EQ(ge::AttrUtils::HasAttr(cond_data3_output_desc, ATTR_NAME_DTYPE_IS_UPDATED), is_updated);
    EXPECT_EQ(cond_data3_output_desc.GetDataType(), while_data3_dtype);

    auto cond_netoutput = cond_subgraph->FindNode(cond_subgraph_name + "netoutput");
    FE_CHECK_NOTNULL(cond_netoutput);
    auto cond_netoutput_opdesc = cond_netoutput->GetOpDesc();
    FE_CHECK_NOTNULL(cond_netoutput_opdesc);
    auto cond_netoutput0 = cond_netoutput_opdesc->GetInputDesc(0);
    EXPECT_EQ(ge::AttrUtils::HasAttr(cond_netoutput0, ATTR_NAME_DTYPE_IS_UPDATED), !is_updated);
    EXPECT_EQ(cond_netoutput0.GetDataType(), DT_FLOAT);

    auto cond_netoutput1 = cond_netoutput_opdesc->GetInputDesc(1);
    EXPECT_EQ(ge::AttrUtils::HasAttr(cond_netoutput1, ATTR_NAME_DTYPE_IS_UPDATED), !is_updated);
    EXPECT_EQ(cond_netoutput1.GetDataType(), DT_FLOAT);

    auto cond_netoutput2 = cond_netoutput_opdesc->GetInputDesc(2);
    EXPECT_EQ(ge::AttrUtils::HasAttr(cond_netoutput2, ATTR_NAME_DTYPE_IS_UPDATED), !is_updated);
    EXPECT_EQ(cond_netoutput2.GetDataType(),DT_FLOAT);

    // body subgraph
    auto body_subgraph = main_graph->GetSubgraph("sub2");
    auto body_subgraph_name = body_subgraph->GetName();
    auto body_data1 = body_subgraph->FindNode(body_subgraph_name + "data1");
    FE_CHECK_NOTNULL(body_data1);
    FE_CHECK_NOTNULL(body_data1->GetOpDesc());
    auto body_data1_output_desc = body_data1->GetOpDesc()->GetOutputDesc(0);
    EXPECT_EQ(ge::AttrUtils::HasAttr(body_data1_output_desc, ATTR_NAME_DTYPE_IS_UPDATED), is_updated);
    EXPECT_EQ(body_data1_output_desc.GetDataType(), while_data1_dtype);

    auto body_data2 = body_subgraph->FindNode(body_subgraph_name + "data2");
    FE_CHECK_NOTNULL(body_data2);
    FE_CHECK_NOTNULL(body_data2->GetOpDesc());
    auto body_data2_output_desc = body_data2->GetOpDesc()->GetOutputDesc(0);
    EXPECT_EQ(ge::AttrUtils::HasAttr(body_data2_output_desc, ATTR_NAME_DTYPE_IS_UPDATED), is_updated);
    EXPECT_EQ(body_data2_output_desc.GetDataType(), while_data2_dtype);

    auto body_data3 = body_subgraph->FindNode(body_subgraph_name + "data3");
    FE_CHECK_NOTNULL(body_data3);
    FE_CHECK_NOTNULL(body_data3->GetOpDesc());
    auto body_data3_output_desc = body_data3->GetOpDesc()->GetOutputDesc(0);
    EXPECT_EQ(ge::AttrUtils::HasAttr(body_data3_output_desc, ATTR_NAME_DTYPE_IS_UPDATED), is_updated);
    EXPECT_EQ(body_data3_output_desc.GetDataType(), while_data3_dtype);

    auto body_netoutput = body_subgraph->FindNode(body_subgraph_name + "netoutput");
    FE_CHECK_NOTNULL(body_netoutput);
    auto body_netoutput_op_desc = body_netoutput->GetOpDesc();
    FE_CHECK_NOTNULL(body_netoutput_op_desc);
    auto body_netouput0 = body_netoutput_op_desc->GetInputDesc(0);
    EXPECT_EQ(ge::AttrUtils::HasAttr(body_netouput0, ATTR_NAME_DTYPE_IS_UPDATED), is_updated);
    EXPECT_EQ(body_netouput0.GetDataType(), while_output0_dtype);

    auto body_netouput1 = body_netoutput_op_desc->GetInputDesc(1);
    EXPECT_EQ(ge::AttrUtils::HasAttr(body_netouput1, ATTR_NAME_DTYPE_IS_UPDATED), is_updated);
    EXPECT_EQ(body_netouput1.GetDataType(), while_output1_dtype);

    auto body_netouput2 = body_netoutput_op_desc->GetInputDesc(2);
    EXPECT_EQ(ge::AttrUtils::HasAttr(body_netouput2, ATTR_NAME_DTYPE_IS_UPDATED), is_updated);
    EXPECT_EQ(body_netouput2.GetDataType(), while_output2_dtype);

    return fe::SUCCESS;
  }

  static fe::Status CheckIfContainIfResult(const ge::ComputeGraphPtr &main_graph) {
    bool is_updated = true;
    ge::DataType if_cond_dtype = ge::DT_FLOAT;
    ge::DataType if_data1_dtype = ge::DT_FLOAT16;
    ge::DataType if_data2_dtype = ge::DT_FLOAT;
    ge::DataType if_output0_dtype = ge::DT_FLOAT16;
    ge::DataType if_output1_dtype = ge::DT_FLOAT;

    auto if1_node = main_graph->FindNode("if1");
    FE_CHECK_NOTNULL(if1_node);
    auto if1_op_desc = if1_node->GetOpDesc();
    FE_CHECK_NOTNULL(if1_op_desc);

    // if1.cond
    auto if1_cond_input_desc = if1_op_desc->GetInputDesc(0);
    EXPECT_EQ(ge::AttrUtils::HasAttr(if1_cond_input_desc, ATTR_NAME_DTYPE_IS_UPDATED), !is_updated);
    EXPECT_EQ(if1_cond_input_desc.GetDataType(), if_cond_dtype);

    // if1.data1
    auto if1_data1_input_desc = if1_op_desc->GetInputDesc(1);
    EXPECT_EQ(ge::AttrUtils::HasAttr(if1_data1_input_desc, ATTR_NAME_DTYPE_IS_UPDATED),
              is_updated);
    EXPECT_EQ(if1_data1_input_desc.GetDataType(), if_data1_dtype);

    // if1.data2
    auto if1_data2_input_desc = if1_op_desc->GetInputDesc(2);
    EXPECT_EQ(ge::AttrUtils::HasAttr(if1_data2_input_desc, ATTR_NAME_DTYPE_IS_UPDATED),
              is_updated);
    EXPECT_EQ(if1_data2_input_desc.GetDataType(), if_data2_dtype);

    // if1.netoutput
    auto if1_output_desc0 = if1_op_desc->GetOutputDesc(0);
    EXPECT_EQ(ge::AttrUtils::HasAttr(if1_output_desc0, ATTR_NAME_DTYPE_IS_UPDATED),
              is_updated);
    EXPECT_EQ(if1_output_desc0.GetDataType(), if_output0_dtype);
    auto if1_output_desc1 = if1_op_desc->GetOutputDesc(1);
    EXPECT_EQ(ge::AttrUtils::HasAttr(if1_output_desc1, ATTR_NAME_DTYPE_IS_UPDATED),
              is_updated);
    EXPECT_EQ(if1_output_desc1.GetDataType(), if_output1_dtype);

    // if1_sub1
    auto if1_sub1 = main_graph->GetSubgraph("if1_sub1");
    auto if1_sub1_name = if1_sub1->GetName();
    // if1_sub1.data1
    auto if1_sub1_data1 = if1_sub1->FindNode(if1_sub1_name + "data1");
    FE_CHECK_NOTNULL(if1_sub1_data1);
    FE_CHECK_NOTNULL(if1_sub1_data1->GetOpDesc());
    auto if1_sub1_data1_output_desc = if1_sub1_data1->GetOpDesc()->GetOutputDesc(0);
    EXPECT_EQ(ge::AttrUtils::HasAttr(if1_sub1_data1_output_desc, ATTR_NAME_DTYPE_IS_UPDATED),
              is_updated);
    EXPECT_EQ(if1_sub1_data1_output_desc.GetDataType(), if_data1_dtype);

    // if1_sub1.data2
    auto if1_sub1_data2 = if1_sub1->FindNode(if1_sub1_name + "data2");
    FE_CHECK_NOTNULL(if1_sub1_data2);
    FE_CHECK_NOTNULL(if1_sub1_data2->GetOpDesc());
    auto if1_sub1_data2_output_desc = if1_sub1_data2->GetOpDesc()->GetOutputDesc(0);
    EXPECT_EQ(ge::AttrUtils::HasAttr(if1_sub1_data2_output_desc,
                                     ATTR_NAME_DTYPE_IS_UPDATED),
              is_updated);
    EXPECT_EQ(if1_sub1_data2_output_desc.GetDataType(), if_data2_dtype);

    // if1_sub1.netoutput
    auto if1_sub1_netoutput = if1_sub1->FindNode(if1_sub1_name + "netoutput");
    FE_CHECK_NOTNULL(if1_sub1_netoutput);
    auto if1_sub1_net_output_op_desc = if1_sub1_netoutput->GetOpDesc();
    FE_CHECK_NOTNULL(if1_sub1_net_output_op_desc);
    auto if1_sub1_netoutput0 = if1_sub1_net_output_op_desc->GetInputDesc(0);
    EXPECT_EQ(ge::AttrUtils::HasAttr(if1_sub1_netoutput0,ATTR_NAME_DTYPE_IS_UPDATED),
              is_updated);
    EXPECT_EQ(if1_sub1_netoutput0.GetDataType(), if_output0_dtype);
    auto if1_sub1_netoutput1 = if1_sub1_net_output_op_desc->GetInputDesc(1);
    EXPECT_EQ(ge::AttrUtils::HasAttr(if1_sub1_netoutput1,ATTR_NAME_DTYPE_IS_UPDATED),
              is_updated);
    EXPECT_EQ(if1_sub1_netoutput1.GetDataType(), if_output1_dtype);

    auto if1_sub2 = main_graph->GetSubgraph("if1_sub2");
    vector<ge::ComputeGraphPtr> all_subgraphs;
    all_subgraphs.push_back(if1_sub2);
    auto if1_sub1_if2 = if1_sub1->FindNode(if1_sub1->GetName()+"if2");
    FE_CHECK_NOTNULL(if1_sub1_if2);
    for(auto& subgraph:if1_sub1->GetAllSubgraphs()){
      all_subgraphs.push_back(subgraph);
    }

    for (const auto &subgraph : all_subgraphs) {
      auto sub_name = subgraph->GetName();
      // data1
      auto sub_data1 = subgraph->FindNode(sub_name + "data1");
      FE_CHECK_NOTNULL(sub_data1);
      FE_CHECK_NOTNULL(sub_data1->GetOpDesc());
      auto sub_data1_output_desc = sub_data1->GetOpDesc()->GetOutputDesc(0);
      EXPECT_EQ(ge::AttrUtils::HasAttr(sub_data1_output_desc, ATTR_NAME_DTYPE_IS_UPDATED),
                is_updated);
      EXPECT_EQ(sub_data1_output_desc.GetDataType(), if_data1_dtype);

      // data2
      auto sub_data2 = subgraph->FindNode(sub_name + "data2");
      FE_CHECK_NOTNULL(sub_data2);
      FE_CHECK_NOTNULL(sub_data2->GetOpDesc());
      auto sub_data2_output_desc = sub_data2->GetOpDesc()->GetOutputDesc(0);
      EXPECT_EQ(ge::AttrUtils::HasAttr(sub_data2_output_desc,
                                       ATTR_NAME_DTYPE_IS_UPDATED),
                is_updated);
      EXPECT_EQ(sub_data2_output_desc.GetDataType(), if_data2_dtype);

      // netoutput
      auto sub_netoutput = subgraph->FindNode(sub_name + "netoutput");
      FE_CHECK_NOTNULL(sub_netoutput);
      auto sub_net_output_op_desc = sub_netoutput->GetOpDesc();
      FE_CHECK_NOTNULL(sub_net_output_op_desc);
      auto sub_netoutput0 = sub_net_output_op_desc->GetInputDesc(0);
      EXPECT_EQ(ge::AttrUtils::HasAttr(sub_netoutput0,ATTR_NAME_DTYPE_IS_UPDATED),
                is_updated);
      EXPECT_EQ(sub_netoutput0.GetDataType(), if_output0_dtype);
      auto sub_netoutput1 = sub_net_output_op_desc->GetInputDesc(1);
      EXPECT_EQ(ge::AttrUtils::HasAttr(sub_netoutput1,ATTR_NAME_DTYPE_IS_UPDATED),
                is_updated);
      EXPECT_EQ(sub_netoutput1.GetDataType(), if_output1_dtype);
    }
    return fe::SUCCESS;
  }

  shared_ptr<fe::FEOpsKernelInfoStore> fe_ops_kernel_info_store_ptr_;
  OpImplTypeJudgePtr op_impl_type_judge_ptr_;
  OpFormatDtypeJudgePtr op_format_dtype_judge_ptr_;
  RefRelationsPtr reflection_builder_ptr_;
};

TEST_F(UTEST_fusion_engine_op_judge_function_op, if_success) {
  ge::GetThreadLocalContext().graph_options_[ge::PRECISION_MODE] = ALLOW_FP32_TO_FP16;

  auto main_graph = BuildMainGraphWithIf();
  main_graph->TopologicalSorting();

  (void)reflection_builder_ptr_->Clear();
  auto status = reflection_builder_ptr_->BuildRefRelations(*main_graph);

  EXPECT_EQ(status, ge::GRAPH_SUCCESS);
  Status ret = op_impl_type_judge_ptr_->MultiThreadJudge(*(main_graph.get()));
  ASSERT_EQ(ret, fe::SUCCESS);
  ret = op_format_dtype_judge_ptr_->Judge(*(main_graph.get()));
  ASSERT_EQ(ret, fe::SUCCESS);
  ret = op_format_dtype_judge_ptr_->SetFormat(*(main_graph.get()));
  ASSERT_EQ(ret, fe::SUCCESS);

  ASSERT_EQ(CheckIfResult(main_graph), fe::SUCCESS);
}

TEST_F(UTEST_fusion_engine_op_judge_function_op, while_success) {
  ge::GetThreadLocalContext().graph_options_[ge::PRECISION_MODE] = ALLOW_FP32_TO_FP16;

  auto main_graph = BuildMainGraphWithWhile();
  main_graph->TopologicalSorting();

  (void)reflection_builder_ptr_->Clear();
  auto status = reflection_builder_ptr_->BuildRefRelations(*main_graph);
  EXPECT_EQ(status, ge::GRAPH_SUCCESS);

  Status ret = op_impl_type_judge_ptr_->MultiThreadJudge(*(main_graph.get()));
  ASSERT_EQ(ret, fe::SUCCESS);
  ret = op_format_dtype_judge_ptr_->Judge(*(main_graph.get()));
  ASSERT_EQ(ret, fe::SUCCESS);
  ret = op_format_dtype_judge_ptr_->SetFormat(*(main_graph.get()));
  ASSERT_EQ(ret, fe::SUCCESS);
  ASSERT_EQ(CheckWhileResult(main_graph), fe::SUCCESS);
}

TEST_F(UTEST_fusion_engine_op_judge_function_op, if_contain_if_success) {
  ge::GetThreadLocalContext().graph_options_[ge::PRECISION_MODE] = ALLOW_FP32_TO_FP16;

  auto main_graph = BuildMainGraphWithIfContainIf();
  main_graph->TopologicalSorting();

  (void)reflection_builder_ptr_->Clear();
  auto status = reflection_builder_ptr_->BuildRefRelations(*main_graph);
  EXPECT_EQ(status, ge::GRAPH_SUCCESS);

  Status ret = op_impl_type_judge_ptr_->MultiThreadJudge(*(main_graph.get()));
  ASSERT_EQ(ret, fe::SUCCESS);
  ret = op_format_dtype_judge_ptr_->Judge(*(main_graph.get()));
  ASSERT_EQ(ret, fe::SUCCESS);
  ret = op_format_dtype_judge_ptr_->SetFormat(*(main_graph.get()));
  ASSERT_EQ(ret, fe::SUCCESS);
  ASSERT_EQ(CheckIfContainIfResult(main_graph), fe::SUCCESS);
}
