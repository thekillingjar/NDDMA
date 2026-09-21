/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <string>
#include <vector>
#include <gtest/gtest.h>

#include "macro_utils/dt_public_scope.h"
#include "graph/build/stream/stream_allocator.h"
#include "graph/build/stream/stream_utils.h"
#include "graph/normal_graph/compute_graph_impl.h"
#include "macro_utils/dt_public_unscope.h"

#include "graph/debug/ge_attr_define.h"
#include "graph/ge_local_context.h"
#include "graph/utils/graph_utils.h"
#include "ge_graph_dsl/graph_dsl.h"
#include "graph/passes/graph_builder_utils.h"
#include "graph/ir_definitions_recover.h"
#include "api/gelib/gelib.h"
#include "graph/build/task_generator_utils.h"
#include "framework/ge_runtime_stub/include/stub/gert_runtime_stub.h"
#include "ge_running_env/op_reg.h"
#include "graph/build/stream/stream_info.h"

extern std::string g_runtime_stub_mock;

namespace ge {
namespace {
void AddTaskDefWithStreamId(std::vector<domi::TaskDef> &task_defs, uint32_t stream_id) {
  auto task = domi::TaskDef();
  task.set_stream_id(stream_id);
  task_defs.emplace_back(task);
}

ComputeGraphPtr BuildGraphWithMultiAttachedStream() {
  std::vector<NamedAttrs> named_attrs;
  NamedAttrs named_attr;
  AttrUtils::SetStr(named_attr, ATTR_NAME_ATTACHED_RESOURCE_NAME, "group");
  AttrUtils::SetInt(named_attr, ATTR_NAME_ATTACHED_RESOURCE_ID, 1);
  AttrUtils::SetStr(named_attr, ATTR_NAME_ATTACHED_RESOURCE_REUSE_KEY, "as0");
  AttrUtils::SetBool(named_attr, ATTR_NAME_ATTACHED_RESOURCE_REQUIRED_FLAG, true);
  named_attrs.emplace_back(named_attr);
  AttrUtils::SetInt(named_attr, ATTR_NAME_ATTACHED_RESOURCE_ID, 2);
  AttrUtils::SetStr(named_attr, ATTR_NAME_ATTACHED_RESOURCE_REUSE_KEY, "as1");
  named_attrs.emplace_back(named_attr);

  DEF_GRAPH(g1) {
                  const auto data_0 = OP_CFG(DATA);
                  auto mc2 = OP_CFG(MATMUL).Attr(ATTR_NAME_ATTACHED_STREAM_INFO_LIST, named_attrs);
                  CHAIN(NODE("data0", data_0)->EDGE(0, 0)->NODE("mc2", mc2)->EDGE(0, 0)->NODE("output0", NETOUTPUT));
                };
  return ToComputeGraph(g1);
}
}
class UtestStreamAllocator : public testing::Test {
 protected:
  void SetUp() {}
  void TearDown() {}

 public:
  /**
   *
   *    A
   *   / \.
   *  B   C
   *  |   |
   *  D  400
   *  |   |
   *  |   E
   *   \ /
   *    F
   *
   */
  void make_graph_active(const ComputeGraphPtr &graph) {
    const auto &a_desc = std::make_shared<OpDesc>("A", DATA);
    a_desc->AddInputDesc(GeTensorDesc());
    a_desc->AddOutputDesc(GeTensorDesc());
    a_desc->SetStreamId(0);
    const auto &a_node = graph->AddNode(a_desc);

    const auto &b_desc = std::make_shared<OpDesc>("B", "testa");
    b_desc->AddInputDesc(GeTensorDesc());
    b_desc->AddOutputDesc(GeTensorDesc());
    b_desc->SetStreamId(1);
    AttrUtils::SetListStr(b_desc, ATTR_NAME_ACTIVE_LABEL_LIST, {"1"});
    const auto &b_node = graph->AddNode(b_desc);

    const auto &c_desc = std::make_shared<OpDesc>("C", "testa");
    c_desc->AddInputDesc(GeTensorDesc());
    c_desc->AddOutputDesc(GeTensorDesc());
    c_desc->SetStreamId(2);
    AttrUtils::SetStr(c_desc, ATTR_NAME_STREAM_LABEL, "1");
    const auto &c_node = graph->AddNode(c_desc);

    const auto &d_desc = std::make_shared<OpDesc>("D", "testa");
    d_desc->AddInputDesc(GeTensorDesc());
    d_desc->AddOutputDesc(GeTensorDesc());
    d_desc->SetStreamId(1);
    const auto &d_node = graph->AddNode(d_desc);

    const auto &e_desc = std::make_shared<OpDesc>("E", "testa");
    e_desc->AddInputDesc(GeTensorDesc());
    e_desc->AddOutputDesc(GeTensorDesc());
    e_desc->SetStreamId(2);
    const auto &e_node = graph->AddNode(e_desc);

    const auto &f_desc = std::make_shared<OpDesc>("F", "testa");
    f_desc->AddInputDesc(GeTensorDesc());
    f_desc->AddInputDesc(GeTensorDesc());
    f_desc->AddOutputDesc(GeTensorDesc());
    f_desc->SetStreamId(2);
    const auto &f_node = graph->AddNode(f_desc);

    std::vector<NodePtr> node_list(400);
    for (int i = 0; i < 400; i++) {
      const auto &op_desc = std::make_shared<OpDesc>("X" + std::to_string(i), DATA);
      op_desc->AddInputDesc(GeTensorDesc());
      op_desc->AddOutputDesc(GeTensorDesc());
      op_desc->SetStreamId(2);
      node_list[i] = graph->AddNode(op_desc);
    }

    GraphUtils::AddEdge(a_node->GetOutDataAnchor(0), b_node->GetInDataAnchor(0));
    GraphUtils::AddEdge(a_node->GetOutDataAnchor(0), c_node->GetInDataAnchor(0));
    GraphUtils::AddEdge(b_node->GetOutDataAnchor(0), d_node->GetInDataAnchor(0));
    GraphUtils::AddEdge(d_node->GetOutDataAnchor(0), f_node->GetInDataAnchor(0));
    GraphUtils::AddEdge(c_node->GetOutDataAnchor(0), node_list[0]->GetInDataAnchor(0));
    for (uint32_t i = 0; i < 399; i++) {
      GraphUtils::AddEdge(node_list[i]->GetOutDataAnchor(0), node_list[i + 1]->GetInDataAnchor(0));
    }
    GraphUtils::AddEdge(node_list[399]->GetOutDataAnchor(0), e_node->GetInDataAnchor(0));
    GraphUtils::AddEdge(e_node->GetOutDataAnchor(0), f_node->GetInDataAnchor(1));
  }
};

namespace {
ComputeGraphPtr MakeFunctionGraph(const std::string &func_node_name, const std::string &func_node_type) {
  DEF_GRAPH(g1) {
    CHAIN(NODE("_arg_0", DATA)->NODE(func_node_name, func_node_type)->NODE("Node_Output", NETOUTPUT));
    CHAIN(NODE("_arg_1", DATA)->NODE(func_node_name));
    CHAIN(NODE("_arg_2", DATA)->NODE(func_node_name));
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
    CHAIN(NODE(prefix + "_arg_2")->Ctrl()->NODE(prefix + "Add"));
  };
  return ToComputeGraph(g2);
}

//                                    Data1   Data2
//                                       \     /
//                                        Add1
//                                          |
//                     ...................Case1 ...................
//                   .`                     |                      `.
//                 .`                   NetOutput                    `.
//   .--------------------------.                       .-----------------------------.
//   |        Sub_Data2         |                       |         Sub_Data3           |
//   |            |             |                       |              |              |
//   |          Relu1           |                       |            Relu3            |
//   |        (stream0)         |                       |          (stream0)          |
//   |            |             |                       |              |              |
//   |          Relu2           |                       |            Relu4            |
//   |        (stream1)         |                       |          (stream2)          |
//   |            |             |                       |              |              |
//   |       Sub_Output1        |                       |         Sub_Output2         |
//   |__________________________|                       |_____________________________|
ComputeGraphPtr MakeMultiDimsGraph() {
  auto data1 = OP_CFG(DATA).StreamId(0);
  auto add1 = OP_CFG(ADD).StreamId(0);
  auto case1 = OP_CFG(CASE).StreamId(0).Attr(ATTR_NAME_BATCH_NUM, 2);
  auto netoutput1 = OP_CFG(NETOUTPUT).StreamId(0);
  auto sub_data = OP_CFG(DATA).Attr(ATTR_NAME_PARENT_NODE_INDEX, 0);
  auto sub_relu1 = OP_CFG(RELU).StreamId(0);
  auto sub_relu2 = OP_CFG(RELU).StreamId(1);
  auto sub_relu3 = OP_CFG(RELU).StreamId(0);
  auto sub_relu4 = OP_CFG(RELU).StreamId(2);
  auto sub_output1 = OP_CFG(NETOUTPUT).StreamId(1);
  auto sub_output2 = OP_CFG(NETOUTPUT).StreamId(2);
  DEF_GRAPH(batch_0) {
    CHAIN(NODE("sub_data1", sub_data)
              ->NODE("sub_relu1", sub_relu1)
              ->NODE("sub_relu2", sub_relu2)
              ->NODE("Sub_Output1", sub_output1));
  };

  DEF_GRAPH(batch_1) {
    CHAIN(NODE("sub_data2", sub_data)
              ->NODE("sub_relu3", sub_relu3)
              ->NODE("sub_relu4", sub_relu4)
              ->NODE("Sub_Output2", sub_output2));
  };

  DEF_GRAPH(g1) {
    CHAIN(NODE("data0", data1)->EDGE(0, 0)->NODE("add1", add1)->NODE("case", case1, batch_0, batch_1)
        ->NODE("netoutput", netoutput1));
    CHAIN(NODE("data1", data1)->EDGE(0, 1)->NODE("add1")->NODE("case"));
  };

  return ToComputeGraph(g1);
}

ComputeGraphPtr MakeMultiDimsGraphWithStreamLabel() {
  auto data1 = OP_CFG(DATA).StreamId(0);
  auto add1 = OP_CFG(ADD).StreamId(0);
  auto case1 = OP_CFG(CASE).StreamId(0).Attr(ATTR_NAME_BATCH_NUM, 2);
  auto netoutput1 = OP_CFG(NETOUTPUT).StreamId(0);
  auto sub_data = OP_CFG(DATA).Attr(ATTR_NAME_PARENT_NODE_INDEX, 0);
  auto sub_relu1 = OP_CFG(RELU).StreamId(0);
  auto sub_relu2 = OP_CFG(RELU).StreamId(1);
  std::string stream_label = "label0";
  sub_relu2.Attr(ATTR_NAME_STREAM_LABEL, stream_label);
  auto sub_relu3 = OP_CFG(RELU).StreamId(0);
  auto sub_relu4 = OP_CFG(RELU).StreamId(2);
  sub_relu4.Attr(ATTR_NAME_STREAM_LABEL, stream_label);
  auto sub_output1 = OP_CFG(NETOUTPUT).StreamId(1);
  auto sub_output2 = OP_CFG(NETOUTPUT).StreamId(2);
  DEF_GRAPH(batch_0) {
                       CHAIN(NODE("sub_data1", sub_data)
                                 ->NODE("sub_relu1", sub_relu1)
                                 ->NODE("sub_relu2", sub_relu2)
                                 ->NODE("Sub_Output1", sub_output1));
                     };

  DEF_GRAPH(batch_1) {
                       CHAIN(NODE("sub_data2", sub_data)
                                 ->NODE("sub_relu3", sub_relu3)
                                 ->NODE("sub_relu4", sub_relu4)
                                 ->NODE("Sub_Output2", sub_output2));
                     };

  DEF_GRAPH(g1) {
                  CHAIN(NODE("data0", data1)->EDGE(0, 0)->NODE("add1", add1)->NODE("case", case1, batch_0, batch_1)
                            ->NODE("netoutput", netoutput1));
                  CHAIN(NODE("data1", data1)->EDGE(0, 1)->NODE("add1")->NODE("case"));
                };

  return ToComputeGraph(g1);
}

void MakeSwitchGraph(ge::ComputeGraphPtr graph, const string &stream_label) {
  auto desc_temp_ptr = make_shared<ge::GeTensorDesc>();
  auto desc_temp = *desc_temp_ptr;
  ge::OpDescPtr op_a = make_shared<ge::OpDesc>("A", DATA);
  op_a->AddInputDesc(desc_temp);
  op_a->AddOutputDesc(desc_temp);
  ge::OpDescPtr op_b = make_shared<ge::OpDesc>("switch", STREAMSWITCH);
  op_b->AddInputDesc(desc_temp);
  op_b->AddOutputDesc(desc_temp);
  ge::OpDescPtr op_c = make_shared<ge::OpDesc>("C", "testa");
  op_c->AddInputDesc(desc_temp);
  op_c->AddOutputDesc(desc_temp);

  ge::NodePtr node_a = graph->AddNode(op_a);
  ge::NodePtr node_b = graph->AddNode(op_b);
  ge::NodePtr node_c = graph->AddNode(op_c);

  ge::GraphUtils::AddEdge(node_a->GetOutDataAnchor(0), node_b->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(node_b->GetOutControlAnchor(), node_c->GetInControlAnchor());

  node_a->GetOpDesc()->SetStreamId(0);
  node_b->GetOpDesc()->SetStreamId(1);
  node_c->GetOpDesc()->SetStreamId(2);
  node_c->GetOpDesc()->SetAttachedStreamId(3);

  (void)AttrUtils::SetListStr(node_b->GetOpDesc(), ATTR_NAME_ACTIVE_LABEL_LIST, {"label1"});
  (void)AttrUtils::SetStr(node_c->GetOpDesc(), ATTR_NAME_STREAM_LABEL, stream_label);
}

void MakeContinuousStreameLableGraph(const ge::ComputeGraphPtr &graph, int node_num, bool diff_stream_flag,
                                     const string &continuous_stream_label) {
  auto desc_temp_ptr = make_shared<ge::GeTensorDesc>();
  auto desc_temp = *desc_temp_ptr;

  std::vector<ge::OpDescPtr> op_desc_list(node_num);

  for (int i = 0; i < node_num; i++) {
    op_desc_list[i] = make_shared<ge::OpDesc>("A" + std::to_string(i), DATA);
    op_desc_list[i]->AddInputDesc(desc_temp);
    op_desc_list[i]->AddOutputDesc(desc_temp);
    if ((i + 10) > node_num) {
      ge::AttrUtils::SetStr(op_desc_list[i], ge::ATTR_NAME_CONTINUOUS_STREAM_LABEL, continuous_stream_label);
    }
  }

  std::vector<ge::NodePtr> node_list(node_num);
  for (int i = 0; i < node_num; i++) {
    node_list[i] = graph->AddNode(op_desc_list[i]);
  }
  for (int i = 0; i <= (node_num - 2); i++) {
    ge::GraphUtils::AddEdge(node_list[i]->GetOutDataAnchor(0), node_list[i + 1]->GetInDataAnchor(0));
  }
  for (int i = 0; i < node_num; i++) {
    if (diff_stream_flag) {
      node_list[i]->GetOpDesc()->SetStreamId(i);
    } else {
      node_list[i]->GetOpDesc()->SetStreamId(0);
    }
  }
}

void MakLargeGraphWithAttachedStream(const ge::ComputeGraphPtr &graph, int node_num,
                                     const std::vector<int64_t> &attached_streams) {
  auto desc_temp_ptr = make_shared<ge::GeTensorDesc>();
  auto desc_temp = *desc_temp_ptr;

  std::vector<ge::OpDescPtr> op_desc_list(node_num + 1);

  for (int i = 0; i < node_num; i++) {
    op_desc_list[i] = make_shared<ge::OpDesc>("A_" + std::to_string(i), "Phony_node");
    op_desc_list[i]->AddInputDesc(desc_temp);
    op_desc_list[i]->AddOutputDesc(desc_temp);
  }
  op_desc_list[node_num] = make_shared<ge::OpDesc>("out", NETOUTPUT);
  node_num += 1;
  std::vector<ge::NodePtr> node_list(node_num);
  for (int i = 0; i < node_num; i++) {
    node_list[i] = graph->AddNode(op_desc_list[i]);
  }
  for (int i = 0; i <= (node_num - 2); i++) {
    ge::GraphUtils::AddEdge(node_list[i]->GetOutDataAnchor(0), node_list[i + 1]->GetInDataAnchor(0));
  }
  std::vector<ge::NamedAttrs> attached_stream_list;
  for (size_t i = 0U; i < attached_streams.size(); i++) {
    ge::GeAttrValue::NAMED_ATTRS attached_stream;
    (void)ge::AttrUtils::SetStr(attached_stream, ATTR_NAME_ATTACHED_RESOURCE_NAME, "group0");
    (void)ge::AttrUtils::SetStr(attached_stream, ge::ATTR_NAME_ATTACHED_RESOURCE_REUSE_KEY, "as0");
    attached_stream_list.emplace_back(attached_stream);
  }

  for (int i = 0; i < node_num - 1; i++) {
    node_list[i]->GetOpDesc()->SetStreamId(0);
    (void)ge::AttrUtils::SetListNamedAttrs(node_list[i]->GetOpDesc(), ge::ATTR_NAME_ATTACHED_STREAM_INFO_LIST,
                                           attached_stream_list);
    node_list[i]->GetOpDesc()->SetAttachedStreamIds(attached_streams);
  }
  node_list[node_num - 1]->GetOpDesc()->SetStreamId(0);

  graph->TopologicalSorting();
}

/*
 * stream id: 0            A--> Send
 *                        /
 * stream id: 1       StreamActiveA
 *                       /
 * stream id: 2      StreamActiveB
 *                       \
 * stream id: 3    Recv-->C
 */

void MakeActiveStreamOptimizeIndirectlyGraph(const ge::ComputeGraphPtr &graph) {
  ge::OpDescPtr op_a = make_shared<ge::OpDesc>("A", DATA);
  auto desc_temp_ptr = make_shared<ge::GeTensorDesc>();
  auto desc_temp = *desc_temp_ptr;
  op_a->AddInputDesc(desc_temp);
  op_a->AddOutputDesc(desc_temp);
  ge::OpDescPtr op_stream_active_a = make_shared<ge::OpDesc>("StreamActiveA", "testa");
  op_stream_active_a->AddInputDesc(desc_temp);
  op_stream_active_a->AddOutputDesc(desc_temp);
  ge::OpDescPtr op_stream_active_b = make_shared<ge::OpDesc>("StreamActiveB", "testa");
  op_stream_active_b->AddInputDesc(desc_temp);
  op_stream_active_b->AddOutputDesc(desc_temp);

  ge::OpDescPtr op_c = make_shared<ge::OpDesc>("C", "testa");
  op_c->AddInputDesc(desc_temp);
  op_c->AddOutputDesc(desc_temp);

  ge::NodePtr node_a = graph->AddNode(op_a);
  ge::NodePtr node_stream_active_a = graph->AddNode(op_stream_active_a);
  ge::NodePtr node_stream_active_b = graph->AddNode(op_stream_active_b);
  ge::NodePtr node_c = graph->AddNode(op_c);

  ge::GraphUtils::AddEdge(node_a->GetOutDataAnchor(0), node_stream_active_a->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(node_stream_active_a->GetOutDataAnchor(0), node_stream_active_b->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(node_stream_active_b->GetOutDataAnchor(0), node_c->GetInDataAnchor(0));

  node_a->GetOpDesc()->SetStreamId(0);
  node_stream_active_a->GetOpDesc()->SetStreamId(1);
  node_stream_active_b->GetOpDesc()->SetStreamId(2);
  node_c->GetOpDesc()->SetStreamId(3);

  (void)AttrUtils::SetListStr(node_stream_active_a->GetOpDesc(), ATTR_NAME_ACTIVE_LABEL_LIST, {"label2"});
  (void)AttrUtils::SetListStr(node_stream_active_b->GetOpDesc(), ATTR_NAME_ACTIVE_LABEL_LIST, {"label3"});
  (void)AttrUtils::SetStr(node_stream_active_a->GetOpDesc(), ATTR_NAME_STREAM_LABEL, "label1");
  (void)AttrUtils::SetStr(node_stream_active_b->GetOpDesc(), ATTR_NAME_STREAM_LABEL, "label2");
  (void)AttrUtils::SetStr(node_c->GetOpDesc(), ATTR_NAME_STREAM_LABEL, "label3");
}

void MakeRefNodeGraphWithAttachedStream(const ge::ComputeGraphPtr &graph) {
  const auto &a_desc = std::make_shared<OpDesc>("A", DATA);
  a_desc->AddInputDesc(GeTensorDesc());
  a_desc->AddOutputDesc(GeTensorDesc());
  const auto &a_node = graph->AddNode(a_desc);

  const auto &b_desc = std::make_shared<OpDesc>("B", REFDATA);
  b_desc->AddInputDesc(GeTensorDesc());
  b_desc->AddOutputDesc(GeTensorDesc());
  b_desc->SetStreamId(1);
  b_desc->SetAttachedStreamId(2);
  const auto &b_node = graph->AddNode(b_desc);

  const auto &d_desc = std::make_shared<OpDesc>("D", "Assign");
  const auto &e_desc = std::make_shared<OpDesc>("E", "Assign");
  std::vector<OpDescPtr> descs{d_desc, e_desc};
  for (auto &desc : descs) {
    desc->AddInputDesc(GeTensorDesc());
    desc->AddInputDesc(GeTensorDesc());
    desc->AddOutputDesc(GeTensorDesc());
    desc->SetStreamId(1);
    desc->SetAttachedStreamId(2);
    ge::AttrUtils::SetBool(desc, ATTR_NAME_REFERENCE, true);
    desc->MutableAllOutputName().erase("__output0");
    desc->MutableAllOutputName()["__input0"] = 0;
  }
  const auto &d_node = graph->AddNode(descs[0]);
  const auto &e_node = graph->AddNode(descs[1]);

  const auto &c_desc = std::make_shared<OpDesc>("C", "IFN");
  c_desc->AddInputDesc(GeTensorDesc());
  c_desc->AddInputDesc(GeTensorDesc());
  c_desc->AddInputDesc(GeTensorDesc());
  c_desc->AddOutputDesc(GeTensorDesc());
  c_desc->SetStreamId(1);
  c_desc->SetAttachedStreamId(3);
  RecoverOpDescIrDefinition(c_desc, "IFN");
  const auto &c_node = graph->AddNode(c_desc);
  NamedAttrs attrs;
  EXPECT_TRUE(AttrUtils::SetStr(attrs, "_attached_stream_key", "tilingsink"));
  EXPECT_TRUE(AttrUtils::SetStr(attrs, "_attached_stream_depend_value_list", "0"));
  EXPECT_TRUE(AttrUtils::SetListNamedAttrs(c_desc, ATTR_NAME_ATTACHED_STREAM_INFO, {attrs}));

  GraphUtils::AddEdge(b_node->GetOutDataAnchor(0), c_node->GetInDataAnchor(0));
  GraphUtils::AddEdge(b_node->GetOutDataAnchor(0), d_node->GetInDataAnchor(0));
  GraphUtils::AddEdge(b_node->GetOutDataAnchor(0), e_node->GetInDataAnchor(0));
  GraphUtils::AddEdge(a_node->GetOutDataAnchor(0), c_node->GetInDataAnchor(1));
  GraphUtils::AddEdge(a_node->GetOutDataAnchor(0), c_node->GetInDataAnchor(2));
  GraphUtils::AddEdge(c_node->GetOutDataAnchor(0), e_node->GetInDataAnchor(1));
  graph->TopologicalSortingGraph();
}

// 需要确保graph里没有重名节点
std::unordered_map<int64_t, std::vector<domi::TaskDef>> MakeTaskDefsByGraph(const ComputeGraphPtr &graph,
                                                                                size_t per_node_task_num, size_t pre_task_sqe_num = 1U) {
  std::unordered_map<int64_t, std::vector<domi::TaskDef>> node_id_to_node_tasks;
  if (graph == nullptr) {
    return node_id_to_node_tasks;
  }
  size_t task_num = per_node_task_num;
  for (const auto &node : graph->GetNodes(graph->GetGraphUnknownFlag())) {
    if (StreamUtils::IsHcclOp(node->GetOpDescBarePtr())) {
      task_num = 245;
    }
    const auto &node_id = node->GetOpDesc()->GetId();
    if (node_id_to_node_tasks.find(node_id) == node_id_to_node_tasks.end()) {
      for (size_t i = 0U; i < task_num; i++) {
        auto task = domi::TaskDef();
        task.set_sqe_num(pre_task_sqe_num);
        task.set_stream_id(node->GetOpDesc()->GetStreamId());
        node_id_to_node_tasks[node_id].emplace_back(task);
      }
    }
  }
  return node_id_to_node_tasks;
}

std::unordered_map<int64_t, std::vector<domi::TaskDef>> MakeTaskDefsByGraphWithoutData(const ComputeGraphPtr &graph,
                                                                            size_t per_node_task_num) {
  std::unordered_map<int64_t, std::vector<domi::TaskDef>> node_id_to_node_tasks;
  if (graph == nullptr) {
    return node_id_to_node_tasks;
  }
  size_t task_num = per_node_task_num;
  for (const auto &node : graph->GetNodes(graph->GetGraphUnknownFlag())) {
    if (StreamUtils::IsHcclOp(node->GetOpDescBarePtr())) {
      task_num = 245;
    }
    const auto &node_id = node->GetOpDesc()->GetId();
    if (node_id_to_node_tasks.find(node_id) == node_id_to_node_tasks.end()) {
      if (node->GetType() == DATA) {
        continue;
      }
      for (size_t i = 0U; i < task_num; i++) {
        auto task = domi::TaskDef();
        task.set_stream_id(node->GetOpDesc()->GetStreamId());
        node_id_to_node_tasks[node_id].emplace_back(task);
      }
    }
  }
  return node_id_to_node_tasks;
}

std::unordered_map<int64_t, std::vector<domi::TaskDef>> MakeTaskDefsByGraphForAttchStream(
    const ComputeGraphPtr &graph, size_t per_node_task_num, std::vector<int64_t> attached_stream_list) {
  std::unordered_map<int64_t, std::vector<domi::TaskDef>> node_id_to_node_tasks;
  if (graph == nullptr) {
    return node_id_to_node_tasks;
  }
  size_t task_num = per_node_task_num;
  for (const auto &node : graph->GetNodes(graph->GetGraphUnknownFlag())) {
    if (StreamUtils::IsHcclOp(node->GetOpDescBarePtr())) {
      task_num = 245;
    }
    const auto &node_id = node->GetOpDesc()->GetId();
    if (node_id_to_node_tasks.find(node_id) == node_id_to_node_tasks.end()) {
      for (size_t i = 0U; i < task_num; i++) {
        auto task = domi::TaskDef();
        task.set_stream_id(0);
        node_id_to_node_tasks[node_id].emplace_back(task);
      }
      if (node->GetOpDesc()->HasValidAttachedStreamId()) {
        for (auto attached_stream: attached_stream_list) {
          for (size_t i = 0U; i < task_num; i++) {
            auto task = domi::TaskDef();
            task.set_stream_id(attached_stream);
            node_id_to_node_tasks[node_id].emplace_back(task);
          }
        }
      }
    }
  }
  return node_id_to_node_tasks;
}

void CheckTaskDefStreamId(const std::unordered_map<int64_t , std::vector<domi::TaskDef>> &node_id_to_node_tasks,
                          const ComputeGraphPtr &graph) {
  for (const auto &iter: node_id_to_node_tasks) {
    for (const auto &task: iter.second) {
      const auto &all_nodes = graph->GetAllNodes();
      NodePtr node = nullptr;
      for(auto n : all_nodes) {
        if (n->GetOpDesc()->GetId() == iter.first) {
          node = n;
          break;
        }
      }
      ASSERT_NE(node, nullptr);
      bool is_equall = (node->GetOpDescBarePtr()->GetStreamId() == task.stream_id());
      if (node->GetOpDescBarePtr()->HasValidAttachedStreamId()) {
        is_equall |= (node->GetOpDescBarePtr()->GetAttachedStreamIds()[0] == task.stream_id());
      }
      EXPECT_EQ(is_equall, true);
    }
  }
}
}  // namespace

TEST_F(UtestStreamAllocator, SetActiveStreamsForSubgraphs_first_active_node_success) {
  // construct graph
  std::string func_node_name = "PartitionedCall_0";
  const auto &root_graph = MakeFunctionGraph(func_node_name, PARTITIONEDCALL);
  const auto &sub_graph = MakeSubGraph("");
  // set stream_id
  for (size_t i = 0U; i < sub_graph->GetDirectNodesSize(); ++i) {
    auto node = sub_graph->GetDirectNode().at(i);
    if (node->GetName() == "Conv2D") {  // set attr
      (void)AttrUtils::SetBool(node->GetOpDesc(), ATTR_NAME_SUBGRAPH_FIRST_ACTIVE, true);
    }
    node->GetOpDesc()->SetStreamId(i);
  }

  ut::GraphBuilder::AddPartitionedCall(root_graph, func_node_name, sub_graph);
  Graph2SubGraphInfoList subgraphs;
  StreamAllocator allocator(root_graph, subgraphs);
  EXPECT_EQ(allocator.SetActiveStreamsForSubgraphs(), SUCCESS);
}

TEST_F(UtestStreamAllocator, InsertSyncEvents_MultiDims_StreamLabel_success) {
  // construct graph
  std::string func_node_name = "PartitionedCall_0";
  const auto &root_graph = MakeMultiDimsGraphWithStreamLabel();
  Graph2SubGraphInfoList subgraphs;
  StreamAllocator allocator(root_graph, subgraphs);
  auto sub_relu2 = root_graph->GetSubgraph("batch_0")->FindNode("sub_relu2");
  ASSERT_NE(sub_relu2, nullptr);
  auto sub_relu4 = root_graph->GetSubgraph("batch_1")->FindNode("sub_relu4");
  ASSERT_NE(sub_relu4, nullptr);
  std::set<NodePtr> specific_activated_nodes = {sub_relu2, sub_relu4};
  std::string stream_label = "label0";
  allocator.specific_activated_labels_[stream_label] = specific_activated_nodes;
  EXPECT_EQ(allocator.InsertSyncEvents(EventType::kEvent), SUCCESS);
  ASSERT_EQ(allocator.event_num_, 2);
  EXPECT_EQ(allocator.OptimizeByStreamActivate(EventType::kEvent, allocator.node_to_send_events_,
                                               allocator.node_to_recv_events_),
            SUCCESS);
}

TEST_F(UtestStreamAllocator, InsertSyncEvents_success) {
  // construct graph
  std::string func_node_name = "PartitionedCall_0";
  const auto &root_graph = MakeFunctionGraph(func_node_name, PARTITIONEDCALL);
  const auto &sub_graph = MakeSubGraph("");
  // set stream_id
  std::string stream_label = "label0";
  for (size_t i = 0U; i < sub_graph->GetDirectNodesSize(); ++i) {
    auto node = sub_graph->GetDirectNode().at(i);
    node->GetOpDesc()->SetStreamId(i);
  }

  ut::GraphBuilder::AddPartitionedCall(root_graph, func_node_name, sub_graph);
  Graph2SubGraphInfoList subgraphs;
  StreamAllocator allocator(root_graph, subgraphs);
  auto node = sub_graph->FindNode("Conv2D");
  EXPECT_NE(node, nullptr);
  (void)AttrUtils::SetStr(node->GetOpDesc(), ATTR_NAME_STREAM_LABEL, stream_label);
  std::set<NodePtr> specific_activated_nodes = {node};
  allocator.specific_activated_labels_[stream_label] = specific_activated_nodes;
  auto end_node = sub_graph->FindNode("Node_Output");
  EXPECT_NE(end_node, nullptr);
  (void)AttrUtils::SetBool(end_node->GetOpDesc(), ATTR_NAME_SUBGRAPH_END_NODE, true);
  EXPECT_EQ(allocator.InsertSyncEvents(EventType::kEvent), SUCCESS);
}

/**
 * Optimization scenario: one stream has multiple send events in one node,
 * and multiple nodes for recv events on another stream
 * Example:
 * Stream0            Stream1
 *   N0 - - - event - > N1
 *     \                |
 *      \               v
 *        - - event - > N2
 */
TEST_F(UtestStreamAllocator, OptimizeBySendEvents_success) {
  std::map<int64_t, std::vector<NodePtr>> stream_nodes;
  auto node_0 = std::make_shared<Node>(std::make_shared<OpDesc>("node0", "testType"), nullptr);
  std::vector<NodePtr> stream0_nodes;
  stream0_nodes.push_back(node_0);
  auto node_1 = std::make_shared<Node>(std::make_shared<OpDesc>("node1", "testType"), nullptr);
  auto node_2 = std::make_shared<Node>(std::make_shared<OpDesc>("node2", "testType"), nullptr);
  std::vector<NodePtr> stream1_nodes;
  stream1_nodes.push_back(node_1);
  stream1_nodes.push_back(node_2);
  stream_nodes[0] = stream0_nodes;
  stream_nodes[1] = stream1_nodes;

  Graph2SubGraphInfoList subgraphs;
  StreamAllocator allocator(nullptr, subgraphs);

  uint32_t event_id = 1U;
  allocator.node_to_recv_events_[node_1] = {event_id};
  allocator.node_to_send_events_[node_0] = {event_id++};
  allocator.node_to_recv_events_[node_2] = {event_id};
  allocator.node_to_send_events_[node_0].push_back(event_id);

  EXPECT_EQ(allocator.node_to_send_events_[node_0].size(), 2);
  EXPECT_EQ(allocator.node_to_send_events_[node_0][0], 1);
  EXPECT_EQ(allocator.node_to_send_events_[node_0][1], 2);

  EXPECT_EQ(StreamUtils::OptimizeBySendEvents(stream_nodes, allocator.node_to_send_events_,
                                           allocator.node_to_recv_events_),
      SUCCESS);
  EXPECT_EQ(allocator.node_to_send_events_[node_0].size(), 1);  // event 2 is optimized
  EXPECT_EQ(allocator.node_to_send_events_[node_0][0], 1);
}

/**
 * Scenario: multiple send nodes on a stream sent to a single recv node on the destination stream
 * Example:
 * Stream0            Stream1
 *   N0 - -
 *   |    |
 *   |    - - event1 - - -
 *   |                  |
 *   V                  V
 *   N1 - - -event2 - > N2
 */
TEST_F(UtestStreamAllocator, OptimizeByRecvEvents_success) {
  std::map<int64_t, std::vector<NodePtr>> stream_nodes;
  auto node_0 = std::make_shared<Node>(std::make_shared<OpDesc>("node0", "testType"), nullptr);
  std::vector<NodePtr> stream0_nodes;
  stream0_nodes.push_back(node_0);
  auto node_1 = std::make_shared<Node>(std::make_shared<OpDesc>("node1", "testType"), nullptr);
  stream0_nodes.push_back(node_1);
  auto node_2 = std::make_shared<Node>(std::make_shared<OpDesc>("node2", "testType"), nullptr);
  std::vector<NodePtr> stream1_nodes;
  stream1_nodes.push_back(node_2);
  stream_nodes[0] = stream0_nodes;
  stream_nodes[1] = stream1_nodes;

  Graph2SubGraphInfoList subgraphs;
  StreamAllocator allocator(nullptr, subgraphs);

  uint32_t event_id = 1U;
  allocator.node_to_recv_events_[node_2] = {event_id};
  allocator.node_to_send_events_[node_0] = {event_id++};
  allocator.node_to_recv_events_[node_2].push_back(event_id);
  allocator.node_to_send_events_[node_1] = {event_id};

  EXPECT_EQ(allocator.node_to_recv_events_[node_2].size(), 2);
  EXPECT_EQ(allocator.node_to_recv_events_[node_2][0], 1);
  EXPECT_EQ(allocator.node_to_recv_events_[node_2][1], 2);

  EXPECT_EQ(StreamUtils::OptimizeByRecvEvents(stream_nodes, allocator.node_to_send_events_,
                                           allocator.node_to_recv_events_),
      SUCCESS);
  // event 1 is optimized，the optimization result is related to the topological order of the nodes.
  EXPECT_EQ(allocator.node_to_recv_events_[node_2].size(), 1);
  EXPECT_EQ(allocator.node_to_recv_events_[node_2][0], 2);
}

/**
 * Example:
 * Stream0            Stream1
 *   N0 - - - event - > N1
 *     \                |
 *      \               v
 *        - - event - > N2
 */
TEST_F(UtestStreamAllocator, OptimizeByStreamActivate_success) {
  Graph2SubGraphInfoList subgraphs;
  StreamAllocator allocator(nullptr, subgraphs);
  auto graph = MakeSubGraph("");
  auto node_0 = graph->FindNode("Conv2D");
  EXPECT_NE(node_0, nullptr);
  auto node_1 = graph->FindNode("Relu");
  EXPECT_NE(node_1, nullptr);
  auto node_2 = graph->FindNode("Add");
  EXPECT_NE(node_2, nullptr);
  uint32_t event_id = 1U;
  allocator.node_to_recv_events_[node_1] = {event_id};
  allocator.node_to_send_events_[node_0] = {event_id++};
  allocator.node_to_recv_events_[node_2] = {event_id};
  allocator.node_to_send_events_[node_0].push_back(event_id);

  EXPECT_EQ(allocator.node_to_send_events_[node_0].size(), 2);
  EXPECT_EQ(allocator.node_to_send_events_[node_0][0], 1);
  EXPECT_EQ(allocator.node_to_send_events_[node_0][1], 2);

  std::int64_t stream_id = 1;
  node_0->GetOpDesc()->SetStreamId(stream_id++);
  node_1->GetOpDesc()->SetStreamId(stream_id);
  AttrUtils::SetStr(node_1->GetOpDesc(), ATTR_NAME_STREAM_LABEL, std::to_string(stream_id));
  node_2->GetOpDesc()->SetStreamId(stream_id);
  AttrUtils::SetStr(node_2->GetOpDesc(), ATTR_NAME_STREAM_LABEL, std::to_string(stream_id));
  std::set<NodePtr> specific_activated_streams_nodes;
  specific_activated_streams_nodes.insert(node_1);
  specific_activated_streams_nodes.insert(node_2);
  allocator.specific_activated_streams_nodes_map_[stream_id] = specific_activated_streams_nodes;
  EXPECT_EQ(allocator.OptimizeByStreamActivate(EventType::kEvent, allocator.node_to_send_events_,
                                               allocator.node_to_recv_events_),
            SUCCESS);
}

TEST_F(UtestStreamAllocator, OptimizeByStreamActivate_fail) {
  graphStatus ret = GRAPH_SUCCESS;

  ge::ComputeGraphPtr graph = make_shared<ge::ComputeGraph>("");
  graph->SetName("OptimizeByStreamActivate_fail");
  MakeActiveStreamOptimizeIndirectlyGraph(graph);

  Graph2SubGraphInfoList subgraphs;
  std::shared_ptr<ge::StreamAllocator> stream_allocator = make_shared<ge::StreamAllocator>(graph, subgraphs);

  ret = stream_allocator->SetActiveStreamsByLabel();
  EXPECT_EQ(ret, GRAPH_SUCCESS);

  ge::NodePtr node_a = graph->FindNode("A");
  ge::NodePtr node_c = graph->FindNode("C");
  StreamUtils::AddSendEventId(node_a, 0, stream_allocator->node_to_send_events_);
  ret = stream_allocator->OptimizeSyncEvents(EventType::kEvent, stream_allocator->node_to_send_events_,
                                             stream_allocator->node_to_recv_events_);
  EXPECT_EQ(ret, ge::PARAM_INVALID);
}

TEST_F(UtestStreamAllocator, SplitStreams_with_continuous_stream_label_success) {
  graphStatus ret = GRAPH_SUCCESS;
  ge::ComputeGraphPtr graph = make_shared<ge::ComputeGraph>("");
  graph->SetName("SplitStreams_CallRtNodeLine_Success_Test");

  Graph2SubGraphInfoList subgraphs;
  std::shared_ptr<ge::StreamAllocator> stream_allocator = make_shared<ge::StreamAllocator>(graph, subgraphs);

  // graph with one stream, 355 nodes
  MakeContinuousStreameLableGraph(graph, 345, false, "label0");
  stream_allocator->stream_num_ = 1;

  // 没有附属流 直接返回
  ret = stream_allocator->CoverAllStreamByNetoutput();
  EXPECT_EQ(ret, GRAPH_SUCCESS);
  EXPECT_EQ(stream_allocator->event_num_, 0U);
  auto node_id_to_node_tasks = MakeTaskDefsByGraph(graph, 3);
  EXPECT_TRUE(node_id_to_node_tasks.size() > 0);
  std::vector<std::set<int64_t>> split_streams(1);
  ret = stream_allocator->SplitStreams(node_id_to_node_tasks, split_streams);
  EXPECT_EQ(ret, GRAPH_SUCCESS);
  CheckTaskDefStreamId(node_id_to_node_tasks, graph);
  int i = 0;
  for (const auto &cur_node : graph->GetDirectNode()) {
    int64_t stream_id = cur_node->GetOpDesc()->GetStreamId();
    if (i < 332) {
      EXPECT_EQ(stream_id, 0);
    } else {
      EXPECT_EQ(stream_id, 1);
    }
    i++;
  }
  EXPECT_EQ(stream_allocator->helper_.attached_stream_.size(), 0U);
  EXPECT_EQ(stream_allocator->event_num_, 1U);
  EXPECT_EQ(stream_allocator->stream_num_, 2U);
}

TEST_F(UtestStreamAllocator, SplitStreams_with_attached_stream_success) {
  graphStatus ret = GRAPH_SUCCESS;
  ge::ComputeGraphPtr graph = make_shared<ge::ComputeGraph>("");
  graph->SetName("SplitStreams_CallRtNodeLine_Success_Test");

  Graph2SubGraphInfoList subgraphs;
  std::shared_ptr<ge::StreamAllocator> stream_allocator = make_shared<ge::StreamAllocator>(graph, subgraphs);

  // graph with 345 nodes, two streams
  MakLargeGraphWithAttachedStream(graph, 345, {1});
  // one main stream, one attached stream
  stream_allocator->stream_num_ = 1 + 1;

  ret = stream_allocator->CoverAllStreamByNetoutput();
  EXPECT_EQ(ret, GRAPH_SUCCESS);
  // 附属流的last node跟模型输出节点之间同步, 所以event资源变成2+1
  EXPECT_EQ(stream_allocator->event_num_, 1U);
  EXPECT_EQ(stream_allocator->attached_node_to_stream_id_to_send_event_id_.size(),1U);
  EXPECT_EQ(stream_allocator->node_to_recv_events_.size(), 1U);
  EXPECT_NE(stream_allocator->attached_node_to_stream_id_to_send_event_id_.find(graph->FindNode("A_344")),
            stream_allocator->attached_node_to_stream_id_to_send_event_id_.end());
  EXPECT_NE(stream_allocator->node_to_recv_events_.find(graph->FindNode("out")),
            stream_allocator->node_to_recv_events_.end());

  std::vector<std::set<int64_t>> split_streams(stream_allocator->stream_num_);
  // each node has 3 task for main stream, 3 task for attached stream
  auto node_id_to_node_tasks = MakeTaskDefsByGraphForAttchStream(graph, 3, {1});
  EXPECT_TRUE(node_id_to_node_tasks.size() > 0);
  ret = stream_allocator->SplitStreams(node_id_to_node_tasks, split_streams);
  EXPECT_EQ(ret, GRAPH_SUCCESS);
  CheckTaskDefStreamId(node_id_to_node_tasks, graph);
  int i = 0;
  for (const auto &cur_node : graph->GetDirectNode()) {
    if (cur_node->GetType() ==NETOUTPUT) {
      continue;
    }
    int64_t stream_id = cur_node->GetOpDesc()->GetStreamId();
    int64_t attached_stream_id = cur_node->GetOpDesc()->GetAttachedStreamIds()[0];
    if (i < 332) {
      EXPECT_EQ(stream_id, 0);
      EXPECT_EQ(attached_stream_id, 1);
    } else {
      EXPECT_EQ(stream_id, 2);
      EXPECT_EQ(attached_stream_id, 3);
    }
    i++;
  }
  // 主流和附属流都拆分了，所以stream个数各自加倍
  EXPECT_EQ(stream_allocator->stream_num_, 4U);
  // helper_.stream_2_nodes_map记录的是拆分前的流和节点的对应关系
  EXPECT_EQ(stream_allocator->helper_.stream_2_nodes_map.size(), 2U);
  EXPECT_EQ(stream_allocator->helper_.stream_2_nodes_map[1U].size(), 345U);
  EXPECT_EQ(stream_allocator->helper_.stream_2_nodes_map[0U].size(), 346U);
  // stream_2_nodes_map记录的流上的节点是按照topo顺序存放
  EXPECT_EQ(stream_allocator->helper_.stream_2_nodes_map[0U].back()->GetName(), "out");
  EXPECT_EQ(stream_allocator->helper_.stream_2_nodes_map[1U].back()->GetName(), "A_344");
  EXPECT_EQ(stream_allocator->helper_.stream_2_nodes_map[0U].front()->GetName(), "A_0");
  EXPECT_EQ(stream_allocator->helper_.stream_2_nodes_map[1U].front()->GetName(), "A_0");
  EXPECT_EQ(stream_allocator->helper_.attached_stream_.size(), 1U);
  EXPECT_EQ(*stream_allocator->helper_.attached_stream_.begin(), 1);

  // 主流和附属流都拆分了，各自需要独立的event资源，所以event资源是2
  EXPECT_EQ(stream_allocator->event_num_, 1U + 2U);
  EXPECT_EQ(stream_allocator->attached_node_to_stream_id_to_recv_event_id_.size(), 1U);
  EXPECT_EQ(stream_allocator->attached_node_to_stream_id_to_send_event_id_.size(), 2U);
  EXPECT_EQ(stream_allocator->attached_node_to_stream_id_to_recv_event_id_.begin()->first->GetName(), "A_332");
  EXPECT_EQ(stream_allocator->attached_node_to_stream_id_to_send_event_id_.begin()->first->GetName(), "A_331");
  EXPECT_EQ(stream_allocator->node_to_recv_events_.size(), 1U+1U);
  EXPECT_EQ(stream_allocator->node_to_send_events_.size(), 1U);
  auto A_332_node = graph->FindNode("A_332");
  EXPECT_TRUE(stream_allocator->node_to_recv_events_.find(A_332_node) != stream_allocator->node_to_recv_events_.end());
  EXPECT_EQ(stream_allocator->node_to_send_events_.begin()->first->GetName(), "A_331");
}

TEST_F(UtestStreamAllocator, SplitStreams_with_multi_attached_stream_success) {
  graphStatus ret = GRAPH_SUCCESS;
  ge::ComputeGraphPtr graph = make_shared<ge::ComputeGraph>("");
  graph->SetName("SplitStreams_CallRtNodeLine_Success_Test");

  Graph2SubGraphInfoList subgraphs;
  std::shared_ptr<ge::StreamAllocator> stream_allocator = make_shared<ge::StreamAllocator>(graph, subgraphs);

  // graph with 345 nodes, two streams
  MakLargeGraphWithAttachedStream(graph, 345, {1, 2});
  // one main stream, two attached stream
  stream_allocator->stream_num_ = 1 + 2;

  ret = stream_allocator->CoverAllStreamByNetoutput();
  EXPECT_EQ(ret, GRAPH_SUCCESS);
  // 2个附属流的last node跟模型输出节点之间同步, 所以event资源变成2
  EXPECT_EQ(stream_allocator->event_num_, 2U);
  EXPECT_EQ(stream_allocator->attached_node_to_stream_id_to_send_event_id_.size(), 1U);
  EXPECT_EQ(stream_allocator->node_to_recv_events_.size(), 1U);
  auto last_node_iter = stream_allocator->attached_node_to_stream_id_to_send_event_id_.find(graph->FindNode("A_344"));
  EXPECT_NE(last_node_iter, stream_allocator->attached_node_to_stream_id_to_send_event_id_.end());
  // attached stream有两个, 都和output之间插了event
  EXPECT_EQ(last_node_iter->second.size(), 2U);
  auto out_iter = stream_allocator->node_to_recv_events_.find(graph->FindNode("out"));
  EXPECT_NE(out_iter, stream_allocator->node_to_recv_events_.end());
  // output前面有两个event
  EXPECT_EQ(out_iter->second.size(), 2U);

  std::vector<std::set<int64_t>> split_streams(stream_allocator->stream_num_);
  // each node has 3 task for main stream, 3 task for attached stream
  // opdesc does not have attached stream 2, but task def has stream id 2
  auto node_id_to_node_tasks = MakeTaskDefsByGraphForAttchStream(graph, 3, {1, 2});
  EXPECT_TRUE(node_id_to_node_tasks.size() > 0);
  ret = stream_allocator->SplitStreams(node_id_to_node_tasks, split_streams);
  EXPECT_EQ(ret, GRAPH_SUCCESS);
  int i = 0;
  for (const auto &cur_node : graph->GetDirectNode()) {
    if (cur_node->GetType() ==NETOUTPUT) {
      continue;
    }
    int64_t stream_id = cur_node->GetOpDesc()->GetStreamId();
    auto attached_stream_ids = cur_node->GetOpDesc()->GetAttachedStreamIds();
    if (i < 332) {
      EXPECT_EQ(stream_id, 0);
      std::vector<int64_t> expect_attached_streams = {1, 2};
      EXPECT_EQ(attached_stream_ids, expect_attached_streams);
    } else {
      EXPECT_EQ(stream_id, 3);
      std::vector<int64_t> expect_attached_streams = {4, 5};
      EXPECT_EQ(attached_stream_ids, expect_attached_streams);
    }
    i++;
  }
  // 主流和附属流都拆分了，所以stream个数各自加倍
  EXPECT_EQ(stream_allocator->stream_num_, 6U);
  // helper_.stream_2_nodes_map记录的是拆分前的流和节点的对应关系
  EXPECT_EQ(stream_allocator->helper_.stream_2_nodes_map.size(), 3U);
  EXPECT_EQ(stream_allocator->helper_.stream_2_nodes_map[2U].size(), 345U);
  EXPECT_EQ(stream_allocator->helper_.stream_2_nodes_map[1U].size(), 345U);
  EXPECT_EQ(stream_allocator->helper_.stream_2_nodes_map[0U].size(), 346U);
  // stream_2_nodes_map记录的流上的节点是按照topo顺序存放
  EXPECT_EQ(stream_allocator->helper_.stream_2_nodes_map[0U].back()->GetName(), "out");
  EXPECT_EQ(stream_allocator->helper_.stream_2_nodes_map[1U].back()->GetName(), "A_344");
  EXPECT_EQ(stream_allocator->helper_.stream_2_nodes_map[2U].back()->GetName(), "A_344");
  EXPECT_EQ(stream_allocator->helper_.stream_2_nodes_map[0U].front()->GetName(), "A_0");
  EXPECT_EQ(stream_allocator->helper_.stream_2_nodes_map[1U].front()->GetName(), "A_0");
  EXPECT_EQ(stream_allocator->helper_.stream_2_nodes_map[2U].front()->GetName(), "A_0");
  EXPECT_EQ(stream_allocator->helper_.attached_stream_.size(), 2U);
  EXPECT_EQ(*stream_allocator->helper_.attached_stream_.begin(), 1);
  EXPECT_EQ(*stream_allocator->helper_.attached_stream_.end(), 2);

  // 主流和附属流都拆分了，各自需要独立的event资源，所以event资源是2+3
  EXPECT_EQ(stream_allocator->event_num_, 2U + 3U);
  EXPECT_EQ(stream_allocator->attached_node_to_stream_id_to_recv_event_id_.size(), 1U);
  // attached stream有两个，都发生了拆流
  EXPECT_EQ(stream_allocator->attached_node_to_stream_id_to_recv_event_id_.begin()->second.size(), 2U);
  EXPECT_EQ(stream_allocator->attached_node_to_stream_id_to_send_event_id_.size(), 2U);
  EXPECT_EQ(stream_allocator->attached_node_to_stream_id_to_send_event_id_.begin()->second.size(), 2U);
  EXPECT_EQ(stream_allocator->attached_node_to_stream_id_to_recv_event_id_.begin()->first->GetName(), "A_332");
  EXPECT_EQ(stream_allocator->attached_node_to_stream_id_to_send_event_id_.begin()->first->GetName(), "A_331");
  EXPECT_EQ(stream_allocator->node_to_recv_events_.size(), 1U + 1U);
  EXPECT_EQ(stream_allocator->node_to_send_events_.size(), 1U);
  auto A_332_node = graph->FindNode("A_332");
  EXPECT_TRUE(stream_allocator->node_to_recv_events_.find(A_332_node) != stream_allocator->node_to_recv_events_.end());
  EXPECT_EQ(stream_allocator->node_to_send_events_.begin()->first->GetName(), "A_331");
  ret = stream_allocator->GenerateSyncEventNodes(false);
  EXPECT_EQ(ret, GRAPH_SUCCESS);
}

// Iterator loop :
// StreamSwitch  ->  StreamActive
TEST_F(UtestStreamAllocator, FindSwitchNodeBeforeLoopActiveNode_iterator_loop) {
  DEF_GRAPH(g1) {
    CHAIN(NODE("StreamSwitch", STREAMSWITCH)->Ctrl()->NODE("StreamActive", STREAMACTIVE));
  };
  auto graph = ToComputeGraph(g1);
  auto stream_active_node = graph->FindNode("StreamActive");
  (void)AttrUtils::SetBool(stream_active_node->GetOpDesc(), ATTR_NAME_IS_LOOP_ACTIVE, true);
  Graph2SubGraphInfoList subgraphs;
  std::shared_ptr<ge::StreamAllocator> stream_allocator = make_shared<ge::StreamAllocator>(graph, subgraphs);
  stream_allocator->stream_num_ = 2;
  auto ret = stream_allocator->SetActiveStreamsForLoop();
  EXPECT_EQ(ret, SUCCESS);
}

TEST_F(UtestStreamAllocator, AfterSplitStreams_FindSwitchNodeBeforeLoopActiveNode_iterator_loop) {
  DEF_GRAPH(g1) {
                  CHAIN(NODE("StreamSwitch", STREAMSWITCH)->Ctrl()->NODE("StreamActive", STREAMACTIVE));
                };
  auto graph = ToComputeGraph(g1);
  auto stream_active_node = graph->FindNode("StreamActive");
  (void)AttrUtils::SetBool(stream_active_node->GetOpDesc(), ATTR_NAME_IS_LOOP_ACTIVE, true);
  Graph2SubGraphInfoList subgraphs;
  std::shared_ptr<ge::StreamAllocator> stream_allocator = make_shared<ge::StreamAllocator>(graph, subgraphs);
  stream_allocator->stream_num_ = 2;
  std::vector<std::set<int64_t>> split_streams(2);
  auto ret = stream_allocator->SetActiveStreamsForLoop(false, split_streams);
  EXPECT_EQ(ret, SUCCESS);
}
// FpBp loop:
// StreamSwitch  ->  AssignAdd  ->  StreamActive
TEST_F(UtestStreamAllocator, FindSwitchNodeBeforeLoopActiveNode_fpbp_loop) {
  DEF_GRAPH(g1) {
    CHAIN(NODE("StreamSwitch", STREAMSWITCH)
              ->Ctrl()
              ->NODE("AssignAdd", ASSIGNADD)
              ->Ctrl()
              ->NODE("StreamActive", STREAMACTIVE));
  };
  auto graph = ToComputeGraph(g1);
  auto stream_active_node = graph->FindNode("StreamActive");
  (void)AttrUtils::SetBool(stream_active_node->GetOpDesc(), ATTR_NAME_IS_LOOP_ACTIVE, true);
  Graph2SubGraphInfoList subgraphs;
  std::shared_ptr<ge::StreamAllocator> stream_allocator = make_shared<ge::StreamAllocator>(graph, subgraphs);
  stream_allocator->stream_num_ = 2;
  auto ret = stream_allocator->SetActiveStreamsForLoop();
  EXPECT_EQ(ret, SUCCESS);
}

TEST_F(UtestStreamAllocator, MultiAttachedStreamAssignLogicStreamSuccess) {
  gert::GertRuntimeStub runtime_stub;
  runtime_stub.GetSlogStub().SetLevelInfo();
  auto graph = BuildGraphWithMultiAttachedStream();
  Graph2SubGraphInfoList sub_graphs;
  std::shared_ptr<ge::StreamAllocator> stream_allocator = make_shared<ge::StreamAllocator>(graph, sub_graphs);
  stream_allocator->stream_num_ = 3U;
  int64_t stream_num;
  int64_t notify_num;
  int64_t event_num;
  ASSERT_EQ(stream_allocator->InsertSyncNodesByLogicStream(stream_num, event_num, notify_num), SUCCESS);
  EXPECT_EQ(event_num, 2U);
  runtime_stub.Clear();
}

TEST_F(UtestStreamAllocator, UpdateActiveStreamsForSwitchNodes_success) {
  graphStatus ret = GRAPH_SUCCESS;

  ge::ComputeGraphPtr graph = make_shared<ge::ComputeGraph>("");
  graph->SetName("HandleSwitchNodes_success");
  MakeSwitchGraph(graph, "label1");

  Graph2SubGraphInfoList subgraphs;
  std::shared_ptr<ge::StreamAllocator> stream_allocator = make_shared<ge::StreamAllocator>(graph, subgraphs);

  stream_allocator->labeled_streams_["label1"].emplace(3);
  stream_allocator->labeled_streams_["label1"].emplace(4);
  stream_allocator->stream_num_ = 5;
  int64_t stream_num;
  int64_t notify_num;
  int64_t event_num;
  ret = stream_allocator->InsertSyncNodesByLogicStream(stream_num, event_num, notify_num);
  EXPECT_EQ(ret, GRAPH_SUCCESS);
  auto node_id_to_node_tasks = MakeTaskDefsByGraph(graph, 3);
  ret = stream_allocator->SplitStreamAndRefreshTaskDef(node_id_to_node_tasks, stream_num, event_num, notify_num);
  EXPECT_EQ(ret, GRAPH_SUCCESS);

  ge::NodePtr node_active = graph->FindNode("switch_StreamActive_0");
  auto op_desc = node_active->GetOpDesc();
  int64_t stream_id = op_desc->GetStreamId();
  EXPECT_EQ(stream_id, 5);

  StreamGraph stream_graph = StreamGraph(graph);
  EXPECT_EQ(stream_graph.ToDotString().empty(), false);
}

TEST_F(UtestStreamAllocator, NoNeedUpdateActiveStreamsForSwitchNodes_success) {
  graphStatus ret = GRAPH_SUCCESS;

  ge::ComputeGraphPtr graph = make_shared<ge::ComputeGraph>("");
  graph->SetName("HandleSwitchNodes_success");
  MakeSwitchGraph(graph, "label1");

  Graph2SubGraphInfoList subgraphs;
  std::shared_ptr<ge::StreamAllocator> stream_allocator = make_shared<ge::StreamAllocator>(graph, subgraphs);

  stream_allocator->labeled_streams_["label1"].emplace(3);
  stream_allocator->stream_num_ = 4;
  int64_t stream_num;
  int64_t event_num;
  int64_t notify_num;
  ret = stream_allocator->InsertSyncNodesByLogicStream(stream_num, event_num, notify_num);
  EXPECT_EQ(ret, GRAPH_SUCCESS);
  auto node_id_to_node_tasks = MakeTaskDefsByGraph(graph, 3);
  ret = stream_allocator->SplitStreamAndRefreshTaskDef(node_id_to_node_tasks, stream_num, event_num, notify_num);
  EXPECT_EQ(ret, GRAPH_SUCCESS);
}

TEST_F(UtestStreamAllocator, SetActivatedStreamsList_fail) {
  graphStatus ret = GRAPH_SUCCESS;

  ge::ComputeGraphPtr graph = make_shared<ge::ComputeGraph>("");
  graph->SetName("HandleSwitchNodes_fail");
  MakeSwitchGraph(graph, "label2");

  Graph2SubGraphInfoList subgraphs;
  std::shared_ptr<ge::StreamAllocator> stream_allocator = make_shared<ge::StreamAllocator>(graph, subgraphs);
  stream_allocator->labeled_streams_["label1"].emplace(3);
  ge::NodePtr node_a = graph->FindNode("A");

  ret = stream_allocator->SetActiveStreamList(node_a, "label2");
  EXPECT_EQ(ret, FAILED);
}

TEST_F(UtestStreamAllocator, AddActiveNodes_fail) {
  graphStatus ret = GRAPH_SUCCESS;

  ge::ComputeGraphPtr graph = make_shared<ge::ComputeGraph>("");
  graph->SetName("HandleSwitchNodes_fail");
  MakeSwitchGraph(graph, "label2");

  Graph2SubGraphInfoList subgraphs;
  std::shared_ptr<ge::StreamAllocator> stream_allocator = make_shared<ge::StreamAllocator>(graph, subgraphs);
  stream_allocator->labeled_streams_["label1"].emplace(3);
  ge::NodePtr node_a = graph->FindNode("A");
  vector<string> ori_active_label_list = {"label3"};
  vector<string> active_label_list;
  vector<NodePtr> added_active_nodes;

  ret = stream_allocator->AddActiveNodes(node_a, ori_active_label_list, active_label_list, added_active_nodes);
  EXPECT_EQ(ret, FAILED);
}

TEST_F(UtestStreamAllocator, test_split_streams_active) {
  const auto &graph = std::make_shared<ComputeGraph>("test_split_streams_active_graph");
  EXPECT_NE(graph, nullptr);
  make_graph_active(graph);

  StreamAllocator allocator(graph, Graph2SubGraphInfoList());
  allocator.stream_num_ = 3;
  EXPECT_EQ(allocator.SetActiveStreamsByLabel(), SUCCESS);
  std::vector<std::set<int64_t>> split_stream(3);
  auto node_id_to_node_tasks = MakeTaskDefsByGraph(graph, 3);
  EXPECT_EQ(allocator.SplitStreams(node_id_to_node_tasks, split_stream), SUCCESS);
  EXPECT_EQ(allocator.UpdateActiveStreams(split_stream), SUCCESS);
  EXPECT_EQ(allocator.SetActiveStreamsForLoop(), SUCCESS);
  EXPECT_EQ(allocator.specific_activated_streams_.count(3), 1);

  const auto &node_b = graph->FindNode("B");
  EXPECT_NE(node_b, nullptr);
  std::vector<uint32_t> active_stream_list;
  EXPECT_TRUE(AttrUtils::GetListInt(node_b->GetOpDesc(), ATTR_NAME_ACTIVE_STREAM_LIST, active_stream_list));
  EXPECT_EQ(active_stream_list.size(), 2);
  const auto &node_e = graph->FindNode("E");
  EXPECT_NE(node_e, nullptr);
  EXPECT_EQ(active_stream_list[0], node_e->GetOpDesc()->GetStreamId());
  EXPECT_EQ(active_stream_list[1], 3);
  auto active_node_id = node_b->GetOpDesc()->GetId();
  auto active_node_stream_id = node_b->GetOpDesc()->GetStreamId();
  EXPECT_EQ(allocator.node_id_to_task_num_infos_[active_node_id][active_node_stream_id], 3);
}

TEST_F(UtestStreamAllocator, AssignSingleStream_disable_single_stream_success) {
  Graph2SubGraphInfoList subgraphs;
  StreamAllocator allocator(nullptr, subgraphs);
  allocator.enable_single_stream_ = false;
  std::unordered_map<int64_t, std::vector<domi::TaskDef>> node_id_to_node_tasks;
  EXPECT_EQ(allocator.AssignSingleStream(node_id_to_node_tasks), SUCCESS);
}

TEST_F(UtestStreamAllocator, AssignSingleStream_stream_num_not_1_fail) {
  Graph2SubGraphInfoList subgraphs;
  StreamAllocator allocator(nullptr, subgraphs);
  allocator.enable_single_stream_ = true;
  allocator.stream_num_ = 2;
  allocator.main_stream_num_ = 2;
  EXPECT_TRUE(allocator.enable_single_stream_);
  std::unordered_map<int64_t, std::vector<domi::TaskDef>> node_id_to_node_tasks;
  EXPECT_NE(allocator.AssignSingleStream(node_id_to_node_tasks), SUCCESS);
}

TEST_F(UtestStreamAllocator, AssignSingleStream_normal_stream_success) {
  auto desc_temp_ptr = make_shared<ge::GeTensorDesc>();
  auto desc_temp = *desc_temp_ptr;
  ge::OpDescPtr op1 = make_shared<ge::OpDesc>("A", RELU);
  op1->AddInputDesc(desc_temp);
  op1->AddOutputDesc(desc_temp);

  ge::ComputeGraphPtr graph = make_shared<ge::ComputeGraph>("");
  graph->AddNode(op1);

  Graph2SubGraphInfoList subgraphs;
  StreamAllocator allocator(graph, subgraphs);
  allocator.enable_single_stream_ = true;
  allocator.stream_num_ = 1;
  allocator.main_stream_num_ = 1;
  auto node_id_to_node_tasks = MakeTaskDefsByGraph(graph, 3);
  EXPECT_EQ(allocator.AssignSingleStream(node_id_to_node_tasks), SUCCESS);
  EXPECT_EQ(allocator.huge_streams_.size(), 0);
}

TEST_F(UtestStreamAllocator, SplitStreams_with_2_stream_success) {
  ge::ComputeGraphPtr graph = make_shared<ge::ComputeGraph>("");

  auto desc_temp_ptr = make_shared<ge::GeTensorDesc>();
  auto desc_temp = *desc_temp_ptr;

  // task count: kTaskNumPerNormalNode=3
  ge::OpDescPtr op1 = make_shared<ge::OpDesc>("A", RELU);
  op1->AddInputDesc(desc_temp);
  op1->AddOutputDesc(desc_temp);
  graph->AddNode(op1);

  // task count: kTaskNumPerHcclNode=245
  for (int i = 0; i < 5; ++i) {
    ge::OpDescPtr op2 = make_shared<ge::OpDesc>("A" + std::to_string(i), HCOMALLREDUCE);
    op2->AddInputDesc(desc_temp);
    op2->AddOutputDesc(desc_temp);
    graph->AddNode(op2);
  }

  Graph2SubGraphInfoList subgraphs;
  StreamAllocator allocator(graph, subgraphs);
  allocator.stream_num_ = 1;
  allocator.main_stream_num_ = 1;
  auto node_id_to_node_tasks = MakeTaskDefsByGraph(graph, 3);
  EXPECT_EQ(allocator.AssignSingleStream(node_id_to_node_tasks), SUCCESS);

  std::vector<std::set<int64_t>> split_stream(1);
  EXPECT_EQ(allocator.SplitStreams(node_id_to_node_tasks, split_stream), SUCCESS);
  EXPECT_EQ(allocator.stream_num_, 2);
}

TEST_F(UtestStreamAllocator, AssignSingleStream_huge_stream_SplitStreams_with_1_stream_success) {
  ge::ComputeGraphPtr graph = make_shared<ge::ComputeGraph>("");

  auto desc_temp_ptr = make_shared<ge::GeTensorDesc>();
  auto desc_temp = *desc_temp_ptr;

  // task count: kTaskNumPerNormalNode=3
  ge::OpDescPtr op1 = make_shared<ge::OpDesc>("A", RELU);
  op1->AddInputDesc(desc_temp);
  op1->AddOutputDesc(desc_temp);
  graph->AddNode(op1);

  // task count: kTaskNumPerHcclNode=245
  for (int i = 0; i < 5; ++i) {
    ge::OpDescPtr op2 = make_shared<ge::OpDesc>("A" + std::to_string(i), HCOMALLREDUCE);
    op2->AddInputDesc(desc_temp);
    op2->AddOutputDesc(desc_temp);
    graph->AddNode(op2);
  }

  Graph2SubGraphInfoList subgraphs;
  StreamAllocator allocator(graph, subgraphs);
  allocator.enable_single_stream_ = true;
  allocator.stream_num_ = 1;
  allocator.main_stream_num_ = 1;
  auto node_id_to_node_tasks = MakeTaskDefsByGraph(graph, 3);
  EXPECT_EQ(allocator.AssignSingleStream(node_id_to_node_tasks), SUCCESS);
  EXPECT_EQ(allocator.huge_streams_.size(), 1);
  EXPECT_EQ(allocator.huge_streams_.front(), 0);

  std::vector<std::set<int64_t>> split_stream(1);
  EXPECT_EQ(allocator.SplitStreams(node_id_to_node_tasks, split_stream), SUCCESS);
  EXPECT_EQ(allocator.stream_num_, 1);
}

TEST_F(UtestStreamAllocator, test_update_active_streams_for_subgraph) {
  const auto &root_graph = std::make_shared<ComputeGraph>("test_update_active_streams_for_subgraph_root_graph");
  EXPECT_NE(root_graph, nullptr);
  root_graph->SetGraphUnknownFlag(false);
  const auto &sub_graph1 = std::make_shared<ComputeGraph>("test_update_active_streams_for_subgraph_sub_graph1");
  EXPECT_NE(sub_graph1, nullptr);
  root_graph->AddSubGraph(sub_graph1);
  const auto &sub_graph2 = std::make_shared<ComputeGraph>("test_update_active_streams_for_subgraph_sub_graph2");
  EXPECT_NE(sub_graph2, nullptr);
  root_graph->AddSubGraph(sub_graph2);
  const auto &sub_graph3 = std::make_shared<ComputeGraph>("test_update_active_streams_for_subgraph_sub_graph3");
  EXPECT_NE(sub_graph3, nullptr);
  root_graph->AddSubGraph(sub_graph3);

  const auto &case_desc = std::make_shared<OpDesc>("case", CASE);
  EXPECT_NE(case_desc, nullptr);
  EXPECT_EQ(case_desc->AddInputDesc(GeTensorDesc()), GRAPH_SUCCESS);
  EXPECT_EQ(case_desc->AddOutputDesc(GeTensorDesc()), GRAPH_SUCCESS);
  case_desc->AddSubgraphName("branch1");
  case_desc->SetSubgraphInstanceName(0, "test_update_active_streams_for_subgraph_sub_graph1");
  case_desc->AddSubgraphName("branch2");
  case_desc->SetSubgraphInstanceName(1, "test_update_active_streams_for_subgraph_sub_graph2");
  case_desc->AddSubgraphName("branch3");
  case_desc->SetSubgraphInstanceName(2, "test_update_active_streams_for_subgraph_sub_graph3");
  const auto &case_node = root_graph->AddNode(case_desc);
  EXPECT_NE(case_node, nullptr);
  sub_graph1->SetParentNode(case_node);
  sub_graph2->SetParentNode(case_node);
  sub_graph3->SetParentNode(case_node);

  const auto &active_desc1 = std::make_shared<OpDesc>("active1", STREAMACTIVE);
  EXPECT_NE(active_desc1, nullptr);
  active_desc1->SetStreamId(2);
  EXPECT_TRUE(AttrUtils::SetListInt(active_desc1, ATTR_NAME_ACTIVE_STREAM_LIST, {0}));
  const auto &active_node1 = sub_graph1->AddNode(active_desc1);
  EXPECT_NE(active_node1, nullptr);

  const auto &active_desc2 = std::make_shared<OpDesc>("active2", STREAMACTIVE);
  EXPECT_NE(active_desc2, nullptr);
  active_desc2->SetStreamId(3);
  EXPECT_TRUE(AttrUtils::SetListInt(active_desc2, ATTR_NAME_ACTIVE_STREAM_LIST, {1, 5}));
  const auto &active_node2 = sub_graph2->AddNode(active_desc2);
  EXPECT_NE(active_node2, nullptr);
  const auto &relu = std::make_shared<OpDesc>("relu", RELU);
  EXPECT_NE(relu, nullptr);
  relu->SetStreamId(5);
  const auto &relu_node = sub_graph2->AddNode(relu);

  const auto &active_desc3 = std::make_shared<OpDesc>("active3", STREAMACTIVE);
  EXPECT_NE(active_desc3, nullptr);
  active_desc3->SetStreamId(4);
  const auto &active_node3 = sub_graph3->AddNode(active_desc3);
  EXPECT_NE(active_node3, nullptr);

  StreamAllocator allocator(root_graph, Graph2SubGraphInfoList());
  allocator.node_split_stream_map_[active_node1] = 2;
  allocator.node_split_stream_map_[active_node2] = 3;
  allocator.node_split_stream_map_[active_node3] = 4;
  allocator.node_split_stream_map_[relu_node] = 5;
  allocator.split_stream_id_to_logic_stream_id_[2] = 0;
  allocator.split_stream_id_to_logic_stream_id_[5] = 1;
  allocator.subgraph_first_active_node_map_[sub_graph1] = active_node1;
  allocator.subgraph_first_active_node_map_[sub_graph2] = active_node2;
  allocator.subgraph_first_active_node_map_[sub_graph3] = active_node3;
  EXPECT_EQ(allocator.UpdateActiveStreamsForSubgraphs(), SUCCESS);
  std::vector<uint32_t> active_stream_list1;
  EXPECT_TRUE(AttrUtils::GetListInt(active_node1->GetOpDesc(), ATTR_NAME_ACTIVE_STREAM_LIST, active_stream_list1));
  EXPECT_EQ(active_stream_list1.size(), 1);
  EXPECT_EQ(active_stream_list1[0], 0);
  std::vector<uint32_t> active_stream_list2;
  EXPECT_TRUE(AttrUtils::GetListInt(active_node2->GetOpDesc(), ATTR_NAME_ACTIVE_STREAM_LIST, active_stream_list2));
  EXPECT_EQ(active_stream_list2.size(), 2);
  EXPECT_EQ(active_stream_list2[0], 1);
  EXPECT_EQ(active_stream_list2[1], 5);
  EXPECT_EQ(allocator.specific_activated_streams_.size(), 3);
  EXPECT_EQ(allocator.specific_activated_streams_.count(3), 1);
  EXPECT_EQ(allocator.AddEventIdWhenStreamSplit({case_node, active_node1, active_node2, true, false}), 0);
  std::vector<uint32_t> active_stream_list3;
  EXPECT_EQ(AttrUtils::GetListInt(active_node3->GetOpDesc(), ATTR_NAME_ACTIVE_STREAM_LIST, active_stream_list3), false);
  EXPECT_EQ(active_stream_list3.size(), 0);
}

TEST_F(UtestStreamAllocator, test_multi_dims_event_reuse) {
  const auto &graph = MakeMultiDimsGraph();
  Graph2SubGraphInfoList subgraphs;
  std::shared_ptr<ge::StreamAllocator> stream_allocator = make_shared<ge::StreamAllocator>(graph, subgraphs);

  stream_allocator->stream_num_ = 3;
  int64_t stream_num;
  int64_t notify_num;
  int64_t event_num;
  auto ret = stream_allocator->InsertSyncNodesByLogicStream(stream_num, event_num, notify_num);
  EXPECT_EQ(ret, GRAPH_SUCCESS);
  auto node_id_to_node_tasks = MakeTaskDefsByGraph(graph, 3);
  ret = stream_allocator->SplitStreamAndRefreshTaskDef(node_id_to_node_tasks, stream_num, event_num, notify_num);
  EXPECT_EQ(ret, GRAPH_SUCCESS);
  EXPECT_EQ(event_num, 1);
}

TEST_F(UtestStreamAllocator, test_multi_dims_with_huge_task_num) {
  const auto &graph = MakeMultiDimsGraph();
  for (const auto node : graph->GetAllNodesPtr()) {
    if (node->GetName() == "sub_relu1") {
      // to split stream from sub_relu1
      AttrUtils::SetInt(node->GetOpDescBarePtr(), ATTR_NAME_NODE_SQE_NUM, 1003);
    }
    if (node->GetType() == DATA) {
      node->GetOpDescBarePtr()->SetStreamId(-1);
    }
  }
  Graph2SubGraphInfoList subgraphs;
  std::shared_ptr<ge::StreamAllocator> stream_allocator = make_shared<ge::StreamAllocator>(graph, subgraphs);

  stream_allocator->stream_num_ = 3;
  int64_t stream_num;
  int64_t notify_num;
  int64_t event_num;
  auto ret = stream_allocator->InsertSyncNodesByLogicStream(stream_num, event_num, notify_num);
  EXPECT_EQ(ret, GRAPH_SUCCESS);
  EXPECT_EQ(event_num, 1);
  auto node_id_to_node_tasks = MakeTaskDefsByGraphWithoutData(stream_allocator->whole_graph_, 3);
  ret = stream_allocator->SplitStreamAndRefreshTaskDef(node_id_to_node_tasks, stream_num, event_num, notify_num);
  EXPECT_EQ(ret, GRAPH_SUCCESS);
  EXPECT_EQ(stream_allocator->event_num_, 3);
  const auto &case_node = graph->FindNode("case");
  ASSERT_NE(case_node, nullptr);
  const auto &out_ctrl_nodes = case_node->GetOutControlNodes();
  EXPECT_EQ(out_ctrl_nodes.size(), 1);
  EXPECT_EQ(out_ctrl_nodes.at(0)->GetType(), "Send");
  int32_t send_id = -1;
  AttrUtils::GetInt(out_ctrl_nodes.at(0)->GetOpDescBarePtr(), "event_id", send_id);
  EXPECT_TRUE(send_id >= 0);

  for (const auto node : graph->GetAllNodesPtr()) {
    if (node->GetName() == "sub_relu1") {
      const auto &in_ctrl_nodes = node->GetInControlNodes();
      EXPECT_EQ(in_ctrl_nodes.size(), 1);
      EXPECT_EQ(in_ctrl_nodes.at(0)->GetType(), "Recv");
      int32_t recv_id = -1;
      AttrUtils::GetInt(in_ctrl_nodes.at(0)->GetOpDescBarePtr(), "event_id", recv_id);
      EXPECT_TRUE(recv_id >= 0);
      EXPECT_TRUE(send_id == recv_id);
    }
  }
}

TEST_F(UtestStreamAllocator, test_ffts_skip_insert_event) {
  const auto &root_graph = std::make_shared<ComputeGraph>("test_ffts_skip_insert_event_root_graph");
  EXPECT_NE(root_graph, nullptr);
  const auto add_desc = std::make_shared<OpDesc>("add", ADD);
  add_desc->SetStreamId(0);
  const auto add_node = root_graph->AddNode(add_desc);

  const auto sub_desc = std::make_shared<OpDesc>("sub", SUB);
  sub_desc->SetStreamId(-1);
  AttrUtils::SetInt(sub_desc, "_thread_scope_id", 0);
  const auto sub_node = root_graph->AddNode(sub_desc);

  Graph2SubGraphInfoList subgraphs;
  StreamAllocator stream_allocator(root_graph, subgraphs);
  auto ret = stream_allocator.InsertOneEventInTwoNodes(EventType::kEvent, add_node, sub_node);
  EXPECT_EQ(ret, GRAPH_SUCCESS);
  EXPECT_EQ(stream_allocator.event_num_, 0);
}

TEST_F(UtestStreamAllocator, test_add_task_num_by_hccl_task_num) {
  ge::ComputeGraphPtr root_graph = make_shared<ge::ComputeGraph>("");
  auto desc_temp_ptr = make_shared<ge::GeTensorDesc>();
  auto desc_temp = *desc_temp_ptr;

  // task count: kTaskNumPerHcclNode=245
  ge::OpDescPtr hccl_op_desc = make_shared<ge::OpDesc>("HcomAllReduce", HCOMALLREDUCE);
  const int64_t hccl_node_task_num = 45;

  hccl_op_desc->AddInputDesc(desc_temp);
  hccl_op_desc->AddOutputDesc(desc_temp);
  AttrUtils::SetInt(hccl_op_desc, ATTR_NAME_HCCL_TASK_NUM, hccl_node_task_num);
  AttrUtils::SetInt(hccl_op_desc, ATTR_NAME_HCCL_ATTACHED_TASK_NUM, hccl_node_task_num);
  auto hccl_node = root_graph->AddNode(hccl_op_desc);

  Graph2SubGraphInfoList subgraphs;
  int64_t task_num = 0;
  StreamAllocator stream_allocator(root_graph, subgraphs);
  stream_allocator.AddTaskNum(hccl_node, task_num, 0U, false);
  EXPECT_EQ(task_num, hccl_node_task_num);
  int64_t attached_task_num = 0;
  stream_allocator.AddTaskNum(hccl_node, attached_task_num, 0U, true);
  EXPECT_EQ(attached_task_num, hccl_node_task_num);
}

TEST_F(UtestStreamAllocator, test_add_task_num_by_sqe_num) {
  ge::ComputeGraphPtr root_graph = make_shared<ge::ComputeGraph>("");
  auto desc_temp_ptr = make_shared<ge::GeTensorDesc>();
  auto desc_temp = *desc_temp_ptr;

  const auto &memcpy_op_desc = make_shared<ge::OpDesc>("memcpy", MEMCPYASYNC);
  const int64_t node_sqe_num = 45;

  memcpy_op_desc->AddInputDesc(desc_temp);
  memcpy_op_desc->AddOutputDesc(desc_temp);
  AttrUtils::SetInt(memcpy_op_desc, ATTR_NAME_NODE_SQE_NUM, node_sqe_num);
  const auto &memcpy_node = root_graph->AddNode(memcpy_op_desc);

  Graph2SubGraphInfoList subgraphs;
  int64_t task_num = 0;
  StreamAllocator stream_allocator(root_graph, subgraphs);
  stream_allocator.AddTaskNum(memcpy_node, task_num, 0U, false);
  EXPECT_EQ(task_num, node_sqe_num);
}

TEST_F(UtestStreamAllocator, SplitStream_OneNodeFiveTask) {
  const auto &graph = std::make_shared<ComputeGraph>("test_need_multi_task_split_streams_");
  EXPECT_NE(graph, nullptr);
  make_graph_active(graph);
  ge::AttrUtils::SetBool(graph, "_op_need_multi_task", true);
  StreamAllocator allocator(graph, Graph2SubGraphInfoList());
  allocator.stream_num_ = 3;
  std::vector<std::set<int64_t>> split_stream(3);
  auto node_id_to_node_tasks = MakeTaskDefsByGraph(graph, 5);
  EXPECT_EQ(allocator.SplitStreams(node_id_to_node_tasks, split_stream), SUCCESS);
  // stream: 400 * 5(mullti task) = 2000
  // 3 + 2000/1048 -> 5
  EXPECT_EQ(allocator.stream_num_, 5);
  auto stream_id_to_logic_stream_id = allocator.GetSplitStreamToLogicStream();
  EXPECT_EQ(stream_id_to_logic_stream_id[2], 2);
  EXPECT_EQ(stream_id_to_logic_stream_id[3], 2);
  EXPECT_EQ(stream_id_to_logic_stream_id[4], 2);
}

TEST_F(UtestStreamAllocator, SplitStream_OneTaskFiveSqeNum) {
  const auto &graph = std::make_shared<ComputeGraph>("test_sqe_num_split_streams_");
  EXPECT_NE(graph, nullptr);
  make_graph_active(graph);
  StreamAllocator allocator(graph, Graph2SubGraphInfoList());
  allocator.stream_num_ = 3;
  std::vector<std::set<int64_t>> split_stream(3);
  auto node_id_to_node_tasks = MakeTaskDefsByGraph(graph, 1, 5);
  EXPECT_EQ(allocator.SplitStreams(node_id_to_node_tasks, split_stream), SUCCESS);
  // stream: 400 * 5(sqe num) = 2000
  // 3 + 2000/1048 -> 5
  EXPECT_EQ(allocator.stream_num_, 5);
}

TEST_F(UtestStreamAllocator, SplitStreams_single_stream_success) {
  const auto &graph = std::make_shared<ComputeGraph>("test_huge_stream");
  EXPECT_NE(graph, nullptr);
  make_graph_active(graph);
  auto desc_temp_ptr = make_shared<ge::GeTensorDesc>();
  auto desc_temp = *desc_temp_ptr;
  ge::OpDescPtr hccl_op_desc = make_shared<ge::OpDesc>("HcomAllReduce", HCOMALLREDUCE);
  hccl_op_desc->AddInputDesc(desc_temp);
  hccl_op_desc->AddOutputDesc(desc_temp);
  AttrUtils::SetInt(hccl_op_desc, ATTR_NAME_HCCL_TASK_NUM, 8080);
  graph->AddNode(hccl_op_desc);
  for (const auto &node : graph->GetAllNodes()) {
    node->GetOpDescBarePtr()->SetStreamId(0);
  }
  StreamAllocator allocator(graph, Graph2SubGraphInfoList());
  allocator.enable_single_stream_ = true;
  allocator.stream_num_ = 1;
  allocator.main_stream_num_ = 1;
  auto node_id_to_node_tasks = MakeTaskDefsByGraph(graph, 3);
  EXPECT_EQ(allocator.AssignSingleStream(node_id_to_node_tasks), SUCCESS);
  EXPECT_EQ(allocator.huge_streams_.size(), 1);
  std::vector<std::set<int64_t>> split_stream(1);
  EXPECT_EQ(allocator.SplitStreams(node_id_to_node_tasks, split_stream), SUCCESS);
  // stream: 400 * 3 + 8080 = 9280
  // 9280 / 8096(when huge stream, rtStream MaxTaskNum is 8096) = 2
  EXPECT_EQ(allocator.stream_num_, 2);
}

TEST_F(UtestStreamAllocator, SplitStream_HcclTaskNum) {
  const auto &graph = std::make_shared<ComputeGraph>("test_need_multi_task_split_streams_");
  EXPECT_NE(graph, nullptr);
  make_graph_active(graph);
  auto desc_temp_ptr = make_shared<ge::GeTensorDesc>();
  auto desc_temp = *desc_temp_ptr;
  ge::OpDescPtr hccl_op_desc = make_shared<ge::OpDesc>("HcomAllReduce", HCOMALLREDUCE);
  hccl_op_desc->AddInputDesc(desc_temp);
  hccl_op_desc->AddOutputDesc(desc_temp);
  AttrUtils::SetInt(hccl_op_desc, ATTR_NAME_HCCL_TASK_NUM, 1010);
  auto hccl_node = graph->AddNode(hccl_op_desc);
  StreamAllocator allocator(graph, Graph2SubGraphInfoList());
  allocator.stream_num_ = 3;
  std::vector<std::set<int64_t>> split_stream(3);
  auto node_id_to_node_tasks = MakeTaskDefsByGraph(graph, 5);
  EXPECT_EQ(node_id_to_node_tasks[hccl_node->GetOpDesc()->GetId()].size(), 245);
  EXPECT_EQ(allocator.SplitStreams(node_id_to_node_tasks, split_stream), SUCCESS);
  CheckTaskDefStreamId(node_id_to_node_tasks, graph);
  // stream: 400 * 5 + 1010 = 3010
  // 3 + 3010/1048 -> 6
  EXPECT_EQ(allocator.stream_num_, 6);
}

TEST_F(UtestStreamAllocator, UpdateActiveStreamsForSwitchNodes_Success_EnableNotify) {
  auto graph_options = GetThreadLocalContext().GetAllGraphOptions();
  graph_options["ge.event"] = "notify";
  GetThreadLocalContext().SetGraphOption(graph_options);

  graphStatus ret = GRAPH_SUCCESS;
  ge::ComputeGraphPtr graph = make_shared<ge::ComputeGraph>("");
  graph->SetName("HandleSwitchNodes_success");
  MakeSwitchGraph(graph, "label1");

  Graph2SubGraphInfoList subgraphs;
  std::shared_ptr<ge::StreamAllocator> stream_allocator = make_shared<ge::StreamAllocator>(graph, subgraphs);

  stream_allocator->labeled_streams_["label1"].emplace(3);
  stream_allocator->labeled_streams_["label1"].emplace(4);
  stream_allocator->stream_num_ = 5;
  int64_t stream_num;
  int64_t notify_num;
  int64_t event_num;
  ret = stream_allocator->InsertSyncNodesByLogicStream(stream_num, event_num, notify_num);
  EXPECT_EQ(ret, GRAPH_SUCCESS);
  auto node_id_to_node_tasks = MakeTaskDefsByGraph(graph, 3);
  ret = stream_allocator->SplitStreamAndRefreshTaskDef(node_id_to_node_tasks, stream_num, event_num, notify_num);
  EXPECT_EQ(ret, GRAPH_SUCCESS);

  ge::NodePtr node_active = graph->FindNode("switch_StreamActive_0");
  auto op_desc = node_active->GetOpDesc();
  int64_t stream_id = op_desc->GetStreamId();
  EXPECT_EQ(stream_id, 5);
}

TEST_F(UtestStreamAllocator, RefreshRealStreamMultiDimsNotifyReuse_Success_EnableNotify) {
  auto graph_options = GetThreadLocalContext().GetAllGraphOptions();
  graph_options["ge.event"] = "notify";
  GetThreadLocalContext().SetGraphOption(graph_options);

  const auto &graph = MakeMultiDimsGraph();
  Graph2SubGraphInfoList subgraphs;
  std::shared_ptr<ge::StreamAllocator> stream_allocator = make_shared<ge::StreamAllocator>(graph, subgraphs);

  stream_allocator->stream_num_ = 3;
  int64_t stream_num;
  int64_t notify_num;
  int64_t event_num;
  auto ret = stream_allocator->InsertSyncNodesByLogicStream(stream_num, event_num, notify_num);
  EXPECT_EQ(ret, GRAPH_SUCCESS);
  EXPECT_EQ(event_num, 0);
  EXPECT_EQ(notify_num, 1);
  auto node_id_to_node_tasks = MakeTaskDefsByGraph(graph, 3);
  ret = stream_allocator->SplitStreamAndRefreshTaskDef(node_id_to_node_tasks, stream_num, event_num, notify_num);
  EXPECT_EQ(ret, GRAPH_SUCCESS);
}

/**
 * Optimization Scenario: multiple send nodes on a stream sent to a single recv node on the destination stream
 * Example:
 * Stream0            Stream1
 *   N0 - -- - notify - -
 *   |                   |
 *   N1 -- -- notify -   |
 *   |                |  |
 *   |                |  |
 *   |                \  |
 *   V                 > V
 *   N2 - - - notify - > N3
 */
TEST_F(UtestStreamAllocator, OptimizeByRecvNotifies_Success_EnableNotify) {
  std::map<int64_t, std::vector<NodePtr>> stream_nodes;
  std::vector<NodePtr> stream0_nodes;
  std::vector<NodePtr> stream1_nodes;
  auto node_0 = std::make_shared<Node>(std::make_shared<OpDesc>("node0", "testType"), nullptr);
  auto node_1 = std::make_shared<Node>(std::make_shared<OpDesc>("node1", "testType"), nullptr);
  auto node_2 = std::make_shared<Node>(std::make_shared<OpDesc>("node2", "testType"), nullptr);
  auto node_3 = std::make_shared<Node>(std::make_shared<OpDesc>("node3", "testType"), nullptr);
  stream0_nodes.push_back(node_0);
  stream0_nodes.push_back(node_1);
  stream0_nodes.push_back(node_2);
  stream1_nodes.push_back(node_3);
  stream_nodes[0] = stream0_nodes;
  stream_nodes[1] = stream1_nodes;

  Graph2SubGraphInfoList subgraphs;
  StreamAllocator allocator(nullptr, subgraphs);

  allocator.node_to_send_notifies_[node_0] = {1U};
  allocator.node_to_send_notifies_[node_1] = {2U};
  allocator.node_to_send_notifies_[node_2] = {3U};

  allocator.node_to_recv_notifies_[node_3] = {1U};
  allocator.node_to_recv_notifies_[node_3].push_back(2U);
  allocator.node_to_recv_notifies_[node_3].push_back(3U);

  EXPECT_EQ(allocator.node_to_send_notifies_[node_0].size(), 1);
  EXPECT_EQ(allocator.node_to_send_notifies_[node_1].size(), 1);
  EXPECT_EQ(allocator.node_to_send_notifies_[node_2].size(), 1);

  EXPECT_EQ(allocator.node_to_recv_notifies_[node_3].size(), 3);
  EXPECT_EQ(allocator.node_to_recv_notifies_[node_3][0], 1);
  EXPECT_EQ(allocator.node_to_recv_notifies_[node_3][1], 2);
  EXPECT_EQ(allocator.node_to_recv_notifies_[node_3][2], 3);

  EXPECT_EQ(StreamUtils::OptimizeByRecvEvents(stream_nodes, allocator.node_to_send_notifies_,
                                           allocator.node_to_recv_notifies_),
            SUCCESS);
  EXPECT_EQ(allocator.node_to_send_notifies_[node_0].size(), 0);  // notify1 notify2 is optimized
  EXPECT_EQ(allocator.node_to_send_notifies_[node_1].size(), 0);
  EXPECT_EQ(allocator.node_to_send_notifies_[node_2].size(), 1);
  EXPECT_EQ(allocator.node_to_send_notifies_[node_2][0], 3);
}

TEST_F(UtestStreamAllocator, InsertSyncNotifies_Success_EnableNotify) {
  // construct graph
  std::string func_node_name = "PartitionedCall_0";
  const auto &root_graph = MakeFunctionGraph(func_node_name, PARTITIONEDCALL);
  const auto &sub_graph = MakeSubGraph("");
  // set stream_id
  std::string stream_label = "label0";
  for (size_t i = 0U; i < sub_graph->GetDirectNodesSize(); ++i) {
    auto node = sub_graph->GetDirectNode().at(i);
    node->GetOpDesc()->SetStreamId(i);
  }

  ut::GraphBuilder::AddPartitionedCall(root_graph, func_node_name, sub_graph);
  Graph2SubGraphInfoList subgraphs;
  StreamAllocator allocator(root_graph, subgraphs);
  auto node = sub_graph->FindNode("Conv2D");
  EXPECT_NE(node, nullptr);
  (void)AttrUtils::SetStr(node->GetOpDesc(), ATTR_NAME_STREAM_LABEL, stream_label);
  std::set<NodePtr> specific_activated_nodes = {node};
  allocator.specific_activated_labels_[stream_label] = specific_activated_nodes;
  auto end_node = sub_graph->FindNode("Node_Output");
  EXPECT_NE(end_node, nullptr);
  (void)AttrUtils::SetBool(end_node->GetOpDesc(), ATTR_NAME_SUBGRAPH_END_NODE, true);
  EXPECT_EQ(allocator.InsertSyncEvents(EventType::kNotify), SUCCESS);
}

TEST_F(UtestStreamAllocator, single_stream_allow_rts_fail) {
  ComputeGraphPtr graph = nullptr;
  Graph2SubGraphInfoList subgraphs;
  StreamAllocator allocator(graph, subgraphs);
  allocator.stream_num_ = 0;
  std::vector<std::set<int64_t>> split_stream(3);
  auto node_id_to_node_tasks = MakeTaskDefsByGraph(graph, 3);
  EXPECT_EQ(allocator.SplitStreams(node_id_to_node_tasks, split_stream), SUCCESS);

  g_runtime_stub_mock = "rtGetMaxStreamAndTask";
  uint32_t max_stream_count = 0U;
  uint32_t max_task_count = 0U;
  EXPECT_NE(allocator.GetMaxStreamAndTask(false, max_stream_count, max_task_count), SUCCESS);
  EXPECT_TRUE(max_stream_count == 0U);
  EXPECT_TRUE(max_task_count == 0U);
  g_runtime_stub_mock = "";
}

TEST_F(UtestStreamAllocator, SetLogicStreamAttr_Ok_OnRefreshRealStream) {
  graphStatus ret = GRAPH_SUCCESS;

  ge::ComputeGraphPtr graph = make_shared<ge::ComputeGraph>("");
  graph->SetName("HandleSwitchNodes_success");
  MakeSwitchGraph(graph, "label1");

  Graph2SubGraphInfoList subgraphs;
  std::shared_ptr<ge::StreamAllocator> stream_allocator = make_shared<ge::StreamAllocator>(graph, subgraphs);

  stream_allocator->labeled_streams_["label1"].emplace(3);
  stream_allocator->labeled_streams_["label1"].emplace(4);
  stream_allocator->stream_num_ = 5;
  int64_t stream_num;
  int64_t notify_num;
  int64_t event_num;
  std::unordered_map<std::string, int64_t> names_to_stream_ids{};
  for(auto &node: graph->GetAllNodes()) {
    names_to_stream_ids[node->GetName()] = node->GetOpDesc()->GetStreamId();
  }
  ret = stream_allocator->InsertSyncNodesByLogicStream(stream_num, event_num, notify_num);
  EXPECT_EQ(ret, GRAPH_SUCCESS);
  auto node_id_to_node_tasks = MakeTaskDefsByGraph(graph, 3);
  ret = stream_allocator->SplitStreamAndRefreshTaskDef(node_id_to_node_tasks, stream_num, event_num, notify_num);
  EXPECT_EQ(ret, GRAPH_SUCCESS);
  std::vector<int64_t> actual_stream_id;
  for(auto &node: graph->GetAllNodes()) {
    if (names_to_stream_ids.find(node->GetName()) != names_to_stream_ids.end()) {
      int64_t logic_stream_id = -1;
      AttrUtils::GetInt(node->GetOpDesc(), "_logic_stream_id", logic_stream_id);
      EXPECT_EQ(logic_stream_id, names_to_stream_ids[node->GetName()]);
    }
  }
}

TEST_F(UtestStreamAllocator, insert_attached_event_between_two_nodes_ok) {
  const auto &graph = std::make_shared<ComputeGraph>("test_attached_event");
  EXPECT_NE(graph, nullptr);
  const auto &a_desc = std::make_shared<OpDesc>("A", VARIABLE);
  a_desc->AddInputDesc(GeTensorDesc());
  a_desc->AddOutputDesc(GeTensorDesc());
  a_desc->SetStreamId(0);
  a_desc->SetAttachedStreamId(-1);
  const auto &a_node = graph->AddNode(a_desc);

  const auto &b_desc = std::make_shared<OpDesc>("B", "testb");
  b_desc->AddInputDesc(GeTensorDesc());
  b_desc->AddOutputDesc(GeTensorDesc());
  b_desc->SetStreamId(1);
  b_desc->SetAttachedStreamId(-1);
  const auto &b_node = graph->AddNode(b_desc);

  Graph2SubGraphInfoList subgraphs;
  // 非event不支持，返回成功
  StreamAllocator allocator1(graph, subgraphs);
  EXPECT_EQ(allocator1.InsertOneEventInTwoNodesWithAttachedStream(EventType::kNotify, a_node, b_node), SUCCESS);

  // 辅流一样，不需要插入event
  a_desc->SetAttachedStreamId(1);
  b_desc->SetAttachedStreamId(1);
  StreamAllocator allocator2(graph, subgraphs);
  EXPECT_EQ(allocator2.InsertOneEventInTwoNodesWithAttachedStream(EventType::kEvent, a_node, b_node), SUCCESS);
  EXPECT_EQ(allocator2.attached_node_to_send_events_[a_node].size(), 0U);
  EXPECT_EQ(allocator2.attached_node_to_recv_events_[b_node].size(), 0U);
  EXPECT_EQ(allocator2.node_to_send_events_[a_node].size(), 0U);
  EXPECT_EQ(allocator2.node_to_recv_events_[b_node].size(), 0U);

  // 辅流不一样，a_node没有辅流，b_node有辅流
  a_desc->SetAttachedStreamId(-1);
  b_desc->SetAttachedStreamId(2);
  StreamAllocator allocator3(graph, subgraphs);
  EXPECT_EQ(allocator3.InsertOneEventInTwoNodesWithAttachedStream(EventType::kEvent, a_node, b_node), SUCCESS);
  EXPECT_EQ(allocator3.attached_node_to_send_events_[a_node].size(), 0U);
  EXPECT_EQ(allocator3.attached_node_to_recv_events_[b_node].size(), 1U);
  EXPECT_EQ(allocator3.node_to_send_events_[a_node].size(), 1U);
  EXPECT_EQ(allocator3.node_to_recv_events_[b_node].size(), 0U);

  // 辅流不一样，a_node、b_node都有辅流
  a_desc->SetAttachedStreamId(2);
  b_desc->SetAttachedStreamId(3);
  StreamAllocator allocator4(graph, subgraphs);
  EXPECT_EQ(allocator4.InsertOneEventInTwoNodesWithAttachedStream(EventType::kEvent, a_node, b_node), SUCCESS);
  EXPECT_EQ(allocator4.attached_node_to_send_events_[a_node].size(), 1U);
  EXPECT_EQ(allocator4.attached_node_to_recv_events_[b_node].size(), 1U);
  EXPECT_EQ(allocator4.node_to_send_events_[a_node].size(), 0U);
  EXPECT_EQ(allocator4.node_to_recv_events_[b_node].size(), 0U);

  // 辅流不一样，a_node有辅流，b_node没有辅流
  a_desc->SetAttachedStreamId(2);
  b_desc->SetAttachedStreamId(-1);
  StreamAllocator allocator5(graph, subgraphs);
  EXPECT_EQ(allocator5.InsertOneEventInTwoNodesWithAttachedStream(EventType::kEvent, a_node, b_node), SUCCESS);
  EXPECT_EQ(allocator5.attached_node_to_send_events_[a_node].size(), 1U);
  EXPECT_EQ(allocator5.attached_node_to_recv_events_[b_node].size(), 0U);
  EXPECT_EQ(allocator5.node_to_send_events_[a_node].size(), 0U);
  EXPECT_EQ(allocator5.node_to_recv_events_[b_node].size(), 1U);

  // a_node为data,没有辅流，b_node有辅流
  a_desc->SetType(DATA);
  a_desc->SetAttachedStreamId(-1);
  b_desc->SetAttachedStreamId(2);
  StreamAllocator allocator6(graph, subgraphs);
  EXPECT_EQ(allocator6.InsertOneEventInTwoNodesWithAttachedStream(EventType::kEvent, a_node, b_node), SUCCESS);
  EXPECT_EQ(allocator6.attached_node_to_send_events_[a_node].size(), 0U);
  EXPECT_EQ(allocator6.attached_node_to_recv_events_[b_node].size(), 0U);
  EXPECT_EQ(allocator6.node_to_send_events_[a_node].size(), 0U);
  EXPECT_EQ(allocator6.node_to_recv_events_[b_node].size(), 0U);
}

TEST_F(UtestStreamAllocator, insert_sync_events_with_attached_stream_ok) {
  const auto &graph = std::make_shared<ComputeGraph>("test_insert_event_with_attached_stream");
  EXPECT_NE(graph, nullptr);
  const auto &a_desc = std::make_shared<OpDesc>("A", DATA);
  a_desc->AddInputDesc(GeTensorDesc());
  a_desc->AddOutputDesc(GeTensorDesc());
  a_desc->SetStreamId(0);
  a_desc->SetAttachedStreamId(-1);
  const auto &a_node = graph->AddNode(a_desc);

  const auto &b_desc = std::make_shared<OpDesc>("B", "testb");
  b_desc->AddInputDesc(GeTensorDesc());
  b_desc->AddOutputDesc(GeTensorDesc());
  b_desc->SetStreamId(1);
  b_desc->SetAttachedStreamId(2);
  const auto &b_node = graph->AddNode(b_desc);

  const auto &c_desc = std::make_shared<OpDesc>("C", "IFN");
  c_desc->AddInputDesc(GeTensorDesc());
  c_desc->AddInputDesc(GeTensorDesc());
  c_desc->AddInputDesc(GeTensorDesc());
  c_desc->AddOutputDesc(GeTensorDesc());
  c_desc->SetStreamId(1);
  c_desc->SetAttachedStreamId(3);
  RecoverOpDescIrDefinition(c_desc, "IFN");
  const auto &c_node = graph->AddNode(c_desc);

  GraphUtils::AddEdge(a_node->GetOutDataAnchor(0), b_node->GetInDataAnchor(0));
  GraphUtils::AddEdge(b_node->GetOutDataAnchor(0), c_node->GetInDataAnchor(0));
  GraphUtils::AddEdge(a_node->GetOutDataAnchor(0), c_node->GetInDataAnchor(1));
  GraphUtils::AddEdge(a_node->GetOutDataAnchor(0), c_node->GetInDataAnchor(2));

  // 构造c_node ir原型和tiling值依赖，并设置资源属性
  NamedAttrs attrs;
  EXPECT_TRUE(AttrUtils::SetStr(attrs, "_attached_stream_key", "tilingsink"));
  EXPECT_TRUE(AttrUtils::SetStr(attrs, "_attached_stream_depend_value_list", "0"));
  EXPECT_TRUE(AttrUtils::SetListNamedAttrs(c_desc, ATTR_NAME_ATTACHED_STREAM_INFO, {attrs}));

  Graph2SubGraphInfoList subgraphs;
  StreamAllocator allocator(graph, subgraphs);
  EXPECT_EQ(allocator.InsertSyncEventsWithAttachedStream(EventType::kEvent), SUCCESS);
  EXPECT_EQ(allocator.attached_node_to_send_events_[b_node].size(), 1U);
  EXPECT_EQ(allocator.attached_node_to_recv_events_[c_node].size(), 1U);
}

TEST_F(UtestStreamAllocator, insert_sync_events_with_attached_stream_ok_v2) {
  dlog_setlevel(0, 0, 0);
  const auto &graph = std::make_shared<ComputeGraph>("test_insert_event_with_attached_stream");
  EXPECT_NE(graph, nullptr);
  const auto &a_desc = std::make_shared<OpDesc>("A", DATA);
  std::vector<ge::NamedAttrs> attached_stream_infos(1);
  AttrUtils::SetListNamedAttrs(a_desc, ATTR_NAME_ATTACHED_STREAM_INFO_LIST, attached_stream_infos);
  a_desc->AddInputDesc(GeTensorDesc());
  a_desc->AddOutputDesc(GeTensorDesc());
  a_desc->SetStreamId(0);
  a_desc->SetAttachedStreamIds({-1});
  const auto &a_node = graph->AddNode(a_desc);

  const auto &b_desc = std::make_shared<OpDesc>("B", "testb");
  AttrUtils::SetListNamedAttrs(b_desc, ATTR_NAME_ATTACHED_STREAM_INFO_LIST, attached_stream_infos);
  b_desc->AddInputDesc(GeTensorDesc());
  b_desc->AddOutputDesc(GeTensorDesc());
  b_desc->SetStreamId(1);
  b_desc->SetAttachedStreamIds({2});
  const auto &b_node = graph->AddNode(b_desc);

  const auto &c_desc = std::make_shared<OpDesc>("C", "IFN");
  AttrUtils::SetListNamedAttrs(c_desc, ATTR_NAME_ATTACHED_STREAM_INFO_LIST, attached_stream_infos);
  c_desc->AddInputDesc(GeTensorDesc());
  c_desc->AddInputDesc(GeTensorDesc());
  c_desc->AddInputDesc(GeTensorDesc());
  c_desc->AddOutputDesc(GeTensorDesc());
  c_desc->SetStreamId(1);
  c_desc->SetAttachedStreamIds({3});
  RecoverOpDescIrDefinition(c_desc, "IFN");
  const auto &c_node = graph->AddNode(c_desc);

  GraphUtils::AddEdge(a_node->GetOutDataAnchor(0), b_node->GetInDataAnchor(0));
  GraphUtils::AddEdge(b_node->GetOutDataAnchor(0), c_node->GetInDataAnchor(0));
  GraphUtils::AddEdge(a_node->GetOutDataAnchor(0), c_node->GetInDataAnchor(1));
  GraphUtils::AddEdge(a_node->GetOutDataAnchor(0), c_node->GetInDataAnchor(2));

  // 构造c_node ir原型和tiling值依赖，并设置资源属性
  NamedAttrs attrs;

  EXPECT_TRUE(AttrUtils::SetStr(attrs, ATTR_NAME_ATTACHED_RESOURCE_NAME, "tilingsink"));
  EXPECT_TRUE(AttrUtils::SetStr(attrs, ATTR_NAME_ATTACHED_RESOURCE_REUSE_KEY, "tilingsink"));
  EXPECT_TRUE(AttrUtils::SetListInt(attrs, ATTR_NAME_ATTACHED_RESOURCE_DEPEND_VALUE_LIST_INT, {0}));
  EXPECT_TRUE(AttrUtils::SetBool(attrs, ATTR_NAME_ATTACHED_RESOURCE_REQUIRED_FLAG, true));
  EXPECT_TRUE(AttrUtils::SetInt(attrs, ATTR_NAME_ATTACHED_RESOURCE_ID, 3));
  EXPECT_TRUE(AttrUtils::SetListNamedAttrs(c_desc, ATTR_NAME_ATTACHED_STREAM_INFO_LIST, {attrs}));

  Graph2SubGraphInfoList subgraphs;
  StreamAllocator allocator(graph, subgraphs);
  EXPECT_EQ(allocator.InsertSyncEventsWithAttachedStream(EventType::kEvent), SUCCESS);
  EXPECT_EQ(allocator.attached_node_to_send_events_[b_node].size(), 1U);
  EXPECT_EQ(allocator.attached_node_to_recv_events_[c_node].size(), 1U);
  dlog_setlevel(0, 3, 0);
}

TEST_F(UtestStreamAllocator, insert_sync_events_with_attached_stream_with_ref_data_ok) {
  const auto &graph = std::make_shared<ComputeGraph>("test_insert_event_with_attached_stream_with_ref_data");
  EXPECT_NE(graph, nullptr);
  MakeRefNodeGraphWithAttachedStream(graph);

  Graph2SubGraphInfoList subgraphs;
  StreamAllocator allocator(graph, subgraphs);
  EXPECT_EQ(allocator.InsertSyncEventsWithAttachedStream(EventType::kEvent), SUCCESS);
  // 在nodeid小于ifn算子的ref算子和ifn之间插入event
  EXPECT_EQ(allocator.attached_node_to_send_events_[graph->FindNode("D")].size(), 1U);
  EXPECT_EQ(allocator.attached_node_to_send_events_[graph->FindNode("B")].size(), 0U);
  EXPECT_EQ(allocator.attached_node_to_recv_events_[graph->FindNode("C")].size(), 1U);
  // nodeid大于ifn算子的ref算子后不插入event
  EXPECT_EQ(allocator.attached_node_to_send_events_[graph->FindNode("E")].size(), 0U);
}

TEST_F(UtestStreamAllocator, insert_sync_events_with_attached_stream_with_stream_active_ok) {
  const auto &graph = std::make_shared<ComputeGraph>("test_insert_event_with_attached_stream_with_stream_active");
  EXPECT_NE(graph, nullptr);
  const auto &a_desc = std::make_shared<OpDesc>("A", DATA);
  a_desc->AddInputDesc(GeTensorDesc());
  a_desc->AddOutputDesc(GeTensorDesc());
  const auto &a_node = graph->AddNode(a_desc);
  a_desc->SetId(0);

  const auto &b_desc = std::make_shared<OpDesc>("B", "StreamActive");
  b_desc->SetStreamId(0);
  b_desc->SetAttachedStreamId(-1);
  const auto &b_node = graph->AddNode(b_desc);
  b_desc->SetId(1);
  (void)AttrUtils::SetBool(b_desc, ATTR_NAME_SUBGRAPH_FIRST_ACTIVE, true);

  const auto &c_desc = std::make_shared<OpDesc>("C", "IFN");
  c_desc->AddInputDesc(GeTensorDesc());
  c_desc->AddInputDesc(GeTensorDesc());
  c_desc->AddInputDesc(GeTensorDesc());
  c_desc->AddOutputDesc(GeTensorDesc());
  c_desc->SetStreamId(1);
  c_desc->SetAttachedStreamId(3);
  RecoverOpDescIrDefinition(c_desc, "IFN");
  const auto &c_node = graph->AddNode(c_desc);
  c_desc->SetId(2);

  GraphUtils::AddEdge(b_node->GetOutControlAnchor(), c_node->GetInControlAnchor());
  GraphUtils::AddEdge(a_node->GetOutDataAnchor(0), c_node->GetInDataAnchor(0));
  GraphUtils::AddEdge(a_node->GetOutDataAnchor(0), c_node->GetInDataAnchor(1));
  GraphUtils::AddEdge(a_node->GetOutDataAnchor(0), c_node->GetInDataAnchor(2));

  // 构造c_node ir原型和tiling值依赖，并设置资源属性
  NamedAttrs attrs;
  EXPECT_TRUE(AttrUtils::SetStr(attrs, "_attached_stream_key", "tilingsink"));
  EXPECT_TRUE(AttrUtils::SetStr(attrs, "_attached_stream_depend_value_list", "0"));
  EXPECT_TRUE(AttrUtils::SetListNamedAttrs(c_desc, ATTR_NAME_ATTACHED_STREAM_INFO, {attrs}));

  Graph2SubGraphInfoList subgraphs;
  StreamAllocator allocator(graph, subgraphs);
  EXPECT_EQ(allocator.InsertSyncEventsWithAttachedStream(EventType::kEvent), SUCCESS);
  // 如果被依赖的节点topo id小于streamactive, 在streamActive和IFN算子之间插入event
  EXPECT_EQ(allocator.node_to_send_events_[b_node].size(), 1U);
  EXPECT_EQ(allocator.attached_node_to_recv_events_[c_node].size(), 1U);
}

TEST_F(UtestStreamAllocator, RefreshTaskDefStreamId_NoAttachedStream_success) {
  std::vector<domi::TaskDef> task_defs;
  AddTaskDefWithStreamId(task_defs, 0);
  AddTaskDefWithStreamId(task_defs, 0);
  RefreshTaskDefStreamId(false, 0, 1, task_defs);
  for (const auto &task: task_defs) {
    ASSERT_EQ(task.stream_id(), 1);
  }
}

TEST_F(UtestStreamAllocator, RefreshTaskDefStreamId_AttachedStream_success) {
  std::vector<domi::TaskDef> task_defs;
  AddTaskDefWithStreamId(task_defs, 0);
  AddTaskDefWithStreamId(task_defs, 1);
  AddTaskDefWithStreamId(task_defs, 0);
  AddTaskDefWithStreamId(task_defs, 1);
  RefreshTaskDefStreamId(true, 0, 2, task_defs);
  RefreshTaskDefStreamId(true, 1, 3, task_defs);
  ASSERT_EQ(task_defs[0].stream_id(), 2);
  ASSERT_EQ(task_defs[1].stream_id(), 3);
  ASSERT_EQ(task_defs[2].stream_id(), 2);
  ASSERT_EQ(task_defs[3].stream_id(), 3);
}
}  // namespace ge
