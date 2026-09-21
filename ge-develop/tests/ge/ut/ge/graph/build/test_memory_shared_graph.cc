/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "test_memory_shared_graph.h"
#include "ge_graph_dsl/graph_dsl.h"
#include "common/mem_conflict_share_graph.h"

namespace ge {
namespace block_mem_ut {
NodePtr GraphBuilder::AddNode(const std::string &name, const std::string &type, int in_cnt, int out_cnt,
                              uint32_t thread_dim, std::vector<int64_t> shape, Format format, DataType data_type) {
  (void)thread_dim;
  auto tensor_desc = std::make_shared<GeTensorDesc>();
  tensor_desc->SetShape(GeShape(shape));
  tensor_desc->SetFormat(format);
  tensor_desc->SetDataType(data_type);
  tensor_desc->SetOriginFormat(format);
  tensor_desc->SetOriginShape(GeShape(shape));
  tensor_desc->SetOriginDataType(data_type);

  int64_t tensor_size = 0;
  TensorUtils::CalcTensorMemSize(tensor_desc->GetShape(), format, data_type, tensor_size);
  tensor_size = (tensor_size + 32 - 1 ) / 32 * 32;
  tensor_size += 32;
  TensorUtils::SetSize(*tensor_desc, tensor_size);

  auto op_desc = std::make_shared<OpDesc>(name, type);
  for (int i = 0; i < in_cnt; ++i) {
    op_desc->AddInputDesc(tensor_desc->Clone());
  }
  for (int i = 0; i < out_cnt; ++i) {
    op_desc->AddOutputDesc(tensor_desc->Clone());
    auto output_op_desc = op_desc->MutableOutputDesc(i);
  }

  return graph_->AddNode(op_desc);
}
Status GraphBuilder::AddDataEdge(const NodePtr &src_node, int src_idx, const NodePtr &dst_node, int dst_idx) {
  return GraphUtils::AddEdge(src_node->GetOutDataAnchor(src_idx), dst_node->GetInDataAnchor(dst_idx));
}
void GraphBuilder::AddControlEdge(const NodePtr &src_node, const NodePtr &dst_node) {
  GraphUtils::AddEdge(src_node->GetOutControlAnchor(), dst_node->GetInControlAnchor());
}
NodePtr GraphBuilder::AddNode(const string &name, const string &type, std::initializer_list<std::string> input_names,
                              std::initializer_list<std::string> output_names, Format format, DataType data_type,
                              std::vector<int64_t> shape) {
  auto tensor_desc = std::make_shared<GeTensorDesc>();
  tensor_desc->SetShape(GeShape(shape));
  tensor_desc->SetFormat(format);
  tensor_desc->SetDataType(data_type);
  tensor_desc->SetOriginFormat(format);
  tensor_desc->SetOriginShape(GeShape(shape));
  tensor_desc->SetOriginDataType(data_type);

  auto op_desc = std::make_shared<OpDesc>(name, type);
  for (auto &input_name : input_names) {
    op_desc->AddInputDesc(input_name, tensor_desc->Clone());
  }
  for (auto &output_name :output_names) {
    op_desc->AddOutputDesc(output_name, tensor_desc->Clone());
  }

  return graph_->AddNode(op_desc);
}
/*
 *                     sub_graph
 *   partitioned_call  +---------------+
 *                     |    a          |
 *                     |    |          |
 *                     |    b          |
 *                     |    |          |
 *                     |  sub_netoutput|
 *                     +---------------+
 *
 */
ComputeGraphPtr BuildPartitionedCallSuspendOut() {
  // root graph builder
  auto root_builder = block_mem_ut::GraphBuilder("root_graph");
  const auto &partitioned_call = root_builder.AddNode("partitioned_call", PARTITIONEDCALL, 0, 1);

  const auto &root_graph = root_builder.GetGraph();

  // partitioncall_0 sub graph build
  auto sub_builder = block_mem_ut::GraphBuilder("partitioned_call_sub");
  const auto &a = sub_builder.AddNode("a", ADD, 0, 1);
  const auto &b = sub_builder.AddNode("b", ADD, 1, 1);
  const auto &sub_netoutput = sub_builder.AddNode("sub_netoutput", NETOUTPUT, 1, 1);
  sub_builder.AddDataEdge(a, 0, b, 0);
  sub_builder.AddDataEdge(b, 0, sub_netoutput, 0);

  AttrUtils::SetInt(sub_netoutput->GetOpDesc()->MutableInputDesc(0), "_parent_node_index", 0);

  const auto &sub_graph = sub_builder.GetGraph();
  sub_graph->SetParentNode(partitioned_call);
  sub_graph->SetParentGraph(root_graph);
  partitioned_call->GetOpDesc()->AddSubgraphName("partitioned_call_sub");
  partitioned_call->GetOpDesc()->SetSubgraphInstanceName(0, "partitioned_call_sub");

  root_graph->AddSubgraph(sub_graph->GetName(), sub_graph);
  root_graph->TopologicalSorting();
  return root_graph;
}

/*
 *                     sub_graph
 *   partitioned_call  +---------------+
 *                     |    a          |
 *                     |    |          |
 *                     |    b          |
 *                     |    |          |
 *                     |  sub_netoutput|
 *                     +---------------+
 *
 */
ComputeGraphPtr BuildNodeWithMemoSizeCalcTypeAndMemoryTypeL1() {
  // root graph builder
  auto root_builder = block_mem_ut::GraphBuilder("root_graph");
  const auto &partitioned_call = root_builder.AddNode("partitioned_call", PARTITIONEDCALL, 0, 1);

  const auto &root_graph = root_builder.GetGraph();

  // partitioncall_0 sub graph build
  auto sub_builder = block_mem_ut::GraphBuilder("partitioned_call_sub");
  const auto &a = sub_builder.AddNode("a", ADD, 0, 1);
  const auto &b = sub_builder.AddNode("b", ADD, 1, 1);
  ge::AttrUtils::SetInt(a->GetOpDesc()->MutableOutputDesc(0), ATTR_NAME_MEMORY_SIZE_CALC_TYPE, 1);
  ge::AttrUtils::SetInt(b->GetOpDescBarePtr()->MutableOutputDesc(0), ATTR_NAME_TENSOR_MEM_TYPE, RT_MEMORY_L1);
  const auto &sub_netoutput = sub_builder.AddNode("sub_netoutput", NETOUTPUT, 1, 1);
  sub_builder.AddDataEdge(a, 0, b, 0);
  sub_builder.AddDataEdge(b, 0, sub_netoutput, 0);

  AttrUtils::SetInt(sub_netoutput->GetOpDesc()->MutableInputDesc(0), "_parent_node_index", 0);

  const auto &sub_graph = sub_builder.GetGraph();
  sub_graph->SetParentNode(partitioned_call);
  sub_graph->SetParentGraph(root_graph);
  partitioned_call->GetOpDesc()->AddSubgraphName("partitioned_call_sub");
  partitioned_call->GetOpDesc()->SetSubgraphInstanceName(0, "partitioned_call_sub");

  root_graph->AddSubgraph(sub_graph->GetName(), sub_graph);
  root_graph->TopologicalSorting();
  return root_graph;
}

/*
 *      root graph
 *         data1(0)
 *          |
 *          a(1)
 *          |            ffts+ sub graph
 *   partitioned_call(2)  +------------+
 *          |             |  data2(3)  |
 *          e(7)          |   /   \    |
 *          |             |  b(4)  |ctrl|
 *        netoutput(8)    |       /    |
 *                        |    c(5)    |
 *                        |    |       |
 *                        |Netoutput(6)|
 *                        +------------+
 */
ComputeGraphPtr BuildGraphForNetoutputNotReuseData() {
  auto root_builder = block_mem_ut::GraphBuilder("root_graph");
  const auto &data = root_builder.AddNode("data1", DATA, 0, 1);
  AttrUtils::SetInt(data->GetOpDesc(), ATTR_NAME_INDEX, 0);

  const auto &a = root_builder.AddNode("a", ADD, 1, 1);
  const auto &partitioned_call = root_builder.AddNode("partitioned_call", PARTITIONEDCALL, 1, 1);
  const auto &e = root_builder.AddNode("e", ADD, 1, 1);
  const auto &netoutput = root_builder.AddNode("netoutput1", NETOUTPUT, 1, 0);

  root_builder.AddDataEdge(data, 0, a, 0);
  root_builder.AddDataEdge(a, 0, partitioned_call, 0);
  root_builder.AddDataEdge(partitioned_call, 0, e, 0);
  root_builder.AddDataEdge(e, 0, netoutput, 0);
  const auto &root_graph = root_builder.GetGraph();
  for (const auto &cur_node : root_graph->GetDirectNode()) {
    cur_node->GetOpDesc()->SetStreamId(0);
  }

  // 1.build partitioncall sub graph
  //      data2
  //      /  |
  //     b   |
  //        /
  //       c
  //       |
  //     netoutput2
  auto p_sub_builder = block_mem_ut::GraphBuilder("partitioned_call_sub");

  const auto
      &data2 = p_sub_builder.AddNode("data2", DATA, 1, 1);
  const auto &b = p_sub_builder.AddNode("b", ADD, 1, 1);
  const auto &c = p_sub_builder.AddNode("c", ADD, 1, 1);
  const auto &netoutput2 = p_sub_builder.AddNode("netoutput2", NETOUTPUT, 1, 0);
  AttrUtils::SetInt(data2->GetOpDesc(), "_parent_node_index", 0);
  AttrUtils::SetInt(netoutput2->GetOpDesc()->MutableInputDesc(0), "_parent_node_index", 0);

  p_sub_builder.AddDataEdge(data2, 0, b, 0);
  p_sub_builder.AddControlEdge(data2, c);
  p_sub_builder.AddDataEdge(c, 0, netoutput2, 0);

  const auto &sub_graph = p_sub_builder.GetGraph();
  sub_graph->SetParentNode(partitioned_call);
  sub_graph->SetParentGraph(root_graph);
  partitioned_call->GetOpDesc()->AddSubgraphName("partitioned_call");
  partitioned_call->GetOpDesc()->SetSubgraphInstanceName(0, "partitioned_call_sub");
  int64_t stream_id = 1;
  for (const auto &cur_node : sub_graph->GetDirectNode()) {
    cur_node->GetOpDesc()->SetStreamId(stream_id++);
  }
  root_graph->AddSubgraph(sub_graph->GetName(), sub_graph);
  root_graph->TopologicalSorting();
  return root_graph;
}

/*
 *  a       b          c
 *  |       |          |
 *  |   hcombroadcast  |
 *   \      |         /
 *     hcombroadcast2
 *          |
 *          d
 */
ComputeGraphPtr BuildRefNodeConnectContinuousInputNode() {
  auto hcombroadcast = OP_CFG(HCOMBROADCAST).Attr(ATTR_NAME_CONTINUOUS_INPUT, true).Attr(ATTR_NAME_REFERENCE, true).InNames({"x"}).OutNames({"x"});
  auto hcombroadcast2 = OP_CFG(HCOMBROADCAST).Attr(ATTR_NAME_CONTINUOUS_INPUT, true).Attr(ATTR_NAME_REFERENCE, true).InNames({"x", "y", "z"}).OutNames({"x", "y", "z"});
  DEF_GRAPH(g1) {
                  CHAIN(NODE("a", CAST)->NODE("hcombroadcast2", hcombroadcast2)->NODE("d", ADD)->NODE("netoutput", NETOUTPUT));
                  CHAIN(NODE("b", CAST)->NODE("hcombroadcast", hcombroadcast)->NODE("hcombroadcast2", hcombroadcast2));
                  CHAIN(NODE("c", CAST)->NODE("hcombroadcast2", hcombroadcast2));
                };

  auto graph = ToComputeGraph(g1);
  for (auto &node : graph->GetAllNodes()) {
    for (size_t i = 0U; i < node->GetOutDataNodesSize(); ++i) {
      auto out_tensor = node->GetOpDescBarePtr()->MutableOutputDesc(i);
      int64_t tensor_size = 0;
      TensorUtils::CalcTensorMemSize(out_tensor->GetShape(), out_tensor->GetFormat(), out_tensor->GetDataType(), tensor_size);
      TensorUtils::SetSize(*out_tensor, tensor_size);
    }

    for (size_t i = 0U; i < node->GetInDataNodesSize(); ++i) {
      auto in_tensor = node->GetOpDescBarePtr()->MutableInputDesc(i);
      int64_t tensor_size = 0;
      TensorUtils::CalcTensorMemSize(in_tensor->GetShape(), in_tensor->GetFormat(), in_tensor->GetDataType(), tensor_size);
      TensorUtils::SetSize(*in_tensor, tensor_size);
    }
  }
  graph->TopologicalSorting();
  return graph;
}
/*
 *     a            b
 *     |            |
 *  PhonyConcat     |
 *           \      |
 *         hcombroadcast
 *              ||
 *               c
 */
ComputeGraphPtr BuildSingleNoPaddingContinuousConnectContinuousInputNode() {
  auto hcombroadcast = OP_CFG(HCOMBROADCAST).Attr(ATTR_NAME_CONTINUOUS_INPUT, true).Attr(ATTR_NAME_REFERENCE, true).InNames({"x", "y"}).OutNames({"x", "y"});
  DEF_GRAPH(g1) {
                  CHAIN(NODE("a", CAST)->NODE("PhonyConcat", PHONYCONCAT)->NODE("hcombroadcast", hcombroadcast)->NODE("c", CAST));
                  CHAIN(NODE("b", CAST)->NODE("hcombroadcast", hcombroadcast)->NODE("c", CAST)->NODE("netoutput", NETOUTPUT));
                };

  auto graph = ToComputeGraph(g1);

  auto pc_node = graph->FindNode("PhonyConcat");
  (void)ge::AttrUtils::SetBool(pc_node->GetOpDesc(), ATTR_NAME_NOPADDING_CONTINUOUS_INPUT, true);
  (void)ge::AttrUtils::SetBool(pc_node->GetOpDesc(), ATTR_NAME_OUTPUT_REUSE_INPUT, true);
  (void)ge::AttrUtils::SetInt(pc_node->GetOpDesc(), ge::ATTR_NAME_REUSE_INPUT_ON_DIM_INDEX, 0);

  for (auto &node : graph->GetAllNodes()) {
    for (size_t i = 0U; i < node->GetOutDataNodesSize(); ++i) {
      auto out_tensor = node->GetOpDescBarePtr()->MutableOutputDesc(i);
      int64_t tensor_size = 0;
      TensorUtils::CalcTensorMemSize(out_tensor->GetShape(), out_tensor->GetFormat(), out_tensor->GetDataType(), tensor_size);
      TensorUtils::SetSize(*out_tensor, tensor_size);
    }

    for (size_t i = 0U; i < node->GetInDataNodesSize(); ++i) {
      auto in_tensor = node->GetOpDescBarePtr()->MutableInputDesc(i);
      int64_t tensor_size = 0;
      TensorUtils::CalcTensorMemSize(in_tensor->GetShape(), in_tensor->GetFormat(), in_tensor->GetDataType(), tensor_size);
      TensorUtils::SetSize(*in_tensor, tensor_size);
    }
  }
  graph->TopologicalSorting();
  return graph;
}
/*
 *         data
 *          |
 *          a(1)---ctrl
 *          |       |
 *          b(2)    |
 *          |      c(3) (stream 1)
 *          d(4)  /
 *           \   /
 *         PhonyConcat(5)
 *            |
 *            g(6)
 *  topo id: b is smaller than c
 *  size: a/PhonyConcat 8M, b/d 4M, others 2k
 */
ComputeGraphPtr BuildNoPaddingContinuousMultiInputDiffStream() {
  DEF_GRAPH(g1) {
    CHAIN(NODE("data", DATA)->NODE("a", CAST)->NODE("b", CAST)->NODE("d", CAST)->NODE("PhonyConcat", PHONYCONCAT)->NODE("g", CAST));
    CHAIN(NODE("a", CAST)->Ctrl()->NODE("c", CAST)->NODE("PhonyConcat", PHONYCONCAT));
  };

  auto graph = ToComputeGraph(g1);
  MemConflictShareGraph::TopologicalSortingMock(graph, {"a", "b", "c", "d"});
  for (auto n : graph->GetDirectNode()) {
    std::cout << n->GetName() << " ";
  }
  std::cout << std::endl;
  auto pc_node = graph->FindNode("PhonyConcat");
  (void)ge::AttrUtils::SetBool(pc_node->GetOpDesc(), ATTR_NAME_NOPADDING_CONTINUOUS_INPUT, true);
  (void)ge::AttrUtils::SetBool(pc_node->GetOpDesc(), ATTR_NAME_OUTPUT_REUSE_INPUT, true);
  (void)ge::AttrUtils::SetInt(pc_node->GetOpDesc(), ge::ATTR_NAME_REUSE_INPUT_ON_DIM_INDEX, 0);

  for (auto &node : graph->GetAllNodes()) {
    int64_t out_tensor_size = 2 * 1024;
    int64_t in_tensor_size = 2 * 1024;
    if (node->GetName() == "a" || node->GetName() == "PhonyConcat") {
      out_tensor_size = 8 * 1024 * 1024;
    }
    if (node->GetName() == "c" || node->GetName() == "d") {
      out_tensor_size = 4 * 1024 * 1024;
      // 设置d/e节点的输入shape和data type
      node->GetOpDescBarePtr()->MutableOutputDesc(0)->SetDataType(ge::DT_FLOAT);
      node->GetOpDescBarePtr()->MutableOutputDesc(0)->SetShape(ge::GeShape({1, 1024, 1024}));
    }
    for (size_t i = 0U; i < node->GetAllOutDataAnchorsSize(); ++i) {
      auto out_tensor = node->GetOpDescBarePtr()->MutableOutputDesc(i);
      TensorUtils::SetSize(*out_tensor, out_tensor_size);
    }
    if (node->GetName() == "b" || node->GetName() == "g") {
      in_tensor_size = 8 * 1024 * 1024;
    }
    // 设置PhonyConcat节点的输入shape和data type
    if (node->GetName() == "PhonyConcat") {
      in_tensor_size = 4 * 1024 * 1024;
      for (size_t i = 0U; i < node->GetAllInDataAnchorsSize(); ++i) {
        auto in_tensor = node->GetOpDescBarePtr()->MutableInputDesc(i);
        in_tensor->SetDataType(ge::DT_FLOAT);
        in_tensor->SetShape(ge::GeShape({1, 1024, 1024}));
      }
      node->GetOpDescBarePtr()->MutableOutputDesc(0)->SetDataType(ge::DT_FLOAT);
      node->GetOpDescBarePtr()->MutableOutputDesc(0)->SetShape(ge::GeShape({2, 1024, 1024}));
    }
    for (size_t i = 0U; i < node->GetAllInDataAnchorsSize(); ++i) {
      auto in_tensor = node->GetOpDescBarePtr()->MutableInputDesc(i);
      TensorUtils::SetSize(*in_tensor, in_tensor_size);
    }
    if (node->GetName() == "c") {
      node->GetOpDesc()->SetStreamId(1);
    } else {
      node->GetOpDesc()->SetStreamId(0);
    }
  }

  return graph;
}

/*
 *(stream 1)a(0)   b(1)
 *          |     / \
 *          |  ctrl  |
 *          | /      /
 *          c(2)    /
 *          |      /
 *          d(3)  /
 *           \  /
 *         PhonyConcat(4)
 *            |
 *            g(5)
 *  topo id: b is smaller than c
 *  size: a/PhonyConcat 8M, b/d 4M, others 2k
 */
ComputeGraphPtr BuildNoPaddingContinuousAndMultiStreamGraph() {
  DEF_GRAPH(g1) {
                  CHAIN(NODE("a", CAST)->NODE("c", CAST)->NODE("d", CAST)->NODE("PhonyConcat", PHONYCONCAT)->NODE("g", CAST));
                  CHAIN(NODE("b", CAST)->Ctrl()->NODE("c", CAST));
                  CHAIN(NODE("b", CAST)->NODE("PhonyConcat", PHONYCONCAT));
                };

  auto graph = ToComputeGraph(g1);
  graph->TopologicalSortingGraph();
  for (auto n : graph->GetDirectNode()) {
    std::cout << n->GetName() << " ";
  }
  std::cout << std::endl;
  auto pc_node = graph->FindNode("PhonyConcat");
  (void)ge::AttrUtils::SetBool(pc_node->GetOpDesc(), ATTR_NAME_NOPADDING_CONTINUOUS_INPUT, true);
  (void)ge::AttrUtils::SetBool(pc_node->GetOpDesc(), ATTR_NAME_OUTPUT_REUSE_INPUT, true);
  (void)ge::AttrUtils::SetInt(pc_node->GetOpDesc(), ge::ATTR_NAME_REUSE_INPUT_ON_DIM_INDEX, 0);

  for (auto &node : graph->GetAllNodes()) {
    int64_t out_tensor_size = 2 * 1024;
    int64_t in_tensor_size = 2 * 1024;
    if (node->GetName() == "a" || node->GetName() == "PhonyConcat") {
      out_tensor_size = 8 * 1024 * 1024;
    }
    if (node->GetName() == "b" || node->GetName() == "d") {
      out_tensor_size = 4 * 1024 * 1024;
      // 设置d/e节点的输入shape和data type
      node->GetOpDescBarePtr()->MutableOutputDesc(0)->SetDataType(ge::DT_FLOAT);
      node->GetOpDescBarePtr()->MutableOutputDesc(0)->SetShape(ge::GeShape({1, 1024, 1024}));
    }
    for (size_t i = 0U; i < node->GetAllOutDataAnchorsSize(); ++i) {
      auto out_tensor = node->GetOpDescBarePtr()->MutableOutputDesc(i);
      TensorUtils::SetSize(*out_tensor, out_tensor_size);
    }
    if (node->GetName() == "c" || node->GetName() == "g") {
      in_tensor_size = 8 * 1024 * 1024;
    }
    // 设置PhonyConcat节点的输入shape和data type
    if (node->GetName() == "PhonyConcat") {
      in_tensor_size = 4 * 1024 * 1024;
      for (size_t i = 0U; i < node->GetAllInDataAnchorsSize(); ++i) {
        auto in_tensor = node->GetOpDescBarePtr()->MutableInputDesc(i);
        in_tensor->SetDataType(ge::DT_FLOAT);
        in_tensor->SetShape(ge::GeShape({1, 1024, 1024}));
      }
      node->GetOpDescBarePtr()->MutableOutputDesc(0)->SetDataType(ge::DT_FLOAT);
      node->GetOpDescBarePtr()->MutableOutputDesc(0)->SetShape(ge::GeShape({2, 1024, 1024}));
    }
    for (size_t i = 0U; i < node->GetAllInDataAnchorsSize(); ++i) {
      auto in_tensor = node->GetOpDescBarePtr()->MutableInputDesc(i);
      TensorUtils::SetSize(*in_tensor, in_tensor_size);
    }
    if (node->GetName() == "a") {
      node->GetOpDesc()->SetStreamId(1);
    } else {
      node->GetOpDesc()->SetStreamId(0);
    }
  }

  return graph;
}
//     data
//      |
//      a (stream 0)
//      |                +---------------+
//  partitioncall0-------| data          |
//      |                |  |            |
//      c (stream 0)     |  b (stream 1) |
//      |                |  |            |
//      d (stream 0)     | netoutput1    |
//      |                +---------------+
//   netoutput
//
ComputeGraphPtr BuildSubGraphWithDiffStream() {
  auto root_builder = block_mem_ut::GraphBuilder("root_graph");
  const auto &data = root_builder.AddNode("data", DATA, 0, 1, 1, {1, 1, 44, 448});
  const auto &a = root_builder.AddNode("a", ADD, 1, 1, 1, {1, 1, 44, 448});
  const auto &partitioncall0 = root_builder.AddNode("partitioncall0", PARTITIONEDCALL, 1, 1, 1, {1, 1, 448, 448});
  const auto &c = root_builder.AddNode("c", ADD, 1, 1, 1, {1, 1, 44, 448});
  const auto &d = root_builder.AddNode("d", ADD, 1, 1, 1, {1, 1, 44, 448});
  const auto &netout = root_builder.AddNode("NETOUTPUT", NETOUTPUT, 1, 0, 1, {1, 1, 44, 448});

  data->GetOpDescBarePtr()->SetOpKernelLibName(kEngineNameGeLocal);
  partitioncall0->GetOpDescBarePtr()->SetOpKernelLibName(kEngineNameGeLocal);
  root_builder.AddDataEdge(data, 0, a, 0);
  root_builder.AddDataEdge(a, 0, partitioncall0, 0);
  root_builder.AddDataEdge(partitioncall0, 0, c, 0);
  root_builder.AddDataEdge(c, 0, d, 0);
  root_builder.AddDataEdge(d, 0, netout, 0);

  const auto &root_graph = root_builder.GetGraph();
  for (auto node : root_graph->GetDirectNode()) {
    node->GetOpDesc()->SetStreamId(0);
  }
  // 1.build partitioncall_0 sub graph
  //   data
  //   b
  //   netoutput1
  auto p0_sub_builder = block_mem_ut::GraphBuilder("partitioncall0_sub");
  const auto &partitioncall_0_data = p0_sub_builder.AddNode("partitioncall_0_data", DATA, 1, 1, 1, {1, 1, 48, 448});
  const auto &b = p0_sub_builder.AddNode("b", ADD, 1, 1, 1, {1, 1, 448, 448});
  const auto &partitioncall_0_netoutput = p0_sub_builder.AddNode("partitioncall_0_netoutput", NETOUTPUT, 1, 0, 1, {1, 1, 448, 448});
  AttrUtils::SetInt(partitioncall_0_data->GetOpDesc(), "_parent_node_index", 0);
  AttrUtils::SetInt(partitioncall_0_netoutput->GetOpDesc()->MutableInputDesc(0), "_parent_node_index", 0);
  partitioncall_0_data->GetOpDescBarePtr()->SetOpKernelLibName(kEngineNameGeLocal);

  // workspace memory size
  p0_sub_builder.AddDataEdge(partitioncall_0_data, 0, b, 0);
  p0_sub_builder.AddDataEdge(b, 0, partitioncall_0_netoutput, 0);
  const auto &sub_graph0 = p0_sub_builder.GetGraph();
  sub_graph0->SetParentNode(partitioncall0);
  sub_graph0->SetParentGraph(root_graph);
  partitioncall0->GetOpDesc()->AddSubgraphName("partitioncall_0");
  partitioncall0->GetOpDesc()->SetSubgraphInstanceName(0, "partitioncall0_sub");
  for (auto node : sub_graph0->GetDirectNode()) {
    node->GetOpDesc()->SetStreamId(1);
  }
  root_graph->AddSubGraph(sub_graph0);
  root_graph->TopologicalSorting();
  return root_graph;
}

//     data
//      |
//      a
//      |                +----------+
//  partitioncall0-------|   data   |
//      |                |    |     |
//      e                |    b     |
//      |                |    |     |
//      f                |    c     |
//      |                |    |     |
//   netoutput           |    d     |
//                       |    |     |
//                       |netoutput1|
//                       +----------+
ComputeGraphPtr BuildKnownSubGraph() {
  auto root_builder = block_mem_ut::GraphBuilder("root_graph");
  const auto &data = root_builder.AddNode("data", DATA, 0, 1, 1, {1, 1, 44, 448});
  const auto &a = root_builder.AddNode("a", ADD, 1, 1, 1, {1, 1, 44, 448});
  const auto &partitioncall0 = root_builder.AddNode("partitioncall0", PARTITIONEDCALL, 1, 1, 1, {1, 1, 448, 448});
  const auto &e = root_builder.AddNode("e", ADD, 1, 1, 1, {1, 1, 44, 448});
  const auto &f = root_builder.AddNode("f", ADD, 1, 1, 1, {1, 1, 44, 448});
  const auto &netout = root_builder.AddNode("NETOUTPUT", NETOUTPUT, 1, 0, 1, {1, 1, 44, 448});

  data->GetOpDescBarePtr()->SetOpKernelLibName(kEngineNameGeLocal);
  partitioncall0->GetOpDescBarePtr()->SetOpKernelLibName(kEngineNameGeLocal);
  root_builder.AddDataEdge(data, 0, a, 0);
  root_builder.AddDataEdge(a, 0, partitioncall0, 0);
  root_builder.AddDataEdge(partitioncall0, 0, e, 0);
  root_builder.AddDataEdge(e, 0, f, 0);
  root_builder.AddDataEdge(f, 0, netout, 0);

  const auto &root_graph = root_builder.GetGraph();
  for (auto node : root_graph->GetDirectNode()) {
    node->GetOpDesc()->SetStreamId(0);
  }
  // 1.build partitioncall_0 sub graph
  //   data
  //   b
  //   c
  //   d
  //   netoutput1
  auto p0_sub_builder = block_mem_ut::GraphBuilder("partitioncall0_sub");
  const auto &partitioncall_0_data = p0_sub_builder.AddNode("partitioncall_0_data", DATA, 1, 1, 1, {1, 1, 48, 448});
  const auto &b = p0_sub_builder.AddNode("b", ADD, 1, 1, 1, {1, 1, 448, 448});
  const auto &c = p0_sub_builder.AddNode("c", ADD, 1, 1, 1, {1, 1, 448, 448});
  const auto &d = p0_sub_builder.AddNode("d", ADD, 1, 1, 1, {1, 2, 448, 448});
  const auto &partitioncall_0_netoutput = p0_sub_builder.AddNode("partitioncall_0_netoutput", NETOUTPUT, 1, 0, 1, {1, 1, 448, 448});
  AttrUtils::SetInt(partitioncall_0_data->GetOpDesc(), "_parent_node_index", 0);
  AttrUtils::SetInt(partitioncall_0_netoutput->GetOpDesc()->MutableInputDesc(0), "_parent_node_index", 0);
  partitioncall_0_data->GetOpDescBarePtr()->SetOpKernelLibName(kEngineNameGeLocal);

  // workspace memory size
  p0_sub_builder.AddDataEdge(partitioncall_0_data, 0, b, 0);
  p0_sub_builder.AddDataEdge(b, 0, c, 0);
  p0_sub_builder.AddDataEdge(c, 0, d, 0);
  p0_sub_builder.AddDataEdge(d, 0, partitioncall_0_netoutput, 0);
  const auto &sub_graph0 = p0_sub_builder.GetGraph();
  sub_graph0->SetParentNode(partitioncall0);
  sub_graph0->SetParentGraph(root_graph);
  partitioncall0->GetOpDesc()->AddSubgraphName("partitioncall_0");
  partitioncall0->GetOpDesc()->SetSubgraphInstanceName(0, "partitioncall0_sub");

  root_graph->AddSubGraph(sub_graph0);
  root_graph->TopologicalSorting();
  (void) AttrUtils::SetBool(root_graph, ATTR_NAME_DYNAMIC_SHAPE_PARTITIONED, true);
  (void)AttrUtils::SetBool(root_graph, ATTR_NAME_GRAPH_UNKNOWN_FLAG, true);
  return root_graph;
}

//     data
//      |
//      a (stream 2)
//      +---------------+
//      |               |
//      b (stream 1)  RefNode (stream 0)
//      |               |
//      |               c (stream 0)
//      |               |
//      |               d (stream 0)
//      |
//   netoutput
//
ComputeGraphPtr BuildSingleOutputConnectMultiStreamAndRefNode() {
  auto root_builder = block_mem_ut::GraphBuilder("root_graph");
  const auto &data = root_builder.AddNode("data", DATA, 0, 1, 1, {1, 1, 44, 448});
  const auto &a = root_builder.AddNode("a", ADD, 1, 1, 1, {1, 1, 44, 448});
  const auto &b = root_builder.AddNode("b", ADD, 1, 1, 1, {1, 1, 448, 448});
  const auto &ref_node = root_builder.AddNode("RefNode", ADD, 1, 1, 1, {1, 1, 448, 448});
  const auto &c = root_builder.AddNode("c", ADD, 1, 1, 1, {1, 1, 44, 448});
  const auto &d = root_builder.AddNode("d", ADD, 1, 1, 1, {1, 1, 44, 448});
  const auto &netout = root_builder.AddNode("NETOUTPUT", NETOUTPUT, 1, 0, 1, {1, 1, 44, 448});

  data->GetOpDescBarePtr()->SetOpKernelLibName(kEngineNameGeLocal);
  root_builder.AddDataEdge(data, 0, a, 0);
  root_builder.AddDataEdge(a, 0, b, 0);
  root_builder.AddDataEdge(a, 0, ref_node, 0);
  root_builder.AddDataEdge(ref_node, 0, c, 0);
  root_builder.AddDataEdge(c, 0, d, 0);
  root_builder.AddDataEdge(b, 0, netout, 0);

  const auto &root_graph = root_builder.GetGraph();
  for (auto node : root_graph->GetDirectNode()) {
    node->GetOpDesc()->SetStreamId(0);
  }
  a->GetOpDesc()->SetStreamId(2);
  b->GetOpDesc()->SetStreamId(1);

  AttrUtils::SetBool(ref_node->GetOpDesc(), ATTR_NAME_REFERENCE, true);
  auto input_name = ref_node->GetOpDescBarePtr()->GetInputNameByIndex(0);
  std::map<std::string, uint32_t> output_name_idx;
  output_name_idx.insert(std::make_pair(input_name, 0));
  ref_node->GetOpDescBarePtr()->UpdateOutputName(output_name_idx);

  root_graph->TopologicalSorting();
  return root_graph;
}

//     data
//      |
//      a (stream 2)
//      +-------------------+------------------------+
//      |                   |                        |
//      b (stream 1)-ctr-> RefNode (stream 0)    RefNode2 (stream 2)
//      |                    |                       |
//      |                    c (stream 0)            e (stream 2)
//      |                    |                       |
//      |                    d (stream 0)            | d不要连接控制边到refnode2
//      |                    |                       |
//      +---------------------+----------------------+
//      f (stream 2)
//      |
//   netoutput
//
// topo order: data(0) a(1) b(2) RefNode(3) c(4) d(5) RefNode2(6) e(7) f(8) NETOUTPUT(9)
//
ComputeGraphPtr BuildSingleOutputConnectMultiStreamAndRefNode3() {
  auto root_builder = block_mem_ut::GraphBuilder("root_graph");
  const auto &data = root_builder.AddNode("data", DATA, 0, 1, 1, {1, 1, 44, 448});
  const auto &a = root_builder.AddNode("a", ADD, 1, 1, 1, {1, 1, 44, 448});
  const auto &b = root_builder.AddNode("b", ADD, 1, 1, 1, {1, 1, 448, 448});
  const auto &ref_node = root_builder.AddNode("RefNode", ADD, 1, 1, 1, {1, 1, 448, 448});
  const auto &c = root_builder.AddNode("c", ADD, 1, 1, 1, {1, 1, 44, 448});
  const auto &d = root_builder.AddNode("d", ADD, 1, 1, 1, {1, 1, 44, 448});
  const auto &ref_node2 = root_builder.AddNode("RefNode2", ADD, 1, 1, 1, {1, 1, 448, 448});
  const auto &e = root_builder.AddNode("e", ADD, 1, 1, 1, {1, 1, 44, 448});
  const auto &f = root_builder.AddNode("f", ADD, 1, 1, 1, {1, 1, 44, 448});
  const auto &netout = root_builder.AddNode("NETOUTPUT", NETOUTPUT, 1, 0, 1, {1, 1, 44, 448});

  data->GetOpDescBarePtr()->SetOpKernelLibName(kEngineNameGeLocal);
  root_builder.AddDataEdge(data, 0, a, 0);
  root_builder.AddDataEdge(a, 0, b, 0);
  root_builder.AddDataEdge(a, 0, ref_node, 0);
  root_builder.AddDataEdge(a, 0, ref_node2, 0);
  root_builder.AddDataEdge(ref_node, 0, c, 0);
  root_builder.AddDataEdge(ref_node2, 0, e, 0);
  root_builder.AddDataEdge(c, 0, d, 0);
  root_builder.AddDataEdge(b, 0, f, 0);
  root_builder.AddDataEdge(f, 0, netout, 0);

  root_builder.AddControlEdge(e, f);
  root_builder.AddControlEdge(d, f);
  root_builder.AddControlEdge(b, ref_node);

  const auto &root_graph = root_builder.GetGraph();
  for (auto node : root_graph->GetDirectNode()) {
    node->GetOpDesc()->SetStreamId(0);
  }
  a->GetOpDesc()->SetStreamId(2);
  f->GetOpDesc()->SetStreamId(2);
  e->GetOpDesc()->SetStreamId(2);
  ref_node2->GetOpDesc()->SetStreamId(2);
  b->GetOpDesc()->SetStreamId(1);

  AttrUtils::SetBool(ref_node->GetOpDesc(), ATTR_NAME_REFERENCE, true);
  auto input_name = ref_node->GetOpDescBarePtr()->GetInputNameByIndex(0);
  std::map<std::string, uint32_t> output_name_idx;
  output_name_idx.insert(std::make_pair(input_name, 0));
  ref_node->GetOpDescBarePtr()->UpdateOutputName(output_name_idx);

  AttrUtils::SetBool(ref_node2->GetOpDesc(), ATTR_NAME_REFERENCE, true);
  auto input_name2 = ref_node2->GetOpDescBarePtr()->GetInputNameByIndex(0);
  std::map<std::string, uint32_t> output_name_idx2;
  output_name_idx2.insert(std::make_pair(input_name2, 0));
  ref_node2->GetOpDescBarePtr()->UpdateOutputName(output_name_idx2);

  MemConflictShareGraph::TopologicalSortingMock(root_graph, {"data", "a", "b", "RefNode", "c", "d", "RefNode2", "e", "f"});
  std::cout << "topo sort: " << std::endl;
  for (auto n : root_graph->GetDirectNode()) {
    std::cout << n->GetName() << "(" << n->GetOpDesc()->GetId() << ") ";
  }
  std::cout << std::endl;
  return root_graph;
}

//     data
//      |
//      a (stream 2)
//      +-------------------+------------------------+
//      |                   |                        |
//      b (stream 1)-ctr-> RefNode (stream 0)  +--> RefNode2 (stream 2)
//      |                    |                 |     |
//      |                    c (stream 0)     ctrl    e (stream 2)
//      |                    |                |       |
//      |                    d (stream 0) ----+      ctrl
//      |                    |                        |
//      +---------------------+------------------------+
//      f (stream 2)
//      |
//   netoutput
//
// topo order: data(0) a(1) b(2) RefNode(3) c(4) d(5) RefNode2(6) e(7) f(8) NETOUTPUT(9)
//
ComputeGraphPtr BuildSingleOutputConnectMultiStreamAndRefNode2() {
  auto root_builder = block_mem_ut::GraphBuilder("root_graph");
  const auto &data = root_builder.AddNode("data", DATA, 0, 1, 1, {1, 1, 44, 448});
  const auto &a = root_builder.AddNode("a", ADD, 1, 1, 1, {1, 1, 44, 448});
  const auto &b = root_builder.AddNode("b", ADD, 1, 1, 1, {1, 1, 448, 448});
  const auto &ref_node = root_builder.AddNode("RefNode", ADD, 1, 1, 1, {1, 1, 448, 448});
  const auto &c = root_builder.AddNode("c", ADD, 1, 1, 1, {1, 1, 44, 448});
  const auto &d = root_builder.AddNode("d", ADD, 1, 1, 1, {1, 1, 44, 448});
  const auto &ref_node2 = root_builder.AddNode("RefNode2", ADD, 1, 1, 1, {1, 1, 448, 448});
  const auto &e = root_builder.AddNode("e", ADD, 1, 1, 1, {1, 1, 44, 448});
  const auto &f = root_builder.AddNode("f", ADD, 1, 1, 1, {1, 1, 44, 448});
  const auto &netout = root_builder.AddNode("NETOUTPUT", NETOUTPUT, 1, 0, 1, {1, 1, 44, 448});

  data->GetOpDescBarePtr()->SetOpKernelLibName(kEngineNameGeLocal);
  root_builder.AddDataEdge(data, 0, a, 0);
  root_builder.AddDataEdge(a, 0, b, 0);
  root_builder.AddDataEdge(a, 0, ref_node, 0);
  root_builder.AddDataEdge(a, 0, ref_node2, 0);
  root_builder.AddDataEdge(ref_node, 0, c, 0);
  root_builder.AddDataEdge(ref_node2, 0, e, 0);
  root_builder.AddDataEdge(c, 0, d, 0);
  root_builder.AddDataEdge(b, 0, f, 0);
  root_builder.AddDataEdge(f, 0, netout, 0);

  root_builder.AddControlEdge(e, f);
  root_builder.AddControlEdge(d, f);

  // 为了保证topo排序时，b ref_node ref_nodes2
  root_builder.AddControlEdge(b, ref_node);
  root_builder.AddControlEdge(d, ref_node2); // 与BuildSingleOutputConnectMultiStreamAndRefNode3的差异就是这个

  const auto &root_graph = root_builder.GetGraph();
  for (auto node : root_graph->GetDirectNode()) {
    node->GetOpDesc()->SetStreamId(0);
  }
  a->GetOpDesc()->SetStreamId(2);
  f->GetOpDesc()->SetStreamId(2);
  e->GetOpDesc()->SetStreamId(2);
  ref_node2->GetOpDesc()->SetStreamId(2);
  b->GetOpDesc()->SetStreamId(1);

  AttrUtils::SetBool(ref_node->GetOpDesc(), ATTR_NAME_REFERENCE, true);
  auto input_name = ref_node->GetOpDescBarePtr()->GetInputNameByIndex(0);
  std::map<std::string, uint32_t> output_name_idx;
  output_name_idx.insert(std::make_pair(input_name, 0));
  ref_node->GetOpDescBarePtr()->UpdateOutputName(output_name_idx);

  AttrUtils::SetBool(ref_node2->GetOpDesc(), ATTR_NAME_REFERENCE, true);
  auto input_name2 = ref_node2->GetOpDescBarePtr()->GetInputNameByIndex(0);
  std::map<std::string, uint32_t> output_name_idx2;
  output_name_idx2.insert(std::make_pair(input_name2, 0));
  ref_node2->GetOpDescBarePtr()->UpdateOutputName(output_name_idx2);

  root_graph->TopologicalSorting();
  std::cout << "topo sort: " << std::endl;
  for (auto n : root_graph->GetDirectNode()) {
    std::cout << n->GetName() << "(" << n->GetOpDesc()->GetId() << ") ";
  }
  std::cout << std::endl;
  return root_graph;
}

//    const
//      |                +---------------+
//  partitioncall0-------| data          |
//      |                |  |            |
//      c (stream 0)     |  b (stream 1) |
//      |                |  |            |
//      d (stream 0)     | netoutput1    |
//      |                +---------------+
//   netoutput
//
ComputeGraphPtr BuildDataRefConst() {
  auto root_builder = block_mem_ut::GraphBuilder("root_graph");
  const auto &a = root_builder.AddNode("a", CONSTANT, 0, 1, 1, {1, 1, 44, 448});
  const auto &partitioncall0 = root_builder.AddNode("partitioncall0", PARTITIONEDCALL, 1, 1, 1, {1, 1, 448, 448});
  const auto &c = root_builder.AddNode("c", ADD, 1, 1, 1, {1, 1, 44, 448});
  const auto &d = root_builder.AddNode("d", ADD, 1, 1, 1, {1, 1, 44, 448});
  const auto &netout = root_builder.AddNode("NETOUTPUT", NETOUTPUT, 1, 0, 1, {1, 1, 44, 448});

  a->GetOpDescBarePtr()->SetOutputOffset({0});
  root_builder.AddDataEdge(a, 0, partitioncall0, 0);
  root_builder.AddDataEdge(partitioncall0, 0, c, 0);
  root_builder.AddDataEdge(c, 0, d, 0);
  root_builder.AddDataEdge(d, 0, netout, 0);

  const auto &root_graph = root_builder.GetGraph();
  for (auto node : root_graph->GetDirectNode()) {
    node->GetOpDesc()->SetStreamId(0);
  }
  // 1.build partitioncall_0 sub graph
  //   data
  //   b
  //   netoutput1
  auto p0_sub_builder = block_mem_ut::GraphBuilder("partitioncall0_sub");
  const auto &partitioncall_0_data = p0_sub_builder.AddNode("partitioncall_0_data", DATA, 1, 1, 1, {1, 1, 48, 448});
  const auto &b = p0_sub_builder.AddNode("b", ADD, 1, 1, 1, {1, 1, 448, 448});
  const auto &partitioncall_0_netoutput = p0_sub_builder.AddNode("partitioncall_0_netoutput", NETOUTPUT, 1, 0, 1, {1, 1, 448, 448});
  AttrUtils::SetInt(partitioncall_0_data->GetOpDesc(), "_parent_node_index", 0);
  AttrUtils::SetInt(partitioncall_0_netoutput->GetOpDesc()->MutableInputDesc(0), "_parent_node_index", 0);

  // workspace memory size
  p0_sub_builder.AddDataEdge(partitioncall_0_data, 0, b, 0);
  p0_sub_builder.AddDataEdge(b, 0, partitioncall_0_netoutput, 0);
  const auto &sub_graph0 = p0_sub_builder.GetGraph();
  sub_graph0->SetParentNode(partitioncall0);
  sub_graph0->SetParentGraph(root_graph);
  partitioncall0->GetOpDesc()->AddSubgraphName("partitioncall_0");
  partitioncall0->GetOpDesc()->SetSubgraphInstanceName(0, "partitioncall0_sub");
  for (auto node : sub_graph0->GetDirectNode()) {
    node->GetOpDesc()->SetStreamId(1);
  }
  root_graph->AddSubGraph(sub_graph0);
  root_graph->TopologicalSorting();
  return root_graph;
}

// a-b-c-d-e
ComputeGraphPtr BuildFiveNodesGraph() {
  auto root_builder = block_mem_ut::GraphBuilder("root_graph");
  const auto &a = root_builder.AddNode("a", ADD, 0, 1, 1, {1, 1, 44, 448});
  const auto &b = root_builder.AddNode("b", ADD, 1, 1, 1, {1, 1, 44, 448});
  const auto &c = root_builder.AddNode("c", ADD, 1, 1, 1, {1, 1, 44, 448});
  const auto &d = root_builder.AddNode("d", ADD, 1, 1, 1, {1, 1, 44, 448});
  const auto &e = root_builder.AddNode("e", ADD, 1, 0, 1, {1, 1, 44, 448});

  root_builder.AddDataEdge(a, 0, b, 0);
  root_builder.AddDataEdge(b, 0, c, 0);
  root_builder.AddDataEdge(c, 0, d, 0);
  root_builder.AddDataEdge(d, 0, e, 0);
  const auto &root_graph = root_builder.GetGraph();
  root_graph->TopologicalSorting();
  return root_graph;
}

// stream0 a-b-c
// stream1 d-e
ComputeGraphPtr BuildFiveNodesDiffStreamGraph() {
  auto root_builder = block_mem_ut::GraphBuilder("root_graph");
  const auto &a = root_builder.AddNode("a", ADD, 0, 1, 1, {1, 1, 44, 448});
  const auto &b = root_builder.AddNode("b", ADD, 1, 1, 1, {1, 1, 44, 448});
  const auto &c = root_builder.AddNode("c", ADD, 1, 1, 1, {1, 1, 44, 448});
  const auto &d = root_builder.AddNode("d", ADD, 1, 1, 1, {1, 1, 44, 448});
  const auto &e = root_builder.AddNode("e", ADD, 1, 0, 1, {1, 1, 44, 448});

  root_builder.AddDataEdge(a, 0, b, 0);
  root_builder.AddDataEdge(b, 0, c, 0);

  root_builder.AddControlEdge(b, d);
  root_builder.AddDataEdge(d, 0, e, 0);
  a->GetOpDescBarePtr()->SetStreamId(0);
  b->GetOpDescBarePtr()->SetStreamId(0);
  c->GetOpDescBarePtr()->SetStreamId(0);

  d->GetOpDescBarePtr()->SetStreamId(1);
  e->GetOpDescBarePtr()->SetStreamId(1);
  const auto &root_graph = root_builder.GetGraph();
  root_graph->TopologicalSorting();
  return root_graph;
}

//     const
//      |
//      a
//      |                +-----------+
//  partitioncall0-------| data      |
//      |                |  |        |
//      d                |  b        |
//      |                |  |        |
//  netoutput            |  c        |
//                       |  |        |
//                       |netoutput1 |
//                       +-----------+
//
ComputeGraphPtr BuildReuseWithOutNodeWrapper() {
  auto root_builder = block_mem_ut::GraphBuilder("root_graph");
  const auto &const_node = root_builder.AddNode("const", CONSTANT, 0, 1, 1, {1, 1, 448, 448});
  const auto &a = root_builder.AddNode("a", ADD, 1, 1, 1, {1, 1, 448, 448});
  const auto &partitioncall0 = root_builder.AddNode("partitioncall0", PARTITIONEDCALL, 1, 1, 1, {1, 1, 448, 448});
  const auto &d = root_builder.AddNode("d", ADD, 1, 1, 1, {1, 1, 448, 448});
  const auto &netout = root_builder.AddNode("NETOUTPUT", NETOUTPUT, 1, 0, 1, {1, 1, 448, 448});

  root_builder.AddDataEdge(a, 0, partitioncall0, 0);
  root_builder.AddDataEdge(partitioncall0, 0, d, 0);
  root_builder.AddDataEdge(d, 0, netout, 0);

  const auto &root_graph = root_builder.GetGraph();
  for (auto node : root_graph->GetDirectNode()) {
    node->GetOpDesc()->SetStreamId(0);
  }
  // 1.build partitioncall_0 sub graph
  //   data
  //   b
  //   netoutput1
  auto p0_sub_builder = block_mem_ut::GraphBuilder("partitioncall0_sub");
  const auto &partitioncall_0_data = p0_sub_builder.AddNode("partitioncall_0_data", DATA, 1, 1, 1, {1, 1, 448, 448});
  const auto &b = p0_sub_builder.AddNode("b", ADD, 1, 1, 1, {1, 1, 448, 448});
  const auto &c = p0_sub_builder.AddNode("c", ADD, 1, 1, 1, {1, 1, 448, 448});
  const auto &partitioncall_0_netoutput = p0_sub_builder.AddNode("partitioncall_0_netoutput", NETOUTPUT, 1, 0, 1, {1, 1, 448, 448});
  AttrUtils::SetInt(partitioncall_0_data->GetOpDesc(), "_parent_node_index", 0);
  AttrUtils::SetInt(partitioncall_0_netoutput->GetOpDesc()->MutableInputDesc(0), "_parent_node_index", 0);

  // workspace memory size
  p0_sub_builder.AddDataEdge(partitioncall_0_data, 0, b, 0);
  p0_sub_builder.AddDataEdge(b, 0, c, 0);
  p0_sub_builder.AddDataEdge(c, 0, partitioncall_0_netoutput, 0);
  const auto &sub_graph0 = p0_sub_builder.GetGraph();
  sub_graph0->SetParentNode(partitioncall0);
  sub_graph0->SetParentGraph(root_graph);
  partitioncall0->GetOpDesc()->AddSubgraphName("partitioncall_0");
  partitioncall0->GetOpDesc()->SetSubgraphInstanceName(0, "partitioncall0_sub");
  for (auto node : sub_graph0->GetDirectNode()) {
    node->GetOpDesc()->SetStreamId(0);
  }
  root_graph->AddSubGraph(sub_graph0);
  root_graph->TopologicalSorting();
  return root_graph;
}

//     const
//      |
//      a
//      |                +-----------+
//  partitioncall0-------| data      |
//      |                |  |        |
//      d                |  b        |
//      |                |  |        |
//  netoutput            |  c        |
//                       |  |        |
//                       | hcom(输入清零)|
//                       |  |        |
//                       |netoutput1 |
//                       +-----------+
//
ComputeGraphPtr BuildAtomicCleanNotReuseSubGraphData() {
  auto root_builder = block_mem_ut::GraphBuilder("root_graph");
  const auto &const_node = root_builder.AddNode("const", CONSTANT, 0, 1, 1, {1, 1, 448, 448});
  const auto &a = root_builder.AddNode("a", ADD, 1, 1, 1, {1, 1, 448, 448});
  const auto &partitioncall0 = root_builder.AddNode("partitioncall0", PARTITIONEDCALL, 1, 1, 1, {1, 1, 448, 448});
  const auto &d = root_builder.AddNode("d", ADD, 1, 1, 1, {1, 1, 448, 448});
  const auto &netout = root_builder.AddNode("NETOUTPUT", NETOUTPUT, 1, 0, 1, {1, 1, 448, 448});

  root_builder.AddDataEdge(a, 0, partitioncall0, 0);
  root_builder.AddDataEdge(partitioncall0, 0, d, 0);
  root_builder.AddDataEdge(d, 0, netout, 0);

  const auto &root_graph = root_builder.GetGraph();
  for (auto node : root_graph->GetDirectNode()) {
    node->GetOpDesc()->SetStreamId(0);
  }
  // 1.build partitioncall_0 sub graph
  //   data
  //   b
  //   c
  //   hcom
  //   netoutput1
  auto p0_sub_builder = block_mem_ut::GraphBuilder("partitioncall0_sub");
  const auto &partitioncall_0_data = p0_sub_builder.AddNode("partitioncall_0_data", DATA, 1, 1, 1, {1, 1, 448, 448});
  const auto &b = p0_sub_builder.AddNode("b", ADD, 1, 1, 1, {1, 1, 448, 448});
  const auto &c = p0_sub_builder.AddNode("c", ADD, 1, 1, 1, {1, 1, 448, 448});
  const auto &memset = p0_sub_builder.AddNode("memset", MEMSET, 0, 0, 1, {1, 1, 448, 448});
  const auto &hcom = p0_sub_builder.AddNode("hcom", HCOMALLREDUCE, 1, 1, 1, {1, 1, 448, 448});
  const auto &partitioncall_0_netoutput = p0_sub_builder.AddNode("partitioncall_0_netoutput", NETOUTPUT, 1, 0, 1, {1, 1, 448, 448});
  AttrUtils::SetInt(partitioncall_0_data->GetOpDesc(), "_parent_node_index", 0);
  AttrUtils::SetInt(partitioncall_0_netoutput->GetOpDesc()->MutableInputDesc(0), "_parent_node_index", 0);
  std::vector<int64_t> atomic_input_index {-1};
  (void)ge::AttrUtils::SetListInt(hcom->GetOpDescBarePtr(), ATOMIC_ATTR_INPUT_INDEX, atomic_input_index);
  // workspace memory size
  p0_sub_builder.AddDataEdge(partitioncall_0_data, 0, b, 0);
  p0_sub_builder.AddDataEdge(b, 0, c, 0);
  p0_sub_builder.AddDataEdge(c, 0, hcom, 0);
  p0_sub_builder.AddDataEdge(hcom, 0, partitioncall_0_netoutput, 0);
  p0_sub_builder.AddControlEdge(memset, c);
  p0_sub_builder.AddControlEdge(memset, d);
  p0_sub_builder.AddControlEdge(memset, hcom);
  const auto &sub_graph0 = p0_sub_builder.GetGraph();
  sub_graph0->SetParentNode(partitioncall0);
  sub_graph0->SetParentGraph(root_graph);
  partitioncall0->GetOpDesc()->AddSubgraphName("partitioncall_0");
  partitioncall0->GetOpDesc()->SetSubgraphInstanceName(0, "partitioncall0_sub");
  for (auto node : sub_graph0->GetDirectNode()) {
    node->GetOpDesc()->SetStreamId(0);
  }
  root_graph->AddSubGraph(sub_graph0);
  root_graph->TopologicalSorting();
  return root_graph;
}

//     data
//      |
//  tensor_list_length DT_VARIANT 数据类型的内存不能被后面的复用
//      |
//     add1
//      |
//     add2
//      |
//     add3
//      |
//    netoutput
//
ComputeGraphPtr BuildGraphWithDtVariant() {
  auto root_builder = block_mem_ut::GraphBuilder("root_graph");
  const auto &data = root_builder.AddNode("data", DATA, 0, 1, 1, {1, 1, 44, 448});
  const auto &tensor_list_length = root_builder.AddNode("tensor_list_length", "TensorListLenght", 1, 1, 1, {1, 1, 44, 448});
  const auto &add1 = root_builder.AddNode("add1", ADD, 1, 1, 1, {1, 1, 44, 448});
  const auto &add2 = root_builder.AddNode("add2", ADD, 1, 1, 1, {1, 1, 44, 448});
  const auto &add3 = root_builder.AddNode("add3", ADD, 1, 1, 1, {1, 1, 44, 448});
  const auto &netout = root_builder.AddNode("NETOUTPUT", NETOUTPUT, 1, 0, 1, {1, 1, 44, 448});
  tensor_list_length->GetOpDescBarePtr()->MutableOutputDesc(0)->SetDataType(DT_VARIANT);

  root_builder.AddDataEdge(data, 0, tensor_list_length, 0);
  root_builder.AddDataEdge(tensor_list_length, 0, add1, 0);
  root_builder.AddDataEdge(add1, 0, add2, 0);
  root_builder.AddDataEdge(add2, 0, add3, 0);
  root_builder.AddDataEdge(add3, 0, netout, 0);
  const auto &root_graph = root_builder.GetGraph();

  root_graph->TopologicalSorting();
  return root_graph;
}

//
//     A:0    B:1
//      \  /
//     stream_merge1:2(s:0) (two out data anchor, but only has one output node)
//        |
//        C:3(s:0)  D:4(s:1)
//        \         /
//     stream_merge2:5(s:2) (C D output use same symbol)
//          |
//       netoutput:6
//
ComputeGraphPtr BuildGraphWithStreamMerge() {
  auto root_builder = block_mem_ut::GraphBuilder("root_graph");
  const auto &node_a = root_builder.AddNode("A", ADD, 0, 1, 1, {1, 1, 44, 448});
  const auto &node_b = root_builder.AddNode("B", ADD, 0, 1, 1, {1, 1, 44, 448});
  const auto &node_stream_merge1 = root_builder.AddNode("stream_merge1", STREAMMERGE, 2, 2, 1, {1, 1, 44, 448});
  const auto &node_c = root_builder.AddNode("C", ADD, 1, 1, 1, {1, 1, 44, 448});
  const auto &node_d = root_builder.AddNode("D", ADD, 0, 1, 1, {1, 1, 44, 448});
  const auto &node_stream_merge2 = root_builder.AddNode("stream_merge2", STREAMMERGE, 2, 2, 1, {1, 1, 44, 448});
  const auto &netout = root_builder.AddNode("NETOUTPUT", NETOUTPUT, 1, 0, 1, {1, 1, 44, 448});

  root_builder.AddDataEdge(node_a, 0, node_stream_merge1, 0);
  root_builder.AddDataEdge(node_b, 0, node_stream_merge1, 1);
  root_builder.AddDataEdge(node_stream_merge1, 0, node_c, 0);
  root_builder.AddDataEdge(node_d, 0, node_stream_merge2, 1);
  root_builder.AddDataEdge(node_c, 0, node_stream_merge2, 0);
  root_builder.AddDataEdge(node_stream_merge2, 0, netout, 0);
  const auto &root_graph = root_builder.GetGraph();

  root_graph->TopologicalSorting();
  node_d->GetOpDescBarePtr()->SetStreamId(1);
  node_stream_merge2->GetOpDescBarePtr()->SetStreamId(2);
  return root_graph;
}

/*
 *    var a
 *     \  /
 *     assign ref_var_src_var_name
 *       |
 *    assign_add ref_var_src_var_name
 *       |
 *     netoutput
 */
ComputeGraphPtr BuildGraphWithRefVarSrcVarName() {
  auto root_builder = block_mem_ut::GraphBuilder("root_graph");
  const auto &node_var = root_builder.AddNode("var", VARIABLE, 0, 1, 1, {1, 1, 44, 448});
  const auto &node_assign = root_builder.AddNode("assign", ASSIGN, 2, 1, 1, {1, 1, 44, 448});
  const auto &node_a = root_builder.AddNode("A", ADD, 0, 1, 1, {1, 1, 44, 448});
  const auto &node_assign_add = root_builder.AddNode("assign_add", ASSIGNADD, 1, 1, 1, {1, 1, 44, 448});
  const auto &netout = root_builder.AddNode("NETOUTPUT", NETOUTPUT, 1, 0, 1, {1, 1, 44, 448});

  root_builder.AddDataEdge(node_var, 0, node_assign, 0);
  root_builder.AddDataEdge(node_a, 0, node_assign, 1);
  root_builder.AddDataEdge(node_assign, 0, node_assign_add, 0);
  root_builder.AddDataEdge(node_assign_add, 0, netout, 0);
  const auto &root_graph = root_builder.GetGraph();

  root_graph->TopologicalSorting();
  auto out_desc = node_assign->GetOpDesc()->MutableOutputDesc(0);
  ge::AttrUtils::SetStr(out_desc, REF_VAR_SRC_VAR_NAME, "var");
  out_desc = node_assign_add->GetOpDesc()->MutableOutputDesc(0);
  ge::AttrUtils::SetStr(out_desc, REF_VAR_SRC_VAR_NAME, "var");
  return root_graph;
}

/*
 *    var a
 *     \  /
 *     assign ref_var_src_var_name
 *       |
 *    assign_add
 *       |
 *     netoutput
 */
ComputeGraphPtr BuildGraphWithRefVarSrcVarNameAbsent() {
  auto root_builder = block_mem_ut::GraphBuilder("root_graph");
  const auto &node_var = root_builder.AddNode("var", VARIABLE, 0, 1, 1, {1, 1, 44, 448});
  const auto &node_assign = root_builder.AddNode("assign", ASSIGN, 2, 1, 1, {1, 1, 44, 448});
  const auto &node_a = root_builder.AddNode("A", ADD, 0, 1, 1, {1, 1, 44, 448});
  const auto &node_assign_add = root_builder.AddNode("assign_add", ASSIGNADD, 1, 1, 1, {1, 1, 44, 448});
  const auto &netout = root_builder.AddNode("NETOUTPUT", NETOUTPUT, 1, 0, 1, {1, 1, 44, 448});

  root_builder.AddDataEdge(node_var, 0, node_assign, 0);
  root_builder.AddDataEdge(node_a, 0, node_assign, 1);
  root_builder.AddDataEdge(node_assign, 0, node_assign_add, 0);
  root_builder.AddDataEdge(node_assign_add, 0, netout, 0);
  const auto &root_graph = root_builder.GetGraph();

  root_graph->TopologicalSorting();
  return root_graph;
}

//    a   b
//     \ /
//     pc
//      |
//  netoutput
//
ComputeGraphPtr BuildContinuousInputAndOutputOffsetForBufferFusion() {
  auto builder = block_mem_ut::GraphBuilder("root_graph");
  const auto &a = builder.AddNode("a", ADD, 1, 1, 1, {1, 1, 48, 448});
  const auto &b = builder.AddNode("b", ADD, 1, 1, 1, {1, 1, 448, 448});
  const auto &pc = builder.AddNode("phonyconcat", "PhonyConcat", 2, 1, 1, {1, 1, 448, 448});
  const auto &netoutput = builder.AddNode("partitioncall_0_netoutput", NETOUTPUT, 1, 0, 1, {1, 1, 448, 448});

  int64_t mem_size = 0;
  MemReuseUtils::GetTensorSize(*a->GetOpDescBarePtr()->MutableOutputDesc(0), mem_size);
  std::vector<int64_t> offsets_of_fusion = {mem_size - 32};
  AttrUtils::SetListInt(a->GetOpDescBarePtr(), ATTR_NAME_OUTPUT_OFFSET_FOR_BUFFER_FUSION, offsets_of_fusion);

  // workspace memory size
  builder.AddDataEdge(a, 0, pc, 0);
  builder.AddDataEdge(b, 0, pc, 1);
  builder.AddDataEdge(pc, 0, netoutput, 0);

  // sgt will crate phonyconcat, and set name reference, update output name
  (void)AttrUtils::SetBool(pc->GetOpDesc(), ATTR_NAME_CONTINUOUS_INPUT, true);
  AttrUtils::SetBool(pc->GetOpDesc(), ATTR_NAME_REFERENCE, true);
  auto input_name = pc->GetOpDescBarePtr()->GetInputNameByIndex(0);
  std::map<std::string, uint32_t> output_name_idx;
  output_name_idx.insert(std::make_pair(input_name, 0));
  pc->GetOpDescBarePtr()->UpdateOutputName(output_name_idx);

  const auto &root_graph = builder.GetGraph();

  root_graph->TopologicalSorting();
  return root_graph;
}

/*
 *       c
 *       |
 *     split
 *      /  \
 *     a    b
 */
ComputeGraphPtr BuildPhonySplitGraph() {
  auto builder = block_mem_ut::GraphBuilder("root_graph");
  const auto &split = builder.AddNode("split", "PhonySplit", 1, 2, 1, {1, 1, 13750}, FORMAT_NCHW, DT_FLOAT16);
  const auto &a = builder.AddNode("a", ADD, 1, 1, 1, {1, 1, 13750}, FORMAT_NCHW, DT_FLOAT16);
  const auto &b = builder.AddNode("b", ADD, 1, 1, 1, {1, 1, 13750}, FORMAT_NCHW, DT_FLOAT16);
  const auto &c = builder.AddNode("c", ADD, 0, 1, 1, {2, 1, 13750}, FORMAT_NCHW, DT_FLOAT16);

  int64_t out_size;
  TensorUtils::GetSize(c->GetOpDescBarePtr()->GetOutputDesc(0), out_size);
  TensorUtils::GetSize(*split->GetOpDescBarePtr()->MutableInputDesc(0), out_size);

  (void)AttrUtils::SetBool(split->GetOpDesc(), ATTR_NAME_NOPADDING_CONTINUOUS_OUTPUT, true);
  (void)AttrUtils::SetBool(split->GetOpDesc(), ATTR_NAME_OUTPUT_REUSE_INPUT, true);
  (void)ge::AttrUtils::SetInt(split->GetOpDesc(), ATTR_NAME_REUSE_INPUT_ON_DIM_INDEX, 0);
  (void)ge::AttrUtils::SetBool(split->GetOpDesc(), ATTR_NAME_NOTASK, true);

  builder.AddDataEdge(c, 0, split, 0);
  builder.AddDataEdge(split, 0, a, 0);
  builder.AddDataEdge(split, 1, b, 0);

  const auto &root_graph = builder.GetGraph();
  root_graph->TopologicalSorting();
  return root_graph;
}

//          a   b   c   d
//           \ /     \ /       abcde output shape: [1,1,48,448]
//           pc1  e  pc2
//[1,2,48,448]\   |  / [1,2,48,448]
//               pc3
//                |  [1,5,48,448]
//                f
//                |
//             netoutput
//
ComputeGraphPtr BuildPhonyConcatCascated() {
  auto builder = block_mem_ut::GraphBuilder("root_graph");
  const auto &a = builder.AddNode("a", ADD, 1, 1, 1, {1, 1, 48, 448});
  const auto &b = builder.AddNode("b", ADD, 1, 1, 1, {1, 1, 48, 448});
  const auto &pc1 = builder.AddNode("pc1", "PhonyConcat", 2, 1, 1, {1, 2, 48, 448});

  const auto &c = builder.AddNode("c", ADD, 1, 1, 1, {1, 1, 48, 448});
  const auto &d = builder.AddNode("d", ADD, 1, 1, 1, {1, 1, 48, 448});
  const auto &pc2 = builder.AddNode("pc2", "PhonyConcat", 2, 1, 1, {1, 2, 48, 448});

  const auto &e = builder.AddNode("e", ADD, 1, 1, 1, {1, 1, 48, 448});
  const auto &pc3 = builder.AddNode("pc3", "PhonyConcat", 3, 1, 1, {1, 5, 48, 448});
  const auto &f = builder.AddNode("f", ADD, 1, 1, 1, {1, 1, 48, 448});
  const auto &netoutput = builder.AddNode("partitioncall_0_netoutput", NETOUTPUT, 1, 0, 1, {1, 1, 48, 448});

  int64_t mem_size = 0;
  MemReuseUtils::GetTensorSize(*a->GetOpDescBarePtr()->MutableOutputDesc(0), mem_size);
  TensorUtils::SetSize(*pc1->GetOpDescBarePtr()->MutableOutputDesc(0), 5 * mem_size);
  TensorUtils::SetSize(*pc2->GetOpDescBarePtr()->MutableOutputDesc(0), 5 * mem_size);
  TensorUtils::SetSize(*pc3->GetOpDescBarePtr()->MutableOutputDesc(0), 5 * mem_size);

  mem_size -= 32;
  std::vector<int64_t> offsets_of_fusion = {mem_size};
  AttrUtils::SetListInt(a->GetOpDescBarePtr(), ATTR_NAME_OUTPUT_OFFSET_FOR_BUFFER_FUSION, offsets_of_fusion);
  AttrUtils::SetListInt(b->GetOpDescBarePtr(), ATTR_NAME_OUTPUT_OFFSET_FOR_BUFFER_FUSION, offsets_of_fusion);
  AttrUtils::SetListInt(e->GetOpDescBarePtr(), ATTR_NAME_OUTPUT_OFFSET_FOR_BUFFER_FUSION, offsets_of_fusion);
  AttrUtils::SetListInt(c->GetOpDescBarePtr(), ATTR_NAME_OUTPUT_OFFSET_FOR_BUFFER_FUSION, offsets_of_fusion);
  AttrUtils::SetListInt(d->GetOpDescBarePtr(), ATTR_NAME_OUTPUT_OFFSET_FOR_BUFFER_FUSION, offsets_of_fusion);

  builder.AddDataEdge(a, 0, pc1, 0);
  builder.AddDataEdge(b, 0, pc1, 1);
  builder.AddDataEdge(pc1, 0, pc3, 0);

  builder.AddDataEdge(e, 0, pc3, 1);

  builder.AddDataEdge(c, 0, pc2, 0);
  builder.AddDataEdge(d, 0, pc2, 1);
  builder.AddDataEdge(pc2, 0, pc3, 2);

  builder.AddDataEdge(pc3, 0, f, 0);
  builder.AddDataEdge(f, 0, netoutput, 0);

  (void)AttrUtils::SetBool(pc1->GetOpDesc(), ATTR_NAME_NOPADDING_CONTINUOUS_INPUT, true);
  (void)AttrUtils::SetBool(pc1->GetOpDesc(), ATTR_NAME_OUTPUT_REUSE_INPUT, true);
  (void)AttrUtils::SetBool(pc1->GetOpDesc(), ATTR_NAME_NOTASK, true);
  (void)ge::AttrUtils::SetInt(pc1->GetOpDesc(), ATTR_NAME_REUSE_INPUT_ON_DIM_INDEX, 0);

  (void)AttrUtils::SetBool(pc2->GetOpDesc(), ATTR_NAME_NOPADDING_CONTINUOUS_INPUT, true);
  (void)AttrUtils::SetBool(pc2->GetOpDesc(), ATTR_NAME_OUTPUT_REUSE_INPUT, true);
  (void)AttrUtils::SetBool(pc2->GetOpDesc(), ATTR_NAME_NOTASK, true);
  (void)ge::AttrUtils::SetInt(pc2->GetOpDesc(), ATTR_NAME_REUSE_INPUT_ON_DIM_INDEX, 0);

  (void)AttrUtils::SetBool(pc3->GetOpDesc(), ATTR_NAME_NOPADDING_CONTINUOUS_INPUT, true);
  (void)AttrUtils::SetBool(pc3->GetOpDesc(), ATTR_NAME_OUTPUT_REUSE_INPUT, true);
  (void)AttrUtils::SetBool(pc3->GetOpDesc(), ATTR_NAME_NOTASK, true);
  (void)ge::AttrUtils::SetInt(pc3->GetOpDesc(), ATTR_NAME_REUSE_INPUT_ON_DIM_INDEX, 0);

  const auto &root_graph = builder.GetGraph();

  root_graph->TopologicalSorting();
  return root_graph;
}

//                        g
//                       /
//          a   b   c   d (output ref input)
//           \ /     \ /       abcde output shape: [1,1,48,448]
//           pc1  e  pc2
//[1,2,48,448]\   |  / [1,2,48,448]
//               pc3
//                |  [1,5,48,448]
//                f
//                |
//             netoutput
//
ComputeGraphPtr BuildPhonyConcatCascatedConnectRefNode() {
  auto builder = block_mem_ut::GraphBuilder("root_graph");
  const auto &a = builder.AddNode("a", ADD, 1, 1, 1, {1, 1, 48, 448});
  const auto &b = builder.AddNode("b", ADD, 1, 1, 1, {1, 1, 48, 448});
  const auto &pc1 = builder.AddNode("pc1", "PhonyConcat", 2, 1, 1, {1, 2, 48, 448});

  const auto &g = builder.AddNode("g", ADD, 1, 1, 1, {1, 1, 48, 448});
  const auto &c = builder.AddNode("c", ADD, 1, 1, 1, {1, 1, 48, 448});
  const auto &d = builder.AddNode("d", ADD, 1, 1, 1, {1, 1, 48, 448});
  const auto &pc2 = builder.AddNode("pc2", "PhonyConcat", 2, 1, 1, {1, 2, 48, 448});

  const auto &e = builder.AddNode("e", ADD, 1, 1, 1, {1, 1, 48, 448});
  const auto &pc3 = builder.AddNode("pc3", "PhonyConcat", 3, 1, 1, {1, 5, 48, 448});
  const auto &f = builder.AddNode("f", ADD, 1, 1, 1, {1, 1, 48, 448});
  const auto &netoutput = builder.AddNode("partitioncall_0_netoutput", NETOUTPUT, 1, 0, 1, {1, 1, 48, 448});

  AttrUtils::SetBool(d->GetOpDescBarePtr(), ATTR_NAME_REFERENCE, true);
  d->GetOpDescBarePtr()->MutableAllInputName() = {{"x", 0}, {"y", 1}};

  int64_t mem_size = 0;
  MemReuseUtils::GetTensorSize(*a->GetOpDescBarePtr()->MutableOutputDesc(0), mem_size);
  mem_size -= 32;
  std::vector<int64_t> offsets_of_fusion = {mem_size};
  AttrUtils::SetListInt(a->GetOpDescBarePtr(), ATTR_NAME_OUTPUT_OFFSET_FOR_BUFFER_FUSION, offsets_of_fusion);
  AttrUtils::SetListInt(b->GetOpDescBarePtr(), ATTR_NAME_OUTPUT_OFFSET_FOR_BUFFER_FUSION, offsets_of_fusion);
  AttrUtils::SetListInt(e->GetOpDescBarePtr(), ATTR_NAME_OUTPUT_OFFSET_FOR_BUFFER_FUSION, offsets_of_fusion);
  AttrUtils::SetListInt(c->GetOpDescBarePtr(), ATTR_NAME_OUTPUT_OFFSET_FOR_BUFFER_FUSION, offsets_of_fusion);
  AttrUtils::SetListInt(d->GetOpDescBarePtr(), ATTR_NAME_OUTPUT_OFFSET_FOR_BUFFER_FUSION, offsets_of_fusion);

  TensorUtils::SetSize(*pc1->GetOpDescBarePtr()->MutableOutputDesc(0), 5 * mem_size);
  TensorUtils::SetSize(*pc2->GetOpDescBarePtr()->MutableOutputDesc(0), 5 * mem_size);

  auto pc3_tensor = pc3->GetOpDescBarePtr()->MutableOutputDesc(0);
  int64_t pc3_out_size;
  TensorUtils::CalcTensorMemSize(pc3_tensor->GetShape(), pc3_tensor->GetFormat(), pc3_tensor->GetDataType(), pc3_out_size);
  TensorUtils::SetSize(*pc3->GetOpDescBarePtr()->MutableOutputDesc(0), pc3_out_size);

  builder.AddDataEdge(a, 0, pc1, 0);
  builder.AddDataEdge(b, 0, pc1, 1);
  builder.AddDataEdge(pc1, 0, pc3, 0);

  builder.AddDataEdge(e, 0, pc3, 1);

  builder.AddDataEdge(g, 0, d, 0);
  builder.AddDataEdge(c, 0, pc2, 0);
  builder.AddDataEdge(d, 0, pc2, 1);
  builder.AddDataEdge(pc2, 0, pc3, 2);

  builder.AddDataEdge(pc3, 0, f, 0);
  builder.AddDataEdge(f, 0, netoutput, 0);

  (void)AttrUtils::SetBool(pc1->GetOpDesc(), ATTR_NAME_NOPADDING_CONTINUOUS_INPUT, true);
  (void)AttrUtils::SetBool(pc1->GetOpDesc(), ATTR_NAME_OUTPUT_REUSE_INPUT, true);
  (void)AttrUtils::SetBool(pc1->GetOpDesc(), ATTR_NAME_NOTASK, true);
  (void)ge::AttrUtils::SetInt(pc1->GetOpDesc(), ATTR_NAME_REUSE_INPUT_ON_DIM_INDEX, 0);

  (void)AttrUtils::SetBool(pc2->GetOpDesc(), ATTR_NAME_NOPADDING_CONTINUOUS_INPUT, true);
  (void)AttrUtils::SetBool(pc2->GetOpDesc(), ATTR_NAME_OUTPUT_REUSE_INPUT, true);
  (void)AttrUtils::SetBool(pc2->GetOpDesc(), ATTR_NAME_NOTASK, true);
  (void)ge::AttrUtils::SetInt(pc2->GetOpDesc(), ATTR_NAME_REUSE_INPUT_ON_DIM_INDEX, 0);

  (void)AttrUtils::SetBool(pc3->GetOpDesc(), ATTR_NAME_NOPADDING_CONTINUOUS_INPUT, true);
  (void)AttrUtils::SetBool(pc3->GetOpDesc(), ATTR_NAME_OUTPUT_REUSE_INPUT, true);
  (void)AttrUtils::SetBool(pc3->GetOpDesc(), ATTR_NAME_NOTASK, true);
  (void)ge::AttrUtils::SetInt(pc3->GetOpDesc(), ATTR_NAME_REUSE_INPUT_ON_DIM_INDEX, 0);

  const auto &root_graph = builder.GetGraph();

  root_graph->TopologicalSorting();
  return root_graph;
}

//
//  a--c
// 添加控制边保证拓扑序为，a b c d
//            d       b
//            |       |
//      ref_node1   ref_node2
//             \     /
//               pc
//                |
//                e
//                |
//             netoutput
//
ComputeGraphPtr BuildPhonyConcatWithRefNodeInputs() {
  auto builder = block_mem_ut::GraphBuilder("root_graph");
  const auto &a = builder.AddNode("a", ADD, 0, 1, 1, {1, 1, 48, 448});
  const auto &b = builder.AddNode("b", ADD, 1, 1, 1, {1, 1, 48, 448});
  const auto &pc = builder.AddNode("pc", "PhonyConcat", 2, 1, 1, {1, 2, 48, 448});
  const auto &c = builder.AddNode("c", ADD, 1, 0, 1, {1, 1, 48, 448});
  const auto &d = builder.AddNode("d", ADD, 1, 1, 1, {1, 1, 48, 448});
  const auto &e = builder.AddNode("e", ADD, 1, 1, 1, {1, 1, 48, 448});

  const auto &ref_node1 = builder.AddNode("ref_node1", "ADD", 1, 1, 1, {1, 1, 48, 448});
  const auto &ref_node2 = builder.AddNode("ref_node2", ADD, 1, 1, 1, {1, 1, 48, 448});
  const auto &netoutput = builder.AddNode("partitioncall_0_netoutput", NETOUTPUT, 1, 0, 1, {1, 1, 48, 448});

  AttrUtils::SetBool(ref_node1->GetOpDescBarePtr(), ATTR_NAME_REFERENCE, true);
  ref_node1->GetOpDescBarePtr()->MutableAllInputName() = {{"x", 0}};
  ref_node1->GetOpDescBarePtr()->MutableAllOutputName() = {{"x", 0}};
  AttrUtils::SetBool(ref_node2->GetOpDescBarePtr(), ATTR_NAME_REFERENCE, true);
  ref_node2->GetOpDescBarePtr()->MutableAllInputName() = {{"x", 0}};
  ref_node2->GetOpDescBarePtr()->MutableAllOutputName() = {{"x", 0}};

  int64_t mem_size = 0;
  MemReuseUtils::GetTensorSize(*b->GetOpDescBarePtr()->MutableOutputDesc(0), mem_size);
  mem_size -= 32;

  TensorUtils::SetSize(*pc->GetOpDescBarePtr()->MutableOutputDesc(0), 2 * mem_size);

  builder.AddDataEdge(a, 0, c, 0);
  builder.AddDataEdge(d, 0, ref_node1, 0);
  builder.AddDataEdge(ref_node1, 0, pc, 0);

  builder.AddDataEdge(b, 0, ref_node2, 0);
  builder.AddDataEdge(ref_node2, 0, pc, 1);

  builder.AddDataEdge(pc, 0, e, 0);
  builder.AddDataEdge(e, 0, netoutput, 0);

  builder.AddControlEdge(a, b);
  builder.AddControlEdge(b, c);
  builder.AddControlEdge(c, d);

  (void)AttrUtils::SetBool(pc->GetOpDesc(), ATTR_NAME_NOPADDING_CONTINUOUS_INPUT, true);
  (void)AttrUtils::SetBool(pc->GetOpDesc(), ATTR_NAME_OUTPUT_REUSE_INPUT, true);
  (void)AttrUtils::SetBool(pc->GetOpDesc(), ATTR_NAME_NOTASK, true);
  (void)ge::AttrUtils::SetInt(pc->GetOpDesc(), ATTR_NAME_REUSE_INPUT_ON_DIM_INDEX, 0);

  const auto &root_graph = builder.GetGraph();

  root_graph->TopologicalSorting();
  return root_graph;
}

/*
               a
              /   \
      ref_node1<--ref_node2
             \     /  \
               pc       b
                |
                c
                |
             netoutput
 用例关注点：
 1 a的同一个输出给到了两个ref_node，并且这两个ref_node的输出又给到了同一个pc
 2 ref_node2上带有input offset，表示从a的某个偏移上读取数据
*/
ComputeGraphPtr BuildPhonyConcatWithSameInputThrougRefNode() {
  auto builder = block_mem_ut::GraphBuilder("root_graph");
  const auto &a = builder.AddNode("a", ADD, 0, 1, 1, {2, 1, 48, 448});
  const auto &pc = builder.AddNode("pc", "PhonyConcat", 2, 1, 1, {1, 2, 48, 448});
  const auto &c = builder.AddNode("c", ADD, 1, 1, 1, {1, 1, 48, 448});
  const auto &b = builder.AddNode("b", ADD, 1, 1, 1, {1, 1, 48, 448});

  const auto &ref_node1 = builder.AddNode("ref_node1", "ADD", 1, 1, 1, {1, 1, 48, 448});
  const auto &ref_node2 = builder.AddNode("ref_node2", ADD, 1, 2, 1, {1, 1, 48, 448});
  const auto &netoutput = builder.AddNode("partitioncall_0_netoutput", NETOUTPUT, 1, 0, 1, {1, 1, 48, 448});

  AttrUtils::SetBool(ref_node1->GetOpDescBarePtr(), ATTR_NAME_REFERENCE, true);
  ref_node1->GetOpDescBarePtr()->MutableAllInputName() = {{"x", 0}};
  ref_node1->GetOpDescBarePtr()->MutableAllOutputName() = {{"x", 0}};
  AttrUtils::SetBool(ref_node2->GetOpDescBarePtr(), ATTR_NAME_REFERENCE, true);
  ref_node2->GetOpDescBarePtr()->MutableAllInputName() = {{"x", 0}};
  ref_node2->GetOpDescBarePtr()->MutableAllOutputName() = {{"x", 0}, {"y", 1}};

  int64_t mem_size = 0;
  MemReuseUtils::GetTensorSize(*ref_node1->GetOpDescBarePtr()->MutableOutputDesc(0), mem_size);
  mem_size -= 32;

  ref_node2->GetOpDescBarePtr()->SetInputOffset({mem_size});
  ge::AttrUtils::SetListInt(ref_node2->GetOpDescBarePtr(), ATTR_NAME_INPUT_MEM_TYPE_LIST, {2});

  TensorUtils::SetSize(*pc->GetOpDescBarePtr()->MutableOutputDesc(0), 2 * mem_size);

  builder.AddDataEdge(a, 0, ref_node1, 0);
  builder.AddDataEdge(a, 0, ref_node2, 0);
  builder.AddDataEdge(ref_node1, 0, pc, 0);
  builder.AddDataEdge(ref_node2, 0, pc, 1);
  builder.AddDataEdge(ref_node2, 1, b, 0);

  builder.AddDataEdge(pc, 0, c, 0);
  builder.AddDataEdge(c, 0, netoutput, 0);

  // 反向刷新时，ref_node2上设置了input offset, 所以代码中要保证ref_node1和ref_node2给a刷新的offset都是正确的
  builder.AddControlEdge(ref_node2, ref_node1);

  (void)AttrUtils::SetBool(pc->GetOpDesc(), ATTR_NAME_NOPADDING_CONTINUOUS_INPUT, true);
  (void)AttrUtils::SetBool(pc->GetOpDesc(), ATTR_NAME_OUTPUT_REUSE_INPUT, true);
  (void)AttrUtils::SetBool(pc->GetOpDesc(), ATTR_NAME_NOTASK, true);
  (void)ge::AttrUtils::SetInt(pc->GetOpDesc(), ATTR_NAME_REUSE_INPUT_ON_DIM_INDEX, 0);

  const auto &root_graph = builder.GetGraph();

  root_graph->TopologicalSorting();
  return root_graph;
}

/*
               a
              /   \
      ref_node1--> ref_node2
             \     /  \
               pc       b
                |
                c
                |
             netoutput
 与上面的用例区别在与ref_node1和ref_node2的拓扑序。反向刷新时，ref_node1和ref_node2谁前进栈都得保证给a刷新的offset时正确的
*/
ComputeGraphPtr BuildPhonyConcatWithSameInputThrougRefNode2() {
  auto builder = block_mem_ut::GraphBuilder("root_graph");
  const auto &a = builder.AddNode("a", ADD, 0, 1, 1, {2, 1, 48, 448});
  const auto &pc = builder.AddNode("pc", "PhonyConcat", 2, 1, 1, {1, 2, 48, 448});
  const auto &c = builder.AddNode("c", ADD, 1, 1, 1, {1, 1, 48, 448});
  const auto &b = builder.AddNode("b", ADD, 1, 1, 1, {1, 1, 48, 448});

  const auto &ref_node1 = builder.AddNode("ref_node1", "ADD", 1, 1, 1, {1, 1, 48, 448});
  const auto &ref_node2 = builder.AddNode("ref_node2", ADD, 1, 2, 1, {1, 1, 48, 448});
  const auto &netoutput = builder.AddNode("partitioncall_0_netoutput", NETOUTPUT, 1, 0, 1, {1, 1, 48, 448});

  AttrUtils::SetBool(ref_node1->GetOpDescBarePtr(), ATTR_NAME_REFERENCE, true);
  ref_node1->GetOpDescBarePtr()->MutableAllInputName() = {{"x", 0}};
  ref_node1->GetOpDescBarePtr()->MutableAllOutputName() = {{"x", 0}};
  AttrUtils::SetBool(ref_node2->GetOpDescBarePtr(), ATTR_NAME_REFERENCE, true);
  ref_node2->GetOpDescBarePtr()->MutableAllInputName() = {{"x", 0}};
  ref_node2->GetOpDescBarePtr()->MutableAllOutputName() = {{"x", 0}, {"y", 1}};

  int64_t mem_size = 0;
  MemReuseUtils::GetTensorSize(*ref_node1->GetOpDescBarePtr()->MutableOutputDesc(0), mem_size);
  mem_size -= 32;

  ref_node2->GetOpDescBarePtr()->SetInputOffset({mem_size});
  ge::AttrUtils::SetListInt(ref_node2->GetOpDescBarePtr(), ATTR_NAME_INPUT_MEM_TYPE_LIST, {2});

  TensorUtils::SetSize(*pc->GetOpDescBarePtr()->MutableOutputDesc(0), 2 * mem_size);

  builder.AddDataEdge(a, 0, ref_node1, 0);
  builder.AddDataEdge(a, 0, ref_node2, 0);
  builder.AddDataEdge(ref_node1, 0, pc, 0);
  builder.AddDataEdge(ref_node2, 0, pc, 1);
  builder.AddDataEdge(ref_node2, 1, b, 0);

  builder.AddDataEdge(pc, 0, c, 0);
  builder.AddDataEdge(c, 0, netoutput, 0);

  // 反向刷新时，ref_node2上设置了input offset, 所以代码中要保证ref_node1和ref_node2给a刷新的offset都是正确的
  builder.AddControlEdge(ref_node1, ref_node2);

  (void)AttrUtils::SetBool(pc->GetOpDesc(), ATTR_NAME_NOPADDING_CONTINUOUS_INPUT, true);
  (void)AttrUtils::SetBool(pc->GetOpDesc(), ATTR_NAME_OUTPUT_REUSE_INPUT, true);
  (void)AttrUtils::SetBool(pc->GetOpDesc(), ATTR_NAME_NOTASK, true);
  (void)ge::AttrUtils::SetInt(pc->GetOpDesc(), ATTR_NAME_REUSE_INPUT_ON_DIM_INDEX, 0);

  const auto &root_graph = builder.GetGraph();

  root_graph->TopologicalSorting();
  return root_graph;
}

/*
 *         data       mem_set
 *          |        /  /(ctrl)
 *     a scatter_nd    /
 *      \  /          /
 *      Reshape+-----+
 *        |
 *     netoutput
 *
 *  Reshape是输出引用输入，但是比较特殊，在建符号表的时候特别判断了节点类型
 */
ComputeGraphPtr AtomicNodeConnectReShapeConnectNetoutput() {
  DEF_GRAPH(graph) {
                     auto atomic_memset = OP_CFG(MEMSET)
                         .InCnt(0)
                         .OutCnt(0);

                     auto scatter_nd = OP_CFG("AtomicNode")
                         .InCnt(1)
                         .OutCnt(1)
                         .TensorDesc(FORMAT_ND, DT_INT32, {16, 448});

                     auto reshape = OP_CFG(RESHAPE)
                         .InCnt(2)
                         .OutCnt(1)
                         .TensorDesc(FORMAT_ND, DT_INT32, {16, 224});

                     auto net_output = OP_CFG(NETOUTPUT)
                         .InCnt(1)
                         .OutCnt(0)
                         .TensorDesc(FORMAT_ND, DT_INT32, {16, 224});

                     CHAIN(NODE("mem_set", atomic_memset)->Ctrl()->NODE("scatter_nd", scatter_nd)->NODE("reshape", reshape)->NODE("netoutput", net_output));
                     CHAIN(NODE("a", RELU)->NODE("reshape", reshape));
                     CHAIN(NODE("mem_set", atomic_memset)->Ctrl()->NODE("reshape", reshape));
                   };
  auto root_graph = ToComputeGraph(graph);

  auto atomic_node = root_graph->FindNode("scatter_nd");
  std::vector<int32_t> data_list = {ge::DataType::DT_INT16};
  std::vector<int32_t> int_list = {0x1};
  std::vector<float32_t> float_list = {};
  (void) AttrUtils::SetListInt(atomic_node->GetOpDesc(), "tbe_op_atomic_dtypes", data_list);
  (void) AttrUtils::SetListInt(atomic_node->GetOpDesc(), "tbe_op_atomic_int64_values", int_list);
  (void) AttrUtils::SetListFloat(atomic_node->GetOpDesc(), "tbe_op_atomic_float_values", float_list);

  EXPECT_EQ(AttrUtils::SetBool(atomic_node->GetOpDesc(), ATOMIC_ATTR_IS_ATOMIC_NODE, true), true);
  EXPECT_EQ(AttrUtils::SetListInt(atomic_node->GetOpDesc(), ATOMIC_ATTR_OUTPUT_INDEX, {0}), true);
  for (auto &node : root_graph->GetAllNodes()) {
    for (auto &input_name : node->GetOpDesc()->GetAllInputNames()) {
      auto input = node->GetOpDesc()->MutableInputDesc(input_name);
      int64_t tensor_size = 0;
      TensorUtils::CalcTensorMemSize(input->GetShape(), input->GetFormat(), input->GetDataType(), tensor_size);
      TensorUtils::SetSize(*input, tensor_size);
    }
    auto out_size = node->GetOpDesc()->GetAllOutputsDescSize();
    for (int32_t id = 0; id < static_cast<int32_t>(out_size); id++) {
      auto output = node->GetOpDesc()->MutableOutputDesc(id);
      int64_t tensor_size = 0;
      TensorUtils::CalcTensorMemSize(output->GetShape(), output->GetFormat(), output->GetDataType(), tensor_size);
      TensorUtils::SetSize(*output, tensor_size);
    }
  }
  root_graph->TopologicalSorting();
  return root_graph;
}

/**
 *      data  data
 *        \   /
 *         add
 *          |
 *         hcom
 *        /
 *     netout
 */
ComputeGraphPtr BuildAtomicCleanInputGraph() {
  vector<std::string> engine_list = {"AIcoreEngine"};
  std::vector<int32_t> input_indexes = {-1};
  auto hcom = OP_CFG(HCOMALLREDUCE).Attr(ATTR_NAME_CONTINUOUS_INPUT, true)
                  .TensorDesc(FORMAT_ND, DT_INT32, {16, 224});
  auto add1 = OP_CFG(ADD).Attr(ATOMIC_ATTR_IS_ATOMIC_NODE, true)
                  .TensorDesc(FORMAT_ND, DT_INT32, {16, 224});
  auto data1 = OP_CFG(DATA).TensorDesc(FORMAT_ND, DT_INT32, {16, 224});
  auto data2 = OP_CFG(DATA).TensorDesc(FORMAT_ND, DT_INT32, {16, 224});
  auto netout = OP_CFG(NETOUTPUT).TensorDesc(FORMAT_ND, DT_INT32, {16, 224});

  DEF_GRAPH(g1) {
    CHAIN(NODE("data_1", data1)->EDGE(0, 0)->NODE("add_1", add1));
    CHAIN(NODE("data_2", data2)->EDGE(0, 1)->NODE("add_1", add1));
    CHAIN(NODE("add_1", add1)->EDGE(0, 0)->NODE("hcom_1", hcom));
    CHAIN(NODE("hcom_1", hcom)->EDGE(0, 0)->NODE("out_1", netout));
  };

  auto compute_graph = ToComputeGraph(g1);
  auto hcom_1 = compute_graph->FindNode("hcom_1");
  auto op_desc = hcom_1->GetOpDesc();
  (void) ge::AttrUtils::SetListInt(op_desc, ATOMIC_ATTR_INPUT_INDEX, input_indexes);

  for (auto &node : compute_graph->GetAllNodes()) {
    for (auto &input_name : node->GetOpDesc()->GetAllInputNames()) {
      auto input = node->GetOpDesc()->MutableInputDesc(input_name);
      int64_t tensor_size = 0;
      TensorUtils::CalcTensorMemSize(input->GetShape(), input->GetFormat(), input->GetDataType(), tensor_size);
      TensorUtils::SetSize(*input, tensor_size);
    }
    auto out_size = node->GetOpDesc()->GetAllOutputsDescSize();
    for (int32_t id = 0; id < static_cast<int32_t>(out_size); id++) {
      auto output = node->GetOpDesc()->MutableOutputDesc(id);
      int64_t tensor_size = 0;
      TensorUtils::CalcTensorMemSize(output->GetShape(), output->GetFormat(), output->GetDataType(), tensor_size);
      TensorUtils::SetSize(*output, tensor_size);
    }
  }
  compute_graph->TopologicalSorting();
  return compute_graph;
}

/*
 *         data       mem_set
 *          |        /  /(ctrl)
 *     a scatter_nd    /
 *      \  /          /
 *        b+-----+
 *        |
 *     netoutput
 *
 */
ComputeGraphPtr BuildAtomicCleanWorkspaceGraph() {
  DEF_GRAPH(graph) {
                     auto atomic_memset = OP_CFG(MEMSET)
                         .InCnt(0)
                         .OutCnt(0);

                     auto scatter_nd = OP_CFG("AtomicNode")
                         .InCnt(1)
                         .OutCnt(1)
                         .TensorDesc(FORMAT_ND, DT_INT32, {16, 448});

                     auto add = OP_CFG(ADD)
                         .InCnt(2)
                         .OutCnt(1)
                         .TensorDesc(FORMAT_ND, DT_INT32, {16, 224});

                     auto net_output = OP_CFG(NETOUTPUT)
                         .InCnt(1)
                         .OutCnt(0)
                         .TensorDesc(FORMAT_ND, DT_INT32, {16, 224});
                     CHAIN(NODE("a", RELU)->NODE("b", add));
                     CHAIN(NODE("mem_set", atomic_memset)->Ctrl()->NODE("scatter_nd", scatter_nd)->NODE("b", add)->NODE("netoutput", net_output));
                     CHAIN(NODE("mem_set", atomic_memset)->Ctrl()->NODE("b", add));
                   };
  auto root_graph = ToComputeGraph(graph);

  auto scatter_nd = root_graph->FindNode("scatter_nd");
  scatter_nd->GetOpDesc()->SetWorkspaceBytes({1024, 2048});
  std::map<std::string, std::map<int64_t, int64_t>> atomic_workspace_index_size;
  atomic_workspace_index_size["scatter_nd"][1] = 2048;
  scatter_nd->GetOpDesc()->SetExtAttr(ge::EXT_ATTR_ATOMIC_WORKSPACE_INFO, atomic_workspace_index_size);
  EXPECT_EQ(AttrUtils::SetBool(scatter_nd->GetOpDesc(), ATOMIC_ATTR_IS_ATOMIC_NODE, true), true);

  for (auto &node : root_graph->GetAllNodes()) {
    for (auto &input_name : node->GetOpDesc()->GetAllInputNames()) {
      auto input = node->GetOpDesc()->MutableInputDesc(input_name);
      int64_t tensor_size = 0;
      TensorUtils::CalcTensorMemSize(input->GetShape(), input->GetFormat(), input->GetDataType(), tensor_size);
      TensorUtils::SetSize(*input, tensor_size);
    }
    auto out_size = node->GetOpDesc()->GetAllOutputsDescSize();
    for (int32_t id = 0; id < static_cast<int32_t>(out_size); id++) {
      auto output = node->GetOpDesc()->MutableOutputDesc(id);
      int64_t tensor_size = 0;
      TensorUtils::CalcTensorMemSize(output->GetShape(), output->GetFormat(), output->GetDataType(), tensor_size);
      TensorUtils::SetSize(*output, tensor_size);
    }
  }
  root_graph->TopologicalSorting();
  return root_graph;
}
//          a
//          |
//          b   c
//           \ /
//           pc1
//            |
//            d
//            |
//         netoutput
//
ComputeGraphPtr BuildPhonyConcatWithOutputOffsetList() {
  auto builder = block_mem_ut::GraphBuilder("root_graph");
  const auto &a = builder.AddNode("a", ADD, 1, 1, 1, {60});
  const auto &b = builder.AddNode("b", ADD, 1, 1, 1, {60});
  const auto &c = builder.AddNode("c", ADD, 1, 1, 1, {60});
  const auto &d = builder.AddNode("d", ADD, 1, 1, 1, {120});
  const auto &pc1 = builder.AddNode("pc1", "PhonyConcat", 2, 1, 1, {120});
  const auto &netoutput = builder.AddNode("partitioncall_0_netoutput", NETOUTPUT, 1, 0, 1, {120});

  int64_t mem_size = 0;
  MemReuseUtils::GetTensorSize(*b->GetOpDescBarePtr()->MutableOutputDesc(0), mem_size);
  mem_size = (mem_size + 32) / 32 * 32 + 32;

  TensorUtils::SetSize(*a->GetOpDescBarePtr()->MutableOutputDesc(0), mem_size);
  TensorUtils::SetSize(*b->GetOpDescBarePtr()->MutableOutputDesc(0), mem_size);
  TensorUtils::SetSize(*c->GetOpDescBarePtr()->MutableOutputDesc(0), mem_size);

  builder.AddDataEdge(a, 0, b, 0);
  builder.AddDataEdge(b, 0, pc1, 0);
  builder.AddDataEdge(c, 0, pc1, 1);
  builder.AddDataEdge(pc1, 0, d, 0);
  builder.AddDataEdge(d, 0, netoutput, 0);

  (void)AttrUtils::SetBool(pc1->GetOpDesc(), ATTR_NAME_NOPADDING_CONTINUOUS_INPUT, true);
  (void)AttrUtils::SetBool(pc1->GetOpDesc(), ATTR_NAME_OUTPUT_REUSE_INPUT, true);
  (void)AttrUtils::SetBool(pc1->GetOpDesc(), ATTR_NAME_NOTASK, true);
  (void)ge::AttrUtils::SetInt(pc1->GetOpDesc(), ATTR_NAME_REUSE_INPUT_ON_DIM_INDEX, 0);

  std::vector<int64_t> offset_list = {0, 240};
  (void)AttrUtils::SetListInt(b->GetOpDescBarePtr(), ATTR_NAME_OUTPUT_OFFSET_LIST_FOR_CONTINUOUS, offset_list);
  (void)AttrUtils::SetListInt(c->GetOpDescBarePtr(), ATTR_NAME_OUTPUT_OFFSET_LIST_FOR_CONTINUOUS, offset_list);

  const auto &root_graph = builder.GetGraph();

  root_graph->TopologicalSorting();
  return root_graph;
}

/*
          a
          |
          ps
         / \
        b   c
        \   /
       netoutput
*/
ComputeGraphPtr BuildPhonySplitWithOutputOffsetList() {
  auto builder = block_mem_ut::GraphBuilder("root_graph");
  const auto &a = builder.AddNode("a", ADD, 0, 1, 1, {120});
  const auto &b = builder.AddNode("b", ADD, 1, 1, 1, {60});
  const auto &c = builder.AddNode("c", ADD, 1, 1, 1, {60});
  const auto &ps = builder.AddNode("ps", "PhonySplit", 1, 2, 1, {60});
  const auto &netoutput = builder.AddNode("partitioncall_0_netoutput", NETOUTPUT, 2, 0, 1, {60});

  int64_t mem_size = 0;
  MemReuseUtils::GetTensorSize(*b->GetOpDescBarePtr()->MutableOutputDesc(0), mem_size);
  mem_size = (mem_size + 32) / 32 * 32 + 32;

  TensorUtils::SetSize(*a->GetOpDescBarePtr()->MutableOutputDesc(0), mem_size);
  TensorUtils::SetSize(*b->GetOpDescBarePtr()->MutableOutputDesc(0), mem_size);
  TensorUtils::SetSize(*c->GetOpDescBarePtr()->MutableOutputDesc(0), mem_size);

  builder.AddDataEdge(a, 0, ps, 0);
  builder.AddDataEdge(ps, 0, b, 0);
  builder.AddDataEdge(ps, 1, c, 0);
  builder.AddDataEdge(b, 0, netoutput, 0);
  builder.AddDataEdge(c, 0, netoutput, 1);

  (void)AttrUtils::SetBool(ps->GetOpDesc(), ATTR_NAME_NOPADDING_CONTINUOUS_OUTPUT, true);
  (void)AttrUtils::SetBool(ps->GetOpDesc(), ATTR_NAME_OUTPUT_REUSE_INPUT, true);
  (void)AttrUtils::SetBool(ps->GetOpDesc(), ATTR_NAME_NOTASK, true);
  (void)ge::AttrUtils::SetInt(ps->GetOpDesc(), ATTR_NAME_REUSE_INPUT_ON_DIM_INDEX, 0);

  std::vector<int64_t> offset_list = {0, 240};
  (void)AttrUtils::SetListInt(b->GetOpDescBarePtr(), ATTR_NAME_INPUT_OFFSET_LIST_FOR_CONTINUOUS, offset_list);
  (void)AttrUtils::SetListInt(c->GetOpDescBarePtr(), ATTR_NAME_INPUT_OFFSET_LIST_FOR_CONTINUOUS, offset_list);

  const auto &root_graph = builder.GetGraph();

  root_graph->TopologicalSorting();
  return root_graph;
}

/*
          a
          |
          ps
         / \
        b
        \
       netoutput
*/
ComputeGraphPtr BuildPhonySplitSuspendOutputWithOutputOffsetList() {
  auto builder = block_mem_ut::GraphBuilder("root_graph");
  const auto &a = builder.AddNode("a", ADD, 0, 1, 1, {120});
  const auto &b = builder.AddNode("b", ADD, 1, 1, 1, {60});
  const auto &ps = builder.AddNode("ps", "PhonySplit", 1, 2, 1, {60});
  const auto &netoutput = builder.AddNode("partitioncall_0_netoutput", NETOUTPUT, 2, 0, 1, {60});

  ps->GetOpDescBarePtr()->MutableInputDesc(0)->SetShape(GeShape(std::vector<int64_t>({120})));
  int64_t tensor_size = 0;
  TensorUtils::GetSize(*a->GetOpDescBarePtr()->MutableOutputDesc(0), tensor_size);
  TensorUtils::SetSize(*ps->GetOpDescBarePtr()->MutableInputDesc(0), tensor_size);

  int64_t mem_size = 0;
  MemReuseUtils::GetTensorSize(*b->GetOpDescBarePtr()->MutableOutputDesc(0), mem_size);
  mem_size = (mem_size + 32) / 32 * 32 + 32;
  TensorUtils::SetSize(*b->GetOpDescBarePtr()->MutableOutputDesc(0), mem_size);

  builder.AddDataEdge(a, 0, ps, 0);
  builder.AddDataEdge(ps, 0, b, 0);
  builder.AddDataEdge(b, 0, netoutput, 0);

  (void)AttrUtils::SetBool(ps->GetOpDesc(), ATTR_NAME_NOPADDING_CONTINUOUS_OUTPUT, true);
  (void)AttrUtils::SetBool(ps->GetOpDesc(), ATTR_NAME_OUTPUT_REUSE_INPUT, true);
  (void)AttrUtils::SetBool(ps->GetOpDesc(), ATTR_NAME_NOTASK, true);
  (void)ge::AttrUtils::SetInt(ps->GetOpDesc(), ATTR_NAME_REUSE_INPUT_ON_DIM_INDEX, 0);

  std::vector<int64_t> offset_list = {0, 240};
  (void)AttrUtils::SetListInt(b->GetOpDescBarePtr(), ATTR_NAME_INPUT_OFFSET_LIST_FOR_CONTINUOUS, offset_list);
  const auto &root_graph = builder.GetGraph();

  root_graph->TopologicalSorting();
  return root_graph;
}

/*
 *             f
 *             |
 *            ps1 [1,5,48,448]
 *          /  |  \
 *        ps2  e  ps3 [1,2,48,448]
 *        / \     /  \
 *       a   b   c    d  abcde output shape: [1,1,48,448]
 */
ComputeGraphPtr BuildPhonySplitCascated() {
  auto builder = block_mem_ut::GraphBuilder("root_graph");
  const auto &f = builder.AddNode("f", ADD, 0, 1, 1, {1, 5, 48, 448});
  const auto &ps1 = builder.AddNode("ps1", "PhonySplit", 1, 3, 1, {1, 5, 48, 448});

  const auto &ps2 = builder.AddNode("ps2", "PhonySplit", 1, 2, 1, {1, 2, 48, 448});
  const auto &e = builder.AddNode("e", ADD, 1, 1, 1, {1, 1, 48, 448});
  const auto &ps3 = builder.AddNode("ps3", "PhonySplit", 1, 2, 1, {1, 2, 48, 448});

  const auto &a = builder.AddNode("a", ADD, 1, 1, 1, {1, 1, 48, 448});
  const auto &b = builder.AddNode("b", ADD, 1, 1, 1, {1, 1, 48, 448});
  const auto &c = builder.AddNode("c", ADD, 1, 1, 1, {1, 1, 48, 448});
  const auto &d = builder.AddNode("d", ADD, 1, 1, 1, {1, 1, 48, 448});
  const auto &netoutput = builder.AddNode("partitioncall_0_netoutput", NETOUTPUT, 5, 0, 1, {1, 1, 48, 448});

  ps1->GetOpDescBarePtr()->UpdateOutputDesc(0, ps2->GetOpDescBarePtr()->GetInputDesc(0));
  ps1->GetOpDescBarePtr()->UpdateOutputDesc(1, e->GetOpDescBarePtr()->GetInputDesc(0));
  ps1->GetOpDescBarePtr()->UpdateOutputDesc(2, ps3->GetOpDescBarePtr()->GetInputDesc(0));

  ps2->GetOpDescBarePtr()->UpdateOutputDesc(0, a->GetOpDescBarePtr()->GetInputDesc(0));
  ps2->GetOpDescBarePtr()->UpdateOutputDesc(1, b->GetOpDescBarePtr()->GetInputDesc(0));

  ps3->GetOpDescBarePtr()->UpdateOutputDesc(0, c->GetOpDescBarePtr()->GetInputDesc(0));
  ps3->GetOpDescBarePtr()->UpdateOutputDesc(1, d->GetOpDescBarePtr()->GetInputDesc(0));

  builder.AddDataEdge(f, 0, ps1, 0);

  builder.AddDataEdge(ps1, 0, ps2, 0);
  builder.AddDataEdge(ps1, 1, e, 0);
  builder.AddDataEdge(ps1, 2, ps3, 0);

  builder.AddDataEdge(ps2, 0, a, 0);
  builder.AddDataEdge(ps2, 1, b, 0);
  builder.AddDataEdge(ps3, 0, c, 0);
  builder.AddDataEdge(ps3, 1, d, 0);

  builder.AddDataEdge(a, 0, netoutput, 0);
  builder.AddDataEdge(b, 0, netoutput, 1);
  builder.AddDataEdge(e, 0, netoutput, 2);
  builder.AddDataEdge(c, 0, netoutput, 3);
  builder.AddDataEdge(d, 0, netoutput, 4);

  (void)AttrUtils::SetBool(ps3->GetOpDesc(), ATTR_NAME_NOPADDING_CONTINUOUS_OUTPUT, true);
  (void)AttrUtils::SetBool(ps3->GetOpDesc(), ATTR_NAME_OUTPUT_REUSE_INPUT, true);
  (void)AttrUtils::SetBool(ps3->GetOpDesc(), ATTR_NAME_NOTASK, true);
  (void)ge::AttrUtils::SetInt(ps3->GetOpDesc(), ATTR_NAME_REUSE_INPUT_ON_DIM_INDEX, 0);

  (void)AttrUtils::SetBool(ps2->GetOpDesc(), ATTR_NAME_NOPADDING_CONTINUOUS_OUTPUT, true);
  (void)AttrUtils::SetBool(ps2->GetOpDesc(), ATTR_NAME_OUTPUT_REUSE_INPUT, true);
  (void)AttrUtils::SetBool(ps2->GetOpDesc(), ATTR_NAME_NOTASK, true);
  (void)ge::AttrUtils::SetInt(ps2->GetOpDesc(), ATTR_NAME_REUSE_INPUT_ON_DIM_INDEX, 0);

  (void)AttrUtils::SetBool(ps1->GetOpDesc(), ATTR_NAME_NOPADDING_CONTINUOUS_OUTPUT, true);
  (void)AttrUtils::SetBool(ps1->GetOpDesc(), ATTR_NAME_OUTPUT_REUSE_INPUT, true);
  (void)AttrUtils::SetBool(ps1->GetOpDesc(), ATTR_NAME_NOTASK, true);
  (void)ge::AttrUtils::SetInt(ps1->GetOpDesc(), ATTR_NAME_REUSE_INPUT_ON_DIM_INDEX, 0);

  const auto &root_graph = builder.GetGraph();

  root_graph->TopologicalSorting();
  return root_graph;
}

//          a
//          |
//          b   c
//           \ /
//           pc1
//            |
//            d
//            |
//         netoutput
//
ComputeGraphPtr BuildPhonyConcatSizeLessThanSumOfAllInputs() {
  auto builder = block_mem_ut::GraphBuilder("root_graph");
  const auto &a = builder.AddNode("a", ADD, 1, 1, 1, {60});
  const auto &b = builder.AddNode("b", ADD, 1, 1, 1, {60});
  const auto &c = builder.AddNode("c", ADD, 1, 1, 1, {60});
  const auto &d = builder.AddNode("d", ADD, 1, 1, 1, {120});
  const auto &pc1 = builder.AddNode("pc1", "PhonyConcat", 2, 1, 1, {120});
  const auto &netoutput = builder.AddNode("partitioncall_0_netoutput", NETOUTPUT, 1, 0, 1, {120});

  int64_t mem_size = 0;
  MemReuseUtils::GetTensorSize(*b->GetOpDescBarePtr()->MutableOutputDesc(0), mem_size);
  mem_size = (mem_size + 32) / 32 * 32 + 32;

  TensorUtils::SetSize(*a->GetOpDescBarePtr()->MutableOutputDesc(0), mem_size);
  TensorUtils::SetSize(*b->GetOpDescBarePtr()->MutableOutputDesc(0), mem_size);
  TensorUtils::SetSize(*c->GetOpDescBarePtr()->MutableOutputDesc(0), mem_size);

  builder.AddDataEdge(a, 0, b, 0);
  builder.AddDataEdge(b, 0, pc1, 0);
  builder.AddDataEdge(c, 0, pc1, 1);
  builder.AddDataEdge(pc1, 0, d, 0);
  builder.AddDataEdge(d, 0, netoutput, 0);

  (void)AttrUtils::SetBool(pc1->GetOpDesc(), ATTR_NAME_NOPADDING_CONTINUOUS_INPUT, true);
  (void)AttrUtils::SetBool(pc1->GetOpDesc(), ATTR_NAME_OUTPUT_REUSE_INPUT, true);
  (void)AttrUtils::SetBool(pc1->GetOpDesc(), ATTR_NAME_NOTASK, true);
  (void)ge::AttrUtils::SetInt(pc1->GetOpDesc(), ATTR_NAME_REUSE_INPUT_ON_DIM_INDEX, 0);
  const auto &root_graph = builder.GetGraph();

  root_graph->TopologicalSorting();
  return root_graph;
}

//    a   b
//     \ /
//     hcom
//      |
//  netoutput
//
ComputeGraphPtr BuildHcomWithP2pMemoryType() {
  auto builder = block_mem_ut::GraphBuilder("root_graph");
  const auto &a = builder.AddNode("a", ADD, 0, 1, 1, {1, 1, 48, 448});
  const auto &b = builder.AddNode("b", ADD, 0, 1, 1, {1, 1, 448, 448});
  const auto &hcom = builder.AddNode("hcom", "hcom", 2, 1, 1, {1, 1, 448, 448});
  const auto &netoutput = builder.AddNode("partitioncall_0_netoutput", NETOUTPUT, 1, 0, 1, {1, 1, 448, 448});

  // workspace memory size
  builder.AddDataEdge(a, 0, hcom, 0);
  builder.AddDataEdge(b, 0, hcom, 1);
  builder.AddDataEdge(hcom, 0, netoutput, 0);

  std::vector<int64_t> inputs_memory_types = {RT_MEMORY_P2P_DDR, RT_MEMORY_P2P_DDR};
  ge::AttrUtils::SetListInt(hcom->GetOpDescBarePtr(), ATTR_NAME_INPUT_MEM_TYPE_LIST, inputs_memory_types);

  const auto &root_graph = builder.GetGraph();
  root_graph->TopologicalSorting();
  return root_graph;
}

//    a   b
//     \ /
//     hcom
//      |
//  netoutput
//
ComputeGraphPtr BuildHcomWithBufferPoolId() {
  auto builder = block_mem_ut::GraphBuilder("root_graph");
  const auto &a = builder.AddNode("a", ADD, 0, 1, 1, {1, 1, 48, 448});
  const auto &b = builder.AddNode("b", ADD, 0, 1, 1, {1, 1, 448, 448});
  const auto &hcom = builder.AddNode("hcom", "hcom", 2, 1, 1, {1, 1, 448, 448});
  const auto &netoutput = builder.AddNode("partitioncall_0_netoutput", NETOUTPUT, 1, 0, 1, {1, 1, 448, 448});

  builder.AddDataEdge(a, 0, hcom, 0);
  builder.AddDataEdge(b, 0, hcom, 1);
  builder.AddDataEdge(hcom, 0, netoutput, 0);

  ge::AttrUtils::SetBool(hcom->GetOpDescBarePtr(), ATTR_NAME_CONTINUOUS_INPUT, true);
  ge::AttrUtils::SetInt(hcom->GetOpDescBarePtr(), ATTR_NAME_BUFFER_POOL_ID, 1);
  ge::AttrUtils::SetInt(hcom->GetOpDescBarePtr(), ATTR_NAME_BUFFER_POOL_SIZE, 804352);
  ge::AttrUtils::SetListInt(hcom->GetOpDescBarePtr(), ATTR_NAME_BUFFER_POOL_NODE_SIZE_AND_OFFSET, {804352, 0});
  const auto &root_graph = builder.GetGraph();
  root_graph->TopologicalSorting();
  return root_graph;
}
/*
 *   a  b
 *    \ /
 *    hcom
 *     |
 *    netoutput
 */
ComputeGraphPtr BuildHcomWithContinuousOutputGraph() {
  auto builder = block_mem_ut::GraphBuilder("root_graph");
  const auto &a = builder.AddNode("a", ADD, 1, 1, 1, {1, 1, 48, 448});
  const auto &b = builder.AddNode("b", ADD, 1, 1, 1, {1, 1, 448, 448});
  const auto &hcom = builder.AddNode("hcom", "hcom", 1, 2, 1, {1, 1, 448, 448});
  const auto &netoutput = builder.AddNode("partitioncall_0_netoutput", NETOUTPUT, 2, 0, 1, {1, 1, 448, 448});

  builder.AddDataEdge(a, 0, netoutput, 0);
  builder.AddDataEdge(b, 0, netoutput, 1);
  builder.AddDataEdge(hcom, 0, a, 0);
  builder.AddDataEdge(hcom, 1, b, 0);

  ge::AttrUtils::SetBool(hcom->GetOpDescBarePtr(), ATTR_NAME_CONTINUOUS_OUTPUT, true);
  const auto &root_graph = builder.GetGraph();
  root_graph->TopologicalSorting();
  return root_graph;
}

//    a   b
//     \ /
//      c
//      |
//  netoutput
//
ComputeGraphPtr BuildGraphWithOutputNotAssign() {
  auto builder = block_mem_ut::GraphBuilder("root_graph");
  const auto &a = builder.AddNode("a", ADD, 0, 1, 1, {1, 1, 48, 448});
  const auto &b = builder.AddNode("b", ADD, 0, 1, 1, {1, 1, 448, 448});
  const auto &c = builder.AddNode("c", "ADD", 2, 2, 1, {1, 1, 448, 448});
  const auto &netoutput = builder.AddNode("partitioncall_0_netoutput", NETOUTPUT, 1, 0, 1, {1, 1, 448, 448});

  // workspace memory size
  builder.AddDataEdge(a, 0, c, 0);
  builder.AddDataEdge(b, 0, c, 1);
  builder.AddDataEdge(c, 0, netoutput, 0);

  ge::AttrUtils::SetInt(c->GetOpDescBarePtr()->MutableOutputDesc(1), ATTR_NAME_TENSOR_MEMORY_SCOPE, 2);

  const auto &root_graph = builder.GetGraph();
  root_graph->TopologicalSorting();
  return root_graph;
}

/*
 *     data
 *      |
 *      a
 *      |
 *  partitioned_call    +----------------------+
 *      |               | inner_data   memset  |
 *      |               |     |       /(ctl)   |
 *      |               | atomic_node          |
 *      |               |     |                |
 *      |               |  reshape             |
 *      |               |     |                |
 *      b               | netoutput2           |
 *      |               +----------------------+
 *    netoutput1
 */
ComputeGraphPtr BuildPartitionedCallWithAtomicNode() {
  const auto inner_data = OP_CFG(DATA).ParentNodeIndex(0);
  DEF_GRAPH(sub_1) {
    CHAIN(NODE("inner_data", inner_data)->NODE("atomic_node", CAST)->NODE("reshape", RESHAPE)->NODE("netoutput2", NETOUTPUT));
    CHAIN(NODE("memset", MEMSET)->Ctrl()->NODE("atomic_node", CAST));
  };
  DEF_GRAPH(g1) {
    CHAIN(NODE("data", DATA)->NODE("a", CAST)->NODE("partitioned_call", PARTITIONEDCALL, sub_1)->NODE("b", CAST)->NODE("netoutput1", NETOUTPUT));
  };

  auto graph = ToComputeGraph(g1);
  auto parent_node = graph->FindNode("partitioned_call");
  auto partitioned_call1_graph = ToComputeGraph(sub_1);
  partitioned_call1_graph->SetParentNode(parent_node);
  graph->AddSubGraph(partitioned_call1_graph);

  auto atomic_node = partitioned_call1_graph->FindNode("atomic_node");
  AttrUtils::SetBool(atomic_node->GetOpDesc(), ATOMIC_ATTR_IS_ATOMIC_NODE, true);
  AttrUtils::SetListInt(atomic_node->GetOpDesc(), ATOMIC_ATTR_OUTPUT_INDEX, {0});

  for (auto &node : graph->GetAllNodes()) {
    if (node->GetType() == NETOUTPUT) {
      if (node->GetOwnerComputeGraph()->GetParentNode() == nullptr) {
        continue;
      }
      auto op_desc = node->GetOpDesc();
      for (uint32_t index = 0U; index < op_desc->GetInputsSize(); ++index) {
        AttrUtils::SetInt(op_desc->MutableInputDesc(index), ATTR_NAME_PARENT_NODE_INDEX, index);
      }
    }
  }
  for (auto &node : graph->GetAllNodes()) {
    for (size_t i = 0U; i < node->GetOutDataNodesSize(); ++i) {
      auto out_tensor = node->GetOpDescBarePtr()->MutableOutputDesc(i);
      int64_t tensor_size = 0;
      TensorUtils::CalcTensorMemSize(out_tensor->GetShape(), out_tensor->GetFormat(), out_tensor->GetDataType(), tensor_size);
      TensorUtils::SetSize(*out_tensor, tensor_size);
    }

    for (size_t i = 0U; i < node->GetInDataNodesSize(); ++i) {
      auto in_tensor = node->GetOpDescBarePtr()->MutableInputDesc(i);
      int64_t tensor_size = 0;
      TensorUtils::CalcTensorMemSize(in_tensor->GetShape(), in_tensor->GetFormat(), in_tensor->GetDataType(), tensor_size);
      TensorUtils::SetSize(*in_tensor, tensor_size);
    }
  }
  graph->TopologicalSorting();
  return graph;
}

/*
 *  data1  data2
 *     \    /
 *       if
 *       |
 *       op
 *       |
 *    netoutput0
 *
 * then_subgraph                             else_subgraph
 * +-------------------------------------+   +------------------------------------+
 * |    data3                            |   |   data5                             |
 * |     |                               |   |    |                               |
 * |    cast                             |   |   cast2                            |
 * |     |                               |   |     |                              |
 * | partitioned_call1   +------------+  |   | partitioned_call2  +------------+  |
 * |     |               |   data4    |  |   |     |              |    data6   |  |
 * |  netoutput1         |     |      |  |   |  netoutput3        |      |     |  |
 * |                     | netoutput2 |  |   |                    | netoutput4 |  |
 * |                     +------------+  |   |                    +------------+  |
 * +-------------------------------------+   +------------------------------------+
 *
 */
ComputeGraphPtr BuildNestingWrapperWithSubgraphNodeDiffStream() {
  const auto data4 = OP_CFG(DATA).ParentNodeIndex(0);
  const auto data3 = OP_CFG(DATA).ParentNodeIndex(0);
  DEF_GRAPH(sub_1) {
    CHAIN(NODE("data4", data4)->NODE("netoutput2", NETOUTPUT));
  };
  DEF_GRAPH(then_) {
    CHAIN(NODE("data3", data3)->NODE("cast", CAST)->NODE("partitioned_call1", PARTITIONEDCALL, sub_1)->NODE("netoutput1", NETOUTPUT));
  };

  const auto data6 = OP_CFG(DATA).ParentNodeIndex(0);
  const auto data5 = OP_CFG(DATA).ParentNodeIndex(1);
  DEF_GRAPH(sub_2) {
    CHAIN(NODE("data6", data6)->NODE("netoutput4", NETOUTPUT));
  };
  DEF_GRAPH(else_) {
    CHAIN(NODE("data5", data5)->NODE("cast2", CAST)->NODE("partitioned_call2", PARTITIONEDCALL, sub_2)->NODE("netoutput3", NETOUTPUT));
  };

  DEF_GRAPH(g1) {
    CHAIN(NODE("data1", DATA)->EDGE(0, 0)->NODE("if", IF, then_, else_)
              ->EDGE(0, 0)->NODE("op", CAST)->EDGE(0, 0)->NODE("netoutput0", NETOUTPUT));
    CHAIN(NODE("data2", DATA)->EDGE(0, 1)->NODE("if"));
  };
  auto graph = ToComputeGraph(g1);

  auto partitioned_call1_graph = ToComputeGraph(sub_1);
  auto partitioned_call2_graph = ToComputeGraph(sub_2);
  then_.Layout();
  else_.Layout();
  auto then_subgraph = graph->GetSubgraph("then_");
  auto else_subgraph = graph->GetSubgraph("else_");
  auto partitioned_call1_node = then_subgraph->FindNode("partitioned_call1");
  auto partitioned_call2_node = else_subgraph->FindNode("partitioned_call2");
  then_subgraph->SetParentGraph(graph);
  else_subgraph->SetParentGraph(graph);
  partitioned_call1_graph->SetParentNode(partitioned_call1_node);
  partitioned_call2_graph->SetParentNode(partitioned_call2_node);
  graph->AddSubGraph(partitioned_call1_graph);
  graph->AddSubGraph(partitioned_call2_graph);

  auto data4_node = partitioned_call1_graph->FindNode("data4");
  data4_node->GetOpDescBarePtr()->SetStreamId(4);
  auto data6_node = partitioned_call2_graph->FindNode("data6");
  data6_node->GetOpDescBarePtr()->SetStreamId(6);
  auto p1 = then_subgraph->FindNode("partitioned_call1");
  p1->GetOpDescBarePtr()->SetStreamId(1);
  auto p2 = else_subgraph->FindNode("partitioned_call2");
  p2->GetOpDescBarePtr()->SetStreamId(2);

  for (auto &node : graph->GetAllNodes()) {
    if (node->GetType() == NETOUTPUT) {
      if (node->GetOwnerComputeGraph()->GetParentNode() == nullptr) {
        continue;
      }
      auto op_desc = node->GetOpDesc();
      for (uint32_t index = 0U; index < op_desc->GetInputsSize(); ++index) {
        AttrUtils::SetInt(op_desc->MutableInputDesc(index), ATTR_NAME_PARENT_NODE_INDEX, index);
      }
    }
  }
  for (auto &node : graph->GetAllNodes()) {
    for (size_t i = 0U; i < node->GetOutDataNodesSize(); ++i) {
      auto out_tensor = node->GetOpDescBarePtr()->MutableOutputDesc(i);
      int64_t tensor_size = 0;
      TensorUtils::CalcTensorMemSize(out_tensor->GetShape(), out_tensor->GetFormat(), out_tensor->GetDataType(), tensor_size);
      TensorUtils::SetSize(*out_tensor, tensor_size);
    }

    for (size_t i = 0U; i < node->GetInDataNodesSize(); ++i) {
      auto in_tensor = node->GetOpDescBarePtr()->MutableInputDesc(i);
      int64_t tensor_size = 0;
      TensorUtils::CalcTensorMemSize(in_tensor->GetShape(), in_tensor->GetFormat(), in_tensor->GetDataType(), tensor_size);
      TensorUtils::SetSize(*in_tensor, tensor_size);
    }
  }
  graph->TopologicalSorting();
  return graph;
}

/*
 *   data_1 data_2
 *       \  /
 *      hcom_1 (ATTR_NAME_CONTINUOUS_OUTPUT)
 *       | |
 *      hcom_2 (ATTR_NAME_CONTINUOUS_INPUT)
 *       | |
 *       a b
 *       | |
 *       c d
 */
ComputeGraphPtr BuildContinuousOutInWithTwoOutIn() {
  vector<std::string> engine_list = {"AIcoreEngine"};
  std::vector<int64_t> memtype_list = {RT_MEMORY_HBM, RT_MEMORY_HBM};
  auto hcom1 = OP_CFG(HCOMALLREDUCE)
                   .InCnt(2)
                   .OutCnt(2)
                   .TensorDesc(FORMAT_NCHW, DT_FLOAT, {1, 1, 224, 224})
                   .Attr(ATTR_NAME_INPUT_MEM_TYPE_LIST, memtype_list)
                   .Attr(ATTR_NAME_OUTPUT_MEM_TYPE_LIST, memtype_list)
                   .Attr(ATTR_NAME_CONTINUOUS_OUTPUT, true);
  auto hcom2 = OP_CFG(HCOMALLREDUCE)
                   .InCnt(2)
                   .OutCnt(2)
                   .TensorDesc(FORMAT_NCHW, DT_FLOAT, {1, 1, 224, 224})
                   .Attr(ATTR_NAME_INPUT_MEM_TYPE_LIST, memtype_list)
                   .Attr(ATTR_NAME_OUTPUT_MEM_TYPE_LIST, memtype_list)
                   .Attr(ATTR_NAME_CONTINUOUS_INPUT, true);

  auto data1 = OP_CFG(DATA);
  auto data2 = OP_CFG(DATA);
  auto type = OP_CFG("type");

  DEF_GRAPH(g1) {
    CHAIN(NODE("data_1", data1)->EDGE(0, 0)->NODE("hcom_1", hcom1));
    CHAIN(NODE("data_2", data2)->EDGE(0, 1)->NODE("hcom_1", hcom1));
    CHAIN(NODE("hcom_1", hcom1)->EDGE(0, 0)->NODE("hcom_2", hcom2));
    CHAIN(NODE("hcom_1", hcom1)->EDGE(1, 1)->NODE("hcom_2", hcom2));
    CHAIN(NODE("hcom_2", hcom2)->EDGE(0, 0)->NODE("a", type));
    CHAIN(NODE("hcom_2", hcom2)->EDGE(1, 0)->NODE("b", type));
    CHAIN(NODE("a", type)->EDGE(0, 0)->NODE("c", type));
    CHAIN(NODE("b", type)->EDGE(0, 0)->NODE("d", type));
  };

  auto compute_graph = ToComputeGraph(g1);
  compute_graph->TopologicalSorting();
  MemConflictShareGraph::SetSizeForAllNodes(compute_graph);
  return compute_graph;
}

/*
 *     hcom1   b
 *    /  | |  /
 *   a   hcom2
 *    \ /
 *     c
 *     |
 *     d
 *     |
 *   netoutput
 */
ComputeGraphPtr BuildContinuousOutInWithOutAsHeader() {
  auto builder = block_mem_ut::GraphBuilder("root_graph");
  const auto &hcom1 = builder.AddNode("hcom1", HCOMALLREDUCE, 0, 3, 1, {1, 1, 48, 448});
  const auto &a = builder.AddNode("a", ADD, 1, 1, 1, {1, 1, 448, 448});
  const auto &b = builder.AddNode("b", ADD, 0, 1, 1, {1, 1, 448, 448});
  const auto &c = builder.AddNode("c", ADD, 2, 1, 1, {1, 1, 448, 448});
  const auto &d = builder.AddNode("d", ADD, 1, 1, 1, {1, 1, 448, 448});
  const auto &hcom2 = builder.AddNode("hcom2", "HCOMALLREDUCE", 3, 1, 1, {1, 1, 448, 448});
  const auto &netoutput = builder.AddNode("netoutput", NETOUTPUT, 1, 0, 1, {1, 1, 448, 448});

  // workspace memory size
  builder.AddDataEdge(hcom1, 0, a, 0);
  builder.AddDataEdge(hcom1, 1, hcom2, 0);
  builder.AddDataEdge(hcom1, 2, hcom2, 1);
  builder.AddDataEdge(b, 0, hcom2, 2);
  builder.AddDataEdge(hcom2, 0, c, 1);
  builder.AddDataEdge(a, 0, c, 0);
  builder.AddDataEdge(c, 0, d, 0);
  builder.AddDataEdge(d, 0, netoutput, 0);

  (void)AttrUtils::SetBool(hcom1->GetOpDesc(), ATTR_NAME_CONTINUOUS_OUTPUT, true);
  (void)AttrUtils::SetBool(hcom2->GetOpDesc(), ATTR_NAME_CONTINUOUS_INPUT, true);
  const auto &root_graph = builder.GetGraph();
  root_graph->TopologicalSorting();
  return root_graph;
}
/*
 *     hcom1
 *    / | |  \
 *   a  hcom2  b
 *   |         ^
 *   c---------|(ctrl edge)
 *   |
 *   d
 *   |
 *   netoutput
 */
ComputeGraphPtr BuildContinuousOutInWithOutAsHeaderAndTail() {
  auto builder = block_mem_ut::GraphBuilder("root_graph");
  const auto &hcom1 = builder.AddNode("hcom1", HCOMALLREDUCE, 0, 4, 1, {1, 1, 48, 448});
  const auto &a = builder.AddNode("a", ADD, 1, 1, 1, {1, 1, 448, 448});
  const auto &b = builder.AddNode("b", ADD, 1, 0, 1, {1, 1, 448, 448});
  const auto &c = builder.AddNode("c", ADD, 1, 1, 1, {1, 1, 448, 448});
  const auto &d = builder.AddNode("d", ADD, 1, 1, 1, {1, 1, 448, 448});
  const auto &hcom2 = builder.AddNode("hcom2", "HCOMALLREDUCE", 2, 0, 1, {1, 1, 448, 448});
  const auto &netoutput = builder.AddNode("netoutput", NETOUTPUT, 1, 0, 1, {1, 1, 448, 448});

  // workspace memory size
  builder.AddDataEdge(hcom1, 0, a, 0);
  builder.AddDataEdge(hcom1, 1, hcom2, 0);
  builder.AddDataEdge(hcom1, 2, hcom2, 1);
  builder.AddDataEdge(hcom1, 3, b, 0);
  builder.AddDataEdge(a, 0, c, 0);
  builder.AddDataEdge(c, 0, d, 0);
  builder.AddDataEdge(d, 0, netoutput, 0);
  builder.AddControlEdge(c, b);

  (void)AttrUtils::SetBool(hcom1->GetOpDesc(), ATTR_NAME_CONTINUOUS_OUTPUT, true);
  (void)AttrUtils::SetBool(hcom2->GetOpDesc(), ATTR_NAME_CONTINUOUS_INPUT, true);
  const auto &root_graph = builder.GetGraph();
  root_graph->TopologicalSorting();
  return root_graph;
}

/*
 *  a  hcom1
 *   \  | | \
 *     hcom2 b
 *       |   ^
 *       c---| (ctrl edge)
 *       |
 *       d
 *       |
 *    netoutput
 */
ComputeGraphPtr BuildContinuousOutInWithInAsHeader() {
  auto builder = block_mem_ut::GraphBuilder("root_graph");
  const auto &hcom1 = builder.AddNode("hcom1", HCOMALLREDUCE, 0, 3, 1, {1, 1, 48, 448});
  const auto &a = builder.AddNode("a", ADD, 0, 1, 1, {1, 1, 448, 448});
  const auto &b = builder.AddNode("b", ADD, 1, 0, 1, {1, 1, 448, 448});
  const auto &c = builder.AddNode("c", ADD, 1, 1, 1, {1, 1, 448, 448});
  const auto &d = builder.AddNode("d", ADD, 1, 1, 1, {1, 1, 448, 448});
  const auto &hcom2 = builder.AddNode("hcom2", "HCOMALLREDUCE", 3, 1, 1, {1, 1, 448, 448});
  const auto &netoutput = builder.AddNode("netoutput", NETOUTPUT, 1, 0, 1, {1, 1, 448, 448});

  // workspace memory size
  builder.AddDataEdge(a, 0, hcom2, 0);
  builder.AddDataEdge(hcom1, 0, hcom2, 1);
  builder.AddDataEdge(hcom1, 1, hcom2, 2);
  builder.AddDataEdge(hcom1, 2, b, 0);
  builder.AddDataEdge(hcom2, 0, c, 0);
  builder.AddDataEdge(c, 0, d, 0);
  builder.AddDataEdge(d, 0, netoutput, 0);
  builder.AddControlEdge(c, b);

  (void)AttrUtils::SetBool(hcom1->GetOpDesc(), ATTR_NAME_CONTINUOUS_OUTPUT, true);
  (void)AttrUtils::SetBool(hcom2->GetOpDesc(), ATTR_NAME_CONTINUOUS_INPUT, true);
  const auto &root_graph = builder.GetGraph();
  root_graph->TopologicalSorting();
  return root_graph;
}

/*
 *           c
 *           |
 *           d
 *           |
 *  a  hcom1 b
 *   \  | | /
 *     hcom2
 *
 *  topo序： "c", "d", "b", "hcom1", "a", "hcom2"
 */
ComputeGraphPtr BuildContinuousOutInWithInAsHeaderAndTail() {
  auto builder = block_mem_ut::GraphBuilder("root_graph");
  const auto &hcom1 = builder.AddNode("hcom1", HCOMALLREDUCE, 0, 2, 1, {1, 1, 48, 448});
  const auto &c = builder.AddNode("c", ADD, 0, 1, 1, {1, 1, 448, 448});
  const auto &d = builder.AddNode("d", ADD, 1, 1, 1, {1, 1, 448, 448});
  const auto &a = builder.AddNode("a", ADD, 0, 1, 1, {1, 1, 448, 448});
  const auto &b = builder.AddNode("b", ADD, 1, 1, 1, {1, 1, 448, 448});
  const auto &hcom2 = builder.AddNode("hcom2", "HCOMALLREDUCE", 4, 1, 1, {1, 1, 448, 448});
  const auto &netoutput = builder.AddNode("netoutput", NETOUTPUT, 1, 0, 1, {1, 1, 448, 448});

  // workspace memory size
  builder.AddDataEdge(c, 0, d, 0);
  builder.AddDataEdge(d, 0, b, 0);
  builder.AddDataEdge(a, 0, hcom2, 0);
  builder.AddDataEdge(hcom1, 0, hcom2, 1);
  builder.AddDataEdge(hcom1, 1, hcom2, 2);
  builder.AddDataEdge(b, 0, hcom2, 3);
  builder.AddDataEdge(hcom2, 0, netoutput, 0);

  (void)AttrUtils::SetBool(hcom1->GetOpDesc(), ATTR_NAME_CONTINUOUS_OUTPUT, true);
  (void)AttrUtils::SetBool(hcom2->GetOpDesc(), ATTR_NAME_CONTINUOUS_INPUT, true);
  const auto &root_graph = builder.GetGraph();
  root_graph->TopologicalSorting();
  for (const auto &node : root_graph->GetDirectNode()) {
    node->GetOpDesc()->SetStreamId(0);
  }
  a->GetOpDesc()->SetStreamId(1);
  for (const auto &node : root_graph->GetDirectNode()) {
    std::cout << "node: " << node->GetName() << ", id: " << node->GetOpDesc()->GetId() << std::endl;
  }
  std::cout << " after sort mock" << std::endl;
  MemConflictShareGraph::TopologicalSortingMock(root_graph, {"c", "d", "b", "hcom1", "a"});
  for (const auto &node : root_graph->GetDirectNode()) {
    std::cout << "node: " << node->GetName() << ", id: " << node->GetOpDesc()->GetId() << std::endl;
  }
  return root_graph;
}

/*
 *      hcom1
 *   / ||   ||  \
 *  a hcom2 hcom3 b
 *                |
 *                c
 *                |
 *                d
 *  a stream1, others: stream0
 *  topo order: hcom1, a, hcom2, hcom3
 */
ComputeGraphPtr BuildContinuousOutInWithTwoContinuousInNode() {
  auto builder = block_mem_ut::GraphBuilder("root_graph");
  const auto &hcom1 = builder.AddNode("hcom1", HCOMALLREDUCE, 0, 6, 1, {1, 1, 48, 448});
  const auto &a = builder.AddNode("a", ADD, 1, 0, 1, {1, 1, 448, 448});
  const auto &b = builder.AddNode("b", ADD, 1, 1, 1, {1, 1, 448, 448});
  const auto &c = builder.AddNode("c", ADD, 1, 1, 1, {1, 1, 448, 448});
  const auto &d = builder.AddNode("d", ADD, 1, 1, 1, {1, 1, 448, 448});
  const auto &hcom2 = builder.AddNode("hcom2", "HCOMALLREDUCE", 2, 0, 1, {1, 1, 448, 448});
  const auto &hcom3 = builder.AddNode("hcom3", "HCOMALLREDUCE", 2, 0, 1, {1, 1, 448, 448});
  const auto &netoutput = builder.AddNode("netoutput", NETOUTPUT, 1, 0, 1, {1, 1, 448, 448});

  // workspace memory size
  builder.AddDataEdge(hcom1, 0, a, 0);
  builder.AddDataEdge(hcom1, 1, hcom2, 0);
  builder.AddDataEdge(hcom1, 2, hcom2, 1);
  builder.AddDataEdge(hcom1, 3, hcom3, 0);
  builder.AddDataEdge(hcom1, 4, hcom3, 1);
  builder.AddDataEdge(hcom1, 5, b, 0);
  builder.AddDataEdge(b, 0, c, 0);
  builder.AddDataEdge(c, 0, d, 0);
  builder.AddDataEdge(d, 0, netoutput, 0);

  (void)AttrUtils::SetBool(hcom1->GetOpDesc(), ATTR_NAME_CONTINUOUS_OUTPUT, true);
  (void)AttrUtils::SetBool(hcom2->GetOpDesc(), ATTR_NAME_CONTINUOUS_INPUT, true);
  (void)AttrUtils::SetBool(hcom3->GetOpDesc(), ATTR_NAME_CONTINUOUS_INPUT, true);
  const auto &root_graph = builder.GetGraph();
  MemConflictShareGraph::TopologicalSortingMock(root_graph, {"hcom1", "a", "hcom2", "hcom3"});
  for (const auto &node : root_graph->GetAllNodes()) {
    node->GetOpDesc()->SetStreamId(0);
  }
  a->GetOpDesc()->SetStreamId(1);

  return root_graph;
}

/*
 *                     e
 *                     |
 *                     f
 *                     |
 *      hcom1        c d
 *   / ||   ||  \  \ | |
 *  a hcom2 hcom3 b hcom4
 *
 *  topo order: "hcom1", "e", "f", "c", "d
 */
ComputeGraphPtr BuildContinuousOutInWithTwoContinuousInNodeAndRefNode() {
  auto builder = block_mem_ut::GraphBuilder("root_graph");
  const auto &hcom1 = builder.AddNode("hcom1", HCOMALLREDUCE, 0, 7, 1, {1, 1, 448, 448});
  const auto &a = builder.AddNode("a", ADD, 1, 1, 1, {1, 1, 448, 448});
  const auto &b = builder.AddNode("b", ADD, 1, 0, 1, {1, 1, 448, 448});
  const auto &c = builder.AddNode("c", ADD, 0, 1, 1, {1, 1, 448, 448});
  const auto &d = builder.AddNode("d", ADD, 1, 1, 1, {1, 1, 448, 448});
  const auto &e = builder.AddNode("e", ADD, 0, 1, 1, {1, 1, 448, 448});
  const auto &f = builder.AddNode("f", ADD, 1, 1, 1, {1, 1, 448, 448});
  const auto &hcom2 = builder.AddNode("hcom2", "HCOMALLREDUCE", 2, 0, 1, {1, 1, 448, 448});
  const auto &hcom3 = builder.AddNode("hcom3", "HCOMALLREDUCE", 2, 0, 1, {1, 1, 448, 448});
  const auto &hcom4 = builder.AddNode("hcom4", "HCOMALLREDUCE", 3, 0, 1, {1, 1, 448, 448});
  const auto &netoutput = builder.AddNode("netoutput", NETOUTPUT, 1, 0, 1, {1, 1, 448, 448});

  const auto &ref_node1 = builder.AddNode("ref_node1", "RefNode", 1, 1, 1, {1, 1, 448, 448});
  const auto &ref_node2 = builder.AddNode("ref_node2", "RefNode", 1, 1, 1, {1, 1, 448, 448});
  const auto &ref_node3 = builder.AddNode("ref_node3", "RefNode", 1, 1, 1, {1, 1, 448, 448});
  const auto &ref_node4 = builder.AddNode("ref_node4", "RefNode", 1, 1, 1, {1, 1, 448, 448});
  const auto &ref_node5 = builder.AddNode("ref_node5", "RefNode", 1, 1, 1, {1, 1, 448, 448});
  const auto &ref_node6 = builder.AddNode("ref_node6", "RefNode", 1, 1, 1, {1, 1, 448, 448});
  const auto &ref_node7 = builder.AddNode("ref_node7", "RefNode", 1, 1, 1, {1, 1, 448, 448});
  const auto &ref_node8 = builder.AddNode("ref_node8", "RefNode", 1, 1, 1, {1, 1, 448, 448});
  const auto &ref_node9 = builder.AddNode("ref_node9", "RefNode", 1, 1, 1, {1, 1, 448, 448});

  builder.AddDataEdge(e, 0, f, 0);
  builder.AddDataEdge(f, 0, d, 0);

  builder.AddDataEdge(hcom1, 0, ref_node1, 0);
  builder.AddDataEdge(ref_node1, 0, a, 0);

  builder.AddDataEdge(hcom1, 1, ref_node2, 0);
  builder.AddDataEdge(ref_node2, 0, hcom2, 0);
  builder.AddDataEdge(hcom1, 2, ref_node3, 0);
  builder.AddDataEdge(ref_node3, 0, hcom2, 1);

  builder.AddDataEdge(hcom1, 3, ref_node4, 0);
  builder.AddDataEdge(ref_node4, 0, hcom3, 0);
  builder.AddDataEdge(hcom1, 4, ref_node5, 0);
  builder.AddDataEdge(ref_node5, 0, hcom3, 1);

  builder.AddDataEdge(hcom1, 5, ref_node6, 0);
  builder.AddDataEdge(ref_node6, 0, b, 0);

  builder.AddDataEdge(hcom1, 6, ref_node7, 0);
  builder.AddDataEdge(ref_node7, 0, hcom4, 0);

  builder.AddDataEdge(c, 0, ref_node8, 0);
  builder.AddDataEdge(ref_node8, 0, hcom4, 1);
  builder.AddDataEdge(d, 0, ref_node9, 0);
  builder.AddDataEdge(ref_node9, 0, hcom4, 2);

  builder.AddDataEdge(a, 0, netoutput, 0);

  AttrUtils::SetBool(ref_node1->GetOpDesc(), ATTR_NAME_REFERENCE, true);
  AttrUtils::SetBool(ref_node2->GetOpDesc(), ATTR_NAME_REFERENCE, true);
  AttrUtils::SetBool(ref_node3->GetOpDesc(), ATTR_NAME_REFERENCE, true);
  AttrUtils::SetBool(ref_node4->GetOpDesc(), ATTR_NAME_REFERENCE, true);
  AttrUtils::SetBool(ref_node5->GetOpDesc(), ATTR_NAME_REFERENCE, true);
  AttrUtils::SetBool(ref_node6->GetOpDesc(), ATTR_NAME_REFERENCE, true);
  AttrUtils::SetBool(ref_node7->GetOpDesc(), ATTR_NAME_REFERENCE, true);
  AttrUtils::SetBool(ref_node8->GetOpDesc(), ATTR_NAME_REFERENCE, true);
  AttrUtils::SetBool(ref_node9->GetOpDesc(), ATTR_NAME_REFERENCE, true);
  auto input_name = ref_node1->GetOpDescBarePtr()->GetInputNameByIndex(0);
  std::map<std::string, uint32_t> output_name_idx;
  output_name_idx.insert(std::make_pair(input_name, 0));
  ref_node1->GetOpDescBarePtr()->UpdateOutputName(output_name_idx);
  ref_node2->GetOpDescBarePtr()->UpdateOutputName(output_name_idx);
  ref_node3->GetOpDescBarePtr()->UpdateOutputName(output_name_idx);
  ref_node4->GetOpDescBarePtr()->UpdateOutputName(output_name_idx);
  ref_node5->GetOpDescBarePtr()->UpdateOutputName(output_name_idx);
  ref_node6->GetOpDescBarePtr()->UpdateOutputName(output_name_idx);
  ref_node7->GetOpDescBarePtr()->UpdateOutputName(output_name_idx);
  ref_node8->GetOpDescBarePtr()->UpdateOutputName(output_name_idx);
  ref_node9->GetOpDescBarePtr()->UpdateOutputName(output_name_idx);

  (void)AttrUtils::SetBool(hcom1->GetOpDesc(), ATTR_NAME_CONTINUOUS_OUTPUT, true);
  (void)AttrUtils::SetBool(hcom2->GetOpDesc(), ATTR_NAME_CONTINUOUS_INPUT, true);
  (void)AttrUtils::SetBool(hcom3->GetOpDesc(), ATTR_NAME_CONTINUOUS_INPUT, true);
  (void)AttrUtils::SetBool(hcom4->GetOpDesc(), ATTR_NAME_CONTINUOUS_INPUT, true);
  const auto &root_graph = builder.GetGraph();
  MemConflictShareGraph::TopologicalSortingMock(root_graph, {"hcom1", "e", "f", "c", "d"});
  return root_graph;
}
/*
 *    a hcom1 b hcom2 c
 *    \  ||   |  ||   |
 *          hcom3
 */
ComputeGraphPtr BuildContinuousOutInWithTwoContinuousOutNodeAndRefNode() {
  auto assign = OP_CFG(ASSIGN).Attr(ATTR_NAME_REFERENCE, true).InNames({"ref"}).OutNames({"ref"});
  DEF_GRAPH(g1) {
    CHAIN(NODE("a", RELU)->NODE("asign1", assign)->NODE("hcom3", HCOMALLREDUCE)->NODE("netoutput", NETOUTPUT));
    CHAIN(NODE("hcom1", HCOMALLREDUCE)->NODE("asign2", assign)->NODE("hcom3", HCOMALLREDUCE));
    CHAIN(NODE("hcom1", HCOMALLREDUCE)->NODE("asign3", assign)->NODE("hcom3", HCOMALLREDUCE));
    CHAIN(NODE("b", RELU)->NODE("asign4", assign)->NODE("hcom3", HCOMALLREDUCE));
    CHAIN(NODE("hcom2", HCOMALLREDUCE)->NODE("asign5", assign)->NODE("hcom3", HCOMALLREDUCE));
    CHAIN(NODE("hcom2", HCOMALLREDUCE)->NODE("asign6", assign)->NODE("hcom3", HCOMALLREDUCE));
    CHAIN(NODE("c", RELU)->NODE("asign7", assign)->NODE("hcom3", HCOMALLREDUCE));
  };

  auto graph = ToComputeGraph(g1);
  MemConflictShareGraph::SetContinuousOutput(graph, "hcom1");
  MemConflictShareGraph::SetContinuousOutput(graph, "hcom2");
  MemConflictShareGraph::SetContinuousInput(graph, "hcom3");
  MemConflictShareGraph::SetShapeForAllNodes(graph, {1, 1, 448, 448});
  MemConflictShareGraph::SetSizeForAllNodes(graph);

  graph->TopologicalSorting();
  return graph;
}

/*
 *    hcom1
 *   / |   \
 *  a hcom2  c
 */
ComputeGraphPtr BuildContinuousOutInWithOneInput() {
  auto assign = OP_CFG(ASSIGN).Attr(ATTR_NAME_REFERENCE, true).InNames({"ref"}).OutNames({"ref"});
  DEF_GRAPH(g1) {
    CHAIN(NODE("hcom1", HCOMALLREDUCE)->NODE("asign1", assign)->NODE("a", RELU)->NODE("netoutput", NETOUTPUT));
    CHAIN(NODE("hcom1", HCOMALLREDUCE)->NODE("asign2", assign)->NODE("hcom2", HCOMALLREDUCE));
    CHAIN(NODE("hcom1", HCOMALLREDUCE)->NODE("asign3", assign)->NODE("c", RELU));
  };

  auto graph = ToComputeGraph(g1);
  MemConflictShareGraph::SetContinuousOutput(graph, "hcom1");
  MemConflictShareGraph::SetContinuousInput(graph, "hcom2");
  MemConflictShareGraph::SetShapeForAllNodes(graph, {1, 1, 448, 448});
  MemConflictShareGraph::SetSizeForAllNodes(graph);

  graph->TopologicalSorting();
  return graph;
}
/*
 *  a  hcom1 b
 *   \ |     /
 *    hcom2
 */
ComputeGraphPtr BuildContinuousOutInWithOneOutput() {
  auto assign = OP_CFG(ASSIGN).Attr(ATTR_NAME_REFERENCE, true).InNames({"ref"}).OutNames({"ref"});
  DEF_GRAPH(g1) {
    CHAIN(NODE("a", RELU)->NODE("asign1", assign)->NODE("hcom2", HCOMALLREDUCE)->NODE("netoutput", NETOUTPUT));
    CHAIN(NODE("hcom1", HCOMALLREDUCE)->NODE("asign2", assign)->NODE("hcom2", HCOMALLREDUCE));
    CHAIN(NODE("b", RELU)->NODE("asign3", assign)->NODE("hcom2", HCOMALLREDUCE));
  };

  auto graph = ToComputeGraph(g1);
  MemConflictShareGraph::SetContinuousOutput(graph, "hcom1");
  MemConflictShareGraph::SetContinuousInput(graph, "hcom2");
  MemConflictShareGraph::SetShapeForAllNodes(graph, {1, 1, 448, 448});
  MemConflictShareGraph::SetSizeForAllNodes(graph);

  graph->TopologicalSorting();
  return graph;
}

/*
 *  a  hcom1 hcom2 hcom3
 *  | |    | |   | |   |
 *  hcom4 hcom5  hcom6 b
 */
ComputeGraphPtr BuildContinuousOutInWithThreeNode() {
  auto assign = OP_CFG(ASSIGN).Attr(ATTR_NAME_REFERENCE, true).InNames({"ref"}).OutNames({"ref"});
  DEF_GRAPH(g1) {
    CHAIN(NODE("a", RELU)->NODE("hcom4", HCOMALLREDUCE));
    CHAIN(NODE("hcom1", HCOMALLREDUCE)->NODE("hcom4", HCOMALLREDUCE));
    CHAIN(NODE("hcom1", HCOMALLREDUCE)->NODE("hcom5", HCOMALLREDUCE));
    CHAIN(NODE("hcom2", HCOMALLREDUCE)->NODE("assign", assign)->NODE("hcom5", HCOMALLREDUCE));
    CHAIN(NODE("hcom2", HCOMALLREDUCE)->NODE("hcom6", HCOMALLREDUCE));
    CHAIN(NODE("hcom3", HCOMALLREDUCE)->NODE("hcom6", HCOMALLREDUCE));
    CHAIN(NODE("hcom3", HCOMALLREDUCE)->NODE("b", RELU)->NODE("netoutput", NETOUTPUT));
  };

  auto graph = ToComputeGraph(g1);
  MemConflictShareGraph::SetContinuousOutput(graph, "hcom1");
  MemConflictShareGraph::SetContinuousOutput(graph, "hcom2");
  MemConflictShareGraph::SetContinuousOutput(graph, "hcom3");
  MemConflictShareGraph::SetContinuousInput(graph, "hcom4");
  MemConflictShareGraph::SetContinuousInput(graph, "hcom5");
  MemConflictShareGraph::SetContinuousInput(graph, "hcom6");
  MemConflictShareGraph::SetShapeForAllNodes(graph, {1, 1, 448, 448});
  MemConflictShareGraph::SetSizeForAllNodes(graph);

  graph->TopologicalSorting();
  return graph;
}
/*
 *  a  hcom1
 *  |  | | \
 *  hcom2 hcom3
 */
ComputeGraphPtr BuildContinuousOutInVisitLastNodeFirst() {
  DEF_GRAPH(g1) {
    CHAIN(NODE("a", RELU)->NODE("hcom2", HCOMALLREDUCE));
    CHAIN(NODE("hcom1", HCOMALLREDUCE)->NODE("hcom2", HCOMALLREDUCE));
    CHAIN(NODE("hcom1", HCOMALLREDUCE)->NODE("hcom3", HCOMALLREDUCE));
    CHAIN(NODE("hcom1", HCOMALLREDUCE)->NODE("hcom3", HCOMALLREDUCE));
  };

  auto graph = ToComputeGraph(g1);
  MemConflictShareGraph::SetContinuousOutput(graph, "hcom1");
  MemConflictShareGraph::SetContinuousInput(graph, "hcom2");
  MemConflictShareGraph::SetContinuousInput(graph, "hcom3");
  MemConflictShareGraph::SetShapeForAllNodes(graph, {1, 1, 448, 448});
  MemConflictShareGraph::SetSizeForAllNodes(graph);

  MemConflictShareGraph::TopologicalSortingMock(graph, {"hcom1", "hcom3", "hcom2"});
  return graph;
}
/*
 *         a
 *         |
 * d   c   b
 *  \ / \ /
 *  pc1 pc2
 *   |  |
 *   e  f
 *   \  /
 *  netoutput
 *
 *   topo order: b < d < c, pc2 < pc1
 */
ComputeGraphPtr BuildOneNodeConnectTwoPhonyConcat() {
  DEF_GRAPH(g1) {
    CHAIN(NODE("c", RELU)->NODE("pc1", PHONYCONCAT));
    CHAIN(NODE("c", RELU)->NODE("pc2", PHONYCONCAT));
    CHAIN(NODE("d", RELU)->NODE("pc1", PHONYCONCAT)->NODE("e", RELU)->NODE("netoutput", NETOUTPUT));
    CHAIN(NODE("a", RELU)->NODE("b", RELU)->NODE("pc2", PHONYCONCAT)->NODE("f", RELU)->NODE("netoutput", NETOUTPUT));
  };

  auto graph = ToComputeGraph(g1);
  MemConflictShareGraph::SetNoPaddingContinuousInput(graph, "pc1");
  MemConflictShareGraph::SetNoPaddingContinuousInput(graph, "pc2");
  MemConflictShareGraph::SetShapeForAllNodes(graph, {1, 1, 448, 224});
  for (auto &node : graph->GetAllNodes()) {
    const auto node_name = node->GetName();
    if (node_name == "pc1" || node_name == "pc2" || node_name == "e" || node_name == "f" || node_name == "a") {
      for (size_t i = 0U; i < node->GetOutDataNodesSize(); ++i) {
        auto out_tensor = node->GetOpDescBarePtr()->MutableOutputDesc(i);
        out_tensor->SetShape(GeShape{{1, 1, 448, 448}});
      }
    }

    if (node_name == "e" || node_name == "f" || node_name == "netoutput"|| node_name == "b") {
      for (size_t i = 0U; i < node->GetInDataNodesSize(); ++i) {
        auto in_tensor = node->GetOpDescBarePtr()->MutableInputDesc(i);
        in_tensor->SetShape(GeShape{{1, 1, 448, 448}});
      }
    }
  }
  MemConflictShareGraph::SetSizeForAllNodes(graph);
  MemConflictShareGraph::TopologicalSortingMock(graph, {"a", "b", "d", "c", "pc2", "pc1"});
  return graph;
}
}  // namespace block_mem_ut
} // namespace ge
